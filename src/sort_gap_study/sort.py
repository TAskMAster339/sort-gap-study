"""Покадровый трекер на основе базового SORT."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sort_gap_study.association import assign_costs, associate_iou
from sort_gap_study.config import (
    SortKalmanConfig,
    SortLifecycleConfig,
    SortRecoveryConfig,
)
from sort_gap_study.kalman import SortKalmanFilter
from sort_gap_study.recovery import recovery_costs
from sort_gap_study.tracker import FrameEvents, TrackEstimate, validate_step


@dataclass
class _Track:
    track_id: int
    kalman: SortKalmanFilter
    last_observed_frame: int
    hits: int = 1


class SortTracker:
    """Обрабатывать последовательные исходные кадры одного видео."""

    def __init__(
        self,
        kalman_config: SortKalmanConfig,
        lifecycle: SortLifecycleConfig,
        *,
        recovery: SortRecoveryConfig | None = None,
    ) -> None:
        self._kalman_config = kalman_config
        self._lifecycle = lifecycle
        self._recovery = recovery
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self._last_frame: int | None = None
        self._last_events = FrameEvents()
        self._gap_start: int | None = None

    @property
    def last_events(self) -> FrameEvents:
        """Вернуть события последнего обработанного кадра."""
        return self._last_events

    def step(
        self,
        frame_index: int,
        detections: ArrayLike,
        *,
        artificial_gap: bool,
    ) -> tuple[TrackEstimate, ...]:
        """Обработать (x1, y1, x2, y2, score) и вернуть оценки по порядку ID."""
        frame_index, rows = validate_step(
            frame_index,
            self._last_frame,
            detections,
            artificial_gap=artificial_gap,
        )
        boxes = rows[:, :-1]

        alive = []
        removed = []
        predictions = []
        for track in self._tracks:
            if frame_index - track.last_observed_frame > self._lifecycle.max_age:
                removed.append(track.track_id)
                continue
            alive.append(track)
            track.kalman.predict()
            predictions.append(track.kalman.bbox)
        self._tracks = alive

        association = associate_iou(predictions, boxes)
        recovery_indices = (
            ()
            if artificial_gap
            else self._recovery_candidates(association.unmatched_tracks)
        )
        recovery_matches = self._match_recovery(
            recovery_indices,
            association.unmatched_detections,
            predictions,
            boxes,
        )
        matched = []
        for track_index, detection_index in association.matches + recovery_matches:
            track = self._tracks[track_index]
            track.kalman.correct(boxes[detection_index])
            track.last_observed_frame = frame_index
            track.hits += 1
            matched.append((track.track_id, detection_index))

        created = []
        recovered_detections = {
            detection_index for _, detection_index in recovery_matches
        }
        unmatched_detections = (
            detection_index
            for detection_index in association.unmatched_detections
            if detection_index not in recovered_detections
        )
        for detection_index in unmatched_detections:
            kalman = SortKalmanFilter(boxes[detection_index], self._kalman_config)
            self._tracks.append(_Track(self._next_track_id, kalman, frame_index))
            created.append((self._next_track_id, detection_index))
            self._next_track_id += 1

        self._last_events = FrameEvents(
            matched=tuple(matched),
            created=tuple(created),
            removed=tuple(removed),
            recovery_candidates=tuple(
                self._tracks[i].track_id for i in recovery_indices
            ),
            recovery_matches=tuple(
                (self._tracks[track_index].track_id, detection_index)
                for track_index, detection_index in recovery_matches
            ),
        )
        if not artificial_gap:
            self._gap_start = None
        elif self._gap_start is None:
            self._gap_start = frame_index
        self._last_frame = frame_index
        return tuple(
            TrackEstimate(
                track.track_id,
                tuple(track.kalman.bbox.tolist()),
                observed=track.last_observed_frame == frame_index,
            )
            for track in self._tracks
            if track.hits >= self._lifecycle.min_hits
        )

    def _match_recovery(
        self,
        track_indices: tuple[int, ...],
        detection_indices: tuple[int, ...],
        predictions: list[NDArray[np.float64]],
        boxes: NDArray[np.float64],
    ) -> tuple[tuple[int, int], ...]:
        if self._recovery is None or not track_indices or not detection_indices:
            return ()
        costs = recovery_costs(
            np.asarray([predictions[i] for i in track_indices]),
            boxes[list(detection_indices)],
            [self._tracks[i].kalman.project_center() for i in track_indices],
            self._recovery,
        )
        return tuple(
            (track_indices[row], detection_indices[col])
            for row, col in assign_costs(costs).matches
        )

    def _recovery_candidates(
        self,
        unmatched_tracks: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Отобрать индексы живых треков для второго назначения."""
        if self._gap_start is None:
            return ()
        last_available = self._gap_start - 1
        return tuple(
            i
            for i in unmatched_tracks
            if self._tracks[i].hits >= self._lifecycle.min_hits
            and self._tracks[i].last_observed_frame == last_available
        )
