"""Покадровый трекер на основе базового SORT."""

from dataclasses import dataclass
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sort_gap_study.association import as_box_array, associate_iou
from sort_gap_study.config import SortKalmanConfig, SortLifecycleConfig
from sort_gap_study.kalman import SortKalmanFilter

_DETECTION_COLUMNS = 5


@dataclass(frozen=True)
class TrackEstimate:
    """Оценка подтверждённого трека на текущем кадре."""

    track_id: int
    bbox: tuple[float, float, float, float]
    observed: bool


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
    ) -> None:
        self._kalman_config = kalman_config
        self._lifecycle = lifecycle
        self._tracks: list[_Track] = []
        self._next_track_id = 1
        self._last_frame: int | None = None

    def step(
        self,
        frame_index: int,
        detections: ArrayLike,
        *,
        artificial_gap: bool,
    ) -> tuple[TrackEstimate, ...]:
        """Обработать (x1, y1, x2, y2, score) и вернуть оценки по порядку ID."""
        frame_index = index(frame_index)
        if frame_index < 0 or (
            self._last_frame is not None and frame_index != self._last_frame + 1
        ):
            raise ValueError(
                "Кадры должны идти подряд. Начальный индекс неотрицателен.",
            )
        boxes = _detection_boxes(detections)
        if artificial_gap and len(boxes):
            raise ValueError("Для искусственного пропуска требуется пустой массив.")

        self._tracks = [
            track
            for track in self._tracks
            if frame_index - track.last_observed_frame <= self._lifecycle.max_age
        ]
        predictions = []
        for track in self._tracks:
            track.kalman.predict()
            predictions.append(track.kalman.bbox)

        association = associate_iou(predictions, boxes)
        for track_index, detection_index in association.matches:
            track = self._tracks[track_index]
            track.kalman.correct(boxes[detection_index])
            track.last_observed_frame = frame_index
            track.hits += 1

        for detection_index in association.unmatched_detections:
            kalman = SortKalmanFilter(boxes[detection_index], self._kalman_config)
            self._tracks.append(_Track(self._next_track_id, kalman, frame_index))
            self._next_track_id += 1

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


def _detection_boxes(detections: ArrayLike) -> NDArray[np.float64]:
    rows = np.asarray(detections, dtype=np.float64)
    if rows.shape == (0,):
        rows = rows.reshape(0, _DETECTION_COLUMNS)
    if rows.shape[1:] != (_DETECTION_COLUMNS,):
        raise ValueError("Ожидается массив детекций размера (N, 5).")
    if not np.isfinite(rows[:, -1]).all():
        raise ValueError("Оценки уверенности детекций должны быть конечными.")
    return as_box_array(rows[:, :-1])
