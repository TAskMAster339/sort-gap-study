"""Общий покадровый интерфейс трекеров."""

from dataclasses import dataclass
from operator import index
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sort_gap_study.association import as_box_array

_DETECTION_COLUMNS = 5


@dataclass(frozen=True)
class TrackEstimate:
    """Оценка подтверждённого трека на текущем кадре."""

    track_id: int
    bbox: tuple[float, float, float, float]
    observed: bool


@dataclass(frozen=True)
class FrameEvents:
    """Сопоставления, создание, удаление и кандидаты после паузы."""

    matched: tuple[tuple[int, int], ...] = ()
    created: tuple[tuple[int, int], ...] = ()
    removed: tuple[int, ...] = ()
    recovery_candidates: tuple[int, ...] = ()
    recovery_matches: tuple[tuple[int, int], ...] = ()


class Tracker(Protocol):
    @property
    def last_events(self) -> FrameEvents: ...

    def step(
        self,
        frame_index: int,
        detections: ArrayLike,
        *,
        artificial_gap: bool,
    ) -> tuple[TrackEstimate, ...]: ...


def validate_step(
    frame_index: int,
    last_frame: int | None,
    detections: ArrayLike,
    *,
    artificial_gap: bool,
) -> tuple[int, NDArray[np.float64]]:
    """Проверить исходный номер кадра и привести детекции к массиву (N, 5)."""
    frame_index = index(frame_index)
    if frame_index < 0 or (last_frame is not None and frame_index != last_frame + 1):
        raise ValueError("Кадры должны идти подряд. Начальный индекс неотрицателен.")
    rows = np.asarray(detections, dtype=np.float64)
    if rows.shape == (0,):
        rows = rows.reshape(0, _DETECTION_COLUMNS)
    if rows.shape[1:] != (_DETECTION_COLUMNS,):
        raise ValueError("Ожидается массив детекций размера (N, 5).")
    if not np.isfinite(rows[:, -1]).all():
        raise ValueError("Оценки уверенности детекций должны быть конечными.")
    as_box_array(rows[:, :-1])
    if artificial_gap and len(rows):
        raise ValueError("Для искусственного пропуска требуется пустой массив.")
    return frame_index, rows
