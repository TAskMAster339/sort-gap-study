"""Ассоциация SORT по IoU и глобальное назначение."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

_BBOX_COORDINATES = 4


@dataclass(frozen=True)
class Association:
    """Пары (индекс трека, индекс детекции) и оставшиеся индексы входов."""

    matches: tuple[tuple[int, int], ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


def pairwise_iou(
    track_boxes: ArrayLike,
    detection_boxes: ArrayLike,
) -> NDArray[np.float64]:
    """Вычислить матрицу IoU для пар bbox в формате (x1, y1, x2, y2)."""
    tracks = as_box_array(track_boxes)
    detections = as_box_array(detection_boxes)
    track_size = tracks[:, 2:] - tracks[:, :2]
    detection_size = detections[:, 2:] - detections[:, :2]

    top_left = np.maximum(tracks[:, None, :2], detections[None, :, :2])
    bottom_right = np.minimum(tracks[:, None, 2:], detections[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    union = (
        track_size.prod(axis=1)[:, None] + detection_size.prod(axis=1) - intersection
    )
    return intersection / union


def associate_iou(
    track_boxes: ArrayLike,
    detection_boxes: ArrayLike,
    *,
    min_iou: float = 0.3,
) -> Association:
    """Максимизировать число допустимых пар, затем их суммарный IoU."""
    if not 0 <= min_iou <= 1:
        raise ValueError("Порог IoU должен принадлежать отрезку [0, 1].")

    iou = pairwise_iou(track_boxes, detection_boxes)
    return assign_costs(np.where(iou >= min_iou, 1 - iou, np.inf))


def assign_costs(pair_costs: NDArray[np.float64]) -> Association:
    """Максимизировать число пар, затем минимизировать сумму стоимостей.

    Допустимые стоимости лежат в диапазоне [0, 1]. Запрещённые пары имеют
    бесконечную стоимость.
    """
    track_count, detection_count = pair_costs.shape
    if not track_count or not detection_count:
        return Association((), tuple(range(track_count)), tuple(range(detection_count)))

    # Дополнительные столбцы позволяют оставить любой трек без детекции.
    # Цена пропуска превышает максимально возможную сумму стоимостей реальных пар.
    unmatched_cost = min(track_count, detection_count) + 1
    costs = np.full((track_count, detection_count + track_count), unmatched_cost, float)
    costs[:, :detection_count] = pair_costs
    rows, columns = linear_sum_assignment(costs)

    real = columns < detection_count
    matches = tuple(zip(rows[real].tolist(), columns[real].tolist(), strict=True))
    unmatched_tracks = tuple(rows[~real].tolist())
    unmatched_detections = tuple(
        sorted(set(range(detection_count)) - set(columns[real])),
    )
    return Association(matches, unmatched_tracks, unmatched_detections)


def as_box_array(value: ArrayLike) -> NDArray[np.float64]:
    """Проверить геометрию боксов и представить их массивом (N, 4)."""
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.shape == (0,):
        return boxes.reshape(0, _BBOX_COORDINATES)
    if boxes.shape[1:] != (_BBOX_COORDINATES,):
        raise ValueError("Ожидается массив координат bbox размера (N, 4).")
    if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError(
            "Для bbox требуются конечные координаты, положительные ширина и высота.",
        )
    return boxes
