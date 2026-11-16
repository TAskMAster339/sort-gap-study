"""Стоимость сопоставления после искусственной паузы."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sort_gap_study.config import SortRecoveryConfig


def recovery_costs(
    predictions: NDArray[np.float64],
    detections: NDArray[np.float64],
    centers: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    config: SortRecoveryConfig,
) -> NDArray[np.float64]:
    """Рассчитать стоимость пар по прогнозам фильтра до коррекции."""
    detection_sizes = detections[:, 2:] - detections[:, :2]
    detection_centers = detections[:, :2] + detection_sizes / 2
    costs = np.empty((len(predictions), len(detections)), dtype=np.float64)
    for row, (bbox, (center, covariance)) in enumerate(
        zip(predictions, centers, strict=True),
    ):
        displacement = detection_centers - center
        weighted = np.linalg.solve(covariance, displacement.T).T
        mahalanobis_squared = (displacement * weighted).sum(axis=1)

        size = bbox[2:] - bbox[:2]
        center_shift = np.linalg.norm(displacement, axis=1) / np.linalg.norm(size)
        ratios = detection_sizes / size
        size_ratio = np.maximum(ratios, 1 / ratios).max(axis=1)
        admissible = (
            (mahalanobis_squared <= config.max_mahalanobis_squared)
            & (center_shift <= config.max_center_shift)
            & (size_ratio <= config.max_size_ratio)
        )
        cost = (
            0.5 * mahalanobis_squared / config.max_mahalanobis_squared
            + 0.3 * center_shift / config.max_center_shift
            + 0.2 * np.log(size_ratio) / np.log(config.max_size_ratio)
        )
        costs[row] = np.where(admissible, cost, np.inf)
    return costs
