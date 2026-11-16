"""Выборки видео и центрированный односторонний bootstrap."""

import numpy as np
from numpy.typing import NDArray

BOOTSTRAP_RESAMPLES = 5000


def resample_indices(count: int, seed: int) -> NDArray[np.int64]:
    return np.random.default_rng(seed).integers(
        count,
        size=(BOOTSTRAP_RESAMPLES, count),
    )


def interval_and_p(
    point: float,
    replicates: NDArray[np.float64],
) -> tuple[tuple[float, float], float]:
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    exceedances = np.count_nonzero(replicates - point >= point)
    p_value = (1 + exceedances) / (len(replicates) + 1)
    return (float(lower), float(upper)), p_value
