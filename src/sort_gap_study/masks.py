"""Маски искусственных пропусков на исходной шкале кадров."""

from math import floor
from typing import Literal

import numpy as np


def make_mask(
    frames: range,
    loss_fraction: float,
    kind: Literal["uniform", "random"],
    *,
    seed: int,
    warmup: int = 10,
) -> tuple[int, ...]:
    """Вернуть исходные номера пропущенных кадров по возрастанию."""
    if frames.step != 1 or frames.start < 0 or not 0 <= warmup <= len(frames):
        message = "Требуются непрерывный диапазон кадров и допустимый прогрев."
        raise ValueError(message)
    if not 0 <= loss_fraction <= 1:
        message = "Доля пропусков должна быть от 0 до 1."
        raise ValueError(message)
    if kind not in ("uniform", "random"):
        message = f"Неизвестный тип маски: {kind}."
        raise ValueError(message)

    evaluation = frames[warmup:]
    count = floor(loss_fraction * len(evaluation) + 0.5)
    if count == 0:
        return ()

    rng = np.random.default_rng(seed)
    if kind == "uniform":
        phase = rng.random()
        size = len(evaluation)
        indices = (floor((j + phase) * size / count) for j in range(count))
        return tuple(evaluation[index] for index in indices)

    indices = rng.choice(len(evaluation), size=count, replace=False)
    return tuple(evaluation[index] for index in sorted(indices.tolist()))
