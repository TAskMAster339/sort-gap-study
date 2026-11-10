"""Маски искусственных пропусков на исходной шкале кадров."""

from dataclasses import dataclass
from itertools import pairwise
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
    evaluation, count = _evaluation_budget(frames, loss_fraction, warmup)
    if kind not in ("uniform", "random"):
        message = f"Неизвестный тип маски: {kind}."
        raise ValueError(message)

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


@dataclass(frozen=True)
class BurstMask:
    """Результат генерации. Серия хранится как пара (первый кадр, длина)."""

    status: Literal["ok", "infeasible"]
    removed_frames: tuple[int, ...] = ()
    runs: tuple[tuple[int, int], ...] = ()
    reason: str | None = None


def make_burst_mask(
    frames: range,
    loss_fraction: float,
    *,
    series_length: int,
    seed: int,
    warmup: int = 10,
) -> BurstMask:
    """Разместить серии пропусков на исходной шкале кадров."""
    evaluation, count = _evaluation_budget(frames, loss_fraction, warmup)
    if series_length < 1:
        message = "Длина серии должна быть положительной."
        raise ValueError(message)
    if count == 0:
        return BurstMask(status="ok")

    full_blocks, remainder = divmod(count, series_length)
    blocks = [series_length] * full_blocks
    if remainder:
        blocks.append(remainder)

    block_count = len(blocks)
    required = count + block_count - 1
    if required > len(evaluation):
        return BurstMask(
            status="infeasible",
            reason=(
                f"Для {count} пропусков и {block_count} серий требуется "
                f"{required} кадров. Доступно: {len(evaluation)}."
            ),
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    spare = len(evaluation) - required
    # Distribute spare frames among the gaps, including before and after the blocks.
    bars = rng.choice(spare + block_count, size=block_count, replace=False)
    positions = [-1, *sorted(bars.tolist())]
    gaps = (right - left - 1 for left, right in pairwise(positions))

    runs = []
    cursor = evaluation.start
    for size, gap in zip(blocks, gaps, strict=True):
        cursor += gap
        runs.append((cursor, size))
        cursor += size + 1

    return BurstMask(
        status="ok",
        removed_frames=tuple(
            frame for start, size in runs for frame in range(start, start + size)
        ),
        runs=tuple(runs),
    )


def _evaluation_budget(
    frames: range,
    loss_fraction: float,
    warmup: int,
) -> tuple[range, int]:
    if frames.step != 1 or frames.start < 0 or not 0 <= warmup <= len(frames):
        message = "Требуются непрерывный диапазон кадров и допустимый прогрев."
        raise ValueError(message)
    if not 0 <= loss_fraction <= 1:
        message = "Доля пропусков должна быть от 0 до 1."
        raise ValueError(message)
    evaluation = frames[warmup:]
    count = floor(loss_fraction * len(evaluation) + 0.5)
    return evaluation, count
