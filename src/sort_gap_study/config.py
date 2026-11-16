"""Чтение конфигурации экспериментального протокола."""

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SequenceSplit:
    tune: tuple[str, ...]
    evaluation: tuple[str, ...]


@dataclass(frozen=True)
class SortKalmanConfig:
    initial_covariance: tuple[float, ...]
    process_noise: tuple[float, ...]
    measurement_noise: tuple[float, ...]
    bbox_epsilon: float


@dataclass(frozen=True)
class SortLifecycleConfig:
    min_hits: int
    max_age: int


@dataclass(frozen=True)
class SortRecoveryConfig:
    max_mahalanobis_squared: float
    max_center_shift: float
    max_size_ratio: float


def load_sort_recovery(path: Path) -> SortRecoveryConfig:
    """Загрузить пороги второго назначения."""
    with path.open("rb") as file:
        sort = tomllib.load(file).get("sort")

    recovery = sort.get("recovery") if isinstance(sort, dict) else None
    if not isinstance(recovery, dict):
        raise TypeError("Ожидается таблица [sort.recovery].")

    max_size_ratio = _positive_number(
        recovery.get("max_size_ratio"),
        "sort.recovery.max_size_ratio",
    )
    if max_size_ratio <= 1:
        raise ValueError(
            "sort.recovery.max_size_ratio: ожидается число больше единицы.",
        )

    return SortRecoveryConfig(
        max_mahalanobis_squared=_positive_number(
            recovery.get("max_mahalanobis_squared"),
            "sort.recovery.max_mahalanobis_squared",
        ),
        max_center_shift=_positive_number(
            recovery.get("max_center_shift"),
            "sort.recovery.max_center_shift",
        ),
        max_size_ratio=max_size_ratio,
    )


def load_sort_lifecycle(path: Path) -> SortLifecycleConfig:
    """Загрузить условия подтверждения и удаления трека SORT."""
    with path.open("rb") as file:
        sort = tomllib.load(file).get("sort")

    lifecycle = sort.get("lifecycle") if isinstance(sort, dict) else None
    if not isinstance(lifecycle, dict):
        raise TypeError("Ожидается таблица [sort.lifecycle].")

    min_hits = lifecycle.get("min_hits")
    max_age = lifecycle.get("max_age")
    if type(min_hits) is not int or min_hits < 1:
        raise ValueError("sort.lifecycle.min_hits: ожидается положительное целое.")
    if type(max_age) is not int or max_age < 0:
        raise ValueError("sort.lifecycle.max_age: ожидается неотрицательное целое.")
    return SortLifecycleConfig(min_hits=min_hits, max_age=max_age)


def load_sort_kalman(path: Path) -> SortKalmanConfig:
    """Загрузить диагонали ковариаций и нижнюю границу площади и пропорций."""
    with path.open("rb") as file:
        sort = tomllib.load(file).get("sort")

    kalman = sort.get("kalman") if isinstance(sort, dict) else None
    if not isinstance(kalman, dict):
        raise TypeError("Ожидается таблица [sort.kalman].")

    return SortKalmanConfig(
        initial_covariance=_covariance_diagonal(kalman, "initial_covariance", 7),
        process_noise=_covariance_diagonal(kalman, "process_noise", 7),
        measurement_noise=_covariance_diagonal(kalman, "measurement_noise", 4),
        bbox_epsilon=_positive_number(
            kalman.get("bbox_epsilon"),
            "sort.kalman.bbox_epsilon",
        ),
    )


def _covariance_diagonal(
    table: dict[str, object],
    name: str,
    size: int,
) -> tuple[float, ...]:
    values = table.get(name)
    key = f"sort.kalman.{name}"
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"{key}: ожидается список из {size} чисел.")
    return tuple(_positive_number(value, key) for value in values)


def _positive_number(value: object, key: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key}: ожидается конечное положительное число.")
    return float(value)


def load_split(path: Path) -> SequenceSplit:
    """Загрузить и проверить списки последовательностей из TOML."""
    with path.open("rb") as file:
        split = tomllib.load(file).get("split")

    if not isinstance(split, dict):
        raise TypeError("Ожидается таблица [split].")

    tune = _sequence_ids(split.get("tune"), "split.tune")
    evaluation = _sequence_ids(split.get("evaluation"), "split.evaluation")
    overlap = sorted(set(tune).intersection(evaluation))
    if overlap:
        raise ValueError(
            f"Настроечные и итоговые последовательности "
            f"пересекаются: {', '.join(overlap)}.",
        )

    return SequenceSplit(tune=tune, evaluation=evaluation)


def _sequence_ids(value: object, key: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(item, str) and re.fullmatch(r"[0-9]{4}", item) for item in value
        )
    ):
        raise ValueError(f"{key}: ожидается непустой список четырёхзначных строк.")
    if len(set(value)) != len(value):
        raise ValueError(f"{key}: список содержит повторяющиеся последовательности.")
    return tuple(value)
