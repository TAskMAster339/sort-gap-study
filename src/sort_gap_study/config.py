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
class EvaluationGrid:
    mask_seeds: tuple[int, ...]
    burst_lengths_by_percent: dict[int, tuple[int, ...]]
    bootstrap_seed: int


def load_evaluation_grid(path: Path) -> EvaluationGrid:
    with path.open("rb") as file:
        grid = tomllib.load(file).get("evaluation_grid")
    if not isinstance(grid, dict):
        raise ValueError("Ожидается таблица [evaluation_grid].")
    lengths = grid.get("burst_lengths_by_percent")
    if not isinstance(lengths, dict) or not lengths:
        raise ValueError("Требуются доли пропусков и длины серий.")
    full_percent = 100
    by_percent = {}
    for key, values in lengths.items():
        percent = int(key)
        if str(percent) != key or not 0 < percent < full_percent:
            raise ValueError(f"Некорректная доля пропусков: {key}.")
        by_percent[percent] = _unique_integers(values, minimum=1)
    bootstrap_seed = grid.get("bootstrap_seed")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed: ожидается неотрицательное целое.")
    return EvaluationGrid(
        _unique_integers(grid.get("mask_seeds"), minimum=0),
        by_percent,
        bootstrap_seed,
    )


def _unique_integers(values: object, *, minimum: int) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("Требуется непустой список целых.")
    if any(type(value) is not int or value < minimum for value in values):
        raise ValueError(f"Ожидаются целые числа от {minimum}.")
    if len(values) != len(set(values)):
        raise ValueError("Список содержит повторяющиеся значения.")
    return tuple(values)


def load_regionlet_threshold(path: Path) -> float:
    with path.open("rb") as file:
        regionlet = tomllib.load(file).get("regionlet", {})
    if not isinstance(regionlet, dict):
        raise ValueError("Ожидается таблица [regionlet].")
    threshold = regionlet.get("score_threshold", 0.0)
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError("regionlet.score_threshold: ожидается конечное число.")
    return float(threshold)


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


@dataclass(frozen=True)
class OCSortConfig:
    delta_t: int
    inertia: float
    iou_threshold: float


def load_ocsort(path: Path) -> OCSortConfig:
    """Загрузить параметры ассоциации OC-SORT."""
    with path.open("rb") as file:
        config = tomllib.load(file).get("ocsort")
    if not isinstance(config, dict):
        raise TypeError("Ожидается таблица [ocsort].")

    delta_t = config.get("delta_t")
    if type(delta_t) is not int or delta_t < 1:
        raise ValueError("ocsort.delta_t: ожидается положительное целое.")
    inertia = _positive_number(config.get("inertia"), "ocsort.inertia")
    iou_threshold = _positive_number(
        config.get("iou_threshold"),
        "ocsort.iou_threshold",
    )
    if iou_threshold > 1:
        raise ValueError("ocsort.iou_threshold: допустимы значения до единицы.")
    return OCSortConfig(delta_t, inertia, iou_threshold)


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
