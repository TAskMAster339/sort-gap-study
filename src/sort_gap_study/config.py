"""Чтение конфигурации экспериментального протокола."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SequenceSplit:
    tune: tuple[str, ...]
    evaluation: tuple[str, ...]


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
