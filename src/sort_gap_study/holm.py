"""Поправка Holm для семейства из шести проверок H1 и H2."""

import json
from collections.abc import Sequence
from pathlib import Path

_FAMILY = (
    ("h1_idf1", "gt"),
    ("h1_idf1", "regionlet"),
    ("h2_length_difference", "gt"),
    ("h2_rfp_slack", "gt"),
    ("h2_length_difference", "regionlet"),
    ("h2_rfp_slack", "regionlet"),
)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Вернуть скорректированные p в порядке входных значений."""
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("Значения p должны принадлежать отрезку [0, 1].")
    count = len(p_values)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(sorted(range(count), key=p_values.__getitem__)):
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def write_holm(h1: Path, h2: Path, output: Path) -> None:
    h1_results = json.loads(h1.read_text(encoding="utf-8"))["sources"]
    h2_results = json.loads(h2.read_text(encoding="utf-8"))["sources"]
    statuses = []
    raw_values = []
    for test, source in _FAMILY:
        record = (
            h1_results[source]
            if test == "h1_idf1"
            else h2_results[source][test.removeprefix("h2_")]
        )
        status = record["status"]
        if status not in ("ok", "unavailable"):
            raise ValueError(f"Неизвестный статус проверки {test}/{source}: {status}.")
        value = record["p_one_sided_raw"] if status == "ok" else 1.0
        if type(value) not in (int, float):
            raise ValueError(f"Ожидается числовое значение p для {test}/{source}.")
        statuses.append(status)
        raw_values.append(value)

    result = {
        "method": "holm",
        "family_size": len(_FAMILY),
        "tests": [
            {
                "test": test,
                "source": source,
                "status": status,
                "p_one_sided_raw": raw,
                "p_holm": adjusted,
            }
            for (test, source), status, raw, adjusted in zip(
                _FAMILY,
                statuses,
                raw_values,
                holm_adjust(raw_values),
                strict=True,
            )
        ],
    }
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(content)
