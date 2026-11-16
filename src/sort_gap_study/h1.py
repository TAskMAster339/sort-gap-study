"""Парный bootstrap разности IDF1 исходного SORT для H1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sort_gap_study.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    interval_and_p,
    resample_indices,
)

_H1_LOSS_FRACTION = 0.5
_H1_SERIES_LENGTH = 3
_REQUIRED = {
    "sequence",
    "source",
    "method",
    "loss_fraction",
    "mask_kind",
    "series_length",
    "NGT",
    "IDF1",
}


def write_h1(by_video: Path, output: Path, *, seed: int) -> None:
    table = pd.read_csv(by_video, dtype={"sequence": "string"})
    missing = _REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {', '.join(sorted(missing))}.")
    if table["sequence"].isna().any():
        raise ValueError("Таблица содержит строки без номера видео.")
    if seed < 0:
        raise ValueError("Seed должен быть неотрицательным.")

    basic = table.loc[(table["method"] == "sort") & (table["NGT"] > 0)]
    result = {
        "effect": "IDF1_uniform_minus_burst_l3",
        "loss_fraction": _H1_LOSS_FRACTION,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "sources": {
            source: _analyze_source(basic.loc[basic["source"] == source], seed)
            for source in ("gt", "regionlet")
        },
    }
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(content)


def _analyze_source(rows: pd.DataFrame, seed: int) -> dict[str, object]:
    selected = rows.loc[rows["loss_fraction"] == _H1_LOSS_FRACTION]
    uniform = selected.loc[selected["mask_kind"] == "uniform"]
    burst = selected.loc[
        (selected["mask_kind"] == "burst")
        & (selected["series_length"] == _H1_SERIES_LENGTH)
    ]
    pairs = (
        uniform.set_index("sequence")["IDF1"]
        .rename("uniform")
        .to_frame()
        .join(
            burst.set_index("sequence")["IDF1"].rename("burst"),
            how="outer",
            validate="one_to_one",
        )
    )
    pairs = pairs.reindex(sorted(set(rows["sequence"])))
    incomplete = pairs.isna().any(axis=1)
    differences = pairs.loc[~incomplete, "uniform"] - pairs.loc[~incomplete, "burst"]
    result = {
        "n_pairs": len(differences),
        "missing_videos": pairs.index[incomplete].to_list(),
        "by_video_pp": differences.to_dict(),
    }
    if pairs.empty or incomplete.any():
        return {
            **result,
            "status": "unavailable",
            "effect_pp": None,
            "ci95_pp": None,
            "p_one_sided_raw": 1.0,
        }
    effect, interval, p_value = bootstrap_mean(differences.to_numpy(), seed=seed)
    return {
        **result,
        "status": "ok",
        "effect_pp": effect,
        "ci95_pp": interval,
        "p_one_sided_raw": p_value,
    }


def bootstrap_mean(
    paired_differences: NDArray[np.float64],
    *,
    seed: int,
) -> tuple[float, tuple[float, float], float]:
    """Вернуть среднее, 95%-й интервал и одностороннее центрированное p."""
    if paired_differences.ndim != 1 or not len(paired_differences):
        raise ValueError("Ожидается непустой вектор парных разностей.")
    if not np.isfinite(paired_differences).all():
        raise ValueError("Парные разности должны быть конечными.")

    indices = resample_indices(len(paired_differences), seed)
    replicates = paired_differences[indices].mean(axis=1)
    point = float(paired_differences.mean())
    interval, p_value = interval_and_p(point, replicates)
    return point, interval, p_value
