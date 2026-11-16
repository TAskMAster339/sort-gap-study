"""Дискретная граница допустимой длины серии и запас RFP для H2."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sort_gap_study.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    interval_and_p,
    resample_indices,
)

SERIES_LENGTHS = (1, 2, 3, 5, 8, 10)
_LENGTHS = np.asarray(SERIES_LENGTHS)
_METHODS = ("sort", "sort-extra")
_LOSS_FRACTION = 0.5
_IDF1_MARGIN = 5
_RFP_MARGIN = 1
_SCORE_COLUMNS = ["IDF1", "RFP"]
_REQUIRED = {
    "sequence",
    "source",
    "method",
    "loss_fraction",
    "mask_kind",
    "series_length",
    "NGT",
    "IDF1",
    "FP",
}


def write_h2(by_video: Path, output: Path, *, seed: int) -> None:
    table = pd.read_csv(by_video, dtype={"sequence": "string"})
    missing = _REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {', '.join(sorted(missing))}.")
    if table["sequence"].isna().any():
        raise ValueError("Таблица содержит строки без номера видео.")
    if seed < 0:
        raise ValueError("Seed должен быть неотрицательным.")

    eligible = table.loc[table["method"].isin(_METHODS) & (table["NGT"] > 0)].copy()
    eligible["RFP"] = 100 * eligible["FP"] / eligible["NGT"]
    result = {
        "loss_fraction": _LOSS_FRACTION,
        "series_lengths": SERIES_LENGTHS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": seed,
        "sources": {
            source: _analyze_source(eligible.loc[eligible["source"] == source], seed)
            for source in ("gt", "regionlet")
        },
    }
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(content)


def _analyze_source(rows: pd.DataFrame, seed: int) -> dict[str, object]:
    videos = sorted(set(rows["sequence"]))
    baseline = rows.loc[(rows["method"] == "sort") & (rows["loss_fraction"] == 0)]
    bursts = rows.loc[
        (rows["mask_kind"] == "burst") & (rows["loss_fraction"] == _LOSS_FRACTION)
    ]
    conditions = {"baseline": baseline}
    conditions.update(
        {
            f"{method}:{length}": bursts.loc[
                (bursts["method"] == method) & (bursts["series_length"] == length)
            ]
            for method in _METHODS
            for length in SERIES_LENGTHS
        },
    )
    tables = {
        condition: frame.set_index("sequence")[_SCORE_COLUMNS].reindex(videos)
        for condition, frame in conditions.items()
    }
    incomplete = {
        condition: table.index[table.isna().any(axis=1)].to_list()
        for condition, table in tables.items()
        if table.isna().any().any()
    }
    if not videos or incomplete:
        return {
            "status": "incomplete_grid" if videos else "no_evaluable_videos",
            "videos": videos,
            "missing": incomplete,
            "length_difference": _inference(None, np.empty(0)),
            "rfp_slack": _inference(None, np.empty(0)),
        }

    baseline_values = tables.pop("baseline").to_numpy(dtype=np.float64)
    curves = np.stack(
        [table.to_numpy(dtype=np.float64) for table in tables.values()],
        axis=1,
    )
    curves = curves.reshape(
        len(videos),
        len(_METHODS),
        len(SERIES_LENGTHS),
        len(_SCORE_COLUMNS),
    )
    if not np.isfinite(baseline_values).all() or not np.isfinite(curves).all():
        raise ValueError("Средние IDF1 и RFP должны быть конечными.")
    return {
        "status": "ok",
        "videos": videos,
        **_estimate(baseline_values, curves, seed),
    }


def _passes(
    baseline: NDArray[np.float64],
    curves: NDArray[np.float64],
) -> NDArray[np.bool_]:
    idf1_floor = baseline[..., 0, None, None] - _IDF1_MARGIN
    rfp_ceiling = baseline[..., 1, None, None] + _RFP_MARGIN
    return (curves[..., 0] >= idf1_floor) & (curves[..., 1] <= rfp_ceiling)


def _estimate(
    baseline: NDArray[np.float64],
    curves: NDArray[np.float64],
    seed: int,
) -> dict[str, object]:
    point = _BoundaryScores(baseline.mean(axis=0), curves.mean(axis=0))
    indices = resample_indices(len(baseline), seed)
    boot = _BoundaryScores(baseline[indices].mean(axis=1), curves[indices].mean(axis=1))

    return {
        "thresholds": {
            "idf1_floor": float(point.baseline[0] - _IDF1_MARGIN),
            "rfp_ceiling": float(point.baseline[1] + _RFP_MARGIN),
        },
        "boundaries": {
            method: _boundary(int(count), curve, allowed, boot_counts)
            for method, count, curve, allowed, boot_counts in zip(
                _METHODS,
                point.counts,
                point.curves,
                point.passes,
                boot.counts.T,
                strict=True,
            )
        },
        **_compare(point, boot),
    }


def _boundary(count, curve, passes, bootstrap_counts) -> dict[str, object]:
    return {
        "status": (
            "undefined"
            if count == 0
            else "lower_bound"
            if count == len(_LENGTHS)
            else "finite"
        ),
        "length_frames": SERIES_LENGTHS[count - 1] if count else None,
        "curve": [
            {
                "length_frames": length,
                "IDF1": idf1,
                "RFP": rfp,
                "passes_thresholds": bool(passes[index]),
            }
            for index, (length, (idf1, rfp)) in enumerate(
                zip(SERIES_LENGTHS, curve.tolist(), strict=True),
            )
        ],
        "bootstrap_status_counts": {
            "undefined": int(np.count_nonzero(bootstrap_counts == 0)),
            "lower_bound": int(np.count_nonzero(bootstrap_counts == len(_LENGTHS))),
            "finite": int(
                np.count_nonzero(
                    (bootstrap_counts > 0) & (bootstrap_counts < len(_LENGTHS)),
                ),
            ),
        },
    }


@dataclass(frozen=True)
class _BoundaryScores:
    baseline: NDArray[np.float64]
    curves: NDArray[np.float64]

    @property
    def passes(self) -> NDArray[np.bool_]:
        return _passes(self.baseline, self.curves)

    @property
    def counts(self) -> NDArray[np.int64]:
        return np.logical_and.accumulate(self.passes, axis=-1).sum(axis=-1)


def _compare(point: _BoundaryScores, boot: _BoundaryScores) -> dict[str, object]:
    point_counts = point.counts
    counts = boot.counts
    length_difference = None
    if ((point_counts > 0) & (point_counts < len(_LENGTHS))).all():
        point_lengths = _LENGTHS[point_counts - 1]
        length_difference = int(point_lengths[1] - point_lengths[0])
    both_finite = ((counts > 0) & (counts < len(_LENGTHS))).all(axis=1)
    lengths = _LENGTHS[counts[both_finite] - 1]

    selected = int(point_counts[1])
    point_slack = (
        float(point.baseline[1] + _RFP_MARGIN - point.curves[1, selected - 1, 1])
        if selected
        else None
    )
    valid = counts[:, 1] > 0
    slack = (
        boot.baseline[valid, 1]
        + _RFP_MARGIN
        - boot.curves[valid, 1, counts[valid, 1] - 1, 1]
    )
    return {
        "length_difference": _inference(
            length_difference,
            lengths[:, 1] - lengths[:, 0],
        ),
        "rfp_slack": {
            "selected_length_frames": SERIES_LENGTHS[selected - 1]
            if selected
            else None,
            **_inference(point_slack, slack),
        },
    }


def _inference(point: float | None, replicates: NDArray) -> dict[str, object]:
    available = point is not None and len(replicates) == BOOTSTRAP_RESAMPLES
    interval, p_value = interval_and_p(point, replicates) if available else (None, 1.0)
    return {
        "status": "ok" if available else "unavailable",
        "effect": point,
        "ci95": interval,
        "p_one_sided_raw": p_value,
        "valid_replicates": len(replicates),
    }
