"""Сводные таблицы и графики IDF1 и FP по результатам пилота."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from sort_gap_study.runner import TRACKING_METHODS

_CONDITION = ["loss_fraction", "mask_kind", "series_length"]
_GROUP = ["source", "method", *_CONDITION]
_REQUIRED = {*_GROUP, "sequence", "seed", "status", "NGT", "IDF1", "FP"}
_MASK_LABELS = {
    "none": "Без потерь",
    "uniform": "Равномерная",
    "random": "Случайная",
    "burst": "Серии",
}
_METHOD_LABELS = {"sort": "SORT", "sort-extra": "SORT-extra", "ocsort": "OC-SORT"}
_METHOD_COLORS = {"sort": "#246795", "sort-extra": "#c66223", "ocsort": "#26836c"}


def summarize_pilot(metrics: Path, output: Path) -> None:
    """Сохранить таблицы и графики в новом каталоге."""
    runs = _read_metrics(metrics)
    by_video, summary = aggregate_metrics(runs)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for table, filename in ((by_video, "by-video.csv"), (summary, "summary.csv")):
            table.to_csv(output / filename, index=False, lineterminator="\n")
        _plot(
            summary,
            "IDF1",
            "IDF1 по условиям",
            "IDF1, %",
            output / "idf1.png",
        )
        _plot(
            summary,
            "FP",
            "Ложные срабатывания по условиям пилота",
            "FP на видео",
            output / "fp.png",
        )
    except BaseException:
        shutil.rmtree(output)
        raise


def _read_metrics(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path, dtype={"sequence": "string"})
    missing = _REQUIRED - set(runs.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {', '.join(sorted(missing))}.")
    if runs.empty:
        raise ValueError("Таблица метрик пуста.")
    if runs[["sequence", "source", "method"]].isna().any().any():
        raise ValueError("Каждая строка должна содержать видео, источник и метод.")
    if (
        not runs["source"].isin(("gt", "regionlet")).all()
        or not runs["method"]
        .isin(
            TRACKING_METHODS,
        )
        .all()
    ):
        raise ValueError("Таблица содержит неизвестный источник или метод.")

    _validate_values(runs)
    runs.loc[runs["loss_fraction"] == 0, "mask_kind"] = "none"
    if not runs["mask_kind"].isin(_MASK_LABELS).all():
        raise ValueError("Таблица содержит неизвестный тип маски.")
    runs["series_length"] = pd.to_numeric(runs["series_length"]).astype("Int64")
    burst = runs["mask_kind"].eq("burst")
    if not runs.loc[burst, "series_length"].ge(1).fillna(value=False).all():
        raise ValueError("Для серийной маски требуется положительная длина.")
    runs.loc[~burst, "series_length"] = pd.NA
    if runs.duplicated([*_GROUP, "sequence", "seed"]).any():
        raise ValueError("Таблица содержит повтор одной реализации маски.")
    return runs


def _validate_values(runs: pd.DataFrame) -> None:
    for key in ("loss_fraction", "NGT", "IDF1", "FP"):
        runs[key] = pd.to_numeric(runs[key])
    if not np.isfinite(runs[["loss_fraction", "NGT", "IDF1", "FP"]]).all().all():
        raise ValueError("Доля пропусков и значения метрик должны быть конечными.")
    if (
        not runs["loss_fraction"].between(0, 1).all()
        or not runs["IDF1"].between(0, 100).all()
        or (runs[["NGT", "FP"]] < 0).any().any()
    ):
        raise ValueError("Таблица содержит метрики или долю пропусков вне диапазона.")
    if not runs["status"].eq(np.where(runs["NGT"] == 0, "no_ground_truth", "ok")).all():
        raise ValueError("Статус прогона не соответствует числу оцениваемых Car.")
    if (runs.groupby("sequence")["NGT"].nunique() > 1).any():
        raise ValueError("Число оцениваемых Car должно совпадать для одного видео.")


def aggregate_metrics(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Усреднить реализации внутри видео, затем дать всем видео равный вес."""
    eligible = runs.loc[runs["NGT"] > 0]
    if eligible.empty:
        raise ValueError("Для всех видео число оцениваемых Car равно нулю.")
    by_video = eligible.groupby(
        [*_GROUP, "sequence"],
        dropna=False,
        as_index=False,
    ).agg(
        NGT=("NGT", "first"),
        n_masks=("FP", "size"),
        IDF1=("IDF1", "mean"),
        FP=("FP", "mean"),
    )
    summary = by_video.groupby(_GROUP, dropna=False, as_index=False).agg(
        n_videos=("sequence", "size"),
        n_runs=("n_masks", "sum"),
        IDF1=("IDF1", "mean"),
        FP=("FP", "mean"),
    )
    return by_video, summary


def _condition_label(fraction: float, kind: str, length: int) -> str:
    label = _MASK_LABELS[kind]
    if kind == "none":
        return label
    if kind == "burst":
        label = f"{label} по {length}"
    return f"{label}\n({fraction:.0%})"


def _plot(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    chart = summary.assign(series_length=summary["series_length"].fillna(0))
    order = tuple(_MASK_LABELS)
    conditions = sorted(
        set(chart[_CONDITION].itertuples(index=False, name=None)),
        key=lambda item: (item[0], order.index(item[1]), item[2]),
    )
    labels = [_condition_label(*condition) for condition in conditions]
    sources = [
        source for source in ("gt", "regionlet") if source in set(chart["source"])
    ]
    methods = [method for method in TRACKING_METHODS if method in set(chart["method"])]
    figure = Figure(
        figsize=(max(6.0, 1.25 * len(conditions)) * len(sources), 4.8),
        layout="constrained",
    )
    axes = figure.subplots(1, len(sources), sharey=True, squeeze=False)[0]
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.supylabel(ylabel)
    for axis, source in zip(axes, sources, strict=True):
        panel = chart.loc[chart["source"] == source].set_index([*_CONDITION, "method"])
        _plot_panel(axis, panel, metric, conditions, methods)
        axis.set_xticks(range(len(conditions)), labels)
        axis.set_title(
            "GT" if source == "gt" else "Regionlet",
            loc="left",
            fontweight="bold",
        )
    maximum = 100 if metric == "IDF1" else max(1.0, 1.08 * float(summary[metric].max()))
    axes[0].set_ylim(0, maximum)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncols=len(methods),
        frameon=False,
    )
    figure.savefig(output, dpi=170)


def _plot_panel(axis, panel, metric, conditions, methods):
    positions = np.arange(len(conditions))
    width = 0.75 / len(methods)
    for index, method in enumerate(methods):
        offsets = positions + (index - (len(methods) - 1) / 2) * width
        values = [
            panel[metric].get((*condition, method), np.nan) for condition in conditions
        ]
        axis.bar(
            offsets,
            values,
            width=width * 0.9,
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            zorder=3,
        )
    axis.grid(axis="y", color="#dce3e9")
    axis.spines[["top", "right"]].set_visible(False)
