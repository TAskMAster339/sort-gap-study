"""Запуск зафиксированной итоговой сетки."""

import csv
import json
import shutil
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from sort_gap_study.config import EvaluationGrid, load_evaluation_grid, load_split
from sort_gap_study.dataset import read_seqmap
from sort_gap_study.evaluation import evaluate_sort
from sort_gap_study.freeze import ProtocolInputs, verify_protocol
from sort_gap_study.h1 import write_h1
from sort_gap_study.h2 import SERIES_LENGTHS, write_h2
from sort_gap_study.holm import write_holm
from sort_gap_study.masks import make_burst_mask
from sort_gap_study.runner import TRACKING_METHODS, MaskSpec, RunInputs, run_sort
from sort_gap_study.summary import summarize_pilot


@dataclass(frozen=True)
class FixedMask(MaskSpec):
    timeline: range
    removed: tuple[int, ...]

    def removed_frames(self, frames: range) -> tuple[int, ...]:
        if frames != self.timeline:
            raise ValueError("Диапазон кадров отличается от диапазона маски.")
        return self.removed


@dataclass(frozen=True)
class Condition:
    name: str
    mask: MaskSpec


def conditions(grid: EvaluationGrid) -> Iterator[Condition]:
    yield Condition("no-loss", MaskSpec("uniform", 0, 0, 3))
    for percent, lengths in sorted(grid.burst_lengths_by_percent.items()):
        for seed in grid.mask_seeds:
            for kind in ("uniform", "random"):
                yield Condition(
                    f"p{percent}-{kind}-s{seed}",
                    MaskSpec(kind, percent / 100, seed, 3),
                )
            for length in lengths:
                yield Condition(
                    f"p{percent}-burst-l{length}-s{seed}",
                    MaskSpec("burst", percent / 100, seed, length),
                )


def run_final(
    inputs: ProtocolInputs,
    manifest: Path,
    output: Path,
    *,
    workers: int,
) -> None:
    """Посчитать только те прогоны, которые разрешены фиксацией."""
    if workers < 1:
        raise ValueError("Число процессов должно быть положительным.")
    verify_protocol(inputs, manifest)
    snapshot, frozen = _snapshot(inputs, manifest, output)
    verify_protocol(snapshot, frozen)
    if (output / "results").is_dir():
        return
    split = load_split(snapshot.config)
    frames = read_seqmap(snapshot.seqmap)
    grid = load_evaluation_grid(snapshot.config)
    if grid.burst_lengths_by_percent.get(50) != SERIES_LENGTHS:
        raise ValueError("Сетка при 50% должна включать длины H2 по порядку.")
    planned = tuple(conditions(grid))
    tasks = [
        (sequence, frames[sequence], condition)
        for sequence in split.evaluation
        for condition in planned
    ]
    rows: list[dict[str, object]] = []
    infeasible = []
    run = partial(_run_condition, snapshot, output)
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        completed = zip(tasks, pool.map(run, tasks), strict=True)
        for index, (task, result) in enumerate(completed, start=1):
            if isinstance(result, str):
                infeasible.append(
                    {"sequence": task[0], "condition": task[2].name, "reason": result},
                )
            else:
                rows.extend(result)
            if index % 50 == 0 or index == len(tasks):
                sys.stdout.write(f"Выполнено условий: {index}/{len(tasks)}\n")
                sys.stdout.flush()
    finally:
        pool.shutdown(cancel_futures=True)
    verify_protocol(snapshot, frozen)
    _write_analysis(output, rows, infeasible, grid.bootstrap_seed)


def _snapshot(
    inputs: ProtocolInputs,
    manifest: Path,
    output: Path,
) -> tuple[ProtocolInputs, Path]:
    root = output / "inputs"
    frozen = root / "protocol-freeze.json"
    snapshot = replace(
        inputs,
        config=root / "protocol.toml",
        seqmap=root / "seqmap.txt",
        labels_dir=root / "gt",
        regionlet_dir=root / "regionlet",
    )
    if output.exists():
        if frozen.read_bytes() != manifest.read_bytes():
            raise ValueError("Каталог относится к другой фиксации протокола.")
        return snapshot, frozen
    output.mkdir(parents=True)
    try:
        snapshot.labels_dir.mkdir(parents=True)
        snapshot.regionlet_dir.mkdir()
        shutil.copyfile(manifest, frozen)
        shutil.copyfile(inputs.config, snapshot.config)
        shutil.copyfile(inputs.seqmap, snapshot.seqmap)
        split = load_split(snapshot.config)
        for sequence in (*split.tune, *split.evaluation):
            filename = f"{sequence}.txt"
            shutil.copyfile(
                inputs.labels_dir / filename,
                snapshot.labels_dir / filename,
            )
            shutil.copyfile(
                inputs.regionlet_dir / filename,
                snapshot.regionlet_dir / filename,
            )
    except BaseException:
        shutil.rmtree(output)
        raise
    return snapshot, frozen


def _run_condition(
    inputs: ProtocolInputs,
    output: Path,
    task: tuple[str, range, Condition],
) -> list[dict[str, object]] | str:
    sequence, frames, condition = task
    target = output / "runs" / sequence / condition.name
    if target.exists():
        return json.loads((target / "rows.json").read_text(encoding="utf-8"))

    spec = condition.mask
    if spec.kind == "burst":
        result = make_burst_mask(
            frames,
            spec.loss_fraction,
            series_length=spec.series_length,
            seed=spec.seed,
        )
        if result.status == "infeasible":
            return result.reason or "Недостаточно кадров для серий."
        removed = result.removed_frames
    else:
        removed = spec.removed_frames(frames)
    mask = FixedMask(
        spec.kind,
        spec.loss_fraction,
        spec.seed,
        spec.series_length,
        frames,
        removed,
    )
    gt = inputs.labels_dir / f"{sequence}.txt"
    regionlet = inputs.regionlet_dir / f"{sequence}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".condition-", dir=target.parent) as directory:
        pending = Path(directory) / condition.name
        rows = []
        for source, detections in (("gt", gt), ("regionlet", regionlet)):
            destination = pending / source
            run_sort(
                RunInputs(
                    inputs.config,
                    inputs.seqmap,
                    sequence,
                    source,
                    detections,
                    evaluation=True,
                ),
                mask,
                destination,
                method="all",
                ocsort_checkout=inputs.ocsort_checkout,
            )
            for method in TRACKING_METHODS:
                path = evaluate_sort(
                    destination / method,
                    gt,
                    inputs.trackeval_checkout,
                    evaluation=True,
                )
                with path.open(encoding="utf-8", newline="") as file:
                    (row,) = csv.DictReader(file)
                rows.append(
                    {
                        "condition": condition.name,
                        "method": method,
                        "loss_fraction": spec.loss_fraction,
                        "mask_kind": spec.kind if spec.loss_fraction else "none",
                        "series_length": spec.series_length
                        if spec.kind == "burst"
                        else None,
                        "seed": spec.seed if spec.loss_fraction else None,
                        **row,
                    },
                )
        (pending / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
        pending.rename(target)
    return rows


def _write_analysis(
    output: Path,
    rows: list[dict[str, object]],
    infeasible: list[dict[str, str]],
    seed: int,
) -> None:
    if not rows:
        raise ValueError("Нет результатов для итогового анализа.")
    with TemporaryDirectory(prefix=".results-", dir=output) as directory:
        root = Path(directory) / "results"
        root.mkdir()
        metrics = root / "metrics.csv"
        with metrics.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (root / "infeasible.json").write_text(
            json.dumps(infeasible, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = root / "summary"
        summarize_pilot(metrics, summary)
        h1, h2 = root / "h1.json", root / "h2.json"
        write_h1(summary / "by-video.csv", h1, seed=seed)
        write_h2(summary / "by-video.csv", h2, seed=seed)
        write_holm(h1, h2, root / "holm.json")
        root.rename(output / "results")
