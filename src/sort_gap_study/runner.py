"""Покадровый запуск SORT и запись артефактов одной последовательности."""

import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import groupby
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sort_gap_study.config import load_sort_kalman, load_sort_lifecycle, load_split
from sort_gap_study.dataset import (
    DETECTION_COLUMNS,
    read_gt_detections,
    read_regionlet_detections,
    read_seqmap,
)
from sort_gap_study.masks import WARMUP_FRAMES, make_burst_mask, make_mask
from sort_gap_study.sort import SortTracker, TrackEstimate


@dataclass(frozen=True)
class RunInputs:
    config: Path
    seqmap: Path
    sequence: str
    source: Literal["gt", "regionlet"]
    detections: Path


@dataclass(frozen=True)
class MaskSpec:
    kind: Literal["uniform", "random", "burst"]
    loss_fraction: float
    seed: int
    series_length: int

    def removed_frames(self, frames: range) -> tuple[int, ...]:
        if self.kind == "burst":
            mask = make_burst_mask(
                frames,
                self.loss_fraction,
                series_length=self.series_length,
                seed=self.seed,
            )
            if mask.status == "infeasible":
                raise ValueError(mask.reason)
            return mask.removed_frames
        return make_mask(frames, self.loss_fraction, self.kind, seed=self.seed)


def run_sort(inputs: RunInputs, mask: MaskSpec, output: Path) -> None:
    """Обработать настроечное видео и сохранить результаты в новом каталоге."""
    frames = _tune_frames(inputs)
    removed = mask.removed_frames(frames)
    reader = {"gt": read_gt_detections, "regionlet": read_regionlet_detections}
    detections = reader[inputs.source](inputs.detections, frames)
    tracker = SortTracker(
        load_sort_kalman(inputs.config),
        load_sort_lifecycle(inputs.config),
    )

    output.mkdir(parents=True, exist_ok=False)
    try:
        predictions_path = output / "predictions" / "data" / f"{inputs.sequence}.txt"
        predictions_path.parent.mkdir(parents=True)
        _write_frames(
            tracker,
            ((frame, detections[frame]) for frame in frames),
            set(removed),
            predictions_path,
            output / "events.jsonl",
        )
        shutil.copyfile(inputs.config, output / "protocol.toml")
        _write_json(output / "mask.json", _mask_record(mask, removed))
        _write_json(
            output / "manifest.json",
            {
                "method": "sort",
                "sequence": inputs.sequence,
                "source": inputs.source,
                "seqmap": str(inputs.seqmap.resolve()),
                "detections": str(inputs.detections.resolve()),
                "first_frame": frames.start,
                "last_frame": frames.stop - 1,
            },
        )
    except BaseException:
        shutil.rmtree(output)
        raise


def _tune_frames(inputs: RunInputs) -> range:
    if inputs.sequence not in load_split(inputs.config).tune:
        raise ValueError("Для запуска SORT требуется настроечная последовательность.")
    sequences = read_seqmap(inputs.seqmap)
    if inputs.sequence not in sequences:
        raise ValueError(f"Последовательность {inputs.sequence} отсутствует в seqmap.")
    return sequences[inputs.sequence]


def _write_frames(
    tracker: SortTracker,
    frame_detections: Iterable[tuple[int, NDArray[np.float64]]],
    removed: set[int],
    predictions_path: Path,
    events_path: Path,
) -> None:
    empty = np.empty((0, DETECTION_COLUMNS), dtype=np.float64)
    with (
        predictions_path.open("w", encoding="utf-8", newline="\n") as predictions_file,
        events_path.open("w", encoding="utf-8", newline="\n") as events_file,
    ):
        for frame, original in frame_detections:
            artificial_gap = frame in removed
            supplied = empty if artificial_gap else original
            estimates = tracker.step(frame, supplied, artificial_gap=artificial_gap)
            event = {
                "frame": frame,
                "artificial_gap": artificial_gap,
                "natural_empty": not artificial_gap and not len(original),
                "source_detection_count": len(original),
                "supplied_detection_count": len(supplied),
                **asdict(tracker.last_events),
                "estimates": [asdict(estimate) for estimate in estimates],
            }
            events_file.write(json.dumps(event, allow_nan=False) + "\n")
            predictions_file.writelines(_kitti_line(frame, item) for item in estimates)


def _kitti_line(frame: int, estimate: TrackEstimate) -> str:
    bbox = " ".join(map(str, estimate.bbox))
    # Поля 3D и углы не оцениваются для этого 2D-трекера.
    return (
        f"{frame} {estimate.track_id} Car -1 -1 -10 {bbox} "
        "-1 -1 -1 -1000 -1000 -1000 -10 1\n"
    )


def _mask_record(mask: MaskSpec, removed: tuple[int, ...]) -> dict[str, object]:
    runs = []
    for _, group in groupby(enumerate(removed), key=lambda item: item[1] - item[0]):
        consecutive = [frame for _, frame in group]
        runs.append((consecutive[0], len(consecutive)))
    return {
        "kind": mask.kind,
        "loss_fraction": mask.loss_fraction,
        "seed": mask.seed,
        "series_length": mask.series_length if mask.kind == "burst" else None,
        "warmup": WARMUP_FRAMES,
        "removed_frames": removed,
        "runs": runs,
    }


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
