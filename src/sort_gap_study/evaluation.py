"""Оценка прогнозов SORT закреплённой версией TrackEval."""

import csv
import json
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np

from sort_gap_study.config import load_split
from sort_gap_study.masks import WARMUP_FRAMES

TRACKEVAL_REVISION = "12c8791b303e0a0b50f753af204249e622d0281a"
_NUMPY_ALIASES = {"float": float, "int": int, "bool": bool}


def evaluate_sort(run: Path, ground_truth: Path, checkout: Path) -> Path:
    """Оценить существующий запуск и сохранить одну строку метрик."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    sequence = manifest.get("sequence")
    if sequence not in load_split(run / "protocol.toml").tune:
        raise ValueError("Для оценки SORT требуется настроечная последовательность.")
    frames = _frame_range(manifest)

    destination = run / "metrics.csv"
    with (
        TemporaryDirectory(prefix="sort-gap-eval-") as directory,
        _pinned_trackeval(checkout, Path(directory)) as package,
    ):
        root = Path(directory)
        gt_root = root / "gt"
        gt_path = gt_root / "label_02" / f"{sequence}.txt"
        predictions = run / "predictions" / "data" / f"{sequence}.txt"
        tracker_path = root / "trackers" / "sort" / "data" / f"{sequence}.txt"
        _stage_kitti(ground_truth, gt_path, frames)
        _stage_kitti(predictions, tracker_path, frames)
        (gt_root / "evaluate_tracking.seqmap.training").write_text(
            f"{sequence} empty 0 {len(frames)}\n",
            encoding="utf-8",
        )
        row = _evaluate(package, root, sequence)

    row = {
        "sequence": sequence,
        "source": manifest["source"],
        "trackeval_revision": TRACKEVAL_REVISION,
        **row,
    }
    with TemporaryDirectory(prefix=".metrics-", dir=run) as directory:
        pending = Path(directory) / destination.name
        with pending.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        destination.hardlink_to(pending)
    return destination


def _frame_range(manifest: dict[str, object]) -> range:
    first = manifest.get("first_frame")
    last = manifest.get("last_frame")
    if type(first) is not int or type(last) is not int or not 0 <= first <= last:
        raise ValueError("Манифест содержит неверный диапазон кадров.")
    return range(first, last + 1)


def _stage_kitti(source: Path, target: Path, frames: range) -> None:
    """Копировать строки KITTI на временную шкалу от нулевого кадра."""
    target.parent.mkdir(parents=True)
    with (
        source.open(encoding="utf-8") as input_file,
        target.open("w", encoding="utf-8") as output_file,
    ):
        for number, line in enumerate(input_file, start=1):
            fields = line.split()
            if not fields:
                continue
            frame, *record = fields
            frame_index = int(frame)
            if frame_index not in frames or not record:
                raise ValueError(f"{source}:{number}: неверная строка KITTI.")
            output_file.write(f"{frame_index - frames.start} {' '.join(record)}\n")


def _evaluate(package: str, root: Path, sequence: str) -> dict[str, object]:
    kitti = import_module(".datasets.kitti_2d_box", package)
    kitti.np = _TrackEvalNumpy()
    trackeval_error = import_module(".utils", package).TrackEvalException
    try:
        dataset = kitti.Kitti2DBox(
            {
                "GT_FOLDER": str(root / "gt"),
                "TRACKERS_FOLDER": str(root / "trackers"),
                "TRACKERS_TO_EVAL": ["sort"],
                "CLASSES_TO_EVAL": ["car"],
                "PRINT_CONFIG": False,
            },
        )
        raw = dataset.get_raw_seq_data("sort", sequence)
        warmup = min(WARMUP_FRAMES, raw["num_timesteps"])
        raw = {
            key: value[warmup:] if isinstance(value, list) else value
            for key, value in raw.items()
        }
        raw["num_timesteps"] -= warmup
        data = dataset.get_preprocessed_seq_data(raw, "car")
        return _metrics(package, data)
    except trackeval_error as error:
        raise ValueError(str(error)) from error


def _metrics(package: str, data: dict) -> dict[str, object]:
    identity_module = import_module(".metrics.identity", package)
    clear_module = import_module(".metrics.clear", package)
    hota_module = import_module(".metrics.hota", package)
    identity_module.np = hota_module.np = _TrackEvalNumpy()

    identity = identity_module.Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    clear = clear_module.CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
    hota = hota_module.HOTA().eval_sequence(data)
    gt_count = int(data["num_gt_dets"])
    fp = int(clear["CLR_FP"])
    return {
        "status": "ok" if gt_count else "no_ground_truth",
        "frames": data["num_timesteps"],
        "NGT": gt_count,
        "IDF1": 100 * float(identity["IDF1"]),
        "HOTA": 100 * float(np.mean(hota["HOTA"])),
        "DetA": 100 * float(np.mean(hota["DetA"])),
        "AssA": 100 * float(np.mean(hota["AssA"])),
        "FP": fp,
        "FN": int(clear["CLR_FN"]),
        "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "RFP": 100 * fp / gt_count if gt_count else None,
    }


class _TrackEvalNumpy:
    """Восстановить псевдонимы типов в модулях старой версии TrackEval."""

    def __getattr__(self, name: str) -> object:
        if name in _NUMPY_ALIASES:
            return _NUMPY_ALIASES[name]
        return getattr(np, name)


@contextmanager
def _pinned_trackeval(checkout: Path, root: Path) -> Iterator[str]:
    if not checkout.is_dir():
        raise FileNotFoundError(f"Копия репозитория TrackEval отсутствует: {checkout}.")
    archive = subprocess.run(  # noqa: S603 — fixed command and revision
        ["git", "-C", str(checkout), "archive", TRACKEVAL_REVISION, "trackeval"],  # noqa: S607
        check=False,
        capture_output=True,
    )
    if archive.returncode:
        details = archive.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"Ревизия TrackEval {TRACKEVAL_REVISION} недоступна. {details}",
        )
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as source:
        source.extractall(root, filter="data")

    package = "_trackeval_" + root.name.replace("-", "_")
    source_dir = root / "trackeval"
    try:
        # Load the KITTI and metric modules without importing unrelated datasets.
        for name, folder in (
            (package, source_dir),
            (package + ".datasets", source_dir / "datasets"),
            (package + ".metrics", source_dir / "metrics"),
        ):
            module = ModuleType(name)
            module.__path__ = [str(folder)]
            sys.modules[name] = module
        yield package
    finally:
        for module in tuple(sys.modules):
            if module == package or module.startswith(package + "."):
                sys.modules.pop(module, None)
