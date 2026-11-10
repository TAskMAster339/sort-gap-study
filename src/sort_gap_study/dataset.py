"""Чтение и аудит входных файлов KITTI Tracking."""

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sort_gap_study.config import SequenceSplit

SEQMAP_FIELDS = 4
GT_FIELDS = 17
REGIONLET_FIELDS = 18
DETECTION_COLUMNS = 5


def read_seqmap(path: Path) -> dict[str, range]:
    """Прочитать исходные диапазоны кадров из seqmap KITTI."""
    sequences = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != SEQMAP_FIELDS:
                message = f"{path}:{line_number}: ожидаются четыре поля seqmap."
                raise ValueError(message)
            sequence, _, first_field, last_field = fields
            first = _read_frame(first_field, path, line_number)
            last = _read_frame(last_field, path, line_number)
            if not 0 <= first <= last:
                message = f"{path}:{line_number}: неверный диапазон {first}..{last}."
                raise ValueError(message)
            if sequence in sequences:
                message = f"{path}:{line_number}: повтор последовательности {sequence}."
                raise ValueError(message)
            sequences[sequence] = range(first, last + 1)
    return sequences


def audit_inputs(
    split: SequenceSplit,
    seqmap: Path,
    labels_dir: Path,
    regionlet_dir: Path,
) -> None:
    """Проверить наличие GT и Regionlet и диапазоны кадров обоих входов."""
    ranges = read_seqmap(seqmap)
    for sequence in (*split.tune, *split.evaluation):
        if sequence not in ranges:
            message = f"{seqmap}: отсутствует последовательность {sequence}."
            raise ValueError(message)
        for directory in (labels_dir, regionlet_dir):
            _audit_frames(directory / f"{sequence}.txt", ranges[sequence])


def read_gt_detections(path: Path, frames: range) -> dict[int, NDArray[np.float64]]:
    """Вернуть наблюдения Car по кадрам. Для оценки используется исходный файл GT."""
    detections: dict[int, list[list[float]]] = {frame: [] for frame in frames}
    for frame, fields, line_number in _frame_records(path, frames):
        if len(fields) < GT_FIELDS:
            message = f"{path}:{line_number}: неполная строка GT."
            raise ValueError(message)
        if fields[2] != "Car":
            continue
        try:
            track_id = int(fields[1])
            x1, y1, x2, y2 = map(float, fields[6:10])
        except ValueError as error:
            message = f"{path}:{line_number}: неверный ID или координаты GT."
            raise ValueError(message) from error
        if track_id >= 0 and _valid_box(x1, y1, x2, y2):
            detections[frame].append([x1, y1, x2, y2, 1.0])

    return _detection_arrays(detections)


def read_regionlet_detections(
    path: Path,
    frames: range,
    *,
    score_threshold: float = 0.0,
) -> dict[int, NDArray[np.float64]]:
    """Вернуть наблюдения Car после фильтрации по исходной уверенности."""
    if not math.isfinite(score_threshold):
        message = "Порог уверенности Regionlet должен быть конечным числом."
        raise ValueError(message)

    detections: dict[int, list[list[float]]] = {frame: [] for frame in frames}
    for frame, fields, line_number in _frame_records(path, frames):
        if len(fields) < REGIONLET_FIELDS:
            message = f"{path}:{line_number}: неполная строка Regionlet."
            raise ValueError(message)
        if fields[2] != "Car":
            continue
        try:
            x1, y1, x2, y2 = map(float, fields[6:10])
            score = float(fields[17])
        except ValueError as error:
            message = f"{path}:{line_number}: неверные координаты или score Regionlet."
            raise ValueError(message) from error
        if (
            math.isfinite(score)
            and score >= score_threshold
            and _valid_box(x1, y1, x2, y2)
        ):
            detections[frame].append([x1, y1, x2, y2, 1.0])

    return _detection_arrays(detections)


def _detection_arrays(
    detections: dict[int, list[list[float]]],
) -> dict[int, NDArray[np.float64]]:
    return {
        frame: np.asarray(rows, dtype=np.float64).reshape(-1, DETECTION_COLUMNS)
        for frame, rows in detections.items()
    }


def _valid_box(x1: float, y1: float, x2: float, y2: float) -> bool:
    return bool(np.isfinite((x1, y1, x2, y2)).all()) and x1 < x2 and y1 < y2


def _audit_frames(path: Path, frames: range) -> None:
    for _ in _frame_records(path, frames):
        pass


def _frame_records(
    path: Path,
    frames: range,
) -> Iterator[tuple[int, list[str], int]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if not fields:
                continue
            frame = _read_frame(fields[0], path, line_number)
            if frame not in frames:
                message = (
                    f"{path}:{line_number}: кадр {frame} вне диапазона "
                    f"{frames.start}..{frames.stop - 1}."
                )
                raise ValueError(message)
            yield frame, fields, line_number


def _read_frame(value: str, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        message = f"{path}:{line_number}: неверный номер кадра {value!r}."
        raise ValueError(message) from error
