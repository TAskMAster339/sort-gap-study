"""Аудит входных файлов KITTI Tracking."""

from pathlib import Path

from sort_gap_study.config import SequenceSplit

SEQMAP_FIELDS = 4


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


def _audit_frames(path: Path, frames: range) -> None:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split(maxsplit=1)
            if not fields:
                continue
            frame = _read_frame(fields[0], path, line_number)
            if frame not in frames:
                message = (
                    f"{path}:{line_number}: кадр {frame} вне диапазона "
                    f"{frames.start}..{frames.stop - 1}."
                )
                raise ValueError(message)


def _read_frame(value: str, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        message = f"{path}:{line_number}: неверный номер кадра {value!r}."
        raise ValueError(message) from error
