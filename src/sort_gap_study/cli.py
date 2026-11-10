"""Интерфейс командной строки эксперимента."""

import argparse
import sys
from pathlib import Path

from sort_gap_study.config import load_split
from sort_gap_study.dataset import audit_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sort-gap-study",
        description="Исследование SORT при серийных пропусках наблюдений.",
    )
    commands = parser.add_subparsers(dest="command")
    audit = commands.add_parser("audit-data", help="Проверить входные файлы KITTI.")
    audit.add_argument("--config", type=Path, default=Path("configs/protocol.toml"))
    audit.add_argument("--seqmap", type=Path, required=True)
    audit.add_argument("--labels-dir", type=Path, required=True)
    audit.add_argument("--regionlet-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "audit-data":
        try:
            split = load_split(args.config)
            audit_inputs(split, args.seqmap, args.labels_dir, args.regionlet_dir)
        except (OSError, TypeError, ValueError) as error:
            audit.error(str(error))
        total = len(split.tune) + len(split.evaluation)
        sys.stdout.write(f"Видео: {total}. Файлы GT и Regionlet: {2 * total}.\n")
