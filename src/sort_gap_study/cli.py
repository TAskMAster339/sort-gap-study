"""Интерфейс командной строки эксперимента."""

import argparse
import sys
from pathlib import Path

from sort_gap_study.config import load_split
from sort_gap_study.dataset import audit_inputs
from sort_gap_study.evaluation import evaluate_sort
from sort_gap_study.runner import MaskSpec, RunInputs, run_sort


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

    run = commands.add_parser(
        "run-sort",
        help="Запустить SORT на одном настроечном видео.",
    )
    run.add_argument("--config", type=Path, default=Path("configs/protocol.toml"))
    run.add_argument("--seqmap", type=Path, required=True)
    run.add_argument("--sequence", required=True)
    run.add_argument("--source", choices=("gt", "regionlet"), required=True)
    run.add_argument("--method", choices=("sort", "sort-extra"), default="sort")
    run.add_argument("--detections", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--mask",
        choices=("uniform", "random", "burst"),
        default="uniform",
    )
    run.add_argument("--loss-fraction", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--series-length", type=int, default=3)

    evaluate = commands.add_parser(
        "evaluate-sort",
        help="Рассчитать метрики TrackEval для готового запуска SORT.",
    )
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--trackeval", type=Path, default=Path("external/TrackEval"))
    args = parser.parse_args()

    if args.command == "audit-data":
        try:
            split = load_split(args.config)
            audit_inputs(split, args.seqmap, args.labels_dir, args.regionlet_dir)
        except (OSError, TypeError, ValueError) as error:
            audit.error(str(error))
        total = len(split.tune) + len(split.evaluation)
        sys.stdout.write(f"Видео: {total}. Файлы GT и Regionlet: {2 * total}.\n")
    elif args.command == "run-sort":
        inputs = RunInputs(
            args.config,
            args.seqmap,
            args.sequence,
            args.source,
            args.detections,
        )
        mask = MaskSpec(args.mask, args.loss_fraction, args.seed, args.series_length)
        try:
            run_sort(inputs, mask, args.output, method=args.method)
        except (OSError, TypeError, ValueError) as error:
            run.error(str(error))
        sys.stdout.write(f"Результаты SORT: {args.output}\n")
    elif args.command == "evaluate-sort":
        try:
            metrics = evaluate_sort(args.run, args.ground_truth, args.trackeval)
        except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
            evaluate.error(str(error))
        sys.stdout.write(f"Метрики SORT: {metrics}\n")
