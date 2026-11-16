"""Интерфейс командной строки эксперимента."""

import argparse
import sys
from pathlib import Path

from sort_gap_study.config import load_split
from sort_gap_study.dataset import audit_inputs
from sort_gap_study.evaluation import evaluate_sort
from sort_gap_study.ocsort import DEFAULT_OCSORT_CHECKOUT
from sort_gap_study.runner import TRACKING_METHODS, MaskSpec, RunInputs, run_sort


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
        help="Запустить трекеры на одном настроечном видео.",
    )
    _add_run_arguments(run)

    evaluate = commands.add_parser(
        "evaluate-sort",
        help="Рассчитать метрики TrackEval для готового запуска SORT.",
    )
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--trackeval", type=Path, default=Path("external/TrackEval"))

    summary = commands.add_parser(
        "summarize-pilot",
        help="Построить сводные таблицы и графики IDF1 и FP.",
    )
    summary.add_argument("--metrics", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
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
            run_sort(
                inputs,
                mask,
                args.output,
                method=args.method,
                ocsort_checkout=args.ocsort_checkout,
            )
        except (ImportError, OSError, TypeError, ValueError) as error:
            run.error(str(error))
        sys.stdout.write(f"Результаты SORT: {args.output}\n")
    elif args.command == "evaluate-sort":
        try:
            metrics = evaluate_sort(args.run, args.ground_truth, args.trackeval)
        except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
            evaluate.error(str(error))
        sys.stdout.write(f"Метрики SORT: {metrics}\n")
    elif args.command == "summarize-pilot":
        from sort_gap_study.summary import summarize_pilot  # noqa: PLC0415 — plotting dependencies

        try:
            summarize_pilot(args.metrics, args.output)
        except (OSError, TypeError, ValueError) as error:
            summary.error(str(error))
        sys.stdout.write(f"Таблицы и графики: {args.output}\n")


def _add_run_arguments(run: argparse.ArgumentParser) -> None:
    run.add_argument("--config", type=Path, default=Path("configs/protocol.toml"))
    run.add_argument("--seqmap", type=Path, required=True)
    run.add_argument("--sequence", required=True)
    run.add_argument("--source", choices=("gt", "regionlet"), required=True)
    run.add_argument(
        "--method",
        choices=(*TRACKING_METHODS, "all"),
        default="sort",
    )
    run.add_argument("--ocsort-checkout", type=Path, default=DEFAULT_OCSORT_CHECKOUT)
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
