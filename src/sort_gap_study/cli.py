"""Интерфейс командной строки эксперимента."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sort-gap-study",
        description="Исследование SORT при серийных пропусках наблюдений.",
    )
    parser.parse_args()
