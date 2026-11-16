"""Извлечение закреплённых исходников из локального Git-репозитория."""

import subprocess
import tarfile
from io import BytesIO
from pathlib import Path


def extract_revision(
    checkout: Path,
    revision: str,
    subtree: str,
    destination: Path,
) -> None:
    if not checkout.is_dir():
        raise FileNotFoundError(f"Репозиторий отсутствует: {checkout}.")
    archive = subprocess.run(  # noqa: S603
        ["git", "-C", str(checkout), "archive", revision, subtree],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if archive.returncode:
        details = archive.stderr.decode(errors="replace").strip()
        raise ValueError(f"Ревизия {revision} недоступна в {checkout}. {details}")
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as source:
        source.extractall(destination, filter="data")
