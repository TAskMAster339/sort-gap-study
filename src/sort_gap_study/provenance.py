"""Версии среды, состояние Git и контрольные суммы входов."""

import hashlib
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version


def environment_versions() -> dict[str, str | None]:
    """Записать версии библиотек, используемых трекерами."""
    versions: dict[str, str | None] = {"python": python_version()}
    for name in ("sort-gap-study", "numpy", "scipy", "filterpy", "lap"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def git_state() -> dict[str, str | bool | None]:
    """Определить коммит исходников, из которых запущен пакет."""
    project = Path(__file__).resolve().parents[2]
    if not (project / ".git").exists():
        return {"commit": None, "dirty": None}
    return {
        "commit": _git(project, "rev-parse", "HEAD").strip(),
        "dirty": bool(_git(project, "status", "--porcelain")),
    }


def _git(project: Path, *arguments: str) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "--no-optional-locks", "-C", str(project), *arguments],  # noqa: S607
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return result.stdout


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
