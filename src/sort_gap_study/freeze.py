"""Фиксация и входная проверка итогового эксперимента."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sort_gap_study.config import load_split
from sort_gap_study.dataset import audit_inputs
from sort_gap_study.evaluation import TRACKEVAL_REVISION, TRACKEVAL_SUBTREE
from sort_gap_study.ocsort import OCSORT_REVISION, OCSORT_SUBTREE
from sort_gap_study.provenance import (
    PROJECT_ROOT,
    environment_versions,
    git_state,
    sha256_file,
)
from sort_gap_study.upstream import extract_revision

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProtocolInputs:
    config: Path
    seqmap: Path
    labels_dir: Path
    regionlet_dir: Path
    ocsort_checkout: Path
    trackeval_checkout: Path


def freeze_protocol(inputs: ProtocolInputs, manifest: Path) -> None:
    """Записать манифест после аудита, сохранив ранее созданные фиксации."""
    state = _protocol_state(inputs)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".freeze-", dir=manifest.parent) as directory:
        temporary = Path(directory) / "protocol.json"
        temporary.write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest.hardlink_to(temporary)


def verify_protocol(inputs: ProtocolInputs, manifest: Path) -> None:
    """Проверить соответствие зафиксированному состоянию."""
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(frozen, dict) or frozen.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest}: неизвестная схема фиксации протокола.")

    commit = frozen.get("git_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError(f"{manifest}: отсутствует ревизия Git.")
    current = _protocol_state(inputs, expected_commit=commit)
    changed = _changed_fields(frozen, current)
    if changed:
        raise ValueError(
            "Изменены поля протокола: " + ", ".join(changed) + ".",
        )


def _changed_fields(
    frozen: dict[str, object],
    current: dict[str, object],
    prefix: str = "",
) -> list[str]:
    changed = []
    for key in sorted(frozen.keys() | current.keys()):
        old, new = frozen.get(key), current.get(key)
        field = prefix + key
        if isinstance(old, dict) and isinstance(new, dict):
            changed.extend(_changed_fields(old, new, field + "."))
        elif key not in frozen or key not in current or old != new:
            changed.append(field)
    return changed


def _clean_commit() -> str:
    code = git_state()
    commit = code["commit"]
    if not isinstance(commit, str) or not commit:
        raise ValueError("Для фиксации протокола требуется исходный Git-репозиторий.")
    if code["dirty"]:
        raise ValueError("Рабочая копия Git содержит изменения.")
    return commit


def _protocol_state(
    inputs: ProtocolInputs,
    *,
    expected_commit: str | None = None,
) -> dict[str, object]:
    commit = _clean_commit()
    if expected_commit is not None and expected_commit != commit:
        raise ValueError("Ревизия Git отличается от зафиксированной.")
    split = load_split(inputs.config)
    audit_inputs(split, inputs.seqmap, inputs.labels_dir, inputs.regionlet_dir)
    files = {
        "config": inputs.config,
        "seqmap": inputs.seqmap,
        "pyproject.toml": PROJECT_ROOT / "pyproject.toml",
        "uv.lock": PROJECT_ROOT / "uv.lock",
    }
    for sequence in split.evaluation:
        files[f"gt/{sequence}.txt"] = inputs.labels_dir / f"{sequence}.txt"
        files[f"regionlet/{sequence}.txt"] = inputs.regionlet_dir / f"{sequence}.txt"

    checksums = {name: sha256_file(path) for name, path in files.items()}
    with TemporaryDirectory(prefix="sort-gap-upstream-") as directory:
        root = Path(directory)
        extract_revision(inputs.ocsort_checkout, OCSORT_REVISION, OCSORT_SUBTREE, root)
        extract_revision(
            inputs.trackeval_checkout,
            TRACKEVAL_REVISION,
            TRACKEVAL_SUBTREE,
            root,
        )
    if _clean_commit() != commit:
        raise ValueError("Ревизия Git изменилась во время проверки протокола.")
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "split": {part: list(sequences) for part, sequences in asdict(split).items()},
        "versions": environment_versions(),
        "upstream": {"ocsort": OCSORT_REVISION, "trackeval": TRACKEVAL_REVISION},
        "sha256": checksums,
    }
