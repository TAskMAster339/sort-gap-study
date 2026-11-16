import json
import sys
from dataclasses import replace

import pytest

from sort_gap_study import freeze
from sort_gap_study.cli import main
from sort_gap_study.freeze import ProtocolInputs, freeze_protocol, verify_protocol
from sort_gap_study.provenance import sha256_file


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    config = tmp_path / "protocol.toml"
    config.write_text('[split]\ntune = ["0000"]\nevaluation = ["0001"]\n')
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text("0000 empty 0 15\n0001 empty 0 17\n")
    labels = tmp_path / "labels"
    regionlet = tmp_path / "regionlet"
    for folder in (labels, regionlet):
        folder.mkdir()
        for sequence in ("0000", "0001"):
            (folder / f"{sequence}.txt").touch()

    for dependency_file in ("uv.lock", "pyproject.toml"):
        (tmp_path / dependency_file).touch()
    state = {"commit": "initial", "dirty": False}
    versions = {"numpy": "2.4.0", "lap": None}
    monkeypatch.setattr(freeze, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(freeze, "git_state", lambda: state)
    monkeypatch.setattr(freeze, "environment_versions", lambda: versions)

    def extract(checkout, revision, subtree, destination):
        if not checkout.is_dir():
            raise FileNotFoundError(checkout)

    monkeypatch.setattr(freeze, "extract_revision", extract)
    inputs = ProtocolInputs(config, seqmap, labels, regionlet, tmp_path, tmp_path)
    return inputs, tmp_path / "runs" / "freeze.json", state, versions


def test_freeze_and_verify_commands(protocol, monkeypatch, capsys):
    argparse_error = 2
    inputs, manifest, _, _ = protocol
    args = [
        "--config",
        str(inputs.config),
        "--seqmap",
        str(inputs.seqmap),
        "--labels-dir",
        str(inputs.labels_dir),
        "--regionlet-dir",
        str(inputs.regionlet_dir),
        "--ocsort-checkout",
        str(inputs.ocsort_checkout),
        "--trackeval",
        str(inputs.trackeval_checkout),
        "--manifest",
        str(manifest),
    ]
    monkeypatch.setattr(sys, "argv", ["sort-gap-study", "freeze-protocol", *args])
    main()
    original = manifest.read_text()
    record = json.loads(original)
    assert record["git_commit"] == "initial"
    assert record["split"] == {"tune": ["0000"], "evaluation": ["0001"]}
    assert record["sha256"] == {
        "config": sha256_file(inputs.config),
        "seqmap": sha256_file(inputs.seqmap),
        "pyproject.toml": sha256_file(inputs.config.parent / "pyproject.toml"),
        "uv.lock": sha256_file(inputs.config.parent / "uv.lock"),
        "gt/0001.txt": sha256_file(inputs.labels_dir / "0001.txt"),
        "regionlet/0001.txt": sha256_file(inputs.regionlet_dir / "0001.txt"),
    }
    assert record["upstream"] == {
        "ocsort": freeze.OCSORT_REVISION,
        "trackeval": freeze.TRACKEVAL_REVISION,
    }
    assert record["versions"]["lap"] is None
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == argparse_error
    assert manifest.read_text() == original

    monkeypatch.setattr(sys, "argv", ["sort-gap-study", "verify-protocol", *args])
    main()
    relocated = inputs.labels_dir.rename(inputs.labels_dir.with_name("other-labels"))
    verify_protocol(replace(inputs, labels_dir=relocated), manifest)
    relocated.rename(inputs.labels_dir)
    (inputs.labels_dir / "0001.txt").write_text("0 1 Car\n")
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == argparse_error
    assert "sha256.gt/0001.txt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ("config", r"sha256\.config"),
        ("seqmap", r"sha256\.seqmap"),
        ("regionlet", r"sha256\.regionlet/0001\.txt"),
        ("lock", r"sha256\.uv\.lock"),
        ("version", r"versions\.numpy"),
        ("dirty", "Рабочая копия Git"),
        ("commit", "Ревизия Git отличается"),
    ],
)
def test_verify_rejects_drift(protocol, change, field):
    inputs, manifest, state, versions = protocol
    freeze_protocol(inputs, manifest)
    changes = {
        "config": (inputs.config, "\n# Changed\n"),
        "seqmap": (inputs.seqmap, "\n"),
        "regionlet": (inputs.regionlet_dir / "0001.txt", "0 1 Car\n"),
        "lock": (inputs.config.parent / "uv.lock", "\n"),
    }
    if change in changes:
        file, suffix = changes[change]
        file.write_text(file.read_text() + suffix)
    elif change == "version":
        versions["numpy"] = "2.5.0"
    elif change == "dirty":
        state["dirty"] = True
    else:
        state["commit"] = "changed"
    with pytest.raises(ValueError, match=field):
        verify_protocol(inputs, manifest)


def test_freeze_does_not_publish_when_input_is_missing(protocol):
    inputs, manifest, _, _ = protocol
    missing = inputs.regionlet_dir / "0001.txt"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match=r"0001\.txt"):
        freeze_protocol(inputs, manifest)
    assert not manifest.exists()


def test_verify_requires_pinned_upstream_checkout(protocol):
    inputs, manifest, _, _ = protocol
    freeze_protocol(inputs, manifest)
    absent = inputs.trackeval_checkout / "absent"
    with pytest.raises(FileNotFoundError, match="absent"):
        verify_protocol(replace(inputs, trackeval_checkout=absent), manifest)
