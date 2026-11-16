import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sort_gap_study.ocsort import OCSORT_REVISION


def test_cli_help():
    executable = shutil.which("sort-gap-study")
    assert executable is not None

    result = subprocess.run(  # noqa: S603 — fixed entry point and arguments
        [executable, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    assert "usage: sort-gap-study" in result.stdout
    assert "--help" in result.stdout
    assert "audit-data" in result.stdout
    assert "run-sort" in result.stdout
    assert "evaluate-sort" in result.stdout


def test_run_sort_exports_original_frames_and_events(tmp_path):
    executable = shutil.which("sort-gap-study")
    assert executable is not None
    config = tmp_path / "protocol.toml"
    original_config = Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"
    config.write_text(
        original_config.read_text(encoding="utf-8").replace(
            "max_age = 30",
            "max_age = 2",
        ),
        encoding="utf-8",
    )
    frames = range(20, 36)
    naturally_empty = 23
    seqmap = tmp_path / "training.seqmap"
    seqmap.write_text("0000 empty 20 35\n", encoding="utf-8")
    detections = tmp_path / "0000.txt"
    detections.write_text(
        "".join(
            f"{frame} 99 Car 0 0 0 0 0 10 10 0 0 0 0 0 0 0\n"
            for frame in frames
            if frame != naturally_empty
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    command = [
        executable,
        "run-sort",
        "--config",
        str(config),
        "--seqmap",
        str(seqmap),
        "--sequence",
        "0000",
        "--source",
        "gt",
        "--detections",
        str(detections),
        "--output",
        str(output),
        "--mask",
        "burst",
        "--loss-fraction",
        "0.5",
    ]

    subprocess.run(command, capture_output=True, check=True)  # noqa: S603
    events_path = output / "events.jsonl"
    events_text = events_path.read_text(encoding="utf-8")
    events = {
        event["frame"]: event for event in map(json.loads, events_text.splitlines())
    }
    mask = json.loads((output / "mask.json").read_text(encoding="utf-8"))
    first_gap, second_gap, last_gap = mask["removed_frames"]

    assert list(events) == list(frames)
    assert mask["runs"] == [[first_gap, 3]]
    assert events[20]["created"] == [[1, 0]]
    assert events[21]["matched"] == [[1, 0]]
    assert events[naturally_empty]["natural_empty"]
    assert not events[naturally_empty]["artificial_gap"]
    for frame in (first_gap, second_gap, last_gap):
        event = events[frame]
        assert event["artificial_gap"]
        assert not event["natural_empty"]
        assert event["source_detection_count"] == 1
        assert event["supplied_detection_count"] == 0
    assert not events[first_gap]["estimates"][0]["observed"]
    assert events[last_gap]["removed"] == [1]
    assert events[last_gap]["estimates"] == []

    kitti = (output / "predictions" / "data" / "0000.txt").read_text(encoding="utf-8")
    rows = [row.split() for row in kitti.splitlines()]
    expected_frames = [frame for frame in frames for _ in events[frame]["estimates"]]
    assert [int(row[0]) for row in rows] == expected_frames
    assert {len(row) for row in rows} == {18}
    assert rows[0][:3] == ["22", "1", "Car"]
    assert tuple(map(float, rows[0][6:10])) == (0, 0, 10, 10)
    assert (output / "protocol.toml").read_bytes() == config.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [manifest["first_frame"], manifest["last_frame"]] == [20, 35]

    duplicate = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    assert duplicate.returncode != 0
    assert events_path.read_text(encoding="utf-8") == events_text


def test_run_all_uses_same_mask_and_records_provenance(tmp_path):
    project = Path(__file__).resolve().parents[1]
    checkout = project / "external" / "OC_SORT"
    if not checkout.is_dir():
        pytest.skip("OC-SORT checkout required. See README.md.")
    executable = shutil.which("sort-gap-study")
    assert executable is not None
    config = project / "configs" / "protocol.toml"
    seqmap = tmp_path / "training.seqmap"
    seqmap.write_text("0000 empty 20 35\n", encoding="utf-8")
    detections = tmp_path / "0000.txt"
    detections.write_text(
        "".join(
            f"{frame} 99 Car 0 0 0 0 0 10 10 0 0 0 0 0 0 0\n" for frame in range(20, 36)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"
    command = [
        executable,
        "run-sort",
        "--config",
        str(config),
        "--seqmap",
        str(seqmap),
        "--sequence",
        "0000",
        "--source",
        "gt",
        "--detections",
        str(detections),
        "--output",
        str(output),
        "--method",
        "all",
        "--ocsort-checkout",
        str(checkout),
        "--mask",
        "burst",
        "--loss-fraction",
        "0.5",
    ]

    subprocess.run(command, capture_output=True, check=True)  # noqa: S603
    manifest_path = output / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    methods = ["sort", "sort-extra", "ocsort"]
    mask_bytes = (output / "mask.json").read_bytes()
    mask = json.loads(mask_bytes)
    expected_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            "config": config,
            "seqmap": seqmap,
            "detections": detections,
        }.items()
    }

    assert manifest["method"] == "all"
    assert manifest["methods"] == methods
    assert manifest["ocsort_revision"] == OCSORT_REVISION
    assert manifest["input_sha256"] == expected_hashes
    assert manifest["mask_sha256"] == hashlib.sha256(mask_bytes).hexdigest()
    assert manifest["versions"]["python"]
    assert manifest["versions"]["numpy"]
    assert manifest["versions"]["scipy"]
    assert manifest["versions"]["filterpy"]
    if (project / ".git").exists():
        assert len(manifest["git"]["commit"]) in {40, 64}
        assert type(manifest["git"]["dirty"]) is bool

    for method in methods:
        run = output / method
        child = json.loads((run / "manifest.json").read_bytes())
        assert child["method"] == method
        for key in ("git", "versions", "input_sha256", "mask_sha256"):
            assert child[key] == manifest[key]
        assert (run / "mask.json").read_bytes() == mask_bytes
        assert (run / "protocol.toml").read_bytes() == config.read_bytes()
        events = [
            json.loads(line)
            for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [event["frame"] for event in events] == list(range(20, 36))
        assert [event["frame"] for event in events if event["artificial_gap"]] == mask[
            "removed_frames"
        ]
        assert (run / "predictions" / "data" / "0000.txt").stat().st_size > 0

    duplicate = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    assert duplicate.returncode != 0
    assert manifest_path.read_bytes() == manifest_bytes
