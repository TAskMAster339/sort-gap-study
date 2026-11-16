import json
from pathlib import Path

import pytest

from sort_gap_study import freeze
from sort_gap_study.config import load_evaluation_grid, load_regionlet_threshold
from sort_gap_study.final_run import conditions, run_final
from sort_gap_study.freeze import ProtocolInputs, freeze_protocol

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "protocol.toml"


def test_final_grid_sizes_and_locked_threshold():
    grid = load_evaluation_grid(_CONFIG)
    planned = list(conditions(grid))
    expected_conditions = 481
    assert len(planned) == len({condition.name for condition in planned})
    assert len(planned) == expected_conditions
    assert grid.mask_seeds == tuple(range(30))
    assert grid.bootstrap_seed == 0
    assert load_regionlet_threshold(_CONFIG) == 0
    assert planned[0].mask.loss_fraction == 0


def test_final_run_requires_freeze_and_uses_only_evaluation_split(
    tmp_path,
    monkeypatch,
):
    ocsort = _ROOT / "external" / "OC_SORT"
    trackeval = _ROOT / "external" / "TrackEval"
    if not ocsort.is_dir() or not trackeval.is_dir():
        pytest.skip("OC-SORT and TrackEval checkouts required. See README.md.")
    config = tmp_path / "protocol.toml"
    settings = _CONFIG.read_text(encoding="utf-8").split("[sort.kalman]", 1)[1]
    settings = settings.split("[evaluation_grid]", 1)[0]
    config.write_text(
        '[split]\ntune = ["0000"]\nevaluation = ["0001"]\n[sort.kalman]'
        + settings
        + "[evaluation_grid]\nmask_seeds = [0]\nbootstrap_seed = 0\n"
        + 'burst_lengths_by_percent = { "50" = [1, 2, 3, 5, 8, 10] }\n',
        encoding="utf-8",
    )
    seqmap = tmp_path / "seqmap"
    seqmap.write_text("0000 empty 0 25\n0001 empty 0 25\n")
    labels, regionlet = tmp_path / "gt", tmp_path / "regionlet"
    for directory, score in ((labels, ""), (regionlet, " 0.5")):
        directory.mkdir()
        (directory / "0000.txt").touch()
        (directory / "0001.txt").write_text(
            "".join(
                f"{frame} 1 Car 0 0 0 0 0 20 40 0 0 0 0 0 0 0{score}\n"
                for frame in range(26)
            ),
        )
    monkeypatch.setattr(freeze, "git_state", lambda: {"commit": "test", "dirty": False})
    inputs = ProtocolInputs(config, seqmap, labels, regionlet, ocsort, trackeval)
    manifest, output = tmp_path / "freeze.json", tmp_path / "run"
    with pytest.raises(FileNotFoundError, match=r"freeze\.json"):
        run_final(inputs, manifest, output, workers=1)
    assert not output.exists()

    freeze_protocol(inputs, manifest)
    run_final(inputs, manifest, output, workers=1)
    assert [path.name for path in (output / "runs").iterdir()] == ["0001"]
    condition = output / "runs" / "0001" / "p50-burst-l3-s0"
    masks = {path.read_bytes() for path in condition.rglob("mask.json")}
    assert len(masks) == 1
    results = output / "results"
    expected_rows = 54
    assert len((results / "metrics.csv").read_text().splitlines()) == expected_rows + 1
    assert json.loads((results / "infeasible.json").read_text()) == []
    assert (results / "summary" / "idf1.png").is_file()
    assert {"h1.json", "h2.json", "holm.json"} <= {
        path.name for path in results.iterdir()
    }

    # Rebuild the aggregate using the completed condition directories.
    (results / "holm.json").rename(tmp_path / "holm-previous.json")
    results.rename(tmp_path / "previous-results")
    run_final(inputs, manifest, output, workers=1)
    assert (results / "holm.json").read_bytes() == (
        tmp_path / "holm-previous.json"
    ).read_bytes()
