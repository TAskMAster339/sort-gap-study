import pandas as pd
import pytest
from matplotlib.figure import Figure

from sort_gap_study.summary import summarize_pilot

_HEADER = (
    "sequence,source,method,loss_fraction,mask_kind,series_length,seed,"
    "status,NGT,IDF1,FP\n"
)


def test_summary_weights_videos_equally_and_excludes_empty_gt(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics.csv"
    regionlet_fp = 125
    metrics.write_text(
        _HEADER
        + "0000,gt,sort,0.5,burst,3,0,ok,100,80,10\n"
        + "0000,gt,sort,0.5,burst,3,1,ok,100,100,30\n"
        + "0001,gt,sort,0.5,burst,3,0,ok,10,30,60\n"
        + f"0000,regionlet,sort,0.5,burst,3,0,ok,100,40,{regionlet_fp}\n"
        + "0017,gt,sort,0.5,burst,3,0,no_ground_truth,0,0,200\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary"

    savefig = Figure.savefig

    def check_fp_range(figure, path, **kwargs):
        if path.name == "fp.png":
            assert all(axis.get_ylim()[1] > regionlet_fp for axis in figure.axes)
        savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_fp_range)
    summarize_pilot(metrics, output)

    videos = pd.read_csv(output / "by-video.csv", dtype={"sequence": str})
    gt = videos[videos["source"] == "gt"].set_index("sequence")
    assert list(gt.index) == ["0000", "0001"]
    assert gt["n_masks"].tolist() == [2, 1]
    assert gt["IDF1"].tolist() == [90, 30]
    assert gt["FP"].tolist() == [20, 60]
    summary = pd.read_csv(output / "summary.csv").set_index("source")
    assert summary["n_videos"].to_dict() == {"gt": 2, "regionlet": 1}
    assert summary["n_runs"].to_dict() == {"gt": 3, "regionlet": 1}
    assert summary["IDF1"].to_dict() == {"gt": 60, "regionlet": 40}
    assert summary["FP"].to_dict() == {"gt": 40, "regionlet": regionlet_fp}
    assert (output / "idf1.png").stat().st_size > 0
    assert (output / "fp.png").stat().st_size > 0


def test_summary_rejects_duplicate_mask_run(tmp_path):
    metrics = tmp_path / "metrics.csv"
    row = "0000,gt,sort,0,none,,,ok,100,80,10\n"
    metrics.write_text(_HEADER + row * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="повтор одной реализации"):
        summarize_pilot(metrics, tmp_path / "summary")
