import json

import numpy as np
import pytest

from sort_gap_study.h1 import BOOTSTRAP_RESAMPLES, bootstrap_mean, write_h1

_HEADER = "sequence,source,method,loss_fraction,mask_kind,series_length,NGT,IDF1\n"


def test_bootstrap_mean_uses_centered_positive_tail():
    values = np.array([-1.0, 3.0])
    effect, interval, p_value = bootstrap_mean(values, seed=42)
    draws = np.random.default_rng(42).integers(2, size=(BOOTSTRAP_RESAMPLES, 2))
    largest_count = sum(int(left == right == 1) for left, right in draws)

    assert effect == 1
    assert interval == (-1, 3)
    assert p_value == (1 + largest_count) / (BOOTSTRAP_RESAMPLES + 1)
    assert bootstrap_mean(-values, seed=42)[2] > p_value


def test_h1_pairs_by_video_and_uses_only_the_two_sort_conditions(tmp_path):
    source = tmp_path / "by-video.csv"
    source.write_text(
        _HEADER
        + "0000,gt,sort,0.5,uniform,,10,70\n"
        + "0001,gt,sort,0.5,burst,3,100,30\n"
        + "0001,gt,sort,0.5,uniform,,100,50\n"
        + "0000,gt,sort,0.5,burst,3,10,60\n"
        + "0000,gt,sort,0.5,burst,10,10,0\n"
        + "0000,gt,ocsort,0.5,uniform,,10,0\n"
        + "0017,gt,sort,0.5,uniform,,0,0\n"
        + "0000,regionlet,sort,0.5,uniform,,10,60\n"
        + "0000,regionlet,sort,0.5,burst,3,10,70\n"
        + "0001,regionlet,sort,0.5,uniform,,100,45\n",
        encoding="utf-8",
    )
    output = tmp_path / "h1.json"
    write_h1(source, output, seed=0)
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    gt = result["sources"]["gt"]
    assert gt["status"] == "ok"
    assert gt["by_video_pp"] == {"0000": 10, "0001": 20}
    assert gt["effect_pp"] == pytest.approx(15)
    assert gt["ci95_pp"] == [10, 20]
    assert gt["p_one_sided_raw"] == 1 / (BOOTSTRAP_RESAMPLES + 1)
    regionlet = result["sources"]["regionlet"]
    assert regionlet["status"] == "unavailable"
    assert regionlet["missing_videos"] == ["0001"]
    assert regionlet["by_video_pp"] == {"0000": -10}
    assert regionlet["effect_pp"] is None
    assert regionlet["p_one_sided_raw"] == 1
