import csv
import json

import pytest

from sort_gap_study.bootstrap import BOOTSTRAP_RESAMPLES
from sort_gap_study.h2 import SERIES_LENGTHS, write_h2


def _run_h2(tmp_path, basic, extra):
    by_video = tmp_path / "by-video.csv"
    output = tmp_path / "h2.json"
    fields = [
        "sequence",
        "source",
        "method",
        "loss_fraction",
        "mask_kind",
        "series_length",
        "NGT",
        "IDF1",
        "FP",
    ]
    with by_video.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for index, (base_idf1, base_rfp, ngt) in enumerate(
            [(88, 9, 1000), (92, 11, 2000)],
        ):
            video = f"{index:04d}"
            writer.writerow(
                [
                    video,
                    "gt",
                    "sort",
                    0,
                    "none",
                    "",
                    ngt,
                    base_idf1,
                    base_rfp * ngt / 100,
                ],
            )
            for method, curves in (("sort", basic), ("sort-extra", extra)):
                for length, (idf1, rfp) in zip(
                    SERIES_LENGTHS,
                    curves[index],
                    strict=False,
                ):
                    writer.writerow(
                        [
                            video,
                            "gt",
                            method,
                            0.5,
                            "burst",
                            length,
                            ngt,
                            idf1,
                            rfp * ngt / 100,
                        ],
                    )
        writer.writerow(["0017", "gt", "sort", 0, "none", "", 0, 0, 21])
    write_h2(by_video, output, seed=0)
    return json.loads(output.read_text(encoding="utf-8"))["sources"]["gt"]


_PASS_LATER = [(90, 9), (90, 9), (90, 9), (90, 9)]
_FINITE_ONE = [(90, 9), (80, 9), *_PASS_LATER]


def test_h2_reselects_length_and_thresholds_in_each_paired_sample(tmp_path):
    basic = [_FINITE_ONE, _FINITE_ONE]
    extra = [
        [(90, 9), (88, 8), (90, 13), *_PASS_LATER[:3]],
        [(90, 11), (86, 11), (90, 13), *_PASS_LATER[:3]],
    ]
    result = _run_h2(tmp_path, basic, extra)

    assert result["videos"] == ["0000", "0001"]
    assert result["thresholds"] == {"idf1_floor": 85, "rfp_ceiling": 11}
    basic_boundary = result["boundaries"]["sort"]
    extra_boundary = result["boundaries"]["sort-extra"]
    assert basic_boundary["length_frames"] == 1
    assert [item["passes_thresholds"] for item in basic_boundary["curve"][:3]] == [
        True,
        False,
        True,
    ]
    assert extra_boundary["length_frames"] == SERIES_LENGTHS[1]
    assert extra_boundary["status"] == "finite"
    difference = result["length_difference"]
    assert difference["effect"] == 1
    assert difference["ci95"] == [0, 1]
    assert difference["p_one_sided_raw"] == 1 / (BOOTSTRAP_RESAMPLES + 1)
    slack = result["rfp_slack"]
    assert slack["effect"] == pytest.approx(1.5)
    assert slack["ci95"] == [1, 2]
    assert slack["interval_method"] == "simultaneous_boundary_envelope"
    assert slack["valid_replicates"] == BOOTSTRAP_RESAMPLES


def test_h2_reports_undefined_and_lower_bound_with_independent_rfp_test(tmp_path):
    basic = [[(80, 9), (90, 9), *_PASS_LATER]] * 2
    extra = [[(90, 9)] * len(SERIES_LENGTHS)] * 2
    result = _run_h2(tmp_path, basic, extra)

    assert result["boundaries"]["sort"]["status"] == "undefined"
    assert result["boundaries"]["sort"]["length_frames"] is None
    assert result["boundaries"]["sort-extra"]["status"] == "lower_bound"
    assert result["boundaries"]["sort-extra"]["length_frames"] == SERIES_LENGTHS[-1]
    assert result["length_difference"]["status"] == "unavailable"
    assert result["length_difference"]["p_one_sided_raw"] == 1
    assert result["rfp_slack"]["status"] == "ok"
    assert result["rfp_slack"]["selected_length_frames"] == SERIES_LENGTHS[-1]


def test_h2_unavailable_if_a_bootstrap_boundary_is_undefined(tmp_path):
    basic = [_FINITE_ONE, _FINITE_ONE]
    extra = [_FINITE_ONE, [(81, 9), (80, 9), *_PASS_LATER]]
    result = _run_h2(tmp_path, basic, extra)

    extra_boundary = result["boundaries"]["sort-extra"]
    assert extra_boundary["status"] == "finite"
    assert extra_boundary["bootstrap_status_counts"]["undefined"] > 0
    difference = result["length_difference"]
    assert difference["effect"] is not None
    assert difference["status"] == "unavailable"
    assert difference["ci95"] is None
    assert difference["p_one_sided_raw"] == 1
    assert result["rfp_slack"]["status"] == "ok"
    assert result["rfp_slack"]["valid_replicates"] == BOOTSTRAP_RESAMPLES


def test_h2_requires_complete_grid(tmp_path):
    truncated = [_FINITE_ONE[:3], _FINITE_ONE]
    result = _run_h2(tmp_path, truncated, [_FINITE_ONE, _FINITE_ONE])

    assert result["status"] == "incomplete_grid"
    assert result["missing"] == {
        "sort:5": ["0000"],
        "sort:8": ["0000"],
        "sort:10": ["0000"],
    }
    assert result["length_difference"]["p_one_sided_raw"] == 1
    assert result["rfp_slack"]["p_one_sided_raw"] == 1


def test_h2_rfp_band_contains_zero_at_selected_threshold(tmp_path):
    extra = [
        [(90, 8), (90, 8), (90, 9), (80, 9), *_PASS_LATER[:2]],
        [(90, 9), (90, 11), (90, 13), (80, 9), *_PASS_LATER[:2]],
    ]
    result = _run_h2(tmp_path, [_FINITE_ONE] * 2, extra)
    assert result["rfp_slack"]["selected_length_frames"] == SERIES_LENGTHS[2]
    assert result["rfp_slack"]["effect"] == 0
    assert result["rfp_slack"]["ci95"][0] == 0
    assert result["rfp_slack"]["ci95"][1] > 0
    assert result["rfp_slack"]["p_one_sided_raw"] == 1


def test_h2_rfp_checks_a_longer_boundary_than_the_point_estimate(tmp_path):
    extra = [
        [(90, 8), (90, 8), (90, 10), (80, 9), *_PASS_LATER[:2]],
        [(90, 8), (90, 9), (90, 14), (80, 9), *_PASS_LATER[:2]],
    ]
    result = _run_h2(tmp_path, [_FINITE_ONE] * 2, extra)
    slack = result["rfp_slack"]
    assert slack["selected_length_frames"] == SERIES_LENGTHS[1]
    assert slack["compatible_lengths_frames"] == list(SERIES_LENGTHS[1:3])
    assert slack["effect"] > 0
    assert slack["ci95"][0] == 0
