import json

import pytest

from sort_gap_study.holm import holm_adjust, write_holm


def test_holm_uses_running_max_and_preserves_input_order():
    raw = [0.9, 0.03, 0.01, 0.031, 0.031, 0.2]
    assert holm_adjust(raw) == pytest.approx([0.9, 0.15, 0.06, 0.15, 0.15, 0.4])
    assert holm_adjust([0.8, 1.0]) == [1.0, 1.0]


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_holm_rejects_invalid_p_values(value):
    with pytest.raises(ValueError, match="отрезку"):
        holm_adjust([value])


def test_h1_and_h2_share_six_slots_including_unavailable_tests(tmp_path):
    h1 = tmp_path / "h1.json"
    h2 = tmp_path / "h2.json"
    output = tmp_path / "holm.json"
    h1.write_text(
        json.dumps(
            {
                "sources": {
                    "gt": {"status": "ok", "p_one_sided_raw": 0.02},
                    "regionlet": {"status": "ok", "p_one_sided_raw": 0.009},
                },
            },
        ),
        encoding="utf-8",
    )
    h2.write_text(
        json.dumps(
            {
                "sources": {
                    "gt": {
                        "length_difference": {
                            "status": "unavailable",
                            "p_one_sided_raw": 0.01,
                        },
                        "rfp_slack": {"status": "ok", "p_one_sided_raw": 0.013},
                    },
                    "regionlet": {
                        "length_difference": {"status": "ok", "p_one_sided_raw": 0.003},
                        "rfp_slack": {"status": "unavailable"},
                    },
                },
            },
        ),
        encoding="utf-8",
    )

    write_holm(h1, h2, output)
    result = json.loads(output.read_text(encoding="utf-8"))

    expected_family = [
        ("h1_idf1", "gt"),
        ("h1_idf1", "regionlet"),
        ("h2_length_difference", "gt"),
        ("h2_rfp_slack", "gt"),
        ("h2_length_difference", "regionlet"),
        ("h2_rfp_slack", "regionlet"),
    ]
    assert result["family_size"] == len(expected_family)
    assert [
        (test["test"], test["source"]) for test in result["tests"]
    ] == expected_family
    assert [test["p_one_sided_raw"] for test in result["tests"]] == [
        0.02,
        0.009,
        1,
        0.013,
        0.003,
        1,
    ]
    assert [test["p_holm"] for test in result["tests"]] == pytest.approx(
        [0.06, 0.045, 1, 0.052, 0.018, 1],
    )
