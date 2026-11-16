import csv
import json
from pathlib import Path

import pytest

from sort_gap_study.evaluation import TRACKEVAL_REVISION, evaluate_sort


@pytest.fixture
def trackeval_checkout():
    path = Path(__file__).resolve().parents[1] / "external" / "TrackEval"
    if not path.is_dir():
        pytest.skip("TrackEval checkout required. See README.md.")
    return path


def _kitti(frame, track_id, box, *, category="Car", flags=(0, 0)):
    truncation, occlusion = flags
    bbox = " ".join(map(str, box))
    return (
        f"{frame} {track_id} {category} {truncation} {occlusion} 0 "
        f"{bbox} 0 0 0 0 0 0 0\n"
    )


def _evaluate(tmp_path, checkout, gt_rows, prediction_rows):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"sequence": "0000", "source": "gt", "first_frame": 20, "last_frame": 32},
        ),
        encoding="utf-8",
    )
    (tmp_path / "protocol.toml").write_text(
        '[split]\ntune = ["0000"]\nevaluation = ["0001"]\n',
        encoding="utf-8",
    )
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("".join(gt_rows), encoding="utf-8")
    prediction_path = tmp_path / "predictions" / "data" / "0000.txt"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_text("".join(prediction_rows), encoding="utf-8")

    result = evaluate_sort(tmp_path, gt_path, checkout)
    assert gt_path.read_text(encoding="utf-8") == "".join(gt_rows)
    assert prediction_path.read_text(encoding="utf-8") == "".join(prediction_rows)
    with result.open(encoding="utf-8", newline="") as file:
        (row,) = csv.DictReader(file)
    return row


def test_evaluates_full_kitti_rules_after_warmup(tmp_path, trackeval_checkout):
    car = (0, 0, 40, 20)  # Short, but matched at frames 30 and 32.
    van = (100, 0, 140, 30)
    occluded = (200, 0, 240, 30)
    truncated = (300, 0, 340, 30)
    ignored = (400, 0, 440, 30)
    half_ignored = (500, 0, 540, 30)
    short = (600, 0, 640, 25)
    tall = (700, 0, 740, 26)
    gt = [
        _kitti(20, 1, car),
        *(_kitti(frame, 1, car, flags=(0, 2)) for frame in range(30, 33)),
        _kitti(30, 2, van, category="Van"),
        _kitti(30, 3, occluded, flags=(0, 3)),
        _kitti(30, 4, truncated, flags=(1, 0)),
        _kitti(30, -1, (400, 0, 430, 30), category="DontCare"),
        _kitti(30, -1, (500, 0, 520, 30), category="DontCare"),
    ]
    predictions = [
        _kitti(20, 90, tall),
        *(
            _kitti(30, track_id, box)
            for track_id, box in enumerate(
                (car, van, occluded, truncated, ignored, half_ignored, short, tall),
                start=10,
            )
        ),
        _kitti(31, 17, tall),
        _kitti(32, 18, car),
    ]

    row = _evaluate(tmp_path, trackeval_checkout, gt, predictions)

    assert row["status"] == "ok"
    assert row["trackeval_revision"] == TRACKEVAL_REVISION
    assert [row[key] for key in ("frames", "NGT", "FP", "FN", "IDSW", "Frag")] == [
        "3",
        "3",
        "3",
        "1",
        "1",
        "1",
    ]
    assert float(row["IDF1"]) == pytest.approx(25)
    assert float(row["HOTA"]) == pytest.approx(100 / 3)
    assert float(row["DetA"]) == pytest.approx(100 / 3)
    assert float(row["AssA"]) == pytest.approx(100 / 3)
    assert float(row["RFP"]) == pytest.approx(100)


def test_no_evaluable_ground_truth_keeps_false_positives(tmp_path, trackeval_checkout):
    box = (0, 0, 40, 30)
    row = _evaluate(
        tmp_path,
        trackeval_checkout,
        [_kitti(20, 1, box)],
        [_kitti(30, 10, box)],
    )

    assert row["status"] == "no_ground_truth"
    assert row["NGT"] == "0"
    assert row["FP"] == "1"
    assert row["RFP"] == ""

    csv_path = tmp_path / "metrics.csv"
    original = csv_path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluate_sort(tmp_path, tmp_path / "gt.txt", trackeval_checkout)
    assert csv_path.read_bytes() == original
