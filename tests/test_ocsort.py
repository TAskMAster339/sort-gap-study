import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sort_gap_study.config import load_ocsort, load_sort_lifecycle
from sort_gap_study.ocsort import OCSortTracker


@pytest.fixture
def tracker_factory():
    root = Path(__file__).resolve().parents[1]
    checkout = root / "external" / "OC_SORT"
    if not checkout.is_dir():
        pytest.skip("OC-SORT checkout required. See README.md.")
    config = root / "configs" / "protocol.toml"
    parameters = load_ocsort(config)
    lifecycle = load_sort_lifecycle(config)

    def create():
        return OCSortTracker(checkout, parameters, lifecycle)

    return create


def _detections(left=0):
    return np.array([[left, 0, left + 10, 10, 1]], dtype=np.float64)


def test_ocsort_confirms_and_keeps_emitted_gap_predictions_after_recovery(
    tracker_factory,
):
    tracker = tracker_factory()
    assert tracker.step(10, _detections(0), artificial_gap=False) == ()
    assert tracker.step(11, _detections(1), artificial_gap=False) == ()
    (confirmed,) = tracker.step(12, _detections(2), artificial_gap=False)
    assert confirmed.track_id == 1
    assert confirmed.observed

    (gap,) = tracker.step(13, [], artificial_gap=True)
    (empty,) = tracker.step(14, [], artificial_gap=False)
    earlier = json.dumps([asdict(item) for item in (gap, empty)])
    (recovered,) = tracker.step(15, _detections(6), artificial_gap=False)

    assert gap.track_id == empty.track_id == recovered.track_id == confirmed.track_id
    assert not gap.observed
    assert not empty.observed
    assert recovered.observed
    assert tracker.last_events.matched == ((1, 0),)
    assert tracker.last_events.created == ()
    np.testing.assert_allclose(gap.bbox, [3, 0, 13, 10], atol=1e-3)
    np.testing.assert_allclose(empty.bbox, [4, 0, 14, 10], atol=1e-3)
    assert recovered.bbox[0] > empty.bbox[0] + 1
    assert json.dumps([asdict(item) for item in (gap, empty)]) == earlier


@pytest.mark.parametrize(
    ("available_frame", "removed", "created"),
    [(32, (), ()), (33, (1,), ((2, 0),))],
)
def test_ocsort_applies_lifetime_before_matching(
    tracker_factory,
    available_frame,
    removed,
    created,
):
    tracker = tracker_factory()
    for frame in range(3):
        tracker.step(frame, _detections(), artificial_gap=False)
    for frame in range(3, available_frame):
        (estimate,) = tracker.step(frame, [], artificial_gap=True)
        assert estimate.track_id == 1
        assert not estimate.observed

    estimates = tracker.step(available_frame, _detections(), artificial_gap=False)

    assert tracker.last_events.removed == removed
    assert tracker.last_events.created == created
    assert [item.track_id for item in estimates] == ([] if removed else [1])


def test_ocsort_uses_actual_rows_for_identical_boxes_and_independent_ids(
    tracker_factory,
):
    tracker = tracker_factory()
    detections = np.repeat(_detections(), 2, axis=0)
    detections[:, -1] = [0.1, 0.2]
    before = detections.copy()
    tracker.step(0, detections, artificial_gap=False)
    assert tracker.last_events.created == ((1, 0), (2, 1))

    other = tracker_factory()
    other.step(0, _detections(), artificial_gap=False)
    assert other.last_events.created == ((1, 0),)

    tracker.step(
        1,
        np.vstack([detections, _detections(40)]),
        artificial_gap=False,
    )
    assert {track_id for track_id, _ in tracker.last_events.matched} == {1, 2}
    assert {detection for _, detection in tracker.last_events.matched} == {0, 1}
    assert tracker.last_events.created == ((3, 2),)
    np.testing.assert_array_equal(detections, before)


@pytest.mark.parametrize("bbox", [[np.nan] * 4, [0, 0, 0, 10]])
def test_ocsort_omits_invalid_bbox_and_keeps_association(
    tracker_factory,
    monkeypatch,
    bbox,
):
    tracker = tracker_factory()
    for frame in range(3):
        tracker.step(frame, _detections(), artificial_gap=False)
    track = tracker._engine.trackers[0]  # noqa: SLF001 — inject an upstream estimate
    with monkeypatch.context() as patch:
        patch.setattr(track, "get_state", lambda: np.array([bbox]))
        assert tracker.step(3, _detections(), artificial_gap=False) == ()
    events = tracker.last_events
    assert events.matched == ((1, 0),)
    assert events.invalid_estimates == (1,)
    assert events.removed == ()

    (estimate,) = tracker.step(4, _detections(), artificial_gap=False)
    assert estimate.track_id == 1
    assert tracker.last_events.invalid_estimates == ()
