from pathlib import Path

import numpy as np
import pytest

from sort_gap_study.config import load_sort_kalman, load_sort_lifecycle
from sort_gap_study.sort import SortTracker


@pytest.fixture
def tracker():
    path = Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"
    return SortTracker(load_sort_kalman(path), load_sort_lifecycle(path))


def _detection(offset=0):
    return [[offset, 0, offset + 10, 10, 1]]


def test_confirmation_counts_only_real_observations_including_creation(tracker):
    assert tracker.step(10, _detection(), artificial_gap=False) == ()
    assert tracker.step(11, [], artificial_gap=True) == ()
    assert tracker.step(12, _detection(), artificial_gap=False) == ()
    assert tracker.step(13, [], artificial_gap=False) == ()

    (confirmed,) = tracker.step(14, _detection(), artificial_gap=False)
    assert confirmed.track_id == 1
    assert confirmed.observed
    assert confirmed.bbox == (0, 0, 10, 10)


def test_predicts_on_artificial_and_natural_empty_frames_then_recovers_id(tracker):
    tracker.step(0, _detection(), artificial_gap=False)
    tracker.step(1, _detection(1), artificial_gap=False)
    (before_gap,) = tracker.step(2, _detection(2), artificial_gap=False)
    (gap,) = tracker.step(3, np.empty((0, 5)), artificial_gap=True)
    (empty,) = tracker.step(4, [], artificial_gap=False)
    (observed,) = tracker.step(5, _detection(6), artificial_gap=False)

    assert before_gap.track_id == gap.track_id == empty.track_id == observed.track_id
    assert before_gap.observed
    assert observed.observed
    assert not gap.observed
    assert not empty.observed
    # A correction at frame 5 must leave previously returned estimates intact.
    np.testing.assert_allclose(gap.bbox, [3, 0, 13, 10], atol=1e-3)
    np.testing.assert_allclose(empty.bbox, [4, 0, 14, 10], atol=1e-3)
    assert observed.bbox[0] > empty.bbox[0] + 1


@pytest.mark.parametrize("return_on_boundary", [True, False])
def test_max_age_boundary_before_association(tracker, *, return_on_boundary):
    for frame in range(3):
        tracker.step(frame, _detection(), artificial_gap=False)
    for frame in range(3, 32):
        (prediction,) = tracker.step(frame, [], artificial_gap=True)
        assert prediction.track_id == 1
        assert not prediction.observed

    if return_on_boundary:
        (matched,) = tracker.step(32, _detection(), artificial_gap=False)
        assert matched.track_id == 1
        assert matched.observed
    else:
        (prediction,) = tracker.step(32, [], artificial_gap=True)
        assert prediction.track_id == 1
        assert not prediction.observed
        assert tracker.step(33, _detection(), artificial_gap=False) == ()
        assert tracker.step(34, _detection(), artificial_gap=False) == ()
        (replacement,) = tracker.step(35, _detection(), artificial_gap=False)
        assert [prediction.track_id, replacement.track_id] == [1, 2]


def test_unmatched_detections_create_tracks_in_input_order(tracker):
    first = _detection(40)
    second = _detection(0)
    tracker.step(0, first + second, artificial_gap=False)
    tracker.step(1, second + first, artificial_gap=False)
    estimates = tracker.step(2, second + first, artificial_gap=False)

    assert [estimate.track_id for estimate in estimates] == [1, 2]
    assert [estimate.bbox for estimate in estimates] == [
        (40, 0, 50, 10),
        (0, 0, 10, 10),
    ]


def test_can_retry_same_frame_after_invalid_input(tracker):
    tracker.step(0, _detection(), artificial_gap=False)
    with pytest.raises(ValueError, match="подряд"):
        tracker.step(2, _detection(), artificial_gap=False)
    with pytest.raises(ValueError, match="искусственного пропуска"):
        tracker.step(1, _detection(), artificial_gap=True)
    with pytest.raises(ValueError, match="bbox"):
        tracker.step(1, [[0, 0, 0, 10, 1]], artificial_gap=False)

    assert tracker.step(1, _detection(), artificial_gap=False) == ()
    (confirmed,) = tracker.step(2, _detection(), artificial_gap=False)
    assert confirmed.track_id == 1
