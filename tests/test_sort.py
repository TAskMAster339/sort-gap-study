from pathlib import Path

import numpy as np
import pytest

from sort_gap_study.config import (
    load_sort_kalman,
    load_sort_lifecycle,
    load_sort_recovery,
)
from sort_gap_study.sort import SortTracker


@pytest.fixture
def protocol():
    return Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"


@pytest.fixture
def tracker(protocol):
    return SortTracker(load_sort_kalman(protocol), load_sort_lifecycle(protocol))


@pytest.fixture
def extra_tracker(protocol):
    return SortTracker(
        load_sort_kalman(protocol),
        load_sort_lifecycle(protocol),
        recovery=load_sort_recovery(protocol),
    )


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


def test_recovery_candidates_require_confirmation_fresh_observation_and_no_match(
    tracker,
):
    first = _detection(0)
    primary_match = _detection(40)
    stale = _detection(80)
    tentative = _detection(120)
    for frame in range(3):
        tracker.step(frame, first + primary_match + stale, artificial_gap=False)
    tracker.step(3, first + primary_match + tentative, artificial_gap=False)

    for frame in (4, 5):
        tracker.step(frame, [], artificial_gap=True)
        assert tracker.last_events.recovery_candidates == ()

    tracker.step(6, primary_match + _detection(160), artificial_gap=False)
    events = tracker.last_events
    assert events.matched == ((2, 0),)
    assert events.created == ((5, 1),)
    assert events.recovery_candidates == (1,)

    tracker.step(7, [], artificial_gap=False)
    assert tracker.last_events.recovery_candidates == ()


def test_available_empty_frame_ends_pause_and_later_gap_has_own_start(tracker):
    for frame in range(3):
        tracker.step(frame, _detection(), artificial_gap=False)

    tracker.step(3, [], artificial_gap=False)
    assert tracker.last_events.recovery_candidates == ()
    tracker.step(4, [], artificial_gap=True)
    tracker.step(5, [], artificial_gap=False)
    assert tracker.last_events.recovery_candidates == ()

    tracker.step(6, _detection(), artificial_gap=False)
    tracker.step(7, [], artificial_gap=True)
    tracker.step(8, [], artificial_gap=False)
    assert tracker.last_events.recovery_candidates == (1,)
    tracker.step(9, _detection(40), artificial_gap=False)
    assert tracker.last_events.recovery_candidates == ()


@pytest.mark.parametrize(
    ("return_frame", "candidates", "removed"),
    [(32, (1,), ()), (33, (), (1,))],
)
def test_recovery_candidates_use_age_at_first_available_frame(
    tracker,
    return_frame,
    candidates,
    removed,
):
    for frame in range(3):
        tracker.step(frame, _detection(), artificial_gap=False)
    for frame in range(3, return_frame):
        tracker.step(frame, [], artificial_gap=True)
    tracker.step(return_frame, _detection(40), artificial_gap=False)

    assert tracker.last_events.recovery_candidates == candidates
    assert tracker.last_events.removed == removed


def test_extra_association_uses_remaining_detections_before_track_creation(
    extra_tracker,
):
    for frame in range(3):
        extra_tracker.step(frame, _detection() + _detection(40), artificial_gap=False)
    for frame in range(3, 6):
        estimates_during_gap = extra_tracker.step(frame, [], artificial_gap=True)

    detections = _detection(100) + _detection(40) + _detection(10)
    estimates = extra_tracker.step(6, detections, artificial_gap=False)
    events = extra_tracker.last_events

    assert events.recovery_candidates == (1,)
    assert events.matched == ((2, 1), (1, 2))
    assert events.recovery_matches == ((1, 2),)
    assert events.created == ((3, 0),)
    assert [estimate.track_id for estimate in estimates] == [1, 2]
    assert all(estimate.observed for estimate in estimates)
    assert estimates[0].bbox[0] > estimates_during_gap[0].bbox[0]
    assert estimates_during_gap[0].bbox == (0, 0, 10, 10)


def test_sort_modes_agree_without_artificial_gaps(tracker, extra_tracker):
    frames = [
        _detection() + _detection(40),
        _detection(41) + _detection(1),
        _detection(2) + _detection(42),
        [],
        _detection(4) + _detection(54),
        _detection(5) + _detection(55),
        _detection(56) + _detection(6),
    ]
    for frame, detections in enumerate(frames):
        basic_estimates = tracker.step(frame, detections, artificial_gap=False)
        extra_estimates = extra_tracker.step(frame, detections, artificial_gap=False)
        assert basic_estimates == extra_estimates
        assert tracker.last_events == extra_tracker.last_events
        assert extra_tracker.last_events.recovery_matches == ()


@pytest.mark.parametrize(
    ("gap_at_frame_six", "recovery_matches", "created"),
    [(True, ((1, 0),), ()), (False, (), ((2, 0),))],
    ids=["first_available", "after_available_empty_frame"],
)
def test_extra_assignment_only_on_first_available_frame(
    extra_tracker,
    gap_at_frame_six,
    recovery_matches,
    created,
):
    for frame in range(3):
        extra_tracker.step(frame, _detection(), artificial_gap=False)
    for frame in range(3, 6):
        extra_tracker.step(frame, [], artificial_gap=True)
        assert extra_tracker.last_events.recovery_matches == ()
    extra_tracker.step(6, [], artificial_gap=gap_at_frame_six)
    assert extra_tracker.last_events.recovery_matches == ()

    extra_tracker.step(7, _detection(10), artificial_gap=False)
    assert extra_tracker.last_events.recovery_matches == recovery_matches
    assert extra_tracker.last_events.created == created


@pytest.mark.parametrize(
    ("return_frame", "recovery_matches", "removed", "created"),
    [(32, ((1, 0),), (), ()), (33, (), (1,), ((2, 0),))],
    ids=["age_30", "age_31"],
)
def test_extra_assignment_respects_max_age(
    extra_tracker,
    return_frame,
    recovery_matches,
    removed,
    created,
):
    for frame in range(3):
        extra_tracker.step(frame, _detection(), artificial_gap=False)
    for frame in range(3, return_frame):
        (prediction,) = extra_tracker.step(frame, [], artificial_gap=True)
        assert prediction.track_id == 1
        assert not prediction.observed

    estimates = extra_tracker.step(return_frame, _detection(10), artificial_gap=False)
    events = extra_tracker.last_events

    assert events.recovery_matches == recovery_matches
    assert events.removed == removed
    assert events.created == created
    assert [(item.track_id, item.observed) for item in estimates] == [
        (track_id, True) for track_id, _ in recovery_matches
    ]
