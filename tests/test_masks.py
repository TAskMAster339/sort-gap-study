from itertools import pairwise

import pytest

from sort_gap_study.masks import make_burst_mask, make_mask


@pytest.mark.parametrize("kind", ["uniform", "random"])
@pytest.mark.parametrize(("fraction", "count"), [(0, 0), (0.25, 1), (0.5, 3), (1, 5)])
def test_mask_budget_and_original_frame_numbers(kind, fraction, count):
    frames = range(100, 115)
    evaluation = frames[10:]

    mask = make_mask(frames, fraction, kind, seed=5)

    assert len(mask) == count
    assert tuple(sorted(set(mask))) == mask
    assert set(mask).issubset(evaluation)
    if fraction == 1:
        assert mask == tuple(evaluation)


@pytest.mark.parametrize("kind", ["uniform", "random"])
def test_mask_repeats_for_same_seed(kind):
    frames = range(60)

    assert make_mask(frames, 0.25, kind, seed=10) == make_mask(
        frames,
        0.25,
        kind,
        seed=10,
    )


def test_random_mask_changes_with_seed():
    frames = range(60)

    assert make_mask(frames, 0.25, "random", seed=10) != make_mask(
        frames,
        0.25,
        "random",
        seed=11,
    )


@pytest.mark.parametrize("seed", [0, 1, 5, 10])
def test_uniform_mask_has_one_gap_in_each_four_frame_block(seed):
    mask = make_mask(range(30), 0.25, "uniform", seed=seed)

    assert tuple((frame - 10) // 4 for frame in mask) == tuple(range(5))
    assert len({(frame - 10) % 4 for frame in mask}) == 1


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="Доля пропусков"):
        make_mask(range(20), fraction, "uniform", seed=0)


def test_warmup_cannot_exceed_sequence_length():
    with pytest.raises(ValueError, match="допустимый прогрев"):
        make_mask(range(9), 0.5, "uniform", seed=0)


@pytest.mark.parametrize("seed", range(10))
def test_burst_mask_preserves_budget_remainder_and_separators(seed):
    frames = range(100, 121)
    mask = make_burst_mask(frames, 0.5, series_length=4, seed=seed)

    assert mask == make_burst_mask(frames, 0.5, series_length=4, seed=seed)
    assert mask.status == "ok"
    assert mask.reason is None
    assert sorted(length for _, length in mask.runs) == [2, 4]
    assert mask.removed_frames == tuple(
        frame for start, length in mask.runs for frame in range(start, start + length)
    )
    assert set(mask.removed_frames).issubset(frames[10:])
    (first, length), (second, _) = mask.runs
    assert first + length < second


def test_burst_mask_uses_minimum_space():
    mask = make_burst_mask(range(17), 0.75, series_length=2, seed=0)

    assert (mask.removed_frames[0], mask.removed_frames[-1]) == (10, 16)
    assert all(
        start + length + 1 == next_start
        for (start, length), (next_start, _) in pairwise(mask.runs)
    )
    assert mask.status == "ok"


def test_burst_mask_marks_infeasible_condition():
    mask = make_burst_mask(range(17), 1, series_length=2, seed=0)

    assert mask.status == "infeasible"
    assert mask.reason is not None
    assert "10 кадров" in mask.reason
    assert mask.removed_frames == mask.runs == ()


def test_burst_mask_with_no_losses():
    mask = make_burst_mask(range(20), 0, series_length=3, seed=0)

    assert mask.status == "ok"
    assert mask.reason is None
    assert mask.removed_frames == mask.runs == ()


def test_burst_mask_rejects_nonpositive_length():
    with pytest.raises(ValueError, match="Длина серии"):
        make_burst_mask(range(20), 0.5, series_length=0, seed=0)
