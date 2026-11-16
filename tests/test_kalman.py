from pathlib import Path

import numpy as np
import pytest

from sort_gap_study.config import load_sort_kalman
from sort_gap_study.kalman import SortKalmanFilter, bbox_to_measurement


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"
    return load_sort_kalman(path)


def test_bbox_round_trip(config):
    bbox = [2.5, 1, 8.5, 4]
    track = SortKalmanFilter(bbox, config)

    np.testing.assert_allclose(bbox_to_measurement(bbox), [5.5, 2.5, 18, 2])
    np.testing.assert_allclose(track.bbox, bbox)
    track.predict()
    np.testing.assert_allclose(track.bbox, bbox)


@pytest.mark.parametrize(
    "bbox",
    [[0, 0, 2, 2, 1], [0, 0, np.inf, 2], [1, 0, 0, 2], [0, 2, 2, 2]],
)
def test_reject_invalid_bbox(bbox):
    with pytest.raises(ValueError, match="bbox"):
        bbox_to_measurement(bbox)


def test_predict_consecutive_frames_and_correct_after_gap(config):
    initial = np.array([0, 0, 8, 4])
    displacement = np.array([6, -2, 6, -2])
    track = SortKalmanFilter(initial, config)
    track.predict()
    track.correct(initial + displacement)

    # For each center coordinate, P⁻ = 10 + 10000 + 1 and R = 1.
    expected_shift = displacement * (10011 / 10012)
    expected_velocity = displacement * (10000 / 10012)
    np.testing.assert_allclose(track.bbox, initial + expected_shift)
    for frames_without_observation in range(1, 4):
        track.predict()
        expected = (
            initial + expected_shift + frames_without_observation * expected_velocity
        )
        np.testing.assert_allclose(track.bbox, expected)

    track.predict()
    before_correction = track.bbox
    observation = initial + 5 * displacement + [3, 2, 3, 2]
    track.correct(observation)

    distance_before = np.linalg.norm(before_correction - observation)
    distance_after = np.linalg.norm(track.bbox - observation)
    assert distance_after < distance_before
    np.testing.assert_allclose(
        before_correction,
        initial + expected_shift + 4 * expected_velocity,
    )


def test_predict_resets_velocity_before_area_becomes_nonpositive(config):
    track = SortKalmanFilter([0, 0, 10, 10], config)
    track.predict()
    track.correct([4, 4, 6, 6])
    corrected = track.bbox
    corrected_area = (corrected[2] - corrected[0]) * (corrected[3] - corrected[1])
    np.testing.assert_allclose(corrected_area, 100 + (10011 / 10021) * (4 - 100))

    track.predict()
    np.testing.assert_allclose(track.bbox, corrected)
    track.predict()
    np.testing.assert_allclose(track.bbox, corrected)
