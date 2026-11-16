from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from sort_gap_study.config import (
    SortRecoveryConfig,
    load_sort_kalman,
    load_sort_recovery,
)
from sort_gap_study.kalman import SortKalmanFilter
from sort_gap_study.recovery import recovery_costs


@pytest.fixture
def protocol():
    return Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"


def test_center_projection_uses_predicted_covariance_and_measurement_noise(protocol):
    kalman = SortKalmanFilter([0, 0, 10, 10], load_sort_kalman(protocol))
    kalman.predict()
    center, covariance = kalman.project_center()

    np.testing.assert_array_equal(center, [5, 5])
    np.testing.assert_array_equal(covariance, np.diag([10012, 10012]))


def test_recovery_cost_uses_full_covariance_and_all_three_terms(protocol):
    projection = (np.array([5.0, 5.0]), np.array([[16.0, 12.0], [12.0, 16.0]]))
    config = load_sort_recovery(protocol)
    costs = recovery_costs(
        np.array([[0, 0, 10, 10]]),
        np.array([[4, 5, 24, 15]]),
        [projection],
        config,
    )

    expected = 0.5 * (5.5 / 9.21034) + 0.3 * (sqrt(106) / sqrt(200)) / 2 + 0.2
    assert costs[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("covariance", "allowed_box", "rejected_box"),
    [
        (4.0, [4, 0, 7, 4], [4.01, 0, 7.01, 4]),
        (100.0, [10, 0, 13, 4], [10.01, 0, 13.01, 4]),
        (100.0, [-1.5, 0, 4.5, 4], [-1.51, 0, 4.51, 4]),
        (100.0, [0.75, 0, 2.25, 4], [0.76, 0, 2.24, 4]),
    ],
    ids=["mahalanobis", "center_shift", "larger_box", "smaller_box"],
)
def test_recovery_gate_boundaries(covariance, allowed_box, rejected_box):
    config = SortRecoveryConfig(4, 2, 2)
    projection = (np.array([1.5, 2]), np.eye(2) * covariance)
    costs = recovery_costs(
        np.array([[0, 0, 3, 4]]),
        np.array([allowed_box, rejected_box]),
        [projection],
        config,
    )

    assert np.isfinite(costs[0, 0])
    assert np.isposinf(costs[0, 1])
