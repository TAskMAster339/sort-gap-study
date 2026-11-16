import numpy as np
import pytest

from sort_gap_study.association import Association, associate_iou, pairwise_iou


def test_pairwise_iou_for_identical_overlapping_and_touching_boxes():
    tracks = [[0, 0, 10, 10], [-5, 0, 5, 10]]
    detections = [[0, 0, 10, 10], [5, 0, 15, 10], [0, 0, 3, 10]]

    np.testing.assert_allclose(
        pairwise_iou(tracks, detections),
        [[1, 1 / 3, 0.3], [1 / 3, 0, 0.3]],
    )


@pytest.mark.parametrize(
    ("tracks", "detections", "shape", "expected"),
    [
        ([], [], (0, 0), Association((), (), ())),
        (np.empty((0, 4)), [[0, 0, 10, 10]], (0, 1), Association((), (), (0,))),
        ([[0, 0, 10, 10]], np.empty((0, 4)), (1, 0), Association((), (0,), ())),
    ],
)
def test_empty_inputs(tracks, detections, shape, expected):
    assert pairwise_iou(tracks, detections).shape == shape
    assert associate_iou(tracks, detections) == expected


def test_competing_tracks_leave_detection_to_highest_iou():
    tracks = [[0, 0, 7, 10], [0, 0, 10, 10], [10, 0, 20, 10]]
    detections = [[0, 0, 10, 10]]

    assert associate_iou(tracks, detections) == Association(((1, 0),), (0, 2), ())


def test_assignment_maximizes_cardinality_before_iou():
    tracks = [[0, 0, 10, 10], [-5, 0, 5, 10]]
    detections = [[0, 0, 10, 10], [5, 0, 15, 10]]

    # Two matches of IoU 1/3 take precedence over one match of IoU 1.
    assert associate_iou(tracks, detections) == Association(((0, 1), (1, 0)), (), ())


def test_assignment_maximizes_total_iou_for_equal_cardinality():
    tracks = [[0, 0, 10, 10], [2, 0, 12, 10]]
    detections = [[2, 0, 12, 10], [0, 0, 10, 10]]

    assert associate_iou(tracks, detections) == Association(((0, 1), (1, 0)), (), ())


def test_below_threshold_pair_leaves_both_indices_unmatched():
    tracks = [[0, 0, 10, 10]]
    detections = [[0, 0, 2, 10], [12, 0, 20, 10]]

    assert associate_iou(tracks, detections) == Association((), (0,), (0, 1))


def test_iou_equal_to_threshold_is_admissible():
    tracks = [[0, 0, 10, 10]]
    detections = [[0, 0, 3, 10], [0, 0, 2, 10]]

    assert associate_iou(tracks, detections) == Association(((0, 0),), (), (1,))
