import numpy as np
import pytest

from sort_gap_study.config import SequenceSplit
from sort_gap_study.dataset import audit_inputs, read_gt_detections, read_seqmap


def test_seqmap_includes_last_frame(tmp_path):
    path = tmp_path / "seqmap"
    path.write_text("0000 empty 000000 000154\n0001 empty 000003 000005\n")

    assert read_seqmap(path) == {"0000": range(155), "0001": range(3, 6)}


@pytest.fixture
def audit_fixture(tmp_path):
    seqmap = tmp_path / "seqmap"
    seqmap.write_text("0000 empty 000000 000002\n0001 empty 000000 000001\n")
    labels = tmp_path / "label_02"
    regionlet = tmp_path / "det_02"
    for directory in (labels, regionlet):
        directory.mkdir()
        (directory / "0000.txt").write_text("0 sample\n2 sample\n")
        (directory / "0001.txt").write_text("")
    split = SequenceSplit(tune=("0000",), evaluation=("0001",))
    return split, seqmap, labels, regionlet


def test_audit_accepts_sparse_and_empty_inputs(audit_fixture):
    audit_inputs(*audit_fixture)


def test_audit_requires_both_inputs(audit_fixture):
    *_, regionlet = audit_fixture
    (regionlet / "0001.txt").unlink()

    with pytest.raises(FileNotFoundError, match=r"0001\.txt"):
        audit_inputs(*audit_fixture)


@pytest.mark.parametrize("frame", ["-1", "3", "invalid"])
def test_audit_rejects_invalid_input_frame(audit_fixture, frame):
    *_, regionlet = audit_fixture
    (regionlet / "0000.txt").write_text(f"{frame} sample\n")

    with pytest.raises(ValueError, match=r"0000\.txt:1:"):
        audit_inputs(*audit_fixture)


def test_audit_requires_configured_sequence_in_seqmap(audit_fixture):
    _, seqmap, _, _ = audit_fixture
    seqmap.write_text("0000 empty 000000 000002\n")

    with pytest.raises(ValueError, match="отсутствует последовательность 0001"):
        audit_inputs(*audit_fixture)


def test_gt_detections_preserve_frames_and_hide_track_ids(tmp_path):
    path = tmp_path / "0000.txt"
    content = (
        "3 42 Car 1 3 0 10 20 30 40 1 2 3 4 5 6 0\n"
        "3 -1 DontCare -1 -1 -10 0 0 3 3 -1 -1 -1 -1 -1 -1 -10\n"
        "3 7 Van 0 0 0 1 2 3 4 1 2 3 4 5 6 0\n"
        "3 99 Car 0 0 0 11 21 31 41 1 2 3 4 5 6 0\n"
        "4 -1 Car 0 0 0 10 20 30 40 1 2 3 4 5 6 0\n"
        "4 10 Car 0 0 0 10 20 10 40 1 2 3 4 5 6 0\n"
        "4 11 Car 0 0 0 10 20 nan 40 1 2 3 4 5 6 0\n"
    )
    path.write_text(content)

    detections = read_gt_detections(path, range(3, 6))

    assert tuple(detections) == (3, 4, 5)
    np.testing.assert_array_equal(
        detections[3],
        [[10, 20, 30, 40, 1], [11, 21, 31, 41, 1]],
    )
    assert detections[4].shape == detections[5].shape == (0, 5)
    assert path.read_text() == content


def test_gt_detections_reject_frame_outside_sequence(tmp_path):
    path = tmp_path / "0000.txt"
    path.write_text("3 0 Car 0 0 0 10 20 30 40 1 2 3 4 5 6 0\n")

    with pytest.raises(ValueError, match=r"0000\.txt:1: кадр 3 вне диапазона"):
        read_gt_detections(path, range(3))
