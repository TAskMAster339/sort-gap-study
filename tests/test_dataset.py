import pytest

from sort_gap_study.config import SequenceSplit
from sort_gap_study.dataset import audit_inputs, read_seqmap


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
