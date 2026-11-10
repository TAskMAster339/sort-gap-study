from pathlib import Path

import pytest

from sort_gap_study.config import load_split


def test_load_project_split():
    path = Path(__file__).resolve().parents[1] / "configs" / "protocol.toml"
    split = load_split(path)

    assert split.tune == ("0000", "0002", "0005", "0009", "0011", "0017", "0019")
    assert len(split.evaluation) == 14
    assert sorted(split.tune + split.evaluation) == [f"{i:04d}" for i in range(21)]


@pytest.mark.parametrize(
    ("tune", "evaluation", "error"),
    [
        ('["0000"]', '["0000"]', r"пересекаются: 0000"),
        ('["0000", "0000"]', '["0001"]', r"split\.tune:.*повторяющиеся"),
        ('["0000"]', '["0001", "0001"]', r"split\.evaluation:.*повторяющиеся"),
        ("[]", '["0001"]', r"split\.tune:.*непустой"),
        ("[0]", '["0001"]', r"split\.tune:.*четырёхзначных"),
        ('["0000"]', '["1"]', r"split\.evaluation:.*четырёхзначных"),
    ],
)
def test_reject_invalid_split(tmp_path, tune, evaluation, error):
    path = tmp_path / "protocol.toml"
    path.write_text(f"[split]\ntune = {tune}\nevaluation = {evaluation}\n")

    with pytest.raises(ValueError, match=error):
        load_split(path)


def test_require_split_table(tmp_path):
    path = tmp_path / "protocol.toml"
    path.write_text("")

    with pytest.raises(TypeError, match=r"таблица \[split\]"):
        load_split(path)
