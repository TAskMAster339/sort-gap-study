from importlib import import_module


def test_package_is_importable():
    package = import_module("sort_gap_study")

    assert package.__name__ == "sort_gap_study"
