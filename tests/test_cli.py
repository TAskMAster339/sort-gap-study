import subprocess


def test_cli_help():
    result = subprocess.run(
        ["sort-gap-study", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    assert "usage: sort-gap-study" in result.stdout
    assert "--help" in result.stdout
