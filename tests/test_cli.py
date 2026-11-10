import shutil
import subprocess


def test_cli_help():
    executable = shutil.which("sort-gap-study")
    assert executable is not None

    result = subprocess.run(  # noqa: S603 — fixed entry point and arguments
        [executable, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    assert "usage: sort-gap-study" in result.stdout
    assert "--help" in result.stdout
    assert "audit-data" in result.stdout
