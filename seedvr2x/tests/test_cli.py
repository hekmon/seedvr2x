"""The command line starts: --version, and python -m seedvr2x."""

import subprocess
import sys

import pytest

from seedvr2x import cli


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("seedvr2x 0.")


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "seedvr2x", "--help"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith("usage: seedvr2x")
