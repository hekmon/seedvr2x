"""The command line starts: --version, and python -m seedvr2x."""

import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from seedvr2x import cli
from seedvr2x.media import ffmpeg


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


def test_ratio_option() -> None:
    assert cli._ratio("32:27") == Fraction(32, 27)
    assert cli._ratio("4/3") == Fraction(4, 3)
    for text in ("0:1", "16:0", "1.5", "-4:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._ratio(text)


def test_refused_source_stops_before_torch(tmp_path: Path) -> None:
    # The build and the source are checked before the GPU is touched: torch isn't even imported.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    source = tmp_path / "tff.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2"),
            *("-vf", "setfield=tff", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    args = [str(source), "-o", str(tmp_path / "out.mkv"), "--model-dir", ".", "--dit-model", "x"]
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " assert 'torch' not in sys.modules; sys.exit(status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1, result.stderr
    assert "interlaced (field order" in result.stderr
