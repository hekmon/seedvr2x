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


def test_numz_padding_not_an_option(capsys: pytest.CaptureFixture[str]) -> None:
    # numz's padding is for seedvr2x's tests only (cli.NUMZ_PADDING): never in the help.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = capsys.readouterr().out
    assert cli.NUMZ_PADDING not in shown and "padding" not in shown.lower()


def test_too_small_to_pad_refused_before_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The target is checked with the source, before the first pass and the model files: at
    # --resolution 16, 64x48 is resized to 21x16, whose 16 rows would take 16 reflected; 48x64 to
    # 16x21, whose 21 rows take 11, and 16 columns none, so the model files are checked next.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    from seedvr2x.media import source as sources

    def first_pass(*args: object, **kwargs: object) -> None:
        raise AssertionError("the first pass ran")

    monkeypatch.setattr(sources, "first_pass", first_pass)
    for size, said in (
        ("64x48", "--resolution 16: the frames resized to 21x16 are too small for the padding"),
        ("48x64", "no such model file"),
    ):
        source = tmp_path / f"{size}.mkv"
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=25:d=0.2"),
                *("-c:v", "ffv1", "-pix_fmt", "yuv420p", str(source)),
            ],
            check=True,
        )
        caplog.clear()
        args = [str(source), "-o", str(tmp_path / "out.mkv"), "--model-dir", str(tmp_path)]
        assert cli.main([*args, "--resolution", "16"]) == 1
        assert said in caplog.text
        assert ("too small" in caplog.text) == (size == "64x48")


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
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " sys.exit(3 if 'torch' in sys.modules else status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1, result.stderr
    assert "interlaced (field order" in result.stderr
