"""The startup check refuses an ffmpeg without what seedvr2x needs: fake builds on PATH."""

from pathlib import Path

import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError

FILTERS = ("zscale", "scdet", "format", "settb", "metadata", "setparams", "idet")


def fake_build(
    directory: Path,
    filters: tuple[str, ...] = FILTERS,
    encoders: tuple[str, ...] = ("ffv1", "png"),
    decoders: tuple[str, ...] = ("ffv1", "h264"),
    ffprobe: bool = True,
) -> None:
    """An ffmpeg answering -filters, -encoders, -decoders and -version as a real one lays them
    out (n9.0.2), header included; shell builtins only, PATH holding nothing else."""
    lists = {
        "-filters": ["Filters:", "  T.. = Timeline support", "  | = Source or sink filter"]
        + [f" .. {name:<16} V->V       A filter." for name in filters],
        "-encoders": ["Encoders:", " V..... = Video", " ------"]
        + [f" V....D {name:<20} A codec." for name in encoders],
        "-decoders": ["Decoders:", " V..... = Video", " ------"]
        + [f" V....D {name:<20} A codec." for name in decoders],
        "-version": ["ffmpeg version n0.0-fake Copyright (c) 2000-2026 the FFmpeg developers"],
    }
    cases = "".join(
        f"  {option}) printf '%s\\n' {' '.join(repr(line) for line in lines)} ;;\n"
        for option, lines in lists.items()
    )
    for name in ("ffmpeg", "ffprobe") if ffprobe else ("ffmpeg",):
        script = directory / name
        script.write_text(f'#!/bin/sh\ncase "$2" in\n{cases}  *) exit 1 ;;\nesac\n')
        script.chmod(0o755)


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PATH", str(tmp_path))
    return tmp_path


def test_complete_build(build: Path) -> None:
    fake_build(build)
    assert ffmpeg.check() == "n0.0-fake"


def test_without_zscale_says_how_to_get_it(build: Path) -> None:
    fake_build(build, filters=tuple(name for name in FILTERS if name != "zscale"))
    with pytest.raises(MediaError, match=r"lacks the zscale filter\. zscale comes with libzimg"):
        ffmpeg.check()


WITHOUT_SCDET = tuple(name for name in FILTERS if name != "scdet")


@pytest.mark.parametrize(
    ("filters", "encoders", "decoders", "missing"),
    [
        (WITHOUT_SCDET, ("ffv1",), ("ffv1",), "the scdet filter"),
        (FILTERS, ("png",), ("ffv1",), "the ffv1 encoder"),
        (FILTERS, ("ffv1",), ("h264",), "the ffv1 decoder"),
    ],
)
def test_incomplete_build(
    build: Path,
    filters: tuple[str, ...],
    encoders: tuple[str, ...],
    decoders: tuple[str, ...],
    missing: str,
) -> None:
    fake_build(build, filters, encoders, decoders)
    with pytest.raises(MediaError, match=f"lacks {missing}") as refused:
        ffmpeg.check()
    assert "libzimg" not in str(refused.value)


def test_without_ffprobe(build: Path) -> None:
    fake_build(build, ffprobe=False)
    with pytest.raises(MediaError, match="ffprobe not found"):
        ffmpeg.check()


def test_png_encoder_needed_only_for_png_output(build: Path) -> None:
    fake_build(build, encoders=("ffv1",))
    assert ffmpeg.check() == "n0.0-fake"
    with pytest.raises(MediaError, match="lacks the png encoder"):
        ffmpeg.check(("png",))
