"""The startup check refuses an ffmpeg without what seedvr2x needs: fake builds on PATH."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError

FILTERS = ("zscale", "format", "scale", "setparams", "split", "select", "idet", "null")

# setparams' options as ffmpeg n9.0.2's -h filter=setparams lists them, each with its help, and
# the named values of two of them (the others' left out): (name, value, help).
SETPARAMS = {
    "field_mode": (
        "select interlace mode (from -1 to 2) (default auto)",
        [
            ("auto", "-1", "keep the same input field"),
            ("bff", "0", "mark as bottom-field-first"),
            ("tff", "1", "mark as top-field-first"),
            ("prog", "2", "mark as progressive"),
        ],
    ),
    "range": ("select color range (from -1 to 2) (default auto)", []),
    "color_primaries": ("select color primaries (from -1 to 256) (default auto)", []),
    "color_trc": ("select color transfer (from -1 to 256) (default auto)", []),
    "colorspace": ("select colorspace (from -1 to 17) (default auto)", []),
    "chroma_location": (
        "select chroma sample location (from -1 to 6) (default auto)",
        [
            ("auto", "-1", "keep the same chroma location"),
            ("unspecified", "0", ""),
            ("unknown", "0", ""),
            ("left", "1", ""),
            ("center", "2", ""),
            ("topleft", "3", ""),
            ("top", "4", ""),
            ("bottomleft", "5", ""),
            ("bottom", "6", ""),
        ],
    ),
    "alpha_mode": ("select alpha moda (from -1 to 2) (default auto)", []),
}
# setparams before ffmpeg 7.1, without chroma_location (libavfilter/vf_setparams.c at n6.1 and
# n7.0).
WITHOUT_CHROMA_LOCATION = tuple(name for name in SETPARAMS if name != "chroma_location")


def setparams_help(options: tuple[str, ...]) -> list[str]:
    """-h filter=setparams's lines as ffmpeg n9.0.2 prints them, listing `options`."""
    lines = [
        "Filter setparams",
        "  Force field, or color property for the output video frame.",
        "    Inputs:",
        "       #0: default (video)",
        "    Outputs:",
        "       #0: default (video)",
        "setparams AVOptions:",
    ]
    for name in options:
        description, values = SETPARAMS[name]
        lines.append(f"   {name:<17} {'<int>':<12} ..FV....... {description}")
        lines += [
            f"     {value:<15} {number:<12} ..FV......." + (f" {said}" if said else "")
            for value, number, said in values
        ]
    return [*lines, "", "", "Exiting with exit code 0"]


ENCODERS = ("ffv1", "rawvideo", "wrapped_avframe", "png")
DECODERS = ("ffv1", "rawvideo", "h264")
MUXERS = ("matroska", "framehash", "null", "rawvideo", "image2")


def fake_build(
    directory: Path,
    filters: tuple[str, ...] = FILTERS,
    encoders: tuple[str, ...] = ENCODERS,
    decoders: tuple[str, ...] = DECODERS,
    ffprobe: bool = True,
    muxers: tuple[str, ...] = MUXERS,
    setparams: tuple[str, ...] = tuple(SETPARAMS),
) -> None:
    """An ffmpeg answering -filters, -encoders, -decoders, -muxers, -version and -h
    filter=setparams as a real one lays them out (n9.0.2), header included; shell builtins only,
    PATH holding nothing else."""
    lists = {
        "-filters": ["Filters:", "  T.. = Timeline support", "  | = Source or sink filter"]
        + [f" .. {name:<16} V->V       A filter." for name in filters],
        "-encoders": ["Encoders:", " V..... = Video", " ------"]
        + [f" V....D {name:<20} A codec." for name in encoders],
        "-decoders": ["Decoders:", " V..... = Video", " ------"]
        + [f" V....D {name:<20} A codec." for name in decoders],
        "-muxers": ["Formats:", " D. = Demuxing supported", " .E = Muxing supported", " ---"]
        + [f"  E  {name:<16} A format." for name in muxers],
        "-version": ["ffmpeg version n0.0-fake Copyright (c) 2000-2026 the FFmpeg developers"],
    }
    cases = "".join(
        f"  {option}) printf '%s\\n' {' '.join(repr(line) for line in lines)} ;;\n"
        for option, lines in lists.items()
    )
    shown = " ".join(repr(line) for line in setparams_help(setparams))
    cases += f"  -h) [ \"$3\" = filter=setparams ] || exit 1; printf '%s\\n' {shown} ;;\n"
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


WITHOUT_ZSCALE = tuple(name for name in FILTERS if name != "zscale")


def test_without_zscale_says_how_to_get_it(build: Path) -> None:
    fake_build(build, filters=WITHOUT_ZSCALE)
    with pytest.raises(MediaError, match=r"lacks the zscale filter\. seedvr2x needs ffmpeg 7\.1"):
        ffmpeg.check()


# The refusal of a build without setparams' chroma_location, after its ffmpeg's path.
WITHOUT_OPTION = (
    " lacks the chroma_location option of its setparams filter (ffmpeg 7.1 and later have it)."
    " seedvr2x needs ffmpeg 7.1 or later with zscale, which comes with libzimg (ffmpeg configured"
    " with --enable-libzimg): the builds of https://github.com/BtbN/FFmpeg-Builds will do, and so"
    " will Ubuntu's ffmpeg package from 25.04 on (24.04's is 6.1)."
)


def test_without_chroma_location(build: Path) -> None:
    # setparams' chroma_location came with ffmpeg 7.1 (ffmpeg.OPTIONS): checked as the filter's
    # help lists it, never by the version, a build without it is refused, saying which ffmpeg to
    # get; without zscale too, both are said, and which ffmpeg to get once.
    fake_build(build, setparams=WITHOUT_CHROMA_LOCATION)
    with pytest.raises(MediaError) as refused:
        ffmpeg.check()
    assert str(refused.value) == f"{build / 'ffmpeg'}{WITHOUT_OPTION}"
    fake_build(build, filters=WITHOUT_ZSCALE, setparams=WITHOUT_CHROMA_LOCATION)
    with pytest.raises(MediaError) as refused:
        ffmpeg.check()
    both = WITHOUT_OPTION.replace(" lacks ", " lacks the zscale filter, ", 1)
    assert str(refused.value) == f"{build / 'ffmpeg'}{both}"


# The command line, exiting 4 should anything after the build's check run: its conversions'
# fingerprint, the source's probe, the first pass or the model files' check; 3 once torch is
# imported. A failed assert would exit 1, as a refusal does (test_cli.py's EARLY).
AFTER_CHECK = """
import sys

from seedvr2x import cli
from seedvr2x.media import fingerprint, source
from seedvr2x.runtime import weights


def ran(*args, **kwargs):
    sys.exit(4)


fingerprint.fingerprint = source.declare = source.first_pass = weights.check_models = ran
status = cli.main(sys.argv[1:])
sys.exit(3 if "torch" in sys.modules else status)
"""


def test_refused_before_anything(tmp_path: Path) -> None:
    # The build is checked first, but for a directory given as the source (cli._run): a build
    # without setparams' chroma_location is refused before the fingerprint, which would fail on
    # it with ffmpeg's own error, the source, the first pass, the model files and torch, exit 1,
    # nothing written.
    build = tmp_path / "build"
    build.mkdir()
    fake_build(build, setparams=WITHOUT_CHROMA_LOCATION)
    source = tmp_path / "in.mkv"
    source.write_bytes(b"")
    files = sorted(tmp_path.rglob("*"))
    for output in ("out", "out.mkv"):
        args = [str(source), "-o", str(tmp_path / output), "--model-dir", str(tmp_path)]
        result = subprocess.run(
            [sys.executable, "-c", AFTER_CHECK, *args],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PATH": str(build)},
        )
        assert result.returncode == 1, result.stderr
        assert f"{build / 'ffmpeg'}{WITHOUT_OPTION}\n" in result.stderr
        assert sorted(tmp_path.rglob("*")) == files


def without(names: tuple[str, ...], name: str) -> tuple[str, ...]:
    return tuple(other for other in names if other != name)


@pytest.mark.parametrize(
    ("filters", "encoders", "decoders", "muxers", "missing"),
    [
        # Each one seedvr2x's own commands use (ffmpeg.FILTERS, ENCODERS, DECODERS, MUXERS):
        # idet, which the first pass runs, select and split, which the reads use, the null graph
        # of an output without filters; rawvideo, the frames raw on a pipe or hashed;
        # wrapped_avframe and null, idet's output; framehash, the frame index's CRC-32s.
        *(
            (without(FILTERS, name), ENCODERS, DECODERS, MUXERS, f"the {name} filter")
            for name in ("idet", "select", "split", "scale", "null")
        ),
        (FILTERS, ("png", "rawvideo", "wrapped_avframe"), DECODERS, MUXERS, "the ffv1 encoder"),
        *(
            (FILTERS, without(ENCODERS, name), DECODERS, MUXERS, f"the {name} encoder")
            for name in ("rawvideo", "wrapped_avframe")
        ),
        (FILTERS, ENCODERS, ("h264", "rawvideo"), MUXERS, "the ffv1 decoder"),
        (FILTERS, ENCODERS, ("ffv1", "h264"), MUXERS, "the rawvideo decoder"),
        *(
            (FILTERS, ENCODERS, DECODERS, without(MUXERS, name), f"the {name} muxer")
            for name in ("framehash", "null", "rawvideo", "matroska")
        ),
    ],
)
def test_incomplete_build(
    build: Path,
    filters: tuple[str, ...],
    encoders: tuple[str, ...],
    decoders: tuple[str, ...],
    muxers: tuple[str, ...],
    missing: str,
) -> None:
    fake_build(build, filters, encoders, decoders, muxers=muxers)
    with pytest.raises(MediaError, match=f"lacks {missing}") as refused:
        ffmpeg.check()
    assert "libzimg" not in str(refused.value)


def test_scdet_no_longer_needed(build: Path) -> None:
    # TransNetV2 detects the cuts (DESIGN.md, Shot detection): a build without scdet runs; nor
    # are settb and metadata, the first pass's before it read the decoder's timestamps, needed.
    assert not {"scdet", "settb", "metadata"} & set(ffmpeg.FILTERS)
    fake_build(build)
    assert ffmpeg.check() == "n0.0-fake"


def test_without_ffprobe(build: Path) -> None:
    fake_build(build, ffprobe=False)
    with pytest.raises(MediaError, match="ffprobe not found"):
        ffmpeg.check()


def test_png_encoder_and_muxer_needed_only_for_png_output(build: Path) -> None:
    fake_build(build, encoders=without(ENCODERS, "png"), muxers=without(MUXERS, "image2"))
    assert ffmpeg.check() == "n0.0-fake"
    with pytest.raises(MediaError, match="lacks the png encoder, the image2 muxer"):
        ffmpeg.check(("png",), ("image2",))
