"""The decode's parameters and refusals, from what a stream declares: no ffmpeg needed."""

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from seedvr2x.media.conversion import conversion_for, guess_matrix
from seedvr2x.media.decode import to_float32
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.probe import VideoStream
from seedvr2x.media.scan import Scan, timing_error
from seedvr2x.media.source import declared_refusal

NTSC_FILM = Fraction(24000, 1001)
UNTAGGED_HD = VideoStream(
    width=1920,
    height=1080,
    pix_fmt="yuv420p",
    frame_rate=NTSC_FILM,
    avg_frame_rate=NTSC_FILM,
    field_order="progressive",
    sample_aspect=Fraction(1),
    color_space="",
    color_range="",
    color_primaries="",
    color_transfer="",
    chroma_location="",
    display_matrix=None,
    cropped=False,
)
ROTATED_90 = (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30)
FLIPPED = (-65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)
UNROTATED = (65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)


def stream(**changes: Any) -> VideoStream:
    return dataclasses.replace(UNTAGGED_HD, **changes)


def test_8bit_through_16_bits_is_numz_read() -> None:
    # zscale expands 8 bits to v * 257 exactly (test_decode.py); v * 257 / 65535 in float32 must
    # be numz's read, uint8 / 255 in float32, bit for bit, or milestone 1 breaks.
    v = np.arange(256, dtype=np.uint16)
    ours = to_float32(v * np.uint16(257))
    numz = v.astype(np.float32) / np.float32(255)
    assert np.array_equal(ours.view(np.uint32), numz.view(np.uint32))


@pytest.mark.parametrize(
    ("width", "height", "matrix"),
    [
        (1920, 1080, "bt709"),
        (1280, 720, "bt709"),
        (1280, 534, "bt709"),
        (720, 578, "bt709"),
        (1024, 576, "smpte170m"),
        (960, 540, "smpte170m"),
        (720, 576, "smpte170m"),
        (720, 480, "smpte170m"),
    ],
)
def test_untagged_matrix_is_mpv_guess(width: int, height: int, matrix: str) -> None:
    assert guess_matrix(width, height) == matrix


def test_untagged_yuv_guesses_matrix_range_and_siting() -> None:
    conversion = conversion_for(stream())
    assert (conversion.matrix, conversion.color_range, conversion.chroma_location) == (
        "709",
        "limited",
        "left",
    )
    assert len(conversion.guessed) == 3
    assert conversion_for(stream(width=720, height=480)).matrix == "170m"


def test_tags_are_taken() -> None:
    conversion = conversion_for(
        stream(color_space="smpte170m", color_range="pc", chroma_location="center")
    )
    assert conversion == dataclasses.replace(
        conversion, planar="yuv420p", matrix="170m", color_range="full", chroma_location="center"
    )
    assert conversion.guessed == ()


def test_matrix_override() -> None:
    assert conversion_for(stream(color_space="bt709"), "bt2020nc").matrix == "2020_ncl"
    assert conversion_for(stream(), "bt470bg").guessed == ("limited range", "chroma sited left")


def test_siting_is_no_guess_without_subsampling() -> None:
    conversion = conversion_for(
        stream(pix_fmt="yuv444p10le", color_space="bt709", color_range="tv")
    )
    assert conversion.guessed == ()


def test_jpeg_family_is_full_range() -> None:
    conversion = conversion_for(
        stream(
            pix_fmt="yuvj420p", color_space="bt470bg", color_range="pc", chroma_location="center"
        )
    )
    assert (conversion.planar, conversion.matrix, conversion.color_range) == (
        "yuvj420p",
        "470bg",
        "full",
    )
    assert conversion_for(stream(pix_fmt="yuvj422p", color_space="bt470bg")).color_range == "full"


@pytest.mark.parametrize(("pix_fmt", "planar"), [("bgr0", "gbrp"), ("rgb48be", "gbrp16le")])
def test_rgb_is_full_range_identity(pix_fmt: str, planar: str) -> None:
    conversion = conversion_for(stream(pix_fmt=pix_fmt, color_space="gbr"))
    assert (conversion.planar, conversion.matrix, conversion.color_range) == (planar, "gbr", "full")
    assert conversion.guessed == ()


@pytest.mark.parametrize(
    ("changes", "matrix", "reason"),
    [
        ({"pix_fmt": "yuva420p"}, None, "alpha"),
        ({"pix_fmt": "bgra"}, None, "alpha"),
        ({"pix_fmt": "gbrap16le"}, None, "alpha"),
        ({"pix_fmt": "pal8"}, None, "palette"),
        ({"pix_fmt": "gray"}, None, "grey"),
        ({"pix_fmt": "gray16le"}, None, "grey"),
        ({"pix_fmt": "xyz12le"}, None, "not supported"),
        ({"color_space": "bt2020c"}, None, "matrix bt2020c"),
        ({"color_space": "fcc"}, None, "matrix fcc"),
        ({"color_space": "ictcp"}, None, "matrix ictcp"),
        ({"color_space": "gbr"}, None, "matrix gbr"),
        ({"pix_fmt": "gbrp", "color_range": "tv"}, None, "RGB tagged limited"),
        ({"pix_fmt": "gbrp"}, "bt709", "RGB, it has no matrix"),
        ({"pix_fmt": "yuvj420p", "color_range": "tv"}, None, "full range by definition"),
    ],
)
def test_refused_conversions(changes: dict[str, Any], matrix: str | None, reason: str) -> None:
    with pytest.raises(MediaError, match=reason):
        conversion_for(stream(**changes), matrix)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"field_order": "tt"}, "interlaced"),
        ({"field_order": "bb"}, "interlaced"),
        ({"field_order": "tb"}, "interlaced"),
        ({"field_order": "bt"}, "interlaced"),
        ({"display_matrix": ROTATED_90}, "rotated or flipped"),
        ({"display_matrix": FLIPPED}, "rotated or flipped"),
        ({"cropped": True}, "cropped"),
    ],
)
def test_declared_refusals(changes: dict[str, Any], reason: str) -> None:
    assert reason in declared_refusal(stream(**changes))


@pytest.mark.parametrize(
    "changes", [{}, {"field_order": ""}, {"display_matrix": UNROTATED}], ids=str
)
def test_declared_accepted(changes: dict[str, Any]) -> None:
    assert declared_refusal(stream(**changes)) == ""


@pytest.mark.parametrize(
    ("scanned", "avg_frame_rate", "reason"),
    [
        (Scan(100, 99, 41000, 42000), NTSC_FILM, ""),  # Matroska's milliseconds at 23.976 fps
        (Scan(100, 99, 41708, 41709), NTSC_FILM, ""),
        (Scan(100, 98, 33333, 41667), NTSC_FILM, "variable frame rate"),  # 24 and 30 fps
        (Scan(100, 99, 40000, 42000), NTSC_FILM, "variable frame rate"),  # 24 and 25 fps
        (Scan(100, 99, 40000, 40000), NTSC_FILM, "not the frames' own"),  # declared wrong
        (Scan(1, 0, 0, 0), NTSC_FILM, ""),  # one frame: the declared rates agree
        (Scan(1, 0, 0, 0), Fraction(24), "can't be checked"),
        (Scan(1, 0, 0, 0), None, "can't be checked"),
        (Scan(0, 0, 0, 0), NTSC_FILM, "no frame"),
    ],
)
def test_timing(scanned: Scan, avg_frame_rate: Fraction | None, reason: str) -> None:
    error = timing_error(scanned, NTSC_FILM, avg_frame_rate)
    assert reason in error if reason else error == ""
