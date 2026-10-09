"""The decode's parameters and refusals, from what a stream declares: no ffmpeg needed."""

import dataclasses
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from test_probe import NO_RECORD, TONE_MAP, WHY_HLG, WHY_PQ

from seedvr2x.media.conversion import conversion_for, guess_matrix
from seedvr2x.media.decode import to_float32
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.probe import DoviRecord, FirstFrame, VideoStream
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
    dovi=None,
    first_frame=FirstFrame(color_transfer="", dovi=False),
)
ROTATED_90 = (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30)
FLIPPED = (-65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)
UNROTATED = (65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30)


def dovi(profile: int, compatibility: int, el: bool = False, bl: bool = True) -> DoviRecord:
    """A Dolby Vision configuration record, its RPU present."""
    return DoviRecord(profile, True, el, bl, compatibility)


def first(transfer: str = "", rpu: bool = False) -> FirstFrame:
    """A first frame tagged with that transfer, Dolby Vision's metadata on it or not."""
    return FirstFrame(transfer, rpu)


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
        ({"color_transfer": "smpte2084"}, "HDR, transfer smpte2084 (PQ): not supported"),
        ({"color_transfer": "arib-std-b67"}, "HDR, transfer arib-std-b67 (HLG): not supported"),
        # An SDR base layer read through, but for a stream tagged HDR all the same.
        ({"dovi": dovi(8, 2), "color_transfer": "smpte2084"}, "HDR, transfer smpte2084 (PQ)"),
        ({"dovi": dovi(8, 2), "color_transfer": "arib-std-b67"}, "HDR, transfer arib-std-b67"),
        # The record refuses an HDR base layer, untagged as it may be.
        ({"dovi": dovi(8, 1)}, "Dolby Vision profile 8, its base layer HDR10 (PQ): not supported"),
        ({"dovi": dovi(7, 6, el=True)}, "profile 7, its base layer UHD Blu-ray's HDR10 (PQ)"),
        ({"dovi": dovi(7, 6, el=True, bl=False)}, "Dolby Vision profile 7 with no base layer"),
        ({"dovi": dovi(8, 3)}, "profile 8, its base layer of compatibility id 3, unknown"),
        # No base layer, whatever the id says: a profile 4 enhancement layer's track.
        (
            {"dovi": dovi(4, 2, el=True, bl=False)},
            "Dolby Vision profile 4 with no base layer: not supported: only an SDR base layer is"
            f" read; {TONE_MAP}",
        ),
        # Id 0 is Dolby's own IPTPQc2 for profiles 5 and 10 alone (source.DOLBYS_OWN).
        (
            {"dovi": dovi(5, 0)},
            "Dolby Vision profile 5, its base layer Dolby's own, viewable only through Dolby's"
            f" processing: not supported: the model was trained on SDR video; {TONE_MAP}",
        ),
        ({"dovi": dovi(10, 0)}, "profile 10, its base layer Dolby's own"),
        (
            {"dovi": dovi(8, 0)},
            "Dolby Vision profile 8, its base layer compatible with no other display (id 0): not"
            f" supported: only an SDR base layer, id 2, is read; {TONE_MAP}",
        ),
        ({"dovi": dovi(7, 0, el=True)}, "profile 7, its base layer compatible with no other"),
        # The first frame's transfer, which the container's tags can hide (media/probe.py).
        (
            {"first_frame": first("smpte2084")},
            "HDR, transfer smpte2084 (PQ) on its first frame, where the stream declares none: not"
            f" supported: {WHY_PQ}; {TONE_MAP}",
        ),
        (
            {"color_transfer": "bt709", "first_frame": first("arib-std-b67")},
            "HDR, transfer arib-std-b67 (HLG) on its first frame, where the stream declares bt709:"
            f" not supported: {WHY_HLG}; {TONE_MAP}",
        ),
        # Dolby Vision's metadata without a record, whatever the tags (provisional,
        # source.NO_RECORD).
        ({"first_frame": first(rpu=True)}, NO_RECORD),
        ({"color_transfer": "bt709", "first_frame": first("bt709", rpu=True)}, NO_RECORD),
    ],
)
def test_declared_refusals(changes: dict[str, Any], reason: str) -> None:
    assert reason in declared_refusal(stream(**changes))


PQ = f"HDR, transfer smpte2084 (PQ): not supported: {WHY_PQ}; {TONE_MAP}"
HLG = f"HDR, transfer arib-std-b67 (HLG): not supported: {WHY_HLG}; {TONE_MAP}"


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        # The record's refusal, which names the profile, over the transfer's and the metadata's.
        (
            {"dovi": dovi(8, 1), "color_transfer": "smpte2084", "first_frame": first(rpu=True)},
            f"Dolby Vision profile 8, its base layer HDR10 (PQ): not supported: {WHY_PQ};"
            f" {TONE_MAP}",
        ),
        # The stream's transfer over its first frame's.
        ({"color_transfer": "smpte2084", "first_frame": first("arib-std-b67")}, PQ),
        # An HDR transfer over Dolby Vision's metadata without a record, which adds nothing.
        ({"color_transfer": "arib-std-b67", "first_frame": first("arib-std-b67", True)}, HLG),
        (
            {"first_frame": first("smpte2084", rpu=True)},
            "HDR, transfer smpte2084 (PQ) on its first frame, where the stream declares none: not"
            f" supported: {WHY_PQ}; {TONE_MAP}",
        ),
    ],
)
def test_one_reason_for_the_pixels(changes: dict[str, Any], refusal: str) -> None:
    assert declared_refusal(stream(**changes)) == refusal


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        # An interlaced HDR source is told both, and so never tone-maps in vain. Numbered, since
        # a reason can hold "; " itself (TONE_MAP's).
        (
            {"field_order": "tt", "color_transfer": "smpte2084"},
            f"2 reasons: (1) interlaced (field order tt): not supported; (2) {PQ}",
        ),
        # An iPhone's HLG, Dolby Vision 8.4, rotated (ffmpeg's FATE sample hevc/dv84.mov).
        (
            {"dovi": dovi(8, 4), "color_transfer": "arib-std-b67", "display_matrix": ROTATED_90},
            "2 reasons: (1) Dolby Vision profile 8, its base layer HLG: not supported:"
            f" {WHY_HLG}; {TONE_MAP}; (2) rotated or flipped by its display matrix {ROTATED_90}:"
            " not supported",
        ),
        (
            {"display_matrix": FLIPPED, "cropped": True},
            f"2 reasons: (1) rotated or flipped by its display matrix {FLIPPED}: not supported;"
            " (2) cropped by its container: not supported",
        ),
        # Every one, in their order.
        (
            {
                "cropped": True,
                "display_matrix": ROTATED_90,
                "first_frame": first(rpu=True),
                "field_order": "bb",
            },
            f"4 reasons: (1) interlaced (field order bb): not supported; (2) {NO_RECORD}; (3)"
            f" rotated or flipped by its display matrix {ROTATED_90}: not supported; (4) cropped by"
            " its container: not supported",
        ),
    ],
)
def test_every_reason_in_one_refusal(changes: dict[str, Any], refusal: str) -> None:
    assert declared_refusal(stream(**changes)) == refusal


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"field_order": ""},
        {"display_matrix": UNROTATED},
        *({"color_transfer": sdr} for sdr in ("bt709", "smpte170m", "bt2020-10", "iec61966-2-1")),
        # Read through an SDR base layer, whatever the rest of the stream holds.
        {"dovi": dovi(8, 2), "color_transfer": "bt709"},
        {"dovi": dovi(9, 2)},
        {"dovi": dovi(4, 2, el=True)},
        # The first frame SDR, Dolby Vision's metadata on it with a record saying SDR.
        {"first_frame": first("bt709")},
        {"color_transfer": "bt2020-10", "first_frame": first("bt2020-10")},
        {"dovi": dovi(8, 2), "color_transfer": "bt709", "first_frame": first("bt709", rpu=True)},
    ],
    ids=str,
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
