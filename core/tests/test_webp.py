"""The lossless WebP writer of a still (seedvr2x/webp.py; DESIGN.md, Still images, Output): 8 bits
exact in Pillow's decode and ffmpeg's, grey as three equal planes, the profile and the orientation
carried, no EXIF at orientation 1, each refusal by its words, the same bytes every time, and the
file read back by picture.read. Uses test_png.py's pictures and ffmpeg check."""

import io
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from test_png import COLOUR, PICTURE, PROFILE, chunk, ffmpeg_decode, needs_ffmpeg

from seedvr2x import picture, png, webp
from seedvr2x.errors import Refusal

RNG = np.random.default_rng(20261104)

# What write() gives by default, a picture with nothing to carry.
NOTHING: dict[str, object] = {"grey": False, "icc": None, "chunks": (), "orientation": 1}


def write(rgb: npt.NDArray[np.float32] = PICTURE, **options: object) -> bytes:
    return webp.write(rgb, **{**NOTHING, **options})  # pyright: ignore[reportArgumentType]


def decoded(data: bytes) -> npt.NDArray[np.uint8]:
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "WEBP" and image.mode == "RGB"
        return np.asarray(image)


def eight(rgb: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """The 8-bit codes of a picture, as png.write's 8-bit path rounds them."""
    return np.rint(np.clip(rgb, 0, 1) * np.float32(255)).astype(np.uint8)


# The pixels ------------------------------------------------------------------------------------


def test_the_codes_are_the_8_bit_ones_exactly() -> None:
    assert np.array_equal(decoded(write()), eight(PICTURE))
    # Every code of the 8-bit grid survives, at its middle and at its rounding edges.
    ramp = (np.arange(256, dtype=np.float32) / np.float32(255)).reshape(16, 16, 1).repeat(3, 2)
    assert np.array_equal(decoded(write(ramp)), eight(ramp))
    assert np.array_equal(eight(ramp)[..., 0].ravel(), np.arange(256))


def test_the_quantisation_is_pngs() -> None:
    rgb = RNG.random((9, 13, 3), dtype=np.float32) * 3 - 1  # beyond [0, 1] on both sides
    expected = png.quantised(rgb, 8, False)
    assert np.array_equal(decoded(write(rgb)), expected)
    assert np.array_equal(decoded(write(rgb)), eight(rgb))


def test_clipping_never_wraps() -> None:
    rgb = np.array([[[-1e9, 2.0, 1e30], [0.0, 1.0, 0.5]]], np.float32)
    assert decoded(write(rgb)).tolist() == [[[0, 255, 255], [0, 255, 128]]]


@pytest.mark.parametrize("layout", ["fortran", "slice", "transposed"])
def test_a_non_contiguous_input_is_written_as_the_same_picture(layout: str) -> None:
    if layout == "fortran":
        odd = np.asfortranarray(PICTURE)
    elif layout == "slice":
        odd = np.repeat(PICTURE, 2, axis=1)[:, ::2]
    else:
        odd = np.ascontiguousarray(PICTURE.transpose(1, 0, 2)).transpose(1, 0, 2)
    assert not odd.flags["C_CONTIGUOUS"]
    assert write(odd) == write()


def test_the_input_is_left_alone() -> None:
    rgb = RNG.random((5, 7, 3), dtype=np.float32) * 2 - 0.5
    kept = rgb.copy()
    write(rgb, grey=True)
    assert np.array_equal(rgb, kept)


@needs_ffmpeg
def test_ffmpeg_decodes_the_same_codes(tmp_path: Path) -> None:
    rgb = RNG.random((11, 17, 3), dtype=np.float32)
    path = tmp_path / "out.webp"
    path.write_bytes(write(rgb, icc=PROFILE, orientation=3))
    assert np.array_equal(ffmpeg_decode(path, "rgb24"), eight(rgb).ravel())


# Grey ------------------------------------------------------------------------------------------


def test_grey_is_written_as_three_equal_planes_of_the_mean() -> None:
    rgb = RNG.random((11, 17, 3), dtype=np.float32)
    rgb[0, 0] = (0.0, 0.3, 0.6)
    rgb[0, 1] = (-5.0, 1.0, 9.0)  # the mean first, then the clip
    out = decoded(write(rgb, grey=True))
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()
    mean = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / np.float32(3)
    assert np.array_equal(out[..., 0], png.quantised(mean[..., np.newaxis], 8, False)[..., 0])
    assert np.array_equal(out[..., 0], png.quantised(rgb, 8, True)[..., 0])
    assert out[0, 0, 0] == round(0.3 * 255) and out[0, 1, 0] == 255


def test_a_grey_comes_back_equal_through_the_reader() -> None:
    plane = RNG.random((11, 17, 1), dtype=np.float32)
    grey = write(np.repeat(plane, 3, axis=2), grey=True)
    read = picture.read(grey, "grey.webp")
    # WebP has no grey mode: RGB with equal planes, equal to the 8-bit codes of the plane.
    assert read.bits == 8 and not read.grey
    codes = np.rint(read.pixels * np.float32(255)).astype(np.uint8)
    assert np.array_equal(codes[..., 0], eight(plane)[..., 0])
    assert (codes[..., 0] == codes[..., 1]).all() and (codes[..., 0] == codes[..., 2]).all()


# The profile and the orientation ----------------------------------------------------------------


def test_the_profile_is_carried_whole() -> None:
    with Image.open(io.BytesIO(write(icc=PROFILE))) as image:
        assert image.info["icc_profile"] == PROFILE
    with Image.open(io.BytesIO(write())) as image:
        assert "icc_profile" not in image.info


@pytest.mark.parametrize("value", range(2, 9))
def test_the_orientation_is_carried_in_an_exif_block_that_holds_it_alone(value: int) -> None:
    data = write(orientation=value)
    with Image.open(io.BytesIO(data)) as image:
        assert dict(image.getexif()) == {0x0112: value}
        # The EXIF chunk holds png.exif's TIFF, with no "Exif\0\0" header (the WebP container's).
        assert image.info["exif"] == png.exif(value)
    # The pixels are not rotated.
    assert np.array_equal(decoded(data), eight(PICTURE))


def test_no_exif_at_orientation_1() -> None:
    data = write(orientation=1)
    with Image.open(io.BytesIO(data)) as image:
        assert "exif" not in image.info and len(image.getexif()) == 0
    assert b"EXIF" not in data


def test_the_profile_and_the_orientation_together() -> None:
    with Image.open(io.BytesIO(write(icc=PROFILE, orientation=6))) as image:
        assert image.info["icc_profile"] == PROFILE and image.getexif()[0x0112] == 6


def test_the_srgb_chunk_alone_is_taken_as_nothing() -> None:
    srgb = (chunk(b"sRGB", b"\x00"),)
    assert write(chunks=srgb) == write()
    webp.check(8, None, srgb)


# Refusals ----------------------------------------------------------------------------------------


def test_a_picture_stored_at_16_bits_is_refused_naming_png() -> None:
    with pytest.raises(Refusal) as refused:
        webp.check(16, None, ())
    assert str(refused.value) == (
        "a lossless WebP holds 8 bits per sample and this picture is stored at 16, which it would"
        " round away: write a PNG (a .png file), which holds them"
    )
    assert webp.why_not(16, PROFILE, ()) == str(refused.value)
    assert webp.why_not(8, None, ()) is None


@pytest.mark.parametrize(
    ("chunks", "named"),
    [
        ((COLOUR[0],), "cICP"),
        ((COLOUR[1],), "gAMA"),
        ((COLOUR[2],), "cHRM"),
        (COLOUR, "cICP, cHRM, gAMA"),
        ((COLOUR[3], COLOUR[1]), "gAMA"),
    ],
)
def test_colour_chunks_without_a_profile_are_refused_naming_png(
    chunks: tuple[bytes, ...], named: str
) -> None:
    words = (
        f"a lossless WebP holds no {named} colour chunk and this picture has no ICC profile to"
        " carry its colour instead, so its colour would change: write a PNG (a .png file), which"
        " holds them"
    )
    with pytest.raises(Refusal) as refused:
        webp.check(8, None, chunks)
    assert str(refused.value) == words
    # The writer itself refuses it too: it never writes a file with the colour tags lost.
    with pytest.raises(Refusal) as again:
        write(chunks=chunks)
    assert str(again.value) == words


def test_colour_chunks_with_a_profile_are_taken_as_the_profile() -> None:
    assert webp.why_not(8, PROFILE, COLOUR) is None
    with Image.open(io.BytesIO(write(icc=PROFILE, chunks=COLOUR))) as image:
        assert image.info["icc_profile"] == PROFILE


def test_a_chunk_is_told_by_its_type_not_by_its_bytes() -> None:
    # A text chunk holding the word cICP in its data is no cICP chunk.
    text = chunk(b"tEXt", b"cICP\x00gAMA")
    assert webp.why_not(8, None, (text,)) is None


def test_a_side_over_webps_limit_is_refused_naming_png() -> None:
    wide = np.zeros((1, webp.MAX_SIDE + 1, 3), np.float32)
    with pytest.raises(Refusal, match=r"at most 16,383 pixels on a side.*16384x1.*write a PNG"):
        write(wide)
    assert decoded(write(wide[:, :-1])).shape == (1, webp.MAX_SIDE, 3)


@pytest.mark.parametrize(
    "arguments",
    [
        {"rgb": np.zeros((4, 4), np.float32)},
        {"rgb": np.zeros((4, 4, 4), np.float32)},
        {"rgb": np.zeros((0, 4, 3), np.float32)},
        {"orientation": 0},
        {"orientation": 9},
    ],
)
def test_a_bad_argument_is_a_value_error(arguments: dict[str, object]) -> None:
    settings: dict[str, object] = {"rgb": PICTURE, **NOTHING, **arguments}
    with pytest.raises(ValueError):
        webp.write(**settings)  # pyright: ignore[reportArgumentType]


# Bytes -------------------------------------------------------------------------------------------


def test_the_same_bytes_every_time() -> None:
    options = {"icc": PROFILE, "orientation": 8}
    assert write(**options) == write(**options)
    assert write(RNG.random((6, 6, 3), dtype=np.float32)) != write()


def test_the_file_is_lossless_webp_of_the_chosen_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    data = write()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP" and b"VP8L" in data[:20]
    assert (webp.METHOD, webp.QUALITY) == (4, 50)
    # The effort is the constants': another is another file, the same pixels.
    rgb = RNG.random((64, 64, 3), dtype=np.float32)
    chosen = write(rgb)
    monkeypatch.setattr(webp, "METHOD", 0)
    monkeypatch.setattr(webp, "QUALITY", 0)
    other = write(rgb)
    assert other != chosen and np.array_equal(decoded(other), decoded(chosen))


def test_a_smooth_picture_is_smaller_than_the_png_of_the_same_8_bits() -> None:
    y, x = np.mgrid[0:96, 0:128]
    rgb = np.stack([x / 128, y / 96, (x + y) / 224], axis=2).astype(np.float32)
    assert len(write(rgb)) < len(
        png.write(rgb, bits=8, grey=False, icc=None, chunks=(), orientation=1)
    )


# Read back by seedvr2x's reader -----------------------------------------------------------------


@pytest.mark.parametrize("orientation", [1, 6])
@pytest.mark.parametrize("icc", [None, PROFILE])
def test_the_reader_gives_back_the_same_pixels(orientation: int, icc: bytes | None) -> None:
    data = write(icc=icc, orientation=orientation)
    read = picture.read(data, "out.webp")
    assert read.format == "WEBP" and read.bits == 8 and not read.grey
    assert read.icc == icc and read.orientation == orientation and read.chunks == ()
    assert not read.opaque_alpha_dropped
    assert read.pixels.shape == PICTURE.shape and read.pixels.dtype == np.float32
    assert np.array_equal(np.rint(read.pixels * np.float32(255)).astype(np.uint8), eight(PICTURE))
    # And written again, the same file.
    assert write(read.pixels, icc=read.icc, orientation=read.orientation) == data


def test_a_16_bit_picture_the_reader_gives_is_refused_for_webp() -> None:
    sixteen = png.write(PICTURE, bits=16, grey=False, icc=None, chunks=(COLOUR[3],), orientation=1)
    read = picture.read(sixteen, "in.png")
    assert read.bits == 16
    with pytest.raises(Refusal, match="stored at 16"):
        webp.check(read.bits, read.icc, read.chunks)
    # Its sRGB chunk alone would have been taken.
    assert webp.why_not(8, read.icc, read.chunks) is None


def test_a_png_with_cicp_and_no_profile_is_read_with_the_chunk_that_webp_refuses() -> None:
    data = png.write(PICTURE, bits=8, grey=False, icc=None, chunks=(COLOUR[0],), orientation=1)
    read = picture.read(data, "in.png")
    assert webp.why_not(read.bits, read.icc, read.chunks) is not None
