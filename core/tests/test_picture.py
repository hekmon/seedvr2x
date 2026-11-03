"""A still picture read from bytes (seedvr2x/picture.py; DESIGN.md, Still images): what a file is
by its content, every format and mode read, the values exact (against the arrays written, and
against ffmpeg's decode where ffmpeg is 7.1 or later), the metadata carried, and every refusal
with its way out, the commands in them run. The files are made here: Pillow writes most, and a
16-bit PNG is written by hand, so that the reader is held to bytes written apart from it."""

import io
import json
import logging
import re
import shlex
import shutil
import struct
import subprocess
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageCms

from seedvr2x import picture
from seedvr2x.errors import Refusal


def _ffmpeg_reason() -> str | None:
    """Why the tests against ffmpeg's decode are skipped, None when they run."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return "ffmpeg and ffprobe are not on PATH"
    banner = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    found = re.search(r"ffmpeg version n?(\d+)\.(\d+)", banner)
    if found is None:
        return f"ffmpeg's version is not a release: {banner.splitlines()[:1]}"
    if (int(found[1]), int(found[2])) < (7, 1):
        return f"ffmpeg {found[1]}.{found[2]} is older than 7.1"
    return None


FFMPEG_REASON = _ffmpeg_reason()
needs_ffmpeg = pytest.mark.skipif(FFMPEG_REASON is not None, reason=str(FFMPEG_REASON))

# ffmpeg's conversion of a 4:2:0 picture upsamples the chroma with a bicubic filter that
# overshoots at sharp edges (161 levels on a test pattern's); with these flags it is bilinear, as
# libjpeg-turbo's and libwebp's are, and the decodes agree within a level or two.
BILINEAR = "scale=flags=bilinear+full_chroma_int+accurate_rnd"

LOGGER = "seedvr2x.picture"

RNG = np.random.default_rng(20261103)
RGB8 = RNG.integers(0, 256, (16, 24, 3), dtype=np.uint8)
GREY8 = RNG.integers(0, 256, (16, 24), dtype=np.uint8)
RGB16 = RNG.integers(0, 65536, (16, 24, 3), dtype=np.uint16)
GREY16 = RNG.integers(0, 65536, (16, 24), dtype=np.uint16)
# A smooth picture for the lossy formats: noise would be 4:2:0's and quantisation's own business.
_Y, _X = np.mgrid[0:32, 0:48]
SMOOTH = np.stack(
    [60 + _X * 2 + _Y, 90 + 50 * np.sin(_X / 7) + _Y * 2, 200 - _X - _Y * 2], axis=2
).astype(np.uint8)

PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# Making files ---------------------------------------------------------------------------------


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def raw_png(
    width: int,
    height: int,
    depth: int,
    colour_type: int,
    rows: list[bytes],
    before: tuple[bytes, ...] = (),
    after: tuple[bytes, ...] = (),
) -> bytes:
    """A PNG written by hand, every row unfiltered; before are whole chunks placed between IHDR
    and IDAT, after between IDAT and IEND."""
    header = struct.pack(">IIBBBBB", width, height, depth, colour_type, 0, 0, 0)
    image = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + b"".join(before)
        + chunk(b"IDAT", zlib.compress(image))
        + b"".join(after)
        + chunk(b"IEND", b"")
    )


def png16(samples: npt.NDArray[np.uint16], colour_type: int, **chunks: tuple[bytes, ...]) -> bytes:
    """A 16-bit PNG of (H, W) or (H, W, C) samples, big-endian."""
    shaped = samples.reshape(*samples.shape[:2], -1)
    rows = [row.astype(">u2").tobytes() for row in shaped.reshape(shaped.shape[0], -1)]
    return raw_png(shaped.shape[1], shaped.shape[0], 16, colour_type, rows, **chunks)


def pillow(image: Image.Image, fmt: str, **options: object) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **options)
    return out.getvalue()


def bmp_v4(bgra: npt.NDArray[np.uint8]) -> bytes:
    """A 32-bit BMP with a V4 header and an alpha mask: the kind of BMP whose fourth byte is alpha
    (a BI_RGB one's is unused, and Pillow reads it as RGB)."""
    height, width = bgra.shape[:2]
    pixels = bgra.tobytes()
    masks = struct.pack("<IIII", 0xFF0000, 0xFF00, 0xFF, 0xFF000000)
    header = struct.pack(
        "<IiiHHIIiiII", 108, width, -height, 1, 32, 3, len(pixels), 2835, 2835, 0, 0
    )
    header += masks + struct.pack("<I", 0x73524742) + bytes(36) + struct.pack("<III", 0, 0, 0)
    return b"BM" + struct.pack("<IHHI", 14 + 108 + len(pixels), 0, 0, 14 + 108) + header + pixels


def inserted(png: bytes, *chunks: bytes) -> bytes:
    """png with the chunks placed right after its IHDR."""
    return png[:33] + b"".join(chunks) + png[33:]


def orientation_exif(value: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif


def codes(pixels: npt.NDArray[np.float32], bits: int) -> npt.NDArray[np.int64]:
    """The codes a writer would make of the pixels: media/writer.py's quantise."""
    scale = 65535 if bits == 16 else 255
    return np.rint(np.clip(pixels, 0, 1) * np.float32(scale)).astype(np.int64)


def reading(data: bytes, name: str = "FILE") -> picture.Picture:
    return picture.read(data, name)


def refusal(data: bytes, name: str = "FILE") -> str:
    with pytest.raises(Refusal) as caught:
        picture.read(data, name)
    return str(caught.value)


def after_marker(message: str, marker: str) -> list[str]:
    """The command in a refusal: what follows the marker, as a shell reads it."""
    assert marker in message
    return shlex.split(message.split(marker, 1)[1])


# The scaling ----------------------------------------------------------------------------------


def test_every_16_bit_code_comes_back() -> None:
    every = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    pic = reading(png16(every, 0))
    assert pic.bits == 16 and pic.grey
    assert pic.pixels.dtype == np.float32 and pic.pixels.flags.c_contiguous
    assert np.array_equal(codes(pic.pixels, 16)[..., 0], every)
    assert pic.pixels.min() == 0.0 and pic.pixels.max() == 1.0


def test_every_8_bit_code_comes_back() -> None:
    every = np.arange(256, dtype=np.uint8).reshape(16, 16)
    pic = reading(pillow(Image.fromarray(every, "L"), "PNG"))
    assert pic.bits == 8
    assert np.array_equal(codes(pic.pixels, 8)[..., 0], every)
    # What a writer makes at 16 bits of an 8-bit picture is the code over 255 rounded: 257 * code.
    assert np.array_equal(codes(pic.pixels, 16)[..., 0], every.astype(np.int64) * 257)


# Identifying ----------------------------------------------------------------------------------

FORMAT_FILES: dict[str, Callable[[], bytes]] = {
    "PNG": lambda: pillow(Image.fromarray(RGB8), "PNG"),
    "JPEG": lambda: pillow(Image.fromarray(SMOOTH), "JPEG"),
    "WEBP": lambda: pillow(Image.fromarray(SMOOTH), "WEBP", lossless=True),
    "TIFF": lambda: pillow(Image.fromarray(RGB8), "TIFF"),
    "BMP": lambda: pillow(Image.fromarray(RGB8), "BMP"),
    "GIF": lambda: pillow(Image.fromarray(RGB8).quantize(16), "GIF"),
}


@pytest.mark.parametrize("fmt", picture.FORMATS)
def test_identify_each_format_by_content(fmt: str) -> None:
    data = FORMAT_FILES[fmt]()
    assert picture.identify(io.BytesIO(data)) == fmt
    assert reading(data).format == fmt


def test_identify_leaves_the_file_open_where_it_was() -> None:
    file = io.BytesIO(FORMAT_FILES["PNG"]())
    file.seek(5)
    assert picture.identify(file) == "PNG"
    assert not file.closed and file.tell() == 5


@pytest.mark.parametrize(
    "content", [b"", b"x", b"plain text, not a picture\n" * 50, bytes(range(256)) * 40]
)
def test_identify_other_content_is_none(content: bytes) -> None:
    assert picture.identify(io.BytesIO(content)) is None


def test_identify_ignores_the_name(tmp_path: Path) -> None:
    path = tmp_path / "photo.mkv"
    path.write_bytes(FORMAT_FILES["JPEG"]())
    with path.open("rb") as file:
        assert picture.identify(file) == "JPEG"


@needs_ffmpeg
@pytest.mark.parametrize(("ext", "codec"), [("mkv", "ffv1"), ("mp4", "mpeg4")])
def test_identify_a_video_is_none(tmp_path: Path, ext: str, codec: str) -> None:
    video = tmp_path / f"v.{ext}"
    made = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=5"),
            *("-t", "1", "-c:v", codec, str(video)),
        ],
        capture_output=True,
    )
    if made.returncode:
        pytest.skip(f"this ffmpeg has no {codec} encoder")
    with video.open("rb") as file:
        assert picture.identify(file) is None
    assert "not a picture" in refusal(video.read_bytes(), video.name)


def test_a_damaged_file_is_identified_by_its_start_and_refused_by_read() -> None:
    data = b"\x89PNG\r\n\x1a\n" + b"junk" * 10
    assert picture.identify(io.BytesIO(data)) == "PNG"
    assert "damaged" in refusal(data)
    assert picture.identify(io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 20)) == "JPEG"


def _bomb() -> bytes:
    header = struct.pack(">IIBBBBB", 100_000, 100_000, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def test_a_picture_too_large_for_pillows_guard_is_identified_and_refused() -> None:
    data = _bomb()
    limit = Image.MAX_IMAGE_PIXELS
    assert picture.identify(io.BytesIO(data)) == "PNG"
    message = refusal(data, "huge.png")
    # Pillow's limit against decompression bombs, the picture's count and the limit, and the way
    # out; never that the picture is beyond what seedvr2x upscales.
    assert "huge.png has 10,000,000,000 pixels" in message
    assert "over the 178,956,970 pixels that Pillow" in message
    assert "limit against decompression bombs" in message
    assert "Make a smaller copy of it first" in message and "huge.png.small.png" in message
    assert "beyond what seedvr2x upscales" not in message
    assert Image.MAX_IMAGE_PIXELS == limit


@needs_ffmpeg
def test_the_smaller_copy_command_of_the_guards_refusal_works(tmp_path: Path) -> None:
    # 13,500 x 13,500 grey pixels, 182 million: over the guard, in a PNG of 177 KB.
    big = tmp_path / "big pic.png"
    Image.new("L", (13_500, 13_500)).save(big)
    message = refusal(big.read_bytes(), "big pic.png")
    assert "big pic.png has 182,250,000 pixels" in message
    command = after_marker(message, "and upscale the copy: ")
    assert command[:3] == ["ffmpeg", "-i", "big pic.png"]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    small = reading((tmp_path / "big pic.png.small.png").read_bytes())
    assert small.pixels.shape[:2] == (8192, 8192)


# The modes, values exact -----------------------------------------------------------------------


def test_rgb_png_is_read_exact() -> None:
    pic = reading(pillow(Image.fromarray(RGB8), "PNG"))
    assert (pic.bits, pic.grey, pic.icc, pic.chunks, pic.orientation) == (8, False, None, (), 1)
    assert not pic.opaque_alpha_dropped
    assert pic.pixels.shape == (16, 24, 3)
    assert np.array_equal(codes(pic.pixels, 8), RGB8)


def test_grey_png_is_read_grey_and_repeated() -> None:
    pic = reading(pillow(Image.fromarray(GREY8, "L"), "PNG"))
    assert pic.grey and pic.bits == 8
    for plane in range(3):
        assert np.array_equal(codes(pic.pixels[..., plane], 8), GREY8)
    assert pic.pixels.flags.c_contiguous


def test_bilevel_png_is_zero_and_one() -> None:
    bits = RNG.integers(0, 2, (16, 24), dtype=np.uint8)
    pic = reading(pillow(Image.fromarray(bits * 255, "L").convert("1"), "PNG"))
    assert pic.grey
    assert set(np.unique(pic.pixels)) <= {0.0, 1.0}
    assert np.array_equal(pic.pixels[..., 0], bits.astype(np.float32))


@pytest.mark.parametrize("depth", [2, 4])
def test_low_depth_grey_png_is_scaled_exactly(depth: int) -> None:
    top = 2**depth - 1
    samples = RNG.integers(0, top + 1, (4, 8), dtype=np.uint8)
    packed = [
        np.packbits(np.unpackbits(row[:, None], axis=1)[:, -depth:]).tobytes() for row in samples
    ]
    pic = reading(raw_png(8, 4, depth, 0, packed))
    assert pic.grey and pic.bits == 8
    assert np.array_equal(codes(pic.pixels[..., 0], 8), samples.astype(np.int64) * (255 // top))


def test_palette_png_is_expanded_exactly() -> None:
    image = Image.fromarray(RGB8).quantize(colors=16, dither=Image.Dither.NONE)
    pic = reading(pillow(image, "PNG"))
    assert not pic.grey and pic.bits == 8
    assert np.array_equal(codes(pic.pixels, 8), np.asarray(image.convert("RGB")))


def test_palette_with_an_opaque_trns_is_dropped_and_said(caplog: pytest.LogCaptureFixture) -> None:
    image = Image.fromarray(RGB8).quantize(colors=16, dither=Image.Dither.NONE)
    data = pillow(image, "PNG", transparency=bytes([255] * 16))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        pic = reading(data, "pal.png")
    assert pic.opaque_alpha_dropped
    assert np.array_equal(codes(pic.pixels, 8), np.asarray(image.convert("RGB")))
    assert [r.getMessage() for r in caplog.records if "alpha" in r.getMessage()] == [
        "pal.png: its alpha plane is opaque everywhere, left out: the picture is unchanged"
    ]


def test_rgb16_png_is_read_at_16_bits_exact() -> None:
    pic = reading(png16(RGB16, 2))
    assert (pic.bits, pic.grey) == (16, False)
    assert np.array_equal(codes(pic.pixels, 16), RGB16)


def test_grey16_png_is_read_exact_and_grey() -> None:
    pic = reading(png16(GREY16, 0))
    assert (pic.bits, pic.grey) == (16, True)
    for plane in range(3):
        assert np.array_equal(codes(pic.pixels[..., plane], 16), GREY16)


@pytest.mark.parametrize(
    ("colour_type", "channels", "grey"), [(6, 4, False), (4, 2, True)], ids=["rgba16", "ga16"]
)
def test_opaque_16_bit_alpha_is_dropped_at_16_bits(
    colour_type: int, channels: int, grey: bool, caplog: pytest.LogCaptureFixture
) -> None:
    samples = RNG.integers(0, 65536, (16, 24, channels), dtype=np.uint16)
    samples[..., -1] = 65535
    with caplog.at_level(logging.INFO, logger=LOGGER):
        pic = reading(png16(samples, colour_type))
    assert (pic.bits, pic.grey, pic.opaque_alpha_dropped) == (16, grey, True)
    expected = samples[..., :1] if grey else samples[..., :3]
    assert np.array_equal(codes(pic.pixels, 16)[..., : expected.shape[2]], expected)
    assert sum("alpha" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_opaque_8_bit_alpha_is_dropped_and_said_once(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    base = RGB8 if mode == "RGBA" else GREY8
    plane = np.dstack([base, np.full((16, 24), 255, np.uint8)])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        pic = reading(pillow(Image.fromarray(plane, mode), "PNG"))
    assert pic.opaque_alpha_dropped and pic.grey == (mode == "LA")
    assert np.array_equal(
        codes(pic.pixels, 8)[..., : (3 if mode == "RGBA" else 1)], plane[..., :-1]
    )
    records = [r for r in caplog.records if r.name == LOGGER]
    assert len(records) == 1 and records[0].levelno == logging.INFO


def test_a_bmp_with_an_alpha_mask_is_dropped_when_opaque() -> None:
    bgra = np.dstack([RGB8[..., ::-1], np.full((16, 24), 255, np.uint8)])
    pic = reading(bmp_v4(bgra))
    assert pic.opaque_alpha_dropped and np.array_equal(codes(pic.pixels, 8), RGB8)


def test_bmp_tiff_and_gif_are_read_exact() -> None:
    assert np.array_equal(codes(reading(FORMAT_FILES["BMP"]()).pixels, 8), RGB8), "BMP"
    assert np.array_equal(codes(reading(FORMAT_FILES["TIFF"]()).pixels, 8), RGB8), "TIFF"
    quantised = Image.fromarray(RGB8).quantize(16)
    assert np.array_equal(
        codes(reading(FORMAT_FILES["GIF"]()).pixels, 8), np.asarray(quantised.convert("RGB"))
    )


def test_grey_tiff_and_bmp_are_grey() -> None:
    assert reading(pillow(Image.fromarray(GREY8, "L"), "TIFF")).grey
    pic = reading(pillow(Image.fromarray(GREY8, "L"), "BMP"))
    assert pic.grey and np.array_equal(codes(pic.pixels[..., 0], 8), GREY8)


def test_lossless_webp_is_exact_and_lossy_close() -> None:
    assert np.array_equal(codes(reading(FORMAT_FILES["WEBP"]()).pixels, 8), SMOOTH)
    lossy = reading(pillow(Image.fromarray(SMOOTH), "WEBP", quality=90))
    assert not lossy.grey and np.abs(codes(lossy.pixels, 8) - SMOOTH).mean() < 4


def test_jpeg_rgb_and_grey() -> None:
    rgb = reading(FORMAT_FILES["JPEG"]())
    assert not rgb.grey and np.abs(codes(rgb.pixels, 8) - SMOOTH).mean() < 4
    grey = reading(pillow(Image.fromarray(SMOOTH[..., 0], "L"), "JPEG"))
    assert grey.grey and np.array_equal(grey.pixels[..., 0], grey.pixels[..., 2])


def test_a_one_frame_animation_is_a_picture() -> None:
    data = pillow(Image.fromarray(RGB8).quantize(16), "GIF", save_all=True, append_images=[])
    assert reading(data).format == "GIF"


# Refusals ------------------------------------------------------------------------------------


def animations() -> dict[str, bytes]:
    # Uneven delays, as a GIF's are: the video path would refuse a video whose timestamps
    # contradict its rate, so the refusal's command fixes one.
    one, two, three = (
        Image.fromarray(RGB8),
        Image.fromarray(RGB8[::-1]),
        Image.fromarray(RGB8[:, ::-1]),
    )
    delays = [50, 120, 80]
    return {
        "GIF": pillow(
            one.quantize(16),
            "GIF",
            save_all=True,
            append_images=[two.quantize(16), three.quantize(16)],
            duration=delays,
        ),
        "PNG": pillow(one, "PNG", save_all=True, append_images=[two, three], duration=delays),
        "WEBP": pillow(
            one, "WEBP", save_all=True, append_images=[two, three], lossless=True, duration=delays
        ),
    }


@pytest.mark.parametrize("fmt", ["GIF", "PNG", "WEBP"])
def test_several_frames_are_refused_with_a_command_that_works(fmt: str, tmp_path: Path) -> None:
    data = animations()[fmt]
    assert picture.identify(io.BytesIO(data)) == fmt
    name = "my anim.x"
    message = refusal(data, name)
    assert "3 frames" in message and "one picture" in message
    # The rate is fixed and said to be the user's to choose, the transparency said dropped.
    assert f"choosing the rate ({picture.VIDEO_RATE} frames a second here;" in message
    assert "its transparency, if any, is dropped" in message
    command = after_marker(message, "under it showing): ")
    assert command[:2] == ["ffmpeg", "-i"] and command[2] == name
    assert f"fps={picture.VIDEO_RATE},format=rgb24" in command
    if FFMPEG_REASON is not None:
        pytest.skip(FFMPEG_REASON)
    (tmp_path / name).write_bytes(data)
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    video = tmp_path / command[-1]
    probe = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "stream=codec_name,r_frame_rate,avg_frame_rate"),
            *("-show_entries", "packet=duration_time", "-of", "json", str(video)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    found = json.loads(probe.stdout)
    [stream] = found["streams"]
    # A constant rate: the stream's two rates equal, every packet as long as the others.
    assert stream["codec_name"] == "ffv1"
    assert stream["r_frame_rate"] == stream["avg_frame_rate"] == f"{picture.VIDEO_RATE}/1"
    durations = {packet["duration_time"] for packet in found["packets"]}
    assert len(found["packets"]) >= 2 and len(durations) == 1


def test_a_multi_page_tiff_is_refused_without_an_ffmpeg_command() -> None:
    one, two = Image.fromarray(RGB8), Image.fromarray(RGB8[::-1])
    message = refusal(pillow(one, "TIFF", save_all=True, append_images=[two]), "pages.tif")
    assert "2 frames" in message and "tiffsplit" in message and "ffmpeg" not in message


def _transparent_files() -> dict[str, bytes]:
    alpha = np.full((16, 24), 255, np.uint8)
    alpha[3, 4] = 254
    rgba = np.dstack([RGB8, alpha])
    palette = Image.fromarray(RGB8).quantize(colors=16, dither=Image.Dither.NONE)
    half = np.full((16, 24, 4), 65535, np.uint16)
    half[2, 2, 3] = 65534
    ga = np.full((16, 24, 2), 65535, np.uint16)
    ga[5, 5, 1] = 0
    key = struct.pack(">H", int(GREY16[1, 1]))
    return {
        "rgba": pillow(Image.fromarray(rgba, "RGBA"), "PNG"),
        "la": pillow(Image.fromarray(np.dstack([GREY8, alpha]), "LA"), "PNG"),
        "palette trns": pillow(palette, "PNG", transparency=bytes([255] * 15 + [0])),
        "rgb colour key": pillow(Image.fromarray(RGB8), "PNG", transparency=tuple(RGB8[0, 0])),
        "gif": pillow(palette, "GIF", transparency=int(np.asarray(palette)[0, 0])),
        "webp": pillow(Image.fromarray(rgba, "RGBA"), "WEBP", lossless=True),
        "tiff": pillow(Image.fromarray(rgba, "RGBA"), "TIFF"),
        "bmp v4": bmp_v4(rgba[..., [2, 1, 0, 3]]),
        "rgba16": png16(half, 6),
        "ga16": png16(ga, 4),
        "grey16 colour key": png16(GREY16, 0, before=(chunk(b"tRNS", key),)),
        "rgb16 colour key": png16(
            RGB16, 2, before=(chunk(b"tRNS", struct.pack(">3H", *map(int, RGB16[4, 4]))),)
        ),
    }


@pytest.mark.parametrize("which", list(_transparent_files()))
def test_any_transparent_pixel_is_refused(which: str) -> None:
    message = refusal(_transparent_files()[which], "t.png")
    assert "transparent" in message and "fringe" in message and "Flatten" in message


def test_a_colour_key_matching_no_pixel_is_opaque() -> None:
    values = {int(v) for v in GREY16.ravel()}
    key = next(v for v in range(65536) if v not in values)
    data = png16(GREY16, 0, before=(chunk(b"tRNS", struct.pack(">H", key)),))
    pic = reading(data)
    assert pic.opaque_alpha_dropped
    assert np.array_equal(codes(pic.pixels[..., 0], 16), GREY16)


@needs_ffmpeg
def test_the_flattening_command_in_the_refusal_works(tmp_path: Path) -> None:
    alpha = np.full((16, 24), 255, np.uint8)
    alpha[:6, :8] = 0
    alpha[8:12] = 128
    rgba = np.dstack([RGB8, alpha])
    name = "see through.png"
    (tmp_path / name).write_bytes(pillow(Image.fromarray(rgba, "RGBA"), "PNG"))
    message = refusal((tmp_path / name).read_bytes(), name)
    command = after_marker(message, "(which drops it): ")
    assert command[:3] == ["ffmpeg", "-i", name]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    flat = reading((tmp_path / command[-1]).read_bytes())
    a = alpha[..., None].astype(np.float64) / 255
    expected = RGB8 * a + 255 * (1 - a)
    assert np.abs(codes(flat.pixels, 8) - expected).max() <= 1


def test_other_colour_modes_are_refused_naming_the_mode() -> None:
    cmyk = pillow(Image.fromarray(RGB8).convert("CMYK"), "JPEG")
    assert "'CMYK'" in refusal(cmyk) and "RGB" in refusal(cmyk)
    assert "'CMYK'" in refusal(pillow(Image.fromarray(RGB8).convert("CMYK"), "TIFF"))
    floats = RNG.random((16, 24), dtype=np.float32)
    assert "'F'" in refusal(pillow(Image.fromarray(floats, "F"), "TIFF"))


@pytest.mark.parametrize("transfer", [16, 18])
def test_hdr_cicp_is_refused(transfer: int) -> None:
    cicp = chunk(b"cICP", bytes([9, transfer, 0, 1]))
    message = refusal(inserted(FORMAT_FILES["PNG"](), cicp), "hdr.png")
    assert {16: "PQ", 18: "HLG"}[transfer] in message and "tone-map" in message


def test_sdr_cicp_is_kept_whole() -> None:
    cicp = chunk(b"cICP", bytes([1, 13, 0, 1]))
    assert reading(inserted(FORMAT_FILES["PNG"](), cicp)).chunks == (cicp,)


@pytest.mark.parametrize("data", [FORMAT_FILES["PNG"](), png16(RGB16, 2)], ids=["8", "16"])
def test_hdr_is_refused_at_16_bits_too(data: bytes) -> None:
    assert "HDR" in refusal(inserted(data, chunk(b"cICP", bytes([9, 16, 0, 1]))))


def test_a_malformed_cicp_is_a_damaged_file() -> None:
    message = refusal(inserted(FORMAT_FILES["PNG"](), chunk(b"cICP", bytes([1, 13, 0]))))
    assert "damaged" in message and "cICP" in message


def test_a_chunk_failing_its_crc_is_a_damaged_file() -> None:
    bad = bytearray(chunk(b"gAMA", struct.pack(">I", 45455)))
    bad[-1] ^= 1
    message = refusal(inserted(FORMAT_FILES["PNG"](), bytes(bad)))
    assert "damaged" in message and "gAMA" in message


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "BMP", "WEBP"])
def test_a_truncated_file_is_refused_as_damaged(fmt: str) -> None:
    data = FORMAT_FILES[fmt]()
    assert picture.identify(io.BytesIO(data[: len(data) // 2])) == fmt
    assert "damaged" in refusal(data[: len(data) // 2])


def test_a_truncated_16_bit_png_is_refused_as_damaged() -> None:
    data = png16(RGB16, 2)
    assert "damaged" in refusal(data[: len(data) - 40])


def test_a_png_ending_before_its_data_is_refused() -> None:
    assert "damaged" in refusal(FORMAT_FILES["PNG"]()[:60])


def _tiff16_grey(profile: bytes | None = None) -> bytes:
    image = Image.fromarray(GREY16)
    return pillow(image, "TIFF", icc_profile=profile) if profile else pillow(image, "TIFF")


def test_a_16_bit_tiff_is_refused_with_a_png_command() -> None:
    message = refusal(_tiff16_grey(), "scan.tif")
    assert "16 bits per sample" in message and "ffmpeg -i scan.tif scan.tif.png" in message


def test_a_16_bit_tiff_with_a_profile_is_sent_to_an_editor() -> None:
    message = refusal(_tiff16_grey(PROFILE), "scan.tif")
    assert "image editor" in message and "drops the profile" in message


@needs_ffmpeg
def test_the_16_bit_tiff_command_makes_a_16_bit_png(tmp_path: Path) -> None:
    raw = tmp_path / "raw.rgb"
    raw.write_bytes(RGB16.astype("<u2").tobytes())
    tiff = tmp_path / "rgb 16.tif"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "24x16"),
            *("-i", str(raw), "-frames:v", "1", str(tiff)),
        ],
        check=True,
        capture_output=True,
    )
    message = refusal(tiff.read_bytes(), tiff.name)
    assert "16 bits per sample" in message
    command = after_marker(message, "seedvr2x reads whole: ")
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    made = reading((tmp_path / command[-1]).read_bytes())
    assert made.bits == 16 and np.array_equal(codes(made.pixels, 16), RGB16)


@needs_ffmpeg
def test_the_16_bit_grey_tiff_command_keeps_grey_at_16_bits(tmp_path: Path) -> None:
    (tmp_path / "g.tif").write_bytes(_tiff16_grey())
    command = after_marker(refusal(_tiff16_grey(), "g.tif"), "seedvr2x reads whole: ")
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    made = reading((tmp_path / command[-1]).read_bytes())
    assert made.bits == 16 and made.grey
    assert np.array_equal(codes(made.pixels[..., 0], 16), GREY16)


def test_an_8_bit_tiff_with_a_profile_is_not_the_16_bit_refusal() -> None:
    pic = reading(pillow(Image.fromarray(RGB8), "TIFF", icc_profile=PROFILE))
    assert pic.bits == 8 and pic.icc == PROFILE


# Metadata ------------------------------------------------------------------------------------


def test_the_icc_profile_is_read_from_every_format_that_holds_one() -> None:
    image = Image.fromarray(SMOOTH)
    for fmt, options in (("PNG", {}), ("JPEG", {}), ("WEBP", {"lossless": True}), ("TIFF", {})):
        assert reading(pillow(image, fmt, icc_profile=PROFILE, **options)).icc == PROFILE, fmt
    assert reading(FORMAT_FILES["PNG"]()).icc is None


def test_the_icc_profile_of_a_16_bit_png_is_read() -> None:
    iccp = chunk(b"iCCP", b"ICC profile\x00\x00" + zlib.compress(PROFILE))
    assert reading(png16(RGB16, 2, before=(iccp,))).icc == PROFILE


@pytest.mark.parametrize("value", range(1, 9))
def test_the_orientation_is_read_and_the_pixels_not_turned(value: int) -> None:
    exif = orientation_exif(value)
    for fmt, options in (("JPEG", {}), ("PNG", {}), ("WEBP", {"lossless": True})):
        pic = reading(pillow(Image.fromarray(SMOOTH), fmt, exif=exif, **options))
        assert pic.orientation == value, fmt
        assert pic.pixels.shape == (32, 48, 3), fmt


def test_a_picture_without_orientation_is_1() -> None:
    assert reading(FORMAT_FILES["JPEG"]()).orientation == 1
    assert reading(pillow(Image.fromarray(SMOOTH), "JPEG", exif=Image.Exif())).orientation == 1


def test_an_orientation_out_of_range_is_taken_as_1_and_said(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = pillow(Image.fromarray(SMOOTH), "JPEG", exif=orientation_exif(9))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pic = reading(data, "odd.jpg")
    assert pic.orientation == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "orientation is 9" in warnings[0].getMessage()


def test_the_orientation_of_a_16_bit_png_is_read() -> None:
    exif = chunk(b"eXIf", orientation_exif(6).tobytes()[6:])
    assert reading(png16(RGB16, 2, after=(exif,))).orientation == 6


def test_png_colour_chunks_are_kept_whole_in_file_order() -> None:
    chunks = (
        chunk(b"sRGB", b"\x00"),
        chunk(b"gAMA", struct.pack(">I", 45455)),
        chunk(b"cICP", bytes([1, 13, 0, 1])),
        chunk(b"cHRM", struct.pack(">8I", *range(31270, 31270 + 8))),
    )
    other = chunk(b"tEXt", b"Comment\x00not kept")
    for base in (FORMAT_FILES["PNG"](), png16(RGB16, 2)):
        pic = reading(inserted(base, chunks[0], other, *chunks[1:]))
        assert pic.chunks == chunks
    assert reading(FORMAT_FILES["JPEG"]()).chunks == ()


# Several images in a JPEG ----------------------------------------------------------------------


def _mpo() -> tuple[bytes, bytes]:
    """(a JPEG with a second image after the first, a plain JPEG of its first image)."""
    first = Image.fromarray(SMOOTH)
    second = Image.fromarray(SMOOTH[::-1, ::-1])
    return (
        pillow(first, "MPO", save_all=True, append_images=[second]),
        pillow(first, "JPEG"),
    )


def test_pillow_opens_an_mpo_as_a_jpeg_of_several_frames() -> None:
    # What the module's MPO_FIRST_IMAGE rule answers: Pillow gives it format "MPO" whatever the
    # formats asked, n_frames 2.
    data, _ = _mpo()
    with Image.open(io.BytesIO(data), formats=picture.FORMATS) as image:
        assert (image.format, getattr(image, "n_frames", 1)) == ("MPO", 2)
    assert picture.identify(io.BytesIO(data)) == "JPEG"


def test_an_mpo_is_read_as_its_first_image_and_said(caplog: pytest.LogCaptureFixture) -> None:
    data, plain = _mpo()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        pic = reading(data, "pair.jpg")
    assert pic.format == "JPEG"
    assert np.array_equal(pic.pixels, reading(plain).pixels)
    said = [r for r in caplog.records if "images" in r.getMessage()]
    # At WARNING, not INFO: information is lost, and the line says what and that the picture is
    # the first image.
    assert len(said) == 1 and said[0].levelno == logging.WARNING
    message = said[0].getMessage()
    assert "pair.jpg" in message and "1 left out" in message
    assert "stereo pair's second view" in message and "gain map" in message
    assert "the picture is the first image alone" in message


def test_an_mpo_is_refused_when_the_rule_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(picture, "MPO_FIRST_IMAGE", False)
    assert "2 frames" in refusal(_mpo()[0])


# JPEGs one after another -------------------------------------------------------------------------


def thumbnail_exif(thumbnail: bytes) -> bytes:
    """The payload of an APP1 segment holding EXIF data with a JPEG thumbnail in its second IFD
    (JPEGInterchangeFormat and its length): a whole JPEG, its SOI and EOI, inside the image."""
    ifd1 = 14
    start = ifd1 + 2 + 2 * 12 + 4
    entries = struct.pack(">HHII", 0x0201, 4, 1, start) + struct.pack(
        ">HHII", 0x0202, 4, 1, len(thumbnail)
    )
    tiff = b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">HI", 0, ifd1)
    tiff += struct.pack(">H", 2) + entries + struct.pack(">I", 0) + thumbnail
    return b"Exif\x00\x00" + tiff


def with_thumbnail(jpeg: bytes) -> bytes:
    """jpeg with an EXIF thumbnail, of its own, in an APP1 segment right after its SOI."""
    thumbnail = pillow(Image.fromarray(SMOOTH[::4, ::4]), "JPEG")
    payload = thumbnail_exif(thumbnail)
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + jpeg[2:]


def jpegs() -> dict[str, bytes]:
    image = Image.fromarray(SMOOTH)
    return {
        "baseline": pillow(image, "JPEG"),
        "progressive": pillow(image, "JPEG", progressive=True),
        "thumbnail": with_thumbnail(pillow(Image.fromarray(SMOOTH[::-1]), "JPEG")),
    }


def test_the_thumbnail_jpeg_is_read_as_one_image() -> None:
    # An EXIF thumbnail holds its own SOI and EOI inside the image: a search for the first FF D9
    # would end the image there and take the rest for a second one.
    data = jpegs()["thumbnail"]
    assert data.count(b"\xff\xd8\xff") == 2 and data.count(b"\xff\xd9") >= 2
    assert reading(data).format == "JPEG"
    assert picture._jpeg_images(data) == 1  # pyright: ignore[reportPrivateUsage]


def test_a_progressive_jpeg_has_several_scans_and_is_one_image() -> None:
    data = jpegs()["progressive"]
    assert data.count(b"\xff\xda") > 1
    assert picture._jpeg_images(data) == 1  # pyright: ignore[reportPrivateUsage]
    assert reading(data).format == "JPEG"


@pytest.mark.parametrize("padding", [b"", bytes(7), bytes(4096)])
def test_a_jpeg_with_padding_after_it_is_one_picture(padding: bytes) -> None:
    for each in jpegs().values():
        pic = reading(each + padding)
        assert np.array_equal(pic.pixels, reading(each).pixels)


def test_a_jpeg_followed_by_other_bytes_is_one_picture() -> None:
    each = jpegs()["baseline"]
    assert reading(each + b"trailing text, no image").format == "JPEG"


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("padding", [b"", bytes(5)], ids=["joined", "padded"])
def test_jpegs_one_after_another_are_refused_as_frames(count: int, padding: bytes) -> None:
    kinds = list(jpegs().values())
    stream = padding.join(kinds[:count])
    message = refusal(stream, "cam.mjpeg")
    assert f"cam.mjpeg holds {count} frames" in message
    command = after_marker(message, "frames a second here): ")
    assert command[:4] == ["ffmpeg", "-f", "mjpeg", "-framerate"]
    assert command[4] == str(picture.MJPEG_RATE) and command[6] == "cam.mjpeg"
    # The start of the file is still a JPEG to identify; read refuses it.
    assert picture.identify(io.BytesIO(stream)) == "JPEG"


def test_every_order_of_the_jpeg_kinds_is_counted() -> None:
    kinds = jpegs()
    for first in kinds.values():
        for second in kinds.values():
            assert picture._jpeg_images(first + second) == 2  # pyright: ignore[reportPrivateUsage]


def test_a_last_jpeg_cut_short_is_still_another_frame() -> None:
    kinds = jpegs()
    stream = kinds["baseline"] + kinds["progressive"][:-200]
    assert "2 frames" in refusal(stream, "dump.mjpeg")


def test_the_scan_walk_follows_restart_markers_stuffing_and_fill_bytes() -> None:
    # A JPEG written by hand: after SOS, entropy-coded data with a stuffed FF00, an FFD0
    # restart marker, and FF fill bytes before the EOI; a DNL-less single image.
    scan = b"\x12\xff\x00\x34\xff\xd0\x56\xff\xd7\x78"
    image = (
        b"\xff\xd8"
        + b"\xff\xe0\x00\x06abcd"
        + b"\xff\xda\x00\x04\x00\x00"
        + scan
        + b"\xff\xff\xff\xd9"
    )
    assert picture._jpeg_end(image, 0) == len(image)  # pyright: ignore[reportPrivateUsage]
    assert picture._jpeg_images(image + image) == 2  # pyright: ignore[reportPrivateUsage]
    # A length beyond the file, and a scan without a marker after it, end nowhere.
    assert picture._jpeg_end(image[:20], 0) is None  # pyright: ignore[reportPrivateUsage]
    assert picture._jpeg_end(image[:-4], 0) is None  # pyright: ignore[reportPrivateUsage]


@needs_ffmpeg
def test_an_ffmpeg_mjpeg_stream_is_refused_and_its_command_works(tmp_path: Path) -> None:
    stream = tmp_path / "cam.mjpeg"
    made = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "3", "-f", "mjpeg", str(stream)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if made.returncode != 0:
        pytest.skip(f"this ffmpeg can't make an MJPEG stream: {made.stderr.strip()[:200]}")
    message = refusal(stream.read_bytes(), "cam.mjpeg")
    assert "cam.mjpeg holds 3 frames" in message
    command = after_marker(message, "frames a second here): ")
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    counted = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0"),
            *("-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "csv=p=0"),
            str(tmp_path / "cam.mjpeg.mkv"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert counted.stdout.strip() == f"{picture.MJPEG_RATE}/1,3"


# Against ffmpeg's decode ------------------------------------------------------------------------


def ffmpeg_decode(path: Path, pix_fmt: str, vf: str | None = None) -> npt.NDArray[np.int64]:
    dtype = "<u2" if pix_fmt.endswith(("48le", "16le")) else "u1"
    command = ["ffmpeg", "-v", "error", "-noautorotate", "-i", str(path), "-frames:v", "1"]
    command += ["-vf", vf] if vf else []
    command += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype).astype(np.int64)


def _smooth_png_palette() -> bytes:
    return pillow(Image.fromarray(RGB8).quantize(colors=32), "PNG")


# name, file, ffmpeg's pixel format, the largest difference allowed (0: bit for bit), ffmpeg's
# filter for the lossy ones.
EXACT = 0
AGAINST_FFMPEG: list[tuple[str, Callable[[], bytes], str, int, str | None]] = [
    ("png rgb", FORMAT_FILES["PNG"], "rgb24", EXACT, None),
    ("png palette", _smooth_png_palette, "rgb24", EXACT, None),
    ("png grey", lambda: pillow(Image.fromarray(GREY8, "L"), "PNG"), "gray", EXACT, None),
    ("png grey 16", lambda: png16(GREY16, 0), "gray16le", EXACT, None),
    ("png rgb 16", lambda: png16(RGB16, 2), "rgb48le", EXACT, None),
    (
        "png rgba opaque",
        lambda: pillow(
            Image.fromarray(np.dstack([RGB8, np.full((16, 24), 255, np.uint8)]), "RGBA"), "PNG"
        ),
        "rgb24",
        EXACT,
        None,
    ),
    ("bmp", FORMAT_FILES["BMP"], "rgb24", EXACT, None),
    ("tiff", FORMAT_FILES["TIFF"], "rgb24", EXACT, None),
    ("tiff grey", lambda: pillow(Image.fromarray(GREY8, "L"), "TIFF"), "gray", EXACT, None),
    ("gif", FORMAT_FILES["GIF"], "rgb24", EXACT, None),
    ("webp lossless", FORMAT_FILES["WEBP"], "rgb24", EXACT, None),
    ("jpeg", FORMAT_FILES["JPEG"], "rgb24", 8, None),
    ("jpeg bilinear", FORMAT_FILES["JPEG"], "rgb24", 3, BILINEAR),
    (
        "jpeg 4:4:4",
        lambda: pillow(Image.fromarray(SMOOTH), "JPEG", subsampling=0),
        "rgb24",
        3,
        None,
    ),
    (
        "webp lossy",
        lambda: pillow(Image.fromarray(SMOOTH), "WEBP", quality=80),
        "rgb24",
        2,
        BILINEAR,
    ),
]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("make", "pix_fmt", "allowed", "vf"),
    [case[1:] for case in AGAINST_FFMPEG],
    ids=[case[0] for case in AGAINST_FFMPEG],
)
def test_the_pixels_are_ffmpegs_decode(
    tmp_path: Path, make: Callable[[], bytes], pix_fmt: str, allowed: int, vf: str | None
) -> None:
    data = make()
    path = tmp_path / "picture"
    path.write_bytes(data)
    pic = reading(data)
    theirs = ffmpeg_decode(path, pix_fmt, vf)
    mine = codes(pic.pixels, pic.bits)
    if pic.grey:
        assert np.array_equal(mine[..., 0], mine[..., 1])
        assert np.array_equal(mine[..., 0], mine[..., 2])
        mine = mine[..., :1]
    assert mine.size == theirs.size
    assert np.abs(mine.ravel() - theirs).max() <= allowed
