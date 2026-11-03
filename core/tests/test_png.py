"""The PNG writer of a still (seedvr2x/png.py; DESIGN.md, Still images, Output): the chunks in their
order with their CRCs, the quantisation of the video writers, 8 and 16 bits, colour and grey, every
filter decoding exactly, read back by Pillow, TorchCodec, seedvr2x's reader and ffmpeg."""

import io
import json
import re
import shutil
import struct
import subprocess
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from PIL import Image, ImageCms, ImageOps
from test_record import read_xmp
from torchcodec.decoders import decode_image  # pyright: ignore[reportPrivateImportUsage]

from seedvr2x import picture, png
from seedvr2x.record import DIGITAL_SOURCE_TYPE, Model, Record

_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
_banner = (
    subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout if _FFMPEG else ""
)
_version = re.search(r"ffmpeg version n?(\d+)\.(\d+)", _banner)
FFMPEG_REASON = (
    "ffmpeg and ffprobe are not on PATH"
    if not _FFMPEG
    else "ffmpeg's version is not a release 7.1 or later"
    if _version is None or (int(_version[1]), int(_version[2])) < (7, 1)
    else None
)
needs_ffmpeg = pytest.mark.skipif(FFMPEG_REASON is not None, reason=str(FFMPEG_REASON))

RNG = np.random.default_rng(20261103)
PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

# A picture with some structure (so that the filters differ) and some noise, as float32 on the
# 16-bit grid, so that its quantisation is exactly known: (11, 17, 3) is odd on purpose.
_Y, _X = np.mgrid[0:11, 0:17]
CODES16 = np.clip(
    np.stack([_X * 3000 + _Y * 40, _Y * 5000 - _X * 100, (_X + _Y) * 1500], axis=2)
    + RNG.integers(0, 600, (11, 17, 3)),
    0,
    65535,
).astype(np.uint16)
PICTURE = (CODES16 / np.float32(65535)).astype(np.float32)

# A record, as the CLI or the server gives one (seedvr2x/record.py).
RECORD = Record(
    "seedvr2x",
    "0.1.0",
    (17, 11),
    (17, 11),
    11,
    Fraction(1),
    Model("dit.safetensors", "1" * 64),
    Model("vae.safetensors", "2" * 64),
    42,
    "split",
    0,
    512,
    "sdpa",
)


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


COLOUR = (
    chunk(b"cICP", bytes([1, 13, 0, 1])),
    chunk(b"gAMA", struct.pack(">I", 45455)),
    chunk(b"cHRM", struct.pack(">8I", *range(31270, 31278))),
    chunk(b"sRGB", b"\x00"),
)


def write(rgb: npt.NDArray[np.float32], bits: int = 16, **options: object) -> bytes:
    settings: dict[str, object] = {
        "grey": False,
        "icc": None,
        "chunks": (),
        **options,
    }
    return png.write(rgb, bits=bits, **settings)  # pyright: ignore[reportArgumentType]


def chunks_of(data: bytes) -> list[tuple[bytes, bytes]]:
    """(type, data) of every chunk, each CRC checked."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    found = []
    position = 8
    while position < len(data):
        (length,) = struct.unpack_from(">I", data, position)
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack_from(">I", data, position + 8 + length)
        assert crc == zlib.crc32(kind + body), kind
        found.append((kind, body))
        position += 12 + length
    assert position == len(data)
    return found


def decoded16(data: bytes) -> npt.NDArray[np.uint16]:
    """(H, W, C) uint16 by TorchCodec's libpng, which turns a picture by its eXIf chunk: the
    writer writes none."""
    tensor = decode_image(
        torch.frombuffer(bytearray(data), dtype=torch.uint8),
        mode="UNCHANGED",
        output_dtype=torch.uint16,
    )
    return tensor.permute(1, 2, 0).numpy()


def quantised(rgb: npt.NDArray[np.float32], bits: int) -> npt.NDArray[np.int64]:
    """What every writer makes of a frame, written apart: clip, scale, round to nearest, the
    product in float32 as media/writer.py's quantise takes it (float64's would round a few
    codes the other way)."""
    scale = np.float32(65535 if bits == 16 else 255)
    return np.rint(np.clip(rgb, np.float32(0), np.float32(1)) * scale).astype(np.int64)


def unfiltered_types(data: bytes) -> list[int]:
    """The filter type byte of every row, from the inflated image data."""
    header = dict(chunks_of(data))[b"IHDR"]
    width, height, depth, colour_type = struct.unpack(">IIBB", header[:10])
    stride = width * (depth // 8) * (1 if colour_type == 0 else 3) + 1
    image = zlib.decompress(b"".join(body for kind, body in chunks_of(data) if kind == b"IDAT"))
    assert len(image) == stride * height
    return [image[row * stride] for row in range(height)]


# The chunks ----------------------------------------------------------------------------------


def test_chunks_in_order_with_their_crcs() -> None:
    data = write(PICTURE, icc=PROFILE, chunks=COLOUR)
    kinds = [kind for kind, _ in chunks_of(data)]
    assert kinds[:6] == [b"IHDR", b"cICP", b"gAMA", b"cHRM", b"sRGB", b"iCCP"]
    assert set(kinds[6:-1]) == {b"IDAT"} and kinds[-1] == b"IEND" and len(kinds) >= 8


def test_the_colour_chunks_are_verbatim_and_in_the_order_given() -> None:
    shuffled = (COLOUR[2], COLOUR[0], COLOUR[3], COLOUR[1])
    data = write(PICTURE, chunks=shuffled)
    start = 8 + 12 + 13  # the signature, the IHDR chunk
    assert data[start : start + sum(map(len, shuffled))] == b"".join(shuffled)


def test_ihdr_says_the_depth_and_the_colour_type() -> None:
    for bits in (8, 16):
        for grey in (False, True):
            header = dict(chunks_of(write(PICTURE, bits, grey=grey)))[b"IHDR"]
            assert header == struct.pack(">IIBBBBB", 17, 11, bits, 0 if grey else 2, 0, 0, 0)


def test_the_idat_is_cut_in_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(png, "IDAT_BYTES", 64)
    noise = RNG.random((40, 40, 3), dtype=np.float32)
    data = write(noise)
    pieces = [body for kind, body in chunks_of(data) if kind == b"IDAT"]
    assert len(pieces) > 5 and all(len(piece) <= 64 for piece in pieces)
    assert np.array_equal(decoded16(data), quantised(noise, 16))


def test_icc_is_a_deflated_iccp_named_icc_profile() -> None:
    body = dict(chunks_of(write(PICTURE, icc=PROFILE)))[b"iCCP"]
    name, rest = body.split(b"\x00", 1)
    assert name == b"ICC profile" and rest[0] == 0
    assert zlib.decompress(rest[1:]) == PROFILE
    assert b"iCCP" not in dict(chunks_of(write(PICTURE)))


def test_no_exif_chunk_is_written() -> None:
    data = write(PICTURE, icc=PROFILE, chunks=COLOUR)
    assert b"eXIf" not in dict(chunks_of(data))
    with Image.open(io.BytesIO(data)) as image:
        assert 0x0112 not in image.getexif()


@pytest.mark.parametrize("value", range(1, 9))
def test_a_picture_turned_upright_reads_back_upright_with_no_orientation(value: int) -> None:
    """The 8 orientations: the picture turned by picture.upright and written reads back, in
    Pillow, as Pillow's exif_transpose of the same picture carrying the orientation, and in
    TorchCodec (16 bits) as the turned array, with no eXIf chunk and no orientation tag."""
    turned = picture.upright(PICTURE, value)
    data = write(turned, 8)
    assert b"eXIf" not in dict(chunks_of(data))
    source = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = value
    Image.fromarray(quantised(PICTURE, 8).astype(np.uint8)).save(source, "PNG", exif=exif)
    with Image.open(io.BytesIO(source.getvalue())) as stored, Image.open(io.BytesIO(data)) as read:
        expected = ImageOps.exif_transpose(stored)
        assert 0x0112 not in read.getexif()
        assert np.array_equal(np.asarray(read), np.asarray(expected))
    assert np.array_equal(decoded16(write(turned)), quantised(turned, 16))


# The pixels ----------------------------------------------------------------------------------


@pytest.mark.parametrize("grey", [False, True])
@pytest.mark.parametrize("bits", [8, 16])
def test_read_back_bit_for_bit(bits: int, grey: bool) -> None:
    rgb = RNG.random((11, 17, 3), dtype=np.float32) * 1.4 - 0.2
    data = write(rgb, bits, grey=grey)
    expected = quantised(rgb, bits)
    if grey:
        mean = ((rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / np.float32(3)).astype(np.float32)
        expected = quantised(mean[..., np.newaxis], bits)
    with Image.open(io.BytesIO(data)) as image:
        # Pillow reads 16-bit grey (as I;16) and 8-bit grey and colour.
        if bits == 16 and not grey:
            theirs = decoded16(data)
        else:
            theirs = np.asarray(image)[..., np.newaxis] if grey else np.asarray(image)
    assert np.array_equal(theirs.astype(np.int64), expected)
    if bits == 16:
        assert np.array_equal(decoded16(data).astype(np.int64).reshape(expected.shape), expected)
    read = picture.read(data)
    assert (read.bits, read.grey) == (bits, grey)
    scale = 65535 if bits == 16 else 255
    mine = np.rint(read.pixels * np.float32(scale)).astype(np.int64)
    assert np.array_equal(mine[..., 0] if grey else mine, expected[..., 0] if grey else expected)


def test_quantisation_is_the_video_writers() -> None:
    # media/writer.py's quantise: clip to [0, 1], times 65535 in float32, rint, 16-bit codes.
    rgb = (RNG.random((11, 17, 3), dtype=np.float32) * 1.6 - 0.3).astype(np.float32)
    ours = decoded16(write(rgb))
    clipped = np.clip(rgb, 0.0, 1.0)
    clipped *= 65535
    np.rint(clipped, out=clipped)
    assert np.array_equal(ours, clipped.astype("<u2"))


def test_every_code_survives_16_and_8_bits() -> None:
    every = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    grey16 = decoded16(
        write((every / np.float32(65535))[..., None].repeat(3, 2).astype(np.float32), grey=True)
    )
    assert np.array_equal(grey16[..., 0], every)
    codes8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.repeat((codes8 / np.float32(255))[..., None], 3, axis=2).astype(np.float32)
    with Image.open(io.BytesIO(write(rgb, 8, grey=True))) as image:
        assert np.array_equal(np.asarray(image), codes8)


def test_a_grey_is_the_mean_of_the_three_planes() -> None:
    rgb = np.zeros((2, 3, 3), np.float32)
    rgb[0, 0] = (0.0, 0.3, 0.6)
    rgb[0, 1] = (1.0, 1.0, 0.0)
    rgb[0, 2] = (-5.0, 1.0, 9.0)  # the mean first, then the clip
    rgb[1] = 0.25
    grey = decoded16(write(rgb, grey=True))[..., 0]
    assert grey[0, 0] == round(0.3 * 65535) and grey[0, 1] == round(2 / 3 * 65535)
    assert grey[0, 2] == 65535
    assert (grey[1] == round(0.25 * 65535)).all()


def test_clipping_never_wraps() -> None:
    rgb = np.array([[[-1e9, 2.0, 1e30], [0.0, 1.0, 0.5]]], np.float32)
    assert decoded16(write(rgb)).tolist() == [[[0, 65535, 65535], [0, 65535, 32768]]]


@pytest.mark.parametrize("layout", ["fortran", "strided", "transposed", "readonly", "float32 view"])
def test_a_non_contiguous_input_is_written_as_the_same_picture(layout: str) -> None:
    reference = write(PICTURE)
    if layout == "fortran":
        other = np.asfortranarray(PICTURE)
    elif layout == "strided":
        base = np.zeros((22, 34, 3), np.float32)
        base[::2, ::2] = PICTURE
        other = base[::2, ::2]
    elif layout == "transposed":
        other = np.ascontiguousarray(PICTURE.transpose(1, 0, 2)).transpose(1, 0, 2)
    elif layout == "readonly":
        other = PICTURE.copy()
        other.flags.writeable = False
    else:
        other = np.moveaxis(np.ascontiguousarray(np.moveaxis(PICTURE, 2, 0)), 0, 2)
    assert not other.flags.c_contiguous or layout == "readonly"
    assert write(other) == reference


def test_the_input_is_left_alone() -> None:
    rgb = RNG.random((11, 17, 3), dtype=np.float32) * 2 - 0.5
    before = rgb.copy()
    write(rgb)
    write(rgb, grey=True)
    assert np.array_equal(rgb, before)


def test_the_same_bytes_every_time() -> None:
    first = write(PICTURE, icc=PROFILE, chunks=COLOUR, xmp=RECORD.xmp(), software=RECORD.creator)
    again = write(PICTURE, icc=PROFILE, chunks=COLOUR, xmp=RECORD.xmp(), software=RECORD.creator)
    assert again == first


# The record ----------------------------------------------------------------------------------


def test_the_record_comes_after_the_colour_and_before_the_image_data() -> None:
    # tEXt Software, then iTXt holding the XMP, after the colour chunks and the profile and
    # before IDAT, where readers that stop at the image data still find them; each CRC checked
    # (chunks_of).
    data = write(PICTURE, icc=PROFILE, chunks=COLOUR, xmp=RECORD.xmp(), software=RECORD.creator)
    kinds = [kind for kind, _ in chunks_of(data)]
    assert kinds[:8] == [b"IHDR", b"cICP", b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"tEXt", b"iTXt"]
    assert set(kinds[8:-1]) == {b"IDAT"} and kinds[-1] == b"IEND"


def test_the_xmp_is_an_uncompressed_itxt_named_as_xmp_asks() -> None:
    data = write(PICTURE, xmp=RECORD.xmp())
    (body,) = [body for kind, body in chunks_of(data) if kind == b"iTXt"]
    # Keyword, NUL, compression flag 0, method 0, no language, no translated keyword, the packet.
    assert body == b"XML:com.adobe.xmp\0\0\0\0\0" + RECORD.xmp()
    tool, mark, fields = read_xmp(body[len(b"XML:com.adobe.xmp") + 5 :])
    assert tool == RECORD.creator and mark == DIGITAL_SOURCE_TYPE
    assert fields == RECORD.fields()


def test_the_software_text_names_the_writer() -> None:
    data = write(PICTURE, software=RECORD.creator)
    (body,) = [body for kind, body in chunks_of(data) if kind == b"tEXt"]
    assert body == b"Software\0seedvr2x 0.1.0 (https://github.com/hekmon/seedvr2x)"


def test_readers_find_the_record() -> None:
    # Pillow reads both texts; the pixels are those written without them.
    data = write(PICTURE, xmp=RECORD.xmp(), software=RECORD.creator)
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        assert image.info["Software"] == RECORD.creator
        assert image.info["XML:com.adobe.xmp"] == RECORD.xmp().decode()
    assert np.array_equal(decoded16(data), decoded16(write(PICTURE)))


def test_nothing_of_the_record_without_it() -> None:
    kinds = {kind for kind, _ in chunks_of(write(PICTURE))}
    assert not kinds & {b"tEXt", b"iTXt", b"zTXt"}


def test_a_software_text_with_a_nul_is_refused() -> None:
    with pytest.raises(ValueError):
        write(PICTURE, software="a\0b")


@needs_ffmpeg
def test_ffprobe_reads_the_software_text(tmp_path: Path) -> None:
    # ffmpeg's PNG decoder shows tEXt as the frame's tags, and no iTXt (n9.0.2, 2026-11-03), so
    # the Software text alone is what ffprobe says of the record.
    path = tmp_path / "record.png"
    path.write_bytes(write(PICTURE, xmp=RECORD.xmp(), software=RECORD.creator))
    shown = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "frame_tags", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tags = json.loads(shown)["frames"][0]["tags"]
    assert tags == {"Software": RECORD.creator}


# The filters ---------------------------------------------------------------------------------


@pytest.mark.parametrize("band", [1 << 22, 40])
@pytest.mark.parametrize("filter_type", range(png.ADAPTIVE + 1))
@pytest.mark.parametrize(("bits", "grey"), [(16, False), (8, False), (16, True), (8, True)])
def test_every_filter_decodes_exactly(
    monkeypatch: pytest.MonkeyPatch, filter_type: int, bits: int, grey: bool, band: int
) -> None:
    # A band of 40 bytes is a row or two: the row above a band's first is the previous band's last.
    monkeypatch.setattr(png, "BAND_BYTES", band)
    data = write(PICTURE, bits, grey=grey, filter_type=filter_type)
    expected = quantised(PICTURE, bits)
    if grey:
        mean = (PICTURE[..., 0] + PICTURE[..., 1] + PICTURE[..., 2]) / np.float32(3)
        expected = quantised(mean[..., np.newaxis], bits)
    with Image.open(io.BytesIO(data)) as image:
        theirs = decoded16(data) if bits == 16 and not grey else np.asarray(image)
    assert np.array_equal(theirs.reshape(expected.shape).astype(np.int64), expected)
    types = unfiltered_types(data)
    if filter_type != png.ADAPTIVE:
        assert set(types) == {filter_type}
    else:
        assert set(types) <= set(range(5))


def test_the_adaptive_filter_picks_by_row() -> None:
    # Rows made for each filter in turn: constant, a ramp along the row, the same row as the
    # one above plus a little, noise: the heuristic must not use one filter for all.
    rows = np.zeros((40, 64, 3), np.float32)
    rows[:10] = 0.5
    rows[10:20] = np.linspace(0, 1, 64, dtype=np.float32)[None, :, None]
    rows[20:30] = RNG.random((10, 64, 3), dtype=np.float32)
    rows[30:] = rows[29] + np.arange(10, dtype=np.float32)[:, None, None] / 1000
    data = write(rows, filter_type=png.ADAPTIVE)
    assert len(set(unfiltered_types(data))) >= 3
    assert np.array_equal(decoded16(data).astype(np.int64), quantised(rows, 16))


def test_the_filters_and_levels_change_the_bytes_not_the_pixels() -> None:
    reference = decoded16(write(PICTURE))
    for level in (0, 1, 9):
        assert np.array_equal(decoded16(write(PICTURE, level=level)), reference)
    assert write(PICTURE, filter_type=png.NONE) != write(PICTURE, filter_type=png.PAETH)


def test_the_defaults_are_the_measured_choice() -> None:
    assert (png.FILTER, png.LEVEL) == (png.ADAPTIVE, 6)
    assert write(PICTURE) == write(PICTURE, filter_type=png.FILTER, level=png.LEVEL)


def test_a_large_picture_in_several_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(png, "BAND_BYTES", 1 << 12)
    rgb = RNG.random((130, 211, 3), dtype=np.float32)
    assert np.array_equal(decoded16(write(rgb)).astype(np.int64), quantised(rgb, 16))


# Refusing what cannot be written ----------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"rgb": np.zeros((4, 4), np.float32)},
        {"rgb": np.zeros((4, 4, 4), np.float32)},
        {"rgb": np.zeros((0, 4, 3), np.float32)},
        {"bits": 12},
        {"filter_type": 6},
    ],
)
def test_a_bad_argument_is_a_value_error(arguments: dict[str, object]) -> None:
    settings: dict[str, object] = {
        "rgb": PICTURE,
        "bits": 16,
        "grey": False,
        "icc": None,
        "chunks": (),
        **arguments,
    }
    with pytest.raises(ValueError):
        png.write(**settings)  # pyright: ignore[reportArgumentType]


# Read back by seedvr2x's reader ------------------------------------------------------------------


@pytest.mark.parametrize(("bits", "grey"), [(16, False), (8, False), (16, True), (8, True)])
def test_the_reader_gives_back_what_the_writer_was_given(bits: int, grey: bool) -> None:
    data = write(PICTURE, bits, grey=grey, icc=PROFILE, chunks=COLOUR)
    read = picture.read(data)
    assert (read.format, read.bits, read.grey) == ("PNG", bits, grey)
    assert (read.icc, read.chunks, read.orientation) == (PROFILE, COLOUR, 1)
    again = write(
        read.pixels,
        read.bits,
        grey=read.grey,
        icc=read.icc,
        chunks=read.chunks,
    )
    assert again == data


# Against ffmpeg and ffprobe ----------------------------------------------------------------------


def ffmpeg_decode(path: Path, pix_fmt: str) -> npt.NDArray[np.int64]:
    dtype = "<u2" if pix_fmt.endswith(("48le", "16le")) else "u1"
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype).astype(np.int64)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("bits", "grey", "pix_fmt"),
    [(16, False, "rgb48le"), (8, False, "rgb24"), (16, True, "gray16le"), (8, True, "gray")],
)
def test_ffmpeg_decodes_the_same_codes(tmp_path: Path, bits: int, grey: bool, pix_fmt: str) -> None:
    rgb = RNG.random((11, 17, 3), dtype=np.float32)
    path = tmp_path / "out.png"
    path.write_bytes(write(rgb, bits, grey=grey, icc=PROFILE, chunks=COLOUR))
    expected = quantised(rgb, bits)
    if grey:
        mean = (rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / np.float32(3)
        expected = quantised(mean[..., np.newaxis], bits)
    assert np.array_equal(ffmpeg_decode(path, pix_fmt), expected.ravel())


def ffprobe_side_data(path: Path) -> dict[str, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_frames", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    frame = json.loads(out)["frames"][0]
    return {
        item["side_data_type"]: json.dumps(item, sort_keys=True)
        for item in frame.get("side_data_list", [])
    }


@needs_ffmpeg
def test_ffprobe_reads_the_profile(tmp_path: Path) -> None:
    # Not with an sRGB or a cICP chunk beside it, which ffmpeg's decoder takes in its place.
    path = tmp_path / "out.png"
    path.write_bytes(write(PICTURE, icc=PROFILE))
    side = ffprobe_side_data(path)
    assert "ICC profile" in side and f'"size": {len(PROFILE)}' in side["ICC profile"]
    assert "3x3 displaymatrix" not in side
    plain = tmp_path / "plain.png"
    plain.write_bytes(write(PICTURE))
    assert "ICC profile" not in ffprobe_side_data(plain)


@needs_ffmpeg
def test_ffprobe_reads_the_cicp(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    path.write_bytes(write(PICTURE, chunks=COLOUR[:1]))
    frame = subprocess.run(
        ["ffprobe", "-v", "error", "-show_frames", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tags = json.loads(frame)["frames"][0]
    assert (tags["color_primaries"], tags["color_transfer"]) == ("bt709", "iec61966-2-1")
    assert tags["color_range"] == "pc"


@needs_ffmpeg
@pytest.mark.parametrize("value", range(1, 9))
def test_ffprobe_sees_no_rotation_in_a_picture_turned_upright(tmp_path: Path, value: int) -> None:
    path = tmp_path / "ours.png"
    path.write_bytes(write(picture.upright(PICTURE, value)))
    assert "3x3 displaymatrix" not in ffprobe_side_data(path)
