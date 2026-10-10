"""The decode against the real ffmpeg: zscale's conversions, the formats read, the first pass and
the refusals (DESIGN.md, Input). Skipped without an ffmpeg seedvr2x accepts."""

import dataclasses
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.conversion import PIXEL_FORMATS, Conversion
from seedvr2x.media.decode import Decoder, decode_command, to_float32
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.probe import probe
from seedvr2x.media.scan import Scan, scan
from seedvr2x.media.source import examine
from seedvr2x.media.writer import FFV1Writer, Tags


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1"
)

WIDTH, HEIGHT = 64, 16  # multiples of 4: 4:1:1 and 4:1:0 included
YUV709 = Conversion("yuv420p", "709", "limited", "left")
RGB = Conversion("gbrp", "gbr", "full", "left")
RGB16 = Conversion("gbrp16le", "gbr", "full", "left")
Planes = tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint16]]


def run(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *args], input=data, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def raw_args(pix_fmt: str, path: Path, width: int = WIDTH, height: int = HEIGHT) -> list[str]:
    """Input options reading frames of pix_fmt laid out raw in path."""
    return [
        *("-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}", "-framerate", "25"),
        *("-i", str(path)),
    ]


def decode(
    input_args: list[str],
    conversion: Conversion,
    frames: int,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> npt.NDArray[np.uint16]:
    with Decoder(input_args, conversion, width, height, frames) as decoder:
        return decoder.read(frames)


def depth(pix_fmt: str) -> int:
    if pix_fmt.startswith(("rgb48", "bgr48")):
        return 16
    if pix_fmt.startswith("p0") or pix_fmt.startswith(("p2", "p4")):
        return int(pix_fmt[2:4])
    match = re.search(r"p(\d+)[lb]e$", pix_fmt)
    return int(match[1]) if match else 8


def subsampling(planar: str) -> tuple[int, int]:
    """Rows and columns per chroma sample of a planar format."""
    for chroma, factors in (
        ("420", (2, 2)),
        ("422", (1, 2)),
        ("444", (1, 1)),
        ("411", (1, 4)),
        ("410", (4, 4)),
        ("440", (2, 1)),
    ):
        if chroma in planar:
            return factors
    return 1, 1  # RGB


def random_planes(planar: str, frames: int, seed: int = 0) -> Planes:
    """Y, U, V (or R, G, B) planes for frames of a format, random over its whole depth."""
    rng = np.random.default_rng(seed)
    top = 1 << depth(planar)
    rows, columns = subsampling(planar)
    small = (frames, HEIGHT // rows, WIDTH // columns)
    first = rng.integers(0, top, (frames, HEIGHT, WIDTH), dtype=np.uint16)
    return (
        first,
        rng.integers(0, top, small, dtype=np.uint16),
        rng.integers(0, top, small, dtype=np.uint16),
    )


def pack(pix_fmt: str, planes: Planes) -> bytes:
    """Planes (T, H, W) of Y, U, V (or R, G, B) at the format's depth, laid out frame after frame
    as pix_fmt lays them."""
    return b"".join(pack_frame(pix_fmt, (a, b, c)) for a, b, c in zip(*planes, strict=True))


def pack_frame(pix_fmt: str, planes: Planes) -> bytes:
    a, b, c = planes
    bits = depth(pix_fmt)
    endian = ">" if pix_fmt.endswith("be") else "<"
    word = np.uint8 if bits == 8 else np.dtype(f"{endian}u2")

    def interleave(*parts: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
        return np.stack(parts, axis=-1)

    if pix_fmt.startswith("yuv"):
        layout = [a, b, c]
    elif pix_fmt.startswith("gbrp"):
        layout = [b, c, a]  # G, B, R
    elif pix_fmt in ("nv12", "nv16", "nv24"):
        layout = [a, interleave(b, c)]
    elif pix_fmt in ("nv21", "nv42"):
        layout = [a, interleave(c, b)]
    elif re.fullmatch(r"p[024]1\dle", pix_fmt):
        shift = np.uint16(16 - bits)  # in the most significant bits
        layout = [a << shift, interleave(b, c) << shift]
    elif pix_fmt == "yuyv422":
        layout = [interleave(a[:, 0::2], b, a[:, 1::2], c)]
    elif pix_fmt == "uyvy422":
        layout = [interleave(b, a[:, 0::2], c, a[:, 1::2])]
    elif pix_fmt == "yvyu422":
        layout = [interleave(a[:, 0::2], c, a[:, 1::2], b)]
    else:  # packed RGB: the letters give the order, 0 a padding byte
        name = pix_fmt.removesuffix("le").removesuffix("be").removesuffix("48").removesuffix("24")
        channels = {"r": a, "g": b, "b": c, "0": np.zeros_like(a)}
        layout = [interleave(*(channels[letter] for letter in name))]
    return b"".join(part.astype(word).tobytes() for part in layout)


def save(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


PLANAR = sorted(name for name, f in PIXEL_FORMATS.items() if f.planar == name)
PACKED = sorted(name for name, f in PIXEL_FORMATS.items() if f.planar != name)


@pytest.mark.parametrize("pix_fmt", PLANAR)
def test_planar_formats_reach_zscale_unconverted(tmp_path: Path, pix_fmt: str) -> None:
    # No scaler of ffmpeg's choosing may sit before zscale: its conversions are what DESIGN.md
    # rules out (swscale's 8 → 16-bit expansion, 255 → 65283).
    path = save(tmp_path, "frames.raw", pack(pix_fmt, random_planes(pix_fmt, 1)))
    conversion = dataclasses.replace(RGB if pix_fmt.startswith("gbrp") else YUV709, planar=pix_fmt)
    log = subprocess.run(
        decode_command(raw_args(pix_fmt, path), conversion, loglevel="verbose"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stderr
    assert "pixfmt:" + pix_fmt in log
    assert "auto-inserting" not in log


@pytest.mark.parametrize("pix_fmt", PACKED)
def test_packed_formats_repack_exactly(tmp_path: Path, pix_fmt: str) -> None:
    planar = PIXEL_FORMATS[pix_fmt].planar
    planes = random_planes(planar, 2, seed=len(pix_fmt))
    conversion = (
        dataclasses.replace(RGB, planar=planar)
        if planar.startswith("gbrp")
        else dataclasses.replace(YUV709, planar=planar)
    )
    packed = save(tmp_path, "packed.raw", pack(pix_fmt, planes))
    twin = save(tmp_path, "planar.raw", pack(planar, planes))
    assert np.array_equal(
        decode(raw_args(pix_fmt, packed), conversion, 2),
        decode(raw_args(planar, twin), conversion, 2),
    )


@pytest.mark.parametrize("pix_fmt", ["gbrp", "rgb24", "bgr24", "rgb0", "bgr0", "0rgb", "0bgr"])
def test_8bit_rgb_expands_exactly(tmp_path: Path, pix_fmt: str) -> None:
    # Every value on every channel, at every position class: v * 257, 255 -> 65535.
    values = np.arange(256, dtype=np.uint16).reshape(HEIGHT // 4, WIDTH)
    r = np.tile(values, (1, 4, 1))
    planes = (r, np.roll(r, 85), r[..., ::-1].copy())
    path = save(tmp_path, "frames.raw", pack(pix_fmt, planes))
    rgb = decode(raw_args(pix_fmt, path), RGB, 1)
    assert np.array_equal(rgb, np.stack(planes, axis=-1).astype(np.uint16) * np.uint16(257))


@pytest.mark.parametrize(("pix_fmt", "exact"), [("gbrp10le", True), ("gbrp12le", False)])
def test_high_depth_rgb_expansion(tmp_path: Path, pix_fmt: str, exact: bool) -> None:
    # zimg expands full-range RGB through float32: 10 bits land on round(v * 65535 / 1023)
    # exactly, 12 bits within one code of it (measured; the fp16 cast that follows rounds by up
    # to 32 codes near white).
    bits = depth(pix_fmt)
    values = np.arange(1 << bits, dtype=np.uint16).reshape(1, -1, WIDTH)
    planes = (values, values, values)
    height = values.shape[1]
    path = save(tmp_path, "frames.raw", pack(pix_fmt, planes))
    rgb = decode(
        raw_args(pix_fmt, path, height=height),
        dataclasses.replace(RGB, planar=pix_fmt),
        1,
        height=height,
    )
    expected = np.floor(values.astype(np.float64) * 65535 / ((1 << bits) - 1) + 0.5)
    difference = np.abs(rgb[..., 0].astype(np.int64) - expected.astype(np.int64))
    assert difference.max() == (0 if exact else 1)


@pytest.mark.parametrize("pix_fmt", ["gbrp16le", "rgb48le", "bgr48be"])
def test_16bit_rgb_is_kept(tmp_path: Path, pix_fmt: str) -> None:
    planes = random_planes("gbrp16le", 1)
    path = save(tmp_path, "frames.raw", pack(pix_fmt, planes))
    conversion = dataclasses.replace(RGB, planar="gbrp16le")
    assert np.array_equal(
        decode(raw_args(pix_fmt, path), conversion, 1)[0], np.stack([p[0] for p in planes], axis=-1)
    )


@pytest.mark.parametrize(
    ("pix_fmt", "white", "black", "neutral"),
    [
        ("yuv420p", 235, 16, 128),
        ("yuv420p10le", 940, 64, 512),
        ("yuv422p10le", 940, 64, 512),
        ("yuv444p12le", 3760, 256, 2048),
        ("yuv420p16le", 60160, 4096, 32768),
    ],
)
@pytest.mark.parametrize("matrix", ["709", "170m"])
def test_limited_white_and_black(
    tmp_path: Path, pix_fmt: str, white: int, black: int, neutral: int, matrix: str
) -> None:
    luma = np.full((1, HEIGHT, WIDTH), black, dtype=np.uint16)
    luma[..., : WIDTH // 2] = white
    rows, columns = subsampling(pix_fmt)
    chroma = np.full((1, HEIGHT // rows, WIDTH // columns), neutral, dtype=np.uint16)
    path = save(tmp_path, "frames.raw", pack(pix_fmt, (luma, chroma, chroma)))
    conversion = Conversion(pix_fmt, matrix, "limited", "left")
    rgb = decode(raw_args(pix_fmt, path), conversion, 1)[0]
    assert (rgb[:, : WIDTH // 2] == 65535).all() and (rgb[:, WIDTH // 2 :] == 0).all()


def test_full_range_white_and_black(tmp_path: Path) -> None:
    luma = np.full((1, HEIGHT, WIDTH), 0, dtype=np.uint16)
    luma[..., : WIDTH // 2] = 255
    chroma = np.full((1, HEIGHT // 2, WIDTH // 2), 128, dtype=np.uint16)
    path = save(tmp_path, "frames.raw", pack("yuvj420p", (luma, chroma, chroma)))
    conversion = Conversion("yuvj420p", "470bg", "full", "center")
    rgb = decode(raw_args("yuvj420p", path), conversion, 1)[0]
    assert (rgb[:, : WIDTH // 2] == 65535).all() and (rgb[:, WIDTH // 2 :] == 0).all()


def ffv1(path: Path, rgb16: npt.NDArray[np.uint16], vf: str, pix_fmt: str, *tags: str) -> Path:
    """An FFV1 file from 16-bit RGB frames (T, H, W, 3), converted by the filters vf to pix_fmt
    and tagged."""
    _, height, width, _ = rgb16.shape
    planes = np.ascontiguousarray(rgb16.transpose(0, 3, 1, 2)[:, [1, 2, 0]]).astype("<u2")
    run(
        *raw_args("gbrp16le", Path("-"), width, height),
        *("-vf", f"{vf},format={pix_fmt}", "-c:v", "ffv1", *tags, str(path)),
        data=planes.tobytes(),
    )
    return path


def yuv_file(path: Path, data: bytes, pix_fmt: str, **tags: str) -> Path:
    """An FFV1 file holding the raw frames data as they are, tagged with setparams' names. The
    frames carry the tags too: else ffmpeg converts untagged frames to the tags asked of the
    encoder (BT.601 to BT.709, seen with n9.0.2)."""
    options = {"range": "-color_range", "chroma_location": "-chroma_sample_location"}
    setparams = ":".join(f"{name}={value}" for name, value in tags.items())
    run(
        *raw_args(pix_fmt, Path("-")),
        *("-vf", f"setparams={setparams}", "-c:v", "ffv1"),
        *(arg for name, value in tags.items() for arg in (options.get(name, f"-{name}"), value)),
        str(path),
        data=data,
    )
    stored = run("-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-")
    assert stored == data  # tagged, not converted
    return path


def test_chroma_siting_is_read_and_applied(tmp_path: Path) -> None:
    # Chroma varying along x, well under the 4:2:0 limit: sited left or centre, the encode
    # differs by half a pixel, which only the right reading undoes.
    x = np.arange(WIDTH) * 2 * np.pi / 16
    red = 0.5 + 0.3 * np.sin(x)
    blue = 0.5 - 0.3 * np.sin(x + 1)
    rgb = np.stack(np.broadcast_arrays(red, np.full(WIDTH, 0.5), blue), axis=-1)
    rgb16 = np.rint(np.broadcast_to(rgb, (2, HEIGHT, WIDTH, 3)) * 65535).astype(np.uint16)
    errors: dict[tuple[str, str], float] = {}
    for sited in ("left", "center"):
        vf = f"zscale=rin=full:m=709:r=limited:c={sited}:f=bilinear:d=none"
        tags = ["-colorspace", "bt709", "-color_range", "tv", "-chroma_sample_location", sited]
        path = ffv1(tmp_path / f"{sited}.mkv", rgb16, vf, "yuv420p10le", *tags)
        source = examine(path)
        assert (source.conversion.chroma_location, source.conversion.guessed) == (sited, ())
        for reading in ("left", "center"):
            conversion = dataclasses.replace(source.conversion, chroma_location=reading)
            decoded = decode(["-i", str(path)], conversion, 2)
            errors[sited, reading] = float(np.abs(decoded.astype(np.float64) - rgb16).mean())
    assert errors["left", "left"] < errors["left", "center"] / 2
    assert errors["center", "center"] < errors["center", "left"] / 2


def test_matrix_applied_primaries_and_transfer_never(tmp_path: Path) -> None:
    luma = np.full((1, HEIGHT, WIDTH), 81, dtype=np.uint16)  # a saturated red, 8-bit BT.601
    data = pack(
        "yuv420p", (luma, np.full((1, 8, 32), 90, np.uint16), np.full((1, 8, 32), 240, np.uint16))
    )
    common = {"range": "tv", "chroma_location": "left"}
    files: dict[str, dict[str, str]] = {
        "709": {"colorspace": "bt709"},
        "601": {"colorspace": "smpte170m"},
        "709, SD primaries": {
            "colorspace": "bt709",
            "color_primaries": "smpte170m",
            "color_trc": "smpte170m",
        },
        # A transfer far from BT.709's, HDR's being refused (test_hdr_refused).
        "709, linear": {"colorspace": "bt709", "color_primaries": "bt2020", "color_trc": "linear"},
    }
    decoded = {}
    for name, tags in files.items():
        path = yuv_file(tmp_path / f"{len(decoded)}.mkv", data, "yuv420p", **tags, **common)
        source = examine(path)
        assert source.conversion.guessed == ()
        assert source.stream.color_transfer == tags.get("color_trc", "")
        with source.decoder() as decoder:
            decoded[name] = decoder.read(1)
    assert not np.array_equal(decoded["709"], decoded["601"])
    assert np.array_equal(decoded["709"], decoded["709, SD primaries"])
    assert np.array_equal(decoded["709"], decoded["709, linear"])


def test_mjpeg_declares_its_centre_siting(tmp_path: Path) -> None:
    # The JPEG family is sited centre (JFIF): ffmpeg declares it, so the tags rule covers it
    # (DESIGN.md, Input).
    path = tmp_path / "white.avi"
    run(
        "-f",
        "lavfi",
        "-i",
        "color=white:s=64x48:r=25:d=0.12",
        "-pix_fmt",
        "yuvj420p",
        "-c:v",
        "mjpeg",
        "-q:v",
        "1",
        str(path),
    )
    source = examine(path)
    assert source.conversion == Conversion("yuvj420p", "470bg", "full", "center")
    with source.decoder() as decoder:
        assert (decoder.read(source.frames) == 65535).all()


def test_8bit_rgb_master_reads_as_numz(tmp_path: Path) -> None:
    # Milestone 1's input is such a file (FFV1 bgr0): every 8-bit value on every channel must
    # read as numz's uint8 / 255 in float32, bit for bit.
    values = np.arange(256, dtype=np.uint8)
    frame = np.stack([values, values[::-1], np.roll(values, 85)], axis=-1)[None, None]
    frames = np.repeat(np.repeat(frame, 2, axis=1), 5, axis=0)  # (5, 2, 256, 3)
    path = tmp_path / "rgb.mkv"
    run(
        *raw_args("rgb24", Path("-"), 256, 2),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgr0",
        str(path),
        data=frames.tobytes(),
    )
    source = examine(path)
    assert (source.stream.pix_fmt, source.conversion.planar, source.frames) == ("bgr0", "gbrp", 5)
    with source.decoder() as decoder:
        decoded = to_float32(decoder.read(5))
    expected = frames.astype(np.float32) / 255.0
    assert np.array_equal(decoded.view(np.uint32), expected.view(np.uint32))


def test_frames_are_counted(tmp_path: Path) -> None:
    path = tmp_path / "count.mkv"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.4", "-c:v", "ffv1", str(path))
    source = examine(path)
    assert source.frames == 10
    with source.decoder() as decoder:
        assert len(decoder.read(4)) == 4
        assert len(decoder.read(10)) == 6
    with pytest.raises(MediaError, match="10 frames, the first pass counted 11"):
        with dataclasses.replace(source, frames=11).decoder() as decoder:
            decoder.read(11)
    with pytest.raises(MediaError, match="more frames than the 9"):
        with dataclasses.replace(source, frames=9).decoder() as decoder:
            decoder.read(10)


def test_cfr_matroska_accepted(tmp_path: Path) -> None:
    # Matroska stores milliseconds: 23.976 fps frames last 41 and 42 ms.
    path = tmp_path / "ntsc.mkv"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001:d=1", "-c:v", "ffv1", str(path))
    source = examine(path)
    assert (source.frames, source.stream.frame_rate) == (24, Fraction(24000, 1001))


def test_vfr_refused(tmp_path: Path) -> None:
    # 12 frames at 24 fps, then 12 at 30: the classic of anime DVDs, which Matroska declares as
    # 24/1 for both rates (sptenc).
    path = tmp_path / "vfr.mkv"
    vf = "setpts='if(lt(N,12),N/24,0.5+(N-12)/30)/TB'"
    run(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=64x48:r=24:d=1",
        "-vf",
        vf,
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        str(path),
    )
    with pytest.raises(MediaError, match="variable frame rate") as refused:
        examine(path)
    # No constant rate holds such a mix within half a frame: no guidance (media/rate.py).
    assert "--frame-rate" not in str(refused.value)


def test_drifted_join_scanned_without_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The join the refusal of a directory gives (cli._directory_refused): 32 frames at
    # 24000/1001 split losslessly, one per segment, joined by ffmpeg's concat demuxer, each
    # segment starting 41 ms after the one before, where a frame lasts 41.708. From frame 30 on,
    # half a frame off the grid: no error for the first pass to report, every frame counted, and
    # the steady 41 ms accepted: read at the rate declared, and warned of (source._strays), a
    # join at every frame being a constant rate of its own, 1000/41 fps, named there, where
    # joins further apart are no rate's (tests/test_rate.py).
    master, joined, listed = tmp_path / "master.mkv", tmp_path / "joined.mkv", tmp_path / "list"
    frames = 32
    run(
        *("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001", "-frames:v", str(frames)),
        *("-c:v", "ffv1", "-g", "1", str(master)),
    )
    run(
        *("-i", str(master), "-map", "0:v", "-c", "copy", "-f", "segment", "-reset_timestamps"),
        *("1", "-segment_frames", ",".join(str(k) for k in range(1, frames))),
        str(tmp_path / "seg_%06d.mkv"),
    )
    listed.write_text("".join(f"file 'seg_{k:06d}.mkv'\n" for k in range(frames)))
    run("-f", "concat", "-i", str(listed), "-c", "copy", str(joined))
    assert scan(joined) == Scan(frames, frames - 1, Fraction(41000), Fraction(41000))
    source = examine(joined)
    assert (source.frames, source.stream.frame_rate) == (frames, Fraction(24000, 1001))
    assert "reported errors" not in caplog.text
    assert (
        f"{joined}: 2 of its 32 frames half a frame or more from their place on the timeline of"
        " the rate it declares, 24000/1001, from frame 30, up to 22 ms (0.53 of a frame): read at"
        " the rate it declares all the same, frame after frame, as a file joined by ffmpeg's"
        " concat demuxer is; its timestamps follow 1000/41 fps (24.39) exactly"
    ) in caplog.text


def test_interlaced_refused(tmp_path: Path) -> None:
    path = tmp_path / "tff.mkv"
    vf = "setfield=tff"  # the option -field_order alone is overridden by the frames' own
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2", "-vf", vf, "-c:v", "ffv1", str(path))
    with pytest.raises(MediaError, match=r"interlaced \(field order"):
        examine(path)


@pytest.mark.parametrize("option", [["-display_rotation", "90"], ["-display_hflip"]])
def test_rotated_or_flipped_refused(tmp_path: Path, option: list[str]) -> None:
    base, path = tmp_path / "base.mkv", tmp_path / "rotated.mp4"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2", "-c:v", "ffv1", str(base))
    run(*option, "-i", str(base), "-c", "copy", str(path))
    with pytest.raises(MediaError, match="rotated or flipped"):
        examine(path)


@pytest.mark.skipif(shutil.which("mkvpropedit") is None, reason="needs mkvpropedit")
def test_declared_crop_refused(tmp_path: Path) -> None:
    path = tmp_path / "crop.mkv"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2", "-c:v", "ffv1", str(path))
    subprocess.run(
        ["mkvpropedit", "-q", str(path), "--edit", "track:v1", "--set", "pixel-crop-bottom=8"],
        check=True,
    )
    with pytest.raises(MediaError, match="cropped by its container"):
        examine(path)


@pytest.mark.parametrize(
    ("transfer", "refusal"),
    [
        (
            "smpte2084",
            "HDR, transfer smpte2084 (PQ): not supported: the model was trained on SDR video, what"
            " it makes of PQ-coded pixels is unmeasured, and HDR10's metadata (mastering display,"
            " MaxCLL) isn't carried",
        ),
        (
            "arib-std-b67",
            "HDR, transfer arib-std-b67 (HLG): not supported: the model was trained on SDR video,"
            " and what it makes of HLG-coded pixels is unmeasured",
        ),
        ("bt2020-10", ""),  # BT.2020's SDR transfer, read
    ],
)
def test_hdr_refused(tmp_path: Path, transfer: str, refusal: str) -> None:
    # BT.2020, as HDR10 and HLG are, and tagged on the frames: n9.0.2 writes the frames' transfer
    # and primaries, which -color_trc and -color_primaries alone leave unknown.
    path = tmp_path / "hdr.mkv"
    tags = f"colorspace=bt2020nc:color_primaries=bt2020:color_trc={transfer}:range=tv"
    vf = f"setparams={tags},format=yuv420p10le"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2", "-vf", vf, "-c:v", "ffv1", str(path))
    assert probe(path).color_transfer == transfer  # as ffprobe reads it back
    if not refusal:
        assert examine(path).frames == 5
        return
    with pytest.raises(MediaError) as refused:
        examine(path)
    tone_map = (
        "for an SDR upscale, tone-map the source to SDR first, a grading choice for your own"
        " tools, as a gamut conversion is"
    )
    assert str(refused.value) == f"{path}: {refusal}; {tone_map}"


def test_every_declared_reason_in_one_refusal(tmp_path: Path) -> None:
    # Interlaced, PQ and rotated: each reason said at once, numbered, interlacing first, refused in
    # every version, so that tone-mapping the source never ends on another of these reasons.
    base, path = tmp_path / "base.mkv", tmp_path / "rotated.mp4"
    tags = "colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084:range=tv"
    vf = f"setfield=tff,setparams={tags},format=yuv420p10le"
    run("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2", "-vf", vf, "-c:v", "ffv1", str(base))
    run("-display_rotation", "90", "-i", str(base), "-c", "copy", str(path))
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == (
        f"{path}: 3 reasons: (1) interlaced (field order tb): not supported; (2) HDR, transfer"
        " smpte2084 (PQ): not supported: the model was trained on SDR video, what it makes of"
        " PQ-coded pixels is unmeasured, and HDR10's metadata (mastering display, MaxCLL) isn't"
        " carried; for an SDR upscale, tone-map the source to SDR first, a grading choice for your"
        " own tools, as a gamut conversion is; (3) rotated or flipped by its display matrix (0,"
        " -65536, 0, 65536, 0, 0, 0, 0, 1073741824): not supported"
    )


def test_sample_aspect_declared_and_overridden(tmp_path: Path) -> None:
    path = tmp_path / "ntsc_wide.mkv"
    run(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=720x480:r=24000/1001:d=0.2",
        "-vf",
        "setsar=32/27",
        "-c:v",
        "ffv1",
        str(path),
    )
    source = examine(path)
    assert (source.sample_aspect, source.display_aspect) == (Fraction(32, 27), Fraction(16, 9))
    assert examine(path, sample_aspect=Fraction(8, 9)).display_aspect == Fraction(4, 3)


def test_skipped_frames_are_counted_not_converted(tmp_path: Path) -> None:
    # Frames skipped are decoded and dropped: the next ones read are those a full read gives
    # there, and the count at the end still holds.
    planes = random_planes("gbrp16le", 7)
    path = save(tmp_path, "rgb.raw", pack("gbrp16le", planes))
    conversion = Conversion("gbrp16le", "gbr", "full", "left")
    whole = decode(raw_args("gbrp16le", path), conversion, 7)
    with Decoder(raw_args("gbrp16le", path), conversion, WIDTH, HEIGHT, 7) as decoder:
        assert decoder.skip(2) == 2
        assert np.array_equal(decoder.read(1), whole[2:3])
        assert decoder.skip(3) == 3 and decoder.decoded == 6
        assert np.array_equal(decoder.read(1), whole[6:])
    with Decoder(raw_args("gbrp16le", path), conversion, WIDTH, HEIGHT, 7) as decoder:
        assert decoder.skip(9) == 7  # fewer only at the end


def ffv1_copy(path: Path, frames: npt.NDArray[np.uint16]) -> Path:
    """frames, (T, H, W, 3), written as split's input copies are (runtime/run.py): FFV1 gbrp16le,
    each slice with its CRC."""
    with FFV1Writer(path, "gbrp16le", WIDTH, HEIGHT, Fraction(25), Tags()) as writer:
        writer.write(to_float32(frames))
    return path


def flipped(path: Path, frame: int) -> Path:
    """A copy of the file at path, a byte flipped in the middle of frame `frame`'s packet."""
    probed = run_probe("-show_entries", "packet=pos,size", "-of", "json", str(path))
    packet = json.loads(probed)["packets"][frame]
    data = bytearray(path.read_bytes())
    data[int(packet["pos"]) + int(packet["size"]) // 2] ^= 0xFF
    damaged = path.with_name(f"flipped-{path.name}")
    damaged.write_bytes(data)
    return damaged


def run_probe(*args: str) -> bytes:
    result = subprocess.run(["ffprobe", "-v", "error", *args], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def test_strict_read_enforces_slice_crcs(tmp_path: Path) -> None:
    # ffmpeg decodes through an FFV1 slice failing its CRC, hides it under the frame before's and
    # exits 0; a strict read fails instead, saying so, and gives no frame wrong before it fails,
    # ffmpeg reporting a frame's damage before writing the frame. 200 frames, so that ffmpeg is
    # still decoding ahead when the damaged one comes.
    frames = np.random.default_rng(0).integers(0, 65536, (200, HEIGHT, WIDTH, 3), dtype=np.uint16)
    copy = ffv1_copy(tmp_path / "copy.mkv", frames)
    damaged = flipped(copy, 150)
    with Decoder(input_args(damaged), RGB16, WIDTH, HEIGHT, 200) as decoder:
        through = decoder.read(200)
    assert np.array_equal(through[:150], frames[:150])
    assert np.array_equal(through[151:], frames[151:])
    assert not np.array_equal(through[150], frames[150])
    given: list[npt.NDArray[np.uint16]] = []
    with (
        pytest.raises(MediaError, match=r"fails a strict read: .*slice CRC mismatch"),
        Decoder(input_args(damaged), RGB16, WIDTH, HEIGHT, 200, strict=True) as decoder,
    ):
        while True:
            given.append(decoder.read(1)[0])
    assert len(given) <= 150
    assert all(np.array_equal(frame, frames[index]) for index, frame in enumerate(given))
    with Decoder(input_args(copy), RGB16, WIDTH, HEIGHT, 200, strict=True) as decoder:
        assert np.array_equal(decoder.read(200), frames)


def test_checksums_catch_what_ffmpeg_decodes_silently(tmp_path: Path) -> None:
    # A damaged slice size makes FFV1 v3 skip slices without a word (libavcodec/ffv1dec.c derives
    # each frame's slice count from the sizes): on a flat frame, whose slices are all one size,
    # ffmpeg reports nothing, even to a strict read, and exits 0, the area skipped keeping what
    # its frame buffer held. The frames' checksums catch it before the frame is given. Random
    # frames but frame 50, flat at 255, the top byte of its slice 14's size flipped: neither black,
    # which a buffer newly allocated holds, nor any frame a reused buffer can hold, so the frame
    # is wrong whatever ffmpeg's frame threads (one per CPU, 16 at most) and buffers.
    width, height = 64, 48
    frames = np.random.default_rng(5).integers(0, 65536, (60, height, width, 3), dtype=np.uint16)
    frames[50] = 255
    copy = tmp_path / "copy.mkv"
    with FFV1Writer(copy, "gbrp16le", width, height, Fraction(25), Tags()) as writer:
        writer.write(to_float32(frames))
    probed = run_probe("-show_entries", "packet=pos,size", "-of", "json", str(copy))
    packet = json.loads(probed)["packets"][50]
    end = int(packet["pos"]) + 4 + int(packet["size"])  # after the block's track, time and flags
    data = bytearray(copy.read_bytes())
    last = int.from_bytes(data[end - 8 : end - 5], "big")  # slice 15's size, in its 8-byte trailer
    data[end - (last + 8) - 8] ^= 0xFF
    damaged = tmp_path / "damaged.mkv"
    damaged.write_bytes(data)
    with Decoder(input_args(damaged), RGB16, width, height, 60, strict=True) as decoder:
        through = decoder.read(60)
    assert np.array_equal(through[:50], frames[:50]) and np.array_equal(through[51:], frames[51:])
    assert not np.array_equal(through[50], frames[50])
    given: list[npt.NDArray[np.uint16]] = []
    checksums = writer.checksums
    with (
        pytest.raises(MediaError, match=r"frame 50: CRC-32 [0-9a-f]{8}, where [0-9a-f]{8} was"),
        Decoder(input_args(damaged), RGB16, width, height, strict=True, checksums=checksums) as d,
    ):
        while True:
            given.append(d.read(1)[0])
    assert len(given) == 50
    assert all(np.array_equal(frame, frames[index]) for index, frame in enumerate(given))
    with Decoder(input_args(copy), RGB16, width, height, strict=True, checksums=checksums) as d:
        assert np.array_equal(d.read(60), frames)
        with pytest.raises(ValueError, match="can't be checked"):
            d.skip(1)
    with pytest.raises(ValueError, match="60 checksums for 59 frames"):
        Decoder(input_args(copy), RGB16, width, height, 59, checksums=writer.checksums)


@pytest.mark.parametrize("kept", [0.5, 0.999])
def test_strict_read_refuses_a_file_cut_short(tmp_path: Path, kept: float) -> None:
    # Cut short, even in what follows the last frame: ffmpeg says so, which fails a strict read.
    frames = np.random.default_rng(1).integers(0, 65536, (40, HEIGHT, WIDTH, 3), dtype=np.uint16)
    data = ffv1_copy(tmp_path / "copy.mkv", frames).read_bytes()
    cut = tmp_path / "cut.mkv"
    cut.write_bytes(data[: int(len(data) * kept)])
    with (
        pytest.raises(MediaError, match="fails a strict read"),
        Decoder(input_args(cut), RGB16, WIDTH, HEIGHT, 40, strict=True) as decoder,
    ):
        decoder.read(40)
