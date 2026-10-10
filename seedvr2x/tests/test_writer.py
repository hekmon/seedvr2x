"""The writers, against the real ffmpeg: exact round trips, the yuv420p10le conversion and its tags
(skipped without a build seedvr2x accepts)."""

import json
import struct
import subprocess
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.checksums import frame_bytes
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.writer import (
    FFV1Writer,
    PNGWriter,
    Tags,
    open_writer,
    to_planar16,
    yuv_matrix,
)


def _usable() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

RATE = Fraction(24000, 1001)
BT709 = Tags("bt709", "bt709")


def frames_of(count: int, width: int, height: int, seed: int = 0) -> npt.NDArray[np.float32]:
    frames = np.random.default_rng(seed).random((count, height, width, 3), dtype=np.float32)
    frames[0, 0, :3] = ((-0.5, 0.0, 1.5), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))  # clipped, extremes
    return frames


def decoded(path: Path, pix_fmt: str) -> bytes:
    """The frames of path as stored, pix_fmt being the stored format: nothing converts."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True,
        check=True,
    ).stdout


def yuv420_planes(
    path: Path, width: int, height: int
) -> list[tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint16]]]:
    data = np.frombuffer(decoded(path, "yuv420p10le"), dtype="<u2")
    luma, chroma = width * height, (width // 2) * (height // 2)
    frames = data.reshape(-1, luma + 2 * chroma)
    return [
        (
            frame[:luma].reshape(height, width),
            frame[luma : luma + chroma].reshape(height // 2, width // 2),
            frame[luma + chroma :].reshape(height // 2, width // 2),
        )
        for frame in frames
    ]


def stream_tags(path: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json", "-show_entries"),
            "stream=pix_fmt,color_space,color_range,color_primaries,color_transfer,"
            "chroma_location,r_frame_rate",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def test_gbrp16le_round_trip_streamed(tmp_path: Path) -> None:
    frames = frames_of(6, 32, 16)
    path = tmp_path / "master.mkv"
    with FFV1Writer(path, "gbrp16le", 32, 16, RATE, BT709) as writer:
        for chunk in (frames[:1], frames[1:4], frames[4:]):
            writer.write(chunk)
    assert writer.written == 6
    planes = np.frombuffer(decoded(path, "gbrp16le"), dtype="<u2").reshape(6, 3, 16, 32)
    assert np.array_equal(planes, np.stack([to_planar16(frame) for frame in frames]))
    assert planes[0, :, 0, 1].tolist() == [65535] * 3 and planes[0, :, 0, 2].tolist() == [0] * 3
    tags = stream_tags(path)
    assert tags["pix_fmt"] == "gbrp16le" and tags["r_frame_rate"] == "24000/1001"
    assert (tags["color_space"], tags["color_range"]) == ("gbr", "pc")
    assert (tags["color_primaries"], tags["color_transfer"]) == ("bt709", "bt709")
    assert list(tmp_path.iterdir()) == [path]  # the temporary file renamed


def test_checksums_as_written(tmp_path: Path) -> None:
    # Each frame's CRC-32 as fed to ffmpeg, computed as it is written: for a gbrp16le master, of
    # the planes it holds and gives back (media/checksums.py).
    frames = frames_of(3, 64, 16)
    path = tmp_path / "m.mkv"
    with FFV1Writer(path, "gbrp16le", 64, 16, RATE, BT709) as writer:
        writer.write(frames[:2])
        writer.write(frames[2:])
    planes = decoded(path, "gbrp16le")
    size = 3 * 64 * 16 * 2
    assert writer.checksums == [zlib.crc32(planes[k * size : (k + 1) * size]) for k in range(3)]


def test_checksums_as_held(tmp_path: Path) -> None:
    # A yuv420p10le master holds what ffmpeg converts: its checksums come from ffmpeg's framehash,
    # the CRC-32 of each frame the file holds, as zlib computes it. A PNG's are of the rgb48be
    # pixels each file holds.
    frames = frames_of(3, 64, 16)
    yuv = tmp_path / "y.mkv"
    with FFV1Writer(yuv, "yuv420p10le", 64, 16, RATE, BT709) as writer:
        writer.write(frames)
    data, size = decoded(yuv, "yuv420p10le"), frame_bytes("yuv420p10le", 64, 16)
    assert writer.checksums == [zlib.crc32(data[k * size : (k + 1) * size]) for k in range(3)]
    assert writer._hashes is not None and not writer._hashes.exists()  # pyright: ignore[reportPrivateUsage]
    with (
        pytest.raises(RuntimeError),
        FFV1Writer(yuv, "yuv420p10le", 64, 16, RATE, BT709) as aborted,
    ):
        aborted.write(frames)
        raise RuntimeError
    assert aborted._hashes is not None and not aborted._hashes.exists()  # pyright: ignore[reportPrivateUsage]
    with PNGWriter(tmp_path / "p", 64, 16, RATE, BT709) as png:
        png.write(frames)
    assert png.checksums == [zlib.crc32(decoded(png.frame_path(k), "rgb48be")) for k in range(3)]


def test_framehash_refused_unless_ffmpeg_s(tmp_path: Path) -> None:
    from seedvr2x.media.writer import _framehash  # pyright: ignore[reportPrivateUsage]

    path = tmp_path / "f.framehash"
    path.write_text(
        "#format: frame checksums\n0,          0,          0,        1,     9216, 18cd7050\n"
    )
    assert _framehash(path) == [0x18CD7050]
    for line in ("0, 0, 0, 1, 9216", "1, 0, 0, 1, 9216, 18cd7050", "0, 0, 0, 1, 9216, 18cd705"):
        path.write_text(f"{line}\n")
        with pytest.raises(MediaError, match="not ffmpeg's framehash"):
            _framehash(path)


def test_stale_checksums_removed_when_replaced(tmp_path: Path) -> None:
    # stale: the checksums of what path held, removed only as the new file replaces it.
    stale = tmp_path / "m.mkv.crc32"
    stale.write_text("00000000\n")
    with (
        pytest.raises(RuntimeError),
        FFV1Writer(tmp_path / "m.mkv", "gbrp16le", 64, 16, RATE, BT709, stale=stale) as writer,
    ):
        writer.write(frames_of(1, 64, 16))
        raise RuntimeError
    assert stale.exists()
    with FFV1Writer(tmp_path / "m.mkv", "gbrp16le", 64, 16, RATE, BT709, stale=stale) as writer:
        writer.write(frames_of(1, 64, 16))
    assert not stale.exists()


def test_yuv_white_black_and_primaries(tmp_path: Path) -> None:
    # BT.709 limited range: white at 940, black at 64, chroma at 512 for both; each primary at
    # 64 + 876 Y', 512 + 896 (B' - Y') / 1.8556, 512 + 896 (R' - Y') / 1.5748, rounded.
    colours = [(1, 1, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    frames = np.stack([np.full((720, 1280, 3), colour, dtype=np.float32) for colour in colours])
    path = tmp_path / "yuv.mkv"
    with FFV1Writer(path, "yuv420p10le", 1280, 720, RATE, BT709) as writer:
        writer.write(frames)
    for (r, g, b), (y, u, v) in zip(colours, yuv420_planes(path, 1280, 720), strict=True):
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        expected = (
            round(64 + 876 * luma),
            round(512 + 896 * (b - luma) / 1.8556),
            round(512 + 896 * (r - luma) / 1.5748),
        )
        assert [np.unique(y).tolist(), np.unique(u).tolist(), np.unique(v).tolist()] == [
            [value] for value in expected
        ], (r, g, b)
    assert yuv420_planes(path, 1280, 720)[0][0][0, 0] == 940


def test_yuv_equals_ffv1_out(tmp_path: Path) -> None:
    # The yuv420p10le master of research/scripts/ffv1_out.py (FORMATS, Writer: lines 61-129),
    # validated in research/docs/output.md, from the same 16-bit planes. Its zscale on one slice
    # too, as ours (threads=1, since 2026-10-05): 720 rows make up to 11 slices.
    frames = frames_of(3, 1280, 720)
    ours = tmp_path / "ours.mkv"
    with FFV1Writer(ours, "yuv420p10le", 1280, 720, RATE, BT709) as writer:
        writer.write(frames)
    reference = tmp_path / "ffv1_out.mkv"
    zscale = "zscale=threads=1:rin=full:pin=709:tin=709:m=709:r=limited:p=709:t=709:d=none"
    subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"),
            *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", "1280x720"),
            *("-framerate", str(RATE), "-i", "-", "-map", "0:v:0", "-fps_mode", "passthrough"),
            *("-vf", f"{zscale}:c=left,format=yuv420p10le"),
            *("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"),
            *("-color_range", "tv", "-chroma_sample_location", "left"),
            *("-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"),
            *("-pix_fmt", "yuv420p10le", str(reference)),
        ],
        input=b"".join(to_planar16(frame).tobytes() for frame in frames),
        check=True,
    )
    assert decoded(ours, "yuv420p10le") == decoded(reference, "yuv420p10le")
    assert stream_tags(ours) == stream_tags(reference)


def test_yuv_chroma_sited_left(tmp_path: Path) -> None:
    # A red rectangle from column 32 and row 24, on black. Sited left, chroma sample j sits on
    # luma column 2j and row 2i + 0.5 (between two rows): bilinear decimation then gives column
    # 15 only black, column 16 a quarter black, and row 11 an eighth red (rows 21 to 24).
    # Centre siting (2j + 0.5) would give column 15 an eighth red; top-left siting, row 11 none.
    frame = np.zeros((48, 64, 3), dtype=np.float32)
    frame[24:, 32:, 0] = 1.0
    path = tmp_path / "sited.mkv"
    with FFV1Writer(path, "yuv420p10le", 64, 48, RATE, Tags()) as writer:
        writer.write(frame[None])
    _, _, v = yuv420_planes(path, 64, 48)[0]
    black, red = int(v[0, 0]), int(v[-1, -1])
    assert red > black + 300
    assert v[-1, 15] == black
    assert abs(int(v[-1, 16]) - (0.25 * black + 0.75 * red)) <= 1
    assert abs(int(v[11, -1]) - (0.875 * black + 0.125 * red)) <= 1
    assert stream_tags(path)["chroma_location"] == "left"


@pytest.mark.parametrize(
    ("tags", "size", "matrix"),
    [
        (BT709, (1280, 720), "bt709"),
        (Tags("smpte170m", "smpte170m"), (720, 480), "smpte170m"),
        (Tags("bt470bg", "bt470bg", "bt470bg"), (720, 576), "bt470bg"),
        (Tags("smpte170m", "smpte170m", "smpte170m"), (720, 480), "smpte170m"),
        (Tags("bt709", "bt709", "bt709"), (960, 540), "smpte170m"),
        (Tags("bt470bg", "bt470bg", "bt470bg"), (1920, 1080), "bt709"),
        (Tags("bt2020", "smpte2084"), (1920, 1080), "bt709"),
        (Tags(), (1280, 720), "bt709"),
    ],
)
def test_tags_copied(tmp_path: Path, tags: Tags, size: tuple[int, int], matrix: str) -> None:
    # Primaries and transfer as the source declares them, untagged staying untagged; the yuv
    # master's matrix from its size, and below HD from the source's BT.601 name (DESIGN.md,
    # Colour and shape).
    width, height = size
    frame = np.full((1, height, width, 3), 0.5, dtype=np.float32)
    for pix_fmt, expected in (
        ("gbrp16le", {"color_space": "gbr", "color_range": "pc"}),
        ("yuv420p10le", {"color_space": matrix, "color_range": "tv", "chroma_location": "left"}),
    ):
        path = tmp_path / f"{pix_fmt}.mkv"
        with FFV1Writer(path, pix_fmt, width, height, RATE, tags) as writer:
            writer.write(frame)
        found = stream_tags(path)
        assert {key: found.get(key) for key in expected} == expected
        assert found.get("color_primaries", "unknown") == (tags.primaries or "unknown")
        assert found.get("color_transfer", "unknown") == (tags.transfer or "unknown")


def test_yuv_matrix_by_size() -> None:
    assert yuv_matrix(1920, 1080) == yuv_matrix(1280, 720) == yuv_matrix(1024, 578) == "bt709"
    assert yuv_matrix(960, 540) == yuv_matrix(720, 576) == yuv_matrix(640, 480) == "smpte170m"


@pytest.mark.parametrize(
    ("source", "sd", "hd"),
    [
        ("bt470bg", "bt470bg", "bt709"),
        ("smpte170m", "smpte170m", "bt709"),
        ("bt709", "smpte170m", "bt709"),
        ("bt2020nc", "smpte170m", "bt709"),
        ("", "smpte170m", "bt709"),
    ],
)
def test_yuv_matrix_below_hd_named_as_the_source(source: str, sd: str, hd: str) -> None:
    # BT.601 below HD, under the source's own name when it has one (DESIGN.md, Colour and shape).
    assert yuv_matrix(720, 576, source) == yuv_matrix(640, 480, source) == sd
    assert yuv_matrix(1280, 720, source) == yuv_matrix(1920, 1080, source) == hd


def test_bt601_names_convert_alike(tmp_path: Path) -> None:
    # bt470bg and smpte170m have the same coefficients: only the tag differs.
    frames = frames_of(2, 64, 48)
    planes: list[bytes] = []
    for matrix in ("bt470bg", "smpte170m"):
        path = tmp_path / f"{matrix}.mkv"
        with FFV1Writer(path, "yuv420p10le", 64, 48, RATE, Tags(matrix=matrix)) as writer:
            writer.write(frames)
        assert stream_tags(path)["color_space"] == matrix
        planes.append(decoded(path, "yuv420p10le"))
    assert planes[0] == planes[1]


def png_chunks(path: Path) -> dict[str, bytes]:
    data, chunks = path.read_bytes()[8:], {}
    while data:
        length, kind = struct.unpack(">I4s", data[:8])
        chunks[kind.decode()] = data[8 : 8 + length]
        data = data[12 + length :]
    return chunks


@pytest.mark.parametrize(("tags", "cicp"), [(BT709, bytes((1, 1, 0, 1))), (Tags(), None)])
def test_png_round_trip_and_tags(tmp_path: Path, tags: Tags, cicp: bytes | None) -> None:
    frames = frames_of(3, 32, 16)
    with open_writer("png", tmp_path / "png", 32, 16, RATE, tags, start=10) as writer:
        writer.write(frames[:2])
        writer.write(frames[2:])
    assert sorted(p.name for p in (tmp_path / "png").iterdir()) == [
        "000010.png",
        "000011.png",
        "000012.png",
    ]
    for index, frame in enumerate(frames):
        path = tmp_path / "png" / f"{10 + index:06d}.png"
        rgb = np.frombuffer(decoded(path, "rgb48le"), dtype="<u2").reshape(16, 32, 3)
        assert np.array_equal(rgb, to_planar16(frame)[[2, 0, 1]].transpose(1, 2, 0))
        chunks = png_chunks(path)
        assert chunks["IHDR"][8:10] == bytes((16, 2))  # 16 bits, RGB
        assert chunks.get("cICP") == cicp


@pytest.mark.parametrize("output_format", ["gbrp16le", "yuv420p10le", "png"])
def test_frames_of_any_layout(tmp_path: Path, output_format: str) -> None:
    # The decode's frames are views of (C, t, H, W) planes, not C-ordered (H, W, 3) arrays: each
    # writer must give the same output for them.
    planes = np.random.default_rng(1).random((3, 2, 720, 1280), dtype=np.float32)
    outputs = []
    for name, frames in (("planar", planes.transpose(1, 2, 3, 0)), ("packed", None)):
        if frames is None:
            frames = np.ascontiguousarray(planes.transpose(1, 2, 3, 0))
        path = tmp_path / name
        with open_writer(output_format, path, 1280, 720, RATE, BT709) as writer:
            writer.write(frames)
        files = sorted(path.iterdir()) if path.is_dir() else [path]
        pix_fmt = "rgb48le" if output_format == "png" else output_format
        outputs.append(b"".join(decoded(file, pix_fmt) for file in files))
    assert outputs[0] == outputs[1]


def test_failure_removes_partial_output(tmp_path: Path) -> None:
    path = tmp_path / "master.mkv"
    with pytest.raises(RuntimeError, match="stop"):
        with FFV1Writer(path, "gbrp16le", 32, 16, RATE, BT709) as writer:
            writer.write(frames_of(2, 32, 16))
            raise RuntimeError("stop")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="the output is 32x16"):
        with FFV1Writer(path, "gbrp16le", 32, 16, RATE, BT709) as writer:
            writer.write(frames_of(1, 16, 16))
    assert list(tmp_path.iterdir()) == []


def test_png_abort_leaves_nothing(tmp_path: Path) -> None:
    # Frames written, then a failure: no directory left that could pass for a segment.
    with pytest.raises(RuntimeError, match="stop"):
        with PNGWriter(tmp_path / "png", 32, 16, RATE, BT709) as writer:
            writer.write(frames_of(3, 32, 16))
            raise RuntimeError("stop")
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("")
    with pytest.raises(MediaError, match="not an empty directory"):
        PNGWriter(tmp_path / "full", 32, 16, RATE, BT709)


def test_ffmpeg_failure_reported(tmp_path: Path) -> None:
    # An unknown tag: ffmpeg refuses its filter graph, and says so.
    with pytest.raises(MediaError, match=r"writing .*master\.mkv with ffmpeg"):
        with FFV1Writer(tmp_path / "master.mkv", "gbrp16le", 32, 16, RATE, Tags("nope")) as w:
            w.write(frames_of(1, 32, 16))
    assert list(tmp_path.iterdir()) == []


def test_png_writer_numbering(tmp_path: Path) -> None:
    writer = PNGWriter(tmp_path, 8, 8, RATE, Tags(), start=7)
    writer.write(np.zeros((1, 8, 8, 3), dtype=np.float32))
    assert writer.close() == 1 and writer.frame_path(7).is_file()
