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

from seedvr2x.media import ffmpeg, files
from seedvr2x.media.checksums import check_frames, frame_bytes
from seedvr2x.media.ffmpeg import MediaError, argument, input_args
from seedvr2x.media.probe import probe
from seedvr2x.media.source import examine
from seedvr2x.media.writer import (
    CommandWriter,
    FFV1Writer,
    PNGWriter,
    Tags,
    count_frames,
    count_packets,
    open_writer,
    to_planar16,
    to_rgb48be,
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


def levels(count: int, width: int, height: int) -> npt.NDArray[np.float32]:
    """count flat frames, each of a level of its own: nothing idet takes for combing."""
    ramp = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.broadcast_to(ramp[:, None, None, None], (count, height, width, 3)).copy()


def packet_times(path: Path) -> list[int]:
    """Each video packet's timestamp, in the stream's ticks, Matroska's milliseconds."""
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pts", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [int(line.rstrip(",")) for line in result.stdout.split()]


@pytest.mark.parametrize(
    ("kind", "rate"),
    [
        (kind, rate)
        for kind in ("gbrp16le", "yuv420p10le", "command")
        for rate in (Fraction(60000, 1001), Fraction(120000, 1001), Fraction(240000, 1001))
    ]
    + [("gbrp16le", Fraction(119)), ("gbrp16le", Fraction(121))],
)
def test_frames_numbered_at_the_rate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str, rate: Fraction
) -> None:
    # From 101 fps up ffmpeg guesses its raw input's rate among its standard ones, 120/1 for
    # 120000/1001, 119 and 121, 240/1 for 240000/1001, unless the input option -r gives it
    # (writer._planes_input): the master then declared the guess and sat on its grid, a place
    # skipped where the two rates part by a frame, at frame 500 of 120000/1001's and 240000/1001's
    # (seedvr2x read 1,101 frames back, one missing), nine times in 1,100 frames at 119; at 121 two
    # frames shared a millisecond. 60000/1001 was written right, and still is. A master's file,
    # and the stream a segment command reads, kept as it comes: every frame on its rate's
    # timeline to the tick, the rate declared the job's, and as many frames read back as written,
    # none missing.
    count = 1100
    path = tmp_path / "master.mkv"
    if kind == "command":
        writer = CommandWriter("cat > {out}", path, "gbrp16le", 32, 16, rate, BT709)
    else:
        writer = FFV1Writer(path, kind, 32, 16, rate, BT709)
    with writer:
        writer.write(levels(count, 32, 16))
    millisecond = Fraction(1, 1000)
    assert packet_times(path) == [
        int(Fraction(k) / rate / millisecond + Fraction(1, 2)) for k in range(count)
    ]
    assert probe(path).frame_rate == rate
    with caplog.at_level("WARNING", logger="seedvr2x"):
        source = examine(path)
    assert source.frame_rate == rate and source.frames == count and source.filled is None
    assert "missing" not in caplog.text


def test_png_holds_no_rate(tmp_path: Path) -> None:
    # A PNG has neither a rate nor a timestamp: at the rates ffmpeg takes for 120/1 or 240/1
    # (test_frames_numbered_at_the_rate), where the guess skips a place at frame 500 or gives
    # two frames one time (121), every frame is written all the same, in files of the same bytes
    # as at 24000/1001. The PNG writer's input has no -r.
    frames = levels(510, 32, 16)
    written: dict[Fraction, list[bytes]] = {}
    for rate in (RATE, Fraction(120000, 1001), Fraction(240000, 1001), Fraction(121)):
        directory = tmp_path / f"{rate.numerator}-{rate.denominator}"
        with PNGWriter(directory, 32, 16, rate, BT709) as writer:
            writer.write(frames)
        written[rate] = [file.read_bytes() for file in sorted(directory.iterdir())]
        assert len(written[rate]) == 510 and written[rate] == written[RATE], rate


# Matroska's elements (matroska.org's ids) that ffmpeg's muxer fills at random in every file it
# writes, each with its size in bytes: the segment's UID, the track's, held twice, in the track
# and in its tags' target, and the CRC-32 of each element that holds one of them (Info, Tracks,
# Tags). 44 bytes, the only ones two runs of one command differ by (measured on ffmpeg n9.0.2,
# 2026-10-12: 480 files).
DRAWN = {0x73A4: 16, 0x73C5: 8, 0x63C5: 8, 0xBF: 4}
# The elements those lie in, walked into: Segment, Info, Tracks, TrackEntry, Tags, Tag, Targets.
HOLDING = {0x18538067, 0x1549A966, 0x1654AE6B, 0xAE, 0x1254C367, 0x7373, 0x63C0}


def vint(data: bytes, at: int, marked: bool) -> tuple[int, int, bool]:
    """EBML's variable-length integer at data[at], its length the leading zeros of its first
    byte plus one: its value, with the length's marker bit for an element's id (`marked`) and
    without it for a size; where it ends; and whether it is a size left unknown, every bit set,
    as the segment's and the clusters' are in a stream written to a pipe."""
    first = data[at]
    length = 9 - first.bit_length()
    value = first if marked else first & ((1 << (8 - length)) - 1)
    unknown = not marked and value == (1 << (8 - length)) - 1
    for byte in data[at + 1 : at + length]:
        value, unknown = value << 8 | byte, unknown and byte == 0xFF
    return value, at + length, unknown


def without_the_drawn(data: bytes) -> bytes:
    """A Matroska file's bytes with those ffmpeg drew at random zeroed (DRAWN), found by walking
    the EBML tree from its start: element after element, id then size then body, into the
    elements that hold them (HOLDING). An element of unknown size runs to its parent's end: the
    segment on a pipe, walked into, and its clusters, where the walk ends, only clusters
    following. Held to 44 bytes, each element found with its size."""
    settled, found = bytearray(data), list[int]()

    def walk(at: int, end: int) -> None:
        while at < end:
            ident, after, _ = vint(data, at, marked=True)
            size, body, unknown = vint(data, after, marked=False)
            if ident in DRAWN:
                assert size == DRAWN[ident]
                settled[body : body + size] = bytes(size)
                found.append(ident)
            if ident in HOLDING:
                walk(body, end if unknown else body + size)
            if unknown:
                return
            at = body + size

    walk(0, len(data))
    assert sorted(found) == sorted([0x73A4, 0x73C5, 0x63C5, 0xBF, 0xBF, 0xBF]), found
    return bytes(settled)


def before_the_rate_option(kind: str, rate: str, out: Path) -> list[str]:
    """ffmpeg's command for a 32x16 output tagged BT.709 as each writer ran it before the input
    option -r came beside -framerate (writer._planes_input, 2026-10-12): written out here, not
    drawn from the writers, so that any option a writer gains or loses shows. `kind`: an FFV1
    master's pixel format, "command" for the stream a segment command reads, a gbrp16le master's
    on stdout, or "png"; `out`, the master's file, the PNG frames' directory, or where the
    yuv420p10le master's second output, its frames' hashes, goes for a stream on stdout."""
    ffmpeg = "ffmpeg -hide_banner -nostdin -nostats -loglevel error -y -f rawvideo".split()
    tagged = "color_primaries=bt709:color_trc=bt709"
    tags = "-color_primaries bt709 -color_trc bt709".split()
    rgb = f"setparams=colorspace=gbr:range=pc:{tagged}"
    rgb_tags = ["-colorspace", "rgb", "-color_range", "pc", *tags]
    if kind == "png":
        return [
            *(*ffmpeg, "-pix_fmt", "rgb48be", "-s", "32x16", "-framerate", rate, "-i", "-"),
            *("-map", "0:v:0", "-fps_mode", "passthrough", "-vf", rgb, *rgb_tags),
            *("-c:v", "png", "-pix_fmt", "rgb48be", "-f", "image2", "-start_number", "0"),
            str(out / "%06d.png"),
        ]
    planes = [*ffmpeg, "-pix_fmt", "gbrp16le", "-s", "32x16", "-framerate", rate, "-i", "-"]
    ffv1 = "-c:v ffv1 -level 3 -g 1 -slices 16 -slicecrc 1 -pix_fmt {} -f matroska"
    if kind != "yuv420p10le":
        master = [*("-fps_mode", "passthrough", *rgb_tags), *ffv1.format("gbrp16le").split()]
        return [
            *planes,
            "-map",
            "0:v:0",
            "-vf",
            rgb,
            *master,
            "pipe:1" if kind == "command" else str(out),
        ]
    yuv = (
        "zscale=threads=1:min=gbr:rin=full:m=170m:r=limited:c=left:pin=unspecified:p=unspecified"
        ":tin=unspecified:t=unspecified:d=none:f=bilinear,format=yuv420p10le,"
        f"setparams=colorspace=smpte170m:range=tv:chroma_location=left:{tagged}"
    )
    yuv_tags = "-fps_mode passthrough -colorspace smpte170m -color_range tv".split()
    yuv_tags += ["-chroma_sample_location", "left", *tags]
    return [
        *(*planes, "-filter_complex", f"[0:v]{yuv},split=2[master][hashed]"),
        *("-map", "[master]", *yuv_tags, *ffv1.format("yuv420p10le").split(), str(out)),
        *("-map", "[hashed]", *yuv_tags, "-c:v", "rawvideo", "-pix_fmt", "yuv420p10le"),
        *("-f", "framehash", "-hash", "crc32", str(out.with_suffix(".framehash"))),
    ]


@pytest.mark.parametrize("rate", ["24000/1001", "24", "25", "30000/1001", "50", "60000/1001"])
@pytest.mark.parametrize("kind", ["gbrp16le", "yuv420p10le", "command", "png"])
def test_files_at_the_rates_in_use_are_those_written_before_the_rate_option(
    tmp_path: Path, kind: str, rate: str
) -> None:
    # The input option -r, which numbers the masters' frames right from 101 fps up
    # (test_frames_numbered_at_the_rate), changes nothing at the six rates in use: each
    # Matroska writer's file, an FFV1 master at gbrp16le and at yuv420p10le and the stream a
    # segment command reads, kept as it comes, is the file of the command run before it,
    # -framerate alone, byte for byte once the 44 bytes ffmpeg draws at random for every
    # Matroska file are zeroed (without_the_drawn); the PNG writer's files, whose command has
    # no -r, as they are. 30 frames at 32x16; measured so through the writers on 30 and 1,100
    # frames, and on 3 at 1280x720 and at 1920x1080 (2026-10-12).
    frames = frames_of(30, 32, 16)
    written, old = tmp_path / "written.mkv", tmp_path / "old.mkv"
    if kind == "png":
        writer = PNGWriter(written, 32, 16, Fraction(rate), BT709)
        old.mkdir()
    elif kind == "command":
        writer = CommandWriter("cat > {out}", written, "gbrp16le", 32, 16, Fraction(rate), BT709)
    else:
        writer = FFV1Writer(written, kind, 32, 16, Fraction(rate), BT709)
    with writer:
        writer.write(frames)
    pack = to_rgb48be if kind == "png" else to_planar16
    fed = b"".join(pack(frame).tobytes() for frame in frames)
    # The stream on a pipe, as the segment command reads it: a file would be seeked in.
    stream = subprocess.run(
        before_the_rate_option(kind, rate, old), input=fed, capture_output=True, check=True
    ).stdout
    if kind == "png":
        names = sorted(file.name for file in old.iterdir())
        assert names == [f"{number:06d}.png" for number in range(30)]
        assert [(written / name).read_bytes() for name in names] == [
            (old / name).read_bytes() for name in names
        ]
        return
    before = stream if kind == "command" else old.read_bytes()
    assert without_the_drawn(written.read_bytes()) == without_the_drawn(before)
    # And the frames numbered at the rate, each on its timeline to the tick.
    millisecond = Fraction(1, 1000)
    assert packet_times(written) == [
        int(Fraction(k) / Fraction(rate) / millisecond + Fraction(1, 2)) for k in range(30)
    ]


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


@pytest.mark.parametrize("kind", ["gbrp16le", "yuv420p10le", "png"])
def test_stale_checksums_stay_until_the_new_file_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # The old file's checksums go once the new file's data is on the disk (files.replace_whole),
    # not before it: a sync that fails, or a stop during it, leaves them with the old file, which
    # they describe (DESIGN.md, Output, Assembly, decided on 2026-10-12). Once the new file is in
    # place they are gone.
    path = tmp_path / ("png" if kind == "png" else "m.mkv")
    stale = tmp_path / "m.crc32"
    stale.write_text("00000000\n")
    if kind != "png":
        path.write_bytes(b"the file before")
    partial = files.partial_path(path)
    sync = files._sync  # pyright: ignore[reportPrivateUsage]
    seen: list[bool] = []

    def failing(synced: Path) -> None:
        if synced == partial:
            seen.append(stale.exists())
            raise OSError(5, "input/output error")
        sync(synced)

    def writer() -> FFV1Writer | PNGWriter:
        if kind == "png":
            return PNGWriter(path, 32, 16, RATE, BT709, stale=stale)
        return FFV1Writer(path, kind, 32, 16, RATE, BT709, stale=stale)

    monkeypatch.setattr(files, "_sync", failing)
    with pytest.raises(OSError, match="input/output error"), writer() as failed:
        failed.write(frames_of(2, 32, 16))
    assert seen == [True] and stale.read_text() == "00000000\n"
    if kind != "png":
        assert path.read_bytes() == b"the file before"
    monkeypatch.setattr(files, "_sync", sync)
    path.unlink(missing_ok=True)
    with writer() as written:
        written.write(frames_of(2, 32, 16))
    assert not stale.exists() and path.exists()


# A relative name whose directory holds ": " (or begins with "-") is a file's for ffmpeg and
# ffprobe: they take "x: y/seg.mkv" for a protocol "x", and "-x" for an option (ffmpeg.argument).
AWKWARD = ["x: y", "-lead: ing"]


def test_argument_is_never_a_protocol_nor_an_option() -> None:
    assert argument(Path("x: y/seg.mkv")) == "./x: y/seg.mkv"
    assert argument(Path("-s.mkv")) == "./-s.mkv"
    assert argument(Path("../a: b.mkv")) == "./../a: b.mkv"
    assert argument(Path("/a: b/seg.mkv")) == "/a: b/seg.mkv"


@pytest.mark.parametrize("directory", AWKWARD)
@pytest.mark.parametrize("pix_fmt", ["gbrp16le", "yuv420p10le"])
def test_ffv1_to_a_relative_name_holding_a_colon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str, pix_fmt: str
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path(directory) / "seg.mkv"
    with FFV1Writer(path, pix_fmt, 64, 16, RATE, BT709) as writer:
        writer.write(frames_of(3, 64, 16))
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [directory]
    assert [entry.name for entry in (tmp_path / directory).iterdir()] == ["seg.mkv"]
    # What reads it back, by the same name: ffprobe's packet count and probe, ffmpeg's decode.
    assert count_packets(path) == count_frames(path) == 3
    assert probe(path).width == 64
    assert check_frames(input_args(path), pix_fmt, 64, 16, writer.checksums) == []


@pytest.mark.parametrize("directory", AWKWARD)
def test_png_to_a_relative_name_holding_a_colon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path(directory) / "png"
    with PNGWriter(path, 32, 16, RATE, BT709) as writer:
        writer.write(frames_of(2, 32, 16))
    assert sorted(entry.name for entry in path.iterdir()) == ["000000.png", "000001.png"]
    # verify's pattern, as cli._verify gives it.
    reading = ["-f", "image2", "-start_number", "0", "-i", argument(path / "%06d.png")]
    assert check_frames(reading, "rgb48be", 32, 16, writer.checksums) == []


@pytest.mark.parametrize("directory", AWKWARD)
def test_command_to_a_relative_name_holding_a_colon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    # {out} reaches the command as the name it writes, which its own ffmpeg takes for a file's.
    monkeypatch.chdir(tmp_path)
    path = Path(directory) / "seg.mkv"
    command = "ffmpeg -v error -f matroska -i - -c copy -f matroska {out}"
    with CommandWriter(command, path, "gbrp16le", 64, 16, RATE, BT709) as writer:
        writer.write(frames_of(3, 64, 16))
    assert [entry.name for entry in (tmp_path / directory).iterdir()] == ["seg.mkv"]
    assert count_frames(path) == 3


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
