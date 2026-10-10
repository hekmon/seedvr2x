"""The frame index and the reads through it (DESIGN.md, Input: frame-exact reading from any frame;
research/docs/seeking.md), on files made here at a small size, a few hundred frames, the kinds of
seek_test.py's synth: x264 and x265 with open GOPs in Matroska, MP4 and MPEG-TS (start time 1.48 s),
MPEG-2 with open GOPs in MPEG-PS (a VOB, 0.54 s) and MPEG-TS (1.44 s); and three of seeking.md's
failure mechanisms made on purpose: an MPEG-2 stream whose colour description appears at its 15th
frame, which makes ffmpeg rebuild its filter graph, as S6's (mechanism 8); an H.264 stream whose
PPS changes in-band, its later keyframes no entry points, as S1's and S10's (mechanism 6); damaged
H.264 and MPEG-2 files (mechanism 9).

Each index is held to a full decode's framemd5, ffmpeg's own (the frames' count and pts), and to
zlib's CRC-32 of the frames as decoded; each read to that decode, by the MD5 of every frame given
as the decoder gives it (Reader's md5). Skipped without an ffmpeg seedvr2x accepts; a kind
without its encoder (libx264, libx265, mpeg2video), saying which.

Seeks hash the frames as decoded, every packed and semi-planar format a file holds included. The
reads never wait on ffmpeg while it waits on them (media/reader.py): a hash held back behind
frames still to come through stdout, as the box's hang had it (2026-10-09), made on purpose; an
ffmpeg that stalls, a fake one on PATH. Each such read runs bounded, a hang failing the test."""

import json
import logging
import os
import queue
import random
import re
import subprocess
import threading
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_probe import has_encoder

from seedvr2x.media import ffmpeg, index
from seedvr2x.media import reader as reading
from seedvr2x.media.conversion import Conversion
from seedvr2x.media.decode import Decoder
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.reader import READ_AHEAD, Reader, read_command
from seedvr2x.media.scan import Scan, _Decoded, read_packets, scan
from seedvr2x.media.source import Source, examine


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1"
)

SIZE = "160x96"
SOURCE = ("-f", "lavfi", "-i", f"testsrc2=s={SIZE}:r=25")
# seek_test.py's SYNTH_ENCODES, open GOPs on purpose: x264's non-IDR I-frames with a recovery
# point, x265's CRA with RASL pictures, MPEG-2's GOPs of 15 with 2 leading B-frames (closed_gop=0,
# ffmpeg's default without +cgop).
X264 = "keyint=48:min-keyint=24:open-gop=1:bframes=3:b-pyramid=normal:log-level=error"
X265 = "keyint=48:min-keyint=24:open-gop=1:bframes=4:log-level=error"
ENCODES = {
    "h264": ("-c:v", "libx264", "-preset", "veryfast", "-x264-params", X264),
    "hevc": ("-c:v", "libx265", "-preset", "veryfast", "-x265-params", X265),
    "mpeg2": ("-c:v", "mpeg2video", "-b:v", "1M", "-g", "15", "-bf", "2"),
}
# Each kind: its encoder and file. TS and MP4 are stream copies of the Matroska or VOB file, as
# seek_test.py's are.
KINDS = {
    "x264 MKV": ("libx264", "h264.mkv"),
    "x264 MP4": ("libx264", "h264.mp4"),
    "x264 TS": ("libx264", "h264.ts"),
    "x265 MKV": ("libx265", "hevc.mkv"),
    "x265 MP4": ("libx265", "hevc.mp4"),
    "x265 TS": ("libx265", "hevc.ts"),
    "MPEG-2 VOB": ("mpeg2video", "mpeg2.vob"),
    "MPEG-2 TS": ("mpeg2video", "mpeg2.ts"),
    "MPEG-2 tagged from frame 14": ("mpeg2video", "tags.mkv"),
}


def run(*arguments: str, cwd: Path | None = None, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments], input=data, capture_output=True, cwd=cwd
    )
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def make(directory: Path, name: str) -> Path:
    """The file `name` of KINDS, made in directory."""
    path = directory / name
    stem, suffix = name.split(".")
    if name == "tags.mkv":
        # Two encodes joined by ffmpeg's concat demuxer: 15 frames untagged, then 105 tagged
        # BT.601, its sequence headers carrying them, as S6's 15th frame brings BT.601.
        tags = ("-color_primaries", "smpte170m", "-color_trc", "smpte170m")
        tags += ("-colorspace", "smpte170m", "-color_range", "tv")
        later = ("-vf", "trim=start_frame=15,setpts=N/TB/25")
        run(*SOURCE, "-frames:v", "15", *ENCODES["mpeg2"], "-f", "mpegts", str(directory / "a.ts"))
        run(*SOURCE, *later, "-frames:v", "105", *ENCODES["mpeg2"], *tags, str(directory / "b.ts"))
        (directory / "joined.txt").write_text("file 'a.ts'\nfile 'b.ts'\n")
        run("-f", "concat", "-i", "joined.txt", "-c", "copy", name, cwd=directory)
    elif suffix in ("mkv", "vob"):
        muxer = ("-f", "vob") if suffix == "vob" else ()
        run(*SOURCE, "-frames:v", "300", *ENCODES[stem], *muxer, str(path))
    else:
        # The file it is copied from, made once: made again, it would replace the one a source's
        # index was made from, and an encode isn't always the one before: of 80 encodes of this
        # clip by libx264, 16 at a time, 2 decoded to other frames than the 78 others
        # (2026-10-10), the reads through the first file's index then retrying on the second.
        base = directory / f"{stem}.{'vob' if stem == 'mpeg2' else 'mkv'}"
        if not base.exists():
            make(directory, base.name)
        run("-i", str(base), "-map", "0", "-c", "copy", str(path))
    return path


@needs_ffmpeg
def test_a_copied_kinds_file_is_made_once(tmp_path: Path) -> None:
    # The MPEG-TS kind is a stream copy of the VOB's file, there already: left as it is, never
    # encoded again under the source whose index was made from it (make).
    vob = make(tmp_path, "mpeg2.vob")
    made = (vob.stat().st_mtime_ns, vob.stat().st_ino)
    assert make(tmp_path, "mpeg2.ts").is_file()
    assert (vob.stat().st_mtime_ns, vob.stat().st_ino) == made


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Source]:
    """kind -> the source of that kind, made and examined once for the module."""
    directory = tmp_path_factory.mktemp("synth")
    made: dict[str, Source] = {}

    def get(kind: str) -> Source:
        encoder, name = KINDS[kind]
        if not has_encoder(encoder):
            pytest.skip(f"needs ffmpeg with {encoder}")
        if kind not in made:
            path = directory / name
            made[kind] = examine(path if path.exists() else make(directory, name))
        return made[kind]

    return get


@dataclass(frozen=True)
class Reference:
    """A full decode's framemd5, seek_test.py's ref: the stream's time base, and each frame's pts
    and MD5 in display order."""

    time_base: Fraction
    pts: list[int]
    md5: list[str]


def reference(path: Path) -> Reference:
    output = run(
        *("-copyts", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"),
        *("-enc_time_base:v", "demux", "-f", "framemd5", "-"),
    ).decode()
    time_base, pts, md5 = Fraction(0), list[int](), list[str]()
    for line in output.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            fields = [field.strip() for field in line.split(",")]
            pts.append(int(fields[2]))
            md5.append(fields[5])
    return Reference(time_base, pts, md5)


def packet_list(path: Path) -> list[dict[str, str]]:
    """Every packet of the first video stream, in decode order: pts and flags, ffprobe's JSON."""
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pts,flags", "-of", "json", str(path)),
        ],
        capture_output=True,
        check=True,
    ).stdout
    return json.loads(probed)["packets"]


def leading(path: Path, pts: list[int]) -> set[int]:
    """The leading pictures: decoded after a keyframe, before the next, displayed before it, the
    open-GOP signature (seek_test.py's analyse); by their place in display order."""
    packets = packet_list(path)
    place = {value: frame for frame, value in enumerate(pts)}
    keys = [i for i, packet in enumerate(packets) if "K" in packet["flags"]]
    found: set[int] = set()
    for number, key in enumerate(keys):
        if "pts" not in packets[key]:
            continue
        own = int(packets[key]["pts"])
        end = keys[number + 1] if number + 1 < len(keys) else len(packets)
        for packet in packets[key + 1 : end]:
            if "pts" in packet and int(packet["pts"]) < own and int(packet["pts"]) in place:
                found.add(place[int(packet["pts"])])
    return found


def targets(source: Source, lead: set[int], seed: int = 1) -> dict[int, str]:
    """seek_test.py's kinds of targets: the first and last 3 frames, every keyframe and the frames
    either side, every leading picture, 5 at random."""
    assert source.index is not None
    frames = source.frames
    pts = source.index.pts.tolist()
    found = {n: "first" for n in range(3)} | {n: "last" for n in range(frames - 3, frames)}
    place = {value: frame for frame, value in enumerate(pts)}
    for key in source.index.keyframes.tolist():
        if key in place:
            for n, kind in ((place[key] - 1, "before-key"), (place[key], "key")):
                found.setdefault(n, kind)
            found.setdefault(place[key] + 1, "after-key")
    for n in lead:
        found.setdefault(n, "leading")
    rng = random.Random(seed)
    for _ in range(5):
        found.setdefault(rng.randrange(frames), "random")
    return {n: kind for n, kind in sorted(found.items()) if 0 <= n < frames}


def read_at(source: Source, first: int, count: int = 3) -> Reader:
    """A read of `count` frames from `first`, fewer at the end, its MD5s kept; stopped after."""
    reader = source.reader(first, md5=True)
    try:
        reader.read(min(count, source.frames - first))
    finally:
        reader.stop()
    return reader


def native_crc32(path: Path, pix_fmt: str, frame_bytes: int) -> list[int]:
    """zlib's CRC-32 of each frame of path, decoded as it is stored (pix_fmt)."""
    data = run("-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-")
    return [zlib.crc32(data[k : k + frame_bytes]) for k in range(0, len(data), frame_bytes)]


@needs_ffmpeg
@pytest.mark.parametrize("kind", KINDS)
def test_index_is_the_full_decodes(synth: Callable[[str], Source], kind: str) -> None:
    # Every frame once, in order, its pts the source's own in the stream's time base, as ffmpeg's
    # framemd5 decodes them; the CRC-32s zlib's of the frames as decoded; the keyframes every
    # packet flagged K that has a pts; the start time ffprobe's.
    source = synth(kind)
    found = source.index
    assert found is not None
    full = reference(source.path)
    assert found.frames == source.frames == len(full.pts)
    assert found.time_base == full.time_base
    assert found.pts.tolist() == full.pts
    assert not found.error.any()
    width, height = 160, 96
    size = width * height * 3 // 2  # yuv420p
    assert found.crc32.tolist() == native_crc32(source.path, "yuv420p", size)
    packets = packet_list(source.path)
    keys = sorted(int(p["pts"]) for p in packets if "K" in p["flags"] and "pts" in p)
    assert found.keyframes.tolist() == keys and len(keys) > 2
    path = source.path
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    start = Fraction(json.loads(probed)["format"]["start_time"])
    assert found.start_time == start * 1_000_000
    expected = {"ts": Fraction(144, 100), "vob": Fraction(54, 100)}.get(source.path.suffix[1:], 0)
    if source.path.name.startswith("h") and source.path.suffix == ".ts":
        expected = Fraction(148, 100)
    assert start == expected


@needs_ffmpeg
@pytest.mark.parametrize("kind", KINDS)
def test_reads_exact_at_every_kind_of_target(synth: Callable[[str], Source], kind: str) -> None:
    # Every target read exactly, at the first try (c_gop was exact on every synthetic target,
    # seeking.md), the read seeking but in the first GOPs.
    source = synth(kind)
    assert source.index is not None
    full = reference(source.path)
    lead = leading(source.path, full.pts)
    assert lead, "open GOPs: leading pictures"
    chosen = targets(source, lead)
    wrong, retried, seeks = [], [], 0
    for n in chosen:
        reader = read_at(source, n)
        if reader.md5s != full.md5[n : n + 3]:
            wrong.append(n)
        if len(reader.attempts) != 1:
            retried.append(n)
        seeks += reader.attempts[0].keyframe is not None
    assert (wrong, retried) == ([], [])
    assert seeks > len(chosen) / 2


@needs_ffmpeg
@pytest.mark.parametrize("kind", [k for k in KINDS if "MKV" not in k and "tagged" not in k])
def test_the_exact_pts_seek_misses_there(synth: Callable[[str], Source], kind: str) -> None:
    # What the reads are held against: seeking to the target's own pts, selected by pts
    # (seeking.md's b_pts, with -copyts), comes out late on leading pictures in MP4 (mechanism 3:
    # mov.c seeks on decode timestamps), MPEG-TS and MPEG-PS (4: no index, a bisection).
    source = synth(kind)
    assert source.index is not None
    full = reference(source.path)
    start = source.index.start_time or 0
    missed = 0
    for n in sorted(leading(source.path, full.pts)):
        seconds = Fraction(full.pts[n]) * full.time_base - Fraction(start, 1_000_000)
        output = run(
            *("-copyts", "-ss", f"{float(seconds):.6f}", "-i", str(source.path)),
            *("-map", "0:v:0", "-vf", f"select=gte(pts\\,{full.pts[n]})"),
            *("-fps_mode", "passthrough", "-frames:v", "3", "-f", "framemd5", "-"),
        ).decode()
        got = [line.split(",")[5].strip() for line in output.splitlines() if line[:1] == "0"]
        missed += got != full.md5[n : n + 3]
    assert missed > 0


@needs_ffmpeg
def test_a_rebuilt_filter_graph_counts_every_frame(synth: Callable[[str], Source]) -> None:
    # The colour description appears on frame 14, which makes ffmpeg rebuild its filter graph:
    # its own frame count restarts there, select's n giving frame 34 for 20 (seeking.md,
    # mechanism 8). The first pass counts the frames as they come: every one, once, in order
    # (test_index_is_the_full_decodes), and the reads around it are exact.
    source = synth("MPEG-2 tagged from frame 14")
    full = reference(source.path)
    frames = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "frame=color_space", "-of", "json", str(source.path)),
        ],
        capture_output=True,
        check=True,
    ).stdout
    tagged = [frame.get("color_space", "unknown") for frame in json.loads(frames)["frames"]]
    assert tagged.index("smpte170m") == 14 and len(tagged) == source.frames == 120
    output = run(
        *("-i", str(source.path), "-map", "0:v:0", "-vf", "select=gte(n\\,20)"),
        *("-fps_mode", "passthrough", "-frames:v", "1", "-f", "framemd5", "-"),
    ).decode()
    [line] = [line for line in output.splitlines() if line[:1] == "0"]
    assert line.split(",")[5].strip() == full.md5[34]
    for n in (13, 14, 15, 20, 34):
        assert read_at(source, n).md5s == full.md5[n : n + 3]


def part(offset: int, frames: int, chroma: int) -> bytes:
    """x264 in Annex B, no B-frames, an IDR every 12 frames, its chroma QP offset in its PPS; its
    SPS and PPS before its first picture only (x264 repeats them at every IDR without a global
    header: libavcodec/libx264.c)."""
    data = run(
        *SOURCE,
        *("-vf", f"trim=start_frame={offset},setpts=N/TB/25", "-frames:v", str(frames)),
        *("-c:v", "libx264", "-preset", "veryfast", "-x264-params"),
        f"keyint=12:min-keyint=12:scenecut=0:bframes=0:chroma-qp-offset={chroma}:log-level=error",
        *("-f", "h264", "-"),
    )
    starts = [m.start() for m in re.finditer(b"\x00\x00\x01", data)]
    kept: list[bytes] = []
    pictures = False
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(data)
        unit = data[start:end].rstrip(b"\x00") if number + 1 < len(starts) else data[start:end]
        kind = unit[3] & 0x1F
        if kind in (7, 8) and pictures:
            continue
        pictures |= kind in (1, 5)
        kept.append(b"\x00" + unit)
    return b"".join(kept)


@pytest.fixture(scope="module")
def parameter_sets(tmp_path_factory: pytest.TempPathFactory) -> Source:
    """An H.264 stream of 300 frames, an IDR every 12, whose PPS changes at frame 96: carried by
    that IDR alone, the extradata (Matroska's) keeping the first. A decoder starting at a later
    IDR uses the extradata's: wrong pictures with the right pts (seeking.md, mechanism 6)."""
    if not has_encoder("libx264"):
        pytest.skip("needs ffmpeg with libx264")
    directory = tmp_path_factory.mktemp("pps")
    (directory / "pps.h264").write_bytes(part(0, 96, 0) + part(96, 204, 10))
    run("-framerate", "25", "-i", "pps.h264", "-c", "copy", "pps.mkv", cwd=directory)
    return examine(directory / "pps.mkv")


@needs_ffmpeg
def test_keyframes_that_are_no_entry_points(parameter_sets: Source) -> None:
    # Read from frame 108 or 109, one keyframe before their own is frame 96's, which carries the
    # PPS: right at the first try. Later frames' reads start at keyframes using the old PPS, and
    # go back, doubling, until one that comes right: frame 299 from keyframes 23, 22, 20, 16 and
    # 8, frame 96's. Every frame read is right.
    source = parameter_sets
    full = reference(source.path)
    assert source.index is not None
    place = {value: frame for frame, value in enumerate(full.pts)}
    assert [place[key] for key in source.index.keyframes.tolist()] == list(range(0, 300, 12))
    for n in (108, 109):
        reader = read_at(source, n)
        assert reader.md5s == full.md5[n : n + 3]
        assert [(a.keyframe, a.back) for a in reader.attempts] == [(8, 1)]
    reader = read_at(source, 299)
    assert reader.md5s == full.md5[299:]
    attempts = [(a.keyframe, a.back) for a in reader.attempts]
    assert attempts == [(23, 1), (22, 2), (20, 4), (16, 8), (8, 16)]
    said = r"frame 299: CRC-32 [0-9a-f]{8}, where the index has [0-9a-f]{8}"
    assert all(re.fullmatch(said, attempt.why) for attempt in reader.attempts[1:])
    for n in list(targets(source, set(), seed=2))[::3]:  # every third: each read takes several
        assert read_at(source, n).md5s == full.md5[n : n + 3], n


@needs_ffmpeg
def test_a_picture_otherwise_doubles_back_then_is_taken_from_the_start(
    parameter_sets: Source, caplog: pytest.LogCaptureFixture
) -> None:
    # An index whose CRC-32 for frame 80 is another, as a decode concealing it otherwise would
    # have it: every seek's frame 80 is a mismatch, the read going back, doubling, then decoding
    # from the start, where the frame, at its place, its pts the index's, is taken as decoded,
    # said once, and the frames after it read on (media/reader.py, _take).
    source = parameter_sets
    assert source.index is not None
    full = reference(source.path)
    altered = source.index.crc32.copy()
    altered[80] ^= 1
    otherwise = replace(source, index=replace(source.index, crc32=altered))
    reader = read_at(otherwise, 80)
    assert [(a.keyframe, a.back) for a in reader.attempts] == [(5, 1), (4, 2), (2, 4), (None, 0)]
    said = r"frame 80: CRC-32 [0-9a-f]{8}, where the index has [0-9a-f]{8}"
    assert all(re.fullmatch(said, attempt.why) for attempt in reader.attempts[1:])
    assert reader.md5s == full.md5[80:83] and reader.otherwise == [80]
    assert re.search(f"{said}, decoded from the start, at its place, .* taken as", caplog.text)
    # A read from the first frame decodes from the start: frame 80 taken there, every frame given.
    caplog.clear()
    with otherwise.decoder() as whole:
        assert len(whole.read(source.frames)) == source.frames
    assert whole.otherwise == [80] and caplog.text.count("taken as decoded") == 1


@needs_ffmpeg
def test_a_timestamp_otherwise_fails_naming_the_frame(parameter_sets: Source) -> None:
    # An index whose pts for frame 80 is another: from the start, a frame at another pts than the
    # index's is no decoder's doing, and fails, naming the frame (media/reader.py, _fall_back).
    source = parameter_sets
    assert source.index is not None
    moved = source.index.pts.copy()
    moved[80] += 1
    wrong = replace(source, index=replace(source.index, pts=moved))
    said = r"frame 80: pts 3200, where the index has 3201, decoded from the start"
    reader = wrong.decoder()
    with pytest.raises(MediaError, match=said):
        reader.read(source.frames)
    assert reader.position == 80 and [a.keyframe for a in reader.attempts] == [None]
    # A read from frame 78 checks 78 and 79, then goes back doubling, its selects from frame 80
    # on taking the index's pts, which passes frame 81 first, and fails from the start, naming
    # frame 80, nothing given past frame 79.
    reader = wrong.reader(78)
    with pytest.raises(MediaError, match=r"frame 80 missing: the next frame is 81, decoded from"):
        reader.read(3)
    assert [(a.keyframe, a.back) for a in reader.attempts] == [(5, 1), (4, 2), (2, 4), (None, 0)]
    assert reader.position == 80


def damaged(path: Path, packets: list[int]) -> tuple[Path, list[int]]:
    """A copy of path, 40 bytes overwritten a third into each packet numbered in `packets`
    (decode order); and those packets' pts."""
    listed = packet_list(path)
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pts,pos,size", "-of", "json", str(path)),
        ],
        capture_output=True,
        check=True,
    ).stdout
    data = bytearray(path.read_bytes())
    for number in packets:
        packet = json.loads(probed)["packets"][number]
        start = int(packet["pos"]) + int(packet["size"]) // 3
        data[start : start + 40] = bytes(b ^ 0x5A for b in data[start : start + 40])
    copy = path.with_name(f"damaged-{path.name}")
    copy.write_bytes(data)
    return copy, [int(listed[number]["pts"]) for number in packets]


def damaged_gop(found: FrameIndex, full: Reference, frames: list[int]) -> range:
    """The GOP of damaged frames, by the keyframes' places in the full decode: from the first
    frame of the first one's own keyframe to the next keyframe's after the last one, leading
    pictures included, which may reference them; the frames a decode may conceal otherwise."""
    keys = [full.pts.index(value) for value in found.keyframes.tolist()]
    first = max(key for key in keys if key <= frames[0])
    after = min((key for key in keys if key > frames[-1]), default=len(full.pts))
    return range(first, after)


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "packets"), [("h264", [100, 101]), ("mpeg2", [100])])
def test_damaged_frames_flagged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, codec: str, packets: list[int]
) -> None:
    # Bytes overwritten in packets 100 and 101 of an x264 file (a P-frame and the B-frame before
    # it), in packet 100 of an MPEG-2 one: the frames ffmpeg reports corrupt are flagged, each tied
    # to its own by the decoder's line after the report (media/scan.py), so none but those
    # packets' frames. x264's frame threads report the P-frame always, the B-frame not always,
    # and conceal both, and the frames after, otherwise from one decode to the next
    # (media/reader.py, _take): every frame outside the damaged GOP is read exactly, every one
    # inside at its place, never failing, from a seek or from the start; MPEG-2's are exact.
    # 320x240: at 160x96, x264's slices took the bytes overwritten without a report.
    encoder = {"h264": "libx264", "mpeg2": "mpeg2video"}[codec]
    if not has_encoder(encoder):
        pytest.skip(f"needs ffmpeg with {encoder}")
    clean = tmp_path / ("clean.vob" if codec == "mpeg2" else "clean.mkv")
    muxer = ("-f", "vob") if codec == "mpeg2" else ()
    size = ("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25", "-frames:v", "200")
    run(*size, *ENCODES[codec], *muxer, str(clean))
    path, hit = damaged(clean, packets)
    source = examine(path)
    found = source.index
    assert found is not None
    full = reference(path)
    flagged = np.flatnonzero(found.error).tolist()
    hit_frames = sorted(full.pts.index(value) for value in hit)
    assert flagged and set(flagged) <= set(hit_frames)
    assert hit_frames[-1] in flagged  # the P-frame, MPEG-2's or x264's
    # One frame is said as one: MPEG-2's always, x264's when its B-frame went unreported.
    several = f"{len(flagged)} frames decoded with an error, from frame {flagged[0]}: a read"
    one = f"frame {flagged[0]} decoded with an error: a read seeking through it decodes"
    assert (f"{several} seeking through them decodes" if len(flagged) > 1 else one) in caplog.text
    assert codec != "mpeg2" or len(flagged) == 1
    zone = damaged_gop(found, full, flagged)
    first, after = zone.start, zone.stop
    clear = [n for n in targets(source, set()) if all(k not in zone for k in range(n, n + 3))]
    for n in clear:
        assert read_at(source, n).md5s == full.md5[n : n + 3], n
    for n in [first, *flagged, after - 1]:
        reader = read_at(source, n)
        assert len(reader.md5s) == min(3, source.frames - n)
        assert set(reader.otherwise) <= set(zone)
        if codec == "mpeg2":
            assert reader.md5s == full.md5[n : n + 3] and not reader.otherwise
    whole = source.reader(0, md5=True)
    whole.read(source.frames)
    whole.finish()
    outside = [n for n in range(source.frames) if n not in zone]
    assert [whole.md5s[n] for n in outside] == [full.md5[n] for n in outside]
    assert set(whole.otherwise) <= set(zone)
    # A frame of that GOP the index has otherwise: the read goes to the start at once, never
    # doubling, and takes it there as decoded, saying so.
    altered = found.crc32.copy()
    altered[flagged[0]] ^= 1
    caplog.clear()
    reader = replace(source, index=replace(found, crc32=altered)).reader(flagged[0])
    assert len(reader.read(1)) == 1
    reader.stop()
    assert [a.keyframe is None for a in reader.attempts] == [False, True]
    assert reader.otherwise == [flagged[0]]
    assert f"frame {flagged[0]} decoded with an error in the first pass" in caplog.text
    assert re.search(f"frame {flagged[0]}: CRC-32 .* taken as decoded", caplog.text)


@needs_ffmpeg
def test_unreported_damage_taken_from_the_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Bytes overwritten in packet 100 of an x265 file: its decoder reports nothing, so the first
    # pass flags no frame, and a frame (98 with n9.0.2) comes out otherwise than the clean file's.
    # ffmpeg's frame threads gave a frame of such a file 2 ways in 6 decodes (media/reader.py,
    # _take): an index holding another CRC-32 for that frame stands for a first pass that
    # concealed it otherwise. A read from the first frame gives every frame, that one taken as
    # decoded, said; a read from it seeks, goes back doubling, then to the start, where it is
    # taken; never a MediaError. Frames are held to the reference outside the damaged GOP only,
    # which a decode may conceal otherwise.
    if not has_encoder("libx265"):
        pytest.skip("needs ffmpeg with libx265")
    clean = tmp_path / "clean.mkv"
    size = ("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25", "-frames:v", "200")
    run(*size, *ENCODES["hevc"], str(clean))
    path, _ = damaged(clean, [100])
    source = examine(path)
    found = source.index
    assert found is not None and not found.error.any()
    full = reference(path)
    pairs = zip(full.md5, reference(clean).md5, strict=True)
    hit = [n for n, (md5, unharmed) in enumerate(pairs) if md5 != unharmed]
    assert hit, "the damage changes a picture"
    zone = damaged_gop(found, full, hit)
    outside = [n for n in range(source.frames) if n not in zone]
    altered = found.crc32.copy()
    altered[hit[0]] ^= 1
    otherwise = replace(source, index=replace(found, crc32=altered))
    with otherwise.reader(0, md5=True) as whole:
        assert len(whole.read(source.frames)) == source.frames
    assert [whole.md5s[n] for n in outside] == [full.md5[n] for n in outside]
    assert hit[0] in whole.otherwise and set(whole.otherwise) <= set(zone)
    assert re.search(f"frame {hit[0]}: CRC-32 .* taken as decoded", caplog.text)
    reader = read_at(otherwise, hit[0])
    attempts = [a.keyframe is None for a in reader.attempts]
    assert attempts[0] is False and attempts[-1] is True and attempts.count(True) == 1
    assert len(reader.md5s) == 3 and reader.otherwise[0] == hit[0]
    assert set(reader.otherwise) <= set(zone)


@needs_ffmpeg
def test_a_timestamp_select_cant_single_out(
    synth: Callable[[str], Source], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two frames on one pts, or one without: select can't tell them apart, so the read decodes
    # from the start and counts, its select keeping every frame from the lowest pts on.
    times = np.array([0, 40, 40, 120, 160], dtype=np.int64)
    found = FrameIndex(
        Fraction(1, 1000), 0, times, np.zeros(5, np.uint32), np.zeros(5, bool), times[[0, 3]]
    )
    # From frame 1 on, its pts selects frames 1 and 2 and the rest: right. From frame 2, frame 1
    # too.
    assert [found.selects(n) for n in range(5)] == [True, True, False, True, True]
    assert [found.lowest_from(n) for n in range(5)] == [0, 40, 40, 120, 160]
    # A select drops a frame without a pts: none before frame 3 selects its frames.
    missing = replace(found, pts=np.array([0, 40, NOPTS, 120, 160], dtype=np.int64))
    assert [missing.selects(n) for n in range(5)] == [False, False, False, True, True]
    assert [missing.lowest_from(n) for n in range(5)] == [None, None, None, 120, 160]
    assert missing.own_keyframe(2) == -1 and missing.own_keyframe(4) == 1
    source = synth("x264 MKV")
    full = reference(source.path)
    monkeypatch.setattr(FrameIndex, "selects", lambda self, frame: False)
    reader = read_at(source, 200)
    assert reader.md5s == full.md5[200:203]
    assert [(a.keyframe, a.back) for a in reader.attempts] == [(None, 0)]


def ffv1(path: Path, pix_fmt: str, frames: int = 3) -> Path:
    """An FFV1 file of random frames in pix_fmt, 256x144."""
    planes = np.random.default_rng(len(pix_fmt)).integers(0, 65536, (frames, 3, 144, 256))
    run(
        *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", "256x144", "-framerate", "25"),
        *("-i", "-", "-vf", f"format={pix_fmt}", "-c:v", "ffv1", str(path)),
        data=planes.astype("<u2").tobytes(),
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize(
    "pix_fmt", ["yuv420p", "yuv420p10le", "yuv422p10le", "gbrp", "gbrp12le", "bgr0"]
)
def test_reads_convert_as_the_decoder(tmp_path: Path, pix_fmt: str) -> None:
    # The conversion's chain, now after a select in an output beside the hashes', gives the frames
    # the decoder's own (decode.decode_command) gives, bit for bit, 10-bit 4:2:0 included, which
    # reads otherwise on more than one zscale slice (DESIGN.md, Input), and milestone 1's input
    # format, FFV1 bgr0 (test_regression.py). These files have one keyframe: the reads decode
    # from the start (seeks: test_seeks_hash_the_frames_as_decoded).
    source = examine(ffv1(tmp_path / f"{pix_fmt}.mkv", pix_fmt))
    conversion = source.conversion
    with Decoder(input_args(source.path), conversion, 256, 144, 3) as decoder:
        expected = decoder.read(3)
    with source.decoder() as reader:
        assert np.array_equal(reader.read(3), expected)
    with source.reader(1) as reader:
        assert np.array_equal(reader.read(2), expected[1:])


# The packed and semi-planar formats a file holds as rawvideo (n9.0.2): Matroska takes 4:2:0
# semi-planar and 4:2:2 packed YUV, NUT packed RGB. nv16, nv24, nv42 and the P0xx, P2xx and P4xx
# formats come back from Matroska, NUT, AVI and MOV as other formats: no source file holds them.
RAW_FILES = {
    **dict.fromkeys(("nv12", "nv21", "uyvy422", "yuyv422", "yvyu422"), "mkv"),
    **dict.fromkeys(("0bgr", "0rgb", "bgr0", "bgr24", "rgb0", "rgb24"), "nut"),
    **dict.fromkeys(("bgr48be", "bgr48le", "rgb48be", "rgb48le"), "nut"),
}
# Each file seeked: its encode's options and suffix. Rawvideo and PNG make every frame a keyframe,
# FFV1 one in 10; FFV1 bgr0 is milestone 1's input format (test_regression.py).
SEEKABLE = {
    **{
        f"{pix_fmt} rawvideo": (("-pix_fmt", pix_fmt, "-c:v", "rawvideo"), suffix)
        for pix_fmt, suffix in RAW_FILES.items()
    },
    "bgr0 FFV1": (("-pix_fmt", "bgr0", "-c:v", "ffv1", "-g", "10"), "mkv"),
    "rgb24 PNG": (("-pix_fmt", "rgb24", "-c:v", "png"), "mov"),
}


@needs_ffmpeg
@pytest.mark.parametrize("kind", SEEKABLE)
def test_seeks_hash_the_frames_as_decoded(tmp_path: Path, kind: str) -> None:
    # In one filter graph with the conversion, behind a split whose links share one list of
    # formats, the conversion's planar format reached back before the split once -ss's trim stood
    # in front of the graph: every seek of a packed or semi-planar source hashed the frame
    # converted, never the index's, and went back to the start, 6 to 7 attempts (2026-10-09). The
    # hashes in outputs of their own: one attempt, a seek; the index holding the frames as
    # decoded, the read the decoder's frames bit for bit, and its MD5s the frames as decoded.
    options, suffix = SEEKABLE[kind]
    path = tmp_path / f"seek.{suffix}"
    run("-f", "lavfi", "-i", "testsrc2=s=256x144:r=25", "-frames:v", "40", *options, str(path))
    source = examine(path)
    found = source.index
    assert found is not None and len(found.keyframes) >= 4 and source.frames == 40
    stored = run(
        "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", source.stream.pix_fmt, "-"
    )
    size = len(stored) // 40
    native = [zlib.crc32(stored[k : k + size]) for k in range(0, len(stored), size)]
    assert found.crc32.tolist() == native
    full = reference(path)
    with Decoder(input_args(path), source.conversion, 256, 144, 40) as decoder:
        expected = decoder.read(40)
    reader = source.reader(25, md5=True)
    try:
        frames = bounded(reader, lambda read: read.read(3))
    finally:
        reader.stop()
    assert [attempt.keyframe is not None for attempt in reader.attempts] == [True]
    assert reader.md5s == full.md5[25:28] and np.array_equal(frames, expected[25:28])


def test_packet_scan() -> None:
    # Packets without a pts are no seek points (MPEG-PS: 189 of 2,878, seeking.md); the start
    # time is ffprobe's, its microseconds exact.
    lines = [
        "packet|pts=48600|flags=K__",
        "packet|pts=N/A|flags=K__",
        "packet|pts=52200|flags=___|",
        "packet|pts=102600|flags=K_C",
        "format|format_name=mpeg|start_time=0.540000",
    ]
    assert read_packets(lines) == ([48600, 102600], 540_000, ["mpeg"])
    assert read_packets(["format|start_time=N/A"]) == ([], None, [""])
    named = "format|format_name=matroska,webm|start_time=N/A"
    assert read_packets([named]) == ([], None, ["matroska", "webm"])


def sample() -> FrameIndex:
    pts = np.arange(0, 400, 40, dtype=np.int64) + 133_470
    crc = np.arange(10, dtype=np.uint32) * 0x01020304
    error = np.zeros(10, dtype=np.bool_)
    error[7] = True
    return FrameIndex(Fraction(1, 90_000), 1_483_000, pts, crc, error, pts[[0, 4, 8]])


def test_index_file(tmp_path: Path) -> None:
    # Exact and versioned: the same index, the same bytes; read back whole; anything else refused,
    # saying why; a column a later version adds skipped by its size.
    written = sample()
    data = written.to_bytes()
    assert data == sample().to_bytes()
    assert data.startswith(index.MAGIC)
    back = FrameIndex.from_bytes(data)
    assert back.to_bytes() == data
    assert (back.time_base, back.start_time, back.frames) == (Fraction(1, 90_000), 1_483_000, 10)
    for name in ("pts", "crc32", "error", "keyframes"):
        assert np.array_equal(getattr(back, name), getattr(written, name)), name
    header, values = data[len(index.MAGIC) :].split(b"\n", 1)
    fields = json.loads(header)
    fields["frames"]["columns"].append(["transnetv2", "<f4"])
    more = np.full(10, 0.5, dtype="<f4").tobytes()
    frame_columns = 10 * (8 + 4 + 1)
    extended = (
        index.MAGIC
        + json.dumps(fields).encode()
        + b"\n"
        + values[:frame_columns]
        + more
        + values[frame_columns:]
    )
    assert np.array_equal(FrameIndex.from_bytes(extended).keyframes, written.keyframes)
    refused = {
        b"seedvr2x frame list\n" + data[len(index.MAGIC) :]: "not a frame index",
        data[:-1]: "cut short",
        data + b"\0": "1 bytes after its last column",
        data.replace(b'"version":1', b'"version":2'): "version 2, where 1 is read",
        data.replace(b'"<u4"', b'"<i4"'): "crc32 of type <i4, not <u4",
        data.replace(b'["error","|u1"]', b'["flag","|u1"]'): "no frames column error",
    }
    for damaged_data, said in refused.items():
        with pytest.raises(ValueError, match=re.escape(said)):
            FrameIndex.from_bytes(damaged_data)
    path = tmp_path / "frame_index.bin"
    written.write(path)
    assert path.read_bytes() == data and not (tmp_path / "frame_index.bin.partial").exists()
    assert FrameIndex.read(path).frames == 10
    with pytest.raises(MediaError, match=r"frame_index\.bin\.gone: missing"):
        FrameIndex.read(tmp_path / "frame_index.bin.gone")


def test_keyframe_arithmetic() -> None:
    found = sample()
    # Frames at pts 133470 + 40k; keyframes at frames 0, 4 and 8.
    assert [found.own_keyframe(n) for n in range(10)] == [0] * 4 + [1] * 4 + [2] * 2
    assert [found.first_frame_from(k) for k in range(3)] == [0, 4, 8]
    assert found.first_error(0, 6) is None and found.first_error(5, 9) == 7
    assert all(found.selects(n) for n in range(10))


@needs_ffmpeg
def test_first_pass_logs_nothing_on_clean_files(
    synth: Callable[[str], Source], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    scanned = scan(synth("x264 TS").path)
    assert scanned.frames == 300 and scanned.index is not None
    assert "reported errors" not in caplog.text and "decoded with an error" not in caplog.text


# What the first pass's ffmpeg writes (media/scan.py), as n9.0.2 writes it: the hashes' header and
# a frame's line on stdout; on stderr, the line it prints before it decodes, the decoder's line
# and its report of a corrupt frame.
HEADER = [
    *("#format: frame checksums", "#version: 2", "#hash: CRC32", "#software: Lavf62.3.100"),
    *("#tb 0: 1/25", "#media_type 0: video", "#codec_id 0: rawvideo", "#dimensions 0: 160x96"),
    *("#sar 0: 1/1", "#stream#, dts,        pts, duration,     size, hash"),
]
REPORT = "[vist#0:0/h264 @ 0x5d1c] [dec:h264 @ 0x5d2f] [warning] corrupt decoded frame"
MAPPED = "[info] Stream mapping:"


def hash_line(pts: int, crc: int) -> str:
    return f"0, {pts:10d}, {pts:10d}, {1:8d}, {23040:8d}, {crc:08x}"


def decoder_line(pts: int, time_base: str = "1/25") -> str:
    said = f"pts:{pts} pts_time:{pts / 25:g} pkt_dts:{pts} pkt_dts_time:{pts / 25:g}"
    frame = f"duration:1 duration_time:0.04 keyframe:1 frame_type:1 time_base:{time_base}"
    return f"[vist#0:0/h264 @ 0x5d1c] [dec:h264 @ 0x5d2f] [info] decoder -> {said} {frame}"


@pytest.fixture
def faked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[list[str], list[str]], Path]:
    """(hashes, logged) -> a file to scan, a real one for ffprobe's packet scan, ffmpeg from then
    on a fake one on PATH that writes the lines `hashes` on stdout and `logged` on stderr, and
    ends: what the first pass reads, chosen."""

    def fake(hashes: list[str], logged: list[str]) -> Path:
        path = tmp_path / "in.mkv"
        run(*SOURCE, "-frames:v", "3", "-c:v", "ffv1", str(path))
        directory = tmp_path / "fake"
        directory.mkdir()
        for name, lines in (("out", hashes), ("err", logged)):
            (directory / name).write_text("".join(f"{line}\n" for line in lines))
        script = directory / "ffmpeg"
        script.write_text(f"#!/bin/sh\ncat '{directory}/out'\ncat '{directory}/err' >&2\n")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
        return path

    return fake


@needs_ffmpeg
def test_pts_are_the_hashes(faked: Callable[[list[str], list[str]], Path]) -> None:
    # Each frame's pts and the time base are read from the hashes on stdout, where ffmpeg writes
    # nothing else: whatever stderr says of them, the source's metadata being printed there, gives
    # no frame another pts, and a time base of 1/0 on a decoder's line stops nothing (it raised
    # in the thread reading stderr, which then drained it no more). The decoder's lines are
    # counted, each tying the report before it to its frame, found by their messages alone: the
    # last frame's glued to the first piece of the muxer's report of a pts going back, its prefix
    # left out, as ffmpeg printed it in 2 first passes of 6 (media/scan.py, DECODED).
    hashes = [*HEADER, hash_line(3, 0xA), hash_line(4, 0xB), hash_line(5, 0xC)]
    glued = "[vost#0:0/rawvideo @ 0x5fe9] [warning] Non-monotonic DTS; previous: 7, current: 4; "
    glued += decoder_line(102, "1/0").split("] ")[-1]
    logged = [decoder_line(100, "1/0"), REPORT.split("] ")[-1], decoder_line(7, "1/0"), glued]
    # Nothing before the line ffmpeg prints once its inputs are dumped is the decoder's: the
    # source's own text is printed there (media/scan.py, MAPPING).
    logged = [decoder_line(55), REPORT, MAPPED, *logged]
    found = scan(faked(hashes, logged)).index
    assert found is not None
    assert found.pts.tolist() == [3, 4, 5] and found.time_base == Fraction(1, 25)
    assert found.crc32.tolist() == [0xA, 0xB, 0xC]
    assert found.error.tolist() == [False, True, False]


@needs_ffmpeg
@pytest.mark.parametrize("decoded", [2, 4])
def test_counts_must_agree(faked: Callable[[list[str], list[str]], Path], decoded: int) -> None:
    # A frame the decoder gave and no hash line, or a hash line and no frame of the decoder's:
    # refused, since the reports of stderr are tied to the frames of stdout by their order.
    hashes = [*HEADER, *(hash_line(pts, pts) for pts in range(3))]
    path = faked(hashes, [MAPPED, *(decoder_line(pts) for pts in range(decoded))])
    with pytest.raises(MediaError) as refused:
        scan(path)
    assert str(refused.value) == f"{path}: ffmpeg decoded {decoded} frames and hashed 3"


@needs_ffmpeg
@pytest.mark.parametrize(
    ("header", "said"),
    [
        ("#tb 0: 1/0", "ffmpeg gave its hashes the time base '#tb 0: 1/0'"),
        ("#tb 0: 0/25", "ffmpeg gave its hashes the time base '#tb 0: 0/25'"),
        ("#tb 0: 1/x", "ffmpeg gave its hashes the time base '#tb 0: 1/x'"),
        ("#no time base", "ffmpeg gave its hashes no time base"),
    ],
)
def test_time_base_refused(
    faked: Callable[[list[str], list[str]], Path], header: str, said: str
) -> None:
    # A time base no pts can be read in is refused, a zero raising nothing (ffmpeg gives none).
    hashes = [header if line.startswith("#tb") else line for line in HEADER]
    path = faked([*hashes, hash_line(0, 1)], [MAPPED, decoder_line(0)])
    with pytest.raises(MediaError) as refused:
        scan(path)
    assert str(refused.value) == f"{path}: {said}"


# The source's own text, as its titles carry it into the lines ffmpeg prints (media/scan.py,
# MAPPING): the decoder's line of a frame, its time base 1/0; a corrupt frame's report; an error;
# and the line that ends the inputs' dumps.
PAYLOAD = [
    "decoder -> pts:7 pts_time:0.28 pkt_dts:7 pkt_dts_time:0.28 duration:1 duration_time:0.04"
    " keyframe:1 frame_type:1 time_base:1/0",
    "corrupt decoded frame",
    "[error] made up",
    MAPPED,
]
# A language tag whose line break makes a line of the tag's own, the one that ends the inputs'
# dumps: ffmpeg prints a stream's language as it is (libavformat/dump.c:638 at n9.0.2).
LANGUAGE = f"x\n{MAPPED}\n"


def titled(path: Path, payload: Sequence[str] = (), language: str | None = None) -> Path:
    """12 frames; with `payload`, its lines as the title of the file, of its video stream and of
    a chapter; with `language`, its video stream's language tag."""
    chapters = path.with_suffix(".txt")
    chapters.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=200\ntitle=one\n")
    tags = [
        argument
        for where in ("", ":s:v:0", ":c:0")
        if payload
        for argument in (f"-metadata{where}", "title=" + "\n".join(payload))
    ]
    if language is not None:
        tags += ["-metadata:s:v:0", f"language={language}"]
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-f", "ffmetadata", "-i", str(chapters), "-map", "0:v", "-map_chapters", "1"),
            *("-frames:v", "12", "-c:v", "ffv1", *tags, str(path)),
        ],
        check=True,
    )
    return path


def tagged(path: Path) -> str:
    """ffprobe's JSON of the titles of path, its video stream's and its chapters', and of that
    stream's language."""
    entries = "format_tags=title:stream_tags=title,language:chapter_tags=title"
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def bounded_scan(path: Path, seconds: float = 60.0) -> Scan:
    """scan(path), failing when it hasn't ended within `seconds`."""
    found: list[Scan] = []
    worker = threading.Thread(target=lambda: found.append(scan(path)), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "the first pass hangs"
    [scanned] = found
    return scanned


@needs_ffmpeg
@pytest.mark.parametrize("language", [None, LANGUAGE], ids=["titles", "a language tag too"])
def test_the_sources_own_text_is_no_line_of_the_pass(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, language: str | None
) -> None:
    # Titles made of the first pass's own lines, in the file's metadata, its stream's and a
    # chapter's, which ffmpeg prints at info level with its input, and with each output it takes
    # them to: none gives the pass a frame, a corrupt frame or an error, and none stops it (a
    # time base of 1/0 raised in the thread reading stderr, ffmpeg then blocked on it for good:
    # bounded here). Nor behind a language tag that prints the line ending the inputs' dumps
    # inside its input's, before the stream's title: the pass then read that title as the
    # decoder's lines, "decoded 13 frames and hashed 12". The pass reads what it reads of the same
    # frames untitled.
    path = titled(tmp_path / "titled.mkv", PAYLOAD, language)
    probed = json.loads(tagged(path))
    assert tagged(path).count("decoder -> pts:7") == 3 and tagged(path).count(MAPPED) == 3 + (
        language is not None
    )
    assert probed["streams"][0]["tags"].get("language") == language
    caplog.set_level(logging.WARNING)
    scanned = bounded_scan(path)
    plain = scan(titled(tmp_path / "plain.mkv"))
    assert scanned.index is not None and plain.index is not None
    assert scanned.frames == 12 and scanned.index.pts.tolist() == plain.index.pts.tolist()
    assert not scanned.index.error.any()
    assert "reported errors" not in caplog.text and "decoded with an error" not in caplog.text


def test_nothing_read_before_the_stream_mapping() -> None:
    # Before the line that ends the inputs' dumps, a line is the source's own text or its like,
    # as n9.0.2 prints a title, a line break in it begun again behind an indent alone: no frame,
    # no report; an error only at its line's start, behind its contexts alone, a parent's too,
    # never behind "[info]" or an indent, where a title is printed. After it, every message
    # wherever it starts.
    frame = decoder_line(7).split("] ")[-1]
    title = "[info]     title           : "
    more = "                    : "
    before = [
        f"{title}{frame}",
        f"{more}{MAPPED}",  # not the line itself, which is whole
        f"{more}{frame}",
        f"{more}{REPORT.split('] ')[-1]}",
        f"{title}[error] made up",
        f"{more}[error] made up",
        "[in#0/matroska @ 0x5d1c] [error] Error opening input",
        "[mpeg2video @ 0x5d2f] [IMGUTILS @ 0x7ffd5e0] [error] Picture size 0x0 is invalid",
        "[fatal] Error opening input files",
    ]
    decoded = _Decoded()
    decoded.read(f"{line}\n" for line in before)
    assert (decoded.pts, decoded.corrupt) == ([], [])
    assert decoded.errors == before[-3:]
    after = [MAPPED, f"[info]   Stream #1{frame}", REPORT, decoder_line(8), f"{more}[error] late"]
    decoded = _Decoded()
    decoded.read(f"{line}\n" for line in [*before, *after])
    assert (decoded.pts, decoded.corrupt) == ([7, 8], [False, True])
    assert decoded.errors == [*before[-3:], f"{more}[error] late".strip()]


def test_each_stream_mapping_line_begins_the_reading_again() -> None:
    # ffmpeg prints the line once, after every input's dump: one before the last is the source's
    # own text, a stream's language printed as it is, and so is what follows it, a title read
    # until then as the decoder's lines (media/scan.py, MAPPING). At the next one, the frames,
    # their reports, a report waiting for its frame and the errors read since are dropped, but
    # the errors at their line's start, as before any such line.
    frame = decoder_line(7).split("] ")[-1]
    more = "                    : "
    dumped = [
        "[matroska,webm @ 0x5d1c] [error] an error of the demuxer's",
        "[info]   Stream #0:0(x",
        MAPPED,
        "): Video: ffv1, yuv420p, 64x48, 25 fps",
        f"[info]       title           : {frame}",
        f"{more}{REPORT.split('] ')[-1]}",
        f"{more}{frame}",
        f"{more}[error] made up",
        "[ffv1 @ 0x5d2f] [error] an error of the decoder's",
        f"{more}{REPORT.split('] ')[-1]}",
    ]
    decoded = _Decoded()
    decoded.read(f"{line}\n" for line in dumped)
    assert (decoded.pts, decoded.corrupt) == ([7, 7], [False, True])
    assert len(decoded.errors) == 3
    decoded = _Decoded()
    again = [MAPPED, decoder_line(0), "[error] real"]
    decoded.read(f"{line}\n" for line in [*dumped, *again])
    assert (decoded.pts, decoded.corrupt) == ([0], [False])
    assert decoded.errors == [dumped[0], dumped[-2], "[error] real"]
    # A third one: the errors dropped at the second are dropped once, their places forgotten,
    # those read since at their line's start kept.
    decoded = _Decoded()
    third = [f"{more}[error] made up", MAPPED, decoder_line(1)]
    decoded.read(f"{line}\n" for line in [*dumped, *again, *third])
    assert (decoded.pts, decoded.corrupt) == ([1], [False])
    assert decoded.errors == [dumped[0], dumped[-2], "[error] real"]


def joined_ts(directory: Path, offsets: tuple[int, int]) -> tuple[Path, list[Path]]:
    """Two MPEG-TS files of 50 frames at 25 fps, their timestamps `offsets` seconds late, and the
    two joined byte after byte, as recordings are: a jump at frame 50."""
    parts = [directory / "a.ts", directory / "b.ts"]
    for part, offset in zip(parts, offsets, strict=True):
        encode = ("-c:v", "mpeg2video", "-g", "10", "-output_ts_offset", str(offset))
        run(*SOURCE, "-frames:v", "50", *encode, "-f", "mpegts", str(part))
    joined = directory / "joined.ts"
    joined.write_bytes(parts[0].read_bytes() + parts[1].read_bytes())
    return joined, parts


def indexed(path: Path) -> FrameIndex:
    found = scan(path).index
    assert found is not None
    return found


@needs_ffmpeg
@pytest.mark.parametrize(
    ("offsets", "wrapped"),
    [((0, 1000), 0), ((1000, 0), 1), ((100, 60), 0)],
    ids=["forwards", "back through the wrap", "backwards"],
)
def test_a_timestamp_jump_refused_with_its_remux(
    tmp_path: Path, offsets: tuple[int, int], wrapped: int
) -> None:
    # MPEG-TS files joined, the second's timestamps 1,000 s after the first's, 1,000 s before
    # (which the demuxer reads as MPEG-TS's 33-bit timestamps wrapping: 2^33 ticks later), or 40 s
    # before (the pts going back, which ffmpeg's muxer holds in the hashes: the decoder's line
    # says where). -copyts keeps the jump, which the first pass before the frame index never saw
    # (media/scan.py, JUMP_FORWARD): refused as what it is, naming frame 50 and both times, with
    # the remux by which ffmpeg takes it out; that command run, the file is read, its 100 frames
    # those decoded from the joined one, 40 ms apart.
    path, parts = joined_ts(tmp_path, offsets)
    first, second = (indexed(part) for part in parts)
    before = float(first.pts[-1] * first.time_base)
    after = float((second.pts[0] + wrapped * 2**33) * second.time_base)
    assert abs(after - before) > 30
    with pytest.raises(MediaError) as refused:
        examine(path)
    remux = "ffmpeg -i SOURCE -map 0 -map -0:d -c copy fixed.mkv"
    assert str(refused.value) == (
        f"{path}: its timestamps jump at frame 50, from {before:.3f} to {after:.3f} s: a"
        " discontinuity, which ffmpeg takes out when it remuxes the file, every frame kept:"
        f" {remux}"
    )
    command = [str(path) if word == "SOURCE" else word for word in remux.split()]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    source = examine(tmp_path / "fixed.mkv")
    assert source.index is not None and source.frames == 100
    assert source.index.crc32.tolist() == indexed(path).crc32.tolist()
    assert set(np.diff(source.index.pts).tolist()) == {40}


@needs_ffmpeg
def test_a_timestamp_jump_no_remux_takes_out(tmp_path: Path) -> None:
    # The same jump in Matroska, whose timestamps ffmpeg takes as they are: refused, naming it,
    # without the remux, which would keep it (media/scan.py, DISCONTINUOUS).
    joined, _ = joined_ts(tmp_path, (0, 1000))
    path = tmp_path / "jump.mkv"
    run("-copyts", "-i", str(joined), "-map", "0", "-c", "copy", str(path))
    found = scan(path)
    assert found.discontinuity is not None and found.frames == 100
    jump = found.discontinuity
    assert (jump.frame, jump.remux) == (50, False) and jump.after - jump.before > 990
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == (
        f"{path}: its timestamps jump at frame 50, from {float(jump.before):.3f} to"
        f" {float(jump.after):.3f} s: a discontinuity, and only a constant frame rate is supported"
    )


@pytest.mark.parametrize(
    ("pts", "decoder", "jump"),
    [
        # ffmpeg's bounds (media/scan.py): more than 10 s on, 250 frames at 25 fps, or more than
        # 0.1 s back, a pts going back being held in the hashes at the frame before's.
        ([0, 1, 251], [0, 1, 251], None),
        ([0, 1, 252], [0, 1, 252], (2, 1, 252)),
        ([5, 6, 6], [5, 6, 4], None),
        ([5, 6, 6], [5, 6, 3], (2, 6, 3)),
        ([5, 6, 6], [5, 6, 6], None),  # two frames of one pts: no jump, a variable frame rate
        ([NOPTS, 300, 301], [NOPTS, 300, 301], None),
        # A step on past the bound is a jump only when it is longer than the frames' shortest
        # too: a frame every 15 s, 375 ticks, a constant rate, has none, whatever its length,
        # nor a single step. Nor has one step longer than the others, however long, the
        # shortest being itself past the bound: a variable frame rate, the timing rule's.
        ([0, 375, 750, 1125], [0, 375, 750, 1125], None),
        ([0, 375], [0, 375], None),
        ([0, 375, 750, 1126], [0, 375, 750, 1126], None),
        ([0, 375, 750, 100_000], [0, 375, 750, 100_000], None),
        # At the bound itself, a frame every 10 s, a longer step is one; and a step back at
        # any rate, the pts held for it in the hashes, or one said twice, being no step on.
        ([0, 250, 500, 751], [0, 250, 500, 751], (3, 500, 751)),
        ([0, 375, 750, 750], [0, 375, 750, 747], (3, 750, 747)),
        ([0, 375, 375, 750], [0, 375, 375, 750], None),
        ([0, 1, 1, 252], [0, 1, 1, 252], (3, 1, 252)),
        # The steps are measured between frames that have a pts, as the timing rule's are.
        ([0, 375, NOPTS, 1125, 1500], [0, 375, NOPTS, 1125, 1500], None),
    ],
)
def test_jump_bounds(pts: list[int], decoder: list[int], jump: tuple[int, int, int] | None) -> None:
    from seedvr2x.media.scan import (  # pyright: ignore[reportPrivateUsage]
        Discontinuity,
        _discontinuity,
    )

    expected = jump and Discontinuity(jump[0], Fraction(jump[1], 25), Fraction(jump[2], 25), True)
    assert _discontinuity(pts, decoder, Fraction(1, 25), True) == expected


def test_a_long_step_within_the_timing_rules_tolerance_is_no_jump() -> None:
    # In milliseconds, a frame every 10 s, ffmpeg's bound: a step past it, within the timing
    # rule's 1 ms of the frames' shortest, is a constant rate's own; 2 ms longer, a jump. A
    # frame every 15 s, the shortest step itself past the bound, has none, 2 ms longer or 20 s.
    from seedvr2x.media.scan import (  # pyright: ignore[reportPrivateUsage]
        Discontinuity,
        _discontinuity,
    )

    base = Fraction(1, 1000)
    steady = [0, 10_000, 20_000, 30_001]
    assert _discontinuity(steady, steady, base, False) is None
    late = [0, 10_000, 20_000, 30_002]
    assert _discontinuity(late, late, base, False) == Discontinuity(
        3, Fraction(20), Fraction(30_002, 1000), False
    )
    for last in (45_001, 45_002, 65_000):
        slow = [0, 15_000, 30_000, last]
        assert _discontinuity(slow, slow, base, False) is None


@needs_ffmpeg
@pytest.mark.parametrize(("rate", "suffix"), [("1/15", "nut"), ("1/15", "mov"), ("1/20", "nut")])
def test_a_constant_rate_under_ffmpegs_bound_is_read(
    tmp_path: Path, rate: str, suffix: str
) -> None:
    # A frame every 15 or 20 s: every step past ffmpeg's bound of 10 s, and none a jump
    # (media/scan.py, JUMP_FORWARD). Read, as before the frame index, where the jump's refusal
    # took each for one, "its timestamps jump at frame 1, from 0.000 to 15.000 s".
    path = tmp_path / f"slow.{suffix}"
    run(
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s=64x48:r={rate}",
        "-frames:v",
        "6",
        "-c:v",
        "ffv1",
        str(path),
    )
    period = 1_000_000 / Fraction(rate)
    assert scan(path) == Scan(6, 5, period, period)
    source = examine(path)
    assert (source.frames, source.stream.frame_rate) == (6, Fraction(rate))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("muxer", "suffix", "start"), [("mpegts", "ts", 1.4), ("matroska", "mkv", 0)]
)
def test_a_variable_rate_under_ffmpegs_bound_is_no_jump(
    tmp_path: Path, muxer: str, suffix: str, start: float
) -> None:
    # A frame every 15 s, one step 2 ms longer: a variable frame rate, refused as one by the
    # timing rule, as before the frame index, and no jump, the shortest step being itself past
    # ffmpeg's bound (media/scan.py, JUMP_FORWARD). Taken for one, the MPEG-TS file was refused
    # as jumping "at frame 3, from 31.400 to 46.402 s", with a remux as its way out, which
    # leaves its last five frames 38 to 42 ms apart.
    base = tmp_path / "base.mkv"
    run(*SOURCE, "-frames:v", "6", "-c:v", "mpeg2video", "-g", "1", str(base))
    path = tmp_path / f"slow.{suffix}"
    # The copy's timestamps set in the base file's milliseconds.
    late = "setts=ts=N*15000+2*gte(N\\,3)"
    run("-i", str(base), "-c", "copy", "-bsf:v", late, "-f", muxer, str(path))
    scanned = scan(path)
    assert scanned.index is not None and scanned.discontinuity is None
    times = [float(pts * scanned.index.time_base) for pts in scanned.index.pts.tolist()]
    assert times == [pytest.approx(start + at) for at in (0, 15, 30, 45.002, 60.002, 75.002)]
    with pytest.raises(MediaError) as refused:
        examine(path)
    assert str(refused.value) == (
        f"{path}: variable frame rate: frames last from 15000 to 15002 ms, and only a constant"
        " frame rate is supported"
    )
    if suffix == "ts":
        fixed = tmp_path / "fixed.mkv"
        run("-i", str(path), "-map", "0", "-map", "-0:d", "-c", "copy", str(fixed))
        remuxed = indexed(fixed)
        assert max(np.diff(remuxed.pts).tolist()[1:]) * remuxed.time_base < Fraction(1, 10)


def bounded[T](reader: Reader, read: Callable[[Reader], T], seconds: float = 60.0) -> T:
    """read(reader), a hang failing the test rather than blocking the suite: after `seconds`,
    ffmpeg is killed, which ends any wait on its pipes."""
    done: list[T] = []
    failed: list[Exception] = []

    def run() -> None:
        try:
            done.append(read(reader))
        except Exception as error:
            failed.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        running = reader._run
        if running is not None:
            running._process.kill()
        thread.join(30)
        pytest.fail(f"the read hung: no end in {seconds:g} s")
    if failed:
        raise failed[0]
    return done[0]


def lagging(lag: int) -> type[reading._Run]:
    """A run whose CRC-32 of each frame it selects comes only once `lag` more frames have been
    taken from its stdout, or at stdout's end, or once ffmpeg has exited: a hash branch holding
    frames back, as rawvideo's frame threads did on the box (media/reader.py, ONE_THREAD), whose
    reader waited for that CRC-32 with stdout unread."""

    class Lagging(reading._Run):
        def _drain(
            self, descriptor: int, into: "queue.Queue[tuple[int, str] | None]"
        ) -> threading.Thread:
            if into is not self.hashes:
                return super()._drain(descriptor, into)
            lines: queue.Queue[tuple[int, str] | None] = queue.Queue()
            drained = super()._drain(descriptor, lines)

            def gate() -> None:
                selected = 0
                while True:
                    line = lines.get()
                    if line is not None and (self.select is None or line[0] >= self.select):
                        selected += 1
                        while (
                            self.taken < selected + lag
                            and self._end is None
                            and self._process.poll() is None
                        ):
                            time.sleep(0.002)
                    into.put(line)
                    if line is None:
                        return

            threading.Thread(target=gate, daemon=True).start()
            return drained

    return Lagging


def stalling(directory: Path, written: int) -> None:
    """An ffmpeg in directory that writes `written` bytes of zeros on stdout, then nothing, and
    never ends: a stalled one, whatever the cause."""
    script = directory / "ffmpeg"
    script.write_text(f"#!/bin/sh\nhead -c {written} /dev/zero\nexec sleep 60\n")
    script.chmod(0o755)


def test_each_output_encodes_in_one_thread() -> None:
    # rawvideo's frame threads held frames back, a frame's CRC-32 behind frames still to come
    # through stdout: the box's hang (media/reader.py, ONE_THREAD). Each output's encoder runs in
    # one thread; and none takes the source's metadata or chapters, as the first pass's doesn't,
    # whose header ffmpeg prints among the lines it reads (ffmpeg.NO_METADATA).
    conversion = Conversion("yuv420p", "709", "limited", "left")
    for md5 in (None, 6):
        command = read_command(Path("in.mkv"), conversion, "1.000000", 1000, 5, md5)
        maps = [k for k, part in enumerate(command) if part == "-map"]
        ends = [k for k, part in enumerate(command) if part.startswith("pipe:")]
        assert len(maps) == len(ends) == (2 if md5 is None else 3)
        for start, end in zip(maps, ends, strict=True):
            options = command[start:end]
            assert ("-threads", "1") in pairwise(options), options
            assert ("-map_metadata", "-1") in pairwise(options), options
            assert ("-map_chapters", "-1") in pairwise(options), options


@needs_ffmpeg
@pytest.mark.parametrize("lag", [1, 5, READ_AHEAD])
def test_a_lagging_hash_is_read_ahead_never_waited_on(
    synth: Callable[[str], Source], monkeypatch: pytest.MonkeyPatch, lag: int
) -> None:
    # A frame's CRC-32 held back until `lag` frames after it have come through stdout: the reader
    # reads ahead meanwhile, READ_AHEAD frames at most, and every frame comes exact. Waiting for
    # it with stdout unread, as the reader did, hangs: the watchdog's bound, made short, then fails
    # the read, and bounded the test.
    monkeypatch.setattr(reading, "_Run", lagging(lag))
    monkeypatch.setattr(reading, "STALL", 10.0)
    source = synth("x264 MKV")
    full = reference(source.path)
    reader = source.reader(150, md5=True)
    try:
        bounded(reader, lambda read: read.read(8))
    finally:
        reader.stop()
    assert reader.md5s == full.md5[150:158]
    assert len(reader.attempts) == 1 and lag <= reader.held <= READ_AHEAD


@needs_ffmpeg
def test_a_hash_lagging_past_the_read_ahead_fails_naming_it(
    synth: Callable[[str], Source],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Further behind than READ_AHEAD frames, more than ffmpeg's queues hold with one encoder
    # thread an output: the read fails, saying so, and logged, rather than holding frames without
    # end.
    monkeypatch.setattr(reading, "_Run", lagging(READ_AHEAD + 1))
    caplog.set_level(logging.ERROR)
    source = synth("x264 MKV")
    reader = source.reader(150)
    try:
        with pytest.raises(MediaError) as raised:
            bounded(reader, lambda read: read.read(8))
    finally:
        reader.stop()
    said = (
        rf"{re.escape(str(source.path))}: frame 150, attempt 1, decoding from frame \d+, keyframe"
        rf" \d+: frame 150's CRC-32 hasn't come with {READ_AHEAD} more frames read on stdout"
    )
    assert re.search(said, str(raised.value)) and re.search(said, caplog.text)
    assert reader.held == READ_AHEAD


@needs_ffmpeg
@pytest.mark.parametrize(("frames", "awaited"), [(0, "frame 150's RGB"), (1, "frame 150's CRC-32")])
def test_a_stalled_ffmpeg_fails_within_the_bound(
    synth: Callable[[str], Source],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    frames: int,
    awaited: str,
) -> None:
    # ffmpeg stalled, whatever the cause: nothing from it for STALL seconds (made 1 s here) while
    # the reader waits on it stops it, and the read fails naming the read and what it waited for,
    # logged, where it hung. Before its first frame, or with that frame taken from stdout and its
    # CRC-32 not come, the next frame being read ahead meanwhile.
    source = synth("x264 MKV")  # its index made by the real ffmpeg
    stalling(tmp_path, frames * 160 * 96 * 6)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(reading, "STALL", 1.0)
    caplog.set_level(logging.ERROR)
    reader = source.reader(150)
    try:
        with pytest.raises(MediaError) as raised:
            bounded(reader, lambda read: read.read(3), 30)
    finally:
        reader.stop()
    said = (
        rf"{re.escape(str(source.path))}: frame 150, attempt 1, decoding from frame \d+, keyframe"
        rf" \d+: nothing from ffmpeg in 1 s, neither RGB nor a hash line, while the read waited"
        rf" for {re.escape(awaited)}(, the next frame read ahead meanwhile)?: stopped"
    )
    assert re.search(said, str(raised.value)) and re.search(said, caplog.text)
    assert reader.position == 150 and reader.held == 0  # no frame given without its CRC-32


@needs_ffmpeg
def test_no_stall_while_the_caller_holds_the_reader(
    synth: Callable[[str], Source], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Between two reads, ffmpeg waits on the reader, its stdout full, for as long as the caller
    # takes (a window on the GPU): no stall. The watchdog counts the reader's own waits only.
    monkeypatch.setattr(reading, "STALL", 2.0)
    source = synth("x264 MKV")
    full = reference(source.path)
    reader = source.reader(150, md5=True)
    try:
        bounded(reader, lambda read: read.read(2))
        time.sleep(5)
        bounded(reader, lambda read: read.read(2))
    finally:
        reader.stop()
    assert reader.md5s == full.md5[150:154] and len(reader.attempts) == 1


@needs_ffmpeg
def test_many_reads_with_md5s(synth: Callable[[str], Source]) -> None:
    # The box's hang came 1 read in about 716 with the MD5 branch: many reads with it here, 8
    # frames each, at random targets, every one exact at the first try, none hung; the frames read
    # ahead while a hash lagged, normally none, printed (pytest -rP).
    source = synth("x264 MKV")
    full = reference(source.path)
    held = 0
    for n in sorted(random.Random(3).sample(range(source.frames - 8), 40)):
        reader = source.reader(n, md5=True)
        try:
            bounded(reader, lambda read: read.read(8))
        finally:
            reader.stop()
        assert reader.md5s == full.md5[n : n + 8] and len(reader.attempts) == 1, n
        held = max(held, reader.held)
    print(f"40 reads of 8 frames with their MD5s: at most {held} frames read ahead")
    assert held <= READ_AHEAD
