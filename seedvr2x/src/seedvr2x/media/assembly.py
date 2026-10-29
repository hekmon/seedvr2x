"""Assembly (DESIGN.md, Output: Assembly): a job's output segments joined into one Matroska file by
ffmpeg's stream copy, with the source's other streams (audio, subtitles, attachments, cover art),
its chapters and its metadata; the video's timestamps on the frame grid, exactly; the file checked
whole, by its packets, before it is put in place.

The join is sptenc's (ffmpeg/concat.go, the user's own tool, whose lessons DESIGN.md names; its
method taken, no code): ffmpeg's concat demuxer given each segment's duration, computed from its
frame count, and every video timestamp snapped to the frame grid by the setts bitstream filter.
Without the durations, each segment starts where the one before ends by its own timestamps, 1 ms
early per join for FFV1 masters (cli._directory_refused); without the snap, a timestamp is rounded
twice, 226 of 1,000 frames 1 ms off a master's in 160 segments at 24000/1001 (measured on
2026-10-10, ffmpeg n9.0.2, as every figure here).

The grid: frame k of the job at start + k / R seconds, R the job's exact rate and `start` the
source's first video frame's time after the file's start time (start), written as Matroska holds
it, rounded once to the millisecond, halves away from zero as ffmpeg's rescaling rounds them:
what a remux of a source whose frames are on that grid holds (an MPEG-TS and an MP4 source: the
assembled video's timestamps were the remux's own video's, every one), and the timestamps of the
same frames written as one master, shifted, when `start` is a whole number of milliseconds.

PROVISIONAL (implementation, 2026-10-10: DESIGN.md says "the finished file's video timestamps
equal the source's" and not where they start): the output's times are the source's less the
source's start time, as ffmpeg's own remux writes them (ffmpeg -i SOURCE -map 0 -c copy out.mkv),
so that a source starting at 0 keeps every time, and the video its offset to the other streams.
The source's absolute times (-copyts) would leave a Matroska file whose first cluster starts
late: measured on an MPEG-TS source starting at 1.4734 s, the file then declares 43.192 s for
41.718 s of streams, its Duration element counting from 0.

What the join needs of the segments (Signature, segment_refusal) and of the source (unmuxable,
rehearse) can be asked before any work, so that no refusal comes after the upscale."""

import filecmp
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from array import array
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, cast

import numpy as np
import numpy.typing as npt

from seedvr2x.media.ffmpeg import MediaError, ends_with_us
from seedvr2x.media.files import partial_path, replace_whole
from seedvr2x.media.index import NOPTS, FrameIndex

logger = logging.getLogger(__name__)

# What assemble writes in the work directory while it runs, removed when it returns, however it
# ends: the concat demuxer's list, the hashes of the packets the join read from the segments, the
# remux of the source's other streams the result is compared with, and the source's attachments,
# each a file named ATTACHED and the stream's index. A killed run may leave them: the caller's
# to know by name (work_file).
LIST = "assembly.ffconcat"
READ = "assembly.framehash"
OTHERS = "assembly.others.mkv"
# And what the question of two segments' codec headers writes there (joins): the two joined,
# and their list.
PAIR = "assembly.pair.mkv"
PAIR_LIST = "assembly.pair.ffconcat"
WORK_FILES = (LIST, READ, OTHERS, PAIR, PAIR_LIST)
ATTACHED = "assembly.attachment."
# And what the check of a cover picture writes there: the picture as the finished file holds it,
# a file named PICTURE and its stream's index in the finished file (work_file).
PICTURE = "assembly.picture."

# The hash of each packet's data, ffmpeg's own (libavutil/hash.c), as its framehash muxer and
# ffprobe's -show_data_hash compute it: MurmurHash3, 128 bits, the fastest of them. Measured on
# 2.16 GiB of 1080p FFV1 in the page cache: a read of the result hashing every packet took 0.38 s
# with murmur3, 0.49 s with crc32, 2.4 s with md5 and 4.0 s with sha256; the join 1.60 s alone,
# and 1.73 s hashing what it reads meanwhile. No adversary here: a damaged, missing or misplaced
# packet is what it must tell.
HASH = "murmur3"

# Matroska's time base as ffmpeg's muxer writes it and its demuxer reads it for a file whose
# TimestampScale is the default (libavformat/matroskaenc.c, mkv_init: avpriv_set_pts_info(st, 64,
# 1, 1000)): the unit the snap computes in.
MILLISECOND = Fraction(1, 1000)
# ffmpeg's AV_TIME_BASE, the unit of a file's start time and of the concat demuxer's durations.
MICROSECOND = Fraction(1, 1_000_000)
# Matroska's unit for chapter times (matroskaenc.c, mkv_write_chapters), which ffprobe reads back.
NANOSECOND = Fraction(1, 1_000_000_000)

# Below it, a quotient of two integers computed in double precision rounds as the exact quotient
# does (snap): 2**52.
EXACT = 2**52

# What ffmpeg prints at the warning level that fails nothing (_complaints), each matched whole.
# - avformat_find_stream_info's own report of a stream it couldn't describe, two lines of one
#   message (libavformat/demux.c, "Could not find codec parameters"): an attachment has no codec
#   to find ("Attachment: none"), on every Matroska file with a font whose MIME type ffmpeg
#   doesn't map (font/ttf, mkvmerge's), and a subtitle stream no size before its first packet.
#   A stream copy needs none of what that analysis looks for: the attachment is copied whole, its
#   bytes compared (tests/test_assembly.py).
# - the MP4 demuxer's note of a cover art stream, which has no time scale (libavformat/mov.c,
#   "timescale not set"): on every MP4 file with a cover, which is copied all the same, its packet
#   compared.
# - fftools' note of an audio stream that declares no channel layout, for which it takes its own
#   guess by the number of channels (fftools/ffmpeg_demux.c, "Guessed Channel Layout"): PCM in
#   Matroska, whose tracks hold a count of channels alone. A copy writes that count again, and
#   the packets as they are, compared.
# All from the source alone, input 0 of every command here, and input 2 of the join, which opens
# the source twice (_join). Anything else printed
# fails the run it is printed by, as the strict reads of the input copies do
# (media/checksums.py).
TOLERATED = (
    re.compile(
        r"\[in#[02](/\S+)? @ 0x[0-9a-f]+\] \[warning\] Could not find codec parameters for stream"
        r" \d+ \(.*\): .*"
    ),
    re.compile(
        r"Consider increasing the value for the 'analyzeduration' \(\d+\) and 'probesize' \(\d+\)"
        r" options"
    ),
    re.compile(r"\[in#[02](/\S+)? @ 0x[0-9a-f]+\] \[warning\] stream \d+, timescale not set"),
    re.compile(r"\[aist#[02]:\d+/\S+ @ 0x[0-9a-f]+\] \[warning\] Guessed Channel Layout: .*"),
)
# And, for a cover picture alone, ffmpeg's note that a stream of one frame has no rate to
# estimate (libavformat/demux.c, avformat_find_stream_info: "not enough frames to estimate rate;
# consider increasing probesize"), printed when the probe of a source reaches its probesize
# before some other stream has its codec parameters (two PGS tracks with no packet in the
# window, a Blu-ray rip's; an audio track whose first packet comes late, tests/test_assembly.py).
# Decided on 2026-10-28 (DESIGN.md, Output, Assembly, "Met on 2026-10-23"): tolerated for a cover
# picture, the picture being attached whole, its bytes compared; for any other stream it fails
# the run as before, the stream being the one the message numbers (_complaints).
NO_RATE = re.compile(
    r"\[in#[02](/\S+)? @ 0x[0-9a-f]+\] \[warning\] Stream #(\d+): not enough frames to estimate"
    r" rate; consider increasing probesize"
)
# A message's prefix at -loglevel level+warning: its contexts, then its level.
PREFIX = re.compile(r"^(\[[^\]]* @ 0x[0-9a-f]+\] )*\[(warning|error|fatal|panic)\] ")
# The muxer's second message when it refuses a stream, which says nothing of why: the first does.
NO_HEADER = "Could not write header"

FFMPEG = ("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "level+warning", "-y")

# The dispositions the upscaled video never takes from the source's: those of a picture attached
# to a file, which Matroska's muxer writes as an attachment (libavformat/matroskaenc.c).
NOT_A_VIDEOS = frozenset({"attached_pic", "timed_thumbnails", "still_image"})
# The tags the upscaled video takes from the source's video stream; its other tags aren't copied:
# mkvmerge's statistics among them (BPS, DURATION, NUMBER_OF_FRAMES, NUMBER_OF_BYTES,
# _STATISTICS_*, each with a language suffix as ffmpeg reads them), which would describe the
# source's video. PROVISIONAL (implementation, 2026-10-10: DESIGN.md names "the source's other
# streams" and not the video's own records).
VIDEO_TAGS = ("language", "title")
# The tag of the first segment's video stream the upscaled video keeps: what encoded it.
KEPT = "ENCODER"


@dataclass(frozen=True)
class Stream:
    """A stream of a file, as ffprobe declares it."""

    index: int  # in the file, ffmpeg's own: -map 0:index
    kind: str  # ffprobe's codec_type: "video", "audio", "subtitle", "attachment", "data"
    codec: str  # ffprobe's codec_name, "" when it has none
    tags: Mapping[str, str]
    dispositions: tuple[str, ...]  # those set, ffprobe's names: "default", "forced"...
    extradata: str  # the hash of its extradata (HASH), "" without any: an attachment's content
    # Its start time as ffprobe declares it, the first of its timestamps, in seconds; None when
    # it declares none (an attachment, a stream no packet of which was read).
    start: Fraction | None = None
    # A cover picture of the source, turned into the attachment the finished file holds it as
    # (_pictured): its kind is then "attachment", its tags the file name and type it is attached
    # with, its index still the source's.
    picture: bool = False


@dataclass(frozen=True)
class Chapter:
    time_base: Fraction  # seconds per unit of start and end
    start: int
    end: int


@dataclass(frozen=True)
class Layout:
    """What a file holds besides its packets: its streams in order, its chapters, its start
    time."""

    streams: tuple[Stream, ...]
    chapters: tuple[Chapter, ...]
    start_time: int | None  # in microseconds, None when the file declares none

    @property
    def video(self) -> Stream:
        """The first video stream, the one seedvr2x reads (v:0)."""
        return next(stream for stream in self.streams if stream.kind == "video")

    @property
    def others(self) -> tuple[Stream, ...]:
        """Every stream but the first video stream, in the file's order."""
        video = self.video
        return tuple(stream for stream in self.streams if stream is not video)


@dataclass(frozen=True)
class Signature:
    """What a job's segments must share for their packets to make one stream by stream copy: the
    concat demuxer gives every packet the first file's stream, its codec parameters and its time
    base, and rescales nothing (libavformat/concatdec.c, copy_stream_props, concat_read_packet:
    each file's offset is added in that file's own time base)."""

    index: int  # the video stream's index in the file
    codec: str
    width: int
    height: int
    pix_fmt: str
    time_base: Fraction
    extradata: str  # the hash of the codec's out-of-band headers (HASH), "" without any


@dataclass(frozen=True)
class Unmuxable:
    """A stream of the source that Matroska's muxer refuses as it is."""

    index: int  # the stream's, in the source
    kind: str  # ffprobe's codec_type
    codec: str  # ffprobe's codec_name, "" when it has none
    reason: str  # ffmpeg's own words


@dataclass(frozen=True)
class Assembled:
    """An assembly done: the file in place, checked."""

    frames: int  # its video's
    size: int  # the file's, in bytes
    streams: int  # all of them, the video among them
    chapters: int
    joined: float  # seconds the join took, what it reads of the source before among them
    checked: float  # seconds its check took, the read of the file among them
    # How far another stream's packets run ahead of the video's in the file's order, at most, in
    # seconds, by their timestamps: the file's interleaving (INTERLEAVED).
    ahead: float


@dataclass
class _Packets:
    """The packets of one stream of a file, in the file's order, as a read of it gives them."""

    # The video's: each packet's pts, size in bytes and data hash.
    pts: list[int] = field(default_factory=lambda: [])
    size: list[int] = field(default_factory=lambda: [])
    hash: list[str] = field(default_factory=lambda: [])
    # Another stream's: 64 bits of each packet's whole record (pts, dts, duration, size, flags,
    # data hash), 8 bytes a packet where a film's audio tracks hold millions.
    digests: "array[int]" = field(default_factory=lambda: array("Q"))
    last: str = ""  # the last record read, for a message

    def __len__(self) -> int:
        return len(self.digests)


# The lead of another stream over the video, in the file's order, in seconds, beyond which an
# assembled file is warned of as poorly interleaved. fftools holds its inputs within 0.1 s of
# each other (fftools/ffmpeg_sched.c, SCHEDULE_TOLERANCE): 0.10 s measured on a join, as on
# ffmpeg's own remux of that source, where libavformat gives up ordering packets at 10 s
# (max_interleave_delta), what a starved input gives (_join). PROVISIONAL (implementation,
# 2026-10-12: DESIGN.md asks nothing of the interleaving): said, the file being whole all the same,
# as a source whose audio starts late or ends early would have it.
INTERLEAVED = 1.0


def _counted(count: int, noun: str) -> str:
    """`count` and its noun, plural unless the count is 1 (cli._plural's rule: this module
    can't import the CLI)."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _absolute(path: Path) -> Path:
    """path as ffmpeg and ffprobe must be given it, absolute: they take what comes before the
    first colon of a name for a protocol, and a relative `Film: 2.mkv` is refused, "Protocol not
    found" (ffmpeg n9.0.2, 2026-10-12), where the same name after a directory's slash is a
    file's. Not resolved: a link stays the name the caller gave."""
    return Path(os.path.abspath(path))


def work_file(name: str) -> bool:
    """Whether a file of this name in a work directory is one assemble writes there."""
    return name in WORK_FILES or any(
        name.startswith(prefix) and name.removeprefix(prefix).isdecimal()
        for prefix in (ATTACHED, PICTURE)
    )


def start(index: FrameIndex) -> Fraction:
    """Where the job's frame grid starts in the assembled file, in seconds: the source's first
    video frame's time less the file's start time, as ffmpeg computes it reading the source
    without -copyts: the start time, in microseconds, is rounded to the stream's time base, then
    taken off each timestamp (fftools/ffmpeg_demux.c, ts_fixup: pkt->pts += av_rescale_q(ts_offset,
    AV_TIME_BASE_Q, pkt->time_base)). 0 for a first frame without a timestamp, as
    rate.first_time has it. Exact: an MPEG-TS source's 902/90000 s, 10.02 ms, is no whole number
    of milliseconds, and the grid is rounded once, with it."""
    pts = int(index.pts[0]) if index.frames else NOPTS
    if pts == NOPTS:
        return Fraction(0)
    return (pts - _rescale(index.start_time or 0, MICROSECOND, index.time_base)) * index.time_base


def grid(frames: int, frame_rate: Fraction, first: Fraction) -> npt.NDArray[np.int64]:
    """The frame grid as Matroska holds it: frame k's timestamp in milliseconds, first + k /
    frame_rate seconds rounded once, halves away from zero (up: no time here is negative), as
    ffmpeg's av_rescale_q rounds the timestamps it gives a muxer. Exact integers."""
    n, d = frame_rate.numerator, frame_rate.denominator
    a, b = first.numerator, first.denominator
    return np.array(
        [(2000 * (a * n + k * d * b) + b * n) // (2 * b * n) for k in range(frames)],
        dtype=np.int64,
    )


def snap(frame_rate: Fraction, first: Fraction, frames: int) -> str:
    """The setts bitstream filter that puts every video timestamp of the join on the frame grid
    (grid), for a job of `frames` frames: each timestamp, in the concat demuxer's timeline, which
    starts at 0, becomes the index of the nearest frame, then that frame's time, first + k /
    frame_rate, in milliseconds. sptenc's snap (ffmpeg/concat.go, frameGridSnap), with the
    grid's start inside the one rounding.

    Exactly, in the double precision ffmpeg's expressions compute in (libavutil/eval.c): the
    time is written as one quotient of integers, (c0 + c1 k) / c2, each below 2**52, so exact in
    a double; the division gives the double nearest the exact quotient, off by at most its
    2**-53rd; a quotient that is no half-integer lies 1 / (2 c2) or more from the nearest one,
    more than that error while its numerator is under 2**52, and a half-integer one is exact in a
    double: round() then rounds as the exact value does, halves away from zero as ffmpeg's
    rescaling. setts itself rounds halves to even (libavcodec/bsf/setts.c, llrint), hence round()
    inside. Checked on every frame by assemble all the same.

    - The filter's unit is the segments' time base, the millisecond (Signature): fftools gives a
      bitstream filter of a copied stream the input stream's time base, and its output goes to
      the muxer in that base (fftools/ffmpeg_mux.c, bsf_init, mux_packet_filter), Matroska's
      own: nothing rounds after it.
    - A decode timestamp is snapped by the same function, which never swaps two timestamps: with
      B-frames they stay in order and none passes its packet's presentation timestamp, so that
      ffmpeg reports neither (fftools/ffmpeg_mux.c, mux_fixup_ts).
    - A missing timestamp, NOPTS, the lowest int64 and no time, stays missing (sptenc's): each
      expression is quoted, its commas otherwise parting the filters of a list.
    - Each packet lasts from its frame's time to the next one's, so that the file ends where the
      grid does: 41.708 s for 1,000 frames at 24000/1001, where the last packet's own 41 or 42
      ms, a master's, end it within a millisecond of that.
    - setts' time_base option, which could make the muxer do the rounding, isn't used: it means
      another thing from ffmpeg 8.1 on, where a result is rescaled from the input time base, than
      in 7.1, where it is taken as it is (libavcodec/bsf/setts.c at n7.1 and n9.0.2).

    Raises MediaError when the rate and the grid's start are too fine for the job's length to be
    computed exactly."""
    n, d = frame_rate.numerator, frame_rate.denominator
    a, b = first.numerator, first.denominator
    common = math.gcd(1000 * a * n, 1000 * d * b, b * n)
    c0, c1, c2 = 1000 * a * n // common, 1000 * d * b // common, b * n // common
    # A frame's index is found from a millisecond timestamp times the rate's numerator: exact
    # too. PROVISIONAL (implementation, 2026-10-10): refused beyond, where no real job is: 10
    # million frames at 24000/1001 after an MPEG-TS start use 3e11 of the 4.5e15.
    last = c0 + c1 * (frames + 1)
    if last >= EXACT or (last // c2 + 1000) * n >= EXACT:
        raise MediaError(
            f"{frames} frames at {n}/{d} fps from {first} s: too fine a rate and start for"
            " ffmpeg's expressions to place the last frames exactly"
        )

    def place(k: str) -> str:
        return f"round(({c0}+{c1}*{k})/{c2})"

    def nearest(timestamp: str) -> str:
        return f"round({timestamp}*{n}/{1000 * d})"

    def at(timestamp: str) -> str:
        return f"'if(eq({timestamp},NOPTS),NOPTS,{place(nearest(timestamp))})'"

    k = nearest("PTS")
    duration = f"'{place(f'({k}+1)')}-{place(k)}'"
    return f"setts=pts={at('PTS')}:dts={at('DTS')}:duration={duration}"


def concat_list(
    segments: Sequence[tuple[Path, int]],
    frame_rate: Fraction,
    firsts: Sequence[int | None] | None = None,
) -> str:
    """The concat demuxer's script for the segments, (path, frames) in order: each with its
    duration, from its frame count (sptenc's: ffmpeg/concat.go, GenerateConcatList). The demuxer
    starts each file where the files before it end by their durations, in microseconds
    (libavformat/concatdec.c, concat_read_header, open_file), whatever the timestamps inside
    them, and takes the file's own start time off its timestamps (file_inpoint), so that a
    segment starting at another time than 0 is placed as one starting at 0.

    firsts: for each segment, its first timestamp in milliseconds when the file doesn't declare
    it as its start time, else None (_signatures): written as the segment's inpoint, which the
    demuxer then takes off in the start time's place. ffmpeg finds no start time for an H.264
    file of 1 to 3 frames with B-frames, none of whose packets gets a decode timestamp, and
    counts from 0: such a segment starting at 0.5 s came out 0.5 s late, on the segment after
    it, ffmpeg reporting decode timestamps out of order.

    A duration is written as the difference of two times rounded to the microsecond, its
    segment's end and its start on the grid, so that every file starts at its exact time rounded
    once: rounded each on its own, the durations of 800 segments of 121 frames at 24000/1001,
    0.33 microsecond short each, would start the last one 0.27 ms early. With 6 decimals, which
    the demuxer reads digit by digit (libavutil/parseutils.c, av_parse_time).

    A path is written whole behind "file:", which the demuxer then opens as it is, resolving
    nothing against the script's own place (concatdec.c, add_file), and in single quotes, where
    everything but a quote is literal (libavutil/avstring.c, av_get_token): a quote closes them,
    is written escaped, and opens them again. A line break can't be written: refused."""
    lines = ["ffconcat version 1.0"]
    done = 0
    for place, (path, frames) in enumerate(segments):
        absolute = os.path.abspath(path)
        if "\n" in absolute or "\r" in absolute:
            raise MediaError(
                f"{absolute!r}: a line break in its name, which ffmpeg's concat demuxer can't be"
                " given: rename it"
            )
        begins = round(done / frame_rate / MICROSECOND)
        done += frames
        lasts = round(done / frame_rate / MICROSECOND) - begins
        quoted = absolute.replace("'", "'\\''")
        lines.append(f"file 'file:{quoted}'")
        first = None if firsts is None else firsts[place]
        if first is not None:
            lines.append(
                f"inpoint {'-' if first < 0 else ''}{abs(first) // 1000}.{abs(first) % 1000:03d}"
            )
        lines.append(f"duration {lasts // 1_000_000}.{lasts % 1_000_000:06d}")
    return "\n".join(lines) + "\n"


def layout(path: Path) -> Layout:
    """What path holds besides its packets (Layout), from ffprobe.

    Raises MediaError when ffprobe fails."""
    path = _absolute(path)
    entries = (
        "stream=index,codec_type,codec_name,extradata_hash,start_time"
        ":stream_tags:stream_disposition"
        ":chapter=time_base,start,end:format=start_time"
    )
    probed = _ffprobe(path, "-show_entries", entries)
    streams: list[Stream] = []
    for stream in cast(list[dict[str, Any]], probed.get("streams") or []):
        flags = cast(dict[str, Any], stream.get("disposition") or {})
        tags = cast(dict[str, Any], stream.get("tags") or {})
        began = str(stream.get("start_time", "N/A"))
        streams.append(
            Stream(
                int(stream["index"]),
                str(stream.get("codec_type", "")),
                str(stream.get("codec_name", "")),
                {str(key): str(value) for key, value in tags.items()},
                tuple(str(name) for name, value in flags.items() if value),
                str(stream.get("extradata_hash", "")),
                None if began in ("N/A", "") else Fraction(began),
            )
        )
    chapters = tuple(
        Chapter(Fraction(str(chapter["time_base"])), int(chapter["start"]), int(chapter["end"]))
        for chapter in cast(list[dict[str, Any]], probed.get("chapters") or [])
    )
    declared = str(cast(dict[str, Any], probed.get("format") or {}).get("start_time", "N/A"))
    # Printed with 6 decimals, the microseconds it holds (media/scan.py, read_packets).
    start_time = None if declared == "N/A" else round(Fraction(declared) / MICROSECOND)
    return Layout(tuple(streams), chapters, start_time)


def signature(path: Path) -> Signature:
    """What the segment at path must share with a job's other segments (Signature), from its
    first video stream. 8 ms a file.

    Raises MediaError when ffprobe fails, or finds no video stream."""
    return _segment(_absolute(path))[0]


def mismatch(first: Signature, other: Signature) -> str:
    """Why a segment of signature `other` can't be joined to one of signature `first`, or "":
    what differs, each with both values."""
    names = {
        "index": "its video stream's index",
        "codec": "codec",
        "width": "width",
        "height": "height",
        "pix_fmt": "pixel format",
        "time_base": "time base",
        "extradata": "codec headers (their hash)",
    }
    differing = [
        f"{said} {getattr(other, name) or 'none'}, not {getattr(first, name) or 'none'}"
        for name, said in names.items()
        if getattr(first, name) != getattr(other, name)
    ]
    return ", ".join(differing)


def joins(
    first: tuple[Path, int], other: tuple[Path, int], frame_rate: Fraction, work: Path
) -> str:
    """Why the segment `other`, its file and its frames, can't follow the job's first segment,
    `first`, in one stream by stream copy, or "": what their signatures differ in (mismatch),
    but codec headers that differ alone and decode alike (_unlike). Asked of each segment a
    user's command wrote as it is finished, and by the assembly of every segment (_signatures):
    one verdict. `work` is an existing directory for the files of the question (PAIR,
    PAIR_LIST), removed before this returns.

    Raises MediaError when ffprobe fails on either, or finds no video stream."""
    first, other = (_absolute(first[0]), first[1]), (_absolute(other[0]), other[1])
    shared, found = _segment(first[0])[0], _segment(other[0])[0]
    return _unlike(shared, found, first, other, frame_rate, _absolute(work))


def _unlike(
    shared: Signature,
    found: Signature,
    first: tuple[Path, int],
    other: tuple[Path, int],
    frame_rate: Fraction,
    work: Path,
) -> str:
    """joins, the two signatures read: `shared` the first segment's, `found` the other's.

    Codec headers are compared by a hash of all of them, and an encoder may write there what
    shapes no picture: libx265's hold its build, the CPU's capabilities and its thread options
    ("x265 (build 217) ... options: cpuid=1111039 frame-threads=5 ..."), which another machine,
    an update of x265 or of ffmpeg, or a resume changes, where libx264's 45 bytes hold neither
    (a review's finding, 2026-10-12). The assembled file has one set of headers, the first
    segment's (Signature): so headers that differ are no obstacle when the first's decode the
    other segment bit for bit, which is asked of ffmpeg itself, no header parsed: the two
    joined as the assembly joins them, and the second's frames decoded there against the same
    frames decoded from its own file (_decoded_otherwise). Measured on 2026-10-12 (ffmpeg
    n9.0.2), two segments of 12 frames at 128x96: libx265 with frame-threads=1 then 4, other
    headers, the same frames; mpeg4 without then with -flags +qpel, other headers and other
    frames, the decoder saying nothing: only the frames tell.

    PROVISIONAL (implementation, 2026-10-12: DESIGN.md has the segments joined by stream copy
    and nothing of their headers): accepted when they decode alike, said at INFO; refused
    otherwise, as any other difference is. It costs a decode of both segments and of the second
    again, each time it is asked."""
    differing = mismatch(shared, found)
    if not differing or replace(found, extradata=shared.extradata) != shared:
        return differing
    otherwise = _decoded_otherwise(shared.index, first, other, frame_rate, work)
    if otherwise:
        return f"{differing}, and under those of {first[0].name} {otherwise}"
    logger.info(
        "%s: its codec headers aren't those of %s, the job's first segment, whose headers the"
        " assembled file takes, and its frames decode the same under them: joined as it is",
        other[0],
        first[0].name,
    )
    return ""


def _decoded_otherwise(
    index: int,
    first: tuple[Path, int],
    other: tuple[Path, int],
    frame_rate: Fraction,
    work: Path,
) -> str:
    """How the frames of the segment `other` decode otherwise once joined to `first` under
    first's codec headers, or "" when they decode the same: the two joined by the concat demuxer
    and stream copy into a Matroska file in work, as the assembly joins every segment (_join:
    -auto_convert 0, the video stream `index` of each), the pair decoded, and its frames after
    the first segment's compared with the other's decoded from its own file, hash for hash
    (HASH). Anything ffmpeg prints, joining or decoding, is a difference."""
    listed, pair = work / PAIR_LIST, work / PAIR
    try:
        firsts = [_undeclared(*_segment(path)[2:]) for path, _ in (first, other)]
        listed.write_bytes(os.fsencode(concat_list([first, other], frame_rate, firsts)))
        command = [
            *(*FFMPEG, "-f", "concat", "-safe", "0", "-auto_convert", "0", "-i", str(listed)),
            *("-map", f"0:{index}", "-c", "copy", "-f", "matroska", str(pair)),
        ]
        _strict(command, f"{other[0]}: joining it to {first[0].name}")
        together, alone = _decoded(pair), _decoded(other[0])
    except MediaError as error:
        return f"it can't be joined and decoded: {error}"
    finally:
        listed.unlink(missing_ok=True)
        pair.unlink(missing_ok=True)
    if len(alone) != other[1] or len(together) != first[1] + other[1]:
        return (
            f"{len(together) - first[1]} of its frames decode, where its own file gives"
            f" {len(alone)} of its {other[1]}"
        )
    unlike = [
        place
        for place, (one, same) in enumerate(zip(together[first[1] :], alone, strict=True))
        if one != same
    ]
    if unlike:
        return (
            f"{len(unlike)} of its {other[1]} frames decode to other pictures, from frame"
            f" {unlike[0]}"
        )
    return ""


def _decoded(path: Path) -> list[str]:
    """The hash (HASH) of each frame of the first video stream of path, decoded, in the order
    the decoder gives them, each once (-fps_mode passthrough).

    Raises MediaError for a failure, or for anything ffmpeg prints."""
    command = [
        *(*FFMPEG, "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"),
        *("-f", "framehash", "-hash", HASH, "-"),
    ]
    with tempfile.TemporaryFile() as log:
        # In a process group of its own, as the join's ffmpeg (_run).
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            process_group=0,
            preexec_fn=ends_with_us(),
        )
        assert process.stdout is not None
        try:
            with process.stdout as lines:
                hashes = [
                    line.rsplit(",", 1)[-1].strip() for line in lines if not line.startswith("#")
                ]
            status = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        said = _complaints(_said(log))
    if status != 0 or said:
        raise MediaError(
            f"{path}: decoding it with ffmpeg: {' / '.join(said[-5:]) or f'exit status {status}'}"
        )
    return hashes


def segment_refusal(path: Path, frames: int, frame_rate: Fraction) -> str:
    """Why the segment at path, of `frames` frames at frame_rate, can't be assembled, or "": for
    a segment a user's command wrote (--segment-cmd), asked as soon as it is written, a read of
    its packets, no decode. A master of FFV1Writer's needs no asking.

    The join places a segment's frames by their timestamps (concat_list, snap): its video
    stream's time base must be Matroska's millisecond, its packets as many as its frames, each
    with a timestamp of its own, and each frame, in presentation order, less than half a frame
    from its place counted from the first packet's, with 1 ms to spare for the two roundings on
    the way, the segment's start's and the timestamp's own. A first timestamp that isn't 0 and a
    millisecond of jitter pass.

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md has a --segment-cmd output's frame count
    checked alone): a segment that fails this is one the assembly's own check would refuse at
    the end of the job (assemble).

    Raises MediaError when ffprobe fails, or finds no video stream."""
    path = _absolute(path)
    found, _, first, _ = _segment(path)
    if found.time_base != MILLISECOND:
        return (
            f"its video stream's time base is {found.time_base} s, where a Matroska file's is"
            f" {MILLISECOND}: written as Matroska with the default timestamp scale, it is"
        )
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", str(found.index)),
            *("-show_entries", "packet=pts", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
        process_group=0,
        preexec_fn=ends_with_us(),
    )
    if result.returncode != 0:
        raise MediaError(f"{path}: ffprobe failed reading its packets: {result.stderr.strip()}")
    read = [line.split(",")[0] for line in result.stdout.split()]
    if len(read) != frames:
        return f"{len(read)} video packets, where the segment has {frames} frames"
    if "N/A" in read:
        return f"its video packet {read.index('N/A')} has no timestamp"
    pts = sorted(map(int, read))
    period = 1 / frame_rate
    slack = period / 2 - MILLISECOND
    # The first packet's, which the join counts the segment's timestamps from (concat_list).
    origin = pts[0] if first is None else first
    for place, timestamp in enumerate(pts):
        off = (timestamp - origin) * MILLISECOND - place * period
        if abs(off) >= slack:
            return (
                f"its frame {place} is at {timestamp} ms, {float(off) * 1000:+.1f} ms from its"
                f" place at {frame_rate} fps counted from its first packet's,"
                f" {origin} ms, where the join needs every frame within"
                f" {float(slack) * 1000:.1f} ms of it: its timestamps aren't that rate's"
            )
    return ""


def unmuxable(source: Path) -> list[Unmuxable]:
    """The streams of source, but its first video stream, that Matroska can't hold as they are,
    in the file's order: by ffmpeg's own verdict, a stream copy of each alone into its Matroska
    muxer, written nowhere, and stopped before its first packet.

    The header is enough: the muxer refuses a codec when it writes the stream's track
    (libavformat/matroskaenc.c, mkv_write_track), and each case made on 2026-10-10 got the same
    verdict stopped there (-t 0) and copied whole: MP4's mov_text ("Subtitle codec mov_text
    (94213) is not supported."), a MOV file's timecode track and an MP4 file's chapter track
    ("Only audio, video, and subtitles are supported for Matroska."), eia_608 captions,
    pcm_bluray and pcm_dvd ("No wav codec tag found for codec pcm_bluray"), an MPEG-TS data
    stream, all refused; AAC, FLAC, Opus, AC-3, MP2, pcm_s16be, pcm_s24le, SubRip, ASS, WebVTT,
    a PNG or JPEG cover and a font, all copied. The output is /dev/null by name, a file to
    ffmpeg: on a pipe, which can't seek, the muxer refuses ADTS AAC from MPEG-TS, whose header
    it completes at the end ("Error parsing AAC extradata, unable to determine samplerate").

    Raises MediaError when ffprobe fails on the source, or it has no video stream."""
    source = _absolute(source)
    found: list[Unmuxable] = []
    whole = layout(source)
    for stream in _others(source, whole):
        # A cover picture isn't muxed as a track: it is attached (_pictured), whatever the
        # muxer says of a one-frame video track.
        refused = _unnamed(stream) if _is_picture(stream, whole.video) else _refusal(source, stream)
        if refused is not None:
            found.append(refused)
    return found


def _refusal(source: Path, stream: Stream) -> Unmuxable | None:
    """Matroska's muxer's verdict on one stream of the source (unmuxable), None when it takes
    it."""
    command = [
        *(*FFMPEG, "-i", str(source), "-map", f"0:{stream.index}", "-c", "copy"),
        *("-t", "0", "-f", "matroska", os.devnull),
    ]
    status, lines = _run(command)
    if status == 0:
        return None
    errors = [PREFIX.sub("", line) for line in lines if "[error]" in line]
    said = [error for error in errors if not error.startswith(NO_HEADER)] or errors or lines
    reason = " / ".join(said) or f"exit status {status}"
    return Unmuxable(stream.index, stream.kind, stream.codec, reason)


def left_out(source: Path, found: Layout) -> list[Unmuxable]:
    """The data streams of the source, `found` its layout, that Matroska can't hold, in the file's
    order: the ones the finished file leaves out (assemble). Decided on 2026-10-12 (DESIGN.md,
    Output, Assembly): a MOV's timecode track, an MP4's chapter track, an MPEG-TS data stream;
    audio, subtitles and the rest Matroska refuses stay refused (unmuxable_refusal). Asked of
    ffmpeg's muxer again wherever the streams are used (the assembly runs in a later process
    than the preflight), the data streams alone, one header each: no record to carry."""
    gone: list[Unmuxable] = []
    for stream in _others(source, found):
        if stream.kind == "data":
            refused = _refusal(source, stream)
            if refused is not None:
                gone.append(refused)
    return gone


def without(found: Layout, gone: Sequence[Unmuxable]) -> Layout:
    """The layout `found` without the streams `gone` (left_out), the others keeping their
    indexes in the source: every use of the source's other streams (the remux the assembly
    compares with, the join, the checks) then leaves them out."""
    out = {stream.index for stream in gone}
    return replace(found, streams=tuple(s for s in found.streams if s.index not in out))


def _kept(source: Path) -> tuple[Layout, list[Unmuxable]]:
    """The source's layout without the data streams the finished file leaves out (left_out,
    without), its cover pictures turned into attachments (_pictured), and those streams."""
    found = layout(source)
    gone = left_out(source, found)
    return _pictured(without(found, gone)), gone


# The file extension and the MIME type a cover picture is attached with, by its codec, when its
# source gives no file name or type: an MP4's cover (a covr atom) has neither, a Matroska
# attachment has both. PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md says "its file name, type and
# bytes the source's" and names no default for a container that has none): cover.jpg and
# cover.png, a second one cover_2.jpg; any other codec (an MP4 may hold a BMP) is refused
# (_unnamed) rather than attached under a type seedvr2x doesn't know.
COVERS = {"mjpeg": ("jpg", "image/jpeg"), "png": ("png", "image/png")}


def _is_picture(stream: Stream, video: Stream) -> bool:
    """Whether a stream of the source is a cover picture, a picture attached to the file as
    ffmpeg's demuxers show one: a video stream of one frame, disposition attached_pic, that is
    not the video seedvr2x reads (DESIGN.md, Output, Assembly, decided on 2026-10-28).
    PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md says "a cover picture"): attached_pic alone;
    the other dispositions of NOT_A_VIDEOS (timed_thumbnails, still_image) stay tracks, as
    ffmpeg's remux copies them."""
    return stream.kind == "video" and "attached_pic" in stream.dispositions and stream is not video


def _unnamed(stream: Stream) -> Unmuxable | None:
    """The refusal of a cover picture that can't be attached, or None: the source gives no file
    name or no type for it and its codec is none of COVERS' (the refusal's remedy, a remux
    without it, is unmuxable_refusal's)."""
    if stream.codec in COVERS or ("filename" in stream.tags and "mimetype" in stream.tags):
        return None
    return Unmuxable(
        stream.index,
        stream.kind,
        stream.codec,
        f"a cover picture of codec {stream.codec or 'unknown'} with no file name or type of its"
        " own: seedvr2x names those of a JPEG and a PNG only",
    )


def _pictured(found: Layout) -> Layout:
    """The layout `found` with each cover picture turned into an attachment (Stream.picture),
    the stream it is attached as: its file name and type the source's tags (a Matroska
    attachment's, ffmpeg's demuxer showing them as "filename" and "mimetype"), else COVERS'; the
    index its own in the source, which the picture is read from (_attachments).

    ffmpeg's remux makes a cover a one-frame video track, "V_MJPEG", which players list as a
    second video; Matroska holds a cover as an attachment, which MakeMKV writes for every file
    (DESIGN.md, Output, Assembly, decided on 2026-10-28, the user)."""
    video = found.video
    streams: list[Stream] = []
    unnamed = 0
    for stream in found.streams:
        if not _is_picture(stream, video):
            streams.append(stream)
            continue
        refused = _unnamed(stream)
        if refused is not None:
            raise MediaError(f"stream {stream.index}: {refused.reason}")
        extension, kind = COVERS.get(stream.codec, ("", ""))
        name = stream.tags.get("filename")
        if name is None:
            unnamed += 1
            name = "cover" if unnamed == 1 else f"cover_{unnamed}"
            name = f"{name}.{extension}"
        tags = {**stream.tags, "filename": name, "mimetype": stream.tags.get("mimetype", kind)}
        streams.append(
            replace(
                stream,
                kind="attachment",
                tags=tags,
                dispositions=(),
                extradata="",
                start=None,
                picture=True,
            )
        )
    return replace(found, streams=tuple(streams))


def covers(source: Path) -> tuple[tuple[Stream, ...], int]:
    """The cover pictures of the source, as the finished file attaches them (_pictured: kind
    "attachment", tags the file name and type, index the source's), and how many other
    attachments (fonts) it has, for the remake a refusal advises (media/rate.py, remake). Those
    seedvr2x can name only (_unnamed): a picture it can't is left to the remake's `-map 0`.

    Raises MediaError when ffprobe fails on the source."""
    found = layout(_absolute(source))
    video = next((stream for stream in found.streams if stream.kind == "video"), None)
    if video is None:
        return (), 0
    named = replace(
        found,
        streams=tuple(
            stream
            for stream in found.streams
            if not _is_picture(stream, video) or _unnamed(stream) is None
        ),
    )
    streams = _pictured(named).streams
    pictures = tuple(stream for stream in streams if stream.picture)
    return pictures, sum(1 for stream in streams if stream.kind == "attachment") - len(pictures)


def _pictures(found: Layout) -> frozenset[int]:
    """The indexes, in the source, of the cover pictures of `found` (_pictured)."""
    return frozenset(stream.index for stream in found.streams if stream.picture)


def left_out_warning(source: Path, gone: Sequence[Unmuxable], chapters: int) -> str:
    """What is said, once and at the start of a job, of the data streams left out of the finished
    file (left_out), or "" without any: each one named with ffmpeg's reason, and, when the source
    has chapters, that they are copied all the same (ffmpeg reads an MP4's chapter track as the
    file's chapters, and the chapters are copied from the source whatever streams are)."""
    if not gone:
        return ""
    several = len(gone) > 1
    plural = "s" if chapters != 1 else ""
    kept = f"; its {chapters} chapter{plural} are copied all the same" if chapters else ""
    return (
        f"{source}: Matroska can't hold {'some' if several else 'one'} of its data streams,"
        f" {'which are' if several else 'which is'} left out of the finished file: {_named(gone)}"
        f"{kept}"
    )


def _named(streams: Sequence[Unmuxable]) -> str:
    return "; ".join(
        f"stream {stream.index} ({stream.kind}{', ' + stream.codec if stream.codec else ''}):"
        f' ffmpeg says "{stream.reason}"'
        for stream in streams
    )


# The way out of each stream Matroska refuses that was tested (tests/test_assembly.py), by its
# codec: ffmpeg's option converting it in a remux, {index} the stream's, and what that keeps.
# Not pcm_dvd's: converted so from a VOB file, whose packets don't all carry a timestamp, the
# remux failed ("Can't write packet with unknown timestamp"); left out, it passes.
CONVERSIONS = {
    "mov_text": (
        "-c:{index} srt",
        "converts it to SubRip, its text and times kept, its styling and placement lost",
    ),
    "pcm_bluray": ("-c:{index} pcm_s24le", "rewraps its samples as they are, nothing lost"),
}


def unmuxable_refusal(source: Path, found: Sequence[Unmuxable]) -> str:
    """The refusal of a one-file output for a source with streams Matroska can't hold (unmuxable),
    every one named with ffmpeg's reason, and the ways out, the user's: a remux of the source
    that converts or leaves out each, given in its place, or a directory output. "" when `found`
    holds data streams alone: those are left out of the finished file (left_out), not refused.

    DESIGN.md (Output, Assembly): audio, subtitles and the other streams Matroska can't hold are
    refused before any work, never dropped in silence; a data stream is left out with a warning,
    decided on 2026-10-12. Where both occur, the refusal names the refused alone.

    PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md leaves open whether the remux a refusal
    advises still leaves the data streams out): it does, as before. A plain `-map 0` remux
    fails on them ("Only audio, video, and subtitles are supported for Matroska."), and what
    is left out there is not warned of again."""
    refused = [stream for stream in found if stream.kind != "data"]
    if not refused:
        return ""
    named = _named(refused)
    several = len(refused) > 1
    options: list[str] = []
    notes: list[str] = []
    left_out = [stream.index for stream in found if stream.codec not in CONVERSIONS]
    for stream in found:
        way = CONVERSIONS.get(stream.codec)
        if way is None:
            options += ["-map", f"-0:{stream.index}"]
            notes.append(f"stream {stream.index} left out")
        else:
            # An option names a stream by its index in the output, the source's less the
            # streams left out before it: -map 0 keeps the source's order.
            placed = stream.index - sum(index < stream.index for index in left_out)
            option = way[0].format(index=placed)
            options += option.split()
            notes.append(f"stream {stream.index}: {option} {way[1]}")
    command = " ".join(["ffmpeg", "-i", "SOURCE", "-map", "0", "-c", "copy", *options])
    return (
        f"{source}: Matroska can't hold {'some' if several else 'one'} of its streams as"
        f" {'they are' if several else 'it is'}: {named}: the one-file output takes every other"
        " stream of the source, and seedvr2x drops none in silence. Give it a remux of the"
        f" source without {'them' if several else 'it'}:\n{command} remuxed.mkv\n"
        f"({'; '.join(notes)}); or give -o a directory, whose segments are then muxed with"
        " the source's streams by your own tools"
    )


def rehearse(source: Path, work: Path) -> str:
    """What ffmpeg says of a remux of the source's other streams into Matroska, whole, the very
    one an assembly makes and compares its result with (assemble), or "" when it says nothing:
    asked before any work, so that the assembly isn't refused for it after the upscale. Written
    in the directory work, which must exist, then removed; one read of the source, no decode.

    PROVISIONAL (implementation, 2026-10-10: the assembly runs ffmpeg strictly, anything it
    prints failing it, and a source's other streams can make it print, at their first packet or
    their last: a timestamp going back, a packet without one): said here, with the way out,
    ffmpeg's own remux of the source, which writes the timestamps it corrects.

    And whether that remux holds each stream where the assembly counts on it, at the source's
    own time less the file's start (moved): a stream that starts before the file's start time
    moves them all, the video with them, and the assembly's check of its frames' places would
    refuse the file once every segment is made.

    Raises MediaError when ffprobe fails on the source, or it has no video stream."""
    source, work = _absolute(source), _absolute(work)
    try:
        found, _ = _kept(source)
        _, _, starts = _reference(source, found, work)
    except MediaError as error:
        return (
            f"{error}: the one-file output takes the source's other streams as an ffmpeg remux"
            " holds them, and takes nothing ffmpeg warns of. Give it a remux of the source,"
            " which ffmpeg corrects as it writes:\nffmpeg -i SOURCE -map 0 -c copy remuxed.mkv\n"
            "or give -o a directory, whose segments are then muxed with the source's streams"
            " by your own tools"
        )
    return moved(source, found, starts)


def moved(source: Path, found: Layout, starts: Mapping[int, Fraction]) -> str:
    """Why the source's other streams, remuxed into Matroska, aren't where an assembly counts on
    them, or "": each one starting, as ffprobe declares it of the remux (`starts`, _reference's,
    by the stream's index in the source, in seconds), at the stream's own start in the source
    less the file's start time, as ffmpeg offsets every timestamp reading the source (start).
    `found` is the source's layout. Start for start, ffprobe's of both files: a first packet
    isn't one (an AAC stream's is 21 ms before the start ffprobe declares, its encoder's
    priming, in the source as in its remux).

    Matroska's timestamps start at 0: given a packet before that, ffmpeg's muxer moves every
    stream later by as much, the video among them in the join, off the frame grid the assembly
    places it on. A stream starts before its file's start time when libavformat leaves its start
    out of the file's. Measured on 2026-10-12 (ffmpeg n9.0.2) on a Matroska file whose video and
    audio start at 5.000 s and whose SubRip stream has a line at 1.000 s: the file's start time
    reads 5.000 s; in the remux of its other streams the subtitle starts at 0.000 s and the
    audio at 4.000, where their own starts less the file's are -4.000 and 0.000; ffmpeg's
    plain remux of the file holds the video at 4.000 s and the subtitle at 0, and is moved the
    same way in its turn; without that stream (-map 0 -map -0:2) every stream is at 0 and
    nothing moves.

    Told by a move every stream shares, within the millisecond of Matroska's rounding, read on
    those whose start ffprobe declares: one stream alone off its place is no move of the
    muxer's, and is left to the assembly's own check.

    Refused before any work, with the remux that mends the source (DESIGN.md, Output, Assembly,
    "Before any work, at every start": a stream that starts before the file's start, by which
    the muxer would move the others, decided on 2026-10-12, kept as built on 2026-10-28)."""
    begins = Fraction(found.start_time or 0) * MICROSECOND
    expected = {
        stream.index: stream.start - begins
        for stream in found.others
        if stream.start is not None and stream.index in starts
    }
    moves = {index: starts[index] - place for index, place in expected.items()}
    if not moves or min(moves.values()) <= MILLISECOND:
        return ""
    if max(moves.values()) - min(moves.values()) > MILLISECOND:
        return ""
    move = min(moves.values())
    early = min(expected, key=lambda index: expected[index])
    stream = next(stream for stream in found.others if stream.index == early)
    assert stream.start is not None
    if expected[early] >= 0:
        # No stream of its own explains it: said as found, with the one way out that is sure.
        return (
            f"{source}: remuxed into Matroska, each of its other streams is"
            f" {float(move):.3f} s later than its own start less the file's start time,"
            f" {float(begins):.3f} s; the upscaled video would be moved off its frame grid with"
            " them, and the one-file output couldn't be assembled. Give -o a directory, whose"
            " segments are then muxed with the source's streams by your own tools"
        )
    return (
        f"{source}: its stream {early} ({stream.kind}, {stream.codec or 'no codec'}) starts at"
        f" {float(stream.start):.3f} s, {float(-expected[early]):.3f} s before the file's start"
        f" time, {float(begins):.3f} s: Matroska's timestamps start at 0, so ffmpeg's muxer moves"
        f" every stream {float(move):.3f} s later, the upscaled video off its frame grid with"
        " them, and the one-file output couldn't be assembled. Give it a remux of the source"
        f" without that stream:\nffmpeg -i SOURCE -map 0 -map -0:{early} -c copy without.mkv\n"
        "(a plain remux keeps the stream, and its start); or give -o a directory, whose segments"
        " are then muxed with the source's streams by your own tools"
    )


# The part of the source a timed read takes to estimate the rehearsal's duration (reading_said):
# 64 MiB, a second or so at the 50 MB/s a Blu-ray rip read through WSL's mount of a Windows disk
# went at (milestone 7, 2026-10-23: 18.5 GB in 6 min 12 s), and 0.1 s or less at the 114 MB/s of a
# 7.5 GB source read from a local disk (66 s).
SAMPLE = 64 * 2**20


def reading_said(source: Path) -> str:
    """What the rehearsal (rehearse) reads before it does, and for about how long, as one line
    for the log: the whole source, once, by ffmpeg. DESIGN.md, Output, Assembly, decided on
    2026-10-28: "a line says what the rehearsal reads before it does".

    PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md leaves the line's wording, and the way its
    duration is known, open): the duration is the source's size over the speed of a timed read
    of SAMPLE bytes from its middle, plain file reads, the range dropped from the page cache
    first where the system lets it, so that a file read a moment ago isn't timed from memory
    (posix_fadvise: a mount that ignores it, WSL's, may still answer from memory, the figure then
    short). The remux itself reads the whole source and writes the other streams, so the figure
    is a floor; the line says "about". A read that fails is said with its reason, and no
    duration given."""
    size = source.stat().st_size
    what = f"{source}: the assembly rehearsed: the source's other streams remuxed once, reading"
    what += f" its {size / 2**30:.1f} GiB"
    try:
        seconds = _read_seconds(source, size)
    except OSError as error:
        return f"{what}, for a time not estimated ({error})"
    if seconds < 10:
        return f"{what}, a few seconds"
    if seconds < 90:
        return f"{what}, about {round(seconds / 5) * 5} s"
    return f"{what}, about {round(seconds / 60)} min"


def _read_seconds(source: Path, size: int) -> float:
    """The seconds to read `size` bytes of source at the speed of a timed read of SAMPLE of them
    (reading_said)."""
    length = min(size, SAMPLE)
    if length == 0:
        return 0.0
    offset = (size - length) // 2
    with source.open("rb", buffering=0) as handle:
        try:
            os.posix_fadvise(handle.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError) as error:
            # The figure may then be a read from memory, short: said in the log, never silently.
            logger.debug("%s: its sample not dropped from the page cache: %s", source, error)
        handle.seek(offset)
        started = time.monotonic()
        left = length
        while left > 0:
            chunk = handle.read(min(left, 4 * 2**20))
            if not chunk:
                break
            left -= len(chunk)
        elapsed = time.monotonic() - started
    read = length - left
    if read == 0:
        raise OSError("nothing read")
    return elapsed * size / read


def left_out_said(source: Path) -> None:
    """Say, as a warning, what the finished file leaves out of the source's data streams
    (left_out_warning), or nothing when it leaves out none: the part of preflight that a start
    trusting its rehearsal's record (cli._run) still runs, the per-stream trial (left_out) alone,
    a header each, well under a second."""
    source = _absolute(source)
    whole = layout(source)
    warning = left_out_warning(source, left_out(source, whole), len(whole.chapters))
    if warning:
        logger.warning("%s", warning)


def preflight(source: Path, work: Path) -> str:
    """Why a one-file output can't take the source's other streams, or "": the streams Matroska
    refuses (unmuxable, unmuxable_refusal), else what ffmpeg says of their remux (rehearse), said
    first with what it reads and for about how long (reading_said). For the start of a job,
    before its first pass; a resume whose record holds a passed rehearsal of the same content
    runs left_out_said alone (cli._copied).

    The data streams Matroska can't hold are no refusal: left out of the finished file, they are
    warned of, once, as the refusal would be said, whenever this runs (left_out_warning).

    PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md says "a warning naming it" and not where, nor
    whether the end of the assembly says it again): said here, at the start of every run of the
    job, and not as a separate warning at the end, whose line counts the streams it left out."""
    source, work = _absolute(source), _absolute(work)
    found = unmuxable(source)
    refusal = unmuxable_refusal(source, found)
    if refusal:
        return refusal
    warning = left_out_warning(source, found, len(layout(source).chapters))
    if warning:
        logger.warning("%s", warning)
    logger.info("%s", reading_said(source))
    return rehearse(source, work)


def assemble(
    output: Path,
    segments: Sequence[tuple[Path, int]],
    frame_rate: Fraction,
    source: Path,
    index: FrameIndex,
    work: Path,
    stale: Path | None = None,
) -> Assembled:
    """Join a job's output segments into the Matroska file `output`, with every other stream of
    `source`, and put it in place once it is proven whole.

    segments: each segment's file and its frames, in order; frame_rate: the job's exact rate R,
    frames per second; index: the source's frame index, for where the frame grid starts (start);
    work: an existing directory for the files this writes meanwhile (work_file), removed before
    it returns; stale: as FFV1Writer's, a file removed just before output is replaced.

    The join is one ffmpeg process, by stream copy (_join): the video from the segments through
    the concat demuxer (concat_list), its timestamps snapped to the grid (snap), every other
    stream of the source in the source's order, the video in its own video's place, with the
    source's chapters and metadata as ffmpeg's remux copies them. Before it, two reads of the
    source: its other streams remuxed alone, what the result is compared with (_reference), and
    its attachments written out, to be attached (_attachments). Written beside output
    (files.partial_path), then checked by its packets, read once, no decode:
    - its video packets are as many as the segments' frames; their data are those the join read
      from the segments, in order, by size and hash, a second output of the join hashing them
      meanwhile; each segment's frames are at its own places of the grid, in the segment's own
      presentation order; its codec headers are the first segment's, which every other
      segment has too, or decodes the same under (joins);
    - every other stream's packets are those of a remux of the source's other streams alone,
      made first: as many, each with its timestamps, duration, size, flags and data hash; an
      attachment's content is the source's; a cover picture is attached, with the source's file
      name and type and its bytes, read back and compared with the source's (_check_picture;
      DESIGN.md, Output, Assembly, decided on 2026-10-28);
    - its chapters are the source's, as many, at the source's times less its start time.
    Then renamed into place (files.replace_whole). A file at output is whole and checked, or the
    one that was there before.

    One unit of a run: it runs in the caller's thread, ffmpeg and ffprobe in process groups of
    their own, out of the terminal's reach (runtime/stop.py); any exception raised in that
    thread, Stopped, Terminated or KeyboardInterrupt among them, kills them, removes the partial
    file and the work files, and leaves what output held.

    The work directory also takes the source's attachments and cover pictures, written out to be
    attached, and the pictures read back from the file.

    Disk: the file takes the segments' size again, and the other streams', where output goes:
    checked first (os.statvfs, _room); an FFV1 output so needs twice its size until the caller
    removes the segments. The work directory takes the other streams' once more, for the time of
    their remux.

    Raises MediaError, naming what is wrong and the first packet that differs: segments that
    can't make one stream (mismatch), a source whose first frame is before its file's start, too
    little room, anything ffmpeg prints (TOLERATED apart), a failed check. ValueError for no
    segment, or one without frames."""
    if not segments or any(frames <= 0 for _, frames in segments):
        raise ValueError("an assembly needs segments, each with frames")
    total = sum(frames for _, frames in segments)
    first = start(index)
    if first < 0:
        # PROVISIONAL (implementation, 2026-10-10): no source measured has one. ffmpeg's muxer
        # would then shift every stream to keep the first frame at 0, the chapters apart.
        raise MediaError(
            f"{source}: its first video frame is {float(-first) * 1000:.3f} ms before the"
            " file's start time: not assembled; give -o a directory"
        )
    began = time.monotonic()
    output, source, work = _absolute(output), _absolute(source), _absolute(work)
    segments = [(_absolute(path), frames) for path, frames in segments]
    partial = partial_path(output)
    listed, read, others = work / LIST, work / READ, work / OTHERS
    attached: dict[int, Path] = {}
    try:
        # What a join killed there left, up to the job's size: gone before the room is read,
        # which it would be counted against.
        partial.unlink(missing_ok=True)
        source_layout, gone = _kept(source)  # refuses a source without a video
        attached = {
            stream.index: work / f"{ATTACHED}{stream.index}"
            for stream in source_layout.others
            if stream.kind == "attachment"
        }
        shared, tags, firsts = _signatures(segments, frame_rate, work)
        video_size = sum(path.stat().st_size for path, _ in segments)
        _room(output, video_size, 0, len(segments))
        reference, reference_size, _ = _reference(source, source_layout, work)
        _room(output, video_size, reference_size, len(segments))
        _attachments(source, source_layout, attached)
        # As bytes, the file system's: a name the locale's encoding can't hold is no error.
        listed.write_bytes(os.fsencode(concat_list(segments, frame_rate, firsts)))
        command = _join(
            *(source, source_layout, listed, shared, tags, attached),
            *(snap(frame_rate, first, total), partial, read),
        )
        _strict(command, f"{output}: joining its segments", _pictures(source_layout))
        joined = time.monotonic()
        ahead = _check(
            *(partial, segments, grid(total, frame_rate, first), read, shared),
            *(source_layout, reference, work, attached),
        )
        size = partial.stat().st_size
        replace_whole(partial, output, stale=stale)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        for path in (listed, read, others, work / PAIR, work / PAIR_LIST, *attached.values()):
            path.unlink(missing_ok=True)
        for path in work.glob(f"{PICTURE}*"):
            path.unlink(missing_ok=True)
    done = time.monotonic()
    result = Assembled(
        *(total, size, len(source_layout.streams), len(source_layout.chapters)),
        *(joined - began, done - joined, ahead / 1000),
    )
    logger.info(
        "%s: assembled, %s of %s with %s%s and %s of %s,"
        " %.2f GiB: joined in %.1f s, checked by its packets in %.1f s; the other streams at"
        " most %.2f s ahead of the video in the file",
        output,
        _counted(total, "frame"),
        _counted(len(segments), "segment"),
        _counted(len(source_layout.streams) - 1, "other stream"),
        (
            f" (its data streams {', '.join(str(stream.index) for stream in gone)} left out)"
            if gone
            else ""
        ),
        _counted(len(source_layout.chapters), "chapter"),
        source,
        size / 2**30,
        result.joined,
        result.checked,
        result.ahead,
    )
    if result.ahead > INTERLEAVED:
        logger.warning(
            "%s: poorly interleaved: another stream's packets run %.1f s ahead of the video's in"
            " the file, as when a stream of the source starts late, ends early or has a gap: the"
            " file is whole, every packet checked, but a player reading it from a slow disk may"
            " stutter there",
            output,
            result.ahead,
        )
    return result


def _written(streams: Sequence[Stream]) -> list[Stream]:
    """The streams of a Matroska file, in the order a read of it gives them: its tracks first,
    then its attachments, each in the order attached (libavformat/matroskadec.c reads the
    tracks, then the attachments). The source's own attachments, fonts, come first, then its
    cover pictures (Stream.picture): ffmpeg's muxer writes a stream of the attachment type as
    an attachment, and a cover that is a video stream to ffmpeg as a track of one frame
    (matroskaenc.c, mkv_write_track), which is why a cover is attached from its file instead
    (_pictured, _join)."""
    tracks = [stream for stream in streams if stream.kind != "attachment"]
    fonts = [stream for stream in streams if stream.kind == "attachment" and not stream.picture]
    return tracks + fonts + [stream for stream in streams if stream.picture]


def _others(source: Path, found: Layout) -> tuple[Stream, ...]:
    """Every stream of the source but its first video stream; MediaError without one."""
    if not any(stream.kind == "video" for stream in found.streams):
        raise MediaError(f"{source}: no video stream")
    return found.others


def _signatures(
    segments: Sequence[tuple[Path, int]], frame_rate: Fraction, work: Path
) -> tuple[Signature, Mapping[str, str], list[int | None]]:
    """The signature every segment shares, the tags of the first one's video stream, and, for
    each segment, its first timestamp in milliseconds when the file doesn't declare it as its
    start time, else None (concat_list).

    Raises MediaError for a segment that can't follow the first in one stream (joins: frame_rate
    and work are its own), or whose time base isn't the millisecond the snap computes in
    (snap)."""
    shared, tags, first, start_time = _segment(segments[0][0])
    if shared.time_base != MILLISECOND:
        raise MediaError(
            f"{segments[0][0]}: its video stream's time base is {shared.time_base} s, where a"
            f" Matroska file's is {MILLISECOND}: not assembled"
        )
    firsts = [_undeclared(first, start_time)]
    for path, frames in segments[1:]:
        found, _, first, start_time = _segment(path)
        differing = _unlike(shared, found, segments[0], (path, frames), frame_rate, work)
        if differing:
            # PROVISIONAL (implementation, 2026-10-10): refused. The concat demuxer gives every
            # packet the first file's codec headers: other headers, and the stream no longer
            # decodes, unless they decode it all the same, which is asked (_unlike). Its own
            # remedy is for H.264 alone (auto_convert, the headers in every
            # keyframe), and changes the packets' data. Not met: 160 segments of 1 to 30 frames
            # each from one command had one set of headers, with libx264, libx265, libsvtav1,
            # libvpx-vp9, mpeg4 and ffv1.
            raise MediaError(
                f"{path}: can't be joined to {segments[0][0]} by stream copy: {differing}:"
                " every segment must come from the same encoder with the same settings"
            )
        firsts.append(_undeclared(first, start_time))
    return shared, tags, firsts


def _undeclared(first: int | None, start_time: int | None) -> int | None:
    """A segment's first timestamp, in milliseconds, unless the file declares it as its start
    time, in microseconds: None then, and for a first packet without a timestamp."""
    return None if first is None or start_time == first * 1000 else first


def _segment(path: Path) -> tuple[Signature, Mapping[str, str], int | None, int | None]:
    """The signature of the segment at path, its video stream's tags, its first video packet's
    pts in the stream's time base, None without one, and the file's start time in microseconds,
    None without one. Its header and one packet read: 8 ms."""
    entries = (
        "stream=index,codec_name,width,height,pix_fmt,time_base,extradata_hash:stream_tags"
        ":format=start_time:packet=pts"
    )
    probed = _ffprobe(
        path, "-select_streams", "v:0", "-show_entries", entries, "-read_intervals", "%+#1"
    )
    streams = cast(list[dict[str, Any]], probed.get("streams") or [])
    if not streams:
        raise MediaError(f"{path}: no video stream")
    stream = streams[0]
    found = Signature(
        int(stream["index"]),
        str(stream.get("codec_name", "")),
        int(stream.get("width", 0)),
        int(stream.get("height", 0)),
        str(stream.get("pix_fmt", "")),
        Fraction(str(stream.get("time_base", "0/1"))),
        str(stream.get("extradata_hash", "")),
    )
    tags = cast(dict[str, Any], stream.get("tags") or {})
    read = cast(list[dict[str, Any]], probed.get("packets") or [])
    first = int(read[0]["pts"]) if read and "pts" in read[0] else None
    declared = str(cast(dict[str, Any], probed.get("format") or {}).get("start_time", "N/A"))
    start_time = None if declared == "N/A" else round(Fraction(declared) / MICROSECOND)
    return found, {str(key): str(value) for key, value in tags.items()}, first, start_time


def _room(output: Path, video: int, others: int, segments: int) -> None:
    """Raise MediaError unless the directory output goes to has room for it: the segments' bytes
    again, `video`, and the other streams' as their remux holds them, `others`. The file is no
    larger: one header for the segments' many (tests/test_assembly.py holds it under that sum)."""
    directory = output.parent
    try:
        stats = os.statvfs(directory)
    except OSError as error:
        raise MediaError(f"{directory}: its free space unreadable: {error}") from error
    free = stats.f_bavail * stats.f_frsize
    needed = video + others
    if free < needed:
        raise MediaError(
            f"{output}: {_gib(needed - free)} more free space needed in {directory} to assemble"
            f" it: {_gib(needed)} for the file, its {segments} segments' {_gib(video)} and"
            f" {_gib(others)} of the source's other streams, where {_gib(free)} is free; the"
            " segments are kept, and the same command assembles them once there is room"
        )


def _gib(size: int) -> str:
    return f"{size / 2**30:.2f} GiB"


def _reference(
    source: Path, found: Layout, work: Path
) -> tuple[dict[int, _Packets], int, dict[int, Fraction]]:
    """The packets of the source's other streams as ffmpeg's remux into Matroska holds them, by
    the stream's index in the source, that remux's size in bytes, and where it has each stream
    start, as ffprobe declares it of the remux, in seconds (moved): the other streams alone,
    copied to a file in work, read back, and removed. Nothing, 0 and nothing for a source whose
    other streams hold no packet (none, or attachments alone). Without the source's cover
    pictures, which the finished file attaches (_pictured): the remux would make them video
    tracks of one frame (DESIGN.md, Output, Assembly, decided on 2026-10-28).

    To a file, not a pipe: Matroska's muxer completes a header at the end, ADTS AAC's from
    MPEG-TS among them, and refuses the stream on an output it can't seek (unmuxable).

    Raises MediaError for anything ffmpeg prints (TOLERATED apart), or a failure."""
    streams = _others(source, found)
    if not any(stream.kind != "attachment" for stream in streams):
        return {}, 0, {}
    streams = tuple(stream for stream in streams if not stream.picture)
    remux = work / OTHERS
    maps = [word for stream in streams for word in ("-map", f"0:{stream.index}")]
    command = [*FFMPEG, "-i", str(source), *maps, "-c", "copy", "-f", "matroska", str(remux)]
    try:
        _strict(command, f"{source}: remuxing its other streams into Matroska", _pictures(found))
        packets, _ = _read_packets(remux, None)
        size = remux.stat().st_size
        held = layout(remux).streams
    finally:
        remux.unlink(missing_ok=True)
    written = _written(streams)
    # Not an attachment's, which has no packet: ffprobe declares it the file's own start time.
    starts = {
        stream.index: start
        for stream, start in zip(written, (stream.start for stream in held), strict=False)
        if start is not None and stream.kind != "attachment"
    }
    return (
        {stream.index: packets.get(place, _Packets()) for place, stream in enumerate(written)},
        size,
        starts,
    )


def _attachments(source: Path, found: Layout, attached: Mapping[int, Path]) -> None:
    """Write each attachment of the source to its file, `attached` by the stream's index, `found`
    its layout (_pictured). A font: ffmpeg's -dump_attachment, an option of the input, which
    wants an output all the same ("At least one output file must be specified", exit 1, the
    files written): the first video packet, copied nowhere. A cover picture, a stream of one
    packet to ffmpeg: that packet, whole, copied to a file by the data muxer, the bytes of the
    attachment (a Matroska file's, compared with mkvextract's by hand on 2026-10-28) or of the
    MP4's covr atom. Nothing to do without attachments.

    Raises MediaError for anything ffmpeg prints (TOLERATED and NO_RATE's cover pictures
    apart), or a failure."""
    if not attached:
        return
    pictures = _pictures(found)
    dumps = [
        word
        for index, path in attached.items()
        if index not in pictures
        for word in (f"-dump_attachment:{index}", str(path))
    ]
    copies = [
        word
        for index, path in attached.items()
        if index in pictures
        for word in ("-map", f"0:{index}", "-c", "copy", "-f", "data", str(path))
    ]
    command = [
        *(*FFMPEG, *dumps, "-i", str(source), "-map", "0:v:0", "-c", "copy"),
        *("-frames:v", "1", "-f", "null", "-", *copies),
    ]
    _strict(command, f"{source}: reading its attachments", pictures)


def _picture(path: Path, index: int, copy: Path) -> None:
    """Write the picture that is stream `index` of the finished file `path`, attached, to `copy`:
    its one packet's bytes (_attachments).

    Raises MediaError for anything ffmpeg prints, or a failure."""
    command = [
        *(*FFMPEG, "-i", str(path), "-map", f"0:{index}", "-c", "copy", "-f", "data", str(copy)),
    ]
    _strict(command, f"{path}: reading back its attached picture {index}")


def _join(
    source: Path,
    found: Layout,
    listed: Path,
    shared: Signature,
    tags: Mapping[str, str],
    attached: Mapping[int, Path],
    snapping: str,
    partial: Path,
    read: Path,
) -> list[str]:
    """The join's ffmpeg command: the source, input 0, the segments through the concat demuxer,
    input 1, and the source once more, input 2, into the Matroska file partial, and the hashes of
    the segments' video packets, as the join reads them, into the file read (HASH).

    - The source first: fftools copies the first input's global metadata by default, less its
      creation time and three tags of its maker (fftools/ffmpeg_mux_init.c, copy_meta), and the
      chapters of the first input that has any, shifted as its timestamps are (copy_chapters):
      what its remux of the source does. Named, -map_metadata 1 would copy those four tags too.
    - -auto_convert 0: the concat demuxer otherwise rewrites H.264 packets with their headers in
      every keyframe (libavformat/concatdec.c, detect_stream_specific), and the file's packets
      are no longer the segments'.
    - The source's streams are mapped apart, by what they give the scheduler, which lets an input
      read on only while one of its streams is the furthest behind, and, while a stream has given
      no packet yet, only the inputs of such streams (fftools/ffmpeg_sched.c, trailing_dts,
      schedule_update_locked). The audio streams come from input 0. The others that hold packets,
      subtitles, a cover, come from input 2, a second opening of the source: a subtitle stream is
      behind between two of its lines, and from one input with the audio it kept that input
      reading while the segments' waited, the audio 1.3 s ahead of the video in the file before
      a first line at 1 s. An attachment gives no packet, ever: mapped from the source, it kept
      the segments waiting until the source's end, libavformat writing the audio every 10 s it
      held (max_interleave_delta), 10.4 s ahead, and ffprobe then found no pixel format for a
      video whose first packet came 279 kB into the file. So each attachment is attached from
      its file (-attach, _attachments), a stream no input feeds, with the source's tags for it,
      its name and MIME type among them, which Matroska's muxer requires. With both: 0.10 s, as
      ffmpeg's own remux of the source. Without audio, the other streams come from input 0.
    - A cover picture is an attachment too (_pictured), attached after the source's own, from
      the file its one packet was copied to: mapped from the source, ffmpeg's muxer would make
      it a track of one frame (DESIGN.md, Output, Assembly, decided on 2026-10-28).
    - The streams are mapped in the order a read of the file gives them back (_written): the
      source's, the video in its video's place.
    - Each other stream's own tags and dispositions are copied by default. The video's are the
      first segment's: its tags are removed, each by name, but its encoder's (KEPT), and the
      source's video's language and title set (VIDEO_TAGS); its dispositions are the source's
      video's, given by name. A disposition given also keeps fftools from marking as default the
      first stream of a type that has none (ffmpeg_mux_init.c, set_dispositions), which a plain
      remux does: the other streams keep theirs as the source has them.
    - -f matroska: the partial file's name doesn't end in .mkv."""
    video = found.video
    written = _written(found.streams)
    tracks = [stream for stream in written if stream.kind != "attachment"]
    apart = any(stream.kind == "audio" for stream in tracks)
    sparse = any(stream.kind != "audio" and stream is not video for stream in tracks)
    inputs = [
        *("-i", str(source)),
        *("-f", "concat", "-safe", "0", "-auto_convert", "0", "-i", str(listed)),
        *(("-i", str(source)) if apart and sparse else ()),
    ]
    maps: list[str] = []
    for stream in tracks:
        if stream is video:
            maps += ["-map", f"1:{shared.index}"]
        else:
            opening = 2 if apart and stream.kind != "audio" else 0
            maps += ["-map", f"{opening}:{stream.index}"]
    place = tracks.index(video)
    records: list[str] = []
    for key in tags:
        # A key holding "=" can't be named to -metadata, which parts key and value at the first.
        if key.upper() != KEPT and key.lower() not in VIDEO_TAGS and "=" not in key:
            records += [f"-metadata:s:{place}", f"{key}="]
    for key in VIDEO_TAGS:
        records += [f"-metadata:s:{place}", f"{key}={video.tags.get(key, '')}"]
    kept = [name for name in video.dispositions if name not in NOT_A_VIDEOS]
    records += [f"-disposition:{place}", "+".join(kept) or "0"]
    attachments: list[str] = []
    for position, stream in enumerate(written[len(tracks) :], len(tracks)):
        attachments += ["-attach", str(attached[stream.index])]
        for key, value in stream.tags.items():
            if "=" not in key:
                attachments += [f"-metadata:s:{position}", f"{key}={value}"]
    return [
        *(*FFMPEG, *inputs, *maps, "-c", "copy", *attachments, *records),
        *(f"-bsf:{place}", snapping, "-f", "matroska", str(partial)),
        *("-map", f"1:{shared.index}", "-c", "copy", "-f", "framehash", "-hash", HASH, str(read)),
    ]


def _check(
    partial: Path,
    segments: Sequence[tuple[Path, int]],
    places: npt.NDArray[np.int64],
    read: Path,
    shared: Signature,
    source: Layout,
    reference: Mapping[int, _Packets],
    work: Path,
    attached: Mapping[int, Path],
) -> int:
    """Check the file the join wrote, by a read of its packets (assemble): MediaError naming the
    stream and the first packet that differs. Returns how far another stream's packets run
    ahead of the video's in the file's order, at most, in milliseconds (_read_packets). `work`
    and `attached`: the work directory, and the source's attachments written there, by their
    stream's index in the source (_attachments), which the file's cover pictures are read back
    against."""
    found = layout(partial)
    if len(found.streams) != len(source.streams):
        raise MediaError(
            f"{partial}: {len(found.streams)} streams, where the source has {len(source.streams)}"
        )
    written = _written(source.streams)
    place = written.index(source.video)
    packets, ahead = _read_packets(partial, place)
    _check_video(partial, segments, places, _read_hashes(read), packets.get(place, _Packets()))
    video = found.streams[place]
    if (video.codec, video.extradata) != (shared.codec, shared.extradata):
        raise MediaError(
            f"{partial}: its video stream is {video.codec} with codec headers"
            f" {video.extradata or 'none'}, where the segments are {shared.codec} with"
            f" {shared.extradata or 'none'}"
        )
    for position, stream in enumerate(written):
        if position == place:
            continue
        said = (
            f"{partial}: stream {position} (the source's stream {stream.index}, {stream.kind},"
            f" {stream.codec or 'no codec'})"
        )
        held = found.streams[position]
        if stream.picture:
            _check_picture(said, stream, held, partial, position, work, attached[stream.index])
            continue
        if held.kind != stream.kind:
            raise MediaError(f"{said}: {held.kind} in the file")
        if stream.kind == "attachment":
            if held.extradata != stream.extradata:
                raise MediaError(f"{said}: not the source's attachment, by its content's hash")
            continue
        _check_other(said, reference.get(stream.index, _Packets()), packets.get(position))
    _check_chapters(partial, source, found)
    return ahead


def _check_picture(
    said: str, stream: Stream, held: Stream, partial: Path, position: int, work: Path, picture: Path
) -> None:
    """A cover picture of the source (Stream.picture) against what the file holds at `position`:
    an attachment, which ffmpeg's demuxer shows as a video stream with the disposition
    attached_pic, its file name and type the source's (`stream`'s tags, _pictured), its bytes
    those of `picture`, the source's, read back from the file. MediaError otherwise
    (DESIGN.md, Output, Assembly, decided on 2026-10-28: "checked against the source's")."""
    if held.kind != "video" or "attached_pic" not in held.dispositions:
        raise MediaError(
            f"{said}: {held.kind} in the file, where the picture is attached (ffprobe shows an"
            " attached picture as a video stream with the disposition attached_pic)"
        )
    for key in ("filename", "mimetype"):
        if held.tags.get(key) != stream.tags[key]:
            raise MediaError(
                f"{said}: attached with the {key} {held.tags.get(key)!r}, where the source's is"
                f" {stream.tags[key]!r}"
            )
    copy = work / f"{PICTURE}{position}"
    _picture(partial, position, copy)
    if not filecmp.cmp(picture, copy, shallow=False):
        raise MediaError(f"{said}: not the source's picture, by its bytes")


def _check_video(
    partial: Path,
    segments: Sequence[tuple[Path, int]],
    places: npt.NDArray[np.int64],
    read: _Packets,
    written: _Packets,
) -> None:
    """The file's video packets against those the join read from the segments, and against the
    frame grid, `places`."""
    total = len(places)
    if len(read.pts) != total:
        raise MediaError(
            f"{partial}: ffmpeg read {len(read.pts)} video packets from the segments, which"
            f" hold {total} frames"
        )
    if len(written.pts) != total:
        raise MediaError(
            f"{partial}: {len(written.pts)} video packets in the file, where the segments hold"
            f" {total} frames"
        )
    begins = 0
    for path, frames in segments:
        ends = begins + frames
        for packet in range(begins, ends):
            if (written.size[packet], written.hash[packet]) != (
                read.size[packet],
                read.hash[packet],
            ):
                raise MediaError(
                    f"{partial}: its video packet {packet} isn't the one read from the segments,"
                    f" packet {packet - begins} of {path}: {written.size[packet]} bytes, hash"
                    f" {written.hash[packet]}, where the segment's is {read.size[packet]} bytes,"
                    f" hash {read.hash[packet]}"
                )
        # The segment's packets in its own presentation order, by the timestamps the join read:
        # the frames of its places on the grid, one each.
        order = np.argsort(np.array(read.pts[begins:ends], dtype=np.int64), kind="stable")
        placed = np.array(written.pts[begins:ends], dtype=np.int64)[order]
        off = np.flatnonzero(placed != places[begins:ends])
        if len(off):
            frame = begins + int(off[0])
            raise MediaError(
                f"{partial}: its video frame {frame}, frame {int(off[0])} of {path}, is at"
                f" {int(placed[off[0]])} ms, where the frame grid has it at"
                f" {int(places[frame])} ms"
            )
        begins = ends


def _check_other(said: str, expected: _Packets, written: _Packets | None) -> None:
    """A stream of the file against the same stream of the remux of the source's other
    streams."""
    found = written or _Packets()
    for packet, (ours, theirs) in enumerate(zip(found.digests, expected.digests, strict=False)):
        if ours != theirs:
            raise MediaError(
                f"{said}: its packet {packet} isn't the one a remux of the source holds, by its"
                " timestamps, duration, size, flags or data hash"
            )
    if len(found) != len(expected):
        raise MediaError(
            f"{said}: {len(found)} packets, where a remux of the source holds {len(expected)}"
        )


def _check_chapters(partial: Path, source: Layout, found: Layout) -> None:
    """The file's chapters against the source's, as fftools copies them (ffmpeg_mux_init.c,
    copy_chapters): the source's start time, rounded to the chapter's time base, taken off both
    ends, a chapter ending before it left out, a start before it held at 0; in nanoseconds,
    Matroska's unit (matroskaenc.c, mkv_write_chapters)."""
    expected: list[tuple[int, int]] = []
    for chapter in source.chapters:
        shift = _rescale(source.start_time or 0, MICROSECOND, chapter.time_base)
        if chapter.end < shift:
            continue
        begins = _rescale(max(0, chapter.start - shift), chapter.time_base, NANOSECOND)
        expected.append((begins, _rescale(chapter.end - shift, chapter.time_base, NANOSECOND)))
    written = [
        (
            _rescale(chapter.start, chapter.time_base, NANOSECOND),
            _rescale(chapter.end, chapter.time_base, NANOSECOND),
        )
        for chapter in found.chapters
    ]
    if len(written) != len(expected):
        raise MediaError(
            f"{partial}: {len(written)} chapters, where the source's are {len(expected)}"
        )
    for number, (ours, theirs) in enumerate(zip(written, expected, strict=True)):
        if ours != theirs:
            raise MediaError(
                f"{partial}: its chapter {number} runs from {ours[0]} to {ours[1]} ns, where the"
                f" source's, less its start time, runs from {theirs[0]} to {theirs[1]} ns"
            )


def _rescale(value: int, source: Fraction, target: Fraction) -> int:
    """value units of `source` seconds in units of `target`, rounded to the nearest, halves away
    from zero: ffmpeg's av_rescale_q (libavutil/mathematics.c, AV_ROUND_NEAR_INF)."""
    scaled = abs(value) * source / target
    rounded = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    return rounded if value >= 0 else -rounded


def _read_hashes(read: Path) -> _Packets:
    """The packets the join read from the segments, from its framehash output (libavformat/
    hashenc.c: stream, dts, pts, duration, size, hash, after # comments)."""
    packets = _Packets()
    try:
        text = read.read_text()
    except OSError as error:
        raise MediaError(f"{read}: the join's hashes unreadable: {error}") from error
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise MediaError(f"{read}: not ffmpeg's framehash: {line[:80]!r}")
        packets.pts.append(int(fields[2]))
        packets.size.append(int(fields[4]))
        packets.hash.append(fields[5])
    return packets


def _read_packets(path: Path, video: int | None) -> tuple[dict[int, _Packets], int]:
    """The packets of every stream of the Matroska file at path, by the stream's index, read by
    ffprobe, no decode, each with its data hash (HASH): the stream `video`'s with their pts,
    size and hash, the others' as digests of their whole records (_Packets). And how far the
    other streams' packets run ahead of the video's in the file's order, at most: a packet's pts
    less that of the last video packet before it, or of the first one for those before any; in
    milliseconds, Matroska's unit for every stream; 0 without `video`. Not those after the
    last video packet: a stream that lasts longer than the video isn't ahead of it.

    Raises MediaError when ffprobe fails or prints anything, or a video packet has no pts."""
    command = [
        *("ffprobe", "-v", "error", "-show_entries"),
        "packet=stream_index,pts,dts,duration,size,flags,data_hash",
        *("-show_data_hash", HASH, "-of", "csv=p=0", str(path)),
    ]
    packets: dict[int, _Packets] = {}
    prefix = f"{HASH}:"
    ahead = 0
    shown: int | None = None  # the last video packet's pts
    early: int | None = None  # the highest pts of the other streams before any video packet
    since = 0  # the largest lead since the last video packet, counted at the next one
    with tempfile.TemporaryFile() as log:
        # In a process group of its own, as the writers' ffmpeg (media/writer.py).
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            process_group=0,
            preexec_fn=ends_with_us(),
        )
        assert process.stdout is not None
        try:
            with process.stdout as lines:
                for line in lines:
                    # stream_index, pts, dts, duration, size, flags, data_hash, ffprobe's order
                    # whatever the entries', then an empty field for a packet with side data.
                    fields = line.rstrip("\n").split(",")
                    if len(fields) < 7:
                        raise MediaError(f"{path}: not ffprobe's packets: {line[:80]!r}")
                    found = packets.setdefault(int(fields[0]), _Packets())
                    if int(fields[0]) == video:
                        if fields[1] == "N/A":
                            raise MediaError(
                                f"{path}: its video packet {len(found.pts)} has no timestamp"
                            )
                        if shown is None and early is not None:
                            since = early - int(fields[1])
                        ahead, since = max(ahead, since), 0
                        shown = int(fields[1])
                        found.pts.append(shown)
                        found.size.append(int(fields[4]))
                        found.hash.append(fields[6].removeprefix(prefix))
                    else:
                        record = ",".join(fields[1:7])
                        digest = hashlib.blake2b(record.encode(), digest_size=8).digest()
                        found.digests.append(int.from_bytes(digest, "little"))
                        found.last = record
                        if video is not None and fields[1] != "N/A":
                            if shown is None:
                                early = max(int(fields[1]), early or int(fields[1]))
                            else:
                                since = max(since, int(fields[1]) - shown)
            status = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        said = _said(log)
    if status != 0 or said:
        raise MediaError(
            f"{path}: ffprobe failed reading its packets: {' / '.join(said[-5:]) or status}"
        )
    return packets, ahead


def _ffprobe(path: Path, *options: str) -> dict[str, Any]:
    """ffprobe's JSON for path with `options`, each stream's extradata hashed (HASH)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", *options, "-show_data_hash", HASH, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
        process_group=0,
        preexec_fn=ends_with_us(),
    )
    if result.returncode != 0:
        raise MediaError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    return cast(dict[str, Any], json.loads(result.stdout))


def _strict(command: Sequence[str], doing: str, pictures: Collection[int] = ()) -> None:
    """Run an ffmpeg command to its end, strictly: MediaError for a failure, or for anything it
    prints (_complaints, `pictures` the indexes of the source's cover pictures), `doing` saying
    what it was at."""
    status, lines = _run(command)
    complaints = _complaints(lines, pictures)
    if status != 0 or complaints:
        said = " / ".join((complaints or lines)[-5:])
        raise MediaError(
            f"{doing} with ffmpeg: "
            + (f"exit status {status}" if status != 0 else "it reported")
            + (f": {said}" if said else "")
        )


def _run(command: Sequence[str]) -> tuple[int, list[str]]:
    """Run an ffmpeg command to its end: its exit status and the lines it printed. In a process
    group of its own, out of reach of the terminal's Ctrl-C (runtime/stop.py); killed when the
    caller's thread raises, a stop among them; and when seedvr2x itself is killed
    (ffmpeg.ends_with_us)."""
    # ffmpeg's messages go to a file, which it never waits on (media/checksums.py).
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            process_group=0,
            preexec_fn=ends_with_us(),
        )
        try:
            status = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        return status, _said(log)


def _said(log: IO[bytes]) -> list[str]:
    """The lines written to log, without the empty ones."""
    log.seek(0)
    lines = (line.decode(errors="replace").strip() for line in log)
    return [line for line in lines if line]


def _complaints(lines: Sequence[str], pictures: Collection[int] = ()) -> list[str]:
    """The lines ffmpeg printed that fail a strict run: every one but those of TOLERATED, each
    matched whole, and NO_RATE's note for a stream that is one of the source's cover pictures
    (`pictures`, their indexes in the source), whichever input of the command opens the source:
    the same note for any other stream stays a complaint."""

    def known(line: str) -> bool:
        if any(pattern.fullmatch(line) for pattern in TOLERATED):
            return True
        noted = NO_RATE.fullmatch(line)
        return noted is not None and int(noted.group(2)) in pictures

    return [line for line in lines if not known(line)]
