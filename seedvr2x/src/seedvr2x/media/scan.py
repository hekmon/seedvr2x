"""The first pass over a source: every frame decoded, counted and timed, before any GPU work, the
frame index made meanwhile (DESIGN.md, Input; media/index.py), and ffmpeg's idet run on every
frame (Idet). Automatic scene detection will join it."""

import logging
import math
import re
import subprocess
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, cast

import numpy as np

from seedvr2x.media.ffmpeg import NO_METADATA, MediaError, input_args
from seedvr2x.media.index import NOPTS, FrameIndex

logger = logging.getLogger(__name__)

# The decoder's line for each frame it gives, which -debug_ts adds (fftools/ffmpeg_dec.c:415-428 at
# n9.0.2): its pts as the filters get it, the decoder's best effort, extrapolated when it has none
# (:398-411), and the time base of that pts, the stream's. Counted, each tying the report before
# it to its frame (CORRUPT); the frames' pts are the hashes' (HASHED), this one only saying where
# a timestamp that went back went (_discontinuity). Found by its message alone, never its
# "[context] [level]" prefix, which ffmpeg leaves out when the message before ended without a
# newline, whatever thread printed it (libavutil/log.c:380-439, its print_prefix one for all):
# the muxer reports a pts going back in two pieces (fftools/ffmpeg_mux.c:180-189), and a decoder's
# line then came glued to the first, "... current: 5536800; decoder -> pts:5572800 ...", in 2
# first passes of 6 over MPEG-TS files joined, their timestamps going back 40 s (2026-10-10). An
# output's header, which av_dump_format prints in pieces, does it too, "[info]   Stream
# #1decoder -> pts:167 ...": 14 first passes of 100 lost a decoder line's prefix with idet's
# output (scan), 0 of 100 without it (2026-10-09). The patterns below take the message wherever
# it starts, in the lines ffmpeg prints once it decodes (MAPPING).
DECODED = re.compile(r"decoder -> pts:(-?\d+|NOPTS) .* time_base:(\d+)/(\d+)$")
# fftools' report of a frame whose decoder flagged it: decode_error_flags set (H.264's slices or
# its concealment, libavcodec/h264dec.c:752-764, 804-812; MPEG-2's concealment,
# error_resilience.c:1127-1130) or AV_FRAME_FLAG_CORRUPT (a reference missing: hevc/refs.c:507-519;
# H.264 before its recovery point, h264_slice.c:1376-1384). It is a warning, logged by the
# decoder's thread right before that frame's own DECODED line (ffmpeg_dec.c:767-772, then
# video_frame_process at :786): the two are written in order by one thread, whatever the others
# print between them, so each report is tied to its frame exactly. By its message alone, as
# DECODED.
CORRUPT = "corrupt decoded frame"
# A message of ffmpeg's at the error level or above, by its level, which -loglevel level prints.
# One glued after a message that ended without a newline comes without it, as DECODED's, and no
# pattern finds a level that isn't printed: what such an error does is then seen where it shows,
# not in its line: ffmpeg's exit status (scan, which then quotes its last lines, _Decoded.last),
# a frame's CORRUPT report, the counts of the frames decoded and hashed.
ERROR = re.compile(r"\[(error|fatal|panic)\]")
# The same at the start of its line, behind its contexts alone, its own and its parent's when it
# has one, each "[name @ 0x...] " (libavutil/log.c:333-345 at n9.0.2: "[mpeg2video @ 0x...]
# [IMGUTILS @ 0x...] [error] Picture size 0x0 is invalid"): what is taken for an error before
# MAPPING, where a line may hold the source's own text behind "[info]".
ERROR_AT_START = re.compile(r"(\[[^\]]* @ 0x[0-9a-f]+\] )*\[(error|fatal|panic)\] ")
# The line ffmpeg prints once every input is opened and dumped, and before its scheduler starts a
# thread (fftools/ffmpeg.c:734-736, 892-896 at n9.0.2, print_stream_maps then sch_start; an
# input's dump at ffmpeg_demux.c:2340), whole, with its level. Before it, ffmpeg prints the
# source's own text at info level. Its metadata and chapters' titles, a line break in one begun
# again behind an indent alone, without "[info]" (libavformat/dump.c:147-165), which a pattern
# matching a message wherever it starts would take for the decoder's or idet's: a title "decoder
# -> pts:7 ... time_base:1/0" counted a frame, and raised in the thread reading stderr, ffmpeg
# then blocked on it for good (a review's finding, 2026-10-09). And some of it raw, a line break
# in it beginning a line of the text's own making: a stream's language (dump.c:638), a program's
# name (:920), a tag's key (:155-156). A language "x\n[info] Stream mapping:\n" printed this very
# line inside its input's dump, and the title after it was read as the decoder's lines, and as
# idet's (a review's finding, 2026-10-10). So nothing before it is read but errors
# (ERROR_AT_START), and each such line begins the reading again (_Decoded.read): fftools prints
# it once (ffmpeg.c:736), after every input's dump, so that the last one read is its own. Nothing
# after it holds the source's text: no output takes its metadata (ffmpeg.NO_METADATA), a
# stream's language among it; and a name with a line break, which ffmpeg prints as it is in its
# input's dump, is refused all the same, before anything is done with the file (name_error).
# The all-zero summary of the idet graph fftools parses first comes before it too.
# A known limit: before it an error is taken at its line's start, and the text printed raw can
# begin a line with one, a language "x\n[error] made up\n" giving the warning "ffmpeg reported
# errors decoding it: [error] made up" (a review's finding, 2026-10-10). A source's text can so
# put words in the errors quoted, by that warning or by a failure's message, and never a frame,
# a flag or a count, which are read after this line alone.
MAPPING = "[info] Stream mapping:"
# The frame hashes' lines (libavformat/hashenc.c:282-317): stream, dts, pts, duration, size, hash.
# Each frame's pts is read here, on stdout, where ffmpeg writes nothing else: no line of stderr,
# which prints the source's metadata, its titles among it, can give a frame a pts or take one
# away. It is the decoder's, in the stream's time base: the encoder takes the demuxer's
# (-enc_time_base:v demux: fftools/ffmpeg_filter.c:2379-2389 at n9.0.2) and the muxer's stream
# the encoder's (ffmpeg_mux.c:613-616), which the header says (TIME_BASE); a frame without one
# would come as AV_NOPTS_VALUE's own number, index.NOPTS (hashenc.c:290-291). But for a pts under
# the frame before's, which fftools' muxer holds at that one's ("Non-monotonic DTS",
# ffmpeg_mux.c:171-196; an equal one passes, the muxer being AVFMT_TS_NONSTRICT,
# hashenc.c:339-340): no source the timing rule accepts has one, every frame lasting within 1 ms
# of the others (timing_error), and the decoder's line says where it went (DECODED).
HASHED = re.compile(r"0, *-?\d+, *(-?\d+), *-?\d+, *\d+, ([0-9a-f]{8})\b")
# The hashes' header line saying the stream's time base (libavformat/framehash.c:35).
TIME_BASE = re.compile(r"#tb 0: (\d+)/(\d+)$")
# idet's counts, printed at info level when its filter is freed (libavfilter/vf_idet.c:358-381 at
# n9.0.2), three lines, each count %6d: [Parsed_idet_0 @ 0x...] [info] Multi frame detection: TFF:
# 0 BFF: 0 Progressive: 240 Undetermined: 1. fftools frees a graph it parses to learn its inputs
# before any frame (an all-zero summary first), and one it rebuilds when the frames' parameters
# change (a summary of the frames before): the summaries are summed (Idet). By the message alone,
# as DECODED: only idet prints these.
IDET = re.compile(r"(Repeated Fields|Single frame detection|Multi frame detection): (.*)$")
IDET_COUNT = re.compile(r"(\w+): *(\d+)")
# Each line's counts, by their names in the line: Idet's fields.
IDET_LINES = {
    "Repeated Fields": ("repeated", ("Neither", "Top", "Bottom")),
    "Single frame detection": ("single", ("TFF", "BFF", "Progressive", "Undetermined")),
    "Multi frame detection": ("multiple", ("TFF", "BFF", "Progressive", "Undetermined")),
}

# How much the time between two frames may vary in a constant frame rate stream: sptenc's
# tolerance (ffmpeg/probe.go, IsConstantFrameRate). Matroska rounds timestamps to the
# millisecond, so 23.976 fps lasts 41 and 42 ms; 2 ms would let a mix of 24 and 25 fps through.
TOLERANCE_US = 1000

# A jump in the timestamps, which -copyts keeps. Without it, ffmpeg takes a jump out of a format
# whose timestamps may restart (AVFMT_TS_DISCONT), a packet more than 10 s from where the one
# before it ended (dts_delta_threshold, fftools/ffmpeg_opt.c:56 at n9.0.2) or more than 0.1 s
# before it (ffmpeg_demux.c:228-256), shifting every timestamp after: the first pass before the
# frame index read such a file as continuous, and with -copyts a jump is corrected no more
# (:234). Two MPEG-TS files joined, their timestamps going from 3.44 to 1,001.4 s, were then
# refused as a variable frame rate, "frames last from 40 to 997960 ms". PROVISIONAL
# (implementation, 2026-10-10, a question for design: DESIGN.md's Input has the 1 ms rule alone):
# a jump past ffmpeg's bounds, in seconds, is refused as what it is, naming the frame and both
# times (Discontinuity), with ffmpeg's own correction as the way out, a remux, for the formats it
# corrects (DISCONTINUOUS); a smaller one stays a variable frame rate. A step on is a jump only
# when it is also longer than the frames' shortest step by more than the timing rule's tolerance
# (TOLERANCE_US), the step that rule refuses as no constant rate's: at a constant rate under 0.1
# fps every step is past ffmpeg's bound, and files at 1/15 and 1/20 fps, read before, were
# refused as jumping "at frame 1, from 0.000 to 15.000 s" (a review's finding, 2026-10-10).
# And none is when the shortest step on is itself past the bound, a rate under 0.1 fps: what
# such a file has is a variable frame rate, the timing rule's to refuse, as before the frame
# index, and ffmpeg's remux is no way out of it. An MPEG-TS file of a frame every 15 s, one
# step 2 ms longer, was refused as jumping "at frame 3, from 31.400 to 46.402 s", and the remux
# it was given left its six frames at 0, 15, 15.04, 15.082, 15.12 and 15.16 s (a review's
# finding, 2026-10-10). A step back stays a jump at any rate.
JUMP_FORWARD = 10
JUMP_BACK = Fraction(1, 10)
# ffprobe's names of the demuxers ffmpeg flags AVFMT_TS_DISCONT (libavformat at n9.0.2:
# mpegts.c:3880, 3894; mpeg.c:707; oggdec.c:997; m4vdec.c:77; dhav.c:501; ty.c:725;
# flvdec.c:2009; hls.c:3082; dvdvideodec.c:1860; rcwtdec.c:114): a remux takes a jump out of
# those alone. Checked on 2026-10-10: the jump of two MPEG-TS files joined, forwards, backwards
# and through the 33-bit wrap, gone after ffmpeg -i SOURCE -map 0 -c copy fixed.mkv, every frame
# kept 40 ms apart; kept, 997.96 s, by the same remux of a Matroska file holding it.
DISCONTINUOUS = frozenset(
    {
        *("mpegts", "mpegtsraw", "mpeg", "ogg", "m4v", "dhav", "ty", "live_flv", "hls"),
        *("dvdvideo", "rcwt"),
    }
)


@dataclass(frozen=True)
class Discontinuity:
    """A jump in a source's timestamps (JUMP_FORWARD, JUMP_BACK): frame `frame`'s, `after`, against
    the frame before's, `before`, both in seconds, exact."""

    frame: int
    before: Fraction
    after: Fraction
    remux: bool  # whether ffmpeg takes it out when it remuxes the file (DISCONTINUOUS)


@dataclass(frozen=True)
class Idet:
    """ffmpeg's idet over the first pass's frames (DESIGN.md, Not in the first version: interlaced
    and telecined sources), its counts summed over its summaries (IDET): frames with a repeated
    field (neither, top, bottom), and each frame's field order by its single-frame and its
    multiple-frame detection (TFF, BFF, progressive, undetermined). About one count a frame: a
    graph rebuilt mid-stream frees the frame idet holds uncounted, 119 counts for 120 frames
    on a stream tagged from its 15th."""

    repeated: tuple[int, ...]  # neither, top, bottom
    single: tuple[int, ...]  # TFF, BFF, progressive, undetermined
    multiple: tuple[int, ...]  # TFF, BFF, progressive, undetermined

    @property
    def combed(self) -> int:
        """The frames its multiple-frame detection finds interlaced, top or bottom field first."""
        return self.multiple[0] + self.multiple[1]

    @property
    def analysed(self) -> int:
        """The frames its multiple-frame detection counts."""
        return sum(self.multiple)

    def record(self) -> dict[str, dict[str, int]]:
        """The counts as the manifest records them, by idet's names, lower-cased."""
        return {
            key: {
                name.lower(): count for name, count in zip(names, getattr(self, key), strict=True)
            }
            for key, names in IDET_LINES.values()
        }

    @classmethod
    def from_record(cls, recorded: object) -> "Idet | None":
        """The counts a manifest records (record), None for anything else."""
        if not isinstance(recorded, Mapping):
            return None
        found: dict[str, tuple[int, ...]] = {}
        for key, names in IDET_LINES.values():
            counts: Any = cast(Mapping[str, Any], recorded).get(key)
            if not isinstance(counts, Mapping):
                return None
            values = [cast(Mapping[str, Any], counts).get(name.lower()) for name in names]
            if not all(type(value) is int for value in values):
                return None
            found[key] = tuple(cast(list[int], values))
        return cls(found["repeated"], found["single"], found["multiple"])


@dataclass(frozen=True)
class Scan:
    """What the first pass measured. The frame times are exact: the source's own timestamps, in
    its stream's time base, turned into microseconds without rounding (Fraction)."""

    frames: int  # decoded and counted, never the container's count (bug 11)
    durations: int  # times measured between consecutive frames that have a timestamp
    shortest: Fraction  # the shortest of them, in microseconds; 0 without any
    longest: Fraction  # the longest, in microseconds; 0 without any
    # The frame index and idet's counts, made meanwhile; not part of what a scan is compared by.
    index: FrameIndex | None = field(default=None, compare=False)
    idet: Idet | None = field(default=None, compare=False)
    # The first jump in the frames' timestamps, which is refused (timing_error); None without one.
    discontinuity: Discontinuity | None = None


def scan(path: Path) -> Scan:
    """Decode every frame of the first video stream of path, counting, timing and indexing them,
    and running idet on them, its packets scanned for keyframes meanwhile.

    One ffmpeg process decodes, prints each frame's line (DECODED) and the reports tied to it on
    stderr, and hashes each decoded frame on stdout, a CRC-32 of its planes as the decoder gives
    them (rawvideo to the framehash muxer, whose crc32 is zlib's: tests/test_index.py), so that no
    frame crosses a pipe; a second output runs idet on every frame, its counts on stderr at the
    end (IDET). The timestamps are the source's own, each read from its frame's hash line
    (HASHED): -copyts, and the hashes' time base the demuxer's. The frames are counted here
    as their lines come, on stdout and on stderr, which must agree, never by ffmpeg's counters,
    which restart when it rebuilds its filter graph mid-stream (research/docs/seeking.md,
    mechanism 8). ffprobe scans the packets in another process meanwhile (packets).

    Raises MediaError when ffmpeg or ffprobe fails, their counts disagree, the hashes come in no
    time base, or the path holds a line break (name_error)."""
    refused = name_error(path)
    if refused:
        raise MediaError(refused)
    command = [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats"),
        # repeat: two reports alike in a row are both printed, never folded into "Last message
        # repeated" (libavutil/log.c), each tied to its frame.
        *("-loglevel", "repeat+level+info", "-debug_ts", "-copyts"),
        *input_args(path),
        # The hashes' time base the stream's, its pts as they are. The encoder's default, a tick
        # per frame at the declared rate, would put two frames of a file drifting off the frame
        # grid on one tick, as joins of ffmpeg's concat demuxer do (cli._directory_refused).
        *("-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux"),
        *(*NO_METADATA, "-c:v", "rawvideo", "-f", "framehash", "-hash", "crc32", "-"),
        # idet on every frame decoded (DESIGN.md, Not in the first version), in an output and a
        # filter graph of its own, so that the frames hashed are the decoder's whatever idet
        # takes: for a format it doesn't (RGB, packed), ffmpeg converts in idet's graph alone. A
        # branch of a split would share one list of formats with the hashed branch
        # (libavfilter/formats.c:1171-1232, ff_default_query_formats), a conversion then falling
        # before the split or after it by the order the links are merged in. Measured on
        # 2026-10-09 (n9.0.2, 16 cores, 3 runs): the pass at 1,237-1,242 and 623-627 fps on 1080p
        # x264 at 10 and 55 Mbit/s, 1,180-1,193 and 612-619 with this output (4.5% and 1.7%),
        # 1,216-1,218 and 621-622 with the split; at 4K, idet is the pass's bottleneck either way,
        # not slice-threaded (libavfilter/vf_idet.c:453-464), about 10-12 ms a frame: 178-207 fps
        # on HEVC 10-bit 4:2:0, 104-106 with it, 198-214 on ProRes 4:2:2 10-bit, 80-83 with it.
        # Detecting the same costs that: idet reads every plane (:148-163), and luma alone would
        # detect otherwise. The null muxer writes nothing; its time base the stream's too, or a
        # join drifting off the frame grid puts two frames on one tick, and the muxer reports an
        # error (tests/test_decode.py, a concat join).
        *("-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux"),
        *(*NO_METADATA, "-vf", "idet", "-f", "null", "-"),
    ]
    scanned: list[object] = []
    scanner = threading.Thread(target=_packets_into, args=(path, scanned), daemon=True)
    scanner.start()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    assert process.stdout is not None and process.stderr is not None
    decoded = _Decoded()
    reader = threading.Thread(target=decoded.read, args=(process.stderr,), daemon=True)
    reader.start()
    hashed = _Hashed()
    hashed.read(process.stdout)
    status = process.wait()
    reader.join()
    scanner.join()
    errors = decoded.errors
    if status != 0:
        said = errors[-5:] or list(decoded.last)
        raise MediaError(f"{path}: ffmpeg failed decoding it: {' / '.join(said)}")
    [packets] = scanned
    if isinstance(packets, OSError):
        raise MediaError(f"{path}: ffprobe failed scanning its packets: {packets}") from packets
    if isinstance(packets, Exception):
        raise packets
    assert isinstance(packets, _Packets)
    hashes = hashed.crc32
    if len(hashes) != len(decoded.pts):
        raise MediaError(
            f"{path}: ffmpeg decoded {len(decoded.pts)} frames and hashed {len(hashes)}"
        )
    if hashed.refused or (hashes and hashed.time_base is None):
        raise MediaError(f"{path}: {hashed.refused or 'ffmpeg gave its hashes no time base'}")
    errors += packets.errors
    if errors:
        logger.warning("%s: ffmpeg reported errors decoding it: %s", path, " / ".join(errors[:5]))
    flagged = [frame for frame, flag in enumerate(decoded.corrupt) if flag]
    if flagged:
        # No seek decodes them as the start does, nor a decode always the same (media/reader.py).
        several = f"{len(flagged)} frames decoded with an error, from frame {flagged[0]}"
        logger.warning(
            "%s: %s: a read seeking through %s decodes the source from its start, and takes %s as"
            " decoded",
            path,
            several if len(flagged) > 1 else f"frame {flagged[0]} decoded with an error",
            *(("them", "them") if len(flagged) > 1 else ("it", "it")),
        )
    time_base = hashed.time_base or Fraction(1)
    index = FrameIndex(
        time_base,
        packets.start_time,
        np.array(hashed.pts, dtype=np.int64),
        np.array(hashes, dtype=np.uint32),
        np.array(decoded.corrupt, dtype=np.bool_),
        np.unique(np.array(packets.keyframes, dtype=np.int64)),
    )
    idet = decoded.idet()
    if idet is None:
        # Said, never silent (AGENTS.md): a build printing its counts otherwise.
        logger.warning("%s: ffmpeg's idet gave no counts: combed frames not looked for", path)
    jump = _discontinuity(hashed.pts, decoded.pts, time_base, packets.remux)
    return replace(_timed(hashed.pts, time_base, index, idet), discontinuity=jump)


def name_error(path: Path) -> str:
    """Why a source's name is refused, or "": a line break in it. ffmpeg prints the name as it
    is, in its input's dump and in its errors: a line of the name's own making among those the
    first pass reads (MAPPING). Known from the command line: refused before anything is done with
    the file (media/source.py, declare), and by the pass itself (scan)."""
    if "\n" in str(path) or "\r" in str(path):
        return (
            f"{str(path)!r}: a line break in its name, which ffmpeg prints as it is among the"
            " lines the first pass reads: rename it"
        )
    return ""


def _timed(
    pts: list[int], time_base: Fraction, index: FrameIndex, idet: Idet | None = None
) -> Scan:
    """The frames' timing from their pts, sptenc's way: a frame without a timestamp breaks the
    chain, the next time measured being between the two frames after it."""
    durations = 0
    shortest = longest = Fraction(0)
    previous: int | None = None
    for value in pts:
        current = None if value == NOPTS else value
        if current is not None and previous is not None:
            duration = (current - previous) * time_base * 1_000_000
            shortest = duration if durations == 0 else min(shortest, duration)
            longest = duration if durations == 0 else max(longest, duration)
            durations += 1
        previous = current
    return Scan(len(pts), durations, shortest, longest, index, idet)


def _discontinuity(
    pts: list[int], decoder: list[int], time_base: Fraction, remux: bool
) -> Discontinuity | None:
    """The first jump in the frames' timestamps, None without one: a step back past JUMP_BACK, or
    a step on past JUMP_FORWARD that is also longer than the frames' shortest step on by more
    than TOLERANCE_US, so that a constant rate's own steps, however long, are none; no step on is
    one when the shortest is itself past JUMP_FORWARD (left to the timing rule). `pts` are the
    hashes', in `time_base`, a frame's going back held at the frame before's (HASHED); `decoder`
    the decoder's own, one a frame too, which then say where it went; `remux` whether a remux
    takes a jump out of the file."""
    # In whole units of pts: 200,000 frames compared in Fractions would take a second.
    forward = JUMP_FORWARD * time_base.denominator
    back = JUMP_BACK.numerator * time_base.denominator
    # The longest step on that is no jump, in pts: the shortest step on plus the tolerance, or
    # any when the shortest is itself past the bound (JUMP_FORWARD); found when needed.
    longer: float | None = None
    for frame in range(1, len(pts)):
        before, after = pts[frame - 1], pts[frame]
        if after == before and NOPTS < decoder[frame] < before:
            after = decoder[frame]
        if before == NOPTS or after == NOPTS:
            continue
        step = (after - before) * time_base.numerator
        if step * JUMP_BACK.denominator < -back:
            return Discontinuity(frame, before * time_base, after * time_base, remux)
        if step > forward:
            if longer is None:
                # The steps on among those the timing rule measures (_timed), the hashes' own: a
                # pts held there for one that went back, or said twice, is none.
                steps = [
                    late - early
                    for early, late in pairwise(pts)
                    if early != NOPTS and late != NOPTS and late > early
                ]
                tolerance = Fraction(TOLERANCE_US, 1_000_000) / time_base
                shortest = min(steps)
                slow = shortest * time_base.numerator > forward
                longer = math.inf if slow else math.floor(shortest + tolerance)
            if after - before > longer:
                return Discontinuity(frame, before * time_base, after * time_base, remux)
    return None


class _Hashed:
    """What the first pass's stdout says, read as it comes: the time base (TIME_BASE), then each
    frame's pts and CRC-32 (HASHED). `refused` says a time base no pts can be read in."""

    def __init__(self) -> None:
        self.pts: list[int] = []
        self.crc32: list[int] = []
        self.time_base: Fraction | None = None
        self.refused = ""

    def read(self, lines: Iterable[str]) -> None:
        for line in lines:
            found = HASHED.match(line)
            if found is not None:
                self.pts.append(int(found[1]))
                self.crc32.append(int(found[2], 16))
            elif line.startswith("#tb "):
                base = TIME_BASE.match(line.rstrip("\n"))
                # A zero would raise in Fraction; ffmpeg refuses such a time base before it
                # encodes (fftools/ffmpeg_filter.c:2380-2385), so this is no source's doing.
                if base is None or int(base[1]) == 0 or int(base[2]) == 0:
                    self.refused = f"ffmpeg gave its hashes the time base {line.strip()!r}"
                elif self.time_base is None:
                    self.time_base = Fraction(int(base[1]), int(base[2]))


class _Decoded:
    """What the first pass's stderr says, read as it comes: each frame the decoder gives, with its
    own pts (DECODED) and whether the decoder flagged it (CORRUPT, the report before it), the
    errors, idet's counts (IDET); of the lines before ffmpeg decodes (MAPPING), the errors alone.
    `last` keeps its last lines, for a failure no error line tells of (ERROR)."""

    def __init__(self) -> None:
        self.pts: list[int] = []  # the decoder's own: the frames' are the hashes' (_Hashed)
        self.corrupt: list[bool] = []
        self.errors: list[str] = []
        self.last: deque[str] = deque(maxlen=5)
        self.counted: dict[str, list[int]] = {}  # idet's, summed, by Idet's fields

    def idet(self) -> Idet | None:
        """idet's counts, summed over its summaries; None without all three lines."""
        if len(self.counted) != len(IDET_LINES):
            return None
        return Idet(*(tuple(self.counted[key]) for key, _ in IDET_LINES.values()))

    def _idet(self, line: str) -> None:
        """Add one of idet's summary lines to the counts; another shape is ignored, idet's counts
        then incomplete (idet)."""
        found = IDET.search(line.rstrip("\n"))
        if found is None:
            return
        key, names = IDET_LINES[found[1]]
        counts = dict(IDET_COUNT.findall(found[2]))
        if set(counts) != set(names):
            return
        summed = self.counted.setdefault(key, [0] * len(names))
        for place, name in enumerate(names):
            summed[place] += int(counts[name])

    def read(self, lines: Iterable[str]) -> None:
        reported = False
        mapped = False
        # Where in errors lie those read since the last MAPPING line that don't start their line
        # (ERROR_AT_START): no errors, should another such line come.
        loose: list[int] = []
        for line in lines:
            self.last.append(line.strip())
            if line.startswith(MAPPING) and line.rstrip("\n") == MAPPING:
                # The reading begins again: what was read since the line before this one, a line
                # of the source's own text then, is its text too (MAPPING).
                for place in reversed(loose):
                    del self.errors[place]
                loose.clear()
                self.pts.clear()
                self.corrupt.clear()
                self.counted.clear()
                reported, mapped = False, True
                continue
            if not mapped:
                if ERROR_AT_START.match(line):
                    self.errors.append(line.strip())
                continue
            # About 14 lines a frame with -debug_ts: the cheapest test first.
            if "decoder -> pts:" in line:
                found = DECODED.search(line.rstrip("\n"))
                if found is None:
                    continue  # a line of another shape: the counts then disagree (scan)
                self.pts.append(NOPTS if found[1] == "NOPTS" else int(found[1]))
                self.corrupt.append(reported)
                reported = False
            elif CORRUPT in line:
                reported = True
            elif "Fields: " in line or "frame detection: " in line:
                self._idet(line)
            elif ERROR.search(line):
                if not ERROR_AT_START.match(line):
                    loose.append(len(self.errors))
                self.errors.append(line.strip())


@dataclass(frozen=True)
class _Packets:
    """What the packet scan found: the keyframes' pts, the file's start time in microseconds, the
    errors ffprobe reported."""

    keyframes: list[int]
    start_time: int | None
    errors: list[str]
    remux: bool = False  # whether ffmpeg's remux takes a timestamp jump out (DISCONTINUOUS)


def _packets_into(path: Path, found: list[object]) -> None:
    """packets(path) into found, or what it raised, which scan raises again: run beside the
    decode."""
    try:
        found.append(packets(path))
    except Exception as error:
        found.append(error)


def packets(path: Path) -> _Packets:
    """Scan the packets of the first video stream of path, demuxed only (3.5 s for 87 GB:
    research/docs/seeking.md, Still to test): the keyframes' pts, the packets flagged K, those
    without a pts left out (an MPEG-PS stream has 189 of 2,878: mechanism 4), since no read can
    seek to them; the file's start time, which -ss counts from (media/reader.py); and whether its
    demuxer is one whose timestamp jumps a remux takes out (DISCONTINUOUS).

    Raises MediaError when ffprobe fails."""
    command = [
        *("ffprobe", "-v", "error", "-select_streams", "v:0"),
        *("-show_entries", "packet=pts,flags:format=start_time,format_name"),
        *("-of", "compact", str(path)),
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace"
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        errors: list[str] = []
        drain = threading.Thread(target=_lines_into, args=(process.stderr, errors), daemon=True)
        drain.start()
        keyframes, start_time, demuxers = read_packets(process.stdout)
        status = process.wait()
        drain.join()
    if status != 0:
        raise MediaError(f"{path}: ffprobe failed scanning its packets: {' / '.join(errors[-5:])}")
    said = [f"ffprobe: {line}" for line in errors]
    return _Packets(keyframes, start_time, said, not DISCONTINUOUS.isdisjoint(demuxers))


def read_packets(lines: Iterable[str]) -> tuple[list[int], int | None, list[str]]:
    """ffprobe's compact output of packet=pts,flags and format=start_time,format_name (packets):
    the pts of the packets flagged K, in their order, those without one left out; the start time
    in microseconds, None without one; and the demuxer's names, ffprobe's format_name split at
    its commas ("matroska,webm")."""
    keyframes: list[int] = []
    start_time: int | None = None
    demuxers: list[str] = []
    for line in lines:
        section, _, rest = line.strip().partition("|")
        values = dict(item.split("=", 1) for item in rest.split("|") if "=" in item)
        if section == "packet":
            pts = values.get("pts", "N/A")
            if "K" in values.get("flags", "") and pts != "N/A":
                keyframes.append(int(pts))
        elif section == "format":
            demuxers = values.get("format_name", "").split(",")
            if values.get("start_time", "N/A") != "N/A":
                # Printed with 6 decimals (fftools/ffprobe.c), the microseconds it holds, exactly.
                start_time = round(Fraction(values["start_time"]) * 1_000_000)
    return keyframes, start_time, demuxers


def _lines_into(stream: IO[str], lines: list[str]) -> None:
    lines.extend(line.strip() for line in stream if line.strip())


def timing_error(scanned: Scan, frame_rate: Fraction, avg_frame_rate: Fraction | None) -> str:
    """Why the frames' timing is refused, or "" when they have the constant rate declared.

    With frame times measured (sptenc's rule), the shortest and longest must be within
    TOLERANCE_US, and the declared rate within them, since the output is written at that rate.
    Without (a single frame, no timestamps), the declared rates are compared, as sptenc does. A
    jump in the timestamps (Discontinuity) is refused first, as what it is."""
    if scanned.frames == 0:
        return "no frame decoded"
    jump = scanned.discontinuity
    if jump is not None:
        # The remux as the colour tags' (media/source.py): every stream but the data streams,
        # which Matroska refuses and MPEG-TS files often carry.
        way = (
            ", which ffmpeg takes out when it remuxes the file, every frame kept: ffmpeg -i SOURCE"
            " -map 0 -map -0:d -c copy fixed.mkv"
            if jump.remux
            else ", and only a constant frame rate is supported"
        )
        return (
            f"its timestamps jump at frame {jump.frame}, from {float(jump.before):.3f} to"
            f" {float(jump.after):.3f} s: a discontinuity{way}"
        )
    if scanned.durations:
        shortest, longest = float(scanned.shortest) / 1000, float(scanned.longest) / 1000
        if scanned.longest - scanned.shortest > TOLERANCE_US:
            return (
                f"variable frame rate: frames last from {shortest:g} to {longest:g} ms, and only"
                " a constant frame rate is supported"
            )
        period = Fraction(1_000_000) / frame_rate
        if not scanned.shortest - TOLERANCE_US <= period <= scanned.longest + TOLERANCE_US:
            return (
                f"the frame rate declared, {frame_rate} fps, is not the frames' own: they last"
                f" {shortest:g} to {longest:g} ms"
            )
        return ""
    if avg_frame_rate is None or abs(frame_rate - avg_frame_rate) > Fraction(1, 1000):
        return (
            f"the frame rate can't be checked (no frame times) and the declared ones differ:"
            f" {frame_rate} and {avg_frame_rate or 'none'} fps"
        )
    return ""
