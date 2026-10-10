"""A source's frames read from any frame on, as 16-bit RGB, each checked against the frame index
(DESIGN.md, Input: frame-exact reading from any frame; research/docs/seeking.md, Recommended
method).

A read from frame n seeks to the keyframe one GOP before n's own, selects by timestamp, and checks
every frame: one ffmpeg process decodes, and gives each decoded frame to two outputs, each with a
filter graph of its own: one selects and converts it to the pipeline's 16-bit RGB on stdout
(Conversion, as Decoder gives it), the other hashes it, a CRC-32 of the frame as the decoder gives
it, on a pipe of its own, as the first pass hashed it (media/scan.py). Each frame given is the
hashed frame its place says,
its pts and CRC-32 those the index holds. A mismatch reads again from one keyframe further back,
doubling; the last resort decodes from the start and counts, the first pass's own way. A mismatch
after a frame the first pass decoded with an error goes there at once (_fall_back). From the
start, a frame at its place, its pts the index's, is taken as decoded whatever its CRC-32, said
(Reader.otherwise): ffmpeg doesn't decode a damaged source the same twice.

n's own keyframe is the last whose pts is at most n's (FrameIndex.own_keyframe), and the read
starts one keyframe before it, as seeking.md's c_gop did, exact on 674 of 676 targets outside S1's
dense-IDR passage, all 54 open-GOP leading pictures of S11 included. A leading picture (displayed
before its keyframe K, decoded after it, referencing the GOP before) thus belongs to the keyframe
before K, and its read starts one keyframe before that one: every picture it references is
decoded from a keyframe's start, one GOP more than its references need. Defined on display order,
the only order the index holds, it needs no decode order.

A read never waits on ffmpeg while ffmpeg waits on it. Its hashes come on pipes read by threads of
their own; each output's encoder runs in one thread, so that no hash is held back behind frames
still to come through stdout (read_command); while a frame's hashes lag its RGB all the same,
stdout is read ahead, READ_AHEAD frames at most, past which the read fails naming the lag
(_Run.line); and a watchdog stops ffmpeg when nothing comes for STALL seconds while the reader
waits on it, the read failing, naming what it waited for (_Run._watch)."""

import io
import logging
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from seedvr2x.media.conversion import Conversion
from seedvr2x.media.ffmpeg import NO_METADATA, MediaError, input_args
from seedvr2x.media.index import FrameIndex

logger = logging.getLogger(__name__)

# A frame's line in ffmpeg's framehash and framemd5 output (libavformat/hashenc.c:282-317 at
# n9.0.2): stream, dts, pts, duration, size, hash.
LINE = re.compile(rb"0, *-?\d+, *(-?\d+), *-?\d+, *\d+, ([0-9a-f]+)")

# The frames a warning lists of those decoded otherwise than in the first pass (Reader.otherwise):
# the first ones, the count said, so that a badly damaged source makes no line per frame.
LISTED = 10

# The frames read ahead at most while a frame's hashes lag its RGB (_Run.line). With one encoder
# thread an output (read_command), the decoder gives a frame to the hash output's graph before or
# right after the RGB output's (ONE_THREAD), and its hash then waits in that output's own queues
# only, which stdout never holds: 2 frames before the graph (DEFAULT_FRAME_THREAD_QUEUE_SIZE,
# fftools/ffmpeg_sched.h:262, ffmpeg_sched.c:894 at n9.0.2), 1 in it, 2 before the encoder, 1 in
# it, 8 packets before the muxer (DEFAULT_PACKET_THREAD_QUEUE_SIZE, :257), 1 in it; with the
# decoder's own, the RGB is at most 16 frames ahead of the hashes written. On 1080p H.264, each
# CRC-32 line came before its frame's RGB, 300 of 300, stdout drained freely, and no read read
# ahead: 150 reads, and 72 with the CPUs oversubscribed 2 to 1, 36 of them with the CRC-32
# encoder put back in 64 frame threads (2026-10-09). 24 leaves room for the pipes and this
# process's threads: 299 MB at 1080p, 1.19 GB at 2160p, held only while a hash lags.
READ_AHEAD = 24

# How long the reader waits for a frame's hash before it reads one more frame ahead, in seconds.
# The hash is normally there before its frame (READ_AHEAD): a wait this long means a hash late
# for another reason, and costs a frame read ahead.
LAG_POLL = 0.05

# How often a reader waiting for a lagging hash looks again, in seconds. It sleeps, as the
# watchdog does (_Run._watch): a timed wait on a lock (Queue.get or Event.wait with a timeout)
# came back up to 5.5 s late under WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2, 2026-10-09: 4
# to 6 of 709 waits of 20 ms over a second), time.sleep never.
POLL = 0.001

# The watchdog's bound, in seconds: ffmpeg stopped and the read failed after this long without
# progress, no RGB and no hash line, while the reader waits on it (_Run._watch); never the read's
# total time. A read from the start hashes every frame it decodes, a line every 6 ms on S5, the
# slowest source, a long-GOP HEVC master decoded at 171 fps, 17 minutes to the end of 2 hours;
# before its first line, ffmpeg probes and seeks, S5's worst read through a seek taking 16 s in
# all (research/docs/seeking.md). 300 s is 19 times that, for slower CPUs and disks. A stalled
# read then fails in 5 minutes, where it hung: 13 minutes at 0% CPU on the box (2026-10-09),
# until stopped by hand.
STALL = 300.0


def listed(frames: list[int]) -> str:
    """The first LISTED of frames, ", " between them, "..." after when there are more."""
    shown = ", ".join(str(frame) for frame in frames[:LISTED])
    return shown + (", ..." if len(frames) > LISTED else "")


@dataclass(frozen=True)
class Attempt:
    """One ffmpeg process a read started: for frame `target`, decoding from keyframe `keyframe` (its
    place in the index's keyframes), `back` keyframes before the target's own, or from the start
    (keyframe None); and why, "" for a read's first."""

    target: int
    keyframe: int | None
    back: int
    command: tuple[str, ...]
    why: str


class _Mismatch(Exception):
    """A frame other than the index's: missing, extra, another picture, or none (ffmpeg ended)."""


class _Stalled(Exception):
    """ffmpeg stopped: no progress while the reader waited on it (_Run._watch), or a frame's hashes
    lagging its RGB by more than READ_AHEAD frames (_Run.line). The message says which."""


# Each output's encoder in one thread. fftools gives every encoder an automatic thread count
# (fftools/ffmpeg_mux_init.c:1384-1386 at n9.0.2), and rawvideo encodes in frame threads
# (libavcodec/rawenc.c:89), as many as the CPUs, 64 at most (frame_thread_encoder.c:167-169),
# which give one packet at most a frame sent, the oldest frame's, none while it isn't encoded:
# once its threads fall behind, the encoder holds up to that many frames until more come
# (:322-326). A frame's hashes are on their way before or right after its RGB (the decoder gives
# each frame to every output's graph in turn: fftools/ffmpeg_sched.c:2403), but a hash encoder
# holding frames gives a frame's CRC-32 only after frames that come through stdout, which a
# reader waiting for that CRC-32 doesn't read: a read hung 13 minutes on the box (2026-10-09),
# its CRC-32 muxer one frame short of the RGB taken. Reading as the box did (one graph, the
# hashes behind a split), the CRC-32 encoder in 64 frame threads and the CPUs oversubscribed 2 to
# 1, 3 reads of 36 hung so, ffmpeg at 0% CPU; in one thread, none of 24. In one thread, each
# frame comes out as it goes in: no hash waits on stdout, and a read's last frames need no later
# ones (8 frames of 1080p H.264 read in 0.342 s, median, from 0.362, the reads paired).
ONE_THREAD = ("-threads", "1")


def read_command(
    path: Path,
    conversion: Conversion,
    seek: str | None,
    select: int | None,
    hashes: int,
    md5: int | None = None,
) -> list[str]:
    """The ffmpeg command reading path: -copyts and -ss `seek` (seconds after the file's start,
    relative to it as seeking.md's method 2 has it, never -seek_timestamp: mechanism 5), without
    accurate seek; the decoded frames given to outputs of their own, each with its own filter
    graph: kept from pts `select` on (select) and converted to gbrp16le on stdout, as Decoder
    gives them; every one hashed (CRC-32) to the pipe of file descriptor `hashes`, as the first
    pass hashes them (media/scan.py); with `md5`, the frames kept also hashed by MD5, as decoded,
    to that descriptor (tests/test_seek_sources.py). Each output's encoder in one thread
    (ONE_THREAD), none taking the source's metadata, as the first pass's (ffmpeg.NO_METADATA)."""
    kept = None if select is None else f"select=gte(pts\\,{select})"
    rgb = conversion.filters() if kept is None else f"{kept},{conversion.filters()}"
    return [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-copyts"),
        # No accurate seek: the select by pts is the selection. Its trim, which fftools puts in
        # front of each graph with -ss (fftools/ffmpeg_demux.c:1140-1142, ffmpeg_filter.c:
        # 1945-1947), only dropped frames before -ss, which select drops too, and is a filter the
        # build would need (:1532-1551).
        *(() if seek is None else ("-noaccurate_seek", "-ss", seek)),
        *input_args(path),
        # As decode.decode_output's, a select before: a frame size changing within the stream
        # fails rather than being scaled by swscale.
        *("-map", "0:v:0", "-vf", rgb, "-fps_mode", "passthrough", "-noautoscale", *ONE_THREAD),
        *(*NO_METADATA, "-f", "rawvideo", "-pix_fmt", "gbrp16le", "pipe:1"),
        # The hashes in outputs of their own, the CRC-32's as the first pass's, each its own graph
        # (an output without filters gets null's: fftools/ffmpeg_mux_init.c:436), so that nothing
        # gives them the conversion's formats. Behind a split in one graph with the conversion,
        # whose links share one list of formats, the conversion's planar format reached back
        # before the split once -ss's trim stood in front of the graph, lavfi then merging the
        # lists in another order: every seek of a packed or semi-planar source hashed the frame
        # converted, never the index's, and went back to the start (6 to 7 attempts on uyvy422,
        # yuyv422 and nv12 rawvideo and rgb24 PNG, 2 to 3 on FFV1 bgr0: 2026-10-09).
        *("-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux", *NO_METADATA),
        *("-c:v", "rawvideo", *ONE_THREAD, "-f", "framehash", "-hash", "crc32", f"pipe:{hashes}"),
        *(
            ()
            if md5 is None
            else (
                *("-map", "0:v:0", *(() if kept is None else ("-vf", kept))),
                *("-fps_mode", "passthrough", "-enc_time_base:v", "demux", *NO_METADATA),
                *("-c:v", "rawvideo", *ONE_THREAD, "-f", "framemd5", f"pipe:{md5}"),
            )
        ),
    ]


def seek_time(index: FrameIndex, keyframe: int) -> str:
    """-ss for a read decoding from keyframe `keyframe`: its time after the file's start, in
    microseconds, as -ss is parsed (libavutil/parseutils.c, av_parse_time), rounded down, so that
    the demuxer's seek, to -ss plus the start time (fftools/ffmpeg_demux.c:2244-2265), never
    lands after the keyframe."""
    exact = int(index.keyframes[keyframe]) * index.time_base * 1_000_000
    microseconds = max(0, math.floor(exact) - (index.start_time or 0))
    return f"{microseconds // 1_000_000}.{microseconds % 1_000_000:06d}"


class _Run:
    """One ffmpeg process of a read (read_command), labelled `label` (the source, frame and
    attempt) for its errors: its RGB frames on stdout, read as needed, and ahead while a frame's
    hashes lag (line); its frames' hashes, and MD5s when asked, read from their pipes as they come
    by threads of their own, so that ffmpeg never waits on them; its errors in a temporary file; a
    watchdog stopping it when nothing comes while the reader waits on it (_watch).

    taken counts the frames taken from stdout, given or read ahead; held is the most read ahead at
    once."""

    def __init__(
        self,
        path: Path,
        conversion: Conversion,
        seek: str | None,
        select: int | None,
        frame_bytes: int,
        md5: bool,
        label: str,
    ) -> None:
        self.frame_bytes = frame_bytes
        self.select = select
        self.label = label
        self.taken = 0
        self.held = 0
        self.stalled = ""  # why the watchdog stopped ffmpeg, once it has
        self._ahead: deque[bytearray] = deque()  # frames read ahead, in their order
        self._spare: list[bytearray] = []  # buffers for them, kept for the next
        # How stdout ended, seen reading ahead: None not yet, "" at a frame's end, else why not.
        self._end: str | None = None
        self._stall = STALL
        # What the reader waits on ffmpeg for, since when (time.monotonic()); None when it doesn't.
        self._waiting: tuple[float, str] | None = None
        self._progress = time.monotonic()  # when stdout or a hash pipe last gave something
        self._over = threading.Event()  # the run stopped or waited for: the watchdog ends
        try:
            self._log = tempfile.TemporaryFile()
        except OSError as error:
            what = f"reading with ffmpeg: no temporary file for its errors: {error}"
            raise MediaError(what) from error
        pipes = [os.pipe() for _ in range(2 if md5 else 1)]
        self.command = read_command(
            path, conversion, seek, select, pipes[0][1], pipes[1][1] if md5 else None
        )
        try:
            # In a process group of its own, out of reach of the terminal's Ctrl-C, which only
            # seedvr2x handles (runtime/stop.py), as Decoder's.
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=self._log,
                pass_fds=[write for _, write in pipes],
                process_group=0,
            )
        except BaseException:
            for read, write in pipes:
                os.close(read)
                os.close(write)
            self._log.close()
            raise
        for _, write in pipes:
            os.close(write)
        assert self._process.stdout is not None
        self._stdout = cast(io.BufferedReader, self._process.stdout)
        self.hashes: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.md5s: queue.Queue[tuple[int, str] | None] | None = queue.Queue() if md5 else None
        self._threads = [self._drain(pipes[0][0], self.hashes)]
        if self.md5s is not None:
            self._threads.append(self._drain(pipes[1][0], self.md5s))
        threading.Thread(target=self._watch, daemon=True).start()

    def _drain(
        self, descriptor: int, into: "queue.Queue[tuple[int, str] | None]"
    ) -> threading.Thread:
        def drain() -> None:
            try:
                with os.fdopen(descriptor, "rb") as lines:
                    for line in lines:
                        self._progress = time.monotonic()
                        found = LINE.match(line)
                        if found is not None:
                            into.put((int(found[1]), found[2].decode()))
            finally:
                into.put(None)  # the end, whatever ended it

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        return thread

    def frame(self, into: memoryview, what: str) -> bool:
        """The next RGB frame into `into`, frame_bytes long, a frame read ahead first: False at the
        end of the stream. `what` names the frame awaited, for a stall. Raises _Mismatch when the
        stream ends inside a frame, _Stalled when the watchdog stopped ffmpeg."""
        if self._ahead:
            buffer = self._ahead.popleft()
            into[:] = buffer
            self._spare.append(buffer)
            return True
        if self._end is not None:
            if self._end:
                raise _Mismatch(self._end)
            return False
        filled = self._fill(into, what)
        if filled == len(into):
            self.taken += 1
            return True
        self._check()
        if filled:
            raise _Mismatch(f"ffmpeg's output ends {filled} bytes into a frame")
        return False

    def line(
        self, lines: "queue.Queue[tuple[int, str] | None]", what: str
    ) -> tuple[int, str] | None:
        """The next of `lines` (hashes or md5s), None at their end; `what` names it. While it lags,
        stdout is read ahead, a frame each LAG_POLL, so that ffmpeg never waits on its stdout for
        it; READ_AHEAD frames at most. Raises _Stalled past them, or when the watchdog stopped
        ffmpeg."""
        with self._wait(what):
            since = time.monotonic()
            while True:
                if self._end is not None:
                    found = lines.get()  # stdout ended: ffmpeg waits on it no more
                    break
                try:
                    found = lines.get_nowait()
                    break
                except queue.Empty:
                    pass
                if time.monotonic() - since < LAG_POLL:
                    time.sleep(POLL)
                    continue
                self._read_ahead(what)
                since = time.monotonic()
        if found is None:
            self._check()
        return found

    def _read_ahead(self, what: str) -> None:
        """One more RGB frame read ahead while `what` lags, or stdout's end noted."""
        if len(self._ahead) >= READ_AHEAD:
            stalled = (
                f"{self.label}: {what} hasn't come with {READ_AHEAD} more frames read on stdout:"
                f" ffmpeg's hashes lag its RGB beyond what its queues hold; stopped"
            )
            logger.error("%s", stalled)
            raise _Stalled(stalled)
        buffer = self._spare.pop() if self._spare else bytearray(self.frame_bytes)
        filled = self._fill(memoryview(buffer), "the next frame read ahead meanwhile")
        if filled == len(buffer):
            self.taken += 1
            self._ahead.append(buffer)
            self.held = max(self.held, len(self._ahead))
            return
        self._spare.append(buffer)
        self._check()
        self._end = f"ffmpeg's output ends {filled} bytes into a frame" if filled else ""

    def _fill(self, into: memoryview, what: str) -> int:
        """Read stdout into `into` until it is full or stdout ends: the bytes read."""
        filled = 0
        with self._wait(what):
            while filled < len(into):
                got = self._stdout.readinto(into[filled:])
                if not got:
                    break
                filled += got
                self._progress = time.monotonic()
        return filled

    def more(self) -> bool:
        """Whether ffmpeg has more RGB output. Raises _Stalled when the watchdog stopped ffmpeg."""
        if self._ahead or self._end is not None:
            return bool(self._ahead) or bool(self._end)
        with self._wait("the end of ffmpeg's output"):
            more = bool(self._stdout.read(1))
        if not more:
            self._check()
        return more

    def wait(self) -> int:
        """ffmpeg's exit status, once it has exited. Raises _Stalled when the watchdog stopped
        ffmpeg."""
        with self._wait("ffmpeg's exit"):
            status = self._process.wait()
        self._over.set()
        self._join()
        self._check()
        return status

    def stop(self) -> None:
        """Stop ffmpeg, whatever is left to read."""
        self._over.set()
        self._process.kill()
        self._process.wait()
        self._join()

    @contextmanager
    def _wait(self, what: str) -> Generator[None]:
        """The reader waiting on ffmpeg for `what` meanwhile, which the watchdog watches; within
        another wait, said after it."""
        before = self._waiting
        self._waiting = (time.monotonic(), what if before is None else f"{before[1]}, {what}")
        try:
            yield
        finally:
            self._waiting = before

    def _check(self) -> None:
        """Raise _Stalled when the watchdog stopped ffmpeg: what it waited for, not a mismatch."""
        if self.stalled:
            raise _Stalled(self.stalled)

    def _watch(self) -> None:
        """The watchdog: ffmpeg stopped (killed) when, while the reader waits on it, neither RGB nor
        a hash line has come for STALL seconds. Between reads, ffmpeg waits on the reader, which
        is no stall. A stall is logged, and said by the reader's next call (_check).

        It ticks by sleeping (POLL says why). This process stopped meanwhile (Ctrl-Z) or its
        machine paused, the threads reading ffmpeg's pipes were too, and saw nothing: a tick
        counts for a second past its due at most, so that such a freeze is never taken for
        ffmpeg's stall, and a stall still shows on a machine that freezes it often."""
        tick = min(1.0, self._stall / 10)
        before = time.monotonic()
        counted: float | None = None  # when the wait or the progress counted from began
        idle = 0.0
        while True:
            time.sleep(tick)
            if self._over.is_set():
                return
            now = time.monotonic()
            step = min(now - before, tick + 1.0)
            before = now
            waiting = self._waiting
            if waiting is None:
                continue
            since, what = waiting
            start = max(since, self._progress)
            if start != counted:
                counted, idle = start, min(now - start, step)
            else:
                idle += step
            if idle < self._stall:
                continue
            self.stalled = (
                f"{self.label}: nothing from ffmpeg in {self._stall:g} s, neither RGB nor a hash"
                f" line, while the read waited for {what}: stopped"
            )
            logger.error("%s", self.stalled)
            self._process.kill()
            return

    def errors(self) -> str:
        """ffmpeg's last 20 lines of errors, " / " between them."""
        if self._log.closed:
            return ""
        self._log.seek(0)  # ffmpeg is done writing at the offset the file shares with it
        return " / ".join(
            line.decode(errors="replace").strip() for line in deque(self._log, maxlen=20)
        )

    def close(self) -> None:
        self._over.set()
        self._stdout.close()
        self._log.close()

    def _join(self) -> None:
        for thread in self._threads:
            thread.join()


class Reader:
    """A source's frames as 16-bit RGB from frame `first` on, read as they are needed, each
    checked against the frame index before it is given (the module's docstring); (n, H, W, 3)
    uint16, as Decoder gives them. At most `frames` of them, the first pass's count; finish checks
    the end. Used as a context manager, the end is checked on a normal exit, and ffmpeg is stopped
    on an exception.

    attempts lists the ffmpeg processes started, each with why. otherwise lists the frames given,
    decoded from the start, whose CRC-32 isn't the index's (_take), in order. With md5, md5s holds
    the MD5 of each frame given as the decoder gives it, before its conversion
    (tests/test_seek_sources.py, against measurement's references). held is the most frames read
    ahead at once while their hashes lagged (_Run.line), over the attempts.

    A read that stalls, nothing from ffmpeg for STALL seconds while the reader waits on it, or a
    frame's hashes lagging its RGB by more than READ_AHEAD frames, raises MediaError naming the
    source, the frame, the attempt and what was awaited, ffmpeg stopped."""

    def __init__(
        self,
        path: Path,
        conversion: Conversion,
        width: int,
        height: int,
        index: FrameIndex,
        first: int = 0,
        frames: int | None = None,
        md5: bool = False,
    ) -> None:
        self.path, self.conversion, self.index = path, conversion, index
        self.width, self.height = width, height
        self.frames = index.frames if frames is None else frames
        if not 0 <= first <= self.frames:
            raise ValueError(f"frame {first} asked, of {self.frames}")
        self.position = first  # the frame the next read gives first
        self.attempts: list[Attempt] = []
        self.otherwise: list[int] = []
        self._summed_up = False  # otherwise said at the read's end (_sum_up)
        self.md5s: list[str] = []
        self._md5 = md5
        self._frame_bytes = width * height * 3 * 2
        # Keyframes back from the target's own: 1, then doubled at each mismatch.
        self._back = 1
        self._run: _Run | None = None
        self._held = 0  # held by the runs done with
        # A run's frames: a counted run's (from the start) by the hashes counted; another's from
        # its target on, by the order of the frames it selects.
        self._counted = False
        self._next = first
        if first < min(self.frames, index.frames):
            self._open(first, "", start=False)

    @property
    def held(self) -> int:
        """The most frames read ahead at once while their hashes lagged, over the attempts."""
        run = self._run
        return max(self._held, run.held if run is not None else 0)

    def read(self, count: int) -> npt.NDArray[np.uint16]:
        """The next `count` frames, (n, H, W, 3) uint16 RGB: fewer only at the end of the
        stream."""
        planes = np.empty((count, 3, self.height, self.width), dtype="<u2")  # G, B, R
        view = memoryview(planes).cast("B")
        given = 0
        while given < count and self._give(view[given * self._frame_bytes :][: self._frame_bytes]):
            given += 1
        frames = planes[:given]
        return np.stack((frames[:, 2], frames[:, 0], frames[:, 1]), axis=-1)

    def finish(self) -> None:
        """Check the end of the stream: ffmpeg done, with no error, after `frames` frames."""
        self._sum_up()
        run = self._run
        if run is not None:
            try:
                left = run.more()
                status = 0 if left else run.wait()
            except _Stalled as stalled:
                raise self._error(run, str(stalled)) from None
            if left:
                raise self._error(run, f"frames left after {self.position}")
            if status != 0:
                raise self._error(run, f"exit status {status}")
            self._done(run)
        if self.position != self.frames:
            raise MediaError(
                f"decoding with ffmpeg: {self.position} frames, the first pass counted"
                f" {self.frames}"
            )

    def stop(self) -> None:
        """Stop ffmpeg, whatever is left to read."""
        self._sum_up()
        run = self._run
        if run is not None:
            run.stop()
            self._done(run)

    def _done(self, run: _Run) -> None:
        """Done with run, stopped or ended: what it held kept, its files closed."""
        self._held = max(self._held, run.held)
        run.close()
        if self._run is run:
            self._run = None

    def _sum_up(self) -> None:
        """At the read's end, the frames decoded otherwise than in the first pass after the first,
        which _take said at once: their count and the first of them, in one line."""
        if self._summed_up or len(self.otherwise) < 2:
            return
        self._summed_up = True
        logger.warning(
            "%s: %d frames of this read decoded otherwise than in the first pass, from the start,"
            " each at its place, its pts the index's: taken as decoded: %s",
            self.path,
            len(self.otherwise),
            listed(self.otherwise),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self.finish()
        else:
            self.stop()

    def _give(self, into: memoryview) -> bool:
        """Read the next frame into `into`, checked: False at the end of the stream. A mismatch
        reads again (_fall_back) until the frame comes right, or raises MediaError; so does a
        stall, ffmpeg stopped."""
        while True:
            run = self._run
            if run is None:
                return False  # at the end already
            try:
                return self._take(run, into)
            except _Mismatch as mismatch:
                self._fall_back(str(mismatch))
            except _Stalled as stalled:
                raise self._error(run, str(stalled)) from None

    def _take(self, run: _Run, into: memoryview) -> bool:
        """The next frame of run into `into`, checked against the index (raises _Mismatch); the
        frames its select keeps before the one wanted, a counted run's, dropped."""
        while True:
            if not run.frame(into, f"frame {self.position}'s RGB"):
                if self.position >= min(self.frames, self.index.frames):
                    return False
                raise _Mismatch(f"ffmpeg's frames end before frame {self.position}")
            frame, pts, crc = self._hashed(run)
            md5 = None
            if run.md5s is not None:
                hashed = run.line(run.md5s, f"frame {frame}'s MD5")
                if hashed is None:
                    raise _Mismatch(f"ffmpeg's MD5s end before frame {frame}")
                md5 = hashed[1]
            if frame < self.position:
                continue  # kept by a counted run's select before the frame wanted
            if self.position >= self.frames:
                raise self._error(run, f"more frames than the {self.frames} the first pass counted")
            if frame > self.position:
                raise _Mismatch(f"frame {self.position} missing: the next frame is {frame}")
            index = self.index
            if frame >= index.frames:
                raise self._error(run, f"frame {frame}: past the {index.frames} of the frame index")
            if pts != int(index.pts[frame]):
                raise _Mismatch(f"frame {frame}: pts {pts}, where the index has {index.pts[frame]}")
            indexed = int(index.crc32[frame])
            if int(crc, 16) != indexed:
                why = f"frame {frame}: CRC-32 {crc}, where the index has {indexed:08x}"
                if not self._counted:
                    # After a seek, the right pts can come with another picture: a keyframe that
                    # is no entry point (in-band parameter sets, seeking.md mechanism 6) or a
                    # leading picture decoded without its references. Strict: further back.
                    raise _Mismatch(why)
                # PROVISIONAL (the orchestrator's choice for design; DESIGN.md has the start as
                # the last resort, counting, and says nothing of a decode that differs): from the
                # start, the frame at its place, counted, its pts the index's, is taken as
                # decoded whatever its CRC-32. Frames are compared by place and pts there, never
                # searched; a resume checks the source by its SHA-256 and the index by its own,
                # so the decoder is the cause left. ffmpeg's frame threads conceal damage
                # otherwise from one full decode to the next: on a damaged x264 file, of 12
                # decodes, 4 reported one of its two damaged frames and 8 both, and the CRC-32s
                # of the B-frame and of the 13 frames after the P-frame varied; on x265 the damage
                # went unreported and a frame came out 2 ways in 6; MPEG-2's slice threads gave
                # the same in 12 (2026-10-09, n9.0.2, 320x240 encodes damaged as
                # tests/test_index.py damages its own). So a source with unreported damage would
                # otherwise fail at random, even a run reading every frame from the first. What
                # it means: an output is bit-identical across a resume only for a source that
                # decodes the same each time. Said at once for a read's first such frame, then
                # counted, the rest in one line at the read's end (_sum_up).
                self.otherwise.append(frame)
                if len(self.otherwise) == 1:
                    logger.warning(
                        "%s: %s, decoded from the start, at its place, its pts the index's: taken"
                        " as decoded, since ffmpeg's frame threads conceal a damaged source"
                        " otherwise from one decode to the next; any more such frames of this"
                        " read are counted and said at its end",
                        self.path,
                        why,
                    )
            if md5 is not None:
                self.md5s.append(md5)
            self.position += 1
            return True

    def _hashed(self, run: _Run) -> tuple[int, int, str]:
        """The frame the next RGB frame of run is, with its pts and hash: the next hashed frame its
        select keeps. Raises _Mismatch when the hashes end first."""
        while True:
            line = run.line(run.hashes, f"frame {self._next}'s CRC-32")
            if line is None:
                raise _Mismatch(f"ffmpeg's hashes end before frame {self.position}")
            pts, crc = line
            frame = self._next
            if self._counted:
                self._next += 1
            if run.select is not None and pts < run.select:
                continue
            if not self._counted:
                self._next += 1
            return frame, pts, crc

    def _open(self, target: int, why: str, start: bool) -> None:
        """Start the run reading from frame `target`: from the keyframe self._back before its own,
        seeking (seek_time) and selecting its pts on; or from the start, counting (`start`, or no
        such keyframe, or a pts select can't single out: FrameIndex.selects)."""
        index = self.index
        own = index.own_keyframe(target)
        keyframe = own - self._back
        # PROVISIONAL (DESIGN.md has the start as the last resort only): a frame before the third
        # keyframe, which a read from the first keyframe would decode from the start anyway, and
        # one select can't single out, which only counting finds, are read from the start at once.
        if start or own < 0 or keyframe < 1 or not index.selects(target):
            if target and not why:
                logger.info(
                    "%s: frame %d decoded from the start of the source, counted: %s",
                    self.path,
                    target,
                    "its timestamp doesn't set it apart from the frames before it"
                    if own >= 0 and keyframe >= 1
                    else "it lies before the source's third keyframe",
                )
            # The frames from the target on, every one, by the lowest pts among them.
            select = None if not target else index.lowest_from(target)
            self._start(target, None, 0, None, select, why, counted=True)
            return
        seek = seek_time(index, keyframe)
        if not why:
            logger.info(
                "%s: frame %d read through the frame index, decoded from frame %d, the keyframe"
                " one before its own",
                self.path,
                target,
                index.first_frame_from(keyframe),
            )
        self._start(target, keyframe, self._back, seek, int(index.pts[target]), why, counted=False)

    def _start(
        self,
        target: int,
        keyframe: int | None,
        back: int,
        seek: str | None,
        select: int | None,
        why: str,
        counted: bool,
    ) -> None:
        where = (
            "decoding from the start"
            if keyframe is None
            else f"decoding from frame {self.index.first_frame_from(keyframe)}, keyframe {keyframe}"
        )
        label = f"{self.path}: frame {target}, attempt {len(self.attempts) + 1}, {where}"
        run = _Run(self.path, self.conversion, seek, select, self._frame_bytes, self._md5, label)
        self._run, self._counted = run, counted
        self._next = 0 if counted else target
        self.attempts.append(Attempt(target, keyframe, back, tuple(run.command), why))

    def _fall_back(self, why: str) -> None:
        """After a mismatch at self.position: read again from one keyframe further back, the
        distance doubled; from the start when the first pass flagged a frame the run decoded, or
        no keyframe is left. A mismatch from the start, a frame missing or extra, a pts other than
        the index's or ffmpeg ending early (a CRC-32 there is taken: _take), raises MediaError,
        naming the frame."""
        run = self._run
        assert run is not None
        run.stop()
        errors = run.errors()
        self._done(run)
        attempt = self.attempts[-1]
        target = self.position
        if attempt.keyframe is None:
            raise MediaError(
                f"{self.path}: {why}, decoded from the start, as the first pass decoded it: the"
                " source isn't the one indexed, or its decode not the same"
                + (f": {errors}" if errors else "")
            )
        index = self.index
        decoded_from = index.first_frame_from(attempt.keyframe)
        flagged = index.first_error(decoded_from, target)
        if flagged is not None:
            # PROVISIONAL (DESIGN.md records the decoder's error report and says no more of it;
            # such a frame "may only match from the start", seeking.md's method): there at once.
            # The decoder conceals a damaged frame one way decoding from the start, another from
            # any seek: no seek gave S9's last frames again (mechanism 9), so going further back
            # would decode, over log2(keyframes) reads, about twice the frames the start does, for
            # nothing.
            logger.warning(
                "%s: %s; frame %d decoded with an error in the first pass, which no seek gives"
                " again: frame %d read again from the start, counted",
                self.path,
                why,
                flagged,
                target,
            )
            self._open(target, why, start=True)
            return
        self._back *= 2
        own = index.own_keyframe(target)
        if own - self._back < 1 or not index.selects(target):
            logger.warning(
                "%s: %s; frame %d read again from the start, counted, the last resort",
                self.path,
                why,
                target,
            )
            self._open(target, why, start=True)
            return
        logger.warning(
            "%s: %s; frame %d read again from frame %d, %d keyframes before its own",
            self.path,
            why,
            target,
            index.first_frame_from(own - self._back),
            self._back,
        )
        self._open(target, why, start=False)

    def _error(self, run: _Run, what: str) -> MediaError:
        """ffmpeg stopped, and what to raise: what, with ffmpeg's last errors."""
        run.stop()
        errors = run.errors()
        self._done(run)
        return MediaError(f"decoding with ffmpeg: {what}" + (f": {errors}" if errors else ""))
