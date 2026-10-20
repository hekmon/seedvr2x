"""The output, written as it comes: FFV1 masters, gbrp16le or yuv420p10le, and 16-bit PNG, each
through an ffmpeg pipe fed the float frames; or a segment by the user's command, --segment-cmd,
fed the master's own stream (DESIGN.md, Output).

The quantisation and the FFV1 settings are those of research/scripts/ffv1_out.py, the master
validated in research/docs/output.md: 16-bit RGB rounded to nearest, FFV1 level 3, every frame a
keyframe, 16 slices with CRCs, the exact rational frame rate. The tags follow DESIGN.md (Colour and
shape): the source's primaries and transfer copied, never converted; the yuv420p10le master's
matrix BT.709 at HD sizes.

ffmpeg n9 converts frames whose tags differ from those asked of the encoder (an untagged frame to
BT.709, for one). So every tag is set twice, on the frames (setparams) and on the encoder, and
nothing converts but zscale.
"""

import contextlib
import errno
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import IO, Self

import numpy as np
import numpy.typing as npt

from seedvr2x.media.ffmpeg import MediaError, zscale
from seedvr2x.media.files import command_partial, partial_path, replace_whole
from seedvr2x.media.probe import VideoStream

logger = logging.getLogger(__name__)

FORMATS = ("gbrp16le", "yuv420p10le", "png")

# The yuv420p10le master's chroma, downsampled by zscale's bilinear, pinned: decimation wants a
# low-pass, not the sharp interpolator the decode uses (DESIGN.md, Output). Sited left, the
# H.264, HEVC and MPEG-2 default, as `sptenc master` tags it.
CHROMA_KERNEL = "f=bilinear"
CHROMA_LOCATION = "left"

# What a segment command names the file it must write, replaced by that file's name.
OUT = "{out}"
# ffmpeg's exit status when what reads its output pipe is gone, 224 on Linux: fftools' main()
# returns the error transcode() gave, AVERROR(EPIPE), whose low byte the status is
# (fftools/ffmpeg.c, the end of main(), at n7.1 and n9.0.2; measured on n9.0.2 with a reader that
# reads 1,000 bytes, and with one that reads nothing, 2026-10-10). ffmpeg ignores SIGPIPE, so no
# signal ends it there.
BROKEN_PIPE = -errno.EPIPE & 0xFF
# The seconds a segment command is left to end on SIGTERM before SIGKILL (CommandWriter).
# PROVISIONAL (implementation, 2026-10-10: DESIGN.md has no figure; nothing it writes is kept, so
# the wait only serves a command that must pass the signal on or remove its own scratch).
STOP_GRACE = 2.0
# What is kept of a segment command's output for the message of its failure: its last lines,
# each cut to as many bytes.
TAIL_LINES = 20
TAIL_WIDTH = 300


@dataclass(frozen=True)
class Tags:
    """The source's primaries and transfer, which the output declares as they are: copied, never
    converted, and "" (untagged) stays untagged (DESIGN.md, Colour and shape); and the matrix the
    source is read with, which names the yuv420p10le master's BT.601 below HD (yuv_matrix). The
    names are ffprobe's, which setparams and the encoders take as they are."""

    primaries: str = ""
    transfer: str = ""
    matrix: str = ""

    @classmethod
    def of(cls, stream: VideoStream, matrix: str = "") -> Self:
        """The tags of stream, read with `matrix` (Conversion.matrix_tag)."""
        return cls(stream.color_primaries, stream.color_transfer, matrix)

    def setparams(self) -> list[str]:
        """setparams' options for the tags declared."""
        return [
            *([f"color_primaries={self.primaries}"] if self.primaries else []),
            *([f"color_trc={self.transfer}"] if self.transfer else []),
        ]

    def options(self) -> list[str]:
        """The encoder's options for the tags declared."""
        return [
            *(["-color_primaries", self.primaries] if self.primaries else []),
            *(["-color_trc", self.transfer] if self.transfer else []),
        ]


# BT.601's two names, which have the same coefficients.
BT601 = ("bt470bg", "smpte170m")


def yuv_matrix(width: int, height: int, source: str = "") -> str:
    """The yuv420p10le master's matrix, ffprobe's name, for a source read with the matrix `source`
    (DESIGN.md, Colour and shape): BT.709 at HD sizes, whatever the source's, since players read
    the tag or assume BT.709 there. Below HD, by mpv's rule, the one the decode guesses an
    untagged matrix by (DESIGN.md, Input), BT.601: tagged as the source is when it says bt470bg or
    smpte170m, else smpte170m."""
    if width >= 1280 or height > 576:
        return "bt709"
    return source if source in BT601 else "smpte170m"


# zscale's names for the matrices yuv_matrix gives.
ZSCALE_MATRICES = {"bt709": "709", "bt470bg": "470bg", "smpte170m": "170m"}


# A frame as the writers take it: (H, W, 3) float32 in [0, 1], or uint16 as quantise makes it
# of one, taken as it is (runtime/run.py's Kept: a shot's frames kept between its runs).
Frame = npt.NDArray[np.float32] | npt.NDArray[np.uint16]


def to_planar16(frame: Frame) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to the G, B, R planes of gbrp16le, round(x * 65535); uint16 as it
    is, quantise's already."""
    return np.ascontiguousarray(_quantised(frame).transpose(2, 0, 1)[[1, 2, 0]])


def to_rgb48be(frame: Frame) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to packed big-endian rgb48be, round(x * 65535), uint16 as it is:
    the 16-bit PNG encoder's own format, so that ffmpeg converts nothing. In C order, whatever the
    frame's: the decode's frames are views of (C, t, H, W) planes."""
    return _quantised(frame).astype(">u2", order="C")


def quantise(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to the 16-bit values every writer is fed: those of gbrp16le's
    planes (to_planar16) and of rgb48be (to_rgb48be), which a yuv420p10le master is converted from
    by ffmpeg (FFV1Writer) and a segment command fed (CommandWriter)."""
    # ffv1_out.py's to_planar, in float32: clipped, scaled, rounded to nearest (milestone 1).
    q = np.clip(frame, 0.0, 1.0)
    q *= 65535
    np.rint(q, out=q)
    return q.astype("<u2")


def _quantised(frame: Frame) -> npt.NDArray[np.uint16]:
    """frame's 16-bit values: quantise's of a float frame, a uint16 frame's own."""
    if frame.dtype == np.uint16:
        return frame.astype(np.uint16, copy=False)
    return quantise(frame.astype(np.float32, copy=False))


class Writer:
    """An ffmpeg process writing the frames it is fed on stdin, (n, H, W, 3) float32 in [0, 1] at a
    time. Used as a context manager: closed and checked on a normal exit; on an exception, ffmpeg
    is stopped and its partial output removed.

    checksums: the CRC-32 of each frame as the output holds it, in order (media/checksums.py;
    DESIGN.md, Output, Checksums). Computed as each frame is written on the planes fed to ffmpeg,
    which a gbrp16le master, split's input copy and a PNG hold as they are; a yuv420p10le master's
    come from ffmpeg once it is closed (FFV1Writer)."""

    # Whether the output holds the planes fed to ffmpeg, so that checksums are theirs.
    _holds_as_fed = True

    def __init__(
        self,
        command: list[str],
        what: Path,
        width: int,
        height: int,
        stdout: int = subprocess.DEVNULL,
    ) -> None:
        self.what, self.width, self.height = what, width, height
        self.written = 0
        self.checksums: list[int] = []
        self._command = command
        # In a process group of its own, out of reach of the terminal's Ctrl-C, which only
        # seedvr2x handles: it lets the segment being written finish (runtime/stop.py).
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=subprocess.PIPE,
            process_group=0,
        )
        assert self._process.stdin is not None and self._process.stderr is not None
        self._stdin = self._process.stdin
        self._errors: deque[str] = deque(maxlen=20)
        # stderr is drained as it comes: a full pipe would stop ffmpeg, and this writer with it.
        self._drain = threading.Thread(
            target=self._read_errors, args=(self._process.stderr,), daemon=True
        )
        self._drain.start()

    def write(self, frames: npt.NDArray[np.float32] | npt.NDArray[np.uint16]) -> None:
        """Write frames (n, H, W, 3) float32 in [0, 1], values outside clipped; or uint16, the
        values quantise gives, taken as they are."""
        if frames.ndim != 4 or frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(f"frames {frames.shape}, the output is {self.width}x{self.height}")
        for frame in frames:
            packed = memoryview(self._pack(frame)).cast("B")
            try:
                self._stdin.write(packed)
            except BrokenPipeError:
                raise self._broken() from None
            if self._holds_as_fed:
                self.checksums.append(zlib.crc32(packed))
            self.written += 1

    def close(self) -> int:
        """Finish the output and check it; returns the frames written."""
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        status = self._process.wait()
        self._drain.join()
        if status != 0:
            raise self._error(f"exit status {status}")
        self._finish()
        return self.written

    def abort(self) -> None:
        """Stop ffmpeg and remove what it wrote."""
        self._process.kill()
        self._process.wait()
        # Closed here, whatever is left in its buffer: flushed later, it would raise in the
        # garbage collector.
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        self._discard()

    def _pack(self, frame: Frame) -> npt.NDArray[np.uint16]:
        raise NotImplementedError

    def _finish(self) -> None:
        """Check the output ffmpeg finished, and put it in place."""

    def _discard(self) -> None:
        """Remove a partial output."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self.close()
        else:
            self.abort()

    def _read_errors(self, stream: IO[bytes]) -> None:
        for line in stream:
            self._errors.append(line.decode(errors="replace").strip())

    def _broken(self) -> MediaError:
        """The error of a write that found ffmpeg's stdin closed."""
        return self._error(f"it stopped after {self.written} frames")

    def _error(self, what: str) -> MediaError:
        self.abort()
        self._drain.join(timeout=5)
        errors = " / ".join(self._errors)
        return MediaError(
            f"writing {self.what} with ffmpeg: {what}" + (f": {errors}" if errors else "")
        )


def master_filters(pix_fmt: str, matrix: str, tags: Tags) -> tuple[str, list[str]]:
    """The filter chain from gbrp16le frames to an FFV1 master's pix_fmt, and the encoder's options
    for its tags; `matrix` is the yuv420p10le master's (yuv_matrix)."""
    if pix_fmt == "gbrp16le":
        # Frame properties only, no pixel conversion: without them the muxer writes no primaries
        # and no transfer (ffv1_out.py).
        chain = _setparams("colorspace=gbr", "range=pc", *tags.setparams())
        return chain, ["-colorspace", "rgb", "-color_range", "pc", *tags.options()]
    if pix_fmt == "yuv420p10le":
        # ffv1_out.py's conversion with every parameter given: the primaries and transfer the same
        # on both sides, so never converted (as the decode does); no dither, rounded to nearest.
        to_yuv = zscale(
            "min=gbr:rin=full",
            f"m={ZSCALE_MATRICES[matrix]}:r=limited:c={CHROMA_LOCATION}",
            "pin=unspecified:p=unspecified:tin=unspecified:t=unspecified",
            "d=none",
            CHROMA_KERNEL,
        )
        chain = f"{to_yuv},format=yuv420p10le," + _setparams(
            f"colorspace={matrix}",
            "range=tv",
            f"chroma_location={CHROMA_LOCATION}",
            *tags.setparams(),
        )
        options = [
            *("-colorspace", matrix, "-color_range", "tv"),
            *("-chroma_sample_location", CHROMA_LOCATION, *tags.options()),
        ]
        return chain, options
    raise ValueError(f"no FFV1 master in {pix_fmt}")


def png_filters(tags: Tags) -> tuple[str, list[str]]:
    """The filter chain from rgb48be frames to 16-bit PNG, and the encoder's options for its tags.
    The PNG encoder writes the primaries and transfer as cICP, cHRM and gAMA chunks, and none for
    an untagged source."""
    chain = _setparams("colorspace=gbr", "range=pc", *tags.setparams())
    return chain, ["-colorspace", "rgb", "-color_range", "pc", *tags.options()]


class FFV1Writer(Writer):
    """An FFV1 master, gbrp16le or yuv420p10le, written to a temporary file beside path, then
    checked (its frames counted by ffprobe) and renamed: a file at path is always whole, even
    after a power cut (files.replace_whole).

    A yuv420p10le master holds ffmpeg's conversion of the planes it is fed, so its checksums come
    from ffmpeg: a second output of the same process hashes the converted frames, framehash's
    CRC-32 of each raw frame being zlib's (tests/test_writer.py). stale, when given, is the file
    of the checksums of what path held before: removed just before path is replaced, so that
    checksums beside a file are always its own, and a run stopped before keeps the old ones."""

    def __init__(
        self,
        path: Path,
        pix_fmt: str,
        width: int,
        height: int,
        frame_rate: Fraction,
        tags: Tags,
        slices: int = 16,
        stale: Path | None = None,
    ) -> None:
        self.path, self._stale = path, stale
        self._partial = partial_path(path)
        self._holds_as_fed = pix_fmt == "gbrp16le"
        self._hashes: Path | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        chain, options = master_filters(pix_fmt, yuv_matrix(width, height, tags.matrix), tags)
        if self._holds_as_fed:
            graph, hashed = ["-map", "0:v:0", "-vf", chain], []
        else:
            try:
                descriptor, name = tempfile.mkstemp(suffix=".framehash")
            except OSError as error:
                raise MediaError(f"{path}: no temporary file for its checksums: {error}") from error
            os.close(descriptor)
            self._hashes = Path(name)
            graph = ["-filter_complex", f"[0:v]{chain},split=2[master][hashed]", "-map", "[master]"]
            hashed = [
                *("-map", "[hashed]", "-fps_mode", "passthrough", *options),
                *("-c:v", "rawvideo", "-pix_fmt", pix_fmt),
                *("-f", "framehash", "-hash", "crc32", str(self._hashes)),
            ]
        command = [
            *_planes_input(width, height, frame_rate),
            *graph,
            *_ffv1_matroska(pix_fmt, slices, options),
            str(self._partial),
            *hashed,
        ]
        try:
            super().__init__(command, path, width, height)
        except BaseException:
            if self._hashes is not None:
                self._hashes.unlink(missing_ok=True)
            raise

    def _pack(self, frame: Frame) -> npt.NDArray[np.uint16]:
        return to_planar16(frame)

    def _finish(self) -> None:
        counted = count_packets(self._partial)
        if counted != self.written:
            self._discard()
            raise MediaError(f"{self.path}: {counted} frames in the file, {self.written} written")
        if self._hashes is not None:
            try:
                self.checksums = _framehash(self._hashes)
            except MediaError:
                self._discard()
                raise
            self._hashes.unlink()
            if len(self.checksums) != self.written:
                self._discard()
                raise MediaError(
                    f"{self.path}: {len(self.checksums)} frames hashed, {self.written} written"
                )
        if self._stale is not None:
            self._stale.unlink(missing_ok=True)
        replace_whole(self._partial, self.path)

    def _discard(self) -> None:
        self._partial.unlink(missing_ok=True)
        if self._hashes is not None:
            self._hashes.unlink(missing_ok=True)


def _planes_input(width: int, height: int, frame_rate: Fraction) -> list[str]:
    """ffmpeg's command up to its input, the gbrp16le planes an FFV1 writer is fed on stdin, its
    frames numbered at frame_rate, frames per second."""
    rate = str(frame_rate)
    return [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"),
        *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{width}x{height}"),
        # The rate twice. -framerate is the raw video demuxer's, which numbers its packets at it;
        # the rate the filters get, and from them the encoder's time base and Matroska's frame
        # duration, is still the one libavformat guesses for the stream among its standard rates
        # once the time base is "unreliable", from 101 fps up and under 5 (n9.0.2:
        # libavformat/demux.c:2311-2333, 2287-2303 and 2411-2442; fftools/ffmpeg_demux.c:1085):
        # 120/1 for 120000/1001, 240/1 for 240000/1001. Such a master declared 120/1, its
        # timestamps on that rate's grid, up to 4.49 ms off 120000/1001's timeline (2.58 ms at
        # 240000/1001) with a place skipped at frame 500, a 17 ms step, and seedvr2x read it back
        # as 1,101 frames for 1,100, "1 frame missing after frame 499"; at 119 fps, 9 places
        # skipped, and at 121 two frames on one millisecond, read back as a variable rate.
        # -r as an input option is ffmpeg's own: the decoded frames are numbered at the rate
        # given, which the filters get as it is (fftools/ffmpeg_dec.c:399-402,
        # ffmpeg_demux.c:1081-1083). Measured through each writer, 2026-10-12: with it, 1,100
        # frames at 28 rates from 3 to 1,000 fps sit on their rate's timeline to the millisecond
        # tick, Matroska's frame duration that rate's; and the files at 24000/1001, 24, 25,
        # 30000/1001, 50 and 60000/1001 are those written without it byte for byte (30 and 1,100
        # frames at 32x16, 3 at 1280x720 and at 1920x1080), but for the 44 bytes ffmpeg draws
        # for every Matroska file: the segment's UID, the track's, held twice, and the three
        # CRC-32s over them. -r alone writes the same files; -framerate stays, so that the
        # demuxer's own packets are at the rate too, not at its default 25.
        *("-framerate", rate, "-r", rate, "-i", "-"),
    ]


def _ffv1_matroska(pix_fmt: str, slices: int, options: list[str]) -> list[str]:
    """ffmpeg's options for an output holding a master's stream, its name left to give: FFV1 as
    research/scripts/ffv1_out.py writes it, in Matroska, with the tags' options (master_filters).
    One list for the master's file and for a segment command's stdin (CommandWriter), which is
    "in the master's format" (DESIGN.md, Output) by being this very stream."""
    return [
        *("-fps_mode", "passthrough", *options),
        *("-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", str(slices)),
        *("-slicecrc", "1", "-pix_fmt", pix_fmt, "-f", "matroska"),
    ]


def _framehash(path: Path) -> list[int]:
    """The CRC-32s of ffmpeg's framehash output at path, a frame per line after its # comments:
    stream, dts, pts, duration, size, hash."""
    checksums: list[int] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or fields[0] != "0" or len(fields[5]) != 8:
            raise MediaError(f"{path}: not ffmpeg's framehash: {line[:60]!r}")
        checksums.append(int(fields[5], 16))
    return checksums


class PNGWriter(Writer):
    """16-bit RGB PNG, one file per frame, directory/NNNNNN.png numbered from start, written to a
    temporary directory beside it, then checked (every frame there) and renamed: a directory at
    path is always whole, even after a power cut (files.replace_whole). directory must not exist,
    or be empty. stale: as FFV1Writer's."""

    def __init__(
        self,
        directory: Path,
        width: int,
        height: int,
        frame_rate: Fraction,
        tags: Tags,
        start: int = 0,
        stale: Path | None = None,
    ) -> None:
        self.directory, self.start, self._stale = directory, start, stale
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise MediaError(f"{directory}: exists, and is not an empty directory")
        self._partial = partial_path(directory)
        # What an interrupted writer left: frames of a segment never finished.
        shutil.rmtree(self._partial, ignore_errors=True)
        self._partial.mkdir(parents=True)
        chain, options = png_filters(tags)
        command = [
            *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"),
            *("-f", "rawvideo", "-pix_fmt", "rgb48be", "-s", f"{width}x{height}"),
            # No -r beside -framerate, as the masters' input has (_planes_input): a PNG holds
            # neither a rate nor a timestamp, and each frame is a file whatever time ffmpeg gives
            # it: 1,100 files of the same bytes at 16 rates, 24000/1001 to 240000/1001 with 119
            # and 121, with the option and without (2026-10-12).
            *("-framerate", str(frame_rate), "-i", "-", "-map", "0:v:0"),
            *("-fps_mode", "passthrough", "-vf", chain, *options),
            *("-c:v", "png", "-pix_fmt", "rgb48be", "-f", "image2"),
            *("-start_number", str(start), str(self._partial / "%06d.png")),
        ]
        super().__init__(command, directory, width, height)

    def frame_path(self, index: int) -> Path:
        return self.directory / f"{index:06d}.png"

    def _pack(self, frame: Frame) -> npt.NDArray[np.uint16]:
        return to_rgb48be(frame)

    def _finish(self) -> None:
        missing = [
            index
            for index in range(self.start, self.start + self.written)
            if not (self._partial / self.frame_path(index).name).is_file()
        ]
        if missing:
            self._discard()
            raise MediaError(f"{self.directory}: {len(missing)} PNG missing, from {missing[0]}")
        if self._stale is not None:  # as FFV1Writer's
            self._stale.unlink(missing_ok=True)
        replace_whole(self._partial, self.directory)

    def _discard(self) -> None:
        shutil.rmtree(self._partial, ignore_errors=True)


class _Tail:
    """The last lines of what a process writes, as a terminal would leave them, in bounded memory
    whatever it writes: TAIL_LINES lines of TAIL_WIDTH bytes at most. A line ends at a newline or
    at a carriage return, and one ended by a carriage return alone is replaced by the next, as on
    a terminal: an encoder's progress line, megabytes without a newline over a segment, counts
    for one line and never hides the error printed after it."""

    def __init__(self) -> None:
        self._kept: deque[bytes] = deque(maxlen=TAIL_LINES)
        self._line = bytearray()
        self._cut = False  # the line being read is longer than TAIL_WIDTH
        self._replaced = False  # the last line kept ended at a carriage return
        # Read by the writer while a process the command left behind may still write.
        self._lock = threading.Lock()

    def feed(self, data: bytes) -> None:
        with self._lock:
            start = 0
            for end in _LINE_ENDS.finditer(data):
                self._extend(data[start : end.start()])
                self._end(end.group())
                start = end.end()
            self._extend(data[start:])

    def lines(self) -> list[str]:
        with self._lock:
            kept = list(self._kept)
            last = self._text()
            if last and self._replaced and kept:
                kept[-1] = last
            elif last:
                kept.append(last)
        return [line.decode(errors="replace") for line in kept[-TAIL_LINES:]]

    def _extend(self, data: bytes) -> None:
        room = TAIL_WIDTH - len(self._line)
        self._line += data[:room]
        self._cut = self._cut or len(data) > room

    def _text(self) -> bytes:
        text = bytes(self._line).strip()
        return text + b" [...]" if self._cut else text

    def _end(self, by: bytes) -> None:
        line = self._text()
        self._line.clear()
        self._cut = False
        if line:
            if self._replaced and self._kept:
                self._kept[-1] = line
            else:
                self._kept.append(line)
            self._replaced = by == b"\r"
        elif by == b"\n":
            # \r\n ends one line, which stays.
            self._replaced = False


_LINE_ENDS = re.compile(rb"[\r\n]")


def check_command(command: str) -> None:
    """Raise MediaError unless command can be a segment command (CommandWriter): not empty, and
    naming {out}, the file it must write, without which it would write nowhere seedvr2x looks."""
    if not command.strip():
        raise MediaError("the segment command is empty")
    if OUT not in command:
        raise MediaError(
            f"the segment command doesn't name {OUT}, the file it must write: {command!r}; write"
            f" {OUT} where its output's name goes, e.g. ffmpeg -i - -c:v libx265 -crf 16 {OUT}"
        )


class CommandWriter(Writer):
    """A segment written by the user's command, --segment-cmd (DESIGN.md, Output, Writer): the
    command reads the segment on stdin and writes the file seedvr2x names, which is then checked,
    its frames decoded and counted by ffprobe, and renamed to path: a file at path is always
    whole, as FFV1Writer's. pix_fmt is an FFV1 master's, gbrp16le or yuv420p10le; stale, as
    FFV1Writer's. No frame has a checksum: the frames are the command's encoder's.

    The stream on stdin is the very one FFV1Writer's master holds, Matroska and FFV1 from the
    same options, on a pipe: its packets are the master's byte for byte, and ffprobe reads from
    the pipe every tag as from the master's file, the pixel format, range, matrix, primaries,
    transfer, chroma location and the exact frame rate (measured on ffmpeg n9.0.2, 2026-10-10;
    tests/test_segment_cmd.py). A Matroska that can't be seeked lacks its duration and its cues,
    which ffmpeg reads without a warning; its timestamps are milliseconds, as the master's. As
    the master's file, it declares no sample aspect (ffprobe: N/A). The other streams ffmpeg
    reads from a pipe lose tags on the way, measured the same day: NUT the range, matrix,
    primaries, transfer and chroma location, with FFV1 as with raw video; yuv4mpegpipe all of
    them but the range, and it takes no RGB.

    The command is run by /bin/sh -c, never parsed. Each {out} in it is replaced by the name of
    the file to write, quoted for the shell: path with `.partial` before its suffix
    (files.command_partial), which doesn't exist when the command starts. Its stdout and stderr
    are read as they come and only their last lines kept, for the message of a failure.

    Nothing times the command: an encoder may be as slow as it likes, and one that stops for good
    stops the run with it, until a Ctrl-C twice or a SIGTERM (runtime/stop.py). It runs in a
    process group of its own, out of the terminal's reach as ffmpeg is, by which abort() stops
    the shell and whatever it started."""

    _holds_as_fed = False

    def __init__(
        self,
        command: str,
        path: Path,
        pix_fmt: str,
        width: int,
        height: int,
        frame_rate: Fraction,
        tags: Tags,
        slices: int = 16,
        stale: Path | None = None,
    ) -> None:
        check_command(command)
        chain, options = master_filters(pix_fmt, yuv_matrix(width, height, tags.matrix), tags)
        self.path, self._stale = path, stale
        self._partial = command_partial(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # What a run killed left there. The command never finds its output's name taken: ffmpeg
        # n9.0.2 reading stdin, without -y, says "File '...' already exists. Exiting." and exits
        # with status 0 (2026-10-10), the old file left as it was. Removed, not written over: a
        # command of that run still writing it then keeps a file of its own, without a name.
        self._partial.unlink(missing_ok=True)
        # PROVISIONAL (implementation, 2026-10-10: DESIGN.md says "the file seedvr2x names" and
        # its example writes {out} bare): every {out} replaced, quoted, as GNU parallel replaces
        # {}; nothing else of the command is read.
        self._run = command.replace(OUT, _word(self._partial))
        self._tail = _Tail()
        self._output = threading.Thread(target=self._read_output, daemon=True)
        stream = [
            *_planes_input(width, height, frame_rate),
            *("-map", "0:v:0", "-vf", chain),
            *_ffv1_matroska(pix_fmt, slices, options),
            "pipe:1",
        ]
        super().__init__(stream, path, width, height, stdout=subprocess.PIPE)
        assert self._process.stdout is not None
        try:
            # A group of its own, not ffmpeg's: abort() signals it whole, the shell and what it
            # started, and ffmpeg must not be among them.
            self._shell = subprocess.Popen(
                ["/bin/sh", "-c", self._run],
                stdin=self._process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                process_group=0,
            )
        except BaseException as error:
            Writer.abort(self)
            if isinstance(error, OSError):
                raise MediaError(f"{path}: the segment command can't be run: {error}") from error
            raise
        finally:
            # The command alone reads the pipe: once it is gone ffmpeg's writes fail (EPIPE)
            # rather than wait for a reader, and ffmpeg's end is the command's end of file.
            self._process.stdout.close()
        try:
            # Read as it comes, as ffmpeg's stderr: a full pipe would stop the command.
            self._output.start()
        except BaseException:
            self.abort()
            raise
        logger.debug("%s: the segment command, run by /bin/sh -c: %s", path, self._run)

    def close(self) -> int:
        """Finish the stream, wait for the segment command to end, however long it takes, check
        the file it wrote and put it in place; returns the frames written."""
        try:
            try:
                self._stdin.close()
            except BrokenPipeError:
                pass
            status = self._process.wait()
            self._drain.join()
            ended = self._ended()
        except BaseException:
            # A stop at once (runtime/stop.py) while the command finishes: it is stopped too.
            self.abort()
            raise
        # The command's failure first: ffmpeg's broken pipe is its consequence.
        if ended != 0:
            raise self._refused(f"{self.path}: the segment command {_ending(ended)}")
        # A command that ends well before the stream's end leaves ffmpeg a broken pipe: only the
        # frames of its file say whether it took them all (DESIGN.md: "only checks the result's
        # frame count"). Any other failure of ffmpeg is its own.
        if status not in (0, BROKEN_PIPE):
            raise Writer._error(self, f"exit status {status}")
        try:
            self._finish()
        except BaseException:
            self.abort()
            raise
        self._shell.wait()
        return self.written

    def abort(self) -> None:
        """Stop the segment command, with whatever it started, and ffmpeg; remove what was
        written."""
        try:
            self._stop()
        finally:
            # After the command: one still running could write the file again once removed.
            super().abort()

    def _pack(self, frame: Frame) -> npt.NDArray[np.uint16]:
        return to_planar16(frame)

    def _finish(self) -> None:
        if not self._partial.is_file():
            raise self._refused(
                f"{self.path}: the segment command exited with status 0 without writing"
                f" {self._partial}, the file its {OUT} names"
            )
        try:
            counted = count_frames(self._partial)
        except MediaError as error:
            raise self._refused(str(error)) from error
        if counted != self.written:
            raise self._refused(
                f"{self.path}: {counted} frames in the file the segment command wrote,"
                f" {self.written} written"
            )
        # The old file's checksums go once the new file's data is on the disk
        # (files.replace_whole).
        replace_whole(self._partial, self.path, stale=self._stale)

    def _discard(self) -> None:
        self._partial.unlink(missing_ok=True)

    def _read_output(self) -> None:
        assert self._shell.stdout is not None
        with self._shell.stdout as stream:
            while data := os.read(stream.fileno(), 1 << 16):
                self._tail.feed(data)

    def _status(self) -> int | None:
        """The shell's status once it has ended, a signal's number negated as Popen's; None while
        it runs. Read without reaping it: until it is reaped its pid, its group's id, is no
        other process's, so that what it left running can still be stopped by the group."""
        if self._shell.returncode is not None:
            return self._shell.returncode
        ended = os.waitid(os.P_PID, self._shell.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        if ended is None:
            return None
        return ended.si_status if ended.si_code == os.CLD_EXITED else -ended.si_status

    def _ended(self) -> int:
        """Wait for the shell to end, however long it takes; its status, as _status gives it."""
        while (status := self._status()) is None:
            os.waitid(os.P_PID, self._shell.pid, os.WEXITED | os.WNOWAIT)
        return status

    def _over(self, seconds: float) -> None:
        """Wait `seconds` at most for the command to be over: its shell ended, and its output
        closed by every process that held it, those the shell started among them."""
        deadline = time.monotonic() + seconds
        while self._status() is None or self._output.is_alive():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

    def _stop(self) -> None:
        """Stop the shell and whatever it started that stayed in its group: SIGTERM, then SIGKILL
        once they are over, or after STOP_GRACE seconds.

        Not SIGKILL at once, as ffmpeg gets: nothing the command writes is kept either, but the
        command is the user's, and may be a script with scratch of its own to remove or the
        client of an encoder running elsewhere (a container, another machine), which a signal it
        can't catch would leave behind. PROVISIONAL (implementation, 2026-10-10: DESIGN.md's
        "stops at once" names no signal)."""
        if self._shell.returncode is not None:
            return  # reaped, by close() or a first abort()
        try:
            self._signal(signal.SIGTERM)
            self._over(STOP_GRACE)
        finally:
            self._signal(signal.SIGKILL)
            self._shell.wait()

    def _signal(self, number: int) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._shell.pid, number)

    def _broken(self) -> MediaError:
        # ffmpeg no longer reads: it failed, or the command no longer reads it. Each is left
        # STOP_GRACE seconds to end by itself, so that the status reported is its own and not the
        # signal of abort().
        try:
            status = self._process.wait(timeout=STOP_GRACE)
        except subprocess.TimeoutExpired:
            return super()._broken()
        if status != BROKEN_PIPE:
            return Writer._error(self, f"exit status {status}")
        self._over(STOP_GRACE)
        ended = self._status()
        how = "stopped reading its input" if ended is None else _ending(ended)
        return self._refused(
            f"{self.path}: the segment command {how} before the segment's end,"
            f" {self.written} frames written"
        )

    def _refused(self, what: str) -> MediaError:
        """The error of a segment refused for `what`, once everything is stopped and its file
        removed: with the command's last lines, and ffmpeg's own when it failed otherwise than
        by the command's end."""
        status = self._process.poll()
        self.abort()
        self._output.join(timeout=5)
        self._drain.join(timeout=5)
        said = " / ".join(self._tail.lines())
        message = what + (f": {said}" if said else "")
        if status is not None and status not in (0, BROKEN_PIPE):
            errors = " / ".join(self._errors)
            message += f"; ffmpeg, which feeds it, failed too: exit status {status}" + (
                f": {errors}" if errors else ""
            )
        return MediaError(message)


def _word(path: Path) -> str:
    """path as one word of a shell command, which no command reads as an option."""
    name = str(path)
    return shlex.quote(f"./{name}" if name.startswith("-") else name)


def _ending(status: int) -> str:
    """How a process ended, from its status as Popen gives it."""
    if status >= 0:
        return f"exited with status {status}"
    try:
        return f"was ended by {signal.Signals(-status).name}"
    except ValueError:
        return f"was ended by signal {-status}"


class SegmentWriter:
    """The output written across consecutive segments, each to a writer of its own: opened at the
    segment's first frame, closed, so checked, after its last (DESIGN.md, Output: output
    segments). segments are (path, frames) in order; open_segment(path) gives a segment's writer;
    finished(index, checksums), when given, is told each segment closed, with its frames'
    checksums (Writer.checksums). Used as a context manager, as a Writer."""

    def __init__(
        self,
        segments: Sequence[tuple[Path, int]],
        open_segment: Callable[[Path], Writer],
        finished: Callable[[int, list[int]], None] | None = None,
    ) -> None:
        self._segments = list(segments)
        self._open = open_segment
        self._finished = finished
        self._index = 0
        self._current: Writer | None = None
        self.written = 0

    def write(self, frames: npt.NDArray[np.float32] | npt.NDArray[np.uint16]) -> None:
        """Write frames (n, H, W, 3) float32 in [0, 1], or uint16 as quantise gives them, the next
        of the output."""
        while frames.shape[0]:
            if self._index == len(self._segments):
                raise ValueError(f"{frames.shape[0]} frames beyond the output's segments")
            path, count = self._segments[self._index]
            if self._current is None:
                self._current = self._open(path)
            take = min(count - self._current.written, frames.shape[0])
            self._current.write(frames[:take])
            frames = frames[take:]
            self.written += take
            if self._current.written == count:
                self._current.close()
                checksums, self._current = self._current.checksums, None
                if self._finished is not None:
                    self._finished(self._index, checksums)
                self._index += 1

    def close(self) -> int:
        """Check that every segment is whole; returns the frames written."""
        if self._index != len(self._segments):
            self.abort()
            raise RuntimeError(
                f"output stopped after {self.written} frames, in segment {self._index + 1} of"
                f" {len(self._segments)}"
            )
        return self.written

    def abort(self) -> None:
        """Stop the segment being written and remove it; the segments closed are whole."""
        if self._current is not None:
            self._current.abort()
            self._current = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self.close()
        else:
            self.abort()


def open_writer(
    output_format: str,
    path: Path,
    width: int,
    height: int,
    frame_rate: Fraction,
    tags: Tags,
    start: int = 0,
    stale: Path | None = None,
    command: str | None = None,
) -> Writer:
    """A writer of output_format (FORMATS): an FFV1 master at path, or PNG in the directory path,
    numbered from start; stale, the checksums of what path held before (FFV1Writer). With
    command, a segment command (CommandWriter), the file at path is the one it writes from the
    FFV1 master's stream: PNG has no stream to give it."""
    if command is not None:
        if output_format == "png":
            raise ValueError("no segment command with PNG: it reads an FFV1 master's stream")
        return CommandWriter(
            command, path, output_format, width, height, frame_rate, tags, stale=stale
        )
    if output_format == "png":
        return PNGWriter(path, width, height, frame_rate, tags, start, stale)
    return FFV1Writer(path, output_format, width, height, frame_rate, tags, stale=stale)


def count_packets(path: Path) -> int:
    """The packets of the first video stream of path, read without decoding: its frames, for
    FFV1, where every frame is one."""
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets"),
            *("-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=False,
        process_group=0,  # as the writers' ffmpeg: it checks the segment being finished
    )
    if result.returncode != 0:
        raise MediaError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    return int(result.stdout.strip() or 0)


def count_frames(path: Path, strict: bool = False) -> int:
    """The frames of the first video stream of path, decoded and counted by ffprobe: those of a
    file a segment command wrote, in any codec and container. What ffprobe reports while it
    decodes, the count being right, is warned of; with `strict`, it fails the count, as it
    fails `seedvr2x verify`'s check of a master (DESIGN.md, Output, Checksums: "Anything ffmpeg
    reports while decoding fails the check").

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md says "only checks the result's frame
    count", not how): decoded frames, not packets. A packet isn't a frame in every codec and
    container, and the decode also shows that the file reads to its end: of a file cut at 56%,
    ffprobe counts 168 frames of 300 and exits 0. The price is a decoder for whatever the command
    wrote, and the decode's time. Measured on 2026-10-10 (ffmpeg n9.0.2, a Ryzen 9 9950X3D, 32
    threads), 300 frames of 1080p: a noisy libx265 -crf 16 (68 Mb/s) in 2.1-2.8 s, a clean one
    (12 Mb/s) in 1.0 s, mpeg4 -q:v 2 -bf 2 (97 Mb/s) in 0.3 s; their packets in 0.01-0.02 s.
    That is with -threads 0: ffprobe otherwise decodes on one thread (it leaves libavcodec's
    default, where ffmpeg asks for all: fftools/ffprobe.c sets `threads` for -show_log only) and
    takes 11.4-12.2 s, 2.1-2.3 s and 1.5-2.1 s. Against about 4 s of GPU time per 1080p frame
    (media/ffmpeg.py), 0.2% of the noisy segment's time, and 1% on one thread.

    Raises MediaError when ffprobe can't read the file, when it holds no video stream, when its
    frames can't be decoded, and with `strict` when ffprobe reports anything."""
    result = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-threads", "0", "-select_streams", "v:0"),
            *("-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"),
            *("-i", str(path)),  # by -i: a name the command was given as ./-x is no option here
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        process_group=0,  # as count_packets'
    )
    # The last lines only: a decoder that fails says so at every frame.
    lines = [line.strip()[:TAIL_WIDTH] for line in result.stderr.splitlines() if line.strip()]
    said = " / ".join(lines[-TAIL_LINES:])
    if result.returncode != 0:
        raise MediaError(
            f"{path}: ffprobe can't read the file the segment command wrote"
            + (f": {said}" if said else "")
        )
    counted = result.stdout.split()
    if not counted:
        raise MediaError(f"{path}: no video stream in the file the segment command wrote")
    # N/A, with exit status 0: no decoder took the stream. Measured on ffmpeg n9.0.2, 2026-10-10:
    # of a codec it doesn't know, 250 packets and nothing said; of AV1 by its own decoder, which
    # needs hardware, "Your platform doesn't support hardware accelerated AV1 decoding." at each
    # frame.
    if not counted[0].isdecimal():
        raise MediaError(
            f"{path}: ffprobe can't decode the file the segment command wrote, to count its"
            f" frames ({counted[0]})" + (f": {said}" if said else "")
        )
    if said and strict:
        raise MediaError(f"{path}: ffprobe, decoding it to count its frames, reported: {said}")
    if said:
        logger.warning("%s: ffprobe, counting its frames, reported: %s", path, said)
    return int(counted[0])


def _setparams(*options: str) -> str:
    return "setparams=" + ":".join(options)
