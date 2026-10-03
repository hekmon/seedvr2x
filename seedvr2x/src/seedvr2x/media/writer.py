"""The output, written as it comes: FFV1 masters, gbrp16le or yuv420p10le, and 16-bit PNG, each
through an ffmpeg pipe fed the float frames (DESIGN.md, Output).

The quantisation and the FFV1 settings are those of research/scripts/ffv1_out.py, the master
validated in research/docs/output.md: 16-bit RGB rounded to nearest, FFV1 level 3, every frame a
keyframe, 16 slices with CRCs, the exact rational frame rate. The tags follow DESIGN.md (Colour and
shape): the source's primaries and transfer copied, never converted; the yuv420p10le master's
matrix BT.709 at HD sizes.

ffmpeg n9 converts frames whose tags differ from those asked of the encoder (an untagged frame to
BT.709, for one). So every tag is set twice, on the frames (setparams) and on the encoder, and
nothing converts but zscale.
"""

import os
import shutil
import subprocess
import tempfile
import threading
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

from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import partial_path, replace_whole
from seedvr2x.media.probe import VideoStream

FORMATS = ("gbrp16le", "yuv420p10le", "png")

# The yuv420p10le master's chroma, downsampled by zscale's bilinear, pinned: decimation wants a
# low-pass, not the sharp interpolator the decode uses (DESIGN.md, Output). Sited left, the
# H.264, HEVC and MPEG-2 default, as `sptenc master` tags it.
CHROMA_KERNEL = "f=bilinear"
CHROMA_LOCATION = "left"


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


def to_planar16(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to the G, B, R planes of gbrp16le, round(x * 65535)."""
    return np.ascontiguousarray(_quantise(frame).transpose(2, 0, 1)[[1, 2, 0]])


def to_rgb48be(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """(H, W, 3) float in [0, 1] to packed big-endian rgb48be, round(x * 65535): the 16-bit PNG
    encoder's own format, so that ffmpeg converts nothing. In C order, whatever the frame's: the
    decode's frames are views of (C, t, H, W) planes."""
    return _quantise(frame).astype(">u2", order="C")


def _quantise(frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    # ffv1_out.py's to_planar, in float32: clipped, scaled, rounded to nearest (milestone 1).
    q = np.clip(frame, 0.0, 1.0)
    q *= 65535
    np.rint(q, out=q)
    return q.astype("<u2")


class Writer:
    """An ffmpeg process writing the frames it is fed on stdin, (n, H, W, 3) float32 in [0, 1] at a
    time. Used as a context manager: closed and checked on a normal exit; on an exception, ffmpeg
    is stopped and its partial output removed.

    checksums: the CRC-32 of each frame as the output holds it, in order (media/checksums.py;
    DESIGN.md, Output, Checksums). Computed as each frame is written on the planes fed to ffmpeg,
    which a gbrp16le master, lab's input copy and a PNG hold as they are; a yuv420p10le master's
    come from ffmpeg once it is closed (FFV1Writer)."""

    # Whether the output holds the planes fed to ffmpeg, so that checksums are theirs.
    _holds_as_fed = True

    def __init__(self, command: list[str], what: Path, width: int, height: int) -> None:
        self.what, self.width, self.height = what, width, height
        self.written = 0
        self.checksums: list[int] = []
        self._command = command
        # In a process group of its own, out of reach of the terminal's Ctrl-C, which only
        # seedvr2x handles: it lets the segment being written finish (runtime/stop.py).
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
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

    def write(self, frames: npt.NDArray[np.float32]) -> None:
        """Write frames (n, H, W, 3) float32 in [0, 1]; values outside are clipped."""
        if frames.ndim != 4 or frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(f"frames {frames.shape}, the output is {self.width}x{self.height}")
        for frame in frames:
            packed = memoryview(self._pack(frame)).cast("B")
            try:
                self._stdin.write(packed)
            except BrokenPipeError:
                raise self._error(f"it stopped after {self.written} frames") from None
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

    def _pack(self, frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
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
        zscale = [
            "min=gbr:rin=full",
            f"m={ZSCALE_MATRICES[matrix]}:r=limited:c={CHROMA_LOCATION}",
            "pin=unspecified:p=unspecified:tin=unspecified:t=unspecified",
            "d=none",
            CHROMA_KERNEL,
        ]
        chain = f"zscale={':'.join(zscale)},format=yuv420p10le," + _setparams(
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
            *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-y"),
            *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{width}x{height}"),
            *("-framerate", str(frame_rate), "-i", "-", *graph),
            *("-fps_mode", "passthrough", *options),
            *("-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", str(slices)),
            *("-slicecrc", "1", "-pix_fmt", pix_fmt, "-f", "matroska", str(self._partial)),
            *hashed,
        ]
        try:
            super().__init__(command, path, width, height)
        except BaseException:
            if self._hashes is not None:
                self._hashes.unlink(missing_ok=True)
            raise

    def _pack(self, frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
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
            *("-framerate", str(frame_rate), "-i", "-", "-map", "0:v:0"),
            *("-fps_mode", "passthrough", "-vf", chain, *options),
            *("-c:v", "png", "-pix_fmt", "rgb48be", "-f", "image2"),
            *("-start_number", str(start), str(self._partial / "%06d.png")),
        ]
        super().__init__(command, directory, width, height)

    def frame_path(self, index: int) -> Path:
        return self.directory / f"{index:06d}.png"

    def _pack(self, frame: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
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

    def write(self, frames: npt.NDArray[np.float32]) -> None:
        """Write frames (n, H, W, 3) float32 in [0, 1], the next of the output."""
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
) -> Writer:
    """A writer of output_format (FORMATS): an FFV1 master at path, or PNG in the directory path,
    numbered from start; stale, the checksums of what path held before (FFV1Writer)."""
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


def _setparams(*options: str) -> str:
    return "setparams=" + ":".join(options)
