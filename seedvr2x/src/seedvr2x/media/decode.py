"""Decoding to 16-bit RGB through an ffmpeg pipe, frames read as they are needed."""

import io
import os
import subprocess
import tempfile
from collections import deque
from collections.abc import Sequence
from types import TracebackType
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from seedvr2x.media.conversion import Conversion
from seedvr2x.media.ffmpeg import MediaError


def decode_command(
    input_args: Sequence[str], conversion: Conversion, loglevel: str = "error"
) -> list[str]:
    """The ffmpeg command writing the frames to stdout as gbrp16le."""
    return [
        *("ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", loglevel),
        *input_args,
        *("-map", "0:v:0", *decode_output(conversion), "-"),
    ]


def decode_output(conversion: Conversion) -> list[str]:
    """ffmpeg's options for the decode's output: the frames converted (Conversion), raw gbrp16le."""
    return [
        *("-vf", conversion.filters(), "-fps_mode", "passthrough"),
        # A frame size changing within the stream fails, instead of being scaled to the first
        # frame's size by swscale.
        "-noautoscale",
        *("-f", "rawvideo", "-pix_fmt", "gbrp16le"),
    ]


class Decoder:
    """The frames of a video stream as 16-bit RGB, from an ffmpeg pipe, read as they are needed.

    zscale converts them with every parameter explicit (Conversion). They are counted as they
    come, never trusting the container's count (bug 11): at the end, there must be `frames` of
    them, the first pass's count, when given. Used as a context manager, the end is checked on a
    normal exit, and ffmpeg is stopped on an exception.

    strict: anything ffmpeg reports fails the read, before the frames read with it are given. That
    is how FFV1's slice CRCs are enforced: on a mismatch, ffmpeg's FFV1 decoder reports it, hides
    the slice under the frame before's and goes on, and ffmpeg exits 0, whatever its options
    (n9.0.2: -err_detect crccheck+explode and -xerror alike, the slice executor of
    libavcodec/ffv1dec.c dropping each slice's error). ffmpeg reports a frame's damage as it
    decodes it, so before writing it: in the file before the frame is in the pipe. ffmpeg's
    reports go to a temporary file: where the temporary directory is full, ffmpeg loses them
    without a word, and a strict read then fails on nothing more than a normal one."""

    def __init__(
        self,
        input_args: Sequence[str],
        conversion: Conversion,
        width: int,
        height: int,
        frames: int | None = None,
        strict: bool = False,
    ) -> None:
        self.width, self.height, self.frames = width, height, frames
        self.decoded = 0
        self._frame_bytes = width * height * 3 * 2
        self._strict = strict
        # ffmpeg's errors go to a file rather than a pipe: ffmpeg never waits on it, and its size
        # says at once whether ffmpeg has reported anything (strict).
        try:
            self._log = tempfile.TemporaryFile()
        except OSError as error:
            what = f"decoding with ffmpeg: no temporary file for its errors: {error}"
            raise MediaError(what) from error
        # In a process group of its own, out of reach of the terminal's Ctrl-C, which only
        # seedvr2x handles (runtime/stop.py).
        self._process = subprocess.Popen(
            decode_command(input_args, conversion),
            stdout=subprocess.PIPE,
            stderr=self._log,
            process_group=0,
        )
        assert self._process.stdout is not None
        # Buffered, as Popen opens it by default.
        self._stdout = cast(io.BufferedReader, self._process.stdout)

    def read(self, count: int) -> npt.NDArray[np.uint16]:
        """The next `count` frames, (n, H, W, 3) uint16 RGB: fewer only at the end of the
        stream."""
        planes = np.empty((count, 3, self.height, self.width), dtype="<u2")  # G, B, R
        view = memoryview(planes).cast("B")
        filled = 0
        while filled < len(view):
            got = self._stdout.readinto(view[filled:])
            if not got:
                break
            filled += got
        whole, rest = divmod(filled, self._frame_bytes)
        self.decoded += whole
        if rest:
            raise self._error(f"its output ends {rest} bytes into frame {self.decoded}")
        if self.frames is not None and self.decoded > self.frames:
            raise self._error(f"more frames than the {self.frames} the first pass counted")
        self._check()
        frames = planes[:whole]
        return np.stack((frames[:, 2], frames[:, 0], frames[:, 1]), axis=-1)

    def skip(self, count: int) -> int:
        """Decode the next `count` frames and drop them, counted but never converted: fewer only
        at the end of the stream. Returns the frames dropped."""
        wanted = count * self._frame_bytes
        # Each read gives at most what the pipe holds: a few MiB at a time do.
        buffer = memoryview(bytearray(min(wanted, 4 << 20)))
        filled = 0
        while filled < wanted:
            got = self._stdout.readinto(buffer[: min(len(buffer), wanted - filled)])
            if not got:
                break
            filled += got
        whole, rest = divmod(filled, self._frame_bytes)
        self.decoded += whole
        if rest:
            raise self._error(f"its output ends {rest} bytes into frame {self.decoded}")
        if self.frames is not None and self.decoded > self.frames:
            raise self._error(f"more frames than the {self.frames} the first pass counted")
        self._check()
        return whole

    def finish(self) -> None:
        """Check the end of the stream: ffmpeg done, with no error, after `frames` frames."""
        if self._stdout.read(1):
            raise self._error(f"frames left after {self.decoded}")
        if self._process.wait() != 0:
            raise self._error(f"exit status {self._process.returncode}")
        self._check()
        if self.frames is not None and self.decoded != self.frames:
            raise self._error(f"{self.decoded} frames, the first pass counted {self.frames}")
        self._log.close()

    def failure(self, what: str) -> MediaError:
        """ffmpeg stopped, and the error to raise: what, with ffmpeg's last errors."""
        return self._error(what)

    def stop(self) -> None:
        """Stop ffmpeg, whatever is left to read."""
        self._process.kill()
        self._process.wait()
        self._log.close()

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

    def _check(self) -> None:
        """When strict, fail if ffmpeg has reported anything yet."""
        if self._strict and os.fstat(self._log.fileno()).st_size:
            raise self._error("an error reported, which fails a strict read")

    def _error(self, what: str) -> MediaError:
        """ffmpeg stopped, and what to raise: what, with ffmpeg's last 20 lines of errors."""
        self._process.kill()
        self._process.wait()
        errors = ""
        if not self._log.closed:
            self._log.seek(0)  # ffmpeg is done writing at the offset the file shares with it
            lines = deque(self._log, maxlen=20)
            errors = " / ".join(line.decode(errors="replace").strip() for line in lines)
            self._log.close()
        return MediaError(f"decoding with ffmpeg: {what}" + (f": {errors}" if errors else ""))


def to_float32(frames: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """Frames of 16-bit RGB (T, H, W, 3) to float32 in [0, 1], v / 65535.

    An 8-bit source, which zscale expands to v * 257 exactly, gives numz's own read, v / 255 in
    float32, bit for bit: both are the correctly rounded value of one quotient
    (tests/test_decode.py)."""
    out = frames.astype(np.float32)
    out /= np.float32(65535)
    return out
