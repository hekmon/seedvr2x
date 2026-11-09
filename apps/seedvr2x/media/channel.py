"""A channel from an ffmpeg process to seedvr2x beside its stdout: a read's frame hashes
(framehash, framemd5: media/reader.py) and the first pass's frames for the shot detector
(media/scan.py), each read by a thread of its own.

On Linux, a pipe whose write end ffmpeg inherits by its number, `pipe:N` (Popen's pass_fds). On
Windows a child inherits no descriptor by its number: `pipe:N` there gave "Bad file descriptor"
for the C runtime's descriptor, the handle's value and a handle passed in Popen's handle_list
alike (impl-win-2, ffmpeg 9.0.1, 2026-11-09). There the channel is a named pipe of seedvr2x's
own, which ffmpeg opens by its name as a file: byte for byte what it wrote, at 4,079 MB/s for
1080p rgb48le with 1 MiB buffers (3,081 at 64 KiB, 4,101 at 16 MiB; stdout's anonymous pipe
1,471), and ffmpeg given -y for it (OVERWRITE)."""

import contextlib
import io
import itertools
import os
import secrets
import subprocess
import sys
from collections.abc import Buffer
from typing import Any, cast

from media.ffmpeg import MediaError

# ffmpeg's options for a command writing to a channel: on Windows -y, since ffmpeg takes the
# named pipe for a file that exists, asks whether to overwrite it, and with stdin closed answers
# no, "Not overwriting - exiting", status 0 and nothing written (impl-win-2, 2026-11-09). On
# Linux none: `pipe:N` is no file. None of the commands that write to a channel has an output
# that is a file.
OVERWRITE: tuple[str, ...] = ("-y",) if sys.platform == "win32" else ()

# The named pipes' buffers, each way: the throughput's knee (above).
BUFFER = 1 << 20

# winerror.h: a client that connected and closed its end (ConnectNamedPipe's), the client's end
# closed with nothing left to read (ReadFile's).
ERROR_NO_DATA = 232
ERROR_BROKEN_PIPE = 109

# CreateNamedPipe's pipe mode: bytes, read as bytes, blocking (all three 0), and no client from
# another machine (winbase.h's PIPE_REJECT_REMOTE_CLIENTS, which _winapi doesn't name).
PIPE_REJECT_REMOTE_CLIENTS = 0x8

_NUMBERS = itertools.count()


class Channel:
    """One channel, made before ffmpeg starts: `output` is the name ffmpeg writes it by, `passed`
    what Popen's pass_fds takes for it. After Popen, `started`; then `stream`, in the thread that
    reads it, its read end as a binary file, read to its end, which comes when ffmpeg closes its
    output or ends, whether or not it ever opened it. `close` frees what neither took (both ends
    where Popen failed, a read end never opened)."""

    def __init__(self) -> None:
        self._read: int | None = None
        self._write: int | None = None
        self._pipe: int | None = None
        if sys.platform == "win32":
            import _winapi

            # The name the process's own and unguessable; the first instance alone, so that no
            # other program's pipe of that name is taken for it.
            self.output = (
                f"\\\\.\\pipe\\seedvr2x-{os.getpid()}-{next(_NUMBERS)}-{secrets.token_hex(8)}"
            )
            try:
                self._pipe = _winapi.CreateNamedPipe(
                    self.output,
                    _winapi.PIPE_ACCESS_INBOUND
                    | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
                    | _winapi.FILE_FLAG_OVERLAPPED,
                    PIPE_REJECT_REMOTE_CLIENTS,
                    1,
                    BUFFER,
                    BUFFER,
                    0,
                    _winapi.NULL,
                )
            except OSError as error:
                raise MediaError(
                    f"ffmpeg's side channel {self.output} can't be made: {error}"
                ) from error
            self.passed: tuple[int, ...] = ()
        else:
            self._read, self._write = os.pipe()
            self.output = f"pipe:{self._write}"
            self.passed = (self._write,)

    def started(self) -> None:
        """ffmpeg has started: the parent's copy of the write end closed, so that the channel ends
        when ffmpeg's output does."""
        if self._write is not None:
            os.close(self._write)
            self._write = None

    def stream(
        self, process: "subprocess.Popen[Any]", buffering: int = -1
    ) -> io.RawIOBase | io.BufferedReader:
        """The read end, opened as a binary file with `buffering` (as open's), the channel's
        caller then its owner. On Windows it first waits for ffmpeg to open its end, or for
        `process` to end without having opened it: a stream at its end at once."""
        if sys.platform == "win32":
            import _winapi

            pipe, self._pipe = self._pipe, None
            assert pipe is not None
            try:
                connected = _connected(pipe, process.pid)
            except BaseException:
                _winapi.CloseHandle(pipe)
                raise
            raw = _NamedPipe(pipe, connected)
            return raw if buffering == 0 else io.BufferedReader(raw, BUFFER)
        read, self._read = self._read, None
        assert read is not None
        return cast(io.RawIOBase | io.BufferedReader, os.fdopen(read, "rb", buffering=buffering))

    def close(self) -> None:
        """Whatever of the channel nothing took."""
        for descriptor in (self._read, self._write):
            if descriptor is not None:
                os.close(descriptor)
        self._read = self._write = None
        if sys.platform == "win32" and self._pipe is not None:
            import _winapi

            _winapi.CloseHandle(self._pipe)
            self._pipe = None


def _connected(pipe: int, pid: int) -> bool:
    """Wait until the client has opened `pipe` (True) or the process `pid` has ended without
    having opened it (False). A client that opened it before the wait leaves it connected
    (ERROR_PIPE_CONNECTED), or gone already with what it wrote still to read (ERROR_NO_DATA)."""
    if sys.platform != "win32":
        raise AssertionError("a named pipe is Windows'")
    import _winapi

    try:
        connecting = _winapi.ConnectNamedPipe(pipe, overlapped=True)
    except OSError as error:
        if error.winerror == ERROR_NO_DATA:
            return True
        raise
    try:
        process = _winapi.OpenProcess(_winapi.SYNCHRONIZE, False, pid)
        try:
            # The first of the two signalled, the connection's when both are.
            first = _winapi.WaitForMultipleObjects(
                [connecting.event, process], False, _winapi.INFINITE
            )
        finally:
            _winapi.CloseHandle(process)
    except BaseException:
        # The connection cancelled and done with before its handle is closed (stream).
        connecting.cancel()
        with contextlib.suppress(OSError):
            connecting.GetOverlappedResult(True)
        raise
    if first != _winapi.WAIT_OBJECT_0:
        connecting.cancel()
    try:
        _, error = connecting.GetOverlappedResult(True)
    except OSError as failed:
        if failed.winerror == ERROR_NO_DATA:
            return True
        raise
    return error == 0  # ERROR_OPERATION_ABORTED where cancelled before a client came


class _NamedPipe(io.RawIOBase):
    """A named pipe's server end, open for overlapped reads (the connection waits on the process
    too, which an overlapped handle alone allows), read as a raw binary stream: its end where
    the client's end is closed, or where no client ever came (connected False)."""

    def __init__(self, pipe: int, connected: bool) -> None:
        super().__init__()
        self._pipe: int | None = pipe
        self._connected = connected

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer) -> int:
        if sys.platform != "win32":
            raise AssertionError("a named pipe is Windows'")
        import _winapi

        view = memoryview(buffer).cast("B")
        if not self._connected or not len(view) or self._pipe is None:
            return 0
        try:
            reading, _ = _winapi.ReadFile(self._pipe, len(view), overlapped=True)
            got, _ = reading.GetOverlappedResult(True)
        except OSError as error:
            if error.winerror == ERROR_BROKEN_PIPE:
                return 0
            raise
        data = reading.getbuffer()
        assert data is not None  # completed
        view[:got] = data
        return got

    def close(self) -> None:
        if sys.platform == "win32" and self._pipe is not None:
            import _winapi

            _winapi.CloseHandle(self._pipe)
            self._pipe = None
        super().close()
