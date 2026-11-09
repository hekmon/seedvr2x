"""media/channel.py: ffmpeg's side channels, a pipe passed by its number on Linux (as before) and
a named pipe of seedvr2x's own on Windows, read to their end whether or not ffmpeg opened them.
The Windows cases run natively only, with an ffmpeg on PATH."""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from media import channel as channels
from media import reader, scan
from media.channel import Channel
from media.conversion import Conversion

on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows' named pipes")
on_linux = pytest.mark.skipif(sys.platform == "win32", reason="POSIX's pass_fds")
with_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")

# 320x240 rgb48le, 10 frames: impl-win-2's measure of the named pipes (2026-11-09). The length
# is the input's (-t before -i), so that it bounds every output: -frames:v after -i bounds the
# next output alone, and an output after it ran on without end (a first draft of this file read
# such an output whole until the machine's memory was full, 2026-11-09).
SOURCE = ("-t", "0.4", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25")
RAW = ("-f", "rawvideo", "-pix_fmt", "rgb48le")
EXPECTED = 320 * 240 * 6 * 10
# The most any read here takes, whatever ffmpeg writes: a byte past what is expected.
LIMIT = EXPECTED + 1


def read_whole(channel: Channel, process: "subprocess.Popen[bytes]", got: list[bytes]) -> None:
    with channel.stream(process) as stream:
        got.append(stream.read(LIMIT))
    if len(got[-1]) == LIMIT:
        process.kill()


@on_linux
def test_on_linux_a_channel_is_a_pipe_passed_by_its_number() -> None:
    before = set(os.listdir("/proc/self/fd"))
    channel = Channel()
    assert channel.output == f"pipe:{channel.passed[0]}" and len(channel.passed) == 1
    number = channel.passed[0]
    process = subprocess.Popen(
        [sys.executable, "-c", f"import os; os.write({number}, b'x' * 70000)"],
        pass_fds=channel.passed,
    )
    channel.started()
    with channel.stream(process) as stream:
        assert stream.read() == b"x" * 70000
    assert process.wait() == 0
    channel.close()
    assert set(os.listdir("/proc/self/fd")) == before


@on_linux
def test_a_channel_nothing_took_is_closed_whole() -> None:
    before = set(os.listdir("/proc/self/fd"))
    channel = Channel()
    channel.close()
    channel.close()
    assert set(os.listdir("/proc/self/fd")) == before


def test_the_commands_say_overwrite_on_windows_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    # ffmpeg takes a named pipe for a file that exists and, stdin closed, writes nothing and exits
    # 0 without -y (impl-win-2, 2026-11-09). On Linux, `pipe:N` is no file: the commands as before.
    conversion = Conversion("yuv420p", "709", "limited", "left")
    path = Path("in.mkv")
    if sys.platform != "win32":
        assert channels.OVERWRITE == ()
        assert "-y" not in reader.read_command(path, conversion, None, 0, "pipe:5", "pipe:6")
        assert "-y" not in scan.command(path, "pipe:5")
    for module in (reader, scan):
        monkeypatch.setattr(module, "OVERWRITE", ("-y",))
    read = reader.read_command(path, conversion, None, 0, "pipe:5")
    first = scan.command(path, "pipe:5")
    assert read[: read.index("-y") + 1][-2:] == ["-copyts", "-y"]
    assert first[:5] == ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-y"]
    assert "-y" not in scan.command(path)  # no channel: stdout and null alone


@on_windows
@with_ffmpeg
def test_on_windows_a_named_pipe_carries_ffmpegs_bytes_beside_stdout() -> None:
    channel = Channel()
    assert channel.output.startswith("\\\\.\\pipe\\seedvr2x-") and channel.passed == ()
    process = subprocess.Popen(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", *channels.OVERWRITE, *SOURCE),
            *("-map", "0:v", *RAW, "pipe:1", "-map", "0:v", *RAW, channel.output),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    channel.started()
    got: list[bytes] = []
    thread = threading.Thread(target=read_whole, args=(channel, process, got))
    thread.start()
    assert process.stdout is not None
    out = process.stdout.read(LIMIT)
    if len(out) == LIMIT:
        process.kill()
    process.wait()
    thread.join()
    assert process.returncode == 0
    assert len(out) == EXPECTED and got == [out]
    channel.close()


@on_windows
@with_ffmpeg
def test_on_windows_two_named_pipes_at_once() -> None:
    first, second = Channel(), Channel()
    assert first.output != second.output
    process = subprocess.Popen(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", *channels.OVERWRITE, *SOURCE),
            *("-map", "0:v", *RAW, first.output, "-map", "0:v", *RAW, second.output),
        ],
        stdin=subprocess.DEVNULL,
    )
    got: list[list[bytes]] = [[], []]
    threads = [
        threading.Thread(target=read_whole, args=(c, process, g))
        for c, g in zip((first, second), got, strict=True)
    ]
    for thread in threads:
        thread.start()
    assert process.wait() == 0
    for thread in threads:
        thread.join()
    assert got[0] == got[1] and len(got[0][0]) == EXPECTED


@on_windows
@with_ffmpeg
def test_on_windows_a_named_pipe_ffmpeg_never_opened_ends_with_it(tmp_path: Path) -> None:
    # ffmpeg opens its inputs before its outputs: one it can't open ends it before the pipe is
    # opened, and the read ends with the process, at once, rather than waiting for a client.
    channel = Channel()
    process = subprocess.Popen(
        [
            *("ffmpeg", "-hide_banner", "-v", "error", *channels.OVERWRITE),
            *("-i", str(tmp_path / "missing.mkv"), *RAW, channel.output),
        ],
        stdin=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    got: list[bytes] = []
    read_whole(channel, process, got)
    assert got == [b""] and process.wait() != 0
    assert time.monotonic() - started < 30
    channel.close()


@on_windows
def test_on_windows_a_named_pipe_of_a_killed_process_ends_and_its_name_is_free_again() -> None:
    channel = Channel()
    name = channel.output
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    killer = threading.Timer(0.5, process.kill)
    killer.start()
    got: list[bytes] = []
    read_whole(channel, process, got)
    killer.join()
    assert got == [b""]
    process.wait()
    channel.close()
    # The first instance alone was asked: closed, the name can be made again.
    if sys.platform != "win32":
        return
    import _winapi

    again = _winapi.CreateNamedPipe(
        name,
        _winapi.PIPE_ACCESS_INBOUND | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE,
        0,
        1,
        1024,
        1024,
        0,
        _winapi.NULL,
    )
    _winapi.CloseHandle(again)
