"""The segment command's writer (media/writer.py, CommandWriter; DESIGN.md, Output, Writer), against
the real ffmpeg and /bin/sh: the stream the command reads on stdin, the master's own; what its
file is checked for; its failures, its stop and what is left of it (skipped without a build
seedvr2x accepts). Linux's /proc tells the processes left."""

import itertools
import json
import logging
import os
import shlex
import signal
import subprocess
import time
import wave
import zlib
from collections.abc import Generator
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media import writer as writer_module
from seedvr2x.media.checksums import frame_bytes
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import command_partial
from seedvr2x.media.writer import (
    TAIL_LINES,
    TAIL_WIDTH,
    CommandWriter,
    FFV1Writer,
    SegmentWriter,
    Tags,
    _Tail,  # pyright: ignore[reportPrivateUsage]
    check_command,
    count_frames,
    open_writer,
)


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

RATE = Fraction(24000, 1001)
BT709 = Tags("bt709", "bt709")
# What ffprobe is asked of a stream: every tag the command's encoder must find on its stdin.
ENTRIES = (
    "stream=pix_fmt,color_range,color_space,color_primaries,color_transfer,chroma_location,"
    "r_frame_rate,avg_frame_rate,sample_aspect_ratio"
)
COPY = "cat > {out}"


def frames_of(count: int, width: int, height: int, seed: int = 0) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed).random((count, height, width, 3), dtype=np.float32)


def tags_of(source: Path | bytes) -> dict[str, str]:
    """ffprobe's reading of the first video stream of a file, or of bytes it reads from a pipe,
    which it can't seek, as a segment command reads its stdin; nothing warned of."""
    piped = isinstance(source, bytes)
    result = subprocess.run(
        [
            *("ffprobe", "-v", "warning", "-select_streams", "v:0", "-of", "json"),
            *("-show_entries", ENTRIES, "-i", "-" if piped else str(source)),
        ],
        input=source if piped else None,
        capture_output=True,
        check=True,
    )
    assert result.stderr == b""
    return json.loads(result.stdout)["streams"][0]


def frame_crcs(path: Path, pix_fmt: str, width: int, height: int) -> list[int]:
    """The CRC-32 of each frame of path as stored, pix_fmt being the stored format."""
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True,
        check=True,
    ).stdout
    size = frame_bytes(pix_fmt, width, height)
    assert len(data) % size == 0
    return [zlib.crc32(data[at : at + size]) for at in range(0, len(data), size)]


def packets(path: Path) -> list[str]:
    """Each packet of path's video stream, not decoded: its timestamps, size and CRC-32."""
    result = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy"),
            *("-f", "framehash", "-hash", "crc32", "-"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if not line.startswith("#")]


def has_encoder(name: str) -> bool:
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
    )
    return any(line.split()[1:2] == [name] for line in listed.stdout.splitlines())


def members(group: int) -> list[int]:
    """The processes of a process group still running: zombies, ended and waiting for their
    parent to read their status, are not."""
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # pid (comm) state ppid pgrp ...: comm may hold spaces and brackets.
            state, _, pgrp = (entry / "stat").read_text().rsplit(")", 1)[1].split()[:3]
        except OSError:
            continue  # ended meanwhile
        if int(pgrp) == group and state != "Z":
            found.append(int(entry.name))
    return found


def gone(group: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while members(group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def told(path: Path, seconds: float = 10.0) -> list[str]:
    """The words a command wrote to path, once it has."""
    deadline = time.monotonic() + seconds
    while not (path.is_file() and path.read_text().endswith("\n")):
        assert time.monotonic() < deadline, f"nothing written to {path}"
        time.sleep(0.01)
    return path.read_text().split()


def appeared(path: Path, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


# The stream on stdin: the proof of 2026-10-10, kept.


@pytest.mark.parametrize("pix_fmt", ["yuv420p10le", "gbrp16le"])
@pytest.mark.parametrize(
    ("tags", "size", "rate", "matrix"),
    [
        (BT709, (1280, 720), Fraction(24000, 1001), "bt709"),
        (BT709, (1280, 720), Fraction(25), "bt709"),
        (BT709, (1280, 720), Fraction(30000, 1001), "bt709"),
        (Tags("smpte170m", "smpte170m"), (720, 480), Fraction(30000, 1001), "smpte170m"),
        (Tags("bt2020", "smpte2084"), (1280, 720), Fraction(24000, 1001), "bt709"),
        (Tags(), (1280, 720), Fraction(24000, 1001), "bt709"),
    ],
)
def test_stream_is_the_master_s(
    tmp_path: Path, pix_fmt: str, tags: Tags, size: tuple[int, int], rate: Fraction, matrix: str
) -> None:
    # `cat > {out}` keeps the bytes the command reads. They hold the frames of FFV1Writer's
    # master, CRC-32 for CRC-32, in its very packets; and ffprobe, reading them from a pipe,
    # finds every tag as in the master's file: an SD size's smpte170m, the rate as an exact
    # rational, an untagged source's primaries and transfer left untagged.
    width, height = size
    frames = frames_of(3, width, height)
    master, segment = tmp_path / "master.mkv", tmp_path / "segment.mkv"
    with FFV1Writer(master, pix_fmt, width, height, rate, tags) as reference:
        reference.write(frames)
    with CommandWriter(COPY, segment, pix_fmt, width, height, rate, tags) as writer:
        writer.write(frames[:1])
        writer.write(frames[1:])
    assert writer.written == 3 and writer.checksums == []
    assert sorted(tmp_path.iterdir()) == [master, segment]  # the partial renamed
    assert frame_crcs(segment, pix_fmt, width, height) == reference.checksums
    assert frame_crcs(master, pix_fmt, width, height) == reference.checksums
    assert packets(segment) == packets(master) and len(packets(segment)) == 3

    piped = tags_of(segment.read_bytes())
    assert piped == tags_of(master)
    exact = f"{rate.numerator}/{rate.denominator}"
    expected = {"pix_fmt": pix_fmt, "r_frame_rate": exact, "avg_frame_rate": exact}
    if pix_fmt == "yuv420p10le":
        expected |= {"color_space": matrix, "color_range": "tv", "chroma_location": "left"}
    else:
        expected |= {"color_space": "gbr", "color_range": "pc"}
    if tags.primaries:
        expected |= {"color_primaries": tags.primaries, "color_transfer": tags.transfer}
    # Nothing else: no primaries nor transfer for an untagged source, and no sample aspect, as
    # the master declares none.
    assert piped == expected


def test_stream_read_to_its_end_without_a_warning(tmp_path: Path) -> None:
    # A Matroska that can't be seeked has no duration (and no cues): ffmpeg reads it from a pipe
    # to its last frame and says nothing. Its timestamps are Matroska's milliseconds, each
    # frame's time at 24000/1001 rounded: 41 or 42 ms apart.
    path = tmp_path / "segment.mkv"
    with CommandWriter(COPY, path, "yuv420p10le", 64, 48, RATE, BT709) as writer:
        writer.write(frames_of(50, 64, 48))
    stream = path.read_bytes()
    read = subprocess.run(
        ["ffmpeg", "-v", "warning", "-i", "-", "-f", "framehash", "-hash", "crc32", "-"],
        input=stream,
        capture_output=True,
        check=True,
    )
    assert read.stderr == b""
    assert sum(not line.startswith(b"#") for line in read.stdout.splitlines()) == 50
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "warning", "-of", "json", "-i", "-"),
            *("-show_entries", "format=duration:stream=duration,time_base:packet=pts"),
        ],
        input=stream,
        capture_output=True,
        check=True,
    )
    assert probed.stderr == b""
    found = json.loads(probed.stdout)
    assert found["format"] == {} and found["streams"] == [{"time_base": "1/1000"}]
    times = [packet["pts"] for packet in found["packets"]]
    assert times == [(index * 1001 + 12) // 24 for index in range(50)]
    assert {after - before for before, after in itertools.pairwise(times)} == {41, 42}


# DESIGN.md's own example, and an encoder any build has.


@pytest.mark.parametrize(
    ("encoder", "pix_fmt"),
    [("libx265 -crf 16", "yuv420p10le"), ("mpeg4 -bf 2 -q:v 2", "yuv420p")],
)
@pytest.mark.parametrize("rate", [Fraction(24000, 1001), Fraction(25), Fraction(30000, 1001)])
def test_encoded_by_the_command(tmp_path: Path, encoder: str, pix_fmt: str, rate: Fraction) -> None:
    # The encoder gets every frame, the rate and the tags, and converts nothing it needn't:
    # libx265 keeps yuv420p10le.
    if not has_encoder(encoder.split()[0]):
        pytest.skip(f"this ffmpeg has no {encoder.split()[0]}")
    path = tmp_path / "segment.mkv"
    command = f"ffmpeg -i - -c:v {encoder} {{out}}"
    with CommandWriter(command, path, "yuv420p10le", 1280, 720, rate, BT709) as writer:
        writer.write(frames_of(8, 1280, 720))
    assert writer.written == 8 and writer.checksums == []
    assert list(tmp_path.iterdir()) == [path]
    assert count_frames(path) == 8
    exact = f"{rate.numerator}/{rate.denominator}"
    found = tags_of(path)
    assert (found["r_frame_rate"], found["avg_frame_rate"]) == (exact, exact)
    assert found["pix_fmt"] == pix_fmt
    assert {key: found.get(key) for key in ("color_range", "color_space", "chroma_location")} == {
        "color_range": "tv",
        "color_space": "bt709",
        "chroma_location": "left",
    }
    assert (found["color_primaries"], found["color_transfer"]) == ("bt709", "bt709")


def test_leftover_partial_is_no_obstacle(tmp_path: Path) -> None:
    # A run killed left its partial file: ffmpeg, without -y, would leave it there and exit with
    # status 0 ("already exists. Exiting."). The command never finds it.
    path = tmp_path / "segment.mkv"
    command_partial(path).write_bytes(b"what a killed run left")
    command = "ffmpeg -i - -c:v mpeg4 -bf 2 -q:v 2 {out}"
    with CommandWriter(command, path, "yuv420p10le", 64, 48, RATE, BT709) as writer:
        writer.write(frames_of(6, 64, 48))
    assert list(tmp_path.iterdir()) == [path] and count_frames(path) == 6


def test_stale_checksums_removed_when_replaced(tmp_path: Path) -> None:
    # As FFV1Writer's: the checksums of what path held go only as the new file replaces it.
    path, stale = tmp_path / "segment.mkv", tmp_path / "segment.crc32"
    stale.write_text("00000000\n")
    failing = "cat > /dev/null; exit 3 # {out}"
    with pytest.raises(MediaError):
        with CommandWriter(failing, path, "gbrp16le", 16, 16, RATE, BT709, stale=stale) as writer:
            writer.write(frames_of(1, 16, 16))
    assert stale.exists()
    with open_writer("gbrp16le", path, 16, 16, RATE, BT709, stale=stale, command=COPY) as writer:
        writer.write(frames_of(1, 16, 16))
    assert not stale.exists() and path.exists()


# What the command's file is checked for.


def test_stale_checksums_stay_while_the_new_file_is_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The old file's checksums go once the new file's data is on the disk, just before the
    # rename (files.replace_whole): a stop during that sync leaves the old file with its own.
    from seedvr2x.media import files
    from seedvr2x.media.files import command_partial

    path, stale = tmp_path / "segment.mkv", tmp_path / "segment.crc32"
    path.write_bytes(b"the file before")
    stale.write_text("00000000\n")
    sync = files._sync  # pyright: ignore[reportPrivateUsage]
    seen: list[bool] = []

    def stopped(synced: Path) -> None:
        if synced == command_partial(path):
            seen.append(stale.exists())
            raise KeyboardInterrupt
        sync(synced)

    monkeypatch.setattr(files, "_sync", stopped)
    with pytest.raises(KeyboardInterrupt):
        with CommandWriter(COPY, path, "gbrp16le", 16, 16, RATE, BT709, stale=stale) as writer:
            writer.write(frames_of(3, 16, 16))
    assert seen == [True] and stale.read_text() == "00000000\n"
    assert path.read_bytes() == b"the file before"


def test_fewer_frames_refused(tmp_path: Path) -> None:
    # The command ends well, having taken 3 frames of 10: only the count tells.
    path = tmp_path / "segment.mkv"
    command = "ffmpeg -i - -frames:v 3 -c:v ffv1 {out}"
    with pytest.raises(
        MediaError, match=r"segment\.mkv: 3 frames in the file the segment command wrote, 10 wri"
    ):
        with CommandWriter(command, path, "yuv420p10le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(10, 16, 16))
    assert list(tmp_path.iterdir()) == []


def test_frames_are_decoded_to_be_counted(tmp_path: Path) -> None:
    # Packets are not frames: a stream no decoder takes holds its 5 packets, and is refused.
    path = tmp_path / "segment.mkv"
    command = "cat > {out}; sed -i s/V_FFV1/V_FFVX/ {out}"
    with pytest.raises(MediaError, match=r"ffprobe can't decode the file the segment command"):
        with CommandWriter(command, path, "yuv420p10le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(5, 16, 16))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["cat > /dev/null # {out}", "exit 0 # {out}"])
def test_nothing_written_refused(tmp_path: Path, command: str) -> None:
    # The command ends well, having read the stream or not: ffmpeg's broken pipe, when it read
    # none of it, is no failure of its own, and the missing file is what is said.
    path = tmp_path / "segment.mkv"
    with pytest.raises(MediaError, match=r"exited with status 0 without writing .*partial\.mkv"):
        with CommandWriter(command, path, "gbrp16le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(2, 16, 16))
    assert list(tmp_path.iterdir()) == []


def test_no_video_stream_refused(tmp_path: Path) -> None:
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-demuxers"], capture_output=True, text=True, check=True
    )
    if not any(line.split()[1:2] == ["wav"] for line in listed.stdout.splitlines()):
        pytest.skip("this ffmpeg reads no WAV")
    sound = tmp_path / "sound.wav"
    with wave.open(str(sound), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(8000)
        file.writeframes(bytes(1600))
    path = tmp_path / "out" / "segment.mkv"
    command = f"cat > /dev/null; cp '{sound}' {{out}}"
    with pytest.raises(MediaError, match=r"no video stream in the file the segment command wr"):
        with CommandWriter(command, path, "gbrp16le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(2, 16, 16))
    assert list(path.parent.iterdir()) == []


def test_unreadable_file_refused(tmp_path: Path) -> None:
    path = tmp_path / "segment.mkv"
    command = "cat > /dev/null; echo no video here > {out}"
    with pytest.raises(MediaError, match=r"ffprobe can't read the file the segment command wr"):
        with CommandWriter(command, path, "gbrp16le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(2, 16, 16))
    assert list(tmp_path.iterdir()) == []


# A failing command.


def test_failure_after_reading(tmp_path: Path) -> None:
    # Its status and its last lines, stdout's and stderr's; nothing left.
    path = tmp_path / "segment.mkv"
    command = "cat > {out}; echo to stdout; echo boom >&2; exit 3"
    with pytest.raises(MediaError) as raised:
        with CommandWriter(command, path, "yuv420p10le", 64, 48, RATE, BT709) as writer:
            writer.write(frames_of(4, 64, 48))
    assert str(raised.value) == (
        f"{path}: the segment command exited with status 3: to stdout / boom"
    )
    assert list(tmp_path.iterdir()) == []


def test_failure_at_once(tmp_path: Path) -> None:
    # The command is gone before it read anything: ffmpeg's broken pipe is its consequence, and
    # the command's status the cause reported, whether the frames were all sent or not.
    path = tmp_path / "segment.mkv"
    command = "echo no such preset >&2; exit 4 # {out}"
    with pytest.raises(MediaError, match=r"the segment command exited with status 4.*: no such"):
        with CommandWriter(command, path, "yuv420p10le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(2, 16, 16))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("command", "how"),
    [
        ("echo no such preset >&2; exit 4 # {out}", "exited with status 4"),
        ("exit 0 # {out}", "exited with status 0"),
        ("head -c 100000 > /dev/null; kill -KILL $$ # {out}", "was ended by SIGKILL"),
    ],
)
def test_failure_while_writing(tmp_path: Path, command: str, how: str) -> None:
    # More frames than the pipes hold: the write finds ffmpeg gone, and the error names the
    # command's end, not a broken pipe.
    path = tmp_path / "segment.mkv"
    frame = frames_of(1, 320, 240)
    writer = CommandWriter(command, path, "gbrp16le", 320, 240, RATE, BT709)
    with pytest.raises(MediaError) as raised, writer:
        for _ in range(400):
            writer.write(frame)
    message = str(raised.value)
    assert f"{path}: the segment command {how} before the segment's end, " in message
    assert "Broken pipe" not in message and "ffmpeg" not in message
    assert writer.written < 400
    assert list(tmp_path.iterdir()) == []


def test_command_not_found(tmp_path: Path) -> None:
    path = tmp_path / "segment.mkv"
    with pytest.raises(MediaError, match=r"exited with status 127: .*no-such-encoder.*not found"):
        with CommandWriter("no-such-encoder {out}", path, "gbrp16le", 16, 16, RATE, BT709) as w:
            w.write(frames_of(1, 16, 16))
    assert list(tmp_path.iterdir()) == []


def test_ffmpeg_s_own_failure_reported(tmp_path: Path) -> None:
    # An unknown tag: seedvr2x's ffmpeg refuses its filter graph. A command that ends well on an
    # empty stream doesn't hide it; one that fails on it is reported first, ffmpeg's lines after.
    path = tmp_path / "segment.mkv"
    with pytest.raises(MediaError, match=r"writing .*segment\.mkv with ffmpeg: exit status"):
        with CommandWriter(COPY, path, "gbrp16le", 16, 16, RATE, Tags("nope")) as writer:
            writer.write(frames_of(1, 16, 16))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(
        MediaError, match=r"command exited with status 5; ffmpeg, which feeds it, failed too: "
    ):
        with CommandWriter(f"{COPY}; exit 5", path, "gbrp16le", 16, 16, RATE, Tags("nope")) as w:
            w.write(frames_of(1, 16, 16))
    assert list(tmp_path.iterdir()) == []


# Stopping it.

# The shell's pid, its group's, and the child's, said once both run.
TELL = "sleep 1000 & echo $$ $! > '{pids}'"


def started(tmp_path: Path, command: str) -> tuple[CommandWriter, int, int]:
    """A writer whose command runs, a child of the shell's with it: the writer, the shell's pid
    and the child's."""
    pids = tmp_path / "pids"
    pids.unlink(missing_ok=True)
    path = tmp_path / "out" / "segment.mkv"
    writer = CommandWriter(
        command.replace("{pids}", str(pids)), path, "yuv420p10le", 64, 48, RATE, BT709
    )
    try:
        writer.write(frames_of(2, 64, 48))
        shell, child = map(int, told(pids))
    except BaseException:
        writer.abort()
        raise
    assert members(shell) and child in members(shell)
    return writer, shell, child


def test_abort_leaves_no_process_and_no_file(tmp_path: Path) -> None:
    # The shell, and the child it started: the whole group.
    writer, shell, _ = started(tmp_path, f"{TELL}; cat > {{out}}; wait")
    assert appeared(command_partial(writer.path))
    writer.abort()
    assert gone(shell)
    assert list(writer.path.parent.iterdir()) == []
    writer.abort()  # once more, as SegmentWriter's exit after a failed close


def test_exception_leaves_no_process_and_no_file(tmp_path: Path) -> None:
    writer, shell, _ = started(tmp_path, f"{TELL}; cat > {{out}}; wait")
    with pytest.raises(RuntimeError, match="stop"):
        with writer:
            raise RuntimeError("stop")
    assert gone(shell)
    assert list(writer.path.parent.iterdir()) == []


def test_groups_of_their_own(tmp_path: Path) -> None:
    # The command leads a group of its own: not the test's, which a terminal's Ctrl-C reaches,
    # nor that of seedvr2x's ffmpeg, which abort() must not signal with it.
    writer, shell, child = started(tmp_path, f"{TELL}; cat > {{out}}; wait")
    try:
        feeding = writer._process.pid  # pyright: ignore[reportPrivateUsage]
        assert os.getpgid(shell) == shell == os.getpgid(child)
        assert os.getpgid(feeding) == feeding
        assert len({shell, feeding, os.getpgid(0)}) == 3
    finally:
        writer.abort()
    assert gone(shell)


def test_stop_is_sigterm_first(tmp_path: Path) -> None:
    # A command with something of its own to undo gets the time to: SIGTERM, which it catches,
    # and the stop waits no longer than it takes.
    undone = tmp_path / "undone"
    command = f"trap 'echo yes > {undone}; exit 143' TERM; {TELL}; wait # {{out}}"
    writer, shell, _ = started(tmp_path, command)
    start = time.monotonic()
    writer.abort()
    assert time.monotonic() - start < writer_module.STOP_GRACE
    assert undone.read_text() == "yes\n"
    assert gone(shell)


def test_stop_is_sigkill_in_the_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A command that ignores SIGTERM, its child with it, is not waited for beyond the grace.
    monkeypatch.setattr(writer_module, "STOP_GRACE", 0.3)
    writer, shell, child = started(tmp_path, f"trap '' TERM; {TELL}; cat > {{out}}; wait")
    start = time.monotonic()
    writer.abort()
    assert 0.3 <= time.monotonic() - start < 3
    assert gone(shell) and child not in members(shell)
    assert list(writer.path.parent.iterdir()) == []


class _Interrupted(BaseException):
    """What a stop at once raises in the run: runtime/stop.py's Terminated, KeyboardInterrupt."""


@pytest.fixture
def alarm() -> Generator[None]:
    def interrupt(number: int, frame: object) -> None:
        raise _Interrupted

    previous = signal.signal(signal.SIGALRM, interrupt)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_stop_at_once_while_the_command_finishes(tmp_path: Path, alarm: None) -> None:
    # The stream is over and the command still encodes: a stop at once, raised in close(),
    # stops it too and removes its file.
    writer, shell, _ = started(tmp_path, f"{TELL}; cat > {{out}}; wait")
    with pytest.raises(_Interrupted):
        with writer:
            signal.setitimer(signal.ITIMER_REAL, 0.3)
    assert gone(shell)
    assert list(writer.path.parent.iterdir()) == []


def test_a_command_s_leftovers_survive_its_success_only(tmp_path: Path) -> None:
    # What the shell left running is the user's business when the segment is good, and stopped
    # with a segment refused: it could still write the file a resume writes again.
    writer, shell, child = started(tmp_path, f"{TELL}; cat > {{out}}")
    assert writer.close() == 2
    try:
        assert writer.path.exists() and members(shell) == [child]
    finally:
        os.kill(child, signal.SIGKILL)
    writer, shell, child = started(tmp_path, f"{TELL}; cat > {{out}}; exit 3")
    with pytest.raises(MediaError, match="exited with status 3"):
        writer.close()
    assert gone(shell)


# The command as run.


@pytest.mark.parametrize("name", ["it's a segment.mkv", 'two  spaces $HOME `x` "q".mkv'])
def test_out_is_one_word(tmp_path: Path, name: str, caplog: pytest.LogCaptureFixture) -> None:
    # {out} is written bare and reaches the command as one word, each time it is named,
    # whatever the path holds; the command as run is logged at DEBUG, and nothing else is said.
    directory = tmp_path / "an output's directory"
    path, words = directory / name, tmp_path / "words"
    command = f"printf '%s\\n' {{out}} > '{words}'; cat > {{out}}"
    with caplog.at_level(logging.DEBUG, logger="seedvr2x.media.writer"):
        with CommandWriter(command, path, "gbrp16le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(1, 16, 16))
    assert words.read_text() == f"{command_partial(path)}\n"
    assert list(directory.iterdir()) == [path] and count_frames(path) == 1
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (
            logging.DEBUG,
            f"{path}: the segment command, run by /bin/sh -c: "
            + command.replace("{out}", shlex.quote(str(command_partial(path)))),
        )
    ]


def test_out_is_never_an_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    words = tmp_path / "words"
    command = f"printf '%s\\n' {{out}} > '{words}'; cat > {{out}}"
    with CommandWriter(command, Path("-y.mkv"), "gbrp16le", 16, 16, RATE, BT709) as writer:
        writer.write(frames_of(1, 16, 16))
    assert words.read_text() == "./-y.partial.mkv\n"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["-y.mkv", "words"]


def test_nothing_printed(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    command = "echo to stdout; echo to stderr >&2; cat > {out}"
    with CommandWriter(command, tmp_path / "s.mkv", "gbrp16le", 16, 16, RATE, BT709) as writer:
        writer.write(frames_of(1, 16, 16))
    assert capfd.readouterr() == ("", "")


def test_command_checked() -> None:
    for empty in ("", " \t\n"):
        with pytest.raises(MediaError, match="the segment command is empty"):
            check_command(empty)
    with pytest.raises(MediaError, match=r"doesn't name \{out\}, the file it must write"):
        check_command("ffmpeg -i - -c:v libx265 out.mkv")
    check_command("ffmpeg -i - -c:v libx265 -crf 16 {out}")
    check_command(COPY)


def test_command_checked_before_anything_runs(tmp_path: Path) -> None:
    command = f"cat > '{tmp_path}/out.mkv'"
    with pytest.raises(MediaError, match=r"doesn't name \{out\}"):
        CommandWriter(command, tmp_path / "d" / "s.mkv", "gbrp16le", 16, 16, RATE, BT709)
    assert list(tmp_path.iterdir()) == []


def test_partial_keeps_the_suffix() -> None:
    assert command_partial(Path("out/seg_000001.mkv")) == Path("out/seg_000001.partial.mkv")
    assert command_partial(Path("a.b/film.mp4")) == Path("a.b/film.partial.mp4")


def test_open_writer_with_a_command(tmp_path: Path) -> None:
    path = tmp_path / "segment.mkv"
    with open_writer("yuv420p10le", path, 64, 48, RATE, BT709, command=COPY) as writer:
        assert isinstance(writer, CommandWriter)
        writer.write(frames_of(2, 64, 48))
    assert count_frames(path) == 2 and writer.checksums == []
    with pytest.raises(ValueError, match="no segment command with PNG"):
        open_writer("png", tmp_path / "png", 64, 48, RATE, BT709, command=COPY)
    assert list(tmp_path.iterdir()) == [path]
    with open_writer("gbrp16le", path, 64, 48, RATE, BT709) as writer:
        assert isinstance(writer, FFV1Writer)
        writer.write(frames_of(1, 64, 48))


def test_segments_by_the_command(tmp_path: Path) -> None:
    # The command runs once per segment, each file whole when its segment closes, without
    # checksums; one that fails stops the output at its segment, the segments before it kept.
    segments = [(tmp_path / f"seg_{index:06d}.mkv", count) for index, count in enumerate((3, 2, 4))]
    counter = tmp_path / "runs"
    command = f"echo run >> '{counter}'; ffmpeg -i - -c:v mpeg4 -bf 2 -q:v 2 {{out}}"
    finished: list[tuple[int, list[int]]] = []

    def open_segment(path: Path) -> CommandWriter:
        return CommandWriter(command, path, "yuv420p10le", 64, 48, RATE, BT709)

    with SegmentWriter(segments, open_segment, lambda *done: finished.append(done)) as writer:
        writer.write(frames_of(4, 64, 48))
        assert segments[0][0].exists() and not segments[1][0].exists()
        writer.write(frames_of(5, 64, 48))
    assert finished == [(0, []), (1, []), (2, [])]
    assert counter.read_text() == "run\n" * 3
    assert [count_frames(path) for path, _ in segments] == [3, 2, 4]
    assert sorted(tmp_path.iterdir()) == sorted([counter, *(path for path, _ in segments)])

    again = [(tmp_path / "again" / path.name, count) for path, count in segments]
    failing = f"cat > {{out}}; test \"$(wc -l < '{counter}')\" != 5 # its second run fails"
    failing = f"echo run >> '{counter}'; {failing}"
    with pytest.raises(MediaError, match=r"seg_000001\.mkv: the segment command exited with st"):
        with SegmentWriter(
            again, lambda path: CommandWriter(failing, path, "gbrp16le", 64, 48, RATE, BT709)
        ) as writer:
            writer.write(frames_of(9, 64, 48))
    assert list((tmp_path / "again").iterdir()) == [again[0][0]] and writer.written == 5
    assert count_frames(again[0][0]) == 3


# The command's output, kept bounded.


def test_megabytes_without_a_newline_neither_block_nor_grow(tmp_path: Path) -> None:
    # 8 MB on stderr before the stream is read, as an encoder's progress over a long segment: a
    # pipe nobody read would stop the command there (GNU timeout ends the wait, and the test).
    path = tmp_path / "segment.mkv"
    noise = "head -c 8000000 /dev/zero | tr '\\0' x"
    command = f'timeout 30 /bin/sh -c "{noise}" >&2 || exit 9; echo; echo done >&2; cat > {{out}}'
    with CommandWriter(command, path, "yuv420p10le", 64, 48, RATE, BT709) as writer:
        writer.write(frames_of(3, 64, 48))
    assert count_frames(path) == 3
    writer._output.join(timeout=5)  # pyright: ignore[reportPrivateUsage]
    kept = writer._tail.lines()  # pyright: ignore[reportPrivateUsage]
    assert kept == ["x" * TAIL_WIDTH + " [...]", "done"]


def test_failure_s_message_is_bounded(tmp_path: Path) -> None:
    # 2,000 progress lines, each replacing the last as on a terminal, 40 lines, then the error:
    # the last 20 lines, the progress one line of them.
    path = tmp_path / "segment.mkv"
    command = (
        "cat > /dev/null; i=0; while [ $i -lt 2000 ]; do printf 'frame %d\\r' $i; i=$((i+1));"
        " done >&2; i=0; while [ $i -lt 40 ]; do echo line $i; i=$((i+1)); done; exit 1 # {out}"
    )
    with pytest.raises(MediaError) as raised:
        with CommandWriter(command, path, "yuv420p10le", 16, 16, RATE, BT709) as writer:
            writer.write(frames_of(1, 16, 16))
    lines = " / ".join(f"line {index}" for index in range(40 - TAIL_LINES, 40))
    assert str(raised.value) == f"{path}: the segment command exited with status 1: {lines}"


def test_tail() -> None:
    tail = _Tail()
    assert tail.lines() == []
    # Lines end at \n, \r\n or \r; a line ended by \r alone is replaced by the next; empty
    # lines are none; the line being written counts.
    tail.feed(b"one\ntwo\r\n\n  \nframe 1\rframe 2\rframe")
    assert tail.lines() == ["one", "two", "frame"]
    tail.feed(b" 3\r")
    assert tail.lines() == ["one", "two", "frame 3"]
    tail.feed(b"\nerror: no\xff\n")  # \r, then \n in the next read: one line end
    assert tail.lines() == ["one", "two", "frame 3", "error: no�"]
    tail.feed(b"last")
    assert tail.lines() == ["one", "two", "frame 3", "error: no�", "last"]
    # The last TAIL_LINES lines only, each cut at TAIL_WIDTH bytes, however it comes.
    tail = _Tail()
    for index in range(3 * TAIL_LINES):
        tail.feed(f"{index}\n".encode())
    tail.feed(b"y" * (TAIL_WIDTH - 1))
    assert tail.lines() == [*map(str, range(2 * TAIL_LINES + 1, 3 * TAIL_LINES)), "y" * 299]
    for _ in range(64):
        tail.feed(b"y" * 65536)
    assert tail.lines()[-1] == "y" * TAIL_WIDTH + " [...]" and len(tail.lines()) == TAIL_LINES
    tail.feed(b"\nz\n")
    assert tail.lines()[-2:] == ["y" * TAIL_WIDTH + " [...]", "z"]
    assert len(tail._line) == 0  # pyright: ignore[reportPrivateUsage]
