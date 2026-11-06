"""Stopping a run (DESIGN.md, Pause and resume), on the CPU: Ctrl-C once lets the unit in progress
finish, twice stops at once, as SIGTERM does; the run stopped resumes as if never stopped.

The runs are processes of their own, the model replaced by test_cli_run.py's stand-in, and the
signals go to their whole process group, as a terminal sends Ctrl-C: ffmpeg's processes must be
out of its reach, else the segment being written would end early."""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import stop as stop_module
from media import ffmpeg
from media.ffmpeg import MediaError
from stop import (
    MARGIN,
    Deadline,
    Stop,
    Stopped,
    Terminated,
    deadline,
    late,
    terminable,
)

TESTS = Path(__file__).resolve().parent

# A run of test_cli_run.py's stand-in, announcing on stderr the step named by argv[2] from inside,
# then taking 2 s in it; a decode, once its first frame is written, so that its segment's ffmpeg
# runs. It runs the command as the script does (cli.command(): the process is its own, and a stop
# at once ends it by os._exit inside main, so nothing printed after main is read): each step
# called is said on stderr as it is called ("CALL name").
#
# The environment variable STOP_MODE (given by upscale's `mode`) makes the step named a hang the
# way the place-study night's decode hung (DESIGN.md, Allocator, the hang of 2026-11-06), on a
# process that "has used CUDA" (waiting.cuda_used replaced):
# - stall: a wait on the GPU that never ends (waiting.wait on an event that never completes);
# - atexit: stall, and an atexit function that blocks in C for ever (pause(2)), as a teardown
#   waiting on the GPU does;
# - finalizer: stall, and a finalizer that blocks in C for ever on an object held by the step's
#   own frame, so that it runs when the stop's traceback is released, as
#   blockswap.HostCopy's does (cudaHostUnregister of pinned memory);
# - cuda-error: the step raises the error CUDA gives after a fault, with the same finalizer;
# - c-call: the step's 2 s spent in C calls (libc's usleep), not in Python's sleep.
DRIVER = r"""
import atexit, ctypes, os, sys, time, weakref
sys.path.insert(0, sys.argv[1])
import test_cli_run as t
import main as cli
import run

slow = sys.argv[2]
mode = os.environ.get("STOP_MODE", "")
libc = ctypes.CDLL(None)

HANGS = ("stall", "atexit", "finalizer", "cuda-error")
if mode in HANGS:
    from seedvr2x import waiting
    waiting.cuda_used = lambda: True

class Holder:
    pass

def inside(name):
    if name == slow:
        # A wait's modes say IN from inside the wait (hang): a signal sent between this line and
        # the wait would find no wait to name.
        if mode not in ("stall", "atexit", "finalizer"):
            print(f"IN {name}", file=sys.stderr, flush=True)
        if mode == "c-call":
            # 2 s in C calls (usleep), where a signal finds the main thread.
            end = time.monotonic() + 2
            while time.monotonic() < end:
                libc.usleep(100000)
            return
        if mode in HANGS:
            if mode == "atexit":
                atexit.register(libc.pause)
            return
        time.sleep(2)

def hang():
    said = []
    def never():
        if not said:
            said.append(True)
            print(f"IN {slow}", file=sys.stderr, flush=True)
        return False
    waiting.wait(never, 0.01)

class Slow(t.Steps):
    def _call(self, name):
        super()._call(name)
        print(f"CALL {name}", file=sys.stderr, flush=True)
        inside(name)

    def decode(self, models, merged, count, target, write, *names, **tiled):
        name = f"decode {int(merged[0, 0, 0, 0])}"
        print(f"CALL {name}", file=sys.stderr, flush=True)
        written = []
        def first_then_inside(frames):
            write(frames)
            written.append(len(frames))
            if len(written) == 1:
                if mode in ("finalizer", "cuda-error"):
                    held = Holder()
                    weakref.finalize(held, libc.pause)
                inside(name)
                if name == slow and mode == "cuda-error":
                    raise RuntimeError("CUDA error: an illegal memory access was encountered")
                if name == slow and mode in HANGS:
                    hang()
        super().decode(models, merged, count, target, first_then_inside, *names, **tiled)

t.stand_in_model(setattr)
steps = Slow()
run.encode_shot, run.sample_windows, run.decode_shot = steps.encode, steps.windows, steps.decode
sys.argv = ["seedvr2x", *sys.argv[3:]]
status = cli.command()
sys.exit(status)
"""

# test_cli_run.py's job: shots of 3, 1 and 21 frames, the last in two windows; segments (0, 3) and
# (4).
UNINTERRUPTED = [
    *("encode 0", "window 0:0", "encode 3", "window 3:0", "decode 0", "decode 3"),
    *("encode 4", "window 4:0", "window 4:1", "decode 4"),
]


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


def test_handlers_while_running() -> None:
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with Stop() as stop:
        stop.check()
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.asked
        with pytest.raises(Stopped):
            stop.check()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)
        with pytest.raises(Terminated):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


# Python leaves Ctrl-C ignored when it starts so, as a shell starts a job with & (its default
# handler, KeyboardInterrupt, isn't installed then), so that test_terminable's Ctrl-C would go
# unseen on any code: read once, as pytest starts and imports this module.
SIGINT_IGNORED = signal.getsignal(signal.SIGINT) is signal.SIG_IGN


@pytest.mark.skipif(
    SIGINT_IGNORED, reason="SIGINT ignored as pytest started (a job started with &): no Ctrl-C"
)
def test_terminable() -> None:
    # Around the model files' fetch, before the run's handlers (main.py): SIGTERM raised as
    # Terminated, so that a download unwinds; Ctrl-C left to Python, KeyboardInterrupt; Python's
    # handlers given back.
    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with terminable():
        assert signal.getsignal(signal.SIGINT) is before[0]
        with pytest.raises(Terminated):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_notice_lost_stop_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # stderr gone (| tee, ended by the same Ctrl-C): the notice is lost, the stop is not.
    from types import SimpleNamespace

    import stop as module

    def broken(descriptor: int, data: bytes) -> int:
        raise BrokenPipeError

    monkeypatch.setattr(module, "os", SimpleNamespace(write=broken))
    with Stop() as stop:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.asked
        with pytest.raises(Stopped):
            stop.check()


@pytest.fixture
def job(tmp_path: Path) -> Path:
    if not _usable():
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "25", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(tmp_path / "in.mkv")),
        ],
        check=True,
    )
    (tmp_path / "cuts.txt").write_text("3\n4\n")
    (tmp_path / "w.safetensors").write_bytes(b"")
    return tmp_path


# The time a process has to end after the signal that stops it at once. A normal exit of a job
# takes 1.4 s at most (DESIGN.md, Allocator, the hang of 2026-11-06), the teardown that hung
# stayed 20 s.
ENDS_WITHIN = 30.0


def upscale(
    directory: Path,
    output: str,
    slow: str = "-",
    signals: tuple[int, ...] = (),
    mode: str = "",
    *,
    within: float | None = None,
) -> tuple[int, list[str], str]:
    """Run the job into output, the signals sent to its process group from inside the step slow
    (DRIVER, hanging as `mode` says), 0.3 s apart; its status, the steps it called, its stderr.
    With `within`, the process is killed that many seconds after the last signal (the step
    itself, with none) if it has not ended, and the test fails saying so."""
    process = subprocess.Popen(
        [
            *(sys.executable, "-c", DRIVER, str(TESTS), slow),
            *("in.mkv", "-o", output, "--model-dir", ".", "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", "42"),
            *("--cuts", "cuts.txt", "--window", "5", "--min-segment", "0.12"),
            # The stand-in's frames say which they are (test_cli_run.py), read back as the RGB
            # planes written (indexes).
            *("--color-correction", "none", "--format", "gbrp16le"),
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=directory,
        start_new_session=True,
        env={**os.environ, "STOP_MODE": mode},
    )
    assert process.stderr is not None
    lines: list[str] = []
    killed = threading.Event()

    def kill() -> None:
        killed.set()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)

    watchdog: threading.Timer | None = None
    try:
        for line in process.stderr:
            lines.append(line)
            if line.startswith(f"IN {slow}"):
                for number in signals:
                    # SIGUSR1 to seedvr2x alone: ffmpeg's default for it is to end.
                    if number == signal.SIGUSR1:
                        os.kill(process.pid, number)
                    else:
                        os.killpg(process.pid, number)
                    time.sleep(0.3)
                if within is not None:
                    watchdog = threading.Timer(within, kill)
                    watchdog.start()
        status = process.wait(timeout=120)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    text = "".join(lines)
    if killed.is_set():
        pytest.fail(f"the process was still running {within} s after the signal:\n{text}")
    calls = [line.split(" ", 1)[1].strip() for line in lines if line.startswith("CALL ")]
    return status, calls, text


def indexes(path: Path) -> list[int]:
    """Each frame's index in the job, as the stand-in writes it (test_cli_run.py)."""
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    values = np.frombuffer(data, dtype="<u2").reshape(-1, 3 * 96 * 128)
    assert (values == values[:, :1]).all()  # each frame one value throughout
    return [round(int(v) / 65535 * 1000) for v in values[:, 0]]


def resumed_as_uninterrupted(directory: Path, resumed: list[str]) -> None:
    status, calls, text = upscale(directory, "out")
    assert status == 0, text
    assert calls == resumed
    out = directory / "out"
    assert indexes(out / "seg_000000.mkv") + indexes(out / "seg_000001.mkv") == list(range(25))
    assert sorted(p.name for p in (directory / "out").iterdir()) == [
        "checksums",
        "frame_index.bin",
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]


def progress(directory: Path) -> tuple[list[bool], list[int]]:
    content = json.loads((directory / "out" / "manifest.json").read_text())
    finished = [segment["finished"] for segment in content["segments"]]
    return finished, [shot["windows_done"] for shot in content["shots"]]


@pytest.mark.parametrize(
    ("slow", "done", "kept", "resumed"),
    [
        # Inside a segment's decode, its ffmpeg running: the segment is finished first.
        ("decode 0", UNINTERRUPTED[:6], ([True, False], [1, 1, 0]), UNINTERRUPTED[6:]),
        # Inside a window: it is finished and kept first.
        ("window 4:1", UNINTERRUPTED[:9], ([True, False], [1, 1, 2]), UNINTERRUPTED[9:]),
    ],
)
def test_ctrl_c_once_finishes_the_unit(
    job: Path,
    slow: str,
    done: list[str],
    kept: tuple[list[bool], list[int]],
    resumed: list[str],
) -> None:
    status, calls, text = upscale(job, "out", slow, (signal.SIGINT,))
    assert status == 130, text
    # The notice starts with the time of day, as every line does (stop.py).
    assert re.search(r"^\d\d:\d\d:\d\d stopping after ", text, re.MULTILINE)
    assert "stopped after" in text
    assert calls == done
    assert progress(job) == kept
    resumed_as_uninterrupted(job, resumed)


@pytest.mark.parametrize(
    ("signals", "status", "said"),
    [
        ((signal.SIGINT, signal.SIGINT), 130, "stopped at once (Ctrl-C)"),
        ((signal.SIGTERM,), 143, "stopped at once (SIGTERM)"),
    ],
)
def test_stopped_at_once(job: Path, signals: tuple[int, ...], status: int, said: str) -> None:
    started = time.monotonic()
    found, calls, text = upscale(job, "out", "decode 0", signals)
    assert found == status, text
    assert said in text
    # A run that has not used CUDA says what it said before the process was ended at once: the
    # line ends there.
    assert re.search(rf"^\d\d:\d\d:\d\d .*{re.escape(said)}$", text, re.MULTILINE), text
    assert time.monotonic() - started < 30
    assert calls == UNINTERRUPTED[:5]  # stopped inside the decode of shot 0
    # The segment being written is gone, partial file and all; the windows are kept.
    assert progress(job) == ([False, False], [1, 1, 0])
    assert sorted(p.name for p in (job / "out").iterdir()) == [
        "frame_index.bin",
        "manifest.json",
        "resume",
    ]
    resumed_as_uninterrupted(job, ["decode 0", "decode 3", *UNINTERRUPTED[6:]])


# A stop at once in a run stalled in a wait on the GPU that never ends, or hung in a teardown
# (DESIGN.md, Allocator, the hang of 2026-11-06): the process ends by os._exit inside main
# (command), once the job's own work is done, so that neither the atexit functions nor the
# finalizers of the objects the stop's traceback holds can wait on the GPU.
SPOKEN = r"\d+ (?:s|min \d+ s|h \d+ min)"


@pytest.mark.parametrize(
    ("signals", "status", "how"),
    [
        ((signal.SIGTERM,), 143, "SIGTERM"),
        ((signal.SIGINT, signal.SIGINT), 130, "Ctrl-C"),
    ],
)
def test_stopped_at_once_in_a_wait_on_the_gpu(
    job: Path, signals: tuple[int, ...], status: int, how: str
) -> None:
    found, calls, text = upscale(
        *(job, "out", "decode 0", signals), mode="stall", within=ENDS_WITHIN
    )
    assert found == status, text
    assert re.search(
        rf"stopped at once \({re.escape(how)}\), {SPOKEN} into a wait on the GPU; the process"
        r" ends without waiting for it$",
        text,
        re.MULTILINE,
    ), text
    assert calls == UNINTERRUPTED[:5]
    assert progress(job) == ([False, False], [1, 1, 0])
    assert sorted(p.name for p in (job / "out").iterdir()) == [
        "frame_index.bin",
        "manifest.json",
        "resume",
    ]
    resumed_as_uninterrupted(job, ["decode 0", "decode 3", *UNINTERRUPTED[6:]])


@pytest.mark.parametrize("mode", ["atexit", "finalizer"])
def test_a_teardown_that_never_ends_is_skipped(job: Path, mode: str) -> None:
    # atexit: a function blocked in C, as CUDA's teardown is when the GPU's work never ends.
    # finalizer: one blocked on an object the stop's traceback holds, which runs when main
    # releases the traceback (blockswap.HostCopy's cudaHostUnregister). Without os._exit inside
    # main's except block, the process stays until it is killed.
    found, _, text = upscale(
        *(job, "out", "decode 0", (signal.SIGTERM,)), mode=mode, within=ENDS_WITHIN
    )
    assert found == 143, text
    assert "stopped at once (SIGTERM)" in text


def test_a_cuda_error_ends_the_process_without_waiting_for_the_gpu(job: Path) -> None:
    found, _, text = upscale(*(job, "out", "decode 0"), mode="cuda-error", within=ENDS_WITHIN)
    assert found == 1, text
    assert "the run failed on a CUDA error" in text
    assert "the process ends without waiting for the GPU" in text
    assert "an illegal memory access was encountered" in text
    assert "Traceback (most recent call last)" in text
    # What was done is kept: the same command resumes.
    assert progress(job) == ([False, False], [1, 1, 0])
    resumed_as_uninterrupted(job, ["decode 0", "decode 3", *UNINTERRUPTED[6:]])


AT = datetime(2026, 10, 10, 22, 0, tzinfo=UTC)


def test_late_is_now_plus_the_margin_times_the_estimate() -> None:
    # --until's rule (DESIGN.md, Pause and resume, Stopping): now plus 1.15 times the unit's
    # estimate passes the time.
    assert MARGIN == 1.15
    assert not late(AT, 99.9, AT + timedelta(seconds=115))
    assert late(AT, 100.1, AT + timedelta(seconds=115))
    # The estimate alone would fit.
    assert late(AT, 100.0, AT + timedelta(seconds=110))
    # Ending exactly at the time is not past it.
    assert not late(AT, 20.0, AT + timedelta(seconds=23))


def test_late_without_an_estimate_is_the_time_alone() -> None:
    assert not late(AT, None, AT)
    assert late(AT + timedelta(microseconds=1), None, AT)
    assert late(AT, 0.0, AT - timedelta(seconds=1))


def test_stopped_before_a_unit_that_would_end_late(monkeypatch: pytest.MonkeyPatch) -> None:
    at = AT
    monkeypatch.setattr(stop_module, "now", lambda: at)
    stop = Stop(AT + timedelta(seconds=115))
    stop.check(99.9)
    with pytest.raises(Deadline) as raised:
        stop.check(100.1, "shot 2/3's encode")
    assert (raised.value.unit, raised.value.cost) == ("shot 2/3's encode", 100.1)
    # A stop by the clock is a stop between two units, as a Ctrl-C's is: what catches one catches
    # the other.
    assert isinstance(raised.value, Stopped)
    # A unit nothing estimates is held to the time alone.
    stop.check()
    stop.check(None)
    at += timedelta(seconds=115)
    stop.check()
    at += timedelta(seconds=1)
    with pytest.raises(Deadline) as raised:
        stop.check()
    assert raised.value.cost is None and raised.value.unit == "the next unit"


def test_no_deadline_no_stop_by_the_clock() -> None:
    Stop().check(1e9)


def test_a_stop_asked_comes_before_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ctrl-C's stop is said as such, whatever the clock says of the next unit.
    monkeypatch.setattr(stop_module, "now", lambda: AT)
    with Stop(AT + timedelta(seconds=10)) as stop:
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(Stopped) as raised:
            stop.check(100.0)
        assert not isinstance(raised.value, Deadline)


@pytest.fixture
def paris() -> Generator[None]:
    """The local zone set to Europe/Paris for a test, the process's given back."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = "Europe/Paris"
    time.tzset()
    try:
        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


def local(text: str) -> datetime:
    """An ISO time taken as the local zone's, aware."""
    return datetime.fromisoformat(text).astimezone()


@pytest.mark.usefixtures("paris")
@pytest.mark.parametrize(
    ("now", "until", "expected"),
    [
        # Later today.
        ("2026-10-10T22:30:00", "23:00", "2026-10-10T23:00:00+02:00"),
        # Past today: tomorrow's, midnight wrapped too.
        ("2026-10-10T22:30:00", "07:00", "2026-10-11T07:00:00+02:00"),
        ("2026-10-10T22:30:00", "7:05", "2026-10-11T07:05:00+02:00"),
        ("2026-12-31T23:59:00", "00:00", "2027-01-01T00:00:00+01:00"),
        # The minute the clock shows now is past too: the next time it shows it.
        ("2026-10-10T07:00:00", "07:00", "2026-10-11T07:00:00+02:00"),
        ("2026-10-10T07:00:30", "07:00", "2026-10-11T07:00:00+02:00"),
        ("2026-10-10T06:59:59", "07:00", "2026-10-10T07:00:00+02:00"),
        # The night the clocks go back (2026-10-25, 03:00 CEST is 02:00 CET): a night's run is
        # an hour longer, and 02:30 is shown twice: the first still to come.
        ("2026-10-24T22:00:00", "07:00", "2026-10-25T07:00:00+01:00"),
        ("2026-10-25T01:00:00", "02:30", "2026-10-25T02:30:00+02:00"),
        ("2026-10-25T02:45:00+02:00", "02:30", "2026-10-25T02:30:00+01:00"),
        ("2026-10-25T02:45:00+01:00", "02:30", "2026-10-26T02:30:00+01:00"),
        # The night they go forward (2027-03-28, 02:00 CET is 03:00 CEST): 02:30 isn't shown
        # that night, so it is the night after's.
        ("2027-03-27T22:00:00", "07:00", "2027-03-28T07:00:00+02:00"),
        ("2027-03-27T22:00:00", "02:30", "2027-03-29T02:30:00+02:00"),
        ("2027-03-27T22:00:00", "03:00", "2027-03-28T03:00:00+02:00"),
    ],
)
def test_deadline_is_the_next_time_shown(now: str, until: str, expected: str) -> None:
    found = deadline(datetime.strptime(until, "%H:%M").time(), local(now))
    assert found == datetime.fromisoformat(expected)
    # A full date in the local zone, as it is logged.
    assert found.isoformat() == expected


@pytest.mark.usefixtures("paris")
def test_deadline_from_another_zone() -> None:
    # now is an instant, whatever zone says it: 20:30 UTC is 22:30 in Paris.
    now = datetime(2026, 10, 10, 20, 30, tzinfo=UTC)
    found = deadline(datetime.strptime("23:00", "%H:%M").time(), now)
    assert found - now == timedelta(minutes=30)
    assert found.isoformat() == "2026-10-10T23:00:00+02:00"


def test_now_is_aware_and_local() -> None:
    assert stop_module.now().utcoffset() is not None


@pytest.mark.parametrize("mode", ["", "c-call"])
def test_sigusr1_prints_every_threads_stack(job: Path, mode: str) -> None:
    # SIGUSR1 (faulthandler, registered by main.command) inside a slow step, the main thread in
    # Python's sleep or in a C call: every thread's stack on stderr, the step's frame in the
    # main thread's, and the run going on to its end (DESIGN.md, Allocator, the hang of
    # 2026-11-06).
    status, calls, text = upscale(job, "out", "decode 0", (signal.SIGUSR1,), mode)
    assert status == 0, text
    assert calls == UNINTERRUPTED
    assert re.search(r"^Current thread 0x[0-9a-f]+ \(most recent call first\):$", text, re.M)
    # The writer's stderr drain and the reader's threads run beside the main one.
    assert re.search(r"^Thread 0x[0-9a-f]+ \(most recent call first\):$", text, re.M)
    assert re.search(r"line \d+ in inside$", text, re.M)
    assert sorted(p.name for p in (job / "out").iterdir()) == [
        "checksums",
        "frame_index.bin",
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
