"""Stopping a run (DESIGN.md, Pause and resume), on the CPU: Ctrl-C once lets the unit in progress
finish, twice stops at once, as SIGTERM does; the run stopped resumes as if never stopped.

The runs are processes of their own, the model replaced by test_cli_run.py's stand-in, and the
signals go to their whole process group, as a terminal sends Ctrl-C: ffmpeg's processes must be
out of its reach, else the segment being written would end early."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime.stop import Stop, Stopped, Terminated

TESTS = Path(__file__).resolve().parent

# A run of test_cli_run.py's stand-in, announcing on stderr the step named by argv[2] from inside,
# then taking 2 s in it; a decode, once its first frame is written, so that its segment's ffmpeg
# runs. The steps called are said last.
DRIVER = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import test_cli_run as t
from seedvr2x import cli
from seedvr2x.runtime import run

slow = sys.argv[2]

def inside(name):
    if name == slow:
        print(f"IN {name}", file=sys.stderr, flush=True)
        time.sleep(2)

class Slow(t.Steps):
    def _call(self, name):
        super()._call(name)
        inside(name)

    def decode(self, models, merged, count, target, write, *names):
        name = f"decode {int(merged[0, 0, 0, 0])}"
        written = []
        def first_then_inside(frames):
            write(frames)
            written.append(len(frames))
            if len(written) == 1:
                inside(name)
        super().decode(models, merged, count, target, first_then_inside, *names)

t.stand_in_model(setattr)
steps = Slow()
run.encode_shot, run.sample_windows, run.decode_shot = steps.encode, steps.windows, steps.decode
status = cli.main(sys.argv[3:])
print("CALLS " + ",".join(steps.calls), file=sys.stderr, flush=True)
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


def test_notice_lost_stop_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # stderr gone (| tee, ended by the same Ctrl-C): the notice is lost, the stop is not.
    from types import SimpleNamespace

    from seedvr2x.runtime import stop as module

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
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
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


def upscale(
    directory: Path, output: str, slow: str = "-", signals: tuple[int, ...] = ()
) -> tuple[int, list[str], str]:
    """Run the job into output, the signals sent to its process group from inside the step slow
    (DRIVER), 0.3 s apart; its status, the steps it called, its stderr."""
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
    )
    assert process.stderr is not None
    lines: list[str] = []
    for line in process.stderr:
        lines.append(line)
        if line.startswith(f"IN {slow}"):
            for number in signals:
                os.killpg(process.pid, number)
                time.sleep(0.3)
    status = process.wait(timeout=120)
    calls = next(line for line in lines if line.startswith("CALLS "))
    return status, calls.split(" ", 1)[1].strip().split(","), "".join(lines)


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
    assert "seedvr2x: stopping after" in text and "stopped after" in text
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
    assert time.monotonic() - started < 30
    assert calls == UNINTERRUPTED[:5]  # stopped inside the decode of shot 0
    # The segment being written is gone, partial file and all; the windows are kept.
    assert progress(job) == ([False, False], [1, 1, 0])
    assert sorted(p.name for p in (job / "out").iterdir()) == ["manifest.json", "resume"]
    resumed_as_uninterrupted(job, ["decode 0", "decode 3", *UNINTERRUPTED[6:]])
