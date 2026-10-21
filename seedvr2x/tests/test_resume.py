"""Milestone 4 on the GPU (DESIGN.md, Validation milestones): a job interrupted inside an encode,
inside a window and inside a segment's decode, by a kill, by Ctrl-C once and twice, and resumed
each time, gives the uninterrupted run's output, bit for bit; each stopped process exits, and
leaves nothing on the GPU. And the same job with its encodes and decodes tiled, killed inside a
tiled encode and inside a tiled decode, their scratch on disk, which each resume discards.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and milestone 1's input,
m1/input_rgb.mkv in SEEDVR2X_REFERENCE_DIR (tests/test_regression.py). Each run goes through the
CLI in a process of its own, as a user's would, its debug log telling which unit begins; Ctrl-C
goes to its whole process group, as a terminal sends it.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
    ),
]

# Milestone 1's 45 frames at 540p, shots of 4, 33 and 8 frames (2, 9 and 3 latents: the second
# in three windows of 5, 4 and 4), segments of 37 and 8 frames (0.2 s is 5 frames at 24000/1001):
# the first holds two shots. In gbrp16le, the masters hold the frames as written, compared as
# stored (frames).
CUTS = "4\n37\n"
OPTIONS = ["--resolution", "540", "--window", "5", "--cuts", "cuts.txt", "--min-segment", "0.2"]
OPTIONS += ["--format", "gbrp16le"]
SEGMENTS = ["seg_000000.mkv", "seg_000001.mkv"]


# seedvr2x's command line with every tiled pass's scratch on disk (python -c): the plan's
# allowance of system memory for it (Plan.scratch_allowance) set to 0, where the planner allows a
# shot this short its whole scratch. The planner imports no torch, so it is patched before the
# command line sets the process up as it does for a user (cli.main: the allocator, before torch
# is imported). No option of seedvr2x's does that.
ON_DISK = """
import dataclasses
import sys

from seedvr2x import cli
from seedvr2x.runtime import planner

plan = planner.plan


def on_disk(*job, **options):
    made, peaks = plan(*job, **options)
    return dataclasses.replace(made, scratch_allowance=0), peaks


planner.plan = on_disk
sys.exit(cli.main())
"""


def upscale(
    directory: Path,
    output: str,
    *acts: tuple[str, Callable[[int], None]],
    options: Sequence[str] = (),
    on_disk: bool = False,
) -> tuple[int, str]:
    """Run the job into output, with `options` besides; for each (text, action) of acts in turn,
    action(pid) once a line of its log holds text. Its status and log, once it has exited and
    left the GPU. on_disk runs it with every tiled pass's scratch on disk (ON_DISK)."""
    assert MODELS is not None
    process = subprocess.Popen(
        [
            *(
                sys.executable,
                *(("-c", ON_DISK) if on_disk else ("-m", "seedvr2x")),
                str(Path(REFERENCE or "") / "m1" / "input_rgb.mkv"),
            ),
            *(
                "-o",
                output,
                "--model-dir",
                MODELS,
                "--dit-model",
                "seedvr2_ema_7b_fp16.safetensors",
                "--vae-model",
                "ema_vae_fp16.safetensors",
            ),
            *(*OPTIONS, *options, "-v"),
        ],
        stderr=subprocess.PIPE,
        text=True,
        cwd=directory,
        start_new_session=True,
    )
    assert process.stderr is not None
    lines: list[str] = []
    pending = list(acts)
    for line in process.stderr:
        lines.append(line)
        if pending and pending[0][0] in line:
            pending.pop(0)[1](process.pid)
    status = process.wait(timeout=1800)
    assert not pending, f"{pending[0][0]!r} never came"
    _left_the_gpu(process.pid)
    return status, "".join(lines)


def _left_the_gpu(pid: int) -> None:
    for _ in range(20):
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        if str(pid) not in apps:
            return
        time.sleep(0.5)
    pytest.fail(f"process {pid} still holds the GPU after its exit")


def kill(delay: float = 0.0) -> Callable[[int], None]:
    """A crash, or a power cut short of the disk: SIGKILL, delay seconds after the unit began."""

    def act(pid: int) -> None:
        time.sleep(delay)
        os.kill(pid, signal.SIGKILL)

    return act


def kill_once_there(path: Path) -> Callable[[int], None]:
    """SIGKILL as soon as path is there: inside a tiled pass, once its scratch is on disk."""

    def act(pid: int) -> None:
        deadline = time.monotonic() + 300
        while not path.exists():
            assert time.monotonic() < deadline, f"{path} never came"
            time.sleep(0.005)
        os.kill(pid, signal.SIGKILL)

    return act


def ctrl_c(delay: float = 0.0) -> Callable[[int], None]:
    """Ctrl-C, delay seconds after the unit began, to the process group, as a terminal sends it."""

    def act(pid: int) -> None:
        time.sleep(delay)
        os.killpg(pid, signal.SIGINT)

    return act


# The notice of a first Ctrl-C (runtime/stop.py: the time of day, then the notice), after which a
# person presses it again.
NOTICE = " stopping after"


def progress(directory: Path) -> tuple[list[bool], list[int], list[bool]]:
    content = json.loads((directory / "manifest.json").read_text())
    shots = content["shots"]
    return (
        [shot["encoded"] for shot in shots],
        [shot["windows_done"] for shot in shots],
        [segment["finished"] for segment in content["segments"]],
    )


def frames(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
        capture_output=True,
        check=True,
    ).stdout


def encoded(log: str) -> list[int]:
    """The shots a run encoded, by number."""
    return [int(n) for n in re.findall(r"run: shot (\d+)/3, frames .* encoded in", log)]


def test_interrupted_and_resumed_bit_identical(tmp_path: Path) -> None:
    (tmp_path / "cuts.txt").write_text(CUTS)
    status, log = upscale(tmp_path, "whole")
    assert status == 0, log[-3000:]
    out = tmp_path / "out"

    # Killed inside the second shot's encode: the first shot's units kept.
    status, log = upscale(tmp_path, "out", ("run: shot 2/3: encoding", kill()))
    assert status == -signal.SIGKILL
    assert progress(out) == ([True, False, False], [1, 0, 0], [False, False])

    # Killed inside its second window: its latent and first window kept.
    status, log = upscale(tmp_path, "out", ("run: shot 2/3: window 2/3", kill()))
    assert status == -signal.SIGKILL
    assert encoded(log) == [2] and "resuming" in log
    assert progress(out) == ([True, True, False], [1, 1, 0], [False, False])

    # Ctrl-C once inside its third window: the window is finished and kept, and the run stops
    # before the first segment's decode.
    status, log = upscale(tmp_path, "out", ("run: shot 2/3: window 3/3", ctrl_c()))
    assert status == 130, log[-3000:]
    assert encoded(log) == [] and "window 2/3, latents 3 to 6" in log
    assert progress(out) == ([True, True, False], [1, 3, 0], [False, False])

    # Ctrl-C twice inside the first segment's decode: stopped at once, the segment gone. The
    # notice of the first comes while the GPU decodes: the run waits for it polling.
    status, log = upscale(
        tmp_path, "out", ("run: shot 2/3: decoding", ctrl_c(1.0)), (NOTICE, ctrl_c(0.2))
    )
    assert status == 130, log[-3000:]
    assert "stopped at once (Ctrl-C)" in log
    assert sorted(p.name for p in out.iterdir()) == ["frame_index.bin", "manifest.json", "resume"]

    # Killed inside it: what ffmpeg left of the segment is discarded on resume.
    status, log = upscale(tmp_path, "out", ("run: shot 2/3: decoding", kill(1.0)))
    assert status == -signal.SIGKILL
    assert progress(out)[2] == [False, False]
    assert (out / "seg_000000.mkv.partial").exists()

    # Ctrl-C once inside it: the segment is finished first, its ffmpeg out of Ctrl-C's reach.
    status, log = upscale(tmp_path, "out", ("run: shot 2/3: decoding", ctrl_c(1.0)))
    assert status == 130, log[-3000:]
    assert encoded(log) == [] and re.search(r"; [1-9]\d* leftovers? discarded", log)
    assert not (out / "seg_000000.mkv.partial").exists()
    assert progress(out) == ([True, True, False], [1, 3, 0], [True, False])

    # The rest.
    status, log = upscale(tmp_path, "out")
    assert status == 0, log[-3000:]
    assert encoded(log) == [3]
    assert progress(out)[2] == [True, True]
    assert sorted(p.name for p in out.iterdir()) == [
        "checksums",
        "frame_index.bin",
        "manifest.json",
        *SEGMENTS,
    ]
    for name in SEGMENTS:
        assert frames(out / name) == frames(tmp_path / "whole" / name), name


# The job with its encodes and decodes tiled: at 540p, padded to 960x576, 4 tiles of 512 px, 2
# rows of 2.
TILED = ["--encode-tile", "512", "--decode-tile", "512"]
GRID = "4 tiles of 512 px sharing 64, 2 across and 2 down,"


def test_tiled_interrupted_and_resumed_bit_identical(tmp_path: Path) -> None:
    # The tiled job uninterrupted, its scratch in system memory, as a user's at this size, then
    # on disk (ON_DISK): the same segments, bit for bit, and nothing of the scratch left.
    (tmp_path / "cuts.txt").write_text(CUTS)
    status, log = upscale(tmp_path, "memory", options=TILED)
    assert status == 0, log[-3000:]
    assert f"tiled encode: {GRID}" in log and f"tiled decode: {GRID}" in log
    # The tiled passes' own line (tiling._said: "its scratch, N bytes, in system memory"), not
    # the plan's, which says where a scratch may go ("scratch: ... in system memory up to ...").
    assert "bytes, in system memory" in log and "scratch.bf16" not in log
    status, log = upscale(tmp_path, "whole", options=TILED, on_disk=True)
    assert status == 0, log[-3000:]
    assert f"tiled encode: {GRID}" in log and f"tiled decode: {GRID}" in log
    assert "resume/shot_000004/scratch.bf16" in log and "bytes, in system memory" not in log
    listing = ["checksums", "frame_index.bin", "manifest.json", *SEGMENTS]
    assert sorted(p.name for p in (tmp_path / "whole").iterdir()) == listing
    for name in SEGMENTS:
        assert frames(tmp_path / "whole" / name) == frames(tmp_path / "memory" / name), name
    out = tmp_path / "out"
    scratch = out / "resume" / "shot_000004" / "scratch.bf16"
    discarded = r"discarded \S*resume/shot_000004/scratch\.bf16"

    # Killed inside the second shot's tiled encode, once its scratch is on disk: the kill leaves
    # it, with the first shot's units.
    status, log = upscale(
        tmp_path,
        "out",
        ("run: shot 2/3: encoding", kill_once_there(scratch)),
        options=TILED,
        on_disk=True,
    )
    assert status == -signal.SIGKILL
    assert progress(out) == ([True, False, False], [1, 0, 0], [False, False])
    assert scratch.exists() and not re.search(discarded, log)

    # Resumed: the scratch left is discarded, never read, and the shot encoded again. Killed
    # inside its tiled decode, the first segment's, once that scratch is on disk.
    status, log = upscale(
        tmp_path,
        "out",
        ("run: shot 2/3: decoding", kill_once_there(scratch)),
        options=TILED,
        on_disk=True,
    )
    assert status == -signal.SIGKILL
    assert re.search(discarded, log) and encoded(log) == [2]
    assert progress(out) == ([True, True, False], [1, 3, 0], [False, False])
    assert scratch.exists()

    # The rest: the decode's scratch discarded in turn, and the tiled job's own output.
    status, log = upscale(tmp_path, "out", options=TILED, on_disk=True)
    assert status == 0, log[-3000:]
    assert re.search(discarded, log) and encoded(log) == [3]
    assert progress(out)[2] == [True, True]
    assert sorted(p.name for p in out.iterdir()) == listing
    for name in SEGMENTS:
        assert frames(out / name) == frames(tmp_path / "whole" / name), name
