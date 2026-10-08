"""Shots from a cut list, on the GPU: each shot's output is the one it gets alone, from its own
frames at its own place in the source; and the output has square pixels at the source's display
aspect (DESIGN.md, Pipeline step 2; Colour and shape).

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and milestone 1's input,
m1/input_rgb.mkv in SEEDVR2X_REFERENCE_DIR (tests/test_regression.py). Each run goes through the
CLI in a process of its own, as test_regression.py's does.
"""

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from seedvr2x.media.probe import probe
from seedvr2x.media.writer import count_packets

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
    ),
]


def upscale(source: Path, output: Path, *options: str) -> None:
    assert MODELS is not None
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(source), "-o", str(output)),
            *("--model-dir", MODELS, "--dit-model", "seedvr2_ema_7b_fp16.safetensors"),
            *("--vae-model", "ema_vae_fp16.safetensors", *options),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]


def test_each_shot_as_run_alone(tmp_path: Path) -> None:
    # 45 frames cut into shots of 10, 2, 18 and 15 frames: padded to 13, 5, 21 and 17, the third
    # in two windows. Each shot run alone, from a lossless copy of its frames with the seed it
    # gets in the job (the seed plus its first frame), must give the job's frames bit for bit.
    assert REFERENCE is not None
    source = Path(REFERENCE) / "m1" / "input_rgb.mkv"
    cuts = tmp_path / "cuts.txt"
    cuts.write_text("10\n12\n30\n")
    options = ("--resolution", "540", "--window", "5")
    job = tmp_path / "job"
    upscale(source, tmp_path / "job.mkv", *options, "--cuts", str(cuts), "--dump-frames", str(job))
    assert count_packets(tmp_path / "job.mkv") == 45
    for start, end in ((0, 10), (10, 12), (12, 30), (30, 45)):
        alone = tmp_path / f"shot_{start}.mkv"
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-i", str(source)),
                *("-vf", f"select=between(n\\,{start}\\,{end - 1})", "-fps_mode", "passthrough"),
                *("-c:v", "ffv1", "-pix_fmt", "bgr0", str(alone)),
            ],
            check=True,
        )
        dump = tmp_path / f"alone_{start}"
        seed = str(42 + start)
        upscale(
            alone,
            tmp_path / f"alone_{start}.mkv",
            *options,
            "--seed",
            seed,
            "--dump-frames",
            str(dump),
        )
        for index in range(start, end):
            ours = np.load(job / f"frame_{index:06d}.npy")
            assert np.array_equal(ours, np.load(dump / f"frame_{index - start:06d}.npy")), index


def test_display_aspect_output(tmp_path: Path) -> None:
    # An NTSC DVD's 720x480 at 16:9 (sample aspect 32:27): 1920x1080 out, square pixels.
    source = tmp_path / "ntsc.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=720x480:r=30000/1001"),
            *("-frames:v", "5", "-vf", "setsar=32/27", "-c:v", "ffv1", "-pix_fmt", "yuv420p"),
            str(source),
        ],
        check=True,
    )
    assert probe(source).sample_aspect == Fraction(32, 27)
    output = tmp_path / "out.mkv"
    upscale(source, output, "--resolution", "1080")
    stream = probe(output)
    assert (stream.width, stream.height) == (1920, 1080)
    assert stream.sample_aspect in (None, Fraction(1))
    assert count_packets(output) == 5
