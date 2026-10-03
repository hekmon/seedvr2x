"""Milestone 1 as a regression test: seedvr2x's output stays bit-identical to numz's.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and milestone 1's reference, in
the m1/ directory of SEEDVR2X_REFERENCE_DIR:
- input_rgb.mkv: 45 frames of 8-bit RGB FFV1 at 960x540, read alike by numz (OpenCV) and by us
- numz.mkv and numz_frames/: numz 4490bd1's output for it, captured by
  research/scripts/ffv1_out.py, from numz's checkout:

    FFV1_OUT_PATH=numz.mkv FFV1_OUT_KEEP=0 FFV1_OUT_DUMP=numz_frames \
    FFV1_OUT_DUMP_FRAMES=$(seq -s, 0 44) python ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction none

The run goes through the CLI in a process of its own, as a user's would: the allocator is set
before torch is imported, which a test process importing torch first wouldn't do.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")


@pytest.mark.gpu
@pytest.mark.skipif(
    not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
)
def test_milestone1_bit_identical(tmp_path: Path) -> None:
    assert MODELS is not None and REFERENCE is not None
    m1 = Path(REFERENCE) / "m1"
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(m1 / "input_rgb.mkv"), "-o", str(master)),
            *("--model-dir", MODELS, "--dit-model", "seedvr2_ema_7b_fp16.safetensors"),
            *("--resolution", "1080", "--seed", "42", "--dump-frames", str(frames)),
            # numz's output milestone 1 holds to, without its lab (StableSR's, not vendored).
            *("--color-correction", "none"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    for kind, reference, ours in (
        ("masters", m1 / "numz.mkv", master),
        ("frames", m1 / "numz_frames", frames),
    ):
        compare = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tools" / "compare.py"),
                kind,
                str(reference),
                str(ours),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compare.returncode == 0, f"{kind}: {compare.stdout[-3000:]}"
