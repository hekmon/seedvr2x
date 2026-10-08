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

- numz_sharp.mkv and numz_sharp_frames/: the same with the sharp 7B (made on 2026-10-08), the
  same command with FFV1_OUT_PATH=numz_sharp.mkv, FFV1_OUT_DUMP=numz_sharp_frames and
  --dit_model seedvr2_ema_7b_sharp_fp16.safetensors.

Each case is a run of its own, both model files named:
- numz-7b: numz's seedvr2_ema_7b_fp16.safetensors and ema_vae_fp16.safetensors, against numz.mkv;
- ours-7b: seedvr2x's seedvr2x_ema_7b_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors,
  the same values as numz's files (DESIGN.md, Weights), so against the same reference, bit for
  bit;
- ours-sharp: seedvr2x_ema_7b_sharp_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors, against
  numz_sharp.mkv, skipped when that reference is absent.
A case whose model file isn't in SEEDVR2X_MODEL_DIR is skipped, naming the file.

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
@pytest.mark.parametrize(
    ("dit", "vae", "reference"),
    [
        pytest.param(
            "seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors", "numz", id="numz-7b"
        ),
        pytest.param(
            "seedvr2x_ema_7b_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz",
            id="ours-7b",
        ),
        pytest.param(
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz_sharp",
            id="ours-sharp",
        ),
    ],
)
def test_milestone1_bit_identical(tmp_path: Path, dit: str, vae: str, reference: str) -> None:
    assert MODELS is not None and REFERENCE is not None
    for name in (dit, vae):
        if not (Path(MODELS) / name).is_file():
            pytest.skip(f"{name}: not in SEEDVR2X_MODEL_DIR")
    m1 = Path(REFERENCE) / "m1"
    references = (m1 / f"{reference}.mkv", m1 / f"{reference}_frames")
    if reference == "numz_sharp":
        for path in references:
            if not path.exists():
                pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(m1 / "input_rgb.mkv"), "-o", str(master)),
            *("--model-dir", MODELS, "--dit-model", dit, "--vae-model", vae),
            *("--resolution", "1080", "--seed", "42", "--dump-frames", str(frames)),
            # numz's output milestone 1 holds to, without its lab (StableSR's, not vendored).
            *("--color-correction", "none"),
            # numz.mkv's format, ffv1_out.py's default: compare.py reads both masters as stored.
            *("--format", "gbrp16le"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    for kind, expected, ours in (
        ("masters", references[0], master),
        ("frames", references[1], frames),
    ):
        compare = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tools" / "compare.py"),
                kind,
                str(expected),
                str(ours),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compare.returncode == 0, f"{kind}: {compare.stdout[-3000:]}"
