"""Milestone 5 as a GPU test (DESIGN.md, Validation milestones 5): seedvr2x's lab, run on milestone
1's input as one shot, is at least as good as numz's lab on the metrics that apply to one window
without a ground truth (ΔE lf, a*/b* spread, Y shift), a difference below the tolerance counting as
equal, and its output stays close to numz's.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and, in the m1/ directory of
SEEDVR2X_REFERENCE_DIR, milestone 1's input_rgb.mkv (tests/test_regression.py) and numz_lab.mkv:
numz 4490bd1's output with its lab, one batch, captured by research/scripts/ffv1_out.py, from
numz's checkout:

    FFV1_OUT_PATH=numz_lab.mkv FFV1_OUT_KEEP=0 python ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction lab

Both outputs are scored alike, from their 16-bit masters, with the metrics of
research/scripts/quality_metrics.py, against the input at its own size (the output brought to it
by 2x2 means) on its 8-bit scale:
- ΔE lf, the low frequencies' colour error: the mean CIE76 distance to the input after a Gaussian
  blur (sigma 4 px) in CIELAB;
- the a*/b* spread: each frame's standard deviation of a* and of b*, averaged over the frames,
  against the input's;
- the Y shift: the output's mean luma (BT.601 weights) minus the input's, signed.
CIELAB is colour.rgb_to_lab, numz's sRGB/D65 formulas (test_colour.py), where quality_metrics.py has
OpenCV's float conversion, which approximates them: up to 0.6 units off on random colours, 0.08 on
average. Both outputs go through the same one, so the comparison holds; on milestone 1's input ours
reads ΔE lf 0.6287 here, 0.6321 by quality_metrics.py.

The run goes through the CLI in a process of its own, as test_regression.py's does.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F

from seedvr2x.runtime import colour

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
    ),
]

# DESIGN.md's tolerances (Validation milestones 5): ΔE; for the a*/b* spread, 1% of the input's
# or 0.1 unit if larger; 8-bit level.
DE_LF, SPREAD, SPREAD_UNITS, SHIFT = 0.1, 0.01, 0.1, 0.1
# The PSNR of ours against numz's output, information in milestone 5's acceptance, is this test's
# own hold, tighter than the tolerances on purpose: 60.18 dB with lab's reference in float32 (59.50
# at the worst frame), 59.63 with the first, the encoder's bfloat16 input. The floor lets the
# output move by about 0.1 level more, 0.04 ΔE at mid-grey (2.5 levels per L* unit), against a
# tolerance of 0.1 ΔE. Measured again and set again after a deliberate change of lab or of what it
# is fed, the decode's precision for one.
PSNR_FLOOR = 59.5
# BT.601 luma, as quality_metrics.py weighs it.
LUMA = (0.299, 0.587, 0.114)
# OpenCV's Gaussian kernel for sigma 4 on float images: cvRound(8 sigma + 1) | 1 = 33 taps.
SIGMA, RADIUS = 4.0, 16


def planes(path: Path, pix_fmt: str, dtype: str) -> npt.NDArray[np.generic]:
    """The frames of an RGB FFV1 file in pix_fmt, (n, 3, H, W), planes R, G, B: a 16-bit master as
    it is stored (gbrp16le), an 8-bit file (bgr0, FFV1's 8-bit RGB) repacked to gbrp, exactly.
    Anything ffmpeg reports fails the read: a damaged FFV1 slice leaves its exit status at 0
    (DESIGN.md, Colour correction, input copy)."""
    probe = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "csv=p=0"),
            *("-show_entries", "stream=width,height", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = (int(x) for x in probe.stdout.strip().split(","))
    decoded = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0"),
            *("-f", "rawvideo", "-pix_fmt", pix_fmt, "-"),
        ],
        capture_output=True,
        check=True,
    )
    if decoded.stderr:
        raise RuntimeError(f"{path}: {decoded.stderr.decode(errors='replace')}")
    return np.frombuffer(decoded.stdout, dtype).reshape(-1, 3, height, width)[:, [2, 0, 1]]


def blur(frames: torch.Tensor) -> torch.Tensor:
    """frames (n, C, H, W) through quality_metrics.py's GaussianBlur (sigma 4), edges reflected
    without repeating the edge pixel, as OpenCV's default border."""
    taps = torch.arange(-RADIUS, RADIUS + 1, dtype=frames.dtype)
    kernel = torch.exp(-(taps**2) / (2 * SIGMA**2))
    kernel /= kernel.sum()
    channels = frames.shape[1]
    padded = F.pad(frames, (RADIUS,) * 4, mode="reflect")
    rows = F.conv2d(padded, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(rows, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def scores(output: npt.NDArray[np.generic], source: npt.NDArray[np.generic]) -> dict[str, float]:
    """ΔE lf, a*/b* spreads and Y shift of output (n, 3, 2h, 2w) against source (n, 3, h, w),
    frame by frame: the spreads in CIELAB units, the output's (a_spread, b_spread) and the
    input's (a_input, b_input)."""
    count, _, height, width = source.shape
    luma = torch.tensor(LUMA, dtype=torch.float64).view(3, 1, 1)
    de_lf, shift = 0.0, 0.0
    spreads = np.zeros((2, 2))  # (output, input) x (a*, b*)
    for index in range(count):
        # The 16-bit codes on the 8-bit scale, v / 257.
        frame = torch.from_numpy(output[index].astype(np.float64) * 255 / 65535)
        small = frame.view(3, height, 2, width, 2).mean((2, 4))
        original = torch.from_numpy(source[index].astype(np.float64))
        shift += float((small * luma).sum(0).mean() - (original * luma).sum(0).mean())
        lab = colour.rgb_to_lab(torch.stack([small, original]).div(255).float())
        lows = blur(lab.double())
        de_lf += float((lows[0] - lows[1]).square().sum(0).sqrt().mean())
        spreads += lab[:, 1:].double().flatten(2).std(2, correction=0).numpy()
    spreads /= count
    return {
        "de_lf": de_lf / count,
        "a_spread": float(spreads[0, 0]),
        "b_spread": float(spreads[0, 1]),
        "a_input": float(spreads[1, 0]),
        "b_input": float(spreads[1, 1]),
        "shift": shift / count,
    }


def psnr(a: npt.NDArray[np.generic], b: npt.NDArray[np.generic]) -> float:
    """PSNR of a against b on the 8-bit scale, from the mean squared error over every frame."""
    error = sum(
        float(np.square((x.astype(np.float64) - y) * 255 / 65535).mean())
        for x, y in zip(a, b, strict=True)
    )
    return 10 * math.log10(255**2 / (error / len(a))) if error else math.inf


def test_lab_against_numz(tmp_path: Path) -> None:
    assert MODELS is not None and REFERENCE is not None
    m1 = Path(REFERENCE) / "m1"
    for path in (m1 / "input_rgb.mkv", m1 / "numz_lab.mkv"):
        assert path.is_file(), f"{path}: missing; this module's docstring says how to make it"
    master = tmp_path / "ours.mkv"
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(m1 / "input_rgb.mkv"), "-o", str(master)),
            *("--model-dir", MODELS, "--dit-model", "seedvr2_ema_7b_fp16.safetensors"),
            *("--vae-model", "ema_vae_fp16.safetensors"),
            *("--resolution", "1080", "--seed", "42", "--color-correction", "lab"),
            *("--format", "gbrp16le"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    source = planes(m1 / "input_rgb.mkv", "gbrp", "u1")
    ours = planes(master, "gbrp16le", "<u2")
    numz = planes(m1 / "numz_lab.mkv", "gbrp16le", "<u2")
    assert ours.shape == numz.shape == (source.shape[0], 3, 1080, 1920)
    distance = psnr(ours, numz)
    mine, theirs = scores(ours, source), scores(numz, source)
    report = f"ours {mine}, numz {theirs}, PSNR ours against numz {distance:.3f} dB"
    print(report)
    assert mine["de_lf"] <= theirs["de_lf"] + DE_LF, report
    for channel in "ab":
        source_spread = mine[f"{channel}_input"]
        tolerance = max(SPREAD * source_spread, SPREAD_UNITS)
        off = abs(mine[f"{channel}_spread"] - source_spread)
        assert off <= abs(theirs[f"{channel}_spread"] - source_spread) + tolerance, report
    assert abs(mine["shift"]) <= abs(theirs["shift"]) + SHIFT, report
    assert distance >= PSNR_FLOOR, report
