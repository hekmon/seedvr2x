"""split on the GPU (DESIGN.md, Validation milestones 5, split's clause).

- test_split_is_the_study: each slice of the decode, corrected in the pipeline, equals the colour
  study's own split() on the same decode and reference, within one 16-bit code, at x2 and x4,
  in one window and in three. x2 is milestone 1's input (960x540 to 1920x1080), x4 a 480x270
  copy of it (to 1920x1080); both cut to their first FRAMES frames, losslessly, to bound the GPU
  time. Each run goes through the CLI in a process of its own, as the other GPU tests' do, the
  allocator set before torch is imported; DRIVER, the process's code, wraps colour.split there to
  hold each slice's content, reference and output against the study's split(), run on the same
  device and on the CPU, and model.decode_stream, to hold the content split receives to the
  VAE's decode, slice for slice, cropped as shot._decoded crops it ("on the same decode": no
  channel moved, frame flipped or value clamped on the way, the decode going in unclamped). A
  wrapper in the tests rather than a hook in seedvr2x: nothing of it reaches a user's run.
- test_split_against_the_input: ours with split, the default, on milestone 1's input as one
  shot, scored against the input with milestone 5's metrics that apply to one window without a
  ground truth (ΔE lf, a*/b* spread, Y shift), as test_lab.py scored lab; every figure printed,
  numz's lab's too when its output is there. Its thresholds are its first run's figures plus
  milestone 5's tolerances.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR: numz's 7B fp16 and VAE), milestone
1's input, m1/input_rgb.mkv in SEEDVR2X_REFERENCE_DIR (tests/test_regression.py), and, for
test_split_is_the_study, the colour study's colour_variants.py and its baseline colour.py: the
repository's research/ and git, or SEEDVR2X_COLOUR_STUDY (the directory holding
colour_variants.py) and COLOUR_BASELINE (its baseline, git blob f7ad9cc6, test_colour.py's
BASELINE_BLOB). numz's lab's output, m1/numz_lab.mkv (milestone 5's), is read when there, for
information; research/scripts/m5_run.sh's stage m1 makes it in its first run, numz 4490bd1's lab
on milestone 1's input, one batch (the command in test_lab.py's docstring: git history, 6c29aa1).

Its scores (ΔE lf, a*/b* spread, Y shift):
- ΔE lf, the low frequencies' colour error: the mean CIE76 distance to the input after a Gaussian
  blur (sigma 4 px) in CIELAB;
- the a*/b* spread: each frame's standard deviation of a* and of b*, averaged over the frames,
  against the input's;
- the Y shift: the output's mean luma (BT.601 weights) minus the input's, signed.
They are research/scripts/quality_metrics.py's, against the input at its own size (the output
brought to it by 2x2 means) on its 8-bit scale, CIELAB computed here (rgb_to_lab).
"""

import json
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
from test_colour import WHY_SKIPPED, load_study

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
    ),
]

# The models the other GPU tests run, those of milestone 1's references; an empty cut list, the
# one shot of numz's one batch, which the detection cuts in three at its default threshold
# (tests/test_regression.py), the shot detector neither run nor its file needed.
MODEL_OPTIONS = (
    *("--dit-model", "seedvr2_ema_7b_fp16.safetensors"),
    *("--vae-model", "ema_vae_fp16.safetensors"),
    *("--cuts", os.devnull),
)
# 33 frames, 9 latents: one window, or three (--window 5: latents 0 to 4, 3 to 6 and 5 to 8, so
# 5, 4 and 4), two shared between each; the run's log names each window it ran (WINDOW_LOGGED).
FRAMES = 33
WINDOWS = {"one window": (), "three windows": ("--window", "5")}
WINDOW_LOGGED = {
    "one window": ["shot 1/1: window 1/1, latents 0 to 8, in "],
    "three windows": [
        *("shot 1/1: window 1/3, latents 0 to 4, in ", "shot 1/1: window 2/3, latents 3 to 6, in "),
        "shot 1/1: window 3/3, latents 5 to 8, in ",
    ],
}
# The output's height and width: milestone 1's 1080p, 16:9.
SIZE = (1080, 1920)
# The input of each factor: milestone 1's 960x540, or a 480x270 copy; the colour's stages there.
FACTORS = {"x2": (None, 3), "x4": ("480:270", 4)}
# One 16-bit code on [0, 1]: DESIGN.md's bound for split against the study's (milestone 5).
CODE = 1 / 65535

# The code of the process each run goes through: the CLI, colour.split wrapped to hold each
# slice against the study's split(), and model.decode_stream to keep each slice the VAE decodes,
# cropped as shot._decoded crops it, which the content split receives must equal. Arguments: the
# tests' directory, a scratch directory, the report's path, the shot's frames, the output's height
# and width, then the CLI's.
DRIVER = """
import json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")  # as cli.main sets it
tests, scratch, report, count, height, width, *argv = sys.argv[1:]
count, height, width = int(count), int(height), int(width)
sys.path.insert(0, tests)
from test_colour import load_study
import torch
from seedvr2x import cli
from seedvr2x.runtime import colour, model

study = load_study(Path(scratch))
assert study is not None
split, stream = colour.split, model.decode_stream
decoded = []  # the decode's slices not corrected yet, (t, C, H, W) as split takes its content
slices = []


def kept(models, latent):
    # Each slice as the VAE decodes it, (C, t, H', W'), cropped as shot._decoded crops it, to the
    # shot's frames and the output's size from the top left, and copied before seedvr2x's code
    # gets it, so that nothing done to it on the way can change what it is held to.
    written = 0
    for frames in stream(models, latent):
        chunk = frames.permute(1, 2, 3, 0)[: count - written, :height, :width]
        if chunk.shape[0]:
            decoded.append(chunk.permute(0, 3, 1, 2).clone())
            written += chunk.shape[0]
        yield frames


def held(content, reference, stages):
    expected = decoded.pop(0) if decoded else None
    same = expected is not None and torch.equal(content, expected)
    ours = split(content, reference, stages)
    content_, reference_ = content.contiguous(), reference.contiguous()
    here = study.split(content_, reference_, "ycc", 4, stages)
    cpu = study.split(content_.cpu(), reference_.cpu(), "ycc", 4, stages)
    slices.append({
        "frames": content.shape[0],
        "size": list(content.shape[2:]),
        "stages": stages,
        "content": str(content.dtype),
        "reference": str(reference.dtype),
        "output": str(ours.dtype),
        "device": str(ours.device),
        "range": [float(ours.min()), float(ours.max())],
        "same_device": float((ours - here).abs().max()),
        "cpu": float((ours.cpu() - cpu).abs().max()),
        "decoded": same,
    })
    return ours


colour.split = held
model.decode_stream = kept
status = cli.main(argv)
Path(report).write_text(json.dumps({"status": status, "slices": slices, "left": len(decoded)}))
"""


def cut(source: Path, output: Path, scale: str | None) -> Path:
    """The first FRAMES frames of source, stored losslessly as FFV1 bgr0 (8-bit RGB, as milestone
    1's input), scaled to `scale` (W:H) by averaging when given."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-frames:v", str(FRAMES)),
            *(("-vf", f"scale={scale}:flags=area") if scale else ()),
            *("-c:v", "ffv1", "-pix_fmt", "bgr0", str(output)),
        ],
        check=True,
    )
    return output


@pytest.mark.parametrize("windows", list(WINDOWS))
@pytest.mark.parametrize("factor", list(FACTORS))
def test_split_is_the_study(tmp_path: Path, factor: str, windows: str) -> None:
    assert MODELS is not None and REFERENCE is not None
    if load_study(tmp_path) is None:
        pytest.skip(WHY_SKIPPED)
    m1 = Path(REFERENCE) / "m1" / "input_rgb.mkv"
    assert m1.is_file(), f"{m1}: missing; tests/test_regression.py says how to make it"
    scale, stages = FACTORS[factor]
    source = cut(m1, tmp_path / "input.mkv", scale)
    report = tmp_path / "report.json"
    run = subprocess.run(
        [
            *(sys.executable, "-c", DRIVER, str(Path(__file__).parent), str(tmp_path)),
            *(str(report), str(FRAMES), *(str(side) for side in SIZE), str(source)),
            *("-o", str(tmp_path / "out.mkv"), "--model-dir", MODELS, *MODEL_OPTIONS),
            *("--resolution", "1080", *WINDOWS[windows]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    # The windows asked for ran, as the run logs them: an option ignored can't pass for three.
    for line in WINDOW_LOGGED[windows]:
        assert line in run.stderr, f"{line!r} not in the run's log"
    found = json.loads(report.read_text())
    assert found["status"] == 0
    slices = found["slices"]
    for held in slices:
        print(f"{factor}, {windows}: {held}")
    # The decode's slices, 5 frames then 4, each corrected once, as it comes out, and as decoded.
    assert [held["frames"] for held in slices] == [5, *[4] * ((FRAMES - 5) // 4)]
    assert found["left"] == 0
    for held in slices:
        assert held["decoded"], "split's content isn't the decode's slice"
        assert held["size"] == list(SIZE) and held["stages"] == stages
        assert (held["content"], held["reference"]) == ("torch.bfloat16", "torch.float32")
        assert held["output"] == "torch.float32" and held["device"].startswith("cuda")
        assert 0 <= held["range"][0] and held["range"][1] <= 1
        assert held["same_device"] < CODE and held["cpu"] < CODE
    worst = max(max(held["same_device"], held["cpu"]) for held in slices)
    print(f"{factor}, {windows}: max |ours - study| {worst} (one 16-bit code: {CODE})")


# The thresholds, set from the first run on the GPU (2026-10-09: numz's 7B fp16 and VAE, milestone
# 1's 45 frames, seed 42), each its figure plus milestone 5's tolerance for the metric (DESIGN.md,
# Validation milestones 5): ΔE lf 0.1307, + 0.1 ΔE; the a* and b* spreads 2.1865 and 5.2446, off
# the input's 2.1611 and 5.2122 by 0.0254 and 0.0324, + 1% of the input's spread or 0.1 unit if
# larger; the Y shift -0.0067 level, + 0.1 level. numz's lab on the same input, for comparison:
# ΔE lf 0.6245, spreads 2.1528 and 5.2313, Y shift +0.0300; PSNR between the two 40.311 dB.
DE_LF_MAX = 0.1307 + 0.1
SPREAD_FIRST = {"a": 0.0254, "b": 0.0324}  # the first run's |spread - input's|
SPREAD_SHARE, SPREAD_UNITS = 0.01, 0.1  # of the input's spread, or SPREAD_UNITS if larger
SHIFT_MAX = 0.0067 + 0.1  # 8-bit levels
# BT.601 luma, as quality_metrics.py weighs it.
LUMA = (0.299, 0.587, 0.114)
# OpenCV's Gaussian kernel for sigma 4 on float images: cvRound(8 sigma + 1) | 1 = 33 taps.
SIGMA, RADIUS = 4.0, 16
# sRGB to CIELAB under D65, from their definitions: IEC 61966-2-1's transfer function; the matrix
# to XYZ computed from the sRGB primaries and the D65 white, to seven decimals as Bruce Lindbloom
# tabulates it (the standard publishes four); CIE 15's L*a*b* with the D65 white (Xn 0.95047, Zn
# 1.08883). The formulas milestone 5 scored with, numz's (colour.py's rgb_to_lab, gone with lab),
# which quality_metrics.py approximates with OpenCV's float conversion: up to 0.6 units off on
# random colours, 0.08 on average.
RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
WHITE = (0.95047, 1.0, 1.08883)
EPSILON = 6 / 29


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """CIELAB under D65 of rgb (n, 3, H, W) float64 in [0, 1], sRGB: (n, 3, H, W) float64, L* then
    a* then b*."""
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    matrix = torch.tensor(RGB_TO_XYZ, dtype=rgb.dtype)
    xyz = torch.einsum("ij,njhw->nihw", matrix, linear)
    xyz = xyz / torch.tensor(WHITE, dtype=rgb.dtype).view(1, 3, 1, 1)
    f = torch.where(xyz > EPSILON**3, xyz.pow(1 / 3), xyz / (3 * EPSILON**2) + 4 / 29)
    lightness = 116 * f[:, 1] - 16
    return torch.stack([lightness, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], dim=1)


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
        lab = rgb_to_lab(torch.stack([small, original]).div(255))
        lows = blur(lab)
        de_lf += float((lows[0] - lows[1]).square().sum(0).sqrt().mean())
        spreads += lab[:, 1:].flatten(2).std(2, correction=0).numpy()
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


def test_split_against_the_input(tmp_path: Path) -> None:
    assert MODELS is not None and REFERENCE is not None
    m1 = Path(REFERENCE) / "m1"
    assert (m1 / "input_rgb.mkv").is_file(), "missing; tests/test_regression.py says how to make it"
    master = tmp_path / "ours.mkv"
    run = subprocess.run(
        [
            *(sys.executable, "-m", "seedvr2x", str(m1 / "input_rgb.mkv"), "-o", str(master)),
            *("--model-dir", MODELS, *MODEL_OPTIONS),
            *("--resolution", "1080", "--seed", "42", "--format", "gbrp16le"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    source = planes(m1 / "input_rgb.mkv", "gbrp", "u1")
    ours = planes(master, "gbrp16le", "<u2")
    assert ours.shape == (source.shape[0], 3, 1080, 1920)
    mine = scores(ours, source)
    report = f"split {mine}"
    if (m1 / "numz_lab.mkv").is_file():
        numz = planes(m1 / "numz_lab.mkv", "gbrp16le", "<u2")
        report += f", numz's lab {scores(numz, source)}, PSNR to it {psnr(ours, numz):.3f} dB"
    print(report)
    assert mine["de_lf"] <= DE_LF_MAX, report
    for channel, first in SPREAD_FIRST.items():
        spread = mine[f"{channel}_input"]
        off = abs(mine[f"{channel}_spread"] - spread)
        assert off <= first + max(SPREAD_SHARE * spread, SPREAD_UNITS), report
    assert abs(mine["shift"]) <= SHIFT_MAX, report
