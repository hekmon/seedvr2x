"""The colour correction (runtime/colour.py), on the CPU: `split` equals the colour study's own
split() on the same frames, within one 16-bit code, at every factor the study ran and around them
(DESIGN.md, Validation milestones 5); the colour's stages follow the upscale factor's rule,
exactly; the à-trous low band is one computed apart in float64 and, away from the edges, the tent
it amounts to; content moved onto itself comes back bit for bit.

The study's script, research/scripts/colour_variants.py, is loaded from the repository with its
baseline, the colour.py it was built on, taken from git (BASELINE_BLOB); or, when both are set,
from STUDY, the directory holding colour_variants.py, and BASELINE, that baseline colour.py
(test_split.py runs on a copy of seedvr2x/ alone). One set without the other fails, naming the
other. With neither the repository nor both set, the comparisons are skipped."""

import importlib.util
import math
import os
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from seedvr2x.runtime import colour
from seedvr2x.runtime.job import target_size

REPOSITORY = Path(__file__).resolve().parents[2]
# runtime/colour.py as the colour study ran it, which colour_variants.py loads as its baseline
# (COLOUR_BASELINE): colour.py as of 6c29aa1, unchanged until split replaced `lab`.
BASELINE_BLOB = "f7ad9cc669b2742c16983de1bb5e836f2fa8aa81"
# The directory holding colour_variants.py, and the baseline colour.py it loads, when the
# repository isn't there.
STUDY = "SEEDVR2X_COLOUR_STUDY"
BASELINE = "COLOUR_BASELINE"
WHY_SKIPPED = (
    f"needs the colour study's colour_variants.py and its baseline: the repository's research/ and"
    f" git, or {STUDY} and {BASELINE}"
)
# One 16-bit code on [0, 1]: DESIGN.md's bound for split against the study's (milestone 5).
CODE = 1 / 65535


def load_study(scratch: Path) -> ModuleType | None:
    """The colour study's colour_variants.py, loaded with its baseline: from STUDY and BASELINE
    when both are set, else from the repository, the baseline taken from git into scratch; None
    when there is no study to load.

    One set without the other fails, naming the other, rather than skip the comparisons or load
    another study or baseline than the one meant: alone, neither says which pair goes together.
    BASELINE is the study's own variable, which colour.md's Reproduce exports for its scoring;
    without it, colour_variants.py loads the repository's runtime/colour.py, split's since split
    replaced lab, not its baseline."""
    directory, baseline = os.environ.get(STUDY), os.environ.get(BASELINE)
    if directory or baseline:
        if not (directory and baseline):
            given, missing = (STUDY, BASELINE) if directory else (BASELINE, STUDY)
            pytest.fail(
                f"{given} is set but {missing} isn't: set both (on a copy of seedvr2x/ alone), or"
                " neither (in the repository)",
                pytrace=False,
            )
        script, base = Path(directory) / "colour_variants.py", Path(baseline)
    else:
        script = REPOSITORY / "research" / "scripts" / "colour_variants.py"
        if not (script.is_file() and (REPOSITORY / ".git").exists()):
            return None
        shown = subprocess.run(
            ["git", "-C", str(REPOSITORY), "cat-file", "-p", BASELINE_BLOB],
            capture_output=True,
            check=False,
        )
        if shown.returncode:
            return None
        base = scratch / "colour_baseline.py"
        base.write_bytes(shown.stdout)
    if not (script.is_file() and base.is_file()):
        return None
    spec = importlib.util.spec_from_file_location("colour_variants", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Read when the module loads (colour_variants.py, BASELINE), then put back as it was.
    before = os.environ.get(BASELINE)
    os.environ[BASELINE] = str(base)
    try:
        spec.loader.exec_module(module)
    finally:
        if before is None:
            del os.environ[BASELINE]
        else:
            os.environ[BASELINE] = before
    return module


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    found = load_study(tmp_path_factory.mktemp("study"))
    if found is None:
        pytest.skip(WHY_SKIPPED)
    return found


# (height, width) of the frames as stored, the target they are resized to, and the colour's
# stages the rule gives (DESIGN.md, Colour correction): the factors the study ran (x1.5, x2, x3,
# x4; x4 from 480x270 to 1080p, its GPU run's), 480p to 1080p (x2.25), below and above them, and
# anamorphic NTSC DVDs, 4:3 and 16:9 (720x480, sample aspects 8:9 and 32:27, to 1080p).
FACTORS = {
    "x1.2": ((90, 160), (108, 192), 2),
    "x1.5": ((72, 128), (108, 192), 3),
    "x2": ((54, 96), (108, 192), 3),
    "x2.25": ((48, 64), (108, 144), 3),
    "x3": ((36, 64), (108, 192), 4),
    "x4": ((27, 48), (108, 192), 4),
    "x4 270p to 1080p": ((270, 480), (1080, 1920), 4),
    "x6": ((18, 32), (108, 192), 5),
    "4:3 DVD": ((480, 720), target_size(720, 480, Fraction(8, 9), 1080), 3),
    "16:9 DVD": ((480, 720), target_size(720, 480, Fraction(32, 27), 1080), 3),
}


def test_colour_stages() -> None:
    for name, (source, target, stages) in FACTORS.items():
        assert colour.colour_stages(source, target) == stages, name
    # The DVDs: each axis its own factor, the geometric mean taken (x2 and x2.25; x2.67 and x2.25).
    assert FACTORS["4:3 DVD"][1] == (1080, 1440) and FACTORS["16:9 DVD"][1] == (1080, 1920)
    # At the boundaries, f² = 2^(2k - 3) exactly, half up: x√2 (2 by 1), x√8 (4 by 2), x√32 (8 by
    # 4); just below, the stage below.
    for (height, width), stages in (((1, 2), 3), ((2, 4), 4), ((4, 8), 5), ((8, 16), 6)):
        assert colour.colour_stages((100, 100), (100 * height, 100 * width)) == stages
        assert colour.colour_stages((100, 100), (100 * height, 100 * width - 1)) == stages - 1
    # The geometric mean, not the larger nor the mean of the two: x1.2 by x6 is x2.68.
    assert colour.colour_stages((100, 100), (120, 600)) == 3
    # Never fewer than 2, at x1 and below.
    for factor in (1, 2, 4, 10):
        assert colour.colour_stages((100 * factor, 100 * factor), (100, 100)) == 2
    # The study's rule, round(2 + log2 f), wherever no float tie can decide it.
    for source_height in range(40, 400, 7):
        for target_height in range(60, 2200, 37):
            factor = target_height / source_height
            exact = 2 + math.log2(factor)
            if abs(exact - math.floor(exact) - 0.5) < 1e-6:
                continue
            expected = max(2, math.floor(exact + 0.5))
            source, target = (source_height, 2 * source_height), (target_height, 2 * target_height)
            assert colour.colour_stages(source, target) == expected, (source, target)


def a_trous(plane: npt.NDArray[np.float64], stages: int) -> npt.NDArray[np.float64]:
    """The low band in float64, written apart from colour.low_bands: each stage's 3-by-3 binomial
    kernel tap by tap, the indices clamped to the plane."""
    height, width = plane.shape
    cap = max(1, min(height, width) // 8)
    rows, columns = np.arange(height), np.arange(width)
    weights = {-1: 0.25, 0: 0.5, 1: 0.25}
    x = plane
    for stage in range(stages):
        d = min(2**stage, cap)
        out = np.zeros_like(x)
        for i, wi in weights.items():
            for j, wj in weights.items():
                taps = x[np.clip(rows + i * d, 0, height - 1)][
                    :, np.clip(columns + j * d, 0, width - 1)
                ]
                out += wi * wj * taps
        x = out
    return x


@pytest.mark.parametrize("stages", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [(128, 144), (40, 57)], ids=["taps 1 to 16", "taps capped at 5"])
def test_low_band_a_trous(size: tuple[int, int], stages: int) -> None:
    frames = torch.rand(2, 3, *size, generator=torch.Generator().manual_seed(3)) * 2 - 1
    low = colour.low_band(frames, stages)
    assert low.dtype == torch.float32 and low.shape == frames.shape
    planes = frames.double().reshape(-1, *size).numpy()
    expected = np.stack([a_trous(plane, stages) for plane in planes]).reshape(frames.shape)
    assert np.abs(low.numpy() - expected).max() < 1e-7


@pytest.mark.parametrize("stages", [4, 5])
def test_low_band_tent(stages: int) -> None:
    # Away from the edges, 2^stages - 1 pixels and more, k stages amount to a separable tent of
    # 2^(k + 1) - 1 taps: 31 for the lightness's 4, 63 for 5.
    frame = torch.rand(130, 150, generator=torch.Generator().manual_seed(4)) * 2 - 1
    half = 1 << stages
    tent = (half - np.abs(np.arange(1 - half, half))) / (half * half)

    def convolve(line: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.convolve(line, tent, mode="valid")

    expected = np.apply_along_axis(
        convolve, 0, np.apply_along_axis(convolve, 1, frame.double().numpy())
    )
    margin = half - 1
    low = colour.low_band(frame, stages)[margin:-margin, margin:-margin].numpy()
    assert low.shape == expected.shape
    assert np.abs(low - expected).max() < 1e-7


def test_low_bands_one_cascade() -> None:
    # Each count's band from one cascade is low_band's, bit for bit; 0 stages are the frames.
    frames = torch.rand(2, 3, 50, 70, generator=torch.Generator().manual_seed(5)) * 2 - 1
    bands = colour.low_bands(frames, (0, 2, 4, 5))
    assert sorted(bands) == [0, 2, 4, 5]
    assert torch.equal(bands[0], frames)
    for stages in (2, 4, 5):
        assert torch.equal(bands[stages], colour.low_band(frames, stages))
    with pytest.raises(ValueError, match="negative"):
        colour.low_bands(frames, (-1, 2))


def frames_for(kind: str, size: tuple[int, int], seed: int) -> tuple[Tensor, Tensor]:
    """content (2, 3, H, W) bfloat16, as the VAE decodes, and reference (2, 3, H, W) float32 in
    [-1, 1]: random, the content out of range in places, or smooth, the content the reference
    brightened and noisier, as the study's self-test makes them."""
    generator = torch.Generator().manual_seed(seed)
    height, width = size
    if kind == "random":
        content = torch.rand(2, 3, height, width, generator=generator) * 2.2 - 1.1
        reference = torch.rand(2, 3, height, width, generator=generator) * 2 - 1
        return content.to(torch.bfloat16), reference
    base = torch.rand(2, 3, max(1, height // 8), max(1, width // 8), generator=generator) * 2 - 1
    reference = F.interpolate(base, size=size, mode="bilinear", align_corners=False)
    noise = 0.1 * torch.randn(2, 3, height, width, generator=generator)
    content = (reference * 1.2 + noise).clamp(-1.05, 1.05)
    return content.to(torch.bfloat16), reference


def cases() -> Iterator[tuple[str, tuple[int, int], int]]:
    for name, (_, target, stages) in FACTORS.items():
        yield name, target, stages


@pytest.mark.parametrize("kind", ["random", "smooth"])
@pytest.mark.parametrize(("name", "target", "stages"), list(cases()), ids=list(FACTORS))
def test_split_is_the_study(
    study: ModuleType, name: str, target: tuple[int, int], stages: int, kind: str
) -> None:
    # Ours against the study's split() on the same frames, `split:ycc:4:SC`, SC the stages the rule
    # gives at the factor (FACTORS): within one 16-bit code, bit for bit in fact.
    source = FACTORS[name][0]
    assert colour.colour_stages(source, target) == stages
    content, reference = frames_for(kind, target, seed=len(name) + target[1])
    ours = colour.split(content, reference, colour.colour_stages(source, target))
    theirs = study.split(content, reference, "ycc", 4, stages)
    assert ours.dtype == torch.float32 and ours.shape == reference.shape
    assert 0 <= float(ours.min()) and float(ours.max()) <= 1
    difference = float((ours - theirs).abs().max())
    print(
        f"{name} ({kind}): {source} to {target}, {stages} stages: max |ours - study| {difference}"
    )
    assert difference < CODE
    # From the decode's layout, (3, T, H, W) seen as (T, 3, H, W), the same values.
    decoded = content.transpose(0, 1).contiguous().transpose(0, 1)
    assert torch.equal(colour.split(decoded, reference, stages), ours)


def test_split_onto_itself() -> None:
    # The split loses nothing: content moved onto itself comes back bit for bit in Y'CbCr, and to
    # float32's rounding of the matrix there and back in RGB.
    content, _ = frames_for("random", (64, 80), seed=6)
    for stages in (2, 3, 4, 5):
        assert torch.equal(colour.transfer(content, content, stages), colour.ycc(content))
        back = colour.split(content, content, stages)
        assert float((back - colour.unit_range(content)).abs().max()) < 1e-6
    with pytest.raises(ValueError):
        colour.split(content, content[:1], 3)  # one frame's low band for all: refused


def test_split_moves_the_low_bands() -> None:
    # The content's detail on the reference's coarse lightness and colour: flat frames take the
    # reference's values; a pattern finer than both scales added to the reference stays, the
    # reference's level under it.
    flat = torch.full((1, 3, 48, 64), 0.25)
    shifted = torch.tensor([0.1, -0.3, 0.6]).view(1, 3, 1, 1).expand(1, 3, 48, 64)
    moved = colour.split(flat, shifted, 3)
    assert float((moved - colour.unit_range(shifted)).abs().max()) < 1e-6
    checks = (torch.arange(64).view(1, -1) + torch.arange(48).view(-1, 1)) % 2 * 0.2 - 0.1
    reference = torch.rand(1, 3, 6, 8, generator=torch.Generator().manual_seed(7)) - 0.5
    reference = F.interpolate(reference, size=(48, 64), mode="bilinear", align_corners=False)
    content = (reference + 0.3 + checks).to(torch.bfloat16)  # brighter, with a fine pattern
    corrected = colour.split(content, reference, 3)[..., 16:-16, 16:-16]
    expected = colour.unit_range(reference + checks)[..., 16:-16, 16:-16]
    assert float((corrected - expected).abs().max()) < 0.01


def test_ycc_bt709() -> None:
    # BT.709's luma weights, and the colour differences scaled to [-0.5, 0.5]: white has no
    # colour, pure blue and pure red the extremes of Cb and Cr.
    white, blue, red = (
        torch.tensor(rgb).view(1, 3, 1, 1) for rgb in ((1.0,) * 3, (0, 0, 1.0), (1.0, 0, 0))
    )
    assert colour.ycc(white).flatten().tolist() == pytest.approx([1, 0, 0], abs=1e-6)
    assert colour.ycc(blue).flatten().tolist() == pytest.approx([0.0722, 0.5, -0.0458], abs=1e-4)
    assert colour.ycc(red).flatten().tolist() == pytest.approx([0.2126, -0.1146, 0.5], abs=1e-4)


def test_unit_range() -> None:
    # [-1, 1] to [0, 1], clamped.
    x = torch.tensor([-math.inf, -1.5, -1.0, 0.0, 1.0, 1.5, math.inf])
    assert torch.equal(colour.unit_range(x), torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))
