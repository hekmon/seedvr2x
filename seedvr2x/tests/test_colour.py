"""The colour correction's parts (runtime/colour.py), on the CPU: the CIELAB conversions are numz's
bit for bit, the matching is within a bin of numz's sort, the wavelet split is an à-trous
computed apart in float64 and, away from the edges, the 63-tap tent it amounts to; and the
correction is deterministic.

numz's code is read from the pinned commit's git object, as tools/vendor.py reads it, and only
its own functions (Apache-2.0) are loaded: the wavelet functions beside them are StableSR's, under
a non-commercial licence, and are neither loaded nor run here."""

import ast
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor

from seedvr2x.runtime import colour

NUMZ = Path(__file__).resolve().parents[2] / "upstream" / "seedvr2-numz"
NUMZ_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"  # tools/vendor.py
# numz's own functions in color_fix.py, the only ones loaded, and the constants lab_color_transfer
# gives them.
NUMZ_FUNCTIONS = ("_rgb_to_lab_batch", "_lab_to_rgb_batch", "_histogram_matching_channel")
NUMZ_CONSTANTS = ("rgb_to_xyz_matrix", "xyz_to_rgb_matrix", "epsilon", "kappa")
CPU = torch.device("cpu")
# One bin of DESIGN.md's 2^16 over [-128, 128), and the float32 rounding of a matched value.
WITHIN = 1 / 256 + 2**-16
match_channel = colour._match  # pyright: ignore[reportPrivateUsage]


def numz_source(path: str) -> ast.Module:
    shown = subprocess.run(
        ["git", "-C", str(NUMZ), "show", f"{NUMZ_COMMIT}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return ast.parse(shown.stdout)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.fixture(scope="module")
def numz() -> dict[str, Any]:
    """numz's functions, and the constants lab_color_transfer gives them, by name."""
    if not (NUMZ / ".git").exists():
        pytest.skip("numz submodule not checked out")
    module = numz_source("src/utils/color_fix.py")
    functions: list[ast.stmt] = [_function(module, name) for name in NUMZ_FUNCTIONS]
    found: dict[str, Any] = {"torch": torch, "Tensor": Tensor}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "color_fix.py", "exec"), found)
    transfer = _function(module, "lab_color_transfer")
    for node in ast.walk(transfer):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in NUMZ_CONSTANTS
        ):
            value = node.value
            if isinstance(value, ast.Call) and value.args:  # torch.tensor([...], dtype=…)
                value = value.args[0]
            expression = ast.Expression(body=value)
            found[node.targets[0].id] = eval(compile(expression, "color_fix.py", "eval"), {})
    # The weight numz runs it with.
    phases = numz_source("src/core/generation_phases.py")
    [call] = [
        node
        for node in ast.walk(phases)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "lab_color_transfer"
    ]
    [weight] = [k.value for k in call.keywords if k.arg == "luminance_weight"]
    found["luminance_weight"] = ast.literal_eval(weight)
    return found


def test_constants_numz(numz: dict[str, Any]) -> None:
    assert numz["rgb_to_xyz_matrix"] == [list(row) for row in colour.RGB_TO_XYZ]
    assert numz["xyz_to_rgb_matrix"] == [list(row) for row in colour.XYZ_TO_RGB]
    assert (numz["epsilon"], numz["kappa"]) == (colour.EPSILON, colour.KAPPA)
    assert numz["luminance_weight"] == colour.LUMINANCE_WEIGHT


def scene(frames: int = 4, height: int = 48, width: int = 64, seed: int = 0) -> Tensor:
    """RGB frames (T, 3, H, W) in [0, 1] that change from frame to frame: gradients, edges, a
    flat area whose pixels are tied, and noise elsewhere."""
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, height).view(1, -1, 1)
    x = torch.linspace(0, 1, width).view(1, 1, -1)
    t = torch.arange(frames).view(-1, 1, 1) / frames
    shape = (frames, height, width)
    red = (0.2 + 0.6 * x * (1 - y) + 0.1 * t).expand(shape)
    green = (0.5 + 0.4 * torch.sin(6 * x + 3 * y + t)).expand(shape)
    blue = (((x + y + 0.1 * t) % 0.5 > 0.25).float() * 0.6 + 0.2).expand(shape)
    rgb = torch.stack([red, green, blue], dim=1)
    rgb = rgb + 0.03 * torch.randn(rgb.shape, generator=generator)
    rgb[:, :, : height // 4, : width // 4] = torch.tensor([0.7, 0.3, 0.4]).view(1, 3, 1, 1)
    return rgb.clamp(0.0, 1.0)


def drifted(rgb: Tensor) -> Tensor:
    """rgb as the model renders it: 30% more saturated, toward blue (research/docs/quality.md)."""
    grey = rgb.mean(dim=1, keepdim=True)
    shift = torch.tensor([-0.02, 0.0, 0.04]).view(1, 3, 1, 1)
    return (grey + 1.3 * (rgb - grey) + shift).clamp(0.0, 1.0)


def rgb_samples() -> Tensor:
    """RGB (6, 3, 37, 53) in [0, 1], random, with the conversions' thresholds and their
    neighbours."""
    rgb = torch.rand(6, 3, 37, 53, generator=torch.Generator().manual_seed(1))
    edges = torch.tensor([0.0, 1.0, 0.04045, 0.0031308, 0.5, 1e-7])
    edges = torch.cat(
        [edges, torch.nextafter(edges, torch.zeros(1)), torch.nextafter(edges, torch.ones(1))]
    )
    rgb.view(-1)[: edges.numel()] = edges
    return rgb


def test_rgb_to_lab_numz(numz: dict[str, Any]) -> None:
    rgb = rgb_samples()
    matrix = torch.tensor(numz["rgb_to_xyz_matrix"], dtype=torch.float32)
    expected = numz["_rgb_to_lab_batch"](rgb.clone(), CPU, matrix, numz["epsilon"], numz["kappa"])
    assert torch.equal(colour.rgb_to_lab(rgb), expected)
    # Laid out as the decode gives frames, (3, T, H, W), seen as (T, 3, H, W).
    assert torch.equal(
        colour.rgb_to_lab(rgb.transpose(0, 1).contiguous().transpose(0, 1)), expected
    )


def test_lab_to_rgb_numz(numz: dict[str, Any]) -> None:
    # In gamut and out of it.
    generator = torch.Generator().manual_seed(2)
    span = torch.tensor([110.0, 260.0, 260.0]).view(1, 3, 1, 1)
    offset = torch.tensor([-5.0, -130.0, -130.0]).view(1, 3, 1, 1)
    lab = torch.cat(
        [
            torch.rand(6, 3, 37, 53, generator=generator) * span + offset,
            colour.rgb_to_lab(rgb_samples()),
        ]
    )
    matrix = torch.tensor(numz["xyz_to_rgb_matrix"], dtype=torch.float32)
    expected = numz["_lab_to_rgb_batch"](lab.clone(), CPU, matrix, numz["epsilon"], numz["kappa"])
    assert torch.equal(colour.lab_to_rgb(lab), expected)
    assert torch.equal(
        colour.lab_to_rgb(lab.transpose(0, 1).contiguous().transpose(0, 1)), expected
    )


def bounds(numz: dict[str, Any], content: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
    """The least and the greatest value numz's sort gives the values of each one's content bin,
    one channel's (B, H, W)."""
    matched = numz["_histogram_matching_channel"](content, reference, CPU)
    bins = ((content.double() + 128) * 256).floor().long().flatten()
    least = torch.full((1 << 16,), math.inf).scatter_reduce(0, bins, matched.flatten(), "amin")
    most = torch.full((1 << 16,), -math.inf).scatter_reduce(0, bins, matched.flatten(), "amax")
    return least[bins].view(content.shape), most[bins].view(content.shape)


@pytest.mark.parametrize("quantised", [False, True], ids=["continuous", "8-bit"])
def test_matching_numz(numz: dict[str, Any], quantised: bool) -> None:
    # Each value within one bin of what numz's sort gives the values of its bin, tied values
    # alike: 8-bit frames tie wherever their RGB values do.
    reference_rgb = scene()
    content_rgb = drifted(scene(seed=1))
    if quantised:
        reference_rgb, content_rgb = (
            (rgb * 255).round() / 255 for rgb in (reference_rgb, content_rgb)
        )
    content, reference = colour.rgb_to_lab(content_rgb), colour.rgb_to_lab(reference_rgb)
    histograms = colour.Histograms(CPU)
    histograms.add(content, reference)
    for channel in range(3):
        ours = match_channel(
            content[:, channel], histograms.content[channel], histograms.reference[channel]
        )
        least, most = bounds(numz, content[:, channel], reference[:, channel])
        assert bool((ours >= least - WITHIN).all() and (ours <= most + WITHIN).all()), channel
        values, groups = torch.unique(content[:, channel], return_inverse=True)
        assert values.numel() < content[:, channel].numel()  # ties there are
        tied = torch.full_like(values, math.inf).scatter_reduce(
            0, groups.flatten(), ours.flatten(), "amin"
        )
        assert torch.equal(tied[groups], ours), channel
    matched = histograms.match(content)
    # L* a fifth matched, by numz's expression (color_fix.py:339) with the weight it runs.
    lightness = match_channel(content[:, 0], histograms.content[0], histograms.reference[0])
    weight = numz["luminance_weight"]
    assert torch.equal(matched[:, 0], content[:, 0].mul(weight).add_(lightness.mul(1.0 - weight)))
    assert torch.equal(
        matched[:, 1:],
        torch.stack(
            [
                match_channel(content[:, c], histograms.content[c], histograms.reference[c])
                for c in (1, 2)
            ],
            dim=1,
        ),
    )


def test_matching_identity() -> None:
    # Linear within a bin: a shot matched onto itself comes back, but for float32's rounding.
    lab = colour.rgb_to_lab(scene())
    histograms = colour.Histograms(CPU)
    histograms.add(lab, lab)
    assert float((histograms.match(lab) - lab).abs().max()) <= 2**-16


def test_histograms_hold_every_rgb() -> None:
    # The CIELAB values of every RGB in [0, 1] are inside the histograms' range: its extremes are
    # at the RGB cube's corners, among this grid's points.
    steps = torch.linspace(0, 1, 65)
    rgb = torch.stack(torch.meshgrid(steps, steps, steps, indexing="ij")).reshape(1, 3, 65, -1)
    lab = colour.rgb_to_lab(rgb)
    assert lab.amin(dim=(0, 2, 3)).tolist() == pytest.approx([0.0, -86.18, -107.86], abs=0.01)
    assert lab.amax(dim=(0, 2, 3)).tolist() == pytest.approx([100.0, 98.23, 94.48], abs=0.01)
    colour.Histograms(CPU).add(lab, lab)  # not refused


def test_histograms_pooled() -> None:
    # Counted in integers: the same frames give the same histograms, and the same matches,
    # added together or one by one, in any order.
    content, reference = colour.rgb_to_lab(drifted(scene(seed=1))), colour.rgb_to_lab(scene())
    together, apart, backwards = (colour.Histograms(CPU) for _ in range(3))
    together.add(content, reference)
    for frame in range(content.shape[0]):
        apart.add(content[frame : frame + 1], reference[frame : frame + 1])
    backwards.add(content.flip(0), reference.flip(0))
    for histograms in (apart, backwards):
        assert torch.equal(histograms.content, together.content)
        assert torch.equal(histograms.reference, together.reference)
        assert torch.equal(histograms.match(content), together.match(content))


def test_histograms_refuse() -> None:
    lab = colour.rgb_to_lab(scene())
    histograms = colour.Histograms(CPU)
    with pytest.raises(ValueError, match="values counted"):
        histograms.match(lab)  # nothing counted
    for wrong in (math.nan, 128.0, -128.5):
        bad = lab.clone()
        bad[0, 1, 0, 0] = wrong
        with pytest.raises(ValueError, match="outside"):
            histograms.add(bad, lab)
        with pytest.raises(ValueError, match="outside"):
            histograms.add(lab, bad)
        assert not histograms.content.any() and not histograms.reference.any()  # nothing counted
    histograms.add(lab, lab)
    histograms.add(lab[:1], lab[:1])
    histograms.reference[0, 0] += 1  # one value more on one side
    with pytest.raises(ValueError, match="values counted"):
        histograms.match(lab)


def a_trous(plane: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The low band in float64, written apart from colour.low_band: each stage's 3-by-3 binomial
    kernel tap by tap, the indices clamped to the plane."""
    height, width = plane.shape
    cap = max(1, min(height, width) // 8)
    rows, columns = np.arange(height), np.arange(width)
    weights = {-1: 0.25, 0: 0.5, 1: 0.25}
    x = plane
    for stage in range(5):
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


@pytest.mark.parametrize("size", [(128, 144), (40, 57)], ids=["taps 1 to 16", "taps capped at 5"])
def test_low_band_a_trous(size: tuple[int, int]) -> None:
    frames = torch.rand(2, 3, *size, generator=torch.Generator().manual_seed(3)) * 2 - 1
    low = colour.low_band(frames)
    assert low.dtype == torch.float32 and low.shape == frames.shape
    planes = frames.double().reshape(-1, *size).numpy()
    expected = np.stack([a_trous(plane) for plane in planes]).reshape(frames.shape)
    assert np.abs(low.numpy() - expected).max() < 1e-7


def test_low_band_tent() -> None:
    # Away from the edges, 31 pixels and more, the stages amount to a separable tent of 63 taps.
    frame = torch.rand(130, 150, generator=torch.Generator().manual_seed(4)) * 2 - 1
    tent = (32 - np.abs(np.arange(-31, 32))) / 1024

    def convolve(line: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.convolve(line, tent, mode="valid")

    expected = np.apply_along_axis(
        convolve, 0, np.apply_along_axis(convolve, 1, frame.double().numpy())
    )
    low = colour.low_band(frame)[31:-31, 31:-31].numpy()
    assert low.shape == expected.shape
    assert np.abs(low - expected).max() < 1e-7


def test_transfer_exact() -> None:
    # The split loses nothing: frames transferred onto themselves come back bit for bit.
    content = (
        torch.rand(2, 3, 64, 80, generator=torch.Generator().manual_seed(5)) * 2.1 - 1.05
    ).to(torch.bfloat16)
    assert torch.equal(colour.transfer(content, content), content.float())
    # Onto another, the content's high band on the reference's low band.
    reference = torch.rand(2, 3, 64, 80, generator=torch.Generator().manual_seed(6)) * 2 - 1
    high = content.float() - colour.low_band(content)
    moved = colour.transfer(content, reference)
    assert torch.allclose(moved - colour.low_band(reference), high, rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        colour.transfer(content, reference[:1])  # one frame's low band for all: refused


def test_unit_range() -> None:
    # [-1, 1] to [0, 1], clamped as numz's clamp to [-1, 1] then its map would.
    x = torch.tensor([-math.inf, -1.5, -1.0, 0.0, 1.0, 1.5, math.inf])
    assert torch.equal(colour.unit_range(x), torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]))


def test_correct() -> None:
    # The model's drift taken out, the same way every time, from the decode's layout too.
    reference = scene() * 2 - 1
    content = (drifted(scene()) * 2 - 1).to(torch.bfloat16)
    corrected = colour.correct(content, reference)
    assert corrected.dtype == torch.float32 and corrected.shape == reference.shape
    assert 0 <= float(corrected.min()) and float(corrected.max()) <= 1
    assert torch.equal(colour.correct(content, reference), corrected)
    decoded = content.transpose(0, 1).contiguous().transpose(0, 1)
    assert torch.equal(colour.correct(decoded, reference), corrected)
    spread = {
        name: colour.rgb_to_lab(rgb)[:, 1:].std(dim=(0, 2, 3))
        for name, rgb in (
            ("input", colour.unit_range(reference)),
            ("model", colour.unit_range(content)),
            ("corrected", corrected),
        )
    }
    assert ((spread["model"] / spread["input"] - 1).abs() > 0.1).all()
    assert ((spread["corrected"] / spread["input"] - 1).abs() < 0.01).all()
