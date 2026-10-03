"""Colour correction (DESIGN.md, Colour correction): numz's `lab`, pooled over a shot.

The model drifts in colour (+30% saturation, toward blue: research/docs/quality.md). `lab` gives
each frame the input's low frequencies under the decoded frame's details (the wavelet split),
then matches the CIELAB values of the result to the input's: a* and b* fully, L* by a fifth.

- The wavelet split is written from its method (DESIGN.md), clean room: numz's comes from
  StableSR, whose licence is non-commercial.
- The CIELAB conversions and the matching are ported from numz (ComfyUI-SeedVR2_VideoUpscaler,
  numz and contributors, src/utils/color_fix.py:249-521 at 4490bd1, Apache-2.0; see NOTICE).
  Changed for seedvr2x: the matching is pooled over the shot through integer histograms, where
  numz sorts each batch's values.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# The wavelet split's à-trous stages: each blurs the one before with the 3-by-3 binomial kernel,
# (1, 2, 1)/4 on each axis, its taps 2^stage pixels apart (DESIGN.md, Colour correction).
STAGES = 5

# sRGB and CIELAB under D65, numz's constants (color_fix.py:303-317).
RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
XYZ_TO_RGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
EPSILON = 6.0 / 29.0
KAPPA = (29.0 / 3.0) ** 3

# The share of its own L* each value keeps, the rest matched: numz's (generation_phases.py:1303).
LUMINANCE_WEIGHT = 0.8

# The histograms: 2^16 bins per channel over [-128, 128), 1/256 unit each. They hold the CIELAB
# values of every RGB in [0, 1]: L* 0 to 100, a* -86.2 to 98.2, b* -107.9 to 94.5, at the RGB
# cube's corners (tests/test_colour.py).
BINS = 1 << 16
LAB_LOW = -128.0
PER_UNIT = BINS / (-2 * LAB_LOW)


def low_band(frames: Tensor) -> Tensor:
    """The low band of the wavelet split of frames (..., H, W), each (H, W) plane apart: STAGES
    à-trous stages, their taps 1, 2, 4, 8 and 16 pixels apart, at most an eighth of the smaller
    side, edges replicated at each stage. Away from the edges, a separable 63-tap tent filter,
    (32 - |k|)/1024, a standard deviation of 13 px. float32, the shape of frames."""
    *lead, height, width = frames.shape
    x = frames.reshape(-1, 1, height, width).to(torch.float32)
    cap = max(1, min(height, width) // 8)
    for stage in range(STAGES):
        d = min(2**stage, cap)
        # (a + 2b + c)/4: the weight of 4 exact, two roundings per axis.
        x = F.pad(x, (d, d, d, d), mode="replicate")
        x = (x[..., :, : -2 * d] + x[..., :, 2 * d :] + 2 * x[..., :, d:-d]) * 0.25
        x = (x[..., : -2 * d, :] + x[..., 2 * d :, :] + 2 * x[..., d:-d, :]) * 0.25
    return x.reshape(*lead, height, width)


def transfer(content: Tensor, reference: Tensor) -> Tensor:
    """content's high band on reference's low band (low_band), both (..., H, W) of one shape:
    float32, unclamped.

    Computed as content plus the difference of the low bands, the same sum without rounding a high
    band on its own: a float32 high band added back to its low band misses the image at 0.4 to
    2.2% of the values of test frames, where content transferred onto itself comes back bit for
    bit."""
    if content.shape != reference.shape:
        raise ValueError(f"content {tuple(content.shape)}, reference {tuple(reference.shape)}")
    return content.to(torch.float32) + (low_band(reference) - low_band(content))


def unit_range(frames: Tensor) -> Tensor:
    """frames in [-1, 1], the model's range, to [0, 1], clamped: float32, as numz maps both its
    inputs (color_fix.py:319-321)."""
    return frames.to(torch.float32).add(1.0).mul_(0.5).clamp_(0.0, 1.0)


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """CIELAB under D65 of rgb (B, 3, H, W) float32 in [0, 1], sRGB: (B, 3, H, W) float32, L*
    then a* then b*. numz's _rgb_to_lab_batch (color_fix.py:368-413), bit for bit.

    Made contiguous first, as lab_to_rgb: from another layout, such as the decode's (3, B, H, W),
    the (B·H·W, 3) pixels are a strided view, which the matrix product takes another way, to other
    last bits. So a frame's values never depend on its layout. On the CPU they depend on the
    call's size and threads, though: torch's pow rounds the scalar tail of each thread's share
    apart from its vectorised body (258 of 1.56M values differ between 16 frames in one call and
    one at a time), so the two passes over a shot convert it in the same calls."""
    rgb = rgb.contiguous()
    matrix = torch.tensor(RGB_TO_XYZ, dtype=torch.float32, device=rgb.device)
    linear = torch.where(rgb > 0.04045, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
    batch, _, height, width = linear.shape
    flat = linear.permute(0, 2, 3, 1).reshape(-1, 3)
    del linear
    xyz = torch.matmul(flat, matrix.T).reshape(batch, height, width, 3).permute(0, 3, 1, 2)
    del flat
    xyz[:, 0].div_(0.95047)
    xyz[:, 2].div_(1.08883)
    f = torch.where(
        xyz > EPSILON**3, torch.pow(xyz, 1.0 / 3.0), xyz.mul(KAPPA).add_(16.0).div_(116.0)
    )
    del xyz
    lightness = f[:, 1].mul(116.0).sub_(16.0)
    a = (f[:, 0] - f[:, 1]).mul_(500.0)
    b = (f[:, 1] - f[:, 2]).mul_(200.0)
    return torch.stack([lightness, a, b], dim=1)


def lab_to_rgb(lab: Tensor) -> Tensor:
    """sRGB of lab (B, 3, H, W) float32, CIELAB under D65 (rgb_to_lab): (B, 3, H, W) float32,
    clamped to [0, 1]. numz's _lab_to_rgb_batch (color_fix.py:416-474), bit for bit."""
    lab = lab.contiguous()
    matrix = torch.tensor(XYZ_TO_RGB, dtype=torch.float32, device=lab.device)
    lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (lightness + 16.0) / 116.0
    fx = a.div(500.0).add_(fy)
    fz = fy - b / 200.0
    x = torch.where(fx > EPSILON, torch.pow(fx, 3.0), fx.mul(116.0).sub_(16.0).div_(KAPPA))
    y = torch.where(fy > EPSILON, torch.pow(fy, 3.0), fy.mul(116.0).sub_(16.0).div_(KAPPA))
    z = torch.where(fz > EPSILON, torch.pow(fz, 3.0), fz.mul(116.0).sub_(16.0).div_(KAPPA))
    del fx, fy, fz
    x.mul_(0.95047)
    z.mul_(1.08883)
    xyz = torch.stack([x, y, z], dim=1)
    del x, y, z
    batch, _, height, width = xyz.shape
    flat = torch.matmul(xyz.permute(0, 2, 3, 1).reshape(-1, 3), matrix.T)
    del xyz
    linear = flat.reshape(batch, height, width, 3).permute(0, 3, 1, 2)
    rgb = torch.where(
        linear > 0.0031308,
        torch.pow(torch.clamp(linear, min=0.0), 1.0 / 2.4).mul_(1.055).sub_(0.055),
        linear * 12.92,
    )
    return torch.clamp(rgb, 0.0, 1.0)


class Histograms:
    """The CIELAB histograms of a shot's corrected frames (transfer) and of its input, pooled over
    the shot: BINS per channel, counted in integers, so the same in whatever order and grouping
    the frames come.

    match maps each value to the reference's at the same quantile, numz's mapping
    (color_fix.py:477-521), linearly within a bin. A sort also orders the values inside a bin,
    and spreads tied values by their position, which a histogram can't see: each value comes out
    within one bin of what numz gives the values of its bin, and tied values alike."""

    def __init__(self, device: torch.device) -> None:
        self.content = torch.zeros(3, BINS, dtype=torch.int64, device=device)
        self.reference = torch.zeros(3, BINS, dtype=torch.int64, device=device)

    def add(self, content: Tensor, reference: Tensor) -> None:
        """Count content and reference, the CIELAB values (B, 3, H, W) float32 (rgb_to_lab) of
        the same frames, corrected and input."""
        if content.shape != reference.shape:
            raise ValueError(f"content {tuple(content.shape)}, reference {tuple(reference.shape)}")
        # Both checked before anything is counted: a refused call leaves the histograms as they
        # were.
        _inside(content)
        _inside(reference)
        for channel in range(3):
            for counts, values in ((self.content, content), (self.reference, reference)):
                index = _positions(values[:, channel]).floor_().long().flatten()
                counts[channel] += torch.bincount(index, minlength=BINS)

    def match(self, content: Tensor) -> Tensor:
        """content, CIELAB values (B, 3, H, W) float32 among those counted, matched: a* and b* to
        the reference's, L* to LUMINANCE_WEIGHT of its own and the rest matched, as numz does
        (color_fix.py:330-343). (B, 3, H, W) float32."""
        totals = [int(total) for total in self.content.sum(dim=1)]
        references = [int(total) for total in self.reference.sum(dim=1)]
        if totals != references or len(set(totals)) != 1 or not totals[0]:
            raise ValueError(f"{totals} content values counted, {references} reference values")
        _inside(content)
        lightness, a, b = (
            _match(content[:, channel], self.content[channel], self.reference[channel])
            for channel in range(3)
        )
        blended = content[:, 0].mul(LUMINANCE_WEIGHT).add_(lightness.mul(1.0 - LUMINANCE_WEIGHT))
        return torch.stack([blended, a, b], dim=1)


def correct(content: Tensor, reference: Tensor) -> Tensor:
    """numz's `lab` (color_fix.py:249-365) on frames held at once, its one batch: content (T, 3,
    H, W) as the VAE decodes it, in [-1, 1] unclamped, against reference (T, 3, H, W), the
    encoder's input, in [-1, 1]. (T, 3, H, W) float32 in [0, 1], kept float32 for the writer."""
    corrected = rgb_to_lab(unit_range(transfer(content, reference)))
    histograms = Histograms(content.device)
    histograms.add(corrected, rgb_to_lab(unit_range(reference)))
    return lab_to_rgb(histograms.match(corrected))


def _inside(values: Tensor) -> None:
    """Refuse CIELAB values outside the histograms' range, NaN included: no RGB in [0, 1] gives
    them."""
    if not bool(((values >= LAB_LOW) & (values < -LAB_LOW)).all()):
        raise ValueError(f"CIELAB values outside [{LAB_LOW:g}, {-LAB_LOW:g}), or NaN")


def _positions(values: Tensor) -> Tensor:
    """values, CIELAB float32 inside the histograms' range (_inside), in bins from LAB_LOW:
    float64, exact but within 2^-22 of 0, where the rounding moves a value by less than a bin."""
    return (values.to(torch.float64) - LAB_LOW) * PER_UNIT


def _match(values: Tensor, content: Tensor, reference: Tensor) -> Tensor:
    """values, one channel's, mapped from the content's histogram to the reference's (BINS, equal
    totals): each to the reference's value at its quantile, linearly within each bin. float32."""
    position = _positions(values)
    index = position.floor()
    within = position - index
    bins = index.long()
    # Its rank among the content's values: below the total, as within < 1, but for a rounding up
    # to it at the top of a long shot.
    rank = (content.cumsum(0) - content).double()[bins] + within * content.double()[bins]
    # The reference's bin holding that rank, never empty: the first whose end passes it, or the
    # last with values for the total itself.
    ends = reference.cumsum(0)
    last = int(reference.nonzero()[-1])
    found = torch.searchsorted(ends.double(), rank, right=True).clamp_(max=last)
    start = (ends - reference).double()[found]
    matched = found.double() + (rank - start) / reference.double()[found]
    return (matched / PER_UNIT + LAB_LOW).to(torch.float32)
