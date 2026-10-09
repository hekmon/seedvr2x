"""Colour correction (DESIGN.md, Colour correction): `split`, frame by frame.

The model drifts in colour (more saturation on 9 of 10 clips, b* toward yellow on 7:
research/docs/colour.md), and the VAE's tiles shift each tile's level and colour. `split` keeps
the decoded frame's detail and takes its coarse lightness and colour from the input, in BT.709
Y'CbCr: Y' below LIGHTNESS_STAGES à-trous stages, Cb and Cr below colour_stages. Nothing is pooled
over the shot, so each frame needs only its own decode and its reference frame, and the decode
streams.

- Ported from the colour study's split(), `split:ycc:4:SC` (research/scripts/colour_variants.py:
  split, split_space, to_space and from_space for ycc, mix, YCC, low_bands), the same operations
  in the same order: on the same frames, the two agree bit for bit on the CPU
  (tests/test_colour.py), and the study's measurements hold for this code.
- The wavelet split is written from its method (DESIGN.md), clean room: numz's comes from
  StableSR, whose licence is non-commercial.
- No numz code is left in the correction: the CIELAB conversions and the histogram matching
  ported from numz for `lab` went with it.
"""

from collections.abc import Iterable
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import Tensor

# The lightness's à-trous stages at every upscale factor: sigma 6.5 px of output. At 13 px (5,
# numz's split) the model's own lightness flickers more than lab's; at 3.2 px (3) fine texture
# starts to go (DESIGN.md, Colour correction; research/docs/colour.md).
LIGHTNESS_STAGES = 4
# The colour's fewest stages, sigma 1.6 px, below x1.41 (colour_stages).
COLOUR_STAGES_MIN = 2

# BT.709's Y'CbCr of gamma-encoded RGB, built as the study builds it (colour_variants.py, YCC):
# float64, cast to float32 where applied. A linear map, so the frames go through it in [-1, 1]
# as they are: only the difference of the low bands moves them.
YCC = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.2126 / 1.8556, -0.7152 / 1.8556, 0.9278 / 1.8556],
        [0.7874 / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748],
    ],
    dtype=torch.float64,
)
YCC_INV = torch.inverse(YCC)  # torch.linalg.inv's alias, which the study calls


def colour_stages(source: tuple[int, int], target: tuple[int, int]) -> int:
    """The à-trous stages of the colour (Cb, Cr) for frames of source (height, width), as stored,
    resized to target (height, width) (job.target_size): the stage whose sigma is nearest 1.6
    source pixels, the input's own colour resolution (a 4:2:0 source has a chroma sample every 2),
    round(2 + log2 f), and COLOUR_STAGES_MIN at the least. f is the upscale factor; where the
    display aspect gives the two axes different factors (anamorphic SD), their geometric mean.

    Rounded on a log scale, half up, from f² exactly, so that no float rounding decides a
    boundary: k stages when 2^(2k - 5) <= f² < 2^(2k - 3). That is 2 below x1.41 (√2), 3 (sigma
    3.2 px) from x1.41 to x2.83 (√8), 4 (sigma 6.5 px) from x2.83 to x5.66 (√32), and 5 above."""
    (source_height, source_width), (height, width) = source, target
    squared = Fraction(height, source_height) * Fraction(width, source_width)
    # Checked at x1.5, x2, x3 and x4, interpolated between (x2.25, 480p to 1080p): DESIGN.md,
    # Colour correction. PROVISIONAL from x5.66 on: 5 stages (sigma 13 px) are the same rule,
    # untested. The stages' own sigmas, √((4^k - 1)/6) (1.58, 3.24, 6.52, 13.06 px), would put
    # the nearest's boundaries at x1.41, x2.87 and x5.77: DESIGN.md's formula decides.
    stages = COLOUR_STAGES_MIN
    while squared >= Fraction(2) ** (2 * stages - 3):
        stages += 1
    return stages


def low_bands(frames: Tensor, stages: Iterable[int]) -> dict[int, Tensor]:
    """The low band of frames (..., H, W), each (H, W) plane apart, at each stage count of
    `stages` (counts >= 0), from one à-trous cascade, each count's band taken as the cascade
    passes it: {count: float32, the shape of frames}. 0 stages: the frames themselves.

    Stage s blurs the one before with the 3-by-3 binomial kernel, (1, 2, 1)/4 on each axis, its
    taps 2^s pixels apart (s from 0), at most an eighth of the smaller side, edges replicated at
    each stage. Away from the edges, k stages amount to a separable tent of 2^(k + 1) - 1 taps,
    sigma √((4^k - 1)/6): 4 stages 31 taps, sigma 6.52 px; 5 stages 63 taps, sigma 13.06 px."""
    *lead, height, width = frames.shape
    wanted = set(stages)
    if any(count < 0 for count in wanted):
        raise ValueError(f"stage counts {sorted(wanted)}: none can be negative")
    x = frames.reshape(-1, 1, height, width).to(torch.float32)
    cap = max(1, min(height, width) // 8)
    bands = {0: x.reshape(*lead, height, width)} if 0 in wanted else {}
    for stage in range(max(wanted, default=0)):
        d = min(2**stage, cap)
        # (a + 2b + c)/4: the weight of 4 exact, two roundings per axis.
        x = F.pad(x, (d, d, d, d), mode="replicate")
        x = (x[..., :, : -2 * d] + x[..., :, 2 * d :] + 2 * x[..., :, d:-d]) * 0.25
        x = (x[..., : -2 * d, :] + x[..., 2 * d :, :] + 2 * x[..., d:-d, :]) * 0.25
        if stage + 1 in wanted:
            bands[stage + 1] = x.reshape(*lead, height, width)
    return bands


def low_band(frames: Tensor, stages: int) -> Tensor:
    """The low band of frames (..., H, W) after `stages` à-trous stages (low_bands): float32, the
    shape of frames."""
    return low_bands(frames, (stages,))[stages]


def ycc(frames: Tensor) -> Tensor:
    """BT.709 Y'CbCr of frames (T, 3, H, W), gamma-encoded RGB in any range, here [-1, 1]: (T, 3,
    H, W) float32, Y' then Cb then Cr.

    Made contiguous first: a frame's values then never depend on its layout, such as the decode's
    (3, T, H, W) seen as (T, 3, H, W), which the matrix product could take another way, to other
    last bits."""
    return _mix(frames.to(torch.float32).contiguous(), YCC)


def rgb(frames: Tensor) -> Tensor:
    """RGB of frames (T, 3, H, W) float32 in Y'CbCr (ycc) of RGB in [-1, 1], brought to [0, 1] and
    clamped: (T, 3, H, W) float32."""
    return unit_range(_mix(frames, YCC_INV))


def _mix(frames: Tensor, matrix: Tensor) -> Tensor:
    """matrix (3, 3) applied to the channels of frames (T, 3, H, W) float32, as the study applies
    it (colour_variants.py, mix): in float32, on the frames' device."""
    weights = matrix.to(device=frames.device, dtype=torch.float32)
    return torch.einsum("ij,tjhw->tihw", weights, frames)


def transfer(content: Tensor, reference: Tensor, colour_stages: int) -> Tensor:
    """content moved onto the low bands of reference, in Y'CbCr (ycc): Y' below LIGHTNESS_STAGES,
    Cb and Cr below colour_stages. content (T, 3, H, W) is the VAE's decode, in [-1, 1]
    unclamped, reference (T, 3, H, W) the same frames of the input, in [-1, 1]: (T, 3, H, W)
    float32, Y'CbCr, unclamped.

    Computed as the study computes it: the content plus the low band of the difference, d =
    reference - content, one cascade per scale. That is content + (low(reference) -
    low(content)), the low band being linear, the same sum as the content's high band on the
    reference's low band without rounding a high band on its own: a float32 high band added back
    to its low band misses the image at 0.4 to 2.2% of the values of test frames, whereas content
    moved onto itself comes back bit for bit (DESIGN.md, Colour correction, Numerics)."""
    if content.shape != reference.shape:
        raise ValueError(f"content {tuple(content.shape)}, reference {tuple(reference.shape)}")
    moved = ycc(content)
    difference = ycc(reference).sub_(moved)
    lightness = moved[:, :1] + low_band(difference[:, :1], LIGHTNESS_STAGES)
    chroma = moved[:, 1:] + low_band(difference[:, 1:], colour_stages)
    return torch.cat([lightness, chroma], dim=1)


def split(content: Tensor, reference: Tensor, colour_stages: int) -> Tensor:
    """The colour correction of content against reference (transfer), back to RGB: (T, 3, H, W)
    float32 in [0, 1], kept float32 for the writer. Each frame is corrected alone. colour_stages
    follows the upscale factor (colour_stages)."""
    return rgb(transfer(content, reference, colour_stages))


def unit_range(frames: Tensor) -> Tensor:
    """frames in [-1, 1], the model's range, to [0, 1], clamped: float32."""
    return frames.to(torch.float32).add(1.0).mul_(0.5).clamp_(0.0, 1.0)
