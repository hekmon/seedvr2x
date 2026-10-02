"""Upscaling of one shot (DESIGN.md, Pipeline, per shot).

The shot is VAE-encoded once. The DiT runs on windows of at most `window` latents, one window
per call, consecutive windows sharing SHARED latents; the shared latents are mixed with cosine
weights, and the whole shot is decoded in one stream. The noise is drawn once for the shot and
sliced per window. Every step keeps numz's order and dtypes (src/core/generation_phases.py at
4490bd1), so a shot that fits one window gives numz's one-batch output, bit for bit
(milestone 1).
"""

import math

import torch
from torch import Tensor

from seedvr2x.runtime import model
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models

# Latents shared by consecutive windows, M (DESIGN.md, Pipeline step 2): M = 2 cut the boundary
# jump by 80% against independent batches in the stitching study (research/docs/stitching.md).
SHARED = 2


def window_layout(latents: int, window: int, shared: int = SHARED) -> list[tuple[int, int]]:
    """[start, end) of the windows over a shot of `latents` latents: as few windows as `window`
    allows, their lengths balanced (differing by at most 1, the longer first), consecutive ones
    sharing `shared` latents (DESIGN.md, Pipeline step 2)."""
    if latents <= window:
        return [(0, latents)]
    if window < 2 * shared + 1:
        raise ValueError(
            f"windows of {window} latents leave none of their own with {shared} shared"
        )
    count = math.ceil((latents - shared) / (window - shared))
    base, longer = divmod(latents + (count - 1) * shared, count)
    layout: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        length = base + 1 if index < longer else base
        layout.append((start, start + length))
        start += length - shared
    # A latent is in at most two windows: a window is at least as long as the two zones it
    # shares with its neighbours.
    assert layout[-1][1] == latents and base >= 2 * shared
    return layout


def mix_weights(shared: int) -> list[float]:
    """The earlier window's weights on the latents two windows share: cosine, 0.75 then 0.25 for
    2, the curve the stitching study measured (blend_patch.py, weights)."""
    return [0.5 + 0.5 * math.cos(math.pi * i / (shared + 1)) for i in range(1, shared + 1)]


def pad_4n1(frames: Tensor) -> Tensor:
    """frames (T, ...) padded at the end to 4n + 1, as numz pads a batch
    (src/core/generation_utils.py, pad_video_temporal): the frames before the last, mirrored,
    and the last one repeated beyond a full mirror. The VAE packs 4 frames per latent after the
    first."""
    count = frames.shape[0]
    missing = (-count + 1) % 4
    if missing == 0:
        return frames
    if missing >= count:
        repeated = frames[-1:].repeat(missing - count + 1, *([1] * (frames.dim() - 1)))
        return torch.cat([frames, frames[1:].flip(0), repeated])
    return torch.cat([frames, frames[-missing - 1 : -1].flip(0)])


def upscale_shot(
    models: Models,
    frames: Tensor,
    resolution: int,
    seed: int,
    window: int | None = None,
    *,
    reseed_windows: bool = False,
) -> Tensor:
    """Upscale one shot, frames (T, H, W, 3) float16 in [0, 1] on the CPU.

    window caps the DiT windows, in latents (4 frames each after the first); None runs the shot
    in one window. Returns (T, H', W', 3) float32 in [0, 1] on the CPU, the short side at
    resolution.

    reseed_windows is for tests only. Each window then reseeds and draws its own noise, as the
    stitching study's reference implementation did (blend_patch.py, STITCH_LATENT), so that the
    windows, the mixing and the decode can be checked against it bit for bit.
    """
    count, height, width, _ = frames.shape
    out_height, out_width = model.resized_size(height, width, resolution)
    with torch.no_grad():
        # Encode (generation_phases.py:329-504). numz seeds here; nothing draws.
        model.set_seed(seed + 1_000_000)
        # (T, 3, H, W): a view, moved with its layout (generation_phases.py:92-104, 380-388)
        video = pad_4n1(frames).permute(0, 3, 1, 2).to(models.device, COMPUTE_DTYPE)
        latent = model.encode(models, model.input_transform(resolution)(video))
        del video
        layout = window_layout(latent.shape[0], window or latent.shape[0])
        if reseed_windows:
            sampled = _sample_reseeded(models, latent, layout, seed)
        else:
            # One draw for the shot, on the latent's layout as numz draws it for a batch
            # (generation_phases.py:663-680), sliced per window.
            model.set_seed(seed)
            noise = torch.randn_like(latent, dtype=COMPUTE_DTYPE)
            sampled = [
                model.sample(models, noise[s:e], model.condition(models, noise[s:e], latent[s:e]))
                for s, e in layout
            ]
        del latent
        merged = sampled[0] if len(sampled) == 1 else _merge(sampled, layout)
        del sampled
        # Decode (generation_phases.py:900-958) and post-process (:1340-1348), slice by slice:
        # (C, t, H, W) to (t, H, W, C) without the padding, then [-1, 1] to [0, 1] in place.
        chunks: list[Tensor] = []
        for decoded in model.decode_stream(models, merged.to(models.device)):
            chunk = decoded.permute(1, 2, 3, 0)[:, :out_height, :out_width]
            chunk.clamp_(-1, 1).mul_(0.5).add_(0.5)
            chunks.append(chunk.to("cpu", torch.float32))
        return torch.cat(chunks)[:count]


def _sample_reseeded(
    models: Models, latent: Tensor, layout: list[tuple[int, int]], seed: int
) -> list[Tensor]:
    """Each window's DiT output with noise drawn per window, as numz ran the study's windows:
    the latent offloaded to the CPU, each window cloned from it, the seed reset, the clone moved
    back and the noise drawn on it (blend_patch.py, latent_upscale; generation_phases.py:654-680).
    """
    offloaded = latent.cpu()
    sampled: list[Tensor] = []
    for s, e in layout:
        window_latent = offloaded[s:e].clone()
        model.set_seed(seed)
        window_latent = window_latent.to(models.device, COMPUTE_DTYPE)
        noise = torch.randn_like(window_latent, dtype=COMPUTE_DTYPE)
        cond = model.condition(models, noise, window_latent)
        sampled.append(model.sample(models, noise, cond))
    return sampled


def _merge(sampled: list[Tensor], layout: list[tuple[int, int]]) -> Tensor:
    """The windows' DiT outputs merged into the shot's latents: on the latents two windows share,
    earlier * w + later * (1 - w). The expression, bfloat16 and the CPU are the study's
    (blend_patch.py, latent_upscale)."""
    merged = sampled[0].cpu()
    for (_, previous_end), (start, _), current in zip(
        layout, layout[1:], sampled[1:], strict=False
    ):
        shared = previous_end - start
        current = current.cpu()
        weights = torch.tensor(mix_weights(shared), dtype=torch.float32).to(current.dtype)
        weights = weights.view(shared, *([1] * (current.dim() - 1)))
        mixed = merged[start:previous_end] * weights + current[:shared] * (1.0 - weights)
        merged = torch.cat([merged[:start], mixed, current[shared:]])
    return merged
