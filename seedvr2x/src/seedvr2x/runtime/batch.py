"""Upscaling of one batch of frames, the way numz runs one batch (milestone 1).

Every step is numz's, in its order and dtypes (src/core/generation_phases.py at 4490bd1): the
frames reach the GPU in bfloat16 through float16, the VAE encodes them, the noise is drawn from
the seed on the latent's memory layout, the DiT samples in one step, the VAE decodes, and the
frames are cropped to the output size and brought to [0, 1] in bfloat16.
"""

import torch
from torch import Tensor

from seedvr2x.runtime import model
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models


def upscale(models: Models, frames: Tensor, resolution: int, seed: int) -> Tensor:
    """Upscale frames (T, H, W, 3) float16 in [0, 1] on the CPU, T = 4n + 1, as one batch.

    Returns (T, H', W', 3) float32 in [0, 1] on the CPU, the short side at resolution
    (model.resized_size).
    """
    count, height, width, _ = frames.shape
    if count % 4 != 1:
        raise ValueError(f"{count} frames: one batch takes 4n + 1 frames")
    out_height, out_width = model.resized_size(height, width, resolution)
    with torch.no_grad():
        # Phase 1, encode (generation_phases.py:329-504). numz seeds here; nothing draws.
        model.set_seed(seed + 1_000_000)
        # (T, 3, H, W): a view, moved with its layout (generation_phases.py:92-104, 380-388)
        video = frames.permute(0, 3, 1, 2).to(models.device, COMPUTE_DTYPE)
        latent = model.encode(models, model.input_transform(resolution)(video))
        del video
        # Phase 2, the DiT (generation_phases.py:663-724). numz draws a second noise right after
        # this one, for its latent noise option: unused at scale 0, and left out.
        model.set_seed(seed)
        noise = torch.randn_like(latent, dtype=COMPUTE_DTYPE)
        cond = model.condition(models, noise, latent)
        sampled = model.sample(models, noise, cond)
        del latent, noise, cond
        # Phase 3, decode (generation_phases.py:900-958): (C, T, H, W) to (T, C, H, W), the
        # DivisiblePad padding cropped away.
        decoded = model.decode(models, sampled).permute(1, 0, 2, 3)
        decoded = decoded[:count, :, :out_height, :out_width]
        # Phase 4 (generation_phases.py:1340-1348): [-1, 1] to [0, 1] in place, in bfloat16.
        frames_out = decoded.permute(0, 2, 3, 1)
        frames_out.clamp_(-1, 1).mul_(0.5).add_(0.5)
        return frames_out.to("cpu", torch.float32)
