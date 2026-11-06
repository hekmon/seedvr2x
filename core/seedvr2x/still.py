"""One picture through the models (DESIGN.md, Still images; Repository and apps, step 3): the
resize, the encode, one latent sampled, the decode, the colour correction against the picture in
memory. The library's brick: it takes models already loaded and never loads, unloads or
reconfigures them, beyond entering the three phases as the video path does (Models.enter), so
that a server can call it in a loop on the models it keeps.

A picture is a shot of one frame at frame 0, encoded alone (DESIGN.md, Still images, the model on
one picture) and seeded as such, so that the output is bit for bit what the shot's pass gives for
it (seedvr2x/shot_pass.py: encode_shot, sample_windows over the one window
[(0, 1)], decode_shot with its correction; tests/test_still_cli.py holds it). The tiled passes
keep their scratch in system memory (tiling.InMemory): a picture's is 191 MiB at 8K."""

from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from seedvr2x import colour, model, sampling, tiling
from seedvr2x.frames import output_size
from seedvr2x.model import COMPUTE_DTYPE, Models
from seedvr2x.passes import Tiles
from seedvr2x.sampling import ENCODE_SEED_OFFSET


class NonFinite(RuntimeError):
    """NaN or inf in what a picture's decode made: the picture isn't returned."""


@torch.no_grad()
def upscale(
    models: Models,
    pixels: npt.NDArray[np.float32],
    target: tuple[int, int],
    seed: int,
    *,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
) -> npt.NDArray[np.float32]:
    """The picture upscaled: (H', W', 3) float32 in [0, 1] on the CPU, (H', W') =
    frames.output_size(target).

    pixels is the picture as stored, (H, W, 3) float32 in [0, 1]. target is the (height, width)
    in pixels the encode resizes it to (frames.target_size, the caller's). seed is the picture's:
    its encode is seeded with seed + ENCODE_SEED_OFFSET and its noise drawn from seed, as a
    shot's first frame is (shot_pass.encode_shot, sample_windows). encode_tiles and decode_tiles,
    the plan's (None: one pass), tile the VAE's passes (tiling). correct runs the colour correction
    (colour.split) against the picture itself.

    The three steps one after the other: encoded, sampled, decoded, which a caller running
    several pictures at once calls apart, each model taking one picture at a time (the VAE's
    causal convolutions keep their memory in the modules). ValueError for pixels that aren't
    (H, W, 3) float32; NonFinite when the decode holds NaN or inf."""
    latent = encoded(models, pixels, target, seed, tiles=encode_tiles)
    return decoded(
        models, sampled(models, latent, seed), pixels, target, tiles=decode_tiles, correct=correct
    )


@torch.no_grad()
def encoded(
    models: Models,
    pixels: npt.NDArray[np.float32],
    target: tuple[int, int],
    seed: int,
    *,
    tiles: Tiles | None,
) -> Tensor:
    """The picture's latent, (1, h, w, 16) in system memory in the encode's channel-major layout
    (model.encode_stream): the encode's phase entered, seeded with seed + ENCODE_SEED_OFFSET,
    the picture resized on the device and encoded in `tiles` (None: one pass). ValueError for
    pixels that aren't (H, W, 3) float32 (upscale)."""
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.float32:
        raise ValueError(f"pixels {pixels.shape} {pixels.dtype}, not (H, W, 3) float32")
    # Encode: a shot of one frame is not padded to 4n + 1 and the VAE takes it in one slice
    # (model.encode_slices, frames.padded_length), so the encode's input is one tensor.
    models.enter("encode")
    model.set_seed(seed + ENCODE_SEED_OFFSET)
    frame = _transformed(models, model.to_input(pixels), target, COMPUTE_DTYPE)
    if tiles is None:
        latent = model.encode_stream(models, [frame], 1)
    else:
        latent = tiling.encode_tiled(models, [frame], 1, tiles, tiling.InMemory())
    del frame
    return latent


@torch.no_grad()
def sampled(models: Models, latent: Tensor, seed: int) -> Tensor:
    """The DiT's output for the picture's latent (encoded), (1, h, w, 16) moved to system memory,
    its layout kept: the DiT's phase entered, the noise drawn in system memory from seed
    (sampling.drawn, a generator of its own: nothing of the pass draws from torch's global
    ones), the one window [(0, 1)], which merge_windows returns as it is."""
    models.enter("dit")
    noise, latent = sampling.drawn(models, latent, seed)
    out = sampling.window(models, noise, latent)
    del noise, latent
    # decode_shot's start: the window's output leaves the device for system memory, and is
    # decoded from there (shot_pass._slices). Here before the decode's phase is entered, so that a
    # picture waiting for the VAE holds nothing on the device.
    model.synchronize(out.device)
    return out.to("cpu")


@torch.no_grad()
def decoded(
    models: Models,
    latent: Tensor,
    pixels: npt.NDArray[np.float32],
    target: tuple[int, int],
    *,
    tiles: Tiles | None,
    correct: bool,
) -> npt.NDArray[np.float32]:
    """The picture from the DiT's output (sampled), as upscale returns it: the decode's phase
    entered, the latent decoded in `tiles` (None: one pass), cropped to the target's even sides,
    and corrected against `pixels`, the picture as stored, where `correct`. NonFinite when the
    decode holds NaN or inf."""
    models.enter("decode")
    slices = (
        model.decode_stream(models, latent)
        if tiles is None
        else tiling.decode_tiled(models, latent, tiles, tiling.InMemory())
    )
    height, width = output_size(target)
    try:
        # One frame: one slice. Cropped to the target's even sides, dropping the padding, as
        # shot_pass._decoded crops, (1, H', W', C) in [-1, 1] unclamped, on the device.
        chunk = next(slices).permute(1, 2, 3, 0)[:1, :height, :width]
        if chunk.shape[0] != 1:
            raise RuntimeError(f"{chunk.shape[0]} frames decoded for a picture of one")
    finally:
        # A generator, as both decodes are; closing it clears the VAE's causal memory and the
        # tiles' scratch at once (shot_pass._slices).
        if isinstance(slices, Generator):
            slices.close()
    # As shot_pass.finite checks it, before the correction and the post-processing; the picture's
    # frame is the one part. Polled first, as finite is: the check's copy to the CPU would wait in
    # C for the whole decode, out of a stop's reach (waiting.py).
    model.synchronize(chunk.device)
    if not bool(torch.isfinite(chunk).all()):
        bad = int((~torch.isfinite(chunk)).sum())
        raise NonFinite(f"the picture's decode holds NaN or inf: {bad} of {chunk.numel()} values")
    if correct:
        stages = colour.colour_stages((pixels.shape[0], pixels.shape[1]), target)
        # split's reference is the picture through the encoder's transform in float32 (no
        # float16 or bfloat16 step), cropped as the decode is (shot_pass.reference_inputs, _frames).
        reference = _transformed(models, torch.as_tensor(pixels), target, torch.float32)
        reference = reference.permute(1, 0, 2, 3)[:, :, :height, :width]
        corrected = colour.split(chunk.permute(0, 3, 1, 2), reference, stages)
        model.synchronize(models.device)
        return corrected.permute(0, 2, 3, 1)[0].to("cpu").numpy()
    # Post-process (numz src/core/generation_phases.py:1340-1348 at 4490bd1): [-1, 1] to [0, 1].
    chunk.clamp_(-1, 1).mul_(0.5).add_(0.5)
    model.synchronize(models.device)
    return chunk[0].to("cpu", torch.float32).numpy()


def _transformed(
    models: Models, frame: Tensor, target: tuple[int, int], dtype: torch.dtype
) -> Tensor:
    """The picture's frame, (H, W, 3) made a tensor on the CPU by the caller (model.to_input for
    the encoder's input, float16 as numz's; torch.as_tensor for the reference), through
    model.input_transform(target) as dtype on the device: (C, 1, H', W') in [-1, 1],
    (H', W') = frames.padded_size(target) (shot_pass._transformed)."""
    frames = frame.unsqueeze(0).permute(0, 3, 1, 2).to(models.device, dtype)
    return model.input_transform(target)(frames)
