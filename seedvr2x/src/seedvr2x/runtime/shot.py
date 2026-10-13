"""Upscaling of one shot (DESIGN.md, Pipeline, per shot), streamed end to end unless the plan
tiles its encode or its decode.

The shot's frames are read as the VAE encodes them, in its slices of 4 frames, and the shot is
encoded once. The DiT runs on windows of at most `window` latents, one window per call,
consecutive windows sharing SHARED latents; the shared latents are mixed with cosine weights, and
the whole shot is decoded in one stream, its frames written as they come. The noise is drawn once
for the shot, on the device, and sliced per window: it waits in system memory with the latent
between two windows, each window's slices sent to the device for its pass, so that the DiT's
phase holds nothing of the shot there beside the window it runs. Every step keeps numz's order
and dtypes (src/core/generation_phases.py at 4490bd1), so a shot that fits one window gives
numz's one-batch output, bit for bit, in numz's padding (milestone 1); in seedvr2x's (DESIGN.md,
Pipeline step 0), numz's patched to pad alike (research/scripts/numerics_patch.py,
NUM_PAD=reflect>=8+black+16). Only the latents and their noise are ever whole in memory, unless
a pass is tiled, and in system memory: on the device, the VAE's passes hold a slice's at a time
and the DiT's a window's, whatever the shot's length, but for the noise's draw, the shot's latent
and its noise whole there for that moment (sample_windows; DESIGN.md, Memory planner).

A tiled pass can't stream (runtime/tiling.py; DESIGN.md, Pipeline steps 1 and 3): each tile runs
over the whole shot, so a tiled encode keeps the shot's transformed input, and a tiled decode
its frames until the last tile, in the shot's scratch, system memory up to its allowance, a
file beyond. The frames are still read once, as the encode's slices come, and the decode's are
handed on in the slices a streamed decode gives, once the last tile is in: what follows a decode
(finite, the crop, the correction, the writers) is the same either way.

The steps are apart, as the resumable units of DESIGN.md (Pause and resume) are: the encode
(encode_shot), each window (sample_windows), the decode (merge_windows, decode_shot). Each takes
what the one before gives, as the units kept it (runtime/units.py); runtime/run.py's run_job runs
them in order.

With `split` (DESIGN.md, Colour correction), each slice of the decode is corrected as it comes out,
against its own frames of the reference, built from the shot's input copy through the encoder's
transform, in float32 (reference_inputs), then written: no second pass, no buffer.
"""

import logging
import math
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from seedvr2x.media.checksums import read_checksums
from seedvr2x.media.conversion import Conversion
from seedvr2x.media.decode import Decoder, to_float32
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.runtime import colour, model, tiling
from seedvr2x.runtime.job import ENCODE_SEED_OFFSET, SHARED, output_size
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models
from seedvr2x.runtime.plan import Tiles

logger = logging.getLogger(__name__)

GIB = 1024**3
CPU = torch.device("cpu")

# A shot's input copy read back: gbrp16le to gbrp16le, which zscale passes through unchanged.
COPY_READ = Conversion("gbrp16le", "gbr", "full", "left")


@dataclass(frozen=True)
class Correction:
    """What a shot's decode with `split` needs besides its latents (DESIGN.md, Colour correction):
    its input copy, the frames its encode read, width x height as the input stores them
    (units.COPY), and their checksums (units.CHECKSUMS)."""

    copy: Path
    checksums: Path
    width: int
    height: int


class CopyError(MediaError):
    """A shot's input copy that its decode can't read whole and intact: missing, cut short,
    failing FFV1's slice CRCs, a frame failing its checksum, or its checksums missing or damaged.
    It is derived data: the run stops, and the next one makes it again from the input, keeping the
    shot's latent and windows (DESIGN.md, Colour correction)."""

    def __init__(self, copy: Path, error: MediaError) -> None:
        super().__init__(f"{copy}: the shot's input copy, not readable whole and intact: {error}")
        self.copy = copy


class NonFinite(RuntimeError):
    """NaN or inf in what a unit made: the unit isn't recorded, and the run stops there, naming it,
    in either colour correction mode (DESIGN.md, Pause and resume). A resume makes the unit again
    once the cause is fixed."""


def finite(tensor: Tensor, what: str) -> None:
    """Raise NonFinite, naming `what`, unless every value of tensor is finite. The device is
    polled first (model.synchronize), so that a Ctrl-C is answered while it works."""
    model.synchronize(tensor.device)
    if not bool(torch.isfinite(tensor).all()):
        bad = int((~torch.isfinite(tensor)).sum())
        raise NonFinite(f"{what}: {bad} of its {tensor.numel()} values NaN or inf")


def moved(tensor: Tensor, device: torch.device, what: str) -> Tensor:
    """tensor on device with its values and its memory layout, its strides: itself when it is
    there already, else a copy. The noise drawn from a latent follows the latent's layout
    (sample_windows), and the decode is fed the windows as the DiT gave them, so neither changes
    layout on its way to system memory or back (runtime/units.py). torch keeps the strides of a
    tensor that is dense, as an encode's latent and a window's output are, and lays the copy of
    any other out anew, dense in the same order of dimensions (a slice of the shot's latent
    along time): RuntimeError then, naming `what`.

    The tensor's device is polled first (model.synchronize), so that a Ctrl-C is answered while
    it finishes what the copy waits for."""
    model.synchronize(tensor.device)
    copy = tensor.to(device)
    if copy.stride() != tensor.stride():
        raise RuntimeError(
            f"{what}: its copy on {device} has strides {tuple(copy.stride())}, not its"
            f" {tuple(tensor.stride())}"
        )
    return copy


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


def padded_length(count: int) -> int:
    """The frames of a shot of `count` frames once padded to 4n + 1: the VAE packs 4 frames per
    latent after the first."""
    return count + (-count + 1) % 4


def shot_layout(count: int, window: int | None) -> list[tuple[int, int]]:
    """The DiT windows of a shot of `count` frames (window_layout), capped at `window` latents;
    None runs the shot in one window."""
    latents = (padded_length(count) - 1) // 4 + 1
    return window_layout(latents, window or latents)


def pad_4n1(frames: Tensor) -> Tensor:
    """frames (T, ...) padded at the end to 4n + 1, as numz pads a batch
    (src/core/generation_utils.py, pad_video_temporal): the frames before the last, mirrored,
    and the last one repeated beyond a full mirror."""
    return torch.cat([frames, _padding(frames[-4:], frames.shape[0])])


def _padding(tail: Tensor, count: int) -> Tensor:
    """The frames pad_4n1 appends to a shot of `count` frames, from tail, its last min(count, 4)
    frames: at most 3 are missing, so the mirror never reaches further back."""
    missing = padded_length(count) - count
    if missing == 0:
        return tail[:0]
    if missing >= count:
        repeated = tail[-1:].repeat(missing - count + 1, *([1] * (tail.dim() - 1)))
        return torch.cat([tail[1:].flip(0), repeated])
    return tail[-missing - 1 : -1].flip(0)


def input_slices(
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    sizes: list[int],
    to_tensor: Callable[[npt.NDArray[np.float32]], Tensor] = model.to_input,
) -> Iterator[Tensor]:
    """The `count` frames of a shot, read as they are needed, padded at the end as pad_4n1 pads
    them, in slices of `sizes` frames (model.encode_slices): (t, H, W, 3) on the CPU, as to_tensor
    makes them, by default float16, numz's input (model.to_input). Only the last 4 frames read are
    kept, for the padding."""
    remaining, given = count, 0
    tail: Tensor | None = None
    padding: Tensor | None = None
    for size in sizes:
        parts: list[Tensor] = []
        real = min(size, remaining)
        if real:
            frames = read(real)
            if frames.shape[0] != real:
                raise RuntimeError(f"{frames.shape[0]} frames read, {real} asked for")
            part = to_tensor(frames)
            tail = part[-4:] if tail is None else torch.cat([tail, part])[-4:]
            parts.append(part)
            remaining -= real
        if real < size:
            if padding is None:
                assert tail is not None
                padding = _padding(tail, count)
            parts.append(padding[given : given + size - real])
            given += size - real
        yield parts[0] if len(parts) == 1 else torch.cat(parts)
    if remaining or (padding is not None and given != padding.shape[0]):
        raise ValueError(f"slices {sizes} for a shot of {count} frames")


@torch.no_grad()
def encode_shot(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
    *,
    numz_padding: bool = False,
    tiles: Tiles | None = None,
    scratch: Path | None = None,
    allowance: int | None = None,
) -> Tensor:
    """The VAE latent of a shot of `count` frames, read as they are needed: (T', h, w, 16)
    bfloat16 in system memory, in the channel-major memory the encode gives it, which the noise
    drawn from it depends on (model.encode). T' is 1 + (padded_length(count) - 1) / 4, and (h, w)
    the padded frames' size (job.padded_size) over 8. read(n) gives the shot's next n frames,
    (n, H, W, 3) float32 in [0, 1] at the source's size; target is the (height, width) the encode
    resizes them to (job.target_size), through model.input_transform.

    numz_padding is for tests only (cli.NUMZ_PADDING): the frames padded as numz pads them
    (model.input_transform), so that milestone 1's regression holds the output to numz's.

    tiles, when the plan tiles the encode (None: one pass, streamed): the shot is then encoded
    tile by tile (tiling.encode_tiled), numz's tiled encode of it, bit for bit, the frames still
    read once, their transformed slices waiting in the shot's scratch, each tile's crop taken from
    there. scratch is the directory the scratch's file goes to when it outgrows its allowance of
    system memory (allowance, in bytes, the plan's; None: tiling.DEFAULT_ALLOWANCE; DESIGN.md,
    Pipeline, per shot, steps 1 and 3), made only then, the file gone when this returns or fails;
    None with nowhere to put one, a larger scratch then refused (tiling.ScratchError)."""
    padded = padded_length(count)
    # Encode (generation_phases.py:329-504). numz seeds here; nothing draws.
    model.set_seed(seed + ENCODE_SEED_OFFSET)
    inputs = encoder_inputs(models, read, count, target, numz_padding=numz_padding)
    if tiles is None:
        return model.encode_stream(models, inputs, padded)
    store = tiling.Scratch(scratch, allowance)
    return tiling.encode_tiled(models, inputs, padded, tiles, store)


def encoder_inputs(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    *,
    numz_padding: bool = False,
) -> Iterator[Tensor]:
    """The encoder's input for a shot of `count` frames, read as they are needed (encode_shot),
    slice by slice: each (C, t, H, W) in [-1, 1], COMPUTE_DTYPE on the device, t following
    model.encode_slices over the shot padded to 4n + 1; numz_padding as encode_shot takes it."""
    return _transformed(
        models, read, count, target, model.to_input, COMPUTE_DTYPE, numz_padding=numz_padding
    )


def reference_inputs(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
) -> Iterator[Tensor]:
    """split's reference for a shot of `count` frames, read from its input copy as they are
    needed (_correct): the frames through the encoder's transform (model.input_transform) in
    float32, slice by slice as encoder_inputs gives the encoder's input: each (C, t, H, W) float32
    in [-1, 1] on the device (DESIGN.md, Colour correction, Numerics). Padded by default whatever
    the encoder's padding (numz_padding): split takes each frame's picture alone, cropped as the
    decode's frames are (_frames, _decoded), which no padding moves.

    Not the encoder's own input: numz's numerics cast the frames to bfloat16 before the resize
    (generation_phases.py:380-413), which rounds 89% of 8-bit codes up, +0.114 level on average,
    and brightened lab's output by its clip's bias (milestone 5's first run). numz's own lab
    transforms the float16 frames without that cast (:127-168), ByteDance the frames in float32
    (projects/inference_seedvr2_7b.py:255-268 at e4de8c2)."""
    return _transformed(models, read, count, target, torch.as_tensor, torch.float32)


def _transformed(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    to_tensor: Callable[[npt.NDArray[np.float32]], Tensor],
    dtype: torch.dtype,
    *,
    numz_padding: bool = False,
) -> Iterator[Tensor]:
    """The frames of a shot of `count` frames, read as they are needed, made tensors by
    to_tensor (input_slices) and through model.input_transform(target) as dtype on the device,
    slice by slice: each (C, t, H, W) in [-1, 1], t following model.encode_slices over the shot
    padded to 4n + 1."""
    transform = model.input_transform(target, numz_padding=numz_padding)
    # (t, 3, H, W): a view, moved with its layout (generation_phases.py:92-104, 380-388), then
    # prepared slice by slice: every step works frame by frame.
    sizes = model.encode_slices(models, padded_length(count))
    return (
        transform(frames.permute(0, 3, 1, 2).to(models.device, dtype))
        for frames in input_slices(read, count, sizes, to_tensor)
    )


def sample_windows(
    models: Models, latent: Tensor, layout: Sequence[tuple[int, int]], seed: int, start: int = 0
) -> Iterator[Tensor]:
    """The DiT's output for each window of layout from window `start` on, in order, each (t, h, w,
    16) on the device: one Euler step per window, from the shot's latent, (T', h, w, 16) in the
    memory layout its encode gave it (encode_shot): in system memory, as its encode gives it and
    the units give it back (units.Units.save_latent, units.Units.latent), or on the device.

    The noise is drawn once for the shot, when the first window is asked for, on the device and
    on the latent's layout, as numz draws it for a batch (generation_phases.py:663-680): its
    values follow the device, the dtype, the shape and the strides (_drawn). Nothing else draws,
    so a window's noise is the same whether the windows before it ran in this process or in an
    earlier one: a resumed shot draws it again, from a latent of the same layout.

    The noise and the latent then wait on the CPU, and each window's slices of them go to the
    device for its pass alone: the device holds the shot whole at the draw only, and beside a
    window nothing of the shot but that window (DESIGN.md, Memory planner, Per-phase peaks). The
    latent in system memory is the one given when it is given there, no copy being made; and no
    output is kept here once it is given. So torch.no_grad is on the steps that compute
    (_drawn, _window), not on this generator: decorating a generator, it keeps the value last
    yielded bound while the next one is computed (torch/utils/_contextlib.py, _wrap_generator),
    a window's output on the device through the next window's pass.

    A slice sent to the device is laid out there on its own, dense, without the strides a
    slice of the shot taken on the device would have; a one-window shot's, the whole shot, has
    the shot's. That shapes no bit: the vendored code reads both slices by value alone before
    anything is computed, the noise copied into a new (L, 16) tensor (na.flatten,
    vendor/models/dit_3b/na.py:102, called at vendor/core/infer.py:356) and the latent into the
    condition's, new too (vendor/core/infer.py:53 and 72, or 57 for a window of one latent),
    and draws nothing."""
    if layout[-1][1] != latent.shape[0]:
        raise ValueError(f"windows {layout} over a latent of {latent.shape[0]}")
    noise, latent = _drawn(models, latent, seed)
    for s, e in layout[start:]:
        yield _window(models, noise[s:e], latent[s:e])


@torch.no_grad()
def _drawn(models: Models, latent: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    """A shot's noise, drawn from seed for its whole latent, and that latent: both on the CPU,
    (T', h, w, 16) in the latent's memory layout (moved), the noise in COMPUTE_DTYPE; the
    latent itself when it is given there.

    The draw is torch.randn_like of the latent on the device: the generator fills the tensor in
    memory order, so that the values follow the latent's strides as well as its shape, and the
    device's generator isn't the CPU's. A latent given on the CPU is sent to the device for it,
    strides kept (moved). Both copies on the device are given up when this returns.

    With -v, the draw is logged, on a GPU: the device then holds the latent and the noise whole
    beside the weights of the DiT's phase, the moment the planner bounds beside a window's peak
    (planner.dit_draw). The peak read is the unit's so far, which the run reset when it began
    (run._started)."""
    device = models.device
    there = moved(latent, device, "the shot's latent")
    model.set_seed(seed)
    noise = torch.randn_like(there, dtype=COMPUTE_DTYPE)
    if device.type == "cuda":
        logger.debug(
            "noise drawn for the shot's %d latents: %.3f GiB allocated on the GPU, its latent and"
            " its noise there; the unit's peak so far %.3f GiB",
            latent.shape[0],
            torch.cuda.memory_allocated(device) / GIB,
            torch.cuda.max_memory_allocated(device) / GIB,
        )
    return moved(noise, CPU, "the shot's noise"), moved(latent, CPU, "the shot's latent")


@torch.no_grad()
def _window(models: Models, noise: Tensor, latent: Tensor) -> Tensor:
    """The DiT's output for one window, (t, h, w, 16) on the device, from the window's slices of
    its shot's noise and latent, on the CPU: sent to the device for the pass, where nothing of
    them is left when this returns but the output."""
    noise, latent = noise.to(models.device), latent.to(models.device)
    return model.sample(models, noise, model.condition(models, noise, latent))


def merge_windows(sampled: list[Tensor], layout: Sequence[tuple[int, int]]) -> Tensor:
    """The shot's latents from its windows' DiT outputs (sample_windows): a single window's as it
    is, else mixed where two windows share latents (_merge)."""
    return sampled[0] if len(sampled) == 1 else _merge(sampled, layout)


@torch.no_grad()
def decode_shot(
    models: Models,
    merged: Tensor,
    count: int,
    target: tuple[int, int],
    write: Callable[[npt.NDArray[np.float32]], None],
    name: str = "the shot",
    first: int = 0,
    correction: Correction | None = None,
    *,
    tiles: Tiles | None = None,
    scratch: Path | None = None,
    allowance: int | None = None,
) -> None:
    """Decode the shot of `count` frames whose latents are merged (merge_windows), in system
    memory or on the device, in one stream, and write its frames as they come, write(frames)
    taking the output's next frames,
    (n, H', W', 3) float32 in [0, 1], the target cropped to even sides (job.output_size); each
    slice corrected first when a correction is given (_correct; DESIGN.md, Colour correction).
    Each slice decoded is checked for NaN or inf first (finite), its frames named for the job's,
    the shot's first being `first`.

    tiles, when the plan tiles the decode (None: one stream): the shot is then decoded tile by
    tile (tiling.decode_tiled), numz's tiled decode of it, bit for bit. Nothing is written before
    the last tile is decoded: the shot's frames wait in its scratch until then, and are handed on
    in the slices a streamed decode gives, each checked, corrected and written as one of those.
    scratch is the directory the scratch's file goes to when it outgrows its allowance of system
    memory (allowance, in bytes, the plan's; None: tiling.DEFAULT_ALLOWANCE; DESIGN.md, Pipeline,
    per shot, step 3), made only then, the file gone when this returns or fails; None with
    nowhere to put one, a larger scratch then refused (tiling.ScratchError)."""
    with _slices(models, merged, tiles, scratch, allowance) as slices:
        chunks = _decoded(slices, count, output_size(target), name, first)
        if correction is not None:
            _correct(models, chunks, count, target, write, correction)
            return
        # Post-process (generation_phases.py:1340-1348): [-1, 1] to [0, 1] in place.
        for chunk in chunks:
            chunk.clamp_(-1, 1).mul_(0.5).add_(0.5)
            model.synchronize(models.device)
            write(chunk.to("cpu", torch.float32).numpy())


@contextmanager
def _slices(
    models: Models, merged: Tensor, tiles: Tiles | None, scratch: Path | None, allowance: int | None
) -> Generator[Iterator[Tensor]]:
    """The VAE's decode of the shot's latents, merged, slice by slice: each (C, t, H', W') in
    [-1, 1], unclamped, on the device; streamed (model.decode_stream), or with tiles, tile by tile
    (tiling.decode_tiled), its scratch's file in the directory scratch when it takes one. A tiled
    decode holds its scratch until its last slice is taken, and either decode the VAE's causal
    memory, on the device: the decode is closed on leaving the block, whatever ends it, a writer
    that fails or a stop, so that the scratch goes and the memory is cleared at once, not with
    the failure's traceback.

    The latents are decoded from system memory, a slice's alone on the device at a time
    (model.decode_sliced): latents given on the device are copied to system memory first, in
    their own layout, as units' files keep them; the device's stay their holder's to free."""
    model.synchronize(merged.device)
    latent = merged.to("cpu")
    if tiles is None:
        slices = model.decode_stream(models, latent)
    else:
        store = tiling.Scratch(scratch, allowance)
        slices = tiling.decode_tiled(models, latent, tiles, store)
    try:
        yield slices
    finally:
        # A generator, as both decodes are; a plain iterator in their place has nothing to close.
        if isinstance(slices, Generator):
            slices.close()


def _decoded(
    slices: Iterator[Tensor], count: int, size: tuple[int, int], name: str, first: int
) -> Iterator[Tensor]:
    """The shot's frames as the VAE decodes them (generation_phases.py:900-958), slice by slice
    (slices, _slices): each (t, H, W, C), a view of the decode's (C, t, H', W') cropped to size
    (height, width) from its top left, which drops the padding, whatever its amount, at the bottom
    and right (job.padding), in [-1, 1] unclamped, on the device; each checked (finite) before it
    goes."""
    height, width = size
    written = 0
    for decoded in slices:
        chunk = decoded.permute(1, 2, 3, 0)[: count - written, :height, :width]
        if chunk.shape[0] == 0:
            continue
        last = first + written + chunk.shape[0] - 1
        finite(chunk, f"{name}'s decode, frames {first + written} to {last}")
        yield chunk
        written += chunk.shape[0]
    if written != count:
        raise RuntimeError(f"{written} frames decoded for a shot of {count}")


def _correct(
    models: Models,
    chunks: Iterator[Tensor],
    count: int,
    target: tuple[int, int],
    write: Callable[[npt.NDArray[np.float32]], None],
    correction: Correction,
) -> None:
    """split on each slice of the shot's decode as it comes out (chunks, _decoded), against the
    slice's own frames of the reference, rebuilt from the input copy as the decode goes
    (reference_inputs), then written, float32 (DESIGN.md, Colour correction): one pass, no
    buffer, on the models' device. The colour's stages follow the upscale factor, from the frames
    as the input stores them to target (colour.colour_stages).
    The shot's last frames are written once the copy is read whole and checked, since they may
    finish the output segment, which is then recorded."""
    height, width = output_size(target)
    stages = colour.colour_stages((correction.height, correction.width), target)
    last: npt.NDArray[np.float32] | None = None
    with _copy_reader(correction, count) as read:
        reference = _frames(reference_inputs(models, read, count, target), height, width)
        done = 0
        for chunk in chunks:
            content = chunk.permute(0, 3, 1, 2)  # (t, C, H, W)
            corrected = colour.split(content, reference(content.shape[0]), stages)
            model.synchronize(models.device)
            frames = corrected.permute(0, 2, 3, 1).to("cpu").numpy()
            done += frames.shape[0]
            if done < count:
                write(frames)
            else:
                last = frames
    if last is not None:
        write(last)


@contextmanager
def _copy_reader(
    correction: Correction, count: int
) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
    """read(n), the next n frames of the shot's input copy, (n, H, W, 3) float32 in [0, 1] as the
    encode read them, each checked against its checksum before it is given (Decoder), `count` of
    them. The copy is read strictly too, so that what ffmpeg reports names the cause: the copy
    missing, a slice failing its CRC, the file cut short. A copy that fails either check, or whose
    checksums are missing or damaged, raises CopyError."""
    copy = correction.copy
    with _copy_failing(copy):
        checksums = read_checksums(correction.checksums)
        if len(checksums) != count:
            raise MediaError(
                f"{correction.checksums}: {len(checksums)} checksums, for {count} frames"
            )
    args = (correction.width, correction.height, count)
    decoder = Decoder(input_args(copy), COPY_READ, *args, strict=True, checksums=checksums)

    def read(n: int) -> npt.NDArray[np.float32]:
        with _copy_failing(copy):
            frames = decoder.read(n)
            if frames.shape[0] != n:
                raise decoder.failure(f"it ended after {decoder.decoded} of its {count} frames")
        return to_float32(frames)

    try:
        yield read
    except BaseException:
        decoder.stop()
        raise
    with _copy_failing(copy):
        decoder.finish()


@contextmanager
def _copy_failing(copy: Path) -> Generator[None]:
    """A failure to read the copy, raised as CopyError; what the reader's user raises isn't."""
    try:
        yield
    except MediaError as error:
        raise CopyError(copy, error) from error


def _frames(slices: Iterator[Tensor], height: int, width: int) -> Callable[[int], Tensor]:
    """take(n), the next n frames of slices (reference_inputs), each slice (C, t, H', W'): (n, C,
    height, width), each frame's top left, without its padding, as the decode's frames are
    cropped (_decoded)."""
    pending: list[Tensor] = []

    def take(n: int) -> Tensor:
        while sum(part.shape[0] for part in pending) < n:
            part = next(slices, None)
            if part is None:
                raise RuntimeError("the reference's frames ran out before the decode's")
            pending.append(part.permute(1, 0, 2, 3)[:, :, :height, :width])
        frames = pending[0] if len(pending) == 1 else torch.cat(pending)
        pending[:] = [frames[n:]] if frames.shape[0] > n else []
        return frames[:n]

    return take


def _merge(sampled: list[Tensor], layout: Sequence[tuple[int, int]]) -> Tensor:
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
