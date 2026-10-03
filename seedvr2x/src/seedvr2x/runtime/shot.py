"""Upscaling of one shot (DESIGN.md, Pipeline, per shot), streamed end to end.

The shot's frames are read as the VAE encodes them, in its slices of 4 frames, and the shot is
encoded once. The DiT runs on windows of at most `window` latents, one window per call,
consecutive windows sharing SHARED latents; the shared latents are mixed with cosine weights, and
the whole shot is decoded in one stream, its frames written as they come. The noise is drawn once
for the shot and sliced per window. Every step keeps numz's order and dtypes
(src/core/generation_phases.py at 4490bd1), so a shot that fits one window gives numz's one-batch
output, bit for bit (milestone 1). Only the latents are ever whole in memory.

The steps are apart, as the resumable units of DESIGN.md (Pause and resume) are: the encode
(encode_shot), each window (sample_windows), the decode (merge_windows, decode_shot). Each takes
what the one before gives, wherever it was kept; upscale_shot runs them in a row.

With `lab` (DESIGN.md, Colour correction), the decode makes two passes over a buffer of its
frames, against the reference rebuilt from the shot's input copy by the encode's own calls.
"""

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
from seedvr2x.runtime import colour, model
from seedvr2x.runtime.job import ENCODE_SEED_OFFSET, SHARED, output_size
from seedvr2x.runtime.model import COMPUTE_DTYPE, Models

# A shot's input copy read back: gbrp16le to gbrp16le, which zscale passes through unchanged.
COPY_READ = Conversion("gbrp16le", "gbr", "full", "left")


@dataclass(frozen=True)
class Lab:
    """What a shot's decode with `lab` needs besides its latents (DESIGN.md, Colour correction):
    its input copy, the frames its encode read, width x height (units.COPY), their checksums
    (units.CHECKSUMS), and where to buffer its decoded frames between the two passes
    (units.BUFFER)."""

    copy: Path
    checksums: Path
    buffer: Path
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
    read: Callable[[int], npt.NDArray[np.float32]], count: int, sizes: list[int]
) -> Iterator[Tensor]:
    """The `count` frames of a shot, read as they are needed, padded at the end as pad_4n1 pads
    them, in slices of `sizes` frames (model.encode_slices): (t, H, W, 3) float16 on the CPU,
    numz's input (model.to_input). Only the last 4 frames read are kept, for the padding."""
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
            part = model.to_input(frames)
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


def upscale_shot(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
    write: Callable[[npt.NDArray[np.float32]], None],
    window: int | None = None,
    *,
    reseed_windows: bool = False,
) -> None:
    """Upscale one shot of `count` frames, read as they are needed, and write its frames as they
    come out: its steps in a row.

    read(n) gives the shot's next n frames, (n, H, W, 3) float32 in [0, 1]. target is the
    (height, width) they are resized to (job.target_size). write(frames) takes the output's next
    frames, (n, H', W', 3) float32 in [0, 1], the target cropped to even sides
    (job.output_size). window caps the DiT windows, in latents (4 frames each after the first);
    None runs the shot in one window.

    reseed_windows is for tests only. Each window then reseeds and draws its own noise, as the
    stitching study's reference implementation did (blend_patch.py, STITCH_LATENT), so that the
    windows, the mixing and the decode can be checked against it bit for bit.
    """
    latent = encode_shot(models, read, count, target, seed)
    layout = shot_layout(count, window)
    if reseed_windows:
        sampled = _sample_reseeded(models, latent, layout, seed)
    else:
        sampled = list(sample_windows(models, latent, layout, seed))
    del latent
    merged = merge_windows(sampled, layout)
    del sampled
    decode_shot(models, merged, count, target, write)


@torch.no_grad()
def encode_shot(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
) -> Tensor:
    """The VAE latent of a shot of `count` frames, read as they are needed (upscale_shot): (T', h,
    w, 16) bfloat16 on the device, in the channel-major memory the encode gives it, which the
    noise drawn from it depends on (model.encode). T' is 1 + (padded_length(count) - 1) / 4."""
    padded = padded_length(count)
    # Encode (generation_phases.py:329-504). numz seeds here; nothing draws.
    model.set_seed(seed + ENCODE_SEED_OFFSET)
    return model.encode_stream(models, encoder_inputs(models, read, count, target), padded)


def encoder_inputs(
    models: Models,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
) -> Iterator[Tensor]:
    """The encoder's input for a shot of `count` frames, read as they are needed (encode_shot),
    slice by slice: each (C, t, H, W) in [-1, 1], COMPUTE_DTYPE on the device, t following
    model.encode_slices over the shot padded to 4n + 1. lab's reference is rebuilt from the
    shot's input copy by the same calls, so it is the tensor the encoder took, bit for bit."""
    transform = model.input_transform(target)
    # (t, 3, H, W): a view, moved with its layout (generation_phases.py:92-104, 380-388), then
    # prepared slice by slice: every step works frame by frame.
    return (
        transform(frames.permute(0, 3, 1, 2).to(models.device, COMPUTE_DTYPE))
        for frames in input_slices(read, count, model.encode_slices(models, padded_length(count)))
    )


@torch.no_grad()
def sample_windows(
    models: Models, latent: Tensor, layout: Sequence[tuple[int, int]], seed: int, start: int = 0
) -> Iterator[Tensor]:
    """The DiT's output for each window of layout from window `start` on, in order, each (t, h, w,
    16) on the device: one Euler step per window, from the shot's latent (encode_shot).

    The noise is drawn once for the shot, on the latent's layout as numz draws it for a batch
    (generation_phases.py:663-680), and sliced per window. Nothing else draws, so a window's
    noise is the same whether the windows before it ran in this process or in an earlier one: a
    resumed shot draws it again, from a latent of the same layout."""
    if layout[-1][1] != latent.shape[0]:
        raise ValueError(f"windows {layout} over a latent of {latent.shape[0]}")
    model.set_seed(seed)
    noise = torch.randn_like(latent, dtype=COMPUTE_DTYPE)
    for s, e in layout[start:]:
        yield model.sample(models, noise[s:e], model.condition(models, noise[s:e], latent[s:e]))


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
    lab: Lab | None = None,
) -> None:
    """Decode the shot of `count` frames whose latents are merged (merge_windows), in one stream,
    and write its frames (upscale_shot): as they come, or with lab corrected once the whole shot
    is decoded (DESIGN.md, Colour correction). Each slice decoded is checked for NaN or inf first
    (finite), its frames named for the job's, the shot's first being `first`."""
    chunks = _decoded(models, merged, count, output_size(target), name, first)
    if lab is not None:
        _correct(models, chunks, count, target, write, lab)
        return
    # Post-process (generation_phases.py:1340-1348): [-1, 1] to [0, 1] in place.
    for chunk in chunks:
        chunk.clamp_(-1, 1).mul_(0.5).add_(0.5)
        model.synchronize(models.device)
        write(chunk.to("cpu", torch.float32).numpy())


def _decoded(
    models: Models,
    merged: Tensor,
    count: int,
    size: tuple[int, int],
    name: str,
    first: int,
) -> Iterator[Tensor]:
    """The shot's frames as the VAE decodes them (generation_phases.py:900-958), slice by slice:
    each (t, H, W, C), a view of the decode's (C, t, H, W) without the padding, cropped to size
    (height, width), in [-1, 1] unclamped, on the device; each checked (finite) before it goes."""
    height, width = size
    written = 0
    for decoded in model.decode_stream(models, merged.to(models.device)):
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
    lab: Lab,
) -> None:
    """lab over a shot (DESIGN.md, Colour correction), pooled: the first pass buffers the decoded
    frames (chunks, _decoded) and counts the histograms, the second reads them back, maps them and
    writes them, float32. Each pass rebuilds the reference from the input copy, and converts in
    the same calls, the decode's slices, so that the second finds the values the first counted.
    The second pass writes its last frames once the copy is read whole and checked, since they
    may finish the output segment, which is then recorded. The buffer goes at the end, whatever
    happens."""
    height, width = output_size(target)
    histograms = colour.Histograms(models.device)
    sizes: list[int] = []
    dtype = COMPUTE_DTYPE  # the decode's, kept as it is: bfloat16 from the VAE
    try:
        with _copy_reader(lab, count) as read, open(lab.buffer, "wb") as buffer:
            reference = _frames(encoder_inputs(models, read, count, target), height, width)
            for chunk in chunks:
                content = chunk.permute(0, 3, 1, 2)  # (t, C, H, W)
                model.synchronize(models.device)
                kept = content.to("cpu").contiguous()
                try:
                    buffer.write(memoryview(kept.view(torch.uint8).numpy()))
                except OSError as error:
                    need = count * kept[0].nbytes
                    raise MediaError(
                        f"{lab.buffer}: {error.strerror}; the shot's decoded frames take"
                        f" {need / 2**30:.1f} GiB there"
                    ) from error
                dtype = kept.dtype
                matched = reference(content.shape[0])
                corrected = colour.rgb_to_lab(colour.unit_range(colour.transfer(content, matched)))
                histograms.add(corrected, colour.rgb_to_lab(colour.unit_range(matched)))
                sizes.append(content.shape[0])
        last: npt.NDArray[np.float32] | None = None
        with _copy_reader(lab, count) as read, open(lab.buffer, "rb") as buffer:
            reference = _frames(encoder_inputs(models, read, count, target), height, width)
            for size in sizes:
                content = torch.empty((size, 3, height, width), dtype=dtype)
                if buffer.readinto(content.view(torch.uint8).numpy()) != content.nbytes:
                    raise RuntimeError(f"{lab.buffer}: shorter than the frames buffered")
                content = content.to(models.device)
                matched = reference(size)
                corrected = colour.rgb_to_lab(colour.unit_range(colour.transfer(content, matched)))
                rgb = colour.lab_to_rgb(histograms.match(corrected))
                model.synchronize(models.device)
                if last is not None:
                    write(last)
                last = rgb.permute(0, 2, 3, 1).to("cpu").numpy()
        if last is not None:
            write(last)
    finally:
        lab.buffer.unlink(missing_ok=True)


@contextmanager
def _copy_reader(lab: Lab, count: int) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
    """read(n), the next n frames of the shot's input copy, (n, H, W, 3) float32 in [0, 1] as the
    encode read them, each checked against its checksum before it is given (Decoder), `count` of
    them. The copy is read strictly too, so that what ffmpeg reports names the cause: the copy
    missing, a slice failing its CRC, the file cut short. A copy that fails either check, or whose
    checksums are missing or damaged, raises CopyError."""
    with _copy_failing(lab.copy):
        checksums = read_checksums(lab.checksums)
        if len(checksums) != count:
            raise MediaError(f"{lab.checksums}: {len(checksums)} checksums, for {count} frames")
    args = (lab.width, lab.height, count)
    decoder = Decoder(input_args(lab.copy), COPY_READ, *args, strict=True, checksums=checksums)

    def read(n: int) -> npt.NDArray[np.float32]:
        with _copy_failing(lab.copy):
            frames = decoder.read(n)
            if frames.shape[0] != n:
                raise decoder.failure(f"it ended after {decoder.decoded} of its {count} frames")
        return to_float32(frames)

    try:
        yield read
    except BaseException:
        decoder.stop()
        raise
    with _copy_failing(lab.copy):
        decoder.finish()


@contextmanager
def _copy_failing(copy: Path) -> Generator[None]:
    """A failure to read the copy, raised as CopyError; what the reader's user raises isn't."""
    try:
        yield
    except MediaError as error:
        raise CopyError(copy, error) from error


def _frames(slices: Iterator[Tensor], height: int, width: int) -> Callable[[int], Tensor]:
    """take(n), the next n frames of slices (encoder_inputs), each slice (C, t, H', W'): (n, C,
    height, width), each frame's top left, as the decode's frames are cropped (_decoded)."""
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


@torch.no_grad()
def _sample_reseeded(
    models: Models, latent: Tensor, layout: Sequence[tuple[int, int]], seed: int
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
