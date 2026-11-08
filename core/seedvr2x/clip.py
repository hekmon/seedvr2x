"""One clip through the models (DESIGN.md, Repository and apps, the ComfyUI nodes: "The shot's pass
moves to the library"): the frames of one shot in memory, upscaled and handed back in memory,
the video's counterpart of still.py. The library's brick: it takes models already loaded and
never loads, unloads or reconfigures them, beyond entering the three phases as the command line
does (Models.enter), so that a node can call it in a loop on the models it keeps.

A clip is a shot, Shot(0, T), seeded as such (shots.Shot.seed), and the pass is the shot's own
(seedvr2x/shot_pass.py): encode_shot, sample_windows over the layout of shots.shot_layout,
merge_windows, decode_shot with the colour correction read from the frames in memory. So the
output is bit for bit what the command line gives the same frames as a shot of one run, and, at
`quality` enhanced or maximum, as a shot of 2 or 4 runs (shotruns; apps/seedvr2x/tests/
test_clip_cli.py holds both): the runs are made one after the other, each fed the shot from its
first frame, whole, and each writes into the one output array only the frames shotruns.run_of
takes from it. A clip cut into shots is one call a shot, `start` giving the shot's place in the
clip (its seed, as the command line seeds it) and `output` the clip's array to write into. What
differs is where things wait, all in system memory: each window's output goes to the CPU as soon
as it is made, as the command line writes it to a file (units.Units.save_window), and is merged
there; a tiled pass keeps its scratch in tiling.InMemory, whole, at 12.7 MB a 1080p frame; the
output is one array, allocated once, nothing kept between a shot's runs but that array, and the
host holds one run's latent and windows at a time, the first run's, the longest. The plan counts
them (plan), at every level."""

import traceback
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x import model, peaks, sampling, tiling
from seedvr2x.errors import HostShort
from seedvr2x.frames import output_size
from seedvr2x.memory import Reading
from seedvr2x.model import Models
from seedvr2x.passes import Plan, Tiles
from seedvr2x.shot_pass import (
    NonFinite,
    decode_shot,
    encode_shot,
    finite,
    merge_windows,
    sample_windows,
)
from seedvr2x.shot_plan import Peaks, Pins, PlanError
from seedvr2x.shot_plan import plan as shot_plan
from seedvr2x.shotruns import QUALITY, shot_runs, taken_from
from seedvr2x.shots import Shot, shot_layout
from seedvr2x.weights import Architecture

__all__ = ["NonFinite", "Stage", "Step", "Tile", "output_bytes", "plan", "upscale"]

# The steps a clip is made of, as `step` is told them: the one encode, each DiT window, and the
# decode's frames, of each run; and, with several runs, the start of each ("run").
Stage = Literal["run", "encode", "window", "decode"]

# step(stage, done, total), called between the steps of upscale: ("encode", 1, 1) after the encode;
# ("window", n, windows) after the n-th window; ("decode", frames, T) after each slice of the
# decode is written to the output, frames being those of the run decoded so far, T the run's.
# With several runs (`quality` enhanced or maximum), ("run", n, runs) is told first, before the
# n-th run's encode, and the steps after it are that run's, counted from its own first frame;
# with one run, standard's, none is told, and the steps are as they were before runs existed.
Step = Callable[[Stage, int, int], None]

# tile(pass, done, total), told by a tiled encode or decode of upscale: the tiles done of the
# pass's grid, 0 before its first (tiling.TileStep, with the pass named). Only tiled passes tell:
# an untiled one, or a grid of one tile, is told by its step alone. It may raise, as a Step.
Tile = Callable[[Literal["encode", "decode"], int, int], None]


def output_bytes(frames: int, target: tuple[int, int]) -> int:
    """The bytes of the output array of a clip of `frames` frames resized to `target`, (height,
    width) as frames.target_size gives it: (T, H', W', 3) float32, (H', W') =
    frames.output_size(target)."""
    height, width = output_size(target)
    return frames * height * width * 3 * 4


def plan(
    free: int | Reading,
    target: tuple[int, int],
    shots: int | Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    correct: bool,
    dit_model: Architecture = peaks.FP16,
    held: bool = False,
) -> tuple[Plan, Peaks]:
    """The plan for one clip in memory, resized to `target`: shot_plan.plan of its shots, with
    `free` (bytes, or the reading the run made), `pins`, `host_available` (bytes of system memory;
    None: unchecked), `correct` (the colour correction, split), the DiT's
    format and `held` (the weights' pinned copies exist already, Models.place) as it takes them.
    `shots` is the clip's cut into shots, each a pass of upscale (the longest shot's windows decide
    the plan, shot_plan.plan), or a number of frames: one shot, Shot(0, frames). PlanError as it
    raises it.

    The system memory is counted for a clip in memory (shot_plan.plan's in_memory): the input
    frames are the caller's and not counted; the output array (output_bytes) is taken off
    host_available before the plan, and a refusal for the host says so; the windows of the shot
    that holds the most are kept in system memory together until its decode, and a tiled pass's
    scratch is the whole of it, never a file, so that a host that can't hold it shortens the
    window or refuses the clip, as for any host short of memory. The plan is the same at every
    `quality` level (upscale): the runs are made one after the other, the first run is the whole
    shot and the longest, and a later one is shorter, so that no run holds more than it, in
    system memory or on the card. ValueError for no frame."""
    cut = [Shot(0, shots)] if isinstance(shots, int) else list(shots)
    if not cut or any(each.frames < 1 for each in cut):
        raise ValueError("a clip of no frame")
    frames = sum(each.frames for each in cut)
    taken = output_bytes(frames, target)
    available = host_available
    if available is not None:
        if taken >= available:
            height, width = output_size(target)
            raise HostShort(
                f"the clip's output, {frames} frames of {width}x{height} in float32, takes"
                f" {taken / peaks.GIB:.2f} GiB of system memory and {available / peaks.GIB:.2f}"
                f" GiB are available, {(taken - available) / peaks.GIB:.2f} GiB short",
                need=taken,
                available=available,
                short=taken - available,
            )
        available -= taken
    try:
        return shot_plan(
            free,
            target,
            cut,
            pins,
            available,
            split=correct,
            dit_model=dit_model,
            held=held,
            in_memory=True,
        )
    except PlanError as refused:
        if refused.phase != "host" or host_available is None:
            raise
        # The figures are of what the output leaves: said, so that the available memory the
        # message gives is no surprise beside the host's own.
        said = (
            f" (the output array's {taken / peaks.GIB:.2f} GiB are taken off the"
            f" {host_available / peaks.GIB:.2f} GiB the host has available)"
        )
        # Bound here: Python unbinds the except clause's name when the clause ends, and the
        # message is worded later, by each app.
        kept = refused
        raise kept.reworded(lambda say: kept.words(say) + said) from None


class _Frames:
    """split's reference for a clip (shot_pass.Reference): the frames themselves, in memory, as
    the encode reads them. `first` is the run's first frame in the clip, which the correction
    skips to (a later run is corrected against the same frames from there)."""

    def __init__(self, frames: npt.NDArray[np.float32], first: int = 0) -> None:
        self.first = first
        self.frames = frames
        self.height, self.width = frames.shape[1], frames.shape[2]

    @contextmanager
    def reader(self, count: int) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        yield _reader(self.frames[:count])


def _reader(frames: npt.NDArray[np.float32]) -> Callable[[int], npt.NDArray[np.float32]]:
    """read(n), the next n of `frames`, (n, H, W, 3) float32 in [0, 1]: views, nothing copied."""
    position = 0

    def read(n: int) -> npt.NDArray[np.float32]:
        nonlocal position
        part = frames[position : position + n]
        position += n
        return part

    return read


@torch.no_grad()
def upscale(
    models: Models,
    frames: npt.NDArray[np.float32],
    target: tuple[int, int],
    seed: int,
    *,
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
    step: Step | None = None,
    start: int = 0,
    output: npt.NDArray[np.float32] | None = None,
    quality: str = "standard",
    tile: Tile | None = None,
    tail_frames: bool = False,
) -> npt.NDArray[np.float32]:
    """The clip upscaled: (T, H', W', 3) float32 in [0, 1] on the CPU, (H', W') =
    frames.output_size(target), one array allocated once, or `output`.

    frames is the clip, one shot, (T, H, W, 3) float32 in [0, 1]; the array is read and left as
    it is. target is the (height, width) in pixels the encode resizes it to (frames.target_size,
    the caller's). seed is the job's: the clip is the shot starting at frame `start` of its
    source (default 0, the plain seed), whose seed it is (shots.Shot.seed, the job's plus the
    shot's first frame, as the command line seeds a shot); `start` names its frames in a
    refusal for NaN too. output, when given, is the (T, H', W', 3) float32 array the frames are
    written into and returned, a slice of a longer clip's (ValueError for another shape): a clip
    of several shots is then one array, never a copy of each. window, encode_tiles and
    decode_tiles are the plan's (plan): the cap on the DiT windows in latents (None: one window)
    and the VAE passes' tiles (None: one pass). correct runs the colour correction
    (colour.split) against the clip's own frames. quality is a level of shotruns.QUALITY
    (ValueError for another): the clip is run 1, 2 or 4 times, a run from the clip's frame 0, 0
    and 2, or 0 to 3 (shotruns.shot_runs; a clip too short to have a run's first frame has no
    such run), each a whole run of its own with its windows and its correction against the same
    input frames from its first frame; the output's frames are those each run gives
    (shotruns.run_of), written into the one array as each run decodes them. tail_frames is the
    plan's (Plan.tail_frames): each run's decode with its tail a frame at a time, on for the
    decode and off after it, whatever ends it.

    The steps one after the other, a phase entered before each (Models.enter): the encode, each
    window with its output sent to system memory as it is made, the windows merged, and the decode
    written into the output slice by slice; a run after the other. step, when given, is told
    after each (Step; and before each run, with several); it may raise anything, BaseException
    included, to stop the clip: nothing is then left on the device by this call, the decode closed
    and the frames of the failing call released, but what the models hold, and the output holds
    the frames the runs made (a stopped call's output is the caller's to drop). tile, when given,
    is told each tile of a tiled encode and decode as it is done (Tile), which `step` is not: a
    bar can move, and Cancel be honoured, between the tiles of a pass of a run; like step it only
    observes and may raise. ValueError for frames that aren't (T, H, W, 3) float32 with T >= 1;
    NonFinite when the encode, a window or the decode holds NaN or inf (shot_pass.finite)."""
    if frames.ndim != 4 or frames.shape[3] != 3 or frames.dtype != np.float32:
        raise ValueError(f"frames {frames.shape} {frames.dtype}, not (T, H, W, 3) float32")
    count = frames.shape[0]
    if count < 1:
        raise ValueError("frames of no frame")
    if quality not in QUALITY:
        raise ValueError(f"quality {quality!r}, not one of {', '.join(QUALITY)}")
    shot = Shot(start, start + count)
    clip_seed = shot.seed(seed)
    height, width = output_size(target)
    if output is None:
        output = np.empty((count, height, width, 3), dtype=np.float32)
    elif output.shape != (count, height, width, 3) or output.dtype != np.float32:
        raise ValueError(
            f"output {output.shape} {output.dtype}, not ({count}, {height}, {width}, 3) float32"
        )
    level = QUALITY[quality]
    runs = shot_runs(Shot(0, count), level)
    # The run each frame is taken from, as the run's first frame in the clip.
    taken = np.asarray(taken_from(count, level))
    try:
        with model.marking(models.device):
            for number, each in enumerate(runs, 1):
                if step is not None and len(runs) > 1:
                    step("run", number, len(runs))
                _run(
                    *(models, frames, each.offset, target, clip_seed, output),
                    *(window, encode_tiles, decode_tiles, correct, step, start),
                    taken == each.offset,
                    tile,
                    tail_frames,
                )
    except BaseException as error:
        # Whatever stopped the clip, a step's cancel included: the frames of the calls it passed
        # through hold the slice being corrected on the device, and would keep it for as long as
        # the exception is kept (decode_shot's slices are closed already, shot_pass._slices). Not
        # on a failed card: what they hold stays with the error, whose holder keeps it, since a
        # free there would report the error again (model.unless_failed).
        traceback_ = error.__traceback__
        model.unless_failed(models.device, lambda: traceback.clear_frames(traceback_))
        raise
    return output


def _run(
    models: Models,
    frames: npt.NDArray[np.float32],
    offset: int,
    target: tuple[int, int],
    seed: int,
    output: npt.NDArray[np.float32],
    window: int | None,
    encode_tiles: Tiles | None,
    decode_tiles: Tiles | None,
    correct: bool,
    step: Step | None,
    start: int,
    mine: npt.NDArray[np.bool_],
    tile: Tile | None = None,
    tail_frames: bool = False,
) -> None:
    """The run of the clip from its frame `offset`: frames [offset, T) of the clip through the
    shot's pass, with the shot's `seed` (the runs of a shot share it, shotruns.Run) and its own
    windows (shots.shot_layout of its frames), the colour correction against the same frames from
    `offset`. Of the frames the decode gives, those whose entry in `mine` (one a frame of the
    clip, True for those taken from this run) is True are written into their place in `output`,
    the others dropped: the one array is all the run keeps. `start` is the clip's first frame in
    its source, which a refusal for NaN names the frames by. `tile` is told each tile of the run's
    tiled passes (upscale)."""
    count = frames.shape[0] - offset
    layout = shot_layout(count, window)
    reference = _Frames(frames, offset) if correct else None
    models.enter("encode")
    latent = encode_shot(
        models,
        _reader(frames[offset:]),
        count,
        target,
        seed,
        tiles=encode_tiles,
        store=tiling.InMemory(),
        tile_step=None if tile is None else partial(tile, "encode"),
    )
    finite(latent, "the clip's encode")
    if step is not None:
        step("encode", 1, 1)
    # The latent is alive through the merge (the caller's and _merged's names hold it): the
    # plan counts it (shot_plan.host_moments, in_memory), L latents beside the merged one.
    first, last = start + offset, start + frames.shape[0] - 1
    merged = _merged(models, latent, layout, seed, step, (first, last))
    del latent
    models.enter("decode")
    written = 0

    def write(part: npt.NDArray[np.float32]) -> None:
        nonlocal written
        # The run's frame i is the clip's frame offset + i.
        place = offset + written
        kept = mine[place : place + part.shape[0]]
        if kept.all():
            output[place : place + part.shape[0]] = part
        elif kept.any():
            output[place : place + part.shape[0]][kept] = part[kept]
        written += part.shape[0]
        if step is not None:
            step("decode", written, count)

    decode_shot(
        models,
        merged,
        count,
        target,
        write,
        "the clip",
        first,
        reference,
        tiles=decode_tiles,
        store=tiling.InMemory(),
        tile_step=None if tile is None else partial(tile, "decode"),
        tail_frames=tail_frames,
    )


def _merged(
    models: Models,
    latent: torch.Tensor,
    layout: list[tuple[int, int]],
    seed: int,
    step: Step | None,
    frames: tuple[int, int],
) -> torch.Tensor:
    """The clip's latents from its windows: each window sampled (shot_pass.sample_windows) with
    the DiT's phase entered before it, its output checked and moved to system memory with its
    memory layout kept (sampling.moved), as units.Units.save_window keeps it, and nothing of it
    left on the device before `step` is told; then merged (shot_pass.merge_windows). `frames` are
    the first and last frame of the shot in the clip, which a refusal for NaN names, as the
    decode's does."""
    sampled: Iterator[torch.Tensor] = sample_windows(models, latent, layout, seed)
    windows: list[torch.Tensor] = []
    try:
        for number in range(len(layout)):
            models.enter("dit")
            output = next(sampled)
            what = (
                f"the clip's window {number + 1}/{len(layout)} (frames {frames[0]} to {frames[1]})"
            )
            finite(output, what)
            windows.append(sampling.moved(output, sampling.CPU, what))
            del output
            if step is not None:
                step("window", number + 1, len(layout))
    finally:
        # The generator holds the noise and the latent: closed at once, a step's stop included.
        if isinstance(sampled, Generator):
            model.unless_failed(models.device, sampled.close)
    return merge_windows(windows, layout)
