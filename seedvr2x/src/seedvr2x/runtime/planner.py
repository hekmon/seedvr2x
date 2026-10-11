"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model: what each phase is predicted to peak at on the GPU, in bytes, by the fits of
research/docs/planner-limits.md and research/docs/vram.md, measured on numz's CLI with the 7B
fp16, the one DiT seedvr2x runs (weights.ACCEPTED). Sizes are (height, width) in pixels; `padded`
is the frame the model takes, the target padded (job.padded_size).

Its choices, in DESIGN.md's order (plan): the window, the biggest quality lever; the fewest
swapped blocks that hold it; the largest VAE tiles, none where the untiled pass fits. A pin is
taken exactly or refused, the refusal naming what fits; a size that doesn't fit is refused, never
degraded (PlanError)."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

from seedvr2x.runtime.job import SHARED, JobError, Shot, check_target, output_size, padded_size
from seedvr2x.runtime.plan import Phase, Plan, Tiles

# The DiT's own window functions, the 7B's: plain Python, and the packages above them hold
# nothing, so importing them brings in no torch.
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)

GIB = 1024**3

# What the plan leaves of the free memory the driver reports, the budget being free - MARGIN
# (DESIGN.md, Memory planner, Budget): bisected on emulated cards, a run needs 0.15-0.45 GiB
# between the free memory it starts with and the torch peak of its bounding phase, the
# decode-bound 8 GB card the most (it failed with 0.37); 0.6 keeps 0.23 over that failure
# (planner-limits.md, The margin over free memory).
MARGIN = round(0.6 * GIB)

# The DiT's peak (planner-limits.md, The DiT's memory: tokens plus windows): what the loaded 7B
# fp16 holds before a forward, the VAE's weights included, as numz has them on the GPU in its DiT
# phase (the DiT 15.35 GiB, its buffers 0.02, the VAE 0.47, the text embeddings and the
# schedule), plus PER_TOKEN a token and PER_WINDOW an attention window, each of which repeats the
# 58 text tokens. It fits 26 window lengths, 1-78 latents at 1080p and 1-19 at 4K, to 0.005 GiB.
DIT_LOADED = round(15.871 * GIB)
PER_TOKEN = 127.16 * 1024
PER_WINDOW = 2.726 * 1024**2
# The attention windows along time, height and width that every block of the 7B asks its window
# function for, the blocks alternating the two layouts (vendor/configs_7b/main.yaml, window and
# window_method).
NUM_WINDOWS = (4, 3, 3)

# The VAE's weights, on the GPU in the VAE's phases, and in the DiT's unless the plan sends them
# away for it: measured on numz, 28.15 to 27.68 GiB (DESIGN.md, Memory planner, Per-phase peaks;
# vram.md, Practical rules).
VAE_WEIGHTS = round(0.47 * GIB)

# The DiT's blocks, which hold 15.19 of its 15.35 GiB (DESIGN.md, Memory planner, Host RAM), and
# the slots on the GPU its swapped blocks stream through (DESIGN.md, BlockSwap, rewritten).
BLOCKS = 36
BLOCK = 15.19 * GIB / BLOCKS
SLOTS = 2
# What of the loaded DiT isn't its blocks: its input and output layers, its buffers, the text
# embeddings and the schedule, 0.211 GiB.
DIT_REST = DIT_LOADED - VAE_WEIGHTS - round(BLOCK * BLOCKS)

# The VAE's peaks above what is on the GPU before the call, each (GiB, GiB per megapixel): of the
# padded frame untiled, flat beyond 9 frames (DESIGN.md, Memory planner, Per-phase peaks, from
# vram.md's fits less the VAE's weights); of a tile's area from 1024-px tiles, refitted at 4K up
# to 2048-px tiles (planner-limits.md, Tiled VAE at 4K: the decode at 1536 px within 0.01 GiB,
# the encode there 0.33 under the fit); and of a tile's area below 1024 px, vram.md's (Tiling,
# residuals under 0.6 GiB), which fell 1.7 GiB short at 2048 px. The overlap doesn't move them.
ENCODE_UNTILED = (1.2, 8.8)
DECODE_UNTILED = (0.8, 16.1)
ENCODE_TILED = (1.36, 8.66)
DECODE_TILED = (0.44, 16.29)
ENCODE_TILED_SMALL = (1.7, 8.4)
DECODE_TILED_SMALL = (1.6, 15.6)
REFITTED = 1024  # the tile's side the refit starts at, in pixels

# The colour correction, split, on the GPU: 0.56 GiB per 3840x2160 output frame it corrects at
# once (research/docs/colour.md, Cost and what streams), in the decode's phase, where it runs on
# each slice as the decode hands it on (shot._correct).
SPLIT_FRAME_4K = round(0.56 * GIB)
PIXELS_4K = 3840 * 2160
# The most frames a slice of the decode holds: its first is the first two latents, 5 frames, the
# next ones a latent each, 4 (model.decode_stream; the VAE's split_size of 4 frames,
# vendor/configs_7b/main.yaml).
SLICE = 5

# The planner's tiles (DESIGN.md, Memory planner, What follows from them): squares, their side a
# multiple of TILE_STEP pixels from TILE_MIN up, the smallest tile it makes.
TILE_MIN = 512
TILE_STEP = 64

# The process's resident system memory during the shots, which stays flat: 2.3-2.4 GiB over 6
# shots (DESIGN.md, Memory planner, Host RAM).
HOST_RESIDENT = round(2.4 * GIB)

# A tiled pass's scratch, per pixel of the padded frame and per frame: 3 channels in bf16, 12.7 MB
# per 1080p frame and 50.5 MB per 4K frame (DESIGN.md, Pipeline, per shot, step 3).
SCRATCH_PIXEL = 6
# What of a scratch stays in system memory, the rest a memory-mapped file. PROVISIONAL
# (implementation, 2026-10-10): DESIGN.md gives the allowance no figure; the tiled passes' own
# replaces this one.
SCRATCH_ALLOWANCE = 8 * GIB


class PlanError(JobError):
    """A job the planner refuses before the run: a pin it can't take exactly, or a size that
    doesn't fit the memory free. Its message counts in free VRAM, never in card sizes (DESIGN.md,
    Memory planner), in GiB: what is taken rounded up, what is there rounded down. PROVISIONAL
    (implementation, 2026-10-10): the messages' wording."""


@dataclass(frozen=True)
class Pins:
    """The choices the command line fixes (DESIGN.md, Memory planner, Pins), each taken exactly or
    refused, never adjusted; None leaves a choice to the planner."""

    window: int | None = None  # --window: the cap on the DiT windows, in latents
    encode_tile: int | None = None  # --encode-tile: a tile's side in output pixels; 0: untiled
    decode_tile: int | None = None  # --decode-tile, likewise
    swap_blocks: int | None = None  # --swap-blocks: the DiT's blocks swapped


def tokens(padded: tuple[int, int], latents: int) -> int:
    """The DiT's tokens for a window of `latents` latents of padded frames: a latent is an eighth
    of the frame a side, patched 1 x 2 x 2 (planner-limits.md, The DiT's attention windows):
    8,280 a latent at 1080p, padded to 1104 rows, 32,880 at 4K, to 2192."""
    height, width = padded
    return latents * (height // 16) * (width // 16)


@cache
def attention_layouts(padded: tuple[int, int], latents: int) -> tuple[int, int]:
    """The attention windows the DiT cuts a window of `latents` latents of padded frames into, by
    its own window functions: (regular, shifted), the counts of the two layouts its blocks
    alternate. They step with the window's length in time, so neither follows the tokens."""
    size = (latents, padded[0] // 16, padded[1] // 16)
    regular = make_720Pwindows_bysize(size, NUM_WINDOWS)
    shifted = make_shifted_720Pwindows_bysize(size, NUM_WINDOWS)
    return len(regular), len(shifted)


def attention_windows(padded: tuple[int, int], latents: int) -> int:
    """The attention windows the DiT's peak counts for a window of `latents` latents: the larger
    of its two layouts' counts (attention_layouts), the shifted one's."""
    return max(attention_layouts(padded, latents))


def dit_peak(
    padded: tuple[int, int], latents: int, swapped: int = 0, vae_in_dit: bool = True
) -> int:
    """The predicted peak of the DiT's phase, in bytes: one window of `latents` latents of padded
    frames, `swapped` of the DiT's blocks swapped, the VAE's weights on the GPU meanwhile unless
    vae_in_dit is False.

    PROVISIONAL (implementation, 2026-10-10): a swapped block is reckoned to take its BLOCK off
    the peak, but for the SLOTS on the GPU (DESIGN.md, BlockSwap, rewritten), so that one or two
    swapped blocks save nothing. numz's own swap, one block there at a time, saved 0.4227 GiB x
    (S - 1) for S blocks, to 0.01 GiB; seedvr2x's saving is measured with milestone 3."""
    windows = PER_WINDOW * attention_windows(padded, latents)
    return round(_dit_least(padded, latents, swapped, vae_in_dit) + windows)


def _dit_least(padded: tuple[int, int], latents: int, swapped: int, vae_in_dit: bool) -> float:
    """dit_peak less its attention windows' part, in bytes: never above the peak, and counted
    without the window functions, which lay out every window of each length they are asked."""
    if not 0 <= swapped <= BLOCKS:
        raise ValueError(f"{swapped} blocks swapped of the DiT's {BLOCKS}")
    least = DIT_LOADED + PER_TOKEN * tokens(padded, latents) - BLOCK * max(0, swapped - SLOTS)
    return least if vae_in_dit else least - VAE_WEIGHTS


def dit_on_gpu(resident: int | None, swapped: int) -> int:
    """What of the DiT is on the GPU in a VAE phase, in bytes, `swapped` of its blocks being
    swapped: `resident` of the others stay there (0: every block away; Plan.resident_encode,
    Plan.resident_decode), all of them when it is None, the DiT staying as its own phase has it.

    PROVISIONAL (implementation, 2026-10-10), the one place to change: only the blocks are
    reckoned to move, the rest of the DiT (DIT_REST) staying, and the slots to stay allocated,
    whatever leaves. BlockSwap's own figures replace it."""
    there = BLOCKS - swapped if resident is None else resident
    if not 0 <= swapped <= BLOCKS or not 0 <= there <= BLOCKS - swapped:
        raise ValueError(f"{resident} blocks resident, {swapped} swapped, of the DiT's {BLOCKS}")
    return round(DIT_REST + BLOCK * (there + min(swapped, SLOTS)))


def tile_area(padded: tuple[int, int], tiles: Tiles) -> float:
    """The area of the largest tile of a tiled pass over padded frames, in megapixels (10^6
    pixels): the square cut off at the picture's edges (DESIGN.md, Memory planner: a 1280-px
    tile at 1080p is 1104x1280)."""
    height, width = padded
    return min(tiles.size, height) * min(tiles.size, width) / 1e6


def grid(padded: tuple[int, int], tiles: Tiles) -> tuple[int, int]:
    """(rows, columns) of the tiles over padded frames: numz's grid (DESIGN.md, Pipeline, per
    shot, step 3; vendor/models/video_vae_v3/modules/attn_video_vae.py:1516-1524), tiles laid from
    the top left corner with a stride of their side less the overlap, one that would lie whole in
    the overlap of the one before left out. 2160x3840 takes 3 x 5 tiles of 1024 px overlapping by
    128, and 2 x 2 of 2048 (planner-limits.md, Tiled VAE at 4K); a side the tile covers takes
    one.

    PROVISIONAL (implementation, 2026-10-10): the tiled passes' own grid replaces this one."""
    height, width = padded
    return _along(height, tiles), _along(width, tiles)


def _along(side: int, tiles: Tiles) -> int:
    """The tiles along a side of `side` pixels (grid)."""
    starts = range(0, side, tiles.size - tiles.overlap)
    return sum(
        start == 0 or min(start + tiles.size, side) - start > tiles.overlap for start in starts
    )


def encode_peak(
    padded: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
) -> int:
    """The predicted peak of the VAE's encode of padded frames, in bytes: its fit, untiled
    (tiles None) or of its largest tile (tile_area), above what is on the GPU before the call,
    the VAE's weights and what of the DiT stays there (dit_on_gpu's resident and swapped). Not the
    shot's frames: seedvr2x feeds the encode slice by slice, where numz moves a batch's there
    first, 0.046 GiB per 4K frame (DESIGN.md, Memory planner, Per-phase peaks)."""
    fit = _fit(padded, tiles, ENCODE_UNTILED, ENCODE_TILED, ENCODE_TILED_SMALL)
    return round(fit) + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
) -> int:
    """The predicted peak of the VAE's decode of padded frames, in bytes, as encode_peak gives
    the encode's; with split, the colour correction too, which corrects each slice of the decode
    at once, SLICE frames at most of the output's size (job.output_size), tiled or not: a tiled
    decode hands its frames on slice by slice as a streamed one does (DESIGN.md, Pipeline, per
    shot, step 3)."""
    fit = _fit(padded, tiles, DECODE_UNTILED, DECODE_TILED, DECODE_TILED_SMALL)
    corrected = SPLIT_FRAME_4K * SLICE * output[0] * output[1] / PIXELS_4K if split else 0
    return round(fit + corrected) + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def _fit(
    padded: tuple[int, int],
    tiles: Tiles | None,
    untiled: tuple[float, float],
    tiled: tuple[float, float],
    small: tuple[float, float],
) -> float:
    """A VAE pass's peak above what is on the GPU before the call, in bytes, by its fits (GiB,
    GiB per megapixel). A tile that covers the whole padded frame is the untiled pass, as numz's
    tiled passes take it (attn_video_vae.py:1480)."""
    if tiles is None or covers(padded, tiles):
        constant, slope = untiled
        area = padded[0] * padded[1] / 1e6
    else:
        constant, slope = tiled if tiles.size >= REFITTED else small
        area = tile_area(padded, tiles)
    return (constant + slope * area) * GIB


def covers(padded: tuple[int, int], tiles: Tiles) -> bool:
    """Whether one tile covers the whole padded frame: no tiled pass, the untiled one."""
    return tiles.size >= max(padded)


def scratch(padded: tuple[int, int], frames: int) -> int:
    """The scratch of a tiled pass over a shot of `frames` frames of padded frames, in bytes:
    SCRATCH_PIXEL per pixel of each of its frames, the shot padded to 4n + 1 as the VAE takes it
    (shot.padded_length)."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import padded_length

    return SCRATCH_PIXEL * padded[0] * padded[1] * padded_length(frames)


# The phases, in a shot's order.
PHASES: tuple[Phase, ...] = ("encode", "dit", "decode")


@dataclass(frozen=True)
class Peaks:
    """What a plan is predicted to take, in bytes but for the last two: which phase bounds it,
    and by how much, is what --plan prints and the manifest keeps for information."""

    free: int  # the memory free on the GPU when the plan was made, as the driver reports it
    budget: int  # free - MARGIN
    encode: int  # each phase's predicted peak on the GPU
    dit: int
    decode: int
    host: int  # the system memory the plan takes
    scratch: int  # the scratch of the longest shot's tiled passes; 0 untiled
    window: int  # the longest window the job really runs, in latents
    tokens: int  # the DiT's tokens per latent

    def peak(self, phase: Phase) -> int:
        """The predicted peak of `phase`."""
        return {"encode": self.encode, "dit": self.dit, "decode": self.decode}[phase]

    @property
    def bound(self) -> Phase:
        """The phase that bounds the plan: the one with the least room under the budget."""
        return max(PHASES, key=self.peak)

    def record(self) -> dict[str, Any]:
        """What the manifest keeps of them beside where the weights are (Plan.record's memory),
        for information, never compared (resume.UNCOMPARED): bytes, the longest window in
        latents. PROVISIONAL (implementation, 2026-10-10): the fields' names."""
        return {
            "free": self.free,
            "budget": self.budget,
            "encode_peak": self.encode,
            "dit_peak": self.dit,
            "decode_peak": self.decode,
            "bound": self.bound,
            "host": self.host,
            "scratch": self.scratch,
            "longest_window": self.window,
        }


@dataclass(frozen=True)
class _Job:
    """What the planner counts a job by: the memory free, its size, and its shots' lengths."""

    free: int  # bytes
    target: tuple[int, int]  # job.target_size
    lengths: tuple[int, ...]  # its shots' lengths in latents, each once, ascending
    frames: int  # its longest shot's frames
    split: bool
    numz_padding: bool

    @property
    def budget(self) -> int:
        return self.free - MARGIN

    @property
    def padded(self) -> tuple[int, int]:
        """The frame the model takes: the target in seedvr2x's padding (job.padded_size), or in
        numz's, zeros up to multiples of 16, for the tests (model.input_transform)."""
        if self.numz_padding:
            return -(-self.target[0] // 16) * 16, -(-self.target[1] // 16) * 16
        return padded_size(self.target)

    @property
    def output(self) -> tuple[int, int]:
        return output_size(self.target)

    @property
    def smallest(self) -> int:
        """The smallest cap the job can run: no cap goes under 2 * SHARED + 1 latents, a window
        in the middle of a shot keeping one of its own (DESIGN.md, Terms, Window); the longest
        shot's length when every shot is shorter, each then one window whatever the cap."""
        return min(2 * SHARED + 1, self.lengths[-1])


def plan(
    free: int,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    split: bool = False,
    numz_padding: bool = False,
    recorded: Plan | None = None,
) -> tuple[Plan, Peaks]:
    """The plan of a job of `shots` resized to target (job.target_size), in `free` bytes of GPU
    memory as the driver reports them, less MARGIN, and what it is predicted to take; refused
    (PlanError) when a pin can't be taken exactly, when no plan fits that memory, or when the plan
    takes more system memory than host_available, in bytes (None: not checked). split: whether the
    decode corrects colours; numz_padding is for tests only (cli.NUMZ_PADDING).

    The choices, each made once for the job (DESIGN.md, Memory planner, Choices, in order):
    1. the window cap, the longest shot's length when the DiT's phase holds it, with as many
       blocks swapped as it takes and the VAE's weights away, else the largest cap whose longest
       window it holds: what it must hold is the longest window the job really runs, a shot of
       80 latents under a cap of 76 running two of 41;
    2. the fewest swapped blocks that hold that window, the VAE leaving the DiT's phase before
       one more block is swapped;
    3. the VAE's tiles, the decode's then the encode's: none where the untiled pass fits, else
       the largest that fit, the DiT's blocks leaving the phase before a tile shrinks; then as
       many of them left there as fit.

    recorded is the plan of a job resumed that has made units: what shapes the output is taken
    from it as it is, the cap, the tiles and the compile flags, and where the weights are planned
    again around it (DESIGN.md, Memory planner, A plan shapes the output a little)."""
    job = _job(free, target, shots, split, numz_padding)
    if pins.swap_blocks is not None and pins.swap_blocks > BLOCKS:
        raise PlanError(
            f"--swap-blocks {pins.swap_blocks}: the DiT has {BLOCKS} blocks, so from 0 to {BLOCKS}"
        )
    if recorded is None:
        made = _chosen(job, pins)
        refused = _unfit(job, made, host_available)
    else:
        made = _around(job, recorded, pins)
        refused = _unfit(job, made, host_available)
        if refused is not None:
            refused = f"the recorded plan no longer fits the memory free now: {refused}"
    if refused is not None:
        raise PlanError(refused)
    return made, _peaks(job, made)


def fits(
    plan: Plan,
    free: int,
    target: tuple[int, int],
    shots: Sequence[Shot],
    host_available: int | None,
    *,
    split: bool = False,
    numz_padding: bool = False,
) -> str | None:
    """None when every phase of `plan` fits `free` bytes of GPU memory less MARGIN, and the plan
    host_available bytes of system memory (None: not checked), for a job as plan() takes it; else
    one sentence naming the first phase that doesn't and what it is short of, in GiB."""
    return _unfit(_job(free, target, shots, split, numz_padding), plan, host_available)


def check_recorded(recorded: Plan, shots: Sequence[Shot]) -> None:
    """Refuse (PlanError) a recorded plan whose cap no layout exists for (shot.window_layout):
    one under 2 * SHARED + 1 latents that a shot is longer than. The planner makes a cap that
    short only for a job whose longest shot is as short, each shot then one window
    (_Job.smallest), so that record was edited; Plan.recorded, which has no shots, can't tell
    the two apart."""
    longest = max(_latents(shot.frames) for shot in shots)
    if recorded.window is not None and recorded.window < min(2 * SHARED + 1, longest):
        raise PlanError(
            f"not a plan as seedvr2x records one: its window, {recorded.window}, with a shot of"
            f" {_count(longest, 'latent')}: a cap that a shot is longer than is"
            f" {2 * SHARED + 1} latents at least"
        )


def said(
    plan: Plan,
    peaks: Peaks,
    card: str,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    *,
    numz_padding: bool = False,
) -> list[str]:
    """The plan as --plan prints it and every run logs it (DESIGN.md, Memory planner, --plan), a
    line each: the card (its name, `card`), its free memory and the budget; the padded size and
    the tokens per latent; the shots and the longest; the window cap, the shots in one window and
    the joins; the swapped blocks and what stays on the GPU in each phase; the pinned system
    memory; the tiles and their grid; each phase's predicted peak and which bounds the plan; the
    scratch of the longest tiled shot; what was pinned. PROVISIONAL (implementation, 2026-10-10):
    the wording; the time estimates' lines come with them."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import shot_layout

    job = _job(peaks.free, target, shots, False, numz_padding)
    height, width = job.padded
    frames = Counter(shot.frames for shot in shots)
    windows = {count: len(shot_layout(count, plan.window)) for count in frames}
    whole = sum(number for count, number in frames.items() if windows[count] == 1)
    joins = sum((windows[count] - 1) * number for count, number in frames.items())
    cap = (
        "no cap on the windows"
        if plan.window is None
        else f"windows of {_count(plan.window, 'latent')} at most"
    )
    lines = [
        f"{card}, {_gib(peaks.free)} GiB free, a budget of {_gib(peaks.budget)} GiB",
        f"frames padded to {width}x{height}, {peaks.tokens:,} tokens a latent",
        f"{_count(len(shots), 'shot')}, the longest of {_count(job.frames, 'frame')}"
        f" ({_count(job.lengths[-1], 'latent')})",
        f"{cap}, the longest run {_count(peaks.window, 'latent')}: {whole} of"
        f" {_count(len(shots), 'shot')} in one window, {_count(joins, 'join')}",
    ]
    swapped = (
        f"{plan.swapped} of the DiT's {BLOCKS} blocks swapped"
        if plan.swapped
        else "no block of the DiT swapped"
    )
    vae = "on" if plan.vae_in_dit else "off"
    lines.append(
        f"{swapped}, the VAE {vae} the GPU in the DiT's phase; in the encode,"
        f" {_resident_said(plan.resident_encode, plan.swapped)}; in the decode,"
        f" {_resident_said(plan.resident_decode, plan.swapped)}"
    )
    leaving = _leaving(plan)
    if leaving or not plan.vae_in_dit:
        left = [_count(leaving, "block")] if leaving else []
        left += [] if plan.vae_in_dit else ["the VAE"]
        pinned = f"{_gib(_pinned(plan))} GiB of system memory pinned for {' and '.join(left)}"
    else:
        pinned = "no system memory pinned, nothing leaving the GPU"
    lines.append(f"{pinned}; {_gib(peaks.host)} GiB of system memory in all")
    lines.append(
        f"encode {_pass_said(job, plan.encode_tiles)}, decode {_pass_said(job, plan.decode_tiles)}"
    )
    names = {"encode": "encode", "dit": "DiT's phase", "decode": "decode"}
    lines.append(
        f"predicted peaks: encode {_gib(peaks.encode)} GiB, DiT {_gib(peaks.dit)} GiB, decode"
        f" {_gib(peaks.decode)} GiB; the {names[peaks.bound]} bounds the plan,"
        f" {_gib(peaks.budget - peaks.peak(peaks.bound))} GiB under the budget"
    )
    if peaks.scratch:
        kept = min(peaks.scratch, SCRATCH_ALLOWANCE)
        where = (
            "in system memory"
            if kept == peaks.scratch
            else f"{_gib(kept)} GiB of it in system memory, the rest in a file"
        )
        lines.append(f"scratch of the longest tiled shot: {_gib(peaks.scratch)} GiB, {where}")
    options = (
        ("--window", pins.window),
        ("--swap-blocks", pins.swap_blocks),
        ("--encode-tile", pins.encode_tile),
        ("--decode-tile", pins.decode_tile),
    )
    given = [f"{option} {value}" for option, value in options if value is not None]
    lines.append(f"pinned: {', '.join(given)}" if given else "nothing pinned")
    return [f"plan: {line}" for line in lines]


def _job(
    free: int, target: tuple[int, int], shots: Sequence[Shot], split: bool, numz_padding: bool
) -> _Job:
    if not shots:
        raise ValueError("a job without a shot")
    lengths = sorted({_latents(shot.frames) for shot in shots})
    frames = max(shot.frames for shot in shots)
    return _Job(free, target, tuple(lengths), frames, split, numz_padding)


def _latents(frames: int) -> int:
    """The latents of a shot of `frames` frames, padded to 4n + 1 (shot.padded_length)."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import padded_length

    return (padded_length(frames) - 1) // 4 + 1


@cache
def _longest(lengths: tuple[int, ...], cap: int | None) -> int:
    """The longest window, in latents, that shots of `lengths` latents really run under a cap
    (None: each in one window): the longest of their balanced layouts' (shot.window_layout),
    which is the cap only where a shot's length makes it so. It never falls as the cap rises:
    fewer windows over a shot are longer ones (tests/test_planner.py holds it)."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import window_layout

    if cap is None or cap >= lengths[-1]:
        return lengths[-1]
    return max(end - start for length in lengths for start, end in window_layout(length, cap))


def _holds(job: _Job, cap: int | None, swapped: int) -> bool:
    """Whether the DiT's phase holds the longest window the job runs under a cap, `swapped`
    blocks swapped and the VAE's weights away: all a window can ask for."""
    peak = dit_peak(job.padded, _longest(job.lengths, cap), swapped, vae_in_dit=False)
    return peak <= job.budget


def _longest_cap(job: _Job, swapped: int) -> int | None:
    """The largest cap, from the longest shot's length down to the smallest the job can run, that
    the DiT's phase holds with `swapped` blocks swapped (_holds); None when it holds none.

    A film has thousands of shots and windows of hundreds of latents: the caps whose longest
    window is over the budget by its tokens alone are bisected away first (_longest never falls
    as the cap rises, nor the tokens with it), the window functions then asked from there down."""

    def possible(cap: int) -> bool:
        least = _dit_least(job.padded, _longest(job.lengths, cap), swapped, vae_in_dit=False)
        return least <= job.budget

    low, high = job.smallest, job.lengths[-1]
    if not possible(low):
        return None
    while low < high:
        middle = (low + high + 1) // 2
        if possible(middle):
            low = middle
        else:
            high = middle - 1
    return next(
        (cap for cap in range(low, job.smallest - 1, -1) if _holds(job, cap, swapped)), None
    )


def _fewest(job: _Job, latents: int) -> tuple[int, bool] | None:
    """The fewest swapped blocks with which the DiT's phase holds a window of `latents` latents,
    and whether the VAE's weights stay on the GPU meanwhile: they leave before one more block is
    swapped. One and two swapped blocks save nothing (dit_peak), so are never chosen. None when
    every block swapped and the VAE away don't hold it."""
    for swapped in (0, *range(SLOTS + 1, BLOCKS + 1)):
        for vae_in_dit in (True, False):
            if dit_peak(job.padded, latents, swapped, vae_in_dit) <= job.budget:
                return swapped, vae_in_dit
    return None


def _chosen(job: _Job, pins: Pins) -> Plan:
    """The plan made whole, around the pins (plan)."""
    floor = _floor(job)
    if floor is not None:
        raise PlanError(_too_large(job, *floor))
    cap, swapped, vae_in_dit = _window(job, pins)
    decode = _pass(job, "decode", pins.decode_tile, swapped)
    encode = _pass(job, "encode", pins.encode_tile, swapped)
    return Plan(
        cap,
        encode,
        decode,
        swapped,
        _resident(job, "encode", encode, swapped),
        _resident(job, "decode", decode, swapped),
        vae_in_dit,
    )


def _around(job: _Job, recorded: Plan, pins: Pins) -> Plan:
    """A recorded plan's shape as it is, where the weights are planned again around it (plan): the
    fewest swapped blocks that hold its windows, or --swap-blocks' when given, and as much of the
    DiT in the VAE's phases as fits; every block swapped and none left there when it no longer
    fits, which _unfit then says."""
    latents = _longest(job.lengths, recorded.window)
    if pins.swap_blocks is None:
        swapped, vae_in_dit = _fewest(job, latents) or (BLOCKS, False)
    else:
        swapped = pins.swap_blocks
        vae_in_dit = dit_peak(job.padded, latents, swapped) <= job.budget
        fewest = _fewest(job, latents)
        if not _holds(job, recorded.window, swapped) and fewest is not None:
            # The pin's doing, not the memory's: the recorded shape still fits with more blocks.
            # PROVISIONAL (implementation, 2026-10-10): refused as the pin is on a new job.
            raise PlanError(_swap_refused(job, swapped, latents, "the recorded plan's", fewest[0]))
    return replace(
        recorded,
        swapped=swapped,
        resident_encode=_resident(job, "encode", recorded.encode_tiles, swapped),
        resident_decode=_resident(job, "decode", recorded.decode_tiles, swapped),
        vae_in_dit=vae_in_dit,
    )


def _window(job: _Job, pins: Pins) -> tuple[int, int, bool]:
    """The window cap, the swapped blocks and whether the VAE's weights stay in the DiT's phase
    (plan's choices 1 and 2), --window and --swap-blocks taken exactly or refused; the job's
    floor holds already (_floor).

    PROVISIONAL (implementation, 2026-10-10), three choices DESIGN.md doesn't make. The cap is
    always a number, the longest shot's length when it fits, never None. With --swap-blocks, the
    cap is the largest that fits with those blocks swapped and the VAE's weights away, as it is
    without the pin, the window being the first choice: --swap-blocks 0 runs 6 latents at 1080p
    on a 24 GB card, where DESIGN.md's 5, every block on the GPU, count the VAE there. And of
    two pins that don't fit together, --window is the one refused while some cap fits the blocks
    pinned, --swap-blocks when none does."""
    most = BLOCKS if pins.swap_blocks is None else pins.swap_blocks
    longest = _longest_cap(job, most)
    if longest is None:
        # The floor holds every block swapped: a --swap-blocks that holds no window at all.
        cap = job.smallest
        whose = "the shortest this job can run"
        if pins.window is not None and _holds(job, pins.window, BLOCKS):
            cap, whose = pins.window, f"--window {pins.window}'s longest"
        latents = _longest(job.lengths, cap)
        fewest = _fewest(job, latents)
        assert fewest is not None
        raise PlanError(_swap_refused(job, most, latents, whose, fewest[0]))
    if pins.window is None:
        cap = longest
    elif _holds(job, pins.window, most):
        cap = pins.window
    else:
        latents = _longest(job.lengths, pins.window)
        need = dit_peak(job.padded, latents, most, vae_in_dit=False) + MARGIN
        blocks = (
            "every block of the DiT swapped"
            if pins.swap_blocks is None
            else f"--swap-blocks {pins.swap_blocks}"
        )
        raise PlanError(
            f"--window {pins.window}: its longest window, {_count(latents, 'latent')}, takes"
            f" {_over(need)} GiB of free VRAM with {blocks}, and the driver reports"
            f" {_under(job.free)} GiB free: the longest window that fits is --window {longest}"
        )
    latents = _longest(job.lengths, cap)
    if pins.swap_blocks is not None:
        return cap, most, dit_peak(job.padded, latents, most) <= job.budget
    fewest = _fewest(job, latents)
    assert fewest is not None
    return cap, *fewest


def _swap_refused(job: _Job, swapped: int, latents: int, whose: str, fewest: int) -> str:
    """--swap-blocks refused: the fewest blocks that hold the window named."""
    need = dit_peak(job.padded, latents, swapped, vae_in_dit=False) + MARGIN
    return (
        f"--swap-blocks {swapped}: a window of {_count(latents, 'latent')}, {whose}, takes"
        f" {_over(need)} GiB of free VRAM with {_count(swapped, 'block')} swapped, and the driver"
        f" reports {_under(job.free)} GiB free: the fewest blocks that hold it are"
        f" --swap-blocks {fewest}"
    )


def _tile_sizes(padded: tuple[int, int]) -> range:
    """The sides of the tiles the planner makes for padded frames, the largest first: multiples
    of TILE_STEP from the largest under the frame's longer side, a tile that covers the frame
    being the untiled pass, down to TILE_MIN; none for a frame that small."""
    largest = (max(padded) - 1) // TILE_STEP * TILE_STEP
    return range(largest, TILE_MIN - 1, -TILE_STEP)


def _vae_peak(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """The predicted peak of the job's encode or decode (encode_peak, decode_peak)."""
    if phase == "encode":
        return encode_peak(job.padded, tiles, resident, swapped)
    return decode_peak(job.padded, job.output, tiles, resident, swapped, job.split)


def _fitting(job: _Job, phase: Phase, swapped: int) -> list[Tiles | None]:
    """The passes of the job's encode or decode that fit the budget with the DiT's blocks away,
    in the planner's order: untiled, then tiled from the largest tiles down."""
    passes = (None, *(Tiles(size) for size in _tile_sizes(job.padded)))
    return [tiles for tiles in passes if _vae_peak(job, phase, tiles, 0, swapped) <= job.budget]


def _pass(job: _Job, phase: Phase, pin: int | None, swapped: int) -> Tiles | None:
    """The tiles of the job's encode or decode (plan's choice 3), None untiled; --encode-tile or
    --decode-tile, `pin`, taken exactly or refused, the refusal naming a pass that fits. The job's
    floor holds already (_floor), so some pass fits: a tile, or the untiled pass alone on a frame
    little larger than a tile (_least_pass)."""
    fitting = _fitting(job, phase, swapped)
    if pin is None:
        return fitting[0]
    option = f"--{phase}-tile {pin}"
    tiles = Tiles(pin) if pin else None
    if tiles is not None and covers(job.padded, tiles):
        height, width = job.padded
        raise PlanError(
            f"{option}: a tile of {pin} px covers the whole padded frame, {width}x{height}:"
            f" --{phase}-tile 0 pins an untiled {phase}"
        )
    need = _vae_peak(job, phase, tiles, 0, swapped) + MARGIN
    if need <= job.free:
        return tiles
    largest = next((fit.size for fit in fitting if fit is not None), None)
    instead = (
        f"no tile fits, the untiled {phase} does, --{phase}-tile 0"
        if largest is None
        else f"the largest tile that fits is --{phase}-tile {largest}"
    )
    raise PlanError(
        f"{option}: the {phase} {_tiles_said(tiles)} takes {_over(need)} GiB of free VRAM, the"
        f" DiT's blocks off the GPU, and the driver reports {_under(job.free)} GiB free: {instead}"
    )


def _resident(job: _Job, phase: Phase, tiles: Tiles | None, swapped: int) -> int | None:
    """The DiT's blocks left on the GPU in the job's encode or decode, in `tiles`: None, the DiT
    staying as its own phase has it, when that fits; else as many of its blocks as fit, none at
    worst (Plan.resident_encode, Plan.resident_decode)."""
    if _vae_peak(job, phase, tiles, None, swapped) <= job.budget:
        return None
    for resident in range(BLOCKS - swapped - 1, 0, -1):
        if _vae_peak(job, phase, tiles, resident, swapped) <= job.budget:
            return resident
    return 0


def _floor(job: _Job) -> tuple[Phase, int] | None:
    """The phase that takes the most where every phase takes its least, and its peak there, when
    that is over the budget, no plan then fitting the job (DESIGN.md, Memory planner: a size that
    doesn't fit is refused, never degraded); else None. The least: the smallest window the job
    can run, every block of the DiT swapped and the VAE's weights away; the encode and the decode
    each by the pass that takes the less of its untiled one and its smallest tiles (_least_pass),
    the DiT's blocks away, its slots there.

    PROVISIONAL (implementation, 2026-10-10): the VAE's floor counts the two slots, as with every
    block swapped, whatever the plan would swap. A job whose smallest window fits unswapped has
    room for them in the VAE's phases anyway: its DiT takes 15.9 GiB and 20 more per 4K frame's
    worth of 5 latents, the decode in 512-px tiles 7.2 and split's 2.8."""
    least: dict[Phase, int] = {
        "encode": _vae_peak(job, "encode", _least_pass(job, "encode"), 0, BLOCKS),
        "dit": dit_peak(job.padded, _longest(job.lengths, job.smallest), BLOCKS, vae_in_dit=False),
        "decode": _vae_peak(job, "decode", _least_pass(job, "decode"), 0, BLOCKS),
    }
    phase = max(PHASES, key=lambda phase: least[phase])
    return (phase, least[phase]) if least[phase] > job.budget else None


def _least_pass(job: _Job, phase: Phase) -> Tiles | None:
    """The pass of the job's encode or decode that takes the least, the DiT's blocks away and its
    slots there: its smallest tiles, or None, the untiled pass, where the frame is too small for
    a tile or little larger than one. The small tiles' fits start above the untiled ones, by 0.5
    GiB for the encode and 0.8 for the decode (ENCODE_TILED_SMALL, DECODE_TILED_SMALL), which a
    tile must save by its area: a padded frame of 544x512 decodes untiled in 0.41 GiB less than
    in tiles of 512 px."""
    sizes = _tile_sizes(job.padded)
    passes = (None, Tiles(sizes[-1])) if sizes else (None,)
    return min(passes, key=lambda tiles: _vae_peak(job, phase, tiles, 0, BLOCKS))


def _too_large(job: _Job, phase: Phase, peak: int) -> str:
    """The floor's refusal (_floor), counted in free VRAM, never in card sizes (DESIGN.md, Memory
    planner): the free VRAM the output size takes, what the driver reports free, and the largest
    output size that fits in it, by its short side, a multiple of 16, the target scaled in
    proportion and the floor checked there with the same shots. --resolution is named first, and
    the size as about that: it is the target's scaled, not the source's, which the command line
    scales, and the two can differ by 2 px (a 1920x1080 source refused at --resolution 600:
    794x448 here, 796x448 by --resolution 448)."""
    height, width = job.output
    if phase == "dit":
        latents = _longest(job.lengths, job.smallest)
        least = (
            f"its smallest window, {_count(latents, 'latent')}, with every block of the DiT"
            " swapped and the VAE off the GPU"
        )
    else:
        tiles = _least_pass(job, phase)
        if tiles is not None:
            how = _tiles_said(tiles)
        elif _tile_sizes(job.padded):
            how = "untiled, which takes less than tiles at that size"
        else:
            how = "untiled, too small for a tile"
        least = f"its {phase} {how}, with the DiT's blocks off the GPU"
    fitting = "no output size fits there"
    target_height, target_width = job.target
    for short in range((min(job.target) - 1) // 16 * 16, 0, -16):
        if target_height <= target_width:
            scaled = (short, short * target_width // target_height)
        else:
            scaled = (short * target_height // target_width, short)
        try:
            check_target(scaled, short)
        except JobError:
            break  # too small for the padding, as every smaller one is
        smaller = replace(job, target=scaled)
        if _floor(smaller) is None:
            fits_height, fits_width = smaller.output
            fitting = (
                f"the largest output that fits there is --resolution {short}, about"
                f" {fits_width}x{fits_height}"
            )
            break
    return (
        f"an output of {width}x{height} takes {_over(peak + MARGIN)} GiB of free VRAM ({least}:"
        f" a peak of {_over(peak)} GiB, and {_gib(MARGIN)} of margin), and the driver reports"
        f" {_under(job.free)} GiB free: {fitting}"
    )


def _leaving(plan: Plan) -> int:
    """The DiT's blocks that leave the GPU in some phase of the plan, each then waiting in a
    pinned copy: its swapped ones, and those the VAE's phases send away."""
    stay = BLOCKS - plan.swapped
    there = (stay if resident is None else resident for resident in _residents(plan))
    return BLOCKS - min(stay, *there)


def _residents(plan: Plan) -> tuple[int | None, int | None]:
    return plan.resident_encode, plan.resident_decode


def _pinned(plan: Plan) -> int:
    """The pinned system memory of the plan, in bytes: a copy of each block that leaves the GPU
    in some phase, and of the VAE's weights when they leave the DiT's (DESIGN.md, BlockSwap,
    rewritten)."""
    return round(BLOCK * _leaving(plan)) + (0 if plan.vae_in_dit else VAE_WEIGHTS)


def _peaks(job: _Job, plan: Plan) -> Peaks:
    """What the plan is predicted to take for the job.

    Its system memory: the process's own during the shots, the pinned copies of what leaves the
    GPU, and a tiled shot's scratch up to its allowance. PROVISIONAL (implementation,
    2026-10-10): the loader's own peak while the weights load, 16.5 GiB, mostly the weights' file
    mapped (DESIGN.md, Memory planner, Host RAM), isn't counted."""
    latents = _longest(job.lengths, plan.window)
    tiled = plan.encode_tiles is not None or plan.decode_tiles is not None
    kept = scratch(job.padded, job.frames) if tiled else 0
    return Peaks(
        job.free,
        job.budget,
        _vae_peak(job, "encode", plan.encode_tiles, plan.resident_encode, plan.swapped),
        dit_peak(job.padded, latents, plan.swapped, plan.vae_in_dit),
        _vae_peak(job, "decode", plan.decode_tiles, plan.resident_decode, plan.swapped),
        HOST_RESIDENT + _pinned(plan) + min(kept, SCRATCH_ALLOWANCE),
        kept,
        latents,
        tokens(job.padded, 1),
    )


def _unfit(job: _Job, plan: Plan, host_available: int | None) -> str | None:
    """Why the plan doesn't fit the job's memory (fits): the first phase over the budget, or the
    system memory it takes over what is available; None when it fits."""
    peaks = _peaks(job, plan)
    blocks = _count(plan.swapped, "block")
    vae = "" if plan.vae_in_dit else " and the VAE off the GPU"
    whats = {
        "encode": f"the encode, {_tiles_said(plan.encode_tiles)}, with"
        f" {_resident_said(plan.resident_encode, plan.swapped)},",
        "dit": f"the DiT's phase, a window of {_count(peaks.window, 'latent')} with {blocks}"
        f" swapped{vae},",
        "decode": f"the decode, {_tiles_said(plan.decode_tiles)}, with"
        f" {_resident_said(plan.resident_decode, plan.swapped)},",
    }
    for phase in PHASES:
        need = peaks.peak(phase) + MARGIN
        if need > job.free:
            return (
                f"{whats[phase]} takes {_over(need)} GiB of free VRAM and the driver reports"
                f" {_under(job.free)} GiB free, {_short(need, job.free)} GiB short"
            )
    if host_available is not None and peaks.host > host_available:
        pinned = _pinned(plan)
        parts = [(HOST_RESIDENT, "for the process")]
        if pinned:
            leaving = _leaving(plan)
            left = [f"the {_count(leaving, 'block')} of the DiT"] if leaving else []
            left += [] if plan.vae_in_dit else ["the VAE's weights"]
            parts.append((pinned, f"pinned for {' and '.join(left)} that leave the GPU"))
        if peaks.scratch:
            parts.append((min(peaks.scratch, SCRATCH_ALLOWANCE), "for a tiled shot's scratch"))
        # Each part rounded up, as what something takes is in every refusal, and the total their
        # sum as said, so that the message's figures add up: up to 0.02 GiB over the total
        # rounded once.
        taken = [(_hundredths(count, up=True), what) for count, what in parts]
        total = sum(count for count, _ in taken)
        there = _hundredths(host_available, up=False)
        said = ", ".join(f"{count / 100:.2f} {what}" for count, what in taken)
        return (
            f"the plan takes {total / 100:.2f} GiB of system memory ({said}) and"
            f" {there / 100:.2f} GiB are available, {(total - there) / 100:.2f} GiB short"
        )
    return None


def _tiles_said(tiles: Tiles | None) -> str:
    return "untiled" if tiles is None else f"in tiles of {tiles.size} px"


def _pass_said(job: _Job, tiles: Tiles | None) -> str:
    """A pass as the plan's lines say it: untiled, or its tiles and their grid, rows x columns."""
    if tiles is None:
        return "untiled"
    rows, columns = grid(job.padded, tiles)
    return f"in tiles of {tiles.size} px, {rows} x {columns}"


def _resident_said(resident: int | None, swapped: int) -> str:
    """What of the DiT's blocks a VAE phase has on the GPU, its slots aside, in words."""
    stay = BLOCKS - swapped
    if resident == 0 or stay == 0:
        return "the DiT's blocks off the GPU"
    if resident is None or resident == stay:
        return f"the DiT's {_count(stay, 'unswapped block' if swapped else 'block')} on the GPU"
    return f"{resident} of the DiT's blocks on the GPU"


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _gib(count: float) -> str:
    """Bytes in GiB, to 2 decimals."""
    return f"{count / GIB:.2f}"


def _over(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded up: what something takes."""
    return f"{_hundredths(count, up=True) / 100:.2f}"


def _under(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded down: what is there, so that what takes more than
    is there never reads as taking the same."""
    return f"{_hundredths(count, up=False) / 100:.2f}"


def _short(need: float, there: float) -> str:
    """What `need` bytes are short of in `there`, in GiB to 2 decimals: the difference of the two
    as _over and _under say them, so that a message's figures add up."""
    return f"{(_hundredths(need, up=True) - _hundredths(there, up=False)) / 100:.2f}"


def _hundredths(count: float, *, up: bool) -> int:
    """Bytes in hundredths of a GiB, rounded up or down, a figure that is one already kept."""
    exact = count / GIB * 100
    return math.ceil(exact - 1e-6) if up else math.floor(exact + 1e-6)
