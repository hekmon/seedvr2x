"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model: what each phase is predicted to peak at on the GPU, in bytes, by the fits of
research/docs/planner-limits.md and research/docs/vram.md, measured on numz's CLI with the 7B
fp16, the one DiT seedvr2x runs (weights.ACCEPTED); where the weights are in each phase, and what
that takes, by what runtime/blockswap.py does with them, measured on seedvr2x itself (milestone
3); the tiles and their scratch by runtime/tiling.py's. Sizes are (height, width) in pixels;
`padded` is the frame the model takes, the target padded (job.padded_size).

Its choices, in DESIGN.md's order (plan): the window, the biggest quality lever; the fewest
swapped blocks that hold it; the largest VAE tiles, none where the untiled pass fits. A pin is
taken exactly or refused, the refusal naming what fits; a size that doesn't fit is refused, never
degraded (PlanError).

PROVISIONAL (implementation, 2026-10-11): four corrections from the planner's first two matrices
on seedvr2x, emulated cards (tests/test_planner.py, test_first_matrix): ENCODE_EXTRA on every
encode, split's term off a tiled decode's peak (decode_peak), no tile chosen above TILE_MAX, and
every VAE pass held to a room under the budget besides its peak (VAE_ROOM)."""

import math
import mmap
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
# By its parts as seedvr2x has them (below): the 36 blocks 15.1925 GiB, the DiT's other
# parameters 0.1550, its buffers and tables with the text embeddings 0.058, the VAE 0.4669, in
# all 15.8723, 0.0013 over the fit's constant, within its 0.005.
# PROVISIONAL (implementation, 2026-10-11): the constants stay numz's fit's. seedvr2x's own peak,
# one window of 1 latent at 1080p, no block swapped, measured 17.016 GiB (milestone 3,
# 2026-10-11; tests/test_placement.py measures it), where this model gives 16.955: 0.061 GiB
# under, of the margin's 0.6. A refit on seedvr2x is measurement's.
DIT_LOADED = round(15.871 * GIB)
PER_TOKEN = 127.16 * 1024
PER_WINDOW = 2.726 * 1024**2
# The attention windows along time, height and width that every block of the 7B asks its window
# function for, the blocks alternating the two layouts (vendor/configs_7b/main.yaml, window and
# window_method).
NUM_WINDOWS = (4, 3, 3)

# The VAE's weights, on the GPU in the VAE's phases, and in the DiT's unless the plan sends them
# away for it, in bytes: 0.4669 GiB, its parameters by the model's shapes in the pipeline's
# bfloat16, each on a multiple of 512 bytes, as the part that leaves lays them out (model.Models'
# vae, a blockswap.Part; tests/test_planner.py builds the model). Measured on numz as DESIGN.md's
# 0.47 GiB: 28.15 to 27.68 GiB (Memory planner, Per-phase peaks; vram.md, Practical rules).
VAE_WEIGHTS = 501_306_880

# The DiT's blocks, and a block's parameters in bytes, 0.4220 GiB, the 36 holding 15.19 of the
# DiT's 15.35 GiB (DESIGN.md, Memory planner, Host RAM): blockswap.BLOCK_BYTES, by the model's
# shapes, a literal here since blockswap imports torch (tests/test_planner.py holds the two
# equal). SLOTS: the buffers on the GPU its swapped blocks stream through (blockswap.SLOTS;
# DESIGN.md, BlockSwap, rewritten).
BLOCKS = 36
BLOCK = 453_133_312
SLOTS = 2
# The DiT's parameters outside its blocks, emb_in, txt_in, vid_in and vid_out, in bytes: 0.155
# GiB, by the model's shapes in fp16 (tests/test_planner.py builds the model). They leave the GPU
# only with the last block a VAE phase keeps (dit_on_gpu).
DIT_OTHER = 166_416_512
# What of the DiT never leaves the GPU (runtime/blockswap.py: only parameters move): RoPE's
# buffers with the text embeddings, 0.025 GiB, and the tables its blocks cache from them once a
# window has run; 0.058 GiB in all, measured on seedvr2x after windows of 1 latent at 1080p
# (milestone 3, 2026-10-11).
DIT_BUFFERS = round(0.058 * GIB)

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

# What seedvr2x's encode holds over its fit, tiled or not, in bytes: the largest excess of a
# measured peak over the fit's in the planner's first matrix (the 96 GB card emulating smaller
# ones, 2026-10-11): at 1080p, 9.66 GiB measured where the fit gave 9.546 in tiles of 896 px
# (+0.11), and 13.91 for 13.511 in tiles of 1216 px (+0.40); untiled at 540p, 22.53 for 22.22
# (+0.31). At 4K the refit is above the measurement: 29.53 for 29.694 in tiles of 1792 px
# (-0.16), 21.26 for 21.648 at 1472 px (-0.39). PROVISIONAL (implementation, 2026-10-11): one
# constant on every encode; a refit of the encode on seedvr2x is measurement's.
ENCODE_EXTRA = round(0.4 * GIB)
# The room every VAE pass, encode or decode, tiled or not, must leave under the budget besides
# its predicted peak, in bytes: 2.1 GiB under the free memory with the margin. Over the
# planner's two matrices on emulated cards (2026-10-11), a VAE pass whose peak comes close to
# the free memory runs out of memory in memory_limit_conv's torch.cat: the device full while
# torch holds 3.3 to 4.9 GiB less than the free memory it started with, at the pass's largest
# request, 2.0 to 4.8 GiB. Every pass that had 1.67 GiB or more between its measured peak and
# the free memory ran (1.67, 1.97, 3.53, 3.69, 3.8, 4.98, 9.97, 13.8, 25.6); every failure had
# 1.6 GiB or less between its predicted, or true, peak and the free memory (0.65, 0.80, 0.86,
# 0.94, 1.09, 1.53, 1.6); passes also ran with 0.63, 0.88, 1.01, 1.28 and 1.32: the band
# between isn't safe. The first matrix's tiled decodes passed only for split's term, which
# padded their predicted peaks then (decode_peak). PROVISIONAL (implementation, 2026-10-11): a
# guard, the cause isn't known; numz's margin, 0.6 GiB, was bisected on passes whose largest
# request was 0.4 GiB (MARGIN).
VAE_ROOM = round(1.5 * GIB)

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
# multiple of TILE_STEP pixels from TILE_MIN up, the smallest tile it makes, to TILE_MAX, the
# largest it chooses: the tiled fits were refitted on tiles of 1024 to 2048 px, and a 4K encode
# in tiles of 2304 px (1 x 2 of 2192x2304) ran out of memory on an emulated 48 GB card, 46.54
# GiB free, at 45.89 GiB or more where its fit gave 45.62 (41.07 allocated and 4.82 asked for;
# the planner's first matrix, 2026-10-11). PROVISIONAL (implementation, 2026-10-11): the fits'
# range as the bound, until larger tiles are measured.
TILE_MIN = 512
TILE_MAX = 2048
TILE_STEP = 64

# The process's resident system memory during the shots, which stays flat: 2.3-2.4 GiB over 6
# shots (DESIGN.md, Memory planner, Host RAM).
HOST_RESIDENT = round(2.4 * GIB)

# A tiled pass's scratch, in bytes per pixel of the padded frame and per frame: 3 channels in
# bfloat16, 12.7 MB per 1080p frame and 50.5 MB per 4K frame (DESIGN.md, Pipeline, per shot, step
# 3). The tiled encode's, its input kept, and the tiled decode's, its frames summed, are the same
# store, of the shot's frames padded to 4n + 1 (tiling.Scratch.needed, which
# tests/test_planner.py holds scratch() to), one at a time.
SCRATCH_PIXEL = 6
# What of a scratch stays in system memory, a larger one going whole to a memory-mapped file:
# tiling.ALLOWANCE, a literal here since tiling imports torch (tests/test_planner.py holds the
# two equal). PROVISIONAL there: DESIGN.md gives the allowance no figure.
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

    A swapped block takes its BLOCK off the peak, but for the SLOTS on the GPU (DESIGN.md,
    BlockSwap, rewritten), so that one or two swapped blocks save nothing: measured on seedvr2x,
    one window of 1 latent at 1080p, the peak with S blocks swapped is the peak with none less
    BLOCK x max(S - 2, 0), to 0.001 GiB (17.016 GiB with none, 16.595 with 3, 12.796 with 12,
    2.668 with 36; milestone 3, 2026-10-11, as tests/test_placement.py measures it). numz's own
    swap, one block there at a time, saved 0.4227 GiB x (S - 1)."""
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
    swapped, by what the phase is entered with (Plan.resident_encode, Plan.resident_decode), as
    blockswap.Placement.enter leaves its parameters there (Placement.device_bytes), with what
    never moves (DIT_BUFFERS):
    - None: nothing moves, the DiT as its own phase has it: its other parameters (DIT_OTHER),
      the blocks that aren't swapped, and the slots, one per swapped block up to SLOTS;
    - 0: nothing of its parameters, the other ones away with the blocks;
    - a number up to the blocks that aren't swapped: that many of them, and its other
      parameters; the slots are freed, and come back with the DiT's phase.
    ValueError for a count the DiT hasn't, as the loader refuses it (model.load_models)."""
    fixed = BLOCKS - swapped
    if not 0 <= swapped <= BLOCKS or not (resident is None or 0 <= resident <= fixed):
        raise ValueError(f"{resident} blocks resident, {swapped} swapped, of the DiT's {BLOCKS}")
    if resident is None:
        return DIT_BUFFERS + DIT_OTHER + BLOCK * (fixed + min(swapped, SLOTS))
    if resident == 0:
        return DIT_BUFFERS
    return DIT_BUFFERS + DIT_OTHER + BLOCK * resident


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

    The tiled passes' own grid, counted in pixels where tiling.grid lays it over the latents,
    which imports torch: tests/test_planner.py holds the two equal, and tile_area to
    tiling.largest."""
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
    the VAE's weights and what of the DiT stays there (dit_on_gpu's resident and swapped), and
    what seedvr2x's encode holds over the fit (ENCODE_EXTRA). Not the shot's frames: seedvr2x
    feeds the encode slice by slice, where numz moves a batch's there first, 0.046 GiB per 4K
    frame (DESIGN.md, Memory planner, Per-phase peaks). Nor the room every VAE pass is held to
    (VAE_ROOM), which the planner judges it by and no peak holds."""
    fit = _fit(padded, tiles, ENCODE_UNTILED, ENCODE_TILED, ENCODE_TILED_SMALL)
    return round(fit) + ENCODE_EXTRA + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
) -> int:
    """The predicted peak of the VAE's decode of padded frames, in bytes, its fit above what is
    on the GPU before the call as encode_peak has the encode's (without ENCODE_EXTRA, the
    encode's); with split, the colour correction too, which corrects a slice of the decode at
    once, SLICE frames at most of the output's size (job.output_size).

    The untiled decode, streamed, corrects each slice while it holds its working memory, and its
    peak is the two together: at 1080p with every weight on the GPU, 50.98 GiB measured for
    51.50 predicted, 50.80 without the correction's term (the 96 GB card, 2026-10-11). A tiled
    decode corrects its frames after its last tile, when the tile's working memory is gone, and
    its peak is the larger of the two, the tile's in practice. PROVISIONAL (implementation,
    2026-10-11), by the planner's first matrix (emulated cards, 2026-10-11), where the tiled
    decodes' peaks were under the sum by the correction's term or about: at 4K, where it is
    2.80 GiB, by 2.85 and 2.84 (26.85 GiB measured for 29.695 in tiles of 1216 px, 19.01 for
    21.845 at 1024 px); at 1080p, where it is 0.70, by 0.75 in tiles of 832 px (12.87 for
    13.624) and by 0.57 at 640 px (9.22 for 9.792), whose prediction is then 0.13 GiB under
    its measured peak, the small tiles' fit that much short there (tests/test_planner.py,
    test_first_matrix). A tile that covers the frame is the untiled pass (_fit)."""
    fit = _fit(padded, tiles, DECODE_UNTILED, DECODE_TILED, DECODE_TILED_SMALL)
    corrected = SPLIT_FRAME_4K * SLICE * output[0] * output[1] / PIXELS_4K if split else 0
    streamed = tiles is None or covers(padded, tiles)
    above = fit + corrected if streamed else max(fit, corrected)
    return round(above) + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def _fit(
    padded: tuple[int, int],
    tiles: Tiles | None,
    untiled: tuple[float, float],
    tiled: tuple[float, float],
    small: tuple[float, float],
) -> float:
    """A VAE pass's peak above what is on the GPU before the call, in bytes, by its fits (GiB,
    GiB per megapixel). A tile that covers the whole padded frame is the untiled pass, as the
    tiled passes run it (covers).

    PROVISIONAL (implementation, 2026-10-11): an untiled pass takes the larger of its own fit
    and the small tiles' at the padded frame's area. The untiled fits come from frames of 720p
    and up, and fall short on a small frame, which behaves as a small tile does. Measured on
    seedvr2x (the 96 GB card, 2026-10-11; every weight on the GPU, 15.87 GiB, split on): 540p,
    padded to 576x960, 0.553 megapixels, peaks at 22.53 GiB in its encode and 26.29 in its
    decode, where the untiled fits give 21.94 and 25.75, and the small tiles' 22.22 and 26.27;
    1080p, 2.12 megapixels, at 35.36 and 50.98, under the untiled fits' 35.72 and 51.50. A
    predicted peak must be an upper bound, the margin covering memory outside torch's tensors,
    not the fits' errors (planner-limits.md, The margin over free memory); the 540p encode's
    measured peak is still 0.31 GiB over its prediction. The two fits cross at 1.25 megapixels
    for the encode and 1.6 for the decode: 1080p and up are unchanged."""
    if tiles is None or covers(padded, tiles):
        area = padded[0] * padded[1] / 1e6
        return max(constant + slope * area for constant, slope in (untiled, small)) * GIB
    constant, slope = tiled if tiles.size >= REFITTED else small
    return (constant + slope * tile_area(padded, tiles)) * GIB


def covers(padded: tuple[int, int], tiles: Tiles) -> bool:
    """Whether one tile covers the whole padded frame, its grid one tile: the pass then runs
    untiled, its scratch untouched (tiling.encode_tiled, tiling.decode_tiled, as numz's tiled
    passes take it, attn_video_vae.py:1315-1317, 1489-1491)."""
    return tiles.size >= max(padded)


def scratch(padded: tuple[int, int], frames: int) -> int:
    """The scratch of a tiled pass over a shot of `frames` frames of padded frames, in bytes:
    SCRATCH_PIXEL per pixel of each of its frames, the shot padded to 4n + 1 as the VAE takes it
    (shot.padded_length): tiling.Scratch.needed for the store either tiled pass opens."""
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

    def need(self, phase: Phase) -> int:
        """What the planner judges `phase` by against the budget: its predicted peak, and for
        the encode and the decode the room a VAE pass is held to (VAE_ROOM)."""
        return self.peak(phase) + (0 if phase == "dit" else VAE_ROOM)

    @property
    def bound(self) -> Phase:
        """The phase that bounds the plan: the one with the least left under the budget, the
        VAE's phases counted with their room (need), as the planner judged them."""
        return max(PHASES, key=self.need)

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
       80 latents under a cap of 76 running two of 41; the plan's cap is then that longest
       window, which lays the shots out the same (_window);
    2. the fewest swapped blocks that hold that window, the VAE leaving the DiT's phase before
       one more block is swapped;
    3. the VAE's tiles, the decode's then the encode's: none where the untiled pass fits, else
       the largest that fit, TILE_MAX at most, each pass with its room (VAE_ROOM), the DiT
       leaving the phase before a tile shrinks; then as much of it left there as fits
       (_resident).

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
    """None when every phase of `plan` fits `free` bytes of GPU memory less MARGIN, a VAE pass
    with its room (VAE_ROOM), and the plan host_available bytes of system memory (None: not
    checked), for a job as plan() takes it; else one sentence naming the first phase that
    doesn't and what it is short of, in GiB."""
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
    memory; the tiles and their grid, and the room the VAE's passes are held to (VAE_ROOM);
    each phase's predicted peak, without that room, and which bounds the plan, a VAE phase
    counted with its room (Peaks.bound) and said so; the scratch of the longest tiled shot; what
    was pinned. PROVISIONAL (implementation, 2026-10-10): the wording; the time estimates' lines
    come with them. The longest window the job runs is said only where it isn't the cap, one
    pinned or recorded above it."""
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
    # The longest window run, where the cap doesn't say it: a cap pinned or recorded above it.
    if peaks.window != plan.window:
        cap += f", the longest run {_count(peaks.window, 'latent')}"
    lines = [
        f"{card}, {_gib(peaks.free)} GiB free, a budget of {_gib(peaks.budget)} GiB",
        f"frames padded to {width}x{height}, {peaks.tokens:,} tokens a latent",
        f"{_count(len(shots), 'shot')}, the longest of {_count(job.frames, 'frame')}"
        f" ({_count(job.lengths[-1], 'latent')})",
        f"{cap}: {whole} of {_count(len(shots), 'shot')} in one window, {_count(joins, 'join')}",
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
    left = _left(plan, "the DiT's other parameters", "the VAE")
    if left:
        pinned = f"{_gib(_pinned(plan))} GiB of system memory pinned for {left}"
    else:
        pinned = "no system memory pinned, nothing leaving the GPU"
    lines.append(f"{pinned}; {_gib(peaks.host)} GiB of system memory in all")
    lines.append(
        f"encode {_pass_said(job, plan.encode_tiles)}, decode"
        f" {_pass_said(job, plan.decode_tiles)}, each held {_gib(VAE_ROOM)} GiB under the budget"
    )
    names = {"encode": "encode", "dit": "DiT's phase", "decode": "decode"}
    lines.append(
        f"predicted peaks: encode {_gib(peaks.encode)} GiB, DiT {_gib(peaks.dit)} GiB, decode"
        f" {_gib(peaks.decode)} GiB; the {names[peaks.bound]} bounds the plan,"
        f" {'' if peaks.bound == 'dit' else 'its room counted, '}"
        f"{_gib(peaks.budget - peaks.need(peaks.bound))} GiB under the budget"
    )
    if peaks.scratch:
        where = (
            "in system memory"
            if peaks.scratch <= SCRATCH_ALLOWANCE
            else f"in a memory-mapped file, over the {_gib(SCRATCH_ALLOWANCE)} GiB a scratch may"
            " take of system memory"
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
    pinned, --swap-blocks when none does.

    PROVISIONAL (implementation, 2026-10-11), a fourth: the planner's own cap, and the one a
    refused --window is told, is the longest window the job really runs under the largest cap
    that fits (_own_cap), not that cap, which a card may hold no window of: a shot of 38 latents
    on a 24 GB card at 1080p runs two windows of 20 under every cap from 20 to 37, and its plan
    says 20. A --window pinned stays the cap as given."""
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
        cap = _own_cap(job, longest)
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
            f" {_under(job.free)} GiB free: the longest window that fits is --window"
            f" {_own_cap(job, longest)}"
        )
    latents = _longest(job.lengths, cap)
    if pins.swap_blocks is not None:
        return cap, most, dit_peak(job.padded, latents, most) <= job.budget
    fewest = _fewest(job, latents)
    assert fewest is not None
    return cap, *fewest


def _own_cap(job: _Job, cap: int) -> int:
    """The cap the planner takes for a job that fits under `cap`: the longest window the job
    really runs under it (_longest), the longest shot's length when every shot is one window. It
    lays every shot out as `cap` does: a shot's balanced windows under a cap are its windows
    under any smaller cap that is still at least the longest of them, as many of that length
    still covering the shot (tests/test_planner.py holds it on shot.shot_layout). 2 * SHARED + 1
    at least when a shot is split: a shot of 6 latents under a cap of 5 runs two windows of 4,
    and no layout exists for a cap that short with a longer shot (check_recorded)."""
    latents = _longest(job.lengths, cap)
    return latents if latents == job.lengths[-1] else max(latents, 2 * SHARED + 1)


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
    being the untiled pass, or from TILE_MAX on a frame longer than it, down to TILE_MIN; none
    for a frame that small."""
    largest = min(TILE_MAX, (max(padded) - 1) // TILE_STEP * TILE_STEP)
    return range(largest, TILE_MIN - 1, -TILE_STEP)


def _vae_peak(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """The predicted peak of the job's encode or decode (encode_peak, decode_peak)."""
    if phase == "encode":
        return encode_peak(job.padded, tiles, resident, swapped)
    return decode_peak(job.padded, job.output, tiles, resident, swapped, job.split)


def _vae_need(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """What the planner judges the job's encode or decode by against the budget, in bytes: its
    predicted peak and the room every VAE pass is held to (VAE_ROOM), wherever it decides
    whether a pass fits and what of the DiT stays there, as _unfit judges the plan made. The
    plan records the peak alone (_peaks)."""
    return _vae_peak(job, phase, tiles, resident, swapped) + VAE_ROOM


def _fitting(job: _Job, phase: Phase, swapped: int) -> list[Tiles | None]:
    """The passes of the job's encode or decode that fit the budget with the DiT away, in the
    planner's order: untiled, then tiled from the largest tiles down, each with its room
    (_vae_need). `swapped` changes nothing to it: a phase that keeps nothing of the DiT frees
    its slots too (dit_on_gpu)."""
    passes = (None, *(Tiles(size) for size in _tile_sizes(job.padded)))
    return [tiles for tiles in passes if _vae_need(job, phase, tiles, 0, swapped) <= job.budget]


def _pass(job: _Job, phase: Phase, pin: int | None, swapped: int) -> Tiles | None:
    """The tiles of the job's encode or decode (plan's choice 3), None untiled; --encode-tile or
    --decode-tile, `pin`, taken exactly or refused, the refusal naming the largest tile that
    fits. The job's floor holds already (_floor), so some pass fits: its smallest tiles at least,
    or the untiled pass on a frame too small for a tile (_least_pass).

    PROVISIONAL (implementation, 2026-10-11): a tile pinned that covers the whole padded frame
    is taken as it is. The plan records the tiles, a setting a resume must find again; the pass
    runs untiled, as the tiled passes run a grid of one tile (covers), its peak the untiled fit
    (_fit), no scratch. DESIGN.md has a pin taken exactly or refused, and doesn't say that one
    is refused for another that runs the same, 0, as the planner refused it before the tiled
    passes were built.

    PROVISIONAL (implementation, 2026-10-11), two more. A tile pinned above TILE_MAX is taken
    exactly when it fits by its fit, outside the fits' range as it is: a pin is the user's, and
    the search alone never chooses one, nor does a refusal name one. And a pass pinned, tiled
    or not, is held to the room the planner's own is (VAE_ROOM), and refused without it."""
    fitting = _fitting(job, phase, swapped)
    if pin is None:
        return fitting[0]
    option = f"--{phase}-tile {pin}"
    tiles = Tiles(pin) if pin else None
    need = _vae_need(job, phase, tiles, 0, swapped) + MARGIN
    if need <= job.free:
        return tiles
    # A frame too small for a tile has the untiled pass alone, whatever is pinned, which the
    # floor holds: a pass refused has tiles, the smallest of which fit.
    largest = next(fit.size for fit in fitting if fit is not None)
    raise PlanError(
        f"{option}: the {phase} {_tiles_said(job, tiles)} takes {_over(need)} GiB of free VRAM,"
        f" the DiT off the GPU{_ROOM_HELD}, and the driver reports {_under(job.free)} GiB"
        f" free: the largest tile that fits is --{phase}-tile {largest}"
    )


# What a refusal adds after the free VRAM a VAE pass takes, its room in it (VAE_ROOM).
_ROOM_HELD = f", {VAE_ROOM / GIB:.2f} of them the room a VAE pass is held to"


def _resident(job: _Job, phase: Phase, tiles: Tiles | None, swapped: int) -> int | None:
    """What of the DiT is left on the GPU in the job's encode or decode, in `tiles`
    (Plan.resident_encode, Plan.resident_decode; dit_on_gpu): None, the DiT staying as its own
    phase has it, when that fits, nothing then moving between the phases; else as many of the
    blocks that aren't swapped as fit, with its other parameters, from all of them down, the
    slots freed (with no block swapped, all of them is what None has there, which didn't fit);
    0, nothing of it, when one block and the other parameters don't fit. What fits leaves the
    pass its room (_vae_need).

    The most that fits is the least to bring back for the DiT's phase, each block one copy from
    its pinned memory, the whole DiT 0.29 s (milestone 3, 2026-10-11), and the fewest blocks
    pinned (_leaving)."""
    if _vae_need(job, phase, tiles, None, swapped) <= job.budget:
        return None
    for resident in range(BLOCKS - swapped, 0, -1):
        if _vae_need(job, phase, tiles, resident, swapped) <= job.budget:
            return resident
    return 0


def _floor(job: _Job) -> tuple[Phase, int] | None:
    """The phase that takes the most where every phase takes its least, and what it takes there,
    when that is over the budget, no plan then fitting the job (DESIGN.md, Memory planner: a size
    that doesn't fit is refused, never degraded); else None. The least: the smallest window the
    job can run, every block of the DiT swapped and the VAE's weights away; the encode and the
    decode each in its smallest tiles, untiled on a frame too small for one (_least_pass), the
    DiT away, whatever the plan swaps: a phase that keeps nothing of it frees its slots too
    (dit_on_gpu). What a phase takes is its peak, and for the encode and the decode their room
    with it (_vae_need)."""
    least: dict[Phase, int] = {
        "encode": _vae_need(job, "encode", _least_pass(job), 0, BLOCKS),
        "dit": dit_peak(job.padded, _longest(job.lengths, job.smallest), BLOCKS, vae_in_dit=False),
        "decode": _vae_need(job, "decode", _least_pass(job), 0, BLOCKS),
    }
    phase = max(PHASES, key=lambda phase: least[phase])
    return (phase, least[phase]) if least[phase] > job.budget else None


def _least_pass(job: _Job) -> Tiles | None:
    """The pass of the job's encode, or of its decode, that takes the least: its smallest tiles;
    None, the untiled pass, on a frame too small for a tile. An untiled pass never takes less
    than a tile of its frame below REFITTED pixels: its fit is at least those tiles' at the
    frame's area, which a tile's is under (_fit; tests/test_planner.py holds it over many
    sizes), the room the same for both (VAE_ROOM)."""
    sizes = _tile_sizes(job.padded)
    return Tiles(sizes[-1]) if sizes else None


def _too_large(job: _Job, phase: Phase, least: int) -> str:
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
        how = (
            f"its smallest window, {_count(latents, 'latent')}, with every block of the DiT"
            " swapped and the VAE off the GPU"
        )
        room = 0
    else:
        tiles = _least_pass(job)
        tiled = "untiled, too small for a tile" if tiles is None else _tiles_said(job, tiles)
        how = f"its {phase} {tiled}, with the DiT off the GPU"
        room = VAE_ROOM
    # The room as what is left of the total once the peak and the margin are said, each rounded
    # as the message has it, so that its figures add up.
    total, peak = _hundredths(least + MARGIN, up=True), _hundredths(least - room, up=True)
    rest = (total - peak - _hundredths(MARGIN, up=True)) / 100
    held = f", {rest:.2f} of room a VAE pass is held to" if room else ""
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
        f"an output of {width}x{height} takes {total / 100:.2f} GiB of free VRAM ({how}:"
        f" a peak of {peak / 100:.2f} GiB{held}, and {_gib(MARGIN)} of margin), and the"
        f" driver reports {_under(job.free)} GiB free: {fitting}"
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
    """The pinned system memory of the plan, in bytes: a copy of each part of the weights that
    leaves the GPU in some phase, of its own size rounded up to a page (blockswap.HostCopy;
    DESIGN.md, BlockSwap, rewritten), as the loader makes them (model.load_models: the parts of
    blockswap.dit_parts, and the VAE): each block of the DiT that is swapped or that a VAE phase
    doesn't keep; the DiT's other parameters when a VAE phase keeps nothing of it; the VAE's
    weights when they leave the DiT's phase. 15.19 GiB for the 36 blocks, 15.35 with the other
    parameters, 15.81 with the VAE (measured, milestone 3, 2026-10-11)."""
    parts = [BLOCK] * _leaving(plan)
    parts += [DIT_OTHER] if 0 in _residents(plan) else []
    parts += [] if plan.vae_in_dit else [VAE_WEIGHTS]
    return sum(-(-part // mmap.PAGESIZE) * mmap.PAGESIZE for part in parts)


def _left(plan: Plan, other: str, vae: str) -> str:
    """What of the weights leaves the GPU in some phase of the plan, pinned (_pinned), in words:
    the DiT's blocks counted, its other parameters and the VAE named as given; empty when
    nothing leaves."""
    leaving = _leaving(plan)
    left = [_count(leaving, "block")] if leaving else []
    left += [other] if 0 in _residents(plan) else []
    left += [] if plan.vae_in_dit else [vae]
    return " and ".join(filter(None, (", ".join(left[:-1]), *left[-1:])))


def _peaks(job: _Job, plan: Plan) -> Peaks:
    """What the plan is predicted to take for the job.

    Its system memory: the process's own during the shots, the pinned copies of what leaves the
    GPU, and a tiled shot's scratch up to its allowance. PROVISIONAL (implementation,
    2026-10-10): the loader's own peak while the weights load, 16.5 GiB, mostly the weights' file
    mapped (DESIGN.md, Memory planner, Host RAM), isn't counted. PROVISIONAL (implementation,
    2026-10-11): a scratch over its allowance goes whole to its memory-mapped file
    (tiling.Scratch), and is still counted for the allowance, as DESIGN.md counts it, for the
    page cache its mapping works in. Nor is what a tiled encode sums beside its scratch
    counted, its tiles' moments on the CPU (tiling.encode_tiled): a byte per pixel of the
    padded frame and per latent of the shot, a twenty-fourth of the scratch, 2.0 GiB for 1,001
    frames at 4K.

    The scratch is one pass's, the encode's and the decode's being the same size and never open
    together; none for a pass whose pinned tile covers the frame (covers)."""
    latents = _longest(job.lengths, plan.window)
    passes = (plan.encode_tiles, plan.decode_tiles)
    tiled = any(tiles is not None and not covers(job.padded, tiles) for tiles in passes)
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
    """Why the plan doesn't fit the job's memory (fits): the first phase over the budget, a VAE
    phase judged with its room as the planner chose it (Peaks.need), or the system memory it
    takes over what is available; None when it fits."""
    peaks = _peaks(job, plan)
    blocks = _count(plan.swapped, "block")
    vae = "" if plan.vae_in_dit else " and the VAE off the GPU"
    whats = {
        "encode": f"the encode, {_tiles_said(job, plan.encode_tiles)}, with"
        f" {_resident_said(plan.resident_encode, plan.swapped)},",
        "dit": f"the DiT's phase, a window of {_count(peaks.window, 'latent')} with {blocks}"
        f" swapped{vae},",
        "decode": f"the decode, {_tiles_said(job, plan.decode_tiles)}, with"
        f" {_resident_said(plan.resident_decode, plan.swapped)},",
    }
    for phase in PHASES:
        need = peaks.need(phase) + MARGIN
        if need > job.free:
            held = "" if phase == "dit" else f"{_ROOM_HELD},"
            return (
                f"{whats[phase]} takes {_over(need)} GiB of free VRAM{held} and the driver reports"
                f" {_under(job.free)} GiB free, {_short(need, job.free)} GiB short"
            )
    if host_available is not None and peaks.host > host_available:
        pinned = _pinned(plan)
        parts = [(HOST_RESIDENT, "for the process")]
        if pinned:
            left = _left(plan, "the DiT's other parameters", "the VAE's weights")
            parts.append((pinned, f"pinned for {left} that leave the GPU"))
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


def _tiles_said(job: _Job, tiles: Tiles | None) -> str:
    """A pass of the job as a refusal says it: untiled, in its tiles, or untiled under a tile
    that covers its frames (covers)."""
    if tiles is None:
        return "untiled"
    if covers(job.padded, tiles):
        return f"untiled (one tile of {tiles.size} px covers the padded frame)"
    return f"in tiles of {tiles.size} px"


def _pass_said(job: _Job, tiles: Tiles | None) -> str:
    """A pass as the plan's lines say it: untiled, or its tiles and their grid, rows x columns;
    a grid of one tile is the untiled pass (covers)."""
    if tiles is None:
        return "untiled"
    rows, columns = grid(job.padded, tiles)
    untiled = ", which is the untiled pass" if covers(job.padded, tiles) else ""
    return f"in tiles of {tiles.size} px, {rows} x {columns}{untiled}"


def _resident_said(resident: int | None, swapped: int) -> str:
    """What of the DiT a VAE phase has on the GPU, in words, as dit_on_gpu counts it: nothing; a
    number of its blocks, with its other parameters; or, nothing moving, what its own phase
    has, the blocks that aren't swapped and the slots."""
    stay = BLOCKS - swapped
    if resident == 0:
        return "the DiT off the GPU"
    if resident is not None:
        return f"{resident} of the DiT's blocks on the GPU"
    if not swapped:
        return f"the DiT's {BLOCKS} blocks on the GPU"
    slots = _count(min(swapped, SLOTS), "slot")
    if not stay:
        return f"the DiT's {slots} on the GPU"
    return f"the DiT's {_count(stay, 'unswapped block')} and its {slots} on the GPU"


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
