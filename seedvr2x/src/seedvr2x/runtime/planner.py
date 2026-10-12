"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model: what each phase is predicted to peak at on the GPU, in bytes. The DiT's and
the VAE's passes by the fits of research/docs/planner-limits.md and research/docs/vram.md,
measured on numz's CLI with the 7B fp16, the one DiT seedvr2x runs (weights.ACCEPTED); a square
tile by the peaks measured on seedvr2x's own tiled passes (MEASURED), the fits serving the tiles
cut off at the picture's edges; and what a VAE pass keeps on the GPU of the shot itself, which
grows with it, counted for the job's longest shot. The DiT's phase keeps nothing of a shot
beside the window it runs (shot.sample_windows); its other moment is the draw of a shot's noise,
the shot's latent and noise whole on the GPU before its first window, counted by the two
tensors' sizes (dit_draw), and the phase takes the larger of the two. Where the weights are in
each phase, and what that takes, by what runtime/blockswap.py does with them, measured on
seedvr2x itself (milestone 3). The tiles and their scratch by runtime/tiling.py's, and what the
scratch may take of system memory from the host's (Plan.scratch_allowance). Sizes are (height,
width) in pixels; `padded` is the frame the model takes, the target padded (job.padded_size).

Its choices, in DESIGN.md's order (plan): the window, the biggest quality lever; the fewest
swapped blocks that hold it; the largest VAE tiles, none where the untiled pass fits. A pin is
taken exactly or refused, the refusal naming what fits; a size that doesn't fit is refused, never
degraded (PlanError).

Four corrections come from the planner's first two matrices on seedvr2x, emulated cards
(tests/test_planner.py, test_first_matrix). Two are DESIGN.md's: no tile chosen above TILE_MAX
(Memory planner, What follows from them), and every VAE pass held to a room under the budget
besides its peak (VAE_ROOM), provisional there until its cause is found (Memory planner, Budget;
To measure, The VAE's room). PROVISIONAL (implementation, 2026-10-11), the two others:
ENCODE_EXTRA on every encode, until the encode is refitted on seedvr2x, and split's term off a
tiled decode's peak (decode_peak)."""

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
MIB = 1024**2

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
# The constants stay numz's fit's (DESIGN.md, Memory planner, Per-phase peaks). seedvr2x's own
# peak, one window of 1 latent at 1080p, no block swapped, measured 17.016 GiB (milestone 3,
# 2026-10-11; tests/test_placement.py measures it), where this model gives 16.955: 0.061 GiB
# under, of the margin's 0.6; and on nine emulated cards and the bare one, 5 to 76 latents, 0 to
# 36 blocks swapped, the DiT's phase came within 0.07 GiB of its prediction (DESIGN.md, Memory
# planner, Budget).
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
# A square tile's measured peaks, one the frame doesn't cut off, by its side in pixels: the
# encode's and the decode's, in GiB above what is allocated before the call, on seedvr2x's own
# tiled passes over 21 frames (2026-10-11; DESIGN.md, Memory planner, Per-phase peaks). The fits
# hold them within 0.3 GiB from 1024 px up, and fall up to 0.43 short below (the encode at 768
# px), most of the margin: DESIGN.md has the planner take the measured peak of every size it may
# choose. PROVISIONAL (implementation, 2026-10-11): five of those 25 sizes; until the table has
# them all, a square between two of these takes the larger of its fit and of the straight line
# between them, by area (_fit).
MEASURED = {
    512: (3.52, 5.89),
    768: (7.08, 11.11),
    1024: (10.47, 17.46),
    1536: (21.50, 38.73),
    2048: (37.76, 68.51),
}

# What a VAE pass keeps on the GPU of the shot itself, which grows with it, each in MiB per
# megapixel of the padded frame and per latent of the shot, as measured on seedvr2x's own passes
# (2026-10-11), and counted for the job's longest shot (DESIGN.md, Memory planner, Per-phase
# peaks: "A pass's peak mustn't grow with the shot ... Each is being taken off the GPU; until
# then the planner counts them"). 1080p is padded to 1104x1920, 2.12 megapixels, and 4K to
# 2192x3840, 8.42.
# A decode, tiled or not, keeps the shot's latent and a copy of it: 2.02 MiB a latent at 1080p,
# 8.03 at 4K. PROVISIONAL (implementation, 2026-10-11), until the decode no longer keeps them on
# the GPU.
DECODE_LATENT = 0.954
# A tiled encode sends its tiles' sums there at its end: 4 MiB a latent at 1080p, 16 at 4K.
# PROVISIONAL (implementation, 2026-10-11), until the tiled encode no longer sends them there.
ENCODE_TILED_SUMS = 1.9
# An untiled encode holds its moments through the pass: 6 MiB a latent at 1080p at its end.
# PROVISIONAL (implementation, 2026-10-11), until the untiled encode no longer keeps them on the
# GPU.
ENCODE_UNTILED_MOMENTS = 2.83

# A latent as the VAE makes it and the DiT takes it: its channels, the pixels of the padded
# frame along a side of one of its positions (vendor/models/video_vae_v3/
# s8_c16_t4_inflation_sd3.yaml: latent_channels 16, spatial_downsample_factor 8), and the bytes
# of one of its values, in the pipeline's bfloat16 (model.COMPUTE_DTYPE; tests/test_planner.py
# holds the three to tensors of that shape). What dit_draw counts a shot's latent and its noise
# by, their sizes: literals here since runtime/model.py imports torch.
LATENT_CHANNELS = 16
LATENT_SIDE = 8
LATENT_VALUE = 2

# What seedvr2x's encode holds over its fit, tiled or not, in bytes: the largest excess of a
# measured peak over the fit's in the planner's first matrix (the 96 GB card emulating smaller
# ones, 2026-10-11): at 1080p, 9.66 GiB measured where the fit gave 9.546 in tiles of 896 px
# (+0.11), and 13.91 for 13.511 in tiles of 1216 px (+0.40); untiled at 540p, 22.53 for 22.22
# (+0.31). At 4K the refit is above the measurement: 29.53 for 29.694 in tiles of 1792 px
# (-0.16), 21.26 for 21.648 at 1472 px (-0.39). PROVISIONAL (implementation, 2026-10-11): one
# constant on every encode, until the encode is refitted on seedvr2x, every tile size measured
# (MEASURED).
ENCODE_EXTRA = round(0.4 * GIB)
# The room every VAE pass, encode or decode, tiled or not, must leave under the budget besides
# its predicted peak, in bytes: 2.1 GiB under the free memory with the margin. Over the
# planner's two matrices on emulated cards (2026-10-11), a VAE pass whose peak comes close to
# the free memory runs out of memory in memory_limit_conv's torch.cat: the device full while
# torch holds 3.3 to 4.9 GiB less than the free memory it started with, at the pass's largest
# request, 2.0 to 4.8 GiB. Every pass that had 1.67 GiB or more between its measured peak and
# the free memory ran (1.67, 1.97, 3.53, 3.69, 3.8, 4.98, 9.97, 13.8, 25.6); each of the six
# failures had 1.6 GiB or less between its predicted, or true, peak and the free memory (0.65,
# 0.80, 0.86, 0.94, 1.09, 1.6); passes also ran with 0.63, 0.88, 1.01, 1.28 and 1.32: the band
# between isn't safe. The first matrix's tiled decodes passed only for split's term, which
# padded their predicted peaks then (decode_peak). PROVISIONAL, as DESIGN.md has it (Memory
# planner, Budget: every VAE pass held 1.5 GiB further under the budget until the cause is
# found; To measure, The VAE's room, its study): a guard, the cause isn't known; numz's margin,
# 0.6 GiB, was bisected on passes whose largest request was 0.4 GiB (MARGIN).
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
# the planner's first matrix, 2026-10-11). The bound is the largest tile measured (DESIGN.md,
# Memory planner, What follows from them: "The planner chooses none above 2048 px, the largest
# measured"); a pin above it is taken by its fit, and warned of (_pass, beyond_measured).
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
# The most a scratch may take of system memory when what the host has available can't be read
# (cli._host_available), in bytes. PROVISIONAL (implementation, 2026-10-11): DESIGN.md sets the
# allowance from the host's memory, and gives no figure without it: 8 GiB, which holds 675 frames
# at 1080p and 170 at 4K, what every scratch was allowed before the planner set it.
ALLOWANCE_UNKNOWN_HOST = 8 * GIB


class PlanError(JobError):
    """A job the planner refuses before the run: a pin it can't take exactly, or a size that
    doesn't fit the memory free. Its message counts in free VRAM, never in card sizes, and gives
    what DESIGN.md has each refusal give (Memory planner: A size that doesn't fit is refused,
    Pins, A plan shapes the output a little, Host RAM), in GiB. Of the GPU's memory, what is
    taken is rounded up and what is there rounded down; of the system's, each part is said to
    the nearest hundredth, as the plan's lines say it, and the total is their sum
    (_host_hundredths)."""


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


def dit_draw(
    padded: tuple[int, int], latents: int, swapped: int = 0, vae_in_dit: bool = True
) -> int:
    """What the GPU holds when the noise of a shot of `latents` latents of padded frames is
    drawn, in bytes: the other moment of the DiT's phase, beside a window's peak (dit_peak),
    once per shot, before its first window, and again when a resumed run takes the shot up
    (shot.sample_windows). The shot's latent and its noise are then whole on the GPU, where
    the noise is drawn so that its values are the device's own; both wait in system memory
    through the windows, of which the GPU holds one at a time.

    The two tensors by their sizes, not by a fit: each (latents, H / 8, W / 8, 16) in bfloat16,
    2 x 16 channels x 2 bytes over the 64 padded pixels of a latent position: 1 byte a padded
    pixel and a latent, 2.02 MiB a latent at 1080p, padded to 1104x1920, and 8.03 at 4K, padded
    to 2192x3840.

    With them, the weights the DiT's phase has on the GPU, entered by then (run._sample enters
    it before the first window is asked for): what of the DiT stays as its own phase has it
    (dit_on_gpu's None): the blocks that aren't swapped, its other parameters and the slots,
    which blockswap.Placement.enter allocates as the phase is entered, before any forward, the
    first swapped blocks on their way into them; and the VAE's weights unless vae_in_dit is
    False, model.Models.enter sending them away before the DiT's come. An upper bound by what
    dit_on_gpu counts of the tables the DiT's blocks cache once a window has run (DIT_BUFFERS,
    0.03 GiB of its 0.058), not there yet at a run's first draw."""
    height, width = padded
    position = (height // LATENT_SIDE) * (width // LATENT_SIDE)
    shot = 2 * LATENT_CHANNELS * LATENT_VALUE * position * latents
    return dit_on_gpu(None, swapped) + (VAE_WEIGHTS if vae_in_dit else 0) + shot


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
    *,
    latents: int = 0,
) -> int:
    """The predicted peak of the VAE's encode of a shot of `latents` latents of padded frames, in
    bytes: its peak untiled (tiles None) or that of its largest tile (_fit), above what is on the
    GPU before the call, the VAE's weights and what of the DiT stays there (dit_on_gpu's resident
    and swapped); what seedvr2x's encode holds over it (ENCODE_EXTRA); and what it keeps there of
    the shot, which grows with it: an untiled encode's moments, a tiled one's sums
    (ENCODE_UNTILED_MOMENTS, ENCODE_TILED_SUMS), none of it with latents left at 0. Not the
    shot's frames: seedvr2x feeds the encode slice by slice, where numz moves a batch's there
    first, 0.046 GiB per 4K frame (DESIGN.md, Memory planner, Per-phase peaks). Nor the room
    every VAE pass is held to (VAE_ROOM), which the planner judges it by and no peak holds.

    PROVISIONAL (implementation, 2026-10-11), until the tiled encode no longer sends its sums to
    the GPU (ENCODE_TILED_SUMS): they are added to its tile's peak, as an upper bound. They go
    to the GPU at its end, once its last tile is done, and on a shot of 301 frames at 540p the
    tiled encode's peak equalled a 21-frame shot's (DESIGN.md): the larger of the two may be all
    it takes.

    ValueError for tiles of which one covers the whole padded frame, which is no tiling
    (covers)."""
    kept = ENCODE_UNTILED_MOMENTS if tiles is None else ENCODE_TILED_SUMS
    above = _fit(padded, tiles, 0) + kept * MIB * _megapixels(padded) * latents
    return round(above) + ENCODE_EXTRA + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
    *,
    latents: int = 0,
) -> int:
    """The predicted peak of the VAE's decode of a shot of `latents` latents of padded frames, in
    bytes, its peak above what is on the GPU before the call as encode_peak has the encode's
    (without ENCODE_EXTRA, the encode's), and the shot's latent with its copy, there through the
    pass, tiled or not (DECODE_LATENT; none of it with latents left at 0); with split, the colour
    correction too, which corrects a slice of the decode at once, SLICE frames at most of the
    output's size (job.output_size). ValueError for tiles of which one covers the whole padded
    frame, as encode_peak.

    The untiled decode, streamed, corrects each slice while it holds its working memory, and its
    peak is the two together: at 1080p with every weight on the GPU, 50.98 GiB measured for
    51.50 predicted, 50.80 without the correction's term (the 96 GB card, 2026-10-11). A tiled
    decode corrects its frames after its last tile, when the tile's working memory is gone, and
    its peak is the larger of the two, the tile's in practice. PROVISIONAL (implementation,
    2026-10-11; DESIGN.md counts the correction in the decode's phase, and doesn't say that a
    tiled decode's peak leaves it out), by the planner's first matrix (emulated cards,
    2026-10-11), where the tiled decodes' peaks were under the sum by the correction's term or
    about: at 4K, where it is 2.80 GiB, by 2.85 and 2.84 (26.85 GiB measured for 29.695 in tiles
    of 1216 px, 19.01 for 21.845 at 1024 px); at 1080p, where it is 0.70, by 0.75 in tiles of
    832 px (12.87 for 13.624) and by 0.57 at 640 px (9.22 for 9.792), whose prediction by the
    fit alone was then 0.13 GiB under its measured peak, the small tiles' fit that much short
    there: a tile of 640 px takes the line between the measured 512 and 768 px, 0.25 GiB over
    its fit (_fit; tests/test_planner.py, test_first_matrix)."""
    fit = _fit(padded, tiles, 1)
    corrected = SPLIT_FRAME_4K * SLICE * output[0] * output[1] / PIXELS_4K if split else 0
    above = fit + corrected if tiles is None else max(fit, corrected)
    kept = DECODE_LATENT * MIB * _megapixels(padded) * latents
    return round(above + kept) + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


# The VAE's fits by MEASURED's columns, the encode's then the decode's: each untiled, of a tile
# from REFITTED pixels, and of one below.
_FITS = (
    (ENCODE_UNTILED, ENCODE_TILED, ENCODE_TILED_SMALL),
    (DECODE_UNTILED, DECODE_TILED, DECODE_TILED_SMALL),
)


def _megapixels(padded: tuple[int, int]) -> float:
    """The padded frame's area, in megapixels (10^6 pixels)."""
    return padded[0] * padded[1] / 1e6


def _fit(padded: tuple[int, int], tiles: Tiles | None, column: int) -> float:
    """A VAE pass's peak above what is on the GPU before the call, in bytes: the encode's
    (column 0) or the decode's (1), untiled (tiles None) or of its largest tile.

    A tile takes (DESIGN.md, Memory planner, Per-phase peaks: "the planner takes the measured
    peak of every size it may choose, the fits serving the tiles cut off at the picture's
    edges"):
    - square, its measured peak when MEASURED has its side;
    - else, square or cut off at the picture's edges (a 1280-px tile at 1080p is 1104x1280), the
      larger of its fit at its area (tile_area) and of the straight line between the two
      measured squares its area lies between, by area (_line); of an area outside them, its fit.
      The cut-off 1280-px tile takes its fit, 13.60 and 23.46 GiB, where 13.55 and 23.34 were
      measured; a 768-px tile cut off at 720p's 752 rows the line, 6.95 and 10.91 GiB, where its
      fit, the small tiles', gives 6.55 and 10.61 and the square measured 7.08 and 11.11.
      PROVISIONAL (implementation, 2026-10-11), until MEASURED has every size the planner may
      choose.
    ValueError for tiles of which one covers the whole padded frame: no tiling, and no pass the
    planner makes or takes (covers).

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
    untiled, tiled, small = _FITS[column]
    if tiles is None:
        area = _megapixels(padded)
        return max(constant + slope * area for constant, slope in (untiled, small)) * GIB
    if covers(padded, tiles):
        raise ValueError(
            f"tiles of {tiles.size} px over frames padded to {padded[1]}x{padded[0]}: one covers"
            " the whole frame, which is no tiling"
        )
    side, area = tiles.size, tile_area(padded, tiles)
    if side <= min(padded) and side in MEASURED:  # a square, measured
        return MEASURED[side][column] * GIB
    constant, slope = tiled if side >= REFITTED else small
    return max(constant + slope * area, _line(area, column)) * GIB


def _line(area: float, column: int) -> float:
    """The straight line between the peaks of the two measured squares (MEASURED) a tile's area,
    in megapixels, lies between, at that area, in GiB: the encode's (column 0) or the decode's
    (1); 0, under any fit, for an area outside the squares measured."""
    pixels = area * 1e6
    below = max((side for side in MEASURED if side**2 <= pixels), default=None)
    above = min((side for side in MEASURED if side**2 > pixels), default=None)
    if below is None or above is None:
        return 0.0
    share = (pixels - below**2) / (above**2 - below**2)
    return MEASURED[below][column] + share * (MEASURED[above][column] - MEASURED[below][column])


def covers(padded: tuple[int, int], tiles: Tiles) -> bool:
    """Whether one tile covers the whole padded frame, its grid one tile, which is no tiling
    (DESIGN.md, Pipeline, per shot, step 3): a pin of one is refused, naming 0, the untiled pass
    (plan), a record of one is no plan's (check_recorded), and the planner never makes one
    (_tile_sizes). The tiled passes would run such a grid untiled (tiling.encode_tiled,
    tiling.decode_tiled, as numz's take it, attn_video_vae.py:1315-1317, 1489-1491)."""
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
    dit: int  # the larger of the longest window's and of the longest shot's draw (dit_draw)
    decode: int
    host: int  # the system memory the plan takes, the scratch up to its allowance
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
        return _padded(self.target, self.numz_padding)

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
    decode corrects colours; numz_padding is for tests only (cli.NUMZ_PADDING). An output
    directory and one file plan alike: a shot's units are off the GPU in both (runtime/units.py).

    The DiT's phase is judged by the larger of two moments, never their sum: the job's longest
    window (dit_peak) and its longest shot's draw of noise, the shot's latent and noise whole on
    the GPU before its first window (dit_draw; _dit_need). A block is swapped, or the VAE sent
    away, for the draw as for a window; a job whose longest shot's draw doesn't fit with every
    block swapped and the VAE away is refused, as a size that doesn't fit (_floor).

    A tile pinned that covers the whole padded frame is no tiling, and is refused, naming 0, the
    untiled pass (DESIGN.md, Pipeline, per shot, step 3), whatever the memory.

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
    Then what a tiled shot's scratch may take of system memory, from host_available
    (_allowance).

    recorded is the plan of a job resumed that has made units: what shapes the output is taken
    from it as it is, the cap, the tiles and the compile flags, and where the weights are and
    the scratch's allowance planned again around it (DESIGN.md, Memory planner, A plan shapes
    the output a little); one with tiles of which one covers the frame is no plan's, refused
    (check_recorded)."""
    job = _job(free, target, shots, split, numz_padding)
    if pins.swap_blocks is not None and pins.swap_blocks > BLOCKS:
        raise PlanError(
            f"--swap-blocks {pins.swap_blocks}: the DiT has {BLOCKS} blocks, so from 0 to {BLOCKS}"
        )
    for phase, pin in (("encode", pins.encode_tile), ("decode", pins.decode_tile)):
        if pin and covers(job.padded, Tiles(pin)):
            height, width = job.padded
            raise PlanError(
                f"--{phase}-tile {pin}: a tile of {pin} px covers the whole padded frame,"
                f" {width}x{height}: --{phase}-tile 0 pins an untiled {phase}"
            )
    if recorded is None:
        made = _chosen(job, pins)
    else:
        _check_tiles(job.padded, recorded)
        made = _around(job, recorded, pins)
    made = replace(made, scratch_allowance=_allowance(job, made, host_available))
    refused = _unfit(job, made, host_available)
    if refused is not None and recorded is not None:
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
    with its room (VAE_ROOM), the DiT's by the larger of its longest window and of its longest
    shot's draw (dit_draw), and the plan host_available bytes of system memory (None: not
    checked), its scratch's allowance as the plan has it, for a job as plan() takes it; else one
    sentence naming the first phase that doesn't and what it is short of, in GiB. ValueError for
    a plan with tiles of which one covers the whole padded frame, which plan() never returns
    (covers)."""
    job = _job(free, target, shots, split, numz_padding)
    return _unfit(job, plan, host_available)


def check_recorded(
    recorded: Plan,
    shots: Sequence[Shot],
    target: tuple[int, int],
    *,
    numz_padding: bool = False,
) -> None:
    """Refuse (PlanError) a recorded plan that the planner makes for no job of `shots` resized to
    target (job.target_size; numz_padding as plan takes it), which Plan.recorded, with neither,
    can't tell from one it does:
    - a cap no layout exists for (shot.window_layout): one under 2 * SHARED + 1 latents that a
      shot is longer than. The planner makes a cap that short only for a job whose longest shot
      is as short, each shot then one window (_Job.smallest), so that record was edited;
    - tiles of which one covers the whole padded frame, which is no tiling (covers): the planner
      makes none, and refuses the pin of one (plan)."""
    longest = max(_latents(shot.frames) for shot in shots)
    if recorded.window is not None and recorded.window < min(2 * SHARED + 1, longest):
        raise PlanError(
            f"not a plan as seedvr2x records one: its window, {recorded.window}, with a shot of"
            f" {_count(longest, 'latent')}: a cap that a shot is longer than is"
            f" {2 * SHARED + 1} latents at least"
        )
    _check_tiles(_padded(target, numz_padding), recorded)


def _check_tiles(padded: tuple[int, int], recorded: Plan) -> None:
    """Refuse (PlanError) a recorded plan with tiles of which one covers the whole padded frame
    (check_recorded)."""
    height, width = padded
    passes = (("encode", recorded.encode_tiles), ("decode", recorded.decode_tiles))
    for phase, tiles in passes:
        if tiles is not None and covers(padded, tiles):
            raise PlanError(
                f"not a plan as seedvr2x records one: its {phase}_tiles, of {tiles.size} px, with"
                f" frames padded to {width}x{height}: a tile that covers the whole frame is no"
                f" tiling, and an untiled {phase} records none"
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
    counted with its room (Peaks.bound) and said so, as the DiT's is where its peak is its
    longest shot's draw of noise, not a window's (dit_draw); the scratch of the longest tiled
    shot, what a scratch may take of system memory (Plan.scratch_allowance) and the shots whose
    scratch goes to its file instead; what was pinned. DESIGN.md's list but for the times, whose
    lines come with the time estimates. The longest window the job runs is said only where it
    isn't the cap, one pinned or recorded above it. The system memory is said as a refusal for
    it says it (_host_hundredths): the pinned copies and the allowance to the nearest hundredth
    of a GiB, and "in all" their sum with the process's own."""
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
    process, pinned, allowance = _host_hundredths(plan)
    left = _left(plan, "the DiT's other parameters", "the VAE")
    if left:
        copies = f"{pinned / 100:.2f} GiB of system memory pinned for {left}"
    else:
        copies = "no system memory pinned, nothing leaving the GPU"
    in_all = process + pinned + allowance
    lines.append(f"{copies}; {in_all / 100:.2f} GiB of system memory in all")
    lines.append(
        f"encode {_pass_said(job, plan.encode_tiles)}, decode"
        f" {_pass_said(job, plan.decode_tiles)}, each held {_gib(VAE_ROOM)} GiB under the budget"
    )
    names = {"encode": "encode", "dit": "DiT's phase", "decode": "decode"}
    # The DiT's peak is its longest shot's draw where that is over its longest window's own
    # (_dit_need), said where the phase bounds the plan.
    own = dit_peak(job.padded, peaks.window, plan.swapped, plan.vae_in_dit)
    counted = "its longest shot's noise drawn whole, " if peaks.dit > own else ""
    lines.append(
        f"predicted peaks: encode {_gib(peaks.encode)} GiB, DiT {_gib(peaks.dit)} GiB, decode"
        f" {_gib(peaks.decode)} GiB; the {names[peaks.bound]} bounds the plan,"
        f" {counted if peaks.bound == 'dit' else 'its room counted, '}"
        f"{_gib(peaks.budget - peaks.need(peaks.bound))} GiB under the budget"
    )
    if peaks.scratch:
        # A scratch over the allowance goes whole to its file (tiling.Scratch).
        allowed = plan.scratch_allowance
        over = sum(
            number for count, number in frames.items() if scratch(job.padded, count) > allowed
        )
        where = (
            f"over which the scratch of {over} of {_count(len(shots), 'shot')} goes whole to a"
            " memory-mapped file"
            if over
            else "which holds every shot's"
        )
        lines.append(
            f"scratch of the longest tiled shot: {_gib(peaks.scratch)} GiB; an allowance of"
            f" {allowance / 100:.2f} GiB of system memory, {where}"
        )
    options = (
        ("--window", pins.window),
        ("--swap-blocks", pins.swap_blocks),
        ("--encode-tile", pins.encode_tile),
        ("--decode-tile", pins.decode_tile),
    )
    given = [f"{option} {value}" for option, value in options if value is not None]
    lines.append(f"pinned: {', '.join(given)}" if given else "nothing pinned")
    return [f"plan: {line}" for line in lines]


def beyond_measured(plan: Plan) -> list[str]:
    """What the log warns of once the plan is made, a line per pass, the encode's then the
    decode's (cli._run): a pass in tiles larger than TILE_MAX, the largest measured, which only a
    pin gives it (_pass), or the record of a job made with one. Its predicted peak is the fit's
    alone there, and the one such pass run, a 4K encode in tiles of 2304 px, took more than its
    fit (TILE_MAX). None for a plan of the planner's own choices."""
    passes = (("encode", plan.encode_tiles), ("decode", plan.decode_tiles))
    return [
        f"the {phase} in tiles of {tiles.size} px: its predicted peak is the fit's, beyond the"
        f" sizes measured (up to {TILE_MAX} px), and the pass may take more"
        for phase, tiles in passes
        if tiles is not None and tiles.size > TILE_MAX
    ]


def _job(
    free: int,
    target: tuple[int, int],
    shots: Sequence[Shot],
    split: bool,
    numz_padding: bool,
) -> _Job:
    if not shots:
        raise ValueError("a job without a shot")
    lengths = sorted({_latents(shot.frames) for shot in shots})
    frames = max(shot.frames for shot in shots)
    return _Job(free, target, tuple(lengths), frames, split, numz_padding)


def _padded(target: tuple[int, int], numz_padding: bool) -> tuple[int, int]:
    """The frame the model takes: the target in seedvr2x's padding (job.padded_size), or in
    numz's, zeros up to multiples of 16, for the tests (model.input_transform)."""
    if numz_padding:
        return -(-target[0] // 16) * 16, -(-target[1] // 16) * 16
    return padded_size(target)


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
    if cap is None or cap >= lengths[-1]:
        return lengths[-1]
    return max(_split(length, cap) for length in lengths)


@cache
def _split(length: int, cap: int) -> int:
    """The longest window, in latents, of a shot of `length` latents under a cap, by its balanced
    layout (shot.window_layout); its length when the cap holds it whole."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import window_layout

    if cap >= length:
        return length
    return max(end - start for start, end in window_layout(length, cap))


def _draw(job: _Job, swapped: int, vae_in_dit: bool) -> int:
    """What the GPU holds at the draw of the job's longest shot's noise, in bytes (dit_draw):
    the job's largest, whatever the cap."""
    return dit_draw(job.padded, job.lengths[-1], swapped, vae_in_dit)


def _dit_need(job: _Job, cap: int | None, swapped: int, vae_in_dit: bool) -> int:
    """The predicted peak of the job's DiT phase under a cap, in bytes, `swapped` blocks swapped
    and the VAE's weights there or not: the larger of its longest window's peak (dit_peak) and
    of its longest shot's draw (_draw), never their sum: the shot is whole on the GPU for the
    draw, before its first window, and nothing of it is there beside a window but the window
    (shot.sample_windows). Wherever the phase is judged, and what the plan records (_peaks):
    real memory, unlike a VAE pass's room."""
    window = dit_peak(job.padded, _longest(job.lengths, cap), swapped, vae_in_dit)
    return max(window, _draw(job, swapped, vae_in_dit))


def _draw_bounds(job: _Job, cap: int | None, swapped: int, vae_in_dit: bool) -> bool:
    """Whether the job's DiT phase under a cap takes its longest shot's draw, more than its
    longest window's peak (_dit_need): what a refusal for the phase then names (_draw_said)."""
    window = dit_peak(job.padded, _longest(job.lengths, cap), swapped, vae_in_dit)
    return _draw(job, swapped, vae_in_dit) > window


def _draw_said(job: _Job) -> str:
    """The draw of the job's longest shot's noise as a refusal names it, where it is what takes
    the free VRAM said (_draw_bounds): what is drawn, where and when, and the shot's length."""
    shot = f"{_count(job.frames, 'frame')} ({_count(job.lengths[-1], 'latent')})"
    return (
        f"the noise of the longest shot, {shot}, drawn whole on the GPU beside the shot's latent"
        " before its first window"
    )


def _holds(job: _Job, cap: int | None, swapped: int) -> bool:
    """Whether the DiT's phase holds the job under a cap, `swapped` blocks swapped and the VAE's
    weights away: all its windows and its draws can ask for (_dit_need)."""
    return _dit_need(job, cap, swapped, vae_in_dit=False) <= job.budget


def _longest_cap(job: _Job, swapped: int) -> int | None:
    """The largest cap, from the longest shot's length down to the smallest the job can run, that
    the DiT's phase holds with `swapped` blocks swapped (_holds); None when it holds none.

    A film has thousands of shots and windows of hundreds of latents: the caps whose longest
    window is over the budget by its tokens alone are bisected away first (_longest never falls
    as the cap rises, nor the tokens with it), the window functions then asked from there down.
    None at once where the longest shot's draw is over the budget, which no cap changes
    (_draw)."""

    def possible(cap: int) -> bool:
        least = _dit_least(job.padded, _longest(job.lengths, cap), swapped, vae_in_dit=False)
        return least <= job.budget

    low, high = job.smallest, job.lengths[-1]
    if _draw(job, swapped, vae_in_dit=False) > job.budget or not possible(low):
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


def _fewest(job: _Job, cap: int | None) -> tuple[int, bool] | None:
    """The fewest swapped blocks with which the DiT's phase holds the job under a cap, its
    windows and its longest shot's draw (_dit_need), and whether the VAE's weights stay on the
    GPU meanwhile: they leave before one more block is swapped. One and two swapped blocks save
    nothing (dit_peak, dit_draw), so are never chosen. None when every block swapped and the
    VAE away don't hold it."""
    for swapped in (0, *range(SLOTS + 1, BLOCKS + 1)):
        for vae_in_dit in (True, False):
            if _dit_need(job, cap, swapped, vae_in_dit) <= job.budget:
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
    cap = recorded.window
    if pins.swap_blocks is None:
        swapped, vae_in_dit = _fewest(job, cap) or (BLOCKS, False)
    else:
        swapped = pins.swap_blocks
        vae_in_dit = _dit_need(job, cap, swapped, vae_in_dit=True) <= job.budget
        fewest = _fewest(job, cap)
        if not _holds(job, cap, swapped) and fewest is not None:
            # The pin's doing, not the memory's: the recorded shape still fits with more blocks.
            # PROVISIONAL (implementation, 2026-10-10): refused as the pin is on a new job.
            raise PlanError(_swap_refused(job, swapped, cap, "the recorded plan's", fewest[0]))
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

    The cap is always a number, never None: the longest shot's length when it fits, else the
    longest window the job really runs under the largest cap that fits (_own_cap), not that cap,
    which a card may hold no window of: a shot of 38 latents on a 24 GB card at 1080p runs two
    windows of 20 under every cap from 20 to 37, and its plan says 20 (DESIGN.md, Memory
    planner, Choices, 1); a refused --window is told the same. A --window pinned stays the cap
    as given (DESIGN.md, Memory planner, Pins). With --swap-blocks, the cap is the largest that
    fits with those blocks swapped and the VAE's weights away, as it is without the pin: the
    VAE leaves the DiT's phase before a window shortens or a block is swapped (Choices, 1), a
    choice the planner makes around the pin as around any other (Pins), so that --swap-blocks 0
    runs 6 latents at 1080p on a 24 GB card (Per-phase peaks: every block on the GPU, the VAE
    away).

    PROVISIONAL (implementation, 2026-10-10), a choice DESIGN.md doesn't make: of two pins that
    don't fit together, --window is the one refused while some cap fits the blocks pinned,
    --swap-blocks when none does.

    A --window refused is refused for its window, never for the longest shot's draw: no cap
    changes the draw (_draw), so that where some cap fits the blocks pinned, the draw fits
    them."""
    most = BLOCKS if pins.swap_blocks is None else pins.swap_blocks
    longest = _longest_cap(job, most)
    if longest is None:
        # The floor holds every block swapped: a --swap-blocks that holds no window at all, or
        # not the longest shot's draw.
        cap = job.smallest
        whose = "the shortest this job can run"
        if pins.window is not None and _holds(job, pins.window, BLOCKS):
            cap, whose = pins.window, f"--window {pins.window}'s longest"
        fewest = _fewest(job, cap)
        assert fewest is not None
        raise PlanError(_swap_refused(job, most, cap, whose, fewest[0]))
    if pins.window is None:
        cap = _own_cap(job, longest)
    elif _holds(job, pins.window, most):
        cap = pins.window
    else:
        latents = _longest(job.lengths, pins.window)
        need = _dit_need(job, pins.window, most, vae_in_dit=False) + MARGIN
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
    if pins.swap_blocks is not None:
        return cap, most, _dit_need(job, cap, most, vae_in_dit=True) <= job.budget
    fewest = _fewest(job, cap)
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


def _swap_refused(job: _Job, swapped: int, cap: int | None, whose: str, fewest: int) -> str:
    """--swap-blocks refused: the fewest blocks that hold the job's DiT phase under a cap, the
    free VRAM said that of the phase with the blocks pinned (_dit_need), and what takes it
    named: the cap's longest window, or the longest shot's draw where that takes more
    (_draw_bounds)."""
    latents = _longest(job.lengths, cap)
    need = _dit_need(job, cap, swapped, vae_in_dit=False) + MARGIN
    if _draw_bounds(job, cap, swapped, vae_in_dit=False):
        what = f"{_draw_said(job)},"
    else:
        what = f"a window of {_count(latents, 'latent')}, {whose},"
    return (
        f"--swap-blocks {swapped}: {what} takes {_over(need)} GiB of free VRAM with"
        f" {_count(swapped, 'block')} swapped, and the driver reports {_under(job.free)} GiB"
        f" free: the fewest blocks that hold it are --swap-blocks {fewest}"
    )


def _tile_sizes(padded: tuple[int, int]) -> range:
    """The sides of the tiles the planner makes for padded frames, the largest first: multiples
    of TILE_STEP from the largest under the frame's longer side, a tile that covers the frame
    being no tiling (covers), or from TILE_MAX on a frame longer than it, down to TILE_MIN; none
    for a frame that small."""
    largest = min(TILE_MAX, (max(padded) - 1) // TILE_STEP * TILE_STEP)
    return range(largest, TILE_MIN - 1, -TILE_STEP)


def _vae_peak(
    job: _Job, phase: Phase, tiles: Tiles | None, resident: int | None, swapped: int
) -> int:
    """The predicted peak of the job's encode or decode (encode_peak, decode_peak), of its
    longest shot's: what a pass keeps on the GPU of the shot grows with it."""
    shot = job.lengths[-1]
    if phase == "encode":
        return encode_peak(job.padded, tiles, resident, swapped, latents=shot)
    return decode_peak(job.padded, job.output, tiles, resident, swapped, job.split, latents=shot)


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
    fits. The job's floor holds already (_floor), so some pass fits (_least_pass); and no tile
    pinned covers the frame, which plan refuses first.

    A pass pinned, tiled or not, is held to the room the planner's own is (VAE_ROOM), and
    refused without it: every VAE pass is (DESIGN.md, Memory planner, Budget).

    PROVISIONAL (implementation, 2026-10-11): DESIGN.md has the planner choose no tile above
    TILE_MAX, the largest measured, and doesn't say what a pin above it gets. It is taken
    exactly when it fits by its fit, beyond the sizes measured as it is: a pin is the user's,
    and the search alone never chooses one, nor does a refusal name one. The log warns of it,
    once per pass (beyond_measured)."""
    fitting = _fitting(job, phase, swapped)
    if pin is None:
        return fitting[0]
    option = f"--{phase}-tile {pin}"
    tiles = Tiles(pin) if pin else None
    need = _vae_need(job, phase, tiles, 0, swapped) + MARGIN
    if need <= job.free:
        return tiles
    # The untiled pass alone fits where a frame little larger than the smallest tile takes less
    # untiled than in that tile (_least_pass).
    largest = next((fit.size for fit in fitting if fit is not None), None)
    fits = (
        f"no tile fits, the untiled {phase} does, --{phase}-tile 0"
        if largest is None
        else f"the largest tile that fits is --{phase}-tile {largest}"
    )
    raise PlanError(
        f"{option}: the {phase} {_tiles_said(tiles)} takes {_over(need)} GiB of free VRAM,"
        f" the DiT off the GPU{_ROOM_HELD}, and the driver reports {_under(job.free)} GiB"
        f" free: {fits}"
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
    that doesn't fit is refused, never degraded); else None. The least: the larger of the
    smallest window the job can run and of its longest shot's draw, every block of the DiT
    swapped and the VAE's weights away (_dit_need); the encode and the decode each in the pass
    that takes the least, of the job's longest shot (_least_pass), the DiT away, whatever the
    plan swaps: a phase that keeps nothing of it frees its slots too (dit_on_gpu). What a phase
    takes is its peak, and for the encode and the decode their room with it (_vae_need)."""
    least: dict[Phase, int] = {
        "encode": _vae_need(job, "encode", _least_pass(job, "encode"), 0, BLOCKS),
        "dit": _dit_need(job, job.smallest, BLOCKS, vae_in_dit=False),
        "decode": _vae_need(job, "decode", _least_pass(job, "decode"), 0, BLOCKS),
    }
    phase = max(PHASES, key=lambda phase: least[phase])
    return (phase, least[phase]) if least[phase] > job.budget else None


def _least_pass(job: _Job, phase: Phase) -> Tiles | None:
    """The pass of the job's encode or decode that takes the least, the DiT away: None, the
    untiled pass, or tiles; untiled on a frame too small for a tile. The least of three, which
    tests/test_planner.py holds to the least of every pass the planner may make, over many
    sizes: the smallest tiles, the least of all in most frames; the untiled pass, under them on
    a frame little larger than one, 512x528, where a square tile takes its measured peak
    (MEASURED) and the untiled pass its fit; and the smallest tiles the refit serves
    (REFITTED), under the smallest of all on a strip over 1024 px long and under 74 px high
    for the encode, 134 for the decode, where the two tiled fits cross."""
    sizes = _tile_sizes(job.padded)
    least = dict.fromkeys(size for size in (*sizes[-1:], REFITTED) if size in sizes)
    passes = [None, *(Tiles(size) for size in least)]
    return min(passes, key=lambda tiles: _vae_need(job, phase, tiles, 0, BLOCKS))


def _too_large(job: _Job, phase: Phase, least: int) -> str:
    """The floor's refusal (_floor), counted in free VRAM, never in card sizes (DESIGN.md, Memory
    planner): the free VRAM the output size takes, what the driver reports free, and the largest
    output size that fits in it, by its short side, a multiple of 16, the target scaled in
    proportion and the floor checked there with the same shots. Where the DiT's phase is the
    floor by its longest shot's draw, more than its smallest window's peak (_draw_bounds), the
    message names the draw in the window's place (_draw_said); the draw takes less of a smaller
    frame, a byte a padded pixel and a latent (dit_draw), so that the size named is one whose
    draw fits. --resolution is named first, and the size as about that: it is the target's
    scaled, not the source's, which the command line scales, and the two can differ by 2 px (a
    1920x1080 source refused at --resolution 600: 794x448 here, 796x448 by --resolution 448)."""
    height, width = job.output
    if phase == "dit":
        latents = _longest(job.lengths, job.smallest)
        if _draw_bounds(job, job.smallest, BLOCKS, vae_in_dit=False):
            how = _draw_said(job)
        else:
            how = f"its smallest window, {_count(latents, 'latent')}"
        how += ", with every block of the DiT swapped and the VAE off the GPU"
        room = 0
    else:
        tiled = _tiles_said(_least_pass(job, phase))
        if not _tile_sizes(job.padded):
            tiled += ", too small for a tile"
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
    GPU, and what a tiled shot's scratch may take of it, the plan's allowance
    (Plan.scratch_allowance). PROVISIONAL (implementation, 2026-10-10): the loader's own peak
    while the weights load, 16.5 GiB, mostly the weights' file mapped (DESIGN.md, Memory planner,
    Host RAM), isn't counted. PROVISIONAL (implementation, 2026-10-11): a scratch over its
    allowance goes whole to its memory-mapped file (tiling.Scratch), and the allowance is still
    counted, as DESIGN.md counts it, for the page cache its mapping works in. Nor is what a
    tiled encode sums beside its scratch counted, its tiles' moments on the CPU
    (tiling.encode_tiled): a byte per pixel of the padded frame and per latent of the shot, a
    twenty-fourth of the scratch, 2.0 GiB for 1,001 frames at 4K. Nor a shot's latent and its
    noise, which wait in system memory through its windows (shot.sample_windows): as much
    again, 7.84 GiB for a shot of 1,000 latents at 4K; nor a one-file output's windows, kept
    there until their shot's decode (units.Units): half a byte, a latent two windows share
    counted in both."""
    latents = _longest(job.lengths, plan.window)
    return Peaks(
        job.free,
        job.budget,
        _vae_peak(job, "encode", plan.encode_tiles, plan.resident_encode, plan.swapped),
        _dit_need(job, plan.window, plan.swapped, plan.vae_in_dit),
        _vae_peak(job, "decode", plan.decode_tiles, plan.resident_decode, plan.swapped),
        HOST_RESIDENT + _pinned(plan) + plan.scratch_allowance,
        _kept(job, plan),
        latents,
        tokens(job.padded, 1),
    )


def _kept(job: _Job, plan: Plan) -> int:
    """The scratch of the job's longest shot under the plan, in bytes (scratch); 0 when neither
    pass is tiled. One pass's: the encode's and the decode's are the same size, and never open
    together."""
    tiled = plan.encode_tiles is not None or plan.decode_tiles is not None
    return scratch(job.padded, job.frames) if tiled else 0


def _allowance(job: _Job, plan: Plan, host_available: int | None) -> int:
    """What a tiled shot's scratch may take of system memory under the plan, in bytes
    (Plan.scratch_allowance), a larger one going to its file: half of what the host has
    available once the pinned copies are counted, and no more than the longest tiled shot needs
    (DESIGN.md, Memory planner, Host RAM: provisional, until measured on a consumer host); 0 when
    no pass is tiled. It shapes no bit.

    PROVISIONAL (implementation, 2026-10-11), two choices DESIGN.md doesn't make. No more either
    than what the host has left once the process's own memory is counted with the pinned copies
    (HOST_RESIDENT), down to nothing, every scratch then going to its file: so that a job is
    refused for the system memory the process and its pinned copies take (_unfit), never for an
    allowance: half of what the copies leave is more than the process leaves of it where they
    leave under 4.8 GiB. And with what the host has available unknown (None), the longest tiled
    shot's scratch, ALLOWANCE_UNKNOWN_HOST at most."""
    kept = _kept(job, plan)
    if host_available is None:
        return min(kept, ALLOWANCE_UNKNOWN_HOST)
    left = host_available - _pinned(plan)
    return max(0, min(kept, left // 2, left - HOST_RESIDENT))


def _unfit(job: _Job, plan: Plan, host_available: int | None) -> str | None:
    """Why the plan doesn't fit the job's memory (fits): the first phase over the budget, a VAE
    phase judged with its room as the planner chose it (Peaks.need), or the system memory it
    takes over what is available, the process's own, the pinned copies and the scratch's
    allowance; None when it fits. A plan of the planner's own is never refused for its
    allowance, which is nothing where the first two leave nothing (_allowance)."""
    peaks = _peaks(job, plan)
    blocks = _count(plan.swapped, "block")
    vae = "" if plan.vae_in_dit else " and the VAE off the GPU"
    # The DiT's phase by what takes its peak: its longest window, or its longest shot's draw.
    if _draw_bounds(job, plan.window, plan.swapped, plan.vae_in_dit):
        dit = f"{_draw_said(job)},"
    else:
        dit = f"a window of {_count(peaks.window, 'latent')}"
    whats = {
        "encode": f"the encode, {_tiles_said(plan.encode_tiles)}, with"
        f" {_resident_said(plan.resident_encode, plan.swapped)},",
        "dit": f"the DiT's phase, {dit} with {blocks} swapped{vae},",
        "decode": f"the decode, {_tiles_said(plan.decode_tiles)}, with"
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
        # Each part as the plan's lines say it, and the total their sum as said, so that the
        # message's figures add up (_host_hundredths).
        process, pinned, allowance = _host_hundredths(plan)
        taken = [(process, "for the process")]
        if pinned:
            left = _left(plan, "the DiT's other parameters", "the VAE's weights")
            taken.append((pinned, f"pinned for {left} that leave the GPU"))
        if allowance:
            taken.append((allowance, "for a tiled shot's scratch"))
        total = process + pinned + allowance
        # What is there is rounded down, and said a hundredth under the total at least: the
        # parts being rounded to the nearest, a host within 0.02 GiB of what the plan takes
        # would else read as having it.
        there = min(_hundredths(host_available, up=False), total - 1)
        said = ", ".join(f"{count / 100:.2f} {what}" for count, what in taken)
        return (
            f"the plan takes {total / 100:.2f} GiB of system memory ({said}) and"
            f" {there / 100:.2f} GiB are available, {(total - there) / 100:.2f} GiB short"
        )
    return None


def _host_hundredths(plan: Plan) -> tuple[int, int, int]:
    """What the plan takes of system memory, as its line and a refusal for it say it (said,
    _unfit), in hundredths of a GiB: the process's own (HOST_RESIDENT), the pinned copies
    (_pinned) and the scratch's allowance (Plan.scratch_allowance), each to the nearest, as the
    plan's other figures are; the total either says is their sum, so that its figures add up.
    One plan's pinned memory then reads the same in both, 15.81 GiB with every part of the
    weights away, DESIGN.md's figure (Memory planner, Host RAM): rounded up, as what is taken of
    the GPU's memory is, a refusal would say 15.82 where the plan's line says 15.81."""
    return _nearest(HOST_RESIDENT), _nearest(_pinned(plan)), _nearest(plan.scratch_allowance)


def _tiles_said(tiles: Tiles | None) -> str:
    """A pass as a refusal says it: untiled, or in its tiles."""
    return "untiled" if tiles is None else f"in tiles of {tiles.size} px"


def _pass_said(job: _Job, tiles: Tiles | None) -> str:
    """A pass as the plan's lines say it: untiled, or its tiles and their grid, rows x columns."""
    if tiles is None:
        return "untiled"
    rows, columns = grid(job.padded, tiles)
    return f"in tiles of {tiles.size} px, {rows} x {columns}"


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


def _nearest(count: float) -> int:
    """Bytes in hundredths of a GiB, to the nearest, as _gib says them."""
    return round(float(_gib(count)) * 100)


def _over(count: float) -> str:
    """Bytes in GiB, to 2 decimals, rounded up: what something takes of the GPU's memory."""
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
