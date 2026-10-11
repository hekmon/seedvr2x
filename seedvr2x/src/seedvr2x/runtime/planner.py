"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model: what each phase is predicted to peak at on the GPU, in bytes, by the fits of
research/docs/planner-limits.md and research/docs/vram.md, measured on numz's CLI with the 7B
fp16, the one DiT seedvr2x runs (weights.ACCEPTED). Sizes are (height, width) in pixels; `padded`
is the frame the model takes, the target padded (job.padded_size).

PROVISIONAL (milestone 3's seam): the planner itself is a stand-in until it is built on the
model. It chooses nothing: the pins are the plan, each shot in one window without --window, every
weight on the GPU, as before milestone 3; and it refuses the pins whose work isn't built yet,
each refusal leaving with the commit that builds it."""

from dataclasses import dataclass
from functools import cache

from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.plan import Plan, Tiles

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
    doesn't fit the memory free."""


@dataclass(frozen=True)
class Pins:
    """The choices the command line fixes (DESIGN.md, Memory planner, Pins), each taken exactly or
    refused, never adjusted; None leaves a choice to the planner."""

    window: int | None = None  # --window: the cap on the DiT windows, in latents
    encode_tile: int | None = None  # --encode-tile: a tile's side in output pixels; 0: untiled
    decode_tile: int | None = None  # --decode-tile, likewise
    swap_blocks: int | None = None  # --swap-blocks: the DiT's blocks swapped


def plan(pins: Pins) -> Plan:
    """The job's plan: the pins as they are, nothing chosen around them yet."""
    return Plan(
        window=pins.window,
        encode_tiles=_tiles("--encode-tile", pins.encode_tile),
        decode_tiles=_tiles("--decode-tile", pins.decode_tile),
        swapped=_swapped(pins.swap_blocks),
    )


def _tiles(option: str, size: int | None) -> Tiles | None:
    """A tile pin's tiles, the pin's size taken as it is: squares of `size` output pixels a side,
    sharing plan.OVERLAP; None without a pin or at 0, an untiled pass. Nothing is refused here
    yet: the tiles' rule, a side a multiple of 64 px from 512 up (DESIGN.md, Memory planner), is
    held where `option` is read (cli._tile), and the pins are taken together as they come, the
    encode tiled without the decode too: no card's plan is so, but that is the planner's choice,
    no rule on a pin."""
    return Tiles(size) if size else None


def _swapped(blocks: int | None) -> int:
    """A --swap-blocks pin's blocks; none without a pin."""
    if blocks is not None and blocks > 36:  # the blocks of the 7B, the DiT v1 runs
        raise PlanError(f"--swap-blocks {blocks}: the DiT has 36 blocks")
    return blocks or 0


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
    if not 0 <= swapped <= BLOCKS:
        raise ValueError(f"{swapped} blocks swapped of the DiT's {BLOCKS}")
    peak = DIT_LOADED + PER_TOKEN * tokens(padded, latents)
    peak += PER_WINDOW * attention_windows(padded, latents)
    peak -= BLOCK * max(0, swapped - SLOTS)
    return round(peak if vae_in_dit else peak - VAE_WEIGHTS)


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
