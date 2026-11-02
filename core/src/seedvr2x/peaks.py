"""What each brick takes of the GPU, predicted (DESIGN.md, Memory planner, Per-phase peaks): the
DiT's peak in a window and where its weights are, the VAE's encode and decode in a pass, whole or
tiled, in bytes, by the fits of research/docs/planner-limits.md and research/docs/vram.md,
measured on numz's CLI with the 7B fp16, the one DiT seedvr2x runs (weights.ACCEPTED), and by the
peaks measured on seedvr2x's own tiled passes (MEASURED); the compiled passes' range; and the
store a tiled pass takes. The library's (DESIGN.md, Repository and apps), as the bricks are; the
command line's planner (runtime/planner.py) chooses a job's plan with them. Sizes are (height,
width) in pixels; `padded` is the frame the model takes, the target padded (frames.padded_size).
No torch at import: the command line plans before it imports torch (tests/test_cli.py)."""

from functools import cache

from seedvr2x.frames import padded_length
from seedvr2x.passes import Phase, Tiles
from seedvr2x.runtime import tilegrid, weights

# The DiT's own window functions, the 7B's: plain Python, and the packages above them hold
# nothing, so importing them brings in no torch.
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)

GIB = 1024**3
MIB = 1024**2

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
# The attention backends, as the run names the one it takes (model.attention_backend), and the
# DiT's fit under each: what it adds to DIT_LOADED, in bytes, and the factor on PER_TOKEN. The
# fit above is FlashAttention 2's (12 latents at 1080p measured 28.316 GiB for 28.320).
# Under SDPA, which a machine without a FlashAttention build runs, the DiT peaks higher
# (DESIGN.md, Memory planner, Per-phase peaks): 4.69% more a token and 0.141 GiB, an upper
# bound of 45 windows measured on two cards (2026-10-12), each window's peak the largest of its
# runs, a predicted peak having to be one (the margin covers memory outside torch's tensors,
# not a fit's errors):
# - a sweep of 29 windows on the 96 GB card (seedvr2x's own forward, one sampling step): 1 to
#   70 latents at 1080p and 1 to 17 at 4K, 22 with no block swapped and the VAE's weights there,
#   on which the factor is fitted (least squares on the tokens, FlashAttention 2's per-window
#   term kept: 1.0469, residuals from -0.042 to +0.072 GiB), and 7 with 12 or 36 blocks swapped
#   or the VAE away, which hold that a swapped block and the VAE's leaving save under SDPA what
#   they save under FlashAttention 2;
# - 16 windows on a desktop card (an RTX 5090 behind WSL2's GPU device,
#   research/scripts/hold_probe.py): 1 to 21 latents at 1080p, 1 to 5 at 4K, with no block
#   swapped and with the 36, the VAE's weights on the GPU. They stand 0.02 to 0.03 GiB over
#   the first card's at the same windows, and set the constant: the 29 alone are bounded by
#   0.124 GiB, which the desktop's 4 latents of 4K pass by 0.016.
# The least margin left is 0.0009 GiB, on that window; on the 96 GB card 0.017, at 8 latents
# of 4K, and 0.13 at the most (tests/test_planner.py holds the 45).
FLASH_ATTN_2, SDPA = "flash_attn_2", "sdpa"
DIT_ATTENTION = {FLASH_ATTN_2: (0, 1.0), SDPA: (round(0.141 * GIB), 1.0469)}

# The VAE's weights, on the GPU in the VAE's phases, and in the DiT's unless the plan sends them
# away for it, in bytes: 0.4669 GiB, its parameters by the model's shapes in the pipeline's
# bfloat16, each on a multiple of 512 bytes, as the part that leaves lays them out (model.Models'
# vae, a blockswap.Part; tests/test_planner.py builds the model). Measured on numz as DESIGN.md's
# 0.47 GiB: 28.15 to 27.68 GiB (Memory planner, Per-phase peaks; vram.md, Practical rules).
VAE_WEIGHTS = 501_306_880

# The DiT seedvr2x runs, the 7B (weights.ACCEPTED), by the model's own constants
# (runtime/weights.py, which imports no torch; the placement moves the weights by the same): its
# blocks, 36, and a block's parameters in bytes, 0.4220 GiB, all alike, the 36 holding 15.19 of
# the DiT's 15.35 GiB (DESIGN.md, Memory planner, Host RAM). SLOTS: the buffers on the GPU its
# swapped blocks stream through (DESIGN.md, BlockSwap, rewritten).
DIT = weights.ACCEPTED["dit"]
BLOCKS = len(DIT.blocks)
BLOCK = DIT.blocks[0]
SLOTS = weights.SLOTS
# The DiT's parameters outside its blocks, emb_in, txt_in, vid_in and vid_out, in bytes: 0.155
# GiB, by the model's shapes in fp16 (tests/test_planner.py builds the model). They leave the GPU
# only with the last block a VAE phase keeps (dit_on_gpu).
DIT_OTHER = DIT.other
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
# A tile's measured peaks, by its rows and columns in pixels: the encode's and the decode's, in
# GiB above what is allocated before the call, on seedvr2x's own tiled passes (2026-10-11;
# DESIGN.md, Memory planner, Per-phase peaks; research/docs/planner-limits.md, "seedvr2x's tiled
# passes: every tile's peak", measured by research/scripts/tile_peaks.py). The 25 squares the
# planner may choose, 512 to 2048 px, and the 20 tiles cut off at 1080p's 1,104 rows (1152 to
# 1856 px) and at 720p's 752 (768 to 1216 px). Over 21 frames, the tiles sharing 64 px: a square
# fed alone, on frames of T rows by T + 128 columns; a cut-off tile through the encoder's own
# padding. Each tile was measured once, on one card, and 512 px's encode was its process's first
# pass. Where a tile has a second measure on file, the larger: the first whole frames in tiles
# (512, 768, 1024, 1536 and 2048 px, and 1280 px at 1080p; 0.03 GiB over the tile alone at the
# most), and shots of 21 to 301 frames in 1024-px tiles (10.4742 and 17.4499 GiB).
# The passes the one-output convolution takes (vendor/models/video_vae_v3/modules/
# causal_inflation_lib.py, ONE_OUTPUT_FROM) take their peak measured with it (2026-10-17;
# DESIGN.md, Memory planner, Budget: "the planner takes the peaks measured with it for the passes
# it takes, those two included, the other passes' as they are"; planner-limits.md, "Under the
# one-output convolution", by research/scripts/one_output_peaks.py), rounded up to the
# hundredth: the decodes of the squares from 1152 px and of the tiles cut off at 1,104 rows from
# 1216 px, and the encodes of the squares from 1664 px. Its tile alone, with the same driver,
# frames and overlap: 0.18 to 5.17 GiB under the measures above, every output bit-identical,
# but for the two decodes just over its floor, 1152 px (21.986 GiB for 21.950 without it) and
# 1104x1216 (22.242 for 22.204). Every other entry is the measure above; those of its passes
# measured again beside it came within 0.005 GiB of it. Its measure is the tile alone: the four
# entries that took a second measure's larger figure, the 2048-px encode and decode, 1536 px's
# decode and 1104x1280's (up to 0.028 GiB over the tile alone), have no whole-frame measure with
# it. Whole frames in those tiles under it, a pass alone (2026-10-17, the frames' floor's
# reference runs on the bare card; vae-room.md), peaked 0.005 to 0.010 GiB under the decode's
# prediction on this table and 0.41 under the encode's.
# The fits don't hold them to the margin (DESIGN.md): from 1024 px up a square's decode without
# the one-output convolution was 0.06 to 0.34 GiB under its fit, and below up to 0.67 over it
# (576 px); an encode moves in steps, from 0.38 GiB under its fit (1472 px) to 0.80 over (832
# px). So a tile takes its measure, and the fits serve the tiles nobody measured (_fit), whose
# peaks under the one-output convolution no measure has.
MEASURED = {
    (512, 512): (3.52, 5.89),
    (576, 576): (4.38, 7.45),
    (640, 640): (4.92, 8.09),
    (704, 704): (5.95, 9.79),
    (768, 768): (7.08, 11.11),
    (832, 832): (8.31, 12.30),
    (896, 896): (8.56, 14.10),
    (960, 960): (9.82, 15.67),
    (1024, 1024): (10.47, 17.46),
    (1088, 1088): (11.81, 19.65),
    (1152, 1152): (13.24, 21.99),
    (1216, 1216): (14.15, 24.14),
    (1280, 1280): (15.67, 26.40),
    (1344, 1344): (16.79, 28.84),
    (1408, 1408): (18.46, 31.32),
    (1472, 1472): (19.74, 33.98),
    (1536, 1536): (21.50, 36.75),
    (1600, 1600): (23.21, 39.65),
    (1664, 1664): (24.88, 42.46),
    (1728, 1728): (26.73, 45.69),
    (1792, 1792): (28.44, 48.93),
    (1856, 1856): (30.46, 52.48),
    (1920, 1920): (32.30, 55.69),
    (1984, 1984): (34.29, 59.25),
    (2048, 2048): (36.33, 63.35),
    # Cut off at 1080p's 1,104 rows.
    (1104, 1152): (12.68, 21.04),
    (1104, 1216): (13.39, 22.25),
    (1104, 1280): (13.55, 23.16),
    (1104, 1344): (14.22, 24.18),
    (1104, 1408): (14.90, 25.21),
    (1104, 1472): (15.58, 26.27),
    (1104, 1536): (16.26, 27.30),
    (1104, 1600): (16.42, 28.26),
    (1104, 1664): (17.08, 29.28),
    (1104, 1728): (17.74, 30.26),
    (1104, 1792): (18.39, 31.29),
    (1104, 1856): (19.05, 32.38),
    # Cut off at 720p's 752 rows.
    (752, 768): (6.94, 10.86),
    (752, 832): (7.51, 11.49),
    (752, 896): (8.09, 12.38),
    (752, 960): (7.70, 12.69),
    (752, 1024): (8.21, 13.54),
    (752, 1088): (8.73, 14.39),
    (752, 1152): (9.24, 14.92),
    (752, 1216): (9.75, 15.59),
}
# The measured squares' sides, ascending: the line between two of them is by area (_line).
_SQUARES = tuple(sorted(rows for rows, columns in MEASURED if rows == columns))
# What a tile cut off at the picture's edges takes over its rule where no measure has it, in
# bytes (_fit): the rule is the larger of its fit and of the squares' line at its area, and the
# most a measured cut-off tile stands above it is 0.27 GiB, the decode of 752x896 (12.38 GiB for
# a fit of 12.11 and a line of 12.09); the encodes stand from 0.68 under it (752x960) to 0.21
# over (1104x1536). tests/test_peaks.py holds the 20 tiles' 40 peaks to it. DESIGN.md's 0.3
# GiB (Memory planner, Per-phase peaks).
CUT_OFF_EXTRA = round(0.3 * GIB)

# What seedvr2x's encode holds over its fit, tiled or not, in bytes: the largest excess of a
# measured peak over the fit's in the planner's first matrix (the 96 GB card emulating smaller
# ones, 2026-10-11): at 1080p, 9.66 GiB measured where the fit gave 9.546 in tiles of 896 px
# (+0.11), and 13.91 for 13.511 in tiles of 1216 px (+0.40); untiled at 540p, 22.53 for 22.22
# (+0.31). At 4K the refit is above the measurement: 29.53 for 29.694 in tiles of 1792 px
# (-0.16), 21.26 for 21.648 at 1472 px (-0.39). PROVISIONAL (implementation, 2026-10-11): one
# constant on every encode, until the encode is refitted on seedvr2x, every tile size measured
# (MEASURED).
ENCODE_EXTRA = round(0.4 * GIB)

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

# --compile, the VAE compiled with its causal convolutions left out of the graph (DESIGN.md,
# Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0; the DiT, not", decided on
# 2026-10-24; research/docs/compile.md). A compiled pass is held to COMPILE_PEAK times its tile's
# plain peak above what is on the GPU before it (the fit the planner takes the tile's peak by,
# _fit): the study's compiled passes, first calls included, peaked 1.09 to 1.10 times the plain
# encode and 1.04 to 1.06 times the plain decode, one clip, 21 frames, 1080p and 4K, and a whole
# job planned at 1.1 ran with no pass out of memory (compile.md, "A fit the planner can use",
# "A whole job, planned for the compiled peak"). The study scaled the fit alone, and so is it
# here: ENCODE_EXTRA, the encode's margin on the fit, and the VAE's weights are not scaled.
COMPILE_PEAK = 1.1
# Only the shapes measured are compiled, until the 96 GB card measures larger ones (DESIGN.md,
# decided on 2026-10-24: "the planner keeps no blind spot"): encode tiles up to 1856 px or the
# 1080p frame whole, decode tiles up to 1152 px. The encode's whole frame is read as the padded
# 1080p frame's two sides at most, rows and columns each (padded_size(1080x1920) = 1104x1920),
# and not as its area, the narrower reading of "the 1080p frame": a frame of its area with
# another shape is not the one measured. PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says "the
# 1080p frame whole" and doesn't say what is as large as it); and an untiled decode is outside
# the range, "decode tiles up to 1152 px" naming tiles alone: the study's decodes were all in
# tiles (compile.md, the VAE's tables).
COMPILE_ENCODE_TILE_MAX = 1856
COMPILE_ENCODE_WHOLE = (1104, 1920)
COMPILE_DECODE_TILE_MAX = 1152


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
    padded: tuple[int, int],
    latents: int,
    swapped: int = 0,
    vae_in_dit: bool = True,
    attention: str = FLASH_ATTN_2,
) -> int:
    """The predicted peak of the DiT's phase, in bytes: one window of `latents` latents of padded
    frames, `swapped` of the DiT's blocks swapped, the VAE's weights on the GPU meanwhile unless
    vae_in_dit is False, by the fit of the attention backend the run takes (DIT_ATTENTION;
    ValueError for another name).

    A swapped block takes its BLOCK off the peak, but for the SLOTS on the GPU (DESIGN.md,
    BlockSwap, rewritten), so that one or two swapped blocks save nothing: measured on seedvr2x,
    one window of 1 latent at 1080p, the peak with S blocks swapped is the peak with none less
    BLOCK x max(S - 2, 0), to 0.001 GiB (17.016 GiB with none, 16.595 with 3, 12.796 with 12,
    2.668 with 36; milestone 3, 2026-10-11, as tests/test_placement.py measures it). numz's own
    swap, one block there at a time, saved 0.4227 GiB x (S - 1)."""
    windows = PER_WINDOW * attention_windows(padded, latents)
    return round(dit_least(padded, latents, swapped, vae_in_dit, attention) + windows)


def dit_least(
    padded: tuple[int, int], latents: int, swapped: int, vae_in_dit: bool, attention: str
) -> float:
    """dit_peak less its attention windows' part, in bytes: never above the peak, and counted
    without the window functions, which lay out every window of each length they are asked."""
    if not 0 <= swapped <= BLOCKS:
        raise ValueError(f"{swapped} blocks swapped of the DiT's {BLOCKS}")
    if attention not in DIT_ATTENTION:
        raise ValueError(f"attention {attention!r}: the DiT's fit is {FLASH_ATTN_2}'s or {SDPA}'s")
    constant, factor = DIT_ATTENTION[attention]
    per_token = PER_TOKEN * factor * tokens(padded, latents)
    least = DIT_LOADED + constant + per_token - BLOCK * max(0, swapped - SLOTS)
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


@cache
def tile_shapes(padded: tuple[int, int], tiles: Tiles) -> tuple[tuple[int, int], ...]:
    """(rows, columns), in pixels, of the tiles of a tiled pass over padded frames, each shape
    once, the largest first: the tiled passes' own grid (tilegrid.grid), whose last tile along
    a side is the shorter one, what the stride leaves of the picture. The first is the square
    cut off at the picture's edges (DESIGN.md, Memory planner: a 1280-px tile at 1080p is
    1104x1280), the one tilegrid.largest gives the area of; not always the tile that takes the
    most (_fit)."""
    laid = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    shapes = {(tile.rows * tilegrid.SCALE, tile.columns * tilegrid.SCALE) for tile in laid}
    return tuple(sorted(shapes, key=lambda shape: (-shape[0] * shape[1], shape)))


def grid_shapes(padded: tuple[int, int], tiles: Tiles) -> tuple[int, int]:
    """(rows, columns) of the tiles of a tiled pass over padded frames, as the plan's lines say
    them: the tiled passes' own grid, counted (tilegrid.grid; numz's, DESIGN.md, Pipeline, per
    shot, step 3). 2160x3840 takes 3 x 4 tiles of 1024 px and 2 x 2 of 2048, overlapping by
    passes.OVERLAP, 64 (planner-limits.md, Tiled VAE at 4K, measured it at 128: 3 x 5 of 1024); a
    side the tile covers takes one."""
    laid = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    return len({tile.top for tile in laid}), len({tile.left for tile in laid})


def encode_peak(
    padded: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    compiled: bool = False,
) -> int:
    """The predicted peak of the VAE's encode of padded frames, in bytes: its peak untiled
    (tiles None) or the most a tile of its grid takes (_fit), above what is on the GPU before
    the call, the VAE's weights and what of the DiT stays there (dit_on_gpu's resident and
    swapped); and what seedvr2x's encode holds over it (ENCODE_EXTRA). Whatever the shot's
    length: no encode keeps anything of the shot on the GPU, each slice's part of the latent
    taken off it as it is made, and a tiled encode summing in system memory (DESIGN.md, Memory
    planner, Per-phase peaks: at 1080p a shot of 301 frames peaked where one of 21 does, to the
    byte, untiled and in 1024-px tiles). Not the shot's frames either: seedvr2x feeds the encode
    slice by slice, where numz moves a batch's there first, 0.046 GiB per 4K frame. Nor the room
    every VAE pass is held to (VAE_ROOM), which the planner judges it by and no peak holds.

    compiled: the VAE compiled (--compile): the tile's peak above the weights is COMPILE_PEAK
    times its plain one.

    ValueError for tiles of which one covers the whole padded frame, which is no tiling
    (covers)."""
    above = _fit(padded, tiles, 0) * (COMPILE_PEAK if compiled else 1.0)
    return round(above) + ENCODE_EXTRA + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


def decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
    compiled: bool = False,
) -> int:
    """The predicted peak of the VAE's decode of padded frames, in bytes, its peak above what is
    on the GPU before the call as encode_peak has the encode's (without ENCODE_EXTRA, the
    encode's), whatever the shot's length too: a decode takes its latents from system memory
    slice by slice; with split, the colour correction too, which corrects a slice of the decode
    at once, SLICE frames at most of the output's size (frames.output_size). ValueError for tiles
    of which one covers the whole padded frame, as encode_peak.

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
    there (the predictions as the planner made them then, by the fits and the five squares
    measured; tests/test_planner.py, test_first_matrix)."""
    # compiled: the tile's peak is COMPILE_PEAK times its plain one (encode_peak); the correction's
    # term, a pass of its own on the GPU, is not compiled.
    fit = _fit(padded, tiles, 1) * (COMPILE_PEAK if compiled else 1.0)
    corrected = SPLIT_FRAME_4K * SLICE * output[0] * output[1] / PIXELS_4K if split else 0
    above = fit + corrected if tiles is None else max(fit, corrected)
    return round(above) + VAE_WEIGHTS + dit_on_gpu(resident, swapped)


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
    (column 0) or the decode's (1), untiled (tiles None) or, tiled, the most a tile of its grid
    takes, each of the grid's shapes counted (tile_shapes, _tile).

    PROVISIONAL (implementation, 2026-10-12; DESIGN.md counts a pass by its largest tile): not
    the largest tile's alone. The grid's last tile along a side is shorter, and a shorter tile
    can take more: an encode moves in steps, not with the tile's area (MEASURED: 752x896 took
    8.09 GiB, 752x960 7.70), so that frames padded to 752x1792 in 960-px tiles, laid 752x960
    then 752x896, were counted 0.39 GiB under the second tile's own measure; and a tile nobody
    measured takes its rule and CUT_OFF_EXTRA, which may pass the measure of a tile little
    larger. The tiles run one after the other: the pass peaks where the one that takes the most
    does.

    A tile takes (DESIGN.md, Memory planner, Per-phase peaks: "the planner takes the measured
    peak of every tile measured ... A cut-off tile of another size takes the larger of its fit
    and of the squares' line at its area, and 0.3 GiB more"):
    - its measured peak where MEASURED has its rows and columns: every square the planner may
      choose, and the tiles cut off at 1080p's and 720p's rows;
    - else its rule (_rule), the larger of its fit at its area and of the line between the two
      measured squares that area lies between; and, cut off at the picture's edges, any tile
      that isn't the square of the tiles' side, CUT_OFF_EXTRA over it. A 1536-px tile at 1440p,
      1472x1536, takes 21.24 and 37.57 GiB:
      its fits' 20.94 and 37.27, over the lines' 20.60 and 37.11, and 0.3;
    - but a tile above TILE_MAX, which only a pin gives, its rule alone, square or cut off, as
      DESIGN.md has it (Pins: "A tile pin above 2048 px is taken by its fit, with a warning: no
      peak is measured there"; beyond_measured).
    PROVISIONAL (implementation, 2026-10-12), a choice DESIGN.md doesn't make: a measure serves
    the tile of its rows and columns alone, never the one of its columns and rows, which nobody
    measured: a tile of 1280 rows by 1,104 columns, an upright 1080p's, isn't the 1104x1280 one,
    and takes its rule and CUT_OFF_EXTRA.

    ValueError for tiles of which one covers the whole padded frame: no tiling, and no pass the
    planner makes or takes (covers); and for tiles the tiled passes lay no grid of, a side or
    an overlap that isn't a multiple of 8 px (tilegrid.grid), which no pin is (cli._tile) and a
    record is refused for (_check_tiles).

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
    untiled, _, small = _FITS[column]
    if tiles is None:
        area = _megapixels(padded)
        return max(constant + slope * area for constant, slope in (untiled, small)) * GIB
    if covers(padded, tiles):
        raise ValueError(
            f"tiles of {tiles.size} px over frames padded to {padded[1]}x{padded[0]}: one covers"
            " the whole frame, which is no tiling"
        )
    return max(_tile(tiles.size, *shape, column) for shape in tile_shapes(padded, tiles))


def _tile(side: int, rows: int, columns: int, column: int) -> float:
    """What one tile of rows x columns pixels takes above what is on the GPU before the call,
    in bytes, in a pass in tiles of `side` px: the encode's (column 0) or the decode's (1), its
    measure, or its rule and what a cut-off tile takes over it (_fit)."""
    if (rows, columns) in MEASURED:
        return MEASURED[rows, columns][column] * GIB
    rule = _rule(side, rows * columns, column) * GIB
    return rule + CUT_OFF_EXTRA if (rows, columns) != (side, side) and side <= TILE_MAX else rule


def _rule(side: int, pixels: int, column: int) -> float:
    """What a tile of `side` px a side and of `pixels` pixels takes where no measure has it, in
    GiB (_fit), before what a cut-off tile takes over it (CUT_OFF_EXTRA): the larger of its fit
    at its area, the refit's from REFITTED px a side and the small tiles' below, and of the
    squares' line there (_line). The encode's (column 0) or the decode's (1)."""
    _, tiled, small = _FITS[column]
    constant, slope = tiled if side >= REFITTED else small
    return max(constant + slope * pixels / 1e6, _line(pixels, column))


def _line(pixels: int, column: int) -> float:
    """The straight line between the peaks of the two measured squares (MEASURED) that a tile's
    area, in pixels, lies between, at that area, in GiB: the encode's (column 0) or the decode's
    (1); 0, under any fit, for an area outside the squares measured."""
    below = max((side for side in _SQUARES if side**2 <= pixels), default=None)
    above = min((side for side in _SQUARES if side**2 > pixels), default=None)
    if below is None or above is None:
        return 0.0
    share = (pixels - below**2) / (above**2 - below**2)
    low, high = MEASURED[below, below][column], MEASURED[above, above][column]
    return low + share * (high - low)


def in_compile_range(padded: tuple[int, int], phase: Phase, tiles: Tiles | None) -> bool:
    """Whether a VAE pass over padded frames is within the shapes measured compiled (COMPILE_*,
    DESIGN.md, Memory planner, decided on 2026-10-24): the encode's tiles up to
    COMPILE_ENCODE_TILE_MAX px, or untiled where the padded frame is no larger than the 1080p
    frame's, rows and columns each; the decode's tiles up to COMPILE_DECODE_TILE_MAX px, never
    untiled. The DiT's phase is no pass of the VAE: always within."""
    if phase == "encode":
        if tiles is None:
            return padded[0] <= COMPILE_ENCODE_WHOLE[0] and padded[1] <= COMPILE_ENCODE_WHOLE[1]
        return tiles.size <= COMPILE_ENCODE_TILE_MAX
    if phase == "decode":
        return tiles is not None and tiles.size <= COMPILE_DECODE_TILE_MAX
    return True


def compile_range_said(phase: str) -> str:
    """The shapes measured compiled for a VAE pass, in words (in_compile_range)."""
    if phase == "encode":
        rows, columns = COMPILE_ENCODE_WHOLE
        return (
            f"encode tiles up to {COMPILE_ENCODE_TILE_MAX} px, or untiled on a frame padded to"
            f" {columns}x{rows} at most"
        )
    return f"decode tiles up to {COMPILE_DECODE_TILE_MAX} px, never untiled"


def covers(padded: tuple[int, int], tiles: Tiles) -> bool:
    """Whether one tile covers the whole padded frame, its grid one tile, which is no tiling
    (DESIGN.md, Pipeline, per shot, step 3): a pin of one is refused, naming 0, the untiled pass
    (plan), a record of one is no plan's (check_recorded), and the planner never makes one
    (_tile_sizes). The tiled passes would run such a grid untiled (tiling.encode_tiled,
    tiling.decode_tiled, as numz's take it, attn_video_vae.py:1315-1317, 1489-1491)."""
    return tiles.size >= max(padded)


def scratch(padded: tuple[int, int], frames: int) -> int:
    """The scratch of a tiled pass over a shot of `frames` frames of padded frames, in bytes
    (tilegrid.scratch_bytes: 3 channels in bfloat16, 12.7 MB per 1080p frame and 50.5 MB per 4K
    frame; DESIGN.md, Pipeline, per shot, step 3), the shot padded to 4n + 1 as the VAE takes it
    (frames.padded_length). The tiled encode's, its input kept, and the tiled decode's, its frames
    summed, are the same store (scratch.Scratch), one at a time."""
    return tilegrid.scratch_bytes(padded_length(frames), padded)
