"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure. Nor torch at import: the command line imports this module before its first refusals,
which don't wait for torch (tests/test_cli.py).

Its memory model: what each phase is predicted to peak at on the GPU, in bytes. The DiT's and
the VAE's passes by the fits of research/docs/planner-limits.md and research/docs/vram.md,
measured on numz's CLI with the 7B fp16, the one DiT seedvr2x runs (weights.ACCEPTED); a tile
by the peak measured on seedvr2x's own tiled passes (MEASURED), the fits serving the tiles cut
off at the picture's edges at a size nobody measured. No VAE pass keeps anything of the shot on
the GPU (runtime/shot.py, runtime/tiling.py), and none is counted by its shot's length. Nor is
the DiT's phase, judged by its longest window alone: it keeps nothing of a shot beside the
window it runs, the shot's noise drawn in system memory (shot.sample_windows; DESIGN.md,
Pipeline step 2). Where the weights are in each phase, and what that takes, by what
runtime/blockswap.py does with them, measured on seedvr2x itself (milestone 3). The tiles and
their scratch by runtime/tilegrid.py's, the arithmetic the tiled passes run on. Of system memory:
the pinned copies of what leaves the GPU, what the run holds there of a shot at the moment it
holds the most, the job's shot that holds the most, by the tensors' sizes (host_moments), and
what the scratch may take of what is left (Plan.scratch_allowance). Sizes are (height, width)
in pixels; `padded` is the frame the model takes, the target padded (job.padded_size).

Its choices, in DESIGN.md's order (plan): the window, the biggest quality lever; the fewest
swapped blocks that hold it; the largest VAE tiles, none where the untiled pass fits; and where
the plan so made takes more system memory than the host has, the VAE's weights kept on the GPU
for the DiT's phase, then the DiT kept there for the VAE's phases and their tiles shrunk, then the
longest window whose plan fits both. A pin is taken exactly or refused, the refusal naming what
fits; a size that doesn't fit is refused, never degraded (PlanError).

Four corrections come from the planner's first two matrices on seedvr2x, emulated cards
(tests/test_planner.py, test_first_matrix). Two are DESIGN.md's: no tile chosen above TILE_MAX
(Memory planner, What follows from them), and every VAE pass held to a room under the budget
besides its peak (VAE_ROOM), judged by the frames against a run with room to spare (Memory
planner, Budget; To measure, The VAE's room). PROVISIONAL (implementation, 2026-10-11), the two
others: ENCODE_EXTRA on every encode, until the encode is refitted on seedvr2x, and split's term
off a tiled decode's peak (decode_peak)."""

import logging
import math
import mmap
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any, Literal

from seedvr2x.runtime import memory, tilegrid, weights
from seedvr2x.runtime.job import (
    QUALITY,
    SHARED,
    JobError,
    Shot,
    check_target,
    kept_bytes,
    output_size,
    padded_length,
    padded_size,
    shot_runs,
    window_layout,
)
from seedvr2x.runtime.memory import Reading
from seedvr2x.runtime.plan import Phase, Plan, Shapes, Tiles

# The DiT's own window functions, the 7B's: plain Python, and the packages above them hold
# nothing, so importing them brings in no torch.
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)

logger = logging.getLogger(__name__)

GIB = 1024**3
MIB = 1024**2

# What the plan leaves of the free memory the driver reports, the budget being free - MARGIN
# (DESIGN.md, Memory planner, Budget): bisected on emulated cards, a run needs 0.15-0.45 GiB
# between the free memory it starts with and the torch peak of its bounding phase, the
# decode-bound 8 GB card the most (it failed with 0.37); 0.6 keeps 0.23 over that failure
# (planner-limits.md, The margin over free memory). Measured where the driver refuses a full
# card, and a driver's alone: a run held to its card, whose driver gives memory the card doesn't
# have, counts on memory.RESERVE less already, which measures the same need under torch's own
# allocator, and takes no margin on top (_Job.margin; DESIGN.md, Allocator, 2026-10-12).
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
# over (1104x1536). tests/test_planner.py holds the 20 tiles' 40 peaks to it. DESIGN.md's 0.3
# GiB (Memory planner, Per-phase peaks).
CUT_OFF_EXTRA = round(0.3 * GIB)

# A latent as the VAE makes it and the DiT takes it: its channels, the pixels of the padded
# frame along a side of one of its positions (vendor/models/video_vae_v3/
# s8_c16_t4_inflation_sd3.yaml: latent_channels 16, spatial_downsample_factor 8), and the bytes
# of one of its values, in the pipeline's bfloat16 (model.COMPUTE_DTYPE; tests/test_planner.py
# holds the three to tensors of that shape). What host_moments counts a shot's latent and its
# noise by in system memory, their sizes: literals here since runtime/model.py imports torch.
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
# its predicted peak, in bytes: 1.85 GiB under the free memory with the margin (DESIGN.md,
# Memory planner, Budget, decided on 2026-10-18; To measure, The VAE's room). A pass packed
# closer than about 1 GiB to its peak can run and give other frames, and say nothing: a
# convolution whose workspace the allocator refuses runs on cuDNN's next engine, kept for the
# process's life (vae-room.md). Judged by the frames against a run with room to spare, on the
# 96 GB card emulating cards of 12 to 48 GB under cudaMallocAsync (2026-10-16 to 18): a pass
# alone kept its frames with 0.97 GiB or less left (decodes 0.67 to 0.97, encodes their latent
# with 0.87 or less), a whole run, the encode, the windows and the decode in one process, with
# 1.08 or less (six tiles; near that floor two runs at one card didn't always agree). So 1.85
# is 0.77 over the most a whole run was seen to need and 0.88 over the most a pass alone needs;
# the 0.25 kept over the 1.0 those measures allow is for what wasn't measured: whole runs in
# other tiles, cards not emulated, other GPU generations and cuDNN builds, whose engines and
# workspaces are their own. An encode is held 0.4 GiB more besides (ENCODE_EXTRA). Before, 1.5
# (2026-10-11), a guard while its cause wasn't known: the large passes refused at
# memory_limit_conv's concatenation, which the one-output convolution spares most of them
# (MEASURED).
VAE_ROOM = round(1.25 * GIB)

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
# The compilation's time, per process, said by --plan and the plan's line: 4 to 6 min on an RTX
# 5090 (DESIGN.md, same section), paid before the DiT loads. Said as the range, before anything
# is compiled; the estimates that follow count the seconds the load measured, the compilation's
# among them, once (cli._say_times): there is no figure to sum before the load. PROVISIONAL
# (impl-v010, 2026-10-29: DESIGN.md gives the range, measured on one card).
COMPILE_MIN = 4 * 60
COMPILE_MAX = 6 * 60
# The slices a compiled shape is warmed with are the shot's: one of 1 or 2 latents is a single
# slice, the encode's and the decode's memory state off, one of 3 or more has a first slice and
# the next ones (model.encode_slices, model.latent_slices), so that 3 latents show every kind
# (compile_shapes).
COMPILE_LATENTS = 3
# The kinds the time estimates' calibration meets besides the job's own (runtime/calibration.py,
# FIRST and FRAMES): an encode and a decode of 5 frames, 2 latents, one slice, and of 13 frames,
# 4 latents, a first slice and next ones (kind 3). Compiled with the rest, so that the
# calibration, which times its passes, never pays for a compilation: it would put it into the
# models of the encode and the decode (tests/test_compile.py holds them to the calibration's).
COMPILE_CALIBRATED = (2, 3)

# The process's resident system memory during the shots, which stays flat: 2.3-2.4 GiB over 6
# shots (DESIGN.md, Memory planner, Host RAM), short ones: what the run holds of a long shot
# there is counted apart (host_moments).
HOST_RESIDENT = round(2.4 * GIB)
# What the check of a tensor's values for NaN and inf holds in system memory over a tensor it
# reads there, an encode's latent (shot.finite: torch.isfinite, then all), a latent of it at a
# time: that latent's absolute values, as large as it, and CHECK_MASKS masks of a byte a value,
# the two tests' and their product. For bfloat16, 2.5 times the latent's bytes, by torch's
# memory profiler on shots of 40 and of 200 latents (torch 2.14.1, 2026-10-12;
# tests/test_shot.py measures it).
CHECK_MASKS = 3
# What a join of two windows holds in system memory while their shared latents are mixed
# (shot._merge), in tensors as large as those latents: the two products and their sum, given up
# before the next join (torch's memory profiler, 2026-10-12; tests/test_shot.py measures it).
MIXING = 3

# The most a scratch may take of system memory when what the host has available can't be read
# (cli._host_available), in bytes. PROVISIONAL (implementation, 2026-10-11): DESIGN.md sets the
# allowance from the host's memory, and gives no figure without it: 8 GiB, which holds 675 frames
# at 1080p and 170 at 4K, what every scratch was allowed before the planner set it.
ALLOWANCE_UNKNOWN_HOST = 8 * GIB


class PlanError(JobError):
    """A job the planner refuses before the run: a pin it can't take exactly, or a size that
    doesn't fit the memory free. Its message counts in free VRAM, never in card sizes, and
    names what is there for what it is (_there): what the driver reports free, or, on a run
    held to the card, what the plan counts on, of which figure, and what would have to be freed
    (_to_free). It gives what DESIGN.md has each refusal give (Memory planner: A size that
    doesn't fit is refused, Pins, A plan shapes the output a little, Host RAM), in GiB. Of the
    GPU's memory, what is taken is rounded up and what is there rounded down; of the system's,
    each part is said to the nearest hundredth, as the plan's lines say it, and the total is
    their sum (_host_hundredths).

    `instead`: where a pin is refused, the pin its message ends with, the one that fits
    ("--window 6"), for a plan made with the tight reserve to name in its budget line (said;
    design, 2026-10-21); else empty.

    `shortfall`: where a resume's recorded shape no longer fits the memory free, what it is
    short of, for whoever words that refusal in plain words with the figures apart
    (cli._kept_refusal; DESIGN.md, Memory planner, A plan shapes the output a little, decided
    on 2026-10-28); else None. The message stays the figures' full sentence.

    `phase`: where the refusal is for want of memory, the phase that doesn't fit ("encode", "dit"
    or "decode") or "host" for the system's, for plan_again to know which part of a shape to
    release (DESIGN.md, Memory planner, A plan shapes the output a little, decided on
    2026-10-28); else empty (a pin that is no pin, a card too small for any plan)."""

    def __init__(
        self,
        message: str,
        instead: str = "",
        shortfall: "Shortfall | None" = None,
        phase: str = "",
    ) -> None:
        super().__init__(message)
        self.instead = instead
        self.shortfall = shortfall
        self.phase = phase


@dataclass(frozen=True)
class Shortfall:
    """What a recorded shape is short of, as a refusal says it first (PlanError.shortfall):
    - what: the phase that doesn't fit, in words ("the encode", "the DiT's phase", "the
      decode"), or "" where it is the system's memory that falls short (host);
    - short: how much, in GiB to 2 decimals as the message's figures say it (_short), for the
      GPU's memory;
    - host: the system's memory is what falls short, not the GPU's;
    - freeable: whether programs closed on the card give the shape its memory back (_freeable),
      or a driver that refuses reports what they leave free; False where the card, whole, has
      no more to give a run held to it."""

    what: str
    short: str
    host: bool = False
    freeable: bool = True


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
    return round(_dit_least(padded, latents, swapped, vae_in_dit, attention) + windows)


def _dit_least(
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


# A shot's moments in system memory, in its order (host_moments).
MOMENTS = ("encode", "check", "windows", "merge", "decode")


def host_moments(padded: tuple[int, int], latents: int, cap: int | None) -> dict[str, int]:
    """What the run holds in system memory of a shot of `latents` latents of padded frames, in
    bytes, at each of its moments (MOMENTS), its windows capped at `cap` latents (None: one
    window). The GPU keeps nothing of a shot between its units (runtime/units.py,
    shot.sample_windows): it waits here, and its noise is drawn here.

    By the tensors' sizes: a latent is (H / 8, W / 8, 16) in bfloat16, half a byte a padded
    pixel, 1.01 MiB at 1080p, padded to 1104x1920, and 4.01 MiB at 4K, padded to 2192x3840. The
    moments, counted in latents, a shot of L in n windows, which share SHARED latents with each
    neighbour, L + SHARED x (n - 1) latents in all:
    - "encode", the encode's end: the latent and the buffer it is made from, as large, the
      posterior's mode or a tiled encode's sums (model.to_latent): 2 L;
    - "check", the latent checked for NaN and inf where the encode left it (run._sample,
      shot.finite), a latent at a time: the latent, and of the one being read its absolute
      values and CHECK_MASKS masks of a byte a value: L + 2.5;
    - "windows", from the noise's draw to the last window: the latent and the noise, drawn
      beside it in place (2 L), and each window's copy as it leaves the GPU, gone once written
      to its file (units.Units.save_window), so the longest window's there;
    - "merge", the decode's start: the windows, read back from their files, and what merging
      them makes (shot.merge_windows): nothing for one window, which is the merged latent; else
      the merged latent, allocated once (L), and at a join the two products and the sum of the
      latents mixed (MIXING x SHARED), given up before the next join;
    - "decode", through the decode: the merged latent, L.
    The latent and the noise are gone before the merge (run._sample), and the windows after it:
    the moments follow one another, and a shot holds the largest, never their sum. One shot at
    a time: a segment's shots are sampled, then decoded, each one's tensors given up before the
    next one's (run.run_job).

    The largest is the windows', 3 L for a shot in one window, 1.5 bytes a padded pixel and a
    latent; the check for a shot of 1 latent; and the merge where a shot's n windows are
    shorter than 2 n + 4 latents, sharing more latents between them than one of them holds: a
    long shot, from 9 windows on under a cap of 20 latents.

    The moments on the CPU alone are measured, by torch's memory profiler (tests/test_shot.py,
    2026-10-12): the check, 2.5 times one latent over the shot's; the merge, this count to the
    few bytes of the mix's weights, for shots of 13 to 100 latents in 1 to 33 windows; the
    encode's end, the latent's memory beside its buffer; a window's file is written from its
    copy and read into one tensor, nothing more allocated. The others move tensors between the
    GPU and system memory, read from the code: a copy each (shot.moved).

    Not here: a tiled pass's scratch, which the plan allows apart (Plan.scratch_allowance). It
    is open through a tiled encode's tiles, beside the sums alone (L), and through a tiled
    decode, beside the merged latent alone (L): never at the moment a shot holds the most."""
    height, width = padded
    latent = LATENT_CHANNELS * LATENT_VALUE * (height // LATENT_SIDE) * (width // LATENT_SIDE)
    if cap is None or cap >= latents:
        windows = [latents]
    else:
        windows = [end - start for start, end in window_layout(latents, cap)]
    kept = sum(windows)
    merging = latents + MIXING * SHARED if len(windows) > 1 else 0
    shot = latent * latents
    return {
        "encode": 2 * shot,
        "check": shot + latent + CHECK_MASKS * latent // LATENT_VALUE,
        "windows": 2 * shot + latent * max(windows),
        "merge": latent * (kept + merging),
        "decode": shot,
    }


def compile_shapes(
    plan: Plan,
    target: tuple[int, int],
    shots: Sequence[Shot],
    runs: int,
    *,
    numz_padding: bool = False,
) -> Shapes:
    """The shapes the plan's VAE passes meet, for the compiled VAE to be warmed with before the
    DiT loads (runtime/compiling.py): one per distinct tile of each pass (_shapes), the whole
    padded frame untiled, and the slice kinds of the job's runs, `runs` a shot (job.QUALITY):
    a run of 1 latent is a slice of 1 frame, of 2 a slice of 5, of 3 or more a first slice of 5
    and the next ones of 4 (model.encode_slices, model.latent_slices), so that the shortest of
    each kind, up to COMPILE_LATENTS, shows them all. A compiled pass compiles once for each
    shape, for each of these (DESIGN.md: "a warm call per shape the plan uses ... at the job's
    slice lengths"), so that no unit of the run compiles; and the calibration's own
    (COMPILE_CALIBRATED), which times the passes right after, a job whose shots are all long
    meeting the 2 latents of its 5-frame passes there only."""
    padded = _padded(target, numz_padding)

    def cut(tiles: Tiles | None) -> tuple[tuple[int, int], ...]:
        return (padded,) if tiles is None else _shapes(padded, tiles)

    latents = sorted(
        {
            *COMPILE_CALIBRATED,
            *(
                min(_latents(run.frames), COMPILE_LATENTS)
                for shot in shots
                for run in shot_runs(shot, runs)
            ),
        }
    )
    return Shapes(padded, cut(plan.encode_tiles), cut(plan.decode_tiles), tuple(latents))


@cache
def _shapes(padded: tuple[int, int], tiles: Tiles) -> tuple[tuple[int, int], ...]:
    """(rows, columns), in pixels, of the tiles of a tiled pass over padded frames, each shape
    once, the largest first: the tiled passes' own grid (tilegrid.grid), whose last tile along
    a side is the shorter one, what the stride leaves of the picture. The first is the square
    cut off at the picture's edges (DESIGN.md, Memory planner: a 1280-px tile at 1080p is
    1104x1280), the one tilegrid.largest gives the area of; not always the tile that takes the
    most (_fit)."""
    laid = tilegrid.grid(tilegrid.latent_size(padded), tiles)
    shapes = {(tile.rows * tilegrid.SCALE, tile.columns * tilegrid.SCALE) for tile in laid}
    return tuple(sorted(shapes, key=lambda shape: (-shape[0] * shape[1], shape)))


def _grid(padded: tuple[int, int], tiles: Tiles) -> tuple[int, int]:
    """(rows, columns) of the tiles of a tiled pass over padded frames, as the plan's lines say
    them: the tiled passes' own grid, counted (tilegrid.grid; numz's, DESIGN.md, Pipeline, per
    shot, step 3). 2160x3840 takes 3 x 4 tiles of 1024 px and 2 x 2 of 2048, overlapping by
    plan.OVERLAP, 64 (planner-limits.md, Tiled VAE at 4K, measured it at 128: 3 x 5 of 1024); a
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
    at once, SLICE frames at most of the output's size (job.output_size). ValueError for tiles
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
    takes, each of the grid's shapes counted (_shapes, _tile).

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
    return max(_tile(tiles.size, *shape, column) for shape in _shapes(padded, tiles))


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


def _compile_range_said(phase: str) -> str:
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
    (job.padded_length). The tiled encode's, its input kept, and the tiled decode's, its frames
    summed, are the same store (tiling.Scratch), one at a time."""
    return tilegrid.scratch_bytes(padded_length(frames), padded)


def scratch_file(
    plan: Plan, target: tuple[int, int], frames: int, *, numz_padding: bool = False
) -> int:
    """What the scratch of a shot of `frames` frames takes on disk under plan, in bytes: its
    memory-mapped file in the shot's unit directory (tiling.Scratch) where the plan tiles a pass
    and the scratch is over the plan's allowance, as tiling.Scratch.where decides; 0 where no pass
    is tiled or the scratch is held in system memory. The encode's and the decode's are the same
    store, one at a time (scratch); a shot's first run is its longest (job.shot_runs)."""
    if plan.encode_tiles is None and plan.decode_tiles is None:
        return 0
    size = scratch(_padded(target, numz_padding), frames)
    return size if size > plan.scratch_allowance else 0


# The phases, in a shot's order.
PHASES: tuple[Phase, ...] = ("encode", "dit", "decode")


@dataclass(frozen=True)
class Shortened:
    """A plan whose window the planner shortened for the host's memory (plan): the cap the card
    alone would have given, in latents, what the plan under it takes of system memory and what
    is available, in hundredths of a GiB, as its refusal would have said them (_unfit)."""

    window: int
    taken: int
    available: int


@dataclass(frozen=True)
class Shrunk:
    """A plan whose VAE tiles the planner shrank for the host's memory, the DiT kept on the GPU
    in the VAE's phases (_dit_kept): the encode's and the decode's tiles the card alone would
    have given at its window (None: untiled), what the plan with them takes of system memory and
    what is available, in hundredths of a GiB, as its refusal would have said them (_unfit)."""

    encode: Tiles | None
    decode: Tiles | None
    taken: int
    available: int


@dataclass(frozen=True)
class Peaks:
    """What a plan is predicted to take, in bytes where nothing else is said: which phase bounds
    it, and by how much, is what --plan prints and the manifest keeps for information."""

    free: int  # the memory the plan was made in: what it may count its peaks in (Reading.counted)
    budget: int  # free - MARGIN; free itself on a run held to the card (_Job.margin)
    encode: int  # each phase's predicted peak on the GPU
    dit: int  # the longest window's (dit_peak)
    decode: int
    host: int  # the system memory the plan takes, a shot's (`shot`) and the scratch's in it
    scratch: int  # the scratch of the longest shot's tiled passes; 0 untiled
    window: int  # the longest window the job really runs, in latents
    tokens: int  # the DiT's tokens per latent
    # How `free` was read: what the driver reported free to the process, what the whole card had
    # free where that was read, and whether the run is held to the card (runtime/memory.py).
    reading: Reading
    attention: str  # the attention backend the DiT's peak is the fit of (DIT_ATTENTION)
    # What the run holds of a shot in system memory at the moment it holds the most, and that
    # moment, one of MOMENTS (host_moments): the job's shot that holds the most, of `holder`
    # latents, the longest but where a shorter one in fewer windows holds more (_shot_held);
    # counted in `host`.
    shot: int
    moment: str
    holder: int
    # Where the planner took a shorter window than the card alone holds, for the host's memory:
    # what the card alone would have given, and why not (plan); else None.
    shortened: Shortened | None = None
    # Where the planner shrank the VAE's tiles at that window for the host's memory, the DiT
    # kept on the GPU in their phases: the tiles the card alone would have given, and why not
    # (_dit_kept); else None.
    shrunk: Shrunk | None = None
    # Where a held run's plan was made with memory.RESERVE_TIGHT, the plan with memory.RESERVE
    # refused (plan): that refusal, for the budget's line; else None. And where that refusal was
    # a pin's, the pin that runs with memory.RESERVE (PlanError.instead); else empty.
    tightened: str | None = None
    instead: str = ""

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
        latents. `free` is the memory the plan was made in; beside it, what the driver reported
        free to the process (driver_free, the same figure on a driver that refuses) and what the
        whole card had free (card_free; null where it wasn't read), and the attention backend
        the DiT's peak was predicted for. Of `host`, the system memory the plan takes, host_shot
        is one shot's part, the shot that holds the most (Peaks.shot). PROVISIONAL
        (implementation, 2026-10-10 and 12): the fields' names; and a window shortened or tiles
        shrunk for the host's memory (Peaks.shortened, Peaks.shrunk) aren't recorded: the plan's
        lines say them."""
        return {
            "free": self.free,
            "driver_free": self.reading.reported,
            "card_free": self.reading.card,
            # What a held run's plan leaves torch's allocator beyond a step's peak; null on a
            # driver that refuses, which leaves none.
            "reserve": self.reading.reserve if self.reading.held else None,
            "attention": self.attention,
            "budget": self.budget,
            "encode_peak": self.encode,
            "dit_peak": self.dit,
            "decode_peak": self.decode,
            "bound": self.bound,
            "host": self.host,
            "host_shot": self.shot,
            "scratch": self.scratch,
            "longest_window": self.window,
        }


@dataclass(frozen=True)
class _Job:
    """What the planner counts a job by: the memory it may count on, its size, its shots'
    lengths, and the attention backend its DiT runs on."""

    free: int  # bytes: what the plan may count its peaks in (reading.counted)
    target: tuple[int, int]  # job.target_size
    lengths: tuple[int, ...]  # its shots' lengths in latents, each once, ascending
    frames: int  # its longest shot's frames
    split: bool
    numz_padding: bool
    reading: Reading  # how `free` was read, for the plan's words and its record
    attention: str  # DIT_ATTENTION's
    compile_vae: bool = False  # the VAE compiled (--compile): its passes at COMPILE_PEAK

    @property
    def margin(self) -> int:
        """What the plan leaves of the memory it may count on, in bytes: MARGIN on a driver that
        refuses; nothing on a run held to the card, where memory.RESERVE, taken off already,
        measures the same need (DESIGN.md, Allocator: "the planner's 0.6 GiB margin is counted
        inside the 4.5"): the budget is then what the plan counts on."""
        return 0 if self.reading.held else MARGIN

    @property
    def budget(self) -> int:
        return self.free - self.margin

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
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    split: bool = False,
    numz_padding: bool = False,
    recorded: Plan | None = None,
    attention: str = FLASH_ATTN_2,
    compile_vae: bool = False,
    compile_fixed: bool = False,
) -> tuple[Plan, Peaks]:
    """The plan of a job of `shots` resized to target (job.target_size), in the GPU memory it may
    count on, less its margin, and what it is predicted to take; refused (PlanError) when a pin
    can't be taken exactly, when no plan fits that memory, or when the plan takes more system
    memory than host_available, in bytes (None: not checked). split: whether the decode corrects
    colours; numz_padding is for tests only (cli.NUMZ_PADDING). An output directory and a .mkv
    path plan alike: both jobs keep their units in a directory (runtime/units.py).

    `free` is that memory: a number of bytes, what a driver that refuses reports free; or the
    reading the run made of it (memory.Reading), whose `counted` the plan is made in, and whose
    other figures only word the plan's first line, its refusals and its record: on a run held
    to the card they name what the driver reports, what the whole card has free where it was
    read, and what the plan counts on. The margin is MARGIN on a driver that refuses, and none
    on a run held to the card, whose `counted` is under torch's limit by what measures the same
    need already (_Job.margin): the budget is then all the plan counts on. attention: the
    backend the run's DiT takes (model.attention_backend), whose fit the DiT's peak is
    predicted by (DIT_ATTENTION).

    The DiT's phase is judged by the job's longest window alone (dit_peak; _dit_need): nothing
    of a shot is on the GPU beside the window it runs, its noise drawn in system memory
    (shot.sample_windows), so that no shot's length bounds the phase. Until 2026-10-16 the noise
    was drawn on the GPU, the shot's latent and noise whole there before its first window, and
    the phase was judged by the larger of that draw and the window, which bounded a shot's
    length on every card.

    A tile pinned that covers the whole padded frame is no tiling, and is refused, naming 0, the
    untiled pass (DESIGN.md, Pipeline, per shot, step 3), whatever the memory.

    The choices, each made once for the job (DESIGN.md, Memory planner, Choices, in order):
    1. the window cap, the longest shot's length when the DiT's phase holds it, with as many
       blocks swapped as it takes and the VAE's weights away, else the largest cap whose longest
       window it holds: what it must hold is the longest window the job really runs, a shot of
       80 latents under a cap of 76 running two of 41; the plan's cap is then that longest
       window, which lays the shots out the same (_window);
    2. the fewest swapped blocks that hold that window, the VAE leaving the DiT's phase before
       one more block is swapped; but where the plan so made takes more system memory than
       host_available, more blocks swapped and the VAE staying, when that takes less of it
       (_vae_kept);
    3. the VAE's tiles, the decode's then the encode's: none where the untiled pass fits, else
       the largest that fit, TILE_MAX at most, each pass with its room (VAE_ROOM), the DiT
       leaving the phase before a tile shrinks; then as much of it left there as fits
       (_resident); but where the plan so made, the VAE's weights kept or not, takes more system
       memory than host_available, the DiT kept on the GPU in the VAE's phases, a block more at
       a time, and their tiles shrunk to keep it, the largest that fit the host too (_dit_kept;
       design's decision of 2026-10-13); the plan's lines say which tiles the card alone gives
       and why smaller ones are taken (Peaks.shrunk). Never a pinned pass's tiles, nor a
       recorded plan's, which shape the output.
    Then the system memory: what the run holds there of a shot at the moment it holds the most
    (_shot_held), counted with the pinned copies, and what a tiled shot's scratch may take of
    what host_available has left once both are counted (_allowance).

    Where the plan so made fits the card and takes more system memory than host_available, its
    tiles shrunk as far as they go, the planner takes by itself the longest window whose plan
    fits both, the other choices made again under it (_window_that_runs), and the plan's lines
    say which window the card alone would have given and why a shorter one was taken
    (Peaks.shortened; DESIGN.md, Memory planner, Choices: design's decision of 2026-10-12): a
    long window may take swapped blocks, each pinned in system memory, where a shorter one takes
    none. Never under --window, a pin being taken exactly or refused, its refusal naming that
    window; nor for a recorded plan, whose window shapes the output. A job is refused for the
    system memory only where no window's plan fits both, its tiles shrunk as far as they go.

    recorded is the plan of a job resumed that has made units: what shapes the output is taken
    from it as it is, the cap, the tiles and the compile flags, and where the weights are and
    the scratch's allowance planned again around it (DESIGN.md, Memory planner, A plan shapes
    the output a little); one with tiles of which one covers the frame is no plan's, refused
    (check_recorded). Where the weights are is planned for the backend given, a resume's too:
    FlashAttention installed since, or gone, the recorded shape is held to today's fit.

    On a run held to the card, a plan so made that is refused, a new job's or a resume's recorded
    shape, whatever refuses it, is made again with the reserve of memory.RESERVE_TIGHT in
    memory.RESERVE's place (memory.tightened), the plan's last resort, slower, as its budget line
    says (Peaks.tightened; DESIGN.md, Allocator, decided on 2026-10-13, for a resume's plan on
    2026-10-21); the VAE's room is in the budget either way. Refused with it too, the job is
    refused with that plan's figures, the closest to running, and the refusal says both reserves
    were tried. Only under torch's own allocator with expandable segments (Reading.expandable),
    under which a step short of room slows down and runs (DESIGN.md, Allocator: "With a reserve
    of 2.5 GiB under those segments"): under a user's allocator without them such a step is
    refused, in a unit, perhaps hours in, and under cudaMallocAsync nothing checks the card (a
    review's finding; DESIGN.md, Allocator, decided on 2026-10-21: "only under the segments, where
    the 2.5 was measured"). Never on a driver that refuses,
    which leaves no reserve; nor for a pin that is no pin, refused before any memory is counted.
    The reserves are memory's, read as the plan is made.

    compile_vae: --compile, the VAE compiled (DESIGN.md, Memory planner, "torch.compile: the
    VAE, opt-in, in v0.1.0"): each VAE pass is held to COMPILE_PEAK times its tile's plain peak
    (encode_peak), so that the tiles are chosen smaller where it doesn't fit, and only passes
    within the shapes measured compiled are made or taken (in_compile_range): the encode's tiles
    up to COMPILE_ENCODE_TILE_MAX px or the 1080p frame whole, the decode's up to
    COMPILE_DECODE_TILE_MAX px. Where a pin gives a pass beyond them, or no compiled plan fits
    where a plain one would, the job is refused, never run plain in its place (DESIGN.md, same
    section, decided on 2026-10-29): the refusal names the pass and the shapes measured and
    advises, each on a line of its own, the same command without --compile and, for a pin, the
    pin within them (_compile_refusal). A job no plain plan fits is refused as it was, whatever
    --compile asked. The plan's compile_vae is the setting's, always. A recorded plan keeps its
    own compile_vae, whatever is asked (the units made are what they are), and is refused as
    any recorded shape; compile_fixed is the same for a plan planned again around a recorded
    shape (plan_again), and for the compiled plan itself, whose refusal is _compile_refusal's."""
    if recorded is not None:
        compile_vae, compile_fixed = recorded.compile_vae, True
    if compile_vae and not compile_fixed:
        try:
            return plan(
                free,
                target,
                shots,
                pins,
                host_available,
                split=split,
                numz_padding=numz_padding,
                recorded=None,
                attention=attention,
                compile_vae=True,
                compile_fixed=True,
            )
        except PlanError as refused:
            raise _compile_refusal(
                refused,
                (free, target, shots, pins, host_available),
                dict(split=split, numz_padding=numz_padding, attention=attention),
            ) from None
    job = _job(free, target, shots, split, numz_padding, attention, compile_vae)
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
    try:
        return _planned(job, pins, recorded, host_available)
    except PlanError as error:
        reading = job.reading
        if not reading.held or reading.tight or memory.RESERVE_TIGHT >= reading.reserve:
            raise
        refused, instead, shortfall = str(error), error.instead, error.shortfall
        phase = error.phase
    tight = memory.tightened(job.reading)
    if not job.reading.expandable:
        # Refused with the usual reserve, as before, saying what torch's default setting would
        # do where it would run the job (design, 2026-10-21).
        try:
            _planned(
                replace(job, free=tight.counted, reading=tight), pins, recorded, host_available
            )
        except PlanError:
            raise PlanError(refused, instead=instead, shortfall=shortfall, phase=phase) from None
        raise PlanError(
            f"{refused}; torch's own allocator with its expandable segments, a held run's default,"
            " which the allocator's setting in the environment replaces here, would run it with"
            f" the reserve of {_gib(memory.RESERVE_TIGHT)} GiB, slower",
            instead=instead,
            shortfall=shortfall,
            phase=phase,
        ) from None
    try:
        made, peaks = _planned(
            replace(job, free=tight.counted, reading=tight), pins, recorded, host_available
        )
    except PlanError as error:
        raise PlanError(
            f"{error}; refused with the reserve of {_gib(memory.RESERVE)} GiB that a held run's"
            f" plan leaves torch's allocator beyond a step's peak, and with"
            f" {_gib(memory.RESERVE_TIGHT)},"
            " the one it takes where that is refused, whose figures these are",
            shortfall=error.shortfall,
            phase=error.phase,
        ) from None
    if instead:
        # Named only where that pin, in its place, runs with the usual reserve: another pin, or
        # a pass the blocks it changes move, can still refuse it (the review, 2026-10-21).
        substituted = _substituted(pins, instead)
        try:
            if substituted is None:
                raise PlanError(instead)
            _planned(job, substituted, recorded, host_available)
        except PlanError:
            instead = ""
    # The refusal the budget's second line names: said whole at debug, not in the line (DESIGN.md,
    # Architecture, What it prints, decided on 2026-10-28).
    logger.debug(
        "the plan with the reserve of %s GiB was refused: %s", _gib(memory.RESERVE), refused
    )
    return made, replace(peaks, tightened=refused, instead=instead)


# The parts of a shape that plan_again releases for a refused phase, in order (Pins' fields): the
# phase's own, and for the system's memory, which no phase names, the tiles before the window
# (the planner's own order for the host, _dit_kept before _window_that_runs: DESIGN.md, Memory
# planner, Choices, decided on 2026-10-13, "before the window shortens, the DiT stays on the GPU
# for the VAE's phases and the tiles shrink"). PROVISIONAL (impl-v010, 2026-10-28): DESIGN.md
# names the phase's part ("the component of the phase the refusal names", the brief's design),
# not what the system's memory releases.
RELEASED = {
    "encode": ("encode_tile",),
    "dit": ("window",),
    "decode": ("decode_tile",),
    "host": ("decode_tile", "encode_tile", "window"),
}


def plan_again(
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    host_available: int | None,
    *,
    recorded: Plan,
    refused: PlanError,
    split: bool = False,
    numz_padding: bool = False,
    attention: str = FLASH_ATTN_2,
) -> tuple[Plan, Peaks]:
    """The plan of a job resumed whose recorded shape no longer fits the memory free (`refused`,
    plan's refusal of `recorded`), planned again for what is left (DESIGN.md, Memory planner, A
    plan shapes the output a little, decided on 2026-10-28, with --accept-env-change): what still
    fits of the shape is kept, the rest is planned as a new job's is.

    The recorded cap and tiles are pinned (a pin is taken exactly or refused: `Pins`), the
    command line's own pins as given. While the plan is refused, the part of the shape the
    refusal's phase names is released (PlanError.phase; RELEASED: the encode's tiles, the decode's
    tiles, the window cap, the system's memory taking the tiles first), unless the command line
    pins it, and the plan made again; at most three rounds, one part each. Refused with nothing
    left to release, `refused` itself when none was released, else the last round's refusal, a new
    job's.

    The result is a plan whose parts are the recorded ones or the planner's own (Plan.window,
    Plan.encode_tiles, Plan.decode_tiles): the caller compares them. The planning is plan's, the
    reserves and the tight reserve included, so that what is refused here is refused as any new job
    is."""
    held = Pins(
        pins.window if pins.window is not None else recorded.window,
        pins.encode_tile if pins.encode_tile is not None else _tile_pin(recorded.encode_tiles),
        pins.decode_tile if pins.decode_tile is not None else _tile_pin(recorded.decode_tiles),
        pins.swap_blocks,
    )
    # The parts the command line leaves to the planner, which are the ones that can be released.
    open_to = {
        "window": pins.window is None and recorded.window is not None,
        "encode_tile": pins.encode_tile is None,
        "decode_tile": pins.decode_tile is None,
    }
    error = refused
    for _ in range(len(open_to)):
        part = next((part for part in RELEASED.get(error.phase, ()) if open_to.get(part)), None)
        if part is None:
            raise error
        open_to[part] = False
        held = replace(held, **{part: None})
        try:
            return plan(
                free,
                target,
                shots,
                held,
                host_available,
                split=split,
                numz_padding=numz_padding,
                attention=attention,
                compile_vae=recorded.compile_vae,
                compile_fixed=True,
            )
        except PlanError as again:
            error = again
    raise error


def _tile_pin(tiles: Tiles | None) -> int:
    """The pin that makes a pass's tiles as recorded: their side, 0 for the untiled pass."""
    return 0 if tiles is None else tiles.size


def _compile_refusal(
    refused: PlanError,
    job: tuple[int | Reading, tuple[int, int], Sequence[Shot], Pins, int | None],
    options: dict[str, Any],
) -> PlanError:
    """The refusal of a job --compile can't run: the compiled plan's `refused`, where a plain
    plan of the same job fits, with the shapes measured and the advice, each on a line of its own
    (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in", decided on 2026-10-29): the same
    command without --compile and, for a tile pinned, the pin within the shapes that the planner
    takes (the untiled pass if it is within them, else the largest tile that fits). `job` is plan's
    free, target, shots, pins and host_available, `options` its split, numz_padding and attention.
    A job no plain plan fits is refused as it was, whatever --compile asked: the plain plan's
    refusal.

    PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says the refusal "names the pass and the shapes
    measured" and doesn't word the lines): the compiled plan's own sentence, which names the
    pass and its tiles, then the shapes measured where it doesn't say them, then the two pieces
    of advice."""
    free, target, shots, pins, host_available = job
    try:
        plan(*job[:5], recorded=None, compile_vae=False, compile_fixed=True, **options)
    except PlanError as plain:
        return plain
    padded = _job(free, target, shots, **options, compile_vae=True).padded
    lines = [str(refused)]
    named = ("encode", "decode") if refused.phase not in ("encode", "decode") else (refused.phase,)
    if not all(_compile_range_said(phase) in lines[0] for phase in named):
        lines.append(
            "--compile compiles only the shapes measured: "
            + "; ".join(_compile_range_said(phase) for phase in ("encode", "decode"))
        )
    lines.append("The same command without --compile fits")
    pinned: dict[Phase, int | None] = {"encode": pins.encode_tile, "decode": pins.decode_tile}
    # The pins refused: a tile beyond the shapes, or the pass the refusal names.
    bad: list[Phase] = [
        phase
        for phase, pin in pinned.items()
        if pin is not None
        and (phase == refused.phase or not in_compile_range(padded, phase, _pin_tiles(pin)))
    ]
    current: Pins = pins
    for index, phase in enumerate(bad):
        later = bad[index + 1 :]
        for candidate in _within(padded, phase):
            substituted = _with_tile(current, phase, candidate)
            try:
                plan(
                    free,
                    target,
                    shots,
                    substituted,
                    host_available,
                    recorded=None,
                    compile_vae=True,
                    compile_fixed=True,
                    **options,
                )
            except PlanError as again:
                # Another pin still beyond the shapes, to be replaced in its turn, refuses it.
                if again.phase not in later:
                    continue
            lines.append(
                f"The same command with --{phase}-tile {candidate} fits, within the shapes measured"
            )
            current = substituted
            break
    return PlanError("\n".join(lines), instead=refused.instead, phase=refused.phase)


def _with_tile(pins: Pins, phase: Phase, pin: int) -> Pins:
    """`pins` with the tile pin of a VAE phase set."""
    return replace(pins, encode_tile=pin) if phase == "encode" else replace(pins, decode_tile=pin)


def _pin_tiles(pin: int) -> Tiles | None:
    """The tiles a tile pin gives: none for 0, the untiled pass."""
    return Tiles(pin) if pin else None


def _within(padded: tuple[int, int], phase: Phase) -> list[int]:
    """The tile pins within the shapes measured compiled (in_compile_range), in the planner's
    own order: 0, the untiled pass, if it is within them, then the tiles it makes (_tile_sizes),
    the largest first."""
    pins = [0, *_tile_sizes(padded)]
    return [pin for pin in pins if in_compile_range(padded, phase, _pin_tiles(pin))]


def _substituted(pins: Pins, instead: str) -> Pins | None:
    """`pins` with the pin a refusal names in place of the one it refused (PlanError.instead,
    "--window 6"); None for words that name no pin."""
    option, _, value = instead.partition(" ")
    field = {
        "--window": "window",
        "--swap-blocks": "swap_blocks",
        "--encode-tile": "encode_tile",
        "--decode-tile": "decode_tile",
    }.get(option)
    if field is None or not value.isdecimal():
        return None
    return replace(pins, **{field: int(value)})


def _planned(
    job: _Job, pins: Pins, recorded: Plan | None, host_available: int | None
) -> tuple[Plan, Peaks]:
    """plan's work, once the pins are checked: the plan of `job`, in the memory its reading
    counts on, and its peaks; refused (PlanError) as plan says."""
    made, refused, shrunk = _made(job, pins, recorded, host_available)
    shortened = None
    instead = ""
    if refused is not None and shrunk is not None:
        # Its tiles shrunk as far as they go, which the figures said before are of.
        refused += _shrunk_said(job, made)
    if refused is not None and recorded is None:
        shorter = _window_that_runs(job, pins, made, host_available)
        if shorter is not None and pins.window is None:
            # The planner's own choice: the longest window whose plan fits the host too.
            shortened = _short_of_host(job, made, host_available)
            (_, made, shrunk), refused = shorter, None
        elif shorter is not None:
            refused += f": the longest window whose plan fits it is --window {shorter[0]}"
            instead = f"--window {shorter[0]}"
    shortfall = None
    if refused is not None and recorded is not None:
        shortfall = _shortfall(job, made, host_available)
        refused = (
            f"the recorded plan no longer fits the memory free now: {refused}"
            f"{_pin_s_part(job, made, pins)}"
        )
    if refused is not None:
        phase = _refusing(job, made, host_available)
        raise PlanError(refused, instead=instead, shortfall=shortfall, phase=phase)
    return made, replace(_peaks(job, made), shortened=shortened, shrunk=shrunk)


def _made(
    job: _Job, pins: Pins, recorded: Plan | None, host_available: int | None
) -> tuple[Plan, str | None, Shrunk | None]:
    """The job's plan around its pins, or around a recorded shape (plan), with its scratch's
    allowance; why it doesn't fit the memory, None when it does (_unfit); and where its tiles
    were shrunk for the host's memory, the DiT kept on the GPU in the VAE's phases, the tiles
    the card alone gives (_dit_kept), else None. PlanError for a pin that can't be taken, or a
    size no plan fits (_chosen, _around)."""
    if recorded is None:
        made = _chosen(job, pins)
    else:
        _check_tiles(job.padded, recorded)
        made = _around(job, recorded, pins)
    made = replace(made, scratch_allowance=_allowance(job, made, host_available))
    refused = _unfit(job, made, host_available)
    ordered = made
    if refused is not None and pins.swap_blocks is None:
        # Where the system memory is what the plan is short of: more blocks swapped, the VAE's
        # weights staying on the GPU, when that takes less of it (_vae_kept).
        spared = _vae_kept(job, made, host_available)
        if spared is not None:
            again = _unfit(job, spared, host_available)
            if again is None or _peaks(job, spared).host < _peaks(job, made).host:
                made, refused = spared, again
    shrunk = None
    if refused is not None and recorded is None:
        # Then the DiT kept on the GPU in the VAE's phases, their tiles shrunk for it, when that
        # takes less of it (_dit_kept); a recorded plan's tiles shape the output. From the plan
        # with the VAE's weights kept first; and where that doesn't fit, from the plan as the
        # choices made it too: the VAE kept swaps one or two blocks more, each pinned whatever
        # the VAE's phases keep, so that the tiles shrunk under it may reach no plan the host
        # holds where they reach one with the VAE away, and the window would shorten for it, or
        # the job be refused: at 4K, shots of 23, 24, 150 and 600 frames with 63.0 GiB free and
        # 11.22 GiB of system memory ran windows of 12 latents where 13 plan, and a 5K shot of
        # 149 frames was refused with 43.0 GiB free where windows of 5 plan
        # (tests/test_planner.py, test_the_vae_kept_first_costs_no_window). PROVISIONAL
        # (implementation, 2026-10-16): DESIGN.md orders the VAE kept before the tiles, the
        # window last; its order is kept where it fits, the plan with the VAE away taken where it
        # alone does, or where it takes less.
        least = None
        for start in dict.fromkeys((made, ordered)):
            kept = _dit_kept(job, start, host_available, pins)
            if kept is None:
                continue
            if least is None or _peaks(job, kept).host < _peaks(job, least[1]).host:
                least = start, kept
            if _unfit(job, kept, host_available) is None:
                least = start, kept
                break
        if least is not None and _peaks(job, least[1]).host < _peaks(job, made).host:
            start, kept = least
            shrunk = Shrunk(
                start.encode_tiles, start.decode_tiles, *_host_short(job, start, host_available)
            )
            made, refused = kept, _unfit(job, kept, host_available)
    return made, refused, shrunk


def _window_that_runs(
    job: _Job, pins: Pins, made: Plan, host_available: int | None
) -> tuple[int, Plan, Shrunk | None] | None:
    """For a new job whose plan fits the card and takes more system memory than the host has,
    its tiles shrunk as far as they go: the longest window under the plan's own with which the
    job is planned on this host, that plan, and the tiles the card alone gives there where they
    were shrunk, the other pins as given, by the planner's own making of each (_made); None where
    no shorter window is, or for another refusal.

    The choices are made for the GPU's memory, the longest window first, and the system memory
    is checked after them: a long window may take swapped blocks, each pinned in system memory,
    where a shorter one takes none, so that a job would be refused on a host that runs it under
    a shorter window, and refused with MORE free VRAM where it was planned with less. 1080p, a
    shot of 301 frames, 16 GiB available and 80 GiB of free VRAM: one window of 76 latents with
    35 blocks pinned takes 17.40 GiB; two windows of 39 run, nothing pinned. The planner takes
    that window by itself (plan); under --window, the refusal names it.

    The window given is the one the job really runs under the cap found (_own_cap), as the
    planner's own cap is: under --window it is told as the pin that plans the same."""
    peaks = _peaks(job, made)
    over = any(peaks.need(phase) + job.margin > job.free for phase in PHASES)
    if host_available is None or over or peaks.host <= host_available or made.window is None:
        return None
    ran = _longest(job.lengths, made.window)
    for cap in range(made.window - 1, job.smallest - 1, -1):
        # A cap under which the job runs the windows it ran under the last one tried plans the
        # same.
        if _longest(job.lengths, cap) == ran:
            continue
        ran = _longest(job.lengths, cap)
        own = _own_cap(job, cap)
        try:
            shorter, short, shrunk = _made(job, replace(pins, window=own), None, host_available)
        except PlanError:
            continue
        if short is None:
            return own, shorter, shrunk
    return None


def _short_of_host(job: _Job, made: Plan, host_available: int | None) -> Shortened:
    """What a plan the host can't hold says of itself where the planner then takes a shorter
    window (plan): its window, the system memory it takes and what is available (_host_short)."""
    assert made.window is not None
    return Shortened(made.window, *_host_short(job, made, host_available))


def _host_short(job: _Job, made: Plan, host_available: int | None) -> tuple[int, int]:
    """The system memory a plan the host can't hold takes and what is available, in hundredths of
    a GiB, as its refusal would have said them (_unfit: each part to the nearest hundredth of a
    GiB, their sum, and what is there rounded down, a hundredth under the total at least)."""
    assert host_available is not None
    taken = sum(_host_hundredths(made, _peaks(job, made).shot))
    return taken, min(_hundredths(host_available, up=False), taken - 1)


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
    - a cap no layout exists for (job.window_layout): one under 2 * SHARED + 1 latents that a
      shot is longer than. The planner makes a cap that short only for a job whose longest shot
      is as short, each shot then one window (_Job.smallest), so that record was edited;
    - tiles of which one covers the whole padded frame, which is no tiling (covers): the planner
      makes none, and refuses the pin of one (plan); or tiles the tiled passes lay no grid of
      (_check_tiles)."""
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
    (check_recorded), or that the tiled passes lay no grid of (tilegrid.latent_tiles: a side or
    an overlap that isn't a multiple of 8 px), which no plan of the planner's has: that record
    was edited, and its pass has no peak (_fit). PROVISIONAL (implementation, 2026-10-12):
    DESIGN.md doesn't say what such a record gets; refused here, in tilegrid.grid's own words,
    which the tiled passes would raise for it."""
    height, width = padded
    passes: tuple[tuple[Phase, Tiles | None], ...] = (
        ("encode", recorded.encode_tiles),
        ("decode", recorded.decode_tiles),
    )
    for phase, tiles in passes:
        if tiles is None:
            continue
        try:
            tilegrid.latent_tiles(tiles)
        except ValueError as error:
            raise PlanError(
                f"not a plan as seedvr2x records one: its {phase}_tiles: {error}"
            ) from None
        if covers(padded, tiles):
            raise PlanError(
                f"not a plan as seedvr2x records one: its {phase}_tiles, of {tiles.size} px, with"
                f" frames padded to {width}x{height}: a tile that covers the whole frame is no"
                f" tiling, and an untiled {phase} records none"
            )
    if recorded.compile_vae:
        # A plan the planner makes with the VAE compiled has no pass beyond the shapes measured
        # (in_compile_range): a record that has one was edited, and its peaks are no one's.
        # PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md doesn't say what such a record gets):
        # refused as the covering tile is, in the same words.
        for phase, tiles in passes:
            if not in_compile_range(padded, phase, tiles):
                raise PlanError(
                    f"not a plan as seedvr2x records one: its compile_vae is true and its"
                    f" {phase} {_tiles_said(tiles)}, beyond the shapes measured compiled"
                    f" ({_compile_range_said(phase)})"
                )


def _quality_said(shots: Sequence[Shot], target: tuple[int, int], quality: str) -> str:
    """The plan's line of the --quality level (said), at every level, standard's too (DESIGN.md,
    Memory planner, Every choice of the plan is said): the runs a shot and in all, how a frame
    is taken, the GPU time against standard's one run, the memory one run's, the disk the frames
    kept between a shot's runs take, and the levels that give or cost more. The time here is a
    multiple of one run's, about runs_per_shot times, a later run being shorter by its first frame
    (DESIGN.md, Several runs of a shot: "4x the GPU time"): each level's time in seconds needs the
    calibration, which comes after the plan's lines, the models loaded (cli._say_times)."""
    runs_per_shot = QUALITY[quality]
    if runs_per_shot == 1:
        return (
            "quality: standard, the default, one run a shot; --quality enhanced or maximum runs 2"
            " or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the"
            " GPU time"
        )
    runs = sum(len(shot_runs(shot, runs_per_shot)) for shot in shots)
    # The frames a shot's earlier runs give wait on disk, at 16 bits, in the shot's unit directory
    # until its last run's decode (run.Kept): the most a shot keeps, said so that a user sees the
    # room it takes before anything runs.
    kept = max(kept_bytes(shot, runs_per_shot, output_size(target)) for shot in shots)
    levels = [(level, count) for level, count in QUALITY.items() if level != quality]
    others = (
        f"{' or '.join(level for level, _ in levels)}:"
        f" {' or '.join(str(count) for _, count in levels)} runs"
    )
    return (
        f"quality: {quality}, {runs_per_shot} runs a shot, {runs} in all, each frame from its best"
        f" run: about {runs_per_shot} times standard's GPU time, one run's memory;"
        f" {kept / 2**30:.2f} GiB of frames on disk for the longest shot; --quality {others}"
    )


# What a run holds of a shot at the moment it holds the most, in the plan's line of the system
# memory (said): _MOMENT_SAID's moments, shorter, the line giving every figure beside them.
_MOMENT_SHORT = {
    "check": "as its latent is checked",
    "windows": "at its longest window",
    "merge": "as its windows are merged",
}


def said(
    plan: Plan,
    peaks: Peaks,
    card: str,
    target: tuple[int, int],
    shots: Sequence[Shot],
    pins: Pins,
    *,
    numz_padding: bool = False,
    quality: str = "standard",
    recorded: bool = False,
    layouts: Sequence[Sequence[tuple[int, int]]] | None = None,
    disk: str | None = None,
) -> list[str]:
    """The plan as --plan prints it and every run and resume logs it, before the first unit,
    a line per choice in plain words (DESIGN.md, Memory planner, Every choice of the plan is
    said), each opening with what it says, "plan: windows: ...": the value with its unit; why it
    was taken, the rule and its figure; what it gives or costs; the option that changes it.
    Every value the planner sets is in it: the card's free memory and the budget, with the
    margin, and on a run held to the card the torch's limit and the reserve, by their names and
    figures (_card_said, _budget_said); the padded frames and the attention backend; the shots;
    the windows; the --quality level, at every level (_quality_said); the swapped blocks, where
    the VAE is in the DiT's phase and what of the DiT stays in the VAE's; the system memory;
    each VAE pass, untiled or its tiles, their overlap and grid; the room a VAE pass is held to
    (VAE_ROOM); the scratch of the longest tiled shot; the disk the job takes (disk.py); the
    predicted peaks and which bounds the plan; with --compile, the VAE compiled, its passes held
    to COMPILE_PEAK times their plain peaks, and the compilation's time, paid before the DiT
    loads; the time, estimated once the models are loaded, by
    the calibration every run makes (DESIGN.md, Memory planner, Time estimates); what was
    pinned. recorded: the plan of a job resumed that has made units, whose window cap and tiles
    were kept from the job's record so that the output stays bit for bit (plan's `recorded`):
    their lines say so in place of the rule. A window the planner shortened for the host's
    memory, tiles it shrank for it, say what the card alone gives and why it isn't taken
    (Peaks.shortened, Peaks.shrunk). The lines stay short, one terminal line where they can, two
    at most; the manual's words.

    layouts: each shot's windows (latents [start, end)) where a job holds runs of its own window
    cap, made under a cap the plan no longer lays out (manifest.WINDOW_CAP; a shape planned again
    for the memory free, DESIGN.md, Memory planner, decided on 2026-10-28): the line's shots in
    one window and its joins are counted from them, not by the plan's cap, which they would
    otherwise misstate (review of the plan's lines, 2026-10-28). None for every other job, whose
    line is the plan's alone.

    disk: the line of the disk the job takes beside its output against what is free (runtime/
    disk.py, DESIGN.md, Memory planner, decided on 2026-10-28), after the scratch's and before the
    predicted peaks, as DESIGN.md lists it; None says none.

    The line of the compiled VAE is the plan's own (plan.compile_vae, the setting's, always: a
    job outside the shapes measured is refused, plan)."""
    # runtime/shot.py imports torch, which this module's import mustn't (the module's docstring).
    from seedvr2x.runtime.shot import shot_layout

    job = _job(peaks.reading, target, shots, False, numz_padding, peaks.attention, plan.compile_vae)
    height, width = job.padded
    out_height, out_width = output_size(target)
    frames = Counter(shot.frames for shot in shots)
    if layouts is None:
        windows = {count: len(shot_layout(count, plan.window)) for count in frames}
        whole = sum(number for count, number in frames.items() if windows[count] == 1)
        joins = sum((windows[count] - 1) * number for count, number in frames.items())
    else:
        whole = sum(1 for layout in layouts if len(layout) == 1)
        joins = sum(len(layout) - 1 for layout in layouts)
    lines = [*_card_said(card, peaks)]
    lines.append(
        f"frames: {out_width}x{out_height} padded to {width}x{height} for the model,"
        f" {peaks.tokens:,} tokens a latent"
    )
    if peaks.attention == FLASH_ATTN_2:
        lines.append("attention: FlashAttention 2, which is installed")
    else:
        lines.append(
            "attention: PyTorch's SDPA, FlashAttention 2 not being installed: a window takes"
            " more memory under it"
        )
    lines.append(
        f"shots: {len(shots)}, the longest of {_count(job.frames, 'frame')}"
        f" ({_count(job.lengths[-1], 'latent')})"
    )
    lines.append(_windows_said(job, plan, peaks, pins, recorded, whole, joins, len(shots)))
    lines.append(_quality_said(shots, target, quality))
    lines.append(_swap_said(plan, pins))
    lines.append(_vae_said(plan))
    lines.append(
        f"the DiT in the VAE's passes: in the encode,"
        f" {_resident_said(plan.resident_encode, plan.swapped)}; in the decode,"
        f" {_resident_said(plan.resident_decode, plan.swapped)}; as much as fits beside each"
    )
    process, pinned, held, allowance = _host_hundredths(plan, peaks.shot)
    left = _left(plan, "the DiT's other parameters", "the VAE")
    copies = f"{pinned / 100:.2f} GiB pinned for {left}" if left else "nothing pinned"
    in_all = process + pinned + held + allowance
    # The shot that holds the most: the longest, but where a shorter one in fewer windows does.
    if peaks.holder == job.lengths[-1]:
        whose = "the longest shot at the most"
    else:
        whose = f"a shot of {_count(peaks.holder, 'latent')} at the most, more than the longest"
    lines.append(
        f"system memory: {in_all / 100:.2f} GiB in all; {copies}; {held / 100:.2f} GiB for"
        f" {whose}, {_MOMENT_SHORT[peaks.moment]}"
    )
    for phase, tiles, pin in (
        ("encode", plan.encode_tiles, pins.encode_tile),
        ("decode", plan.decode_tiles, pins.decode_tile),
    ):
        lines.append(_vae_pass_said(job, phase, tiles, pin, recorded, peaks.shrunk))
    lines.append(
        f"the VAE's room: {_gib(VAE_ROOM)} GiB of the budget left free in each VAE pass, for the"
        " large pieces of memory a pass asks for at once"
    )
    if peaks.scratch:
        # A scratch over the allowance goes whole to its file (tiling.Scratch).
        allowed = plan.scratch_allowance
        over = sum(
            number for count, number in frames.items() if scratch(job.padded, count) > allowed
        )
        where = (
            f"over it, the scratch of {over} of {_count(len(shots), 'shot')} goes whole to a"
            " memory-mapped file"
            if over
            else "which holds every shot's"
        )
        lines.append(
            f"scratch: {_gib(peaks.scratch)} GiB for the longest tiled shot, what a tiled pass"
            f" keeps between its tiles; in system memory up to an allowance of"
            f" {allowance / 100:.2f} GiB, {where}"
        )
    if disk is not None:
        lines.append(disk)
    names = {"encode": "encode", "dit": "DiT's phase", "decode": "decode"}
    lines.append(
        f"predicted peaks: encode {_gib(peaks.encode)} GiB, DiT {_gib(peaks.dit)} GiB, decode"
        f" {_gib(peaks.decode)} GiB; the {names[peaks.bound]} bounds the plan,"
        f"{'' if peaks.bound == 'dit' else ' its room counted,'}"
        f" {_gib(peaks.budget - peaks.need(peaks.bound))} GiB under the budget"
    )
    if plan.compile_vae:
        minutes = f"{COMPILE_MIN // 60} to {COMPILE_MAX // 60} min"
        lines.append(
            f"compile: the VAE compiled (--compile), each pass held to {COMPILE_PEAK} times its"
            f" plain peak, in the shapes measured only; compiling it takes about {minutes} per"
            " process, paid before the DiT loads, at every start and resume, its cache kept in"
            " the job's directory"
        )
    lines.append(
        "time: measured, not guessed: a calibration of the job's own steps once the models are"
        " loaded, its cost said then; it gives the job's time and each unit's time left and end;"
        " --until HH:MM stops before a unit that wouldn't finish by then"
    )
    options = (
        ("--window", pins.window),
        ("--swap-blocks", pins.swap_blocks),
        ("--encode-tile", pins.encode_tile),
        ("--decode-tile", pins.decode_tile),
    )
    given = [f"{option} {value}" for option, value in options if value is not None]
    lines.append(
        f"pinned: {', '.join(given)}, taken exactly"
        if given
        else "pinned: nothing; --window, --swap-blocks, --encode-tile and --decode-tile each fix"
        " a choice"
    )
    return [f"plan: {line}" for line in lines]


# What the plan's lines say of a value a resume takes from the job's record (said).
_KEPT = "kept from the job's record, so that the output stays bit for bit"


def _windows_said(
    job: _Job,
    plan: Plan,
    peaks: Peaks,
    pins: Pins,
    recorded: bool,
    whole: int,
    joins: int,
    shots: int,
) -> str:
    """The plan's line of the windows (said): the cap, in latents and frames, and the longest
    window run where the cap doesn't say it, a cap pinned or recorded above it; why: pinned,
    kept from the record, shortened for the host's memory (Peaks.shortened), the longest shot
    whole, or the longest window the budget holds (DESIGN.md, Memory planner, Choices, 1); the
    shots in one window and the joins; what a longer window gives; --window."""
    if plan.window is None:
        cap = "no cap"
    else:
        cap = f"{_count(plan.window, 'latent')} at most ({_count(_frames(plan.window), 'frame')})"
    # The longest window run, where the cap doesn't say it: a cap pinned or recorded above it.
    if peaks.window != plan.window:
        cap += f", the longest run {_count(peaks.window, 'latent')}"
    if pins.window is not None:
        why = "pinned by --window"
    elif recorded:
        why = _KEPT
    elif peaks.shortened is not None:
        short = peaks.shortened
        why = (
            f"shortened from {_count(short.window, 'latent')} for the host's memory (their plan"
            f" {short.taken / 100:.2f} GiB of {short.available / 100:.2f} available)"
        )
    elif plan.window is not None and plan.window >= job.lengths[-1]:
        why = "the longest shot whole, which the budget holds"
    else:
        why = "the longest the budget holds"
    gives = "; a longer window, fewer joins" if joins else ""
    return (
        f"windows: {cap}, {why}: {whole} of {_count(shots, 'shot')} in one window,"
        f" {_count(joins, 'join')}{gives}; --window N caps them"
    )


def _frames(latents: int) -> int:
    """The frames `latents` latents hold: 1 in the first, 4 in each next (DESIGN.md, Terms)."""
    return 4 * latents - 3


def _swap_said(plan: Plan, pins: Pins) -> str:
    """The plan's line of the swapped blocks (said): how many; why: pinned, none needed, the VAE
    kept on the GPU for the host's memory (_vae_kept), or the fewest that hold the window
    (DESIGN.md, Memory planner, Choices, 2); what it costs; --swap-blocks."""
    if pins.swap_blocks is not None:
        why = "pinned by --swap-blocks"
    elif not plan.swapped:
        why = "the window fitting with the DiT whole on the GPU"
    elif plan.vae_in_dit:
        why = "more than the window needs: the VAE stays on the GPU, which spares system memory"
    else:
        why = "the fewest that hold the window"
    if not plan.swapped:
        return f"blocks swapped: none, {why}; --swap-blocks N swaps some"
    return (
        f"blocks swapped: {plan.swapped} of the DiT's {BLOCKS}, {why}: slower, not worse;"
        " --swap-blocks N"
    )


def _vae_said(plan: Plan) -> str:
    """The plan's line of the VAE's weights in the DiT's phase (said): off the GPU, given to the
    window, before one more block is swapped; or on it, beside the window."""
    if plan.vae_in_dit:
        return "the VAE: on the GPU in the DiT's phase, the window fitting beside it"
    return (
        f"the VAE: off the GPU in the DiT's phase, its {_gib(VAE_WEIGHTS)} GiB given to the window"
    )


def _vae_pass_said(
    job: _Job,
    phase: str,
    tiles: Tiles | None,
    pin: int | None,
    recorded: bool,
    shrunk: Shrunk | None,
) -> str:
    """The plan's line of a VAE pass, the encode or the decode (said): untiled, or its tiles,
    their overlap and grid; why: pinned, kept from the record, shrunk for the host's memory
    (Peaks.shrunk), the whole frame fitting, or the largest tiles that fit (DESIGN.md, Memory
    planner, Choices, 3); what tiles cost; --encode-tile or --decode-tile."""
    option = f"--{phase}-tile"
    alone = None if shrunk is None else (shrunk.encode if phase == "encode" else shrunk.decode)
    if pin is not None:
        why = f"pinned by {option}"
    elif recorded:
        why = _KEPT
    elif shrunk is not None and alone != tiles:
        why = (
            f"shrunk from {'untiled' if alone is None else f'{alone.size} px'} for the host's"
            f" memory (that plan {shrunk.taken / 100:.2f} GiB of {shrunk.available / 100:.2f}"
            " available)"
        )
    elif tiles is None:
        why = "the whole frame fitting the budget"
    else:
        why = f"the whole frame not fitting the budget, the largest that do, up to {TILE_MAX} px"
    if tiles is None:
        return f"{phase}: untiled, {why}; {option} PX tiles it"
    rows, columns = _grid(job.padded, tiles)
    return (
        f"{phase}: tiles of {tiles.size} px overlapping by {tiles.overlap}, {columns} across,"
        f" {rows} down, {why}: a slight colour drift, which split corrects; {option} PX"
    )


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
    free: int | Reading,
    target: tuple[int, int],
    shots: Sequence[Shot],
    split: bool,
    numz_padding: bool,
    attention: str = FLASH_ATTN_2,
    compile_vae: bool = False,
) -> _Job:
    if not shots:
        raise ValueError("a job without a shot")
    if attention not in DIT_ATTENTION:
        raise ValueError(f"attention {attention!r}: the DiT's fit is {FLASH_ATTN_2}'s or {SDPA}'s")
    # A number is a driver that refuses: what it reports free is the card's, and all the plan's.
    reading = free if isinstance(free, Reading) else Reading(free, None, free, False)
    lengths = sorted({_latents(shot.frames) for shot in shots})
    frames = max(shot.frames for shot in shots)
    return _Job(
        *(reading.counted, target, tuple(lengths), frames, split, numz_padding, reading),
        attention,
        compile_vae,
    )


def _padded(target: tuple[int, int], numz_padding: bool) -> tuple[int, int]:
    """The frame the model takes: the target in seedvr2x's padding (job.padded_size), or in
    numz's, zeros up to multiples of 16, for the tests (model.input_transform)."""
    if numz_padding:
        return -(-target[0] // 16) * 16, -(-target[1] // 16) * 16
    return padded_size(target)


def _latents(frames: int) -> int:
    """The latents of a shot of `frames` frames, padded to 4n + 1 (job.padded_length)."""
    return (padded_length(frames) - 1) // 4 + 1


@cache
def _longest(lengths: tuple[int, ...], cap: int | None) -> int:
    """The longest window, in latents, that shots of `lengths` latents really run under a cap
    (None: each in one window): the longest of their balanced layouts' (job.window_layout),
    which is the cap only where a shot's length makes it so. It never falls as the cap rises:
    fewer windows over a shot are longer ones (tests/test_planner.py holds it)."""
    if cap is None or cap >= lengths[-1]:
        return lengths[-1]
    return max(_split(length, cap) for length in lengths)


@cache
def _split(length: int, cap: int) -> int:
    """The longest window, in latents, of a shot of `length` latents under a cap, by its balanced
    layout (job.window_layout); its length when the cap holds it whole."""
    if cap >= length:
        return length
    return max(end - start for start, end in window_layout(length, cap))


def _dit_need(job: _Job, cap: int | None, swapped: int, vae_in_dit: bool) -> int:
    """The predicted peak of the job's DiT phase under a cap, in bytes, `swapped` blocks swapped
    and the VAE's weights there or not: its longest window's (dit_peak), on the attention
    backend the job runs; nothing of a shot is on the GPU beside the window, its noise drawn in
    system memory (shot.sample_windows). Wherever the phase is judged, and what the plan records
    (_peaks): real memory, unlike a VAE pass's room."""
    latents = _longest(job.lengths, cap)
    return dit_peak(job.padded, latents, swapped, vae_in_dit, job.attention)


def _holds(job: _Job, cap: int | None, swapped: int) -> bool:
    """Whether the DiT's phase holds the job under a cap, `swapped` blocks swapped and the VAE's
    weights away: its longest window (_dit_need)."""
    return _dit_need(job, cap, swapped, vae_in_dit=False) <= job.budget


def _longest_cap(job: _Job, swapped: int) -> int | None:
    """The largest cap, from the longest shot's length down to the smallest the job can run, that
    the DiT's phase holds with `swapped` blocks swapped (_holds); None when it holds none.

    A film has thousands of shots and windows of hundreds of latents: the caps whose longest
    window is over the budget by its tokens alone are bisected away first (_longest never falls
    as the cap rises, nor the tokens with it), the window functions then asked from there down."""

    def possible(cap: int) -> bool:
        latents = _longest(job.lengths, cap)
        return _dit_least(job.padded, latents, swapped, False, job.attention) <= job.budget

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


def _fewest(job: _Job, cap: int | None) -> tuple[int, bool] | None:
    """The fewest swapped blocks with which the DiT's phase holds the job under a cap, its
    longest window (_dit_need), and whether the VAE's weights stay on the GPU meanwhile: they
    leave before one more block is swapped. One and two swapped blocks save nothing (dit_peak),
    so are never chosen. None when every block swapped and the VAE away don't hold it."""
    for swapped in (0, *range(SLOTS + 1, BLOCKS + 1)):
        for vae_in_dit in (True, False):
            if _dit_need(job, cap, swapped, vae_in_dit) <= job.budget:
                return swapped, vae_in_dit
    return None


def _chosen(job: _Job, pins: Pins) -> Plan:
    """The plan made whole, around the pins (plan)."""
    floor = _floor(job)
    if floor is not None:
        raise PlanError(_too_large(job, *floor), phase=floor[0])
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
        compile_vae=job.compile_vae,
    )


def _around(job: _Job, recorded: Plan, pins: Pins) -> Plan:
    """A recorded plan's shape as it is, where the weights are planned again around it (plan): the
    fewest swapped blocks that hold its windows, or --swap-blocks' when given, and as much of the
    DiT in the VAE's phases as fits; every block swapped and none left there when it no longer
    fits, which _unfit then says, and under --swap-blocks the pin's blocks, whose part of the
    shortfall the refusal names (_pin_s_part)."""
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
            raise PlanError(
                _swap_refused(job, swapped, cap, "the recorded plan's", fewest[0]),
                instead=f"--swap-blocks {fewest[0]}",
            )
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
    --swap-blocks when none does."""
    most = BLOCKS if pins.swap_blocks is None else pins.swap_blocks
    longest = _longest_cap(job, most)
    if longest is None:
        # The floor holds every block swapped: a --swap-blocks that holds no window at all.
        cap = job.smallest
        whose = "the shortest this job can run"
        if pins.window is not None and _holds(job, pins.window, BLOCKS):
            cap, whose = pins.window, f"--window {pins.window}'s longest"
        fewest = _fewest(job, cap)
        assert fewest is not None
        raise PlanError(
            _swap_refused(job, most, cap, whose, fewest[0]),
            instead=f"--swap-blocks {fewest[0]}",
            phase="dit",
        )
    if pins.window is None:
        cap = _own_cap(job, longest)
    elif _holds(job, pins.window, most):
        cap = pins.window
    else:
        latents = _longest(job.lengths, pins.window)
        need = _dit_need(job, pins.window, most, vae_in_dit=False) + job.margin
        blocks = (
            "every block of the DiT swapped"
            if pins.swap_blocks is None
            else f"--swap-blocks {pins.swap_blocks}"
        )
        raise PlanError(
            f"--window {pins.window}: its longest window, {_count(latents, 'latent')}, takes"
            f" {_over(need)} GiB of free VRAM with {blocks}, and {_there(job)}:"
            f" {_to_free(job, need)}the longest window that fits is --window"
            f" {_own_cap(job, longest)}",
            instead=f"--window {_own_cap(job, longest)}",
            phase="dit",
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
    named: the cap's longest window."""
    latents = _longest(job.lengths, cap)
    need = _dit_need(job, cap, swapped, vae_in_dit=False) + job.margin
    what = f"a window of {_count(latents, 'latent')}, {whose},"
    return (
        f"--swap-blocks {swapped}: {what} takes {_over(need)} GiB of free VRAM with"
        f" {_count(swapped, 'block')} swapped, and {_there(job)}: {_to_free(job, need)}the"
        f" fewest blocks that hold it are --swap-blocks {fewest}"
    )


def _vae_kept(job: _Job, plan: Plan, host_available: int | None) -> Plan | None:
    """The plan's shape with the VAE's weights left on the GPU in the DiT's phase and as many
    more blocks swapped as that takes, the fewest, where the plan as made fits the GPU's memory
    and not the system's; None where it doesn't apply: the system memory unchecked or enough,
    a phase over the budget, the VAE's weights there already, or no count of swapped blocks
    holding the window with them. Not chosen under --swap-blocks, taken exactly (plan).

    DESIGN.md has the VAE leave the DiT's phase before a block is swapped (Memory planner,
    Choices, 1: "the VAE off the GPU for the DiT's phase first, then as many blocks swapped as
    it takes"): a swapped block costs time at every window, the VAE's leaving a copy per phase.
    But the VAE's weights away are 0.47 GiB more of pinned system memory, and a swapped block
    costs none where the VAE's phases send that block away anyway, as most plans of a small
    card do, or 0.42. So DESIGN.md has this one too, provisional there (Choices, the paragraph
    after 3: "Where the plan so made takes more system memory than the host has, the VAE's
    weights stay on the GPU for the DiT's phase and the fewest more blocks are swapped in their
    place, when that fits the host or takes less of it"): taken only where the host's memory
    refuses the plan as the choices order it, so that no plan moves on a host that holds it; it
    shapes no bit. 1080p on a 24 GB card, a shot of 301 frames, 18.3 GiB available: 29 blocks
    swapped and the VAE staying take 17.92 GiB, where 28 and the VAE away take 18.38 and were
    refused (tests/test_planner.py)."""
    peaks = _peaks(job, plan)
    over = any(peaks.need(phase) + job.margin > job.free for phase in PHASES)
    if host_available is None or over or peaks.host <= host_available or plan.vae_in_dit:
        return None
    counts = range(max(plan.swapped, SLOTS) + 1, BLOCKS + 1)
    fits = (count for count in counts if _dit_need(job, plan.window, count, True) <= job.budget)
    swapped = next(fits, None)
    if swapped is None:
        return None
    kept = replace(
        plan,
        swapped=swapped,
        resident_encode=_resident(job, "encode", plan.encode_tiles, swapped),
        resident_decode=_resident(job, "decode", plan.decode_tiles, swapped),
        vae_in_dit=True,
    )
    return replace(kept, scratch_allowance=_allowance(job, kept, host_available))


def _dit_kept(job: _Job, plan: Plan, host_available: int | None, pins: Pins) -> Plan | None:
    """The plan's shape with more of the DiT kept on the GPU in the VAE's phases, their tiles
    shrunk to keep it, where the plan as made, the VAE's weights kept for the DiT's phase where
    that takes less (_vae_kept), fits the GPU's memory and not the system's: every block a VAE
    phase sends away waits in a pinned copy (_pinned), and the largest tiles, chosen first,
    send the most away (DESIGN.md, Memory planner, Choices, decided on 2026-10-13: "before the
    window shortens, the DiT stays on the GPU for the VAE's phases and the tiles shrink, the
    window being the larger lever"). One block more kept in both phases at a time, one block
    fewer pinned, each pass in the largest of its tiles, from its own down, that keeps as many
    (_keeping), up to every block the DiT's phase keeps: the first plan that fits the host is
    taken, with the largest tiles that do; where none does, the last one made, which takes the
    least, for the shorter windows' search and the refusal's figures (plan). None where it
    doesn't apply: the system memory unchecked or enough, a phase over the budget, or no tile
    under a pass's own that keeps one block more.

    A pinned pass keeps its tiles, as on any host ("never under a tile pin"), the other one
    still shrinking. The swapped blocks stay as they are, under --swap-blocks or not: each is
    pinned whatever the VAE's phases keep. Jobs refused for the system memory with more free
    VRAM than one that planned, short shots whose untiled VAE pass fits only with the DiT sent
    away, 26 to 36 blocks pinned (DESIGN.md, Memory planner, Choices: 828 of a grid of 144,000
    plans); and windows that fell as the free VRAM grew (DESIGN.md, To measure, The planner on a
    short host): at 1440p a decode's tiles growing from 1600 to 1728 px kept one block less on
    the GPU, which pinned 0.42 GiB more, the window then shortened for 0.01 GiB of system memory
    (tests/test_planner.py, test_on_a_short_host_the_tiles_shrink_before_the_window).

    PROVISIONAL (implementation, 2026-10-16): DESIGN.md orders the steps, not how far the tiles
    shrink. They shrink as little as the host needs, the same count of blocks kept in both
    phases, the largest tiles that keep it, where the smallest tiles would pin the least: tiles
    drift in colour more as they get smaller (DESIGN.md, Memory planner, A plan shapes the
    output a little)."""
    peaks = _peaks(job, plan)
    over = any(peaks.need(phase) + job.margin > job.free for phase in PHASES)
    if host_available is None or over or peaks.host <= host_available:
        return None
    passes: tuple[tuple[Phase, Tiles | None, bool], ...] = (
        ("encode", plan.encode_tiles, pins.encode_tile is not None),
        ("decode", plan.decode_tiles, pins.decode_tile is not None),
    )
    least = None
    keep = BLOCKS - _leaving(plan) + 1
    while keep <= BLOCKS - plan.swapped:
        chosen = [_keeping(job, *each, plan.swapped, keep) for each in passes]
        if not all(found for found, _ in chosen):
            break
        (_, encode), (_, decode) = chosen
        kept = replace(
            plan,
            encode_tiles=encode,
            decode_tiles=decode,
            resident_encode=_resident(job, "encode", encode, plan.swapped),
            resident_decode=_resident(job, "decode", decode, plan.swapped),
        )
        least = replace(kept, scratch_allowance=_allowance(job, kept, host_available))
        if _unfit(job, least, host_available) is None:
            break
        keep = BLOCKS - _leaving(least) + 1
    return least


def _keeping(
    job: _Job, phase: Phase, tiles: Tiles | None, pinned: bool, swapped: int, keep: int
) -> tuple[bool, Tiles | None]:
    """Whether a pass of the job's encode or decode at or under `tiles`, in the planner's order
    (_fitting: untiled, then the largest tiles down), keeps `keep` of the DiT's blocks on the GPU
    or more (_resident; None, the DiT as its own phase has it, keeps every block that isn't
    swapped), and the largest that does (_dit_kept); a pinned pass's tiles alone."""
    passes = [tiles] if pinned else _fitting(job, phase, swapped)
    for option in passes[passes.index(tiles) :]:
        resident = _resident(job, phase, option, swapped)
        if (BLOCKS - swapped if resident is None else resident) >= keep:
            return True, option
    return False, None


def _shrunk_said(job: _Job, plan: Plan) -> str:
    """What a refusal for the system memory adds where the plan's tiles were shrunk to keep the
    DiT on the GPU in the VAE's phases and it still takes more than is there (_dit_kept): the
    tiles its figures are of, and what of the DiT they keep."""
    encode = _resident_said(plan.resident_encode, plan.swapped)
    decode = _resident_said(plan.resident_decode, plan.swapped)
    return (
        f", its VAE's tiles shrunk to keep the DiT on the GPU: the encode"
        f" {_tiles_said(plan.encode_tiles)}, with {encode}; the decode"
        f" {_tiles_said(plan.decode_tiles)}, with {decode}"
    )


def _pin_s_part(job: _Job, plan: Plan, pins: Pins) -> str:
    """What a resume's refusal adds where --swap-blocks is pinned and the DiT's phase is what
    doesn't fit: the pin's part of the shortfall said, and what the recorded shape is short of
    by itself, every block swapped and the VAE's weights away. The pin isn't refused as on a
    new job there (_around), the shape fitting under no count of swapped blocks, and the
    shortfall counted with the blocks it pins would pass for the shape's: a recorded window of
    21 latents at 1080p with 22 GiB free is 1.07 GiB short, and 15.41 with --swap-blocks 0.
    Empty elsewhere: no pin, another phase refused first, which no swapped block changes (its
    pass with the DiT away), or the system memory."""
    if pins.swap_blocks is None:
        return ""
    peaks = _peaks(job, plan)
    first = next((phase for phase in PHASES if peaks.need(phase) + job.margin > job.free), None)
    if first != "dit":
        return ""
    least = _dit_need(job, plan.window, BLOCKS, vae_in_dit=False) + job.margin
    there = _hundredths(job.free, up=False)
    short = _hundredths(least, up=True) - there
    part = _hundredths(peaks.need("dit") + job.margin, up=True) - there - short
    if part <= 0 or short <= 0:
        return ""
    return (
        f", {part / 100:.2f} GiB of them --swap-blocks {pins.swap_blocks}'s: with every block of"
        f" the DiT swapped and the VAE off the GPU, the recorded plan is {short / 100:.2f} GiB"
        " short"
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
    """The predicted peak of the job's encode or decode (encode_peak, decode_peak), whatever
    its shots' lengths: no VAE pass keeps anything of the shot on the GPU."""
    if phase == "encode":
        return encode_peak(job.padded, tiles, resident, swapped, job.compile_vae)
    return decode_peak(job.padded, job.output, tiles, resident, swapped, job.split, job.compile_vae)


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
    return [
        tiles
        for tiles in passes
        if _vae_need(job, phase, tiles, 0, swapped) <= job.budget
        # With the VAE compiled, only the passes within the shapes measured (in_compile_range):
        # the largest tiles that fit and are measured, smaller where the compiled peak needs it.
        and (not job.compile_vae or in_compile_range(job.padded, phase, tiles))
    ]


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
        if not fitting:
            # Only with the VAE compiled: the passes within the shapes measured don't fit (the
            # untiled pass or the smallest tile may, outside them), or the frame is too small
            # for a tile and its pass can't be untiled.
            raise PlanError(
                f"the VAE compiled has no {phase} pass within the shapes measured"
                f" ({_compile_range_said(phase)}) that fits the free VRAM at {COMPILE_PEAK} times"
                " its plain peak",
                phase=phase,
            )
        return fitting[0]
    option = f"--{phase}-tile {pin}"
    tiles = Tiles(pin) if pin else None
    if job.compile_vae and not in_compile_range(job.padded, phase, tiles):
        raise PlanError(
            f"{option}: the VAE compiled has the {phase} {_tiles_said(tiles)} beyond the shapes"
            f" measured ({_compile_range_said(phase)})",
            phase=phase,
        )
    need = _vae_need(job, phase, tiles, 0, swapped) + job.margin
    if need <= job.free:
        return tiles
    # The untiled pass alone fits where a frame little larger than the smallest tile takes less
    # untiled than in that tile (_least_pass).
    largest = next((fit.size for fit in fitting if fit is not None), None)
    # With the VAE compiled, the untiled pass is named only where it is within the shapes
    # measured (in_compile_range): a refusal never advises a pin the compiled plan refuses.
    if largest is None:
        fits = (
            f"no tile within the shapes measured fits ({_compile_range_said(phase)})"
            if job.compile_vae and not in_compile_range(job.padded, phase, None)
            else f"no tile fits, the untiled {phase} does, --{phase}-tile 0"
        )
    else:
        fits = f"the largest tile that fits is --{phase}-tile {largest}"
    raise PlanError(
        f"{option}: the {phase} {_tiles_said(tiles)} takes {_over(need)} GiB of free VRAM,"
        f" the DiT off the GPU{_ROOM_HELD}, and {_there(job)}: {_to_free(job, need)}{fits}",
        instead=f"--{phase}-tile {0 if largest is None else largest}",
        phase=phase,
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
    job can run, every block of the DiT swapped and the VAE's weights away (_dit_need), whatever
    its shots' lengths; the encode and the decode each in the pass
    that takes the least (_least_pass), the DiT away, whatever the plan swaps: a phase that
    keeps nothing of it frees its slots too (dit_on_gpu). What a phase takes is its peak, and
    for the encode and the decode their room with it (_vae_need)."""
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
    a frame little larger than one: 512x528, where a square tile takes its measured peak
    (MEASURED) and the untiled pass its fit, or 496x528, whose tiles, cut off, take
    CUT_OFF_EXTRA over their rule; and the smallest tiles the refit serves (REFITTED), under
    the smallest of all on a strip over 1024 px long and under 74 px high for the encode, 134
    for the decode, where the two tiled fits cross."""
    sizes = _tile_sizes(job.padded)
    least = dict.fromkeys(size for size in (*sizes[-1:], REFITTED) if size in sizes)
    passes = [None, *(Tiles(size) for size in least)]
    return min(passes, key=lambda tiles: _vae_need(job, phase, tiles, 0, BLOCKS))


def _too_large(job: _Job, phase: Phase, least: int) -> str:
    """The floor's refusal (_floor), counted in free VRAM, never in card sizes (DESIGN.md, Memory
    planner): the free VRAM the output size takes, what is there for it (_there), and the largest
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
        tiled = _tiles_said(_least_pass(job, phase))
        if not _tile_sizes(job.padded):
            tiled += ", too small for a tile"
        how = f"its {phase} {tiled}, with the DiT off the GPU"
        room = VAE_ROOM
    # The room as what is left of the total once the peak and the margin are said, each rounded
    # as the message has it, so that its figures add up. No margin on a run held to the card
    # (_Job.margin), and none said.
    total, peak = _hundredths(least + job.margin, up=True), _hundredths(least - room, up=True)
    rest = (total - peak - _hundredths(job.margin, up=True)) / 100
    held = f", {rest:.2f} of room a VAE pass is held to" if room else ""
    margin = f", and {_gib(job.margin)} of margin" if job.margin else ""
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
        f" a peak of {peak / 100:.2f} GiB{held}{margin}), and"
        f" {_there(job)}: {_to_free(job, least + job.margin)}{fitting}"
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
    GPU, what the run holds of a shot at the moment it holds the most (_shot_held), and what a
    tiled shot's scratch may take of it, the plan's allowance (Plan.scratch_allowance): their
    sum. PROVISIONAL (implementation, 2026-10-12): a sum, as DESIGN.md counts the allowance
    after the rest (Memory planner, Host RAM), though a scratch is never open at the moment a
    shot holds the most: beside it the run holds one latent's worth of the shot, a tiled
    encode's sums or the latent a tiled decode reads (host_moments), so that the sum is over
    what any moment takes by the rest of the shot's part, 1 byte a padded pixel and a latent at
    least for a shot in one window. The load of the weights isn't counted apart: it reads their
    file at its own offsets, never mapped, the parts that leave the GPU straight into their
    pinned copies and every other tensor through one buffer the size of the largest
    (blockswap.read), so that it peaks at what it leaves, the copies counted here and about 1.1
    GiB of the process's own, under HOST_RESIDENT's 2.4: 1.1 GiB with nothing pinned, 7.5 with
    6.38 GiB pinned, 16.3 with the 36 blocks' 15.19 (DESIGN.md, Memory planner, Host RAM;
    BlockSwap, rewritten, the loader). Mapped, as until 2026-10-16, a load peaked at 16.3 to
    16.8 GiB whatever the plan pinned, which no plan could count. PROVISIONAL (implementation,
    2026-10-11): a scratch over its allowance goes whole to its memory-mapped file
    (tiling.Scratch), and the allowance is still counted, as DESIGN.md counts it, for the page
    cache its mapping works in."""
    latents = _longest(job.lengths, plan.window)
    held, moment, holder = _shot_held(job, plan)
    return Peaks(
        job.free,
        job.budget,
        _vae_peak(job, "encode", plan.encode_tiles, plan.resident_encode, plan.swapped),
        _dit_need(job, plan.window, plan.swapped, plan.vae_in_dit),
        _vae_peak(job, "decode", plan.decode_tiles, plan.resident_decode, plan.swapped),
        HOST_RESIDENT + _pinned(plan) + held + plan.scratch_allowance,
        _kept(job, plan),
        latents,
        tokens(job.padded, 1),
        job.reading,
        job.attention,
        held,
        moment,
        holder,
    )


def _shot_held(job: _Job, plan: Plan) -> tuple[int, str, int]:
    """What the run holds in system memory of one of the job's shots under the plan's cap, in
    bytes, at the moment it holds the most, that moment (host_moments), the first in the shot's
    order where two hold as much, and the shot's length in latents: the job's shot that holds
    the most (_most_held), which the plan counts with the pinned copies, the run holding one
    shot's tensors at a time.

    As the run is, a shot in one window holds the most through that window, its latent and its
    noise beside the window's output, 1.5 bytes a padded pixel and a latent: 3.03 MiB a latent
    at 1080p (3,179,520 bytes), 12.04 at 4K (12,625,920); a long shot in many windows as its
    decode merges them, its windows beside the merged latent, 1 byte a padded pixel and a
    latent and the latents its windows share: a shot of 1,000 latents takes 2.09 GiB at 1080p
    in windows of 20, and 10.47 at 4K in windows of 5. Until 2026-10-12 the check read the
    latent whole, 1.75 bytes a padded pixel and a latent, and a merge copied the shot at each
    join: 3.45 GiB for that shot at 1080p and 14.36 at 4K (shot.finite, shot._merge)."""
    return _most_held(job.padded, job.lengths, plan.window)


@cache
def _most_held(
    padded: tuple[int, int], lengths: tuple[int, ...], cap: int | None
) -> tuple[int, str, int]:
    """(bytes, moment, latents): what the run holds in system memory of the shot that holds the
    most among shots of `lengths` latents (ascending) under a cap, the moment it does and that
    shot's length (_shot_held, host_moments). The longest shot's, but where a shorter one holds
    more: the most a shot holds can fall as the shot grows past the cap, its windows then
    shorter than the shot: 12 latents in one window hold 36 latents' worth, 13 in two windows
    of 8 and 7 hold 34 (tests/test_planner.py). Of two that hold as much, the longer.

    No shot holds more than 3 latents' worth a latent and MIXING x SHARED more (its latent and
    noise beside a window no longer than itself; its windows, which share SHARED latents and
    are 2 x SHARED + 1 long at least, with the merged latent and a join's mix): the shots are
    read from the longest down, until none left can hold more than the most found."""
    height, width = padded
    latent = LATENT_CHANNELS * LATENT_VALUE * (height // LATENT_SIDE) * (width // LATENT_SIDE)
    most = (0, MOMENTS[0], lengths[-1])
    for latents in reversed(lengths):
        if latent * (3 * latents + MIXING * SHARED) <= most[0]:
            break
        moments = host_moments(padded, latents, cap)
        moment = max(MOMENTS, key=lambda moment: moments[moment])
        # The encode's end and the decode never hold the most: a shot's windows hold a window
        # more than the first, which holds twice the second (host_moments). No sentence says
        # them (_MOMENT_SAID).
        assert moment in _MOMENT_SAID, moment
        if moments[moment] > most[0]:
            most = (moments[moment], moment, latents)
    return most


def _kept(job: _Job, plan: Plan) -> int:
    """The scratch of the job's longest shot under the plan, in bytes (scratch); 0 when neither
    pass is tiled. One pass's: the encode's and the decode's are the same size, and never open
    together."""
    tiled = plan.encode_tiles is not None or plan.decode_tiles is not None
    return scratch(job.padded, job.frames) if tiled else 0


def _allowance(job: _Job, plan: Plan, host_available: int | None) -> int:
    """What a tiled shot's scratch may take of system memory under the plan, in bytes
    (Plan.scratch_allowance), a larger one going to its file: half of what the host has
    available once the pinned copies and what the run holds of a shot at the most (_shot_held)
    are counted, and no more than the longest tiled shot needs (DESIGN.md, Memory planner, Host
    RAM: provisional, until measured on a consumer host); 0 when no pass is tiled. It shapes no
    bit.

    PROVISIONAL (implementation, 2026-10-11), two choices DESIGN.md doesn't make. No more either
    than what the host has left once the process's own memory is counted with them
    (HOST_RESIDENT), down to nothing, every scratch then going to its file: so that a job is
    refused for the system memory the process, its pinned copies and its longest shot take
    (_unfit), never for an allowance: half of what they leave is more than the process leaves
    of it where they leave under 4.8 GiB. And with what the host has available unknown (None),
    the longest tiled shot's scratch, ALLOWANCE_UNKNOWN_HOST at most."""
    kept = _kept(job, plan)
    if host_available is None:
        return min(kept, ALLOWANCE_UNKNOWN_HOST)
    left = host_available - _pinned(plan) - _shot_held(job, plan)[0]
    return max(0, min(kept, left // 2, left - HOST_RESIDENT))


def _unfit(job: _Job, plan: Plan, host_available: int | None) -> str | None:
    """Why the plan doesn't fit the job's memory: the first phase over the budget, a VAE
    phase judged with its room as the planner chose it (Peaks.need), or the system memory it
    takes over what is available, the process's own, the pinned copies, what the run holds of
    a shot at the most and the scratch's allowance, each named with what it takes where that is a
    hundredth of a GiB or more; None when it fits. A plan of the planner's own is never refused
    for its allowance, which is nothing where the first three leave nothing (_allowance). The
    shot named is the one that holds the most (Peaks.holder)."""
    peaks = _peaks(job, plan)
    blocks = _count(plan.swapped, "block")
    vae = "" if plan.vae_in_dit else " and the VAE off the GPU"
    dit = f"a window of {_count(peaks.window, 'latent')}"
    whats = {
        "encode": f"the encode, {_tiles_said(plan.encode_tiles)}, with"
        f" {_resident_said(plan.resident_encode, plan.swapped)},",
        "dit": f"the DiT's phase, {dit} with {blocks} swapped{vae},",
        "decode": f"the decode, {_tiles_said(plan.decode_tiles)}, with"
        f" {_resident_said(plan.resident_decode, plan.swapped)},",
    }
    for phase in PHASES:
        need = peaks.need(phase) + job.margin
        if need > job.free:
            room = "" if phase == "dit" else f"{_ROOM_HELD},"
            freed = ", which would have to be freed on the card" if _freeable(job, need) else ""
            return (
                f"{whats[phase]} takes {_over(need)} GiB of free VRAM{room} and {_there(job)},"
                f" {_short(need, job.free)} GiB short{freed}"
            )
    if host_available is not None and peaks.host > host_available:
        # Each part as the plan's lines say it, and the total their sum as said, so that the
        # message's figures add up (_host_hundredths).
        process, pinned, held, allowance = _host_hundredths(plan, peaks.shot)
        taken = [(process, "for the process")]
        if pinned:
            left = _left(plan, "the DiT's other parameters", "the VAE's weights")
            taken.append((pinned, f"pinned for {left} that leave the GPU"))
        if held:
            # The shot that holds the most: the longest, but where a shorter one in fewer
            # windows does (_most_held).
            if peaks.holder == job.lengths[-1]:
                shot = f"{_count(job.frames, 'frame')} ({_count(job.lengths[-1], 'latent')})"
                whose = f"the longest shot, {shot},"
            else:
                shot = _count(peaks.holder, "latent")
                whose = f"a shot of {shot}, which holds more than the longest,"
            taken.append((held, f"for {whose} {_MOMENT_SAID[peaks.moment]}"))
        if allowance:
            taken.append((allowance, "for a tiled shot's scratch"))
        total = process + pinned + held + allowance
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


def _shortfall(job: _Job, plan: Plan, host_available: int | None) -> Shortfall | None:
    """What `plan` is short of, as _unfit finds it (the same order: the first phase over the
    budget, then the system memory), in the parts a refusal words apart (Shortfall); None where
    it fits."""
    names = {"encode": "the encode", "dit": "the DiT's phase", "decode": "the decode"}
    phase = _refusing(job, plan, host_available)
    if phase == "host":
        return Shortfall("", "", host=True)
    if phase:
        need = _peaks(job, plan).need(phase) + job.margin
        # A driver that refuses reports what other programs leave: closing them gives it back; an
        # early reading, held or not, judges it by what they hold by NVML (_freeable).
        freeable = (not job.reading.held and job.reading.nvml is None) or _freeable(job, need)
        return Shortfall(names[phase], _short(need, job.free), freeable=freeable)
    return None


def _refusing(job: _Job, plan: Plan, host_available: int | None) -> Phase | Literal["host", ""]:
    """The phase `plan` first doesn't fit in, as _unfit finds it (the same order): "encode", "dit"
    or "decode" when its peak is over the budget, "host" when the system's memory is what falls
    short, else empty."""
    peaks = _peaks(job, plan)
    for phase in PHASES:
        if peaks.need(phase) + job.margin > job.free:
            return phase
    if host_available is not None and peaks.host > host_available:
        return "host"
    return ""


def _host_hundredths(plan: Plan, shot: int) -> tuple[int, int, int, int]:
    """What the plan takes of system memory, as its line and a refusal for it say it (said,
    _unfit), in hundredths of a GiB: the process's own (HOST_RESIDENT), the pinned copies
    (_pinned), what the run holds of a shot at the most, `shot` bytes (Peaks.shot), and the
    scratch's allowance (Plan.scratch_allowance), each to the nearest, as the plan's other
    figures are; the total either says is their sum, so that its figures add up.
    One plan's pinned memory then reads the same in both, 15.81 GiB with every part of the
    weights away, DESIGN.md's figure (Memory planner, Host RAM): rounded up, as what is taken of
    the GPU's memory is, a refusal would say 15.82 where the plan's line says 15.81."""
    return (
        _nearest(HOST_RESIDENT),
        _nearest(_pinned(plan)),
        _nearest(shot),
        _nearest(plan.scratch_allowance),
    )


# A shot's moments in system memory (MOMENTS) as the plan's line and a refusal name the one
# that holds the most, after the shot: the three that can (_most_held).
_MOMENT_SAID = {
    "check": "as its latent is checked after its encode",
    "windows": "at its longest window, its latent and its noise beside the window's output",
    "merge": "as its decode merges its windows",
}


# The attention backends as the plan's lines name them (model.load_models' own words for SDPA).
_ATTENTION_SAID = {FLASH_ATTN_2: "FlashAttention 2", SDPA: "PyTorch's SDPA"}


def _held_under(reading: Reading) -> int:
    """The free memory a held run is held under, in bytes: the smaller of what its driver
    reports free to the process and of what the whole card has free; the driver's figure where
    the card's wasn't read (runtime/memory.py)."""
    if reading.card is None:
        return reading.reported
    return min(reading.reported, reading.card)


def _named(reading: Reading, said: Callable[[float], str]) -> str:
    """Which of a held reading's figures is the free memory, in words for a line that has just
    given it as "N GiB free", with the other figure: the words of runtime/memory.py's own log
    lines. `said` puts a figure in GiB, as the line around it does."""
    if reading.card is None:
        return "as its driver reports it to this process (the whole card's, NVML's, wasn't read)"
    if reading.card <= reading.reported:
        reported = said(reading.reported)
        return f"on the whole card by NVML (its driver reports {reported} GiB free to this process)"
    return (
        "as its driver reports it to this process (the whole card has"
        f" {said(reading.card)} GiB free by NVML)"
    )


def _card_said(card: str, peaks: Peaks) -> list[str]:
    """The plan's first lines (said): the card, its free memory and the budget, with the margin
    and the reserve by their names and figures. On a driver that refuses, one line: what it
    reports free, the budget that less the margin (MARGIN), and no reserve. On a run held to the
    card (memory.Reading.held), two: the free memory the run is held under, the driver's figure
    or the whole card's, with the other (_named); then the budget, what the plan counts on: that
    less what torch's limit leaves under it (Reading.left) and the reserve the plan leaves
    torch's allocator beyond a step's peak (memory.RESERVE), two figures that were once given as
    one (a review's finding, 2026-10-12), the margin counted in the reserve (_Job.margin); where
    the plan took memory.RESERVE_TIGHT, the plan with memory.RESERVE refused, that refusal and
    the cost, slow, in a second line (Peaks.tightened)."""
    reading = peaks.reading
    # An early reading (memory.early) says what NVML read and what was counted off it, never
    # "its driver reports", which it did not ask.
    early = None if reading.nvml is None else _early_said(reading, _gib)
    if not reading.held:
        free_said = (
            f"{_gib(peaks.free)} GiB free"
            if early is None
            else f"{early}, {_gib(peaks.free)} GiB left"
        )
        return [
            f"card: {card}, {free_said}, a budget of {_gib(peaks.budget)} GiB:"
            f" the margin of {_gib(MARGIN)} GiB left to the desktop and other programs; no"
            " reserve, the driver refusing memory the card doesn't have"
        ]
    free = _held_under(reading)
    budget = (
        f"budget: {_gib(peaks.budget)} GiB, the {_gib(free)} GiB free less torch's limit,"
        f" {_gib(reading.left)} GiB under it, and the reserve of"
        f" {_gib(free - reading.left - peaks.free)} GiB left to torch's allocator beyond a step's"
        f" peak; the margin of {_gib(MARGIN)} GiB is in the reserve"
    )
    lines = [
        f"card: {card}, {_gib(free)} GiB free"
        f" {_named(reading, _gib) if early is None else f'({early})'}: the run is held to the"
        " card's own memory, its driver giving more than the card has",
        budget,
    ]
    if peaks.tightened is not None:
        # The plan's last resort (plan; DESIGN.md, Allocator): said slow, on a line of its own,
        # the budget's second (DESIGN.md, Architecture, What it prints, decided on 2026-10-28:
        # the budget's line took 851 characters with the usual reserve's refusal inside it, which
        # plan() now says at debug). PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says "the two
        # lines a plan's line may take" and leaves the split open): the figures in the first, the
        # reserve of 2.5, its cost and the pin that runs at full speed in the second, so that the
        # first reads as any held plan's does.
        tight, usual = _gib(memory.RESERVE_TIGHT), _gib(memory.RESERVE)
        reserve = (
            f"reserve: {tight} GiB and not {usual}, the plan with {usual} having been refused"
            " (--verbose says why); slow: a step short of room beyond its peak slows down where"
            f" it would be refused, and a job that fits with {usual} runs up to 1.5 times as"
            f" long with {tight}; memory freed on the card gives the {usual} back"
        )
        if peaks.instead:
            # The pin that runs with the usual reserve, at full speed (design, 2026-10-21).
            reserve += f"; {peaks.instead} would run with the full reserve, at full speed"
        lines.append(reserve)
    return lines


def _early_said(reading: Reading, said: Callable[[float], str]) -> str:
    """An early reading's figures in words (memory.early): what NVML reads free on the whole card
    and what is counted off it for the CUDA context this process is yet to make. `said` puts a
    figure in GiB, as the line around it does."""
    assert reading.nvml is not None
    return (
        f"NVML's {said(reading.nvml)} GiB free on the whole card, less {said(reading.context)} GiB"
        " counted for this process's CUDA context to come"
    )


def _there(job: _Job) -> str:
    """What is there for the plan, as a refusal says it after what something takes: on a driver
    that refuses, what it reports free; on a run held to the card, what the plan counts on, of
    which figure, the driver's or the whole card's, with the other (_named). Rounded down, as
    what is there is."""
    reading = job.reading
    if reading.nvml is not None:
        # An early reading (memory.early): NVML's figure and the context counted off it, not a
        # driver's report, which no process of ours has yet been given.
        if not reading.held:
            return f"{_early_said(reading, _under)}, leaves {_under(job.free)} GiB"
        return f"the plan counts on {_under(job.free)} GiB of {_early_said(reading, _under)}"
    if not reading.held:
        return f"the driver reports {_under(job.free)} GiB free"
    return (
        f"the plan counts on {_under(job.free)} GiB of the {_under(_held_under(reading))} GiB"
        f" free {_named(reading, _under)}, the run held to the card's own memory"
    )


def _to_free(job: _Job, need: float) -> str:
    """What a held run's refusal says before what fits: the memory that would have to be freed
    on the card for what was asked, `need` bytes of free VRAM, to fit what the plan counts on,
    by the message's own figures (_short); where no memory freed on the card gives it
    (_freeable), what it is short of, and no more. Nothing on a driver that refuses, whose
    refusals say what they said."""
    if not job.reading.held:
        return ""
    if not _freeable(job, need):
        return f"{_short(need, job.free)} GiB short; "
    return f"{_short(need, job.free)} GiB more would have to be freed on the card for it; "


def _freeable(job: _Job, need: float) -> bool:
    """Whether memory freed on the card, other programs closed, can give a held run's plan the
    `need` bytes of free VRAM it is refused for: only where the whole card's figure is the one
    the plan was made under, the smaller of the two, and only up to the driver's, which the plan
    is held under from there on (memory.reading). Elsewhere a refusal that said "N GiB more
    would have to be freed on the card" gave advice nobody can follow (a review's finding,
    2026-10-12): the driver's figure the smaller, or the card's not read, it is the process's
    own while the card has room, and other programs' memory isn't in it; and a request 16 GiB
    over a card that has 2 in use is short of more than the card holds."""
    reading = job.reading
    if reading.nvml is not None:
        # PROVISIONAL (impl-v010, 2026-10-29: the refusal's rule with NVML's used for the driver's
        # figure, which an early reading hasn't asked): what quitting programs can give back is
        # what they hold of the card by NVML (memory.card_state), there being no CUDA context of
        # ours on it yet, as a held run's is what the driver reports free beyond the card's; where
        # used wasn't read, the early reading cannot tell, and does not say it.
        return reading.nvml_used is None or need - job.free <= reading.nvml_used
    if not reading.held or reading.card is None or reading.card >= reading.reported:
        return False
    return need - job.free <= reading.reported - reading.card


def _tiles_said(tiles: Tiles | None) -> str:
    """A pass as a refusal says it: untiled, or in its tiles."""
    return "untiled" if tiles is None else f"in tiles of {tiles.size} px"


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
