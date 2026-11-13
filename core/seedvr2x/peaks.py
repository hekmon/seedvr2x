"""What each brick takes of the GPU, predicted (DESIGN.md, Memory planner, Per-phase peaks): the
DiT's peak in a window and where its weights are, the VAE's encode and decode in a pass, whole or
tiled, in bytes, by the fits of research/docs/planner-limits.md and research/docs/vram.md,
measured on numz's CLI with the 7B fp16 (FP16), and by the peaks measured on seedvr2x's own tiled
passes (MEASURED); the DiT in seedvr2x's int8 files (INT8) by its weights, read from its files'
headers, and the fp16 fit's other terms, provisional until measured (INT8_*); the compiled
passes' range; and the store a tiled pass takes. The library's (DESIGN.md, Repository and
apps), as the bricks are; the planner (shot_plan.py) chooses a job's plan with them.
Sizes are (height, width) in pixels; `padded` is the frame the model takes, the target padded
(frames.padded_size). No torch at import: the command line plans before it imports torch
(tests/test_cli.py)."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import cache, lru_cache

from seedvr2x import tilegrid, weights
from seedvr2x.frames import padded_length
from seedvr2x.memory import PLAIN, POOL, SEGMENTS
from seedvr2x.passes import Phase, Tiles

# The DiT's own window functions, the 7B's: plain Python, and the packages above them hold
# nothing, so importing them brings in no torch.
from seedvr2x.vendor.models.dit_7b.window import (
    make_720Pwindows_bysize,
    make_shifted_720Pwindows_bysize,
)

GIB = 1024**3
MIB = 1024**2

# The DiT's peak on seedvr2x's lean attention (DESIGN.md, Allocator, Small cards; research/docs/
# small-cards.md, The lean attention's peak, refitted): what the loaded 7B fp16 holds before a
# forward, the VAE's weights included, in its DiT phase (the DiT 15.35 GiB, its buffers 0.02, the
# VAE 0.47, the text embeddings and the schedule), plus PER_TOKEN a token and PER_WINDOW an
# attention window, each of which repeats the 58 text tokens. Measured by research/scripts/
# hold_probe.py (seedvr2x's own forward, one sampling step, the VAE's weights on the GPU, the
# sharp 7B fp16, an RTX 5090, 2026-11-06): 1080p (padded 1104x1920) with 36 blocks swapped, 1 to
# 19 latents by 2, with none swapped 1 to 7; 4K (2192x3840) with 36 swapped 1 to 4 and with none
# 1 latent. Least squares on the tokens and the windows of peak less (loaded less the swapped
# blocks' saving), the 1-latent 1080p window left out: 61.243 KiB a token, 0.996 MiB a window,
# +0.0068 GiB, every other point within 0.0004 GiB. The 1-latent 1080p window peaks 0.073 GiB over
# that line (its peak is elsewhere in the step): the constant of DIT_ATTENTION that bounds every
# point is 0.0798 GiB, taken as 0.081, which leaves 1 MiB over that window with the slopes
# rounded. FlashAttention 2 and SDPA gave the same peaks to the MiB at every point: on the lean
# code the step's peak is no longer in the attention, and one fit serves both.
# For the record, the plain attention's fit was 127.16 KiB a token and 2.726 MiB a window, SDPA's
# 4.69% more and 0.141 GiB.
# By its parts as seedvr2x has them (below): the 36 blocks 15.1925 GiB, the DiT's other
# parameters 0.1550, its buffers with the text embeddings and cuBLAS's workspace 0.0555
# (DIT_BUFFERS, the RoPE tables apart: a window makes them, and the fit holds them), the VAE
# 0.4669, in all 15.8699, 0.0011 under DIT_LOADED, within the fit's 0.005.
# Other measures: the plain code's one window of 1 latent at 1080p, no block swapped, 17.016 GiB
# (milestone 3, 2026-10-11); the lean code's 16.464. In int8 and in the sharp dynamic GGUF,
# 5 latents at 1080p with 36 swapped measured 3.649 and 3.433 GiB, under the predicted 4.063 and
# 3.576 (fp16: 4.035 for 4.108).
DIT_LOADED = round(15.871 * GIB)
PER_TOKEN = 61.243 * 1024
PER_WINDOW = 0.996 * 1024**2
# The attention windows along time, height and width that every block of the 7B asks its window
# function for, the blocks alternating the two layouts (vendor/configs_7b/main.yaml, window and
# window_method).
NUM_WINDOWS = (4, 3, 3)
# What the DiT's fit adds to DIT_LOADED beside its per-token and per-window terms, in bytes: the
# constant that bounds every point, 0.081 GiB, the 1-latent 1080p window's. PyTorch's SDPA is the
# one attention kernel (DESIGN.md, Vendored model code); FlashAttention 2, gone on 2026-11-08,
# peaked at the same figures on the lean attention.
DIT_ATTENTION = round(0.081 * GIB)

# The VAE's weights, on the GPU in the VAE's phases, and in the DiT's unless the plan sends them
# away for it, in bytes: 0.4669 GiB, its parameters by the model's shapes in the pipeline's
# bfloat16, each on a multiple of 512 bytes, as the part that leaves lays them out (model.Models'
# vae, a blockswap.Part; tests/test_shot_plan.py builds the model). Measured on numz as DESIGN.md's
# 0.47 GiB: 28.15 to 27.68 GiB (Memory planner, Per-phase peaks; vram.md, Practical rules).
VAE_WEIGHTS = 501_306_880

# The DiT seedvr2x runs, the 7B, in any of its files' formats (weights.ACCEPTED): FP16, the
# format the fits were measured in, INT8, seedvr2x's int8 files, and the dynamic GGUF files, the
# regular 7B's and the sharp one's (GGUF_7B, GGUF_7B_SHARP). Each figure of a format's weights by
# the model's own constants, its files' headers' (seedvr2x/weights.py, which imports no torch; the
# placement moves the weights by the same): its blocks, 36 in all, and a block's parameters in
# bytes. In fp16 and int8 the 36 are alike, 0.4220 GiB in fp16, the 36 holding 15.19 of the DiT's
# 15.35 GiB (DESIGN.md, Memory planner, Host RAM), and 0.2113 GiB in int8, 7.61 of its 7.76. In a
# dynamic GGUF file they differ, from 0.105 to 0.121 GiB: a type per matrix, so a block's bytes
# are its own (block_sizes), the 36 holding 4.28 of the DiT's 4.43 GiB. The module's
# BLOCK and DIT_OTHER are fp16's; block_bytes, block_sizes and other_bytes give any format's.
# SLOTS: the buffers on the GPU its swapped blocks stream through (DESIGN.md, BlockSwap,
# rewritten), each as large as the largest block swapped (slot_bytes: blockswap.Placement's
# block_bytes).
FP16 = weights.DIT_7B
INT8 = weights.DIT_7B_INT8
GGUF_7B = weights.DIT_7B_DYN
GGUF_7B_SHARP = weights.DIT_7B_SHARP_DYN
DITS = (FP16, INT8, GGUF_7B, GGUF_7B_SHARP)
DIT = FP16
BLOCKS = len(FP16.blocks)
BLOCK = FP16.blocks[0]
SLOTS = weights.SLOTS
# The DiT's parameters outside its blocks, emb_in, txt_in, vid_in and vid_out, in bytes: 0.155
# GiB, by the model's shapes in fp16 (tests/test_shot_plan.py builds the model), the same in an int8
# file, which holds them in fp16. They leave the GPU only with the last block a VAE phase keeps
# (dit_on_gpu).
DIT_OTHER = FP16.other
# What of the DiT never leaves the GPU (seedvr2x/blockswap.py: only parameters move): the 36
# blocks' rotary_embedding_torch caches ((8192, 21) float32 each, 24.77 MB), the two text
# embeddings (1.25 MB) and the RoPE buffers, 26,022,898 bytes of tensors, and cuBLAS's workspace
# (CUBLAS_WORKSPACE and 66 KB more, which no tensor holds), 33,620,838 bytes: the CUDA tensors
# alive at a decode's entry and torch's allocated figure beside them, the VAE's weights and the
# vendored RoPE's tables apart (RTX 5090, impl-place-2, 2026-11-07,
# research/scripts/decode_peaks.py --census, a 1080p clip after windows of 19 and of 7 latents:
# 59,643,736 and 59,644,616 bytes; 59,613,696 at both once the tables are cleared at the DiT's
# end, the same census on the patch's tree). The tables (model.forget_rope_tables) are cleared
# where the DiT's work on a shot ends, so no VAE phase holds one; the windows' fit holds those a
# window makes. Before,
# 0.058 GiB, measured after windows of 1 latent at 1080p (milestone 3, 2026-10-11), that
# window's 2.6 MB of tables in it: a longer window's passed it (a 16 GB card's clip, 2026-11-06,
# its decode 1.1 MiB over its prediction, the tables kept into the VAE's phase).
DIT_BUFFERS = 59_643_736

# The DiT in seedvr2x's int8 files (INT8; DESIGN.md, Weights, "int8 before v0.1.0 is tagged"):
# its peak is the fp16 fit's with its weights' bytes in the place of fp16's (loaded), exact, and
# these terms, PROVISIONAL (impl-int8, 2026-11-02): nothing of the int8 DiT is measured yet, and
# DESIGN.md's release gate has the 96 GB card measure them before v0.1.0 is tagged, the planner
# keeping no blind spot. All of them are here:
# - INT8_PER_TOKEN and INT8_PER_WINDOW: the fp16 fit's PER_TOKEN and PER_WINDOW, the attention's
#   constant beside them as beside fp16's (DIT_ATTENTION). Between the linears the activations are
#   the same tensors, in the pipeline's bfloat16, and a linear's output is the same size; a
#   linear's input is rounded to bfloat16 W8A8_ROWS rows at a time, where autocast rounds an fp16
#   linear's input whole. Measured, they would replace these;
# - INT8_BUFFERS: what the W8A8 path holds beside a linear's output while it runs, W8A8_ROWS rows
#   at a time (seedvr2x/int8.py's linear), by the 7B's block matrices' shapes (W8A8_MATRICES): the
#   most of three moments, max(8 K, K + 4 N, 8 N) bytes a row for a matrix of K inputs and N
#   outputs: the input in float32 and rotated, 4 bytes a value each, then the codes and their
#   int32 product, 1 byte an input and 4 an output, then the product and its float32 rescaling,
#   4 and 4 an output; 96 KiB a row for either of the MLP's matrices (K or N 12,288), 0.375 GiB at
#   4,096 rows; beside them the bias in float32 and the rows' scales (W8A8_SMALL), and H, 0.25
#   MiB. The weight itself is no buffer: torch's int8 multiply is given a view of the file's
#   own (int8.transposed), never a copy. Those are CUDA's, whose kernels convert their operands'
#   dtypes as they read and write them; on the CPU, torch makes a float32 copy of the int32
#   product for its multiply by the token's scale and a float32 output for the last operation,
#   rounded into the output's rows after (TensorIterator.cpp at v2.14.1, compute_types:
#   temporaries for the CPU alone), 12 bytes an output at the most. With them counted, torch's
#   memory profiler finds the path's buffers on the CPU those within W8A8_SMALL
#   (tests/test_int8.py), and the card measures them on CUDA (its -m gpu tests). Where several
#   linears run one after the other, one at a time: once. The fp16 linear's copy of its weight in
#   bfloat16 under autocast, up to 72 MiB, which int8 doesn't make, is left in the fp16 fit's
#   constant: on the side of too much.
INT8_PER_TOKEN = PER_TOKEN
INT8_PER_WINDOW = PER_WINDOW
# The rows of a linear's input the W8A8 path takes at a time, int8.ROWS (its PROVISIONAL choice):
# a literal here, since seedvr2x/int8.py imports torch; tests/test_int8.py holds the two equal.
W8A8_ROWS = 4096
# The 7B's block matrices, (inputs, outputs), the video's and the text's alike: the attention's
# qkv and output projections and the MLP's two (tests/test_int8.py holds them to the model).
W8A8_MATRICES = ((3072, 9216), (3072, 3072), (3072, 12288), (12288, 3072))
# H, the rotation's matrix, float32, kept on the device once made (int8.hadamard).
W8A8_ROTATION = 256 * 256 * 4
# The small buffers beside the rows': the bias in its dtype and in float32, 6 bytes an output,
# and the rows' scales, about 20 bytes a row, 0.15 MiB for the 7B at 4,096 rows; 1 MiB.
W8A8_SMALL = 1 << 20


def w8a8_row_bytes(inputs: int, outputs: int) -> int:
    """The most the W8A8 path holds beside a linear's output for a row of its input, in bytes, a
    matrix of `inputs` and `outputs` (INT8_BUFFERS' three moments)."""
    return max(8 * inputs, inputs + 4 * outputs, 8 * outputs)


INT8_BUFFERS = (
    W8A8_ROWS * max(w8a8_row_bytes(*shape) for shape in W8A8_MATRICES) + W8A8_SMALL + W8A8_ROTATION
)


# The DiT in seedvr2x's dynamic GGUF files (GGUF_7B; DESIGN.md, Weights, "The dynamic GGUF before
# v0.1.0 too"): its peak is the fp16 fit's with its weights' bytes in the place of fp16's
# (loaded), exact, and one term, GGUF_BUFFERS. The activations are the fp16 fit's: between the
# linears the tensors are the same, in the pipeline's bfloat16, a linear's output the same size,
# and the attention in 16 bits as in fp16 (GGUF_PER_TOKEN, GGUF_PER_WINDOW).
GGUF_PER_TOKEN = PER_TOKEN
GGUF_PER_WINDOW = PER_WINDOW
# What a GGUF linear holds beside its output while it runs (seedvr2x/kquant.py: GGUFLinear
# dequantises each matrix straight into bfloat16 just before its multiply and frees it after):
# the matrix in bfloat16, the largest of the 7B's, 3,072 x 12,288 values. The dequantisation's
# temporaries, one chunk of kquant.BLOCKS blocks in float32 and the step's intermediates, are
# freed before the multiply's output is allocated, so they add nothing at the peak. Measured on
# the RTX 5090 (2026-11-04, tests/test_kquant.py's GPU test
# test_a_linear_s_memory_beside_its_output_is_printed, Q3_K, Q4_K and Q5_K alike): 12,288 x 3,072
# matrix, a linear of 8,280 rows (a 1080p latent) in bfloat16, 75,522,048 bytes beside the output
# at the peak, the bfloat16 matrix (75,497,472) plus 24,576. A 1080p job of 239 frames with the
# sharp GGUF (one window of 17 latents at most, FA2) peaked 0.13 GiB under the plan made with the
# earlier, provisional term (the matrix plus 52 MB of temporaries, 0.12 GiB). A literal here, since
# seedvr2x/kquant.py imports torch (tests/test_peaks.py holds the matrix equal to its size).
GGUF_MATRIX = 3072 * 12288 * 2
GGUF_BUFFERS = 75_522_048


def block_sizes(dit_model: weights.Architecture) -> tuple[int, ...]:
    """Each block's parameters in bytes in one of the DiT's formats (DITS), in the DiT's order: 36
    figures of 453,133,312 in fp16, of 226,862,080 in int8, from 112,509,952 to 129,909,760 in a
    dynamic GGUF file. ValueError for another architecture."""
    return _format(dit_model).blocks


def block_bytes(dit_model: weights.Architecture) -> int:
    """The largest block's parameters in bytes in one of the DiT's formats (DITS): 453,133,312 in
    fp16, 226,862,080 in int8, every block alike; 129,909,760 in a dynamic GGUF file, where blocks
    differ (block_sizes). ValueError for another architecture."""
    return max(block_sizes(dit_model))


def leaving_blocks(dit_model: weights.Architecture, count: int) -> tuple[int, ...]:
    """The bytes of the DiT's last `count` blocks, those that leave the GPU in some phase (the
    swapped ones come last, then those a VAE phase sends away: blockswap.dit_parts), in the DiT's
    order. ValueError for another architecture."""
    sizes = block_sizes(dit_model)
    return sizes[len(sizes) - count :]


def slot_bytes(dit_model: weights.Architecture, swapped: int) -> int:
    """The bytes of one slot of the GPU with the last `swapped` blocks swapped: the largest of
    them, as blockswap.Placement.block_bytes has it (0 with none swapped). ValueError for another
    architecture."""
    return max(leaving_blocks(dit_model, swapped), default=0)


def swapped_saving(dit_model: weights.Architecture, swapped: int) -> int:
    """What the last `swapped` blocks swapped take off the DiT's phase, in bytes: their own bytes
    less the SLOTS slots they stream through, each the largest of them (slot_bytes). In fp16 and
    int8, where the blocks are alike, one block's bytes for each beyond the SLOTS (0 for one and
    two); in a dynamic GGUF file the same sum less what the slots' size adds, so two swapped
    blocks of unequal sizes save a little less than nothing. ValueError for another
    architecture."""
    left = leaving_blocks(dit_model, swapped)
    return sum(left) - min(swapped, SLOTS) * max(left, default=0)


def other_bytes(dit_model: weights.Architecture) -> int:
    """The DiT's parameters outside its blocks in bytes in one of its formats (DITS): 166,416,512,
    fp16 in both. ValueError for another architecture."""
    return _format(dit_model).other


def weights_bytes(dit_model: weights.Architecture) -> int:
    """The DiT's parameters in bytes in one of its formats (DITS), by its files' headers: 15.35
    GiB in fp16, 7.76 in int8, 4.43 in a dynamic GGUF file. ValueError for another
    architecture."""
    found = _format(dit_model)
    return sum(found.blocks) + found.other


def loaded(dit_model: weights.Architecture) -> int:
    """What the loaded DiT holds on the GPU before a forward in one of its formats (DITS), the
    VAE's weights included, in bytes: DIT_LOADED, the fp16 fit's constant, its weights' bytes
    exchanged for the format's own (weights_bytes): 15.871 GiB in fp16, 8.285 in int8, 4.95 in a
    dynamic GGUF file. ValueError for another architecture."""
    return DIT_LOADED - weights_bytes(FP16) + weights_bytes(dit_model)


def _format(dit_model: weights.Architecture) -> weights.Architecture:
    """dit_model when it is one of the DiT's formats (DITS); ValueError for another."""
    if dit_model not in DITS:
        raise ValueError(
            f"{dit_model.name}: the DiT's figures are for {FP16.name} in fp16, in int8 or in the"
            " dynamic GGUF files"
        )
    return dit_model


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
# each slice as the decode hands it on (shot_pass._correct).
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

# What the plan leaves of the free memory the driver reports, the budget being free - MARGIN
# (DESIGN.md, Memory planner, Budget): bisected on emulated cards, a run needs 0.15-0.45 GiB
# between the free memory it starts with and the torch peak of its bounding phase, the
# decode-bound 8 GB card the most (it failed with 0.37); 0.6 keeps 0.23 over that failure
# (planner-limits.md, The margin over free memory). Measured where the driver refuses a full
# card, and a driver's alone: a run held to its card, whose driver gives memory the card doesn't
# have, counts on memory.RESERVE less already, which measures the same need under torch's own
# allocator, and takes no margin on top (shot_plan._Job.margin; DESIGN.md,
# Allocator, 2026-10-12). The planner's, here so that the library's plan for one picture
# (still_plan.py) takes the same. Measured under cudaMallocAsync, taken unchanged under torch's
# own allocator with expandable segments, the default there since 2026-11-12 (memory.startup),
# and held under them on the 96 GB card (impl-box-4, 2026-11-13; DESIGN.md, Allocator, decided
# that day): the cards emulated from 8 to 48 GB at 1080p and 4K ran every step at or under its
# plan with no out of memory, and so did 8 runs held beside a desktop.
MARGIN = round(0.6 * GIB)

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
# (MEASURED). Measured under cudaMallocAsync, taken unchanged under torch's own allocator with
# expandable segments where the driver refuses, the default there since 2026-11-12
# (memory.startup), and held under them on the 96 GB card (impl-box-4, 2026-11-13; DESIGN.md,
# Allocator, decided that day): every step of the cards emulated from 8 to 48 GB at or under
# its plan, no out of memory, the regression's frames those of the references made under the
# pool.
VAE_ROOM = round(1.25 * GIB)

# The VAE passes' room at the plan's last rung (shot_plan.plan, "the floor"): the reserve floor of
# a VAE pass, VAE_RESERVE_FLOOR under the segments and the pool and VAE_ROOM elsewhere, lowered to
# 0.5 GiB for a clip that fits no other way. DESIGN.md (Allocator, Small cards, "The floor, last
# rungs"): "the VAE's room down to 0.5 GiB, where its frames still hold (its decode about 1.2
# times as long)". PROVISIONAL: the 0.5 is design's figure, the frames and the time to be measured
# again on the GPU for each tile side.
VAE_ROOM_LEAST = round(0.5 * GIB)

# The decoder's tail a frame at a time (model.tail_frames; the plan's first rung): the decode's
# peak above the models resident, times a ratio by tile side. Allocated peaks of the shipped tail
# against the plain decoder, measured on the RTX 5090 (2026-11-06, research/scripts/hold_probe.py's
# decode of 5 latents at 1080p, the DiT off the GPU, expandable segments; research/docs/
# small-cards.md, The decoder's tail by tile side), in GiB, the VAE's 0.50 allocated before the
# step under both, by tile side: plain, tail. The ratio is taken on the part above the weights,
# (tail - 0.50) / (plain - 0.50), from 0.714 at 512 px to 0.800 at 832. Every tile side the
# planner takes up to 1088 px is measured (at 1152 px the plain decode does not fit the card
# beside its desktop's 4 GiB, the tail's 17.07); a side above the largest measured, and the
# untiled decode, take the largest ratio measured, a side under 512 px 512's (tail_ratio).
TAIL_MEASURED = (
    (512, 6.385, 4.704),
    (576, 7.946, 5.815),
    (640, 8.587, 6.574),
    (704, 10.287, 7.845),
    (768, 11.602, 9.237),
    (832, 12.797, 10.337),
    (896, 14.599, 11.304),
    (960, 16.174, 12.890),
    (1024, 17.949, 14.164),
    (1088, 20.147, 15.924),
)
TAIL_WEIGHTS_GIB = 0.50


def tail_ratio(side: int | None) -> float:
    """The share of a decode's peak above the models resident that the decoder's tail a frame at
    a time leaves (decode_peak's `tail`), for tiles of `side` pixels, None for the untiled decode:
    the ratio measured at that side (TAIL_MEASURED); for a side not measured above 512 px and for
    the untiled decode, the largest ratio measured; under 512 px, 512's."""
    ratios = {
        known: (tail - TAIL_WEIGHTS_GIB) / (plain - TAIL_WEIGHTS_GIB)
        for known, plain, tail in TAIL_MEASURED
    }
    if side is not None and side in ratios:
        return ratios[side]
    if side is not None and side < min(ratios):
        return ratios[min(ratios)]
    return max(ratios.values())


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
# Only the shapes measured are compiled (DESIGN.md, decided on 2026-10-24: "the planner keeps no
# blind spot"): encode tiles up to 2048 px or the 1080p frame whole, decode tiles up to 2048 px
# or the 1080p frame whole.
# The tiles' range was widened from 1856 and 1152 px on 2026-11-11 (design's decision), measured
# on the 96 GB card (RTX PRO 6000 Blackwell, Linux's driver) that day: a 4K job (x2 from 1080p,
# 120 frames, 2 shots, standard), the VAE compiled, encode and decode pinned to one tile size,
# the range lifted for that process alone, each step's allocated peak at or under its plan (GiB,
# plan / measured): in tiles of 1536 px (2 down, 3 across) the encode 39.92 / 39.62, the DiT
# window 56.67 / 56.59, the decode 56.29 / 55.23; in tiles of 2048 px the encode 56.23 / 56.20,
# the DiT window 56.67 / 56.59, the decode 85.55 / 82.89, a budget of 93.58 GiB; the 2048-px
# encode the tightest, 0.03 GiB under its plan. The range is read as a tile's side, the tiles
# cut off at a frame's edge included (a 1080p decode in tiles of 1856 px is 1104 rows by 1856
# and 128 columns): PROVISIONAL (impl-box-2, 2026-11-11: those measured were 4K's squares, and
# their cut-off tiles at the frame's edges). The encode's whole frame is read as the padded
# 1080p frame's two sides at most, rows and columns each (padded_size(1080x1920) = 1104x1920),
# and not as its area, the narrower reading of "the 1080p frame": a frame of its area with
# another shape is not the one measured. PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md says "the
# 1080p frame whole" and doesn't say what is as large as it).
# The decode's whole frame is the encode's, COMPILE_ENCODE_WHOLE, one constant for both (the VAE
# is compiled once and a frame is the same frame on either side). Measured on the 96 GB card
# (RTX PRO 6000 Blackwell, Linux's driver) on 2026-11-11 (impl-box-2, research/scripts/
# compile_vae.py, the VAE compiled with its causal convolutions left out of the graph as --compile
# ships, 1080p frames, 21 frames): the decode untiled peaked +33.31 GiB above the VAE's weights
# plain and +35.31 GiB compiled, 1.06 times plain (under COMPILE_PEAK), its first call no higher;
# 22% faster than plain (steady seconds a frame, compiled over plain 0.785); 143 compilations in
# 41 s. Decided by design, 2026-11-11: the compiled decode may be untiled up to the padded 1080p
# frame, as the encode.
COMPILE_ENCODE_TILE_MAX = 2048
COMPILE_ENCODE_WHOLE = (1104, 1920)
COMPILE_DECODE_TILE_MAX = 2048


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
    dit_model: weights.Architecture = FP16,
) -> int:
    """The predicted peak of the DiT's phase, in bytes: one window of `latents` latents of padded
    frames, `swapped` of the DiT's blocks swapped, the VAE's weights on the GPU meanwhile unless
    vae_in_dit is False, the DiT in the format of its file (dit_model, one of DITS;
    ValueError for another): in int8, its weights' bytes and the INT8_* terms (provisional); in a
    dynamic GGUF file, its weights' bytes and GGUF_BUFFERS (measured).

    A swapped block takes its BLOCK off the peak, but for the SLOTS on the GPU (DESIGN.md,
    BlockSwap, rewritten), so that one or two swapped blocks save nothing (swapped_saving: in a
    GGUF file, whose blocks differ, the slots are the largest swapped block's size): measured on
    seedvr2x,
    one window of 1 latent at 1080p, the peak with S blocks swapped is the peak with none less
    BLOCK x max(S - 2, 0), to 0.001 GiB (17.016 GiB with none, 16.595 with 3, 12.796 with 12,
    2.668 with 36; milestone 3, 2026-10-11, as tests/test_placement.py measures it). numz's own
    swap, one block there at a time, saved 0.4227 GiB x (S - 1)."""
    per_window = INT8_PER_WINDOW if _format(dit_model) == INT8 else PER_WINDOW
    windows = per_window * attention_windows(padded, latents)
    least = dit_least(padded, latents, swapped, vae_in_dit, dit_model)
    return round(least + windows)


def dit_least(
    padded: tuple[int, int],
    latents: int,
    swapped: int,
    vae_in_dit: bool,
    dit_model: weights.Architecture = FP16,
) -> float:
    """dit_peak less its attention windows' part, in bytes: never above the peak, and counted
    without the window functions, which lay out every window of each length they are asked."""
    if not 0 <= swapped <= BLOCKS:
        raise ValueError(f"{swapped} blocks swapped of the DiT's {BLOCKS}")
    int8 = _format(dit_model) == INT8
    per_token = (INT8_PER_TOKEN if int8 else PER_TOKEN) * tokens(padded, latents)
    buffers = INT8_BUFFERS if int8 else GGUF_BUFFERS if dit_model.layout == weights.GGUF else 0
    least = (
        loaded(dit_model) + DIT_ATTENTION + buffers + per_token - swapped_saving(dit_model, swapped)
    )
    return least if vae_in_dit else least - VAE_WEIGHTS


def dit_working(
    padded: tuple[int, int],
    latents: int,
    dit_model: weights.Architecture = FP16,
    windows: int | None = None,
) -> int:
    """P of a DiT step, in bytes: what one window of `latents` latents of padded frames takes of the
    GPU over the weights the DiT keeps resident, in the format of `dit_model` (one of DITS;
    ValueError for another): the attention constant (DIT_ATTENTION), the per-token term, the
    per-window term and the format's scratch buffers (INT8_BUFFERS, GGUF_BUFFERS: scratch, no
    weights). Never the DiT's weights, the swapped blocks' slots or the VAE's weights, which
    dit_peak counts besides (loaded, swapped_saving, VAE_WEIGHTS): so it depends on neither the
    blocks swapped nor the VAE's place, and dit_peak(...) equals what is resident (loaded(dit_model)
    less swapped_saving, less VAE_WEIGHTS where the VAE is away) plus this, to a byte (dit_peak
    rounds the sum, this its own part). The P of the DiT's reserve (reserve; DESIGN.md, Allocator,
    Small cards). `windows`: the attention windows the per-window term counts, where the caller
    bounds a set of pictures by the most any takes (still_plan.windows_most); None counts the
    window's own (attention_windows)."""
    int8 = _format(dit_model) == INT8
    per_token = (INT8_PER_TOKEN if int8 else PER_TOKEN) * tokens(padded, latents)
    buffers = INT8_BUFFERS if int8 else GGUF_BUFFERS if dit_model.layout == weights.GGUF else 0
    per_window = INT8_PER_WINDOW if int8 else PER_WINDOW
    counted = attention_windows(padded, latents) if windows is None else windows
    return round(DIT_ATTENTION + buffers + per_token + per_window * counted)


def vae_working(
    peak: int, resident: int | None, swapped: int, dit_model: weights.Architecture = FP16
) -> int:
    """P of a VAE pass, in bytes: its predicted `peak` (encode_peak, decode_peak) over the
    models resident in its phase, the VAE's weights and what of the DiT stays on the GPU
    (dit_on_gpu's `resident`, `swapped` and `dit_model`). The P of the VAE's reserve (reserve);
    the pass's own terms (ENCODE_EXTRA, the colour correction's) are in it."""
    return peak - VAE_WEIGHTS - dit_on_gpu(resident, swapped, dit_model)


def dit_on_gpu(resident: int | None, swapped: int, dit_model: weights.Architecture = FP16) -> int:
    """What of the DiT is on the GPU in a VAE phase, in bytes, `swapped` of its blocks being
    swapped, by what the phase is entered with (Plan.resident_encode, Plan.resident_decode), as
    blockswap.Placement.enter leaves its parameters there (Placement.device_bytes), with what
    never moves (DIT_BUFFERS), the DiT in the format of its file (dit_model, one of DITS):
    - None: nothing moves, the DiT as its own phase has it: its other parameters (DIT_OTHER),
      the blocks that aren't swapped, and the slots, one per swapped block up to SLOTS;
    - 0: nothing of its parameters, the other ones away with the blocks;
    - a number up to the blocks that aren't swapped: that many of them, and its other
      parameters; the slots are freed, and come back with the DiT's phase.
    ValueError for a count the DiT hasn't, as the loader refuses it (model.load_models)."""
    fixed = BLOCKS - swapped
    if not 0 <= swapped <= BLOCKS or not (resident is None or 0 <= resident <= fixed):
        raise ValueError(f"{resident} blocks resident, {swapped} swapped, of the DiT's {BLOCKS}")
    sizes, other = block_sizes(dit_model), other_bytes(dit_model)
    if resident is None:
        slots = min(swapped, SLOTS) * slot_bytes(dit_model, swapped)
        return DIT_BUFFERS + other + sum(sizes[:fixed]) + slots
    if resident == 0:
        return DIT_BUFFERS
    return DIT_BUFFERS + other + sum(sizes[:resident])


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
    dit_model: weights.Architecture = FP16,
) -> int:
    """The predicted peak of the VAE's encode of padded frames, in bytes: its peak untiled
    (tiles None) or the most a tile of its grid takes (_fit), above what is on the GPU before
    the call, the VAE's weights and what of the DiT stays there (dit_on_gpu's resident, swapped
    and dit_model); and what seedvr2x's encode holds over it (ENCODE_EXTRA). Whatever the shot's
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
    there = dit_on_gpu(resident, swapped, dit_model)
    return round(above) + ENCODE_EXTRA + VAE_WEIGHTS + there


def decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
    compiled: bool = False,
    dit_model: weights.Architecture = FP16,
    *,
    frames: int = SLICE,
    tail: bool = False,
) -> int:
    """The predicted peak of the VAE's decode of padded frames, in bytes, its peak above what is
    on the GPU before the call as encode_peak has the encode's (without ENCODE_EXTRA, the
    encode's), whatever the shot's length too: a decode takes its latents from system memory
    slice by slice; with split, the colour correction too, which corrects a slice of the decode
    at once, `frames` frames of the output's size (frames.output_size): a video's slice holds
    SLICE at most, which the video's planner (shot_plan.py) passes, a still picture one. ValueError
    for tiles of which one covers the whole padded frame, as encode_peak, and for frames under 1.

    tail: the decoder's tail a frame at a time (model.tail_frames, a plan's last rungs): the
    tile's peak above the weights is tail_ratio of its tile side times its plain one, the
    colour correction's term unchanged (it is a pass of its own after the decoder's).

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
    measured; tests/test_shot_plan.py, test_first_matrix)."""
    # compiled: the tile's peak is COMPILE_PEAK times its plain one (encode_peak); the correction's
    # term, a pass of its own on the GPU, is not compiled.
    if frames < 1:
        raise ValueError(f"the colour correction's term over {frames} frames")
    fit = _fit(padded, tiles, 1) * (COMPILE_PEAK if compiled else 1.0)
    if tail:
        fit *= tail_ratio(None if tiles is None else tiles.size)
    corrected = SPLIT_FRAME_4K * frames * output[0] * output[1] / PIXELS_4K if split else 0
    above = fit + corrected if tiles is None else max(fit, corrected)
    return round(above) + VAE_WEIGHTS + dit_on_gpu(resident, swapped, dit_model)


# ---- The reserve of a held run, by the step (DESIGN.md, Allocator, "The reserve under the segments
# is measured by the step", decided 2026-11-06; research/docs/small-cards.md) ---------------------

# What a held run's plan leaves torch beyond a step's predicted peak P (dit_working, vae_working)
# under torch's own allocator with expandable segments (memory.SEGMENTS): max(floor, share x P).
# Measured by place-study on an RTX 5090 under WSL2 with a card left exactly its size, at night:
# the lean windows want 0.18, 0.20 then 0.29 x P beyond their peak to keep their speed (a room
# short of it costs 2.7 to 3.4 times the window's time), and keep it from there down to no bit
# changed; a VAE pass keeps its speed from 0.2 x P. The DiT's share is 0.33 (design's ruling of
# 2026-11-07), the least to two decimals that leaves every want two sessions measured 0.04 GiB
# more, with every block swapped, the VAE away and torch held 4.5 GiB under the card's free memory
# (hold_probe.py): on 2026-11-06, the int8 sharp 7B at 1080p, windows of 19, 27 and 31 latents (P
# 9.34, 13.21, 15.14 GiB) at speed down to 3.0, 4.0 and 3.25 GiB (the 19-latent one slower at
# 2.75); on 2026-11-07 (impl-place-2, the room from 0.40 to 0.20 x P by 0.05 GiB, the usual speed
# within 5% of the roomy runs'), int8 at 19 and 27 latents down to 2.70 and 3.95 GiB (P 9.343,
# 13.212), fp16 down to 2.80 and 4.05 (P 9.342, 13.211). The same window wanted 0.30 GiB more on
# one night than on the other: 0.33 x 9.34 = 3.08 GiB covers its 3.0 by 0.08, where 0.32 leaves
# 2.99, and fp16's 27-latent window, 4.36 for 4.05. A room 0.05 GiB under a want cost 1.1 to 2.1
# times the window's time. The DiT's rule is capped (DIT_RESERVE_CAP, below); a VAE pass's is
# not. The floors: the planner's MARGIN for a window (0.6 GiB) and VAE_ROOM for a pass (1.25
# GiB), the figures a driver that refuses has always left, kept as the rule's lower bound.
DIT_RESERVE_FLOOR = MARGIN
DIT_RESERVE_SHARE = 0.33
# The cap on a DiT step's reserve under the segments: max(0.6, min(0.33 x P, 4.5)) GiB, the cap
# from P = 13.6 GiB. Measured by
# hold_probe.py's scan (RTX 5090, 2026-11-06, the sharp 7B in int8, every block swapped, the VAE
# away, expandable segments, torch held 4.5 GiB under the card's free memory, each window's room
# scanned down by 0.25 GiB): at 1080p, windows of 19, 27 and 31 latents (P 9.34, 13.21, 15.14 GiB)
# kept their usual speed (13.3, 19.1, 25.4 s) down to 3.0, 4.0 and 3.25 GiB of room (0.32, 0.30,
# 0.21 x P), slower below: the want stops growing past 27 latents, so the rule stops at 4.5 GiB,
# 0.45 GiB over the largest want measured since (fp16's 27-latent window, 4.05 GiB, 2026-11-07;
# the share above). PROVISIONAL beyond P = 15 GiB, unmeasured (design's ruling of 2026-11-06
# asked for this scan). The VAE passes' rule has no cap.
DIT_RESERVE_CAP = round(4.5 * GIB)
VAE_RESERVE_FLOOR = VAE_ROOM
VAE_RESERVE_SHARE = 0.2
# The same under cudaMallocAsync (memory.POOL), on a driver that lends: torch's limit binds the
# allocated bytes alone, and the pool held more of the card beside them, slowing nothing and moving
# no bit: that pool is the reserve there, counted on the card. The DiT's: max(0.85, 0.24 x P).
# Measured by hold_probe.py's scan (RTX 5090, 2026-11-06, the lean attention, 36 blocks swapped,
# the VAE away, torch's limit lowered from 3.0 to 0 GiB over the peak by 0.25: the bits equal and
# no step slower at any rung); the pool held beyond the allocated peak, at the most over the
# rungs, 0.35 GiB at 1 latent of 1080p (P 0.59), 0.74 at 3 (P 1.55), 0.71 at 5 (2.51), 0.81 at 7
# (3.51), 1.22 at 11 (5.47), 1.70 at 19 (9.34), and 0.80 at one latent of 4K (2.03): the rule
# covers every one by 0.03 GiB at the least (7 latents: 0.842 for 0.811). A VAE pass: 0.35 x P
# (DESIGN.md, Allocator, Small cards), under the same floor as the segments' (1.25 GiB), a floor
# design has not measured below: PROVISIONAL, a question for design.
POOL_DIT_FLOOR = round(0.85 * GIB)
POOL_DIT = 0.24
POOL_VAE = 0.35
# The phases a reserve is asked for: a DiT window, a VAE encode, a VAE decode.
DIT_PHASE, ENCODE_PHASE, DECODE_PHASE = "dit", "encode", "decode"
PHASES = (DIT_PHASE, ENCODE_PHASE, DECODE_PHASE)


def reserve(
    phase: str,
    working: int,
    allocator: str,
    margin: float | None = None,
    vae_floor: int | None = None,
) -> int:
    """What a held run's plan leaves torch beyond a step's predicted peak, in bytes, for the
    step `phase` (DIT_PHASE, ENCODE_PHASE or DECODE_PHASE; ValueError for another) whose peak
    over the models resident in it is `working` bytes (P: dit_working, vae_working), under
    `allocator` (memory.SEGMENTS, PLAIN or POOL; ValueError for another):
    - SEGMENTS: a DiT step max(DIT_RESERVE_FLOOR, min(DIT_RESERVE_SHARE x P, DIT_RESERVE_CAP)),
      a VAE pass max(VAE_RESERVE_FLOOR, VAE_RESERVE_SHARE x P);
    - POOL: a DiT step max(POOL_DIT_FLOOR, POOL_DIT x P), a VAE pass max(VAE_RESERVE_FLOOR,
      POOL_VAE x P);
    - PLAIN (torch's own allocator without the segments, and a driver that refuses): what the
      plan leaves beyond the peak besides the flat memory.RESERVE it takes off the counted figure
      (memory.counted), which is not in this one: 0 for a DiT step, its margin being inside that
      figure, and VAE_ROOM for a VAE pass, the room it is judged by besides, as before.
    `margin`: the allocator's margin the environment probe measured, a share of P (memory.Reading.
    margin, probe.Probed.allocator_margin), None where none is: where it is larger than the
    rule's share (SEGMENTS and POOL), it is the share, the plan counting the larger of the two.
    PLAIN's flat reserve takes none.
    `vae_floor`: the plan's last rung (VAE_ROOM_LEAST), bytes: the floor of a VAE pass's rule
    (VAE_RESERVE_FLOOR, or VAE_ROOM under PLAIN) replaced by it, the share kept; a DiT step's rule
    is unchanged; None, the rule's own.
    The step needs its peak plus this under the figure the plan counts on."""
    floor, share = reserve_terms(phase, allocator, margin, vae_floor)
    wanted = round(share * working)
    # A probe's margin over the rule's share is a measure of this environment's allocator: it
    # is counted whole, the cap being the rule's own.
    if phase == DIT_PHASE and allocator == SEGMENTS and share == DIT_RESERVE_SHARE:
        wanted = min(wanted, DIT_RESERVE_CAP)
    return max(floor, wanted)


def reserve_terms(
    phase: str, allocator: str, margin: float | None = None, vae_floor: int | None = None
) -> tuple[int, float]:
    """The floor, in bytes, and the share of P of reserve's rule max(floor, share x P) for the
    step `phase` under `allocator` (ValueError for a name that isn't one): reserve's own table,
    less the DiT step's cap under SEGMENTS (DIT_RESERVE_CAP, which reserve applies),
    for whoever solves the rule for P (still_plan._largest_input). `margin`: the probe's measured
    share of P (reserve), which replaces the rule's share where larger, under SEGMENTS and POOL;
    PLAIN's takes none. `vae_floor`: reserve's. Nothing builds the measurement yet (probe.py):
    this is the reading side."""
    floor, share = _terms(phase, allocator)
    if vae_floor is not None and phase != DIT_PHASE:
        floor = vae_floor
    if margin is not None and allocator != PLAIN:
        share = max(share, margin)
    return floor, share


def _terms(phase: str, allocator: str) -> tuple[int, float]:
    """The rule's own floor and share (reserve_terms without the probe's margin)."""
    if phase not in PHASES:
        raise ValueError(f"phase {phase!r}: one of {', '.join(PHASES)}")
    dit = phase == DIT_PHASE
    if allocator == SEGMENTS:
        return (
            (DIT_RESERVE_FLOOR, DIT_RESERVE_SHARE)
            if dit
            else (VAE_RESERVE_FLOOR, VAE_RESERVE_SHARE)
        )
    if allocator == POOL:
        return (POOL_DIT_FLOOR, POOL_DIT) if dit else (VAE_RESERVE_FLOOR, POOL_VAE)
    if allocator == PLAIN:
        return (0 if dit else VAE_ROOM), 0.0
    raise ValueError(f"allocator {allocator!r}: one of {SEGMENTS}, {PLAIN}, {POOL}")


# ---- One still picture: one frame (DESIGN.md, Still images, "The plan for one picture") -------

# One frame's VAE pass against the video slice's figure (_fit), by the tile's side in pixels:
# the encode's then the decode's, the most measured over the pictures at that side, rounded up to
# the hundredth. Measured on the RTX 5090 (2026-11-03, design's agent, the desktop live; SDPA,
# torch 2.14.1, backend:native,expandable_segments:True), a fresh process a case, at 1080p, 4K,
# 6000x4000 and 8K: a tile's peak is the same at every picture size to the MiB (the 512-px
# encode 1.037 GiB in all four), the tiles cut off at 1080p's 1,104 rows taking the most at 1536
# px (0.283 and 0.239), and the 1024-px decode's 1024x960 tile more than its square (4.784 GiB
# where 6000x4000's grid, without it, took 4.430). Between two measured sides a tile takes the
# larger of their two ratios; past 2048 px, which only a pin gives, the slice's own figure
# (ONE_FRAME_UNTILED_MEASURED). PROVISIONAL until the card's batch measures one frame at every
# tile size.
ONE_FRAME_TILED: dict[int, tuple[float, float]] = {
    512: (0.30, 0.33),
    768: (0.33, 0.33),
    1024: (0.34, 0.28),
    1536: (0.29, 0.24),
    2048: (0.24, 0.20),
}
# The 4K picture's one-frame decode at every side the planner takes, 512 to 2048 px by 64: its peak
# against the slice's figure (_fit) at the 4K frame (2192x3840), rounded up to the thousandth.
# Measured on the RTX 5090 (impl-place-2, 2026-11-07, research/scripts/decode_peaks.py through
# seedvr2x's own command; every DiT block swapped, SDPA, expandable segments; the same bits and
# the same peak to the byte twice at each side), the peak over what was allocated before the
# decode with cuBLAS's workspace (CUBLAS_WORKSPACE) counted in it, as the 2026-11-03 figures
# counted it, which these reproduce at 768, 1024, 1536 and 2048 px to 0.1 MiB. The whole decode
# step, its copies and colour correction as a run makes them, peaked during its tiles at every
# side: the copies' moment and the correction (still_plan.COPIES, colour_one_frame) come once the
# tiles' working memory is gone, as the plan counts them. A side takes the larger of this and of
# ONE_FRAME_TILED's ratio (one_frame_ratio), so that no frame's prediction falls (no other frame
# was measured at these sides): 832 px, which took 768's 0.33, measured 0.3463 (4.26 GiB above
# the models where 4.06 was predicted; the emulated 8 GB card's 4K picture ran 0.17 GiB over it
# on 2026-11-06); at every other side the ratio before stands at or over the measure, by 1.31
# times at the most (1984 px).
ONE_FRAME_DECODE_4K: dict[int, float] = {
    512: 0.321,
    576: 0.320,
    640: 0.314,
    704: 0.313,
    768: 0.328,
    832: 0.347,
    896: 0.278,
    960: 0.287,
    1024: 0.275,
    1088: 0.255,
    1152: 0.255,
    1216: 0.235,
    1280: 0.238,
    1344: 0.224,
    1408: 0.227,
    1472: 0.217,
    1536: 0.218,
    1600: 0.210,
    1664: 0.204,
    1728: 0.206,
    1792: 0.201,
    1856: 0.197,
    1920: 0.195,
    1984: 0.184,
    2048: 0.192,
}
# An untiled pass, by the frame it was measured on. At 1080p (1104x1920): the encode 0.255 and the
# decode 0.207 of their figures (2026-11-03). At 4K (2192x3840), on the RTX 5090 (place-study's
# night, 2026-11-05 to 06, picture_phases.py --then, 36 blocks swapped, a fresh process, the same
# bits twice): the encode 14.621 GiB above the weights and the picture's frame (16.193 allocated
# over 1.571 before it), 0.194 of its figure (75.27 GiB), and the decode 23.654 GiB above the
# weights (25.178 over 1.524), 0.1735 of its figure (136.32 GiB): the ratio falls with the frame,
# the figure growing with the area faster than the pass does, so one ratio would count the 4K
# decode at 28.6 GiB (0.21) or the 1080p one at 6.1 (0.1735, under its 7.22 measured). Each
# measured frame has its own, rounded up to the thousandth for the 4K pair (14.68 and 23.72 GiB
# predicted) and to the hundredth for the 1080p pair (as before: 5.16 and 7.34).
ONE_FRAME_UNTILED_1080P = (0.26, 0.21)
ONE_FRAME_UNTILED_4K = (0.195, 0.174)
# Kept for the 1080p frame: the ratios of a frame within it, and of `padded` None.
ONE_FRAME_UNTILED = ONE_FRAME_UNTILED_1080P
# The largest padded frames one frame's untiled pass was measured on, (rows, columns), the
# encode's and the decode's, in either orientation (4K's, both, since 2026-11-06): past them, and
# for tiles past TILE_MAX, which only a pin gives, a picture's pass takes the video slice's
# figure, 3 to 5.6 times what one frame took where it was measured, until the card's batch
# measures them (DESIGN.md, Still images, Measured before it ships: "the planner keeping no blind
# spot"; the review of 2026-11-03).
ONE_FRAME_UNTILED_MEASURED = ((2192, 3840), (2192, 3840))
# The bytes the picture's own frame takes on the GPU beside the encode, per pixel of the padded
# frame: 3 channels in bfloat16 (model.COMPUTE_DTYPE), alive through the pass (still.upscale's
# `frame`), and allocated before it, so outside the measures above (6 B a pixel: 0.047 GiB at 4K,
# the case's own figure).
ONE_FRAME_INPUT = 3 * 2
# The colour correction at one frame, beside split's own term (SPLIT_FRAME_4K at the output's
# area): the decoded frame, 3 channels in bfloat16 at the padded frame's size; its float32
# reference, 3 channels at the padded frame's size (a crop of it); and cuBLAS's workspace, which
# stays allocated after the decode (CUBLAS_WORKSPACE). Measured on the RTX 5090 (2026-11-03): split
# itself 0.116, 0.465, 1.343 and 1.856 GiB at 1080p, 4K, 6000x4000 and 8K, under its term (0.14,
# 0.56, 1.62 and 2.24), the whole 0.183, 0.637, 1.780 and 2.448, which this counts 0.207, 0.732,
# 2.060 and 2.833.
ONE_FRAME_DECODED = 3 * 2
ONE_FRAME_REFERENCE = 3 * 4
# cuBLAS's workspace, allocated by torch's allocator at the first matrix product on a stream and
# kept for the process: the 32 MiB that stayed allocated after every decode measured on the RTX
# 5090 (2026-11-03; torch's default for compute capability 9.0 and up).
CUBLAS_WORKSPACE = 32 * MIB

# What torch's allocator reserves over a phase's allocated peak, as a share of what the phase
# allocates above the models on the GPU in it: the most measured on the RTX 5090 (2026-11-03,
# reserved and allocated peaks, a fresh process a case) is 0.182, the DiT under FlashAttention 2
# at 1080p (0.204 GiB over 1.12 above the models), then 0.167, the 2048-px decode (2.02 over
# 12.10), and 0.163, the DiT at 8K with its 36 blocks swapped (2.75 over 16.82); SDPA's DiT 0.13
# at the most, the encodes 0.09. The image server's reservation counts it beside each phase's
# peak (still_plan.Peak.reserved): what it holds for torch is what torch reserves, not what it
# allocates. PROVISIONAL until the card's batch.
RESERVED_OVER = 0.2


def one_frame_ratio(
    tiles: Tiles | None, column: int, padded: tuple[int, int] | None = None
) -> float:
    """One frame's VAE pass against the video slice's figure (_fit), the encode's (column 0) or
    the decode's (1), for the pass in `tiles` (None: untiled) over frames padded to `padded`:
    ONE_FRAME_TILED at a measured side, the larger of its two neighbours' between them, the
    smallest's below, and for the decode the larger of that and of the 4K frame's measure at the
    side (ONE_FRAME_DECODE_4K); ONE_FRAME_UNTILED_1080P / _4K untiled within the frame measured
    (ONE_FRAME_UNTILED_MEASURED, either orientation; `padded` None: within, the 1080p's). 1.0,
    the slice's own figure, past what was measured: an untiled pass over a larger frame, a tile
    past the largest side measured."""
    if tiles is None:
        measured = ONE_FRAME_UNTILED_MEASURED[column]
        if padded is not None and (max(padded) > max(measured) or min(padded) > min(measured)):
            return 1.0
        # The 4K ratio at a frame as large as 4K's in area. A frame between 1080p's and 4K's takes
        # the 1080p ratio, but never more than the 4K frame's whole prediction (a pass takes no
        # more on a smaller frame than on the larger one it was measured on), so that the
        # prediction grows with the frame, the proof of still_plan._worst resting on that.
        fitted = ONE_FRAME_UNTILED_MEASURED[column]
        if padded is None:
            return ONE_FRAME_UNTILED_1080P[column]
        if padded[0] * padded[1] >= fitted[0] * fitted[1]:
            return ONE_FRAME_UNTILED_4K[column]
        cap = ONE_FRAME_UNTILED_4K[column] * _fit(fitted, None, column) / _fit(padded, None, column)
        return min(ONE_FRAME_UNTILED_1080P[column], cap)
    sides = sorted(ONE_FRAME_TILED)
    if tiles.size > sides[-1]:
        return 1.0
    if tiles.size in ONE_FRAME_TILED:
        rule = ONE_FRAME_TILED[tiles.size][column]
    elif tiles.size < sides[0]:
        rule = ONE_FRAME_TILED[sides[0]][column]
    else:
        above = min(side for side in sides if side > tiles.size)
        below = max(side for side in sides if side < tiles.size)
        rule = max(ONE_FRAME_TILED[below][column], ONE_FRAME_TILED[above][column])
    if column == 1:
        return max(rule, ONE_FRAME_DECODE_4K.get(tiles.size, 0.0))
    return rule


def colour_one_frame(padded: tuple[int, int], output: tuple[int, int]) -> int:
    """The colour correction of one picture on the GPU, in bytes, above the models: split's own
    term at the output's size, the decoded frame and the float32 reference at the padded frame's,
    and cuBLAS's workspace (ONE_FRAME_DECODED, ONE_FRAME_REFERENCE, CUBLAS_WORKSPACE)."""
    split = SPLIT_FRAME_4K * output[0] * output[1] / PIXELS_4K
    frames = (ONE_FRAME_DECODED + ONE_FRAME_REFERENCE) * padded[0] * padded[1]
    return round(split) + frames + CUBLAS_WORKSPACE


def picture_encode_peak(
    padded: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    dit_model: weights.Architecture = FP16,
) -> int:
    """The predicted peak of one picture's encode, in bytes: one frame's pass (one_frame_ratio of
    the slice's _fit, the measures holding what seedvr2x's encode holds over the fit, so no
    ENCODE_EXTRA), the picture's frame beside it (ONE_FRAME_INPUT), the VAE's weights and what of
    the DiT stays on the GPU (dit_on_gpu). Without the room every VAE pass is held to (VAE_ROOM).
    ValueError as encode_peak."""
    above = one_frame_ratio(tiles, 0, padded) * _fit(padded, tiles, 0)
    frame = ONE_FRAME_INPUT * padded[0] * padded[1]
    return round(above) + frame + VAE_WEIGHTS + dit_on_gpu(resident, swapped, dit_model)


def picture_decode_peak(
    padded: tuple[int, int],
    output: tuple[int, int],
    tiles: Tiles | None = None,
    resident: int | None = None,
    swapped: int = 0,
    split: bool = False,
    dit_model: weights.Architecture = FP16,
) -> int:
    """The predicted peak of one picture's decode, in bytes: one frame's pass (one_frame_ratio),
    or with split the colour correction after it (colour_one_frame), the larger, tiled or not: a
    picture's decode is closed before its correction starts (still.upscale), so their working
    memory is never alive at once; then the VAE's weights and what of the DiT stays on the GPU.
    Without VAE_ROOM. ValueError as decode_peak."""
    above = round(one_frame_ratio(tiles, 1, padded) * _fit(padded, tiles, 1))
    if split:
        above = max(above, colour_one_frame(padded, output))
    return above + VAE_WEIGHTS + dit_on_gpu(resident, swapped, dit_model)


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
    an overlap that isn't a multiple of 8 px (tilegrid.grid), which no pin is (main._tile) and a
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
    tile = _tile if _remembered_tile is None else _remembered_tile
    return max(tile(tiles.size, *shape, column) for shape in tile_shapes(padded, tiles))


# The tiles' figures remembered while a search plans the same tiles again and again (the image
# server's start, apps/imgserver/reservation.py's make): the start refused for two 4K pictures on 22
# GiB makes 301 plans of one picture, 270 of them distinct, whose floor proofs (still_plan._worst,
# 460 of them, one per bound, phase and tile side) count 19.0 million tiles, 353 thousand of them
# distinct (2026-11-08). The memo is the search's alone, made at its start and dropped at its end
# (tiles_remembered), never the process's: a figure of this module a test sets (CUT_OFF_EXTRA,
# MEASURED) is read again by the next search, and memory is held only while a search runs. At most
# TILE_MEMO tiles, the least recently used dropped past it: about 120 MiB at the most, extrapolated
# from the refused start's 353 thousand measured at 81 MiB (its process's peak, 550 MiB without the
# memo, 631 with), the start 67 s without and 28 s with. It changes no figure: a tile's is a
# function of its four arguments and of this module's figures, which nothing changes during a
# search.
TILE_MEMO = 2**19
_remembered_tile: Callable[[int, int, int, int], float] | None = None


@contextmanager
def tiles_remembered() -> Generator[None]:
    """Inside it, each tile's figure (_tile) made once and remembered (TILE_MEMO), for a search that
    plans the same tiles again and again; nested, the outer memo serves."""
    global _remembered_tile
    if _remembered_tile is not None:
        yield
        return
    _remembered_tile = lru_cache(maxsize=TILE_MEMO)(_tile)
    try:
        yield
    finally:
        _remembered_tile = None


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
    frame's, rows and columns each (COMPILE_ENCODE_WHOLE); the decode's the same, with
    COMPILE_DECODE_TILE_MAX px for its tiles. The DiT's phase is no pass of the VAE: always
    within."""
    if phase == "encode":
        if tiles is None:
            return padded[0] <= COMPILE_ENCODE_WHOLE[0] and padded[1] <= COMPILE_ENCODE_WHOLE[1]
        return tiles.size <= COMPILE_ENCODE_TILE_MAX
    if phase == "decode":
        if tiles is None:
            return padded[0] <= COMPILE_ENCODE_WHOLE[0] and padded[1] <= COMPILE_ENCODE_WHOLE[1]
        return tiles.size <= COMPILE_DECODE_TILE_MAX
    return True


def compile_range_said(phase: str) -> str:
    """The shapes measured compiled for a VAE pass, in words (in_compile_range)."""
    rows, columns = COMPILE_ENCODE_WHOLE
    tile_max = COMPILE_ENCODE_TILE_MAX if phase == "encode" else COMPILE_DECODE_TILE_MAX
    kind = "encode" if phase == "encode" else "decode"
    return (
        f"{kind} tiles up to {tile_max} px, or untiled on a frame padded to"
        f" {columns}x{rows} at most"
    )


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
