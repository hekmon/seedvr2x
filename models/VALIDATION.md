# Validation: what phase 2's files do to the video

> Status: **measured on the GPU** on 2026-10-07, for [DESIGN.md](../seedvr2x/DESIGN.md#weights)
> (Weights: phase 2's files are validated by GPU runs before they ship). Tier 1 (1080p, 8 clips)
> done for every file of the 7B and the sharp 7B, in every way it multiplies; the 4K slice done
> for the 7B's fp8 (W8A16) and Q8_0 and for five of the sharp's files (int8, the dynamic GGUF,
> Q4_K, fp8 W8A8, Q8_0); the user's eyes given on the 7B's 1080p crops, not yet on the 4 GB
> files' nor on the sharp's. numz's SeedVR2 at `4490bd1` ran our safetensors files through
> [`gpu/ck_patch.py`](gpu/ck_patch.py) (comfy-kitchen 0.2.37's layers) and our GGUF files through
> its own loader; colour.md's tools scored them ([colour.md](../research/docs/colour.md#reproduce)).
> What each format does to the weights: [FORMATS.md](FORMATS.md). The GPU has a power-cap fault:
> no speed is measured here.

In short (each file against its own model's float16 output at the same seed, 1080p, 8 clips of
45 frames, a ×2 upscale of a mildly degraded input; judged against how much the float16 model's
own seeds differ):

- **Every 8-bit file is as close to its float16 model as another seed of it, or closer**, for
  both models: no score past the calibrated line but one, on one clip (the 7B's int8: LPIPS
  +0.0002 on live-slow, 2.85 times the seed spread, the line at 2.7). Distance to the float16
  output at the same seed: Q8_0 56.3 dB (the sharp's 54.4), int8 53.8 (51.9), fp8 multiplied in
  16 bits 50.6 (48.5), in 8 bits 48.0 (46.0), where two seeds of the float16 model are 40.6 dB
  apart (the sharp's 39.2). At 4K too, against two seeds: the 7B's fp8 (W8A16) and Q8_0 (50.4
  and 56.1 dB; its seeds 39.5 apart), the sharp's Q8_0, int8 and fp8 W8A8 (55.5, 52.8 and 46.8
  dB; its seeds 39.2), fp8 W8A8 past the line on 3 cells only, where the sharp's two seeds agree
  almost exactly. Its first 4K runs came out NaN: comfy-kitchen's quantizer of the activations
  breaks from 2^32 values, which ck_patch.py now works around
  ([below](#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices)).
- **At 4 bits, a file moves the output about as much as a seed does** (37.3 to 44.2 dB), and
  where it departs shows a pattern:
  - the 7B's Q4_K (ours, the same as numz's Q4_K_M on video too) is softer: fine detail 16% lower
    on one anime clip, lower on live action, within the calibrated line (2.68 times the spread at
    worst); the full-reference scores put it closer to the source (PSNR-Y +0.30 dB);
  - the 7B's dynamic GGUF stays the closest of the 4 GB files to float16 (44.2 dB, 0.8 to 3.3 dB
    closer than Q4_K on every clip) and keeps the detail Q4_K loses on that anime clip, but the
    cartoon clip's DISTS goes past the line (3.0 times the spread); its control, every matrix
    Q4_K with the same importance, sits beside it (43.7 dB, within it on nearly every score);
  - NVFP4 is clearly worse multiplied in 4 bits (W4A4: DISTS on the cartoon clip 11.8 times the
    spread, on an anime clip 3.6 times; 38.9 dB, further than a seed), and softer on live action
    multiplied in 16 bits (W4A16, 42.1 dB).
- **The sharp 7B's 4-bit files all go past the line somewhere**, its own seeds scattering more
  than the 7B's notwithstanding: the cartoon clip's DISTS on all five, live action's finest
  texture on four, an anime clip's flicker on three. The closest: its dynamic GGUF (42.5 dB, 6
  cells past the line) and its Q4_K with importance (42.1 dB, 4 cells); its Q4_K has 9 (41.4 dB).
  **The 4 GB pick is its dynamic GGUF:** the closest to float16 at 1080p, and at 4K too (43.3 dB
  against Q4_K's 42.0, closer on all 5 shots; the Q4_K with importance did not run there), where
  it crosses the line only on digital live action's finest texture (lower, as on live action at
  1080p) and on one colour cell of near-zero spread. The eyes have yet to judge it.
- **The eyes:** on 10 windows of 5 clips at 1080p, the 7B's fp8 (both ways), int8, Q8_0, Q4_K,
  numz's Q4_K_M and NVFP4 W4A4 beside float16: nothing to report on any window (the user,
  2026-10-07). The 4 GB files' crops and the sharp's, at 1080p and 4K: not given when this was
  written.
- **The runtime path is the one meant:** exactly the 288 marked matrices become comfy-kitchen
  layers, every other tensor equal to the float16 file's; each mode runs its kernel on all 288
  and none dequantizes where it must not (W8A8 torch's scaled_mm, int8 comfy-kitchen's
  int8_linear, W4A4 its scaled_mm_nvfp4). On the run's own activations a layer's output moves by
  0.76% in int8, 1.50% in fp8 W8A16, 2.09% in W8A8, 4.93% in NVFP4 W4A16 and 7.18% in W4A4
  (medians). The check mode and BlockSwap change no bit of the output.

## Method

### The runs

- **numz's CLI at `4490bd1`**, one batch of 45 frames, `flash_attn_2`, `--color_correction
  none`, seed 42: the float16 reference runs' flags (colour's dumps, [colour.md](../research/docs/colour.md),
  steps 1 and 6), only `--dit_model`, `--model_dir`, the output and our wrapper changing. Each run
  dumps its latents and its decode ([`colour_dump.py`](../research/scripts/colour_dump.py)); the
  encoder input and the float32 reference of the colour correction are the float16 run's (the
  sharp 7B's equal the 7B's bit for bit, colour.md step 6).
- **[`gpu/ck_patch.py`](gpu/ck_patch.py)**, a wrapper patching numz's modules at import (numz's
  checkout unchanged): `--dit_model` takes the files of `--model_dir` (a directory of links to
  our 14 files and numz's VAE); each layer marked by a `comfy_quant` tensor becomes a CKLinear
  holding the file's tensors, numz's own loader loading the rest; CKLinear keeps them as buffers
  (moved as any buffer, never cast), computes with autocast off in the dtype numz's autocast
  would give (bfloat16). Modes: fp8 `w8a16` (dequantized, a 16-bit multiply) or `w8a8` (the input
  quantized per tensor at run time, torch's scaled_mm); `int8` (comfy-kitchen rotates the input
  and quantizes it per token); NVFP4 `w4a4` (the input in NVFP4, cuBLASLt's FP4 multiply) or
  `w4a16`. In `w8a8`, `int8` and `w4a4`, any dequantization of a comfy-kitchen tensor during the
  multiply aborts the run: comfy-kitchen falls back to dequantizing when its kernel fails, some
  of its handlers without a word. In `w8a8` and `w4a4`, an input of more than 2^31 − 1 elements
  is quantized in row chunks, at the scale comfy-kitchen gives the whole tensor: its quantizers
  index in 32 bits ([below](#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices)). GGUF files go
  through numz's own GGUF loader, untouched.
- **First on the CPU** ([`gpu/ck_patch_test.py`](gpu/ck_patch_test.py)): the wrapper chain and its
  guards through numz's CLI (each refusal exits with status 3), each format loaded into numz's 7B
  (288 CKLinear holding the file's tensors bit for bit, every other parameter and buffer the
  float16 model's), each mode's multiply on a few layers against float32.

### The smoke test

anime-clean (below), the 7B's files in every mode, with `CK_PATCH_CHECK=1`: at each layer's first
forward, on up to 16,384 rows of its real input, in float32, y the layer's output as run,
y_deq = x W_deq^T + b (the file's tensors decoded by our own code, not comfy-kitchen's) and
y_16 = x W16^T + b (the float16 file's tensor):

- **kernel** ||y - y_deq|| / ||y_deq||: the kernel and the rounding of the activations;
- **total** ||y - y_16|| / ||y_16||: the whole format against float16, on real activations;
- **weights** ||y_deq - y_16|| / ||y_16||: the weights alone;
- **floor**: the float16 file's layer computed as numz computes it (bfloat16) against y_16.

And the proof of the path: the aten operations comfy-kitchen's tensor met, the backend its
registry chose for each of its functions, torch's multiplies called, the dequantizations. At
load, the 840 tensors kept in 16 bits must equal the float16 file's. Then:

- **the check mode changes nothing:** tier 1's run of the fp8 file in `w8a16` on the same clip,
  without it, gives the same latents and decode, bit for bit (so the smoke's decodes served as
  tier 1's on that clip);
- **BlockSwap:** the int8 file with 18 of the 36 blocks swapped to the CPU (`--blocks_to_swap 18
  --dit_offload_device cpu`) gives the same latents and decode as without, bit for bit.

### Tier 1: 1080p

- **8 clips** of 45 frames, ×2 from d1 (the 1080p ground truth at half size, Mitchell, x264 CRF
  20) to 1920×1080, by kind: anime (anime-clean, anime-grain, anime-sky, anime-bright), cartoon
  (cartoon-bright), live action (live-vfx, live-slow), dark (anime-dark). Every file of both
  models, in every mode; numz's own Q4_K_M on the 4 clips models.md had not run it on.
- **Scored as colour's step 6:** each decode with no colour correction (`none`) and with
  seedvr2x's default `split:ycc:4:3`, against the ground truth
  ([`colour_eval.py`](../research/scripts/colour_eval.py)): PSNR-Y, SSIM-Y, VMAF (v1) and CAMBI,
  LPIPS, DISTS (every 9th frame; every frame by [`fr_metrics.py`](../research/scripts/fr_metrics.py)),
  the low frequencies' ΔE00, the temporal error (T-err, and T-err of the low frequencies), the
  luma Laplacian variance over the GT's (detail), the finest band's energy over the GT's
  ([`colour_bands.py`](../research/scripts/colour_bands.py)); and the distance in dB to the float16
  output at the same seed: RGB PSNR between their 16-bit masters
  ([`ffv1_out.py`](../research/scripts/ffv1_out.py) `--diff`).
- **The reference:** each file paired frame by frame with its own model's float16 at seed 42 (the
  7B's: colour's runs; the sharp's: colour's `sh42`). **The band:** that float16 model's spread
  over 3 seeds (42, 43, 1234: max − min of the per-seed means), per clip, variant and score; the
  7B's seeds 43 and 1234 are colour's runs, the sharp's ours. The sharp's spread is wider on most
  scores (median over the clips: LPIPS 1.78 times the 7B's, DISTS 1.3 to 1.5, VMAF 1.25, PSNR-Y
  1.14).
- **The pipeline checked:** colour's sharp decodes scored through it give colour's own JSONs,
  series for series (56 of 56 comparisons), and colour's table of the sharp against the 7B, cell
  for cell (240 of 240); at 4K, 28 of 28 for the 7B's shots and 22 of 22 for the sharp's.

### Slice B: 4K

- **The 7B:** 6 shots of 45 frames (41 for one), ×2 from d1 to 4K: Sol Levante (painted anime:
  painted, line, action, dark) and digital live action (sunrise, space); VAE decode in tiles of
  2048 overlapping 64; scored as colour's step 5 (whole frames, VMAF's 4K model); the band: the
  7B's two float16 seeds there (42 and 43, colour's step 5).
- **The sharp 7B:** its 5 shots with two float16 seeds (colour's step 6: digital-cockpit,
  ouatia-face, digital-space, sollevante-painted, cel4k-detail), each of its files the GPU's slot
  reached: int8, the dynamic GGUF, Q4_K, fp8 W8A8 (run twice: the first runs came out NaN,
  [below](#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices)), Q8_0. Not run at 4K (the slot ended
  first): fp8 W8A16, NVFP4 (W4A4 and W4A16) and the Q4_K with importance.

### The guards and two rules

- **Guards**, each on `none` and `split`, per kind of source: banding (CAMBI added), flicker
  (T-err, T-err lf), colour drift (ΔE00 lf), detail (Laplacian over the GT's, either way), band
  energy (the finest band over the GT's, either way), LPIPS, DISTS (every frame where scored).
  PSNR-Y, SSIM-Y and VMAF are reported, not guarded: the model redraws, and closer to the source
  can mean redrawing less ([numerics.md](../research/docs/numerics.md#the-model-re-renders)).
- **Strict rule:** a cell (a score on a clip and variant) is worse when colour_eval's 95%
  moving-block bootstrap interval (blocks of 8 frames, 2,000 draws) excludes 0 and the difference
  exceeds the band; detail, band energy and DISTS on 5 frames have no interval: the band alone
  decides. A guard fails on a kind of source when one of its cells is worse.
- **The strict rule fails the float16 model itself:** in a Monte Carlo of a further float16 seed
  (the seeds' scatter taken as Gaussian, both directions), it fails such a seed on 29% of the
  cells with a 3-seed band (on 94% of 8-clip runs at least once), on 50% with a 2-seed band (98%
  of 6-shot runs).
- **Calibrated rule:** strict, and the difference beyond K times the band, K the multiple a
  further float16 seed exceeds with 5% probability per cell: 2.7 with 3 seeds, 10.9 with 2 (4K).
  A further seed would still go past it on about one cell in twenty: a file is read by its count
  of cells past the line, its worst multiple, its distance in dB and the pattern, not by any one
  cell.

### The dynamic GGUF

FORMATS.md's [Dynamic GGUF](FORMATS.md#dynamic-gguf-a-type-per-matrix-chosen-with-an-importance-matrix)
says how its types are chosen. Here, the importance:

- **[`gpu/imatrix_hook.py`](gpu/imatrix_hook.py)** in numz's runs of the model's own float16 file
  (the 7B's for the 7B, the sharp's for the sharp): on each of the 288 block matrices, the input's
  sum of squares per channel and its token count, at every DiT forward; the importance of a
  channel is the mean of its square, llama.cpp's imatrix.
- **Calibration:** 4 runs of 45 frames at 1080p, seed 42, the d1 inputs at a quarter of the size
  of four 4K shots outside tier 1 and the 7B's 4K slice
  ([`colour_clips.py`](../research/scripts/colour_clips.py) `--factor 4`): cel4k-detail (anime),
  cel4k-flat (flat cel), ouatia-street (live action), ouatia-dark (dark). One DiT forward each:
  403,104 video tokens and 232 text tokens per matrix in all, merged into one file per model
  (12,459,016 and 12,459,160 bytes).
- **The hooks change nothing:** a control run with them on anime-clean gives the unhooked run's
  decode, SHA-256 for SHA-256 (the 7B's `1eda5ac6…`, the sharp's `5273f3f1…`).
- **The files:** the dynamic one and its control, every matrix Q4_K with the same importance,
  built from the fp32 masters (`seedvr2_gguf_dyn.py`), read by numz's own loader
  ([`numz_gguf_check.py`](numz_gguf_check.py): 288 GGUF linears, one layer's forward equal to
  F.linear's), run as our static GGUF files are. The runs read a first build, whose metadata held
  the importance file's path; the pinned files, built again without it (twice, the same bytes),
  hold the same tensors, byte for byte.

### The eyes

[`colour_crops.py`](../research/scripts/colour_crops.py) on colour's windows: at 1080p, 10 windows
of 384×216 at 1:1 on 5 clips, each strip GT, bicubic, the float16 model, then the files, all with
`split:ycc:4:3`; at 4K, colour's windows on the sharp's 5 shots. The user's verdict is recorded
as given.

## Results

### Per layer, on real activations (the smoke test, the 7B)

anime-clean, the 288 block matrices' first forward; median (worst) in percent of the layer's
output:

| Mode | Kernel | Total, against float16 | The weights alone | Floor: float16 in bfloat16 | Decode, dB to float16 |
|---|---|---|---|---|---|
| fp8, W8A16 | 0.22 (0.38) | 1.50 (2.92) | 1.48 (2.90) | 0.22 (0.38) | 47.6 |
| fp8, W8A8 | 1.48 (2.90) | 2.09 (4.10) | 1.48 (2.90) | 0.22 (0.38) | 45.1 |
| int8, rotated | 0.59 (1.32) | 0.76 (1.63) | 0.47 (1.07) | 0.22 (0.38) | 50.6 |
| NVFP4, W4A16 | 0.22 (0.38) | 4.93 (9.61) | 4.92 (9.61) | 0.22 (0.38) | 39.3 |
| NVFP4, W4A4 | 5.16 (10.3) | 7.18 (14.0) | 4.94 (9.59) | 0.22 (0.38) | 36.1 |
| GGUF Q4_K (numz's loader: no check mode) | | | | | 40.8 |
| float16, seeds 43 and 1234 | | | | | 37.3, 37.8 |

| Mode | comfy-kitchen's functions (backend) | The multiply | Dequantizations |
|---|---|---|---|
| fp8, W8A16 | dequantize_per_tensor_fp8 (cuda) | torch's addmm (216 layers with a bias), mm (72 without) | one per layer, as meant |
| fp8, W8A8 | quantize_per_tensor_fp8 (cuda) | torch's scaled_mm, fp32 accumulation, the bias in it | none |
| int8, rotated | int8_linear (cuda) | comfy-kitchen's own, the input rotated and quantized per token | none |
| NVFP4, W4A4 | quantize_nvfp4, scaled_mm_nvfp4 (cuda) | cuBLASLt's FP4 multiply, the bias added after | none |
| NVFP4, W4A16 | dequantize_nvfp4 (cuda) | a 16-bit multiply | one per layer, as meant |

- In W8A16 and W4A16 the kernel adds nothing to numz's own float16 computation (0.22%, the
  floor): the format's error is its weights'.
- On real activations a layer's output moves less than its weights do (fp8's weights 1.48% here,
  2.65% per weight in FORMATS.md; int8's 0.47% against 0.86%; NVFP4's 4.94% against 8.80%).
  Rounding the activations adds 0.6 points in W8A8 and 2.2 in W4A4; int8's rotation keeps its
  activation rounding small (total 0.76%).
- The worst layer against float16 (total) is `blocks.22.attn.proj_out.vid` in every mode.
- Every mode ran on all 288 layers through the path in the table, the 840 16-bit tensors equal
  to the float16 file's, numz's float16 VAE and `vid_in` untouched.

### 1080p, the 7B

Each file against the 7B's float16 at seed 42, the band its 3 seeds' spread. Cells: a score on a
clip and variant (8 clips × 2 variants). PSNR-Y and VMAF: the mean difference against the ground
truth with `split:ycc:4:3`, the file's minus the float16's (+: closer to the source).

| File, mode | Cells past the strict / calibrated line | Guards past the calibrated line (kinds) | Worst multiple of the band | dB to float16, seed 42: mean (worst) | PSNR-Y, VMAF |
|---|---|---|---|---|---|
| fp8, W8A16 | 0 / 0 | none | – | 50.56 (45.50) | −0.01, −0.08 |
| fp8, W8A8 | 6 / 0 | none | 2.36× | 47.97 (42.69) | −0.03, −0.15 |
| int8, rotated | 4 / 1 | LPIPS (live action: +0.0002 on live-slow) | 2.85× | 53.79 (49.22) | −0.01, −0.14 |
| GGUF Q8_0 | 0 / 0 | none | – | 56.28 (51.64) | −0.04, −0.17 |
| GGUF Q4_K | 11 / 0 | none (strict: detail and finest band lower on anime and live action; DISTS on cartoon) | 2.68× | 42.87 (37.68) | +0.30, +0.47 |
| GGUF Q4_K, importance | 11 / 6 | detail ↓, finest band ↓ (live action); DISTS (cartoon) | 2.90× | 43.74 (37.89) | +0.10, +0.32 |
| GGUF dynamic | 10 / 2 | DISTS (cartoon) | 3.02× | 44.15 (38.54) | +0.08, +0.22 |
| NVFP4, W4A16 | 5 / 4 | detail ↓, finest band ↓ (live action) | 4.10× | 42.09 (36.16) | +0.13, +0.24 |
| NVFP4, W4A4 | 12 / 8 | detail ↓, finest band ↓ (live action); DISTS (anime, cartoon) | 11.8× | 38.89 (33.02) | +0.16, +0.24 |
| numz's Q4_K_M (4 clips) | 7 / 4 | detail ↓, finest band ↓ (live action) | 2.91× | 43.15 (37.73) | +0.36, +0.24 |
| float16, seeds 43 and 1234 | | | | 40.62 (36.48) | |

- **8-bit:** Q8_0 and fp8 W8A16 have no cell past even the strict line. fp8 W8A8's strict cells
  (flicker and colour drift on anime-clean, LPIPS and DISTS on the cartoon clip) stay under 2.4
  times the band; int8's are LPIPS and DISTS on the cartoon clip and live-slow, +0.0001 to
  +0.0003, one of them 2.85 times the band (live-slow's LPIPS band is 0.0001): a further seed
  would cross that line on one cell in twenty.
- **Q4_K** lowers fine detail: anime-sky's Laplacian variance 5.08 times the GT's against the
  float16's 6.04 (its seeds 5.42 to 6.12), live-slow's 1.15 against 1.22; and it lands closer to
  the source on most full-reference scores (without correction: PSNR-Y +0.41 to +0.69 dB beyond
  the band on 4 clips, ΔE00 lf lower on all 8, flicker lower on 6). numz's Q4_K_M does the same
  (paired with ours: within the band on nearly every score, 48 to 57 dB apart), a little further
  on live-slow (2.91 times).
- **NVFP4 W4A4** rounds every activation to 4 bits: DISTS +0.0086 on the cartoon clip (11.8
  times the band), +0.0080 on anime-bright (3.6 times); further from float16 than a seed on 6 of
  8 clips.

### 1080p, the sharp 7B

Each file against the sharp's float16 at seed 42, the band the sharp's own 3 seeds' spread.

| File, mode | Cells past the strict / calibrated line | Guards past the calibrated line (kinds) | Worst multiple of the band | dB to float16, seed 42: mean (worst) | PSNR-Y, VMAF |
|---|---|---|---|---|---|
| fp8, W8A16 | 4 / 0 | none | 2.06× | 48.52 (43.28) | −0.04, −0.15 |
| fp8, W8A8 | 5 / 0 | none | 1.53× | 45.96 (40.41) | −0.08, −0.26 |
| int8, rotated | 2 / 0 | none | 1.13× | 51.94 (46.59) | +0.02, +0.07 |
| GGUF Q8_0 | 0 / 0 | none | – | 54.43 (49.04) | +0.00, −0.04 |
| GGUF Q4_K | 28 / 9 | flicker, detail ↑, finest band ↑ (live action); colour drift (anime); LPIPS, DISTS (cartoon) | 6.51× | 41.43 (35.30) | −0.20, −0.33 |
| GGUF Q4_K, importance | 20 / 4 | finest band ↓ (live action); DISTS (cartoon) | 4.20× | 42.05 (35.82) | +0.02, +0.01 |
| GGUF dynamic | 23 / 6 | flicker, colour drift (anime); finest band ↓ (live action); LPIPS, DISTS (cartoon) | 4.37× | 42.48 (36.36) | −0.07, −0.13 |
| NVFP4, W4A16 | 13 / 4 | flicker (anime); finest band ↓ (live action); DISTS (cartoon) | 5.23× | 40.36 (34.21) | +0.20, +0.42 |
| NVFP4, W4A4 | 18 / 7 | flicker (anime); colour drift (live action); LPIPS (cartoon); DISTS (anime, cartoon) | 12.9× | 37.28 (30.99) | +0.23, +0.05 |
| float16, seeds 43 and 1234 | | | | 39.15 (34.59) | |

- **8-bit:** as the 7B's, closer than a seed: no cell past the calibrated line, the strict ones
  under 2.1 times the band.
- **4-bit:** past the line on the cartoon clip's DISTS in every file (+0.0032 to +0.0169), on
  live-slow's finest band in four (the static Q4_K adds texture there, 6.5 times the band; the
  others remove it), on an anime clip's flicker in three. The sharp's band is the wider of the
  two, yet its 4-bit files cross it more often (30 cells in all) than the 7B's cross the 7B's
  (20): the sharp 7B looks the more sensitive to 4-bit weights.

### Distances to float16

RGB PSNR between the 16-bit masters of a file's output and its float16 model's at seed 42, per
clip, the 7B's 4-bit files, and its float16's own seeds for scale:

| Clip | Q4_K | Q4_K, importance | dynamic | NVFP4 W4A16 | NVFP4 W4A4 | float16, seeds 43 / 1234 |
|---|---|---|---|---|---|---|
| anime-clean | 40.80 | 41.31 | 41.78 | 39.32 | 36.08 | 37.34 / 37.75 |
| anime-grain | 43.11 | 43.95 | 43.98 | 42.52 | 39.24 | 40.56 / 40.57 |
| anime-sky | 46.09 | 48.71 | 49.38 | 46.66 | 44.35 | 42.88 / 42.78 |
| anime-bright | 37.68 | 37.89 | 38.54 | 36.16 | 33.02 | 36.60 / 36.48 |
| cartoon-bright | 40.23 | 40.48 | 41.18 | 38.75 | 35.00 | 39.91 / 39.99 |
| live-vfx | 45.93 | 46.85 | 47.12 | 45.53 | 42.38 | 41.60 / 41.88 |
| live-slow | 42.38 | 43.04 | 43.67 | 41.12 | 37.64 | 41.40 / 41.43 |
| anime-dark | 46.72 | 47.67 | 47.52 | 46.69 | 43.43 | 44.43 / 44.38 |

- Every 4-bit file but NVFP4 W4A4 is closer to float16 than a seed on most clips; W4A4 is
  further on 6 of 8.
- A distance is not a direction: Q4_K departs from float16 toward a smoother image, closer to
  the source; the dynamic file departs less, and not that way.

### The dynamic GGUF

| | the 7B | the sharp 7B |
|---|---|---|
| Types of the 288 matrices | Q3_K 17, Q4_K 203, Q5_K 68 | Q3_K 17, Q4_K 204, Q5_K 67 |
| Size against our Q4_K's | −1,252,288 bytes | −1,178,560 bytes |
| dB to float16: dynamic / control / Q4_K | 44.15 / 43.74 / 42.87 | 42.48 / 42.05 / 41.43 |
| Cells past the calibrated line: dynamic / control / Q4_K | 2 / 6 / 0 | 6 / 4 / 9 |
| At 4K, 5 shots: dB to float16, dynamic / Q4_K; cells past the calibrated line | not run | 43.31 / 41.96; 3 / 4 |

- **Where the bytes went** (the 7B; the sharp's choice differs on 5 matrices,
  [FORMATS.md](FORMATS.md#dynamic-gguf-a-type-per-matrix-chosen-with-an-importance-matrix)):
  Q5_K for every video attention output projection and 32 of the 36 text ones; Q3_K for the 3
  matrices of block 35 whose output nothing reads, the text MLP's output projection of blocks 0
  to 10 (but 3) and 34, block 3's text MLP input and block 0's video MLP.
- **The 7B:** the dynamic file is closer to float16 than Q4_K on all 8 clips (+0.8 to +3.3 dB),
  keeps anime-sky's fine detail (5.88 times the GT's against the float16's 6.04 and Q4_K's 5.08),
  still lowers anime-grain's (2.63 against 3.01) and live-slow's (1.16 against 1.22); against
  Q4_K it is further from the source on several full-reference scores (LPIPS on 6 of 16 cells,
  flicker on 7): Q4_K's smoothing moves it toward the source. Against its control it is within
  the band on nearly every score (DISTS on all 16 cells, LPIPS on 14): the importance does most of
  the work, the mix of types adds 0.4 dB.
- **The sharp:** the three files fail differently (above); none is past the line on fewer than
  two kinds of source. At 4K the dynamic file is closer to float16 than Q4_K on every shot (0.8
  to 1.8 dB) and takes a little of digital live action's finest texture away where Q4_K adds
  some.

### 4K, the 7B

| File, mode | Shots | Cells past the strict / calibrated line | Worst multiple of the band | dB to float16, seed 42: mean (worst) | PSNR-Y, VMAF |
|---|---|---|---|---|---|
| fp8, W8A16 | 6 | 19 / 0 | 4.62× | 50.44 (44.41) | +0.02, +0.00 |
| GGUF Q8_0 | 6 | 7 / 0 | 5.10× | 56.13 (50.57) | −0.02, −0.08 |
| float16, seed 43 | | | | 39.53 (34.66) | |

With 2 seeds the strict line is crossed by a further seed on half the cells: the strict cells
here (flicker, colour drift, detail, finest band, LPIPS, DISTS) are within what a seed would
show; none comes near the calibrated line (5.1 times the band at worst, the line at 10.9).

### 4K, the sharp 7B

The sharp's files on its 5 shots, against its own two seeds (39.22 dB apart, 34.18 to 42.25 per
shot); kinds: digital live action (digital-cockpit, digital-space), the first film's close-up
(ouatia-face), Sol Levante (sollevante-painted), cel (cel4k-detail).

| File, mode | Shots | Cells past the strict / calibrated line | Guards past the calibrated line (kinds) | Worst multiple of the band | dB to float16, seed 42: mean (worst) | PSNR-Y, VMAF |
|---|---|---|---|---|---|---|
| int8, rotated | 5 | 14 / 0 | none | 6.92× | 52.77 (46.08) | +0.05, +0.18 |
| GGUF dynamic | 5 | 18 / 3 | finest band ↓ (digital); colour drift (first film, a near-zero spread) | 29.3× | 43.31 (37.02) | +0.14, +0.38 |
| GGUF Q4_K | 5 | 18 / 4 | detail ↑, finest band ↑ (digital) | 62.6× | 41.96 (35.95) | +0.08, +0.27 |
| fp8, W8A8 | 5 | 12 / 3 | finest band ↑ (digital); colour drift (first film, a near-zero spread) | 87.5× | 46.83 (40.25) | −0.02, −0.13 |
| GGUF Q8_0 | 5 | 12 / 0 | none | 10.2× | 55.47 (48.81) | −0.01, −0.05 |
| float16, seed 43 | | | | | 39.22 (34.18) | |

Not run at 4K: fp8 W8A16, NVFP4 (W4A4 and W4A16) and the Q4_K with importance.

- **Every file is closer to float16 than the sharp's other seed, on every shot:** Q8_0 48.8 to
  58.6 dB, int8 46.1 to 55.8, fp8 W8A8 40.3 to 49.4, the dynamic GGUF 37.0 to 46.0, Q4_K 36.0 to
  45.1, the seed 34.2 to 42.3; the dynamic GGUF 0.8 to 1.8 dB closer than Q4_K on each shot.
- **The calibrated line is crossed only where the sharp's two seeds agree almost exactly:**
  digital-cockpit's detail and finest band (its seeds at 3.22 and 3.23 times the GT's, and both
  at 1.78: Q8_0 lands at 3.2 and 10.2 times these spreads, int8 at 1.5 and 4.7) and ouatia-face's
  low-frequency ΔE00 with the correction (a spread near 0.00001: fp8 W8A8's +0.001 is 87.5 times
  it). The direction still shows: on digital-cockpit Q4_K adds fine texture (detail 3.40 times
  the GT's against 3.23, the finest band 1.82 against 1.78: 18 to 63 times the spreads), the
  dynamic GGUF removes some (the finest band 1.76 to 1.77, 20 times), as both do on live action
  at 1080p.
- **Fidelity:** the 4-bit files land a little closer to the source (PSNR-Y +0.08 and +0.14,
  LPIPS −0.004 and −0.006 with the correction), the 8-bit ones within 0.05 dB of float16's.

### fp8 W8A8 at 4K: comfy-kitchen's 32-bit indices

The sharp's fp8 file multiplied in 8 bits first came out NaN at 4K: on digital-cockpit and
ouatia-face every value of every frame (their 16-bit renders 12.2 dB from float16), on
digital-space too (numz's warning of a NaN cast in all three runs), where it ran fine at 1080p
(46.0 dB).

- **The cause:** comfy-kitchen 0.2.37's CUDA quantizers of the activations index in 32 bits. The
  fp8 one takes the element count as a 32-bit integer and leaves its output past 2^32 elements
  unwritten: whatever memory it was given, NaN codes among it, and one NaN makes the next layer's
  per-tensor scale NaN, then the whole video. The NVFP4 one quantizes, from 2^32 elements on, the
  values found 2^32 elements earlier: finite, wrong, without a word.
- **When:** a linear's input of 2^32 elements or more. In the 7B only the MLP's output projection
  gets there (12,288 values per token), from 349,526 video tokens in one forward: at 3840×2160 a
  batch of 41 frames or more (45 frames: 388,800 tokens), so all five of the sharp's 4K shots
  (362,880 to 388,800 tokens); at 1080p from 169 frames (45 frames: 97,920 tokens, so tier 1 is
  unaffected). Only W8A8 and W4A4: int8 quantizes its input in its own kernel with 64-bit
  offsets, and W8A16 and W4A16 quantize no input.
- **The fix, in ck_patch.py:** above 2^31 − 1 elements, the input goes to comfy-kitchen in chunks
  of a multiple of 128 rows, each quantized by comfy-kitchen's own quantizer at the scale its
  formula gives the whole tensor: its values bit for bit wherever its kernel is right (on the
  GPU, chunks forced on a smaller input equal its unchunked output; a 1080p layer's forward is
  unchanged, bit for bit; on a 4K input the rows past 2^32 elements are as accurate as the
  others). The five shots ran again with it, each run 36 chunked inputs (the 36 blocks' video
  MLP, 3 chunks each), no NaN: the table above holds those runs. NVFP4 W4A4 never ran at 4K
  here.
- **Beyond this repository:** any runtime that quantizes a layer's input through comfy-kitchen
  0.2.37's CUDA kernels meets it (comfy-kitchen is ComfyUI's kernel library): with our fp8 and
  NVFP4 files in W8A8 or W4A4, from a 3840×2160 batch of 41 frames. An upstream report is
  pending.

### The eyes

- **The 7B at 1080p** (10 windows on anime-clean, anime-bright, cartoon-bright, live-vfx,
  live-slow: GT, bicubic, float16, fp8 W8A16, fp8 W8A8, int8, Q8_0, Q4_K, numz's Q4_K_M, NVFP4
  W4A4): nothing to report on any window, NVFP4 W4A4 and Q4_K included (the user, 2026-10-07).
- **The 7B's 4 GB files at 1080p** (the same windows: float16, Q4_K, Q4_K with importance, the
  dynamic file, NVFP4 W4A16 and W4A4, numz's Q4_K_M): not given when this was written.
- **The sharp 7B at 1080p** (the same windows: its float16 and its 9 file-modes): not given when
  this was written.
- **The sharp 7B at 4K** (colour's 15 windows on its 5 shots, 512×288 at 1:1: GT, bicubic,
  float16, int8, the dynamic GGUF, Q4_K, fp8 W8A8, Q8_0): not given when this was written.

## Caveats

- **One seed per file** (42), judged against its float16 model's spread over 3 seeds: an effect
  that changed sign with the seed would not show. The patterns hold across clips (Q4_K softer on
  3, NVFP4 W4A4's DISTS on 2 kinds), which argues against chance.
- **4K bands from 2 seeds:** |s42 − s43| is a noisy spread, the calibrated line sits at 10.9 times
  it, and the sharp's is much wider than the 7B's (PSNR-Y 2.3 times, VMAF 5, LPIPS 6.7), but
  where its two seeds happen to agree almost exactly (digital-cockpit's finest band), every file
  lands past the strict line, Q8_0 at 10.2 times the spread: 4K results say less than 1080p's.
- **Short clips, one degradation:** 8 clips of 45 frames, d1 only, ×2 at 1080p; 6 and 5 shots at
  4K.
- **The scores are guards, not the verdict:** they flag a file that departs from its float16
  model, not which output looks better; the eyes judge, and the user wants the image redrawn
  (crisper lines, cleaner colours), not a washed-out copy of the source.
- **No speed:** the GPU ran its SM clock at about 580 MHz all night under a false power-cap
  reading; values are unaffected, times are not. FORMATS.md gives NVIDIA's peak rates.
- **numz's GGUF runtime:** its forward decodes Q4_K's values in float16, rounding each twice
  (FORMATS.md: one rounding's value on 29% of them, up to 0.17% of the tensor's max away). The
  GGUF results are that runtime's; a float32 decode would sit closer to the file.
- **NVFP4 W4A4's bias:** comfy-kitchen 0.2.37 has no NVFP4 addmm (F.linear with a bias would
  dequantize both operands, a 16-bit multiply in disguise), so ck_patch.py multiplies without the
  bias and adds it after, in bfloat16: one rounding more than cuBLASLt's bias epilogue.
- **comfy-kitchen's 32-bit quantizers:** fp8 W8A8's 4K results are those of ck_patch.py's row
  chunks, comfy-kitchen's own values wherever its kernel is right; comfy-kitchen 0.2.37 alone
  gives an all-NaN video there, and NVFP4 W4A4, not run at 4K here, a silently wrong one
  ([above](#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices)).
- **The importance's calibration** is disjoint from tier 1 and from the 7B's 4K shots, but
  cel4k-detail is also one of the sharp's 4K shots (at a quarter size for the importance, half
  size at 4K), and the two cel4k shots come from the film of a tier-1 clip (other scenes).
- **numz's runtime, not seedvr2x's:** these paths are comfy-kitchen's kernels driven by
  ck_patch.py in numz's CLI; seedvr2x's own runtime must reproduce them, and its own kernels need
  their own check.

## Reproduce

numz's checkout as the working directory of its runs, its venv's interpreter; GPU runs one at a
time; scoring in colour.md's environment (`COLOUR_BASELINE`, `MEAS_SCRIPTS`: colour.md's
[Reproduce](../research/docs/colour.md#reproduce)):

```bash
S=research/scripts; M=models; N=/path/to/numz; PY=$N/.venv/bin/python; P=/path/to/pylib; MD=/path/to/modeldir
C=/path/to/clips; R=/path/to/colour-dumps; D=/path/to/dumps; O=/path/to/out; I=/path/to/imatrix
F16=/path/to/seedvr2x_ema_7b_fp16.safetensors     # numz's arguments and the dump paths: absolute (numz's cwd)
# comfy-kitchen for numz's interpreter, outside its venv; the model directory: links to our files and numz's VAE
uv pip install --python $PY --target $P --no-deps comfy-kitchen==0.2.37
ln -s /path/to/phase2/seedvr2x_ema_7b*.safetensors /path/to/phase2/seedvr2x_ema_7b*.gguf \
  /path/to/phase2-dyn/seedvr2x_ema_7b*.gguf /path/to/numz-models/ema_vae_fp16.safetensors $MD/
# the patch on the CPU: wrapper chain and guards, every format loaded into numz's 7B, every mode's multiply
(cd $N && CUDA_VISIBLE_DEVICES= PYTHONPATH=$P $PY /path/to/models/gpu/ck_patch_test.py --out ck_test.json \
  --model-dir $MD --phase2 /path/to/phase2 --fp16 $F16 --colour-dump /path/to/research/scripts/colour_dump.py \
  --clip $C/anime-clean.d1.lr.mkv --scratch /path/to/scratch)
# the smoke test: one run per mode with the check mode (here int8; w8a16, w8a8, w4a4, w4a16 the same)
$PY $S/bench.py run model-smoke-int8 --seedvr2-dir $N --wrap $S/colour_dump.py --wrap $M/gpu/ck_patch.py \
  --env PYTHONPATH=$P --env CK_PATCH_MODE=int8 --env CK_PATCH_EXPECT=288 --env CK_PATCH_CHECK=1 \
  --env CK_PATCH_FP16=$F16 --env CK_PATCH_LOG=$O/model-smoke-int8.ck_patch.json \
  --env COLOUR_DUMP=$D/anime-clean-d1/int8-s42 --env COLOUR_DUMP_INPUTS=0 -- $C/anime-clean.d1.lr.mkv \
  --output $O/smoke-int8/ --model_dir $MD --dit_model seedvr2x_ema_7b_int8_convrot.safetensors --resolution 1080 \
  --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none --debug --seed 42
#   BlockSwap: the same with --blocks_to_swap 18 --dit_offload_device cpu; its dump equal bit for bit
# tier 1 and slice B: the same run per file, mode and clip, without CK_PATCH_CHECK and CK_PATCH_FP16; a GGUF file
# without CK_PATCH_MODE and CK_PATCH_EXPECT (numz's own loader); the sharp's files: seedvr2x_ema_7b_sharp_*;
# 4K: --resolution 2160 (2016 for the digital shots, 2048 for cel4k-detail) --vae_decode_tiled
# --vae_decode_tile_size 2048 --vae_decode_tile_overlap 64
# the sharp's float16 seeds 43 and 1234 (its band): numz's own file, colour_dump.py alone
$PY $S/bench.py run model-a-anime-clean-d1-sharp-s43 --seedvr2-dir $N --wrap $S/colour_dump.py \
  --env COLOUR_DUMP=$D/anime-clean-d1/sharp-s43 --env COLOUR_DUMP_INPUTS=0 -- $C/anime-clean.d1.lr.mkv \
  --output $O/sharp-s43/ --model_dir /path/to/numz-models --dit_model seedvr2_ema_7b_sharp_fp16.safetensors \
  --resolution 1080 --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none \
  --debug --seed 43
# the importance: 4 calibration runs per model, then merged; the dynamic GGUF and its control; numz's loader
python3 $S/colour_clips.py make /path/to/clips4k/cel4k-detail --factor 4 --out $I/clips   # and the other three
$PY $S/bench.py run model-c-imx-cel4k-detail --seedvr2-dir $N --wrap $M/gpu/imatrix_hook.py \
  --env IMATRIX_OUT=$I/cel4k-detail-d1x4.imatrix.safetensors -- $I/clips/cel4k-detail.d1x4.lr.mkv \
  --output $O/imx-cel4k-detail/ --model_dir /path/to/numz-models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --attention_mode flash_attn_2 --batch_size 45 --load_cap 45 --color_correction none \
  --debug --seed 42
#   the control: the same on anime-clean with --wrap $S/colour_dump.py too: its decode's SHA-256 = the unhooked run's
$PY $M/gpu/imatrix_hook.py merge $I/seedvr2_ema_7b_fp16.imatrix.safetensors $I/*-d1x4.imatrix.safetensors
uv run $M/seedvr2_gguf_dyn.py --imatrix $I/seedvr2_ema_7b_fp16.imatrix.safetensors \
  --masters /path/to/bytedance --static /path/to/phase2/seedvr2x_ema_7b_Q4_K.gguf --out /path/to/phase2-dyn \
  --threads 8
#   the sharp's: --model sharp, its own importance, --static /path/to/phase2/seedvr2x_ema_7b_sharp_Q4_K.gguf
(cd $N && $PY /path/to/models/numz_gguf_check.py . /path/to/phase2-dyn/*.gguf)
# scoring, per file and clip (here Q4_K on anime-grain): scores, 16-bit masters, VMAF, distance, bands, DISTS
python3 $S/colour_eval.py score --clip anime-grain-d1 --gt $C/anime-grain.gt.mkv --bars 30:29 \
  --ref q4k=$R/anime-grain-d1/s42/ref_f32.pt --content s42=$D/anime-grain-d1/q4k-s42/decode.pt \
  --variants none,split:ycc:4:3 --out eval/q4k/anime-grain-d1 --lpips --dists-every 9
python3 $S/colour_eval.py render --clip anime-grain-d1 --gt $C/anime-grain.gt.mkv \
  --ref q4k=$R/anime-grain-d1/s42/ref_f32.pt --content s42=$D/anime-grain-d1/q4k-s42/decode.pt \
  --variants none,split:ycc:4:3 --out masters/q4k --pix-fmt gbrp16le
python3 $S/colour_eval.py vmaf --clip anime-grain-d1 --gt $C/anime-grain.gt.mkv --master-content s42 \
  --master 'none@q4k=masters/q4k/anime-grain-d1.s42.none~q4k.gbrp16le.mkv' \
  --master 'split:ycc:4:3@q4k=masters/q4k/anime-grain-d1.s42.split_ycc_4_3~q4k.gbrp16le.mkv' \
  --out vmaf/q4k/anime-grain-d1
python3 $S/ffv1_out.py --diff masters/q4k/anime-grain-d1.s42.none~q4k.gbrp16le.mkv \
  masters/7b/anime-grain-d1.s42.none~f32.gbrp16le.mkv      # the float16's master, rendered the same way
python3 $S/colour_bands.py scan --clip anime-grain-d1 --gt $C/anime-grain.gt.mkv \
  --ref $R/anime-grain-d1/s42/ref_f32.pt --content $D/anime-grain-d1/q4k-s42/decode.pt \
  --bicubic $C/anime-grain.d1.bicubic.mkv --rows 32:1048 --every 3 --out bands/q4k/anime-grain-d1.json
python3 $S/fr_metrics.py $C/anime-grain.gt.mkv --clip anime-grain-d1 --json-dir fr/q4k/anime-grain-d1 --no-vmaf \
  --out none@q4k 42 masters/q4k/anime-grain-d1.s42.none~q4k.gbrp16le.mkv \
  --out split:ycc:4:3@q4k 42 masters/q4k/anime-grain-d1.s42.split_ycc_4_3~q4k.gbrp16le.mkv
# crops: several decodes in one strip, one window per --at
python3 $S/colour_crops.py --clip anime-clean-d1 --gt $C/anime-clean.gt.mkv \
  --bicubic $C/anime-clean.d1.bicubic.mkv --ref $R/anime-clean-d1/s42/ref_f32.pt \
  --content 7B=$R/anime-clean-d1/s42/decode.pt --content int8=$D/anime-clean-d1/int8-s42/decode.pt \
  --content q4k=$D/anime-clean-d1/q4k-s42/decode.pt --variants split:ycc:4:3 --no-stretch \
  --at 400,480 --at 800,864 --at-frame 22 --size 384x216 --out crops/anime-clean
```

The tables pair each file's per-clip means with its float16's at seed 42 and apply the two rules
of Method; the script that made them (`ms_sum.py`, `ms_sum4k.py`, `ms_sum4ksh.py`) is the GPU
box's and not in this repository.
