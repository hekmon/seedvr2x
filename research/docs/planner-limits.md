# Planner limits: the DiT at 4K and in long windows, tiled VAE at 4K, and the margin over free memory

> Status: **measured** for the open question "4K and long windows" of
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure) and the memory planner's margin over the free
> memory the driver reports ([DESIGN.md](../../seedvr2x/DESIGN.md#memory-planner): "≈ 0.6 GiB, to be
> re-derived"). The DiT through [`scripts/dit_probe.py`](../scripts/dit_probe.py) (numz's own
> runner and Phase 2 code, random latents); the tiled VAE through the CLI with
> [`scripts/vae_probe.py`](../scripts/vae_probe.py); the margin by bisection of the emulated card
> size with [`scripts/vram_cap.py`](../scripts/vram_cap.py), every CLI run recorded by
> [`scripts/bench.py`](../scripts/bench.py). numz `4490bd1`, 7B fp16 unless noted, `flash_attn_2`,
> one GPU (RTX PRO 6000 Blackwell 96 GB, torch 2.14.1+cu130, cuDNN 9.24, flash-attn 2.8.3), default
> `cudaMallocAsync` allocator. The GPU is power-capped and its speed drifts within minutes: DiT
> times are normalised by a reference forward measured around every point. seedvr2x's own tiled
> passes were measured later on the same GPU, tile by tile (2026-10-11), through
> [`scripts/tile_peaks.py`](../scripts/tile_peaks.py), on seedvr2x's runtime, not numz's:
> [their section](#seedvr2xs-tiled-passes-every-tiles-peak).

In short (7B fp16, `flash_attn_2`; DiT forwards on random latents through numz's own code, the
VAE and the margin through the CLI):

- **The DiT's per-token model holds at 4K and in long windows:** 26 window lengths from 8,160 to
  636,480 tokens (1080p windows of 1–78 latents, 4K of 1–19) peak within −0.17 … +0.83 GiB of
  16.05 GiB + 128.5 KiB per token. The rest is the attention windows, each of which repeats the 58
  text tokens: **peak = 15.87 GiB + 127.16 KiB × tokens + 2.73 MiB × windows** fits every point
  to 0.005 GiB ([results](#the-dits-memory-tokens-plus-windows)).
- **Time per token is flat** (±5%) from 1- to 8-latent attention windows, from 32,400 to 636,480
  tokens, at 1080p and 4K; the session's clock sets the absolute speed (0.33–0.45 ms per token
  here, 0.23 in vram.md's fastest session).
- **On the full 96 GB card** the longest windows are 78 latents (309 frames) at 1080p, with
  0.56 GiB left, and 19 (73 frames) at 4K; one latent more fails within 3 s, in the DiT's first
  block.
- **Tiled VAE at 4K** follows vram.md's tiled fits up to 1536 tiles (±0.45 GiB) and exceeds them
  at 2048 (decode 68.8 GiB, +1.7); a 4K frame takes the same time with 1024, 1536 or 2048 tiles
  (5.3–5.6 s to encode, 11.5–12.3 s to decode here).
- **The margin, bisected on emulated cards:** a run needs 0.15–0.45 GiB between the free memory it
  starts with and its torch peak. Decode-bound (8 GB) fails at +0.37 and passes at +0.45,
  encode-bound at +0.14 / +0.24, DiT-bound at +0.15 / +0.23 with BlockSwap and +0.21 / +0.26
  without. Below that the run fails in its bounding phase, after dozens of silent allocator
  retries; numz's VAE retry doesn't save it, the DiT doesn't retry.
- **seedvr2x's own tiled passes, tile by tile:** the 25 square tiles from 512 to 2048 px and 20
  tiles cut off at 1080p's and 720p's rows, an encode and a decode each. The tiled fits are no
  upper bound there: an encode stands up to 0.797 GiB over its fit (832 px) and a decode 0.671
  (576 px), so seedvr2x's planner takes each tile's measured peak
  ([results](#seedvr2xs-tiled-passes-every-tiles-peak)).
- **For seedvr2x:** budget = free memory at start − 0.6 GiB (on these cards, the validated
  "N − 2 GiB" again), DiT peaks by tokens and windows, the tiled VAE fits refitted above 1536
  tiles and, on seedvr2x's own passes, each tile's measured peak in their place, DiT time per
  token calibrated on the machine ([What the planner should use](#what-the-planner-should-use)).

## Why it matters for seedvr2x

The planner of [DESIGN.md](../../seedvr2x/DESIGN.md#memory-planner) budgets the free memory the
driver reports once the CUDA context exists (`mem_get_info`) minus a margin, and fills it with
per-phase peaks predicted by [vram.md](vram.md#practical-rules)'s formulas, the DiT's as
16.05 GiB + 128.5 KiB per token. Three things were open:

- **The DiT where the planner will push it,** windows as long as the memory allows, at 1080p and
  at 4K: the per-token model had been measured up to 171,360 tokens (1080p, 21 latents), and at 4K
  up to 7 latents (226,800 tokens, 44.44 GiB in the 48 GB recipe). From the code it should hold:
  the attention windows have a fixed size, at most 15 × 27 tokens in space and ⌈min(L, 30) / 4⌉
  latents in time ([below](#the-dits-attention-windows)).
- **Tiled VAE at 4K:** the tiled fits came from 512–1536 tiles at batch 9; the planner will pick
  larger tiles on large cards.
- **The margin:** every emulated run of vram.md passed, the tightest with 0.41 GiB between the
  free memory the CLI saw and its torch peak (0.25 GiB with `expandable_segments`). The failure
  edge, and so the margin, was never measured.

## The DiT's attention windows

From the code (`src/models/dit_7b/`): latents [L, H/8, W/8, 16] get the condition's 17
channels and are patched 1 × 2 × 2 (`patch.py:71-85`): **tokens = L × H/16 × W/16** on the
padded output (8,160 per latent frame at 1080p, padded to 1088 × 1920; 32,400 at 4K). The 36
blocks alternate two window layouts (`configs_7b/main.yaml`: `720pwin_by_size_bysize` and its
shifted variant, window (4, 3, 3)), computed in `window.py:28-83`: the frame is rescaled to
45 × 80 patches and cut 3 × 3, so a window holds **at most 15 × 27 tokens per latent frame at any
resolution**, and ⌈min(L, 30) / 4⌉ latent frames (1 up to L = 4, 8 from L = 29). The shifted
layout has more, smaller windows.

Each window attends to its video tokens **plus the 58 text tokens** of `pos_emb.pt`:
`nablocks/mmsr_block.py:117-134` builds q, k and v as video and text concatenated per window
(`na.repeat_concat_idx`, `na.py:320`), and the text outputs are averaged back. The text is
therefore repeated once per window, a cost that follows the window count, not the tokens.

| Output | L | Tokens | Temporal window | Windows (regular / shifted) | Largest window (tokens) |
|---|---|---|---|---|---|
| 1080p | 1 / 3 / 6 / 12 / 21 | 8,160 … 171,360 | 1 / 1 / 2 / 3 / 6 | 25/30, 75/90, 75/120, 100/150, 100/120 | 405 … 2,430 |
| 1080p | 30 / 50 / 70 / 78 | 244,800 … 636,480 | 8 | 100/150, 175/210, 225/300, 250/330 | 3,240 |
| 4K | 1 / 2 / 4 / 6 / 8 | 32,400 … 259,200 | 1 / 1 / 1 / 2 / 2 | 81/100, 162/200, 324/400, 243/400, 324/500 | 405 / 810 |
| 4K | 10 / 12 / 14 / 16 / 18 / 19 / 20 | 324,000 … 648,000 | 3 / 3 / 4 / 4 / 5 / 5 / 5 | 324 / 400–500 | 1,215 … 2,025 |

## Method

**DiT.** [`dit_probe.py`](../scripts/dit_probe.py) builds what the CLI builds before its Phase 2:
`setup_generation_context` + `prepare_runner` (CLI defaults, no BlockSwap, no compile, tensors
offloaded to the CPU), `materialize_model` for the VAE (it stays on the GPU through Phase 2, as in
the CLI) and the DiT, the text embeddings, `configure_diffusion` with the CLI's one-step settings.
One forward is Phase 2's body for one batch (`generation_phases.py:654-750`): random latents
[L, H/8, W/8, 16] to the GPU in bf16, `set_seed`, the noise and augmentation noise,
`get_condition(task="sr")`, `runner.inference` under bf16 autocast (the weights are fp16), the
output back to the CPU.

- Per forward: torch peak allocated (reset before), GPU time between CUDA events around
  `runner.inference`, device memory, SM clock and power sampled by NVML every 0.1 s. The token
  count is read on the DiT's patch-in output (it always matched the formula), the windows come from
  numz's own window functions.
- Two forwards per point (the steady time is the second; the first was within 2%), one for edge
  refinements; the allocator's pool is emptied before each point. A sweep stops at its first OOM,
  then bisects the edge.
- **Clock drift:** a reference forward (1080p, L = 6, 48,960 tokens) runs at the start and end of
  every invocation (and after every 3 points). A point's time is divided by the reference time
  interpolated at its moment and multiplied by the median reference: all times read as if run at
  one speed. Within one lock hold the reference drifted by up to 36% (16.0 → 21.8 s in 4 minutes,
  SM clock 696 → 580 MHz at the 600 W cap).
- **4K** (3840 × 2160, no padding): L = 1, 2, 3, 4, 6, 8, …, 18, 20 then 19, in one invocation.
  **1080p:** L = 1, 2, 3, 6, 12 in one invocation, then one invocation per point (the GPU lock is
  shared with other jobs): 21, 30, 40, 50, 60, 70, 80, then 75, 77, 78 and 79 by bisection (one
  forward each, the references still two), merged with `dit_probe.py table --merge`.

**Tiled VAE at 4K.** One CLI run per tile size through [`vae_probe.py`](../scripts/vae_probe.py),
which records every VAE call, tile and temporal slice (torch peak above what was allocated before
it, time): a clean 1080p anime clip (8-bit, as the CLI reads it) upscaled to 3840 × 2160
(`--resolution 2160`, no padding needed), 21 frames in one batch (6 latents: the sliced regime of
vram.md's batch ≥ 9 fits), encode and decode tiled with the same tile size, **1024, 1536 and
2048**, overlap 128; 7B fp16, `flash_attn_2`, `--color_correction none`.

**Margin.** [`vram_cap.py`](../scripts/vram_cap.py) emulates an N GB card on the 96 GB one: after
the CLI's CUDA context exists it allocates a driver-level ballast that leaves N − 0.35 − 0.3 GiB
(hidden memory, other processes), so the free memory the CLI reads at start, its `Initial CUDA
memory` line, is N − 1.44 GiB ([vram.md](vram.md#emulating-a-smaller-gpu)). Four configurations
from vram.md's [validated recipe](vram.md#recipe-per-card-size-validated), each bound by a
different phase, run as one batch (the smallest frame count with the recipe's peaks):

| Config | Card | Model | Batch | Tiles encode / decode | Torch peaks encode / DiT / decode (GiB) | Bound by |
|---|---|---|---|---|---|---|
| `dec8` | 8 GB | Q4_K_M, BlockSwap 36 | 13 | 640 / 512 | 5.71 / 5.06 / **6.11** | decode |
| `enc12` | 12 GB | Q4_K_M, BlockSwap 36 | 29 | 896 / 640 | **9.72** / 9.09 / 8.65 | encode |
| `dit24` | 24 GB | 7B fp16, BlockSwap 16 | 45 | 1344 / 1024 | 15.54 / **21.81** / 18.18 | DiT |
| `dit30` | 30 GB | 7B fp16, no BlockSwap | 45 | 1024 / 1024 | 12.01 / **28.15** / 18.18 | DiT |

1080p output from a 1080p anime segment, overlap 64, `--color_correction none`. N goes down in
steps of 0.15 GiB (0.2 for `enc12`, 0.1 for `dit30`) from the recipe's passing value until a run
fails, then once halfway back. **Margin = free memory at start − the bounding phase's torch
peak** (its unconstrained value, as a planner would predict it).

## Results

### The DiT's memory: tokens plus windows

Torch peak of one forward (GiB, the loaded model included), against vram.md's per-token model
(16.05 GiB + 128.5 KiB × tokens) and the fit below; windows = the shifted layout's count (the
larger); L latents = 4L − 3 frames:

| Output | L (frames) | Tokens | Windows | Peak | Per-token model | Fit | Peak − per-token model |
|---|---|---|---|---|---|---|---|
| 1080p | 1 (1) | 8,160 | 30 | 16.94 | 17.05 | 16.94 | −0.11 |
| 1080p | 2 (5) | 16,320 | 60 | 18.01 | 18.05 | 18.01 | −0.04 |
| 1080p | 3 (9) | 24,480 | 90 | 19.08 | 19.05 | 19.08 | +0.03 |
| 1080p | 6 (21) | 48,960 | 120 | 22.13 | 22.05 | 22.13 | +0.08 |
| 1080p | 12 (45) | 97,920 | 150 | 28.15 | 28.05 | 28.15 | +0.10 |
| 1080p | 21 (81) | 171,360 | 120 | 36.97 | 37.05 | 36.97 | −0.08 |
| 1080p | 30 (117) | 244,800 | 150 | 45.96 | 46.05 | 45.96 | −0.09 |
| 1080p | 40 (157) | 326,400 | 180 | 55.93 | 56.05 | 55.93 | −0.12 |
| 1080p | 50 (197) | 408,000 | 210 | 65.91 | 66.05 | 65.91 | −0.14 |
| 1080p | 60 (237) | 489,600 | 240 | 75.88 | 76.05 | 75.88 | −0.17 |
| 1080p | 70 (277) | 571,200 | 300 | 85.94 | 86.05 | 85.94 | −0.11 |
| 1080p | 75 (297) | 612,000 | 300 | 90.89 | 91.05 | 90.89 | −0.16 |
| 1080p | 77 (305) | 628,320 | 330 | 92.95 | 93.05 | 92.95 | −0.10 |
| 1080p | 78 (309) | 636,480 | 330 | 93.94 | 94.05 | 93.94 | −0.11 |
| 1080p | 79, 80 | 644,640, 652,800 | 330 | OOM | 95.05, 96.05 | 94.93, 95.92 | – |
| 4K | 1 (1) | 32,400 | 100 | 20.07 | 20.02 | 20.07 | +0.05 |
| 4K | 2 (5) | 64,800 | 200 | 24.26 | 23.99 | 24.26 | +0.27 |
| 4K | 3 (9) | 97,200 | 300 | 28.46 | 27.96 | 28.46 | +0.50 |
| 4K | 4 (13) | 129,600 | 400 | 32.65 | 31.93 | 32.65 | +0.72 |
| 4K | 6 (21) | 194,400 | 400 | 40.51 | 39.87 | 40.51 | +0.64 |
| 4K | 8 (29) | 259,200 | 500 | 48.64 | 47.81 | 48.64 | +0.83 |
| 4K | 10 (37) | 324,000 | 400 | 56.23 | 55.76 | 56.23 | +0.47 |
| 4K | 12 (45) | 388,800 | 500 | 64.35 | 63.70 | 64.35 | +0.65 |
| 4K | 14 (53) | 453,600 | 400 | 71.94 | 71.64 | 71.94 | +0.30 |
| 4K | 16 (61) | 518,400 | 500 | 80.07 | 79.58 | 80.07 | +0.49 |
| 4K | 18 (69) | 583,200 | 500 | 87.93 | 87.52 | 87.93 | +0.41 |
| 4K | 19 (73) | 615,600 | 500 | 91.86 | 91.49 | 91.86 | +0.37 |
| 4K | 20 (77) | 648,000 | 500 | OOM | 95.46 | 95.79 | – |

- **The per-token model holds where the planner will push it,** from 8,160 to 636,480 tokens,
  1080p up to 78 latents (309 frames) and 4K up to 19 (73 frames), within −0.17 … +0.83 GiB.
  Where the CLI measured the same shapes ([vram.md](vram.md#weights-and-the-per-token-model):
  1080p batch 1, 5, 9, 21, 45, 81, 4K batch 21), the probe's peaks are identical to the hundredth.
- **Its error is the windows:** at 1080p it is within −0.17 … +0.10 GiB; at 4K, with about three
  times the windows of a 1080p forward of the same token count (500 against 150 around 250,000
  tokens), it underestimates by up to 0.83 GiB.
- **Tokens plus windows fit all 26 passing points to 0.005 GiB:**
  **peak = 15.871 GiB + 127.16 KiB × tokens + 2.726 MiB × windows.** 15.87 GiB is what the loaded
  model holds before the forward (DiT 15.35 + buffers 0.02 + VAE 0.47 + text embeddings and
  schedule); 2.73 MiB per window is 48 KiB per repeated text token. The per-token part is
  0.474 GiB per output Mpx and latent frame (3,906 tokens); the windows add 0.08–0.9 GiB at
  1080p, 0.27–1.33 GiB at 4K. The window count comes from the model's own window functions
  ([table above](#the-dits-attention-windows)); it steps with the temporal window, so it is not
  proportional to the tokens.

### The DiT's time per token

Each point's time per token divided by the reference's (1080p, L = 6, 48,960 tokens), measured
in the same process around it:

| Output | L | Temporal window (latents) / largest window (tokens) | Time per token ÷ reference's |
|---|---|---|---|
| 4K | 1, 2, 3, 4 | 1 / 405 | 1.03, 1.00, 0.99, 1.00 |
| 4K | 6, 8 | 2 / 810 | 0.99, 0.99 |
| 4K | 10, 12 | 3 / 1,215 | 0.97, 0.97 |
| 4K | 14, 16 | 4 / 1,620 | 0.98, 1.00 |
| 4K | 18, 19 | 5 / 2,025 | 1.01, 1.00 |
| 1080p | 21 | 6 / 2,430 | 0.90–0.95 |
| 1080p | 30 | 8 / 3,240 | 1.00–1.07 |
| 1080p | 40, 50, 60, 70 | 8 / 3,240 | 1.02, 1.02, 1.02, 1.03 |
| 1080p | 75, 77, 78 | 8 / 3,240 | 1.03, 1.03, 1.03 (one forward each) |

The 4K points come from one process with a reference every 3 points (interpolated); each 1080p
point from 21 on had its own invocation, the reference just before and just after it (both at
21.74–21.85 s from L = 40 on). L = 21 and 30 started on a still-falling clock (reference 19.2–19.8
→ 21.8 s), hence a range: against the reference interpolated at the point, or the one right
after. The 1080p points under 21 ran in the first, fastest-drifting hold and are left out.

- **Flat within ±5%:** time per token doesn't grow with the temporal window (1 → 8 latents, 405 →
  3,240 tokens per window), the token count (32,400 → 636,480) or the resolution. The attention's
  quadratic part doesn't show (attention is 4–9% of the DiT's time,
  [attention.md](attention.md#where-the-dit-time-goes)); 8-latent windows cost 2–3% more per
  token than the reference's 2-latent ones.
- **The absolute speed is the session's:** the same reference forward took 16.0–21.9 s over these
  sessions (0.33–0.45 ms per token; vram.md's 0.231 ms came from its fastest session), and within
  one lock hold it went from 16.0 to 21.8 s as the card reached its power cap.
- **A full card costs no measurable time:** from 4K L = 16 and 1080p L = 70 the allocator trims
  its pool about once per block (NVML shows the whole 95 GiB in use), and the ratio stays at
  1.00–1.03.

### The full-card edges

On the 96 GB card itself, no ballast: the CLI reads 94.50 GiB free at import, and 94.36 GiB are
left once numz's import-time bf16 check has run.

| Output | Longest window that ran | Its peak | Left of the 94.50 GiB | First to fail | Fit predicts | The failing allocation |
|---|---|---|---|---|---|---|
| 1080p | L = 78 (309 frames) | 93.94 | 0.56 | L = 79 | 94.93 | 90.88 GiB allocated + 3.77 requested, 31 MiB free |
| 4K | L = 19 (73 frames) | 91.86 | 2.64 | L = 20 | 95.79 | 91.44 + 3.82, 31 MiB free |

- **A window that doesn't fit fails within seconds,** in its first block (3 s into the forward,
  on a 3.8 GiB allocation), "Forward pass error", nothing retried. 1080p L = 80 failed the same
  way (91.82 + 3.82).
- **The real card agrees with the emulation:** 1080p L = 78 ran with 0.56 GiB left, where the
  emulated DiT-bound runs pass from 0.23 GiB on ([below](#the-margin-over-free-memory)).
- **The window length is bounded by the card, not the code:** on 96 GB, 309 frames per window at
  1080p, 73 at 4K, as the fit predicts.

### Tiled VAE at 4K

3840 × 2160, 21 frames in one batch, encode and decode tiled alike, overlap 128. Peaks in GiB above
what was allocated before the call (the largest tile), against vram.md's tiled fits for batch ≥ 9
(encode 1.7 + 8.4 × T², decode 1.6 + 15.6 × T², T² = tile area in Mpx):

| Tile | Tiles (rows × columns) | Tiled area / frame | Encode peak (fit) | Decode peak (fit) | Decode caches | Encode / decode phase peak | Encode / decode s per frame |
|---|---|---|---|---|---|---|---|
| 1024 | 15 (3 × 5) | 1.27 | 10.44 (10.51) | 17.52 (17.96) | 8.44 | 12.15 / 18.06 | 5.59 / 11.59 |
| 1536 | 6 (2 × 3) | 1.13 | 21.45 (21.52) | 38.86 (38.41) | 18.99 | 23.17 / 39.40 | 5.37 / 12.27 |
| 2048 | 4 (2 × 2) | 1.09 | 37.67 (36.93) | 68.76 (67.03) | 33.75 | 39.39 / 69.31 | 5.33 / 11.53 |

- **The fits hold up to 1536 tiles** (−0.44 … +0.45 GiB, and vram.md's own 4K batch-9 runs gave
  the same 10.4 / 17.4 and 21.4 / 38.7) **and run short at 2048:** +0.74 GiB encode, +1.73 GiB
  decode. Refitted on 1024–2048 at 4K: **decode ≈ 0.44 + 16.29 × T²** (1536 within 0.01 GiB),
  **encode ≈ 1.36 + 8.66 × T²** (1536 0.33 GiB below). The decode's causal caches stay exactly
  8.05 GiB per tile Mpx; the 1.7 GiB the old fit misses at 2048 is in the activations.
- **The phase peak adds what sits on the GPU before the call:** 0.54 GiB for decode (VAE weights
  and latents), 1.72 GiB for encode, because numz moves the batch's input frames to the GPU first
  (21 frames × 0.046 GiB per 4K frame in bf16, plus the weights). A 101-frame 4K batch's frames
  alone would take 4.7 GiB.
- **Time per frame barely depends on the tile size:** 5.3–5.6 s to encode and 11.5–12.3 s to
  decode a 4K frame, not in the order of the tiled area (1.27 → 1.09). Large tiles take more time
  per tiled pixel than 1024 ones, +8–11% to encode and +16–19% to decode (decode 1.10 s per tiled
  Mpx with 1024 tiles, 1.31 and 1.28 with 1536 and 2048), which cancels their smaller overlap.
  Same session and clock for the three runs: their DiT forwards (4K, L = 6) took 84.36–84.40 s.
- The grid starts at 0 with a stride of tile − overlap: 2048 tiles leave a 240-pixel bottom row
  (2 × 2 tiles, two of 240 × 2048 and 240 × 1920), 1024 tiles a 368-pixel row and a 256-pixel
  column. The sliver tiles cost little to decode (2.3–12.3 s against 22–110 s for a full tile).

### The margin over free memory

The emulated card size where each configuration stops completing (free memory at start in GiB;
margin against the bounding phase's unconstrained peak):

| Config (bound, peak) | Passed | Last pass: card → free, margin | First fail: card → free, margin | Edge |
|---|---|---|---|---|
| `dec8` (decode, 6.11) | 8.00 | 8.00 → 6.56, **+0.45** | 7.92 → 6.48, **+0.37** (7.85: +0.30) | 0.37–0.45 |
| `enc12` (encode, 9.72) | 12.00, 11.80, 11.60, 11.40 | 11.40 → 9.96, **+0.24** | 11.30 → 9.86, **+0.14** (11.20: +0.04) | 0.14–0.24 |
| `dit24` (DiT, 21.81) | 24.00, 23.85, 23.70, 23.55, 23.48 | 23.48 → 22.04, **+0.23** | 23.40 → 21.96, **+0.15** | 0.15–0.23 |
| `dit30` (DiT, 28.15) | 30.00, 29.90, 29.85 | 29.85 → 28.41, **+0.26** | 29.80 → 28.36, **+0.21** | 0.21–0.26 |

- **The edge is a narrow band, and it moves with the bounding phase:** a run needs 0.15–0.45 GiB
  between the free memory it starts with and its torch peak. The decode-bound small card needs the
  most (it fails with 0.37 GiB left), the encode- and DiT-bound ones pass down to 0.23–0.26, with
  or without BlockSwap.
- **Passing runs show nothing but silent allocator retries** ("`[cudaMallocAsync] recovered from
  an allocation failure by trimming the pool and retrying`"), no OOM event: `dec8` 118, all in
  decode; `enc12` 67, 108, 134 and 169 as its margin goes 0.84 → 0.24; `dit24` 34–45 and `dit30`
  32–36 at every cap, nearly all in the DiT (with BlockSwap, numz empties the cache after each
  swapped block once less than 5% is free, then the pool regrows,
  [vram.md](vram.md#recipe-per-card-size-validated)). They cost no measurable time: `enc12` took
  141–143 s and `dit24` 195–198 s at every cap.
- **The encode peak gives way under pressure:** 9.52 GiB instead of 9.72 at the 11.40 GB cap,
  9.52 and 9.18 reached in the two failing runs. The DiT and decode peaks never moved. Plausibly
  cuDNN choosing convolution algorithms with smaller workspaces when memory is short (not
  verified); a planner can't count on it.

**How a run fails.** At the failing allocation the device was full while torch held 0.8–1.0 GiB
less than the free memory the run started with:

| Run | Free at start | Torch allocated | Request | Device free | Outside torch's tensors | Where |
|---|---|---|---|---|---|---|
| `dec8` 7.92 GB | 6.48 | 5.51 | 0.25 | 21 MiB | 0.95 | decode: a causal conv's cache copy (`InflatedCausalConv3d`, `causal_inflation_lib.py:276`) |
| `dec8` 7.85 GB | 6.41 | 5.51 | 0.25 | 13 MiB | 0.89 | same place |
| `enc12` 11.30 GB | 9.86 | 8.81 | 0.39 | 27 MiB | 1.02 | encode: a causal conv's padded input (`pad_and_forward`, `causal_inflation_lib.py:141`) |
| `enc12` 11.20 GB | 9.76 | 8.81 | 0.39 | 21 MiB | 0.93 | same place |
| `dit24` 23.40 GB | 21.96 | 21.19 | 0.61 | 1 MiB | 0.77 | DiT: flash-attn's varlen forward output, first forward |
| `dit30` 29.80 GB | 28.36 | 27.53 | 0.61 | 25 MiB | 0.81 | same place |

GiB unless noted; "outside torch's tensors" = free at start − allocated − device free: CUDA
context growth since the start (libraries and kernels load on first use; on the full card the
CLI reads 94.50 GiB free at import, and 94.36 GiB are left once numz's import-time bf16 matmul
check has run) and memory the allocator's pool still holds.

- **VAE phases retry once, uselessly:** numz wraps every VAE conv in `retry_on_oom`
  (`src/optimization/memory_manager.py:361-401`): it logs "OOM during
  InflatedCausalConv3d.pad_and_forward", empties the cache, waits 0.5 s and retries; at the edge
  the retry fails the same way (9 OOM log lines in decode, 6 in encode, after 27–84 allocator
  retries) and the CLI exits with status 1.
- **The DiT doesn't retry:** "Forward pass error: … out of memory" (`compatibility.py:933`) after
  only 2 allocator retries, at the first forward, exit status 1.
- A run that fails costs only the time up to the failing phase (19–69 s here).

## seedvr2x's tiled passes: every tile's peak

The tiled fits above come from numz's CLI and from a few tile sizes. seedvr2x runs tiled passes
of its own (`runtime/tiling.py`: numz's grid and blend, each tile run over the whole shot, the
frames and the latents waiting in system memory), and its planner needs an upper bound for
whichever tile it picks. [`tile_peaks.py`](../scripts/tile_peaks.py) measured every tile it may
pick, on seedvr2x's own runtime (2026-10-11; the same GPU, torch 2.14.1+cu130, `cudaMallocAsync`,
`flash_attn_2`, 7B fp16).

**How measured.** The two models loaded whole, as seedvr2x loads them without a plan; a shot of
**21 frames** of a 1080p clip (6 latents, encoded in slices of 5, 4, 4, 4 and 4 frames), the
tiles sharing **64 px**, the scratch in system memory. A pass's peak is
`torch.cuda.max_memory_allocated()` above `memory_allocated()` before the pass, the statistics
reset first: the DiT's and the VAE's weights sit on the card before it and are not counted. The
encode is `tiling.encode_tiled`, its latent returned in system memory; the decode is
`tiling.decode_tiled` of that latent, each slice dropped as it comes. One process, the tiles in
the tables' order, each measured once.

- **The 25 squares,** 512 to 2048 px in steps of 64, each fed alone: the clip's frames resized
  with torch (bicubic, antialias) to T rows × T + 128 columns, normalised to [−1, 1], bf16, in
  the VAE's slices. The grid is 1 × 2, a whole T × T tile and one of T × 192.
- **The 20 tiles cut off at the picture's edge,** through the encoder's own transform and padding
  (`shot.encoder_inputs`): 1080p, padded to 1104 × 1920, in tiles of 1152 to 1856 px (12), and
  720p, padded to 752 × 1280, in tiles of 768 to 1216 px (8). The grid is 1 × 2, its largest
  tile the padded rows × T.
- **The tree:** seedvr2x at `e6450d8` with the VAE passes that `2adcea7` then brought, before
  that commit. By a diff of the measured tree against `ca9e6e1`: the vendored model code,
  `runtime/job.py` (the padding) and `runtime/tilegrid.py` (the grid) are the same to the letter;
  `runtime/tiling.py` differs in the scratch allowance's name, comment and default alone;
  `runtime/model.py` in its imports, in `Models.enter` and in the log of a plan's load, none of
  them in the VAE's passes; `runtime/shot.py` outside the encoder's inputs.

Peaks in GiB (2³⁰ bytes) above what was allocated before the pass; Mpx is the largest tile's
area (10⁶ pixels). The fits are taken at that area: from 1024-px tiles, encode 1.36 + 8.66 × T²
and decode 0.44 + 16.29 × T² ([above](#tiled-vae-at-4k)); below, encode 1.7 + 8.4 × T² and
decode 1.6 + 15.6 × T² ([vram.md](vram.md#tiling)); a cut-off tile takes the fit of its side.
The squares' line is the straight line, by area, between the two measured squares whose areas
are around the tile's; "the larger" is the larger of the fit and that line. The tables are
`tile_peaks.py table`'s output.

### The squares: a whole T × T tile (25)

| Tile | Mpx | Encode | Encode's fit | Encode − fit | Decode | Decode's fit | Decode − fit |
|---|---|---|---|---|---|---|---|
| 512 | 0.2621 | 3.518 | 3.902 | −0.384 | 5.886 | 5.689 | +0.197 |
| 576 | 0.3318 | 4.380 | 4.487 | −0.107 | 7.446 | 6.776 | +0.671 |
| 640 | 0.4096 | 4.921 | 5.141 | −0.220 | 8.087 | 7.990 | +0.097 |
| 704 | 0.4956 | 5.953 | 5.863 | +0.090 | 9.788 | 9.332 | +0.456 |
| 768 | 0.5898 | 7.083 | 6.655 | +0.428 | 11.089 | 10.801 | +0.288 |
| 832 | 0.6922 | 8.311 | 7.515 | +0.797 | 12.298 | 12.399 | −0.101 |
| 896 | 0.8028 | 8.560 | 8.444 | +0.116 | 14.100 | 14.124 | −0.024 |
| 960 | 0.9216 | 9.817 | 9.441 | +0.375 | 15.675 | 15.977 | −0.302 |
| 1024 | 1.0486 | 10.459 | 10.441 | +0.019 | 17.450 | 17.521 | −0.071 |
| 1088 | 1.1837 | 11.806 | 11.611 | +0.195 | 19.647 | 19.723 | −0.076 |
| 1152 | 1.3271 | 13.235 | 12.853 | +0.383 | 21.950 | 22.059 | −0.108 |
| 1216 | 1.4787 | 14.152 | 14.165 | −0.013 | 24.401 | 24.527 | −0.126 |
| 1280 | 1.6384 | 15.673 | 15.549 | +0.125 | 26.982 | 27.130 | −0.148 |
| 1344 | 1.8063 | 16.790 | 17.003 | −0.213 | 29.707 | 29.865 | −0.158 |
| 1408 | 1.9825 | 18.456 | 18.528 | −0.072 | 32.568 | 32.734 | −0.166 |
| 1472 | 2.1668 | 19.736 | 20.124 | −0.388 | 35.552 | 35.737 | −0.185 |
| 1536 | 2.3593 | 21.496 | 21.792 | −0.295 | 38.702 | 38.873 | −0.171 |
| 1600 | 2.5600 | 23.209 | 23.530 | −0.320 | 41.957 | 42.142 | −0.185 |
| 1664 | 2.7689 | 25.103 | 25.339 | −0.236 | 45.302 | 45.545 | −0.243 |
| 1728 | 2.9860 | 27.012 | 27.219 | −0.207 | 48.794 | 49.082 | −0.287 |
| 1792 | 3.2113 | 29.003 | 29.170 | −0.166 | 52.478 | 52.751 | −0.273 |
| 1856 | 3.4447 | 31.121 | 31.191 | −0.070 | 56.317 | 56.555 | −0.238 |
| 1920 | 3.6864 | 33.256 | 33.284 | −0.028 | 60.165 | 60.491 | −0.327 |
| 1984 | 3.9363 | 35.462 | 35.448 | +0.014 | 64.219 | 64.562 | −0.342 |
| 2048 | 4.1943 | 37.756 | 37.683 | +0.073 | 68.482 | 68.765 | −0.283 |

### The tiles cut off at the picture's edge: the encode (20)

| Tile | Largest tile | Mpx | Encode | Fit | Measured − fit | Squares' line | Between | Measured − line | Measured − the larger |
|---|---|---|---|---|---|---|---|---|---|
| 1152 | 1104 × 1152 | 1.2718 | 12.684 | 12.374 | +0.311 | 12.684 | 1088 and 1152 | +0.000 | +0.000 |
| 1216 | 1104 × 1216 | 1.3425 | 13.389 | 12.986 | +0.403 | 13.328 | 1152 and 1216 | +0.061 | +0.061 |
| 1280 | 1104 × 1280 | 1.4131 | 13.546 | 13.598 | −0.051 | 13.756 | 1152 and 1216 | −0.209 | −0.209 |
| 1344 | 1104 × 1344 | 1.4838 | 14.223 | 14.210 | +0.014 | 14.201 | 1216 and 1280 | +0.023 | +0.014 |
| 1408 | 1104 × 1408 | 1.5544 | 14.901 | 14.821 | +0.079 | 14.874 | 1216 and 1280 | +0.027 | +0.027 |
| 1472 | 1104 × 1472 | 1.6251 | 15.578 | 15.433 | +0.145 | 15.546 | 1216 and 1280 | +0.032 | +0.032 |
| 1536 | 1104 × 1536 | 1.6957 | 16.255 | 16.045 | +0.210 | 16.055 | 1280 and 1344 | +0.201 | +0.201 |
| 1600 | 1104 × 1600 | 1.7664 | 16.422 | 16.657 | −0.235 | 16.525 | 1280 and 1344 | −0.102 | −0.235 |
| 1664 | 1104 × 1664 | 1.8371 | 17.079 | 17.269 | −0.190 | 17.081 | 1344 and 1408 | −0.002 | −0.190 |
| 1728 | 1104 × 1728 | 1.9077 | 17.736 | 17.881 | −0.145 | 17.749 | 1344 and 1408 | −0.013 | −0.145 |
| 1792 | 1104 × 1792 | 1.9784 | 18.393 | 18.493 | −0.100 | 18.418 | 1344 and 1408 | −0.025 | −0.100 |
| 1856 | 1104 × 1856 | 2.0490 | 19.050 | 19.105 | −0.055 | 18.918 | 1408 and 1472 | +0.131 | −0.055 |
| 768 | 752 × 768 | 0.5775 | 6.936 | 6.551 | +0.384 | 6.936 | 704 and 768 | +0.000 | +0.000 |
| 832 | 752 × 832 | 0.6257 | 7.513 | 6.956 | +0.558 | 7.513 | 768 and 832 | +0.000 | +0.000 |
| 896 | 752 × 896 | 0.6738 | 8.091 | 7.360 | +0.731 | 8.090 | 768 and 832 | +0.001 | +0.001 |
| 960 | 752 × 960 | 0.7219 | 7.700 | 7.764 | −0.064 | 8.378 | 832 and 896 | −0.678 | −0.678 |
| 1024 | 752 × 1024 | 0.7700 | 8.213 | 8.029 | +0.185 | 8.486 | 832 and 896 | −0.273 | −0.273 |
| 1088 | 752 × 1088 | 0.8182 | 8.727 | 8.445 | +0.281 | 8.722 | 896 and 960 | +0.004 | +0.004 |
| 1152 | 752 × 1152 | 0.8663 | 9.240 | 8.862 | +0.378 | 9.232 | 896 and 960 | +0.008 | +0.008 |
| 1216 | 752 × 1216 | 0.9144 | 9.753 | 9.279 | +0.474 | 9.741 | 896 and 960 | +0.012 | +0.012 |

### The tiles cut off at the picture's edge: the decode (20)

| Tile | Largest tile | Mpx | Decode | Fit | Measured − fit | Squares' line | Between | Measured − line | Measured − the larger |
|---|---|---|---|---|---|---|---|---|---|
| 1152 | 1104 × 1152 | 1.2718 | 21.035 | 21.158 | −0.122 | 21.062 | 1088 and 1152 | −0.026 | −0.122 |
| 1216 | 1104 × 1216 | 1.3425 | 22.204 | 22.309 | −0.105 | 22.198 | 1152 and 1216 | +0.006 | −0.105 |
| 1280 | 1104 × 1280 | 1.4131 | 23.336 | 23.460 | −0.124 | 23.341 | 1152 and 1216 | −0.005 | −0.124 |
| 1344 | 1104 × 1344 | 1.4838 | 24.503 | 24.611 | −0.108 | 24.484 | 1216 and 1280 | +0.019 | −0.108 |
| 1408 | 1104 × 1408 | 1.5544 | 25.670 | 25.762 | −0.092 | 25.625 | 1216 and 1280 | +0.045 | −0.092 |
| 1472 | 1104 × 1472 | 1.6251 | 26.778 | 26.913 | −0.135 | 26.767 | 1216 and 1280 | +0.011 | −0.135 |
| 1536 | 1104 × 1536 | 1.6957 | 27.942 | 28.064 | −0.122 | 27.912 | 1280 and 1344 | +0.029 | −0.122 |
| 1600 | 1104 × 1600 | 1.7664 | 29.060 | 29.215 | −0.154 | 29.059 | 1280 and 1344 | +0.001 | −0.154 |
| 1664 | 1104 × 1664 | 1.8371 | 30.223 | 30.366 | −0.143 | 30.206 | 1344 and 1408 | +0.016 | −0.143 |
| 1728 | 1104 × 1728 | 1.9077 | 31.385 | 31.517 | −0.131 | 31.354 | 1344 and 1408 | +0.031 | −0.131 |
| 1792 | 1104 × 1792 | 1.9784 | 32.486 | 32.668 | −0.182 | 32.502 | 1344 and 1408 | −0.015 | −0.182 |
| 1856 | 1104 × 1856 | 2.0490 | 33.646 | 33.819 | −0.172 | 33.646 | 1408 and 1472 | +0.001 | −0.172 |
| 768 | 752 × 768 | 0.5775 | 10.859 | 10.610 | +0.249 | 10.920 | 704 and 768 | −0.061 | −0.061 |
| 832 | 752 × 832 | 0.6257 | 11.493 | 11.360 | +0.133 | 11.512 | 768 and 832 | −0.019 | −0.019 |
| 896 | 752 × 896 | 0.6738 | 12.377 | 12.111 | +0.266 | 12.080 | 768 and 832 | +0.296 | +0.266 |
| 960 | 752 × 960 | 0.7219 | 12.694 | 12.862 | −0.168 | 12.782 | 832 and 896 | −0.087 | −0.168 |
| 1024 | 752 × 1024 | 0.7700 | 13.540 | 12.984 | +0.556 | 13.566 | 832 and 896 | −0.026 | −0.026 |
| 1088 | 752 × 1088 | 0.8182 | 14.386 | 13.768 | +0.618 | 14.304 | 896 and 960 | +0.083 | +0.083 |
| 1152 | 752 × 1152 | 0.8663 | 14.919 | 14.552 | +0.367 | 14.942 | 896 and 960 | −0.022 | −0.022 |
| 1216 | 752 × 1216 | 0.9144 | 15.585 | 15.336 | +0.249 | 15.580 | 896 and 960 | +0.005 | +0.005 |

### Where the fits stand against them

- **The fits are no upper bound.** 11 of the 25 squares' encodes and 5 of their decodes stand
  over their fit, as do 13 and 7 of the 20 cut-off tiles'. The largest shortfalls of the
  encode's fit: 0.797 GiB at 832 px, 0.731 at 752 × 896, 0.558 at 752 × 832, 0.474 at
  752 × 1216, 0.428 at 768 px, 0.403 at 1104 × 1216. Of the decode's: 0.671 GiB at 576 px, 0.618
  at 752 × 1088, 0.556 at 752 × 1024, 0.456 at 704 px.
- **A decode follows its tile's area from 1024 px up, not below.** From 1024 px every square's
  decode is 0.071 to 0.342 GiB under the refit and grows by 15.9 to 16.5 GiB per megapixel from
  one square to the next (the refit's slope is 16.29); the 1080p cut-off tiles' are 0.092 to
  0.182 under. Below 1024 px a square's decode goes from 0.302 GiB under its fit (960 px) to
  0.671 over (576 px), growing by 8.2 to 22.4 GiB per megapixel from one square to the next.
- **An encode moves in steps:** from 0.388 GiB under its fit (1472 px) to 0.797 over (832 px), a
  square adding 2.2 to 12.4 GiB per megapixel to the one before it (the least from 832 to
  896 px, the most from 512 to 576). A larger tile may take less: 752 × 960 encodes in 0.391 GiB
  less than 752 × 896.
- **A cut-off tile's decode lies on the squares' line,** 19 of the 20 within 0.09 GiB of it and
  one 0.296 over (752 × 896). The encodes stand from 0.678 GiB under that line (752 × 960) to
  0.201 over (1104 × 1536).
- **Over the larger of its fit and of the squares' line,** a cut-off tile stands by 0.266 GiB at
  the most for a decode (752 × 896: 12.377 for a fit of 12.111 and a line of 12.080) and by 0.201
  for an encode (1104 × 1536).
- **Alone or in a whole frame, the same tile.** Whole 1080p and 4K frames in tiles of 512, 768,
  1024, 1280 (cut off: 1104 × 1280), 1536 and 2048 px, measured earlier that day by the same
  peak over 21 frames, on the passes as they then stood, and kept to three decimals: their
  encodes read within 0.015 GiB of the tile fed alone, their decodes 0.006 to 0.026 over it.
  And 1024-px tiles over shots of 21 and 301 frames at 1080p and of 21 and 101 at 4K read the
  same bytes each time, 10.4742 and 17.4499 GiB, for 10.4593 and 17.4499 fed alone: a pass's
  peak doesn't grow with the shot.
- **seedvr2x's planner takes the larger where a tile has two measures,** to the hundredth
  (`runtime/planner.py`, `MEASURED`). That moves five of its 90 entries from the tables' value:
  768 px's decode (11.11 for 11.089), 1024 px's encode and decode (10.47 for 10.459, 17.46 for
  17.450), 1536 px's decode (38.73 for 38.702) and 2048 px's (68.51 for 68.482).

### Their caveats

- **Each tile was measured once, on one card,** with one driver and one cuDNN: no band from run
  to run, and a card of another generation may choose other convolution kernels, with other
  workspaces ([the encode's peak gave way under pressure](#the-margin-over-free-memory) on this
  one).
- **512 px's encode was its process's first VAE pass,** which left 32 MiB (0.031 GiB) allocated
  behind it where every later pass left none: its peak counts them, and may be that much high.
- **One overlap, one shot length:** 64 px and 21 frames. The cut-off tiles are 1080p's and
  720p's alone: a tile cut off at another picture's rows has no measure.
- **Allocated, not reserved:** the allocator's pool held more than the peaks say, up to
  94.25 GiB of the card's 95.05 at the 1984- and 2048-px decodes (64.219 and 68.482 allocated).
- **Times aren't in the tables:** the card is power-capped. The 45 tiles took 2,787 s in all:
  2.96 s to encode and 6.68 s to decode at 512 px, 55.9 s and 126.4 s at 2048 px.

### Under the one-output convolution (2026-10-17)

The tables above are the peaks of a VAE whose split convolutions concatenate their pieces. Since
`82cbd0d` a convolution cut in pieces allocates its whole output ahead and writes each piece into
it (`InflatedCausalConv3d.whole_output` in the vendored `causal_inflation_lib.py`), where the
output a slice of four frames gives is **2.5 GiB or more**, judged at each convolution's own call,
so that every slice of a pass takes one path at that convolution. Of the tiles above, the rule
takes these passes: the decodes in squares of 1152 to 2048 px (15), in tiles of 1104 rows × 1216
to 1856 px (11) and untiled at 1080p; the encodes in squares of 1664 to 2048 px (7) and untiled
at 4K (and the untiled 4K decode, which the 96 GB card cannot hold). It takes others the tables
don't have: 1440p's tiles of 1,472 rows and its untiled passes, and tiles cut off at 4K's edges.

**How measured.** The method of the section above, by
[`one_output_peaks.py`](../scripts/one_output_peaks.py) (`tile_peaks.py`'s measure unchanged, with
a list of steps, the outputs' digests and the path counted per convolution): the same card, torch
2.14.1+cu130, cuDNN 92400, `flash_attn_2` 2.8.3.post1, `cudaMallocAsync`; the two models loaded
whole (15.8386 GiB allocated, 78.4484 of 95.0512 free); 21 frames of the same 1080p clip from
frame 48, the tiles sharing 64 px; a pass's peak above what was allocated before it. The 35 passes
of the list were measured on **two codes**, the same steps in the same order: with the one output
and without it. Each code ran in four processes, one hold of the card for each group of tiles
(711 and 708 s, 796 and 792, 655 and 652, 893 and 892 s: eight holds, 2026-10-16 22:57 to
2026-10-17 00:38 UTC, no other compute application on the card, 10 MiB used before each), the
code with the one output first. Every process began with a 512-px pair, as 512 px was the first
pass of the process of 2026-10-11. The untiled passes are the run's own calls
(`model.encode_stream` of `shot.encoder_inputs`, `model.decode_stream` of its latent; 1080p
padded to 1104 × 1920, 4K to 2192 × 3840); the DiT's weights were moved to the CPU before the
untiled 4K encode on both codes (free memory 78.37 to 93.68 GiB: the encode's fit, 75.27 GiB,
leaves 3.1 GiB of the 78.37 beside them), the peaks counting from what is allocated before the
pass either way. The SHA-256 of each tile's latent and of each decoded slice was kept, with the
convolutions that took the one output.

**The tree.** With the one output: seedvr2x at `82cbd0d`; without it: its parent `c9afce4`. The
two differ in `causal_inflation_lib.py` and `tests/test_causal_conv.py` alone (the trees of
`seedvr2x/` without `DESIGN.md` and `AGENTS.md`, the ones the measure ran: `0a81d0a9` and
`d045524f`).

Peaks in GiB above what was allocated before the pass, from the records' bytes. "Before" is the
planner's figure at `fbf1ee4` (`runtime/planner.py`, `MEASURED`; for an untiled pass, its fit); the
differences are the peak with the one output less the figure, and less the peak without it. A
decode square or encode square is named by its side, a cut-off tile by its size. The convolutions
that took the one output, on all five slices of the pass's largest tile: **D1** =
`decoder.up_blocks.2.upsamplers.0.conv`; **D9** = D1, `decoder.up_blocks.1.upsamplers.0.conv`,
`decoder.up_blocks.3.resnets.{0,1,2}.conv1` and `.conv2`, and `resnets.0.conv_shortcut`; **E4** =
`encoder.down_blocks.0.resnets.{0,1}.conv1` and `.conv2`; **E9** = E4, and
`encoder.down_blocks.1.resnets.0.conv1`, `.conv2` and `.conv_shortcut`, and `resnets.1.conv1` and
`.conv2`. On a square the T × T tile takes it and the T × 192 one concatenates: the two paths are
never mixed within a tile's pass. "Equal": the latent, or the five decoded slices, has one
SHA-256 in every run of both codes.

**The decodes (27 measured)**

| Tile | Before | Without | With | With − before | With − without | Convolutions | Outputs |
|---|---|---|---|---|---|---|---|
| 1152 | 21.950 | 21.950 | 21.986 | +0.036 | +0.036 | D1 | equal |
| 1216 | 24.400 | 24.401 | 24.138 | −0.262 | −0.263 | D1 | equal |
| 1280 | 26.980 | 26.982 | 26.396 | −0.584 | −0.586 | D1 | equal |
| 1344 | 29.710 | 29.707 | 28.838 | −0.872 | −0.869 | D1 | equal |
| 1408 | 32.570 | 32.568 | 31.313 | −1.257 | −1.255 | D1 | equal |
| 1472 | 35.550 | 35.552 | 33.974 | −1.576 | −1.577 | D1 | equal |
| 1536 | 38.730 | 38.702 | 36.743 | −1.987 | −1.958 | D1 | equal |
| 1600 | 41.960 | 41.957 | 39.649 | −2.311 | −2.309 | D1 | equal |
| 1664 | 45.300 | 45.302 | 42.455 | −2.845 | −2.847 | D9 | equal |
| 1728 | 48.790 | 48.794 | 45.681 | −3.109 | −3.113 | D9 | equal |
| 1792 | 52.480 | 52.478 | 48.928 | −3.552 | −3.550 | D9 | equal |
| 1856 | 56.320 | 56.317 | 52.479 | −3.841 | −3.838 | D9 | equal |
| 1920 | 60.160 | 60.165 | 55.687 | −4.473 | −4.478 | D9 | equal |
| 1984 | 64.220 | 64.219 | 59.249 | −4.971 | −4.970 | D9 | equal |
| 2048 | 68.510 | 68.482 | 63.342 | −5.168 | −5.140 | D9 | equal |
| 1104 × 1216 | 22.200 | 22.204 | 22.242 | +0.042 | +0.037 | D1 | equal |
| 1104 × 1280 | 23.340 | 23.336 | 23.158 | −0.182 | −0.178 | D1 | equal |
| 1104 × 1344 | 24.500 | 24.503 | 24.180 | −0.320 | −0.323 | D1 | equal |
| 1104 × 1408 | 25.670 | 25.670 | 25.209 | −0.461 | −0.461 | D1 | equal |
| 1104 × 1472 | 26.780 | 26.778 | 26.266 | −0.514 | −0.512 | D1 | equal |
| 1104 × 1536 | 27.940 | 27.942 | 27.299 | −0.641 | −0.643 | D1 | equal |
| 1104 × 1600 | 29.060 | 29.060 | 28.259 | −0.801 | −0.801 | D1 | equal |
| 1104 × 1664 | 30.220 | 30.223 | 29.272 | −0.948 | −0.951 | D1 | equal |
| 1104 × 1728 | 31.390 | 31.385 | 30.254 | −1.136 | −1.131 | D1 | equal |
| 1104 × 1792 | 32.490 | 32.486 | 31.280 | −1.210 | −1.206 | D1 | equal |
| 1104 × 1856 | 33.650 | 33.646 | 32.380 | −1.270 | −1.267 | D1 | equal |
| untiled 1104 × 1920 | 34.927 | 34.828 | 33.315 | −1.612 | −1.514 | D1 | equal |

The untiled 4K decode (fit 136.318 GiB, the card 95.05) ran on neither code.

**The encodes (8 measured)**

| Tile | Before | Without | With | With − before | With − without | Convolutions | Outputs |
|---|---|---|---|---|---|---|---|
| 1664 | 25.100 | 25.103 | 24.877 | −0.223 | −0.226 | E4 | equal |
| 1728 | 27.010 | 27.012 | 26.720 | −0.290 | −0.291 | E4 | equal |
| 1792 | 29.000 | 29.003 | 28.436 | −0.564 | −0.567 | E4 | equal |
| 1856 | 31.120 | 31.121 | 30.458 | −0.662 | −0.664 | E4 | equal |
| 1920 | 33.260 | 33.256 | 32.291 | −0.969 | −0.965 | E4 | equal |
| 1984 | 35.460 | 35.462 | 34.283 | −1.177 | −1.179 | E4 | equal |
| 2048 | 37.760 | 37.756 | 36.327 | −1.433 | −1.429 | E4 | equal |
| untiled 2192 × 3840 | 75.272 | 75.376 | 70.791 | −4.481 | −4.585 | E9 | equal |

**The controls.** The 22 passes off the list (the 512-px pair, the encodes of the squares of 1152
to 1600 px, the 11 encodes of the 1104-row tiles and the untiled 1080p encode): no convolution
took the one output, the peak is the same to the byte with and without it, the outputs are equal,
and they read within −0.005 to +0.003 GiB of the table's figure; the untiled 1080p encode, 19.350
GiB on both codes, is 0.503 under its fit (19.853).

**Without it, the same tiles.** The 33 tiled passes of the list read −0.028 to +0.005 GiB from
the table's figure (the 0.028 under are the 1536-px and 2048-px decodes, where the table keeps the
larger of two measures, to the hundredth); the untiled 1080p decode 34.828 for a fit of 34.927,
and the untiled 4K encode **75.376 for a fit of 75.272**, 0.104 over it (the planner adds 0.4 GiB
to every encode, `ENCODE_EXTRA`).

**Determinism.** The 512-px pair ran in all eight processes: the encode's peak 3,777,345,824 B
and the decode's 6,320,068,752 B in all eight, one latent digest (`bbec29f9…`), one digest of the
decode (`bd9324b5…`) and one set of slice digests.

**Just over the floor.** Two passes stand over their figure with the one output: the decode in
1152-px squares (21.986 for 21.95, +0.036; a slice's largest output there 2.531 GiB) and in tiles
of 1104 × 1216 (22.242 for 22.20, +0.042; 2.561 GiB). Each takes the path at one convolution,
`decoder.up_blocks.2.upsamplers.0.conv`, and its peak is the higher by 0.036 and 0.037 GiB than
without it: where the pieces were small, the output allocated ahead costs more than they did.
Their outputs are equal. Every other pass of the list is lower, 0.18 GiB (the 1104 × 1280 decode)
to 5.17 (the 2048-px decode) under its figure, the untiled 4K encode 4.48 under its fit.

**The allocator.** `cudaMallocAsync` logged "recovered from an allocation failure by trimming the
pool and retrying" five times in all, once in the first group with the one output (in the 2048-px
decode, the pool's reserved peak 94.25 GiB, as on 2026-10-11), three times in the first group
without it and once in the second: no output moved.

**What the planner takes from it.** The peak with the one output, rounded up to the hundredth,
for the passes it takes that were measured (the untiled ones keep their fits: one point each, no
refit). Decodes in squares: 1152 21.99, 1216 24.14, 1280 26.40, 1344 28.84, 1408 31.32, 1472
33.98, 1536 36.75, 1600 39.65, 1664 42.46, 1728 45.69, 1792 48.93, 1856 52.48, 1920 55.69, 1984
59.25, 2048 63.35. Decodes in tiles of 1104 rows × 1216 22.25, 1280 23.16, 1344 24.18, 1408
25.21, 1472 26.27, 1536 27.30, 1600 28.26, 1664 29.28, 1728 30.26, 1792 31.29, 1856 32.38.
Encodes in squares: 1664 24.88, 1728 26.73, 1792 28.44, 1856 30.46, 1920 32.30, 1984 34.29, 2048
36.33. The other passes keep their figures.

**Their caveats.** Each pass was measured once on each code (the 512-px pair four times), on one
card, with one driver and one cuDNN, as before. The tile was fed alone: the whole-frame second
measures of 2026-10-11, up to 0.03 GiB over the tile alone, were not repeated with the one output.

## What the planner should use

- **Budget = the free memory read once the CUDA context exists, minus 0.6 GiB.** Runs failed with
  up to 0.37 GiB between that free memory and the bounding phase's torch peak and passed from
  0.23–0.45 GiB, the decode-bound small card needing the most (DiT-bound runs failed at
  0.15–0.21 and passed from 0.23–0.26, with or without BlockSwap); 0.6 GiB keeps 0.23 GiB over the
  worst failure measured (0.37) and 0.15 GiB over the tightest pass (0.45). On the emulated cards
  that is the validated rule again (N − 1.44 − 0.6 ≈ N − 2 GiB), and on the real 96 GB card a
  1080p window passed with 0.56 GiB left. Read it before anything else touches the GPU: numz's
  own import-time bf16 check already takes 0.14 GiB of it.
- **The margin covers memory outside torch's tensors, not the formulas' errors:** the predicted
  peaks must be upper bounds. The DiT's are exact with the window term; the VAE fits are within
  ≈ 0.5 GiB up to 1536 tiles and need the refit beyond. On seedvr2x's own tiled passes neither
  tiled fit is one, the refit included: up to 0.797 GiB short.
- **DiT peak = F + 127.16 KiB × tokens + 2.726 MiB × windows,** with F what the loaded model holds
  (15.87 GiB for 7B fp16 with the VAE resident; the other models differ by their weights,
  [vram.md](vram.md#practical-rules)) and the windows counted with the model's own window
  functions (the larger of its two layouts). The per-token model alone is short by up to 0.83 GiB
  at 4K. The window length is then bounded by memory alone: 78 latents (309 frames) at 1080p and
  19 (73 frames) at 4K on 96 GB.
- **DiT time = tokens × the machine's time per token,** flat within ±5% across window lengths and
  resolutions. That time moved between 0.23 and 0.45 ms on this GPU with its clock: measure it on
  the first window rather than trusting a constant.
- **Tiled VAE:** decode ≈ 0.44 + 16.29 × T², encode ≈ 1.36 + 8.66 × T² for tiles from 1024 up (the
  old fits for smaller ones), plus what is on the GPU before the call: the VAE weights, and for
  encode the frames numz moves there first (0.046 GiB per 4K frame). seedvr2x, which encodes a
  whole shot in one causal pass, should feed the frames slice by slice rather than count them.
  Time per frame hardly depends on the tile size (5.3–5.6 s to encode, 11.5–12.3 s to decode a 4K
  frame here), so the largest tile that fits costs nothing and blends the fewest tile edges
  ([vram.md](vram.md#quality-tiled-vs-untiled-1080p-batch-9-png-output): tile size is what drives
  the tiled decode's drift).
- **seedvr2x's tiled passes: a tile's measured peak, not its fit**
  ([every tile's peak](#seedvr2xs-tiled-passes-every-tiles-peak)). Each of the 45 tiles measured
  takes its own, the larger where it was measured twice, since the fits fall up to 0.797 GiB
  short for an encode and 0.671 for a decode. A tile cut off at another picture's rows, which
  nobody measured, takes the larger of its fit and of the line between the two squares around
  its area, and 0.3 GiB more ([DESIGN.md](../../seedvr2x/DESIGN.md#memory-planner)): the measured
  cut-off tiles stand 0.266 GiB over the larger of the two at the most. Beyond 2048 px nothing
  is measured.
  Each adds what is on the GPU before the call, the VAE's weights and the DiT blocks left there,
  and nothing of the shot, which seedvr2x's passes feed from system memory and keep there.

## Caveats

- **Emulated cards.** The margin edges come from a driver-level ballast on a 96 GB card, which
  reproduces the memory a smaller card leaves, not its context size, its driver's hidden memory
  or its kernels' workspaces ([vram.md](vram.md#emulating-a-smaller-gpu)). On a real card that
  drives a display, other programs' memory changes while a run lasts: the budget is read once at
  start.
- **One GPU, one driver, one software stack** (Blackwell, torch 2.14.1+cu130, cuDNN 9.24,
  flash-attn 2.8.3, `cudaMallocAsync`). Lazy kernel loading, cuDNN's algorithm choices and the
  allocator's pool all feed the margin; another stack can need more. `expandable_segments` was not
  bisected.
- **One edge per configuration,** each refined once: the edge is a band of 0.05–0.10 GiB, and a
  rerun at the edge could land on either side.
- **Power-capped clocks:** the GPU sits at its 600 W cap and its clock drifts within minutes (the
  reference forward from 16.0 to 21.8 s within one 4-minute lock hold) and between sessions.
  Times are normalised by the reference around each point; their scatter (±5%) is mostly that
  drift. Absolute times only hold for this GPU in these sessions.
- **Random latents:** the DiT's memory and time don't depend on the latents' values (dense compute
  over a fixed sequence), so random latents stand for real ones; the peaks match the CLI's
  measured runs to the hundredth of a GiB where both exist.
- **7B fp16 only for the DiT sweeps.** The per-window term comes from the text repetition, which
  every model has; the other models' constants (weights, per-token slope) are in
  [vram.md](vram.md#practical-rules).
- **seedvr2x's tile peaks** have caveats of their own, each tile measured once, on this one card
  ([their caveats](#their-caveats)).

## Reproduce

```bash
# cwd = the numz checkout, its venv's python; GPU commands one at a time
S=/path/to/scripts; M=/path/to/models; O=out/q6
# DiT, 4K: one invocation, the edge bisected
python $S/dit_probe.py run --model-dir $M --attention-mode flash_attn_2 \
  --sweep 2160:1,2,3,4,6,8,10,12,14,16,18,20,22,24 --ref 1080:6 --ref-every 3 --repeats 2 \
  --refine --out $O/dit-2160.json
# DiT, 1080p: one invocation per point (bracketed by the reference), then the edge by hand
python $S/dit_probe.py run --model-dir $M --sweep 1080:1,2,3,6,12 --ref 1080:6 --ref-every 3 \
  --repeats 2 --out $O/dit1080-L1-2-3-6-12.json
for L in 21 30 40 50 60 70 80; do
  python $S/dit_probe.py run --model-dir $M --sweep 1080:$L --ref 1080:6 --ref-every 0 \
    --repeats 2 --out $O/dit1080-L$L.json
done
python $S/dit_probe.py run --model-dir $M --sweep 1080:75 --ref 1080:6 --ref-every 0 \
  --repeats 1 --ref-repeats 2 --out $O/dit1080-refine-L75.json      # and so on, halving
python $S/dit_probe.py table $O/dit-2160.json                       # per file
python $S/dit_probe.py table --merge $O/dit1080-*.json              # one table, all references
python $S/dit_probe.py plan --sweep 1080:1,6 --sweep 2160:1,2       # points and GPU time, no GPU

# tiled VAE at 4K: one CLI run per tile size (VAE probe JSON next to each log)
python3 $S/bench.py run q6-vae-2160-bs21-t1536o128 --wrap $S/vae_probe.py -- /path/to/clip-1080p.mkv \
  --output out/ --model_dir $M --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 2160 \
  --batch_size 21 --load_cap 21 --attention_mode flash_attn_2 --color_correction none \
  --vae_encode_tiled --vae_decode_tiled --vae_encode_tile_size 1536 --vae_decode_tile_size 1536 \
  --vae_encode_tile_overlap 128 --vae_decode_tile_overlap 128
python3 $S/vae_probe.py --summary runs/q6-vae-2160-*.vae.json

# margin: the same CLI run on emulated cards, N lowered until the run fails (here dec8)
python3 $S/bench.py run q6m-dec8-c7.92 --wrap $S/vram_cap.py --env VRAM_CAP_GIB=7.92 -- \
  /path/to/segment-1080p.mkv --output out/ --model_dir $M --skip_first_frames 48 \
  --attention_mode flash_attn_2 --color_correction none --resolution 1080 \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf --dit_offload_device cpu --blocks_to_swap 36 \
  --batch_size 13 --load_cap 13 --vae_encode_tiled --vae_decode_tiled \
  --vae_encode_tile_size 640 --vae_decode_tile_size 512 \
  --vae_encode_tile_overlap 64 --vae_decode_tile_overlap 64
python3 $S/bench.py table q6m-dec8-c8.00 q6m-dec8-c7.92 q6m-dec8-c7.85
```

`bench.py` takes `--seedvr2-dir` and `--runs-dir` (or `SEEDVR2_DIR`, `BENCH_RUNS_DIR`) and
`--env PATH=…` for the CLI's ffmpeg ([benchmarking.md](benchmarking.md#usage)). The other
configurations change only the CLI arguments of the configuration table above (`enc12`: batch 29,
tiles 896 / 640; `dit24`: `--dit_model seedvr2_ema_7b_fp16.safetensors --dit_offload_device cpu
--blocks_to_swap 16`, batch 45, tiles 1344 / 1024; `dit30`: 7B fp16 without BlockSwap, batch 45,
tiles 1024 / 1024).

seedvr2x's tiled passes run with the python of seedvr2x's environment, not numz's, `run` alone
on the GPU; `table` needs no environment:

```bash
S=/path/to/scripts; M=/path/to/models
python $S/tile_peaks.py plan                      # the 45 tiles and their grids, no GPU
python $S/tile_peaks.py run --dit $M/seedvr2_ema_7b_fp16.safetensors \
  --vae $M/ema_vae_fp16.safetensors --source /path/to/clip-1080p.mkv --out tiles.jsonl
python3 $S/tile_peaks.py table tiles.jsonl        # the section's three tables
```

The same tiles with and without the one-output convolution ([the subsection](#under-the-one-output-convolution-2026-10-17))
run in two trees of seedvr2x, a group of steps in one process each, the 512-px pair first; `run`
alone on the GPU, `measured` and `compare` need no GPU, `table` no environment:

```bash
S=/path/to/scripts; M=/path/to/models
python $S/one_output_peaks.py measured --steps square-512,square-2048,untiled-1080p \
  --out measured.json                             # the planner's figures for the steps
python $S/one_output_peaks.py run --dit $M/seedvr2_ema_7b_fp16.safetensors \
  --vae $M/ema_vae_fp16.safetensors --source /path/to/clip-1080p.mkv --first 48 \
  --steps square-512,square-2048,square-1600,untiled-1080p --out runs/g1-with.jsonl   # tree with it
python $S/one_output_peaks.py run ... --out runs/g1-without.jsonl                      # tree without it
python3 $S/one_output_peaks.py table --with runs/*-with.jsonl --without runs/*-without.jsonl \
  --measured measured.json --logs runs/*.log      # the subsection's tables
```
