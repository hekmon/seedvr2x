# Attention in the SeedVR2 DiT: what the backends really do

> Status: **analysis from the code** (SeedVR2 `4490bd1`, 7B DiT), **checked by instrumented
> runs** on the reference stack (RTX PRO 6000, sm_120). See [Measurements](#measurements): the
> window predictions hold exactly, `sageattn_3` never runs SA3, attention is 4–9% of DiT time,
> and the four backends give the same DiT time (±1.5%) up to batch 81. Their outputs differ by
> 55–56 dB PSNR, which VMAF doesn't see. The length-grouping patch is not worth writing.

## Windowed attention

Code: `src/models/dit_7b/window.py`, `src/models/dit_7b/nablocks/mmsr_block.py`,
`configs_7b/main.yaml`.

- The latent video (VAE: 8× spatial, 4× temporal) is patchified with `patch_size [1, 2, 2]`. One DiT
  token is therefore 2×2 latent pixels, i.e. **16×16 output pixels**, and one latent frame (≈ 4
  video frames).
- Attention is **3D-windowed**: the token grid `(t, h, w)` is cut into windows, and tokens only
  attend within their window. Each window, plus a copy of the text-prompt tokens
  (`na.repeat_concat_idx`, averaged back afterwards), is one independent sequence. All windows are
  packed into a single **varlen** attention call (`cu_seqlens`).
- Window config: `window: (4,3,3)` for all 36 layers, alternating `720pwin_by_size_bysize`
  (regular) and `720pswin_by_size_bysize` (shifted by half a window, Swin-style) between layers.
- **Window size is defined relative to 720p, so it doesn't change with output resolution.** The
  token grid is rescaled to a 45×80 area, which gives windows of `ceil(45/3) × ceil(80/3)` =
  **15 × 27 tokens** (240 × 432 output pixels) for 16:9 content. In time a window spans
  `wt = ceil(min(t, 30) / 4)` latent frames. **A higher resolution means more windows, not larger
  ones.** Only the batch size, through `t`, changes the window length.

| `--batch_size` | Latent frames `t` | `wt` | Max video tokens per window |
|---|---|---|---|
| 5 | 2 | 1 | 405 |
| 9 | 3 | 1 | 405 |
| 21 | 6 | 2 | 810 |
| 45 | 12 | 3 | 1215 |
| 81 | 21 | 6 | 2430 |
| ≥ 117 | ≥ 30 | 8 | 3240 (cap) |

Each sequence also contains the text tokens: a fixed embedding, `pos_emb.pt`, **58 tokens**
(measured).

## Windows are almost never all the same size

The window size rarely divides the token grid, and the shifted layers cut half-windows at the
borders. Example: 1080p output (padded to 1088 → 68 × 120 tokens):

- regular layers: heights 15,15,15,15,**8** × widths 27,27,27,27,**12**, so 4 distinct sizes
- shifted layers: heights **7**,15,15,15,15,**1** × widths **13**,27,27,27,**26**, down to windows
  1 token tall

720p gives widths 27/27/26. 4K (135 × 240 tokens) gives heights 9×15 (even) but widths
8×27 + 24. **Practically every attention call is variable-length.** Measured: not a single
uniform call in any run, 4K included (its regular layers still mix 2 lengths).

## Consequences for `--attention_mode`

- `sageattn_3`: SeedVR2's wrapper (`call_sage_attn_3_varlen`) only uses SA3 when every sequence
  in the call has the same length. Otherwise it falls back to SA2's `sageattn_varlen`.
  **Measured: SA3 never runs**, and `sageattn_3` behaves exactly like `sageattn_2`.
- `sageattn_2`: `sageattn_varlen` is a Triton INT8-QK/FP16-PV kernel. SA2's fast CUDA kernels
  only exist behind the batched API (see [environment.md](environment.md), pitfall 9). Making them
  usable means grouping each call's windows by length and calling the batched API once per group.
  The same grouping would let SA3 actually run. **Measured: at these lengths neither batched
  kernel beats FlashAttention 2's varlen kernel**, so the grouping buys nothing
  ([below](#is-the-length-grouping-patch-worth-it)).
- **Attention is a small share of DiT compute at these lengths.** Per token and per layer, the
  linear and MLP layers cost about `24·d²` FLOPs and attention about `4·L·d`. That makes the
  attention share about `L / (6·d + L)` with `d = 3072`: **≈ 2% at L ≈ 405** (batch 5–9),
  ≈ 15% for full 3240-token windows. With the real window mix (border windows are shorter) it
  is 2.2% at batch 5, 10% at batch 81 and 13% at the cap. *(The first version of this note said
  ≈ 18% at the cap: that was `L / 6d`, which overestimates long windows.)*

## Measurements

7B fp16, `--skip_first_frames 48 --color_correction none`, VAE tiling for batch 81 and 4K (only
the DiT matters here). Two tools:

- [`scripts/attn_probe.py`](../scripts/attn_probe.py) runs the CLI in-process with import-time
  patches (nothing in the checkout changes): every `FlashAttentionVarlen` call is recorded (number
  of windows, length multiset, uniform or not, which kernel really ran, GPU time from CUDA events),
  and so is every DiT forward. One JSON per run, next to the bench log (`bench.py run --wrap`).
- Clean timing runs through [`bench.py`](../scripts/bench.py), without the probe.

### What each attention call contains (`--attention_mode sageattn_3`, instrumented)

| Run | Token grid `t×h×w` | Windows per call (regular / shifted) | Sequence lengths (min–max, distinct) | Uniform calls | Kernel that ran | Attention GPU time, steady | DiT GPU time | Attention share of DiT time | Attention share, FLOP model |
|---|---|---|---|---|---|---|---|---|---|
| 1080p, batch 5 | 2×68×120 | 50 / 60 | 71–463 (12) | 0 / 36 | `sageattn_varlen` 36 / 36 | 0.20 s | 4.9 s | 4.1% | 2.1% |
| 1080p, batch 81 | 21×68×120 | 100 / 120 | 97–2488 (21) | 0 / 36 | `sageattn_varlen` 36 / 36 | 4.23 s | 49.9 s | 8.5% | 9.9% |
| 4K, batch 5 | 2×135×240 | 162 / 200 | 135–463 (10) | 0 / 36 | `sageattn_varlen` 36 / 36 | 0.91 s | 19.7 s | 4.6% | 2.3% |
| 4K, batch 45 | 12×135×240 | 324 / 500 | 135–1273 (28) | 0 / 36 | `sageattn_varlen` 36 / 36 | 9.22 s | 115.0 s | 8.0% | 5.8% |

Lengths include the 58 text tokens. One DiT forward per batch (one sampling step, CFG off).
"Steady" charges every call the median time of its layout: the very first attention call of a
run takes 0.3–0.4 s more (Triton JIT). 4K batch 81 was not run: the DiT would need about 113 GiB
(see [VRAM](#vram-seen-on-the-way)).

Layouts (length × number of windows), 1080p:

| Run | Regular layers | Shifted layers |
|---|---|---|
| batch 5 | 154×2 238×8 274×8 463×32 | 71×2 84×2 85×6 149×2 240×2 247×6 253×8 448×8 463×24 |
| batch 81 | 346×1 598×4 634×3 706×4 1138×12 1273×16 1354×12 2488×48 | 97×1 136×4 139×3 214×3 220×9 331×1 604×4 625×3 643×4 1150×3 1192×9 1228×16 1273×12 2398×12 2488×36 |

Replaying `window.py`'s two window functions on the token grid reproduces all four runs'
layouts exactly, so other configurations can be predicted without a GPU:

| Output, batch | Windows (regular / shifted) | Max video tokens | Distinct lengths (regular / shifted) | Attention share, FLOP model |
|---|---|---|---|---|
| 720p, 5 | 18 / 32 | 405 | 2 / 6 | 2.1% |
| 720p, 81 | 36 / 64 | 2430 | 4 / 12 | 9.5% |
| 1080p, 21 | 75 / 120 | 810 | 4 / 15 | 3.6% |
| 1080p, 121 | 100 / 150 | 3240 | 8 / 24 | 12.7% |
| 4K, 81 | 324 / 400 | 2430 | 4 / 18 | 10.5% |
| 4K, 121 | 324 / 500 | 3240 | 4 / 27 | 13.5% |

### Where the DiT time goes

`torch.profiler` on one steady DiT forward (1080p, batch 5, `sageattn_2`, `ATTN_PROBE_PROFILE=1`),
by self GPU time (4.13 s of kernels):

| Kernels | Share |
|---|---|
| GEMMs (`addmm`/`mm`, CUTLASS bf16 `sm80` kernels) | 45% |
| Element-wise and data movement: `add`, `copy_` (dtype casts), `cat`, `gather`, `mul`, `div`, `pow`, `mean`, … | ≈ 53% |
| Attention: `_attn_fwd` (SA2 Triton) 1.8% + INT8 quantization 0.4% | 2.2% |

Several small ops run once per window (1980 windows per forward here, hence kernels called
1980, 5940 or 11 880 times for ≈ 1 µs each), so part of the forward is launch-bound. q/k/v also
reach the attention wrapper as **float32** and are cast to bf16 on every call.

Half of the GPU time is memory-bound glue around the GEMMs: that is where DiT speed-ups are to be
found (the kind of code `torch.compile`, i.e. `--compile_dit`, can fuse: −26 to −32% DiT time,
measured in [vram.md](vram.md#torchcompile)), not in the attention kernel.

### Backend comparison (clean runs)

Same inputs, no probe. Batch 5 runs process 2 batches and the table shows the second (the first
includes warmup); batch 81 runs are a single batch, whose warmup is < 1 s of 54 s. The GPU drifts
by up to 9% between back-to-back runs (power cap, see [benchmarking.md](benchmarking.md#caveats)),
so the last column divides by VAE encode time of the same run, which attention can't affect.

| Output, batch | `--attention_mode` | DiT inference | VAE encode | DiT / encode | DiT torch peak |
|---|---|---|---|---|---|
| 1080p, 5 | `sdpa` | 4.58 s · rerun 4.98 s | 7.59 s · 8.27 s | 0.603 · 0.602 | 18.1 GiB |
| 1080p, 5 | `flash_attn_2` | 4.65 s | 7.74 s | 0.601 | 18.0 GiB |
| 1080p, 5 | `sageattn_2` | 5.05 s · rerun 5.03 s | 8.29 s · 8.43 s | 0.609 · 0.597 | 18.4 GiB |
| 1080p, 5 | `sageattn_3` | 5.18 s | 8.59 s | 0.603 | 18.4 GiB |
| 4K, 5 | `sdpa` | 19.72 s | 36.44 s | 0.541 | 24.7 GiB |
| 4K, 5 | `flash_attn_2` | 19.93 s | 36.48 s | 0.546 | 24.3 GiB |
| 4K, 5 | `sageattn_2` | 20.19 s | 36.88 s | 0.547 | 25.6 GiB |
| 4K, 5 | `sageattn_3` | 19.99 s | 36.62 s | 0.546 | 25.6 GiB |
| 1080p, 81 | `sdpa` | 53.93 s | 67.12 s | 0.803 | 38.0 GiB |
| 1080p, 81 | `flash_attn_2` | 53.58 s | 67.94 s | 0.789 | 37.0 GiB |
| 1080p, 81 | `sageattn_2` | 54.30 s | 68.67 s | 0.791 | 40.0 GiB |

**The four backends give the same DiT time within ±1.5%, below the run-to-run drift.** The rerun
pair (same settings, minutes apart) moved `sdpa` by 9% and `sageattn_2` by 0.4%. SageAttention
costs memory instead: +1.3 GiB at 4K batch 5 and +3 GiB at 1080p batch 81 over `flash_attn_2`.

One call through SeedVR2's own wrapper (`FlashAttentionVarlen`, fp32 q/k/v as the DiT passes
them, regular-layer layouts, ms per call; the fp32 → bf16 casts alone take 1.0 / 9.7 / 4.1 ms):

| Layout | `flash_attn_2` | `sageattn_2` | `sageattn_3` | `sdpa` |
|---|---|---|---|---|
| 1080p batch 5 | **3.2** | 4.7 | 4.8 | 5.1 |
| 1080p batch 81 | **92.6** | 106.7 | 106.4 | 93.8 |
| 4K batch 5 | **24.8** | 30.9 | 30.6 | 28.2 |

`flash_attn_2` is the fastest per call, but over 36 layers its lead over `sageattn_2` is worth
≈ 1% of DiT time, which matches the end-to-end runs. Inside the pipeline the probe measured the
same `sageattn_3` calls at 5.6 / 117 / 25 ms: the same order, shifted by the clocks under
sustained load and by the probe's synchronizations.

<details><summary>bench.py table of the runs</summary>

| Run | Key args | Frames | Batches | Encode | DiT | DiT inference | Decode | Post | Total s | FPS | NVML peak | Max RSS | OOM / retries | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe-1080-bs5-sa3 | 7b_fp16, res 1080, bs 5, sageattn_3, color_correction=none, wrap=attn_probe.py | 5 @ 1920x1080 | 1 | 3.98 s · 14.6 / 17.4 / 18.2 | 10.03 s · 18.4 / 18.6 / 19.4 | 5.20 s (1×, first 5.20) | 8.41 s · 21.7 / 24.6 / 25.3 | 0.01 s · 0.1 / 24.6 / 25.3 | 23.35 | 0.21 | 25.3 | 18.0 | 0 / 0 | ok |
| probe-1080-bs81-sa3 | 7b_fp16, res 1080, bs 81, sageattn_3, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128, wrap=attn_probe.py | 81 @ 1920x1080 | 1 | 62.53 s · 12.8 / 15.9 / 16.6 | 55.14 s · 40.0 / 42.7 / 43.4 | 50.12 s (1×, first 50.12) | 155.30 s · 18.4 / 42.7 / 43.4 | 0.34 s · 1.0 / 42.7 / 43.4 | 275.86 | 0.29 | 43.4 | 20.6 | 0 / 0 | ok |
| probe-2160-bs5-sa3 | 7b_fp16, res 2160, bs 5, sageattn_3, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128, wrap=attn_probe.py | 5 @ 3840x2160 | 1 | 17.27 s · 8.8 / 10.9 / 11.7 | 24.71 s · 25.6 / 26.3 / 27.1 | 19.99 s (1×, first 19.99) | 41.09 s · 11.3 / 26.3 / 27.1 | 0.03 s · 0.3 / 26.3 / 27.1 | 84.65 | 0.06 | 27.1 | 18.0 | 0 / 0 | ok |
| probe-2160-bs45-sa3 | 7b_fp16, res 2160, bs 45, sageattn_3, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128, wrap=attn_probe.py | 45 @ 3840x2160 | 1 | 152.14 s · 13.6 / 16.3 / 17.1 | 120.00 s · 71.5 / 76.6 / 77.4 | 115.09 s (1×, first 115.09) | 357.94 s · 18.2 / 76.6 / 77.4 | 0.75 s · 2.1 / 76.6 / 77.4 | 634.71 | 0.07 | 77.4 | 19.4 | 0 / 0 | ok |
| probe-1080-bs5-cap10-sa2-prof | 7b_fp16, res 1080, bs 5, sageattn_2, color_correction=none, wrap=attn_probe.py | 10 @ 1920x1080 | 2 | 7.63 s · 14.6 / 17.6 / 18.4 | 32.53 s · 18.4 / 18.6 / 19.4 | 27.68 s (2×, first 5.00) | 16.34 s · 21.7 / 24.6 / 25.4 | 0.01 s · 0.1 / 24.6 / 25.4 | 57.53 | 0.17 | 25.4 | 20.2 | 0 / 0 | ok |
| clean-1080-bs5-cap10-sdpa | 7b_fp16, res 1080, bs 5, sdpa, color_correction=none | 10 @ 1920x1080 | 2 | 7.59 s · 14.6 / 17.6 / 18.4 | 14.02 s · 18.1 / 18.6 / 19.4 | 9.30 s (2×, first 4.72) | 16.54 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 39.21 | 0.26 | 25.3 | 18.1 | 0 / 0 | ok |
| clean-1080-bs5-cap10-flash_attn_2 | 7b_fp16, res 1080, bs 5, flash_attn_2, color_correction=none | 10 @ 1920x1080 | 2 | 7.74 s · 14.6 / 17.6 / 18.4 | 14.22 s · 18.0 / 18.6 / 19.4 | 9.46 s (2×, first 4.81) | 17.27 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 40.35 | 0.25 | 25.3 | 18.1 | 0 / 0 | ok |
| clean-1080-bs5-cap10-sageattn_2 | 7b_fp16, res 1080, bs 5, sageattn_2, color_correction=none | 10 @ 1920x1080 | 2 | 8.29 s · 14.6 / 17.6 / 18.4 | 15.39 s · 18.4 / 18.7 / 19.5 | 10.55 s (2×, first 5.50) | 17.98 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 42.76 | 0.23 | 25.3 | 18.1 | 0 / 0 | ok |
| clean-1080-bs5-cap10-sageattn_3 | 7b_fp16, res 1080, bs 5, sageattn_3, color_correction=none | 10 @ 1920x1080 | 2 | 8.59 s · 14.6 / 17.6 / 18.4 | 15.53 s · 18.4 / 18.7 / 19.5 | 10.76 s (2×, first 5.58) | 18.46 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 43.68 | 0.23 | 25.3 | 18.1 | 0 / 0 | ok |
| clean-2160-bs5-cap10-sdpa | 7b_fp16, res 2160, bs 5, sdpa, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 10 @ 3840x2160 | 2 | 36.44 s · 8.8 / 11.0 / 11.8 | 44.59 s · 24.7 / 26.3 / 27.1 | 39.83 s (2×, first 20.11) | 85.40 s · 11.3 / 26.3 / 27.1 | 0.17 s · 0.3 / 26.3 / 27.1 | 168.42 | 0.06 | 27.1 | 18.1 | 0 / 0 | ok |
| clean-2160-bs5-cap10-flash_attn_2 | 7b_fp16, res 2160, bs 5, flash_attn_2, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 10 @ 3840x2160 | 2 | 36.48 s · 8.8 / 11.0 / 11.8 | 45.08 s · 24.3 / 26.3 / 27.1 | 40.33 s (2×, first 20.40) | 86.81 s · 11.3 / 26.3 / 27.1 | 0.17 s · 0.3 / 26.3 / 27.1 | 170.42 | 0.06 | 27.1 | 18.1 | 0 / 0 | ok |
| clean-2160-bs5-cap10-sageattn_2 | 7b_fp16, res 2160, bs 5, sageattn_2, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 10 @ 3840x2160 | 2 | 36.88 s · 8.8 / 11.0 / 11.8 | 45.86 s · 25.6 / 26.3 / 27.1 | 41.11 s (2×, first 20.92) | 86.22 s · 11.3 / 26.3 / 27.1 | 0.18 s · 0.3 / 26.3 / 27.1 | 170.95 | 0.06 | 27.1 | 18.1 | 0 / 0 | ok |
| clean-2160-bs5-cap10-sageattn_3 | 7b_fp16, res 2160, bs 5, sageattn_3, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 10 @ 3840x2160 | 2 | 36.62 s · 8.8 / 11.0 / 11.8 | 45.41 s · 25.6 / 26.3 / 27.1 | 40.68 s (2×, first 20.69) | 84.34 s · 11.3 / 26.3 / 27.1 | 0.06 s · 0.3 / 26.3 / 27.1 | 168.30 | 0.06 | 27.1 | 18.2 | 0 / 0 | ok |
| clean-1080-bs81-sdpa | 7b_fp16, res 1080, bs 81, sdpa, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 81 @ 1920x1080 | 1 | 67.12 s · 12.8 / 15.9 / 16.6 | 58.66 s · 38.0 / 42.7 / 43.4 | 53.93 s (1×, first 53.93) | 163.09 s · 18.4 / 42.7 / 43.4 | 0.36 s · 1.0 / 42.7 / 43.4 | 291.74 | 0.28 | 43.4 | 20.6 | 0 / 0 | ok |
| clean-1080-bs81-flash_attn_2 | 7b_fp16, res 1080, bs 81, flash_attn_2, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 81 @ 1920x1080 | 1 | 67.94 s · 12.8 / 15.9 / 16.6 | 58.27 s · 37.0 / 42.7 / 43.4 | 53.58 s (1×, first 53.58) | 164.08 s · 18.4 / 42.7 / 43.4 | 0.36 s · 1.0 / 42.7 / 43.4 | 293.21 | 0.28 | 43.4 | 20.6 | 0 / 0 | ok |
| clean-1080-bs81-sageattn_2 | 7b_fp16, res 1080, bs 81, sageattn_2, color_correction=none, vae_encode_tiled=True, vae_decode_tiled=True, enc tile 1024/128, dec tile 1024/128 | 81 @ 1920x1080 | 1 | 68.67 s · 12.8 / 15.9 / 16.6 | 59.22 s · 40.0 / 42.7 / 43.4 | 54.30 s (1×, first 54.30) | 164.66 s · 18.4 / 42.7 / 43.4 | 0.34 s · 1.0 / 42.7 / 43.4 | 295.44 | 0.27 | 43.4 | 20.6 | 0 / 0 | ok |
| clean-1080-bs5-cap10-sageattn_2-r2 | 7b_fp16, res 1080, bs 5, sageattn_2, color_correction=none | 10 @ 1920x1080 | 2 | 8.43 s · 14.6 / 17.6 / 18.4 | 15.18 s · 18.4 / 18.7 / 19.5 | 10.45 s (2×, first 5.42) | 17.78 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 42.47 | 0.24 | 25.3 | 18.1 | 0 / 0 | ok |
| clean-1080-bs5-cap10-sdpa-r2 | 7b_fp16, res 1080, bs 5, sdpa, color_correction=none | 10 @ 1920x1080 | 2 | 8.27 s · 14.6 / 17.6 / 18.4 | 14.90 s · 18.1 / 18.6 / 19.4 | 10.14 s (2×, first 5.16) | 17.66 s · 21.7 / 24.6 / 25.3 | 0.02 s · 0.1 / 24.6 / 25.3 | 41.92 | 0.24 | 25.3 | 18.1 | 0 / 0 | ok |

Phase cells: time · torch peak allocated / torch reserved at phase end / NVML device peak (GiB).
The probe runs' DiT times include the probe's synchronizations; `-prof` includes the profiler.

</details>

### Output differences between backends

Measured on lossless 16-bit RGB masters written by [`ffv1_out.py`](../scripts/ffv1_out.py) from
the float frames, so the CLI's lossy or truncated output doesn't blur the comparison
([output.md](output.md#do-attention-backends-change-the-output)). 1080p, batch 5, 21 frames,
`--color_correction lab`, seed 42:

| Pair | PSNR (RGB) | Worst frame | Max \|difference\| | Samples that differ | VMAF fidelity |
|---|---|---|---|---|---|
| `flash_attn_2` vs its rerun | ∞ (bit-identical) | ∞ | 0 | 0% | 100 |
| `sageattn_2` vs `sageattn_3` | ∞ (bit-identical) | ∞ | 0 | 0% | – |
| `flash_attn_2` vs `sdpa` | 56.20 dB | 55.33 dB | 10.0 levels | 34% | 100 |
| `flash_attn_2` vs `sageattn_2` | 55.08 dB | 54.21 dB | 14.4 levels | 40% | 100 |
| `sdpa` vs `sageattn_2` | 55.02 dB | 54.10 dB | 12.5 levels | 41% | – |

- `flash_attn_2` is deterministic, and `sageattn_3` produces `sageattn_2`'s output bit for bit,
  as the fallback above predicts.
- The backends change the output by 55–56 dB PSNR: small, measurable, invisible to VMAF
  (fidelity 100 at every percentile, CAMBI adds ≤ 0.003). `sdpa` and `flash_attn_2` are the
  closest pair.
- Compared through the CLI's mp4s instead, PSNR drops to 47 dB: the mp4 is itself 44.6 dB from
  the frames it encodes, so x264 dominates the difference.

### Is the length-grouping patch worth it?

Micro-benchmark of one attention call on the real layouts (one-off script, not kept: regular
layer, bf16 inputs, random data, ms per call). "Grouped" = gather the windows of each length, one batched call per length,
scatter back, as the patch would.

| Layout | `flash_attn_varlen` (FA2) | `sageattn_varlen` (SA2 Triton) | Grouped SA2 CUDA `sageattn()` | Grouped SA3 | Grouped SDPA | Gather + scatter alone |
|---|---|---|---|---|---|---|
| 1080p batch 5 (50 windows, 19k tokens) | **2.0** | 3.3 | 4.2 | 7.3 | 4.0 | 1.9 |
| 1080p batch 81 (100 windows, 177k tokens) | **64.8** | 78.0 | 79.6 | 120.9 | 83.7 | 19.8 |
| 4K batch 5 (162 windows, 74k tokens) | **11.0** | 19.2 | 25.5 | 45.7 | 19.5 | 8.3 |
| 4K batch 45 (324 windows, 408k tokens) | **101.2** | 139.2 | 159.5 | 267.1 | 148.2 | 45.1 |

Cosine against FA2: SA2 varlen 0.99991, grouped SA2 CUDA 0.9993, grouped SA3 0.973–0.980.

- FlashAttention 2's varlen kernel is the fastest option at every SeedVR2 length: 1.2–1.7×
  faster than SA2's Triton kernel.
- Even if the gather/scatter were free (it could be folded into the window permutation the
  block already does), grouped SA2 CUDA would beat FA2 only at 1080p batch 81, by ≈ 8%, and lose
  in the three other layouts. SA3 stays slower than FA2 at every length here, and is the least
  accurate.
- Attention is 4–9% of DiT time. A kernel 8% faster than FA2 would save < 1% of the DiT.

**Decision: don't write the patch.** Use `flash_attn_2`: as fast as anything else end to end,
fastest per call, exact, and the smallest DiT memory peak. `sageattn_3` and `sageattn_2` bring
nothing in SeedVR2 on this GPU.

### VRAM seen on the way

DiT phase, torch peak allocated (15.9 GiB of it is the fp16 weights), `sageattn_3` probe runs:

| Run | DiT tokens | Peak allocated | Above the weights, per token |
|---|---|---|---|
| 1080p, batch 5 | 16 320 | 18.4 GiB | ≈ 160 KiB |
| 4K, batch 5 | 64 800 | 25.6 GiB | ≈ 157 KiB |
| 1080p, batch 81 | 171 360 | 40.0 GiB | ≈ 148 KiB |
| 4K, batch 45 | 388 800 | 71.5 GiB | ≈ 150 KiB |

The DiT's activation peak grows linearly with the token count, ≈ 150 KiB per token with
SageAttention, ≈ 130 KiB with `flash_attn_2` (37.0 GiB at 1080p batch 81), whatever the
resolution or window length (fitted over batch 1–81 in [vram.md](vram.md#the-dit): 16.05 GiB +
128.5 KiB per token). 4K batch 81 (680 400 tokens) would need ≈ 100 GiB with
`flash_attn_2` and ≈ 113 GiB with SageAttention, so it was not run: it can't fit in 96 GB with
the weights on the GPU. Swapping all 36 blocks out would not be enough with SageAttention; with
`flash_attn_2`, [vram.md](vram.md#blockswap)'s all-swapped model (1.3 GiB + 128.5 KiB per token)
predicts ≈ 85 GiB, not tested. With the weights on the GPU, the largest 4K batch on a 96 GB card
is about 61 frames (16 latent frames) with SageAttention, 73 (19) with `flash_attn_2`. NVML peaks with VAE
tiling: 43.4 GiB at 1080p batch 81, 77.4 GiB at 4K batch 45, both set by the DiT phase.

## Reproduce

```bash
# instrumented run: <runs-dir>/<name>.attn.json next to the log
python3 scripts/bench.py run probe-1080-bs5 --wrap scripts/attn_probe.py ... -- <input> ... \
  --resolution 1080 --batch_size 5 --load_cap 5 --attention_mode sageattn_3
python3 scripts/attn_probe.py --summary runs/*.attn.json      # the tables above
ATTN_PROBE_PROFILE=1 python3 scripts/bench.py run ... --wrap scripts/attn_probe.py ...   # + .prof.txt
```
