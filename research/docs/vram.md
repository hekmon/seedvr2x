# VRAM per phase: models of the VAE and the DiT, and what each memory option costs

> Status: **VAE and DiT analysed from the code** (SeedVR2 `4490bd1`, `ema_vae_fp16`, shared by
> the 3B and 7B models) and **measured** on the reference stack (RTX PRO 6000 96 GB, torch
> 2.14.1, cuDNN 9.24). The VAE, not the DiT, sets the VRAM peak of untiled runs from 1080p batch
> 5 up: decoding needs ≈ 16 GiB per output megapixel as soon as the batch is ≥ 9 frames, whatever
> the batch size. 4K needs ≈ 134 GiB untiled at batch ≥ 9. Tiling makes the peak depend on the
> tile size only. The user's 4K OOM came from `--compile_vae`. The DiT needs its weights plus
> 128.5 KiB per token; BlockSwap, fp8 and GGUF weights bring the 7B under 8 GB. On a full card
> the default `cudaMallocAsync` allocator (or `expandable_segments`) trims its cache and needs
> little more than the torch peak; `backend:native` fragments and fails. A
> [recipe per card size](#recipe-per-card-size-validated), validated on emulated 8–48 GB cards,
> fits each phase's torch peak under N − 2 GiB.

## The picture per phase

The phases run one after the other, and each has its own peak. With the CLI's default
flow, the DiT is not on the GPU during the VAE phases. It is loaded at the start of phase 2,
copied to the CPU at its end and deleted. The VAE weights (≈ 0.5 GiB) stay throughout.

| Phase | What sets the peak | Torch peak allocated |
|---|---|---|
| 1. VAE encode | Input resized to the **output** resolution, then encoded | ≈ 1 + 8.8 GiB × output Mpx (batch ≥ 9), see below |
| 2. DiT | Weights + activations ∝ tokens | 7B fp16: 16.05 GiB + 128.5 KiB/token (`flash_attn_2`), ≈ 150 KiB/token with SageAttention ([attention.md](attention.md#vram-seen-on-the-way)); see [The DiT](#the-dit) |
| 3. VAE decode | Activations of the first temporal slice + causal caches | ≈ 1 + 16.1 GiB × output Mpx (batch ≥ 9), see below |
| 4. Post-processing | Output frames are on the CPU | ≤ 1 GiB at 1080p, 2 GiB at 4K batch 45 |

DiT tokens = latent frames × ⌈H/16⌉ × ⌈W/16⌉, with latent frames = 1 + (batch − 1)/4.
Example: 1080p batch 9 = 3 × 68 × 120 = 24 480 tokens.

Measured (7B fp16, `flash_attn_2`, no tiling, one batch, `--skip_first_frames 48`):

| Output, batch | Encode | DiT | Decode | NVML run peak |
|---|---|---|---|---|
| 720p, 5 | 7.6 | 16.8 | 10.1 | 17.9 |
| 720p, 9 | 10.4 | 17.3 | 16.2 | 20.5 |
| 1080p, 1 | 5.8 | 16.9 | 7.6 | 18.0 |
| 1080p, 5 | 14.6 | 18.0 | 21.7 | 25.3 |
| 1080p, 9 | 19.7 | 19.1 | 34.9 | 40.0 |
| 1080p, 21 | 20.0 | 22.1 | 35.0 | 40.3 |
| 1440p, 5 | 24.9 | 19.6 | 37.6 | 42.4 |
| 1440p, 9 | 34.0 | 21.5 | 60.7 | 75.4 |
| 4K, 5 | 54.7 | 24.3 | 83.5 | 92.7 |
| 4K, 9 | 75.0 | 28.5 | **OOM** (> 93.7) | 95.0 |

Torch peak allocated per phase, GiB. NVML peak = device memory, including the allocator's
cache and the CUDA context.

- From 1080p batch 5 upwards, **decode is the largest phase**, 1.2–3.4× the DiT. Below that
  (720p, batch 1) the DiT's 15.9 GiB of weights dominate. Above it, the DiT only dominates once
  the VAE is tiled (e.g. 1080p batch 81: DiT 37 GiB, tiled decode
  18 GiB).
- **NVML ≈ 1.1–1.25 × the torch peak** with the default `cudaMallocAsync` on this 96 GB card:
  reserved minus allocated was 4.4 GiB at 1080p batch 9, 8.5 GiB at 4K batch 5 and 14 GiB at
  1440p batch 9, plus 0.8 GiB of context. That is cache nothing asked back: on a full card the
  pool trims itself down to the torch peak ([The allocator](#the-allocator)). Budget a smaller
  GPU on the torch peak plus the context and ≈ 0.5 GiB.

## How the VAE processes a batch

Code: `src/models/video_vae_v3/modules/attn_video_vae.py` (`VideoAutoencoderKL`),
`causal_inflation_lib.py` (`InflatedCausalConv3d`), `configs_7b/main.yaml` (`vae:` section),
`src/core/generation_phases.py`, `src/core/infer.py`.

- **One call per batch.** Phase 1 encodes each batch of `--batch_size` frames, already resized to
  the output resolution (padded to a multiple of 16). Phase 3 decodes each batch's latents. The
  VAE computes in bf16, and decoded frames go to the CPU (`--tensor_offload_device`, default
  `cpu`) batch by batch.
- **Architecture:** 8× spatial, 4× temporal, 16 latent channels. Block widths 128/256/512/512,
  so the full-resolution layers have 128 channels (256 at half resolution). The mid block has a
  per-frame spatial self-attention over the whole (latent) frame. GroupNorm statistics are taken
  per frame over the whole frame.
- **Temporal "causal slicing"** (`vae.slicing: split_size 4, memory_device same`): a batch is cut
  into slices that go through the network one after the other. Each causal 3D conv keeps the
  last 2 frames of its input (`kernel − stride`) as a cache for the next slice.
  - Encode: first slice = frames 0–4 (5 frames), then 4 frames per slice. It only kicks in when
    `frames − 1 > 4`.
  - Decode: first slice = latents 0–1 (→ 5 frames), then 1 latent (→ 4 frames) per slice. It
    only kicks in when `latents − 1 > 1`.
  - So **batch 5 runs in one pass without caches, and batch ≥ 9 runs a 5-frame first slice
    while the caches fill up.** The caches stay on the GPU (`memory_device: same`) until the
    end of the batch. Later slices are smaller, so the peak is in the first slice, and it doesn't
    grow with the batch size.
- **"Memory limits for causal convolutions"** (`vae.memory_limit: conv_max_mem 0.5,
  norm_max_mem 0.5`, in GiB): a conv whose padded input exceeds 0.5 GiB is split along H, then
  W, recursively, and the outputs are concatenated. A GroupNorm input above 0.5 GiB is
  normalized in 4 channel chunks. This bounds the conv workspace, not the activations: the full
  input and output of every layer still exist for the whole slice.
- **OOM retry** (`retry_on_oom`, `memory_manager.py`): every conv, norm and upsample call is
  wrapped. On OOM it logs "OOM during …", empties the caches, waits 0.5 s and retries once.
  That works when the OOM came from cached-but-free memory. It can't fix a real shortfall.
- **Conv3d workaround** (`compatibility.py`): for torch ≥ 2.9 with cuDNN ≥ 9.10.2, every causal
  conv calls `torch.cudnn_convolution` directly to dodge a "3× memory" bug in fp16/bf16 Conv3d
  dispatch. It is active on the reference stack.
- **Tiling** (`--vae_{encode,decode}_tiled`, `_tile_size`, `_tile_overlap`, in output pixels,
  converted to latent pixels ÷ 8). It is **spatial only**: each tile runs through the full
  temporal slicing above with its own caches, which are freed between tiles. The tiles overlap,
  and the overlaps are blended with a cosine ramp on the interior edges. The blended result is
  accumulated on `--tensor_offload_device` (the CPU by default), so the GPU only holds one
  tile's work. The decoder's last tile row and column are smaller (the grid starts at 0 with
  stride `tile − overlap`). Tiling is skipped when the frame fits in one tile.

## Untiled VAE: the model

Measured with [`scripts/vae_probe.py`](../scripts/vae_probe.py), which records each VAE call,
slice and tile (torch peak, cache bytes, GPU time) without touching the checkout. P = output
megapixels after padding (1080p → 1088 × 1920 = 2.09 Mpx, 4K = 8.29 Mpx). "Peak" here = the
call's own peak minus what was allocated before it, VAE weights excluded.

| Output | P | Encode b1 | Encode b5 | Encode b≥9 | Decode b1 | Decode b5 | Decode b≥9 | Caches b≥9 (enc / dec) |
|---|---|---|---|---|---|---|---|---|
| 720p | 0.92 | – | 7.0 | 9.8 | – | 9.6 | 15.7 | 3.3 / 7.4 |
| 1080p | 2.09 | 5.3 | 14.0 | 19.0 | 7.1 | 21.2 | 34.3 | 7.6 / 16.8 |
| 1440p | 3.69 | – | 24.2 | 33.2 | – | 37.1 | 60.2 | 13.3 / 29.7 |
| 4K | 8.29 | – | 53.9 | 74.0 | – | 83.0 | OOM | 30.0 / – |

Linear fits (GiB, P in Mpx, residuals < 0.6 GiB):

| Call | Batch 1 | Batch 5 | Batch ≥ 9 | Of which causal caches |
|---|---|---|---|---|
| Encode | 2.55 × P | 0.9 + 6.4 × P | 1.2 + 8.8 × P | 3.62 × P (exact) |
| Decode | 3.4 × P | 0.4 + 9.95 × P | 0.8 + 16.1 × P | 8.05 × P (exact) |

- **Linear in pixels, flat in batch size from 9 up.** 1080p batch 9 / 13 / 21: decode
  34.34 / 34.39 / 34.48 GiB, encode 19.02 / 19.02 / 19.03 GiB. A longer batch only adds slices,
  not memory.
- **Decode ≈ 1.8× encode.** Batch 5 (no slicing) costs 9.95 GiB/Mpx to decode its 5 frames,
  ≈ 2 GiB per output Mpx and frame, i.e. ≈ 8 full-resolution 128-channel bf16 tensors. Batch ≥ 9
  costs 16.1 GiB/Mpx: the same 5-frame first slice plus up to 8 GiB/Mpx of caches.
- **Prediction holds at 4K.** The fit from 720p–1440p predicted 82.9 GiB for 4K batch-5 decode
  (measured 83.0) and 72.1 GiB for batch-9 encode (measured 74.0). 4K batch ≥ 9 decode needs
  ≈ 134 GiB: it can't fit on 96 GB untiled.
- **Time is linear in pixels × frames:** decode ≈ 0.83 s and encode ≈ 0.36 s per output Mpx
  and frame (720p–1440p, 0.80–0.90 and 0.34–0.42 at the extremes). That makes 1.7 s per 1080p
  frame to decode and 7.5 s per 4K frame, against 0.75 s and 3.4 s to encode. Batch 1 is slower
  per frame (2.3 s at 1080p).

### Keeping the causal caches on the CPU

`memory_device: cpu` is a config value the CLI doesn't expose. Set through `vae_probe.py`
(`VAE_PROBE_CACHE_DEVICE=cpu`), it brings batch ≥ 9 back to the batch-5 cost:

| Run | Encode peak / time | Decode peak / time |
|---|---|---|
| 1080p, 9, caches on GPU (default) | 19.0 GiB / 6.9 s | 34.3 GiB / 15.5 s |
| 1080p, 9, caches on CPU | 14.0 GiB / 9.3 s | 21.2 GiB / 21.0 s |
| 4K, 9, caches on GPU (default) | 74.0 GiB / 30.1 s | OOM |
| 4K, 9, caches on CPU | 54.1 GiB / 40.4 s | 83.0 GiB / 89.7 s |

−26% encode and −38% decode memory, for +35% VAE time (synchronous GPU→CPU copies of every
cache). At 4K, tiling does better on both counts (below).

## The 4K OOM, reproduced

The first real run was 1080p → 4K, batch 5, `--temporal_overlap 1 --prepend_frames 1`,
`sageattn_2`, compile, CPU offload, no tiling: 93 GB during encode (saved by an OOM retry),
then an OOM in decode. Reproduced on 10 frames (`--load_cap 10`):

| Run | Encode peak | Decode peak | Result |
|---|---|---|---|
| `vae-2160-bs5`: batch 5, eager | 54.7 GiB | 83.5 GiB (reserved 91.9, NVML 92.7) | ok, 2.3 GiB spare |
| `vae-2160-bs5-ov1-pp1`: + overlap 1, prepend 1 | 54.7 GiB | 83.5 GiB | ok (3 batches instead of 2) |
| `vae-2160-bs5-userflags`: + `sageattn_2 --compile_dit --compile_vae --dit_offload_device cpu --vae_offload_device cpu` (closest match to the description) | **93.6 GiB** first batch (OOM, retry ok), then 64.8 GiB | **OOM at 88.2 GiB** (pool 94.2 GiB) | OOM in decode batch 1 |
| `vae-2160-bs9`: batch 9, eager | 75.0 GiB | OOM at 93.7 GiB, in `Upsample3D.conv` → `InflatedCausalConv3d.pad_and_forward`, first slice | OOM |

- **`--compile_vae` is what breaks the user's run.** Eager 4K batch 5 fits, with 2.3 GiB to
  spare. The compiled VAE needs 39 GiB more on its first (compiling) encode call and 10 GiB more
  afterwards (64.8 vs 54.7 GiB). Its decode doesn't fit. `--temporal_overlap` and
  `--prepend_frames` change only the number of batches.
- Without compile, 4K untiled only works at batch ≤ 5 on 96 GB. Batch ≥ 9 needs ≈ 134 GiB.
- `--compile_vae`'s effect at 1080p was measured later: about twice the VAE memory for −16–19%
  VAE time, after minutes of compilation ([`torch.compile`](#torchcompile)).

## Tiling

4K, batch 9 (the sliced regime), encode and decode tiled with the same settings, one batch:

| Tile / overlap | Tiles | Encode peak | Decode peak | Decode caches | Encode time | Decode time | Tiled area |
|---|---|---|---|---|---|---|---|
| 512 / 128 | 60 | 3.5 | 5.9 | 2.1 | 35.2 s | 87.7 s | 1.69× |
| 768 / 128 | 24 | 7.1 | 11.1 | 4.8 | 32.2 s | 77.0 s | 1.37× |
| 1024 / 64 | 12 | 10.4 | 17.4 | 8.4 | 27.4 s | 66.7 s | 1.11× |
| 1024 / 128 | 15 | 10.4 | 17.4 | 8.4 | 30.7 s | 74.1 s | 1.27× |
| 1024 / 256 | 15 | 10.4 | 17.4 | 8.4 | 39.3 s | 93.9 s | 1.57× |
| 1536 / 128 | 6 | 21.4 | 38.7 | 19.0 | 29.3 s | 68.1 s | 1.13× |
| untiled | 1 | 74.0 | ≈ 134 (model) | – | 30.1 s | ≈ 62 s (model) | 1× |

Peaks in GiB above what was allocated before the call. "Tiled area" = sum of the tile areas /
frame area.

- **The tiled peak depends on the tile size only.** 1080p batch 9 gives exactly the same
  figures as 4K for 512 and 1024 tiles (5.89 / 17.45 GiB decode, 3.52 GiB encode), and the
  caches match the untiled per-Mpx rates on a T × T tile exactly. Fit for batch ≥ 9 (residuals
  < 0.6 GiB): **decode ≈ 1.6 + 15.6 × T² GiB, encode ≈ 1.7 + 8.4 × T² GiB**, T² in Mpx. For
  batch ≤ 5 use the batch-5 rates: 4K batch 5 with 1024 tiles decoded at 11.3 GiB
  ([attention.md](attention.md) runs).
- The result accumulates on the CPU, so resolution and batch length add nothing on the GPU. 4K
  batch 45 with 1024 tiles decoded at 18.2 GiB, against 18.0 for batch 9.
- **Overlap costs time, not memory.** Time follows the tiled area: decode time ÷ tiled area is
  52–60 s for these 9 frames, i.e. 0.70–0.81 s per Mpx and frame, no slower per pixel than
  untiled (0.83 at 1080p–1440p, 0.90 at 4K). Small tiles and large overlaps waste up to 70% more work. With the
  default 1024/128 the cost is +27% area. At 4K, 1024/64 is cheaper (12 tiles instead of 15,
  because the last column no longer needs a sliver tile).
- Encode needs about half the decode memory at the same tile size, so encode can usually stay
  untiled longer (or use larger tiles) than decode.

### Quality: tiled vs untiled (1080p, batch 9, PNG output)

Same seed and settings, `--output_format png`, compared with
[`scripts/frame_diff.py`](../scripts/frame_diff.py) to the untiled run. Two untiled runs are
bit-identical, so every difference comes from tiling. "Decode only" keeps the same latents
(pure decode effect). Encode tiling changes the latents the DiT sees.

| Tiling | PSNR vs untiled (worst frame) | Mean abs. diff (0–255) | Boundary / interior abs. diff | Gradient excess at boundaries / interior | Per-tile mean offset (min…max) |
|---|---|---|---|---|---|
| decode 512 / 64 | 38.3 dB (35.0) | 2.18 | 2.17 / 2.18 (x) | +0.05 / +0.03 | −2.1 … +0.7 |
| decode 512 / 128 | 38.3 dB (35.2) | 2.19 | 2.19 / 2.19 (x) | +0.04 / +0.03 | −2.2 … +0.5 |
| decode 1024 / 128 | 43.6 dB (40.9) | 1.12 | 1.23 / 1.12 (x) | +0.04 / +0.03 | −0.8 … +0.5 |
| encode 512 / 128 | 33.4 dB (31.2) | 3.98 | 4.01 / 3.98 (x) | 0.00 / −0.01 | −2.3 … +3.8 |
| both 512 / 128 | 35.3 dB (32.3) | 3.05 | 3.10 / 3.04 (x) | +0.03 / +0.01 | −1.6 … +1.4 |

Boundary = within 16 px of an interior tile edge. Gradient excess = mean |step between
neighbouring pixels| of the tiled output minus the untiled one, in 8-bit levels.

- **No seams:** the differences are not concentrated at the tile edges, and the tiled output is
  not sharper there. The cosine blending works.
- **Per-tile drift instead:** each tile is shifted by up to 2 levels (decode) or 4 levels
  (encode, through the DiT), uniformly across the tile. Both global operations of the VAE (the
  mid-block's whole-frame attention and the per-frame GroupNorm statistics) see only the tile.
  The tiles therefore differ in low frequencies, and the blending turns that into smooth
  gradients. This was not inspected visually. On flat areas (skies, anime backgrounds) it could
  show as faint patches: [quality.md](quality.md#vae-tiling-on-flat-areas) measures it there,
  and the default `lab` colour correction removes it.
- **Tile size matters, overlap barely does:** 512 → 1024 gains 5.3 dB, 64 → 128 overlap gains
  nothing. Use the largest tile that fits, and the smallest overlap that hides edges (64 here).
- Encode tiling hurts more than decode tiling (33 vs 38 dB at 512), and it is the cheaper phase
  in memory: tile the decode first.

## Other knobs

| Variant (1080p, batch 9) | Encode peak | Decode peak | Reserved after decode | NVML peak | Decode time |
|---|---|---|---|---|---|
| default (`cudaMallocAsync`, Conv3d workaround on), 3 runs | 19.7 | 34.9 | 39.2 | 40.0 | 14.1–16.3 s |
| Conv3d workaround forced off | 19.7 | 34.9 | 39.2 | 40.0 | 13.5 s |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 19.7 | 34.9 | 40.9 | 41.6 | 13.2 s |
| `PYTORCH_CUDA_ALLOC_CONF=backend:native` | 19.8 | 34.9 | 49.5 | 50.3 | 13.5 s |

- **The Conv3d workaround changes nothing on this stack:** identical memory, time within the
  run-to-run spread (decode 14.1–16.3 s for the untouched config over the session, and all four
  variants ran last, when the GPU was faster). The
  "3× memory" bug it works around isn't triggered with torch 2.14.1 / cuDNN 9.24. Toggled
  through `VAE_PROBE_CONV3D_WORKAROUND=0`.
- **Allocator:** the torch peak is the same, only the overhead changes. The native caching
  allocator without expandable segments holds 15 GiB more. At 4K batch 5, `expandable_segments`
  ended decode at 84.3 GiB reserved / 85.0 GiB NVML against 91.9 / 92.7 for `cudaMallocAsync`,
  with one failed 20 MiB mapping it recovered from. On a card with no spare memory the overhead
  mostly disappears: see [The allocator](#the-allocator).

## The DiT

Code: `src/core/generation_phases.py` (`upscale_all_batches`), `src/optimization/blockswap.py`,
`src/optimization/memory_manager.py` (`cleanup_dit`, `manage_model_device`, `clear_memory`),
`src/core/model_loader.py` (`materialize_model`), `src/optimization/compatibility.py`
(`CompatibleDiT`), `src/utils/model_registry.py`. Runs `dit-*`: 1080p, `flash_attn_2`,
`--skip_first_frames 48`, `--color_correction none`, VAE tiled 1024/64 from batch 45 up. Time =
"DiT inference" per batch; with two or more batches, the mean of the batches after the first.

### Weights and the per-token model

Phase 2 materializes the DiT straight from the safetensors file to the GPU (1.2 s for 15.4 GiB,
page cache warm), runs one forward per batch, then **copies the whole DiT to the CPU before
deleting it** (`cleanup_dit`: "releasing GPU memory"). That copy takes 3.5 s for 7B fp16 and
leaves ≈ 16 GiB in the process RSS until the end (max RSS 19 GiB for a 3-frame run). The VAE
(0.47 GiB) stays on the GPU throughout unless `--vae_offload_device` is set.

7B fp16, batch size sweep (`dit-1080-bs{1,5,9,21,45,81}`, 2–3 batches up to 21, one above):

| Batch | Latent frames | Tokens | DiT torch peak | Time per batch | Per token | Per frame | Throughput |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 8 160 | 16.94 GiB | 1.71 s | 0.210 ms | 1.71 s | 0.58 fps |
| 5 | 2 | 16 320 | 18.01 GiB | 3.68 s | 0.226 ms | 0.74 s | 1.36 fps |
| 9 | 3 | 24 480 | 19.08 GiB | 5.82 s | 0.238 ms | 0.65 s | 1.55 fps |
| 21 | 6 | 48 960 | 22.13 GiB | 11.29 s | 0.231 ms | 0.54 s | 1.86 fps |
| 45 | 12 | 97 920 | 28.15 GiB | 22.66 s | 0.231 ms | 0.50 s | 1.99 fps |
| 81 | 21 | 171 360 | 36.97 GiB | 39.59 s | 0.231 ms | 0.49 s | 2.05 fps |

- **Memory: peak = 16.05 GiB + 128.5 KiB × tokens** (least squares, residuals ≤ 0.11 GiB). The
  16.05 GiB are the DiT weights (15.35 GiB), the resident VAE (0.47) and ≈ 0.2 GiB of buffers.
  Per output Mpx (3 906 tokens per Mpx and latent frame): **0.48 GiB per Mpx and latent frame**,
  i.e. 1.0 GiB per latent frame at 1080p, 4.0 GiB at 4K. Window length doesn't matter: the
  same slope holds from batch 1 (1 latent frame per window) to 81 (6).
- **Time is linear in tokens too: ≈ 0.23 ms per token** (7B fp16, this GPU), from batch 5 to 81.
  The attention's quadratic part (window length 405 → 2430 tokens) doesn't show.
- **Throughput comes from the 4× temporal compression:** a batch of b frames has 1 + (b − 1)/4
  latent frames, so the cost per frame is ∝ (1 + (b − 1)/4) / b: 1 at batch 1, 0.4 at batch 5,
  0.29 at batch 21, 0.26 at batch 81. Batch 21 already gets 91% of the batch-81 throughput;
  batch 5 gets 66%, batch 1 only 28%. Batch size's effect on temporal consistency is measured in
  [quality.md](quality.md#batches-and-temporal-consistency).
- The batch-81 run of [attention.md](attention.md#backend-comparison-clean-runs) took 53.6 s
  against 39.6 s here, with VAE encode 41% slower too: GPU clocks differ a lot between sessions.
  Within this session, `dit-1080-bs45` and its repeat `-r2` (run 18 min later) differ by 0.3%,
  but `dit-1080-bs21-r2`, 56 min after `dit-1080-bs21-png`, was 14% slower (VAE encode +12%):
  compare runs made back to back, or normalize by encode time.

### BlockSwap

`--blocks_to_swap N` (0–36 for 7B, 0–32 for 3B, clamped) needs `--dit_offload_device cpu`
(validated, else `ValueError`). What it does (`blockswap.py`):

- At load, blocks 0 … N−1 stay on the offload device, the others and the I/O components
  (`vid_in`, `txt_in`, `emb_in`, `vid_out`: 0.16 GiB for 7B) go to the GPU.
- Each swapped block's `forward` is wrapped: `block.to(cuda, non_blocking=False)`, run the
  block, `block.to(cpu, non_blocking=False)`, then `clear_memory(force=False)` (empties the
  cache only when < 5% of VRAM is free). **Every forward moves each swapped block in and out,
  synchronously, from pageable memory: no prefetch, no overlap with compute.** Moving back
  re-copies weights that never changed.
- `--swap_io_components` does the same for the I/O components.
- The log's "BlockSwap overhead" / "Block swaps: avg … ms" time the whole wrapped forward,
  compute included: they are not the swap cost.

1080p batch 45, 7B fp16 (`dit-1080-bs45-swap*`):

| `--blocks_to_swap` | DiT torch peak | DiT NVML peak | Time per batch | Max RSS |
|---|---|---|---|---|
| 0 (no offload) | 28.15 GiB | 32.1 | 22.66 s | 19.3 GiB |
| 12 | 23.50 GiB | 27.4 | 23.46 s | 23.8 GiB |
| 24 | 18.43 GiB | 22.3 | 24.46 s | 28.8 GiB |
| 36 | 13.35 GiB | 16.9 | 25.34 s | 33.9 GiB |
| 0 + `--swap_io_components` | 27.99 GiB | 31.9 | 22.79 s | 19.2 GiB |
| 36 + `--swap_io_components` | 13.20 GiB | 16.7 | 26.10 s | 34.5 GiB |

- **Memory: −0.41 GiB per swapped block** (a 7B fp16 block is 0.42 GiB), linear. With all 36
  swapped, what is left is the activations, one block in flight, the I/O components and the
  VAE: peak ≈ 1.3 GiB + 128.5 KiB × tokens.
- **Time: + 0.074 s per block and per batch** at batch 45 and at batch 21 (`dit-1080-bs21-swap36`
  against `dit-1080-bs21-r2`, run back to back: 15.48 vs 12.86 s), 0.08–0.10 s at batch 5. The
  cost is per forward, not per token, so it weighs on small batches: 36 blocks add 12% at batch
  45, 20% at batch 21, 97% at batch 5. That is 0.84 GiB moved per block (in and out) at 9–12 GB/s; a
  standalone copy measured 25 GB/s host→device and 4 GB/s device→new pageable host memory
  (57 GB/s each way from pinned memory) on this PCIe 5.0 ×16 link. On a PCIe 4.0 card, or with
  slower RAM, expect up to twice the cost.
- **Smaller weights swap for less.** 1080p batch 5, all blocks swapped (`dit-1080-bs5-*`):

  | Model | Peak, no swap → all swapped | Per block | Time per batch, no swap → all swapped | Per block |
  |---|---|---|---|---|
  | 7B fp16 (36) | 18.01 → 3.22 GiB | 0.41 GiB | 3.68 → 7.23 s | +0.099 s |
  | 7B fp8 (36) | 10.55 → 3.14 GiB | 0.21 GiB | 3.98 → 5.19 s | +0.034 s |
  | 7B Q4_K_M (36) | 7.10 → 2.92 GiB | 0.12 GiB | 4.57 → 4.61 s | +0.001 s |
  | 3B fp16 (32) | 9.27 → 3.30 GiB | 0.19 GiB | 3.11 → 4.05 s | +0.029 s |

  The fp8 and 3B blocks cost 0.15–0.17 s per GiB moved, the 7B fp16 ones 0.24. The GGUF blocks
  cost nothing measurable end to end; the log even reports "Total VRAM saved: 5.10MB", since it
  only counts parameters, but the peak drops by 4.2 GiB.
- **The GGUF blocks really move; their moves just don't show in the batch time.**
  [`swap_probe.py`](../scripts/swap_probe.py) times every `.to()` of a swapped block,
  synchronized (1080p batch 5, 2 batches, run back to back; the GPU was ≈ 60% slower than for
  the table above):

  | Model | Block (params + buffers) | Largest tensor | In: CPU → GPU | Out: GPU → CPU | Moves per batch (36 blocks) | Block compute per batch |
  |---|---|---|---|---|---|---|
  | 7B fp16 | 0.423 GiB | 72 MiB | 24.7 ms, 18 GB/s | 95 ms, 4.8 GB/s | 4.3 s | 5.4 s |
  | 7B fp8 | 0.212 GiB | 36 MiB | 13.4 ms, 17 GB/s | 52 ms, 4.5 GB/s | 2.4 s | 5.3 s |
  | 7B Q4_K_M | 0.119 GiB | 20 MiB | 6.1 ms, 21 GB/s | 18.2 ms, 7.0 GB/s | 0.88 s | 7.1 s |

  - The quantized weights are buffers (`GGUFTensor`), moved with the block by
    `_GGUFQuantizedBase._apply`: 72 moves each way, none pinned. BlockSwap does swap them, and the
    memory saving is the weights leaving the GPU, as for fp16.
  - The way back (into fresh pageable memory) is the slow direction. Q4_K_M's tensors (≤ 20 MiB)
    come back at 7 GB/s against 4.5–4.8 GB/s for the 36–72 MiB fp8/fp16 ones, plausibly because
    glibc serves allocations under its 32 MiB mmap threshold from recycled heap pages.
  - Q4_K_M's compute is the slowest (per-layer dequantization: 7.1 s against 5.4 s per batch).
  - End to end, in the same session: Q4_K_M without swap 7.90 and 7.61 s per batch
    (`swapctl-1080-bs5-q4`, `-r2`), with 36 blocks swapped 7.63 s (`-swap36`). The 0.88 s of
    synchronous moves don't add up. The other models show the same gap: their end-to-end swap
    cost (table above) is ≈ 1 s per batch below their measured move time. The likely reason is
    the 600 W power cap this GPU always hits: compute is energy-bound, and the idle gaps of the
    copies let the following kernels run at higher clocks. **On a GPU that isn't power-limited,
    expect Q4_K_M swap 36 to cost up to its move time, ≈ 0.9 s per batch at this PCIe 5.0
    bandwidth** (≈ +20% at batch 5, +3% at batch 45); still the cheapest swap by far.
- `--swap_io_components` saves 0.16 GiB for ≈ 0.08 s per batch: not worth it. **Used without
  `--blocks_to_swap`, it leaks the whole DiT:** the 36 blocks (15.2 GiB) stay allocated after
  phase 2, so decode ran at 33.4 GiB instead of 18.2 (`dit-1080-bs45-swap0io`, NVML peak 40.2
  GiB instead of 32.1). `cleanup_dit` checks the first parameter's device, finds an I/O
  component on the CPU and skips the move.
- RAM: swapped blocks live in RSS (≈ +0.4 GiB per block, plus allocator churn): 34 GiB max RSS
  with 36 blocks against 19 GiB without.
- Phase 2 as a whole got *shorter* with swapping (25.4 s at 36 blocks vs 27.4 s) because the
  3.5 s end-of-phase copy of the GPU-resident DiT to the CPU shrinks with it.

### Offload devices

`dit-1080-bs45-{ditoff,vaeoff,tnone}`, against `dit-1080-bs45` (and `-r2`):

| Option | What it does in a single CLI run | DiT peak | Time cost |
|---|---|---|---|
| `--dit_offload_device cpu` | Weights materialized to the CPU, moved to the GPU at phase 2 (0.71 s), back at its end (3.65 s), then deleted. Required for BlockSwap | 28.15 GiB (same) | +0.7 s |
| `--vae_offload_device cpu` | VAE moved to the CPU after encode and decode, back to the GPU before decode | 27.68 GiB (−0.47) | +0.2 s |
| `--tensor_offload_device none` | Latents stay on the GPU between phases (tens of MB). Decoded frames still go to the CPU (`final_video`), tiled-decode accumulation stays on the GPU | 28.15 GiB (same), decode +0.1 GiB | none |

- The DiT is never on the GPU during the VAE phases by default (it is loaded at phase 2 and
  deleted after), so `--dit_offload_device` saves nothing on its own. It matters with
  `--cache_dit` (the DiT then stays on the offload device between runs, or chunks) and as the
  BlockSwap prerequisite.
- `--vae_offload_device cpu` is the only one that lowers the DiT peak: by the VAE's 0.47 GiB.
- `--cache_dit` / `--cache_vae` only act when the CLI processes several inputs or
  `--chunk_size` chunks in one process (`runner_cache`); they then default the offload device to
  `cpu`. `--chunk_size N` streams the input N frames at a time through all four phases: it bounds
  host RAM, not VRAM, and without `--cache_dit` it reloads the DiT for every chunk.
- `--uniform_batch_size` pads the last batch to the full batch size: more compute, same peak
  as a full batch.

### `torch.compile`

`--compile_dit` wraps the whole DiT in `torch.compile` (after BlockSwap, if any); the first
forward compiles. `--compile_vae` compiles the VAE encoder and decoder submodules.

| Run (default `--compile_mode`) | First batch | Steady batch | Eager steady batch | Peak (torch), eager in () |
|---|---|---|---|---|
| `dit-1080-bs5-compile` (3 batches) | 35.0 s | 2.74 s | 3.68 s (−26%) | DiT 18.11 (18.01) |
| `dit-1080-bs21-compile` | 56.3 s | 7.93 s | 11.29 s (−30%) | DiT 22.77 (22.13) |
| `dit-1080-bs21-7bfp8-compile` | 19.5 s | 7.80 s | 10.95 s (−29%) | DiT 15.31 (14.67) |
| `dit-1080-bs21-compile-swap36` | 24.6 s | 16.02 s | 15.48 s (+3%, but the eager run was ≈ 10% slower GPU-wide) | DiT 7.98 (7.34) |
| `cap32-1080-fp16-bs45-t1280-x2-compile` (32 GB card, 2 batches) | 84.5 s | 22.01 s | 32.28 s (−32%) | DiT 29.56 (28.15) |
| `vae-1080-bs9-compilevae` | encode 108.5 s | encode 4.79 s, decode 9.30 s | 5.7 s, 11.4 s (−16%, −19%) | encode 33.8 (19.7), decode 68.6 (34.9) |

- **`--compile_dit` is the one big speed-up: −26 to −32% DiT time** for 10–50 s of compilation
  per process (less when inductor's on-disk cache already holds kernels from an earlier run;
  every new shape, such as a shorter last batch, recompiles). Its memory grows with the
  tokens: +0.1 GiB at 1080p batch 5, +0.6 at batch 21, +1.4 at batch 45. At 1080p batch 21 it
  pays for itself after ≈ 15 batches on a cold cache, at batch 45 after ≈ 5. It fuses the
  element-wise glue that is half the DiT time ([attention.md](attention.md#where-the-dit-time-goes)).
- **It fits the 32 GB recipe at 1080p** (no swap, batch 45, run back to back with the eager
  `cap32-1080-fp16-bs45-t1280-x2`): DiT 29.56 GiB against a room of 30.56, still under N − 2,
  with no allocator trim (48 for the eager run). Budget the extra 1.4 GiB before adding it to a
  recipe that runs closer to its limit.
- **Compile and BlockSwap don't combine:** with all 36 blocks swapped, the compiled DiT is no
  faster than the eager one (16.0 vs 15.5 s per batch, ≈ 14 s once the eager run is normalized
  by its VAE encode time). Each swapped block's wrapper calls
  `@torch._dynamo.disable` timing helpers and moves the block's weights inside `forward`, so
  the graph breaks at every swapped block.
- **`--compile_vae` costs about twice the VAE memory** (encode +71%, decode +97% at 1080p batch 9)
  and leaves 5.5 GiB allocated after encode (DiT peak 24.6 instead of 19.1 GiB), for −16–19%
  VAE time and minutes of compilation. Never use it to fit a smaller GPU.
- `max-autotune` with the default `cudaMallocAsync` allocator crashes (CUDA graphs); see
  [environment.md](environment.md). `max-autotune-no-cudagraphs` was not measured (autotuning
  time).

### Other models

The CLI downloads any model of its registry (`--dit_model`, from Hugging Face, sha256-checked)
into `--model_dir`. Same VAE for all. 1080p batch 21, 42 frames (2 batches), PNG output, compared
with [`frame_diff.py`](../scripts/frame_diff.py) to the 7B fp16 output and to the input frames
(the output is 1080p too, so "vs input" measures how much the model changes the image):

| `--dit_model` | File | DiT torch peak | Time per batch | PSNR vs 7B fp16 (worst frame) | PSNR vs input |
|---|---|---|---|---|---|
| `seedvr2_ema_7b_fp16` | 16.5 GB | 22.13 GiB | 11.29 s | – | 27.7 dB |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16` | 8.5 GB | 14.67 GiB | 10.95 s | 43.7 dB (40.0) | 27.5 dB |
| `seedvr2_ema_7b-Q4_K_M.gguf` | 4.8 GB | 11.21 GiB | 11.68 s | 44.6 dB (41.2) | 27.7 dB |
| `seedvr2_ema_3b_fp16` | 6.8 GB | 13.80 GiB | 8.45 s | 32.1 dB (27.8) | 26.4 dB |
| `seedvr2_ema_3b_fp8_e4m3fn` | 3.4 GB | 10.64 GiB | 7.87 s | 32.1 dB (28.1); 43.6 dB vs 3B fp16 | 26.5 dB |

- **Weights are the only difference in memory for a given model size:** each peak is the 7B fp16
  peak minus the weight-size difference, within 0.01 GiB. fp8 weights stay fp8 on the GPU and
  are cast to bf16 per layer under autocast; GGUF weights are dequantized per layer
  (`GGUFQuantizedLinear`); neither adds a visible transient.
- **Quantization is nearly free in speed and close in output:** 7B fp8 −3%, Q4_K_M +3% time
  (−5% and +1% normalized by encode time, these runs' GPU being 2–4% slower); both
  ≈ 44 dB from fp16, the same order as the drift of 1024-pixel decode tiling (43.6 dB, above).
  The 7B fp16 model itself runs under bf16 autocast (its weights are cast on every layer too).
- **The 3B model is a different model, not a cheaper 7B:** 32 dB from the 7B, only 25% faster
  (0.17 ms per token), and ≈ 140 KiB per token (3B fp16 batch 45: 20.34 GiB, slope from batch
  21). PSNR is not a quality measure: none of these is a ground truth, the numbers only say how
  far apart the outputs are.

## The allocator

The CLI sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` unless the variable is already
set. The backend is fixed when torch is imported: a `--wrap` script that imports torch before
the CLI must apply the same default first, or torch asserts "Allocator backend parsed at runtime
!= allocator backend parsed at load time".

A standalone test (allocate 40 × 0.5 GiB, free every other one, allocate 10 × 0.9 GiB, free
everything, `torch.cuda.empty_cache()`), device memory used in GiB:

| `PYTORCH_CUDA_ALLOC_CONF` | CUDA context | 19 GiB live after the pattern | All freed | After `empty_cache` | `set_per_process_memory_fraction` |
|---|---|---|---|---|---|
| `backend:cudaMallocAsync` (CLI default) | 0.79 | 20.8 | 20.8 | 0.79 | honoured |
| `backend:native` | 0.69 | 29.7 | 29.7 | 0.69 | honoured |
| `expandable_segments:True` | 0.69 | 29.7 | 29.7 | 0.69 | honoured |

- **`cudaMallocAsync`** is CUDA's stream-ordered pool. Torch sets its release threshold to
  "never", so freed memory stays in the pool and reserved memory only grows during a run. But
  the driver maps the pool in pages and reuses them for any size: the 0.9 GiB blocks fit in the
  0.5 GiB holes, so it doesn't fragment. When an allocation fails, torch synchronizes, trims the
  pool down to the live tensors and retries, and logs "[cudaMallocAsync] recovered from an
  allocation failure by trimming the pool and retrying". The run goes on with no exception.
  **On a full card the pool shrinks to what is live.** The 1.1–1.25 × NVML overhead seen on 96
  GB is cache that nothing asked back; it is not needed. The context is 0.1 GiB larger.
- **`native`** caches `cudaMalloc` segments. A freed block can only serve requests that fit in
  it, and a segment goes back to the driver only when it is entirely free (on OOM, torch frees
  those and retries). Partly used segments stay: fragmentation (+9 GiB in the test, +10 GiB
  NVML at 1080p batch 9).
- **`expandable_segments:True`** (native allocator) maps one growing virtual segment in 2 MiB
  pages. On a failed mapping it logs "expandable_segments: memory mapping failed", releases its
  cached pages and retries. Its `allocated` figures are ≈ 0.3 GiB higher (block rounding), the
  context is the `native` one.
- `garbage_collection_threshold:X` (native only) frees cached blocks once usage passes X × the
  allowed memory. Not tested: under the ballast emulation below, the allowed memory stays the
  whole 95 GiB device, so it would never trigger.
- **SeedVR2 empties the cache rarely:** `clear_memory` calls `empty_cache` at the end of a run,
  between `--chunk_size` chunks, in the VAE's OOM retry, and after a swapped block when less
  than 5% of the card is free. Never between phases. Each call releases all unused cached memory,
  whatever the backend.

Same configuration on an emulated card ([next section](#emulating-a-smaller-gpu)), only the
allocator changes:

| Run | Allocator | Room for torch | Torch peak | Result | Allocator retries | NVML peak |
|---|---|---|---|---|---|---|
| 8 GB card, 1080p, Q4_K_M swap 36, batch 13, tiles 640 / 512, 26 frames | `cudaMallocAsync` | 6.56 | 6.11 (decode) | ok, 87.7 s | 253 pool trims, all in decode | 7.35 (full) |
| same | `expandable_segments` | 6.66 | 6.40 (decode) | ok, 87.0 s | 127 failed mappings | 7.34 (full) |
| same | `native` | 6.66 | – | **OOM** in decode with 1.40 GiB reserved but unallocated | – | 7.34 |
| 38 GB card, 1080p, 7B fp16, batch 9, untiled, 9 frames | `cudaMallocAsync` | 36.56 | 34.85 (decode) | ok, 40.8 s | 4 pool trims | 37.35 |
| same | `expandable_segments` | 36.66 | 34.85 (decode) | ok, 40.3 s | 4 failed mappings | 36.55 |
| same | `native` | 36.66 | – | **OOM** in decode | – | 37.25 |
| 24 GB card, 1080p, 7B fp16 swap 16, batch 45, tiles 1344 / 1024, 45 frames | `cudaMallocAsync` | 22.56 | 21.81 (DiT) | ok, DiT / encode time 0.90 | 35 pool trims (DiT) | 23.34 |
| same | `expandable_segments` | 22.66 | 21.81 (DiT) | ok, 0.89 | 14 failed mappings | 23.35 |
| same, no cap | `cudaMallocAsync` | – | 21.81 (DiT) | ok, 0.91 | – | 25.72 |
| 30 GB card, 1080p, 7B fp16 no swap, batch 45, tiles 1024 / 1024, 45 frames | `cudaMallocAsync` | 28.56 | 28.15 (DiT) | ok, 0.86 | 33 pool trims (DiT) | 29.35 |
| same | `expandable_segments` | 28.66 | 28.15 (DiT) | ok, 0.90 | 63 failed mappings | 29.35 |

Runs `cap8-1080-q4-bs13-swap36-t512[-expseg|-native]`, `alloc-1080-bs9-cap38-{default,expseg,native}`,
`cap24-1080-fp16-bs45-swap16-t1024[-expseg]`, `alloc-1080-bs45-swap16-nocap`, `alloc-1080-bs45-cap30-{default,expseg}`.
The same untiled run took 50.3 GiB NVML with `native` on the full 96 GB card.

- **Keep the default `cudaMallocAsync`, or use `expandable_segments:True`.** Both use the card up
  to the torch peak plus a few hundred MiB, with no measurable time cost from the retries
  (decode 51.6 s with 253 trims vs 48.8 s with expandable segments, within run-to-run drift).
  Never set `backend:native` alone: it took 25% more device memory at 1080p batch 9 (50.3 vs
  40.0 GiB) and fails where the other two pass.
- Both keep what they grabbed until the end of the run (or the next `empty_cache`): other
  programs on the same card (desktop, browser) can't get that memory back while SeedVR2 runs.

## Emulating a smaller GPU

[`scripts/vram_cap.py`](../scripts/vram_cap.py) is a `--wrap` script that makes the 96 GB card
look like a card of N GB before SeedVR2 starts. After creating the CUDA context it allocates a
**ballast** with the driver API (`cuMemAlloc`, outside torch's allocator, so torch's figures are
untouched) that leaves exactly what an N GB card would leave, and it patches
`torch.cuda.mem_get_info` so that the total reads N GB too. Everything that needs device memory
then competes for the same space as on the small card: torch's pool, cuBLAS/cuDNN workspaces,
the allocator's own retries, and SeedVR2's "less than 5% free" check. bench.py subtracts the
ballast from its NVML figures.

| Quantity | Assumption | 8 GB card | 24 GB card |
|---|---|---|---|
| Capacity (what `cudaMemGetInfo` reports as total) | N − 0.35 GiB (`VRAM_CAP_HIDDEN_GIB`; 0.95 on this 96 GB card) | 7.65 | 23.65 |
| Held by other processes (desktop, compositor) | 0.3 GiB (`VRAM_CAP_OTHER_GIB`; 0 for a headless card) | 7.35 usable | 23.35 usable |
| CUDA context of the CLI | measured: 0.79 GiB with `cudaMallocAsync`, 0.69 with `native` | | |
| **Room for torch** | **≈ N − 1.44 GiB** | 6.56 | 22.56 |

The alternative mode `VRAM_CAP_MODE=fraction` uses `torch.cuda.set_per_process_memory_fraction`
(honoured by both backends on torch 2.14). It caps torch's allocations only, and SeedVR2 then
sees the whole card free; the ballast is the more faithful of the two.

Limits:
- Only memory is emulated. Speed, PCIe bandwidth (BlockSwap) and host RAM are this machine's
  (RTX PRO 6000, PCIe 5.0, 377 GiB RAM): times are not those of a smaller card.
- The context, cuDNN's algorithm choices and the driver's hidden memory vary per GPU and driver.
  The desktop's share varies over time. Keep a few hundred MiB more on a real card that drives
  a display.
- `torch.cuda.get_device_properties` still reports 95 GiB (SeedVR2 only prints it), and
  `garbage_collection_threshold` sees the whole device.
- Windows' WDDM driver can spill to system memory ("sysmem fallback") instead of failing: slower,
  not an OOM. Not emulated.

## Practical rules

The phases peak one after the other, so a run fits when **each phase's torch peak ≤ room for
torch ≈ N − 1.44 GiB on an N GB card, minus ≈ 0.5 GiB of margin**, with the default
`cudaMallocAsync` or `expandable_segments` (validated below; the previous "device memory / 1.2"
rule came from the cache an unconstrained pool keeps). On a card that also drives a display,
subtract what the desktop holds beyond the 0.3 GiB assumed. P = padded output Mpx (720p 0.92,
1080p 2.09, 1440p 3.69, 4K 8.29), T = tile size in Mpx (T² for a square tile), L = latent frames
= 1 + (batch − 1)/4, S = blocks swapped. Torch peaks in GiB, VAE weights (0.47) included:

| Phase | Peak |
|---|---|
| VAE encode, batch ≥ 9 | 1.7 + 8.8 × P untiled, 2.2 + 8.4 × T² tiled |
| DiT | F − s × S + k × P × L (table below) |
| VAE decode, batch ≥ 9 | 1.3 + 16.1 × P untiled, 2.1 + 15.6 × T² tiled |

| `--dit_model` | F | k (per Mpx and latent frame) | s (per swapped block) | Blocks | Swap time per block and batch | Floor, all blocks swapped |
|---|---|---|---|---|---|---|
| 7B fp16 | 16.05 | 0.48 | 0.41 | 36 | 0.07–0.10 s | 1.2 |
| 7B fp8 (`mixed_block35_fp16`) | 8.6 | 0.48 | 0.21 | 36 | 0.034 s | 1.1 |
| 7B Q4_K_M (GGUF) | 5.15 | 0.48 | 0.12 | 36 | ≈ 0 here, up to 0.024 s (moves) | 0.9 |
| 3B fp16 | 7.2 | 0.52 | 0.19 | 32 | 0.029 s | 1.1 |
| 3B fp8 | 4.1 | 0.52 | ≈ 0.1 (not measured) | 32 | – | – |

`--vae_offload_device cpu` lowers F by 0.47 GiB. One latent frame of 7B activations costs 1.0
GiB at 1080p and 4.0 GiB at 4K. 7B DiT time ≈ 0.23 ms per token (26–32% less with
`--compile_dit`), VAE time ≈ 1.2 s per output Mpx and frame (RTX PRO 6000 in its faster
sessions; other sessions ran up to 40–60% slower, see [benchmarking.md](benchmarking.md#caveats)).

**VAE**

1. **Decode is the VAE phase to plan for.** Its peak is ≈ 1 + 16 GiB per output Mpx for batch ≥ 9
   (any length), ≈ 10 GiB/Mpx for batch 5, ≈ 3.4 GiB/Mpx for batch 1. Encode is ≈ 55% of that.
2. **Batch size doesn't change the VAE peak beyond 9.** Pick the batch for the DiT (memory,
   temporal consistency), not for the VAE.
3. **With tiling, only the tile size matters:** decode ≈ 1.6 + 15.6 × T² GiB (batch ≥ 9), for any
   output resolution and batch length.
4. Use the largest tile that fits, overlap 64–128. Tile the decode before the encode.
5. Don't use `--compile_vae` near the memory limit: about twice the VAE peak (1080p batch 9:
   encode 19.7 → 33.8, decode 34.9 → 68.6 GiB; 4K batch 5: +10 GiB, +39 GiB on the compiling
   call).

| Card (room for torch) | Untiled VAE, batch ≥ 9 | Untiled VAE, batch 5 | Decode tile (batch ≥ 9) |
|---|---|---|---|
| 8 GB (6.6) | < 0.3 Mpx | < 0.5 Mpx | 512 (6.1 GiB measured) |
| 12 GB (10.6) | < 0.55 Mpx | < 0.9 Mpx (720p decode 10.1: too tight) | 640 (8.7) |
| 16 GB (14.6) | < 0.8 Mpx | ≤ 720p (10.1) | 768 (11.6) |
| 24 GB (22.6) | ≤ 720p (16.2) | ≤ 1080p (21.7, tight) | 1024 (18.2) |
| 32 GB (30.6) | < 1.8 Mpx | ≤ 1080p (21.7) | 1280 (23.8 at 1080p, ≈ 27.7 square) |
| 48 GB (46.6) | ≤ 1080p (35.7 at batch 81, measured) | ≤ 1440p (37.6) | 1536 (39.2) |
| 80 GB (78.6) | ≤ 1440p (60.7) | ≤ 1440p | 1536+ |
| 96 GB (93.5) | ≤ 1440p | ≤ 4K (83.5) | 1536+ |

Limits from the fits with torch peak ≤ room − 0.5 GiB (room ≈ N − 1.44 GiB, see
[Emulating a smaller GPU](#emulating-a-smaller-gpu); 93.5 for the 96 GB card measured here).
Torch peaks in parentheses, measured unless marked ≈. Mpx is the padded output area (720p =
0.92, 1080p = 2.09, 1440p = 3.69, 4K = 8.29).

**DiT**

6. **Batch for throughput first:** DiT time per frame ∝ (1 + (b − 1)/4) / b. At 1080p, batch 21
   already gets 91% of the batch-81 throughput, batch 5 66%, batch 1 28%. Each latent frame (4
   video frames) costs 1.0 GiB at 1080p, 4.0 GiB at 4K (7B).
7. **Shrink the weights before swapping them.** 7B fp8 and Q4_K_M run at the fp16 speed (−3%,
   +3%) with half and a third of the weights, and their output is ≈ 44 dB from fp16. With
   Q4_K_M, swapping all 36 blocks cost nothing measurable here (the blocks do move: ≈ 0.9 s of
   copies per batch, hidden on this power-capped GPU, see [BlockSwap](#blockswap)): on a small GPU use
   `--dit_model seedvr2_ema_7b-Q4_K_M.gguf --dit_offload_device cpu --blocks_to_swap 36`, and
   spend the memory on the batch. The 3B model is a different model (32 dB from the 7B), not a
   cheaper copy, and saves only 25% of the time.
8. **BlockSwap costs a fixed time per block and per batch** (7B fp16: 0.07–0.10 s, fp8: 0.03 s),
   so it hurts small batches most: 36 fp16 blocks add 97% at 1080p batch 5, 20% at batch 21,
   12% at batch 45. Swapping blocks to afford batch ≥ 21 beats dropping to batch 5. Expect up to twice the cost on
   PCIe 4.0. Host RAM: + the swapped weights (7B fp16 all swapped: 34 GiB max RSS against 19).
9. `--vae_offload_device cpu` saves 0.47 GiB in the DiT phase for 0.2 s. `--dit_offload_device`
   alone saves nothing (the DiT is never on the GPU during the VAE phases); it is the BlockSwap
   prerequisite. `--tensor_offload_device none` changes nothing measurable. Never use
   `--swap_io_components` without `--blocks_to_swap`: it leaks the whole DiT into the decode
   phase.
10. `--compile_dit` saves 26–32% of DiT time for 10–50 s of compilation and +0.1–1.4 GiB (1080p
    batch 5–45), but nothing with BlockSwap. Use it for long clips when the model fits without
    swapping (it fits the 32 GB 1080p recipe below).

### Recipe per card size, validated

Each configuration ran under [`vram_cap.py`](../scripts/vram_cap.py) (default `cudaMallocAsync`,
room for torch ≈ N − 1.44 GiB), then one notch up. Tiles: encode / decode, overlap 64. Torch
peaks per phase in GiB; NVML = the emulated card's device memory used (ballast subtracted; the
card is full at N − 0.65). "Trims" = allocation failures the allocator recovered from. Times
are on the RTX PRO 6000, which ran 40–60% slower than in the earlier steps (power cap at
577 MHz, uncapped controls included): they only compare runs of this table.

**1080p output**

| Card (room) | Recommended | Encode / DiT / decode peak | Smallest margin | NVML | Trims | Time (frames) | Also passed |
|---|---|---|---|---|---|---|---|
| 8 GB (6.56) | Q4_K_M, swap 36, **batch 17**, tiles 640 / 512 | 5.81 / 5.97 / 6.11 | 0.45 | 7.35 | 204 | 93 s (22) | batch 13: 5.71 / 5.06 / 6.11 |
| 12 GB (10.56) | Q4_K_M, swap 36, **batch 33**, tiles 896 / 640 | 9.81 / 10.00 / 8.66 | 0.56 | 11.35 | 86 | 154 s (38) | batch 29: 9.72 / 9.09 / 8.65 |
| 16 GB (14.56) | Q4_K_M, swap 36, **batch 49**, tiles 1024 / 768 | 12.10 / 13.96 / 11.61 | 0.60 | 15.35 | 36 | 222 s (54) | batch 41: 11.91 / 12.06 / 11.61 |
| 24 GB (22.56) | 7B fp16, **swap 24**, batch 45, tiles 1344 / 1024 | 15.54 / 18.43 / 18.18 | 4.1 | 23.35 | 1 | 183 s (50) | swap 16: DiT 21.81 (margin 0.75, ≈ −0.6 s per batch); keep swap 24 on a desktop card |
| 32 GB (30.56) | 7B fp16, no swap, **batch 45**, encode untiled, decode tile **1280** | 20.58 / 28.15 / 23.83 | 2.4 | 31.34 | 15 | 188 s (45) | batch 29, decode 1152: 20.20 / 24.19 / 21.38; + `--compile_dit`: DiT 29.56, −32% DiT time |
| 48 GB (46.56) | 7B fp16, no swap, batch 81, untiled | 21.44 / 36.97 / 35.72 | 9.6 | 45.50 | 0 | 355 s (81) | – |

**4K output**

| Card (room) | Recommended | Encode / DiT / decode peak | Smallest margin | NVML | Trims | Time (frames) |
|---|---|---|---|---|---|---|
| 8 GB (6.56) | Q4_K_M, swap 36, batch 1, tiles 640 / 512 | 2.12 / 4.97 / 2.36 | 1.6 | 6.78 | 0 | 63 s (2) |
| 12 GB (10.56) | Q4_K_M, swap 36, batch 5, tiles 896 / 640 | 7.58 / 9.17 / 6.22 | 1.4 | 11.34 | 21 | 100 s (5) |
| 16 GB (14.56) | Q4_K_M, swap 36, **batch 9**, tiles 1024 / 768 | 11.47 / 13.36 / 11.61 | 1.2 | 15.35 | 35 | 176 s (9) |
| 24 GB (22.56) | 7B fp16, swap 36, batch 13, tiles 1344 / 1024 | 18.00 / 17.86 / 18.00 | 4.6 | 23.00 | 0 | 241 s (13) |
| 32 GB (30.56) | 7B fp16, swap 36, batch 25, tiles 1600 / 1280 | 25.11 / 29.65 / 27.68 | 0.91 | 31.35 | 71 | 349 s (25) |
| 48 GB (46.56) | 7B fp16, no swap, **batch 25**, tiles 2048 / 1536 | 39.63 / 44.44 / 39.46 | 2.1 | 47.35 | 42 | 363 s (25) |

Runs `cap<N>-<res>-<model>-bs<B>[-swap<S>]-t<decode tile>`. The 4K 32 and 48 GB rows ran in a
later session, ≈ 11% faster (`cap32-1080-fp16-bs45-t1280` repeated: 32.9 s per DiT batch
against 37.1).

- **The model and the per-phase formulas hold under the cap.** Every measured DiT and decode
  peak is within 0.9 GiB of the formulas, and every capped configuration ran without an
  exception with `cudaMallocAsync` (22 runs, notch-ups, allocator and compile tests included).
  Tiled encode with long batches reads above its fit, because the batch's input frames are on
  the GPU: up to 0.9 GiB at 1080p batch ≥ 29, 1.4–2.2 GiB at 4K batch 21–25.
- **48 GB at 4K takes batch 25, not the predicted 21:** batch 21 passed with 6.0 GiB to spare
  (39.39 / 40.51 / 39.40, `cap48-2160-fp16-bs21-t1536`). Batch 29 would need ≈ 48.4 GiB in the
  DiT. At 32 GB the DiT is the limit too (29.65 GiB, N − 2.35): batch 29 would need ≈ 33.6.
- **The old "torch peak ≤ card / 1.2" rule was too strict.** With the allocator trimming its
  pool, runs pass with 0.41–0.6 GiB between the torch peak and the room (8, 12, 16 GB rows,
  each one notch above the earlier recipe, and `alloc-1080-bs45-cap30-default`: DiT 28.15 GiB
  on a 30 GB card, room 28.56). Plan **torch peak ≤ N − 2 GiB** (room minus 0.5 GiB; the
  passing runs went up to N − 1.85). The next notch (one more latent frame = +1.0 GiB at 1080p)
  would exceed the room on 8, 12 and 16 GB.
- **Running near the limit costs no measurable time.** When less than 5% of the card is free,
  SeedVR2 empties the cache after every swapped block (`clear_memory(force=False)`), and the
  next block's allocations make the pool trim and regrow: about one trim per swapped block (35
  trims and 16 cache flushes for `cap24-…-swap16`). The same run without a cap
  (`alloc-1080-bs45-swap16-nocap`, run 30 min later) had the same DiT / encode time ratio (0.91 vs
  0.90). Decode on small cards trims hundreds of times (253 at 8 GB) with no visible cost either.

## Reproduce

```bash
# untiled point (VAE probe: <runs-dir>/<name>.vae.json next to the log)
python3 scripts/bench.py run vae-1080-bs9 --wrap scripts/vae_probe.py ... -- <input> ... \
  --resolution 1080 --batch_size 9 --load_cap 9 --skip_first_frames 48 \
  --attention_mode flash_attn_2 --color_correction none
# tiled: add --vae_encode_tiled --vae_encode_tile_size 1024 --vae_encode_tile_overlap 128
#            --vae_decode_tiled --vae_decode_tile_size 1024 --vae_decode_tile_overlap 128
# config overrides: --env VAE_PROBE_CACHE_DEVICE=cpu, --env VAE_PROBE_CONV3D_WORKAROUND=0
python3 scripts/vae_probe.py --summary runs/vae-*.vae.json        # per-call table
# quality: --output_format png, then
python3 scripts/frame_diff.py <untiled>/<stem> <tiled>/<stem> --tile 512 --overlap 128 --band 16
# DiT: plain bench.py runs, e.g. BlockSwap
python3 scripts/bench.py run dit-1080-bs45-swap36 ... -- <input> ... --resolution 1080 \
  --batch_size 45 --load_cap 45 --skip_first_frames 48 --attention_mode flash_attn_2 \
  --color_correction none --vae_encode_tiled --vae_decode_tiled --vae_encode_tile_size 1024 \
  --vae_decode_tile_size 1024 --vae_encode_tile_overlap 64 --vae_decode_tile_overlap 64 \
  --dit_offload_device cpu --blocks_to_swap 36
# other models: --dit_model seedvr2_ema_7b-Q4_K_M.gguf (downloaded by the CLI into --model_dir)
# emulated 8 GB card (ballast; NVML figures net of it), optionally another allocator
python3 scripts/bench.py run cap8-1080-q4-bs17-swap36-t512 --wrap scripts/vram_cap.py \
  --env VRAM_CAP_GIB=8 [--env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True] ... -- <input> ... \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf --dit_offload_device cpu --blocks_to_swap 36 \
  --resolution 1080 --batch_size 17 --load_cap 22 --vae_encode_tiled --vae_decode_tiled \
  --vae_encode_tile_size 640 --vae_decode_tile_size 512 --vae_encode_tile_overlap 64 \
  --vae_decode_tile_overlap 64 --skip_first_frames 48 --attention_mode flash_attn_2 --color_correction none
# BlockSwap moves: <runs-dir>/<name>.swap.json
python3 scripts/bench.py run swapprobe-1080-bs5-q4-swap36 --wrap scripts/swap_probe.py ... -- <input> ... \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf --dit_offload_device cpu --blocks_to_swap 36 \
  --resolution 1080 --batch_size 5 --load_cap 10
```

<details>
<summary>Run names (all 7B fp16, <code>flash_attn_2</code>, <code>--color_correction none</code>,
<code>--skip_first_frames 48</code>, <code>--load_cap</code> = batch size unless noted)</summary>

`vae-<res>-bs<N>[-<variant>]`: `vae-1080-bs{1,5,9,13,21}`, `vae-720-bs{5,9}`,
`vae-1440-bs{5,9}`, `vae-2160-bs5` (cap 10), `vae-2160-bs5-ov1-pp1` (cap 10),
`vae-2160-bs5-userflags` (cap 10), `vae-2160-bs9`, `vae-2160-bs9-t{512,768,1024,1536}o128`,
`vae-2160-bs9-t1024o{64,256}`, `vae-1080-bs9-png`, `-png-r2`, `-dt512o64-png`, `-dt512o128-png`,
`-dt1024o128-png`, `-et512o128-png`, `-t512o128-png` (`dt` = decode tiled only, `et` = encode
only, `t` = both), `vae-1080-bs9-{cachecpu,noconvwa,expseg,native}`, `vae-2160-bs5-expseg`,
`vae-2160-bs9-cachecpu`.

`dit-1080-bs<N>[-<variant>]` (`--load_cap` = 2 batches up to batch 21, one batch from 45, VAE
tiled 1024/64 from 45): `dit-1080-bs{1 (cap 3),5,9,45,81}`, `dit-1080-bs21-png`, `-r2`;
BlockSwap `dit-1080-bs45-swap{12,24,36,0io,36io}`, `dit-1080-bs5-swap36`, `dit-1080-bs21-swap36`;
offload `dit-1080-bs45-{ditoff,vaeoff,tnone,r2}`; models `dit-1080-bs21-{7bfp8,3bfp16,3bfp8,7bq4}-png`,
`dit-1080-bs45-3bfp16`, `dit-1080-bs5-{7bfp8,7bq4,3bfp16}`, `-7bfp8-swap36`, `-7bq4-swap36`,
`-3bfp16-swap32`; compile `dit-1080-bs5-compile` (cap 15), `dit-1080-bs21-compile`,
`-compile-swap36`, `-7bfp8-compile`, `vae-1080-bs9-compilevae` (cap 18).

Allocator and emulation (`--load_cap` = batch + 5, or 2 batches, unless noted):
`cap8-1080-q4-bs13-swap36-t512[-expseg|-native]`, `cap8-1080-q4-bs17-swap36-t512`,
`cap12-1080-q4-bs{29,33}-swap36-t640`, `cap16-1080-q4-bs{41,49}-swap36-t768`,
`cap24-1080-fp16-bs45-swap{24,16}-t1024`, `cap24-1080-fp16-bs45-swap16-t1024-expseg`,
`cap32-1080-fp16-bs29-t1152`, `cap32-1080-fp16-bs45-t1280`, `cap48-1080-fp16-bs81` (one batch),
`cap8-2160-q4-bs1-swap36-t512`, `cap12-2160-q4-bs5-swap36-t640`, `cap16-2160-q4-bs9-swap36-t768`,
`cap24-2160-fp16-bs13-swap36-t1024`, `cap32-2160-fp16-bs25-swap36-t1280`,
`cap48-2160-fp16-bs{21,25}-t1536` (4K: one batch, two at batch 1),
`cap32-1080-fp16-bs45-t1280-x2[-compile]` (cap 90, run back to back),
`alloc-1080-bs9-cap38-{default,expseg,native}`, `alloc-1080-bs45-cap30-{default,expseg}`,
`alloc-1080-bs45-swap16-nocap`; BlockSwap probe `swapprobe-1080-bs5-{q4,fp8,fp16}-swap36`,
controls `swapctl-1080-bs5-q4[-swap36|-r2]`.

</details>
