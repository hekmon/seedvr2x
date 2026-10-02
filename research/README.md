# Research

Research notes and experiments on **SeedVR2**, ByteDance-Seed's one-step diffusion video
upscaler/restorer, through the
[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) CLI.
`file:line` references point to numz `4490bd1`, pinned with ByteDance's SeedVR `e4de8c2` as
submodules under `../upstream/`.

Goals:
- understand what each of the CLI's 44 options does, and measure its effect
- understand and model **VRAM** use per phase (VAE encode, DiT, VAE decode), and derive rules that
  carry over to GPUs with less memory
- isolate the options (offload, caching, BlockSwap, batch size, temporal overlap, VAE tiling,
  attention backend, torch.compile…) and measure their impact on memory, speed and quality

## Key findings

Measured on one RTX PRO 6000 (96 GB) with the 7B fp16 model unless noted; the CLI's own default
is the 3B fp8 model ([cli-flags.md](docs/cli-flags.md#models)).

- **The attention backend doesn't matter:** all four give the same DiT time within ±1.5%, `sageattn_3` never runs SA3, and attention is 4–9% of DiT time. Use `flash_attn_2` ([attention.md](docs/attention.md#backend-comparison-clean-runs)).
- **VAE decode, not the DiT, sets the VRAM peak** of untiled runs from 1080p batch 5 up: ≈ 16 GiB per output megapixel as soon as the batch is ≥ 9, whatever the batch size; 4K needs ≈ 134 GiB untiled ([vram.md](docs/vram.md#untiled-vae-the-model)).
- **With VAE tiling, the peak depends on the tile size only:** decode ≈ 1.6 + 15.6 × T² GiB (batch ≥ 9); no seams, but a per-tile drift of up to ≈ 2 levels that `lab` colour correction removes ([vram.md](docs/vram.md#tiling), [quality.md](docs/quality.md#vae-tiling-on-flat-areas)).
- **`--compile_vae` costs about twice the VAE memory** and caused the 4K OOM; `--compile_dit` saves 26–32% of DiT time, but nothing with BlockSwap ([vram.md](docs/vram.md#the-4k-oom-reproduced), [torch.compile](docs/vram.md#torchcompile)).
- **DiT peak = 16.05 GiB + 128.5 KiB per token** (7B fp16, `flash_attn_2`); batch 21 already gets 91% of the batch-81 throughput ([vram.md](docs/vram.md#weights-and-the-per-token-model)).
- **BlockSwap: −0.41 GiB per swapped 7B fp16 block** for +0.07–0.10 s per block and batch, a fixed cost per forward that hurts small batches most ([vram.md](docs/vram.md#blockswap)).
- **8–16 GB cards: `seedvr2_ema_7b-Q4_K_M.gguf` with all 36 blocks swapped**, plus VAE tiling, validated on emulated cards; plan each phase's torch peak ≤ N − 2 GiB ([vram.md](docs/vram.md#recipe-per-card-size-validated)).
- **Keep the default `cudaMallocAsync` allocator** (or `expandable_segments:True`): on a full card it trims its pool; `backend:native` fragments and fails ([vram.md](docs/vram.md#the-allocator)).
- **Keep `--color_correction lab`:** it removes the model's colour drift (+30% saturation) without losing detail and halves the low-frequency jumps at batch boundaries ([quality.md](docs/quality.md#colour-correction)). It runs StableSR-derived code under a non-commercial licence, like every mode but `hsv` and `none` ([cli-flags.md](docs/cli-flags.md#quality)).
- **`--temporal_overlap` only cross-fades with odd values ≥ 3:** 1, 2 and 4 cost compute and change nothing at the boundary ([quality.md](docs/quality.md#--temporal_overlap)).
- **`--prepend_frames` frames are not removed on one GPU:** 45 frames in, 49 out with `--prepend_frames 4` ([quality.md](docs/quality.md#--prepend_frames)).
- **Both noise options degrade the output:** input noise turns into texture and grain, latent noise washes the image out ([quality.md](docs/quality.md#noise-scales)).

## Contents

| Path | What |
|---|---|
| [docs/environment.md](docs/environment.md) | How to build a clean environment with every attention backend (Blackwell / sm_120), the pitfalls, and reference results |
| [docs/attention.md](docs/attention.md) | DiT windowed attention: sequence lengths, why `sageattn_3` and `sageattn_2` don't do what their names say, measured attention share of DiT time and backend comparison |
| [docs/cli-flags.md](docs/cli-flags.md) | Reference of the CLI's 44 options, from the code: what each does, defaults, interactions, silent fallbacks and ignored combinations, environment variables, input/output handling, models, multi-GPU, and the bugs found (prepended frames kept in the output, `--swap_io_components` leak, overlaps that don't blend, `--10bit` on 8-bit frames…), with links to the measurements |
| [bugs/](bugs/README.md) | One file per bug found in the CLI (23, from hangs and wrong output to memory leaks and misleading help): reproduction, root cause with `file:line` references, impact and workaround, a possible fix and how to test it. Candidate upstream reports, not yet filed |
| [scripts/setup_env.sh](scripts/setup_env.sh) | Reproducible environment build: uv venv, stable torch, SageAttention 2/3 and FlashAttention 2 built as wheels |
| [scripts/probe_env.py](scripts/probe_env.py) | Checks an environment: versions, which attention backend really runs, accuracy and throughput |
| [docs/benchmarking.md](docs/benchmarking.md) | How runs are measured: the record's fields, what torch and NVML memory figures mean, caveats, reference runs |
| [scripts/bench.py](scripts/bench.py) | Measurement harness: runs the CLI, samples device memory (NVML), parses the debug log into per-phase JSON records and Markdown tables |
| [scripts/attn_probe.py](scripts/attn_probe.py) | Attention probe: runs the CLI with import-time patches and records every DiT attention call (window lengths, uniform or not, kernel that really ran, GPU time), optionally a `torch.profiler` table |
| [docs/vram.md](docs/vram.md) | VRAM per phase, how the VAE encodes/decodes (temporal slicing, causal caches, tiling), a per-megapixel VRAM model, tiling cost and quality, the 4K OOM; the DiT's per-token model, batch-size throughput, BlockSwap, offload, `torch.compile` and the fp8 / GGUF / 3B models (memory, speed, PSNR); the allocators (`cudaMallocAsync`, native, expandable segments) on a full card; emulating smaller GPUs; rules and a recipe for 8–48 GB GPUs validated by emulation |
| [scripts/vae_probe.py](scripts/vae_probe.py) | VAE probe: runs the CLI with import-time patches and records every VAE encode/decode call, temporal slice and tile (torch peak, causal-cache bytes, GPU time); can override the VAE's slicing / memory-limit config and the Conv3d workaround |
| [scripts/vram_cap.py](scripts/vram_cap.py) | Emulates a smaller GPU: a driver-level ballast leaves only what an N GB card would (context included), and SeedVR2 sees that card's free and total memory; bench.py nets the ballast out of its NVML figures |
| [scripts/swap_probe.py](scripts/swap_probe.py) | BlockSwap probe: times every move of a swapped DiT block between CPU and GPU (bytes, bandwidth, compute in between) |
| [scripts/frame_diff.py](scripts/frame_diff.py) | Compares two PNG sequences: PSNR, and a seam / per-tile drift check against SeedVR2's tile grid |
| [docs/quality.md](docs/quality.md) | Quality options measured with no-reference proxies: colour correction modes, batch size and batch-boundary flicker, `--temporal_overlap`, `--prepend_frames`, `--uniform_batch_size`, input/latent noise, seed, VAE tiling on flat areas; recommendations |
| [scripts/quality_metrics.py](scripts/quality_metrics.py) | Quality proxies for a PNG sequence vs its input and a reference run: PSNR/SSIM, colour and CIELAB stats, sharpness and flat-area grain, per-transition temporal change split at batch boundaries (held-frame flicker), per-tile offsets on flat areas |

Experiments run on a remote GPU machine. Where that machine is and how it's reached is
deployment-specific and deliberately left out of this repo.
