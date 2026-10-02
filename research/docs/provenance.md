# Where the SeedVR2 code comes from

> Status: **from comparing the code**. numz/ComfyUI-SeedVR2_VideoUpscaler `4490bd1` (v2.5.24)
> against ByteDance-Seed/SeedVR `e4de8c2`, both pinned under [`../../upstream/`](../../upstream/).
> Line counts are Python lines.

## Lineage

- **ByteDance** published the SeedVR2 inference code on 2025-06-17 (`790ea60`). Its last code
  change is from 2025-07-02, a fix contributed by numz (`common/utils.py`,
  `safe_pad_operation` / `safe_interpolate_operation`). Later commits only touch the README.
- **numz's first commit** (`fc1d24e`, 2025-06-20) copies ByteDance's `common/`, `data/`,
  `models/dit`, `models/dit_v2`, `models/video_vae_v3`, the configs and the text embeddings,
  plus one ComfyUI node (320 lines). There is no other port in between; ByteDance's README later
  links to numz's repo.
- Most of the current code is by Adrien Toupet (AInVFX): 306 of 423 commits, against 62 for numz.
- Naming: numz's `dit_3b` is ByteDance's `models/dit_v2`, and `dit_7b` is `models/dit`.

## Three layers

| Layer | Path in numz | Lines (ByteDance) | Provenance |
|---|---|---|---|
| **a. Model** | `src/models/dit_3b/` | 2,364 (1,709) | ByteDance, modified: 7 files verbatim, 9 changed |
| | `src/models/dit_7b/` | 2,419 (1,813) | ByteDance, modified: 6 verbatim, 9 changed |
| | `src/models/video_vae_v3/` | 3,623 (3,293) | ByteDance, modified: see [below](#what-numz-changed-in-the-model-code) |
| | `src/common/diffusion/` (Euler sampler, schedule, timesteps) | 787 (776) | ByteDance: 7 files verbatim, 3 trivially changed |
| | `configs_3b/`, `configs_7b/`, VAE yaml | 207 yaml | ByteDance, modified: class paths, fp16 VAE, file names |
| | `pos_emb.pt`, `neg_emb.pt` | — | byte-identical: precomputed text embeddings, so no text encoder is needed |
| ByteDance plumbing | `src/common/{cache,config,decorators,logger,partition,seed}.py`, `src/common/distributed/` | 1,322 (1,299) | mostly verbatim; ~880 lines of sequence-parallel code unused on one GPU |
| | `src/data/image/transforms/` | 350 (279) | modified: `max_size`, `DivisiblePad`, MPS |
| | `src/core/infer.py` (`VideoDiffusionInfer`) | 394 (347) | ByteDance's `projects/video_diffusion_sr/infer.py`, modified: model loading removed, tiling, autocast, debug added |
| **b. numz runtime** | `src/core/` (except `infer.py`): generation phases, model loading/cache, alpha | 5,422 | numz |
| | `src/optimization/`: BlockSwap, memory manager, attention dispatch, performance | 3,291 | numz |
| | `src/optimization/gguf_dequant.py` | 343 | adapted from city96/ComfyUI-GGUF (Apache-2.0), credited |
| | `src/optimization/gguf_ops.py` | 283 | numz |
| | `src/utils/color_fix.py` | 872 | wavelet and AdaIN part (~250 lines) adapted from StableSR, not credited; LAB, HSV, wavelet_adaptive are numz (see [Licences](#licences)) |
| | `src/utils/` (others: debug, downloads, constants, model registry) | 1,332 | numz |
| | `src/common/half_precision_fixes.py` | 159 | numz (grown from its own upstream fix) |
| **c. Wrappers** | `inference_cli.py` | 1,711 | numz: options, video I/O, multi-GPU split |
| | `src/interfaces/` (ComfyUI nodes) | 1,134 | numz |

Totals: **25,814** lines in numz, **11,318** in ByteDance's repo.
- Model code: about 9.2k lines (7.6k in ByteDance for the same files, so ~1.6k lines of numz
  changes inside the model files).
- ByteDance plumbing: about 2.1k.
- numz runtime: about 11.7k.
- Wrappers: about 2.9k.

ByteDance files numz did not carry over: `projects/inference_seedvr*_{3b,7b}.py` (the
reference inference scripts), `video_diffusion_sr/utils.py`, `common/utils.py`,
`data/video/transforms/rearrange.py`.

## What numz changed in the model code

- **Attention** (`dit_*/attention.py`): the hard `flash_attn` import is replaced by a backend
  switch (SDPA, FA2, FA3, SageAttention 2/3, through `src/optimization/compatibility.py`). The
  `.bfloat16()` casts on q/k/v are removed, so attention runs in the pipeline's dtype.
  `.item()` calls are removed for `torch.compile`.
- **Windowing** (`na.py`, the same file for both sizes): rewritten so `torch.compile` can trace
  it (`tensor_split` instead of `.tolist()`); meant to be equivalent.
- **Normalisation** (`normalization.py`): Apex's fused LayerNorm/RMSNorm replaced by plain
  PyTorch, with fp8 weight upcasts. ByteDance's configs use the fused versions, so ByteDance's
  code needs Apex.
- **Modulation, MMSR block:** fp8 parameter upcasts and dtype casts.
- **RoPE:** the 7B casts the frequencies to the query dtype (half precision) before applying
  them, where ByteDance keeps fp32. `get_freqs` is excluded from `torch.compile`.
- **VAE:**
  - spatial tiling (`tiled_encode`/`tiled_decode`, ~330 lines); ByteDance raises
    `NotImplementedError`
  - OOM retry around convolutions, norms and concatenations (10 call sites)
  - the cuDNN Conv3d memory workaround
  - some dtype casts removed
  - encoding takes the distribution's mode instead of a sample, so it is deterministic
  - the sequence-parallel code removed or stubbed

The changes that alter numerics (RoPE precision, attention dtype, VAE mode instead of sample)
were not measured separately.

## Licences

- **ByteDance code:** Apache-2.0, with a "Copyright (c) 2025 Bytedance Ltd." header on every
  model, common and data file. The VAE files also carry "Copyright 2023 HuggingFace Team"
  (Apache-2.0).
- **numz repo:** Apache-2.0 (its LICENSE is identical to ByteDance's; it started as MIT and
  switched to match ByteDance). numz kept ByteDance's headers on derived files; its own files
  have no headers.
- **GGUF dequantisation:** adapted from city96/ComfyUI-GGUF, Apache-2.0, credited in the file
  header.
- **Colour correction:** the wavelet and AdaIN functions in `color_fix.py` are adapted from
  StableSR / sd-webui-stablesr `colorfix.py`, under the **S-Lab License 1.0 (non-commercial)**
  (sd-webui-stablesr also ships CC BY-NC-SA 4.0), without attribution. `lab`, `wavelet`,
  `wavelet_adaptive` and `adain` all run that code; only `hsv` and `none` don't
  ([cli-flags.md](cli-flags.md)). ByteDance doesn't ship this file: its README tells users to
  download it from sd-webui-stablesr.
- **Weights:** the Hugging Face model cards for ByteDance-Seed/SeedVR2-3B and SeedVR2-7B, and the
  repackaged fp16/fp8/GGUF weights (numz/SeedVR2_comfyUI, AInVFX/SeedVR2_comfyUI), state
  Apache-2.0.

## Using the model code outside numz

The model modules import without ComfyUI (`src.models.dit_3b.nadit`, `src.models.dit_7b.nadit`,
the VAE, `src.common.diffusion`, `src.core.infer`), but not without numz's runtime:
- the DiT imports `src.optimization.compatibility` (attention dispatch) and
  `src.optimization.memory_manager`, plus `src.common.{cache, distributed.ops,
  half_precision_fixes}`
- the VAE imports `memory_manager.retry_on_oom`,
  `compatibility.NVIDIA_CONV3D_MEMORY_BUG_WORKAROUND` and `common.logger`
- importing prints banners and applies triton / flash_attn / bitsandbytes shims as a side
  effect
- config loading goes through `src.utils.model_registry`

Third-party dependencies of the model layer: torch, diffusers (VAE base classes, RMSNorm,
timestep embedding), einops, rotary_embedding_torch, omegaconf; flash_attn and sageattention are
optional.

Two routes for reusing it (see [`seedvr2x/DESIGN.md`](../../seedvr2x/DESIGN.md)):
- **ByteDance's code as is:** needs flash_attn, Apex, absolute `common.`/`models.` imports, and
  has no tiling.
- **numz's model files, vendored:** stub or replace the four runtime functions above and drop
  the rest of `src/core` and `src/optimization`. `VideoDiffusionInfer` (`src/core/infer.py`) is
  the natural seam for our own orchestration.
