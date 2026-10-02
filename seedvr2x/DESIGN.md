# seedvr2x design

> Status: **draft for review**, nothing implemented. It collects the decisions taken so far and
> the questions still open. Each decision links to the measurement it rests on in
> [../research/](../research/).

## Goal

A SeedVR2 video upscaler for **long runs** (whole episodes or films) that:
- produces a lossless, correctly tagged master that the next encoding step can trust
- has no visible seams inside a shot
- fits the GPU it runs on without the user tuning memory options
- can be paused and resumed (run at night, give the computer back in the morning)

It replaces numz's orchestration and I/O (about 14.6k lines, where nearly all of the
[23 bugs](../research/bugs/README.md) live). It keeps ByteDance's model code.

### Not in the first version
- Image inputs, ComfyUI nodes, macOS/MPS, AMD/ROCm
- Multi-GPU in one process (separate processes on separate shot ranges can come later, see
  [Pause and resume](#pause-and-resume))
- Options that only work around numz's own design (see [Options](#options-kept-and-dropped))

## Architecture

| Layer | Contents | Origin |
|---|---|---|
| Model | DiT 7B/3B, VAE, windowed attention, RoPE, Euler sampler, configs, text embeddings | **vendored** from `upstream/seedvr2-numz` (ByteDance code, numz changes) |
| Runtime | shot pipeline, latent stitching, memory planner, BlockSwap, resume | ours |
| I/O | ffmpeg pipes in and out, FFV1/PNG writers, manifest | ours |
| CLI | options, logging | ours |

**One long-running process** handles a whole job: models are loaded (and compiled) once and
stay on the GPU when the plan allows it.

### Vendored model code
- Copied from `upstream/seedvr2-numz` at `4490bd1`: `src/models/{dit_7b,dit_3b,video_vae_v3}`,
  `src/common/diffusion`, the parts of `src/common` and `src/core/infer.py`
  (`VideoDiffusionInfer`) that the model needs, the configs and `pos_emb.pt`/`neg_emb.pt`.
  About 11k lines ([provenance](../research/docs/provenance.md)).
- The submodules stay untouched. A script diffs our copy against numz and ByteDance, so every
  change stays visible.
- Changes we expect to make in our copy:
  - remove the dependencies on numz's runtime: `retry_on_oom`, the attention dispatch, the
    Conv3d flag, MPS detection, the banners and import-time shims
  - attention: SDPA, or FlashAttention 2 when installed, nothing else
    ([why](../research/docs/attention.md))
  - drop the ~880 lines of sequence-parallel code
  - decide per change whether to keep numz's numerics or go back to ByteDance's: RoPE in half
    precision (ByteDance: fp32), attention in the pipeline dtype (ByteDance: bf16), VAE encode
    taking the mode instead of a sample. Each one is measured against the other before choosing.
- Licence: Apache-2.0. We keep the copyright headers, add a NOTICE, and mark modified files.
  The StableSR-derived colour code (`color_fix.py`, non-commercial licence) is **not**
  vendored: see [Colour correction](#colour-correction).

### Weights
7B fp16, 7B fp8, 7B Q4_K_M (GGUF), 3B fp16/fp8, all Apache-2.0
([memory and quality per model](../research/docs/vram.md)). GGUF needs the dequantisation code
adapted from city96/ComfyUI-GGUF (Apache-2.0, credited).

## Input

Two forms, one internal model (a list of shots):
1. **One master plus a list of cuts**: frame numbers or timestamps, e.g. from sptenc's scene
   detection.
2. **A directory of segments** (sptenc split output), plus the information of which joins are
   real cuts. Joins that aren't real cuts are stitched like a long shot.

Decoding goes through an ffmpeg pipe:
- frame-accurate, counting the frames actually decoded (never trusting the container's count,
  bug 11)
- 16-bit RGB, from a known matrix (BT.709 for HD unless the tags say otherwise)
- exact rational frame rate

## Pipeline, per shot

1. **VAE encode** of the whole shot in one causal pass. The VAE already streams in 4-frame
   slices; resetting it at each cut is correct (no context should cross a cut).
2. **DiT** on windows:
   - one window per shot when it fits
   - otherwise equal windows sharing **M = 2 latents** (8 frames), the shared latents mixed
     before decoding
   - noise drawn once per shot from the seed, sliced per window

   [Measured](../research/docs/stitching.md): −80% boundary jump against independent batches,
   no softening, output 40.5–40.8 dB from the single-window result, about +3–9% compute
   (the VAE is most of the time).
3. **VAE decode** of the shot in one streaming pass, tiled when the planner says so.
4. **Colour correction** against the input (see below).
5. **Write** frames as they come out.

Fallback when latent stitching can't be used: a linear pixel cross-fade over K = 4 frames
(−67% boundary jump, mixed frames ~20% softer).

## Colour correction

`lab` is the recommended mode on quality grounds
([quality.md](../research/docs/quality.md)): it removes the model's colour drift (+30%
saturation, toward blue), halves boundary jumps and cancels the VAE tile drift.
Its first step is StableSR's wavelet split (non-commercial licence). We **rewrite** that split
(an iterated blur separating low and high frequencies, or a Gaussian low-pass), keep numz's LAB
histogram matching (Apache-2.0), and validate against numz's `lab` with the same metrics
(ΔE low-frequency, a*/b* spread, boundary jumps).

## Output

- **FFV1 masters**, every frame a keyframe, per-slice CRCs, exact frame rate, from the float
  frames (no 8-bit step, bugs 09/19):
  - `gbrp16le`: research and archive master, closest to the model
  - `yuv420p10le`, BT.709, limited range, explicit conversion: the master handed to sptenc
    (same layout as `sptenc master`, no implicit RGB→YUV anywhere downstream)

  Validated with the [`ffv1_out.py` wrap](../research/docs/output.md): bit-exact round trip,
  tags checked by ffprobe.
- **PNG** (16-bit) as an alternative.
- Written as **segment files plus a manifest**, concatenated at the end (`sptenc concat`).

## Memory planner

Built into the CLI, from the validated models in [vram.md](../research/docs/vram.md):
- **Budget:** free memory reported by the driver after the CUDA context exists
  (`mem_get_info`), minus a margin. That covers the desktop and other programs without
  guessing. The validated rule was "torch peak ≤ card size − 2 GiB"; the margin over measured
  free memory is to be re-derived from the emulation data (≈ 0.6 GiB).
- **Per-phase peaks** (P = output megapixels, T = tile size in megapixels, L = latent frames
  per window):
  - VAE encode ≈ 1.2 + 8.8·P GiB, decode ≈ 0.8 + 16.1·P GiB (flat beyond 9 frames)
  - tiled: encode ≈ 1.7 + 8.4·T², decode ≈ 1.6 + 15.6·T²
  - DiT (7B fp16) ≈ 16.05 GiB + 128.5 KiB per token (≈ 0.48 GiB per Mpx per latent frame),
    constants per model in vram.md
- **Choices, in order:** model (if the user allows a smaller one), window length (the biggest
  quality lever: fewer boundaries), BlockSwap blocks, VAE tile sizes, then what's left goes to
  speed: `compile_dit` (−26 to −32% DiT time, +0.1 to +1.4 GiB), and `compile_vae` only when
  its memory fits (it about doubles VAE activation memory, for −16 to −19% VAE time).
- `--plan` prints the plan and the time estimate without running.
- Validated with `vram_cap.py` emulation of 8–48 GB cards before release.

### BlockSwap, rewritten
numz moves each block to the GPU and back to pageable memory at every forward pass,
synchronously ([measured](../research/docs/vram.md)). Weights never change, so:
- one **pinned** host copy per block, host→device only (57 GB/s pinned vs 4 GB/s back to
  pageable memory)
- **prefetch** the next block on a side stream while the current one computes

Expected: most of the swap cost hidden. To measure.

### Allocator
`cudaMallocAsync` stays the default (it trims its pool on a full card; `native` fragments and
fails). Set by us, documented, overridable.

## Pause and resume

Work is saved in resumable units; a stop loses only the unit in progress.

| Stage | Saved | Resume granularity |
|---|---|---|
| VAE encode | the shot's latents (~1 MB per 4 frames at 1080p) | per shot |
| DiT | each window's output latents | per window |
| VAE decode | output frames, in segment files | per shot (restart the shot's decode), or per sub-segment with warm-up latents if that proves bit-identical |

- **Manifest:** settings, seed, model hashes, finished shots and windows. A resume with
  different settings is refused.
- **Stopping:** Ctrl-C once finishes the current unit and exits; twice stops at once.
  `--until HH:MM` stops cleanly before a unit that wouldn't finish in time, using the
  planner's time estimates.
- Exiting frees the GPU entirely.
- **Resume must be bit-identical** to an uninterrupted run: deterministic noise per shot, exact
  latents, deterministic attention (FA2 reruns are bit-identical). Automated test: run,
  interrupt, resume, compare the FFV1 masters bit for bit.
- The same manifest lets separate processes take separate shot ranges (several GPUs or
  machines) later.

## Options kept and dropped

| numz option | seedvr2x |
|---|---|
| `--attention_mode` | dropped: FA2 when installed, else SDPA |
| `--color_correction` | `lab` (rewritten) by default; `none` available |
| `--input_noise_scale` | dropped (numz's own addition, only degrades) |
| `--latent_noise_scale` | dropped from v1 (ByteDance uses 0; corrected version possibly later as an experiment, see bug 08) |
| `--batch_size`, `--temporal_overlap`, `--prepend_frames`, `--uniform_batch_size` | replaced by shots, windows and latent stitching, chosen by the planner; window length can be capped by the user |
| `--blocks_to_swap`, tile sizes, offload devices | chosen by the planner; overridable |
| `--compile_dit`, `--compile_vae` | chosen by the planner when memory allows; overridable |
| `--cache_dit`, `--cache_vae`, `--chunk_size` | dropped: one long-running process, streaming by design |
| `--seed` | kept (comparisons, reproducibility) |
| `--output_format`, `--video_backend`, `--10bit` | replaced by FFV1 (`gbrp16le` / `yuv420p10le`) and 16-bit PNG |
| `--resolution`, `--max_resolution` | kept |
| `--dit_model`, `--model_dir` | kept; models identified by hash, no silent deletion (bug 22) |
| `--cuda_device` | one device per process |

## Validation milestones

1. **Reproduce numz:** same settings (one batch, no tiling, `flash_attn_2`, same seed) →
   bit-identical or within quantisation of numz's float frames, checked with the FFV1 masters.
   Then each numerics choice (RoPE precision, attention dtype, VAE mode vs sample) measured
   separately.
2. **Stitching:** reproduce the latent-stitching results from the study.
3. **Planner:** every card size passes under `vram_cap.py` emulation; plan estimates within a
   few percent of measured time and memory.
4. **Resume:** interrupted and resumed runs bit-identical to uninterrupted ones.
5. **Colour correction:** rewritten `lab` matches numz's `lab` on the quality metrics.
6. **Visual review** of long runs by the user.

## Open questions

- **Language split:** all Python, or Go orchestration (sptenc-like: scene splitting, ffmpeg,
  VMAF, concat) around a small Python worker that turns one segment into one master?
- **Scene list format:** what sptenc exports (frame numbers, timestamps), and whether sptenc
  ever splits inside a shot (fades, maximum segment length).
- **Model scope for v1:** 7B fp16 only, or fp8/Q4 and 3B from the start (needed for small
  GPUs).
- **Numerics:** numz's or ByteDance's for RoPE, attention dtype and VAE encode, once measured.
- **Decode resume granularity:** whether sub-segment decoding with warm-up latents is
  bit-identical.
- **4K and long windows:** the planner's limits on large outputs, where the DiT window is the
  constraint.
