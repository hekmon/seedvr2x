# Bugs found in the SeedVR2 CLI

One file per issue found while measuring the
[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) CLI at
commit `4490bd1` (v2.5.24, 2025-12-24): what goes wrong, how to reproduce it, the code path with
`file:line` references, the impact and workaround, and a possible fix with a way to test it.

These are **candidate upstream reports, not yet filed**. Before filing, check that the issue still
exists on the current upstream `main` (line numbers refer to `4490bd1`), and search the existing
issues.

Status: **measured** = reproduced by a run (numbers in the file); **from code** = read from the
code, not reproduced. Severity, from worst: crash, wrong output, memory, performance, UX-doc.

| # | Title | Severity | Status | Summary |
|---|---|---|---|---|
| [01](01-multi-gpu-image-hang.md) | Multi-GPU run on an image hangs forever | crash (hang) | measured | One frame gives fewer chunks than GPUs; the parent waits forever for the missing workers, and for any worker that dies |
| [02](02-cuda-graph-compile-modes-crash.md) | CUDA-graph compile modes crash under `cudaMallocAsync` | crash | measured | `reduce-overhead`, `max-autotune` and the `cudagraphs` backend abort at the first DiT forward under the CLI's default allocator |
| [03](03-image-input-grayscale-and-16-bit.md) | Grayscale images crash, 16-bit images come out white | crash, wrong output | measured | `frame.shape[2]` on a 2-D array; 16-bit divided by 255 |
| [04](04-image-mp4-output-crash.md) | `--output_format mp4` on an image crashes after processing | crash | measured | The image is saved with `cv2.imwrite` to a `.mp4` path |
| [05](05-prepend-frames-not-removed.md) | `--prepend_frames` frames never removed on one GPU | wrong output | measured | 45 frames in, 49 out: removal only exists in the multi-GPU path |
| [06](06-temporal-overlap-blend-weights.md) | `--temporal_overlap` 1, 2, 4 never blend | wrong output | measured | Weights are exactly 1 then 0; 3 and 5 mix one frame 50/50 |
| [07](07-chunk-overlap-not-blended.md) | `--chunk_size` overlap is context only, never blended | wrong output | from code | Hard seam at every chunk boundary despite "seamless" docstrings |
| [08](08-latent-noise-timestep-shift.md) | `--latent_noise_scale` shift from (h, w, c) | wrong output | from code, effect measured | The noise strength depends on resolution only; 4× too strong a shift at 1080p batch 5. Call copied from ByteDance's SeedVR2 scripts, dormant there (scale hard-coded to 0); training convention unknown |
| [09](09-uint8-truncation.md) | Frames truncated to uint8 instead of rounded | wrong output | measured | Dark content half a level too dark |
| [10](10-ffmpeg-untagged-bt601.md) | ffmpeg writes untagged BT.601 YUV | wrong output | measured | Players assuming BT.709 for HD shift greens by 15% |
| [11](11-frame-count-trusted.md) | Container frame count trusted | wrong output | from code | An under-reported count silently drops the last frames |
| [12](12-swap-io-dit-leak.md) | `--swap_io_components` leaks unswapped DiT blocks into decode | memory | measured | +0.42 GiB per unswapped 7B block through decode (+15.2 GiB with no block swapped) |
| [13](13-release-memory-frees-nothing.md) | `release_model_memory` frees nothing | memory | measured | `.data.set_()` empties an alias, not the tensor; the log claims otherwise |
| [14](14-compile-vae-doubles-memory.md) | `--compile_vae` about doubles VAE memory | memory | measured | +71% encode, +97% decode at 1080p; the cause of the 4K OOM; cause not isolated |
| [15](15-compile-vae-exclusion-dead-code.md) | `--compile_vae` conv exclusion is dead code | memory (suspected) | from code | `_dynamo_disable` is an attribute torch never reads |
| [16](16-dit-copied-to-cpu-before-deletion.md) | DiT copied to the CPU before deletion | performance | measured | 3.5 s and ≈ 16 GiB RSS per generation for 7B fp16, for nothing |
| [17](17-blockswap-synchronous-copies.md) | BlockSwap: synchronous pageable copies, no prefetch | performance | measured | 4.3 s of copies per batch for 36 fp16 blocks; the copy back is unnecessary |
| [18](18-attention-modes-misleading.md) | `sageattn_3` never runs SA3; `sageattn_2` runs Triton varlen | UX-doc | measured | The names promise kernels that never run (no speed impact: all backends within ±1.5%) |
| [19](19-10bit-output-is-8-bit.md) | `--10bit` encodes 8-bit frames; ignored with OpenCV | UX-doc | from code | bf16 → uint8 before a 10-bit encode |
| [20](20-dynamo-cache-size-limit-no-effect.md) | `--compile_dynamo_cache_size_limit` has no effect | UX-doc | confirmed (torch config) | Alias of `recompile_limit`, overwritten right after |
| [21](21-cuda-device-nonzero-mask.md) | `--cuda_device N`: mask set after CUDA init | UX-doc | mechanism confirmed, rest from code | Context and memory figures on GPU 0 |
| [22](22-model-lookup-and-validation.md) | Model lookup, architecture by name, deletion on hash mismatch | UX-doc (data-loss risk) | from code | `./models/SEEDVR2` shadows `--model_dir`; "7b" in the name picks the config |
| [23](23-help-text-errors.md) | Help text and log messages that don't match the code | UX-doc | from code | `--output` default, `None` choice, 4n+1, cache "requires", misleading logs |

## Considered and not filed

Measured or read in the notes, but design choices or limitations rather than bugs:

- **Batches are independent**, so each batch boundary is a visible jump on still content
  ([quality.md](../docs/quality.md#batches-and-temporal-consistency)): inherent to batching;
  `--temporal_overlap` (once [06](06-temporal-overlap-blend-weights.md) is fixed) and larger
  batches are the remedies.
- **Per-tile colour drift with VAE tiling** (up to ≈ 2 levels, no seams): GroupNorm and the
  mid-block attention only see the tile; `lab` colour correction removes it
  ([quality.md](../docs/quality.md#vae-tiling-on-flat-areas)).
- **Multi-GPU without `--temporal_overlap` concatenates with a hard seam**, and the parent gathers
  the whole output in RAM ([cli-flags.md](../docs/cli-flags.md#multi-gpu)): documented trade-offs.
- **Every batch gets the same diffusion noise**, and `RANK` is added to the seed: deliberate,
  for reproducibility and distributed runs.
- **`--compile_dynamic` off by default** (recompiles per shape): a choice; the recompile limit is
  generous.
- **The `cudaMallocAsync` pool keeps 10–25% above the torch peak** on a large card: it trims
  itself on a full card ([vram.md](../docs/vram.md#the-allocator)); not a leak.
- **The default DiT is the 3B fp8 model**: a choice, documented in
  [cli-flags.md](../docs/cli-flags.md#models).
- **`--compile_fullgraph` with BlockSwap** should fail (graph breaks in the swap wrappers): an
  incompatible combination, not run.
- **The Conv3d workaround** changes nothing on torch 2.14.1 / cuDNN 9.24
  ([vram.md](../docs/vram.md#other-knobs)): harmless.
