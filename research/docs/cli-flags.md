# CLI options: what each one really does

> Status: **read from the code** (SeedVR2 `4490bd1`, `inference_cli.py` and `src/`), with links to
> the notes where an effect was measured. Lists all 44 options of `inference_cli.py` plus the
> positional input, grouped by theme. Main findings:
> - `--prepend_frames` is never removed from the output of a single-GPU run (measured: 10 frames
>   in, 11 out; 45 in, 49 out in [quality.md](quality.md#--prepend_frames)).
> - `--swap_io_components` leaks the DiT into the decode phase unless every block is swapped.
> - `--temporal_overlap` 1, 2 and 4 switch hard from one batch to the next instead of blending
>   (measured in [quality.md](quality.md#--temporal_overlap)).
> - `--compile_vae` doubles the VAE's memory: it caused the 4K OOM
>   ([vram.md](vram.md#the-4k-oom-reproduced)).
> - The default DiT is the 3B fp8 model, not the 7B fp16 the other notes measure ([Models](#models)).
> - `--10bit` only re-encodes 8-bit frames.
> - `--compile_dynamo_cache_size_limit` is overwritten by `--compile_dynamo_recompile_limit`.
>
> Many combinations are ignored or overridden without an error ([table](#ignored-and-overridden-combinations)).
> Items marked *(code)* come from reading the code and were not run.

## How a run uses the options

`main()` parses the options and downloads the models. It then calls `process_single_file` for the
input, or for each file of an input directory. OpenCV reads a video in chunks of `--chunk_size`
frames (the whole clip by default). Each chunk goes through `_process_frames_core`, which builds the
runner and then runs the four phases:

| Step | Function | Options read there |
|---|---|---|
| Setup | `prepare_runner` → `configure_runner` (`src/core/model_configuration.py`) | `--dit_model`, `--model_dir`, offload devices, BlockSwap, VAE tiling, `--attention_mode`, compile, caching |
| Frame preparation | `compute_generation_info` (`src/core/generation_utils.py`) | `--prepend_frames`, `--resolution`, `--max_resolution` |
| 1. VAE encode | `encode_all_batches` (`src/core/generation_phases.py`) | `--batch_size`, `--uniform_batch_size`, `--temporal_overlap`, `--input_noise_scale`, `--seed` |
| 2. DiT | `upscale_all_batches` | `--seed`, `--latent_noise_scale` |
| 3. VAE decode | `decode_all_batches` | `--temporal_overlap` (blending), `--tensor_offload_device` |
| 4. Post-processing | `postprocess_all_batches` | `--color_correction`, `--tile_debug` |

No option controls the following:
- The sampler: one step, CFG forced to 1.0, and the bundled text embedding (`pos_emb.pt`, 58 tokens).
- The VAE (`ema_vae_fp16.safetensors`) and its temporal slicing (`vae.slicing` in the config, see [vram.md](vram.md#how-the-vae-processes-a-batch)).
- The compute dtype: bfloat16, or float16 if a bf16 cuBLAS matmul probe at import fails.

## Options

Links to [quality.md](quality.md) in the "Measured" column give the measured effect on output quality.

### Input and output

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `input` (positional) | path | A video (`.mp4 .avi .mov .mkv .webm .flv .wmv .m4v`), an image (`.png .jpg .jpeg .bmp .tiff .tif .webp`) or a directory, matched on the extension. A file with any other extension is an error | See [Input handling](#input-handling) | – |
| `--output` | path, none | **None:** the output goes next to the input as `<stem>_upscaled.mp4`, `<stem>_upscaled/` (PNG sequence) or `<stem>_upscaled.png` (image). For a directory input it goes to a sibling `<dir>_upscaled/` and keeps the original names. **Treated as a directory** when it has no extension, when the input is a directory, or for a video written as PNG; any dot counts as an extension, so a video written to `out/clip-x1.5/` fails at the save ([bug 26](../bugs/26-output-directory-with-a-dot.md)). It then holds `<stem>.mp4`, `<stem>.png` or a `<stem>/` PNG directory, without the `_upscaled` suffix. Existing files are overwritten | The help's "auto-generated in 'output/' directory" is wrong | – |
| `--output_format` | `mp4` / `png`, auto | **Auto:** `mp4` for a video, `png` for an image, decided per file in a directory. **`png` on a video:** `<stem>_NNNNNN.png`, 6 digits, 8-bit. **`mp4` on an image:** the path gets `.mp4`, but the image is still saved with `cv2.imwrite`, which fails after all the processing *(code)* | The help lists `None` as a choice, but argparse rejects the string `None` | – |
| `--video_backend` | `opencv` / `ffmpeg`, `opencv` | **`opencv`:** `cv2.VideoWriter`, fourcc `mp4v` (MPEG-4 Part 2). **`ffmpeg`:** RGB24 piped to `ffmpeg -c:v libx264 -pix_fmt yuv420p -preset medium -crf 12`, with stderr discarded. With `ffmpeg`, the CLI checks at start-up that the binary is on `PATH` | `mp4v` at OpenCV's default quality is much lossier than x264 at CRF 12. For comparisons, use `ffmpeg` or PNG | not measured |
| `--10bit` | flag | With `ffmpeg`: `libx265`, `yuv420p10le`, CRF 12. **Silently ignored with `opencv`** | The frames are 8-bit before encoding ([Output](#output)): only the RGB→YUV step gains precision | not measured |

### Model

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--model_dir` | path, `./models/SEEDVR2` | Where models are looked up and downloaded. The default is relative to the **current directory**, not to the script. `./models/SEEDVR2` is always searched first, even when `--model_dir` is set ([Models](#models)) | – | – |
| `--dit_model` | name, `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | Selects the DiT checkpoint, downloaded at start-up if missing. The architecture comes from the **file name**: a name containing `7b` gets the 7B config, any other name the 3B config | Weight memory, speed, output | [vram.md](vram.md#other-models) |

### Resolution, frames and batching

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--resolution` | int, 1080 | Target length of the output's short side. Bicubic antialiased resize, **up or down** (`downsample_only=False`), run on the GPU in bf16 after the batch is moved there. The size is rounded down to even numbers, then zero-padded to a multiple of 16 for processing and cropped back after decode | Every phase scales with the output pixel count | [vram.md](vram.md#the-picture-per-phase), [practical rules](vram.md#practical-rules) |
| `--max_resolution` | int, 0 (off) | If the long edge after the `--resolution` resize exceeds this value, a **second** bicubic resize brings it down. The frame is resampled twice | Lowers the output size | not measured |
| `--batch_size` | int, 5 | Frames per batch through all four phases. **Not validated.** A batch that isn't 4n+1 is padded to the next 4n+1 with mirrored frames, which are dropped after decode: batch 8 computes 9 frames per 8. A larger batch means longer DiT windows in time and more temporal context. A tip logs the largest 4n+1 ≤ the frame count | DiT: peak = weights + 128.5 KiB per token; cost per frame ∝ (1 + (b − 1)/4) / b. VAE peak stops growing from 9 up | [vram.md](vram.md#weights-and-the-per-token-model), [VAE](vram.md#untiled-vae-the-model), [attention.md](attention.md#windowed-attention); [quality.md](quality.md#batches-and-temporal-consistency) |
| `--uniform_batch_size` | flag | Pads the last, shorter batch to `--batch_size` with frames mirrored at its end, instead of only to the next 4n+1. The padding is dropped after decode. No effect when the frames fill whole batches | Up to one extra batch of compute; same peak as a full batch | [vram.md](vram.md#offload-devices) (from the code); [quality.md](quality.md#--uniform_batch_size) |
| `--skip_first_frames` | int, 0 | Videos only: seeks the capture to frame N (`CAP_PROP_POS_FRAMES`). With several GPUs, the split starts there | – | used by most measured runs |
| `--load_cap` | int, 0 (all) | Videos only: processes at most N frames after the skipped ones | – | – |
| `--chunk_size` | int, 0 (off) | **Streaming**, videos only. Reads and processes N frames at a time; each chunk is a full four-phase run, and its output is written before the next chunk is read. `--temporal_overlap` raw frames from the end of the previous chunk are prepended as context, and their output is discarded. **There is no blending between chunks.** Batches restart in each chunk, so a chunk that isn't a whole number of batches ends with a short batch | Bounds host RAM, not VRAM. Without `--cache_dit` / `--cache_vae`, the models are re-read from disk for every chunk | [vram.md](vram.md#offload-devices) (from the code); not measured |

### Temporal

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--temporal_overlap` | int, 0 | **Three roles:**<br>1. Between the batches of a chunk: batch k starts at k × (batch − overlap). In Phase 3, the first `overlap` frames of each batch after the first are blended with the last frames already written (`blend_overlapping_frames`). A last batch made only of overlap frames is skipped. If overlap ≥ batch size, it is reset to 0 for batching, with a warning.<br>2. Between `--chunk_size` chunks: context frames, not blended (see above).<br>3. Between GPUs: each worker's range is extended, and the parent blends the results.<br>**Blend weights:** linear for overlap < 3, otherwise a raised cosine over the **middle third** of the overlap. So overlap 1 keeps the previous batch's frame, and overlap 2 and 4 switch hard at the middle: no frame is mixed. Overlap 3 and 5 mix one frame 50/50, overlap 6 mixes two frames (≈ 0.9/0.1, 0.1/0.9), overlap 9 mixes three | Each overlap frame is encoded, upscaled and decoded twice: compute × batch / (batch − overlap). No memory effect | [vram.md](vram.md#the-4k-oom-reproduced) (only the batch count changes); [quality.md](quality.md#--temporal_overlap) (1, 2, 4 leave the boundary jump unchanged; 3 cuts it by about a third); [stitching.md](stitching.md) (corrected weights, latent-space stitching) |
| `--prepend_frames` | int, 0 | Prepends N frames mirrored around frame 0 (frames N…1, then 0…), or repeats the last frame when N ≥ the frame count. Only the first chunk and the first GPU get them. **They are removed only in the multi-GPU path** (`_gpu_processing`). `_process_frames_core` passes `prepend_frames=0` to Phase 4, with the comment "Worker mode handles this in main process". **On one GPU the output keeps them:** it has N extra mirrored frames at the start, and the frame timing shifts by N | N more frames to process | **Bug, measured:** `vae-2160-bs5-ov1-pp1`, 10 input frames with `--prepend_frames 1`, 11 output frames (log and `ffprobe`); [quality.md](quality.md#--prepend_frames): 45 in, 49 out with `--prepend_frames 4` |

### Quality

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--color_correction` | `lab` / `wavelet` / `wavelet_adaptive` / `hsv` / `adain` / `none`, `lab` | Phase 4, batch by batch, on the GPU. The reference is the input batch resized to the output size, rebuilt from the input frames. **`wavelet`:** the output's high frequencies plus the input's low frequencies (5-level blur pyramid). **`lab`:** `wavelet`, then CIELAB histogram matching of a\* and b\*, and L\* = 0.8 × output + 0.2 × matched. The histograms pool all frames of a batch, so the mapping can change from one batch to the next. **`wavelet_adaptive`:** `wavelet`, plus HSV saturation matching blended in where the output is > 0.15 more saturated than the input. **`hsv`:** saturation histograms matched per 30° hue bin. **`adain`:** per-frame, per-channel mean and std set to the input's. **`none`:** Phase 4 only rescales to [0, 1]. **Licence:** the wavelet functions (`wavelet_blur`, `wavelet_decomposition`, `wavelet_reconstruction`) and the AdaIN ones (`calc_mean_std`, `adaptive_instance_normalization`) in `src/utils/color_fix.py:72-247` are adapted without attribution from StableSR's `colorfix.py` (S-Lab License 1.0, non-commercial; sd-webui-stablesr, which ByteDance's readme points to for it, is also CC BY-NC-SA 4.0). `lab` (`:280`), `wavelet`, `wavelet_adaptive` (`:812`) and `adain` all run that code; only `hsv` and `none` are free of it. ByteDance doesn't ship the file | `lab` at 1080p: 5.0 s for one 9-frame batch on the first call (1.8 GiB peak), 0.5 s for 21 frames in a warm run (batch 5), 0.6 s and 4.26 GiB per 21-frame batch. `lab` removes the model's colour drift without losing detail; `hsv` and `adain` are worse | [environment.md](environment.md#smoke-test-9-frames-19201080--19201080-7b-fp16-sageattn_3-no-offload-no-tiling), [benchmarking.md](benchmarking.md#reference-two-runs-on-the-reference-stack); [quality.md](quality.md#colour-correction) |
| `--input_noise_scale` | float, 0.0 (range not checked) | Phase 1, on the resized input normalized to [−1, 1]: x + 0.025 × scale × N(0, 1) (the code blends x and x + 0.05·noise with weight scale/2). At scale 1 that is Gaussian noise of ≈ 3.2 8-bit levels on the VAE input | No memory or time cost | [quality.md](quality.md#noise-scales) |
| `--latent_noise_scale` | float, 0.0 (range not checked) | Phase 2: moves the encoded input latent, which the DiT uses as its condition, toward a small noise (0.1 × the sampling noise + 0.05 × fresh noise, std ≈ 0.11): x' = (1 − t)·x + t·n, with t = shift·s / (1 + (shift − 1)·s), the sampler's resolution-dependent timestep shift. **The shift is computed from `x.shape[1:]` of a (t, h, w, c) latent**, i.e. from (h, w, c) instead of (frames, h, w), a call copied from ByteDance's SeedVR2 scripts, where the scale is hard-coded to 0. It is ≈ 2.7 at 720p, 5.0 at 1080p and 17 at 4K whatever the batch size, so s = 0.1 already gives t ≈ 0.36 at 1080p. With (frames, h, w) the shift would be ≈ 1.2 at 1080p batch 5 *(code)* | No cost. Measured: 0.1 already washes the image out | [quality.md](quality.md#noise-scales) |
| `--seed` | int, 42 | `set_seed` seeds Python, NumPy and torch with seed + 1 000 000 before Phase 1 (VAE posterior sampling, input noise), and with seed before **every** DiT batch: batches of the same shape get the same noise. The environment's `RANK` is added to the seed. Every chunk and every GPU worker restarts from the same seed | Two identical runs give bit-identical output | [vram.md](vram.md#quality-tiled-vs-untiled-1080p-batch-9-png-output); [quality.md](quality.md#seed) |

### Devices, offload and BlockSwap

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--cuda_device` | `N` or `N,M,…`, none (= `0`) | Checked against the device count before the main torch import, exit 1 if invalid (skipped when `CUDA_VISIBLE_DEVICES` is set). One device: sets `CUDA_VISIBLE_DEVICES` and runs in-process on `cuda:<N>`. Several: one spawned worker per GPU ([Multi-GPU](#multi-gpu)). Absent on macOS | For a single non-zero device, see [Multi-GPU](#multi-gpu) *(code)* | not measured (one GPU) |
| `--dit_offload_device` | `none` / `cpu` / GPU id / `cuda:N`, `none` | Where the DiT is materialized and parked. **`none`:** loaded straight to the GPU at Phase 2. **`cpu`:** loaded to the CPU, moved to the GPU at Phase 2 and back at its end. **Required by BlockSwap:** `ValueError` if missing or equal to the inference device. `--cache_dit` turns `none` into `cpu` | Alone it saves nothing (the DiT is never on the GPU during the VAE phases) and costs +0.7 s | [vram.md](vram.md#offload-devices) |
| `--vae_offload_device` | same, `none` | The VAE moves there after encode, back to the GPU before decode, and there again after decode. `--cache_vae` turns `none` into `cpu` | −0.47 GiB on the DiT peak, +0.2 s | [vram.md](vram.md#offload-devices) |
| `--tensor_offload_device` | `cpu` / `none` / GPU id, `cpu` | Where latents wait between phases (and an RGBA input's alpha and RGB guides), and where tiled VAE results accumulate. **`none`:** they stay on the GPU, but the decoded frames still go to the CPU (`final_video` falls back to `cpu`). On MPS, `cpu` is changed to `none` | `none`: nothing measurable at 1080p (+0.1 GiB in decode). The tiled accumulation on the GPU grows with the resolution | [vram.md](vram.md#offload-devices) |
| `--blocks_to_swap` | int, 0 | Blocks 0…N−1 stay on `--dit_offload_device`. Every forward moves each of them to the GPU and back, synchronously. Clamped to the block count (36 for 7B, 32 for 3B); ≤ 0 disables it. Needs `--dit_offload_device` (`ValueError` otherwise, raised after the input is read). Disabled with a warning on macOS | 7B fp16: −0.41 GiB and +0.07–0.10 s per block and batch. Cheaper with fp8 / GGUF. The swapped weights add to host RAM. `--compile_dit` gains nothing with it | [vram.md](vram.md#blockswap) |
| `--swap_io_components` | flag | Same treatment for `vid_in`, `txt_in`, `emb_in` and `vid_out` (0.16 GiB for 7B). Needs `--dit_offload_device`. **Leak:** `cleanup_dit` looks at the first parameter (`vid_in`, now on the CPU) and skips moving the model off the GPU. `release_model_memory` then logs "Released memory from 1080 params" but frees nothing (`param.data.set_()` empties an alias, not the parameter). Every block left on the GPU stays allocated through decode | Saves 0.16 GiB for ≈ 0.08 s per batch. Without `--blocks_to_swap`, decode needs +15.2 GiB. With any count below the block total, the unswapped blocks leak the same way *(code)* | [vram.md](vram.md#blockswap) |

### VAE tiling

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--vae_encode_tiled` | flag | Spatial tiling of Phase 1. Each tile goes through the full temporal slicing, and the overlaps are blended with a cosine ramp. Skipped when the padded frame fits in one tile | The peak depends on the tile size only. Per-tile colour drift, no seams; encode tiling costs more PSNR than decode tiling. `lab` colour correction removes the drift | [vram.md](vram.md#tiling), [quality](vram.md#quality-tiled-vs-untiled-1080p-batch-9-png-output), [flat areas](quality.md#vae-tiling-on-flat-areas) |
| `--vae_encode_tile_size` | int, 1024 | Square tile in output pixels, rounded down to a multiple of 8 (one latent pixel). Ignored without `--vae_encode_tiled` | Encode ≈ 1.7 + 8.4 × T² GiB (T² in Mpx, batch ≥ 9) | [vram.md](vram.md#tiling) |
| `--vae_encode_tile_overlap` | int, 128 | In output pixels, rounded down to a multiple of 8. Must be < the tile size when tiling is on (exit 1), not checked otherwise | Costs time (tiled area), not memory; 64 is enough | [vram.md](vram.md#tiling) |
| `--vae_decode_tiled` | flag | Same for Phase 3; the "fits in one tile" check is done in latent pixels | Decode ≈ 1.6 + 15.6 × T² GiB (batch ≥ 9) | [vram.md](vram.md#tiling) |
| `--vae_decode_tile_size` | int, 1024 | As for encode | – | [vram.md](vram.md#tiling) |
| `--vae_decode_tile_overlap` | int, 128 | As for encode | – | [vram.md](vram.md#tiling) |
| `--tile_debug` | `false` / `encode` / `decode`, `false` | Phase 4 **draws numbered tile outlines into the output frames**. Only when that phase is tiled and was actually split | Debug only: the overlay stays in the output | not measured |

### Attention and `torch.compile`

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--attention_mode` | `sdpa` / `flash_attn_2` / `flash_attn_3` / `sageattn_2` / `sageattn_3`, `sdpa` | Set on every `FlashAttentionVarlen` module after `validate_attention_mode`. A backend that can't be imported **falls back silently**. `sageattn_3` runs SA2 on every variable-length call, and the DiT's window attention only makes such calls | All within ±1.5% of DiT time. SageAttention costs +1.3–3 GiB. `flash_attn_2` is the best choice | [attention.md](attention.md#backend-comparison-clean-runs), [fallbacks](environment.md#2-seedvr2-falls-back-silently-between-attention-backends) |
| `--compile_dit` | flag | `torch.compile` of the whole DiT, applied after BlockSwap. The compilation runs at the first forward; the CLI's "falling back to uncompiled" only catches errors at wrap time, so a compilation error aborts the run | −26 to −32% DiT time for 10–50 s of compilation and +0.1–1.4 GiB (grows with the batch). Nothing gained with BlockSwap | [vram.md](vram.md#torchcompile) |
| `--compile_vae` | flag | Compiles the VAE `encoder` and `decoder` submodules. The helper meant to exclude the causal 3D convs sets a `_dynamo_disable` attribute that torch never reads, so they are compiled too *(code)* | ≈ 2× VAE memory and −16–19% VAE time, after minutes of compilation. It caused the 4K OOM | [vram.md](vram.md#torchcompile), [4K OOM](vram.md#the-4k-oom-reproduced) |
| `--compile_backend` | `inductor` / `cudagraphs`, `inductor` | One setting shared by DiT and VAE. **`inductor`** needs Triton. Without it, the `RuntimeError` is caught by the CLI's fallback: an ERROR and a WARNING are logged and the model runs **uncompiled**. **`cudagraphs`** records CUDA graph trees without generating kernels. It ignores `--compile_mode`: torch passes `mode=` to it and it logs "cudagraphs backend ignoring extra kwargs" *(code)* | – | not measured |
| `--compile_mode` | `default` / `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs`, `default` | Passed to `torch.compile`. `max-autotune` uses CUDA graphs and crashes under the CLI's default `cudaMallocAsync` allocator. `reduce-overhead` (CUDA graphs too) and the `cudagraphs` backend go through the same CUDA graph trees and should fail the same way *(code)* | – | `max-autotune` crash: [environment.md](environment.md#runtime-notes-that-depend-on-the-environment); others not measured |
| `--compile_fullgraph` | flag | `fullgraph=True`: any graph break is an error at the first forward. BlockSwap's wrappers break the graph (`@torch._dynamo.disable` timing helpers, block moves), so it should fail with BlockSwap *(code)* | – | not measured |
| `--compile_dynamic` | flag | `dynamic=True`: symbolic shapes. The default `False` is stricter than torch's own default (automatic dynamic after the second shape): every new shape recompiles until the recompile limit, then that code runs eager. New shapes include a shorter last batch, another resolution, or the next file of a directory | – | recompilation seen in [vram.md](vram.md#torchcompile) |
| `--compile_dynamo_cache_size_limit` | int, 64 | Sets the global `torch._dynamo.config.cache_size_limit`. On torch 2.14 that is an alias of `recompile_limit`, which the next option sets right after: **this option has no effect** | – | – |
| `--compile_dynamo_recompile_limit` | int, 128 | Sets the global `torch._dynamo.config.recompile_limit` (torch's default is 8) | – | – |

### Model caching

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--cache_dit` | flag | Keeps the DiT between the generations of one process: the files of a directory on one GPU, or `--chunk_size` chunks (on one GPU, or inside each multi-GPU worker). **Ignored elsewhere:** single file without streaming (a tip is logged), several GPUs without `--chunk_size` (disabled with a warning). The DiT is parked on `--dit_offload_device`, which defaults to `cpu`: it can't stay on the GPU between files. Keeps the BlockSwap setup and the compiled model | Saves re-reading the checkpoint (1.2 s for 7B fp16 from the page cache) and re-applying BlockSwap and compile. It doesn't save the GPU→CPU copy at the end of Phase 2, which happens in both cases (3.5 s for 7B fp16). With a warm page cache and no compile, the gain is small *(from measured parts)* | not measured |
| `--cache_vae` | flag | Same for the VAE. Because it defaults `--vae_offload_device` to `cpu`, the VAE also leaves the GPU between encode and decode | – | not measured |

### Debugging

| Option | Type, default | What the code does | Effect | Measured |
|---|---|---|---|---|
| `--debug` | flag | Verbose log: the "Arguments" block, per-phase timers and their `└─` breakdowns, memory snapshots (torch allocated / reserved / peak of `cuda:0`, process RSS; the peak counter is reset after each snapshot), and the BlockSwap summary. Adds no CUDA synchronization | The source of every per-phase figure in these notes | overhead not measured; [benchmarking.md](benchmarking.md#the-record) |
| `-h`, `--help` | – | Prints the help. Running the CLI without arguments does the same | – | – |

## Input handling

- **Videos** are read by OpenCV (`cv2.VideoCapture`) and converted to **8-bit RGB**. A 10-bit or HDR
  source loses its extra precision and its colour metadata before processing. The audio is ignored.
- The frame count comes from `CAP_PROP_FRAME_COUNT`, and the reader stops after that many frames
  (minus `--skip_first_frames`, capped by `--load_cap`). If the container under-reports the count,
  the last frames are silently dropped *(code)*. The frame rate is `CAP_PROP_FPS`, or 30 when it
  reads 0.
- Frames are float32 while being read (`np.stack` briefly doubles that), then float16. Each batch
  is moved to the GPU in bf16 and resized there.
- **Images** are read with `cv2.imread(IMREAD_UNCHANGED)`. RGB and RGBA work. With RGBA, the alpha
  channel is upscaled separately ("edge-guided", `src/core/alpha_upscaling.py`) and colour
  correction applies to RGB only. *(code)* problems:
  - A grayscale image is a 2-D array, and the code indexes `shape[2]`: `IndexError`.
  - A 16-bit PNG or TIFF is divided by 255 instead of 65535 and then clamped: the output is nearly
    white.

  An image is a one-frame batch. `--skip_first_frames`, `--load_cap` and `--chunk_size` don't
  apply to images.
- **Directories** are read without recursion, in name order. Only the extensions above are kept
  (case-insensitive) and the output format is chosen per file. A failing file aborts the whole run.

## Output

- **The frames are 8-bit at best.** The final frames are a bfloat16 tensor in [0, 1]: 8 significant
  bits, so steps of 1/256 above mid-grey. They are converted to float32, then
  `(x * 255).astype(uint8)` **truncates** instead of rounding: on average half a level darker than
  rounding. `--10bit` therefore encodes 8-bit RGB into 10-bit YUV: it avoids the rounding of the
  RGB→YUV step, but adds no source precision.
- **ffmpeg:** rawvideo `rgb24` in, x264 (or x265 with `--10bit`) out, CRF 12, preset `medium`. The
  stream gets no colour tags and swscale's default matrix converts RGB to YUV. Most players assume
  BT.709 for HD, so a small colour shift is possible *(code, not measured)*. ffmpeg's stderr is
  discarded: a non-zero exit code only produces a WARNING.
- **opencv:** fourcc `mp4v`.
- **PNG:** 8-bit RGB(A), `<stem>_000000.png`. For an image input, `cv2.imwrite` picks the format from
  the extension of `--output` (`.jpg` writes a JPEG).
- Output size: short side = `--resolution`, rounded down to even. The frame rate is the input's,
  and there is no audio.
- For comparisons, [`ffv1_out.py`](../scripts/ffv1_out.py) writes a lossless 16-bit RGB master from
  the float frames, rounded and tagged ([output.md](output.md)).

## Models

| `--dit_model` | Size | Hugging Face repo | Notes |
|---|---|---|---|
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3.4 GB | `numz/SeedVR2_comfyUI` | **Default** |
| `seedvr2_ema_3b_fp16.safetensors` | 6.8 GB | `numz/SeedVR2_comfyUI` | |
| `seedvr2_ema_3b-Q8_0.gguf` | 3.7 GB | `AInVFX/SeedVR2_comfyUI` | not tested |
| `seedvr2_ema_3b-Q4_K_M.gguf` | 2.0 GB | `AInVFX/SeedVR2_comfyUI` | not tested |
| `seedvr2_ema_7b_fp16.safetensors` | 16.5 GB | `numz/SeedVR2_comfyUI` | Reference model of these notes |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 8.5 GB | `AInVFX/SeedVR2_comfyUI` | |
| `seedvr2_ema_7b-Q4_K_M.gguf` | 4.8 GB | `AInVFX/SeedVR2_comfyUI` | |
| `seedvr2_ema_7b_sharp_fp16.safetensors` | 16.5 GB | `numz/SeedVR2_comfyUI` | "sharp" variant, not tested |
| `seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 8.5 GB | `AInVFX/SeedVR2_comfyUI` | not tested |
| `seedvr2_ema_7b_sharp-Q4_K_M.gguf` | 4.8 GB | `AInVFX/SeedVR2_comfyUI` | not tested |
| VAE `ema_vae_fp16.safetensors` (no option) | 0.5 GB | `numz/SeedVR2_comfyUI` | Shared by all |

- **The default DiT is the 3B fp8**, not the 7B fp16 used in the other notes. The 3B is a different
  model (32 dB from the 7B), only 25% faster ([vram.md](vram.md#other-models)). 7B: 36 blocks; 3B:
  32.
- **Download:** at start-up, before the input is read, from
  `https://huggingface.co/<repo>/resolve/main/<file>`:
  - resumable (`<file>.download`), 3 attempts, sha256 checked;
  - a validated file is recorded in `.validation_cache.json` (size, mtime), so it isn't hashed again
    on later runs;
  - **a file with a registry name but another hash is deleted and downloaded again**;
  - after a download, the cache entry goes to `./models/SEEDVR2/.validation_cache.json` in the
    current directory instead of `--model_dir`. The directory gets created, and the next run hashes
    the file once more *(code)*.
- **Lookup:** `./models/SEEDVR2` in the current directory (or ComfyUI's model paths when running
  inside ComfyUI) comes first; `--model_dir` is only the fallback. A file in the former shadows the
  latter.
- **Choices:** argparse builds the `--dit_model` choices before parsing: the registry, plus model
  files found in `./models/SEEDVR2`, **not** in `--model_dir`. A custom checkpoint is accepted only
  if it sits in `./models/SEEDVR2` of the current directory. It isn't hash-checked, and its name
  picks the 3B or 7B architecture.
- Precision: fp16 DiT weights stay fp16 and run under bf16 autocast. fp8 and GGUF weights stay
  quantized and are converted per layer. The VAE weights are converted to bf16 at load.

## Multi-GPU

`--cuda_device 0,1,…` goes through `_gpu_processing`. Nothing here could be measured on the one-GPU
reference host; this section comes from the code.

- One process is spawned per GPU, with `CUDA_VISIBLE_DEVICES` set for each. Each worker loads its
  own models, so each GPU needs the full single-GPU memory.
- **Videos:** the frame range is split evenly, and every worker but the last gets
  `--temporal_overlap` extra frames. Each worker reads its own segment and applies `--chunk_size`
  inside it. The results come back through shared memory and are joined in the parent:
  - with `--temporal_overlap`: blended over those frames (same weights as between batches);
  - without it: concatenated, which leaves a hard seam with no shared context.

  `--prepend_frames` applies to worker 0 only and is removed in the parent: this is the only path
  where the removal works.
- The parent gathers the **whole output in RAM** before writing it: here `--chunk_size` bounds the
  workers' input, not the output.
- Caching works only with `--chunk_size`, inside each worker.
- **Images hang.** `torch.chunk` of a one-frame tensor gives fewer chunks than GPUs, so fewer
  workers start, and the parent waits forever for the missing results. With `--temporal_overlap`
  the split depends on the batch size and can give a worker an empty chunk. That worker crashes
  without returning anything, and the parent waits again. The same happens to the images of a
  directory.
- The parent also keeps a CUDA context on GPU 0 (import-time probes, see below).

**A single non-zero device (`--cuda_device 1`):**
- The validation calls `torch.cuda.is_available()`, which runs `cuInit`, before
  `CUDA_VISIBLE_DEVICES=1` is set. The mask therefore has no effect in that process: all GPUs stay
  visible.
- The run still lands on GPU 1, because the code then uses `cuda:1`. Had the mask worked, `cuda:1`
  would not exist, and that is what to expect with `PYTORCH_NVML_BASED_CUDA_CHECK=1`.
- Module imports have already created a context on GPU 0: the "Initial CUDA memory" line
  (`mem_get_info`) and the bf16 probe.
- The memory figures of `--debug` and BlockSwap's "< 5% free" check read `cuda:0`, i.e. the wrong
  GPU.

## Environment variables

| Variable | Set or read | Effect |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | Set with `setdefault` to `backend:cudaMallocAsync` before torch is imported (not on macOS) | A value you set wins. Effects measured in [vram.md](vram.md#the-allocator); CUDA graphs fail under this default ([environment.md](environment.md#runtime-notes-that-depend-on-the-environment)) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `setdefault` 0.0 (macOS only) | No MPS memory cap |
| `CUDA_VISIBLE_DEVICES` | Read: when set, `--cuda_device` is neither validated nor applied. Set: for a single device, and per worker | See [Multi-GPU](#multi-gpu) |
| `PYTHONPATH` | Script directory prepended | Lets the spawned workers import `src` |
| `LOCAL_RANK` | `setdefault` 0 | – |
| `RANK`, `WORLD_SIZE` | Read by `src/common/distributed` | `RANK` is added to every seed: a stray `RANK` in the environment changes the output |
| `SEEDVR2_OPTIMIZATIONS_LOGGED` | Set to 1 | Prints the "optimizations check" banner once per process tree |
| `PYTORCH_NVML_BASED_CUDA_CHECK` | Read by torch | Changes how `--cuda_device N` behaves ([Multi-GPU](#multi-gpu)) |

The CLI also forces the multiprocessing start method to `spawn`. The imports themselves touch GPU
0: `mem_get_info` prints the initial free memory, a bf16 matmul probe sets the compute dtype, and
the Conv3d workaround check reads the cuDNN version
([environment.md](environment.md#runtime-notes-that-depend-on-the-environment)).

## Host memory

From the code, not measured separately (max RSS figures are in [vram.md](vram.md#blockswap)):

- Without `--chunk_size`, the whole clip is in RAM: the input frames (float32 while reading, then
  float16), and the output as a bf16 tensor. At the end that tensor is converted to float32, and
  writing it adds a float32 ×255 copy and a uint8 array: about 0.2 GB per 4K output frame or 50 MB
  per 1080p frame at that point.
- At the end of Phase 2, the DiT is copied to the CPU before being deleted (≈ 16 GiB for 7B fp16).
  This happens in every run, cached or not.
- BlockSwap keeps the swapped blocks in RAM.
- With `--chunk_size` on one GPU, input and output are bounded per chunk: each chunk is written
  before the next one is read.

## Bugs and surprises

Each issue has a write-up with a possible fix in [bugs/](../bugs/README.md), which also covers
issues not listed here: CUDA-graph compile modes, the `--chunk_size` overlap, untagged ffmpeg
colours, `--compile_vae` memory, BlockSwap copies, attention modes and model lookup.

| Issue | When | Consequence | Evidence |
|---|---|---|---|
| [Prepended frames stay in the output](../bugs/05-prepend-frames-not-removed.md) | `--prepend_frames N`, one GPU | N extra mirrored frames at the start | Measured: 10 frames in, 11 out; 45 in, 49 out ([quality.md](quality.md#--prepend_frames)) |
| [`--swap_io_components` leaks the DiT](../bugs/12-swap-io-dit-leak.md) | Without swapping every block | The unswapped blocks stay on the GPU through decode: +15.2 GiB with `--blocks_to_swap 0` | Measured: +15.2 GiB with 0 blocks swapped ([vram.md](vram.md#blockswap)), +7.6 GiB with 18 of 36 |
| [`release_model_memory` frees nothing](../bugs/13-release-memory-frees-nothing.md) | Always (it only matters when the model wasn't moved off the GPU first) | `param.data.set_()` empties an alias; the log still says "Released memory from N params" | Seen in the leak's log; checked standalone on torch 2.14.1 |
| [DiT copied to the CPU before deletion](../bugs/16-dit-copied-to-cpu-before-deletion.md) | Every run | 3.5 s and ≈ 16 GiB of RSS for 7B fp16, with nothing reusing the copy | Measured ([vram.md](vram.md#weights-and-the-per-token-model)) |
| [Overlap without blending](../bugs/06-temporal-overlap-blend-weights.md) | `--temporal_overlap` 1, 2, 4 | Hard switch: the overlap costs compute but gives no crossfade | Measured ([quality.md](quality.md#--temporal_overlap)): boundary jump unchanged; the weights are in the code |
| [`--10bit`](../bugs/19-10bit-output-is-8-bit.md) | Always | 8-bit frames encoded as 10-bit | *(code)* |
| [Truncation to uint8](../bugs/09-uint8-truncation.md) | Always | Half a level darker on average than rounding, below mid-grey | *(code)* |
| [`--compile_dynamo_cache_size_limit`](../bugs/20-dynamo-cache-size-limit-no-effect.md) | torch 2.14 | No effect (alias overwritten) | Checked on torch 2.14.1 |
| [`--compile_vae` exclusions](../bugs/15-compile-vae-exclusion-dead-code.md) | `--compile_vae` | The causal convs it means to exclude are compiled anyway | *(code)* |
| [Latent-noise timestep shift](../bugs/08-latent-noise-timestep-shift.md) | `--latent_noise_scale` > 0 | Shift computed from (h, w, c) instead of (frames, h, w), as in ByteDance's scripts (dormant there: scale 0): far stronger noise than intended, growing with resolution | *(code)*; its effect measured at 1080p ([quality.md](quality.md#noise-scales)) |
| [Single non-zero `--cuda_device`](../bugs/21-cuda-device-nonzero-mask.md) | `--cuda_device N`, N ≠ 0 | The mask has no effect; context and memory figures on GPU 0 | *(code)*; the ignored mask checked on one GPU |
| [Multi-GPU images](../bugs/01-multi-gpu-image-hang.md) | Image input, several GPUs | Hang | Measured with `--cuda_device 0,0` |
| [`--output_format mp4` on an image](../bugs/04-image-mp4-output-crash.md) | Image input | Crash at save, after the processing | Measured |
| [`--output` directory with a dot](../bugs/26-output-directory-with-a-dot.md) | Video output to a directory named like `clip-x1.5/` | Taken for a file name: the writer fails at save, after the processing | Met in a run; both writers checked on CPU |
| [Grayscale or 16-bit images](../bugs/03-image-input-grayscale-and-16-bit.md) | Image input | `IndexError` / near-white output | Measured |
| [Frame count under-reported by the container](../bugs/11-frame-count-trusted.md) | Some videos | The last frames are dropped silently | *(code)* |
| [`--cuda_device` debug message](../bugs/23-help-text-errors.md) | `--debug` | "Using device index 0 inside script (mapped to selected GPU)" is not what the code does | *(code)* |
| [Help text](../bugs/23-help-text-errors.md) | – | The `--output` default is wrong. `None` is listed as a format choice but rejected. `--batch_size` "must follow 4n+1" is padded instead. `--cache_*` "Requires --*_offload_device" is defaulted to `cpu` instead | *(code)* |

## Ignored and overridden combinations

| Combination | What happens |
|---|---|
| Tile size or overlap without `--vae_encode_tiled` / `--vae_decode_tiled` | Ignored (overlap ≥ size isn't checked either) |
| Tiling on when the frame fits in one tile | The untiled path runs |
| `--tile_debug` without tiling in that phase | Ignored |
| `--10bit` with `--video_backend opencv` | Ignored |
| `--cache_dit` / `--cache_vae` on one file without `--chunk_size` | Ignored (tip logged) |
| `--cache_dit` / `--cache_vae` with several GPUs and no `--chunk_size` | Disabled (warning) |
| `--cache_dit` / `--cache_vae` with the offload device `none` | Offload device forced to `cpu` (info line) |
| `--blocks_to_swap` or `--swap_io_components` without `--dit_offload_device` | `ValueError`; the run aborts after reading the input |
| `--blocks_to_swap` above the block count | Clamped to 36 (7B) or 32 (3B) |
| BlockSwap on macOS | Disabled (warning) |
| `--temporal_overlap` ≥ `--batch_size` | Reset to 0 for batching (warning); still used as chunk context and GPU overlap |
| `--batch_size` not 4n+1, or larger than the clip | Each batch is padded to the next 4n+1 with mirrored frames, dropped after decode |
| `--prepend_frames` after the first chunk or on workers other than 0 | Set to 0 |
| `--skip_first_frames`, `--load_cap`, `--chunk_size` on an image | Ignored |
| `--attention_mode` whose library is missing | Silent fallback ([environment.md](environment.md#2-seedvr2-falls-back-silently-between-attention-backends)) |
| `--attention_mode sageattn_3` | SA2 on every variable-length call, i.e. always ([attention.md](attention.md#consequences-for---attention_mode)) |
| `--compile_*` settings without `--compile_dit` / `--compile_vae` | Ignored |
| `--compile_backend inductor` without Triton | Error logged, then the model runs uncompiled |
| `--compile_mode` other than `default` with `--compile_backend cudagraphs` | Ignored (torch warning) |
| `--input_noise_scale`, `--latent_noise_scale` outside [0, 1] | Not checked |
| `PYTORCH_CUDA_ALLOC_CONF` already set | The CLI's `cudaMallocAsync` default isn't applied |
| `--cuda_device` with `CUDA_VISIBLE_DEVICES` set | Not validated, mask unchanged |
