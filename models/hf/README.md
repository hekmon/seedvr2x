---
license: apache-2.0
base_model:
- ByteDance-Seed/SeedVR2-7B
base_model_relation: quantized
pipeline_tag: video-to-video
tags:
- seedvr2
- video-super-resolution
- shot-boundary-detection
---

# seedvr2x's model files

The files seedvr2x runs. seedvr2x is a SeedVR2 video upscaler for long runs; it downloads these
files itself, at a revision of this repository and with SHA-256s pinned in its code.

**These are unofficial conversions: neither ByteDance's files nor TransNetV2's authors'.** Each
was made from its original by a script that pins its inputs and checks its output, and running
the scripts again gives the same bytes (`SHA256SUMS`).

## Which file to use

| | File | Size | What you give up |
|---|---|---|---|
| **Recommended** | `seedvr2x_ema_7b_sharp_fp16.safetensors` (float16) | 16.5 GB | nothing: the reference |
| 8 GB tier | `seedvr2x_ema_7b_sharp_int8_convrot.safetensors` (int8, rotated) | 8.3 GB | almost nothing: as close to the float16 model as another seed of it, or closer; faster on GeForce RTX 20 to 50 |
| 4 GB tier | `seedvr2x_ema_7b_sharp_dyn.gguf` (GGUF, a type per matrix) | 4.8 GB | some fidelity: the closest 4 GB file to the float16 model at 1080p, closer than Q4_K at 4K too, yet further from it than another seed on some sources (at 1080p a cartoon's LPIPS and DISTS, live action's finest texture, an anime clip's flicker and colour; at 4K live action's finest texture); saves memory only, a little slower |

Take float16 when your GPU has room for it, else the 8 GB file, else the 4 GB one. These are the
sharp 7B's, the model seedvr2x runs by default; ByteDance's regular 7B has the same files, named
without `_sharp`. The other formats, and how each was measured, are
[further down](#every-format-in-detail).

| File | Bytes | SHA-256 | Made from | Change |
|---|---|---|---|---|
$files

## SeedVR2 in float16

[SeedVR2](https://github.com/ByteDance-Seed/SeedVR) is ByteDance Seed's one-step video
restoration model. ByteDance publishes its weights in float32
([ByteDance-Seed/SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)): the 7B DiT, a
"sharp" 7B DiT of the same architecture, and the VAE.

Each file here is one of them with every tensor rounded to the nearest float16, ties to even,
under the same name and shape. Nothing else changes. numz's float16 files
([numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI)), which
ComfyUI-SeedVR2_VideoUpscaler runs, hold the same values: `seedvr2_fp16.py` checks ours against
them, equal element for element. Only the header differs, which holds the metadata, and the file
names: numz's downloader deletes a file that bears one of its names with another SHA-256.

## Precision

ByteDance's masters are float32: 33 GB for a 7B DiT. The files here hold them in float16, 16.5 GB.

- Rounding to float16 changes each weight by 0.05% at most, and it doesn't show: in seedvr2x's
  measurements, the float32 weights moved the output by 0.38 8-bit levels on average, where
  another seed moves it by 1.90.
- float16 rather than bfloat16: float16 keeps 3 more bits of each weight, and every SeedVR2
  weight fits its range. The model computes in bfloat16 anyway.
- The VAE stays in float16 too: quantizing its weights (0.47 GiB) would save nothing, and its 3D
  convolutions have no 8- or 4-bit path.

## Every format, in detail

The 7B DiT, the model that does the upscaling (the VAE turns frames into its input and back),
can be stored in several formats. A smaller file saves memory, sometimes time, and loses some
precision. These tables measure that loss on the weights themselves, against ByteDance's
originals, and this repository's files on video too: first the files of this repository, then
the SeedVR2 files other repositories publish, each against ours.

**This repository's files**

| Format | Size (7B) | Multiply speed, against float16 | Error per weight: typical (worst layer) | On video: the sharp 7B | On video: the 7B | Here |
|---|---|---|---|---|---|---|
| **float16** | 16.5 GB | 1x on every GPU: the reference | 0.02% (0.02%) | the reference | the reference | **yes** |
| fp8, with a scale per tensor | 8.3 GB | 2x on RTX 40 and 50 (up to 3x measured on RTX 50) and on workstation Ada and Blackwell cards; 1x before RTX 40: memory only | 2.6% (2.7%) | as close as another seed or closer, multiplied in 8 or 16 bits | the same | yes |
| int8, rotated, with a scale per row | 8.3 GB | 4x on GeForce RTX 20 to 50; 2x on workstation cards | 0.86% (1.08%) | as close as another seed or closer | the same (one score on one clip just past the line) | yes |
| GGUF Q8_0 | 8.8 GB | 1x: memory only | 0.6% (0.6%) | as close as another seed or closer: the closest of all | the same | yes |
| GGUF Q4_K | 4.8 GB | 1x: memory only, a little slower | 7.3% (7.9%); weighted 7.15% (9.0%) | further than a seed on cartoon, live action and anime | as close as another seed, softer on some sources | yes |
| GGUF Q4_K, with an importance matrix | 4.8 GB | 1x: memory only, a little slower | 7.55% (8.8%); weighted 6.64% (7.4%) | further than a seed on a cartoon's DISTS and live action's finest texture | further than a seed on a cartoon's DISTS and live action's fine detail | yes |
| GGUF, a type per matrix (Q3_K to Q5_K), with an importance matrix | 4.8 GB | 1x: memory only, a little slower | 7.55% (18%); weighted 6.31% (12%) | further than a seed on cartoon, live action and an anime clip's flicker | the closest 4 GB file: further than a seed on a cartoon's DISTS only | yes |
| NVFP4, with searched scales | 4.8 GB | 8x on RTX 50; 4x on workstation Blackwell cards (RTX PRO); 1x before Blackwell: memory only | 8.8% (8.9%) | multiplied in 4 bits: clearly worse on DISTS (cartoon, anime); in 16 bits: further than a seed on anime, cartoon and live action | in 4 bits: clearly worse on DISTS (cartoon, anime); in 16 bits: softer on live action | yes |

> **Multiply speed is not job speed.**
>
> The speed column gives the GPU's peak multiply rate for the DiT's matrices, from NVIDIA's
> specifications, against the same card's 16-bit rate. RTX 20 has no bf16 support, so there the
> reference is fp16.
> - GeForce cards run 16-bit and fp8 multiplies at half rate when they add up in 32 bits, as
>   they do here, but integer and 4-bit multiplies at full rate.
> - Workstation cards (Quadro RTX, RTX A6000, RTX 6000 Ada, RTX PRO 6000) halve nothing.
> - That is why int8 gets 4x on a GeForce card, twice fp8's on an RTX 40. RTX 50's fp8 is 2x in
>   NVIDIA's table; others measured up to 3x.
>
> A faster multiply still makes a job only a little faster:
> - rounding each input to 8 or 4 bits takes a pass of its own;
> - attention stays in 16 bits;
> - only the DiT gets faster, and at 1080p it takes about a fifth of a job (the VAE, which stays
>   in float16, takes the rest).
>
> By our estimate, multiplies 3 to 4 times faster make the DiT about twice as fast, and a job
> about 10% shorter. No speed is measured here. The main gain of a smaller file is memory: on a
> 16–32 GB card, seedvr2x can process more frames at a time and move less of the model out to
> system memory.

> **At 4K, the fp8 file in W8A8 and the NVFP4 file in W4A4 need care with comfy-kitchen.**
>
> comfy-kitchen 0.2.37, ComfyUI's kernel library, rounds a layer's input wrongly once it holds
> 2^32 values or more. With the fp8 file multiplied in 8 bits (W8A8), the video then comes out
> all NaN; with NVFP4 multiplied in 4 bits (W4A4), silently wrong. In the 7B that is the MLP's
> output projection, from about 350,000 video tokens in one pass: at 3840×2160, a batch of 41
> frames or more. Until comfy-kitchen fixes it, keep such batches to 37 frames, or multiply in 16
> bits (W8A16, W4A16). int8 and the GGUF files are not affected. seedvr2x's validation runs
> quantize such inputs in row chunks at comfy-kitchen's own scale.

How to read it:

- **Error per weight:** how far each stored weight is from ByteDance's original, relative to
  the weights' own size. Typical is the median over the 288 matrices that hold 99% of the 7B's
  weights, worst is the worst of them. A low typical error with a high worst one, as for fp8
  without scale, means a few layers are badly damaged: there, the weights are so small that most
  of them fall into fp8's coarsest range. A scale per tensor lifts them out of it.
- **Memory only:** the weights are stored small but widened to 16 bits for every
  multiplication. The file saves memory, not time.
- **Weighted:** for the GGUF files, each weight's error weighted by how much the layer's input
  uses its channel, measured on runs of the float16 model (an importance matrix, llama.cpp's
  method): close to how much the layer's output moves. The files made with an importance matrix
  trade error on rarely used inputs for less on the busy ones: their plain error is a little
  higher than the plain Q4_K's, their weighted one lower.
- **8-bit or 4-bit multiply:** the GPU multiplies in 8 or 4 bits, which is faster, but each
  layer's input is rounded to 8 or 4 bits too: a second loss, which the error per weight doesn't
  show, and the videos do.
- **On video** is measured separately (seedvr2x's `models/VALIDATION.md`): each file against its
  own model's float16 output at the same seed, on 8 clips at 1080p, judged against how much the
  float16 model's own seeds differ. "As close as another seed" means no score past 2.7 times the
  spread of 3 seeds, a line another seed of the float16 model crosses on one score in twenty;
  "further than a seed" names where a file crosses it. "Clearly worse": many times past it. At
  4K, against 2 seeds (the line at 10.9 times their spread): the sharp 7B's int8 and Q8_0 and
  the 7B's fp8 (W8A16) and Q8_0 within it; the sharp's fp8 W8A8, dynamic GGUF and Q4_K past it
  only where its two seeds agree almost exactly, Q4_K adding fine texture to live action, the
  dynamic GGUF taking a little away; the other files were not run at 4K.
- Measured on the 7B (the sharp 7B's files are within 0.01%; those made with an importance
  matrix within 0.05%, their weighted worst within 0.4%), against its float32 master: this
  repository's files themselves; the others' files where checked, their method otherwise (fp8
  without scale, Comfy-Org's NVFP4 and int8: each repository's own way, applied to the master).
  int8's rotation spreads each row's largest values before rounding, the input's too when it
  runs. NVFP4 has a scale per 16 weights: comfy-kitchen sets each from its block's largest
  weight; here each is the one, of that scale and the 7 below it, that minimises its block's
  error, in the same layout, which comfy-kitchen runs as it is.

**Other repositories' files, against ours**

| File | Repository | Size (7B) | Error per weight: typical (worst layer) | Against ours |
|---|---|---|---|---|
| float32, the original | ByteDance-Seed/SeedVR2-7B | 33 GB | 0, the reference | the source of every file here; too big to run |
| float16 (7B, sharp 7B, VAE) | numz/SeedVR2_comfyUI | 16.5 GB | 0.02% (0.02%) | **the same values**, checked element for element; only the header differs |
| float16 | Comfy-Org/SeedVR2 | 16.5 GB | 0.02% (0.02%) | the same values (one layer checked), with two text embeddings added |
| GGUF Q4_K_M | AInVFX/SeedVR2_comfyUI | 4.8 GB | 7.3% (7.9%) | **the same**: Q4_K on the same matrices, the same error; on video too (4 clips: within the seed band on nearly every score) |
| fp8, without scale, the last block in float16 | AInVFX/SeedVR2_comfyUI (numz's 7B fp8) | 8.5 GB | 2.8% (11%) | further: no scale, and its biases, embeddings and output layer in fp8 too (up to 15%) |
| fp8, without scale | Comfy-Org/SeedVR2 | 8.2 GB | 2.8% (17%) | further: every tensor in fp8, with no scale |
| NVFP4 | Comfy-Org/SeedVR2 | 4.8 GB | 9.5% (10.1%) | further: each block's scale from its largest weight |
| int8, rotated | Comfy-Org/SeedVR2 | 8.3 GB | 0.86% (1.08%) | the same method, comfy-kitchen's quantizer (one layer checked) |
| MXFP8 | Comfy-Org/SeedVR2 | 8.6 GB | not measured | Blackwell only; not made here |
| 3B (float16, fp8, GGUF) | numz/SeedVR2_comfyUI, AInVFX/SeedVR2_comfyUI | | | ByteDance's first 3B weights, which ByteDance replaced on 2025-06-22 (checked on the float16 and fp8 files) |

"The same" means weights as close to ByteDance's as ours; the files' bytes still differ, by their
header and by the precision they were made from.

## TransNetV2

[TransNetV2](https://github.com/soCzech/TransNetV2) finds the cuts between shots; seedvr2x splits
a video into shots with it. Its official weights are a TensorFlow SavedModel.
`transnetv2.safetensors` holds them as TransNetV2's own `inference-pytorch/convert_weights.py`
converts them for its PyTorch model, values unchanged: that model loads the file as it is
(`load_state_dict`). On the same frames, it gives the TensorFlow model's probabilities within
1e-6, and the same detections.

## Licences

- SeedVR2's files: the Apache License 2.0, in `LICENSE`, as ByteDance's. `NOTICE` gives each
  file's origin and what was changed, and credits ggml (MIT), whose quantizer made the GGUF files.
- `transnetv2.safetensors`: the MIT License, in `transnetv2.LICENSE` beside it.

Each file's metadata says the same: a safetensors file's `source`, `source_url` (at its
revision), `source_sha256`, `change`, `license`, `copyright` and `conversion`; a GGUF file's
`general.license`, `general.source.url` and `seedvr2x.source`, `.source_sha256`, `.change`,
`.copyright` and `.conversion`.

## How they were made

By the scripts in seedvr2x's `models/` directory, on a CPU, each with its dependencies inline
(`uv run models/<script>.py`):

- `seedvr2_fp16.py` downloads each master at the pinned revision and checks its SHA-256, rounds
  it (torch's conversion, checked bit for bit against numpy's), reads the file back with the
  safetensors library, and compares it with numz's file, tensor by tensor.
- `transnetv2_weights.py` downloads TransNetV2 at commit `85cef72`, runs its
  `convert_weights.py` unchanged (TensorFlow 2.21.0, PyTorch 2.14.1), checks the result byte for
  byte against an earlier conversion, saves it as safetensors, and runs both models on the same
  frames.
- `seedvr2_fp8.py` stores the 288 block matrices of the float32 master in fp8 with one scale per
  tensor (comfy-kitchen's layout); every other tensor stays the float16 file's, here and below.
- `seedvr2_int8.py` rotates the same matrices (comfy-kitchen's Hadamard) and rounds them to int8
  with a scale per row, checked against comfy-kitchen's own quantizer.
- `seedvr2_nvfp4.py` stores them in NVFP4 (comfy-kitchen's layout), each block's scale the one of
  8 that minimises the block's error; comfy-kitchen decodes every layer as the script does.
- `seedvr2_gguf.py` quantizes them to Q4_K or Q8_0 by ggml's own quantizer (llama.cpp `abeada3`,
  built from source); ggml's and gguf-py's decoders agree on every tensor.
- `seedvr2_gguf_dyn.py` chooses a type per matrix among Q3_K to Q8_0, the least error weighted by
  an importance matrix (from runs of the float16 model on a GPU) within the Q4_K file's bytes, and
  quantizes them by ggml with that importance; and Q4_K with the same importance.
- `dist.py` writes `LICENSE`, `NOTICE`, this card and `SHA256SUMS`.
