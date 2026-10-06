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

## Which file to choose

The 7B DiT, the model that does the upscaling (the VAE turns frames into its input and back),
can be stored in several formats. A smaller file saves memory, sometimes time, and loses some
precision. This table measures that loss on the weights themselves, against
ByteDance's originals.

| Format | Size (7B) | Where it runs faster | Error per weight: typical (worst layer) | Here |
|---|---|---|---|---|
| float32, ByteDance's original | 33 GB | none: the source, too big to run | 0 (the reference) | no |
| **float16** | 16.5 GB | every GPU, at the reference speed | 0.02% (0.02%) | **yes** |
| fp8, with a scale per tensor | 8.3 GB | RTX 40 and 50: 8-bit multiply | 2.6% (2.7%) | to come |
| fp8, without scale | 8.2 GB | none: numz widens it to 16 bits | 2.8% (17%) | no: in numz's and Comfy-Org's repos |
| int8, rotated, with a scale per row | 8.3 GB | RTX 20 to 50: 8-bit multiply | 0.86% (1.08%) | to come |
| GGUF Q8_0 | 8.8 GB | none: memory only | 0.6% (0.6%) | to come |
| GGUF Q4_K_M | 4.8 GB | none: memory only, a little slower | 7.3% (7.9%) | to come |
| NVFP4 | 4.8 GB | RTX 50: 4-bit multiply | 9.5% (10.1%) | being studied |

How to read it:

- **Error per weight:** how far each stored weight is from ByteDance's original, relative to
  the weights' own size. Typical is the median over the 288 matrices that hold 99% of the 7B's
  weights, worst is the worst of them. A low typical error with a high worst one, as for fp8
  without scale, means a few layers are badly damaged: there, the weights are so small that most
  of them fall into fp8's coarsest range. A scale per tensor lifts them out of it.
- **Memory only:** the weights are stored small but widened to 16 bits for every
  multiplication. The file saves memory, not time.
- **8-bit or 4-bit multiply:** the GPU multiplies in 8 or 4 bits, which is faster, but each
  layer's input is rounded to 8 or 4 bits too: a second loss, which this table doesn't show.
- **What you see** is measured separately, on videos, against how much two seeds of the float16
  model differ. Each smaller file is published once it passes that test, its result added here.
- **Speed:** only the DiT gets faster. At 1080p it takes about a fifth of a job (the VAE, which
  stays in float16, takes the rest), so even a DiT twice as fast shortens a job by about 10%.
  The main gain of a smaller file is memory: on a 16–32 GB card, seedvr2x can process more
  frames at a time and move less of the model out to system memory.
- Measured on the 7B (the sharp 7B has the same architecture), against its float32 master: fp8,
  int8, Q8_0 and Q4_K_M on the files made for this repository, NVFP4 simulated with
  comfy-kitchen's scales. int8's rotation spreads each row's largest values before rounding, the
  input's too when it runs.

## TransNetV2

[TransNetV2](https://github.com/soCzech/TransNetV2) finds the cuts between shots; seedvr2x splits
a video into shots with it. Its official weights are a TensorFlow SavedModel.
`transnetv2.safetensors` holds them as TransNetV2's own `inference-pytorch/convert_weights.py`
converts them for its PyTorch model, values unchanged: that model loads the file as it is
(`load_state_dict`). On the same frames, it gives the TensorFlow model's probabilities within
1e-6, and the same detections.

## Licences

- SeedVR2's files: the Apache License 2.0, in `LICENSE`, as ByteDance's. `NOTICE` gives each
  file's origin and what was changed.
- `transnetv2.safetensors`: the MIT License, in `transnetv2.LICENSE` beside it.

Each file's safetensors metadata says the same: `source`, `source_url` (at its revision),
`source_sha256`, `change`, `license`, `copyright` and `conversion`.

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
- `dist.py` writes `LICENSE`, `NOTICE`, this card and `SHA256SUMS`.
