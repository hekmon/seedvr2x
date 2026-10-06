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

Smaller files, 8-bit and 4-bit, come later, each published once GPU runs have measured how close
it stays to the float16 model, against the spread between seeds. This card will then compare
them, to help choose: size, how each one multiplies on each GPU generation, speed and quality.

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
