# models

The scripts that make the files of seedvr2x's Hugging Face repository,
[hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x) ([DESIGN.md](../seedvr2x/DESIGN.md#weights),
Weights). Each runs on its own, on a CPU, its dependencies inline (`uv run models/<script>.py`:
seedvr2x stays the repository's only uv project). Each pins its inputs (URL, revision, SHA-256),
runs its check, and writes into `models/dist/`, which git ignores; the upload is one directory
with every file hard-linked into it (`dist.py`).
The scripts of `gpu/` run in numz's environment instead (its checkout at `4490bd1` and its venv),
around numz's own CLI: the GPU runs that validate phase 2's files, and the importance matrices.

| Script | Makes | From | Check |
|---|---|---|---|
| `seedvr2_fp16.py` | `seedvr2x_ema_7b_fp16.safetensors`, `seedvr2x_ema_7b_sharp_fp16.safetensors`, `seedvr2x_ema_vae_fp16.safetensors` | ByteDance's fp32 masters (ByteDance-Seed/SeedVR2-7B at `eb0c428`), every tensor rounded to the nearest float16, ties to even | equal to numz's fp16 files (numz/SeedVR2_comfyUI at `09ced71`), element for element |
| `transnetv2_weights.py` | `transnetv2.safetensors`, `transnetv2.LICENSE` | TransNetV2's TensorFlow weights (soCzech/TransNetV2 at `85cef72`), through its own `convert_weights.py` | the converter's output byte for byte as measurement's ([scene-detection.md](../research/docs/scene-detection.md#transnetv2)); PyTorch against TensorFlow on the same frames |
| `dist.py` | `LICENSE`, `NOTICE`, `README.md` (the model card, from `hf/README.md`), `SHA256SUMS` | each file's metadata, safetensors or GGUF; an importance matrix's runs | each GGUF file made with an importance matrix finds it beside it, by name and SHA-256, and each importance matrix a GGUF file made with it |
| `seedvr2_fp8.py` | phase 2: `seedvr2x_ema_7b{,_sharp}_fp8_scaled.safetensors` | the fp32 masters: the 288 block matrices in E4M3 with one scale per tensor (comfy-kitchen's layout), the rest our fp16 values | each matrix's error; read back; the 16-bit tensors byte for byte our fp16 file's |
| `seedvr2_gguf.py` | phase 2: `seedvr2x_ema_7b{,_sharp}_{Q4_K,Q8_0}.gguf` | the fp32 masters, by ggml's own quantizer (llama.cpp `abeada3`, built from source), the rest float16 | ggml's and gguf-py's decoders agree; each matrix's error, numz's Q4_K_M's beside; read back |
| `seedvr2_int8.py` | phase 2: `seedvr2x_ema_7b{,_sharp}_int8_convrot.safetensors` | the fp32 masters: the 288 block matrices rotated (comfy-kitchen's Hadamard), int8 with one scale per row, the rest our fp16 values | each matrix's error; codes against comfy-kitchen's own quantizer; read back; the 16-bit tensors byte for byte |
| `seedvr2_nvfp4.py` | phase 2: `seedvr2x_ema_7b{,_sharp}_nvfp4.safetensors` | the fp32 masters: the 288 block matrices in NVFP4 (comfy-kitchen's layout), each block's scale the one of 8 that minimises its error, the rest our fp16 values | each matrix's error, beside comfy-kitchen's own quantizer's; read back; comfy-kitchen decodes every layer bit for bit as we do; the 16-bit tensors byte for byte |
| `seedvr2_gguf_dyn.py` | phase 2: `seedvr2x_ema_7b{,_sharp}_dyn.gguf` (a type per matrix) and `seedvr2x_ema_7b{,_sharp}_Q4_K_imatrix.gguf` (its control) | the fp32 masters and an importance matrix (`gpu/imatrix_hook.py`'s; by default ours, pinned: the one uploaded to `hekmon/seedvr2x` at `c14a2bc4`, its size and SHA-256 checked), by ggml's own quantizer with the importance (llama.cpp `abeada3`, MIT), the types among Q3_K to Q8_0 chosen by activation-weighted error within our Q4_K's bytes, the rest float16 | each matrix's plain and weighted error, our static Q4_K's beside (its bytes reproduced); ggml's and gguf-py's decoders agree; read back; the size within our Q4_K's; numz's loader (`numz_gguf_check.py`); outputs pinned by the importance file's SHA-256 |
| `seedvr2_fp16_3b.py` | research, not uploaded: `seedvr2x_ema_3b_fp16.safetensors` | the 3B's current master (ByteDance-Seed/SeedVR2-3B at `37255ff`), rounded as `seedvr2_fp16.py` rounds | read back; numz's 3B fp16 file's tensor names, dtypes and shapes, in its order at its offsets, 4.9% of the values equal (numz's holds the first 3B weights); `--first`: the first master rounded the same way gives numz's data section byte for byte |
| `ck_check.py` | | our fp8, int8 and NVFP4 files | comfy-kitchen (0.2.37, CPU) decodes every marked layer as we do, and multiplies one |
| `numz_gguf_check.py` | | our GGUF files, run in numz's environment | numz's own loader: every tensor, the 7B DiT loaded, one layer's forward |
| `numz_check.py` | | numz's files and ByteDance's masters | how numz's 3B and 7B files were made |
| `formats_study.py` | | the 7B's fp32 master | every format's error per weight: [FORMATS.md](FORMATS.md) |
| `gpu/ck_patch.py` | | our fp8, int8 and NVFP4 files in numz's CLI on the GPU, through comfy-kitchen's layers (0.2.37), and our model directory: a wrapper patching numz's modules at import; in W8A8 and W4A4, an input of more than 2^31 − 1 elements quantized in row chunks at the scale comfy-kitchen gives the whole tensor (its CUDA quantizers index in 32 bits) | its check mode: each layer's output against float32 on the run's own activations, and the kernels and dequantizations each multiply met ([VALIDATION.md](VALIDATION.md)) |
| `gpu/ck_patch_test.py` | | `gpu/ck_patch.py` in numz's environment, on the CPU | the wrapper chain and its guards through numz's CLI; each format loaded into numz's 7B, the file's tensors bit for bit, every other tensor the fp16 model's; each mode's multiply against float32; the row chunks against comfy-kitchen's own quantization, bit for bit |
| `gpu/imatrix_hook.py` | an importance matrix (safetensors: per block matrix, the inputs' sum of squares per channel and the tokens) | numz's runs of an fp16 7B on the GPU, through a wrapper of numz's CLI (as `gpu/ck_patch.py`); `merge` adds runs | on the GPU, a control run's decode equal to the unhooked run's |
| `gpu/imatrix_hook_test.py` | | `gpu/imatrix_hook.py` in numz's environment, on the CPU | the wrapper chains and guards; numz's 7B with and without the hooks, bit for bit; the sums against an independent computation; the file read back and merged |
| `gpu/validation/` | the GPU runs' job lines, their scores and VALIDATION.md's pages | the GPU queue, the CPU pool scoring each run with colour.md's tools, the summaries, pairings and crops that ran on the GPU box, every path from `glue.env` ([gpu/validation/README.md](gpu/validation/README.md)) | the box's files once glue.env's values are put back, byte for byte; on the box, the same job lines and pages |
| `VALIDATION.md` | the GPU validation of phase 2's files | numz's runs of every file, through `gpu/ck_patch.py` or numz's GGUF loader, scored with colour.md's tools | each file against its own model's fp16 output and that model's seed band, at 1080p and 4K; the 3B's current and first weights against the GT, beside the sharp 7B's 4 GB file; the user's eyes |

Phase 2's files are made ahead of phase 2, on the CPU, and stay out of `models/dist/` until GPU
runs validate them (DESIGN.md, Weights): [VALIDATION.md](VALIDATION.md). Every format is to ship
in the one upload, each with its measured quality on the card. What each format does to the
weights, and why it is made as it is: [FORMATS.md](FORMATS.md).

`common.py` holds what they share: downloads checked by SHA-256 (resumed when cut), and the
safetensors writer.

- **The same bytes on every run.** The safetensors library writes a file's metadata in a random
  order (version 0.8.0: three saves, three SHA-256s), so `common.py` writes the files itself, in
  the library's layout (tensors by decreasing element size then name, compact JSON header, the
  metadata's keys sorted, padded with spaces to 8 bytes), and each script reads its file back
  with the library. Each script pins its outputs' SHA-256 (`OUTPUTS`, `OUTPUT`): a run whose
  bytes differ fails.
- **The metadata** of each file says where it comes from and what changed: `source`,
  `source_url` (at its revision), `source_sha256`, `change`, `license`, `copyright`,
  `conversion`, and `format` (`pt`); a GGUF file the same as `general.license`,
  `general.source.url` and `seedvr2x.*`. `dist.py` writes `NOTICE` and the card's table from it.
- **The names** are not numz's: numz's downloader deletes a file named like one of its own whose
  SHA-256 differs (`src/utils/downloads.py:216-235` at `4490bd1`), and ours differ from numz's
  by their header.
- **Masters and references are read where they are**, never moved: `--masters`,
  `--reference` and `--imatrices` (our importance matrices) name directories holding them; what
  is missing is downloaded into `--cache` (default `~/.cache/seedvr2x-models`).
- **Others' files are checks, not models.** numz's files serve as a reference to compare with,
  never as a source: every file starts from its original (ByteDance's fp32 masters, TransNetV2's
  TensorFlow weights). What other tools' files show (tensor names, layouts, which tensors they
  quantize) is taken only where a runtime needs it to load ours, and checked, never copied as a
  choice.
- **Third-party code:** none is copied into `models/`. `transnetv2_weights.py` downloads and
  runs TransNetV2's own `convert_weights.py` and model code (MIT, Tomáš Souček) at the pinned
  commit, unchanged; `NOTICE`, the card and `transnetv2.LICENSE` credit it. Code used or ported
  from elsewhere is named here, with its licence, and in `NOTICE` when an uploaded file is made
  with it.
  - comfy-kitchen (Comfy-Org, Apache-2.0, Copyright (c) 2025 Comfy Org; 0.2.37 at `be003b7`): a
    run-time dependency of `ck_check.py` (inline, installed by uv) and `gpu/ck_patch.py`
    (installed for numz's interpreter, outside its venv), not vendored; our fp8, int8 and NVFP4
    files use its layouts.
  - ggml / llama.cpp (MIT, Copyright (c) 2023-2026 The ggml authors): `seedvr2_gguf.py` and
    `seedvr2_gguf_dyn.py` build ggml-base from llama.cpp at a pinned commit and call its quantizer;
    the importance matrix is llama.cpp's method (tools/imatrix: each input channel's mean square,
    weighed by ggml's quantizers), collected by our own hooks; no code copied. `dist.py` credits
    it in `NOTICE` when GGUF files are uploaded.

```bash
uv run models/seedvr2_fp16.py --masters /path/to/bytedance --reference /path/to/numz
uv run models/transnetv2_weights.py --video FILE START COUNT   # any number of excerpts, or none
uv run models/dist.py --out DIR    # DIR: the upload, every file hard-linked into it (dist.py)
(cd DIR && sha256sum -c SHA256SUMS)
hf upload hekmon/seedvr2x DIR . --repo-type model --commit-message "..."   # re-run if cut
```
