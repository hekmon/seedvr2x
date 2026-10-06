# models

The scripts that make the files of seedvr2x's Hugging Face repository,
[hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x) ([DESIGN.md](../seedvr2x/DESIGN.md#weights),
Weights). Each runs on its own, on a CPU, its dependencies inline (`uv run models/<script>.py`:
seedvr2x stays the repository's only uv project). Each pins its inputs (URL, revision, SHA-256),
runs its check, and writes into `models/dist/`, which git ignores and the upload takes as it is.

| Script | Makes | From | Check |
|---|---|---|---|
| `seedvr2_fp16.py` | `seedvr2x_ema_7b_fp16.safetensors`, `seedvr2x_ema_7b_sharp_fp16.safetensors`, `seedvr2x_ema_vae_fp16.safetensors` | ByteDance's fp32 masters (ByteDance-Seed/SeedVR2-7B at `eb0c428`), every tensor rounded to the nearest float16, ties to even | equal to numz's fp16 files (numz/SeedVR2_comfyUI at `09ced71`), element for element |
| `transnetv2_weights.py` | `transnetv2.safetensors`, `transnetv2.LICENSE` | TransNetV2's TensorFlow weights (soCzech/TransNetV2 at `85cef72`), through its own `convert_weights.py` | the converter's output byte for byte as measurement's ([scene-detection.md](../research/docs/scene-detection.md#transnetv2)); PyTorch against TensorFlow on the same frames |
| `dist.py` | `LICENSE`, `NOTICE`, `README.md` (the model card, from `card.md`), `SHA256SUMS` | each file's safetensors metadata | |

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
  `conversion`, and `format` (`pt`). `dist.py` writes `NOTICE` and the card's table from it.
- **The names** are not numz's: numz's downloader deletes a file named like one of its own whose
  SHA-256 differs (`src/utils/downloads.py:216-235` at `4490bd1`), and ours differ from numz's
  by their header.
- **Masters and references are read where they are**, never moved: `--masters` and
  `--reference` name directories holding them; what is missing is downloaded into `--cache`
  (default `~/.cache/seedvr2x-models`).
- **Others' files are checks, not models.** numz's files serve as a reference to compare with,
  never as a source: every file starts from its original (ByteDance's fp32 masters, TransNetV2's
  TensorFlow weights). What other tools' files show (tensor names, layouts, which tensors they
  quantize) is taken only where a runtime needs it to load ours, and checked, never copied as a
  choice.
- **Third-party code:** none is copied into `models/`. `transnetv2_weights.py` downloads and
  runs TransNetV2's own `convert_weights.py` and model code (MIT, Tomáš Souček) at the pinned
  commit, unchanged; `NOTICE`, the card and `transnetv2.LICENSE` credit it. Code used or ported
  from elsewhere is named here and in `NOTICE`, with its licence.

```bash
uv run models/seedvr2_fp16.py --masters /path/to/bytedance --reference /path/to/numz
uv run models/transnetv2_weights.py --video FILE START COUNT   # any number of excerpts, or none
uv run models/dist.py
hf upload hekmon/seedvr2x models/dist . --repo-type model --commit-message "..."
```
