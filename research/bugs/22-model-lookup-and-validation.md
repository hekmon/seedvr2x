# 22. Model lookup: `./models/SEEDVR2` shadows `--model_dir`, the architecture comes from the file name, a hash mismatch deletes the file, and missing weights pass silently

| | |
|---|---|
| Severity | UX-doc (with a data-loss risk); silent wrong output with an incomplete custom checkpoint (6) |
| Status | from code, not reproduced; 6 run on CPU with numz's loader (no real checkpoint) |
| Affected options | `--model_dir`, `--dit_model` |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

Several surprises in how the CLI finds, checks and interprets model files:

1. `./models/SEEDVR2` in the **current directory** is always searched first; `--model_dir` is
   only the fallback, so a file there shadows the one in `--model_dir`.
2. `--dit_model` choices are the registry plus files found in `./models/SEEDVR2`, not in
   `--model_dir`: a custom checkpoint is only accepted from the current directory's
   `./models/SEEDVR2`.
3. The 3B or 7B architecture is chosen by the substring `"7b"` in the file name (case-sensitive).
4. A registry file whose sha256 doesn't match is **deleted** and downloaded again.
5. After a download, the validation cache is written to `./models/SEEDVR2/.validation_cache.json`
   of the current directory (created if needed) instead of `--model_dir`, so the next run hashes
   the 16 GB file again.
6. Nothing checks that a `.safetensors` or `.pth` checkpoint holds the model's keys: missing and
   unexpected keys are ignored. A missing buffer becomes zeros (the RoPE frequencies: no positional
   encoding, no error). A missing weight stays on the meta device: moving its block later fails
   with an error that doesn't name the file, and a forward through it on CPU returns garbage
   without error. A checkpoint whose keys carry a prefix loads nothing. Only GGUF files are
   checked.

## Reproduction

From the code (not run):

```bash
mkdir -p ./models/SEEDVR2 && cp old_or_other/seedvr2_ema_7b_fp16.safetensors ./models/SEEDVR2/
python inference_cli.py input.mp4 --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors --debug
#   → "DiT model found: ./models/SEEDVR2/seedvr2_ema_7b_fp16.safetensors" (not /path/to/models)

cp my_finetune.safetensors /path/to/models/
python inference_cli.py input.mp4 --model_dir /path/to/models --dit_model my_finetune.safetensors
#   → argparse: invalid choice (only ./models/SEEDVR2 is scanned for extra files)

cp my_finetune.safetensors ./models/SEEDVR2/my_finetune_7B.safetensors   # a 7B fine-tune
python inference_cli.py input.mp4 --dit_model my_finetune_7B.safetensors
#   → accepted, but built with configs_3b ("7B" doesn't contain "7b"): the 7B weights don't fit
```

Item 6, run on CPU (torch 2.14.1) through numz's own `_load_standard_weights` and
`initialize_meta_buffers`, on a 2-block 7B DiT built on the meta device as numz builds it, with a
complete checkpoint of 74 tensors minus some keys:

| Checkpoint | `load_state_dict` result (discarded) | After numz's load | Then |
|---|---|---|---|
| Without the 2 `rope.freqs` buffers | 2 missing | both buffers zeros, now non-persistent; the only trace is "Initialized 6 non-persistent buffers" (4 otherwise), in `--debug` | angle table all zeros: RoPE returns q and k unchanged, no error |
| Without one weight | 1 missing | the weight still on meta | `block.to(device)` (BlockSwap, offload): `NotImplementedError: Cannot copy out of meta tensor; no data!`; a forward through it on CPU: no error, garbage values (not tried on CUDA) |
| Every key prefixed `model.` | 74 missing, 74 unexpected | all 72 weights on meta, no message | as above |

## Root cause

- Search order: `find_model_file(filename, fallback_dir)` returns the first match among
  `get_all_model_paths()` and only then `fallback_dir` (`src/utils/constants.py:110-132`).
  Outside ComfyUI, `get_all_model_paths()` is `[get_base_cache_dir()]` = `"./models/SEEDVR2"`
  (`constants.py:38-54`, `57-86`). Both the download check (`src/utils/downloads.py:187`) and the
  loader (`src/core/model_configuration.py:1033`, `1054`, `1106`, `1134`) pass `--model_dir` as
  the fallback.
- Choices: `choices=get_available_dit_models()` is evaluated when the parser is built
  (`inference_cli.py:1364-1366`), and lists the registry plus `get_all_model_files()`, i.e. files
  in the same search paths (`src/utils/model_registry.py:67-86`); `--model_dir` isn't parsed yet.
- Architecture: `'./configs_7b' if "7b" in dit_model else './configs_3b'`
  (`model_configuration.py:718-720`). Files outside the registry are not hash-checked either
  (`downloads.py:177-181`).
- Deletion (`downloads.py:216-235`):
  ```python
  if validate_file(filepath, expected_hash, cache_dir):
      ...
  else:
      # File is corrupted
      ...
      os.remove(filepath)
  ```
  The mismatching file may be a user's own file under a registry name, or the result of an
  upstream re-upload with a new hash; it is removed without confirmation, wherever it was found
  (including `./models/SEEDVR2`).
- Cache location: after a download, `validate_file(filepath, expected_hash)` is called without
  `cache_dir` (`downloads.py:260`), so `get_validation_cache_path(None)` →
  `get_base_cache_dir()` (`constants.py:135-147`). The next run looks in `--model_dir`'s cache
  (`downloads.py:209`), misses, and hashes the file again (then stores it in the right place).
- Unchecked keys: the model is built on the meta device (`src/core/model_loader.py:451-452`) and
  loaded with `model.load_state_dict(state, strict=False, assign=True)` (`model_loader.py:823`),
  whose result (the missing and unexpected keys) is dropped. Then `initialize_meta_buffers`
  (`model_loader.py:598-600`, `777-815`) replaces every buffer still on meta with
  `torch.zeros_like(...)` on the target device, re-registered as non-persistent (`811-812`): meant
  for the non-persistent buffers no checkpoint holds (`rotary_embedding_torch`'s `dummy` and
  `cached_freqs`), it also zeroes the persistent ones the file lacks: the RoPE `freqs`, the DiTs'
  only persistent buffers (36 on the 7B, 32 on the 3B; the VAE has none). Weights aren't touched
  and stay on meta. Shape mismatches still raise (`load_state_dict` checks sizes even with
  `strict=False`). Later checks only look at the first parameter (`model_loader.py:503`,
  `src/core/generation_phases.py:620`, `src/optimization/memory_manager.py:711`). The GGUF path
  does check: three key shapes (`model_loader.py:897-933`) and a forced WARNING listing missing
  and unmatched names (`model_loader.py:749-765`, `867`).

## Impact

- Running the CLI from a directory that happens to contain `models/SEEDVR2` (for example the
  SeedVR2 checkout itself, or a ComfyUI tree) silently uses other weights than `--model_dir`'s.
- Custom or renamed checkpoints need to be copied into the current directory; a name without
  lowercase "7b" picks the wrong architecture.
- A file that doesn't match the registry hash (a re-upload, a manual edit) is deleted: 16.5 GB to
  download again, or a lost custom file.
- One extra full hash (tens of seconds for 16.5 GB) after each download, and a stray
  `./models/SEEDVR2/` directory in the working directory.
- Registry files pass the hash check, so item 6 concerns custom and converted files: a conversion
  that only exports parameters drops the RoPE buffers, and the DiT then runs without positional
  encoding, with no message (expected to degrade the output; not run on a real checkpoint). A
  checkpoint saved from a wrapper (`model.` or `module.` prefix) loads nothing; what follows
  depends on the path: a meta-tensor error that doesn't point at the file when a block is moved
  (BlockSwap, offload), garbage on CPU (CUDA not tried).

## Possible fix

- Search `--model_dir` first when it is given, then the default paths:
  `find_model_file(name, preferred_dir=model_dir)`; log the chosen path at INFO level, not only
  in `--debug`.
- Build `--dit_model` choices after a pre-parse of `--model_dir`, or drop `choices=` and validate
  after parsing (registry name, or an existing file in the search paths, or a path).
- Choose the architecture from the checkpoint's contents (for example the number of blocks or a
  7B-only key in the state dict), or at least `"7b" in name.lower()`, and add a `--dit_arch
  {3b,7b}` override.
- On hash mismatch, rename to `<file>.mismatch` (or ask), and download to a temporary name;
  never delete a file the CLI didn't download itself.
- Pass `cache_dir` to `validate_file` at `downloads.py:260`.
- Keep `load_state_dict`'s result at `model_loader.py:823`: raise on missing keys, naming the file
  and the first keys, and log unexpected keys as a forced WARNING (as the GGUF path does). In
  `initialize_meta_buffers_impl`, only initialize buffers registered as non-persistent, and raise
  for any other tensor still on meta.

Test: the three commands above; a deliberately modified registry file must survive (renamed)
and be reported. For item 6, the three checkpoints of the table must fail at load with the missing
keys named; check that the registry files load without a warning.

## References

- [cli-flags.md, Models](../docs/cli-flags.md#models)
- [cli-flags.md, Model](../docs/cli-flags.md#model)
