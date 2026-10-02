# 02. CUDA-graph compile modes crash under the CLI's default `cudaMallocAsync` allocator

| | |
|---|---|
| Severity | crash |
| Status | measured (`max-autotune`, `reduce-overhead`, `--compile_backend cudagraphs`) |
| Affected options | `--compile_dit` / `--compile_vae` with `--compile_mode reduce-overhead` or `max-autotune`, or `--compile_backend cudagraphs` |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

The CLI sets `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` before importing torch. Torch's
CUDA graph trees (used by `reduce-overhead`, `max-autotune` and the `cudagraphs` backend) don't
support that allocator, so every compile mode the help offers apart from `default` and
`max-autotune-no-cudagraphs` aborts the run at the first DiT forward. The CLI's "falling back
to uncompiled model" handler never sees the error, so the run dies after the VAE encode.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --model_dir /path/to/models \
  --dit_model seedvr2_ema_3b_fp16.safetensors --resolution 480 --batch_size 5 --load_cap 5 \
  --compile_dit --compile_mode reduce-overhead        # or: --compile_backend cudagraphs
```

- Expected: the DiT is compiled with CUDA graphs, or the CLI refuses the combination up front.
- Actual: Phase 2 fails with
  `RuntimeError: cudaMallocAsync does not yet support checkPoolLiveAllocations. If you need it,
  please file an issue describing your use case.`, raised from
  `torch/_inductor/cudagraph_trees.py` (`cudagraphify` → `add_function` → `run_eager`), exit 1.
- Measured: runs `bugs-480-3bfp16-compile-ro` (`reduce-overhead`) and
  `bugs-480-3bfp16-compile-cg` (`--compile_backend cudagraphs`) both fail on the first DiT batch;
  `max-autotune` failed the same way earlier ([environment.md](../docs/environment.md#runtime-notes-that-depend-on-the-environment)).
- The same `reduce-overhead` command with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (`bugs-480-3bfp16-compile-ro-expseg`, 10 frames) completes: first batch 36.2 s (compilation),
  second 2.7 s.

## Root cause

1. `inference_cli.py:77`, before torch is imported:
   ```python
   os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
   ```
2. `--compile_mode` and `--compile_backend` are passed unchanged to `torch.compile`
   (`inference_cli.py:890-907` → `src/core/model_configuration.py:1307-1312`, `1379`). With
   `reduce-overhead` / `max-autotune` (inductor) or `backend="cudagraphs"`, the first call goes
   through CUDA graph trees, which call `checkPoolLiveAllocations` on the caching allocator; the
   `cudaMallocAsync` backend raises.
3. `torch.compile` is lazy: `_apply_torch_compile` (`model_configuration.py:1373-1391`) only
   wraps the model, so its `except` ("Falling back to uncompiled model") catches nothing. The
   error surfaces in `upscale_all_batches`, which logs it and re-raises
   (`src/core/generation_phases.py:760-762`).

The allocator default is good for memory ([vram.md](../docs/vram.md#the-allocator)): the
problem is only that it is applied unconditionally, including with options that can't work with
it.

## Impact

- Anyone who tries the help's "best runtime" `max-autotune`, `reduce-overhead`, or the
  `cudagraphs` backend: the run aborts after the VAE encode, every time.
- Workaround: `--compile_mode default` or `max-autotune-no-cudagraphs`; or set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` yourself (the CLI then keeps it), which
  worked for `reduce-overhead` above. Avoid plain `backend:native`, which fragments on a full
  card ([vram.md](../docs/vram.md#the-allocator)).

## Possible fix

Decide the allocator after looking at the compile options, in the pre-parse block that already
runs before the torch import:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -74,12 +74,27 @@ if platform.system() == "Darwin":
     os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
     os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
 else:
-    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
-
     # Pre-parse arguments that must be handled before torch import
-    _pre_parser = argparse.ArgumentParser(add_help=False)
+    _pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
     _pre_parser.add_argument("--cuda_device", type=str, default=None)
+    _pre_parser.add_argument("--compile_dit", action="store_true")
+    _pre_parser.add_argument("--compile_vae", action="store_true")
+    _pre_parser.add_argument("--compile_mode", type=str, default="default")
+    _pre_parser.add_argument("--compile_backend", type=str, default="inductor")
     _pre_args, _ = _pre_parser.parse_known_args()
+
+    # CUDA graph trees don't support cudaMallocAsync (checkPoolLiveAllocations)
+    _uses_cuda_graphs = (_pre_args.compile_dit or _pre_args.compile_vae) and (
+        _pre_args.compile_backend == "cudagraphs"
+        or _pre_args.compile_mode in ("reduce-overhead", "max-autotune"))
+    if not _uses_cuda_graphs:
+        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
+    elif "cudaMallocAsync" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
+        print("❌ [ERROR] CUDA graphs (--compile_mode reduce-overhead/max-autotune, --compile_backend "
+              "cudagraphs) don't work with PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync")
+        sys.exit(1)
+    else:
+        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

And make the fallback message honest: either remove "Falling back to uncompiled model" (it only
covers wrap-time errors), or run a warm-up forward inside the `try` when compile is requested.

Risks: CUDA graphs keep their own memory pool and static input/output buffers, so
`reduce-overhead` costs memory; the DiT's per-batch shapes (a shorter last batch) re-record
graphs. Measure it before recommending it.

Test: the two commands above must either run (allocator switched) or exit at start-up with the
message, never after Phase 1; a run with `PYTORCH_CUDA_ALLOC_CONF` already set by the user must
keep it.

## References

- [environment.md, Runtime notes](../docs/environment.md#runtime-notes-that-depend-on-the-environment)
- [vram.md, The allocator](../docs/vram.md#the-allocator), [`torch.compile`](../docs/vram.md#torchcompile)
- [cli-flags.md, Attention and `torch.compile`](../docs/cli-flags.md#attention-and-torchcompile)
