# 16. The DiT is copied to the CPU before being deleted, in every run

| | |
|---|---|
| Severity | performance (time and host RAM) |
| Status | measured |
| Affected options | every run without `--cache_dit` (with it, the copy is the caching itself) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

At the end of Phase 2, `cleanup_dit` moves the whole DiT to the CPU ("releasing GPU memory") and
then deletes it. Nothing uses the CPU copy. For the 7B fp16 model that is 3.5 s of synchronous
device → host copy per generation, and ≈ 16 GiB of process RSS that stays until the end. The VAE
gets the same treatment after decode (0.5 GiB).

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 5 --load_cap 5 --debug
```

The debug log shows "Moving DiT from CUDA:0 to CPU (releasing GPU memory)", then "DiT model
deleted". Measured:

- 7B fp16, 1080p: the copy takes 3.5 s (3.65 s with `--dit_offload_device cpu`); max RSS 19 GiB
  for a 3-frame run, of which ≈ 16 GiB is this copy
  ([vram.md](../docs/vram.md#weights-and-the-per-token-model)).
- With all 36 blocks swapped, Phase 2 is *shorter* than without swap (25.4 s vs 27.4 s at batch
  45) because there is less to copy back ([vram.md](../docs/vram.md#blockswap)).
- At 1080p batch 5 the DiT inference itself is 3.7 s per batch: on a one-batch clip (a shot of
  ≤ 5 frames, an image) the useless copy costs as much as the inference.

- Expected: without caching, the GPU memory is released in place, in milliseconds.
- Actual: a full-model copy to pageable host memory first.

## Root cause

`cleanup_dit` (`src/optimization/memory_manager.py:1047-1080`):

```python
param_device = next(runner.dit.parameters()).device
if param_device.type not in ['meta', 'cpu']:
    ...
        offload_target = getattr(runner, '_dit_offload_device', None)
        if offload_target is None or offload_target == 'none':
            offload_target = torch.device('cpu')
        reason = "model caching" if cache_model else "releasing GPU memory"
        manage_model_device(model=runner.dit, target_device=offload_target, ...)
...
if not cache_model:
    release_model_memory(model=runner.dit, debug=debug)
    runner.dit = None
```

The move is done whether or not the model is cached (only MPS skips it, lines 1054-1056). In
effect the copy is what empties the GPU, since `release_model_memory` frees nothing
([13](13-release-memory-frees-nothing.md)). The CPU
copy then lives in the process until the cyclic GC collects the model (BlockSwap) or forever in
glibc's heap (pageable allocations are rarely returned to the OS).

`cleanup_vae` does the same (`memory_manager.py:1127-1150`).

## Impact

- Every CLI run, and every file of a directory and every `--chunk_size` chunk without
  `--cache_dit`: +3.5 s (7B fp16; proportionally less for the smaller fp8 and GGUF weights, not
  measured) and +16 GiB RSS at the peak, which matters on hosts with 32 GB of RAM.
- Workaround: none from the CLI. `--cache_dit` makes the copy useful, but only for directories
  and streaming.

## Possible fix

For the non-cached case, free the GPU tensors in place instead of copying them:

```diff
--- a/src/optimization/memory_manager.py
+++ b/src/optimization/memory_manager.py
@@ -1050,7 +1050,8 @@ def cleanup_dit(runner: Any, debug: Optional['Debug'] = None, cache_model: bool
         # Move model off GPU if needed
-        if param_device.type not in ['meta', 'cpu']:
+        # Only a cached model needs a copy on its offload device; otherwise it is freed below
+        if cache_model and param_device.type not in ['meta', 'cpu']:
             # MPS: skip CPU movement before deletion (unified memory, just causes sync)
@@ -1075,7 +1076,11 @@ def cleanup_dit(runner: Any, debug: Optional['Debug'] = None, cache_model: bool
     # 4. Complete cleanup if not caching
     if not cache_model:
-        release_model_memory(model=runner.dit, debug=debug)
+        runner.dit.to_empty(device="meta")   # drop every storage, no copy
         runner.dit = None
+        gc.collect()                         # BlockSwap leaves reference cycles
```

`to_empty` replaces each parameter and buffer through `Module._apply` (bypassing BlockSwap's
`.to()` guard); GGUF tensor subclasses need checking. Same change in `cleanup_vae`. This also
removes the leak of [12](12-swap-io-dit-leak.md) for the non-cached case.

Test: the command above must show Phase 2 shorter by ≈ 3.5 s, "After phase 2" at 0.5 GiB
allocated (VAE only), and max RSS ≈ 16 GiB lower (bench.py's `max_rss_gib`); `--cache_dit
--chunk_size N` runs must be unchanged.

## References

- [vram.md, Weights and the per-token model](../docs/vram.md#weights-and-the-per-token-model)
- [cli-flags.md, Host memory](../docs/cli-flags.md#host-memory)
