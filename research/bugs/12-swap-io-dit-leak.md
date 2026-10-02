# 12. `--swap_io_components` leaks every unswapped DiT block into the VAE decode

| | |
|---|---|
| Severity | memory |
| Status | measured (`--blocks_to_swap` 0 and 18) |
| Affected options | `--swap_io_components` with `--blocks_to_swap` below the block count (36 for 7B, 32 for 3B) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

After Phase 2, `cleanup_dit` decides whether the DiT is on the GPU by looking at its first
parameter only. With `--swap_io_components` that parameter belongs to an I/O module parked on
the CPU, so the move off the GPU is skipped. The model is then "released" by a function that
frees nothing ([13](13-release-memory-frees-nothing.md)), and the blocks that were not swapped
stay allocated through the decode and post-processing phases: 0.42 GiB per 7B fp16 block, 15.2
GiB without `--blocks_to_swap`. This is the option meant to save memory.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 5 --load_cap 5 \
  --dit_offload_device cpu --blocks_to_swap 18 --swap_io_components --debug
```

Measured (7B fp16, 1080p, torch peak allocated per phase, GiB):

| Run | Blocks swapped | `--swap_io_components` | Allocated after Phase 2 | Decode peak | Phase 4 peak | NVML run peak |
|---|---|---|---|---|---|---|
| `bugs-1080-bs5-swap18` | 18 | no | 0.50 | 21.7 | 0.1 | 25.4 |
| `bugs-1080-bs5-swap18io` | 18 | yes | **8.11** | **29.3** | **7.7** | 33.0 |
| `dit-1080-bs45-swap0io` (batch 45, tiled decode) | 0 | yes | – | **33.4** (18.2 expected) | – | 40.2 (32.1 expected) |

- Expected: after Phase 2 only the VAE (0.5 GiB) stays on the GPU.
- Actual: the 18 unswapped blocks (18 × 0.42 = 7.6 GiB) stay; with 0 blocks swapped, all 36
  (15.2 GiB). The log says "Released memory from 540 params and 54 buffers" and "DiT model
  deleted", but the allocation doesn't move. Without `--swap_io_components` the DiT is moved to
  the CPU ("Moving DiT from CUDA:0 to CPU") and nothing leaks.

## Root cause

1. BlockSwap places the I/O modules (`vid_in`, `txt_in`, `emb_in`, `vid_out`, …) on the offload
   device when `--swap_io_components` is set (`src/optimization/blockswap.py:317-326`), and
   blocks `> blocks_to_swap` on the GPU (`blockswap.py:370-378`).
2. `cleanup_dit` checks one parameter (`src/optimization/memory_manager.py:1048-1063`):
   ```python
   param_device = next(runner.dit.parameters()).device

   # Move model off GPU if needed
   if param_device.type not in ['meta', 'cpu']:
       ...
       manage_model_device(model=runner.dit, target_device=offload_target, ...)
   ```
   The first parameter is in `vid_in`, now on the CPU: the move is skipped, for any number of
   GPU-resident blocks.
3. `cleanup_blockswap` restores the forward methods and moves the I/O modules to the offload
   device (`src/optimization/blockswap.py:907-916`), not the blocks.
4. `release_model_memory` (`memory_manager.py:544-577`) calls `param.data.set_()` on every CUDA
   parameter. That empties a temporary alias and leaves the parameter's storage allocated
   ([13](13-release-memory-frees-nothing.md)); it still counts and logs them as released.
5. `runner.dit = None` (`memory_manager.py:1078`) doesn't free the model either: BlockSwap
   stored bound methods on the modules (`block.forward = types.MethodType(...)`,
   `model.to = types.MethodType(...)`; restored the same way at `blockswap.py:871` and `920`),
   which creates reference cycles. Only the cyclic garbage collector can free it, and
   `clear_memory` runs `gc.collect` only in "deep" mode (`memory_manager.py:308-313`), between
   chunks and at the end. The blocks therefore stay through Phases 3 and 4.

## Impact

- Users following the low-VRAM advice (`--swap_io_components` with a partial
  `--blocks_to_swap`): the decode, usually the largest phase, runs with 0.42 GiB × (36 − N)
  extra; on a small card that is an OOM in decode. With `--blocks_to_swap 0` the whole 7B
  (15.2 GiB) stays.
- The option itself saves only 0.16 GiB for ≈ 0.08 s per batch
  ([vram.md](../docs/vram.md#blockswap)).
- Workaround: don't use `--swap_io_components`, or only with every block swapped
  (`--blocks_to_swap 36` for 7B, 32 for 3B); the help's 8 GB example does the latter.

## Possible fix

Check every tensor, and free the GPU storage for real when the model is not cached:

```diff
--- a/src/optimization/memory_manager.py
+++ b/src/optimization/memory_manager.py
@@ -1046,9 +1046,14 @@ def cleanup_dit(runner: Any, debug: Optional['Debug'] = None, cache_model: bool
     # 2. Handle model offloading (for caching or before deletion)
     try:
-        param_device = next(runner.dit.parameters()).device
+        # BlockSwap can leave parameters on several devices: find any accelerator tensor
+        tensors = itertools.chain(runner.dit.parameters(), runner.dit.buffers())
+        param_device = next((t.device for t in tensors if t.device.type not in ('cpu', 'meta')),
+                            next(runner.dit.parameters()).device)
         
         # Move model off GPU if needed
         if param_device.type not in ['meta', 'cpu']:
```

(plus `import itertools`). With that, the existing path moves the whole model to the CPU
(BlockSwap bypass included) and the
leak is gone. It still pays the useless GPU → CPU copy of a model that is deleted right after
([16](16-dit-copied-to-cpu-before-deletion.md)); the better fix for the non-cached case is to
drop the GPU storage without copying, which also needs [13](13-release-memory-frees-nothing.md):

```python
if not cache_model:
    # free every parameter/buffer storage in place, whatever its device
    runner.dit.to_empty(device="meta")     # replaces each tensor; nothing is copied
    runner.dit = None
    gc.collect()                           # BlockSwap's bound methods form cycles
    clear_memory(debug=debug, deep=False, force=True)
```

`to_empty` goes through `Module._apply`, not the `.to()` that BlockSwap overrides; check that it
handles the GGUF weights (a tensor subclass, `GGUFTensor`), or free those with
`t.data = torch.empty(0, device=t.device)` instead. In
`cleanup_blockswap`, restore the methods with `del block.forward` / `del model.to` instead of
re-assigning bound methods, so the modules stop referencing themselves.

Test: the two `bugs-1080-bs5-swap18*` runs must show the same "After phase 2" allocation
(0.50 GiB) and the same decode peak; with `--cache_dit` and `--chunk_size`, the cached model must
still come back on the next chunk.

## References

- [vram.md, BlockSwap](../docs/vram.md#blockswap)
- [cli-flags.md, Devices, offload and BlockSwap](../docs/cli-flags.md#devices-offload-and-blockswap)
