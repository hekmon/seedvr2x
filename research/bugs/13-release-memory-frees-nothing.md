# 13. `release_model_memory` and `release_tensor_memory` free nothing (`.data.set_()` on an alias)

| | |
|---|---|
| Severity | memory |
| Status | measured (in the leak of [12](12-swap-io-dit-leak.md), and a standalone check) |
| Affected options | none directly; every model and tensor cleanup relies on it |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

The cleanup helpers release a tensor's memory with `tensor.data.set_()`. `.data` returns a new
tensor object that shares the storage; `set_()` empties that temporary object and leaves the
original tensor (and its storage) untouched. The helpers then log "Released memory from N params
and M buffers" while releasing nothing. Most of the time the memory goes away anyway because the
last reference is dropped right after; when a reference survives (the BlockSwap case), the model
stays on the GPU.

## Reproduction

Standalone (torch 2.14.1, CPU is enough):

```python
import torch
p = torch.nn.Parameter(torch.ones(1000))
p.data.set_()
print(p.numel(), p.untyped_storage().nbytes())   # 1000 4000: unchanged
t = torch.ones(1000); t.data.set_(); print(t.numel())   # 1000: unchanged
q = torch.nn.Parameter(torch.ones(1000)); q.data = torch.empty(0); print(q.numel())   # 0: freed
```

In a real run (`bugs-1080-bs5-swap18io`, see [12](12-swap-io-dit-leak.md)): "Released memory from
540 params and 54 buffers", then 8.11 GiB still allocated after Phase 2 instead of 0.50.

- Expected: after `release_model_memory`, the model's CUDA tensors hold no memory.
- Actual: nothing changes; the log claims otherwise.

## Root cause

`src/optimization/memory_manager.py:563-574`:

```python
for param in model.parameters():
    if param.is_cuda or param.is_mps:
        if param.numel() > 0:
            param.data.set_()
            released_params += 1
...
for buffer in model.buffers():
    if buffer.is_cuda or buffer.is_mps:
        if buffer.numel() > 0:
            buffer.data.set_()
```

and `release_tensor_memory`, `memory_manager.py:458-464`:

```python
if tensor.numel() > 0:
    tensor.data.set_()
```

`Tensor.data` is a detached shallow copy (new `TensorImpl`, same storage). `set_()` with no
argument points *that* object at an empty storage. The parameter keeps its own `TensorImpl` and
its storage reference. Assigning `param.data = …` (which swaps the parameter's data) would work;
calling an in-place method on `param.data` doesn't.

Callers: `cleanup_dit` and `cleanup_vae` (`memory_manager.py:1076-1078`, `1148-1150`), the latent
and sample lists in the phases (`src/core/generation_phases.py:747`, `1027`, `1051`, `1461`,
`1472`), text embeddings (`memory_manager.py:500-513`). Where the caller also drops its reference
(`ctx[...] = None`, `runner.vae = None`), refcounting frees the memory and the bug is invisible.

## Impact

- Silent: the helpers' only effect is a misleading log line, until something else holds a
  reference (BlockSwap's reference cycles, a cached runner, a debug structure); then the memory
  stays allocated for the rest of the phase or run, as in [12](12-swap-io-dit-leak.md).
- Anyone debugging memory trusts "Released memory from N params" and looks elsewhere.

## Possible fix

```diff
--- a/src/optimization/memory_manager.py
+++ b/src/optimization/memory_manager.py
@@ -460,7 +460,7 @@ def release_tensor_memory(tensor: Optional[torch.Tensor]) -> None:
     if tensor is not None and torch.is_tensor(tensor):
         # Release storage for all devices (CPU, CUDA, MPS)
         if tensor.numel() > 0:
-            tensor.data.set_()
+            tensor.set_()          # in place on the tensor itself, not on a .data alias
         tensor.grad = None
@@ -563,14 +563,14 @@ def release_model_memory(model: Optional[torch.nn.Module], debug: Optional['Debu
         for param in model.parameters():
             if param.is_cuda or param.is_mps:
                 if param.numel() > 0:
-                    param.data.set_()
+                    param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                     released_params += 1
                 param.grad = None
                 
         for buffer in model.buffers():
             if buffer.is_cuda or buffer.is_mps:
                 if buffer.numel() > 0:
-                    buffer.data.set_()
+                    buffer.data = torch.empty(0, dtype=buffer.dtype, device=buffer.device)
                     released_buffers += 1
```

Checked on torch 2.14.1: `tensor.set_()` directly on a plain tensor empties it; on a parameter
it raises "a leaf Variable that requires grad is being used in an in-place operation" unless run
under `torch.no_grad()`, hence the `.data =` assignment (or wrap the loop in `no_grad`).
`Module.to_empty(device="meta")` is the one-call alternative for a whole model.

Risks: freeing a tensor that is still in use elsewhere now really frees it (that code path would
have been wrong anyway). GGUF weights are a tensor subclass: check that `.data =` and `set_()`
behave on them.

Test: the snippet above, adapted to the helpers (assert `numel() == 0` and
`torch.cuda.memory_allocated()` back to its baseline); then the `bugs-1080-bs5-swap18io` run
must show 0.50 GiB allocated after Phase 2 even without the fix of [12](12-swap-io-dit-leak.md).

## References

- [vram.md, BlockSwap](../docs/vram.md#blockswap)
- [cli-flags.md, Devices, offload and BlockSwap](../docs/cli-flags.md#devices-offload-and-blockswap)
