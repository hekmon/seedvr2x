# 24. RoPE "stability" wrapper: every DiT block uses the last block's RoPE table (late-binding closure)

| | |
|---|---|
| Severity | wrong output with numz's 7B fp8 file (its blocks 0–34 run on block 35's fp16 table, not their own fp8-rounded one); latent with the other files checked, whose blocks hold equal tables; the 7B's RoPE cache is bypassed |
| Status | confirmed on CPU with numz's own model classes (no weights) and the files' tables; effect on output from code |
| Affected options | every DiT run, 3B and 7B (`CompatibleDiT` wraps every loaded DiT); BlockSwap |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1, rotary-embedding-torch 0.9.1 |

## Summary

`CompatibleDiT._stabilize_rope_computations` replaces `get_axial_freqs` on each RoPE module with a
wrapper that calls `original_method`, a variable of the enclosing loop. Python closures bind late:
all wrappers read that variable when they are called, after the loop, so they all call the last
wrapped module's method. The module filter matches two modules per block, the block's
`NaRotaryEmbedding3d` (3B: `NaMMRotaryEmbedding3d`) and the `rotary_embedding_torch.RotaryEmbedding`
inside it, and the last match is the library module of the last block. As a result:

- every block's RoPE angle table is built from the last block's `rope.freqs` (block 35 on the 7B,
  block 31 on the 3B), on that block's device;
- the `lru_cache` on `get_axial_freqs` is never used: the wrappers call the library's uncached
  method. On the 7B, whose per-forward cache is disabled, every block recomputes the table of every
  window at every DiT forward.

The frequencies are constants, never trained, so the blocks of most files hold the same table and
their output is unaffected: 36 equal tensors of 10 values in the 7B fp16 file (and the sharp
7B's, and the Q4_K_M's, which keeps them in F16), 32 of 21 values in each 3B file. The exception
is numz's 7B fp8 file, `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors`, a plain
`e4m3fn` cast without scale tensors: blocks 0–34 share one table rounded to fp8, up to 5.8% off
the fp16 values (136.125 → 144, 269.25 → 256), and block 35 holds the fp16 table. numz runs every
block of that file on block 35's fp16 table.

## Reproduction

The pattern, standard library only (python 3.12):

```python
import types

class Rope:                                     # stands for each block's RoPE module
    def __init__(self, i): self.i = i
    def get_axial_freqs(self, *dims): return f"block {self.i}'s table"

def stabilize(modules):                         # the loop of compatibility.py:870-890
    for module in modules:
        original_method = module.get_axial_freqs
        def stable_rope_computation(self, *args, **kwargs):
            return original_method(*args, **kwargs)
        module.get_axial_freqs = types.MethodType(stable_rope_computation, module)

blocks = [Rope(i) for i in range(36)]
stabilize(blocks)
print(blocks[0].get_axial_freqs(4, 3, 3))       # block 35's table
```

On numz's own classes, on CPU (torch 2.14.1, rotary-embedding-torch 0.9.1): the 7B and 3B DiTs
built from `configs_7b/main.yaml` and `configs_3b/main.yaml` with numz's `create_object` on the
meta device, wrapped by `CompatibleDiT`, their closures inspected; then 4-block models with real
CPU tensors, block i's `rope.freqs` multiplied by 1 + i, each block's table compared with the one
its own unwrapped method returns:

| | 7B | 3B |
|---|---|---|
| Modules matching the filter | 72: `blocks.N.attn.rope` and `blocks.N.attn.rope.rope` | 64 |
| `*rope.freqs` tensors in the state dict | 36, 10 values each | 32, 21 values each |
| Closure cells / wrappers | 1 / 72 | 1 / 64 |
| Method every wrapper calls | `blocks.35.attn.rope.rope`'s `RotaryEmbedding.get_axial_freqs` | `blocks.31.attn.rope.rope`'s |
| 4-block model: blocks given the last block's table | 4 / 4 (their own: block 3 only) | 4 / 4 |
| `lru_cache` hits / misses after wrapping (8 calls) | 0 / 0 | |

The same holds through `get_freqs()` (the path the forward takes) and with BlockSwap's own wrapper
on top. With the last block's RoPE module on another device (meta in the test), block 0's table
comes out on that device.

## Root cause

`src/optimization/compatibility.py:870-890`:

```python
for name, module in self.dit_model.named_modules():
    if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
        ...
        original_method = module.get_axial_freqs
        ...
        def stable_rope_computation(self, *args, **kwargs):
            try:
                return original_method(*args, **kwargs)
            except Exception:
                return call_rope_with_stability(original_method, *args, **kwargs)

        module.get_axial_freqs = types.MethodType(stable_rope_computation, module)
```

- `original_method` is a local of `_stabilize_rope_computations`: the 72 closures share one cell,
  which holds the last value once the loop ends. The wrapper's `self` is unused.
- The filter matches `blocks.N.attn.rope` and its child `blocks.N.attn.rope.rope`
  (`named_modules()` yields a module before its children), so the last value is the library
  method of the last block's inner module.
- The wrapper is an instance attribute: it shadows the class-level `@lru_cache(maxsize=128)`
  `get_axial_freqs` (`src/models/dit_7b/rope.py:44-46`, `dit_3b/rope.py:44-46`). The library method
  it calls builds positions on `self.device` (its `dummy` buffer) and angles from `self.freqs`,
  uncached.
- 7B: `NaRotaryEmbedding3d.get_freqs` calls `self.get_axial_freqs(f, h, w)` once per window
  (`dit_7b/rope.py:95-111`), under `cache("rope_freqs_3d", ...)` (`rope.py:84`), but the 7B DiT's
  cache is disabled (`dit_7b/nadit.py:159`, `disable_cache: bool = True`; no caller passes it, as
  in ByteDance's code). Each block thus computes its own tables, and ByteDance's code gives each
  block its own `freqs`.
- 3B: the cache is enabled (`dit_3b/nadit.py:197`), keyed per window layout
  (`dit_3b/nablocks/attention/mmattn.py:188`, `dit_3b/rope.py:107-110`): the first block of each layout computes the table and
  all blocks reuse it. ByteDance's 3B code therefore already shares one table across blocks (the
  first block's); numz's makes it the last block's.
- `CompatibleDiT` wraps every DiT at load (`src/core/model_configuration.py:1185`).
- The wrapper can't do what its docstring says: NaNs raise no exception, and the retry
  (`call_rope_with_stability`, `compatibility.py:701-717`) repeats the same computation with
  autocast disabled, which `RotaryEmbedding.forward` already disables.

## Impact

- Output: none with tables equal in every block: the 7B fp16, sharp 7B fp16, 7B Q4_K_M, 3B fp16
  and 3B fp8 files (read on CPU). The frequencies are computed at construction (7B:
  `linspace(1, 128, 10)·π`; 3B: `10000^(−2k/42)`, 21 values) and never trained (ByteDance turns
  them into buffers, `dit_7b/rope.py:33-42`). The 3B fp8 file, also a plain `e4m3fn` cast, holds
  them rounded (5 of the 21 values at zero, the others up to 41% off), but alike in all 32 blocks.
- A checkpoint whose blocks hold different tables runs on the 7B with block 35's table in every
  block, where ByteDance's code uses each block's own. numz's 7B fp8 file is one (a conversion
  that stores blocks at different precisions): all 36 blocks run on block 35's fp16 table, angles
  in fp16 as with the 7B fp16 file, where blocks 0–34 would use their fp8-rounded table, converted
  at load to the compute dtype, bf16 (`compatibility.py:787-804`; rotary_embedding_torch computes
  the angles in the table's dtype). A fine-tune that trains or rescales the frequencies would be
  another case.
- Speed (7B, not measured): 1,980 table computations per DiT forward at 1080p batch 5 (50 and 60
  windows in the regular and shifted layers, 18 of each: [18](18-attention-modes-misleading.md)),
  about 30 small tensor operations each, after a GPU → CPU sync (`shape.tolist()`), where an intact
  cache computes each distinct window shape once per block. Negligible on the 3B.
- BlockSwap with all blocks swapped (blocks 0 to N − 1 are swapped: `blockswap.py:233`, `257`): the
  last block sits on the CPU while the others run, so their tables are computed on the CPU and
  copied to the GPU by `freqs.to(device=q.device, ...)` (`dit_7b/rope.py:85`). From code.
- `--debug` logs "Stabilized 72 RoPE modules" for 36.
- Workaround: none needed; with the 7B fp8 file the bug keeps the fp16 table, within 0.05% of
  the fp32 values, where the file's own is up to 5.8% off.

## Possible fix

Simplest: delete `_stabilize_rope_computations` (it only re-raises in practice). Or bind each
module's own method, and wrap only the modules that own the cache:

```diff
--- a/src/optimization/compatibility.py
+++ b/src/optimization/compatibility.py
@@ -870,5 +870,6 @@ class CompatibleDiT(torch.nn.Module):
         for name, module in self.dit_model.named_modules():
-            if "rope" in name.lower() and hasattr(module, "get_axial_freqs"):
+            # The block's RoPE module, not the rotary_embedding_torch module inside it
+            if "rope" in name.lower() and hasattr(module, "get_axial_freqs") and hasattr(module, "rope"):
                 # Check if already wrapped
                 if hasattr(module, '_rope_wrapped'):
                     continue
@@ -883,5 +884,6 @@ class CompatibleDiT(torch.nn.Module):
-                def stable_rope_computation(self, *args, **kwargs):
+                # Bind this module's method now: a plain closure reads the loop's last value
+                def stable_rope_computation(self, *args, _original=original_method, **kwargs):
                     try:
-                        return original_method(*args, **kwargs)
+                        return _original(*args, **kwargs)
                     except Exception:
-                        return call_rope_with_stability(original_method, *args, **kwargs)
+                        return call_rope_with_stability(_original, *args, **kwargs)
```

Either way each block computes its table from its own `freqs` through ByteDance's cache again (one
`lru_cache` of 128 entries for all blocks, keyed by module and window shape).

Test: the 4-block check above (block i's table equals its own unwrapped one, `lru_cache` hits > 0);
done on CPU with this diff applied: 7B and 3B, 4 of 4 blocks get their own table, 4 cache hits in 8
calls. With numz's 7B fp16 checkpoint the output should stay bit-identical (equal tables), and the
7B DiT time per batch drop by the table computations (to be measured). The 3B fp8's output
doesn't change either (equal tables); the 7B fp8's does: blocks 0–34 then run on their own
fp8-rounded table, in bf16, instead of block 35's fp16 one. Keeping its current output would take
the fp16 table in every block.

## References

- [numerics.md, What differs from ByteDance on the 7B fp16 path](../docs/numerics.md#what-differs-from-bytedance-on-the-7b-fp16-path) (the RoPE table's precision)
- [provenance.md, What numz changed in the model code](../docs/provenance.md#what-numz-changed-in-the-model-code)
