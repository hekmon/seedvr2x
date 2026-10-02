# 15. `--compile_vae`: the exclusion of the causal 3D convs is dead code

| | |
|---|---|
| Severity | memory (suspected; see [14](14-compile-vae-doubles-memory.md)) |
| Status | from code (torch source checked) |
| Affected options | `--compile_vae` |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

Before compiling the VAE encoder and decoder, the code marks every `InflatedCausalConv3d` with
`submodule._dynamo_disable = True` to keep these dynamic-shape modules out of the graph. Torch
never reads such an attribute: the convs are traced and compiled with everything else, which is
what the comment says must not happen ("prevents recompilation issues with variable tensor
sizes").

## Reproduction

From the code. To see it in a run:

```bash
TORCH_LOGS=graph_breaks,recompiles python inference_cli.py input.mp4 --output out/ \
  --model_dir /path/to/models --resolution 720 --batch_size 9 --load_cap 9 --compile_vae
```

- Expected (per the code's intent): a graph break around each `InflatedCausalConv3d`, which runs
  eagerly.
- Actual (expected from the code, not run): no break at those modules; the convs are inside the
  compiled graphs, and new slice shapes recompile them.

`grep -rn _dynamo_disable` over torch 2.14.1's sources finds no reader of a module attribute with
that name (only an unrelated `__dynamo_disable` cache attribute on functions, in
`torch/_compile.py`).

## Root cause

`src/core/model_configuration.py:1394-1402`:

```python
def _disable_compile_for_dynamic_modules(module: torch.nn.Module) -> None:
    """
    Mark modules with dynamic shapes to be excluded from torch.compile.
    This prevents recompilation issues with variable tensor sizes.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, InflatedCausalConv3d):
            # Mark module to skip compilation
            submodule._dynamo_disable = True
```

called at `model_configuration.py:1437` and `1444` right before
`torch.compile(model.encoder, ...)` / `torch.compile(model.decoder, ...)`. Dynamo decides what to
skip from `torch._dynamo.disable` / `torch.compiler.disable` wrappers on the callable, skip files,
and config; an arbitrary attribute on an `nn.Module` instance has no effect.

## Impact

- `--compile_vae` users get a fully compiled VAE, convs included, with whatever memory and
  recompilation behaviour that implies; the measured ≈ 2× VAE memory
  ([14](14-compile-vae-doubles-memory.md)) is the prime suspect.
- Readers of the code believe the convs are excluded.

## Possible fix

Wrap the forward of each conv in a real disable:

```diff
--- a/src/core/model_configuration.py
+++ b/src/core/model_configuration.py
@@ -1399,4 +1399,5 @@ def _disable_compile_for_dynamic_modules(module: torch.nn.Module) -> None:
     for name, submodule in module.named_modules():
         if isinstance(submodule, InflatedCausalConv3d):
-            # Mark module to skip compilation
-            submodule._dynamo_disable = True
+            # Run these convs eagerly: graph break around each call
+            if not getattr(submodule, "_compile_disabled", False):
+                submodule.forward = torch.compiler.disable(submodule.forward)
+                submodule._compile_disabled = True
```

(or decorate `InflatedCausalConv3d.forward` with `@torch.compiler.disable` at class level, in
`src/models/video_vae_v3/modules/causal_inflation_lib.py`).

Trade-off: the VAE has many such convs, so the compiled graph will be cut into many small pieces
between them, and the speed-up (−16–19% VAE time today) may mostly disappear. That is the
experiment to run before deciding: if excluding the convs brings memory back to eager levels and
keeps a useful speed-up, keep it; otherwise drop the helper (and the misleading comment) and
document `--compile_vae`'s memory cost.

Test: `TORCH_LOGS=graph_breaks` shows the breaks; `vae_probe.py` peaks and times for
`vae-1080-bs9` eager, compiled as today, and compiled with the exclusion.

## References

- [vram.md, `torch.compile`](../docs/vram.md#torchcompile)
- [cli-flags.md, Attention and `torch.compile`](../docs/cli-flags.md#attention-and-torchcompile)
