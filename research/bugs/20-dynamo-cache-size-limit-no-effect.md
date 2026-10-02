# 20. `--compile_dynamo_cache_size_limit` has no effect (overwritten by `--compile_dynamo_recompile_limit`)

| | |
|---|---|
| Severity | UX-doc |
| Status | confirmed on torch 2.14.1 (config check) |
| Affected options | `--compile_dynamo_cache_size_limit`, `--compile_dynamo_recompile_limit` |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

The CLI sets `torch._dynamo.config.cache_size_limit` and then `recompile_limit`. In current torch,
`cache_size_limit` is a deprecated alias of `recompile_limit`: the second assignment overwrites
the first, so `--compile_dynamo_cache_size_limit` is ignored and both read the recompile limit
(128 by default).

## Reproduction

```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 128
print(torch._dynamo.config.cache_size_limit, torch._dynamo.config.recompile_limit)
# torch 2.14.1: 128 128
```

- Expected: two independent limits, 64 and 128, as the help describes ("Max cached compiled
  versions per function" / "Max recompilation attempts before fallback to eager mode").
- Actual: one limit, set by whichever option is applied last (always the recompile one).

## Root cause

`src/core/model_configuration.py:1314-1350`:

```python
dynamo_cache_size_limit = compile_args.get('dynamo_cache_size_limit', 64)
dynamo_recompile_limit = compile_args.get('dynamo_recompile_limit', 128)
...
torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
torch._dynamo.config.recompile_limit = dynamo_recompile_limit
```

The two names refer to the same setting in torch (`cache_size_limit` was renamed
`recompile_limit`, the old name kept as an alias). The log line "Dynamo cache_size_limit: 64 |
recompile_limit: 128" (`model_configuration.py:1343-1344`) reports a value that isn't in effect.
The settings are global (`torch._dynamo.config`), shared by the DiT and VAE compiles.

## Impact

- Users tuning recompilation (several resolutions, a shorter last batch, directories) change
  `--compile_dynamo_cache_size_limit` and see no effect.
- Harmless otherwise: the effective limit is 128, well above torch's default of 8.

## Possible fix

Keep one option, or map the old one onto the new name:

```diff
--- a/src/core/model_configuration.py
+++ b/src/core/model_configuration.py
@@ -1346,8 +1346,11 @@ def _configure_torch_compile(compile_args: Dict[str, Any], model_type: str,
     try:
         import torch._dynamo
-        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
-        torch._dynamo.config.recompile_limit = dynamo_recompile_limit
+        # cache_size_limit is an alias of recompile_limit in recent torch: set one value
+        torch._dynamo.config.recompile_limit = max(dynamo_cache_size_limit, dynamo_recompile_limit)
```

and deprecate `--compile_dynamo_cache_size_limit` in the help (the ComfyUI torch-compile node has
the same pair of inputs, `src/interfaces/torch_compile_settings.py:63-85`). On torch versions
older than the rename, only `cache_size_limit` exists: set it with `hasattr` as a fallback.

Test: the snippet above after `_configure_torch_compile`, for both option orders.

## References

- [cli-flags.md, Attention and `torch.compile`](../docs/cli-flags.md#attention-and-torchcompile)
