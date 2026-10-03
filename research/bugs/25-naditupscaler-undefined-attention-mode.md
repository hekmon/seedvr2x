# 25. `NaDiTUpscaler` (7B) uses an undefined `attention_mode`: dead code that raises `NameError` if built

| | |
|---|---|
| Severity | crash, latent (nothing builds the class) |
| Status | confirmed on CPU with numz's code (no weights) |
| Affected options | none: no option or config reaches the class |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

`src/models/dit_7b/nadit.py` holds a second DiT class, `NaDiTUpscaler` (ByteDance's, with an extra
`downscale` embedding; unused in ByteDance's code too). When numz added the attention backend to
the DiT, it passed `attention_mode=attention_mode` to the blocks in both classes, but added the
parameter to `NaDiT` only. In `NaDiTUpscaler.__init__` the name is undefined: building the class
raises `NameError`, also when `attention_mode=` is passed (it lands in `**kwargs`). Nothing builds
it: the configs name `NaDiT`, the model registry ignores the class name anyway, and nothing imports
`NaDiTUpscaler`.

## Reproduction

On CPU, with numz's code and the 7B config, on the meta device (no memory):

```python
cfg = load_config("configs_7b/main.yaml")                 # src.common.config
params = OmegaConf.to_object(cfg.dit.model); params.pop("__object__")
with torch.device("meta"):
    NaDiTUpscaler(**params)          # NameError: name 'attention_mode' is not defined
```

- `dis`: `NaDiTUpscaler.__init__` loads `attention_mode` with `LOAD_GLOBAL`, `NaDiT.__init__` with
  `LOAD_FAST`; the module has no global of that name.
- `create_object` on the 7B config with `__object__.name` set to `NaDiTUpscaler` builds a `NaDiT`.

## Root cause

- The signature (`src/models/dit_7b/nadit.py:200-227`) has no `attention_mode`; `NaDiT`'s does
  (`nadit.py:73`). The name is used at `nadit.py:290`, in the list comprehension that builds the
  blocks. Lines 131 and 290 come from the same numz commit (`e735c2a`, "V3 migration with GGUF
  fixes and attention optimizations"); otherwise the class is ByteDance's (`models/dit/nadit.py:191`
  at `e4de8c2`).
- Unreachable:
  - the configs name `NaDiT` (`configs_7b/main.yaml:7-10`, `configs_3b/main.yaml:7-10`), and are
    loaded without overrides (`src/core/model_configuration.py:718-721`);
  - `import_item` returns the registry entry for a registered path and ignores `name`
    (`src/common/config.py:102-104`); the registry maps `"dit_7b.nadit"` to `NaDiT`
    (`src/utils/model_registry.py:13`, `17-21`);
  - nothing imports `NaDiTUpscaler`.
- The constructor parameter only sets a default anyway: the attention mode is applied after loading,
  on every `FlashAttentionVarlen` (`model_configuration.py:1206-1209`).

## Impact

- None for users today.
- A trap for whoever reuses the class (ByteDance's `downscale`-conditioned variant), and for
  whoever edits `__object__.name` in a config: the registry silently builds `NaDiT` instead.

## Possible fix

Delete `NaDiTUpscaler` (unused here and in ByteDance's code), or give it the parameter:

```diff
--- a/src/models/dit_7b/nadit.py
+++ b/src/models/dit_7b/nadit.py
@@ -224,5 +224,6 @@ class NaDiTUpscaler(nn.Module):
         window_method: Optional[Tuple[str]] = None,
         temporal_window_size: int = None,
         temporal_shifted: bool = False,
+        attention_mode: str = 'sdpa',
         **kwargs,
     ):
```

and have `import_item` check the registry class's `__name__` against `name` (or key the registry on
both).

Test: the snippet above builds a 36-block model after the change (checked on CPU, on the meta
device); `create_object` with a `name` that doesn't match the registry raises.
