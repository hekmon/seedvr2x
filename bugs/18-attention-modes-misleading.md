# 18. `--attention_mode sageattn_3` never runs SageAttention 3, and `sageattn_2` doesn't run SA2's CUDA kernels

| | |
|---|---|
| Severity | UX-doc (misleading option; no speed impact measured) |
| Status | measured |
| Affected options | `--attention_mode sageattn_3`, `sageattn_2` (and silent fallbacks of all modes) |
| Version | SeedVR2 `4490bd1` (v2.5.24), SageAttention 2.2.0, sageattn3 1.0.0 |

## Summary

- `sageattn_3` only calls SA3 when every sequence of a varlen call has the same length, and
  otherwise falls back, per call and silently, to SA2's varlen kernel. The DiT's window attention
  never produces uniform calls, so SA3 never runs: `sageattn_3` is `sageattn_2` under another
  name.
- `sageattn_2` calls `sageattention.sageattn_varlen`, a Triton kernel (INT8 QK, FP16 PV: the
  SageAttention 1 algorithm). SA2's CUDA kernels (INT8 QK + FP8 PV) are only reachable through the
  batched `sageattn()` API. The mode also accepts the PyPI `sageattention` 1.0.6 (which is SA1).
- A backend that can't be imported falls back to another with a single setup-time warning.

None of this costs time today: all four backends give the same DiT time within ±1.5%, attention
being 4–9% of it. The problem is that the option names promise something else.

## Reproduction

```bash
python scripts/bench.py run probe-1080-bs5-sa3 --wrap scripts/attn_probe.py ... -- input.mp4 \
  --model_dir /path/to/models --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --batch_size 5 --load_cap 5 --attention_mode sageattn_3
python scripts/attn_probe.py --summary runs/probe-1080-bs5-sa3.attn.json
```

Measured with the probe (every DiT attention call recorded):

| Run | Windows per call | Distinct lengths | Uniform calls | Kernel that ran |
|---|---|---|---|---|
| 1080p, batch 5 | 50 / 60 | 12 | 0 / 36 | `sageattn_varlen` 36 / 36 |
| 1080p, batch 81 | 100 / 120 | 21 | 0 / 36 | `sageattn_varlen` 36 / 36 |
| 4K, batch 5 | 162 / 200 | 10 | 0 / 36 | `sageattn_varlen` 36 / 36 |
| 4K, batch 45 | 324 / 500 | 28 | 0 / 36 | `sageattn_varlen` 36 / 36 |

- Expected (from the name and help text "'sageattn_3' (Blackwell GPUs)"): SA3 kernels.
- Actual: SA2's Triton varlen kernel on every call, no log line.

## Root cause

1. The DiT's 3D windows are cut on a token grid that the window size rarely divides, and the
   shifted (Swin-style) layers add half windows at the borders: at 1080p the regular layers mix 4
   window sizes and the shifted ones 9 ([attention.md](../docs/attention.md#windows-are-almost-never-all-the-same-size)).
2. `call_sage_attn_3_varlen` (`src/optimization/compatibility.py:487-507`):
   ```python
   uniform_q = (seq_lens_q == seq_lens_q[0]).all()
   uniform_k = (seq_lens_k == seq_lens_k[0]).all()

   if not (uniform_q and uniform_k):
       # Fall back to SA2 for variable-length sequences
       # This is expected behavior - SA3 Blackwell doesn't support varlen natively
       if SAGE_ATTN_2_AVAILABLE:
           return call_sage_attn_2_varlen(...)
   ```
   The check also forces a GPU → CPU synchronization on every call (`.all()` used as a Python
   bool).
3. `call_sage_attn_2_varlen` calls `sageattn_varlen` (`compatibility.py:438-443`), which is
   Triton in SageAttention 2.2.0.
4. Availability is tested by import only (`compatibility.py:147-172`); `validate_attention_mode`
   (`compatibility.py:175-…`) returns a different mode with a WARNING when the import fails.

## Impact

- Users pick `sageattn_3` on Blackwell expecting FP4 speed, pay +1.3–3 GiB of DiT memory over
  `flash_attn_2` for it, and get SA2's Triton path.
- Benchmarks and bug reports about "SA3" in SeedVR2 are about SA2.
- Not a speed problem: at SeedVR2's window lengths (≈ 400–3300 tokens) FlashAttention 2's varlen
  kernel is faster than every alternative, grouped SA2 CUDA and grouped SA3 included
  ([attention.md](../docs/attention.md#is-the-length-grouping-patch-worth-it)).
- Workaround: use `--attention_mode flash_attn_2`.

## Possible fix

Documentation and logging rather than kernels (making SA3 run by grouping windows by length was
measured slower than FA2 at these lengths):

- Help text: say that `sageattn_3` falls back to `sageattn_2` for variable-length windows (always,
  in the DiT), that `sageattn_2` uses SageAttention's Triton varlen kernel, and recommend
  `flash_attn_2` when available.
- Log once per run, at the first fallback: "sageattn_3: variable-length windows, using
  SageAttention 2 varlen" (a module-level flag in `call_sage_attn_3_varlen`).
- Check the SageAttention version at import (`importlib.metadata.version("sageattention")`) and
  warn when it is < 2 (SA1 from PyPI).
- Cache the uniformity test per window layout (the layout only changes with the token grid),
  instead of `.all()` on GPU tensors at every call, to drop the per-call sync.

Test: `attn_probe.py` on a `sageattn_3` run shows the single log line; `probe_env.py` warns with
SA1 installed.

## References

- [attention.md, Consequences for `--attention_mode`](../docs/attention.md#consequences-for---attention_mode)
- [attention.md, Backend comparison](../docs/attention.md#backend-comparison-clean-runs)
- [environment.md, pitfall 2](../docs/environment.md#2-seedvr2-falls-back-silently-between-attention-backends), [pitfall 9](../docs/environment.md#9-seedvr2s-sageattn_2-doesnt-use-sageattention-2s-fast-cuda-kernels), [pitfall 1](../docs/environment.md#1-pip-install-sageattention-gives-you-sageattention-1-and-seedvr2-runs-it-as-sageattn_2)
