# 14. `--compile_vae` about doubles the VAE's memory (the cause of the 4K OOM)

| | |
|---|---|
| Severity | memory |
| Status | measured; cause not isolated |
| Affected options | `--compile_vae` |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

With `--compile_vae`, the VAE encode and decode peaks are 1.7–2× the eager ones, the compiling
call needs even more, and 5.5 GiB stay allocated after the encode. The help sells it as a 15–25%
speed-up; nothing says it costs memory. It is what turned a 4K batch-5 run that fits a 96 GB card
eagerly (2.3 GiB to spare) into an OOM.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 9 --load_cap 18 \
  --compile_vae --debug
```

Measured (torch peak allocated, GiB):

| Run | Encode | Decode | Allocated after encode / DiT peak | VAE time per batch |
|---|---|---|---|---|
| 1080p batch 9, eager | 19.7 | 34.9 | – / 19.1 | encode 5.7 s, decode 11.4 s |
| 1080p batch 9, `--compile_vae` (`vae-1080-bs9-compilevae`) | 33.8 | 68.6 | +5.5 / 24.6 | 4.79 s, 9.30 s (−16%, −19%), first encode 108.5 s |
| 4K batch 5, eager (`vae-2160-bs5`) | 54.7 | 83.5 | – | – |
| 4K batch 5, `--compile_vae` + user flags (`vae-2160-bs5-userflags`) | 93.6 on the compiling call (OOM, retry), then 64.8 | OOM at 88.2 | – | – |

- Expected: memory close to the eager peaks (compiling changes the kernels, not the math; the
  DiT, compiled the same way, needs only +0.1–1.4 GiB), plus a bounded overhead while compiling.
- Actual: +71% encode, +97% decode, and allocations left behind.

## Root cause

Not isolated. What the code does (`src/core/model_configuration.py:1405-1460`):

```python
_disable_compile_for_dynamic_modules(model.encoder)
model.encoder = torch.compile(model.encoder, **settings)
...
_disable_compile_for_dynamic_modules(model.decoder)
model.decoder = torch.compile(model.decoder, **settings)
```

Candidates, to check in this order:

1. **The causal 3D convs are compiled although the code means to exclude them**
   ([15](15-compile-vae-exclusion-dead-code.md)). Each `InflatedCausalConv3d` does its own
   memory management in eager mode: it splits its input along H and W above 0.5 GiB
   (`vae.memory_limit.conv_max_mem`), concatenates the causal cache, and with the Conv3d
   workaround calls `torch.cudnn_convolution` directly
   ([vram.md](../docs/vram.md#how-the-vae-processes-a-batch)). Traced into one graph, those
   splits and the cache concatenations can become simultaneously live buffers.
2. **One graph per temporal slice shape**: the first slice (5 frames) and the following ones
   (4 frames) have different shapes; with `dynamic=False` each compiles separately, and each
   compiled graph keeps its own workspace/constant buffers. The 5.5 GiB left after encode point
   to buffers owned by compiled code (cudagraph pools aren't used in `default` mode).
3. **Inductor's memory planning** of a very large graph (the whole decoder for one slice) can keep
   more intermediates alive than eager's layer-by-layer freeing.

[`vae_probe.py`](../scripts/vae_probe.py) can measure each VAE call and slice; combined with
`TORCH_LOGS=graph_breaks,recompiles` and `torch.cuda.memory._record_memory_history()` on one
compiled decode, it should show which buffers are live at the peak.

## Impact

- Anyone enabling `--compile_vae` for speed on a card near its limit: an OOM in decode (or a
  slow OOM-retry in encode), after minutes of compilation.
- Even when it fits, it removes the headroom the VAE phases need; the gain is only 16–19% of VAE
  time.
- Workaround: don't use `--compile_vae` near the memory limit; use `--compile_dit` (−26 to −32%
  DiT time for +0.1–1.4 GiB).

## Possible fix

1. Make the exclusion real ([15](15-compile-vae-exclusion-dead-code.md)) and re-measure: if the
   memory comes back to the eager level, that was it.
2. Otherwise compile smaller units (each `ResnetBlock3D` / up/down block) instead of the whole
   encoder/decoder, so inductor's buffers are bounded by one block.
3. Until then, document the cost in the help ("≈ 2× VAE memory") and warn when `--compile_vae`
   is combined with an untiled VAE at a resolution where the eager peak is already above half the
   card.

Test: `vae-1080-bs9` eager vs compiled with `vae_probe.py`; the compiled peaks must be within a
few percent of 19.7 / 34.9 GiB and "After phase 1" must return to the VAE weights (0.5 GiB).

## References

- [vram.md, The 4K OOM, reproduced](../docs/vram.md#the-4k-oom-reproduced)
- [vram.md, `torch.compile`](../docs/vram.md#torchcompile)
- [cli-flags.md, Attention and `torch.compile`](../docs/cli-flags.md#attention-and-torchcompile)
