# 08. `--latent_noise_scale`: timestep shift computed from (h, w, c) instead of (frames, h, w)

| | |
|---|---|
| Severity | wrong output, unless training used the same convention ([caveat](#caveat-the-training-convention-is-unknown)) |
| Status | from code; its effect measured at 1080p. Inherited from ByteDance's reference scripts, dormant there |
| Affected options | `--latent_noise_scale` > 0 |
| Version | SeedVR2 `4490bd1` (v2.5.24); ByteDance SeedVR `e4de8c2` |

## Summary

The latent noise level goes through the sampler's resolution-dependent timestep shift, but the
shape passed to it is `x.shape[1:]` of a `(t, h, w, c)` latent, i.e. `(h, w, c)`, which the
shift reads as `(frames, height, width)`. The shift then depends on the resolution only: it is
what a 64-frame batch should get, so smaller batches get too much noise. At the default batch 5
at 1080p the shift is 4× too large, and `--latent_noise_scale 0.1` replaces 36% of the condition
with noise instead of 12%, which visibly washes the image out.

The call comes from ByteDance's SeedVR2 inference scripts, which hard-code the scale to 0, so
it never runs there; numz exposed the scale as an option and kept the call.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --output_format png --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --batch_size 21 --load_cap 21 \
  --color_correction none --latent_noise_scale 0.1
```

Measured (clip A, 1080p, batch 21, 21 frames; runs `q-a-ln{0.1,0.25,0.5}` against no noise):

| `--latent_noise_scale` | t applied (shift ≈ 5.0) | t with the intended shift (2.26 at batch 21) | PSNR vs no noise | Y shift (levels) | Laplacian variance (no noise: 11.6) |
|---|---|---|---|---|---|
| 0.1 | 0.36 | 0.20 | 35.4 dB | +2.94 (+1.5 vs no noise) | 8.5 |
| 0.25 | 0.63 | 0.43 | 24.4 dB | +8.63 | 5.2 |
| 0.5 | 0.83 | 0.69 | 16.5 dB | +23.87 | 8.4 |

- Expected: a small scale adds a little variation; the help presents the range 0.0–1.0.
- Actual: 0.1 is already 1.5 levels brighter and 27% softer; 0.5 is unusable (ΔE 20).

## Root cause

`src/core/generation_phases.py:689-697`:

```python
def _add_noise(x, aug_noise):
    if latent_noise_scale == 0.0:
        return x
    t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
    shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
    t = runner.timestep_transform(t, shape)
    x = runner.schedule.forward(x, aug_noise, t)
```

`x` is the encoded latent of one batch, `(t, h, w, c)` with c = 16: `get_condition` unpacks it
the same way (`src/core/infer.py:55`, `t, h, w, c = latent.shape`). `x.shape[1:]` is therefore
`(h, w, 16)`.

`timestep_transform` (`src/core/infer.py:281-311`) reads its argument as `(frames, h, w)` in
latent units:

```python
frames = (latents_shapes[:, 0] - 1) * vt + 1      # vt = 4
heights = latents_shapes[:, 1] * vs                # vs = 8
widths = latents_shapes[:, 2] * vs
...
vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
shift = torch.where(frames > 1, vid_shift_fn(heights * widths * frames), img_shift_fn(heights * widths))
...
timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
```

At 1080p (latent 136 × 240 × 16) it computes frames = 541, height = 1920, width = 128: a
"video" of 133 M pixel-frames, shift ≈ 4.98, whatever the batch size. With the right
`(t, h, w)`, batch 5 is 5 × 1088 × 1920 = 10.4 M pixel-frames, shift ≈ 1.24. In general the
wrong product is ≈ 4h × 8w × 128 = 4096·h·w pixel-frames (h, w = latent size) against
F × 8h × 8w = 64·F·h·w for F frames: they match for F ≈ 64 at any resolution.

| Output, batch | Shift today | Intended shift |
|---|---|---|
| 720p, any | 2.7 | – |
| 1080p, 1 (image) | 5.0 | 5.5 (image formula) |
| 1080p, 5 | 5.0 | 1.24 |
| 1080p, 21 | 5.0 | 2.26 |
| 1080p, 45 | 5.0 | 3.79 |
| 1080p, 81 | 5.0 | 6.08 |
| 4K, any | 17.1 | – |

`schedule.forward` then mixes `(1 − t)·x + t·noise` with t ∈ [0, 1] after dividing by T = 1000,
and `aug_noise` is `0.1 × sampling noise + 0.05 × fresh noise` (std ≈ 0.11,
`generation_phases.py:683`): t is the fraction of the condition replaced by a near-zero signal.

`timestep_transform` itself is ByteDance's code, unchanged, and consistent with the latent
layout: the VAE encode moves channels last (`b c t h w -> b t h w c`,
`src/core/infer.py:187`; the encode log prints e.g. `Latents shape: torch.Size([3, 136, 240,
16])`, i.e. t, h, w, c). The bug is in the caller.

The caller is copied from ByteDance's reference scripts (SeedVR `e4de8c2`,
`projects/inference_seedvr2_3b.py:95-115`, identical in `inference_seedvr2_7b.py:94-114`):
same `_add_noise`, same `shape = torch.tensor(x.shape[1:])[None]` and
`runner.timestep_transform(t, shape)`. There `cond_noise_scale = 0.0` is hard-coded, so the
wrong shape never matters. Two differences: ByteDance's `aug_noise` is plain `randn_like`
(std 1), and its `_add_noise` has no early return for 0 (t = 0 leaves x unchanged anyway). The
SeedVR (v1) scripts (`inference_seedvr_3b.py:93-113`, `inference_seedvr_7b.py:95-115`) use the
same call with `cond_noise_scale = 0.1`, so there it is live.

### Caveat: the training convention is unknown

ByteDance's training code isn't published. If training noised the condition with the same
`(h, w, c)` shape, the current shift is what the model saw, and "fixing" it would move
inference away from it. The v1 scripts ship 0.1 with this call, which doesn't tell whether the
convention is deliberate. The default 0 (ByteDance's own value for SeedVR2) is unaffected
either way.

## Impact

- Anyone using `--latent_noise_scale` at the default batch 5 gets about 3× the intended noise
  level at 1080p (t = 0.36 instead of 0.12 for s = 0.1), and more at 4K (shift 17 instead of
  2.2: t = 0.65 instead of 0.20). The strength doesn't follow the batch size as designed.
- Workaround: divide the scale. To get the intended t at 1080p batch 5, use s' = t / (shift −
  (shift − 1)·t) with shift = 4.98, e.g. s = 0.1 (t = 0.12) → s' ≈ 0.027. Or leave it at 0: on
  our content it only degraded the output ([quality.md](../docs/quality.md#noise-scales)).

## Possible fix

Fix it in the orchestration layer, where the noise is added; `timestep_transform` needs no
change. Given the [caveat](#caveat-the-training-convention-is-unknown), ship it as an
experiment (for instance for heavily compressed sources, where some condition noise could
help), not as a plain fix, or behind a separate option.

```diff
--- a/src/core/generation_phases.py
+++ b/src/core/generation_phases.py
@@ -690,7 +690,8 @@ def upscale_all_batches(
                 if latent_noise_scale == 0.0:
                     return x
                 t = torch.tensor([1000.0], device=ctx['dit_device'], dtype=ctx['compute_dtype']) * latent_noise_scale
-                shape = torch.tensor(x.shape[1:], device=ctx['dit_device'])[None]
+                # x is (t, h, w, c): the shift wants the latent (t, h, w)
+                shape = torch.tensor(x.shape[:-1], device=ctx['dit_device'])[None]
                 t = runner.timestep_transform(t, shape)
```

Risk: existing workflows tuned around the current strength change output (much weaker noise for
batches under ≈ 64 frames, slightly stronger above), and the result may be further from the
training conditions than today. Mention it in the changelog.

Test: log `t` for 1080p batch 5 and batch 21 (expect 0.12 and 0.20 for s = 0.1); rerun
`q-a-ln0.1` and check with `quality_metrics.py --ref` that the output is now closer to the
no-noise run than 35.4 dB.

## References

- [quality.md, Noise scales](../docs/quality.md#noise-scales)
- [cli-flags.md, Quality](../docs/cli-flags.md#quality)
