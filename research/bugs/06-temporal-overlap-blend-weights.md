# 06. `--temporal_overlap` 1, 2 and 4 never blend; 3 and 5 mix a single frame

| | |
|---|---|
| Severity | wrong output |
| Status | measured |
| Affected options | `--temporal_overlap` (between batches, and between GPUs) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

The cross-fade weights of `blend_overlapping_frames` start at exactly 1 and end at exactly 0, and
from overlap 3 up they only ramp over the middle third. So overlap 1 keeps the previous batch's
frame, overlap 2 and 4 switch hard from one batch to the next, and overlap 3 and 5 mix one frame
50/50. The overlap frames are still computed twice: users pay for a crossfade they don't get.

## Reproduction

Weights of the previous batch on the overlap frames, from the code:

| Overlap | Weights | Frames really mixed |
|---|---|---|
| 1 | 1 | none (the new batch's first frame is discarded) |
| 2 | 1, 0 | none |
| 3 | 1, 0.5, 0 | one, 50/50 |
| 4 | 1, 1, 0, 0 | none |
| 5 | 1, 1, 0.5, 0, 0 | one, 50/50 |
| 6 | 1, 1, ≈ 0.9, ≈ 0.1, 0, 0 | two |

Measured on held drawings (frames where the input doesn't change, so any output change is
flicker), 45 frames, `--batch_size 21`, `--color_correction none`:

```bash
python inference_cli.py input.mp4 --output out/ --output_format png --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --load_cap 45 --batch_size 21 \
  --color_correction none --temporal_overlap 2
python scripts/quality_metrics.py out/input --input input.mp4 --batch 21 --temporal-overlap 2
```

| Overlap | Frames computed | Jump on held frames at the boundary (input noise 0.45) |
|---|---|---|
| 0 | 47 | 2.31 |
| 1 | 47 | 2.36 |
| 2 | 51 | 2.40 |
| 3 | 51 | 1.48 |
| 4 | 55 | 2.97 |

- Expected: the boundary jump decreases as the overlap grows.
- Actual: 1, 2 and 4 leave it unchanged (or worse) for up to 17% more compute at batch 21 and up
  to 60% at batch 5; only 3 reduces it, by about a third (runs `q-b-bs21-ov{1,2,3,4}`,
  `q-b-bs5-ov{1,2,3}`).

## Root cause

`src/core/generation_utils.py:299-312`:

```python
if overlap >= 3:
    t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
    blend_start = 1.0 / 3.0
    blend_end = 2.0 / 3.0
    u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
    w_prev_1d = 0.5 + 0.5 * torch.cos(torch.pi * u)  # Hann window
else:
    w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)
```

- `linspace(1, 0, steps=n)` includes both end points: the first overlap frame is 100% previous
  batch, the last 100% new batch. With n = 1 it is `[1]`; with n = 2, `[1, 0]`.
- From 3 up, `u` is clamped outside the middle third of `t ∈ [0, 1]`, so a third of the frames
  get weight 1 and a third weight 0. For n = 4, `t = [0, 1/3, 2/3, 1]` falls exactly on the
  clamp limits: `[1, 1, 0, 0]`.

The function is used for batch boundaries in Phase 3 (`src/core/generation_phases.py:988`) and to
join GPU workers (`inference_cli.py:1260`).

## Impact

- Users who set `--temporal_overlap 1`, `2` or `4` pay up to 60% more compute for an output
  whose boundaries are as visible as without overlap. The ComfyUI node shares the function.
- Even odd overlaps blend far less than the option suggests: overlap 9 mixes only three frames.
- Workaround: use 0, or an odd value ≥ 3 (3 cuts the jump by a third).

## Possible fix

Weights strictly between 0 and 1 for every overlap frame, with a smooth ramp over the whole
overlap:

```diff
--- a/src/core/generation_utils.py
+++ b/src/core/generation_utils.py
@@ -296,16 +296,11 @@ def blend_overlapping_frames(prev_tail: torch.Tensor, cur_head: torch.Tensor, ov
     device = prev_tail.device
     dtype = prev_tail.dtype
     
-    # Smooth crossfade with Hann window for overlap >= 3, linear for smaller overlaps
-    if overlap >= 3:
-        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
-        blend_start = 1.0 / 3.0
-        blend_end = 2.0 / 3.0
-        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
-        w_prev_1d = 0.5 + 0.5 * torch.cos(torch.pi * u)  # Hann window
-    else:
-        w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)
+    # Raised-cosine crossfade over every overlap frame, end points excluded:
+    # overlap 1 → [0.5], 2 → [0.75, 0.25], 3 → [0.85, 0.5, 0.15], ...
+    i = torch.arange(1, overlap + 1, device=device, dtype=torch.float32)
+    w_prev_1d = (0.5 + 0.5 * torch.cos(torch.pi * i / (overlap + 1))).to(dtype)
     
     w_prev = w_prev_1d.view(overlap, 1, 1, 1)
     w_cur = 1.0 - w_prev
```

Risks and notes:
- A blended frame averages two reconstructions and is slightly softer (measured −10% Laplacian
  variance at batch 5 / overlap 3, −15 to −20% on the mixed frames with the fix). A linear ramp
  is an alternative, measured slightly better (below).
- The first frame of every batch is less restored ([quality.md](../docs/quality.md#--prepend_frames)),
  and it is an overlap frame: giving it weight < 1 for the new batch is desirable.
- Outputs of existing workflows that use overlap change.

Test: rerun the table above with `quality_metrics.py --temporal-overlap N`; the boundary jump
should decrease monotonically with N, and the frame count must not change.

Measured with the corrected weights, patched in by [`blend_patch.py`](../scripts/blend_patch.py)
(`cosine` = the diff above, `linear` = 1 − i / (K + 1)), against a single-batch reference
([stitching.md](../docs/stitching.md#pixel-space-results-clip-b)): clip B, 81 frames, batch 21,
`--color_correction lab`; the jump a boundary adds on held drawings over the reference (0.78
without overlap):

| Overlap | Compute | Today's weights | Raised cosine (the diff) | Linear |
|---|---|---|---|---|
| 2 | +6% | 0.79 | 0.34 | 0.34 |
| 3 | +11% | 0.48 | 0.38 | 0.37 |
| 4 | +16% | 0.72 | 0.32 | 0.25 |
| 8 | +45% | 0.44 | 0.30 | 0.25 |

- The fix works: every overlap now reduces the jump, by 55–67% from K = 2 up, against 0–45% today.
- A linear ramp does slightly better than the raised cosine (its largest frame-to-frame step is
  smaller), and K = 4 is enough: K = 8 brings nothing more. Prefer `linear` in the fix.
- Measured softening of the mixed frames: −15 to −20% Laplacian variance (linear a little more
  than cosine), no ghosting (both renderings are of the same input frame).

## References

- [quality.md, `--temporal_overlap`](../docs/quality.md#--temporal_overlap)
- [cli-flags.md, Temporal](../docs/cli-flags.md#temporal)
