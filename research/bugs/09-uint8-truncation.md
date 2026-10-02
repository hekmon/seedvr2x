# 09. Output frames are truncated to 8 bits instead of rounded: half a level darker

| | |
|---|---|
| Severity | wrong output (small, systematic) |
| Status | measured (real frames against the float output) |
| Affected options | every output (MP4 with both backends, PNG, images) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

The final [0, 1] frames are converted with `(x * 255).astype(np.uint8)`, which truncates. Below
mid-grey, pixels lose 0.5 level on average (up to 1) compared with rounding: a systematic
darkening of dark content. (Above mid-grey the bfloat16 grid of the frames makes truncation and
rounding agree, see below.)

## Reproduction

From the code; any run shows it. With `--color_correction lab`, which pulls the low frequencies
back to the input, the output's mean luma stayed 0.59 level below the input on clip A (and 0.63
on clip B), of which about 0.5 is this truncation:

```bash
python inference_cli.py input.mp4 --output out/ --output_format png --model_dir /path/to/models \
  --batch_size 21 --load_cap 21
python scripts/quality_metrics.py out/input --input input.mp4 --batch 21   # "Y shift" column
```

- Expected: `round(x * 255)`, mean error 0.
- Actual: `floor(x * 255)`, mean error −0.5 level below mid-grey.

## Root cause

Three conversions in `inference_cli.py`, one per output path:

```python
frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)        # :590, image
frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)   # :763, video
frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)   # :809, PNG sequence
```

NumPy's float → uint8 cast truncates toward zero. The same pattern is in
`src/core/alpha_upscaling.py:152` (`(images_np * 255).clip(0, 255).astype(np.uint8)`, the alpha
path) and `src/core/generation_utils.py:736` (tile debug overlay).

The values come from a bfloat16 `final_video` (`src/core/generation_phases.py:879`), converted
to float32 at `inference_cli.py:1009-1010`. In [0.5, 1) bf16 values are k/256, so `x * 255`
= k − k/256 has a fractional part above 0.5: rounding and truncation both give k − 1. Below 0.5
the bf16 grid is finer than 1/255, and truncation is on average 0.5 level below rounding. Dark
content (our anime clips: mean L\* 14–20) is almost entirely in that range.

## Impact

- The dark half of every output is on average half a level darker than it should be; on dark,
  flat content (where colour correction otherwise matches the input within 0.1 level) it is a
  constant offset.
- Small, but it biases every quality comparison against the input, and it is free to fix.
- The bf16 `final_video` itself limits the output to about 8 significant bits; see
  [19](19-10bit-output-is-8-bit.md).
- Measured on real frames (1080p, `lab`): the CLI's PNGs are exactly `floor(x × 255)`, 53–56% of
  the samples one level below `round(x × 255)`, mean −0.53 to −0.56 level
  ([output.md](../docs/output.md#validation)). At or above 0.5 the only samples that differ are
  exactly 0.5 (127.5: rounds to 128, truncates to 127).
- Workaround: [`ffv1_out.py`](../scripts/ffv1_out.py) writes a 16-bit RGB FFV1 master from the
  float frames, rounded (bit-exact `round(x × 65535)`).

## Possible fix

One helper used by every writer:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -719,6 +719,12 @@ def _stream_video_chunks(
+def _to_uint8(frames: torch.Tensor) -> np.ndarray:
+    """[0, 1] float frames → uint8, rounded to nearest."""
+    return (frames.float().mul(255.0).round_().clamp_(0, 255)
+            .to(torch.uint8).cpu().numpy())
+
+
 def _save_image_bgr(frame_np: np.ndarray, file_path: str) -> None:
@@ -760,7 +766,7 @@ def save_frames_to_video(
-    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
+    frames_np = _to_uint8(frames_tensor)
@@ -806,7 +812,7 @@ def save_frames_to_image(
-    frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
+    frames_np = _to_uint8(frames_tensor)
```

and the same at `inference_cli.py:590` and `src/core/alpha_upscaling.py:152`. Doing the
conversion in torch also avoids the float32 ×255 NumPy copy of the whole clip (≈ 50 MB per
1080p frame at that point, [cli-flags.md](../docs/cli-flags.md#host-memory)).

Risk: none functional; outputs change by at most one level.

Test: rerun the command above; the "Y shift" with `lab` should move from ≈ −0.6 to ≈ −0.1, and
a synthetic test can assert `_to_uint8(torch.tensor([0.2, 0.5, 0.998]))` = `[51, 128, 254]`.

## References

- [cli-flags.md, Output](../docs/cli-flags.md#output)
- [quality.md, Colour correction](../docs/quality.md#colour-correction) (the −0.59 level)
