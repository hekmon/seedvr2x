# 03. Image input: grayscale images crash, 16-bit images come out white

| | |
|---|---|
| Severity | crash (grayscale), wrong output (16-bit) |
| Status | measured |
| Affected options | image input (`.png .jpg .jpeg .bmp .tiff .tif .webp`), directly or in a directory |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

`extract_frames_from_image` assumes an 8-bit, 3- or 4-channel array. A grayscale image is a 2-D
array and crashes with `IndexError`; a 16-bit PNG or TIFF is divided by 255 instead of 65535, so
almost every pixel is above 1 and the output is white.

## Reproduction

```bash
python -c "import cv2, numpy as np; g = np.tile(np.linspace(20, 220, 320, dtype=np.uint8), (240, 1)); \
cv2.imwrite('gray8.png', g); rgb = np.dstack([g, g[::-1], np.full_like(g, 128)]); \
cv2.imwrite('rgb8.png', rgb); cv2.imwrite('rgb16.png', rgb.astype(np.uint16) * 257)"
python inference_cli.py gray8.png --output out_gray.png --model_dir /path/to/models --resolution 240
python inference_cli.py rgb16.png --output out16.png --model_dir /path/to/models --resolution 240
```

Measured (3B fp8, 320×240):

| Input | Expected | Actual |
|---|---|---|
| `gray8.png` (8-bit grayscale) | an upscaled image | `IndexError: tuple index out of range` at `inference_cli.py:329`, exit 1 |
| `rgb8.png` (control) | – | mean 121.8, min 19 (input mean 122.3) |
| `rgb16.png` (same picture, 16-bit) | ≈ the 8-bit result | mean 255.0, min 253: white |

## Root cause

`inference_cli.py:324-336`:

```python
frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
...
if frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
...
else:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to float32 and normalize
frame = frame.astype(np.float32) / 255.0
```

- `IMREAD_UNCHANGED` returns grayscale as `(H, W)`: `frame.shape[2]` raises. A grayscale image
  with alpha (2 channels) would pass the index but fail in `cvtColor(COLOR_BGR2RGB)`.
- `IMREAD_UNCHANGED` keeps the bit depth: a 16-bit file is `uint16` up to 65535, and `/ 255.0`
  gives values up to 257. These go through the whole pipeline, and Phase 4's clamp to [0, 1]
  (`src/core/generation_phases.py:1348`) turns the result white.
- The output is 8-bit anyway (`inference_cli.py:590`), so a 16-bit input loses its extra
  precision even once this is fixed; see [19](19-10bit-output-is-8-bit.md) for the same limit on
  video.

## Impact

- Grayscale scans and line art (common upscaling inputs) can't be processed; in a directory, the
  first grayscale file aborts the whole run.
- 16-bit PNG/TIFF (renders, scans, HDR-ish exports) silently produce a white image after the full
  processing time.
- Workaround: convert inputs to 8-bit RGB first (`convert in.png -depth 8 -type TrueColor out.png`,
  or `ffmpeg -i in.png -pix_fmt rgb24 out.png`).

## Possible fix

Normalize by the real dtype range and expand gray to RGB:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -325,15 +325,23 @@ def extract_frames_from_image(image_path: str) -> Tuple[torch.Tensor, float]:
     if frame is None:
         raise ValueError(f"Cannot open image file: {image_path}")
     
-    # Convert BGR(A) to RGB(A) based on channel count
-    if frame.shape[2] == 4:
+    # Grayscale (H, W) or (H, W, 1) / gray + alpha (H, W, 2) → RGB(A)
+    if frame.ndim == 2 or frame.shape[2] == 1:
+        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
+    elif frame.shape[2] == 2:
+        gray, alpha = frame[..., 0], frame[..., 1]
+        frame = np.dstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), alpha])
+    elif frame.shape[2] == 4:
         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
         debug.log(f"Detected RGBA image (alpha channel preserved)", category="file")
     else:
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
-    # Convert to float32 and normalize
-    frame = frame.astype(np.float32) / 255.0
+    # Convert to float32 and normalize by the dtype's range (uint8, uint16, float TIFF)
+    if np.issubdtype(frame.dtype, np.integer):
+        frame = frame.astype(np.float32) / np.iinfo(frame.dtype).max
+    else:
+        frame = np.clip(frame.astype(np.float32), 0.0, 1.0)
```

Optionally write 16-bit PNG/TIFF when the input was 16-bit (`(x * 65535).round().astype(np.uint16)`
in `_save_image_bgr`); the pipeline itself is bf16, so the gain is limited (about 9 significant
bits).

Test: the three images above; `rgb16.png` is `rgb8.png` × 257, so its result must be
bit-identical to the 8-bit one, and
the grayscale one must be processed like `rgb8.png` (same output size, no error).

## References

- [cli-flags.md, Input handling](../docs/cli-flags.md#input-handling)
