# 04. `--output_format mp4` on an image crashes after the processing

| | |
|---|---|
| Severity | crash |
| Status | measured |
| Affected options | `--output_format mp4` with an image input (or a directory with images) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

For an image, the output path gets an `.mp4` extension, but the image is still written with
`cv2.imwrite`, which has no writer for `.mp4`. The run fails at the very end, after all the
processing, and leaves an empty output directory. In a directory run with
`--output_format mp4`, the first image aborts everything.

## Reproduction

```bash
python inference_cli.py image.png --output_format mp4 --output out/ --model_dir /path/to/models
```

- Expected: either a one-frame MP4, or an error before processing ("mp4 needs a video input").
- Actual (measured, 320×240 PNG, 3B fp8): the four phases complete ("Output assembled: 1
  frames"), then
  `cv2.error: ... could not find a writer for the specified extension in function 'imwrite_'`
  from `_save_image_bgr` (`inference_cli.py:733`), exit 1. `out/` is created and stays empty.

## Root cause

1. The output path for `mp4` is `<stem>.mp4` whatever the input type
   (`generate_output_path`, `inference_cli.py:413-419`):
   ```python
   if output_format == "png":
       ...
   else:
       output_path = base_dir / f"{input_name}{file_suffix}.mp4"
   ```
2. The image branch of `process_single_file` ignores `--output_format` and always saves a single
   image (`inference_cli.py:588-591`):
   ```python
   frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)
   _save_image_bgr(frame_np, output_path)
   ```
3. `_save_image_bgr` calls `cv2.imwrite(file_path, frame_bgr)` (`inference_cli.py:733`), which
   picks the encoder from the extension and raises for `.mp4`. (It also ignores `imwrite`'s
   `False` return, so other failures, such as an unwritable path, are silent.)

Nothing validates the combination before the models are loaded.

## Impact

- Users batch-converting a mixed folder with `--output_format mp4`: the first image kills the run
  after it has been processed.
- Wasted compute (the whole pipeline runs first) and no partial output.
- Workaround: leave `--output_format` unset for images (auto-detect gives PNG), or use `png`.

## Possible fix

Either write the frame as a one-frame video, or reject the combination early. The first keeps the
user's intent:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -586,9 +586,16 @@ def process_single_file(
     debug.log(f"Processing time: {time.time() - processing_start:.2f}s", category="timing")
     
-    # Save single image
-    os.makedirs(Path(output_path).parent, exist_ok=True)
-    frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)
-    _save_image_bgr(frame_np, output_path)
+    if args.output_format == "mp4":
+        # One-frame video, same writer as for video inputs
+        writer = save_frames_to_video(result[:1], output_path, 30.0,
+                                      video_backend=args.video_backend, use_10bit=args.use_10bit)
+        writer.release()
+    else:
+        os.makedirs(Path(output_path).parent, exist_ok=True)
+        frame_np = (result[0].cpu().numpy() * 255.0).astype(np.uint8)
+        _save_image_bgr(frame_np, output_path)
```

and make `_save_image_bgr` raise when `cv2.imwrite` returns `False`. An RGBA image would need its
alpha dropped (or a PNG fallback) for MP4.

If one-frame videos aren't wanted, check `input_type == "image" and args.output_format == "mp4"`
in `main()` and exit with an error before `download_weight`.

Test: the command above must produce a playable one-frame MP4 (or fail in under a second); a
directory with a video and an image and `--output_format mp4` must process both.

## References

- [cli-flags.md, Input and output](../docs/cli-flags.md#input-and-output)
