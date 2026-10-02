# 05. `--prepend_frames` frames are never removed on a single GPU

| | |
|---|---|
| Severity | wrong output |
| Status | measured |
| Affected options | `--prepend_frames` (single GPU: no `--cuda_device` or one device) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

`--prepend_frames N` mirrors N frames before the first frame so that the clip's first frame is not
the first (less restored) frame of a batch. The help says they are "auto-removed", but they are
only removed in the multi-GPU path. On one GPU the output starts with the N mirrored frames: the
video is N frames longer, starts with a back-and-forth twitch, and every frame is shifted by N.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --output_format png --model_dir /path/to/models \
  --load_cap 45 --batch_size 21 --prepend_frames 4
ls out/input/ | wc -l
```

- Expected: 45 PNGs, the first one computed from input frame 0.
- Actual: 49 PNGs. The first four are computed from the mirrored frames 4, 3, 2, 1; frame 0 is
  `input_000004.png`.
- Measured twice: 45 frames in, 49 out with `--prepend_frames 4` (run `q-b-bs21-pp4`); 10 frames
  in, 11 out with `--prepend_frames 1` (run `vae-2160-bs5-ov1-pp1`, log and `ffprobe`). The log
  says "Output assembled: 49 frames" and never "Removing 4 prepended frames".

## Root cause

The frames are prepended inside the core, and removal is delegated to a caller that only exists
for multi-GPU:

1. `_process_frames_core` passes the option to `compute_generation_info`
   (`inference_cli.py:960`), which mirrors the frames in
   (`src/core/generation_utils.py:197-198`):
   ```python
   if prepend_frames > 0:
       images = pad_video_temporal(images, count=prepend_frames, temporal_dim=0, prepend=True, debug=debug)
   ```
2. Phase 4 has the removal code (`src/core/generation_phases.py:1391-1394`), but the CLI disables
   it (`inference_cli.py:996-1000`):
   ```python
   ctx = postprocess_all_batches(
       ctx=ctx, debug=debug, progress_callback=None,
       color_correction=args.color_correction,
       prepend_frames=0,  # Worker mode handles this in main process
   ```
3. Only `_gpu_processing` (multi-GPU) removes them, after joining the workers
   (`inference_cli.py:1279-1283`):
   ```python
   if args.prepend_frames > 0:
       if args.prepend_frames < result_tensor.shape[0]:
           ...
           result_tensor = result_tensor[args.prepend_frames:]
   ```
4. The single-GPU path, `process_single_file` → `_stream_video_chunks` → `_process_frames_core`
   (`inference_cli.py:535-560`, `696-714`), writes each chunk as returned and never strips them.
   `_stream_video_chunks` already zeroes `prepend_frames` after the first chunk
   (`inference_cli.py:673-675`), so only the first chunk is affected.

Images go through `_single_gpu_direct_processing` and are saved with `result[0]`
(`inference_cli.py:590`), i.e. the first prepended frame. For a one-frame input that frame is a
copy of the image, so the output happens to be right.

## Impact

- Everyone who uses `--prepend_frames` on one GPU, which is the common case. The help's own
  multi-GPU example uses `--prepend_frames 4`, so users copy it to single-GPU commands.
- The output has N extra frames at the start (a visible stutter: frames N…1 then 0…), audio muxed
  afterwards is off by N frames, and any frame-accurate comparison with the input is shifted.
- Workaround: cut the first N frames yourself (`ffmpeg -vf trim=start_frame=N,setpts=PTS-STARTPTS`,
  or delete the first N PNGs). `quality_metrics.py --drop-first N` does it for the metrics.

## Possible fix

Remove the frames where they were added, in the core, and drop the second removal in the
multi-GPU parent. Worker 0 is the only one with prepended frames (`inference_cli.py:1060-1063`),
and the parent's overlap blending uses the tail of each worker's result, not its head, so removing
them in the worker is equivalent.

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -996,7 +996,7 @@ def _process_frames_core(
     ctx = postprocess_all_batches(
         ctx=ctx, debug=debug, progress_callback=None,
         color_correction=args.color_correction,
-        prepend_frames=0,  # Worker mode handles this in main process
+        prepend_frames=args.prepend_frames,
         temporal_overlap=args.temporal_overlap,
         batch_size=args.batch_size
     )
@@ -1276,15 +1276,6 @@ def _gpu_processing(
         # Simple concatenation without overlap
         result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float32)
 
-    # Handle prepend_frames removal (multi-GPU safe - done after all workers complete)
-    if args.prepend_frames > 0:
-        if args.prepend_frames < result_tensor.shape[0]:
-            debug.log(f"Removing {args.prepend_frames} prepended frames from output", category="generation")
-            result_tensor = result_tensor[args.prepend_frames:]
-        else:
-            debug.log(f"prepend_frames ({args.prepend_frames}) >= total frames ({result_tensor.shape[0]}), skipping removal", 
-                     level="WARNING", category="generation", force=True)
-    
     return result_tensor
```

Points to check:
- `args` in `_process_frames_core` is the per-chunk / per-worker copy, whose `prepend_frames` is
  already 0 after the first chunk and on workers other than 0. The pre-loaded-frames worker mode
  (`inference_cli.py:1088-1097`) does not zero it for workers ≠ 0; it is only reached for images
  today ([01](01-multi-gpu-image-hang.md)), but zero it there too.
- Phase 4's removal keeps the frames when N ≥ the frame count (one-frame image): unchanged.

Test: 45 frames with `--prepend_frames 4` must give 45 output frames on one GPU, and the same 45
frames on `--cuda_device 0,1` with and without `--temporal_overlap`. With PNG output,
`quality_metrics.py <out> --input input.mp4 --batch 21 --prepend 4` (no `--drop-first`) must
find every output frame aligned with its input frame (frame 0 at ≈ 32 dB PSNR in, like the
others, instead of a mirrored frame).

## References

- [quality.md, `--prepend_frames`](../docs/quality.md#--prepend_frames): measurement and the effect on the first frame
- [cli-flags.md, Temporal](../docs/cli-flags.md#temporal)
