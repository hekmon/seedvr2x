# 01. Multi-GPU run on an image hangs forever

| | |
|---|---|
| Severity | crash (hang) |
| Status | measured (two workers on one GPU, `--cuda_device 0,0`) |
| Affected options | `--cuda_device N,M,…` with an image input, or a directory containing images |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

With several GPUs, an image (one frame) is split with `torch.chunk`, which returns fewer chunks
than GPUs. Fewer workers start than the parent waits for, and the parent blocks forever on
`Queue.get()` without a timeout. With `--temporal_overlap` a worker can get an empty chunk, crash,
and leave the parent waiting the same way. A directory with any image in it hangs at that file.

## Reproduction

```bash
python inference_cli.py image.png --output out.png --model_dir /path/to/models --cuda_device 0,1
```

- Expected: `out.png` written, process exits.
- Actual: one worker processes the image, then the CLI never returns (no output file, no error).
- Measured on a one-GPU host with `--cuda_device 0,0` (two workers on the same GPU; the hang
  doesn't depend on the device ids), a 320×240 PNG, 3B fp8: the log shows one "Starting upscaling
  generation", the image is done 8 s after start, and the process is still waiting when
  `timeout 120` kills it (exit 124).

## Root cause

`process_single_file` sends images to `_gpu_processing` when there is more than one device
(`inference_cli.py:582-583`). There, in pre-loaded frames mode:

1. The barrier and the collection loop are sized for `num_devices`
   (`inference_cli.py:1152-1156`, `1227-1232`):
   ```python
   done_barrier = mp.Barrier(num_devices + 1)
   ...
   while collected < num_devices:
       proc_idx, result_tensor = return_queue.get()
   ```
2. Without overlap, the frames are split with `torch.chunk` (`inference_cli.py:1213-1214`), which
   returns `min(frames, num_devices)` chunks: one chunk for one frame. `zip(device_list, chunks)`
   then starts one worker (`inference_cli.py:1216-1223`).
3. That worker returns its result and waits on `done_barrier.wait()` for `num_devices + 1`
   parties (`inference_cli.py:1100-1104`); the parent waits for a second result that never
   comes. Both block.
4. With `--temporal_overlap` (`inference_cli.py:1199-1212`), `chunk_with_overlap = 1 // N +
   overlap`, rounded up to the batch size, so the last worker's slice
   `frames_tensor[base_chunk_size:1]` is empty. That worker fails in `compute_generation_info`
   (`images[0]` of an empty tensor) and exits without putting anything in the queue: same hang.

The parent never checks whether a worker died, so any worker crash (OOM, missing model, bad
input) hangs multi-GPU runs, video included.

Related: in this pre-loaded mode, `prepend_frames` is not zeroed for workers other than 0
(compare `inference_cli.py:1060-1063` for videos with `1088-1097`). Only images reach this mode
today, so it has no effect yet.

## Impact

- Anyone running a multi-GPU command on an image or on a directory that mixes videos and images:
  the run hangs at the first image, with GPUs idle and no message.
- Any worker failure on a multi-GPU video also turns into a silent hang instead of an error.
- Workaround: process images with a single `--cuda_device`.

## Possible fix

An image can't be split, so process it on one device; and never wait for a worker that is gone.

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -580,7 +580,8 @@ def process_single_file(
     processing_start = time.time()
     # Process frames (multiprocessing only for multi-GPU)
     if len(device_list) > 1:
-        result = _gpu_processing(frames_tensor, device_list, args)
+        # One frame can't be split across GPUs: use a single worker on the first device
+        result = _gpu_processing(frames_tensor, device_list[:1], args)
     else:
         result = _single_gpu_direct_processing(frames_tensor, args, device_list[0], runner_cache)
@@ -1227,8 +1228,20 @@ def _gpu_processing(
     results_np = [None] * num_devices
     collected = 0
-    while collected < num_devices:
-        proc_idx, result_tensor = return_queue.get()
+    while collected < len(workers):
+        try:
+            proc_idx, result_tensor = return_queue.get(timeout=5)
+        except queue.Empty:
+            failed = [(i, p.exitcode) for i, p in enumerate(workers) if p.exitcode not in (None, 0)]
+            if failed:
+                for p in workers:
+                    p.terminate()
+                raise RuntimeError(f"GPU worker(s) exited before returning a result: {failed}")
+            continue
         results_np[proc_idx] = result_tensor.numpy()
         collected += 1
```

Also:
- size `done_barrier` with the number of workers actually started (create it after the chunks
  are known), and skip empty chunks in the pre-loaded mode;
- zero `prepend_frames` for workers ≠ 0 in the pre-loaded mode, as the video mode does;
- `import queue` at the top.

A worker that put its result and waits on the barrier has `exitcode is None`, so it isn't
mistaken for a failure.

Test: `--cuda_device 0,0` (works on a one-GPU machine) with an image, with and without
`--temporal_overlap 3`; a directory with one video and one image; a worker forced to fail (for
example a `--dit_model` file that isn't a valid checkpoint) must exit non-zero within seconds.

## References

- [cli-flags.md, Multi-GPU](../docs/cli-flags.md#multi-gpu)
