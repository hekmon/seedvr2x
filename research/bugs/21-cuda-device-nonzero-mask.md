# 21. `--cuda_device N` (N ≠ 0): `CUDA_VISIBLE_DEVICES` is set after CUDA is initialized

| | |
|---|---|
| Severity | UX-doc (extra context on GPU 0, memory figures, checks and the compute dtype from the wrong GPU) |
| Status | mechanism confirmed on one GPU; the two-GPU behaviour from code, not reproduced |
| Affected options | `--cuda_device` with a single non-zero id; with a list, the parent process |
| Version | SeedVR2 `4490bd1` (v2.5.24), torch 2.14.1 |

## Summary

To validate `--cuda_device`, the CLI calls `torch.cuda.is_available()`, which initializes the
CUDA driver, and only then sets `CUDA_VISIBLE_DEVICES=N`. The runtime has already enumerated
every GPU, so the mask has no effect in this process. The run still lands on GPU N because the
code then uses `cuda:N`, but import-time code creates a context on GPU 0, the `--debug` memory
figures and BlockSwap's "< 5% free" check read GPU 0, and until CUDA is fully initialized torch's
NVML-based `device_count()` (which does read the mask) disagrees with the runtime. With a list of
devices the parent sets no mask, by design, but the same import-time code gives it a context on
GPU 0 for the whole run.

## Reproduction

On a two-GPU machine (not available here):

```bash
python inference_cli.py input.mp4 --output out/ --model_dir /path/to/models --cuda_device 1 --debug
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv    # during the run
```

- Expected: the process sees one GPU (physical 1, as `cuda:0`), uses only it, and reports its
  memory.
- Actual (from the code): contexts on GPU 0 and GPU 1; the work runs on GPU 1; the "[VRAM]" lines
  and the memory-pressure check report GPU 0. The debug log even says "Using device index 0 inside
  script (mapped to selected GPU)", which isn't what the code does.

The mechanism, checked on the one-GPU host (torch 2.14.1):

```python
import os, torch
torch.cuda.is_available()                       # what the CLI's validation does
os.environ["CUDA_VISIBLE_DEVICES"] = "1"        # GPU 1 doesn't exist on this host
torch._C._cuda_getDeviceCount()                 # 1: the runtime ignores the mask
torch.cuda.device_count()                       # 0: torch's NVML count reads the mask
torch.zeros(1, device="cuda:0")                 # works, on physical GPU 0
```

With `CUDA_VISIBLE_DEVICES=1` set before any CUDA call, the runtime sees 0 devices; and with
`PYTORCH_NVML_BASED_CUDA_CHECK=1`, `is_available()` and `device_count()` leave CUDA
uninitialized (`torch.cuda.is_initialized()` is `False`), so a mask set afterwards does apply.

## Root cause

`inference_cli.py:84-105`:

```python
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    # Temporary torch import for CUDA device validation only
    # Must happen before setting CUDA_VISIBLE_DEVICES and before main torch import
    import torch as _torch_check
    if _torch_check.cuda.is_available():
        available_count = _torch_check.cuda.device_count()
        ...
    # Set CUDA_VISIBLE_DEVICES for single GPU after validation
    if len(device_list_env) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]
```

- `_torch_check` is the same `torch` module as the main import: "temporary" doesn't undo the
  initialization. `is_available()` calls `cudaGetDeviceCount`, which runs `cuInit` and fixes the
  device list for the process.
- The device used is then built from the id itself: `device_list = ["1"]`
  (`inference_cli.py:1575-1576`) → `_device_id_to_name("1")` → `cuda:1` (`inference_cli.py:856`).
  Had the mask worked, `cuda:1` would not exist; the code relies on the mask *not* working.
- The context on GPU 0 is created at import, by the project imports that follow the mask
  (`inference_cli.py:115-134`): `src/optimization/memory_manager.py:132-138` calls
  `get_basic_vram_info(device=None)`, i.e. `torch.cuda.mem_get_info(cuda:0)`
  (`memory_manager.py:110-114`), to print the "Initial CUDA memory" line, and
  `src/optimization/compatibility.py:684-698` runs a bf16 matmul on `cuda:0` to choose the
  pipeline's compute dtype (`COMPUTE_DTYPE`, used at `src/core/generation_utils.py:380`). Every
  later `get_basic_vram_info` / `get_vram_usage` with `device=None` defaults to `cuda:0` too
  (`memory_manager.py:97-174`), as used by `Debug.log_memory_state` (`src/utils/debug.py:485-486`)
  and `clear_memory`'s pressure check (`memory_manager.py:263`).
- With a list (`--cuda_device 0,1`), only workers are masked: the parent masks a single id only
  (`inference_cli.py:103-105`), and sets each worker's id in the environment before spawning it
  (`inference_cli.py:1184`, `1217`). The parent runs the same imports, so it creates a context on
  physical GPU 0 and keeps it while it waits for the workers, even with `--cuda_device 1,2`. The
  NVML change below doesn't affect this one.
- The debug message is at `inference_cli.py:1566`.

## Impact

- Users who pick a second GPU because GPU 0 is busy (desktop, another job) still allocate a
  context (≈ 0.5–0.8 GiB) on GPU 0, and get memory logs for the wrong card.
- BlockSwap's `clear_memory(force=False)` decides on GPU 0's free memory: it may never empty the
  cache when GPU 1 is full, or empty it after every block when GPU 0 is full.
- The compute dtype is probed on GPU 0: with GPUs of different generations, the selected one may
  get bf16 without bf16 cuBLAS support, or fp16 for nothing. When GPU 0 can't take a context (full,
  or reserved in exclusive-process mode), the probe raises (only `CUBLAS_STATUS_NOT_SUPPORTED` is
  caught, `compatibility.py:692-695`) and the CLI fails at import, although another GPU was
  selected. Both from code.
- With a device list, the orchestrating parent holds a context on GPU 0 for the whole run, next to
  worker 0's, or on a GPU outside the list.
- Workaround: set `CUDA_VISIBLE_DEVICES=N` yourself and don't pass `--cuda_device` (the CLI then
  uses `cuda:0`, which is physical GPU N).

## Possible fix

Validate without initializing CUDA, then use index 0 inside the masked process:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -88,7 +88,9 @@ else:
         if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
-            # Temporary torch import for CUDA device validation only
-            # Must happen before setting CUDA_VISIBLE_DEVICES and before main torch import
+            # Count devices through NVML: cudaGetDeviceCount() would initialize CUDA,
+            # and CUDA_VISIBLE_DEVICES set afterwards would be ignored by this process
+            os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
             import torch as _torch_check
             if _torch_check.cuda.is_available():
                 available_count = _torch_check.cuda.device_count()
@@ -103,5 +105,6 @@ else:
             # Set CUDA_VISIBLE_DEVICES for single GPU after validation
             if len(device_list_env) == 1:
                 os.environ["CUDA_VISIBLE_DEVICES"] = device_list_env[0]
+                _CLI_MASKED_SINGLE_GPU = True    # in-process device is then cuda:0
```

and in `main()`, use `device_list = ["0"]` when `_CLI_MASKED_SINGLE_GPU` (defined `False` at
module level) is set (only then: a
user-set `CUDA_VISIBLE_DEVICES` keeps today's meaning of `--cuda_device` as an index into the
visible devices). Fix the debug message, and pass the inference device to
`get_basic_vram_info` / `get_vram_usage` instead of defaulting to `cuda:0`. Make the import-time
work lazy, at setup and on the inference device (`memory_manager.py:132-138`,
`compatibility.py:697-698`): that also removes the multi-GPU parent's context.

Test (two GPUs): `--cuda_device 1` must show a single compute process on GPU 1 in `nvidia-smi`,
and the "[VRAM]" totals of GPU 1; `--cuda_device 0,1` (workers) unchanged, and no compute process
of the parent left in `nvidia-smi` (three GPUs: none on GPU 0 with `--cuda_device 1,2`).

## References

- [cli-flags.md, Multi-GPU](../docs/cli-flags.md#multi-gpu) (single non-zero device)
- [cli-flags.md, Environment variables](../docs/cli-flags.md#environment-variables)
