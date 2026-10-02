# Setting up the SeedVR2 environment (NVIDIA Blackwell, sm_120)

How to build a clean, reproducible Python environment for the SeedVR2 CLI
([numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)),
with **every attention backend SeedVR2 can use actually compiled and verified**, and the traps
found along the way.

Automated by [`scripts/setup_env.sh`](../scripts/setup_env.sh) and verified with
[`scripts/probe_env.py`](../scripts/probe_env.py).

## TL;DR

```bash
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /path/to/seedvr2
SEEDVR2_DIR=/path/to/seedvr2 scripts/setup_env.sh all
```

The venv is created directly in its final place (`$SEEDVR2_DIR/.venv`). Sources, wheels and logs
go to `$BUILD_DIR` (default: a `build/` directory next to the SeedVR2 checkout). No root is needed:
only `uv`, `git`, a CUDA toolkit and a host compiler.

## Reference stack

| Component | Version | Notes |
|---|---|---|
| GPU | RTX PRO 6000 Blackwell Workstation (sm_120, 96 GB) | Other sm_120 cards (RTX 50xx) should behave the same |
| SeedVR2 | `4490bd1` (v2.5.24, 2025-12-24) | |
| Python | 3.13 (uv-managed CPython) | SeedVR2 needs ≥ 3.12. SageAttention 3 was reported to need 3.13 to build |
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 | Stable releases from `download.pytorch.org/whl/cu130`, not nightlies |
| CUDA toolkit (nvcc) | 13.0 | Must match torch's CUDA major version (cu130 → 13.x). Only used to compile kernels |
| Host compiler | gcc 13.3 | |
| SageAttention 2 | `sageattention` 2.2.0 built from [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) `d1a57a5` | **Not** the PyPI package (see pitfalls) |
| SageAttention 3 | `sageattn3` 1.0.0 built from the same repo (`sageattention3_blackwell/`) | CUTLASS pinned to v4.8.0 |
| FlashAttention 2 | `flash-attn` v2.8.3.post1 built from source | Built for sm_120 only |
| FlashAttention 3 | not installable | Hopper (sm_90) only |
| ffmpeg | any build with `libx265` on `PATH` | Needed for `--video_backend ffmpeg --10bit` |

## Steps

The script runs these phases in order (`setup_env.sh <phase>...`):

1. **`venv`**: `uv venv --seed --python 3.13`, then torch/torchvision from the cu130 index, then
   SeedVR2's `requirements.txt`. uv keeps the already-installed torch because it satisfies the
   unpinned `torch` requirement. Finally the build dependencies (`setuptools wheel ninja packaging
   psutil`).
2. **`src`**: clone SageAttention, CUTLASS (for SA3) and flash-attention at pinned refs.
3. **`sage2`, `sage3`, `flash2`**: build each kernel as a **wheel** with
   `pip wheel --no-build-isolation --no-deps`, so it compiles against the venv's torch.
4. **`install`**: `uv pip install --no-deps` the wheels. They are then regular, tracked
   packages with metadata.
5. **`verify`**: `probe_env.py` (see [Verification](#verification)).

## Pitfalls (each one was hit for real)

### 1. `pip install sageattention` gives you SageAttention **1**, and SeedVR2 runs it as "sageattn_2"

The PyPI package `sageattention` stops at 1.0.6, which is SageAttention 1: Triton kernels only, no
compiled CUDA modules. SeedVR2's `sageattn_2` mode only checks that
`from sageattention import sageattn_varlen` works, which SA1 also satisfies. So the run "works",
but on SA1 kernels, which are no faster than SDPA on Blackwell.
SageAttention 2 (2.2.0) only exists as source: build it from the GitHub repo. A real SA2 install
contains compiled modules: `sageattention/_qattn_sm80*.so`, `_qattn_sm89*.so`, `_fused*.so`.

### 2. SeedVR2 falls back silently between attention backends

`src/optimization/compatibility.py::validate_attention_mode`:

| `--attention_mode` | Falls back to | When |
|---|---|---|
| `flash_attn_3` | `flash_attn_2`, then `sdpa` | `flash_attn_interface` not importable. Always the case on sm_120, since FA3 is Hopper-only |
| `flash_attn_2` | `sdpa` | `flash_attn` / `flash_attn_2_cuda` not importable |
| `sageattn_3` | `sageattn_2`, then `sdpa` | `sageattn3` not importable |
| `sageattn_2` | `sdpa` | `sageattention.sageattn_varlen` not importable |

On top of that, **`sageattn_3` falls back to `sageattn_2` per call** (`call_sage_attn_3_varlen`)
whenever the sequences of a varlen batch don't all have the same length. SA3's API is batched
only. The DiT's window attention produces such batches, so a "`sageattn_3`" run is in practice a
mix of SA3 and SA2.
The only log trace is a WARNING at setup time, so a run that works proves nothing: check with
`probe_env.py`.

### 3. Recent torch needs C++20, but the kernels' `setup.py` hard-code `-std=c++17`

`torch.utils.cpp_extension` (2.14) compiles extensions with `-std=c++20`, but only if the extension
doesn't pass its own `-std`. SageAttention 2, SageAttention 3 and flash-attention all pass
`-std=c++17`, which then overrides torch's choice and breaks the build against torch's headers.
The script rewrites `-std=c++17` → `-std=c++20` in each `setup.py` (`sed`, idempotent).

### 4. `ninja` must be on `PATH`, not just installed in the venv

`torch.utils.cpp_extension` uses ninja (parallel compilation, honoring `MAX_JOBS`) only if a
`ninja` binary is found on `PATH`. If you call the venv's python directly without activating the
venv, the venv's `ninja` isn't on `PATH`, and torch **silently compiles one file at a time**: one
`nvcc` process, load average around 2 on a 48-core machine. The script puts `$VENV/bin` first in
`PATH`. With it, about 80 compiler processes run in parallel.

### 5. Build wheels and install them, never copy `.so` files

Copying a `.so` into `site-packages` (or `setup.py build_ext --inplace` followed by a copy)
"works" but leaves a package with no metadata. uv/pip don't know it's there, `pip list` doesn't
show it, and nothing records which torch it was built against. A kernel built against one torch
version must be rebuilt for another (C++ ABI): keep the wheels, named and stored per stack.

### 6. Pin what the build scripts fetch, and limit the target architectures

- **SageAttention 3**'s `setup.py` `git clone`s CUTLASS **HEAD** if `csrc/cutlass` is missing. Clone a
  pinned tag there first. It also picks its target architecture from the GPU visible at build
  time (`sm_100a` / `sm_120a` / `sm_121a`), so build it on the target machine.
- **SageAttention 2** reads `TORCH_CUDA_ARCH_LIST` (e.g. `12.0`) and compiles `sm_120a` kernels.
- **flash-attention** compiles for `80;90;100;120` by default. Set `FLASH_ATTN_CUDA_ARCHS=120` to
  build only what you need, which is far faster. Also set `FLASH_ATTENTION_FORCE_BUILD=TRUE`,
  otherwise its `setup.py` first tries to download a prebuilt wheel from GitHub.

### 7. Don't move a venv after creating it

`uv venv` hard-codes the absolute path in the console-script shebangs (`.venv/bin/hf`,
`accelerate`, …) and in `activate` (`VIRTUAL_ENV=...`). After a move, `.venv/bin/python` still
works (it computes its prefix from its own location), but those scripts and `activate` point to
the old path. Create the venv where it will live.

### 8. Prefer stable torch to nightlies

The previous environment used torch `2.15.0.dev` nightly. It ran, but nightlies make results hard
to reproduce, and every compiled kernel is tied to the exact torch build. Stable cu130 builds
support sm_120 and Python 3.13.

### 9. SeedVR2's `sageattn_2` doesn't use SageAttention 2's fast CUDA kernels

SeedVR2 calls `sageattention.sageattn_varlen`. Even in the real SageAttention 2.2.0, that function is
a **Triton** kernel: INT8 QK with FP16 PV, i.e. the SageAttention 1 algorithm. SA2's CUDA kernels
(INT8 QK + FP8 PV, `_qattn_sm89`, the ones selected on sm_120) are only reachable through the
batched `sageattention.sageattn()` API, which SeedVR2 doesn't call for this mode. So
`--attention_mode sageattn_2` gains little over FlashAttention 2 (see the
[throughput table](#reference-results)); in SeedVR2 itself it gains nothing (measured in
[attention.md](attention.md#measurements)). Building SA2 properly still matters if you select
`sageattn_2` or `sageattn_3`: it is the per-call fallback of `sageattn_3`, and in practice the
only kernel that mode runs. [attention.md](attention.md#is-the-length-grouping-patch-worth-it)
recommends `flash_attn_2`.

### 10. `uv` "failed to hardlink" warning

It appears when the uv cache (in `$HOME`) and the venv sit on different filesystems. It's harmless:
uv copies the files instead. The script sets `UV_LINK_MODE=copy` to silence it.

## Runtime notes that depend on the environment

- **Memory allocator.** `inference_cli.py` does
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")`. Setting
  `PYTORCH_CUDA_ALLOC_CONF` yourself overrides it. `cudaMallocAsync` doesn't support CUDA graphs
  (`checkPoolLiveAllocations`), so `--compile_mode max-autotune` crashes with
  `RuntimeError: cudaMallocAsync does not yet support checkPoolLiveAllocations`.
  Use `max-autotune-no-cudagraphs`, or the native allocator.
- **Conv3d / cuDNN.** With torch ≥ 2.9 and cuDNN ≥ 9.10.2, SeedVR2 detects a known Conv3d bug
  that makes Conv3d use about 3× the memory. It enables its own workaround and prints
  `🔧 Conv3d workaround active`. The bundled cuDNN version therefore matters for VAE memory
  (torch 2.14.1+cu130 ships cuDNN 9.24).

## Verification

`scripts/probe_env.py <SEEDVR2_DIR>` (run with the venv's python) prints:
- the interpreter, torch, CUDA and cuDNN versions, the GPU, and the architectures torch was built for
- the installed attention/compile distributions
- SeedVR2's own `*_AVAILABLE` flags and the result of `validate_attention_mode` for each mode
- for each backend, a run of SeedVR2's own varlen wrapper on uniform and on variable-length
  batches, with the error against an SDPA reference
- a small throughput test

Then an end-to-end smoke test: 9 frames, no upscale, debug logs giving time and VRAM per phase.

```bash
cd "$SEEDVR2_DIR" && PATH=/path/to/ffmpeg/bin:$PATH .venv/bin/python inference_cli.py input.mkv \
  --output out/ --output_format mp4 --video_backend ffmpeg --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 9 --load_cap 9 \
  --attention_mode flash_attn_2 --debug
```

The `--dit_model` matters: the CLI's default is the 3B fp8 model, not the 7B fp16 used in these
notes ([cli-flags.md](cli-flags.md#models)). The [reference smoke test](#smoke-test-9-frames-19201080--19201080-7b-fp16-sageattn_3-no-offload-no-tiling) was
run with `--attention_mode sageattn_3`, before [attention.md](attention.md#measurements) showed
that SA3 never runs inside SeedVR2 and that all backends give the same DiT time (±1.5%):
use `flash_attn_2`.

## Reference results

Measured on 2026-10-02 with the reference stack above and an idle GPU.

### Build times (48-core host, `MAX_JOBS=16`, `NVCC_THREADS=4`, single target architecture)

| Wheel | Wall time | CPU time | Size |
|---|---|---|---|
| `sageattention-2.2.0` | 1 min 31 s | 8 min 46 s | 15.5 MB |
| `sageattn3-1.0.0` | 2 min 15 s | 2 min 13 s (a single big file) | 1.8 MB |
| `flash_attn-2.8.3.post1` | 5 min 24 s | 70 min 25 s | 66.1 MB |

The SA2 and FA2 builds ran concurrently. Without ninja on `PATH` (pitfall 4), they compile one
file at a time and take many times longer.

### Correctness: SeedVR2's wrappers against an SDPA reference (bf16, 24 heads × 128)

| Backend | Uniform 4×2048 (cosine) | Varlen 1024/3000/2048 (cosine) | Notes |
|---|---|---|---|
| `flash_attn_2` | 1.000000 | 1.000000 | Exact |
| `sageattn_2` | 0.999908 | 0.999908 | INT8 quantization noise |
| `sageattn_3` | **0.9817** (mean abs error 15× SA2) | 0.999908 | FP4. The varlen batch silently ran SA2 (identical numbers) |
| `flash_attn_3` | not available | not available | Falls back to `flash_attn_2` |

Random Gaussian inputs are a harsh test, but SA3's error is an order of magnitude above SA2's.
In SeedVR2 it doesn't show: SA3 never runs there
([attention.md](attention.md#consequences-for---attention_mode)).

### Throughput (ms per call, bf16, 24 heads × 128, uniform batches)

| Kernel | 8 × 4096 tokens | 4 × 16384 tokens |
|---|---|---|
| `sdpa` (SeedVR2's per-sequence loop) | 22.5 | 157.1 |
| `flash_attn_2` (SeedVR2) | 20.5 | 147.2 |
| `sageattn_2` (SeedVR2, Triton varlen) | 21.5 | 132.0 |
| `sageattn_3` (SeedVR2) | 20.0 | **84.5** |
| *SDPA batched (not used by SeedVR2)* | *21.5* | *154.3* |
| *SA2 batched CUDA `sageattn()` (not used by SeedVR2)* | *15.6* | *95.1* |

The low-bit kernels only pay off on long sequences, where SA3 reaches 1.75× FA2. At 4096 tokens
everything is within 10%, because quantization overhead eats the gain. **SeedVR2's attention
sequences are much shorter:** windowed attention gives about 400 to 3300 tokens depending on
batch size, whatever the resolution. Every call is also variable-length, which disables SA3:
measured, SA3 never runs and the backend doesn't change DiT time. See [attention.md](attention.md).

### Smoke test (9 frames, 1920×1080 → 1920×1080, 7B fp16, `sageattn_3`, no offload, no tiling)

| Phase | Time | Torch peak allocated |
|---|---|---|
| 1. VAE encode | 8.7 s | 19.7 GiB |
| 2. DiT (one sampling step) | 12.2 s | 19.6 GiB (15.9 GiB is weights) |
| 3. VAE decode | 13.4 s | 34.9 GiB |
| 4. Color correction (lab) | 5.0 s | 1.8 GiB |

Total 47 s, peak process RSS 19 GB. These are first-call timings, so they include warmup.
The same log's phase totals, which also count what surrounds these timers (model loading for the
DiT), are in
[benchmarking.md](benchmarking.md#reference-two-runs-on-the-reference-stack) (encode 8.80 s, DiT
16.97 s, decode 15.53 s). The GPU is power-capped and its speed varies by up to 40–60% between
sessions ([benchmarking.md](benchmarking.md#caveats)): compare absolute times only with runs made
back to back. SeedVR2 prints these memory figures as "GB"; they are GiB.
