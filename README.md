# seedvr2-research

Research notes and experiments on **SeedVR2**, ByteDance-Seed's one-step diffusion video
upscaler/restorer, through the
[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) CLI.

Goals:
- understand what each of the CLI's ~45 options does, and measure its effect
- understand and model **VRAM** use per phase (VAE encode, DiT, VAE decode), and derive rules that
  carry over to GPUs with less memory
- isolate the options (offload, caching, BlockSwap, batch size, temporal overlap, VAE tiling,
  attention backend, torch.compile…) and measure their impact on memory, speed and quality

## Contents

| Path | What |
|---|---|
| [docs/environment.md](docs/environment.md) | How to build a clean environment with every attention backend (Blackwell / sm_120), the pitfalls, and reference results |
| [docs/attention.md](docs/attention.md) | DiT windowed attention: sequence lengths, why `sageattn_3` and `sageattn_2` don't do what their names say, what still needs verifying |
| [scripts/setup_env.sh](scripts/setup_env.sh) | Reproducible environment build: uv venv, stable torch, SageAttention 2/3 and FlashAttention 2 built as wheels |
| [scripts/probe_env.py](scripts/probe_env.py) | Checks an environment: versions, which attention backend really runs, accuracy and throughput |
| [docs/benchmarking.md](docs/benchmarking.md) | How runs are measured: the record's fields, what torch and NVML memory figures mean, caveats, reference runs |
| [scripts/bench.py](scripts/bench.py) | Measurement harness: runs the CLI, samples device memory (NVML), parses the debug log into per-phase JSON records and Markdown tables |

Experiments run on a remote GPU machine. Where that machine is and how it's reached is
deployment-specific and deliberately left out of this repo.
