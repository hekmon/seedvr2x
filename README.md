# seedvr2-research

Research on **SeedVR2**, ByteDance-Seed's one-step diffusion video upscaler/restorer, and our own
command-line upscaler built from what the research found.

| Path | What |
|---|---|
| [research/](research/) | Measurements and analysis of the [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) CLI: every option, VRAM per phase, attention, quality, the bugs found, and the scripts used to measure them |
| [seedvr2x/](seedvr2x/) | Our own CLI (not started yet) |
| [upstream/](upstream/) | Pinned, untouched submodules used as the code reference: `seedvr2-numz` (numz, `4490bd1`), `seedvr-bytedance` (ByteDance-Seed/SeedVR, `e4de8c2`) |

Clone with `git clone --recurse-submodules`, or run `git submodule update --init` after a plain clone.
