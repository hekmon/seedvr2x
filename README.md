# seedvr2-research

Research on **SeedVR2**, ByteDance-Seed's one-step diffusion video upscaler/restorer, and our own
command-line upscaler built from what the research found.

| Path | What |
|---|---|
| [research/](research/) | Measurements and analysis of the [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) CLI: every option, VRAM per phase, attention, quality, the bugs found, and the scripts used to measure them |
| [seedvr2x/](seedvr2x/) | Our own CLI, in progress: [DESIGN.md](seedvr2x/DESIGN.md) is its specification |
| [models/](models/) | The scripts that make seedvr2x's model files from ByteDance's fp32 masters (and TransNetV2's), their formats ([FORMATS.md](models/FORMATS.md)) and their validation on the GPU ([VALIDATION.md](models/VALIDATION.md)); the files themselves are on Hugging Face, [hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x) |
| [upstream/](upstream/) | Pinned, untouched submodules used as the code reference: `seedvr2-numz` (numz, `4490bd1`), `seedvr-bytedance` (ByteDance-Seed/SeedVR, `e4de8c2`) |

Clone with `git clone --recurse-submodules`, or run `git submodule update --init` after a plain clone.

## Licence

Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) credits the code adapted from other
projects, and [seedvr2x/NOTICE](seedvr2x/NOTICE) the SeedVR2 model code seedvr2x vendors
(ByteDance-Seed's SeedVR and numz's ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0). The submodules
in [upstream/](upstream/) keep their own licences. The model files, on Hugging Face
([hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x)), carry their own card and notices.
