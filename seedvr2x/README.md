# seedvr2x

Our own SeedVR2 command-line upscaler. **In progress**: [DESIGN.md](DESIGN.md) is the specification, with the decisions so far and the open questions; [AGENTS.md](AGENTS.md) says how to work on the code.

It needs ffmpeg and ffprobe 7.1 or later on PATH, with zscale (libzimg), idet and ffv1: seedvr2x checks the build at startup.

Its model files come from its own Hugging Face repo, [hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x), at a revision pinned in the code. A run downloads only the files it uses, by default the sharp 7B (16.5 GB) and the VAE (0.5 GB), once, into Hugging Face's cache (`~/.cache/huggingface`, or `HF_HOME`), its disk's free space checked first; `HF_HUB_OFFLINE=1` keeps the network out. Each file is checked by its pinned size and SHA-256 before it loads, and never deleted. The same files downloaded at another revision of the repo (`hf download` without `--revision`) are taken from the cache as they are. No token is sent: the repo is public. Hugging Face's telemetry is off: seedvr2x sets `HF_HUB_DISABLE_TELEMETRY=1` unless you have set it. `--model-dir DIR` reads the files from a directory instead, by name, and downloads nothing.

Its shot detector, being built, is TransNetV2 (Tomáš Souček, MIT, [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)): its official PyTorch model, vendored unchanged in `src/seedvr2x/transnetv2/`, with its official weights converted to PyTorch by its own script, `transnetv2.safetensors` in seedvr2x's Hugging Face repo ([NOTICE](NOTICE)).

Planned direction, from the findings in [../research/](../research/):
- all Python, one process per job; standalone (a video file in, a finished file out), and connected to sptenc through files when used with it (the source → seedvr2x → `sptenc encode <dir>`)
- vendor the model code (DiT, VAE, sampler) from [../upstream/seedvr2-numz](../upstream/seedvr2-numz) and fix it in our copy, keeping the Apache-2.0 headers and a NOTICE; the submodules stay the untouched reference
- write our own orchestration and I/O: FFV1/PNG output from the float tensor, BT.709 tags, exact frame rate, the source as the only input, one video file (cuts detected), models kept loaded across segments
- a VRAM planner built on the measured memory models
- proper crossfades between batches, and batches aligned with scene cuts
