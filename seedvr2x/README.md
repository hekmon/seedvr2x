# seedvr2x

Our own SeedVR2 command-line upscaler. **In progress**: [DESIGN.md](DESIGN.md) is the specification, with the decisions so far and the open questions; [AGENTS.md](AGENTS.md) says how to work on the code.

It needs ffmpeg and ffprobe 7.1 or later on PATH, with zscale (libzimg) and ffv1: seedvr2x checks the build at startup.

Planned direction, from the findings in [../research/](../research/):
- all Python, one process per job; standalone (a video file in, a finished file out), and connected to sptenc through files when used with it (the source → seedvr2x → `sptenc encode <dir>`)
- vendor the model code (DiT, VAE, sampler) from [../upstream/seedvr2-numz](../upstream/seedvr2-numz) and fix it in our copy, keeping the Apache-2.0 headers and a NOTICE; the submodules stay the untouched reference
- write our own orchestration and I/O: FFV1/PNG output from the float tensor, BT.709 tags, exact frame rate, the source as the only input, one video file (cuts detected), models kept loaded across segments
- a VRAM planner built on the measured memory models
- proper crossfades between batches, and batches aligned with scene cuts
