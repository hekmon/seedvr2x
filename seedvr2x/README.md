# seedvr2x

A [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) video upscaler for long runs: whole
episodes and films, on one GPU, as a job that stops and resumes. It takes one video file and
gives back its upscale as a lossless, correctly tagged master, or as a finished file with the
source's audio and subtitles.

> **Being built.** [DESIGN.md](DESIGN.md) is the specification, and this README and the
> [manual](MANUAL.md) describe version 1 as it specifies it. What they describe that the code
> doesn't do yet is listed under [Status](#status), and marked in the manual.

SeedVR2 is ByteDance Seed's one-step diffusion model for video restoration. seedvr2x runs
ByteDance's model code, as
[numz's ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
modified it, and replaces what surrounds the model: reading and writing video, cutting it into
pieces, fitting the GPU's memory. That is where nearly all of the
[26 bugs](../research/bugs/README.md) our research found in numz's CLI live, and where a long run
is won or lost.

- **Shot by shot.** seedvr2x finds the cuts itself and upscales each shot on its own, from cut
  to cut. Nothing crosses a cut, and no batch boundary is left inside a shot: a long shot runs
  in overlapping windows, blended before the frames are rebuilt, which removes 80% of a
  boundary's jump without softening anything ([stitching.md](../research/docs/stitching.md)).
- **A lossless master.** FFV1, 10-bit 4:2:0 by default, written from the model's float frames
  with no 8-bit step, tagged so that it looks like its source in any player, each frame with a
  checksum. seedvr2x has no encoder options: you compress the master your own way, afterwards,
  or during the run with your own command (`--segment-cmd`).
- **Colours corrected against the source.** The model shifts colours. seedvr2x's own
  correction, `split`, takes the source's coarse colour and brightness and keeps the model's
  detail: a lower colour error than numz's `lab` at every scale, and less flicker
  ([colour.md](../research/docs/colour.md)).
- **It stops and resumes.** Every piece of work is saved as it is made. The same command
  resumes a stopped job, and the result is the uninterrupted run's, bit for bit. `--until 07:30`
  stops cleanly before a set time.
- **It fits your GPU without tuning.** A planner sizes the work from the memory free on the
  card: how many frames the model takes at once, how much of the model waits in system memory,
  how the picture is tiled. It is developed on a 96 GB card and made for 16–32 GB ones too.
- **It refuses what it can't do right.** Version 1 is for SDR, progressive, constant-frame-rate
  sources. An HDR, interlaced or variable-frame-rate file is refused with a message that says
  why, and the manual says what to do with each.

The output looks redrawn, not merely enlarged: crisper lines and text, cleaner colours than a
plain upscale, the source's style kept. The [manual](MANUAL.md#what-the-model-does-to-a-picture)
says what that means on live action and at 4K, with the measurements.

## Two workflows

Both start from the source, one video file, and differ only at the end.

- **The regular workflow: a file in, a finished file out.** seedvr2x finds the cuts, upscales,
  and assembles one file with the source's audio, subtitles and chapters. ffmpeg is the only
  other tool. The video is lossless unless you give `--segment-cmd` your own encoder's command.
- **The sptenc workflow: the source in, a directory of segments out.** The segments are
  lossless masters, cut where the shots are, for `sptenc encode`, which encodes them as they are
  and takes the audio and subtitles back from the source. It stays lossless up to sptenc's
  encode, and takes the disk space that goes with it. seedvr2x never requires sptenc.

## Requirements

- **Linux**, x86_64. Windows comes later, as a best-effort port.
- **A CUDA GPU**: one NVIDIA card per job. How much memory a job takes depends on the output
  size: see the manual's [GPU memory](MANUAL.md#gpu-memory), and [Status](#status) below.
- **ffmpeg and ffprobe 7.1 or later** on `PATH`, built with zscale (libzimg), idet and FFV1.
  seedvr2x checks the build at startup and refuses to run without them, saying what is missing.
  The builds of [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) will do, and so will
  Ubuntu's ffmpeg package from 25.04 on (24.04's is 6.1). Why zscale and nothing else:
  [the manual](MANUAL.md#why-zscale-is-required).
- **[uv](https://docs.astral.sh/uv/)**, which installs Python 3.13 and the locked dependencies,
  PyTorch's CUDA 13.0 build among them.
- **Disk space**: 17 GB for the model files, downloaded once, and room for the output. A
  lossless master takes about 100–135 GiB per hour of 1080p, and four times that at 4K
  ([disk use](MANUAL.md#disk-use)).
- **FlashAttention 2 is optional.** seedvr2x uses it when it is installed, else PyTorch's own
  attention (SDPA), and logs which one runs. It buys no speed: on numz's CLI the backends gave
  the same DiT time within 1.5% ([attention.md](../research/docs/attention.md)). Their outputs
  differ slightly (55–56 dB apart), so a job is resumed with the backend it started with.
  PyPI only has FlashAttention's source, a long CUDA build:
  [setup_env.sh](../research/scripts/setup_env.sh) builds a wheel for your GPU, for Blackwell
  cards unless its `ARCH` says otherwise
  ([environment.md](../research/docs/environment.md)), installed with
  `uv pip install --no-deps <wheel>`. `uv sync` removes what its lock doesn't list: install the
  wheel again after a sync, or sync with `--inexact`.

## Install

```
git clone https://github.com/hekmon/seedvr2x.git
cd seedvr2x/seedvr2x
uv sync
uv run seedvr2x --version
```

`uv run seedvr2x` runs it from that directory. From anywhere else, name the project:
`uv run --project /path/to/seedvr2x/seedvr2x seedvr2x …`.

## Quick start

`uv run seedvr2x --help` lists every option; the [manual](MANUAL.md#options) explains them.

### A file in, a finished file out

```
uv run seedvr2x film.mkv -o film.1080p.mkv
```

The output's short side is 1080 pixels by default (`--resolution`), with square pixels at the
source's display aspect. The first run downloads the model files (17 GB). Then:

1. A first pass decodes the whole source: it counts the frames, checks the frame rate against
   them, and finds the cuts. It takes minutes, up to an hour for a long 4K film; the upscale
   takes hours or days.
2. Each shot is upscaled, and its frames are written as they come out.
3. The pieces are joined into `film.1080p.mkv`, with the source's other streams.

To get a compressed file instead of a lossless one, give your encoder's command. It runs once
per piece of the output, reads it lossless on its standard input, and writes the file seedvr2x
names:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16 {out}'
```

To run a job night after night, stop it at a set time, and run the same command again the next
evening:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --until 07:30
```

Ctrl-C, once, also stops cleanly: it finishes the piece of work in progress first. The same
command resumes.

### The source in, a directory of segments out, for sptenc

```
uv run seedvr2x film.mkv -o film.segments
uv run seedvr2x verify film.segments && sptenc encode film.segments film.1080p.mkv -f film.mkv
```

Any `-o` that isn't a video file's name is a directory (`.mp4`, `.mov` and the like are
refused): it gets the output segments, FFV1 masters of 5 seconds or more each
(`--min-segment`), their checksums, and `manifest.json`, the job's record. `seedvr2x verify`
decodes every frame and checks it against its checksum before sptenc trusts the directory.

### Looking before running

```
uv run seedvr2x film.mkv -o film.segments --plan film.cuts.txt
```

`--plan` stops before the upscale. It prints what the job would be on your card (the windows,
the memory, the time) and writes the cuts it found to `film.cuts.txt`, the doubtful ones
commented, for you to [check and edit](MANUAL.md#how-shots-are-found). The same command without
`--plan` then runs the job as planned; with `--cuts film.cuts.txt`, it runs with your cuts.

## Model files

seedvr2x's model files come from its own Hugging Face repository,
[hekmon/seedvr2x](https://huggingface.co/hekmon/seedvr2x), at a revision pinned in the code.
They are ByteDance's SeedVR2 weights and TransNetV2's, each converted by a script of
[../models/](../models/) that pins its inputs and checks its output: unofficial conversions,
neither ByteDance's files nor TransNetV2's authors'.

A run downloads only the files it uses, one by one, never the repository whole (122 GB):

| File | Size | Downloaded |
|---|---|---|
| `seedvr2x_ema_7b_sharp_fp16.safetensors` | 16.5 GB | by default: the sharp 7B, the model that upscales |
| `seedvr2x_ema_7b_fp16.safetensors` | 16.5 GB | only with `--dit-model seedvr2x_ema_7b_fp16.safetensors`: the regular 7B |
| `seedvr2x_ema_vae_fp16.safetensors` | 0.5 GB | always: the VAE, which turns frames into the model's input and back |
| `transnetv2.safetensors` | 30 MB | unless `--cuts` gives the cuts: the shot detector |

- **Where they go.** Into Hugging Face's cache, each file once: `~/.cache/huggingface`, or the
  directory `HF_HOME` (or `HF_HUB_CACHE`) names, to put it on another disk. The log names it.
  Before a download, the free space of its disk is checked against the file's size.
- **Checked before every run.** Each file is checked by its pinned size and SHA-256 before it
  loads, so a version of seedvr2x always runs the same bytes. A file that fails is refused,
  with both hashes and how to fetch it again; seedvr2x never deletes a model file.
- **Offline.** `HF_HUB_OFFLINE=1` keeps the network out: the files are taken from the cache.
  The same files downloaded another way, at another revision of the repository (`hf download`
  without `--revision`), are taken from the cache as they are.
- **A directory of your own.** `--model-dir DIR` reads the files from DIR instead, by name, and
  downloads nothing.
- **Nothing is sent.** The repository is public: no token is sent, even one you have stored.
  Hugging Face's telemetry is off: seedvr2x sets `HF_HUB_DISABLE_TELEMETRY=1` unless you have
  set it.
- **Recognised by content.** A model file is checked by its own tensors, not by its name.
  Anything but SeedVR2's 7B in float16, sharp or regular, is refused, saying what the file is.

### Which file to use

Version 1 runs the float16 files, the sharp 7B by default. Judged by eye on 75 crops at 4K and
1080p, it was preferred to the regular 7B or no different on 67 (preferred on 27, the regular
7B on 8), and it has the same memory and time ([the model](MANUAL.md#the-model)). The
repository also holds smaller files of both 7Bs, which seedvr2x will run in a later phase. The
model card gives the same tables.

| | File | Size | What you give up |
|---|---|---|---|
| **Recommended** | `seedvr2x_ema_7b_sharp_fp16.safetensors` (float16) | 16.5 GB | nothing: the reference |
| 8 GB tier (later) | `seedvr2x_ema_7b_sharp_int8_convrot.safetensors` (int8, rotated) | 8.3 GB | almost nothing: as close to the float16 model as another seed of it, or closer; faster on GeForce RTX 20 to 50 |
| 4 GB tier (later) | `seedvr2x_ema_7b_sharp_dyn.gguf` (GGUF, a type per matrix) | 4.8 GB | some fidelity: the closest 4 GB file to the float16 model at 1080p, yet further from it than another seed on some sources there (a cartoon's LPIPS and DISTS, an anime clip's flicker and colour); at 4K the closest 4 GB file too, and as close as another seed; saves memory only, a little slower |

These are the sharp 7B's; the regular 7B has the same files, named without `_sharp`.

**Every format.** A smaller file saves memory, sometimes time, and loses some precision. The
table measures that loss on the weights themselves, against ByteDance's float32 originals, and
on video, against the float16 model's own output.

| Format | Size (7B) | Multiply speed, against float16 | Error per weight: typical (worst layer) | On video: the sharp 7B | On video: the 7B | seedvr2x runs it |
|---|---|---|---|---|---|---|
| **float16** | 16.5 GB | 1x on every GPU: the reference | 0.02% (0.02%) | the reference | the reference | **version 1** |
| fp8, with a scale per tensor | 8.3 GB | 2x on RTX 40 and 50 (up to 3x measured on RTX 50) and on workstation Ada and Blackwell cards; 1x before RTX 40: memory only | 2.6% (2.7%) | as close as another seed or closer, multiplied in 8 or 16 bits | the same | later |
| int8, rotated, with a scale per row | 8.3 GB | 4x on GeForce RTX 20 to 50; 2x on workstation cards | 0.86% (1.08%) | as close as another seed or closer | the same | later |
| GGUF Q8_0 | 8.8 GB | 1x: memory only | 0.6% (0.6%) | as close as another seed or closer: the closest of all | the same | later |
| GGUF Q4_K | 4.8 GB | 1x: memory only, a little slower | 7.3% (7.9%); weighted 7.15% (9.0%) | further than a seed on cartoon, live action and anime | as close as another seed, softer on some sources | later |
| GGUF Q4_K, with an importance matrix | 4.8 GB | 1x: memory only, a little slower | 7.55% (8.8%); weighted 6.64% (7.4%) | further than a seed on a cartoon's DISTS | further than a seed on a cartoon's DISTS and live action's fine detail | later |
| GGUF, a type per matrix (Q3_K to Q5_K), with an importance matrix | 4.8 GB | 1x: memory only, a little slower | 7.55% (18%); weighted 6.31% (12%) | further than a seed on cartoon and an anime clip's flicker and colour | the closest 4 GB file: further than a seed on a cartoon's DISTS only | later |
| NVFP4, with searched scales | 4.8 GB | 8x on RTX 50; 4x on workstation Blackwell cards (RTX PRO); 1x before Blackwell: memory only | 8.8% (8.9%) | multiplied in 4 bits: clearly worse on DISTS (cartoon, anime); in 16 bits: further than a seed on anime and cartoon | in 4 bits: clearly worse on DISTS (cartoon, anime); in 16 bits: softer on live action | later |

How to read it:

- **Multiply speed is not job speed.** The column gives the GPU's peak multiply rate for the
  model's matrices, from NVIDIA's specifications, against the same card's 16-bit rate. Only the
  DiT gets faster, and at 1080p it takes about a fifth of a job: the VAE, which stays in
  float16, takes the rest. By our estimate, multiplies 3 to 4 times faster make the DiT about
  twice as fast, and a job about 10% shorter. No speed is measured here. The main gain of a
  smaller file is memory: on a 16–32 GB card, more frames at a time and less of the model moved
  out to system memory.
- **Memory only:** the weights are stored small but widened to 16 bits for every
  multiplication. The file saves memory, not time.
- **Error per weight:** how far each stored weight is from ByteDance's original, relative to
  the weights' own size: the median over the 288 matrices that hold 99% of the 7B's weights,
  and the worst of them. **Weighted,** for the GGUF files: each weight's error weighted by how
  much the layer's input uses its channel.
- **On video:** each file against its own model's float16 output at the same seed, on 8 clips
  at 1080p, judged against how much the float16 model's own seeds differ. "As close as another
  seed" means no score past 2.7 times the spread of 3 seeds, a line another seed of the float16
  model crosses on one score in twenty; "further than a seed" names where a file crosses it;
  "clearly worse" is many times past it.

The [model card](https://huggingface.co/hekmon/seedvr2x) has the full notes, the 4K results
and each file's SHA-256, origin and change; [VALIDATION.md](../models/VALIDATION.md) has the
runs, and [FORMATS.md](../models/FORMATS.md) the formats.

**Other repositories' files, against ours:**

| File | Repository | Size (7B) | Error per weight: typical (worst layer) | Against ours |
|---|---|---|---|---|
| float32, the original | ByteDance-Seed/SeedVR2-7B | 33 GB | 0, the reference | the source of every file here; too big to run |
| float16 (7B, sharp 7B, VAE) | numz/SeedVR2_comfyUI | 16.5 GB | 0.02% (0.02%) | **the same values**, checked element for element; only the header differs |
| float16 | Comfy-Org/SeedVR2 | 16.5 GB | 0.02% (0.02%) | the same values (one layer checked), with two text embeddings added |
| GGUF Q4_K_M | AInVFX/SeedVR2_comfyUI | 4.8 GB | 7.3% (7.9%) | **the same**: Q4_K on the same matrices, the same error; on video too (4 clips: within the seed band on nearly every score) |
| fp8, without scale, the last block in float16 | AInVFX/SeedVR2_comfyUI (numz's 7B fp8) | 8.5 GB | 2.8% (11%) | further: no scale, and its biases, embeddings and output layer in fp8 too (up to 15%) |
| fp8, without scale | Comfy-Org/SeedVR2 | 8.2 GB | 2.8% (17%) | further: every tensor in fp8, with no scale |
| NVFP4 | Comfy-Org/SeedVR2 | 4.8 GB | 9.5% (10.1%) | further: each block's scale from its largest weight |
| int8, rotated | Comfy-Org/SeedVR2 | 8.3 GB | 0.86% (1.08%) | the same method, comfy-kitchen's quantizer (one layer checked) |
| MXFP8 | Comfy-Org/SeedVR2 | 8.6 GB | not measured | Blackwell only; not made here |
| 3B (float16, fp8, GGUF) | numz/SeedVR2_comfyUI, AInVFX/SeedVR2_comfyUI | | | ByteDance's first 3B weights, which ByteDance replaced on 2025-06-22 (checked on the float16 and fp8 files); on video, at 1080p, the current weights' output differs from the first weights' more than two seeds of a 7B do, about as much as the 7B's from the sharp 7B's ([VALIDATION.md](../models/VALIDATION.md)) |

"The same" means weights as close to ByteDance's as ours; the files' bytes still differ, by
their header and by the precision they were made from.

Of these, version 1 runs numz's float16 7B files, given with `--model-dir` and `--dit-model`:
the log says such a file isn't one of the pinned ones, and the job's record keeps its SHA-256.
Comfy-Org's float16 7B is refused, for the text embeddings it holds beside the model's
tensors. seedvr2x runs no 3B, now or later: neither 3B in float16 (6.8 GB) comes as close to
the source as the sharp 7B's 4.8 GB file ([VALIDATION.md](../models/VALIDATION.md)).

## Documentation

- [MANUAL.md](MANUAL.md), for users: the terms in plain words, colour correction, the pipeline
  for SeedVR2 users coming from numz's CLI, how shots are found, what the model does to a
  picture, disk and memory, the refused sources and what to do with each, resume, integrity,
  and every option.
- [DESIGN.md](DESIGN.md): what is built and why, with the measurement behind each decision.
- [AGENTS.md](AGENTS.md): how to work on the code.
- [../research/](../research/): the measurements.

## Credits and licence

seedvr2x is under the Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) credits what it
includes:

- **SeedVR2's model code**, vendored in `src/seedvr2x/vendor/`: ByteDance Seed's
  [SeedVR](https://github.com/ByteDance-Seed/SeedVR), as modified by
  [numz's ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler),
  both Apache-2.0. The files changed for seedvr2x say so under their original header.
- **The shot detector, TransNetV2** (Tomáš Souček, MIT,
  [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)): its official PyTorch model,
  vendored unchanged in `src/seedvr2x/transnetv2/` with its licence, and its official weights
  converted to PyTorch by its own script, `transnetv2.safetensors` in seedvr2x's Hugging Face
  repository.
- **The colour correction's wavelet split** is StableSR's method, whose code is under a
  non-commercial licence: seedvr2x's is rewritten from the method, with none of that code.

The model files are under the Apache License 2.0, as ByteDance's, and TransNetV2's under the MIT
License: the [model card](https://huggingface.co/hekmon/seedvr2x) and its notices say what each
file is made from, and what was changed.

## Status

Built, and held by the tests: the pipeline per shot with its windows, the colour correction,
resume, the lossless masters and their checksums, `seedvr2x verify`, the model files' download
and checks, the refusals of sources, and the first pass: the shot detector (`--cut-threshold`,
`--cuts`, `--plan FILE`'s cut list), the frame index a resume seeks by, the frame rate's check
and `--frame-rate`, the combing warning.

Not built yet, though this README and the manual describe them:

- **The memory planner, BlockSwap and VAE tiling**, with their options (`--swap-blocks`,
  `--encode-tile`, `--decode-tile`), `--until`, and the plan and time estimate `--plan` prints.
  Until then nothing fits a job to the card. A shot runs in one window unless `--window` caps
  them, and the VAE runs untiled, with the whole DiT beside it: about 35 GiB for a 1080p decode
  and 15 GiB of weights. It is developed and measured on a 96 GB card, and 4K doesn't run at
  all (its untiled decode needs about 134 GiB).
- **Assembly and `--segment-cmd`**, the regular workflow's end. Until then `-o x.mkv` writes
  one FFV1 master, video only, and can't resume; a directory output can, and the sptenc
  workflow works as described.
- **In the first pass:** the frames' timeline held strictly to the declared rate, missing
  frames filled, the join on the frame grid that a refused directory is told of, and the
  playlist of possible cuts for VLC.
