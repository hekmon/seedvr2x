# seedvr2x

A [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) video upscaler for long runs: whole
episodes and films, on one GPU, as a job that stops and resumes. It takes one video file and
gives back its upscale as a lossless, correctly tagged master, or as a finished file with the
source's audio and subtitles.

> **Version 0.1.0.** [DESIGN.md](DESIGN.md) is the specification, and this README and the
> [manual](MANUAL.md) describe version 1 as it is built. What comes in later versions is listed
> under [Status](#status).

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
  stops cleanly before a set time. Run once the job is finished, the same command says so at
  once and makes nothing.
- **It fits your GPU without tuning, and says how.** A planner sizes the work from the memory
  free on the card: how many frames the model takes at once, how much of the model waits in
  system memory, how the picture is tiled. Every run prints its plan, a line per choice with
  its reason and the option that changes it, then each unit's time left. It is developed on a
  96 GB card and checked on an RTX 5090, a 32 GB card driving a desktop.
- **Crisper frames, for more GPU time.** The model rebuilds 3 frames of every 4 softer, with
  doubled lines on fast motion. `--quality enhanced` or `maximum` runs each shot 2 or 4 times,
  shifted a frame each time, and keeps each frame from its best run
  ([the manual](MANUAL.md#several-runs-of-a-shot---quality) shows it).
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
- **The [sptenc](https://github.com/hekmon/sptenc) workflow: the source in, a directory of
  segments out.** sptenc, by seedvr2x's author, encodes a film shot by shot. The segments are
  lossless masters, cut where the shots are, for `sptenc encode`, which encodes them as they
  are and takes the audio and subtitles back from the source. It stays lossless up to
  sptenc's encode, and takes the disk space that goes with it. seedvr2x never requires sptenc.

## Requirements

- **Linux**, x86_64, WSL2 included ([under WSL2](MANUAL.md#under-wsl2)). Windows itself comes
  later, as a best-effort port.
- **An NVIDIA driver of the 580 series or later** (580.65.06 on Linux; under WSL2, the
  Windows driver), which PyTorch's CUDA 13.0 build needs, and a card that CUDA 13 supports:
  NVIDIA's release notes list them.
- **A CUDA GPU**: one NVIDIA card per job. What counts is the memory free at the start: at
  least 8.3 GiB for a 1080p output and 22.4 GiB for 4K (23.5 without FlashAttention 2) on
  Linux, more for longer windows and fewer swapped blocks ([GPU memory](MANUAL.md#gpu-memory)).
  [Under WSL2](MANUAL.md#under-wsl2), 10.7 and 24.8 GiB (25.9) to run, up to 1.5 times as
  long, and 12.7 and 26.8 GiB (27.9) to run at full speed.
- **System memory**: the plans shown in the manual took 19 to 28 GiB: up to 15.8 GiB pinned for
  what leaves the card, 2.3–2.5 GiB for the process, and a tiled shot's scratch up to an
  allowance ([small cards](MANUAL.md#small-cards)). A host short of it gets a plan that fits,
  or a refusal naming what it needs.
- **ffmpeg and ffprobe 7.1 or later** on `PATH`, built with zscale (libzimg), idet and FFV1.
  seedvr2x checks the build at startup and refuses to run without them, saying what is missing.
  The builds of [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) will do, and so will
  Ubuntu's ffmpeg package from 25.04 on (24.04's is 6.1). Why zscale and nothing else:
  [the manual](MANUAL.md#why-zscale-is-required).
- **[uv](https://docs.astral.sh/uv/)**, which installs Python 3.13 and the locked dependencies,
  PyTorch's CUDA 13.0 build among them.
- **Disk space**: 17 GB for the model files, downloaded once, and room for the output. A
  lossless master takes about 100–135 GiB per hour of 1080p, and four times that at 4K; a
  finished file needs about twice that while it is assembled, beside it. `--plan` and every run
  say how much the job will take against the disk's free space ([disk use](MANUAL.md#disk-use)).
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
source's display aspect: a 1080p source comes out at its own size, redrawn rather than
enlarged ([colour and shape](MANUAL.md#colour-and-shape-of-the-output)). The first run downloads
the model files (17 GB). Then:

1. The source's other streams (audio, subtitles, a cover) are copied once, ahead, so that one the
   finished file can't take is found at once rather than after the night's work. A first pass
   then decodes the whole source: it counts the frames, checks the frame rate against them, and
   finds the cuts. It takes minutes, up to an hour for a long 4K film; the upscale takes hours
   or days.
2. The plan is made and printed, a line per choice, the disk the job will take among them; the
   models load; a calibration, about a minute at 1080p and two and a half at 4K, measures the
   job's own steps. From then on each line says the time left and when the job should end
   ([reading a run](MANUAL.md#reading-a-run)). The log goes to the terminal (standard error)
   only: keep it with `2>&1 | tee -a film.log`.
3. Each shot is upscaled, and its frames are written as they come out.
4. The pieces are joined into `film.1080p.mkv`, with the source's other streams, and checked
   before it is put in place. Until then the job waits in `film.1080p.mkv.work`.

To get a compressed file instead of a lossless one, give your encoder's command. It runs once
per piece of the output, reads it lossless on its standard input, and writes the file seedvr2x
names:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --segment-cmd 'ffmpeg -i - -c:v libx265 -crf 16 {out}'
```

To run a job night after night, stop it at a set time, and run the same command again the next
evening; stopped by the clock, it exits with 75:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --until 07:30
```

Exit status 0 means finished. Run again then, the same command says
`film.1080p.mkv: finished already` within seconds and makes nothing: the job's record stays
beside the file, `film.1080p.mkv.manifest.json`, and another job to that name is refused.

For crisper frames at 4 times the GPU time, `--quality maximum`; `enhanced` takes twice the
time. On a long job, `--compile` saves about a tenth of the GPU time, for minutes of
compilation at each start: a short job loses ([the manual](MANUAL.md#compiling-the-vae---compile)
says where the line falls).

Ctrl-C, once, also stops cleanly: it finishes the piece of work in progress first. The same
command resumes.

### When a resume is refused: `--accept-env-change`

A resume promises the uninterrupted run's output, bit for bit. So the same command refuses to
go on when something that changes bits has changed since the job began, and says what. Two
such refusals name `--accept-env-change` as a way out:

```
uv run seedvr2x film.mkv -o film.1080p.mkv --accept-env-change
```

- **The environment changed:** another GPU, another PyTorch, CUDA or ffmpeg, an upgraded
  package. The NVIDIA driver isn't compared: updating it is no change. Without the flag, a job
  of tens of GPU hours would be lost to an upgrade or to a new card.
- **The card no longer has the memory the job's plan took,** another program holding part of
  it. The job's windows and tiles, which its record keeps so that its output stays the same,
  no longer fit. The refusal says the job is kept, what it lacks, and the ways out: quit what
  holds the card ([GPU memory](MANUAL.md#what-else-holds-the-card)) and run the same command,
  the output kept bit for bit; or the flag, by which the rest of the job is planned again for
  the memory free now, with shorter windows or smaller tiles, or on another card with more of
  it.

With the flag, what was made is kept, and the change is written in the job's record: what
changed, when, and how much was made before it. It is needed once: from there the new
environment, or the new windows and tiles, are the job's own, so a nightly script shouldn't
pass it every night. What it gives up: from there the frames are not the uninterrupted run's,
as another card's aren't. The upscale is as correct; with smaller tiles the colour drifts
slightly, which the colour correction mostly removes. A shot keeps its encode, and its
windows where all were done; windows begun under a cap that no longer fits are made again from
the shot's first. Nobody, the same command included, can make that output again bit for bit, and
`seedvr2x verify` checks that the files are as written, not that they equal another run.

It is not the default because the bit-for-bit promise is the point of a resume: leave it off
where an output must be made again exactly, and prefer freeing the card's memory when that is
possible. It doesn't cover seedvr2x itself: another version of its code is another job, refused
with or without the flag, so update seedvr2x between jobs, or finish a job with the commit it
began with (`git rev-parse HEAD` before it, `git checkout` and `uv sync` after an update). A
finished job needs no flag: run in another environment, it says the difference and makes nothing
(a file's job with only its assembly left isn't finished yet, and needs it under another PyTorch
or ffmpeg). The manual has the details ([resume](MANUAL.md#--accept-env-change)).

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

`--plan` stops before the upscale. It prints what the job would be on your card, a line per
choice, and how long it would take; on an RTX 5090, for a 4.4-minute short to 1080p:

```
05:37:34 plan: windows: 19 latents at most (73 frames), the longest the budget holds: 40 of 62 shots in one window, 60 joins; a longer window, fewer joins; --window N caps them
05:37:34 plan: blocks swapped: 35 of the DiT's 36, the fewest that hold the window: slower, not worse; --swap-blocks N
05:37:34 plan: decode: tiles of 1088 px overlapping by 64, 2 across, 2 down, the whole frame not fitting the budget, the largest that do, up to 2048 px: a slight colour drift, which split corrects; --decode-tile PX
05:40:01 time: about 1 h 31 min on this GPU as it runs now: 21 min 19 s for 62 encodes, 18 min 44 s for 122 windows, 51 min 22 s for the decode and write of 24 segments
05:40:01 time at each --quality: standard about 1 h 31 min (this job's), enhanced about 3 h 1 min, maximum about 6 h 0 min
```

It also writes the cuts it found to `film.cuts.txt`, the doubtful ones commented, for you to
[check and edit](MANUAL.md#how-shots-are-found), with two playlists to check them in VLC. The
same command without `--plan` then runs the job as planned, with no second first pass; with
`--cuts film.cuts.txt`, it runs with your cuts. The manual reads a whole run line by line
([reading a run](MANUAL.md#reading-a-run)).

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

- **Where they go.** Into Hugging Face's cache, each file once: `~/.cache/huggingface/hub`, or
  `$HF_HOME/hub`, or the directory `HF_HUB_CACHE` names, to put it on another disk. The log
  names it. Before a download, the free space of its disk is checked against every file left
  to download, so that a run short of room is refused before its first 16.5 GB.
- **Checked before every run.** Each file is checked by its pinned size and SHA-256 before it
  loads, so a version of seedvr2x always runs the same bytes. A file that fails is refused,
  with both sizes or both hashes and how to fetch it again; seedvr2x never deletes a model
  file.
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

Version 0.1.0 runs the float16 files, the sharp 7B by default,
`seedvr2x_ema_7b_sharp_fp16.safetensors`. Judged by eye on 75 crops at 4K and 1080p, it was
preferred to the regular 7B or no different on 67 (preferred on 27, the regular 7B on 8), and
it has the same memory and time ([the model](MANUAL.md#the-model)).

- **Smaller files come later.** The repository also holds int8, fp8, GGUF and NVFP4 files of
  both 7Bs, each measured against the float16 model on video; seedvr2x will run them in a later
  phase. The [model card](https://huggingface.co/hekmon/seedvr2x) has their tables: what each
  gives up, its speed, its error per weight, and other repositories' files against ours, with
  each file's SHA-256, origin and change. [FORMATS.md](../models/FORMATS.md) holds the formats
  and [VALIDATION.md](../models/VALIDATION.md) the runs.
- **Other repositories' files.** numz's float16 7B files hold the same values as ours, only
  their header differs: version 1 runs them, given with `--model-dir` and `--dit-model`, the log
  saying such a file isn't one of the pinned ones and the job's record keeping its SHA-256.
  Comfy-Org's float16 7B is refused, for the text embeddings it holds beside the model's
  tensors.
- **No 3B,** now or later: neither 3B in float16 (6.8 GB) comes as close to the source as the
  sharp 7B's 4.8 GB dynamic GGUF ([VALIDATION.md](../models/VALIDATION.md)).

## Documentation

- [MANUAL.md](MANUAL.md), for users: the terms in plain words, colour correction, the pipeline
  for SeedVR2 users coming from numz's CLI, how shots are found, what the model does to a
  picture, several runs of a shot (`--quality`), the finished file, disk and memory, the refused
  sources and what to do with each, resume and `--until`, integrity, a real run read line by
  line, and every option.
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

The manual's example picture, [images/quality-teeth.png](images/quality-teeth.png), is made
from frames of [Sol Levante](https://opencontent.netflix.com/), Netflix and Production I.G,
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/): frames of the SDR UHD master,
downscaled, compressed and upscaled with seedvr2x to compare its `--quality` levels.

The model files are under the Apache License 2.0, as ByteDance's, and TransNetV2's under the MIT
License: the [model card](https://huggingface.co/hekmon/seedvr2x) and its notices say what each
file is made from, and what was changed.

## Status

Version 0.1.0 is version 1 of [DESIGN.md](DESIGN.md), built and held by the tests on an RTX
5090, and on a 96 GB card until 2026-10-23: every item built since was checked on the RTX 5090
alone, the 96 GB card being away for repair, and v0.1.0 waits for its checks on it. Built: the
first pass and its shot detector, the pipeline per shot with its windows, `--quality`, the
colour correction, the memory planner with BlockSwap and VAE tiling, the time estimates and
`--until`, resume and `--accept-env-change`, the lossless masters and their checksums, the
finished file's assembly and its record beside it, `--segment-cmd`, `--compile` (the VAE alone),
and `seedvr2x verify`.

Later versions: HDR, the smaller model files of [Which file to use](#which-file-to-use), and
Windows itself.
