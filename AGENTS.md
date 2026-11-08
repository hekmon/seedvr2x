# seedvr2x: context for contributors and coding agents

Read this, then [DESIGN.md](DESIGN.md), before changing any code.

## What this is

seedvr2x is a SeedVR2 video upscaler for long runs, built from what [research/](research/)
measured on numz's CLI. DESIGN.md is its specification: what to build and why, with the
measurement behind each decision. The code follows it. A decision the code shows to be wrong or
missing is settled in DESIGN.md first, not changed in passing.

## Layout

```
AGENTS.md DESIGN.md README.md LICENSE NOTICE
core/                    the library: distribution seedvr2x-core, import seedvr2x (uv_build)
  pyproject.toml .python-version LICENSE NOTICE
  seedvr2x/              no src/ folder: the package sits at core/seedvr2x/
    model.py weights.py pull.py blockswap.py int8.py sampling.py tiling.py tilegrid.py
    compiling.py colour.py memory.py detector.py      the bricks: models and their files,
                         placement, the int8 linear, sampling, the VAE passes, colour, the
                         memory model, the shot detector
    gguf_file.py kquant.py      a dynamic GGUF file: its header read without torch, and the
                         K-quants' dequantisation on the card with the linear that uses it;
                         pull.py also holds the smaller files of a DiT file, which a plan names
                         (the facts: each file, its size, what its plan swaps; an app words them)
    shots.py shot_plan.py shot_pass.py     a video's shots: a shot and its DiT windows, the
                         job's plan for its shots (the memory planner, no torch at import), and a
                         shot's pass, encode, windows and decode, on models already loaded
    picture.py png.py webp.py still.py still_plan.py     a still picture: read, written as PNG
                         or lossless WebP, upscaled on models already loaded, its plan for a bound
    clip.py              a clip's frames in memory, a call a shot (its seed by the shot's first
                         frame, the clip's one output array written into), upscaled on models
                         already loaded, its plan for the clip's shots (shot_plan's, the output
                         and the windows and scratch kept in system memory counted)
    shotruns.py          the runs of a shot (the levels of `--quality` and the nodes' quality,
                         each run's first frame, the run each frame is taken from, the frames
                         kept between runs), plain Python; the CLI's job.py imports it, clip.py
                         runs a clip by it
    shotcuts.py          the cuts drawn from the shot detector's probabilities (Cut, detections,
                         detected, possible, the default and the possible cuts' thresholds), plain
                         Python and NumPy; the CLI's cuts.py keeps their timecodes, the cut list
                         file and the playlists
    record.py            seedvr2x's record in each output: its fields and the AI mark, a picture's
                         XMP packet, a video's stream tags
    probe.py             the environment probe: the kernels a DiT file needs checked on the device
                         against the CPU, cached on disk by a fingerprint of the environment
    frames.py passes.py peaks.py files.py clock.py errors.py     what the bricks share (errors.py:
                         a refusal's message worded by each app, and the plan's refusals by kind)
    waiting.py           a wait on the GPU, polled and timed: said once past 4 times its unit's
                         estimate, the unit and its estimate named by each app, and an app's
                         check called at each poll, which may end the wait
    vendor/              model layer, vendored from numz (below)
    transnetv2/          the shot detector's model, vendored from TransNetV2 (below)
  tests/                 the library's, with library_fixtures.py, stand_ins.py and conftest.py
  tools/vendor.py        copy, diff and check of the vendored code
  tools/compare.py       two runs' outputs compared bit for bit
  tools/plan_matrix.py   the memory planner's plans over a fixed matrix, two runs' compared
apps/seedvr2x/          the seedvr2x command: distribution seedvr2x (setuptools), flat: its
                         modules at the app's root, no package folder
  main.py                options, logging (the script `seedvr2x = "main:command"`, which owns
                         its process: `main(owns_process=True)`)
  shot run cuts planner job manifest resume units scratch environment calibration timing
  stop disk .py          a job: its shots through the library's pass, unit by unit (shot.py: the
                         input copy a decode is corrected against), stitching, the plan's lines
                         and a resume's checks (planner.py), saving and resume, its environment
                         record
  stills.py              a still picture's run: its options, -o, plan, load, pass, PNG or WebP
  media/                 I/O layer: ffprobe, ffmpeg pipes, FFV1 and PNG writers
  tests/                 the CLI's, with levels.py and conftest.py
  tools/plan_lines.py    the plan matrix with the CLI's lines, on core's tools/plan_matrix.py
  pyproject.toml uv.lock .python-version LICENSE NOTICE
  README.md MANUAL.md images/
apps/imgserver/          the image server: distribution seedvr2x-imgserver (setuptools), flat
  main.py                options, environment, the server's life (`seedvr2x-imgserver = "main:command"`)
  service.py reservation.py admission.py     the start (files, plan, reservation, warm-up), a
                         picture's run under a lock per model, the queue
  web.py answers.py metrics.py     HTTP (ThreadingHTTPServer), a request's headers and the
                         server's words, /metrics
  tests/                 the server's, with serving.py and conftest.py
  pyproject.toml uv.lock .python-version LICENSE NOTICE README.md
apps/comfyui/            seedvr2x's ComfyUI nodes: not a uv project, they run in ComfyUI's own
                         environment; a package ComfyUI imports from its custom_nodes (relative
                         imports, any folder name)
  __init__.py nodes.py   V3 registration (`comfy_entrypoint`, a ComfyExtension; `io.ComfyNode`,
                         `io.Schema`); the nodes (Models, Cuts, Upscale Image, Upscale Video, and
                         the value nodes Tile, Blocks Swapped and Window Length: Upscale Video
                         takes the Cuts node's cut list, a required input of its own type, and
                         a quality widget: the model's runs over each shot; the settings are the
                         upscale nodes' widgets, the plan's numbers their optional sockets of
                         seedvr2x's own types, no Tuning node) and ComfyUI's
                         calls (the bar and text it gets from progress.py, Cancel, the plan said
                         on the node, plan only); the one
                         module that imports `comfy_api` (unstable)
  models.py plans.py     the files, the models kept in pinned memory and the one registration
                         with ComfyUI's model manager, a run's room and placement, keep in RAM
                         auto/on/off against ComfyUI's cap; the inputs' values as pins and
                         the plan in the nodes' words
  progress.py            what the nodes show while they run: the upscale's bar as its expected
                         time, the text on the node (the plan, where the run is, the time), the
                         previews, the download and the reading of the files on one bar, the Cuts
                         node's scan; on the library's callbacks, which only observe; a thread
                         for the text alone (the time each second between events); the units
                         named for the library's waits on the GPU, ComfyUI's Cancel their check
  words.py               the nodes' words for each refusal, with the way out in ComfyUI's terms
  workflows/ README.md   the two example workflows (picture.json, clip.json: ComfyUI's UI format,
                         Models wired in), and the nodes' README, for users
  ruff.toml pyrightconfig.json pytest.ini     the checks' settings, the CLI's
  tests/                 the nodes', with standins.py, conftest.py and comfy_stand_in/ (ComfyUI's
                         few modules, comfy, comfy_api.latest and server, written for the tests)
research/ models/ upstream/
```

The layers are DESIGN.md's (Architecture), the layout its Repository and apps (step 2): the
library is bricks that import nothing of the CLI, the CLI assembles them
(`core/tests/test_library.py` holds the rule). Two uv projects: `core/` declares the library's
dependencies as ranges; `apps/seedvr2x/` is the command's, with the exact pins, the lock and the
dev tools, and installs `core/` by path, editable. Its environment is the one shipped, and the one
the library is checked and tested in. The build backends differ: `core/` is built by uv_build, a
folder package; the CLI by setuptools, which lists each top-level module (`py-modules`,
`packages`), and `apps/seedvr2x/tests/test_names.py` holds the list to the files and to
`manifest.MODULES`, and each name to its own file in the environment, so that no dependency's
`run` or `media` shadows the CLI's. The scripts in `research/` stay standalone.

- The image server, `apps/imgserver/`, is a third uv project: its lock was started from the
  CLI's and holds the same versions of everything the two share (`apps/imgserver/tests/
  test_lock.py`; a moved one comes back by copying the CLI's lock and `uv lock`), `core/` by
  path, editable, and its own venv, `apps/imgserver/.venv`, where its checks and tests run:
  `uv sync`, then `uv run seedvr2x-imgserver --help`. Its loggers are `imgserver.<module>`.
- The CLI, from `apps/seedvr2x/`: `uv sync`, then `uv run seedvr2x --help`. There is no
  `python -m seedvr2x`: the tests start the venv's `seedvr2x` script.
- The checks and the tests, in the CLI's environment: from `apps/seedvr2x/` as written in
  [Checks](#checks), from `core/` with `uv run --project ../apps/seedvr2x` before each command.
- No environment of `core/`'s own: its pyproject says `managed = false`, so that a plain
  `uv run` there makes no lock or environment from the ranges, which wouldn't be the one shipped.
- Paths below are relative to the package they name: a library module is `seedvr2x/X.py` (in
  `core/`), a CLI module `X.py` or `media/X.py` (in `apps/seedvr2x/`), `tools/` is `core/`'s, and
  `tests/` is the directory of the package the test is for ([Tests](#tests)). README.md and
  MANUAL.md are `apps/seedvr2x/`'s.
- The loggers: the library's are `seedvr2x.<module>`, the CLI's `cli.<module>` (`cli.main`,
  `cli.media.reader`), the nodes' `seedvr2x_comfyui.<module>` (ComfyUI's console shows INFO and
  above, as bare messages).
- The ComfyUI nodes, `apps/comfyui/`, are checked and tested in the CLI's environment, from
  `apps/comfyui/` with `uv run --project ../seedvr2x` before each command ([Checks](#checks)):
  they have no lock of their own. ComfyUI imports the folder as a package, its folder not on
  `sys.path`, so their modules import each other by relative imports, and the tests import the
  package as `comfyui` (pytest.ini puts `apps/` on the path, with `core/tests` for the library's
  helpers and `tests/comfy_stand_in` for the stand-in `comfy` and `server`).

## The vendored model layer

`core/seedvr2x/vendor/` holds the model code: ByteDance's SeedVR (`e4de8c2`) as modified by numz
(`4490bd1`), under Apache-2.0. [provenance.md](research/docs/provenance.md) says what each file
is and what numz changed in it.

- **Path for path.** numz's `src/X` is `vendor/X`, and its configs and text embeddings
  (`configs_7b/`, `configs_3b/`, `pos_emb.pt`, `neg_emb.pt`) sit at the root of `vendor/`. numz's
  model code imports its own modules with relative imports only, so the copy imports unchanged,
  and its diff against numz only shows real changes.
- **Never reformatted**, and linted for errors only (see [Checks](#checks)): a reformat would bury
  seedvr2x's changes in the diff.
- **Modified files are marked**, as Apache-2.0 §4(b) requires: one line right under the original
  header, `# Modified for seedvr2x: <summary>.`, and a `# seedvr2x: <why>` comment on each change.
- **The lean DiT attention** (DESIGN.md, Allocator, Small cards): `models/dit_7b/attention.py`,
  `mlp.py`, `mm.py` and `nablocks/mmsr_block.py` are modified so that a window's forward holds
  fewer buffers at once (the norm and RoPE in chunks, the varlen attention writing into one
  output, the MLP's activation in place); the arithmetic is the plain code's. Each carries the
  marker. `core/tests/test_lean_attention.py` holds them bit for bit to the plain code, which it
  reads from the repository's git objects at main's commit before the change (`2ba0857`, through
  `tools/vendor.py`'s `git_show`) and loads beside the vendored modules: not numz's own
  `mmsr_block.py`, which imports the sequence parallelism the vendored layer removed and does not
  run here. The test skips where that commit is not in the repository; the reference stays at
  that commit whatever the vendored files become, and a later change to these files is held to
  it or moves the reference with a measured reason. The sizes are stand-ins chosen so that the
  windows are several and of unequal lengths and the chunks end partial; the pipeline's chunked
  path, the branch without a compute dtype and both under autocast are held.
  `models/video_vae_v3/modules/attn_video_vae.py` is modified the same way for the decoder's
  tail a frame at a time (below).
- **Nothing in it imports numz's runtime** (`src/optimization`, `src/utils`, the rest of
  `src/core`). DESIGN.md (Vendored model code) lists what goes.
- The StableSR-derived colour code is not vendored: its licence is non-commercial (DESIGN.md,
  Colour correction). `seedvr2x/colour.py` writes its wavelet split from the method, clean room:
  whoever changes it doesn't read StableSR's functions in numz's `src/utils/color_fix.py`, lines
  25-247 at 4490bd1 (`adain_color_fix`, `wavelet_color_fix`, `calc_mean_std`,
  `adaptive_instance_normalization`, `wavelet_blur`, `wavelet_decomposition`,
  `wavelet_reconstruction`), nor ComfyUI's `comfy/ldm/seedvr/color_fix.py`, which holds functions
  of the same names (`wavelet_blur` among them), under GPL-3.0 besides. numz's own `lab` code
  there (lines 249-521, Apache-2.0) may be read; no numz code is left in the correction since
  `split` replaced `lab`.
- `tools/vendor.py diff` compares the copy with numz and with ByteDance, both read from the
  submodules' git objects at the pinned commits, never from their work trees: a moved submodule
  can't shift the reference. `tools/vendor.py check`, also run by the tests, verifies that a file
  carries the marker exactly when it differs from numz, that the original headers are kept, that
  nothing imports numz's runtime, and that seedvr2x's edits add no syntax error or undefined
  name.

## The vendored TransNetV2

`core/seedvr2x/transnetv2/` holds the shot detector's model (DESIGN.md, Shot detection), a
brick of the library:
TransNetV2's official PyTorch code (github.com/soCzech/TransNetV2 at `85cef72`, MIT),
`inference-pytorch/transnetv2_pytorch.py` as `transnetv2_pytorch.py`, with TransNetV2's `LICENSE`
beside it; the `__init__.py` is seedvr2x's.

- **A directory of its own,** not `vendor/`: `vendor/` is numz's tree path for path, which
  `tools/vendor.py` diffs against numz, and TransNetV2 is another upstream under another licence.
  The file's own directory is left out, `inference-pytorch` being no name Python imports.
- **Byte for byte,** never reformatted, excluded from ruff and pyright as `vendor/` is:
  `tests/test_detector.py` holds the file and the licence to their SHA-256 at that commit, those
  `models/transnetv2_weights.py` pins (`SOURCES`), so any change is deliberate. It runs with
  the locked torch unchanged; a change would be marked as `vendor/`'s are, its pin changed with it.
- `seedvr2x/detector.py` builds it, loads seedvr2x's `transnetv2.safetensors` into it (pinned in
  `seedvr2x/pull.py`, made by `models/transnetv2_weights.py`) and runs measurement's windows on
  frames pushed to it; the CLI's `media/scan.py` feeds it the frames and the library's
  `shotcuts.py` draws the cuts from its probabilities. The ComfyUI nodes' frames, already
  decoded, `detector.shrink` takes to its 48x27 in torch (antialiased bicubic, no ffmpeg) and
  `detector.probabilities` runs through a new detector, whole. Its licence is the library's:
  `core/`'s pyproject says "Apache-2.0 AND MIT" and ships `transnetv2/LICENSE`; `core/`'s NOTICE
  and `apps/seedvr2x/`'s README.md credit it.

## Environment

- The CLI's uv project, on Python 3.13: `uv sync` in `apps/seedvr2x/` creates
  `apps/seedvr2x/.venv` with the locked versions, `core/` installed editable, and the dev tools.
  The scripts of `research/` that import seedvr2x run with its `bin/python`.
- Its dependencies are pinned to the versions seedvr2x's own references were made with (milestone
  1's regression, below): torch and torchvision (CUDA 13.0 builds, from PyTorch's index),
  diffusers, rotary-embedding-torch and the rest. A bump can move the output's bits: it remakes
  the references, with the measured reason said in its commit (DESIGN.md).
- Linux x86_64 only for now (DESIGN.md: Linux first).
- `gguf` (llama.cpp's own gguf-py, 0.19.0) is a dev dependency of `apps/seedvr2x` alone, pinned
  there: the tests' reference for the GGUF reader and the dequantisation, never the library's or
  the shipped environment's; the tests that use it skip without it.
- huggingface_hub is declared, at the version the lock held already through diffusers: it pulls
  seedvr2x's own model files from its repo at the pinned revision, into Hugging Face's cache
  (`seedvr2x/pull.py`). seedvr2x imports it to look in the cache on every run without
  `--model-dir`, which reads the files from a directory instead; diffusers, which the vendored
  model imports, brings it into every run anyway. Its telemetry is off: `main.main` sets
  `HF_HUB_DISABLE_TELEMETRY` before either import, unless the user has. hf_xet, its Xet
  downloader, seedvr2x never imports: its import writes a log file under `HF_HOME`.
- TorchCodec is declared (PyPI's wheel) for one use: the decode of a 16-bit colour PNG, which
  Pillow cuts to 8 bits and whose torchvision decoder is deprecated since 0.29
  (`seedvr2x/picture.py`). It is imported there alone, inside the function, never at import:
  its import loads the system's FFmpeg shared libraries where there are any, which no other
  picture and no video run should do; its image decoders bundle libpng and need no FFmpeg. It
  is never in a job's environment record (a video run never imports it), and it turns a picture
  by its eXIf chunk with no switch to leave it, so the chunk is taken out of the bytes it decodes.
- ffmpeg and ffprobe on PATH, 7.1 or later (for setparams' chroma_location), with zscale
  (libzimg), idet and ffv1: seedvr2x checks the build at startup and refuses to run without
  them, or without any other filter, encoder, decoder or muxer it uses (`media/ffmpeg.py`;
  DESIGN.md, Input).
- **FlashAttention 2 is not a dependency.** PyPI only has its source, a long CUDA build, and
  seedvr2x runs without it: it uses FA2 when installed, else PyTorch's SDPA, and logs which one
  runs; the two peak at the same memory since the lean attention (below), so a plan is one for
  both. To use it, build a wheel for your GPU ([setup_env.sh](research/scripts/setup_env.sh),
  [environment.md](research/docs/environment.md)) and install it into `apps/seedvr2x/.venv` with `uv
  pip install --no-deps <wheel>` (from `apps/seedvr2x/`). `uv run` keeps it, but `uv sync` removes
  what the lock doesn't list: install it again after a sync, or sync with `--inexact`.

## Checks

From `core/`, in the CLI's environment:

```
uv run --project ../apps/seedvr2x ruff format --check
uv run --project ../apps/seedvr2x ruff check
uv run --project ../apps/seedvr2x pyright
uv run --project ../apps/seedvr2x tools/vendor.py check
uv run --project ../apps/seedvr2x pytest
```

From `apps/seedvr2x/`:

```
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

From `apps/imgserver/`, in its own environment, the same four as the CLI's. From
`apps/comfyui/`, in the CLI's environment, the same four with `uv run --project ../seedvr2x`
before each (ruff.toml and pyrightconfig.json there are the CLI's settings).

- ruff: line length 100, rules E, F, W, I, B, UP and RUF, set in each pyproject. The vendored
  code, `seedvr2x/vendor/` and `seedvr2x/transnetv2/`, is excluded from formatting and from that
  lint (`core/`'s pyproject). `tools/vendor.py check`
  lints `vendor/` for errors only (syntax errors, undefined names), and reports those that numz's
  own files don't have: numz has some already, false positives and a bug in a class nothing
  builds. `transnetv2/` is held to its pinned bytes (`core/tests/test_detector.py`). The CLI's ruff
  counts `seedvr2x` as first-party, so that its imports sort with its own modules.
- pyright: strict on `seedvr2x` in `core/` and on the CLI's modules (`*.py`, `media/`) in
  `apps/seedvr2x/`, standard on the tests and tools. The vendored code is excluded: the tests show
  that it imports and runs. The CLI's pyright reads the library as source and the library's test
  helpers (`extraPaths`).
- A change that may move a plan (`seedvr2x/shot_plan.py` or any module it imports, `peaks.py`,
  `memory.py`, `passes.py`, `weights.py` and the vendored window code among them, or the CLI's
  `planner.py`): the plan matrix for the four DiT formats on the tree before it and on the tree
  after it, then `diff`, which lists the cases it moves: none where none was meant, and the list
  shown where some were. Core's `tools/plan_matrix.py run`, from `core/`, gives the library's
  plans; the CLI's `tools/plan_lines.py run`, from `apps/seedvr2x/`, the same with the CLI's lines
  (the tools' docstrings).

## Tests

- `pytest` (as in [Checks](#checks)) runs on the CPU and needs no model. The four projects run at
  once, each in its own process, basetemp and log, with a quarter of the machine's threads
  (`OMP_NUM_THREADS`), and started with SIGINT at its default (a shell's background job has it
  ignored, which `tests/test_stop.py`'s `test_terminable` reads): 28 minutes on 2026-11-07 on a
  desktop of 32 threads, nothing else running, the CLI's minutes the wall: the library 5.5, the CLI
  28.4, the image server 2.4, the nodes 0.8 (62 minutes the same day before, the projects one after
  another, the stand-in VAE at its shipped widths, ffmpeg's listings asked by every run). While
  building, run only the test files of the code just written, with ruff and pyright of the project
  touched; the whole suite once at the end. A failure there is fixed in its patch and only its files
  run again, never a second suite; a patch re-cut on a moved main runs the files its conflicts
  touched. Tests that need ffmpeg skip without a build seedvr2x accepts (see
  [Environment](#environment)).
- The stand-in VAE of the CPU tests, `cpu_models` (`core/tests/stand_ins.py`), is the shipped one
  narrowed to 32, 32, 64 and 64 channels a block (`NARROW`: two widths, so that its resnets'
  shortcuts are built, the decoder's tail's among them; the library reads no channel count of
  the VAE), about nine times faster a pass; `shipped_cpu_models`, at the shipped widths, runs
  one case of `tests/test_clip_cli.py` and one of `tests/test_still_cli.py`, and
  `tests/test_tiling.py`'s `real` its own. In the CLI's tests, ffmpeg's listings
  (`media/ffmpeg.py`'s check) and the conversions' fingerprint are asked once per build
  (`tests/conftest.py`: the program PATH finds, by its path, inode, size, times and, under a
  mebibyte, its bytes, its loader's environment, the options or the conversions): a fake build
  is asked again only when its own bytes change, not files it prints from. They are asked every
  time in `tests/test_ffmpeg_check.py`, `test_fingerprint.py` and `test_ends_with_us.py`
  (`MEMO_OFF`), whose subject they are, and in any process of its own.
- `pytest -m gpu` runs the tests that need CUDA, never selected by default; run it in each
  directory.
- **Where a test file sits.** `tests/test_x.py` is in the directory of the package it tests:
  `core/tests/` for the library's, `apps/seedvr2x/tests/` for the CLI's. File names are unique
  across the two, a helper being imported by name. A file that held both halves is two: the
  library's, and its `_cli` half beside the command (`test_pull_cli.py`, `test_weights_cli.py`,
  `test_compile_cli.py`, `test_detector_cli.py`, `test_padding_cli.py`, `test_int8_cli.py`,
  `test_gguf_cli.py`, `test_shot_cli.py`), which use the library's half's helpers and say so in
  their docstring. The shot's pass and its plan moved to the library on 2026-11-04 (DESIGN.md,
  Repository and apps, "The shot's pass moves to the library"), their tests with them:
  `core/tests/test_shot_plan.py` holds the plan (`seedvr2x/shot_plan.py`, importing no torch) and
  `core/tests/test_shot_pass.py` the pass on the CPU; the CLI's `tests/test_planner.py` what the
  command line does with a plan (its lines, the smaller DiT files, a resume's record checked or
  planned again) and `tests/test_shot_cli.py` the pass with the units' files, the scratch and the
  input copy, and the GPU test of a shot's windows through the real DiT. The helpers both
  directories use are `core/tests/`'s (`library_fixtures.py`, `stand_ins.py`, the fixtures
  `conftest.py` imports from the first); the CLI's pytest puts `../../core/tests` on its
  `pythonpath`.
- The rule between them: `core/tests/test_library.py` holds that no library file imports the
  CLI (a scan and a fresh process); `apps/seedvr2x/tests/test_names.py` holds the CLI's top-level
  names to its files, the standard library's and what else the environment provides (see
  [Layout](#layout)).
- Edit nothing under `core/seedvr2x/` or `apps/seedvr2x/` while a test run is going: a job's
  settings record a hash of both packages' files (`manifest.code_sha256`, the library's
  `seedvr2x/<path>` and the CLI's `cli/<path>`), so a file changed between a test's first run and
  its resume makes the resume another job, and the failure looks like a bug in resume.
- Tests that need the model weights read their directory from `SEEDVR2X_MODEL_DIR`, and skip without
  it. It holds numz's files and seedvr2x's own side by side (links will do): the GPU tests run
  numz's 7B fp16 and VAE, the weights their references were made with, and the regression seedvr2x's
  own as well, and the shot detector's, `transnetv2.safetensors`. With it, the CPU suite also checks
  every known model file there by its header (`tests/test_weights.py`; `test_weights_cli.py` has the
  command line refuse a model, saying what the file is), and runs the shot detector on TransNetV2's
  weights (`tests/test_detector.py`: the windows, and measurement's own run reproduced bit for bit;
  `tests/test_detector_images.py`: the same detector on frames held in memory, `shrink` and
  `probabilities` on a clip equal bit for bit to the frames pushed to a `Detector`;
  `test_detector_cli.py`: the cuts of a synthetic clip; `tests/test_detection.py`: the first pass
  recording the module's probabilities bit for bit); without, the detector's windows run on the
  untrained model's weights, written by the test.
- `tests/test_int8.py` holds seedvr2x's int8 linear (`seedvr2x/int8.py`, W8A8 with
  comfy-kitchen's rotation) on the CPU to a reference written there apart: the rotation in
  float64, the quantization in NumPy with bfloat16's roundings by their bits, the integer
  product in int64, the rescaling in float64; codes, scales and the product equal, the output
  within bfloat16's rounding, the rows taken at a time changing no bit, its module under
  autocast, the 7B's 288 block matrices swapped on the meta device, and its buffers measured by
  torch's memory profiler within `peaks.W8A8_SMALL` of the memory model's count. The CPU path is
  the tests' alone: the pipeline runs int8 on CUDA. `pytest -m gpu`, skipped without CUDA: the
  path on CUDA against the CPU's for the 7B's four shapes, 1 to 4,101 rows (cuBLASLt's int32
  product exact, the codes 99.999% equal at least, the outputs the CPU's but where codes or
  scales differ, a call's memory within its output and `peaks.INT8_BUFFERS`); opt-in, the same
  activations through comfy-kitchen 0.2.37's CUDA `int8_linear`, skipped unless it is installed
  (`uv pip install --no-deps comfy-kitchen==0.2.37`; never a dependency); and
  `tests/test_blockswap.py`'s gate on a real int8 file in `SEEDVR2X_MODEL_DIR`, its blocks
  swapped giving the bits of the DiT loaded whole, skipped without one. `tests/test_weights.py`
  holds the int8 files' headers and markers; `test_int8_cli.py` the command line over a
  synthetic int8 file: its check, the plan by int8's figures and its line, the job's record, a
  resume judged early by int8's figures.
- `tests/test_gguf_file.py` holds `seedvr2x/gguf_file.py`, the GGUF header read without torch or
  the tensors' data, to files written there by hand with struct from the format's specification
  (every size and type, every refusal, every prefix cut short) and to `gguf.GGUFReader` on the
  same files; where `SEEDVR2X_MODEL_DIR` has seedvr2x's two dynamic files, to their real headers.
  `tests/test_kquant.py` holds `seedvr2x/kquant.py` on the CPU: each type's dequantisation (Q3_K,
  Q4_K, Q5_K) equal in float32, bit for bit, to the `gguf` package's, on synthetic blocks of every
  type (random bytes, the fp16 scales of several magnitudes, every 6-bit scale value in every
  position) and, with the files, on real matrices; the dtype rounding per chunk equal to the
  float32 matrix rounded whole, the blocks taken at a time changing no bit, the module under
  autocast, the 7B's 288 block matrices swapped on the meta device. `pytest -m gpu`: the
  dequantisation on CUDA against the CPU's, bit for bit, and a linear's memory printed for the
  planner. `tests/test_weights.py` holds the dynamic files' pins, headers (the types per matrix
  by the test's own reading), recognition by content and every refusal (other types, other
  tensors, versions, a file cut short), the header read without torch; `tests/test_blockswap.py`
  the blocks of unequal sizes swapped as any others, the file read into its parts and hashed once
  (the one read, below), a file cut short refused in safetensors' words, and, `-m gpu`, a real
  GGUF file's blocks swapped giving the bits of the DiT loaded whole, and `HostCopy`'s two
  platforms by their calls with the platform faked (Windows: the anonymous `mmap.mmap(-1,
  length)` with no POSIX flag; Linux: `MAP_PRIVATE | MAP_ANONYMOUS` and `MADV_DONTFORK`);
  `tests/test_peaks.py` and
  `tests/test_still_plan.py` the GGUF DiT's figures by its own blocks (`peaks.leaving_blocks`)
  and the smaller files' line in a picture's plan; `tests/test_pull.py` the pins and
  `pull.smaller_files` (the facts: files, sizes, `pull.Swaps`); `tests/test_planner.py` and
  `test_gguf_cli.py` (the CLI's) the plan by
  the GGUF's figures and its line, the job's record, a resume judged early, the smaller files
  named in a refusal for memory (with `--model-dir` said once) and in the plan where they swap
  fewer blocks or run a longer window, and the file never
  changed by itself, for a video and for a picture; their sentences are the CLI's
  (`planner.smaller_way`, `planner.smaller_line`, held by `tests/test_planner.py`); the image
  server's `tests/test_reservation.py` and `test_service.py` the same way out in a refused
  start, in the server's own words ("a smaller DiT file that plans this bound";
  `reservation.smaller_way`), a start that then takes the file, and the start's line of the
  smaller files (`reservation.smaller_line`, "smaller DiT files plan this bound better", blocks
  alone) at INFO after its blocks-swapped line, and none where the plan swaps none or runs the
  dynamic GGUF already.
- A refusal's message worded by each app (`seedvr2x/errors.py`): a library refusal names the way
  out as a `Setting` (`SETTINGS`: the library's name for the window, the swapped blocks, the
  tiles, the resolution, the DiT file and compiling), never as an option; `str()` of it is the
  library's own words (`plain`, no `--`), `Refusal.words(say)` the message worded by an app's
  `say`, the CLI's `planner.option` (`--` and the name with hyphens, then the value), which
  `planner.worded` applies to any error before it reaches the user. The plan's refusals are
  `PlanRefused` and its kinds, each carrying its facts apart (`PinRefused`, `MemoryShort`,
  `HostShort`, `InputTooLarge`, `CompileRefused`; `shot_plan.PlanError` and
  `still_plan.PlanRefusal` are the base's names). `core/tests/test_errors.py` holds the
  mechanism, `test_shot_plan.py` and `test_still_plan.py` each kind raised by its sites with its
  facts and its plain words naming no option; a library test that holds a refusal's text with
  its options renders it with `library_fixtures.cli_words`, the CLI's own with `planner.worded`.
  Library logs of a refusal (the plan with the usual reserve refused, said at debug) are in the
  plain words.
- The model files', the picture's and the host's refusals are of a kind too (`seedvr2x/errors.py`):
  `ModelError` and its kinds (`ModelMissing`, `ModelUnknown`, `ModelDamaged`, `ModelChanged`,
  `ModelNotPinned`, `ModelUnreachable`, `ModelNoRoom`, `ModelUnreadable`, `ModelsRefused` for
  several at once), carrying the role, the file's name (not its path, which stays in the
  message), the pinned and the found figure, the settings named as the way out (`vae_model`,
  `model_dir`, `cuts` among `SETTINGS`) and the command that fetches the file again (`hf download`
  and Hugging Face's variables stay text, as the library's tools); `HostMemoryError` (need and
  available, in bytes), `PictureRefused` (kind, reason and way as `picture.KINDS` has them, with
  the pixels and frames counted) and `TargetTooSmall` (`frames.check_target`: the resolution
  asked, the frames' size, the least the padding takes). `weights.ModelError` and
  `blockswap.HostMemoryError` stay importable. `pull.advice` gives a message of `Say` for each
  role, `Hashes.vouch`'s and `expect`'s refusals are functions returning a `ModelError`, and the
  server words its start's refusals with `reservation.worded`, the CLI every error with
  `planner.worded`. `core/tests/test_errors.py` holds the kinds and their plain words (no `--`
  but in a path or the advised command); `test_weights.py`, `test_pull.py`, `test_blockswap.py`
  and `test_picture.py` each kind raised by its sites, their texts with options rendered by
  `cli_words`.
- Models kept in their pinned copies and placed per run (`passes.HELD`, `Models.place` and
  `Models.release` in `seedvr2x/model.py`; DESIGN.md, Repository and apps, the nodes):
  `tests/test_held.py` on `test_blockswap.py`'s toys and simulated streams, loaded as
  `load_models` loads them for `HELD`: models placed for each of five plans, in turn forwards and
  backwards, with a release between or place alone, giving through `Models.enter` the bits of
  models loaded directly for that plan and the bytes the placement keeps; nothing read again
  (reads counted) and the pinned copies the same memory; the VAE staying in the DiT's phase where
  the plan says so, and a job's models keeping the VAE away as before; release leaving nothing on
  the device, idempotent, refusing an `enter` before the next `place`, safe after a forward that
  raised and after a release that failed half-way; a `place` that can't be made changing
  nothing; the VAE's causal memory and the vendored RoPE's `lru_cache` dropped by release and by
  `model.forget`, for any `Models` (a dropped job's models leave no RoPE module alive), and
  the same tables cleared at each shot's start (below); only
  `HELD` itself, never a plan equal to it, making models kept; the
  modules' buffers and the text embeddings copied to the CPU by release and back by place, equal
  to the bit, with a stand-in card and a walk of the models (`strays`) finding no tensor left on
  a device or in a cache. The
  loader with `HELD` on the real weights is a GPU test, not yet written. The planners' `held`
  (`shot_plan.plan`, `still_plan.plan`): `tests/test_shot_plan.py` and `tests/test_still_plan.py`
  hold the plan the card alone gives where the host is short, no pinned copy counted, a job the
  host refused planned, `held=False` changing no plan (the plan matrix's 5,120 cases too), and
  `_vae_kept` and `_dit_kept` returning nothing for weights held.
- The vendored RoPE's tables where the DiT's work on a shot starts and where it ends
  (`model.forget_rope_tables`; DESIGN.md, Allocator, "The vendored RoPE's tables"):
  `NaRotaryEmbedding3d.get_axial_freqs` (rope.py:44) memoises the tables of each window shape for
  the class, so a long run of many shot lengths grew them without end, and they stayed into the
  VAE's phases (a 16 GB card's clip decode 1.1 MiB over its prediction, 2026-11-06).
  `shot_pass.sample_windows` clears them before a shot's first window and again in a `finally`
  once its generator is exhausted, closed or left by a raise (`clip.py` closes it, `run.py` drops
  it, the calibration's is dropped after its one window), and `still.sampled` before its window
  and after it, a raise included; `still.upscale` and `clip.upscale` go through those. No VAE
  pass holds a table, so no VAE peak counts one (`peaks.DIT_BUFFERS`, the census's rest).
  `core/tests/test_rope_tables.py` runs the real rotary module, asked for each window's tables
  by a stand-in DiT, through those entries: a long run of many shot lengths holding at most one
  shot's tables while its windows run and none after, the same run growing past one without the
  clearing (the test proving the clearing is what holds it), a picture run step by step holding
  one table while it samples and none after, the tables gone however a shot's windows end
  (exhausted, closed early, dropped, a window raising; a picture's window raising), every entry
  clearing once before a shot's first window and once after its last, and no output bit moved by
  it; its autouse fixture empties the class's cache before and after each test, so that no test
  sees another's tables. `test_held.py` has one line of isolation for the same reason.
- A clip of frames in memory (`seedvr2x/clip.py`; DESIGN.md, Repository and apps, the nodes), on
  the CPU: `core/tests/test_clip.py` holds it on `cpu_models`' VAE in bfloat16 and `test_still.py`'s
  stand-in DiT: the output's shape and dtype and the input left as it was, the clip's seeds (a
  shot at frame 0, and a shot of a longer clip seeded by its first frame, `start`, as the command
  line seeds it, written into a slice of the clip's one output array, `output`), the phases
  entered in order and nothing loaded, 1, 2 and 3 windows, the steps told in order with their
  counts, a step raising an `Exception` or a `BaseException` at each stage leaving no decode
  open, none of the decode's slices alive (the traceback's frames cleared; `Watched`) and nothing
  for `test_held.strays` to find, the refusals (not `(T, H, W, 3)` float32, no frame, `NonFinite`
  from an encode, a window or a decode, a window's naming the shot by its frames), the correction
  against the clip's own frames, the scratch in `tiling.InMemory` and no file. The plan
  (`clip.plan`, `shot_plan.plan`'s `in_memory`; the clip's shots planned as the planner takes them,
  the output of all of them taken off the host): the output array taken off the host and a host
  short of it refused saying so, a tiled pass's allowance the whole scratch, the windows counted
  all together at the DiT's last window and the latent beside the merge (`host_moments`'
  `in_memory`), and `in_memory=False`
  changing no plan (the plan matrix's cases too). `apps/seedvr2x/tests/test_clip_cli.py` holds
  `clip.upscale` bit for bit to the shot's pass as the command line runs it (`encode_shot`,
  `sample_windows`, `merge_windows`, `decode_shot` with `shot.py`'s `Correction` reading the same
  frames), in 1 to 3 windows, each tiled pass alone and both, corrected and not, and under
  `-m gpu` on CUDA with the sharp 7B in two windows. The runs of a shot (`seedvr2x/shotruns.py`;
  `quality` of `clip.upscale`, the nodes' `quality`): `core/tests/test_shotruns.py` holds the levels
  (standard 1 run, enhanced 2 from frames 0 and 2, maximum 4 from 0 to 3), the run each frame is
  taken from against the closed forms written out apart, a short shot having fewer runs, the
  frames kept between runs and their bytes, and the module importing no torch (the rule held to
  measurement's composer stays the CLI's `tests/test_runs.py`, through `job`'s re-exports, which
  `test_clip_cli.py` holds to be the library's own objects); `test_clip.py` a level's output
  equal to the standard clips of the runs' frames composed by `run_of`, in 1 to 3 windows,
  corrected and not, a clip too short for a run's first frame having no such run, the steps told
  run by run ("run" first, with several runs only: standard's are as before), the output allocated
  once and given back when given, the seed shared by the runs and the correction reading the
  same frames from each run's first, a Cancel between runs leaving nothing alive, an unknown level
  refused; `test_clip_cli.py` each level bit for bit to the shot's pass run run by run as
  `run.py` does (`runs_pipeline`: the shot from the run's first frame, `Correction`'s `first`,
  the shot's seed), 1 to 3 windows, tiled and not, corrected and not, and under `-m gpu` the
  sharp 7B at standard and at maximum. The plan is the same at
  every level (the runs one after the other, the first the longest): the plan matrix's both halves
  unmoved by the move of the choice to the library.
- `tests/test_pull.py` holds the pull of seedvr2x's own files (`seedvr2x/pull.py`) on the CPU,
  huggingface_hub's two calls replaced, and the pins to the revision itself;
  `test_pull_cli.py` holds what the command line does over it.
- One read per load (DESIGN.md, Weights, decided on 2026-11-03): `tests/test_blockswap.py` holds
  `blockswap.read` hashing every byte of the file once, in its order (the header, the tensors
  skipped, bytes after the data), its SHA-256 hashlib's of the whole file and its tensors those
  read without the hash; the hash thread a piece behind the read, the staging buffer read into
  again only once its last piece is hashed (a slow hash, pieces of 5 bytes; the test fails
  without that wait); a file cut short refused in the same words; a GGUF file's header parsed again from the bytes
  hashed, a header read differently the second time refused as changed while it loaded; and, where
  `SEEDVR2X_MODEL_DIR` has them, seedvr2x's VAE and TransNetV2's weights read to their pins and to
  safetensors' own tensors. `tests/test_pull.py` the pin's check with `load` reading nothing, its
  size first, its SHA-256 proved at the load's end or refused there in the pin's words; a SHA-256
  known before (a record's, a whole read's) held to the pin at once and proved by the load, each
  refused in its own words, but a pinned file's a record vouches for, in the pin's; with
  `whole`, a SHA-256 nothing vouches for read at once once its size has passed; a file unknown before taken with the load's; `load_models` refusing a
  file whose bytes aren't those expected at the end of its own load, before the next.
  `tests/test_detector.py` the detector's file opened once for its bytes (and once for its
  header's check), hashed, loaded from them and proved. The picture's run and the server's start
  read no model file whole (`test_still_cli.py`, `apps/imgserver/tests/test_service.py`), their
  stand-in loads proving the files as the library's does. The pins: `SHA256SUMS`, a few
  kilobytes, downloaded into a temporary `HF_HOME` behind a link, and the sizes read from the
  Hub's API, skipped offline (no network to huggingface.co, or `HF_HUB_OFFLINE` set). A test
  that runs seedvr2x in a process of its own gives it its own `HF_HOME` and `HF_HUB_OFFLINE=1`,
  but that one; hf_xet is imported in such a process only. Its runs of the CLI give an empty cut
  list, so that the shot detector, which it replaces, neither fetches nor checks its file, but
  where a test holds that pull. With `SEEDVR2X_TESTS_PULL=1`, it pulls the four v1 files into
  Hugging Face's cache too, about 33 GB the first time; `test_regression.py`'s
  `test_milestone1_from_the_cache` (GPU) then runs the default models from the cache,
  `HF_HUB_OFFLINE=1`, and skips until the three it takes are there, saying how to fetch them
  alone (`hf download`, 17 GB).
- `tests/test_regression.py` (GPU) holds every change to the runtime or the vendored code to
  milestone 1: the output must stay bit-identical to seedvr2x's own references, FFV1 master and
  float32 frames, made once on main's code in seedvr2x's defaults (its own padding, its tail
  repeating a shot's last frame, its noise drawn on the CPU) and checked equal to numz's in the
  same GPU session (2026-11-04, the last comparison with numz): milestone 1's input with the 7B
  fp16 of numz and of seedvr2x (one reference, `7b`) and with seedvr2x's sharp 7B (`sharp`), at
  1080p, at 720p and on a crop whose width isn't a multiple of 16 (the padding cases), and under
  three pairs of VAE tiles. The references are per card, in `seedvr2x/<card>/` of
  `SEEDVR2X_REFERENCE_DIR` with a `cases.json` giving each case's size on that card, the inputs
  in `m1/`; a card with no directory skips every case, naming it, and one with a directory fails
  a case whose reference is missing, unless the card can't run it. A change that moves the
  output's bits is allowed with a measured reason, and remakes the references, said in its
  commit (DESIGN.md). Its runs detect their shots, as a user's does without `--cuts`, the shot
  detector on the GPU before the models load, at `--cut-threshold 0.5`, where milestone 1's input
  is one shot (0.3 cuts it in three): each case skips, naming the file, without
  `transnetv2.safetensors` in `SEEDVR2X_MODEL_DIR`. The other GPU tests' runs give a cut list,
  empty for one shot. How to make the references is in the file's docstring;
  `tests/test_regression_cases.py` holds, on the CPU, what a card takes of each case.
- `test_milestone1_unpinned` (same file) pins no tile: its two calls, `--plan` then the run, are
  each compared with the plan the planner makes for the memory that call recorded itself
  (`compare_with_planner`), never with each other: a desktop that drives the card moves its free
  memory between them, and a plan that follows it is right.
  `test_plan_and_run_read_the_same_free_memory` holds the two readings to one figure, back to
  back, the run stopped once it has planned, and skips, saying the figures, where the other
  programs' memory on the card moved between them. Its
  name lacks `milestone1`: a gate that selects by `-k milestone1` doesn't run it, so the gate's
  hold names it. Both calls run with `-v`: the free memory is the `--plan` call's from its
  manifest and the run's from its log; the system memory available is each call's own, from its
  log, in bytes (`logged_host`: a plan bound by it moves at steps of a few MiB, which two decimals
  of GiB can't rebuild); the disk line is compared by its masters (size at most, assembled or
  not, rebuilt from the manifest by `recorded_disk`), without the free space and the verdict
  after it. The run's comparison comes after its tiling check, so that a run that tiles where
  `--plan` did not skips or fails as `unpinned()` says. `tests/test_regression_unpinned.py` holds
  the helpers on the CPU, on synthetic inputs: a reading found in a log or a manifest, the
  host's reading found in a log or said unread, the planner's plan rebuilt for them (a plan bound
  by the system memory from the logged bytes), the disk line compared by its masters alone and
  rebuilt from a manifest, errors found with and without `--verbose`, the comparison and its
  failure message (each call's reading, windows, blocks and disk masters), the run stopped.
- seedvr2x runs in no mode for the tests only: numz's padding, tail and noise draw, which the
  regression ran in until 2026-11-04 to reproduce numz's references, retired with them. A
  manifest written before, holding `numz_padding`, `numz_draw` or `numz_tail` among its settings is
  another job (`tests/test_cli_run.py`), refused naming the setting.
- `split`, the colour correction (`seedvr2x/colour.py`), is held to the colour study's own `split()`
  (`research/scripts/colour_variants.py`): on the CPU by `tests/test_colour.py` (`core/tests/`), on
  the GPU in the pipeline by `tests/test_split.py` (the CLI's), at x2 and x4, in one window and in
  three, within one 16-bit code. Both load the study's script with its baseline `colour.py` from the
  repository (the baseline from git), or from `SEEDVR2X_COLOUR_STUDY` (the directory holding
  `colour_variants.py`) and `COLOUR_BASELINE` (that `colour.py`) on a copy of `core/` alone, and
  skip without them; one of the two set without the other fails. `tests/test_split.py` also scores
  `split` against milestone 1's input with milestone 5's metrics, its thresholds its first run's
  figures (2026-10-09) plus milestone 5's tolerances.
- The frame index and the reads through it (`media/index.py`, `media/reader.py`) are held by
  `tests/test_index.py` to full decodes of files it makes, the kinds of
  `research/scripts/seek_test.py`'s `synth` and three of `seeking.md`'s failure mechanisms
  made on purpose; the first pass's reading of ffmpeg's lines (`media/scan.py`) to a fake ffmpeg
  on `PATH` writing chosen ones, and to a source whose titles and language tag are made of its own
  lines; and a jump in the timestamps to MPEG-TS files joined, with the remux its refusal gives,
  a constant rate under 0.1 fps having none. `tests/test_idet.py` holds idet's counts and the
  combing warning to clips made there, and their reading to such a source, its titles idet's own
  lines.
  `tests/test_seek_sources.py` holds the index and its reads to measurement's real sources, on the
  CPU: opt-in, it reads the list of sources named by `SEEDVR2X_SEEK_SOURCES`, each with
  `seek_test.py`'s outputs for it (its docstring says the list's format), and skips without it.
- The shot detector in the first pass (`media/scan.py`) is held by `tests/test_detection.py`, a
  recording stand-in in its place: its frames are the official extraction command's byte for
  byte, on 18 pixel formats whatever the official command's scaler threads (1, 4, or ffmpeg's
  own count), every frame of the index once, its scaler's graph on 4 threads of ffmpeg's and its
  encoder on none (counted by the names ffmpeg gives its threads, `/proc`), a slow detector
  holding ffmpeg back, one that stops moving stopped by the watchdog, a thread reading one of
  ffmpeg's pipes that raises stopping the pass at once, the detector's queue full or not; and
  the detector itself, on the untrained model's weights, its probabilities recorded bit for
  bit. The CLI's runs on the CPU take a stand-in detector with
  the stand-in model (`tests/test_cli_run.py`'s `StandInDetector`, each frame's probability its
  mean value): its file an empty `transnetv2.safetensors` beside the stand-in's model files,
  pinned so in place of TransNetV2's weights, which the pin's check then checks.
- The cuts drawn from the probabilities (`seedvr2x/shotcuts.py`) are held by
  `core/tests/test_shotcuts.py` to measurement's own detection
  (`research/scripts/scd_scores.py`'s `tnet_detections`, copied there, on the float32 values read
  as float64, as its analysis reads them), and the possible cuts; the cut list `--plan` writes
  with them, read back by `--cuts`, is `tests/test_cuts.py`'s (the CLI's). The CLI's
  (`--cut-threshold`, `--plan`, a plan planned again or taken up in another environment) by
  `tests/test_cli_run.py`, a `ScriptedDetector` giving chosen probabilities in place of the
  stand-in's, about 0.25 on its clips, which cut nothing at 0.3; with `SEEDVR2X_MODEL_DIR`,
  TransNetV2 itself cuts `test_detector_cli.py`'s synthetic clip at its cuts through the CLI.
  The playlists `--plan` writes beside the list for VLC, `FILE.cuts.m3u8` and
  `FILE.possible.m3u8`, are held by `tests/test_cuts.py` (entries, names, what is said of them)
  and by `tests/test_cli_run.py` (written, left from an earlier threshold and said so, refused
  when they would take the place of a file the job reads or writes).
- A source's rate against its frames' timestamps (`media/rate.py`, `media/source.py`'s
  `timing_refusal`) is held by `tests/test_rate.py` to `seeking.md`'s mechanism 7 made on purpose,
  S9's timestamps (`mechanism7`), and to indexes made there. A file whose timestamps contradict
  its declared rate is refused, never read on a guess: the guidance names the rate the timestamps
  are exactly, or the declared rate's NTSC twin they follow (N for N x 1000/1001, nothing
  else), and its two fixes, a remake (its command, a cover picture attached as the finished file
  holds it) and `--frame-rate`; the join sentence is said only where the twin rule alone names N
  for a file declared N x 1000/1001. Files with no rate to name get sptenc's refusal as it is.
  Other rates declared are held to seedvr2x's own test of what sptenc's rule passes (rates
  contradicted within its 1 ms: 24000/1001 declared 24/1, 48 declared 50/1, 33 ms a frame
  declared 30/1; joins by ffmpeg's concat demuxer that drift half a frame, refused with the
  remake that puts them back), and what stays read (constant rates to their time base's
  rounding, joins that keep the frame grid, a video starting after its audio). A source
  declared or read at more than 1,000 fps is refused once before the first pass, `--frame-rate`
  too, the master written at 2,000 fps included; 1,000 itself holds. Matroska's readings of the
  NTSC rates are held to files written there. Frames missing on the grid are filled, not
  refused: timestamps made there that sit on a rate's grid with places left empty, read in
  places, with what the source's warning says, and what stays refused. The command line's runs
  are `tests/test_cli_run.py`'s. Opt-in, a real source: `SEEDVR2X_RATE_SOURCE` names it,
  `SEEDVR2X_RATE_EXPECTED` the rate its timestamps follow (N/D); one first pass, then the rule
  without and with `--frame-rate` (its docstring says how).
- Missing frames filled, places and the frame index (`media/grid.py`, `media/index.py`,
  `media/reader.py`): `tests/test_grid.py` holds each decoded frame's nearest place on a rate's
  timeline, the index in places (held rows, maps, what was filled), the detector's probabilities
  and the cuts at the places, and the reads through it, on files with holes made on purpose
  (packets dropped by the noise bitstream filter, the others' timestamps kept) held to the
  frames of the file they were made from; `tests/test_cli_run.py` runs such a source through
  a run, its resumes and a finished job run again.
- A job run again once finished (`tests/test_cli_run.py`) makes nothing: said finished, exit 0,
  with no first pass, detector, plan, memory read, model or unit, its manifest and index
  untouched (whether the index is there, missing, damaged or cut short), still held to its
  settings, model files, source and directory, and to its environment (accepted or refused).
  The files are known by size and modification time (`st_mtime_ns`), recorded beside their
  SHA-256 (the `Finished` fixture; `tests/test_pull.py` for the model files): read where both
  match, nothing; hashed where one differs, the job said finished if the content is the same and
  refused if another file has the same size and a new time; a source vouched for only at its
  recorded path (a copy elsewhere with the same size and time is hashed); a record without the
  time hashed as before. A job with anything left to make vouches for its model files the same
  way (DESIGN.md, Pause and resume, decided on 2026-11-03; `tests/test_cli_run.py`): nothing read
  at a resume's start where size and time match, its load reading each once and proving the
  record's SHA-256 (the stand-in's loads prove as the library's do: `proved`, `LOADED`); a file
  changed with its size and time kept refused at the end of its load, before any unit, the
  record as it was; a touched one hashed at the start and by its load; a new job's unpinned file
  hashed at its start, its pinned one (the stand-in detector's) by its load alone; a job with its
  assembly alone left reading no model file. A pinned file of its size and other bytes is refused
  at the end of its load, the first pass kept, what is kept said before the command the refusal
  advises, which stays its last line (`tests/test_pull_cli.py`). The source is hashed at each start of such a job,
  as before.
  A `.mkv` path's finished file keeps its own manifest beside it (`tests/test_assembled.py`): the
  same command makes nothing, another job or a changed file is refused, `--plan` writes nothing,
  and a stop after each of the end steps (file, checksums, work record, manifest beside it,
  removal) is finished by the next run, whose frames are the uninterrupted run's.
- The assembly (`media/assembly.py`) is held by `tests/test_assembly.py` to ffmpeg's own remux of
  the source and to the source's packets: data streams Matroska can't hold left out with one
  warning, a refusal naming the other streams alone; a cover picture attached to the finished
  file, not a second video track, and checked there by bytes, name and type; ffmpeg's note that
  a cover has no rate to estimate tolerated for the cover alone, its fixture (`unrated`) a
  source whose late MP3 track makes ffmpeg's probe stop short, which the test checks ffmpeg
  does print; the rehearsal's line and its estimate from a timed read. `tests/test_assembled.py`
  holds what the command line does with it: a segment's hash recorded when it is finished
  (`tests/test_units.py`, `tests/test_cli_segment_cmd.py` for a command's), a damaged one made
  again before the join; the rehearsal said before it reads, recorded, trusted by a resume under
  the same ffmpeg and run again under another; the assembly's ffmpeg processes ending with a
  killed seedvr2x (a control variant without the helper, where ffmpeg survives); a finished
  segment found missing or of another size at a start made again, every segment finished and the
  assembly refused for room included, then the file assembled (the start's side in
  `tests/test_cli_run.py`: the segments changed told once, a PNG segment, `--plan` saying it and
  writing nothing).
- Every ffmpeg and ffprobe seedvr2x starts ends with it (`media/ffmpeg.py`'s `ends_with_us`), held
  for all of them by `tests/test_ends_with_us.py`: a static scan of the CLI's files
  (`apps/seedvr2x/`), a function (`scan`) held on its own to synthetic sources, one proof per rule
  (an aliased `subprocess` or `os`, a name imported from either, `os`'s starters by family,
  `spawn*`, `exec*`, `posix_spawn*`, `fork*`, `system` and `popen`; a starter named without being
  called, bound or in a `functools.partial`; the `subprocess` or `os` module itself bound or passed;
  asyncio's and pty's starters by name; multiprocessing whole), one leaving subprocess's exceptions,
  result type and pipe constants alone, and one counting the hooked and unhooked starts and leaving
  plain uses alone; on the CLI's files it finds no violation, 16 starts with the hook (a new one is
  counted on purpose) and one without, the segment command's shell, and on the library's
  (`core/seedvr2x/`, the vendored code left out) none at all, the processes being the CLI's; `prctl`
  resolved in the parent before any fork, the hook holding the typed C function and no `CDLL` (a
  lookup in the child would allocate); a child process per kind of site (the startup check, probes,
  fingerprint, packet count and the rest) with a fake ffmpeg and ffprobe on `PATH` that only wait,
  SIGKILLed alone, and its control with the hook replaced by none in that site's module, where they
  stay, the reader's and the writer's real ffmpeg (skipped without a build seedvr2x accepts), and
  the segment command's shell, left running by a kill, as DESIGN.md has it. `/proc` tells the
  processes left, so Linux only.
- A stop at once ends the process (DESIGN.md, Allocator, the hang of 2026-11-06): the scripts
  are `main:command`, which call `main(owns_process=True)`; `main(argv)` returns its status in
  every other call, as the in-process tests take it. Where the process has used CUDA, a second
  Ctrl-C or SIGTERM, and a CUDA error other than out of memory, end it with `os._exit` inside
  main's except block, once the work directories made are removed, the locks released and the
  lines flushed: the traceback still holds the run's frames there, so no finalizer of the models
  (`blockswap.HostCopy`'s cudaHostUnregister) runs. `apps/seedvr2x/tests/test_stop.py`'s DRIVER
  runs `main.command()`, each step said as it is called (`CALL`), in modes (`STOP_MODE`): a
  decode whose stand-in event never completes (`waiting.wait(lambda: False, ...)`, the process
  taken for one that used CUDA), with an atexit and with a finalizer on an object of the decode's
  own frame each calling libc's `pause()` (a teardown blocked in C: both fail on the code before,
  the process still there 30 s after the signal), and a decode raising a CUDA error: exit 143 or
  130 within 30 s, the line with the wait's length, the job's files as `test_stopped_at_once`
  checks them and the resume as if uninterrupted; exit 1 and the CUDA error's line; a run that
  never used CUDA keeping the line before (`stopped at once (SIGTERM)`) and its exit. Ctrl-C and
  SIGTERM are ignored from the line to the exit; a CUDA error in `--compile`'s warm-up is the
  card's, not a CompileError (`seedvr2x/compiling.py`). The image
  server's `command` ends the process the same way once `serve` has returned
  (`apps/imgserver/main.py`'s `_end`), its first SIGTERM still waiting for the pictures running.
- Every thread's stack on SIGUSR1: `command` registers `faulthandler` (all threads, to stderr,
  not chained) in the CLI and the image server; `tests/test_stop.py` sends it inside a slow step,
  the main thread in Python's sleep and in C calls (libc's `usleep`, mode `c-call`): the main
  thread's stack with the step's frame and the other threads', the run ending 0 with every
  segment. SIGUSR1 goes to seedvr2x alone there, never its process group: ffmpeg ends on it.
- A wait on the GPU (`seedvr2x/waiting.py`; DESIGN.md, Allocator, the hang of 2026-11-06), the poll
  `model.synchronize` and `blockswap.Cuda.wait` go through: `core/tests/test_waiting.py` holds it on
  a fake clock whose sleep advances it, nothing sleeping: the line once past 4 times its unit's
  estimate and never under it, the floor of 60 s, nothing said without a unit or an estimate, NVML's
  reading or its absence in the line, the stacks with the process's first such line only (and the
  real `faulthandler` dump to a file), the end of a wait said once it was, a stop raised inside the
  poll (an exception, a real signal's handler) propagating with the wait recorded (`ended`,
  `ended_by`: by the exception's identity, the exception itself not kept, its frames with it), a
  line that fails (a replaced NVML reading that raises, as a research script's did on the GPU) said
  at debug and the wait going on, the waits running seen from another thread (`longest`, `stalled`),
  units per thread, an app's check (`check`, `checking`: a callable the poll calls after each sleep,
  the calling thread's alone, that may raise and so end the wait; the nodes set ComfyUI's Cancel)
  called at each poll and never by a wait done at once, a wait past its bound that an exception
  ended, or a wait inside a release (`releasing`) that one ended whatever its bound and with no
  unit, abandoned per card (`abandoned(index)`, CUDA's numbering; cleared only by a release of that
  card whose waits all completed, never by another wait or card); `Models.release` sends nothing
  back while its own card's mark holds (`core/tests/test_held.py`,
  `apps/comfyui/tests/test_release.py`), `cuda_used` and `cuda_failed`, and its import making no
  torch; `tests/test_memory.py` NVML's activity (`memory.activity`) on its stand-in. The CLI names
  each unit in `run._begin` with the calibration's estimate; the image server each step of a picture
  inside its model's lock, the warm-up's longest time of that step its estimate:
  `tests/test_service.py` the estimates and the units named, `tests/test_web.py` /health's `stalled`
  (503, a ready server only, ready again after) and the gauge of the longest wait,
  `tests/test_metrics.py` its family.
- A card a CUDA error failed (`waiting.fail`, `waiting.failed`, `model.marking`, `model.unless_failed`;
  the nodes on Windows, 2026-11-07: "Fatal Python error: Aborted" at a free, where under
  cudaMallocAsync a tensor's free reports a pending error inside torch's deleter, which aborts:
  shown alone on the RTX 5090, a device-side assert then a `del`, exit code 3; torch's own
  allocator frees silently and the error comes at the next synchronise): a CUDA error other than
  torch's `OutOfMemoryError` (`waiting.cuda_failed`: a "CUDA error: out of memory" a kernel met is
  one) marks its card, abandoned too and never cleared. From then on nothing seedvr2x controls
  frees, closes or waits on that card: each free on an error's way out first asks the card
  (`model.unless_failed`: the poll, where a pending error comes back and marks it), since a generator's
  own `finally` runs as the error unwinds through its consumer, before any outer block has seen
  it. Frees inside a forward pass before the error surfaces, ComfyUI's own clean-up and the
  interpreter's exit are not seedvr2x's. A decode's normal end polls the card before its causal
  memory is freed (`model.decode_sliced`); `synchronize` marks an error its event's record or
  query meets; `Models.release` polls before its first free; the environment probe passes such an
  error up uncached (`probe._run`), and the detector's close keeps its model on a failed card.
  `core/tests/test_waiting.py` holds the mark (a wait's error, no bound needed; out of memory and
  other errors unmarked; a completed release leaving it); `tests/test_shot_pass.py` a poll after
  the last write with the memory held, a CUDA error at a poll inside a tile's pass or the
  consumer's, or in the writer, tiled or not, leaving what the decode held and marking the card,
  out of memory freeing as before; `tests/test_clip.py` the decode left open and the frames kept;
  `tests/test_env_probe.py` the probe's. `library_fixtures.no_card_failed` (autouse in the four
  projects) clears the marks before each test. The nodes (`apps/comfyui/tests/test_card_failed.py`): the error kept with its frames
  (`models._failures`), the models held, ComfyUI's cache not emptied, the error's first line on
  the node (`progress.FAILED`), at ERROR in the console and in ComfyUI's error with the restart
  (`words.card_failed`), every later run refused at its start with nothing read, a CUDA error at
  a run's end poll holding models not kept, out of memory freeing as before; the Cuts node
  refused on a failed card and marking one (`tests/test_cuts_node.py`). The CLI's and the
  server's CUDA errors still end their process (`main.py`, `service.py`).
- The card read on Windows itself (`seedvr2x/memory.py`: `NVML_WINDOWS`, `CUDA_WINDOWS`,
  `_libraries`, `_unread`): NVML through `nvml.dll` (System32, or NVSMI's directory) and the
  driver through `nvcuda.dll`, the probe's too; a reading NVML can't make said at INFO the first
  time a process meets its reason, at debug after. There NVML counts what a process has made
  resident, not what it allocated (the comment at `NVML_WINDOWS`), and a reading takes the smaller
  figure. `core/tests/test_memory.py` holds the names asked in order, the probe's library and the
  INFO line once; `probe.NVML_WINDOWS` is the same tuple. `library_fixtures.no_card_failed` clears
  the reasons said before each test, so that each test's first is at INFO.
- Windows itself taken for a driver that lends (`memory.windows`, `memory.startup`; the nodes'
  first Windows runs, 2026-11-07 and 08): unasked, so that a run is held to the smaller of the
  driver's and NVML's figures, less `HELD_BACK`, and each step keeps its allocator's reserve
  (`peaks.reserve`): planned for a driver that refuses, the nodes' 1080p clip left nothing beside
  cudaMallocAsync's pool, which went past the card in ComfyUI's process (`startup`'s docstring has
  the figures). The expandable segments are never read as on there, torch's Windows build having
  none (`_allocator`), and the nodes' allocator note says what applies on Windows
  (`comfyui/models.py`, `_allocator_note`); without a setting of the user's, the CLI and the
  server run cudaMallocAsync there. `core/tests/test_memory.py` holds the startup on Windows (not
  asked, the warning, the reading held under the pool, a resume's early reading held, the default
  allocator with nothing said of the environment's) and `apps/comfyui/tests/test_machine.py` the
  note on Windows for both allocators. A resume's early reading on Windows is NVML's alone, which
  there counts what is resident: PROVISIONAL, the CLI on Windows being the next baton's.
- The one clock (`seedvr2x/clock.py`): `tests/test_clock.py` holds it on fake monotonic and wall
  readers, nothing sleeping (`Readers`, `Pace`): the ratio of the wall to the run's clock before and
  after its minimum span, steps of the wall told from drift (a few seconds either way, an hour set,
  a VM asleep, a long unit), the band outside which a ratio is not used, and every duration printed
  (time left, the end, `--until`, time spent, the assembly's reading) said in the wall's seconds. A
  span said before the ratio is measured is the wall's own seconds over it, from the readings taken
  at each line logged (`main._sampled`, the handler's filter, tested where `main._logging` installs
  it, with and without `--verbose`); a step of the wall inside a span, or wall seconds outside 0.8
  to 1.25 of the span's own, leave the run's own seconds, said at debug. The autouse fixture
  `one_to_one_clock` (`core/tests/library_fixtures.py`, imported by each directory's `conftest.py`)
  replaces the clock's readers by a frozen pair, so that every other test sees a ratio of 1.0: a
  long session's real drift would leak into their expected durations.
- `tests/test_disk.py` holds the disk a job takes beside its output (`disk.py`): the
  estimate of the masters at 1080p, at 4K and at an odd size, from the segments a resume has
  written, a segment command's unknown and then measured, the room free to a user, and the line
  among `--plan`'s and a run's warning, which never refuses.
- `--compile` is held on the CPU by `tests/test_compile.py` (`seedvr2x/compiling.py`, `peaks.py`'s
  range of shapes) for the library's side and by `test_compile_cli.py` (the compiled
  passes of `seedvr2x/shot_plan.py`'s plan, `planner.py`'s shapes, the job's plan and record)
  for the CLI's: the plan holding a compiled pass to 1.1 times
  its tile's plain peak and to the measured range of shapes, the shapes warmed (the decoder's latent
  in one layout, the DiT's, dense channels last, for the run's decodes and the calibration's: the
  run's own calls, shots of 1 to 64 latents, are compared with the warm-up's), the order of the load
  with fakes (the VAE compiled before the DiT loads, the compile workers shut before its blocks),
  the VAE's exclusion of the causal convolutions from the graph, inductor's deterministic mode kept
  on, the flag recorded and a resume with the other value another job, the cache in the job's own
  directory, and a graph compiled, or tried, after the warm-up said at warning. The recompile limits
  (per code object and for all) are raised past any warm-up and a higher one the process set is
  kept; a limit hit in the warm-up is a refusal naming the way out, with
  `fail_on_recompile_limit_hit` on and `suppress_errors` off while it runs (torch calls them
  incompatible), both as they were afterwards, on the refusal too. A job outside the shapes measured
  (a pin beyond them, a frame too small for a tile, an untiled decode) is refused, never run plain,
  the refusal naming the pass, its tiles and the shapes, then its advice; a plan with no unit made
  is planned again for the other value of `--compile` without a flag, and units made make it another
  job's settings; inductor's modules count in the environment's comparison only between two compiled
  jobs. The plan's compile line gives the time by the shapes the warm-up compiles (an untiled pass
  is one, never counted by tiles), by kind of job (new, or a resume with the cache warm, found so
  only where the recorded plan is kept as it is: a re-planned resume or a start killed while
  compiling is a new job's figure) and by disk, the line saying only "on a Windows drive" and the
  cache's many small files at debug. A Windows drive is found in `/proc/self/mountinfo`, the longest
  mount point deciding, a 9p mount counting only with `aname=drvfs`. Compiled for real only on a
  tiny module, in a subprocess so that no state of inductor stays in the suite (two tests, skipped
  without a C++ compiler); the VAE's compile on a GPU is not in the default suite.
- The default output (DESIGN.md, Architecture, What it prints): `tests/test_cli.py` holds
  `TORCH_CPP_LOG_LEVEL` set before torch is imported (seen by a finder at each import, in a
  subprocess) and the user's own value winning, and the early reading of the card in a
  subprocess that must import no torch; `tests/test_model.py` the sampler's progress bars off
  but under `--verbose`, and a download's kept; `tests/test_cli_run.py` the lines saying the date
  or the zone's offset changed, and what a user does nothing with (the conversions' fingerprint,
  the assembly's rate and count) at debug; `tests/test_timing.py` the calibration's working at
  debug and a duration under a second said so, its line saying what it is for;
  `tests/test_assembly.py` and `tests/test_assembled.py` the rehearsal's line saying the same.
  The help texts are `tests/test_cli.py`'s: `--compile`'s with no figure in it (no digit, dB, min
  or px; its four points asserted in words), `--accept-env-change`'s naming both its uses, and the
  main help naming `seedvr2x verify`. The system memory the plan counts on is said at debug alone
  by `--plan` and by a run, or that none was read, and is not recorded in the manifest
  (`tests/test_cli_run.py`'s `test_the_system_memory_the_plan_counts_on_is_logged_at_debug`).
  The card line leaves the driver's figure where NVML's is the smaller to a debug record beside
  it (`tests/test_planner.py`; `tests/test_cli_run.py`'s
  `test_the_drivers_figure_is_said_under_verbose`).
  The spilling driver's warnings are written out whole in `tests/test_memory.py`
  (`LENDS_LINE`, `HOLDS_LINE`, and `PLAIN_HOLDS_LINE` for torch's own allocator without the
  segments, which keeps the flat 4.5 GiB), not built from `memory`'s constants, so that a change
  of wording fails the tests that compare them. `HOLDS_LINE` carries the autouse `round_figures`
  fixture's figures, so `SHIPPED_HOLDS_LINE` is the line with the shipped ones (each step planned
  with a reserve by its own peak), held by `test_the_shipped_holds_line_says_the_shipped_figures`,
  and `SHIPPED_PLAIN_HOLDS_LINE` the plain allocator's, held by
  `test_the_plain_allocators_holds_line_keeps_its_4_5_gib`.
- The calibration's measured lengths (`calibration.py`): `tests/test_timing.py`'s
  stand-ins `Steady`, `Heavy` (slow windows, and the first calls costing what the calibration's
  constants say) and `SlowEncode` (slow encodes) give the times, and the tests hold the longer
  window's length by time, never under the attention window's 30 latents or over the job's
  longest, and the warm-up counted as a sample where its first call is small against it. The
  calibration's decodes take the latent in the DiT's layout, `(T', h, w, 16)` dense, whatever the
  encode gave (`Steady` hands on the encode's channel-major view) and a dense one uncopied: a
  compiled decoder's guards read the strides.
- The writers: `tests/test_writer.py` holds the stale checksums, removed only once the new file
  is safe, and relative names holding `": "` or a leading `-` never taken by ffmpeg for a
  protocol or an option (`media/ffmpeg.argument`), for FFV1 and PNG; `tests/test_segment_cmd.py`
  the segment command's stop, 5 s between SIGTERM and SIGKILL with a line saying so, a second
  Ctrl-C killing it at once.
- seedvr2x's record on a video (DESIGN.md, The record in each output; `seedvr2x/record.py`'s
  tags): `tests/test_writer.py` holds it as the master's stream tags, gbrp16le and yuv420p10le,
  read back by ffprobe (Matroska writes the names in upper case), the master's packets and
  checksums those written without it; none by default; on the stream `--segment-cmd` reads, and
  carried by an ffmpeg that re-encodes it; none in a PNG directory's frames.
  `tests/test_assembly.py` holds the finished file's record written by the join from the job's
  (`assemble`'s `record`), whatever the segments carry, the first segment's encoder kept
  (`KEPT`), the source's video tags dropped but language and title. The command line's jobs
  (`tests/test_cli_run.py`, `tests/test_cli_segment_cmd.py`) hold each field against the job's
  manifest (`main._job_record`), on the masters, on the finished `.mkv`, on a command's segments,
  the whole record on the finished file of a command writing MP4, which drops stream tags; a
  re-planned resume's segments `replanned`, with the new plan's values; a cut list's SHA-256 or
  the detection's threshold; and none in a PNG job's frames. The tags change the masters' bytes and sizes, not their frames, packets or
  checksums: what compares masters (the milestone-1 regression's `tools/compare.py`, the
  segments' SHA-256 and sizes, each against its own file's record) compares frames, or a file
  with its own record.
- The environment probe (`seedvr2x/probe.py`, DESIGN.md, Repository and apps, "ComfyUI's process is
  not seedvr2x's"): `tests/test_env_probe.py` holds, on the CPU, the fingerprint's parts from stand-ins
  for torch's and NVML's readings (Linux's library, Windows' two names, macOS none), the cache
  directory of each platform, the cache (written, vouching, run again for each part changed with
  the other environment kept, a check added for another layout, a damaged or unreadable file
  ignored and written again, the write atomic, the margin's field kept), each check on the CPU
  device for the three layouts, and a kernel made wrong on purpose per check, refused with its
  facts, the refusal cached and refusing at once on the next run, a kernel that raises a failed
  check and an out-of-memory error uncached; importing the module imports no torch. The
  matrix product's and the attention's tolerances are PROVISIONAL (probe.py); `pytest -m gpu`
  prints every error against its tolerance on CUDA for each layout. The allocator's margin is not
  measured: the cache has the field, null; its reading side is built (`Probed.allocator_margin`,
  carried by an app into `memory.Reading.margin`, which `peaks.reserve` counts where it is
  larger than the rule's share, under the segments and the pool; the nodes do it,
  `apps/comfyui/models.py`), held by `test_a_cached_margin_counts_in_the_plan_where_it_is_over_the_rules_share`
  (under, equal to and over the 0.33 share) and, through the plans, by `test_peaks.py`,
  `test_shot_plan.py`, `test_still_plan.py` and the nodes' `test_machine.py`.
- A still picture (DESIGN.md, Still images), on the CPU, in `core/tests/`: `tests/test_picture.py`
  holds `seedvr2x/picture.py` to files it makes, bytes in: every format and mode read (PNG in
  RGB, palette, grey at 1 to 16 bits, 16-bit RGB by TorchCodec's decoder, alpha; JPEG, WebP,
  TIFF, BMP, GIF), values exact and, where an ffmpeg seedvr2x accepts is on `PATH`, ffmpeg's
  decode bit for bit (JPEG and lossy WebP within 3 levels of its bilinear conversion); the
  format told by content (`identify`); the profile, the PNG's colour chunks verbatim and the
  EXIF orientation; an opaque alpha dropped and said; an MPO's first image read and warned of;
  JPEGs one after another (a raw MJPEG stream) refused as frames, found by the markers' walk, an
  EXIF thumbnail no frame; a camera's raw file built on TIFF refused by its markers, read from the
  IFDs' bytes with no pixel decoded (Canon CR2's header, DNGVersion, a first image marked a
  reduced-resolution one, a colour filter array or linear raw photometric interpretation in the
  chain or a SubIFD), on files written by hand there, without a real raw file: which of them
  Pillow opens and which it refuses, a multi-page TIFF whose first page is a preview refused as
  raw and not as pages, one whose reduced-resolution image comes later refused as pages, ordinary
  TIFFs (RGB, grey, LZW, with a profile) still read, a cut or looping file never called raw,
  ORF's and RW2's magic numbers no TIFF; a picture over Pillow's limit and under its refusal said
  in the library's own INFO line, from a header-only PNG, the library touching neither Pillow's
  limit nor the warning filters; every refusal, the commands it gives run. `tests/test_frames.py`
  holds `frames.short_side` (the `--resolution` that keeps the displayed size) to `target_size`,
  and `frames.check_picture_target`: a picture's output is its target cropped to even sides, and
  the plan's bound refuses one under the padding's 17 rows (resolution 17 on a 40x24 picture, a
  28x17 target and a 28x16 output), so the picture is refused first as `TargetTooSmall`, its size
  the output's and its `least` 18 rows, while `check_target` passes the same target for a video,
  whose frames are padded at the target's size and cropped after the decode; the CLI's
  `test_still_cli.py`, the server's `test_web.py` and `test_answers.py` and the nodes'
  `test_nodes.py` hold each caller's refusal in its words, and a video at 17 rows planned.
  `tests/test_png.py` holds `seedvr2x/png.py`: the chunks'
  order and CRCs, the record's tEXt Software and uncompressed iTXt XMP after the colour chunks
  and the profile and before IDAT (read back by Pillow, the Software text by ffprobe, which reads
  no iTXt), the five filters and the adaptive one, 8 and 16 bits, grey and RGB, read back
  by Pillow, torchvision, `picture.read` and ffmpeg, the profile by ffprobe; no eXIf chunk ever,
  and each of the 8 orientations written upright (`picture.upright`, below) read back by Pillow as
  its `exif_transpose` of the same picture with the orientation, no tag, no `displaymatrix` for
  ffprobe. `picture.upright`, an exact permutation per EXIF orientation and a view, is held in
  `tests/test_picture.py` to Pillow's `ImageOps.exif_transpose` for all 8 on a picture whose
  samples all differ, the sides swapped for 5 to 8, turned back by its inverse, 0 and 9 refused.
  `tests/test_webp.py` holds `seedvr2x/webp.py`: the record's XMP chunk, last, flagged in VP8X,
  read from the RIFF chunks by hand and by Pillow; the 8-bit codes of `png.quantised` exact in
  Pillow's decode and ffmpeg's (`rgb24`), grey as three equal planes of the mean, the profile
  carried and no EXIF block written whatever the orientation, the 8 orientations written upright
  and read back by Pillow as its `exif_transpose` of the same picture, the refusals by
  their words (a source stored at 16 bits, cICP, cHRM or gAMA without a profile, a side over
  16,383; an sRGB chunk alone taken, a chunk told by its type), the bytes the same every time
  and the effort constants (`METHOD`, `QUALITY`, measured in the module's comment) pinned,
  `picture.read` giving the pixels back.
  `tests/test_record.py` holds `seedvr2x/record.py`: its XMP packet parsed back by the standard
  library's XML parser (`read_xmp`, which `test_png.py`, `test_webp.py` and `test_still_cli.py`
  read a file's packet with), every field at its place under seedvr2x's namespace, the mark's
  URI in full; the app named in the creator (`seedvr2x`, `seedvr2x-imgserver`); the resolution
  asked and the sample aspect as `N:D`; a video's tags, the same fields prefixed `seedvr2x_`, with
  its window cap, level, compile, `replanned` and cuts (a threshold, or `cuts_sha256`, the hash of
  the plain list's text); the bytes the fields' alone; a field out of range refused; a model's
  name that XML can't hold, or a path, written so that it parses, as its file name.
  `tests/test_still.py` holds `seedvr2x/still.py` on `stand_ins.cpu_models`' VAE cast to bfloat16
  and a stand-in DiT: one latent, the two seeds, nothing loaded, the phases in order, tiled
  passes in memory, the correction against the picture, NaN refused. `tests/
  test_still_plan.py` holds `seedvr2x/still_plan.py`, importing no torch: every picture within
  8K and 6000x4000 bounds on cards of 16 to 48 GB given tiles that fit, by `peaks.py` alone; the
  attention windows counted as the DiT's window functions count them, and their most within a
  bound; the floor proved, and what of the DiT a pinned VAE phase keeps lowered for it; the DiT's order, int8, SDPA (the fit of FlashAttention 2's),
  pins, the host's trades and refusal, held readings and each phase's reserve (below); the picture's own
  copies and resize on the GPU (`input_pixels`, a 100 MP input's 1.2 GB in float32 and 0.6 in
  bfloat16, the transform's tensors counted from their sizes, unmeasured): the residents lowered
  where a moment needs it, the refusal naming the largest input that fits, a small input changing
  nothing; the CLI's bound its one picture as stored (`plan`'s `picture`: its own frame, windows
  and pins, a landscape 1080p pinned untiled taken where its plan is untiled, a portrait one
  judged by its own frame, `tiles` taking its target alone), the server's every picture in either
  orientation, unchanged without it. `tests/test_peaks.py`
  holds the colour correction's term by its frames, a video's `SLICE` unchanged.
  `still_plan.said`'s lines point at the DiT file, the swapped blocks and the tiles by a `say`,
  as a refusal's message does (default `errors.plain`: the pointers left out, no `--` in a line):
  `tests/test_still_plan.py` holds the library's own words naming no option and the command
  line's (`library_fixtures.cli_say`) equal to the lines of before; the CLI's `stills.py` gives
  `planner.option`, the server's `service.py` `reservation.option` (its own `--dit-model` and
  `--swap-blocks`: `tests/test_service.py`).
- The command line's still path (`stills.py`, `main.py`'s branch by content): `tests/
  test_still_cli.py` holds `still.upscale` bit for bit to `shot_pass.py`'s pass over a shot of one
  frame, tiled and corrected, and under `-m gpu` on CUDA with the sharp 7B and the VAE from
  `SEEDVR2X_MODEL_DIR`, untiled at 720p, the plan's own at 1080p, and both passes tiled at 1080p
  pinned to 1728 and 1024 px, the models placed by the plan for one picture in the card's free
  memory less 4 GiB left to the desktop (the smaller of the driver's and NVML's readings); a picture told by its content whatever its name, a video named
  `.png` sent to the video path; every option of the parser taken or refused on a still (a new
  one unclassified fails), each refusal; `-o`'s rules and a write that fails leaving nothing;
  `--plan` alone saying the plan, loading and writing nothing, and hashing no model file for the
  record; seedvr2x's record read
  back from each file, PNG and WebP, upright or turned (the app and version, the mark, the source's
  size as stored and the output's as written, the resolution asked and the sample aspect, 48 and
  49 giving the same size, the model files by name and SHA-256, hashed by their load
  alone, no file read whole before it, the seed, the colour correction, the tiles of the pass, the attention kernel), written out
  in the test; runs on the stand-ins whose file is `png.write` of `still.upscale` with its record
  byte for byte, at 16 and 8 bits (`--format png8`), grey, the
  profile kept, an opaque alpha dropped and its line reaching the user; every picture written
  upright, in PNG and in WebP, the 8 orientations of a source (an eXIf chunk put in the stand-in's
  file by the test) giving the output of the same picture stored upright turned by
  `picture.upright`, the file `png.write` or `webp.write` of the turned `still.upscale` byte for
  byte, no eXIf or EXIF chunk and no tag in Pillow, the read line saying the source's
  orientation and the result line `, turned upright from its orientation N` (nothing at 1) with
  the written size, sides swapped for 5 to 8; `-o x.webp`
  (any case) whose file is `webp.write` of `still.upscale` byte for byte, `--format` refused with
  it, a source stored at 16 bits or with cICP and no profile refused naming `.png` before the card
  is read and taken for a `.png`, the profile carried, grey as equal planes, a
  video with `-o x.webp` refused; no
  ffmpeg on `PATH`; and the video path's refusals: `-o x.png`, `png8`, `--plan` alone, and image
  formats ffprobe reads that seedvr2x doesn't as pictures (JPEG XL, AVIF and HEIC by their
  demuxer or major brand, a video listing a sequence brand among its compatible ones kept;
  files made by ffmpeg where its build can, skipped without); a tile pin covering the picture's
  own padded frame refused in the video planner's words; a seed out of range refused in a
  picture's words (no shot, no source), the bounds taken; Pillow's size warning replaced by the
  library's line in a run of the `seedvr2x` script (its stderr holding the line, no "decompression
  bomb", no `DecompressionBombWarning`), the filter `main._pillow_size_warning_off` installs once
  for the process, in front of any thread (the library's `warnings.catch_warnings` would not be
  thread-safe); the animated refusal's ffmpeg command, run on a GIF of uneven delays, making a
  file the video path's `declare` accepts (FFV1, bgr0, 10/1), skipped without an ffmpeg seedvr2x
  accepts. The video's `--format`, `--quality`, `--min-segment` and `--resolution` have no parser
  default, applied after the branch (`main._video_arguments`; the help says 1080, a picture takes
  it in `stills.py`), and `tests/test_cli_run.py` holds a video job's record unchanged, its
  settings recording 1080 with and without the option. The size line (`job.default_smaller`: the
  default makes the output smaller than the source, naming the `--resolution` that keeps its size,
  by the displayed short side) is held for a picture, with `--plan`, with a sample aspect, with a
  size given (nothing more) and where the output is not smaller (nothing), and for a video in the
  run and in `--plan`, a source made larger than 1080 there, with a sample aspect too.
- One picture's figures on the RTX 5090 (DESIGN.md, Still images, decided on 2026-11-03):
  `tests/test_still_measured.py` (`core/tests/`) holds every one-frame VAE pass, colour correction
  and copies measured on that card that day under `peaks.picture_encode_peak`,
  `peaks.picture_decode_peak`, `peaks.colour_one_frame` and `still_plan.copies_peak`, each
  prediction at or over its measure and near it, the untiled 4K encode and decode among them
  (`test_the_4k_untiled_passes_are_within_a_tenth_of_a_gib_over_their_measure`); `tests/
  test_peaks.py` the one-frame ratios by tile side, the untiled ratios by the frame they were
  measured on (`ONE_FRAME_UNTILED_1080P` and `_4K`, a frame between the two taking the smaller
  of the 1080p ratio and a capped 4K one, never falling as the frame grows:
  `test_one_frame_untiled_between_the_two_measured_frames_never_falls_as_the_frame_grows`), the
  video slice's own figure past what one frame was measured on (an untiled pass over a larger
  frame than `ONE_FRAME_UNTILED_MEASURED`, a tile past 2048 px), and a video's figures unmoved;
  `tests/test_still_plan.py`
  the plan choosing no untiled pass past the frame measured (`still_plan._untiled_measured`, 4K
  for both passes now), a pin taking one there at the slice's figure, and a landscape 1080p
  picture pinned untiled taken where its own plan is untiled (the padded 1080p and 1440p
  frames within 4K), the refusal shown on a 4K bound; `tests/test_still_plan.py` the copies'
  measured rule (`still_plan.RESIZE`, cuBLAS's workspace held) and what a phase takes for a reservation
  (`still_plan.Peak`, `bound_peaks`, `picture_peaks`, a picture's passes within a `budget`). That
  patch moved no video plan: the plan matrix's 5,480 cases were main's byte for byte (5,120
  since the tests' numz modes retired, 2026-11-04; since 2026-11-05 in the repository, core's
  `tools/plan_matrix.py`'s 4,760 and the CLI's `tools/plan_lines.py`'s 360 checks of a record:
  4,760 + 360 = 5,120, the counts still). The patches of 2026-11-06 (the lean attention, the
  reserve by the step, the floor's rungs) move plans on purpose, each commit listing what the
  diff of the two matrices moved.
- A picture's refusals carry their kind, their reason and their way out apart (`errors.Refusal`'s
  `kind`, `reason`, `way`; `picture.KINDS`, and `frames.check_target`'s `too-small`), so that the
  image server words its own way out with no file name and no command (DESIGN.md, The image
  server, Errors): `tests/test_picture.py` holds each refusal's message word for word, its reason
  free of the file's name, of `ffmpeg` and of options, and every kind raised; `tests/test_frames.py`
  `too-small`'s.
- The image server (DESIGN.md, The image server), on the CPU in its own venv, from
  `apps/imgserver/`: `tests/serving.py` puts a stand-in machine in `service.Card`'s place (the
  card's figures chosen, the stand-in models: `cpu_models`' VAE in bfloat16 and the DiT of
  arithmetic of `core/tests/test_still.py`, failures put in on purpose) and runs the server over
  a real socket on 127.0.0.1 (`main.launch`, `bring_up`, `shut`: `main.serve` less its signals,
  which the main thread alone takes); `tests/test_web.py` every endpoint, header, status and
  state, the limits before a body is read, the deadlines, a picture's bytes those the library
  makes of it with the same seed and tiles; `tests/test_service.py` the steps under a lock per
  model, one picture in the DiT while another is in the VAE, each with its bits alone, the
  reservation held again when idle, the warm-up in both orientations, the start refused; `tests/test_stop.py` the stop in process and by
  SIGTERM in a process of its own; `tests/test_reservation.py`, `test_admission.py`,
  `test_answers.py`, `test_metrics.py` (read back by prometheus_client's parser, a dependency of
  the tests alone), `test_main.py`, `test_names.py` and `test_lock.py` the parts alone. The
  library's three steps (`still.encoded`, `sampled`, `decoded`) are held to `still.upscale` by
  the CLI's `tests/test_still_cli.py`, through `upscale`, which composes them.
- The library's callbacks for a caller's bar, text and Cancel (`core/tests/test_progress_hooks.py`),
  all observing only: `tiling.TileStep` told by a tiled encode and decode (0 before the first
  tile, then each; a decode tells every tile before its first slice), through `shot_pass`,
  `clip.upscale`'s `tile` and `still.encoded` and `decoded`, the bits of a tiled picture and clip
  those of the same call without it, an untiled pass telling none, a tile raising a
  `BaseException` stopping the pass with nothing alive; `blockswap.read`'s `progress` (the bytes of
  the file read, in order, ending at its size, hashing or not, the tensors equal to a read
  without); `pull.fetch`'s `progress` through the progress bar class `hf_hub_download` is given
  (huggingface_hub 1.33.0 and 2.4.0's, read in their sources): a plain download's one bar with its
  chunks and a resume's negative step, a Xet download's two bars with the keywords the reporter
  gives them and no `update_transfer`, the most either counted told in order to the file's size,
  0 told for every file before the first begins, none passed to the library without a callback,
  and a huggingface_hub without `tqdm_class` (0.36.2) downloading as before with a warning;
  `detector.probabilities`' `progress` per batch with the same probabilities.
- The ComfyUI nodes (DESIGN.md, Repository and apps), on the CPU in the CLI's environment, from
  `apps/comfyui/`: `tests/comfy_stand_in/` writes ComfyUI's few modules from the reading of
  v0.44.0, no ComfyUI code (`comfy.model_management`: `current_loaded_models` of weak
  references, `load_models_gpu`, `free_memory` in ComfyUI's order and partial, `unload_all_models`,
  Cancel as a flag that raises a `BaseException`, torch's cache that a partial unload fills and
  `soft_empty_cache` empties; `comfy.model_patcher`; `comfy.utils`' ProgressBar, recording;
  `comfy_api.latest`: `io`'s types, `Schema`, `ComfyNode`, `NodeOutput`, `Hidden` and
  `ComfyExtension`, with the executor's run of a node on a clone of its class that holds the
  hidden values; `server`'s `send_progress_text`, recording), and `tests/standins.py`
  the machine on the CPU (`models.Machine`: the card is the stand-in manager's, the models those
  of `core/tests/test_still.py` with place and release recorded) and ComfyUI's other models on the
  stand-in card. `tests/test_registration.py` holds the one patcher (its sizes, strong
  references, Unload models and POST /free dropping the weights, the next run loading again,
  keep in RAM off, `--disable-smart-memory`); `tests/test_room.py` the room (the plan made for the
  card without ComfyUI's models and less the room ComfyUI keeps for the system
  (`extra_reserved_memory`, read at each run; the stand-in's `EXTRA_RESERVED` is 0 unless a test
  sets it, so that the other tests' figures stay theirs; asked of ComfyUI beside the plan's need,
  not twice; a card short once it is kept refused; a reserve over the free memory counting 0; the
  card line saying it only where there is one), `free_memory` asked for the plan's largest phase plus
  `ASK_MARGIN` (a partial unload's buffer) and only when the card has less than the phase alone,
  never more, seedvr2x's registration kept, a card still short of the phase refused, the
  library's command-line allocator and limit warnings left out of ComfyUI's console for the
  nodes' own lines, once); `tests/test_release.py`
  the card freed at the end, on Cancel between steps and on an error, and `strays` finding
  nothing on the real `Models.place` and `release` over test_held's toy models, and the models
  of a run not kept gone while the exception that ended it is held (the frames cleared, a
  refusal raised from the library's error included), with what they keep on the device (the
  RoPE's cache, `model.forget`); `tests/test_nodes.py` each picture of a batch and a clip equal
  bit for bit to `still.upscale` and `clip.upscale`, the batch's output allocated once and
  counted in the plan's `process`, the alpha rule, the plan said on the node, in the console and
  as text with what each auto became and the host memory beside the host's, the attention,
  colour correction and seed, the probe, a clip of 1, 2 and 3 shots equal bit for bit to
  `clip.upscale` of each shot one after the other (each seeded by its first frame, its frames
  written into the clip's one output array, no window across a cut, the plan made once and said
  shot by shot, the bar over all the shots), a cut list for other frames refused, a clip the host
  can't hold refused before a model loads with the command line's way named, a seed some shot would
  seed past 2**32 (the seed plus the last shot's first frame and the encode's offset) refused
  before a model loads, a picture at resolution 17 refused and a clip at 17 planned, four channels
  with an opaque alpha a view of the same memory and not a copy, the input said in the
  plan's RAM line and out of the comparison, and the INFO line naming the command line said once
  at import and never by a run; a clip at enhanced and maximum equal to `clip.upscale` at the
  level shot by shot (the shot of 2 frames with one run of its two), a level that is no option
  refused at the node's start, the runs writing into the one output array, the bar over the
  runs' units moving forward and ending full, a Cancel between runs (`test_release.py`) freeing
  the card with no later run made; `apps/comfyui/tests/test_progress.py` what the nodes show
  (`progress.py`) on a fake clock and the stand-in bar and text box: the bar the upscale's
  expected time (each tile, window and decode tile of every run of every shot, or picture, by its
  size, a run being a pass over the shot from its first frame with its own windows and tiles; a
  kind's cost the card's figure until its first unit is made, then the run's own, the rest spread
  again over what remains, monotone, full at the last unit only, an untiled decode moving it by
  its frames written), the text of each form ("Shot 2 of 4 (frames 121 to 240): encoding, tile 3
  of 6", "Shot 2 of 4, run 3 of 4 (frames 123 to 240): upscaling, window 2 of 3" (a later run
  names its own frames from its first and counts its decode in them, "decoding, 60 of 118
  frames"), "Frames 3 to 480, run 2 of 2: ...", a clip of one shot, "Picture 3 of 8: decoding, tile
  2 of 4", "Picture:
  decoding", the time so far and "about ... left" only once a window is measured, the plan and
  "480 frames in 4 min 32 s" at the end, the box replaced and not appended to), the previews (one
  a decode slice or a picture, the newest frame, ComfyUI's preview size, a smaller frame at its
  own), the download and the reading on one bar of a constant total with their lines, Cancel as
  the real hook raises it (the stand-in bar clears the flag and raises) honoured on the node's
  thread and, from another thread, left pending under the flag's lock with the bar and text
  untouched, the Cuts node's scan, and the nodes' runs over the stand-in models with 4 tiles to
  each VAE pass: the bits the library's, the bar, text and previews through the tiles (at
  maximum too, each preview a frame the output then holds, the slice's newest that its run keeps),
  Cancel after a tile freeing the card, the load told on a bar of its own before the upscale's
  (Cancel between runs is `test_release.py`'s, on the run's text); the time said between events
  (`Progress.running`, `Ticker`: a daemon thread, joined at the end with no time cap (it ends at
  once at the stop, its sends never block), on Cancel and on an error,
  its pause cued by the test and its clock the progress's own: the text the events' own with the
  time moved on, nothing before the first event or after the end, the bar never touched, no
  thread alive after a run that ended, was cancelled or failed, a text that can't be sent ending
  the thread and not the run; `conftest.py`'s autouse `no_ticks` keeps it from ticking in the
  other tests, whose runs are slower than its second); `tests/test_stall.py` a wait on the GPU
  that lasts, on the nodes (an event that never completes, `waiting.wait(lambda: False, ...)` on a
  fake wait clock):
  each event naming the next unit with its estimate once a unit of its kind is measured and none
  before, the text and ComfyUI's console saying a wait past its bound with the way out (restart
  ComfyUI) and the text normal again when it ends, Cancel leaving it at a poll, torch's cache not
  emptied and models not kept held rather than dropped after it (Unload models too), `running`
  giving the check and the unit back, and every later run refused at its start, nothing read or
  placed, until ComfyUI restarts; the release after a run an exception ended (Cancel or an error)
  run inside ComfyUI's interrupt as the check, the node's text `progress.FREEING` meanwhile, a
  second Cancel leaving it with the card marked whatever the bound, torch's cache not emptied and
  the next run refused, `progress.STOPPED` after it, models not kept waiting for the device in the
  poll first (`model.synchronize`) under the same text and second Cancel, a healthy release putting
  the text back (none said before: cleared) and marking nothing, and one after a normal end with no
  check and no text; `tests/test_cuts_node.py` seedvr2x Cuts over the untrained detector's
  weights and scripted probabilities (the cut list and the text of the cuts and the
  possible cuts, one shot mode running no detector, the threshold, the file found in a folder or
  fetched by its pin, the detector closed after, the alpha left out, its refusals, its list into
  Upscale Video, its file's refusals naming seedvr2x Cuts and its own model folder, sizes with
  separators); `tests/test_plans.py`
  the DiT files' names, the inputs (the pictures first, then the models, the widgets, then the
  plan's numbers as optional sockets of seedvr2x's own types, no value standing for auto, each
  type given by its value node alone: Tile, Blocks Swapped and Window Length on the planner's
  grid of tiles, blocks and windows, the whole image at once, a window set in frames on the
  model's grid and given in latents, a number off the grid refused queued or linked, a value no
  value node gives refused at the upscale node's start; every input and output with a
  label and a tooltip, none in the library's terms; the colour correction a BOOLEAN, split or
  none, `quality` a combo (standard, enhanced, maximum) on Upscale Video alone, its tooltip's
  points and the plan's quality line (the level and its runs, a short shot's missing runs not
  counted), the seed's fixed control; the Models node's model, folder and RAM tooltips; a value the
  prompt holds or a link gives refused in the nodes' words at queue time and at the node's
  start, an unwired one not given), the settings' pins
  reaching the plan exactly, the text's lines, the V3 registration (the entrypoint, the seven
  ids, names, inputs, outputs and hidden id), `nodes.py` the one user of `comfy_api`;
  `tests/test_keeping.py` keep in RAM auto/on/off against ComfyUI's cap (its MAX_PINNED_MEMORY
  or its rule), what is said, the first file's memory freed before the second is pinned,
  Unload models under a cached Models output said once, the Models node reading nothing;
  `tests/test_plan_only.py` the plan alone (the headers read and nothing loaded, registered or
  freed, an execution blocker for the pictures); `tests/
  test_words.py` every refusal class worded without "--", and the library's real refusals (a
  damaged or unpinned file in a cache and in a directory written there, the plan with tiny
  figures, the target check) worded with the way out; the four kinds of a plan (memory, host,
  input, pin) worded from their facts and never by the library's sentence, in none of the library's
  terms, with the value node and the socket that take the number that fits, and a clip's host
  refusal naming what the command line does instead; `tests/test_machine.py` the reading
  shifted by ComfyUI's models, torch's allocator setting given to `memory.startup` and the
  environment left as it was; `tests/test_process_state.py` torch's memory fraction never set
  and the global random generators as they were after a run; `tests/test_room.py` also the card
  read again after torch's cache is emptied; `tests/test_workflows.py` the two example workflows
  (`workflows/picture.json`, `clip.json`, ComfyUI's UI format): each parses, every node seedvr2x's
  or a stock one of a table read from v0.44.0, seedvr2x's sockets, outputs and widgets equal to
  their V3 schemas' in count, order and defaults, every optional input unwired, every link
  joining an output to an input of its type, Models wired into the upscale node and no Tuning
  node, the plan shown as text, the clip's Cuts between Load Video and Upscale Video. Smoke in
  the real ComfyUI, CPU only, by hand: the folder linked into its `custom_nodes`, then
  `CUDA_VISIBLE_DEVICES= python main.py --cpu
  --quick-test-for-ci` in its environment with `PYTHONPATH` at `core/`, which lists the nodes
  among the custom nodes imported with no error, and a script that imports ComfyUI's `nodes`
  as the server does, awaits `init_external_custom_nodes` and checks the seven ids registered from
  the extension with their schemas (`GET_NODE_INFO_V1`) and run on the executor's locked clone.
- The image server's container (DESIGN.md, The image server, The Docker image):
  `apps/imgserver/Dockerfile`, built from the repository's root (`docker build -f
  apps/imgserver/Dockerfile .`), its ignore file `Dockerfile.dockerignore` beside it, and
  `.github/workflows/imgserver-image.yml`, which pushes it to ghcr.io on release tags. No test
  builds it: a build downloads torch's CUDA wheels, several GB.
- A resume short of the memory free is held by `tests/test_shot_plan.py` (the shortfall and the
  phase a refusal names, planning importing no torch) and the CLI's `tests/test_planner.py` (a
  recorded shape planned again releasing that part), `tests/test_memory.py` (the early NVML
  reading, off this process's own context),
  `tests/test_units.py` and `tests/test_timing.py` (each run's windows decided, kept or made
  again, from the layouts the manifest records) and `tests/test_cli_run.py`: the refusal naming
  `--accept-env-change`, the warning given before the checks, the re-plan under the flag
  (`peaks.PER_TOKEN` multiplied, so that the DiT phase refuses at the stand-in's size), the
  segments and latents kept, `--plan` looking and writing nothing.
- The DiT's fit on the lean attention (`seedvr2x/peaks.py`: `PER_TOKEN`, `PER_WINDOW`, the int8
  and GGUF terms, `DIT_ATTENTION`; `research/docs/small-cards.md`, The lean attention's peak):
  `core/tests/test_peaks.py` holds the prediction at or over the allocated peak of every window
  of `library_fixtures.LEAN_WINDOWS` (one sampling step at 1080p and 4K padding, blocks swapped
  or not, in bytes, from `research/scripts/hold_probe.py`; `research/scripts/lean_fit.py` refits
  from its files) and near it, the two kernels sharing one fit (FlashAttention 2 and SDPA peaked
  at the same figure, so the plan is one), int8 needing less and SDPA the same. A refit moves
  the tests that held plans or peaks measured on the plain code. Those that hold a run measured
  there (numz's recipes, the first matrices, the windowed shots in `test_shot_plan.py`,
  `test_peaks.py`, `test_shot_pass.py` and the CLI's `test_planner.py`) take the
  `library_fixtures.plain_fit` fixture, which sets the plain fit back (and the SDPA surcharge)
  and keep their assertions as main had them; the others had their literals updated and the
  boundaries of their scenarios re-picked, the direction of each move checked (a refused case
  planned, a window longer, never the reverse), and nothing loosened. The ledger of the moves
  is in the commit's message, not here.
- The reserve by the step (DESIGN.md, Allocator, Small cards; `seedvr2x/peaks.py`'s `reserve`,
  `dit_working`, `vae_working`, `memory.counted`): a held run no longer takes a flat 4.5 GiB
  off the card's free memory; each step (a DiT window, an encode, a decode) needs its peak over
  the models resident in it (P) plus a reserve of its own, a floor or a share of P (the DiT's
  capped at 4.5 GiB under expandable segments, where its want stops growing), by the run's
  allocator (`memory.SEGMENTS`, torch's own with expandable segments, the default;
  `PLAIN`, without them, and a driver that refuses; `POOL`, cudaMallocAsync), which
  `Reading.allocator` names. `PLAIN` is today's behaviour byte for byte (the flat reserve in
  `memory.counted`, the VAE's room), by construction, and the tests that held it keep their
  literals under it. `core/tests/test_peaks.py` holds the working sets (`dit_working` equal to
  the DiT's peak less what is resident, over 72 parameter sets, `vae_working`), the table of
  floors and shares by phase and allocator, the DiT's cap at 4.5 GiB, the pool
  scan's coverage, `PLAIN`'s zero and refusals, and a probe's margin counted where larger;
  `tests/test_memory.py` `counted` without the flat reserve, the reading naming its allocator
  and a refusing driver `PLAIN`, `hold` taking the plan's reserves by step (a mapping, or the
  flat figure under `PLAIN`) and `check` judging the step about to run (stop and re-hold at each
  side of its reserve; no step, the smallest; one figure, all steps), and the pool's check
  (stopped where the card no longer holds the step, the pool's bytes counted with the card's,
  the driver's figure where the card's was not read: PROVISIONAL, no re-hold); `tests/
  test_shot_plan.py` and `test_still_plan.py` the plans by step (the reserve each `Peaks`
  carries, need equal to peak plus reserve within the budget, the floors on both sides, the
  record's per-phase reserve, a picture's phase room and the copies judged by theirs, the
  reserve a picture's plan hands the limit, `_largest_input` exact to the pixel) and the
  direction of the change: where no reserve passes 4.5 GiB the segments need no more than the
  flat reserve (a window no shorter, blocks swapped no more), and cards `PLAIN` refuses that the
  segments run. The 2.5 GiB reserve and its `tightened` and `instead` fields are retired with
  its advice (`test_a_check_judges_the_step_about_to_run` replaces it). The server's start holds
  torch to each step's reserve and says it (`apps/imgserver/tests/test_service.py`; the
  reservation per phase and `Peak.over` by allocator in `test_reservation.py`, its
  `RESERVED_OVER` slack under `PLAIN` alone) and the nodes' plan text says the spare memory of
  each step (`apps/comfyui/tests/test_machine.py`; ComfyUI's models gone counted by the
  allocator's rule, a refusing driver none). The CLI's budget line is per step
  (`apps/seedvr2x/tests/test_planner.py`). A plan refused under `PLAIN` says where the segments
  would run it (`test_a_plan_refused_under_torch_s_own_allocator_says_where_the_segments_would_run_it`).
  The pool's VAE floor (1.25 GiB) is design's open question, not a measure.
- The floor's last rungs (DESIGN.md, Allocator, Small cards, "The floor, last rungs";
  `shot_plan.plan`'s `tail_frames` and `rungs`, `Plan.tail_frames` and `Plan.vae_floor`,
  `peaks.VAE_ROOM_LEAST`, `peaks.TAIL_MEASURED`, `peaks.tail_ratio`, `model.tail_frames`): a clip
  that fits no other way is planned again, in this order: a plan refused for the decode takes the
  decoder's tail a frame at a time, then the tail with the VAE passes' room (their reserve floor)
  lowered to 0.5 GiB, under expandable segments only (where it was measured), elsewhere the tail
  alone; one refused for the encode takes the room alone, never the tail; the refusal of a plan
  no rung helps is the last rung's, saying what it counted. The tail shapes the output (its frames
  differ from the plain decoder's at tiles of 512, 576 and 960 px, the same in every run), so the
  plan records it and a resume takes it as recorded, the key written only when it is on and
  every other plan's record as before; the room is planned again. `core/tests/
  test_tail_frames.py` holds the vendored `Decoder3D.tail_frames` on the CPU against the plain
  decoder bit for bit, on `cpu_models`' VAE in float32 (with oneDNN's convolutions off for the
  test: it picks its algorithm by tensor shape, so a frame's convolution and a slice's differ by
  an ulp) and in bfloat16 (the pipeline's, bit for bit as it runs), in the slices
  `model.latent_slices` cuts, the causal memory carried slice to slice and frame to frame, the
  switch on and off again giving today's bits, a VAE without a decoder refused; the cards'
  sizes are measured on the GPU (`research/scripts/tail_probe.py`, hold_probe.py with the tail
  switched on), not here. `test_peaks.py` holds `tail_ratio` at each measured side and off
  them, the prediction at or over each measure, the ratio taken on the part above the weights,
  the colour correction's term untouched and the least room replacing the VAE floor alone;
  `test_shot_plan.py` the rungs in order over the cards, none taken where the plain plan fits
  or under `--compile` (a compiled decode with the tail is not measured), a compiled job only a
  rung would fit refused as `CompileRefused`, saying that it runs without `--compile` and with
  which rungs, named as the plan's lines name them (its facts `tail_frames` and `vae_floor`; the
  CLI's words in `test_compile_cli.py`; the image server and the nodes compile nothing), one no
  plain plan fits, or whose compiled plan is refused for a pin beyond the shapes and not a
  pass's memory, refused as before the rungs, a pinned decode tile helped too, the DiT's and the
  host's refusals no help, `PLAIN` and a refusing driver taking the tail alone and never the
  room, the record read back, a resume holding the tail keeping it and one without never taking
  it, the refusal after every rung (`rungs=False` is the plain plan a compiled job's refusal is
  judged by first); `test_shot_pass.py` and `test_clip.py` the
  switch on for the decode alone and off after, a raise included, and untouched by a plan
  without it; the CLI's `test_planner.py` the plan's lines saying the tail and the room and what
  they cost and `test_cli_run.py` a resume run with the tail; the nodes' `test_plans.py` a clip
  that fits only with the rungs said in a user's words. The tests of refusal scenarios the rungs
  now plan (`test_every_plan_fits`, the brute force, the floor) were re-picked or extended, with
  the plan checked to fit by its own needs. The 0.5 GiB floor and the tail's ratios beyond the
  sides measured are PROVISIONAL (peaks.py).
- The plan matrix, in two tools ([Checks](#checks)). Core's `tools/plan_matrix.py`, the
  library's alone: `shot_plan.plan` over a fixed matrix, 4,760 cases today (cards, targets, shots,
  system memory, pins, kernels, readings, plans recorded on a larger card planned again on smaller
  ones, as many as the larger card plans) and the peaks' bricks, as text, for each DiT format
  (`--dit`), and `diff` of two such texts. The CLI's `tools/plan_lines.py` (`apps/seedvr2x/`),
  on core's `Matrix` and its hooks: the same text with what the command line does with each plan,
  which core can't import: `planner.said`'s lines (with the smaller DiT files' line, as a new
  job's plan says it; `--without-smaller` leaves it out), `card_detail`, `beyond_measured`,
  `check_recorded` on each recorded case and 360 checks of a record (5,120 cases), `plan_again`
  after a recorded plan refused, `compile_shapes` and `on_windows_drive`. `--without-smaller`'s
  text is the private copies' of before (impl-shotpass's, 2026-11-05) for the four formats, their
  messages on several lines written on one and one path renamed. `tests/test_plan_matrix.py`
  (`core/tests/`) keeps core's running: its `--quick` part for each format, in a process of its
  own, with no crash and no torch imported, each format's text its own, and `diff` on two outputs
  made to differ (a plan moved, a case gone, a brick item moved, the recorded cases numbered
  again and compared by what they are, an output unread or cut short). `tests/test_plan_lines.py`
  (the CLI's) the CLI's: the same, core's text in it unchanged but for the lines that hold a
  refusal, which this half words as the CLI does (a plan's refusal under `PlanError`, its
  message in the CLI's options, its parts `instead` as the option string, `phase` and
  `shortfall`, and the Peaks printed as before, `tightened=None, instead=''` kept where the
  retired fields stood, `Matrix.peaks_text`, so that two runs differ in the plans alone): the
  proof that the CLI's words did not move when the refusals got kinds, core's text printing the
  kind, the library's words and the kind's
  facts; the smaller files' line alone left out by `--without-smaller`, and a check of a record
  compared by what it is.
  The matrix's held readings are `memory.SEGMENTS` (kind 2) and `memory.PLAIN` (kind 3,
  `counted` from `memory.counted`); its cases and their counts did not move.
- Met running the suite from a worktree: `git read-tree -u --reset` empties the `upstream/`
  submodule directories (link them again); `--compile`'s cache on a Windows drive under WSL2
  costs minutes, so compile times are measured with the job on Linux's disk; `--until` takes
  a time of day to the minute, so a test that stops a run inside a shot needs a shot of minutes.

## Comments and documents

- A comment gives the reason for the code it sits on, with the measurement or source behind it:
  a section of `research/docs/`, a numz `file:line` at `4490bd1`, a bug of `research/bugs/`.
  Every figure is measured or sourced. A comment repeating what the code says is noise.
- The docstrings of what other modules use give the units (frames, latent frames, GiB), and for a
  tensor its shape, dtype and range: `(T, H, W, C) float32 in [0, 1]`.
- Prose is in British English, as in `research/` and DESIGN.md. Identifiers keep the spelling of
  the libraries they deal with (`color_range`, as ffmpeg, torch and numz write it).
- The documents speak as the project, never "we", "our" or "ours": seedvr2x's model files, found
  by seedvr2x's research, this repository's files (the user, 2026-11-02).
- No silent fallback: a fallback is logged with its reason. numz's silent ones are among its bugs
  ([cli-flags.md](research/docs/cli-flags.md)).
- Each document has its job: DESIGN.md decides; this file says how to work on the code, and
  records the approaches rejected, and why, as they come; README.md is for users;
  `research/docs/` holds the measurements.

## Commits

- Subject: `seedvr2x: <what changed>[: <why, with the figure>]`, on one line.
- A body only when the subject can't carry it, wrapped at about 90 columns.
- One logical change per commit. A vendoring commit names the upstream commit (`numz 4490bd1`).
