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
    picture.py png.py webp.py still.py still_plan.py     a still picture: read, written as PNG
                         or lossless WebP, upscaled on models already loaded, its plan for a bound
    record.py            seedvr2x's record in each output: its fields and the AI mark, a picture's
                         XMP packet, a video's stream tags
    frames.py passes.py peaks.py files.py clock.py errors.py     what the bricks share
    vendor/              model layer, vendored from numz (below)
    transnetv2/          the shot detector's model, vendored from TransNetV2 (below)
  tests/                 the library's, with library_fixtures.py, stand_ins.py and conftest.py
  tools/vendor.py        copy, diff and check of the vendored code
  tools/compare.py       two runs' outputs compared bit for bit
apps/seedvr2x/          the seedvr2x command: distribution seedvr2x (setuptools), flat: its
                         modules at the app's root, no package folder
  main.py                options, logging (the script `seedvr2x = "main:main"`)
  shot run cuts planner plan job manifest resume units scratch environment calibration timing
  stop disk .py          the pipeline per shot, then stitching, planner, a job's plan, saving and
                         resume, its environment record
  stills.py              a still picture's run: its options, -o, plan, load, pass, PNG or WebP
  media/                 I/O layer: ffprobe, ffmpeg pipes, FFV1 and PNG writers
  tests/                 the CLI's, with levels.py and conftest.py
  pyproject.toml uv.lock .python-version LICENSE NOTICE
  README.md MANUAL.md images/
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
  `cli.media.reader`).

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
- **Nothing in it imports numz's runtime** (`src/optimization`, `src/utils`, the rest of
  `src/core`). DESIGN.md (Vendored model code) lists what goes.
- The StableSR-derived colour code is not vendored: its licence is non-commercial (DESIGN.md,
  Colour correction). `seedvr2x/colour.py` writes its wavelet split from the method, clean room:
  whoever changes it doesn't read StableSR's functions in numz's `src/utils/color_fix.py`, lines
  25-247 at 4490bd1 (`adain_color_fix`, `wavelet_color_fix`, `calc_mean_std`,
  `adaptive_instance_normalization`, `wavelet_blur`, `wavelet_decomposition`,
  `wavelet_reconstruction`). numz's own `lab` code there (lines 249-521, Apache-2.0) may be read;
  no numz code is left in the correction since `split` replaced `lab`.
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
  frames pushed to it; the CLI's `media/scan.py` feeds it the frames and `cuts.py` draws the cuts
  from its probabilities. Its licence is the library's: `core/`'s pyproject says "Apache-2.0 AND
  MIT" and ships `transnetv2/LICENSE`; `core/`'s NOTICE and `apps/seedvr2x/`'s README.md credit
  it.

## Environment

- The CLI's uv project, on Python 3.13: `uv sync` in `apps/seedvr2x/` creates
  `apps/seedvr2x/.venv` with the locked versions, `core/` installed editable, and the dev tools.
  The scripts of `research/` that import seedvr2x run with its `bin/python`.
- Its dependencies are pinned to the versions of the numz environment milestone 1 is compared
  with, so that the comparison differs by code only: torch and torchvision (CUDA 13.0 builds, from
  PyTorch's index), diffusers, rotary-embedding-torch and the rest. Bump one on purpose, and run
  that comparison again.
- Linux x86_64 only for now (DESIGN.md: Linux first).
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
  runs. To use it, build a wheel for your GPU ([setup_env.sh](research/scripts/setup_env.sh),
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

## Tests

- `pytest` (as in [Checks](#checks)) runs on the CPU, needs no model and is quick. Tests that
  need ffmpeg skip without a build seedvr2x accepts (see [Environment](#environment)).
- `pytest -m gpu` runs the tests that need CUDA, never selected by default; run it in each
  directory.
- **Where a test file sits.** `tests/test_x.py` is in the directory of the package it tests:
  `core/tests/` for the library's, `apps/seedvr2x/tests/` for the CLI's. File names are unique
  across the two, a helper being imported by name. A file that held both halves is two: the
  library's, and its `_cli` half beside the command (`test_pull_cli.py`, `test_weights_cli.py`,
  `test_compile_cli.py`, `test_detector_cli.py`, `test_padding_cli.py`, `test_int8_cli.py`),
  which use the library's half's helpers and say so in their docstring. The helpers both
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
- `tests/test_pull.py` holds the pull of seedvr2x's own files (`seedvr2x/pull.py`) on the CPU,
  huggingface_hub's two calls replaced, and the pins to the revision itself;
  `test_pull_cli.py` holds what the command line does over it. The pins: `SHA256SUMS`, a few
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
  milestone 1: the output must stay bit-identical to numz's, FFV1 master and float32 frames,
  with numz's 7B fp16, with seedvr2x's own 7B, and with seedvr2x's sharp 7B against numz's run
  of its sharp 7B, all in numz's padding. seedvr2x's own padding, the default, is held
  bit-identical to numz patched to pad the same way (`numerics_patch.py`'s `NUM_PAD`) at 1080p,
  at 720p and on a crop whose width isn't a multiple of 16. Its runs detect their shots, as a
  user's does without `--cuts`, the shot detector on the GPU before the models load, at
  `--cut-threshold 0.5`, where milestone 1's input is one shot, as numz's one batch (0.3 cuts it
  in three): each case skips, naming the file, without `transnetv2.safetensors` in
  `SEEDVR2X_MODEL_DIR`. The other GPU tests' runs give a cut list, empty for one shot. Its
  references, in the `m1/` directory of `SEEDVR2X_REFERENCE_DIR`, and how to make them are in its
  docstring.
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
- numz's padding is a mode for the tests only, never a user's: `test_regression.py`'s milestone-1
  cases run the CLI with `SEEDVR2X_TESTS_NUMZ_PADDING=1` in its environment (`main.NUMZ_PADDING`),
  which an output directory's manifest records, so that a job is never resumed in the other
  padding. `test_regression.py`'s padding cases and the other GPU tests run in the default
  padding.
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
- The cuts drawn from the probabilities (`cuts.py`) are held by `tests/test_cuts.py` to
  measurement's own detection (`research/scripts/scd_scores.py`'s `tnet_detections`, copied
  there, on the float32 values read as float64, as its analysis reads them), the possible cuts
  and the cut list `--plan` writes with them, read back by `--cuts`. The CLI's
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
  time hashed as before; a job with anything left to make, its assembly included, hashing
  everything at each start.
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
  range of shapes) for the library's side and by `test_compile_cli.py` (`planner.py`'s compiled
  passes, the job's plan and record) for the CLI's: the plan holding a compiled pass to 1.1 times
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
  The spilling driver's two warnings are written out whole in `tests/test_memory.py`
  (`LENDS_LINE`, `HOLDS_LINE`), not built from `memory`'s constants, so that a change of wording
  fails the tests that compare them. `HOLDS_LINE` carries the autouse `round_figures` fixture's
  figures, so `SHIPPED_HOLDS_LINE` is the line with the shipped ones, held by
  `test_the_shipped_holds_line_says_the_shipped_figures`.
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
  holds `frames.short_side` (the `--resolution` that keeps the displayed size) to `target_size`.
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
  bound; the floor proved, and what of the DiT a pinned VAE phase keeps lowered for it; the DiT's order, int8, SDPA,
  pins, the host's trades and refusal, held readings and the tight reserve; the picture's own
  copies and resize on the GPU (`input_pixels`, a 100 MP input's 1.2 GB in float32 and 0.6 in
  bfloat16, the transform's tensors counted from their sizes, unmeasured): the residents lowered
  where a moment needs it, the refusal naming the largest input that fits, a small input changing
  nothing. `tests/test_peaks.py`
  holds the colour correction's term by its frames, a video's `SLICE` unchanged.
- The command line's still path (`stills.py`, `main.py`'s branch by content): `tests/
  test_still_cli.py` holds `still.upscale` bit for bit to `shot.py`'s pass over a shot of one
  frame, tiled and corrected, and under `-m gpu` on CUDA with the sharp 7B and the VAE from
  `SEEDVR2X_MODEL_DIR`, untiled at 720p and both passes tiled at 1080p, placed by the plan for
  one picture in the card's free memory less 4 GiB left to the desktop (the smaller of the
  driver's and NVML's readings); a picture told by its content whatever its name, a video named
  `.png` sent to the video path; every option of the parser taken or refused on a still (a new
  one unclassified fails), each refusal; `-o`'s rules and a write that fails leaving nothing;
  `--plan` alone saying the plan, loading and writing nothing, and hashing no model file for the
  record; seedvr2x's record read
  back from each file, PNG and WebP, upright or turned (the app and version, the mark, the source's
  size as stored and the output's as written, the resolution asked and the sample aspect, 48 and
  49 giving the same size, the model files by name and SHA-256, hashed before the
  load, the seed, the colour correction, the tiles of the pass, the attention kernel), written out
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
  prediction at or over its measure and near it; `tests/test_peaks.py` the one-frame ratios by
  tile side, the video slice's own figure past what one frame was measured on (an untiled pass
  over a larger frame, a tile past 2048 px), and a video's figures unmoved; `tests/test_still_plan.py`
  the plan choosing no untiled pass past the frame measured (`still_plan._untiled_measured`), a
  pin taking one there at the slice's figure; `tests/test_still_plan.py` the copies' measured rule
  (`still_plan.RESIZE`, cuBLAS's workspace held) and what a phase takes for a reservation
  (`still_plan.Peak`, `bound_peaks`, `picture_peaks`, a picture's passes within a `budget`). No
  video plan moves: the plan matrix's
  5,480 cases are main's byte for byte.
- A picture's refusals carry their kind, their reason and their way out apart (`errors.Refusal`'s
  `kind`, `reason`, `way`; `picture.KINDS`, and `frames.check_target`'s `too-small`), so that the
  image server words its own way out with no file name and no command (DESIGN.md, The image
  server, Errors): `tests/test_picture.py` holds each refusal's message word for word, its reason
  free of the file's name, of `ffmpeg` and of options, and every kind raised; `tests/test_frames.py`
  `too-small`'s.
- A resume short of the memory free is held by `tests/test_planner.py` (the shortfall and the
  phase a refusal names, a recorded shape planned again releasing that part, planning importing
  no torch), `tests/test_memory.py` (the early NVML reading, off this process's own context),
  `tests/test_units.py` and `tests/test_timing.py` (each run's windows decided, kept or made
  again, from the layouts the manifest records) and `tests/test_cli_run.py`: the refusal naming
  `--accept-env-change`, the warning given before the checks, the re-plan under the flag
  (`peaks.PER_TOKEN` multiplied, so that the DiT phase refuses at the stand-in's size), the
  segments and latents kept, `--plan` looking and writing nothing.
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
