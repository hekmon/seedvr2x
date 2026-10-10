# seedvr2x: context for contributors and coding agents

Read this, then [DESIGN.md](DESIGN.md), before changing any code.

## What this is

seedvr2x is a SeedVR2 video upscaler for long runs, built from what [../research/](../research/)
measured on numz's CLI. DESIGN.md is its specification: what to build and why, with the
measurement behind each decision. The code follows it. A decision the code shows to be wrong or
missing is settled in DESIGN.md first, not changed in passing.

## Layout

```
seedvr2x/
  AGENTS.md DESIGN.md README.md LICENSE NOTICE
  pyproject.toml uv.lock .python-version
  src/seedvr2x/
    cli.py __main__.py   CLI layer: options, logging
    media/               I/O layer: ffprobe, ffmpeg pipes, FFV1 and PNG writers
    runtime/             runtime layer: the pipeline per shot, then stitching, planner, BlockSwap,
                         resume
    vendor/              model layer, vendored from numz (below)
  tests/
  tools/vendor.py        copy, diff and check of the vendored code
```

The layers are DESIGN.md's (Architecture). This is the only uv project of the repository: the
scripts in `../research/` stay standalone.

## The vendored model layer

`src/seedvr2x/vendor/` holds the model code: ByteDance's SeedVR (`e4de8c2`) as modified by numz
(`4490bd1`), under Apache-2.0. [provenance.md](../research/docs/provenance.md) says what each file
is and what numz changed in it.

- **Path for path.** numz's `src/X` is `vendor/X`, and its configs and text embeddings
  (`configs_7b/`, `configs_3b/`, `pos_emb.pt`, `neg_emb.pt`) sit at the root of `vendor/`. numz's
  model code imports its own modules with relative imports only, so the copy imports unchanged,
  and its diff against numz only shows real changes.
- **Never reformatted**, and linted for errors only (see [Checks](#checks)): a reformat would bury
  our changes in the diff.
- **Modified files are marked**, as Apache-2.0 §4(b) requires: one line right under the original
  header, `# Modified for seedvr2x: <summary>.`, and a `# seedvr2x: <why>` comment on each change.
- **Nothing in it imports numz's runtime** (`src/optimization`, `src/utils`, the rest of
  `src/core`). DESIGN.md (Vendored model code) lists what goes.
- The StableSR-derived colour code is not vendored: its licence is non-commercial (DESIGN.md,
  Colour correction). `runtime/colour.py` writes its wavelet split from the method, clean room:
  whoever changes it doesn't read StableSR's functions in numz's `src/utils/color_fix.py`, lines
  25-247 at 4490bd1 (`adain_color_fix`, `wavelet_color_fix`, `calc_mean_std`,
  `adaptive_instance_normalization`, `wavelet_blur`, `wavelet_decomposition`,
  `wavelet_reconstruction`). numz's own `lab` code there (lines 249-521, Apache-2.0) may be read;
  no numz code is left in the correction since `split` replaced `lab`.
- `tools/vendor.py diff` compares the copy with numz and with ByteDance, both read from the
  submodules' git objects at the pinned commits, never from their work trees: a moved submodule
  can't shift the reference. `tools/vendor.py check`, also run by the tests, verifies that a file
  carries the marker exactly when it differs from numz, that the original headers are kept, that
  nothing imports numz's runtime, and that our edits add no syntax error or undefined name.

## Environment

- A uv project on Python 3.13: `uv sync` creates `.venv` with the locked versions and the dev
  tools.
- The dependencies are pinned to the versions of the numz environment milestone 1 is compared
  with, so that the comparison differs by code only: torch and torchvision (CUDA 13.0 builds, from
  PyTorch's index), diffusers, rotary-embedding-torch and the rest. Bump one on purpose, and run
  that comparison again.
- Linux x86_64 only for now (DESIGN.md: Linux first).
- huggingface_hub is declared, at the version the lock held already through diffusers: it pulls
  seedvr2x's own model files from its repo at the pinned revision, into Hugging Face's cache
  (`runtime/pull.py`). seedvr2x imports it to look in the cache on every run without
  `--model-dir`, which reads the files from a directory instead; diffusers, which the vendored
  model imports, brings it into every run anyway. Its telemetry is off: `cli.main` sets
  `HF_HUB_DISABLE_TELEMETRY` before either import, unless the user has. hf_xet, its Xet
  downloader, seedvr2x never imports: its import writes a log file under `HF_HOME`.
- ffmpeg and ffprobe on PATH, 7.1 or later (for setparams' chroma_location), with zscale
  (libzimg), scdet and ffv1: seedvr2x checks the build at startup and refuses to run without
  them (DESIGN.md, Input).
- **FlashAttention 2 is not a dependency.** PyPI only has its source, a long CUDA build, and
  seedvr2x runs without it: it uses FA2 when installed, else PyTorch's SDPA, and logs which one
  runs. To use it, build a wheel for your GPU ([setup_env.sh](../research/scripts/setup_env.sh),
  [environment.md](../research/docs/environment.md)) and install it into the venv with
  `uv pip install --no-deps <wheel>`. `uv run` keeps it, but `uv sync` removes what the lock
  doesn't list: install it again after a sync, or sync with `--inexact`.

## Checks

```
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

- ruff: line length 100, rules E, F, W, I, B, UP and RUF. The vendored code is excluded from
  formatting and from that lint. `uv run tools/vendor.py check` lints it for errors only (syntax
  errors, undefined names), and reports those that numz's own files don't have: numz has some
  already, false positives and a bug in a class nothing builds.
- pyright: strict on `src/seedvr2x`, standard on the tests and tools. The vendored code is
  excluded: the tests show that it imports and runs.

## Tests

- `uv run pytest` runs on the CPU, needs no model and is quick. Tests that need ffmpeg skip
  without a build seedvr2x accepts (see [Environment](#environment)).
- `uv run pytest -m gpu` runs the tests that need CUDA, never selected by default.
- Tests that need the model weights read their directory from `SEEDVR2X_MODEL_DIR`, and skip
  without it. It holds numz's files and seedvr2x's own side by side (links will do): the GPU
  tests run numz's 7B fp16 and VAE, the weights their references were made with, and the
  regression seedvr2x's own as well. With it, the CPU suite also checks every known model file
  there by its header (`tests/test_weights.py`).
- `tests/test_pull.py` holds the pull of seedvr2x's own files (`runtime/pull.py`) on the CPU,
  huggingface_hub's two calls replaced, and the pins to the revision itself: `SHA256SUMS`, a few
  kilobytes, downloaded into a temporary `HF_HOME` behind a link, and the sizes read from the
  Hub's API, skipped offline (no network to huggingface.co, or `HF_HUB_OFFLINE` set). A test
  that runs seedvr2x in a process of its own gives it its own `HF_HOME` and `HF_HUB_OFFLINE=1`,
  but that one; hf_xet is imported in such a process only. With `SEEDVR2X_TESTS_PULL=1`, it
  pulls the three v1 files into Hugging Face's cache too, about 33 GB the first time;
  `test_regression.py`'s `test_milestone1_from_the_cache` (GPU) then runs the default models
  from the cache, `HF_HUB_OFFLINE=1`, and skips until both are there, saying how to fetch the
  two alone (`uv run hf download`, 17 GB).
- `tests/test_regression.py` (GPU) holds every change to the runtime or the vendored code to
  milestone 1: the output must stay bit-identical to numz's, FFV1 master and float32 frames,
  with numz's 7B fp16, with seedvr2x's own 7B, and with seedvr2x's sharp 7B against numz's run
  of its sharp 7B, all in numz's padding. seedvr2x's own padding, the default, is held
  bit-identical to numz patched to pad the same way (`numerics_patch.py`'s `NUM_PAD`) at 1080p,
  at 720p and on a crop whose width isn't a multiple of 16. Its references, in the `m1/`
  directory of `SEEDVR2X_REFERENCE_DIR`, and how to make them are in its docstring.
- numz's padding is a mode for the tests only, never a user's: `test_regression.py`'s milestone-1
  cases run the CLI with `SEEDVR2X_TESTS_NUMZ_PADDING=1` in its environment (`cli.NUMZ_PADDING`),
  which an output directory's manifest records, so that a job is never resumed in the other
  padding. `test_regression.py`'s padding cases and the other GPU tests run in the default
  padding.
- `split`, the colour correction, is held to the colour study's own `split()`
  (`../research/scripts/colour_variants.py`): on the CPU by `tests/test_colour.py`, on the GPU
  in the pipeline by `tests/test_split.py`, at x2 and x4, in one window and in three, within one
  16-bit code. Both load the study's script with its baseline `colour.py` from the repository
  (the baseline from git), or from `SEEDVR2X_COLOUR_STUDY` (the directory holding
  `colour_variants.py`) and `COLOUR_BASELINE` (that `colour.py`) on a copy of `seedvr2x/` alone,
  and skip without them; one of the two set without the other fails. `tests/test_split.py` also
  scores `split` against milestone 1's input with milestone 5's metrics, its thresholds its first
  run's figures (2026-10-09) plus milestone 5's tolerances.

## Comments and documents

- A comment gives the reason for the code it sits on, with the measurement or source behind it:
  a section of `../research/docs/`, a numz `file:line` at `4490bd1`, a bug of `../research/bugs/`.
  Every figure is measured or sourced. A comment repeating what the code says is noise.
- The docstrings of what other modules use give the units (frames, latent frames, GiB), and for a
  tensor its shape, dtype and range: `(T, H, W, C) float32 in [0, 1]`.
- Prose is in British English, as in `../research/` and DESIGN.md. Identifiers keep the spelling of
  the libraries they deal with (`color_range`, as ffmpeg, torch and numz write it).
- No silent fallback: a fallback is logged with its reason. numz's silent ones are among its bugs
  ([cli-flags.md](../research/docs/cli-flags.md)).
- Each document has its job: DESIGN.md decides; this file says how to work on the code, and
  records the approaches rejected, and why, as they come; README.md is for users;
  `../research/docs/` holds the measurements.

## Commits

- Subject: `seedvr2x: <what changed>[: <why, with the figure>]`, on one line.
- A body only when the subject can't carry it, wrapped at about 90 columns.
- One logical change per commit. A vendoring commit names the upstream commit (`numz 4490bd1`).
