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
                         resume; the shot detector
    vendor/              model layer, vendored from numz (below)
    transnetv2/          the shot detector's model, vendored from TransNetV2 (below)
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

## The vendored TransNetV2

`src/seedvr2x/transnetv2/` holds the shot detector's model (DESIGN.md, Shot detection):
TransNetV2's official PyTorch code (github.com/soCzech/TransNetV2 at `85cef72`, MIT),
`inference-pytorch/transnetv2_pytorch.py` as `transnetv2_pytorch.py`, with TransNetV2's `LICENSE`
beside it; the `__init__.py` is seedvr2x's.

- **A directory of its own,** not `vendor/`: `vendor/` is numz's tree path for path, which
  `tools/vendor.py` diffs against numz, and TransNetV2 is another upstream under another licence.
  The file's own directory is left out, `inference-pytorch` being no name Python imports.
- **Byte for byte,** never reformatted, excluded from ruff and pyright as `vendor/` is:
  `tests/test_detector.py` holds the file and the licence to their SHA-256 at that commit, those
  `../models/transnetv2_weights.py` pins (`SOURCES`), so any change is deliberate. It runs with
  the locked torch unchanged; a change would be marked as `vendor/`'s are, its pin changed with it.
- `runtime/detector.py` builds it, loads seedvr2x's `transnetv2.safetensors` into it (pinned in
  `runtime/pull.py`, made by `../models/transnetv2_weights.py`) and runs measurement's windows on
  it. NOTICE and README.md credit it.

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
  (libzimg), idet and ffv1: seedvr2x checks the build at startup and refuses to run without
  them, or without any other filter, encoder, decoder or muxer it uses (`media/ffmpeg.py`;
  DESIGN.md, Input).
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

- ruff: line length 100, rules E, F, W, I, B, UP and RUF. The vendored code, `vendor/` and
  `transnetv2/`, is excluded from formatting and from that lint. `uv run tools/vendor.py check`
  lints `vendor/` for errors only (syntax errors, undefined names), and reports those that numz's
  own files don't have: numz has some already, false positives and a bug in a class nothing
  builds. `transnetv2/` is held to its pinned bytes (`tests/test_detector.py`).
- pyright: strict on `src/seedvr2x`, standard on the tests and tools. The vendored code is
  excluded: the tests show that it imports and runs.

## Tests

- `uv run pytest` runs on the CPU, needs no model and is quick. Tests that need ffmpeg skip
  without a build seedvr2x accepts (see [Environment](#environment)).
- `uv run pytest -m gpu` runs the tests that need CUDA, never selected by default.
- Edit nothing under `src/` while a test run is going: a job's
  settings record a hash of the package's files (`manifest.code_sha256`), so a file changed
  between a test's first run and its resume makes the resume another job, and the failure looks
  like a bug in resume.
- Tests that need the model weights read their directory from `SEEDVR2X_MODEL_DIR`, and skip
  without it. It holds numz's files and seedvr2x's own side by side (links will do): the GPU
  tests run numz's 7B fp16 and VAE, the weights their references were made with, and the
  regression seedvr2x's own as well, and the shot detector's, `transnetv2.safetensors`. With it,
  the CPU suite also checks every known model file there by its header
  (`tests/test_weights.py`), and runs the shot detector on TransNetV2's weights
  (`tests/test_detector.py`: the cuts of a synthetic clip, and measurement's own run reproduced
  bit for bit; `tests/test_detection.py`: the first pass recording the module's probabilities
  bit for bit); without, the detector's windows run on the untrained model's weights, written by
  the test.
- `tests/test_pull.py` holds the pull of seedvr2x's own files (`runtime/pull.py`) on the CPU,
  huggingface_hub's two calls replaced, and the pins to the revision itself: `SHA256SUMS`, a few
  kilobytes, downloaded into a temporary `HF_HOME` behind a link, and the sizes read from the
  Hub's API, skipped offline (no network to huggingface.co, or `HF_HUB_OFFLINE` set). A test
  that runs seedvr2x in a process of its own gives it its own `HF_HOME` and `HF_HUB_OFFLINE=1`,
  but that one; hf_xet is imported in such a process only. Its runs of the CLI give an empty cut
  list, so that the shot detector, which it replaces, neither fetches nor checks its file, but
  where a test holds that pull. With `SEEDVR2X_TESTS_PULL=1`, it pulls the four v1 files into
  Hugging Face's cache too, about 33 GB the first time; `test_regression.py`'s
  `test_milestone1_from_the_cache` (GPU) then runs the default models from the cache,
  `HF_HUB_OFFLINE=1`, and skips until the three it takes are there, saying how to fetch them
  alone (`uv run hf download`, 17 GB).
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
  each compared with the plan the planner makes for the free memory that call recorded itself
  (`compare_with_planner`; the `--plan` call's reading from its manifest, the run's from its
  `--verbose` log), never with each other: a desktop that drives the card moves its free memory
  between them, and a plan that follows it is right. `test_plan_and_run_read_the_same_free_memory`
  holds the two readings to one figure, back to back, the run stopped once it has planned, and
  skips, saying the figures, where the other programs' memory on the card moved between them. Its
  name lacks `milestone1`: a gate that selects by `-k milestone1` doesn't run it, so the gate's
  hold names it. `tests/test_regression_unpinned.py` holds the helpers on the CPU, on synthetic
  inputs: a reading found in a log or a manifest, the planner's plan rebuilt for it, the
  comparison and its failure message (each call's reading, windows and blocks), the run stopped.
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
- The frame index and the reads through it (`media/index.py`, `media/reader.py`) are held by
  `tests/test_index.py` to full decodes of files it makes, the kinds of
  `../research/scripts/seek_test.py`'s `synth` and three of `seeking.md`'s failure mechanisms
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
- The cuts drawn from the probabilities (`runtime/cuts.py`) are held by `tests/test_cuts.py` to
  measurement's own detection (`../research/scripts/scd_scores.py`'s `tnet_detections`, copied
  there, on the float32 values read as float64, as its analysis reads them), the possible cuts
  and the cut list `--plan` writes with them, read back by `--cuts`. The CLI's
  (`--cut-threshold`, `--plan`, a plan planned again or taken up in another environment) by
  `tests/test_cli_run.py`, a `ScriptedDetector` giving chosen probabilities in place of the
  stand-in's, about 0.25 on its clips, which cut nothing at 0.3; with `SEEDVR2X_MODEL_DIR`,
  TransNetV2 itself cuts `tests/test_detector.py`'s synthetic clip at its cuts through the CLI.
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
- Every ffmpeg and ffprobe seedvr2x starts ends with it (`media/ffmpeg.py`'s `ends_with_us`),
  held for all of them by `tests/test_ends_with_us.py`: a static scan of `src/`, a function
  (`scan`) held on its own to synthetic sources, one proof per rule (an aliased `subprocess` or
  `os`, a name imported from either, `os`'s starters by family, `spawn*`, `exec*`, `posix_spawn*`,
  `fork*`, `system` and `popen`), and one counting the hooked and unhooked starts and leaving plain
  uses alone; on `src/` it finds no violation, 16 starts with the hook (a new one is counted on
  purpose) and one without, the segment command's shell; `prctl` resolved in the parent before
  any fork, the hook holding the typed C function and no `CDLL` (a lookup in the child would
  allocate); a child process per kind of site (the startup check, probes, fingerprint, packet count
  and the rest) with a fake ffmpeg and ffprobe on `PATH` that only wait, SIGKILLed alone, and its
  control with the hook replaced by none in that site's module, where they stay, the reader's and
  the writer's real ffmpeg (skipped without a build seedvr2x accepts), and the segment command's
  shell, left running by a kill, as DESIGN.md has it. `/proc` tells the processes left, so Linux
  only.
- The one clock (`clock.py`): `tests/test_clock.py` holds it on fake monotonic and wall readers,
  nothing sleeping (`Readers`, `Pace`): the ratio of the wall to the run's clock before and after
  its minimum span, steps of the wall told from drift (a few seconds either way, an hour set, a
  VM asleep, a long unit), the band outside which a ratio is not used, and every duration
  printed (time left, the end, `--until`, time spent, the assembly's reading) said in the wall's
  seconds. A span said before the ratio is measured is the wall's own seconds over it, from the
  readings taken at each line logged (`cli._sampled`, the handler's filter, tested where
  `cli._logging` installs it, with and without `--verbose`); a step of the wall inside a span, or
  wall seconds outside 0.8 to 1.25 of the span's own, leave the run's own seconds, said at debug.
  `tests/conftest.py`'s autouse fixture `one_to_one_clock` replaces the clock's readers by a
  frozen pair, so that every other test sees a ratio of 1.0: a long session's real drift would
  leak into their expected durations.
- `tests/test_disk.py` holds the disk a job takes beside its output (`runtime/disk.py`): the
  estimate of the masters at 1080p, at 4K and at an odd size, from the segments a resume has
  written, a segment command's unknown and then measured, the room free to a user, and the line
  among `--plan`'s and a run's warning, which never refuses.
- `--compile` is held on the CPU by `tests/test_compile.py` (`runtime/compiling.py`, the
  planner's compiled passes): the plan holding a compiled pass to 1.1 times its tile's plain peak
  and to the measured range of shapes, the shapes warmed (the decoder's latent in one layout, the
  DiT's, dense channels last, for the run's decodes and the calibration's: the run's own calls,
  shots of 1 to 64 latents, are compared with the warm-up's), the order of the load with fakes (the
  VAE compiled before the DiT loads, the compile workers shut before its blocks), the VAE's
  exclusion of the causal convolutions from the graph, inductor's deterministic mode kept on, the
  flag recorded and a resume with the other value another job, the cache in the job's own
  directory, and a compilation after the warm-up said at warning. A job outside the shapes
  measured (a pin beyond them, a frame too small for a tile, an untiled decode) is refused, never
  run plain, the refusal naming the pass, its tiles and the shapes, then its advice; a plan with
  no unit made is planned again for the other value of `--compile` without a flag, and units made
  make it another job's settings; inductor's modules count in the environment's comparison only
  between two compiled jobs. The plan's compile line gives the time by kind of job (new, or a
  resume with the cache warm) and by disk: a Windows drive is found in `/proc/self/mountinfo`, the
  longest mount point deciding, a 9p mount counting only with `aname=drvfs`. Compiled for real
  only on a tiny module, in a subprocess so that no state of inductor stays in the suite (two
  tests, skipped without a C++ compiler); the VAE's compile on a GPU is not in the default suite.
- The default output (DESIGN.md, Architecture, What it prints): `tests/test_cli.py` holds
  `TORCH_CPP_LOG_LEVEL` set before torch is imported (seen by a finder at each import, in a
  subprocess) and the user's own value winning, and the early reading of the card in a
  subprocess that must import no torch; `tests/test_model.py` the sampler's progress bars off
  but under `--verbose`, and a download's kept; `tests/test_cli_run.py` the lines saying the date
  or the zone's offset changed, and what a user does nothing with (the conversions' fingerprint,
  the assembly's rate and count) at debug; `tests/test_timing.py` the calibration's working at
  debug and a duration under a second said so, its line saying what it is for;
  `tests/test_assembly.py` and `tests/test_assembled.py` the rehearsal's line saying the same.
  The card line leaves the driver's figure where NVML's is the smaller to a debug record beside
  it (`tests/test_planner.py`; `tests/test_cli_run.py`'s
  `test_the_drivers_figure_is_said_under_verbose`).
  The spilling driver's two warnings are written out whole in `tests/test_memory.py`
  (`LENDS_LINE`, `HOLDS_LINE`), not built from `memory`'s constants, so that a change of wording
  fails the tests that compare them.
- The calibration's measured lengths (`runtime/calibration.py`): `tests/test_timing.py`'s
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
- A resume short of the memory free is held by `tests/test_planner.py` (the shortfall and the
  phase a refusal names, a recorded shape planned again releasing that part, planning importing
  no torch), `tests/test_memory.py` (the early NVML reading, off this process's own context),
  `tests/test_units.py` and `tests/test_timing.py` (each run's windows decided, kept or made
  again, from the layouts the manifest records) and `tests/test_cli_run.py`: the refusal naming
  `--accept-env-change`, the warning given before the checks, the re-plan under the flag
  (`planner.PER_TOKEN` multiplied, so that the DiT phase refuses at the stand-in's size), the
  segments and latents kept, `--plan` looking and writing nothing.
- Met running the suite from a worktree: `git read-tree -u --reset` empties the `upstream/`
  submodule directories (link them again); `--compile`'s cache on a Windows drive under WSL2
  costs minutes, so compile times are measured with the job on Linux's disk; `--until` takes
  a time of day to the minute, so a test that stops a run inside a shot needs a shot of minutes.

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
