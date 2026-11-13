"""Milestone 1 as a regression test: seedvr2x's output stays bit-identical to its own references,
masters and float32 frames, by core/tools/compare.py, in seedvr2x's defaults (its own padding,
DESIGN.md, Pipeline step 0; its tail repeating a shot's last frame, Pipeline step 1; its noise
drawn on the CPU, Pipeline step 2). And the VAE tiling (DESIGN.md, Pipeline step 3): seedvr2x
tiled, to its own tiled references.

The references were made on 2026-11-04 on the RTX 5090 by main 9a18423, in one GPU session: first
this test as it stood there, its 12 cases held to numz's references bit for bit (on that card,
seedvr2x's frames in numz's modes, which numz 4490bd1, run there on milestone 1's case, had
matched on 2026-10-22), the last comparison with numz (DESIGN.md, Vendored model code, "seedvr2x's
own model since 2026-11-04"); then each case's command (command()) in seedvr2x's defaults, its
master and frames kept as the case's reference.
The references hold seedvr2x's output from then on: a change that moves its bits is allowed with
a measured reason, and makes them again, said in its commit.

Those were made under FlashAttention 2. When it went (2026-11-08: PyTorch's SDPA the one attention
kernel, DESIGN.md, Vendored model code), they were made again under SDPA on the RTX 5090 by main
666a6cb, each case's command in an environment where FlashAttention 2 doesn't import, at the
card's sizes of cases.json, and this test held its every case to them on that code, then on the
code that removed FlashAttention 2. SDPA's are the only references since; the FlashAttention 2
ones are kept aside, under a name that says so, unused. The 96 GB card's (RTX PRO 6000 Blackwell, on
Linux's driver) were made under SDPA on 2026-11-10 (impl-box-1), as the RTX 5090's were.

Until 2026-11-04 the references were numz's, and seedvr2x ran in three modes for the tests only to
reproduce them: numz's padding, numz's mirrored tail and numz's draw of the noise on the GPU.
Those modes are gone, and so is numz's own output as a reference; the cases' input and their
names are the same.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and the references, in
SEEDVR2X_REFERENCE_DIR: the inputs in m1/, and seedvr2x's references in seedvr2x/<card>/, <card>
being the name torch gives the card (torch.cuda.get_device_name) with each run of characters
other than letters and digits a hyphen (seedvr2x/NVIDIA-GeForce-RTX-5090/). Each card holds to
references of its own (decided on 2026-10-21): the noise is the same on every card (Pipeline
step 2), but the kernels the GPU picks are not, so that another card's frames differ from a
card's in the last bits. Windows' own, under a platform level, seedvr2x/windows/<card>/
(decided on 2026-11-09, DESIGN.md, Allocator: a picture made natively differed from the same run
under WSL2 on the same card, 54 to 57 dB PSNR, and Windows repeats its own; torch's Windows wheel
has no FlashAttention backend in SDPA, so another kernel runs there, inferred and not isolated),
made natively as below; Linux's,
WSL2's among them, stay at seedvr2x/<card>/ (platform_level). A card with no directory there has
every case skipped, naming the card.
A card with one runs every case, and a reference missing fails it. Inputs, in m1/:
- input_rgb.mkv: 45 frames of 8-bit RGB FFV1 at 960x540, read alike by numz (OpenCV) and by
  seedvr2x: frames 48-92 of SEGMENT, a 1920x1080 yuv420p FFV1 segment of an animated episode
  (limited range, chroma left, no colour tags, 23.976 fps, 377 frames; its first 26 black, then a
  fade-in to about frame 50), BT.709 limited range to full-range RGB, its chroma upsampled and
  the picture brought to 960x540 by zimg's spline36, with ffmpeg n9.0.2:

    VF=trim=start_frame=48:end_frame=93,setpts=PTS-STARTPTS
    VF=$VF,zscale=w=960:h=540:filter=spline36:matrixin=709:transferin=709:primariesin=709
    VF=$VF:rangein=limited:transfer=709:primaries=709:range=full:dither=none,format=gbrp
    ffmpeg -nostdin -y -i SEGMENT -map 0:v:0 -vf "$VF" -fps_mode passthrough \
      -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 -an -sn input_rgb.mkv

  Remade so on 2026-10-09: the same frames and packets, the two files differing only in the
  Matroska muxer's random UIDs.
- input_rgb_956x530.mkv, input_rgb.mkv's top left 956x530, cropped losslessly:

    ffmpeg -i input_rgb.mkv -vf crop=956:530:0:0 -c:v ffv1 -pix_fmt bgr0 input_rgb_956x530.mkv

numz's own references (numz.mkv, numz_sharp.mkv, numz_pad*.mkv, numz_padcols.mkv and the tiled
ones, made on numz 4490bd1 through ffv1_out.py, in m1/, m3/ and cards/) stay on disk, unused.

Each reference is <name>.mkv, the master (FFV1, gbrp16le: --format gbrp16le), and <name>_frames/,
the float32 frames (--dump-frames), as core/tools/compare.py reads them. Each case is a run of its
own, both model files named, in seedvr2x's defaults, with --color-correction none, --seed 42 and
the tiles pinned (untiled unless the case pins them):
- 7b: seedvr2x_ema_7b_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors, the same values as
  numz's seedvr2_ema_7b_fp16.safetensors and ema_vae_fp16.safetensors (DESIGN.md, Weights), so
  that numz-7b and ours-7b, the two cases, are held to the one reference, bit for bit;
- sharp: seedvr2x_ema_7b_sharp_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors (the cases
  ours-sharp and test_milestone1_from_the_cache);
- sharp_dyn: seedvr2x_ema_7b_sharp_dyn.gguf, the sharp 7B's dynamic GGUF, and
  seedvr2x_ema_vae_fp16.safetensors (the case ours-sharp-gguf), made on 2026-11-04 by the code
  that brought the GGUF in (impl-gguf), not by main 9a18423: numz never ran it. The reference
  only holds that output still (a regression guard, made by the code it guards). How far it is
  from float16 is not held here: DESIGN.md's claim, the output within the model card's figures
  for the file, is measured on the RTX 5090 by a job at 1080p (impl-gguf's GPU window), against
  models/VALIDATION.md ("1080p, the sharp 7B": the file's RGB PSNR to the sharp float16's output
  at seed 42, tier 1's 8 clips, is 42.48 dB on average and 36.36 dB at the worst clip).
Every run detects its shots, as a user's does without --cuts (DESIGN.md, Shot detection): the
shot detector runs on the GPU in the first pass, before the models load, so that its settings,
scoped to its own forwards, and the memory it gives back are held to the bits too. At
--cut-threshold 0.5, where its cuts are none, so that the job is one shot, which the log must
say. At the default, 0.3, it isn't: input_rgb.mkv ends on two short shots, TransNetV2 peaking at
0.3242 on frame 41 and 0.3880 on frame 43, and at 0.1171 on frame 0, on the CPU and on the GPU
alike (1.19e-07 apart; the same on its 956x530 crop): cut at 42 and 44, 3 shots, where at 0.5 it
is 1 shot and 2 possible cuts (the GPU box, 2026-10-10). A case whose model file, or the shot
detector's (transnetv2.safetensors), isn't in SEEDVR2X_MODEL_DIR is skipped, naming the file.
One more, test_milestone1_from_the_cache, runs sharp's case with the default models taken from
Hugging Face's cache (seedvr2x/pull.py): no --model-dir, no --dit-model or --vae-model, and
HF_HUB_OFFLINE=1 in its environment, so that only the cache can serve them; it needs no
SEEDVR2X_MODEL_DIR, and is skipped, saying how to fill the cache (hf download, the three files,
17 GB), unless the run would find them there: at the pinned revision, or downloaded at another
(seedvr2x/pull.py).

Every case pins the passes its reference was made with, untiled or in its tiles, but one,
test_milestone1_unpinned: the 7b case with no tile pinned, so that a bit comparison goes through
the planner's own choice of passes (DESIGN.md, Memory planner, Choices, 3). The plan is made
first, by the same command with --plan, which runs the first pass and plans, loading no model
today (DESIGN.md has --plan load them for the time estimates' calibration, which isn't built): its
lines of the passes, "plan: encode: untiled, ..." and "plan: decode: untiled, ...", must say both
untiled, as the reference ran them; the run then makes a plan of its own (each call's held to the
planner's for the free memory that call read, not to the other's: the desktop moves it), and the
master and frames are the reference's bit for bit. A card that can't run untiled can't make the
untiled master (the untiled decode of these frames, padded to 1920x1104, takes 37.31 GiB of free
VRAM by the planner, the DiT off the GPU, its room and the margin counted; asked of the planner
with --decode-tile 0 on 2026-11-04): where the driver reports under 40 GiB free (UNTILED_FREE), a
plan that tiles a pass, or is refused, skips the case before any run, saying that line or the
refusal. With 40 GiB or more reported, such a plan fails it: the plan is wrong there, or what the
log says of the free memory no longer reads (reported).

seedvr2x's own padding (at least 8 rows reflected from the picture, up to a multiple of 16, then
16 black rows; columns alike when the width isn't a multiple of 16) has its three cases on the
7b files, each with the sizes numz logs (src/core/generation_utils.py:274-279) and seedvr2x's
--resolution gives (tests/test_padding.py):
- pad1080: input_rgb.mkv at --resolution 1080, 1920x1080 padded to 1920x1104 (8 rows reflected,
  no column);
- pad720: input_rgb.mkv at --resolution 720, 1280x720 padded to 1280x752 (16 rows reflected);
- padcols: input_rgb_956x530.mkv at --resolution 1060, 1912x1060 padded to 1936x1088 (12 rows and
  8 columns reflected).
The milestone 1 cases themselves run at 1080 in the same padding, and a card's cases.json may
name another size for them.

The VAE tiling (seedvr2x/tiling.py) is held on milestone 1's input in the 7b files: seedvr2x with
its tiles pinned, the whole shot under each tile, with the 64 px overlap seedvr2x fixes, which
was numz's tiling bit for bit when the references were numz's (DESIGN.md, Pipeline step 3):
- d1024: --decode-tile 1024: on the frames padded to 1920x1104, 4 tiles, 2 rows of 2;
- e1344-d1024: --encode-tile 1344 --decode-tile 1024: the encode in 2 tiles, one row;
- e512-d512: --encode-tile 512 --decode-tile 512: 15 tiles each way, 3 rows of 5.
Each run's log must say those grids (the debug line of seedvr2x/tiling.py, with -v).

A card's directory holds cases.json, which says what each case takes on that card:
{"card": NAME, "made": {...}, "milestone1": {"resolution": R}, "padding": {"pad1080":
{"resolution": R}, ...}, "tiled": {"d1024": {"resolution": R, "grids": [...], "over": "..."},
...}}. "made" is free-form: the commit and code subset that made the references, the date, and what
checked them. "milestone1" gives the resolution for all its cases, the unpinned one and the one
from the cache included, which run untiled at the largest size the card's untiled passes fit;
"padding" and "tiled" give each case's by its reference name; a tiled case at another size than
1080 says what its debug log says of each tiled pass there: the start of each grid's line
("grids") and the end every one shares ("over"). A key absent is the test's own (1080 and the
grids above, over 45 frames padded to 1920x1104).

To make a card's references again, or for a new card: run each case's command (command() gives it:
the options above, in seedvr2x's defaults, with --dump-frames and the master's output) at the
card's sizes, on a commit whose regression passed on a card that has references, and move each
run's ours.mkv and frames/ to <name>.mkv and <name>_frames/ in the card's directory; write
cases.json. Windows' natively, the run's own platform giving the directory (platform_level).
A gate's case keeps 2 GiB of the card's budget (DESIGN.md, Validation, decided on 2026-10-21):
each case at a size whose plan beside a second program holding 2 GiB of the card is neither
refused nor planned with the reserve of 2.5 GiB, its window and tiles the same. A change
that moves the output's bits is allowed with a measured reason, which remakes the references, said
in its commit (DESIGN.md). A case refused at its run for the memory it takes is skipped, with the
refusal's figures (upscaled), and one without its reference where the card can't run it likewise
(no_reference); a case planned with the reserve of 2.5 GiB fails (upscaled), the card no longer
leaving it the room its reference was made in.

The run goes through the CLI in a process of its own, as a user's would: the allocator is set
before torch is imported, which a test process importing torch first wouldn't do. And this
process creates no CUDA context: its memory would be taken from the free memory a run reads for
its plan (cli._free_memory).
"""

import functools
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

import pytest

import disk
import planner
from manifest import NAME
from seedvr2x import memory, peaks, pull
from seedvr2x.frames import target_size
from seedvr2x.shots import Shot
from seedvr2x.weights import Architecture

# core/tools/compare.py: the repository is three levels up (apps/seedvr2x/tests/).
COMPARE = Path(__file__).resolve().parents[3] / "core" / "tools" / "compare.py"
# The installed command, the script beside the interpreter.
COMMAND = str(Path(sys.executable).with_name("seedvr2x"))
MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not REFERENCE, reason="needs SEEDVR2X_REFERENCE_DIR"),
]

# numz's model files: the same values as seedvr2x's own (DESIGN.md, Weights), so that the 7B
# fp16 and the VAE of either give the references' bits.
NUMZ_DIT, NUMZ_VAE = "seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors"
# The command line's defaults, the sharp 7B and the VAE (main.py).
DEFAULT_DIT, DEFAULT_VAE = (
    "seedvr2x_ema_7b_sharp_fp16.safetensors",
    "seedvr2x_ema_vae_fp16.safetensors",
)


def models_present(*names: str) -> None:
    """Skip, naming the file, unless SEEDVR2X_MODEL_DIR holds each of names, and the shot
    detector's file, which every run here takes (the module's docstring)."""
    if MODELS is None:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    for name in (*names, pull.DETECTOR):
        if not (Path(MODELS) / name).is_file():
            pytest.skip(f"{name}: not in SEEDVR2X_MODEL_DIR")


# The directory of SEEDVR2X_REFERENCE_DIR that holds seedvr2x's own references, one directory a
# card (card_directory), and the file in it that says what each case takes on that card.
OWN = "seedvr2x"
CASES = "cases.json"


@functools.cache
def card() -> str:
    """The name of the card the runs take, device 0, as torch gives it, asked in a process of its
    own: this one creates no CUDA context (the module's docstring)."""
    found = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 0, found.stderr[-3000:]
    return found.stdout.strip().splitlines()[-1]


def platform_level() -> tuple[str, ...]:
    """The level of OWN that holds this platform's cards: none on Linux, `windows` on Windows
    (the module's docstring)."""
    return ("windows",) if sys.platform == "win32" else ()


def card_directory(name: str) -> Path:
    """Where the references of the card `name` are: OWN/<name>/ in SEEDVR2X_REFERENCE_DIR on
    Linux, OWN/windows/<name>/ on Windows (platform_level), its name's runs of other characters
    than letters and digits each a hyphen ("NVIDIA GeForce RTX 5090":
    seedvr2x/NVIDIA-GeForce-RTX-5090/, seedvr2x/windows/NVIDIA-GeForce-RTX-5090/)."""
    assert REFERENCE is not None
    card = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return Path(REFERENCE, OWN, *platform_level(), card)


@dataclass(frozen=True)
class Case:
    """A case as this card runs it: its input, the resolution it is upscaled to, and its reference,
    <reference>.mkv and <reference>_frames/ in the card's directory; for a tiled case, what its
    debug log says of each tiled pass at the card's size where that isn't the test's own, the start
    of each grid's line and the end every one shares (grids, over), else None."""

    source: Path
    resolution: int
    reference: Path
    grids: tuple[str, ...] | None = None
    over: str | None = None


def case(kind: str, source: str, resolution: int, reference: str) -> Case:
    """The case of `kind` (milestone1, padding or tiled) on this card (card()), skipped where the
    card has no directory of references (card_directory), naming the card. The card's CASES says
    what the case takes there (the module's docstring): milestone1's resolution, for all its
    cases; each padding case's resolution, and each tiled case's, with its tiled passes' "grids"
    and what they are "over", by the case's reference name; the parameters for what it doesn't
    say. The inputs are the same on every card: m1/ of SEEDVR2X_REFERENCE_DIR."""
    assert REFERENCE is not None
    directory = card_directory(card())
    if not directory.is_dir():
        pytest.skip(
            f"{card()}: no references of seedvr2x's own for this card ({OWN}/{directory.name}/);"
            " this module's docstring says how to make them"
        )
    said: Any = {}
    if (directory / CASES).is_file():
        said = json.loads((directory / CASES).read_text()).get(kind, {})
    entry: Any = said if kind == "milestone1" else said.get(reference, {})
    grids = entry.get("grids")
    return Case(
        Path(REFERENCE) / "m1" / source,
        int(entry.get("resolution", resolution)),
        directory / reference,
        None if grids is None else tuple(str(grid) for grid in grids),
        entry.get("over"),
    )


def refusal(log: str) -> str | None:
    """The refusal of the planner's for the memory a job takes (REFUSED) in a run's log, its
    error's message, or None: not the disk's (cli._room_on_disk), whose scratch's part names the
    system memory too, and whose way out is the machine's, not the card's (a review's finding).
    A line without --verbose is "12:00:00 ERROR message"; under it, "... ERROR cli.main:
    message" (cli.VERBOSE_FORMAT)."""
    for line in log.splitlines():
        found = re.match(r"(?:\d\d:\d\d:\d\d|.* ERROR [\w.]+:) ?(?:ERROR )?(.*)", line)
        if found and " ERROR " in f" {line} " and any(what in found[1] for what in REFUSED):
            if "free on its disk" not in found[1]:
                return found[1]
    return None


def no_reference(tmp_path: Path, job: Case, models: tuple[str, str], *more: str) -> NoReturn:
    """A case this card has no reference for, on a card that has references: skipped with the
    refusal's figures where the card can't run it, its --plan refused (the planner's own words),
    else failed, naming the reference missing."""
    missing = f"{job.reference}.mkv"
    arguments, environment = command(*(tmp_path, job.source, job.resolution, models, *more))
    plan = subprocess.run(
        [*arguments, "--plan", str(tmp_path / "cuts.txt")],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    refused = refusal(plan.stderr)
    if plan.returncode == 1 and refused is not None:
        pytest.skip(f"{card()}: the case can't run there: {refused}")
    pytest.fail(f"{card()}: no reference for it, {missing}; this module's docstring says how")


def referenced(tmp_path: Path, job: Case, models: tuple[str, str], *more: str) -> None:
    """Go on with a case only where its input is there (skipped, naming it) and its reference
    (no_reference): a card that has references and lacks one fails the case, but where it can't
    run it."""
    if not job.source.exists():
        pytest.skip(f"{job.source}: absent; this module's docstring says how to make it")
    for path in (
        job.reference.parent / f"{job.reference.name}.mkv",
        job.reference.parent / f"{job.reference.name}_frames",
    ):
        if not path.exists():
            no_reference(tmp_path, job, models, *more)


# The tile pins of a run (main.py): one not given is pinned untiled (upscaled), but where a case
# leaves its passes to the planner.
TILE_PINS = ("--encode-tile", "--decode-tile")


def command(
    tmp_path: Path,
    source: Path,
    resolution: int,
    models: tuple[str, str] | None,
    *more: str,
    pin_untiled: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """seedvr2x's command line on source at resolution with the DiT and VAE files of models in
    SEEDVR2X_MODEL_DIR, or with None, its default models from Hugging Face's cache, offline
    (HF_HUB_OFFLINE=1), in seedvr2x's defaults, with the options of `more` besides, to tmp_path's
    ours.mkv and frames/ (same_bits); and the environment to run it in. A pass that `more` doesn't
    pin is pinned untiled, unless pin_untiled is False, which leaves it to the planner."""
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    environment = dict(os.environ)
    if models is None:
        options = []
        environment["HF_HUB_OFFLINE"] = "1"
    else:
        assert MODELS is not None
        options = ["--model-dir", MODELS, "--dit-model", models[0], "--vae-model", models[1]]
    arguments = [
        *(COMMAND, str(source), "-o", str(master), *options),
        *("--resolution", str(resolution), "--seed", "42", "--dump-frames", str(frames)),
        # The shots detected, and none found: at the default 0.3, the input is cut at 42 and
        # 44, 3 shots where the references are one (the module's docstring).
        *("--cut-threshold", "0.5"),
        # The references are made without a colour correction, so that they hold the model's
        # output alone.
        *("--color-correction", "none"),
        # The references' format, gbrp16le: compare.py reads both masters as stored.
        *("--format", "gbrp16le"),
        # The passes the reference was made with, pinned: untiled unless `more` pins tiles, so
        # that a card too small for them refuses the job (DESIGN.md, Memory planner, Pins)
        # where the planner would tile what the reference ran untiled, bits that aren't its.
        # None pinned for the case that leaves its passes to the planner, and looks at what it
        # chose (test_milestone1_unpinned).
        *(
            pin
            for option in TILE_PINS
            if pin_untiled and option not in more
            for pin in (option, "0")
        ),
        *more,
    ]
    return arguments, environment


def upscaled(
    tmp_path: Path,
    source: Path,
    resolution: int,
    models: tuple[str, str] | None,
    *more: str,
    pin_untiled: bool = True,
    planned: str | None = None,
) -> str:
    """Run seedvr2x as command has it, to tmp_path's ours.mkv and frames/ (same_bits), its first
    pass read (detected) in the log of the call that ran it: the run's own, or `planned`, the log
    of the same command's --plan call before it, where the run says it resumes a job. Returns the
    run's log."""
    arguments, environment = command(
        *(tmp_path, source, resolution, models, *more), pin_untiled=pin_untiled
    )
    run = subprocess.run(arguments, capture_output=True, text=True, check=False, env=environment)
    # A card that can't run the case: skipped with the refusal's figures (decided on 2026-10-21).
    refused = refusal(run.stderr)
    if run.returncode == 1 and refused is not None:
        pytest.skip(f"{card()}: the case can't run there: {refused}")
    assert run.returncode == 0, run.stderr[-3000:]
    shared_card(run.stderr)
    # The plan's reserve is the step's own (peaks.reserve; DESIGN.md, Allocator, Small cards,
    # 2026-11-06): the 2.5 GiB reserve and its second line, "reserve: ... slow", retired, so no
    # plan line of a case says a slower reserve was taken.
    slow = [line for line in plan_lines(run.stderr) if line.startswith("reserve: ")]
    assert not slow, f"{card()}: the case was planned with a slower reserve: {slow[0]}"
    # ours.mkv is made in ours.mkv.work, a directory's job, and --plan makes that job with its
    # first pass (cli._run): the run after it resumes the job, and scores no frame again. Without
    # a --plan call, or where the run doesn't say it resumes, the first pass is the run's own.
    # (A line without --verbose starts with the time of day, "12:00:00 resuming ..."; under it, in
    # the logger's format, "... INFO cli.main: resuming ...": cli.VERBOSE_FORMAT.)
    resumes = re.search(r"^(\d\d:\d\d:\d\d|.* INFO cli\.main:) resuming ", run.stderr, re.M)
    detected(planned if planned is not None and resumes is not None else run.stderr)
    return run.stderr


def shared_card(log: str) -> None:
    """A run held for another program on its card under Linux's driver (memory.SHARED; DESIGN.md,
    Allocator, decided on 2026-11-11) fails the case: the references under Linux's driver were made
    on a card nothing else used, a refusing driver's plans, and a held run plans otherwise. The
    other program may be this pytest process, whose in-process GPU tests (test_clip_cli.py,
    test_still_cli.py, test_shot_cli.py) leave it a CUDA context on the card: run this file in a
    pytest process of its own, nothing else on the card."""
    said = next((line for line in log.splitlines() if memory.SHARED in line), None)
    assert said is None, (
        f"{card()}: the run was held, its card shared ({said}): run tests/test_regression.py in a"
        " pytest process of its own, with no desktop or other program on the card"
    )


def detected(log: str) -> None:
    """A run's first pass, by its log: the shot detector ran, on the GPU, and gave its memory
    back; no cut at 0.5, the input one shot (the module's docstring)."""
    assert "the shot detector scored" in log, log[-3000:]
    assert "shot detector: VRAM peak" in log, log[-3000:]
    assert "1 shot: 0 cuts detected at --cut-threshold 0.5" in log, log[-3000:]


def errors_of(log: str) -> list[str]:
    """The messages of a log's errors, as the command logs them: without --verbose a line is
    "12:00:00 ERROR message"; under it, in the logger's format, "2026-10-31 12:00:00,123 ERROR
    cli.main: message" (cli.VERBOSE_FORMAT)."""
    return [
        found[1] or found[2]
        for line in log.splitlines()
        if (
            found := re.match(
                r"(?:\d\d:\d\d:\d\d ERROR (.*)"
                r"|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} ERROR [\w.]+: (.*))",
                line,
            )
        )
    ]


def plan_lines(log: str) -> list[str]:
    """The plan's lines in a run's log, as planner.said gives them, without their "plan: " (their
    time of day, or under --verbose their date and logger, dropped with it)."""
    return [line.split("plan: ", 1)[1] for line in log.splitlines() if "plan: " in line]


def choices(lines: list[str]) -> list[str]:
    """A plan's lines (plan_lines) without what follows the free memory read that time: the
    card's line and, on a run held to the card, its budget's, and how far under the budget the
    plan is, the end of the peaks' line. What is left is the plan: its windows, where the weights
    are, its passes, its predicted peaks and its pins."""
    peaks = "predicted peaks: "
    return [
        line.split(";", 1)[0] if line.startswith(peaks) else line
        for line in lines
        if not line.startswith(("card: ", "budget: "))
    ]


# What every refusal of the planner's for the memory a job takes says (shot_plan.PlanError): the
# free VRAM it takes, or the system memory.
REFUSED = (" GiB of free VRAM", " GiB of system memory")

# The free memory, in GiB as the driver reports it, from which test_milestone1_unpinned holds a
# card to its untiled passes: the plan takes 37.31 GiB of free VRAM for them (the module's
# docstring), and a card of 48 GB reports 46.5 or more (cli._free_memory: N - 1.49 on a card
# emulated at N GB).
UNTILED_FREE = 40.0

# What the driver reported free to a run, in the words of the plan's first line and of its
# refusals, the first that is there (reported): on a run held to the card, the driver's own
# figure, said beside the whole card's (at debug, under --verbose: planner.card_detail) or in
# its place; on a driver that refuses, the one figure
# a refusal gives, or the plan's first line (planner._card_said, shot_plan._there).
REPORTED = (
    r"driver reports ([\d.]+) GiB free to this process",
    r"([\d.]+) GiB free as its driver reports it to this process",
    r"the driver reports ([\d.]+) GiB free",
    r"plan: [^\n]*, ([\d.]+) GiB free, a budget of ",
)


def reported(log: str) -> float | None:
    """What the driver reported free to the run whose log this is, in GiB, as its plan's first
    line or its refusal says it (REPORTED); None where neither does, a refusal for the system
    memory alone."""
    for pattern in REPORTED:
        said = re.search(pattern, log)
        if said is not None:
            return float(said[1])
    return None


def unpinned(log: str, why: str, free: float | None = None) -> None:
    """The unpinned case on a card whose plan can't make the untiled master, `why` saying
    it, the refusal or the plan's line of its passes: skipped where the driver reported under
    UNTILED_FREE GiB free to the --plan run whose log this is (reported), failed where it
    reported that or more, or where the log doesn't say. `free`, in GiB, is that figure where
    the caller has it from the call's own record (Free), not from the log's words."""
    free = reported(log) if free is None else free
    if free is not None and free < UNTILED_FREE:
        pytest.skip(
            f"the driver reports {free:.2f} GiB free, under the {UNTILED_FREE:.0f} a card makes"
            f" the untiled master from: {why}"
        )
    there = (
        "what the driver reports free isn't in the log"
        if free is None
        else (f"the driver reports {free:.2f} GiB free, {UNTILED_FREE:.0f} or more")
    )
    pytest.fail(f"{there}, where the plan must run both passes untiled: {why}")


@dataclass(frozen=True)
class Free:
    """The free memory a call's plan was made in, as that call read it, in bytes: what the driver
    reported to the process, what the whole card had by NVML (None where it was not read), and
    whether the run was held to the card (memory.Reading). `exact`: read from the job's manifest,
    to the byte; else from a log, which says GiB to two decimals (logged_free)."""

    reported: int
    card: int | None
    held: bool
    exact: bool = True

    def said(self, counted: int | None = None) -> str:
        """In words, GiB to two decimals, for a failure's message."""
        card = "not read" if self.card is None else f"{self.card / memory.GIB:.2f} GiB"
        more = "" if counted is None else f", counted {counted / memory.GIB:.2f} GiB"
        return f"reported {self.reported / memory.GIB:.2f} GiB, card {card}{more}"


def recorded_free(record: dict[str, Any]) -> Free:
    """The free memory a plan was made in, from the manifest's plan record, its "memory" part
    (shot_plan.Peaks.record): the driver's figure, the card's, and a reserve only on a held run."""
    return Free(record["driver_free"], record["card_free"], record["reserve"] is not None)


def gib(figure: str) -> int:
    """A log's figure in GiB, to bytes."""
    return round(float(figure) * memory.GIB)


# The line memory.reading logs at debug level, which --verbose shows: what the plan counts on, of
# the free memory it was made under (the smaller of two figures) and which figure that is, with
# the other (memory._named).
PLAN_COUNTS = re.compile(
    r"the plan counts on [\d.]+ GiB of the GPU's memory, of the ([\d.]+) GiB free (.*?):"
    r" [\d.]+ GiB are left to what torch"
)
THE_CARD_SMALLER = re.compile(
    r"on the whole card by NVML \(its driver reports ([\d.]+) GiB free to this process\)"
)
THE_DRIVER_SMALLER = re.compile(
    r"as its driver reports it to this process \(the whole card has ([\d.]+) GiB free by NVML\)"
)
THE_CARD_UNREAD = "as its driver reports it to this process (the whole card's, NVML's, wasn't read)"


def logged_free(log: str) -> Free | None:
    """The free memory a run's plan was made in, from its --verbose log (PLAN_COUNTS; two
    decimals of GiB), the run's manifest being removed with its work directory once the file is
    assembled. On a driver that refuses, memory.reading logs nothing: the plan's card line, "N GiB
    free, a budget of". None where the log says neither."""
    found = PLAN_COUNTS.search(log)
    if found is not None:
        taken, named = gib(found[1]), found[2]
        if card := THE_CARD_SMALLER.fullmatch(named):
            return Free(gib(card[1]), taken, True, exact=False)
        if reported := THE_DRIVER_SMALLER.fullmatch(named):
            return Free(taken, gib(reported[1]), True, exact=False)
        if named == THE_CARD_UNREAD:
            return Free(taken, None, True, exact=False)
        return None
    refuses = re.search(r"plan: card: [^\n]*?, ([\d.]+) GiB free, a budget of ", log)
    return None if refuses is None else Free(gib(refuses[1]), None, False, exact=False)


# The line cli._run logs at debug beside memory.reading's, the system memory available the plan
# was made with (cli._host_available), in GiB and in bytes; or that nothing was read.
HOST_COUNTS = re.compile(
    r"the plan counts on [\d.]+ GiB \((\d+) bytes\) of system memory available"
)
HOST_UNREAD = "the system memory available was not read: the plan counts on none"


def logged_host(log: str) -> int | None:
    """The system memory available a run's plan was made with, exactly, in bytes, from its
    --verbose log (HOST_COUNTS: not two decimals of GiB, which a plan bound by the system memory,
    moving at steps of a few MiB, cannot be rebuilt from), or None where the log says it was not
    read (HOST_UNREAD). Fails where it says neither: a call whose host reading is unknown can't
    have its plan rebuilt."""
    found = HOST_COUNTS.search(log)
    if found is not None:
        return int(found[1])
    if HOST_UNREAD in log:
        return None
    pytest.fail(f"no system memory reading in the call's --verbose log: {log[-3000:]}")


@dataclass(frozen=True)
class Inputs:
    """What a job's plan is made from besides the memory, read from the manifest --plan left
    (recorded_inputs): the target size, the shots and the DiT's format the plan was predicted
    for."""

    target: tuple[int, int]
    shots: tuple[Shot, ...]
    # The disk line the plan is given (disk.Disk.line), for the job's masters; its free space,
    # which compared() drops, is nothing here. None: no disk line (the CPU tests of the rest).
    disk: str | None = None
    # The DiT's format (planner.recorded_dit_model): fp16, the milestone-1 runs', by default.
    dit_model: Architecture = peaks.FP16


def recorded_disk(content: dict[str, Any]) -> str | None:
    """The disk line of a manifest's job as the run says it (cli._disk, disk.reading, Disk.line)
    with nothing free there: its segments and the bytes of those finished, the output's format and
    size, a segment command's files or not, the file assembled (the tests' output is a .mkv).
    None where the content records no output or segments (the CPU tests' synthetic content)."""
    if "output" not in content or "segments" not in content:
        return None
    width, height = content["output"]["size"]
    segments = content["segments"]
    taken = [entry["bytes"] if entry["finished"] else None for entry in segments]
    masters = disk.masters(
        [entry["end"] - entry["start"] for entry in segments],
        taken,
        content["output"]["format"],
        (height, width),
        content["settings"].get("segment_cmd") is not None,
    )
    return disk.Disk(masters, 0, True).line()


def recorded_inputs(content: dict[str, Any]) -> tuple[Inputs, Free]:
    """The inputs of the plan a manifest's content records, with the free memory it was made in:
    the target from the input's size and sample aspect at the setting's resolution (cli._run, job.
    target_size), the shots as recorded, the DiT's format of the plan's
    record (fp16 where it names none, planner.recorded_dit_model), the disk line of its masters
    (recorded_disk)."""
    source, settings = content["input"], content["settings"]
    width, height = source["size"]
    target = target_size(width, height, Fraction(source["sample_aspect"]), settings["resolution"])
    shots = tuple(Shot(shot["start"], shot["end"]) for shot in content["shots"])
    memory_record = content["plan"]["memory"]
    inputs = Inputs(
        target,
        shots,
        recorded_disk(content),
        planner.recorded_dit_model(memory_record),
    )
    return inputs, recorded_free(memory_record)


def readings(free: Free, monkeypatch: pytest.MonkeyPatch) -> list[memory.Reading]:
    """The readings the planner is given for `free`, made by memory.reading itself (its driver
    and NVML replaced by free's figures, restored by monkeypatch): the first for the figures as
    they are; for a log's (not exact), then those that two decimals of GiB leave possible, each
    figure up or down half a hundredth, which the plan must hold within."""
    half = 0 if free.exact else round(0.005 * memory.GIB)
    shifts = (0,) if free.exact else (0, -half, half)
    found: list[memory.Reading] = []
    for by_driver in shifts:
        for by_card in shifts if free.card is not None else (0,):
            card = None if free.card is None else free.card + by_card
            monkeypatch.setattr(memory, "_spills", free.held)
            # The allocator the command runs with by default on this platform (memory.startup):
            # torch's own with expandable segments on Linux; cudaMallocAsync on Windows, whose
            # torch has no segments, each step's reserve then the pool's (impl-win-3, 2026-11-09:
            # Linux's taken there made other tiles than the run's).
            windows = sys.platform == "win32"
            monkeypatch.setattr(memory, "_native", not windows)
            monkeypatch.setattr(memory, "_segments", not windows)
            monkeypatch.setattr(memory, "_footing", None)
            monkeypatch.setattr(memory, "_planned", None)
            monkeypatch.setattr(
                memory, "_twice", lambda index, card=card: "not read" if card is None else card
            )
            found.append(memory.reading(free.reported + by_driver))
    return found


def planner_says(reading: memory.Reading, inputs: Inputs, host: int | None) -> list[str]:
    """The plan lines (plan_lines, choices) the planner makes for `reading` and the job's
    inputs: nothing pinned, the decode splitting no colours (the command's --color-correction
    none), `host` the system memory available, the disk's line given for the
    job's masters (compared keeps that part and drops the free space, which follows the disk). A
    refusal is one line, "refused: ...", which no run's plan equals."""
    try:
        made, peaks = planner.plan(
            reading,
            inputs.target,
            inputs.shots,
            planner.Pins(),
            host,
            dit_model=inputs.dit_model,
        )
    except planner.PlanError as error:
        return [f"refused: {planner.worded(error)}"]
    said = planner.said(
        made,
        peaks,
        "a card",
        inputs.target,
        inputs.shots,
        planner.Pins(),
        disk=inputs.disk,
    )
    return compared(plan_lines("\n".join(said)))


# What a disk line (disk.Disk.line) says of the disk beside the masters: the free space, always
# the line's last part, and the verdict after it. The masters' part before it is the job's.
FREE_ON_DISK = re.compile(r"; [^;]*free on its disk.*$")


def compared(lines: list[str]) -> list[str]:
    """A plan's lines as two plans are compared: choices, the disk's line kept to its masters'
    part (the job's: how many bytes at most, whether the file is assembled), without the free
    space and the verdict after it, which follow the disk (design, baton 4, 2026-10-31)."""
    return [
        FREE_ON_DISK.sub("", line) if line.startswith("disk: ") else line for line in choices(lines)
    ]


def line_of(lines: list[str], start: str) -> str:
    """The first of a plan's lines that starts with `start`, or a note that there is none."""
    return next((line for line in lines if line.startswith(start)), f"(no line {start!r})")


@dataclass(frozen=True)
class Verdict:
    """A call's plan against the planner's for the call's own reading (compare_with_planner)."""

    call: str
    reading: str
    logged: tuple[str, str, str]  # the call's windows, blocks-swapped and disk (masters) lines
    planner: tuple[str, str, str]  # the planner's, for the reading as given
    agrees: bool


def compare_with_planner(
    call: str,
    log: str,
    free: Free,
    inputs: Inputs,
    host: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> Verdict:
    """Whether the plan lines in `log` are those the planner makes for `free`, the reading that
    call planned in: for a reading from a log, for any of the readings its two decimals leave
    possible (readings), and `host`, the system memory available that call planned with
    (logged_host, exact). Not compared with another call's plan: the desktop's memory moves
    between two calls, and a plan that follows it is right (decided by design on 2026-10-30)."""
    logged = compared(plan_lines(log))
    options = readings(free, monkeypatch)
    made = [planner_says(reading, inputs, host) for reading in options]
    first = made[0]

    def pair(lines: list[str]) -> tuple[str, str, str]:
        return (
            line_of(lines, "windows: "),
            line_of(lines, "blocks swapped: "),
            line_of(lines, "disk: "),
        )

    return Verdict(
        call,
        free.said(options[0].counted),
        pair(logged),
        pair(first),
        agrees=logged in made,
    )


def _bracketed(lines: tuple[str, ...]) -> str:
    """Lines each in brackets, for a failure's message."""
    return " ".join(f"[{line}]" for line in lines)


def assert_planned_as_read(verdicts: list[Verdict]) -> None:
    """Fail unless each call's plan is the planner's for its own reading, saying each call's
    reading, windows, blocks swapped and disk's masters, and what the planner makes of it."""
    if all(verdict.agrees for verdict in verdicts):
        return
    pytest.fail(
        "a call's plan is not the planner's for the reading that call made:\n"
        + "\n".join(
            f"{verdict.call}: {'as the planner makes it' if verdict.agrees else 'NOT'}; read"
            f" {verdict.reading}; logged {_bracketed(verdict.logged)}; the planner makes"
            f" {_bracketed(verdict.planner)}"
            for verdict in verdicts
        )
    )


def work_of(tmp_path: Path) -> Path:
    """The work directory of the job the tests' command makes for tmp_path's ours.mkv
    (cli._work_directory), where its manifest is."""
    return tmp_path / "ours.mkv.work"


# Two readings of the free memory taken back to back are the same to the byte on a steady card
# (cli._free_memory: the same for a new job and a resume), the log saying them to 0.01 GiB; the
# tolerance takes that rounding twice over, so that a few MiB of the driver's own figure do not
# fail the test. PROVISIONAL (impl-v010, 2026-10-30: design asks the readings equal, a gap a bug,
# and leaves the tolerance open).
READING_TOLERANCE = round(0.02 * memory.GIB)
# How far the card's other programs may move between the calls (other_programs) for the card to be
# steady: the same figure. A desktop moving more is why the readings differ, not seedvr2x.
# PROVISIONAL (impl-v010, 2026-10-30): the same.
STEADY = READING_TOLERANCE


def other_programs(settle: float = 15.0) -> int:
    """What the card's programs hold of it by NVML (memory.card_state), in bytes, once it stops
    moving: read every half second until two readings are within 8 MiB, at most `settle`
    seconds. Taken where no seedvr2x process has the card (this one makes no CUDA context, and the
    calls are over), so that it is the other programs'. Skips the test where NVML says nothing:
    a steady card can't be told."""
    previous = None
    deadline = time.monotonic() + settle
    while True:
        state = memory.card_state()
        if state is None:
            pytest.skip("NVML doesn't answer: a steady card can't be told")
        used = state[1]
        if previous is not None and abs(used - previous) <= 8 << 20:
            return used
        if time.monotonic() >= deadline:
            return used
        previous = used
        time.sleep(0.5)


def run_until_planned(
    arguments: list[str],
    environment: dict[str, str],
    marker: str = "plan: pinned: ",
    wait: float = 3600,
) -> tuple[str, int | None]:
    """Run a command that logs to stderr and stop it once a line holds `marker`, the plan's last
    line by default (planner.said): the run has read its free memory and planned, the models not
    yet loaded. Returns its stderr to there and its exit status, None where it was stopped
    here. A command that ends first, a refusal, returns its whole stderr and its status."""
    process = subprocess.Popen(
        arguments,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    seen: list[str] = []
    stop = threading.Event()

    def read() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            seen.append(line)
            if marker in line:
                stop.set()
        stop.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    stop.wait(wait)
    try:
        # A command that ended (the stream closed) is reaped at once; one that logged the marker
        # is still running.
        status = process.wait(0.5)
    except subprocess.TimeoutExpired:
        status = None
        process.terminate()
        try:
            process.wait(60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    reader.join(30)
    return "".join(seen), status


def same_bits(tmp_path: Path, reference: Path) -> None:
    """Compare the master and float32 frames of the run made in tmp_path (upscaled) with the
    reference's, <reference>.mkv and <reference>_frames/, bit for bit."""
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    expected = (
        reference.parent / f"{reference.name}.mkv",
        reference.parent / f"{reference.name}_frames",
    )
    for kind, theirs, ours in (("masters", expected[0], master), ("frames", expected[1], frames)):
        compare = subprocess.run(
            [sys.executable, str(COMPARE), kind, str(theirs), str(ours)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compare.returncode == 0, f"{kind}: {compare.stdout[-3000:]}"


def bit_identical(
    tmp_path: Path,
    source: Path,
    resolution: int,
    models: tuple[str, str] | None,
    reference: Path,
    *more: str,
) -> str:
    """Run seedvr2x as upscaled does, the passes `more` doesn't pin pinned untiled, and compare
    its master and float32 frames with the reference's, bit for bit (same_bits). Returns its
    log."""
    log = upscaled(tmp_path, source, resolution, models, *more)
    same_bits(tmp_path, reference)
    return log


@pytest.mark.parametrize(
    ("dit", "vae", "reference"),
    [
        pytest.param(NUMZ_DIT, NUMZ_VAE, "7b", id="numz-7b"),
        pytest.param(
            "seedvr2x_ema_7b_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "7b",
            id="ours-7b",
        ),
        pytest.param(
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "sharp",
            id="ours-sharp",
        ),
        # The sharp 7B's dynamic GGUF (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0
        # too"): its matrices dequantised on the card before each multiply, held to its own
        # reference as float16 is to its own.
        pytest.param(
            "seedvr2x_ema_7b_sharp_dyn.gguf",
            "seedvr2x_ema_vae_fp16.safetensors",
            "sharp_dyn",
            id="ours-sharp-gguf",
        ),
    ],
)
def test_milestone1_bit_identical(tmp_path: Path, dit: str, vae: str, reference: str) -> None:
    assert REFERENCE is not None
    models_present(dit, vae)
    job = case("milestone1", "input_rgb.mkv", 1080, reference)
    referenced(tmp_path, job, (dit, vae))
    bit_identical(tmp_path, job.source, job.resolution, (dit, vae), job.reference)


def test_milestone1_unpinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # numz-7b with no tile pinned: the passes are the planner's own choice, and the bits go
    # through it. Decided before any run, by the same command with --plan: the first pass and
    # the plan, its cut list written and no master (main.py); --plan loads no model today. A card
    # where the plan is refused, or tiles a pass, can't make the untiled master: the case is
    # skipped where the driver reports under 40 GiB free, saying the refusal, or the plan's line
    # of the pass it tiles (planner.said), and fails where it reports more (unpinned). Else the
    # encode's and the decode's lines say untiled, as the reference ran them, and nothing is
    # pinned; the run then makes a plan of its own, and its bits are the reference's. The run
    # resumes the job the --plan call made, so its first pass is read in that call's log
    # (upscaled).
    # The two calls are not compared with each other: a desktop that drives the card moves its
    # free memory between them (0.3 GiB on an RTX 5090 under WSL2, 5 blocks swapped in one call
    # and 4 in the other), and a plan that follows it is right. Each call's plan is compared with
    # the planner's for the reading that call made (compare_with_planner): the --plan call's from
    # the manifest it leaves (exact), the run's from its --verbose log, its manifest being gone
    # with its work directory once the file is assembled (design, 2026-10-30). That the two calls
    # read the same memory on a steady card is test_plan_and_run_read_the_same_free_memory.
    # The disk line's masters (its first part: their size at most, assembled or not) are compared
    # too, rebuilt from the manifest's output and segments (recorded_disk); its free space, and
    # the verdict after it, follow the disk and are not (compared; design, baton 4, 2026-10-31).
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    own = case("milestone1", "input_rgb.mkv", 1080, "7b")
    referenced(tmp_path, own, (NUMZ_DIT, NUMZ_VAE))
    job = (tmp_path, own.source, own.resolution, (NUMZ_DIT, NUMZ_VAE))
    arguments, environment = command(*job, pin_untiled=False)
    listed = tmp_path / "cuts.txt"
    plan = subprocess.run(
        [*arguments, "-v", "--plan", str(listed)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    # A plan refused: exit status 1, the refusal the log's error (cli._run). Any other failure
    # fails the case. The call runs with -v, so that its log holds the system memory it planned
    # with (logged_host) as the run's does: its lines are the logger's (errors_of).
    refusals = [error for error in errors_of(plan.stderr) if any(w in error for w in REFUSED)]
    # A plan refused, or tiling a pass, is held to UNTILED_FREE (unpinned): a card's reference is
    # made untiled at the size its untiled passes fit (the module's docstring), a card reporting
    # less than that being skipped, saying it. A refusal records no reading: its figure is the
    # one its words give (reported).
    if plan.returncode == 1 and refusals:
        unpinned(plan.stderr, f"the plan is refused: {refusals[0]}")
    assert plan.returncode == 0, plan.stderr[-3000:]
    detected(plan.stderr)
    planned = plan_lines(plan.stderr)
    passes = [line for line in planned if line.startswith(("encode: ", "decode: "))]
    assert len(passes) == 2 and any(line.startswith("pinned: nothing") for line in planned), planned
    # The --plan call's reading and the inputs of its plan, from the manifest it left, read
    # before the run, whose plan may write it again.
    inputs, free_planned = recorded_inputs(json.loads((work_of(tmp_path) / NAME).read_text()))
    tiled = [line for line in passes if "tiles of" in line]
    if tiled:
        unpinned(
            plan.stderr,
            f"the plan tiles a pass: plan: {tiled[0]}",
            free_planned.reported / memory.GIB,
        )
    assert passes[0].startswith("encode: untiled, "), planned
    assert passes[1].startswith("decode: untiled, "), planned
    assert listed.is_file() and not (tmp_path / "ours.mkv").exists()
    log = upscaled(*job, "-v", pin_untiled=False, planned=plan.stderr)
    free_ran = logged_free(log)
    assert free_ran is not None, f"no free memory read in the run's --verbose log: {log[-3000:]}"
    assert_planned_as_read(
        [
            compare_with_planner(
                *("--plan", plan.stderr, free_planned, inputs, logged_host(plan.stderr)),
                monkeypatch=monkeypatch,
            )
        ]
    )
    # The run's own passes: the card it planned for may hold less than the --plan call's did.
    ran = plan_lines(log)
    run_passes = [line for line in ran if line.startswith(("encode: ", "decode: "))]
    assert len(run_passes) == 2 and any(line.startswith("pinned: nothing") for line in ran), ran
    run_tiled = [line for line in run_passes if "tiles of" in line]
    if run_tiled:
        unpinned(
            log,
            f"the run's plan tiles a pass: plan: {run_tiled[0]}",
            free_ran.reported / memory.GIB,
        )
    # After the run's tiling check: a run that tiles where --plan did not skips or fails as
    # unpinned() says, not on the planner comparison (review of baton 3, 2026-10-31).
    assert_planned_as_read(
        [
            compare_with_planner(
                *("the run", log, free_ran, inputs, logged_host(log)),
                monkeypatch=monkeypatch,
            )
        ]
    )
    same_bits(tmp_path, own.reference)


def test_plan_and_run_read_the_same_free_memory(tmp_path: Path) -> None:
    # The free memory --plan reads and the run reads, back to back on the same case, are the same
    # figure: both read it once the first pass is done and before any model loads, with a first
    # allocation and a warm-up in place (cli._free_memory), the same to the byte for a new job
    # and a resume (its docstring). A gap is a bug. The run is stopped as soon as it has planned,
    # its models never loaded: its reading is read in its --verbose log (logged_free), the --plan
    # call's in the manifest it leaves, copied before the run. The test cannot make the desktop
    # steady: the other programs' memory on the card (NVML's used, with none of seedvr2x's processes
    # on it) is read before the first call, between the two and after the second; where the readings
    # differ and that memory moved by more than STEADY, the desktop moved, not seedvr2x, and the
    # test skips, saying the figures. (Design's check, baton 3, 2026-10-30.)
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    own = case("milestone1", "input_rgb.mkv", 1080, "7b")
    if not own.source.exists():
        pytest.skip(f"{own.source}: absent; this module's docstring says how to make it")
    arguments, environment = command(
        tmp_path, own.source, own.resolution, (NUMZ_DIT, NUMZ_VAE), pin_untiled=False
    )
    before = other_programs()
    plan = subprocess.run(
        [*arguments, "--plan", str(tmp_path / "cuts.txt")],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if plan.returncode == 1 and (refused := refusal(plan.stderr)) is not None:
        pytest.skip(f"{card()}: the case can't run there: {refused}")
    assert plan.returncode == 0, plan.stderr[-3000:]
    _, free_planned = recorded_inputs(json.loads((work_of(tmp_path) / NAME).read_text()))
    between = other_programs()
    log, status = run_until_planned([*arguments, "-v"], environment)
    after = other_programs()
    assert status is None, f"the run ended before its plan was made: {log[-3000:]}"
    free_ran = logged_free(log)
    assert free_ran is not None, f"no free memory read in the run's --verbose log: {log[-3000:]}"
    gaps = [abs(free_ran.reported - free_planned.reported)]
    if free_ran.card is not None and free_planned.card is not None:
        gaps.append(abs(free_ran.card - free_planned.card))
    same = (free_ran.card is None) == (free_planned.card is None) and max(gaps) <= READING_TOLERANCE
    if same:
        return
    moved = max(abs(between - before), abs(after - between))
    figures = (
        f"--plan read {free_planned.said()}, the run {free_ran.said()}; the card's other"
        f" programs held {before / memory.GIB:.2f}, {between / memory.GIB:.2f} and"
        f" {after / memory.GIB:.2f} GiB before the first call, between the two and after the"
        f" second"
    )
    if moved > STEADY:
        pytest.skip(f"the desktop moved, not seedvr2x: {figures}")
    pytest.fail(f"the card was steady and the two calls read other memory: {figures}")


def test_milestone1_from_the_cache(tmp_path: Path) -> None:
    # ours-sharp, its files taken from Hugging Face's cache, as a user's run takes them by
    # default, the shot detector's with them: offline, so that the cache alone can serve them,
    # bit for bit as SEEDVR2X_MODEL_DIR does. Skipped unless the run would find all three there
    # (pull.cached).
    assert REFERENCE is not None
    for name in (DEFAULT_DIT, DEFAULT_VAE, pull.DETECTOR):
        if pull.cached(name) is None:
            pytest.skip(
                f"{name}: not in Hugging Face's cache; fill it with the three default files, 17"
                f" GB: uv run --project ../apps/seedvr2x hf download {pull.REPO} {DEFAULT_DIT}"
                f" {DEFAULT_VAE} {pull.DETECTOR} --revision {pull.REVISION}"
            )
    job = case("milestone1", "input_rgb.mkv", 1080, "sharp")
    referenced(tmp_path, job, (DEFAULT_DIT, DEFAULT_VAE))
    log = bit_identical(tmp_path, job.source, job.resolution, None, job.reference)
    for name in (DEFAULT_DIT, DEFAULT_VAE, pull.DETECTOR):
        assert f"{name}: in the cache, " in log
        pinned = f"its size and SHA-256 as seedvr2x pins them, {pull.REPO} at {pull.REVISION[:8]}"
        assert f"{name}: {pinned}" in log
    assert "downloading" not in log


@pytest.mark.parametrize(
    ("source", "resolution", "reference"),
    [
        pytest.param("input_rgb.mkv", 1080, "pad1080", id="pad1080"),
        pytest.param("input_rgb.mkv", 720, "pad720", id="pad720"),
        pytest.param("input_rgb_956x530.mkv", 1060, "padcols", id="padcols"),
    ],
)
def test_padding_bit_identical(
    tmp_path: Path, source: str, resolution: int, reference: str
) -> None:
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    job = case("padding", source, resolution, reference)
    models = (NUMZ_DIT, NUMZ_VAE)
    referenced(tmp_path, job, models)
    bit_identical(tmp_path, job.source, job.resolution, models, job.reference)


@pytest.mark.parametrize(
    ("pins", "reference", "grids"),
    [
        pytest.param(
            ("--decode-tile", "1024"),
            "d1024",
            ["tiled decode: 4 tiles of 1024 px sharing 64, 2 across and 2 down,"],
            id="d1024",
        ),
        pytest.param(
            ("--encode-tile", "1344", "--decode-tile", "1024"),
            "e1344-d1024",
            [
                "tiled encode: 2 tiles of 1344 px sharing 64, 2 across and 1 down,",
                "tiled decode: 4 tiles of 1024 px sharing 64, 2 across and 2 down,",
            ],
            id="e1344-d1024",
        ),
        pytest.param(
            ("--encode-tile", "512", "--decode-tile", "512"),
            "e512-d512",
            [
                "tiled encode: 15 tiles of 512 px sharing 64, 5 across and 3 down,",
                "tiled decode: 15 tiles of 512 px sharing 64, 5 across and 3 down,",
            ],
            id="e512-d512",
        ),
    ],
)
def test_tiled_bit_identical(
    tmp_path: Path, pins: tuple[str, ...], reference: str, grids: list[str]
) -> None:
    # seedvr2x's tiled passes against their own references (the module's docstring): the whole
    # shot under each tile. The debug log says each tiled pass's grid, over the 45 frames padded to
    # 1920x1104, its scratch in system memory at that size (572,313,600 bytes).
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    job = case("tiled", "input_rgb.mkv", 1080, reference)
    models = (NUMZ_DIT, NUMZ_VAE)
    referenced(tmp_path, job, models, *pins)
    log = bit_identical(tmp_path, job.source, job.resolution, models, job.reference, *pins, "-v")
    # The job's own passes: those of its time estimates' calibration come before, said too
    # (calibration.calibrate, which every run makes: DESIGN.md, Memory planner, Time estimates).
    job_lines = log.split("time estimates: the job's own steps timed once, in", 1)[-1]
    said = [
        line.split("tiling: ", 1)[1] for line in job_lines.splitlines() if "tiling: tiled " in line
    ]
    # At 1080, or at the card's own size where its cases.json gives one, with what each tiled
    # pass's line says there (case). The default's grids are tilegrid.grid's over the 1104x1920
    # frame (138 x 240 latents, tiles of 128, 168 and 64 latents sharing 8, a stride of 120, 160
    # and 56), as tiling._tiled logs them, and its scratch is tilegrid.scratch_bytes: 3 channels
    # x 45 frames x 1104 x 1920 x 2 bytes, 564,019,200 bytes at numz's 1088 rows, 572,313,600
    # now.
    grids = list(job.grids) if job.grids is not None else grids
    over = job.over or (
        "over 45 frames padded to 1920x1104; its scratch, 572313600 bytes, in system memory"
    )
    assert len(said) == len(grids), said
    for line, grid in zip(said, grids, strict=True):
        assert line.startswith(grid), said
        assert line.endswith(over), said
    assert not (tmp_path / "ours.mkv.work").exists()
