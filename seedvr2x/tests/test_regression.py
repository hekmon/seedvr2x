"""Milestone 1 as a regression test: seedvr2x's output stays bit-identical to numz's, in numz's
padding; and in seedvr2x's own (DESIGN.md, Pipeline step 0), to numz's with that padding patched
in. And the VAE tiling (DESIGN.md, Pipeline step 3): seedvr2x tiled, to numz tiled. Every case
runs in numz's draw of the noise (cli.NUMZ_DRAW, for the tests only): numz drew each reference's
noise on the GPU, whose generator follows the card, where seedvr2x draws it on the CPU (DESIGN.md,
Pipeline step 2), so that the references hold on the card that made them. And in numz's tail
(cli.NUMZ_TAIL): numz mirrors a shot's end, seedvr2x repeats its last frame by default (DESIGN.md,
Pipeline step 1, decided on 2026-10-19). The input's 45 frames are one shot of 4n + 1 frames, which
neither pads: the mode changes no bit of these cases, and keeps them numz's whatever input a
reference is remade with.

Needs a GPU (pytest -m gpu), the weights (SEEDVR2X_MODEL_DIR) and milestone 1's reference, in
the m1/ directory of SEEDVR2X_REFERENCE_DIR:
- input_rgb.mkv: 45 frames of 8-bit RGB FFV1 at 960x540, read alike by numz (OpenCV) and by us:
  frames 48-92 of SEGMENT, a 1920x1080 yuv420p FFV1 segment of an animated episode (limited
  range, chroma left, no colour tags, 23.976 fps, 377 frames; its first 26 black, then a fade-in
  to about frame 50), BT.709 limited range to full-range RGB, its chroma upsampled and the
  picture brought to 960x540 by zimg's spline36, with ffmpeg n9.0.2:

    VF=trim=start_frame=48:end_frame=93,setpts=PTS-STARTPTS
    VF=$VF,zscale=w=960:h=540:filter=spline36:matrixin=709:transferin=709:primariesin=709
    VF=$VF:rangein=limited:transfer=709:primaries=709:range=full:dither=none,format=gbrp
    ffmpeg -nostdin -y -i SEGMENT -map 0:v:0 -vf "$VF" -fps_mode passthrough \
      -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 -an -sn input_rgb.mkv

  Remade so on 2026-10-09: the same frames and packets, the two files differing only in the
  Matroska muxer's random UIDs.
- numz.mkv and numz_frames/: numz 4490bd1's output for input_rgb.mkv, captured by
  research/scripts/ffv1_out.py, from numz's checkout:

    FFV1_OUT_PATH=numz.mkv FFV1_OUT_KEEP=0 FFV1_OUT_DUMP=numz_frames \
    FFV1_OUT_DUMP_FRAMES=$(seq -s, 0 44) python ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction none

- numz_sharp.mkv and numz_sharp_frames/: the same with the sharp 7B (made on 2026-10-08), the
  same command with FFV1_OUT_PATH=numz_sharp.mkv, FFV1_OUT_DUMP=numz_sharp_frames and
  --dit_model seedvr2_ema_7b_sharp_fp16.safetensors.

Each case is a run of its own, both model files named, in numz's padding (cli.NUMZ_PADDING, for
the tests only) and numz's draw:
- numz-7b: numz's seedvr2_ema_7b_fp16.safetensors and ema_vae_fp16.safetensors, against numz.mkv;
- ours-7b: seedvr2x's seedvr2x_ema_7b_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors,
  the same values as numz's files (DESIGN.md, Weights), so against the same reference, bit for
  bit;
- ours-sharp: seedvr2x_ema_7b_sharp_fp16.safetensors and seedvr2x_ema_vae_fp16.safetensors, against
  numz_sharp.mkv, skipped when that reference is absent.
Every run detects its shots, as a user's does without --cuts (DESIGN.md, Shot detection): the
shot detector runs on the GPU in the first pass, before the models load, so that its settings,
scoped to its own forwards, and the memory it gives back are held to the bits too. At
--cut-threshold 0.5, where its cuts are none, so that the job is one shot, as numz's one batch,
which the log must say. At the default, 0.3, it isn't: input_rgb.mkv ends on two short shots,
which numz's batch ran through, TransNetV2 peaking at 0.3242 on frame 41 and 0.3880 on frame 43,
and at 0.1171 on frame 0, on the CPU and on the GPU alike (1.19e-07 apart; the same on its
956x530 crop): cut at 42 and 44, 3 shots, none of the run's 45 frames was numz's, where at 0.5,
1 shot and 2 possible cuts, all 45 are (the GPU box, 2026-10-10). A case whose model file, or the
shot detector's (transnetv2.safetensors), isn't in SEEDVR2X_MODEL_DIR is skipped, naming the file.
One more,
test_milestone1_from_the_cache, runs ours-sharp with the default models taken from Hugging Face's
cache (runtime/pull.py): no --model-dir, no --dit-model or --vae-model, and HF_HUB_OFFLINE=1 in
its environment, so that only the cache can serve them; it needs no SEEDVR2X_MODEL_DIR, and is
skipped, saying how to fill the cache (uv run hf download, the three files, 17 GB), unless the
run would find them there: at the pinned revision, or downloaded at another (runtime/pull.py).

Every case pins the passes its reference was made with, untiled or in its tiles, but one,
test_milestone1_unpinned: numz-7b with no tile pinned, so that a bit comparison goes through the
planner's own choice of passes (DESIGN.md, Memory planner, Choices, 3). The plan is made first,
by the same command with --plan, which runs the first pass and plans, loading no model today
(DESIGN.md has --plan load them for the time estimates' calibration, which isn't built): its
lines of the passes, "plan: encode: untiled, ..." and "plan: decode: untiled, ...", must say both
untiled, as numz ran them; the run then makes a plan of its own (each
call's held to the planner's for the free memory that call read, not to the other's: the desktop
moves it), and the master and frames are numz's bit for bit. A card that can't run untiled
can't make numz's untiled
master (the untiled decode of these frames, padded to 1920x1088, takes 37.09 GiB of free VRAM by
the planner, the DiT off the GPU, its room and the margin counted): where the driver reports
under 40 GiB free (UNTILED_FREE), a plan that tiles a pass, or is refused, skips the case before
any run, saying that line or the refusal. With 40 GiB or more reported, such a plan fails it: the
plan is wrong there, or what the log says of the free memory no longer reads (reported).

seedvr2x's own padding, the default (at least 8 rows reflected from the picture, up to a multiple
of 16, then 16 black rows; columns alike when the width isn't a multiple of 16), is held to numz
patched by research/scripts/numerics_patch.py with NUM_PAD=reflect>=8+black+16, which pads the
same way, numz's 7B fp16 and VAE on both sides. Each reference was made on 2026-10-09, from numz's
checkout, with numerics_patch.py running ffv1_out.py running numz's CLI:
- pad1080: input_rgb.mkv at --resolution 1080, 1920x1080 padded to 1920x1104 (8 rows reflected,
  no column), against numz_pad1080.mkv and numz_pad1080_frames/:

    NUM_PAD='reflect>=8+black+16' FFV1_OUT_PATH=numz_pad1080.mkv FFV1_OUT_KEEP=0 \
    FFV1_OUT_DUMP=numz_pad1080_frames FFV1_OUT_DUMP_FRAMES=$(seq -s, 0 44) \
    python numerics_patch.py ffv1_out.py inference_cli.py input_rgb.mkv \
      --output out --output_format png --model_dir MODELS \
      --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
      --seed 42 --attention_mode flash_attn_2 --color_correction none

- pad720: input_rgb.mkv at --resolution 720, 1280x720 padded to 1280x752 (16 rows reflected),
  against numz_pad720.mkv and numz_pad720_frames/: the same command with
  FFV1_OUT_PATH=numz_pad720.mkv, FFV1_OUT_DUMP=numz_pad720_frames and --resolution 720;
- padcols: input_rgb_956x530.mkv, input_rgb.mkv's top left 956x530, cropped losslessly,

    ffmpeg -i input_rgb.mkv -vf crop=956:530:0:0 -c:v ffv1 -pix_fmt bgr0 input_rgb_956x530.mkv

  at --resolution 1060, 1912x1060 padded to 1936x1088 (12 rows and 8 columns reflected), against
  numz_padcols.mkv and numz_padcols_frames/: the same command with input_rgb_956x530.mkv,
  FFV1_OUT_PATH=numz_padcols.mkv, FFV1_OUT_DUMP=numz_padcols_frames and --resolution 1060.
numz logs each size (src/core/generation_utils.py:274-279):
"Padded: 1920x1104px → Output: 1920x1080px", "Padded: 1280x752px → Output: 1280x720px",
"Padded: 1936x1088px → Output: 1912x1060px"; seedvr2x's --resolution gives the same
(tests/test_padding.py). A case whose input or reference is absent is skipped, naming the file.

The VAE tiling (runtime/tiling.py) is held to numz's own, on milestone 1's input in numz's padding,
numz's 7B fp16 and VAE on both sides: seedvr2x with its tiles pinned, the whole shot under each
tile, against numz 4490bd1's CLI tiling its one batch of 45 frames, bit for bit. The references are
in the m3/ directory of SEEDVR2X_REFERENCE_DIR, each made on 2026-10-10 by numz.mkv's command
above with numz's tiling options added, its overlap the 64 px seedvr2x fixes, and
--tensor_offload_device left at its default, cpu: numz then weights each tile on the GPU, sums
them on the CPU and divides the sums on the GPU, as runtime/tiling.py does:
- d1024: --decode-tile 1024, against numz_d1024.mkv and numz_d1024_frames/ (FFV1_OUT_PATH and
  FFV1_OUT_DUMP so named, as for each case), made with
  --vae_decode_tiled --vae_decode_tile_size 1024 --vae_decode_tile_overlap 64: on the frames
  padded to 1920x1088, 4 tiles, 2 rows of 2 (rows 0-1024 and 960-1088, columns 0-1024 and
  960-1920);
- e1344-d1024: --encode-tile 1344 --decode-tile 1024, against numz_e1344-d1024.mkv and
  numz_e1344-d1024_frames/, made with d1024's three options and
  --vae_encode_tiled --vae_encode_tile_size 1344 --vae_encode_tile_overlap 64: the encode in 2
  tiles, one row (columns 0-1344 and 1280-1920);
- e512-d512: --encode-tile 512 --decode-tile 512, against numz_e512-d512.mkv and
  numz_e512-d512_frames/, made with the six options at 512 and 64: 15 tiles each way, 3 rows of 5.
Each run's log must say those grids. A case whose reference is absent is skipped, naming the file.

Each card holds to references of its own (decided on 2026-10-21): numz drew each reference's noise
on the card that made it, and another card's frames differ from them (3 cases ran on an RTX 5090
and none of their 45 frames was the 96 GB card's). Another card's references are seedvr2x's own
frames on it, made by the code the 96 GB card gated that day, whose frames there are numz's bit
for bit (main 35435f2, its seedvr2x subset 13979c69, that of 490baf1), by the commands command()
gives, in these cases and modes; numz itself is run there once as a cross-check (DESIGN.md,
Validation milestones 3). A test takes its card's, by name as torch gives it (card();
torch.cuda.get_device_name). The 96 GB card's are those above, in m1/ and m3/ (ROOT_CARD).
Another card's are in cards/<its name>/m1/ and m3/, its name's runs of characters other than
letters and digits each a hyphen (cards/NVIDIA-GeForce-RTX-5090/), the inputs staying m1/'s,
with cards/<its name>/cases.json saying what each case takes there (case): "milestone1":
{"resolution": R, "references": {"numz": NAME, "numz_sharp": NAME}} for the milestone 1 cases,
the unpinned one and the one from the cache, which run untiled at the largest size the card's
untiled passes fit; "padding", by each case's id, its "source", "resolution" and "reference";
"tiled", by each case's id, its "reference", at 1080 as above or at its own "resolution", with
what its debug log says of each tiled pass there, the start of each grid's line ("grids") and the
end every one shares ("over"); and "made", the commit and subset that made them. A gate's case
keeps 2 GiB of the card's budget (DESIGN.md, Validation, decided on 2026-10-21): on the RTX 5090
(2026-10-22, a desktop holding 5.5 GB of the card, 20.41 GiB the plan counts on), each case at a
size whose plan beside a second program holding 2 GiB of the card is neither refused nor made
with the reserve of 2.5 GiB, its window and tiles the same: milestone 1 at 720 (1280x720; 756
takes 2.5 there, 792 too, with nothing beside it); the padding cases pad1080 at 648 (1152x648
padded to 1152x672, 8 rows reflected and no column, as pad1080 at 1080), pad720 at 576
(1024x576 padded to 1024x608, its height already a multiple of 16, as pad720 at 720), padcols
at 530 (956x530 padded to 992x560, rows and columns reflected); the tiled cases d1024 and
e1344-d1024 at 864 (1536x864, a decode in 2 tiles of 1024 px, 2 across and 1 down; an encode in
2 tiles of 1344 px likewise), e512-d512 at 1080. The references made again at a size kept
(padcols, e512-d512) are those of 2026-10-21 frame for frame. Its numz cross-check, numz 4490bd1
through ffv1_out.py with its block swap (--blocks_to_swap 36 --dit_offload_device cpu
--vae_offload_device cpu): in each case but d1024 at 2026-10-21's sizes, and on milestone 1's
case at 720, the same frames, 45 of 45 in each. A case a card has no reference for is skipped,
naming the card and the reference missing, or, where the card can't run it, with its refusal's
figures (no_reference); a case refused there at its run is skipped the same way, and a case
planned there with the reserve of 2.5 GiB fails (upscaled), the card no longer leaving it the
room its reference was made in. On the 96 GB card nothing of this changes, and nothing skips:
a refusal there fails the case.

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

from seedvr2x.cli import NUMZ_DRAW, NUMZ_PADDING, NUMZ_TAIL
from seedvr2x.runtime import memory, planner, pull
from seedvr2x.runtime.job import Shot, target_size
from seedvr2x.runtime.manifest import NAME

PROJECT = Path(__file__).resolve().parents[1]
MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not REFERENCE, reason="needs SEEDVR2X_REFERENCE_DIR"),
]

# numz's model files, which the references were made with.
NUMZ_DIT, NUMZ_VAE = "seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors"
# The command line's defaults, the sharp 7B and the VAE (cli.py).
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


# The card whose references are m1/ and m3/ of SEEDVR2X_REFERENCE_DIR, numz having made them on it
# (the module's docstring), as torch names it (torch.cuda.get_device_name). Another card's are in
# cards/<its name>/ beside them (card_directory), with what each case of it takes (CASES).
ROOT_CARD = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
CARDS = "cards"
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


def card_directory(name: str) -> Path:
    """Where the references of the card `name` are: SEEDVR2X_REFERENCE_DIR itself for ROOT_CARD;
    for another, cards/<name>/ in it, its name's runs of other characters than letters and digits
    each a hyphen ("NVIDIA GeForce RTX 5090": cards/NVIDIA-GeForce-RTX-5090/)."""
    assert REFERENCE is not None
    if name == ROOT_CARD:
        return Path(REFERENCE)
    return Path(REFERENCE) / CARDS / re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


@dataclass(frozen=True)
class Case:
    """A case as this card runs it: its input, the resolution it is upscaled to, and its reference,
    <reference>.mkv and <reference>_frames/; for a tiled case at another size than 1080 on a card
    other than ROOT_CARD, what its debug log says of each tiled pass there, the start of each
    grid's line and the end every one shares (grids, over), else None: the test's own, at 1080."""

    source: Path
    resolution: int
    reference: Path
    grids: tuple[str, ...] | None = None
    over: str | None = None


def case(kind: str, name: str, source: str, resolution: int, reference: str, where: str) -> Case:
    """The case `name` of `kind` (milestone1, padding or tiled) on this card (card()). On
    ROOT_CARD, as the test's parameters give it: its input in m1/, its reference in `where` (m1
    or m3) of SEEDVR2X_REFERENCE_DIR. On another, as its directory's CASES has it (the module's
    docstring): milestone1's resolution for all its cases; each padding case's source, resolution
    and reference by its id; each tiled case's reference by its id, at 1080 as on ROOT_CARD or at
    its own "resolution", with its tiled passes' "grids" and what they are "over" there; the
    parameters for what the entry doesn't say, its reference in `where` of the card's directory.
    The inputs are the same on every card: m1/ of SEEDVR2X_REFERENCE_DIR."""
    assert REFERENCE is not None
    inputs = Path(REFERENCE) / "m1"
    directory = card_directory(card())
    if card() == ROOT_CARD:
        return Case(inputs / source, resolution, directory / where / reference)
    said: Any = {}
    if (directory / CASES).is_file():
        said = json.loads((directory / CASES).read_text()).get(kind, {})
    if kind == "milestone1":
        entry: Any = {**said, "reference": said.get("references", {}).get(name, reference)}
    else:
        entry = said.get(name, {})
    grids = entry.get("grids")
    return Case(
        inputs / entry.get("source", source),
        int(entry.get("resolution", resolution)),
        directory / where / entry.get("reference", reference),
        None if grids is None else tuple(str(grid) for grid in grids),
        entry.get("over"),
    )


def refusal(log: str) -> str | None:
    """The refusal of the planner's for the memory a job takes (REFUSED) in a run's log, its
    error's message, or None: not the disk's (cli._room_on_disk), whose scratch's part names the
    system memory too, and whose way out is the machine's, not the card's (a review's finding).
    A line without --verbose is "12:00:00 ERROR message"; under it, "... ERROR seedvr2x:
    message" (cli.VERBOSE_FORMAT)."""
    for line in log.splitlines():
        found = re.match(r"(?:\d\d:\d\d:\d\d|.* ERROR seedvr2x:) ?(?:ERROR )?(.*)", line)
        if found and " ERROR " in f" {line} " and any(what in found[1] for what in REFUSED):
            if "free on its disk" not in found[1]:
                return found[1]
    return None


def no_reference(
    tmp_path: Path, job: Case, models: tuple[str, str], *more: str, numz_padding: bool
) -> NoReturn:
    """Skip a case this card has no reference for: with the refusal's figures where the card
    can't run it, its --plan refused (the planner's own words), else naming the card and the
    reference missing. On ROOT_CARD a reference absent is skipped as before, naming it."""
    missing = f"{job.reference}.mkv"
    if card() == ROOT_CARD:
        pytest.skip(f"{missing}: absent; this module's docstring says how to make it")
    arguments, environment = command(
        *(tmp_path, job.source, job.resolution, models, *more), numz_padding=numz_padding
    )
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
    pytest.skip(f"{card()}: no reference for it, {missing}; this module's docstring says how")


def referenced(
    tmp_path: Path,
    job: Case,
    models: tuple[str, str],
    *more: str,
    numz_padding: bool,
    required: bool = False,
) -> None:
    """Go on with a case only where its input and reference are there (no_reference). On
    ROOT_CARD, a case `required` goes on whatever is there, as it always did, its comparison
    failing where its reference is missing: milestone 1's own references, numz.mkv's, are the
    module's; the others were skipped when absent and still are."""
    if card() == ROOT_CARD and required:
        return
    if not job.source.exists():
        pytest.skip(f"{job.source}: absent; this module's docstring says how to make it")
    for path in (
        job.reference.parent / f"{job.reference.name}.mkv",
        job.reference.parent / f"{job.reference.name}_frames",
    ):
        if not path.exists():
            no_reference(tmp_path, job, models, *more, numz_padding=numz_padding)


# The tile pins of a run (cli.py): one not given is pinned untiled (upscaled), but where a case
# leaves its passes to the planner.
TILE_PINS = ("--encode-tile", "--decode-tile")


def command(
    tmp_path: Path,
    source: Path,
    resolution: int,
    models: tuple[str, str] | None,
    *more: str,
    numz_padding: bool,
    pin_untiled: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """seedvr2x's command line on source at resolution with the DiT and VAE files of models in
    SEEDVR2X_MODEL_DIR, or with None, its default models from Hugging Face's cache, offline
    (HF_HUB_OFFLINE=1), in numz's padding or in its own, with the options of `more` besides, to
    tmp_path's ours.mkv and frames/ (same_bits); and the environment to run it in, numz's draw
    of the noise in it whatever the padding (NUMZ_DRAW: the module's docstring), and numz's tail
    (NUMZ_TAIL: the same). A pass that `more` doesn't pin is pinned untiled, unless pin_untiled is
    False, which leaves it to the planner."""
    master, frames = tmp_path / "ours.mkv", tmp_path / "frames"
    modes = (NUMZ_PADDING, NUMZ_DRAW, NUMZ_TAIL)
    environment = {name: value for name, value in os.environ.items() if name not in modes}
    environment[NUMZ_DRAW] = "1"
    environment[NUMZ_TAIL] = "1"
    if numz_padding:
        environment[NUMZ_PADDING] = "1"
    if models is None:
        options = []
        environment["HF_HUB_OFFLINE"] = "1"
    else:
        assert MODELS is not None
        options = ["--model-dir", MODELS, "--dit-model", models[0], "--vae-model", models[1]]
    arguments = [
        *(sys.executable, "-m", "seedvr2x", str(source), "-o", str(master), *options),
        *("--resolution", str(resolution), "--seed", "42", "--dump-frames", str(frames)),
        # The shots detected, and none found: at the default 0.3, the input is cut at 42 and
        # 44, 3 shots where numz ran one batch (the module's docstring).
        *("--cut-threshold", "0.5"),
        # numz's output milestone 1 holds to, without its lab (StableSR's, not vendored).
        *("--color-correction", "none"),
        # numz.mkv's format, ffv1_out.py's default: compare.py reads both masters as stored.
        *("--format", "gbrp16le"),
        # The passes the reference was made with, pinned: untiled unless `more` pins tiles, so
        # that a card too small for them refuses the job (DESIGN.md, Memory planner, Pins)
        # where the planner would tile what numz ran untiled, bits that aren't its. None pinned
        # for the case that leaves its passes to the planner, and looks at what it chose
        # (test_milestone1_unpinned).
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
    numz_padding: bool,
    pin_untiled: bool = True,
    planned: str | None = None,
) -> str:
    """Run seedvr2x as command has it, to tmp_path's ours.mkv and frames/ (same_bits), its first
    pass read (detected) in the log of the call that ran it: the run's own, or `planned`, the log
    of the same command's --plan call before it, where the run says it resumes a job. Returns the
    run's log."""
    arguments, environment = command(
        *(tmp_path, source, resolution, models, *more),
        numz_padding=numz_padding,
        pin_untiled=pin_untiled,
    )
    run = subprocess.run(arguments, capture_output=True, text=True, check=False, env=environment)
    # A card other than ROOT_CARD that can't run the case: skipped with the refusal's figures
    # (decided on 2026-10-21). On ROOT_CARD, which made the references, a refusal fails it.
    refused = refusal(run.stderr)
    if run.returncode == 1 and refused is not None and card() != ROOT_CARD:
        pytest.skip(f"{card()}: the case can't run there: {refused}")
    assert run.returncode == 0, run.stderr[-3000:]
    # A held run's plan made with the reserve of 2.5 GiB, the one with 4.5 refused (planner.plan):
    # not the case its reference was made in, which keeps 2 GiB of the card's budget (DESIGN.md,
    # Validation, decided on 2026-10-21), the card's desktop having grown.
    budget = [line for line in plan_lines(run.stderr) if line.startswith("budget: ")]
    tight = [line for line in budget if " and not " in line]
    assert not tight, f"{card()}: the case was planned with the tight reserve: {tight[0]}"
    # ours.mkv is made in ours.mkv.work, a directory's job, and --plan makes that job with its
    # first pass (cli._run): the run after it resumes the job, and scores no frame again. Without
    # a --plan call, or where the run doesn't say it resumes, the first pass is the run's own.
    # (A line without --verbose starts with the time of day, "12:00:00 resuming ..."; under it, in
    # the logger's format, "... INFO seedvr2x: resuming ...": cli.VERBOSE_FORMAT.)
    resumes = re.search(r"^(\d\d:\d\d:\d\d|.* INFO seedvr2x:) resuming ", run.stderr, re.M)
    detected(planned if planned is not None and resumes is not None else run.stderr)
    return run.stderr


def detected(log: str) -> None:
    """A run's first pass, by its log: the shot detector ran, on the GPU, and gave its memory
    back; no cut at 0.5, the input one shot, as numz's one batch (the module's docstring)."""
    assert "the shot detector scored" in log, log[-3000:]
    assert "shot detector: VRAM peak" in log, log[-3000:]
    assert "1 shot: 0 cuts detected at --cut-threshold 0.5" in log, log[-3000:]


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


# What every refusal of the planner's for the memory a job takes says (planner.PlanError): the
# free VRAM it takes, or the system memory.
REFUSED = (" GiB of free VRAM", " GiB of system memory")

# The free memory, in GiB as the driver reports it, from which test_milestone1_unpinned holds a
# card to numz's untiled passes: the plan takes 37.09 GiB of free VRAM for them (the module's
# docstring), and a card of 48 GB reports 46.5 or more (cli._free_memory: N - 1.49 on a card
# emulated at N GB).
UNTILED_FREE = 40.0

# What the driver reported free to a run, in the words of the plan's first line and of its
# refusals, the first that is there (reported): on a run held to the card, the driver's own
# figure, said beside the whole card's (at debug, under --verbose: planner.card_detail) or in
# its place; on a driver that refuses, the one figure
# a refusal gives, or the plan's first line (planner._card_said, planner._there).
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
    """The unpinned case on a card whose plan can't make numz's untiled master, `why` saying
    it, the refusal or the plan's line of its passes: skipped where the driver reported under
    UNTILED_FREE GiB free to the --plan run whose log this is (reported), failed where it
    reported that or more, or where the log doesn't say. `free`, in GiB, is that figure where
    the caller has it from the call's own record (Free), not from the log's words."""
    free = reported(log) if free is None else free
    if free is not None and free < UNTILED_FREE:
        pytest.skip(
            f"the driver reports {free:.2f} GiB free, under the {UNTILED_FREE:.0f} a card makes"
            f" numz's untiled master from: {why}"
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
    (planner.Peaks.record): the driver's figure, the card's, and a reserve only on a held run."""
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


@dataclass(frozen=True)
class Inputs:
    """What a job's plan is made from besides the memory, read from the manifest --plan left
    (recorded_inputs): the target size, the shots and the attention backend the plan was
    predicted for."""

    target: tuple[int, int]
    shots: tuple[Shot, ...]
    attention: str


def recorded_inputs(content: dict[str, Any]) -> tuple[Inputs, Free]:
    """The inputs of the plan a manifest's content records, with the free memory it was made in:
    the target from the input's size and sample aspect at the setting's resolution (cli._run, job.
    target_size), the shots as recorded, the attention backend of the plan's record."""
    source, settings = content["input"], content["settings"]
    width, height = source["size"]
    target = target_size(width, height, Fraction(source["sample_aspect"]), settings["resolution"])
    shots = tuple(Shot(shot["start"], shot["end"]) for shot in content["shots"])
    memory_record = content["plan"]["memory"]
    return Inputs(target, shots, memory_record["attention"]), recorded_free(memory_record)


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
            monkeypatch.setattr(memory, "_native", True)
            monkeypatch.setattr(memory, "_segments", True)
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
    none), numz's padding, `host` the system memory available. The disk's line is not compared:
    it follows the disk. A refusal is one line, "refused: ...", which no run's plan equals."""
    try:
        made, peaks = planner.plan(
            reading,
            inputs.target,
            inputs.shots,
            planner.Pins(),
            host,
            numz_padding=True,
            attention=inputs.attention,
        )
    except planner.PlanError as error:
        return [f"refused: {error}"]
    said = planner.said(
        made, peaks, "a card", inputs.target, inputs.shots, planner.Pins(), numz_padding=True
    )
    return compared(plan_lines("\n".join(said)))


def compared(lines: list[str]) -> list[str]:
    """A plan's lines as two plans are compared: choices, without the disk's line."""
    return [line for line in choices(lines) if not line.startswith("disk: ")]


def line_of(lines: list[str], start: str) -> str:
    """The first of a plan's lines that starts with `start`, or a note that there is none."""
    return next((line for line in lines if line.startswith(start)), f"(no line {start!r})")


@dataclass(frozen=True)
class Verdict:
    """A call's plan against the planner's for the call's own reading (compare_with_planner)."""

    call: str
    reading: str
    logged: tuple[str, str]  # the call's windows line and blocks-swapped line
    planner: tuple[str, str]  # the planner's, for the reading as given
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
    possible (readings). Not compared with another call's plan: the desktop's memory moves
    between two calls, and a plan that follows it is right (decided by design on 2026-10-30)."""
    logged = compared(plan_lines(log))
    options = readings(free, monkeypatch)
    made = [planner_says(reading, inputs, host) for reading in options]
    first = made[0]

    def pair(lines: list[str]) -> tuple[str, str]:
        return line_of(lines, "windows: "), line_of(lines, "blocks swapped: ")

    return Verdict(
        call,
        free.said(options[0].counted),
        pair(logged),
        pair(first),
        agrees=logged in made,
    )


def assert_planned_as_read(verdicts: list[Verdict]) -> None:
    """Fail unless each call's plan is the planner's for its own reading, saying each call's
    reading, windows and blocks swapped, and what the planner makes of it."""
    if all(verdict.agrees for verdict in verdicts):
        return
    pytest.fail(
        "a call's plan is not the planner's for the reading that call made:\n"
        + "\n".join(
            f"{verdict.call}: {'as the planner makes it' if verdict.agrees else 'NOT'}; read"
            f" {verdict.reading}; logged [{verdict.logged[0]}] [{verdict.logged[1]}]; the planner"
            f" makes [{verdict.planner[0]}] [{verdict.planner[1]}]"
            for verdict in verdicts
        )
    )


def work_of(tmp_path: Path) -> Path:
    """The work directory of the job the tests' command makes for tmp_path's ours.mkv
    (cli._work_directory), where its manifest is."""
    return tmp_path / "ours.mkv.work"


def host_available() -> int | None:
    """The system memory available now, as the plan reads it (cli._host_available), in bytes."""
    from seedvr2x.runtime import blockswap

    return blockswap.available()


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
    seconds. Taken where no process of ours has the card (this one makes no CUDA context, and the
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
            [sys.executable, str(PROJECT / "tools" / "compare.py"), kind, str(theirs), str(ours)],
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
    numz_padding: bool,
) -> str:
    """Run seedvr2x as upscaled does, the passes `more` doesn't pin pinned untiled, and compare
    its master and float32 frames with the reference's, bit for bit (same_bits). Returns its
    log."""
    log = upscaled(tmp_path, source, resolution, models, *more, numz_padding=numz_padding)
    same_bits(tmp_path, reference)
    return log


@pytest.mark.parametrize(
    ("dit", "vae", "reference"),
    [
        pytest.param(NUMZ_DIT, NUMZ_VAE, "numz", id="numz-7b"),
        pytest.param(
            "seedvr2x_ema_7b_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz",
            id="ours-7b",
        ),
        pytest.param(
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            "seedvr2x_ema_vae_fp16.safetensors",
            "numz_sharp",
            id="ours-sharp",
        ),
    ],
)
def test_milestone1_bit_identical(tmp_path: Path, dit: str, vae: str, reference: str) -> None:
    assert REFERENCE is not None
    models_present(dit, vae)
    job = case("milestone1", reference, "input_rgb.mkv", 1080, reference, "m1")
    referenced(tmp_path, job, (dit, vae), numz_padding=True, required=reference == "numz")
    bit_identical(
        tmp_path, job.source, job.resolution, (dit, vae), job.reference, numz_padding=True
    )


def test_milestone1_unpinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # numz-7b with no tile pinned: the passes are the planner's own choice, and the bits go
    # through it. Decided before any run, by the same command with --plan: the first pass and
    # the plan, its cut list written and no master (cli.py); --plan loads no model today. A card
    # where the plan is refused, or tiles a pass, can't make numz's untiled master: the case is
    # skipped where the driver reports under 40 GiB free, saying the refusal, or the plan's line
    # of the pass it tiles (planner.said), and fails where it reports more (unpinned). Else the
    # encode's and the decode's lines say untiled, as numz ran them, and nothing is pinned; the
    # run then makes a plan of its own, and its bits are numz's. The run resumes the job the
    # --plan call made, so its first pass is read in that call's log (upscaled).
    # The two calls are not compared with each other: a desktop that drives the card moves its
    # free memory between them (0.3 GiB on an RTX 5090 under WSL2, 5 blocks swapped in one call
    # and 4 in the other), and a plan that follows it is right. Each call's plan is compared with
    # the planner's for the reading that call made (compare_with_planner): the --plan call's from
    # the manifest it leaves (exact), the run's from its --verbose log, its manifest being gone
    # with its work directory once the file is assembled (design, 2026-10-30). That the two calls
    # read the same memory on a steady card is test_plan_and_run_read_the_same_free_memory.
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    numz = case("milestone1", "numz", "input_rgb.mkv", 1080, "numz", "m1")
    referenced(tmp_path, numz, (NUMZ_DIT, NUMZ_VAE), numz_padding=True, required=True)
    job = (tmp_path, numz.source, numz.resolution, (NUMZ_DIT, NUMZ_VAE))
    arguments, environment = command(*job, numz_padding=True, pin_untiled=False)
    listed = tmp_path / "cuts.txt"
    host_planned = host_available()
    plan = subprocess.run(
        [*arguments, "--plan", str(listed)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    # A plan refused: exit status 1, the refusal the log's error, after the logger's name
    # (cli._run). Any other failure fails the case.
    # A line without --verbose is "12:00:00 ERROR message", this command's.
    errors = [
        found[1]
        for line in plan.stderr.splitlines()
        if (found := re.match(r"\d\d:\d\d:\d\d ERROR (.*)", line))
    ]
    refusals = [error for error in errors if any(what in error for what in REFUSED)]
    # Another card than ROOT_CARD, whose reference is made untiled at the size its untiled passes
    # fit (the module's docstring): its plan there refused, or tiling a pass, skips the case,
    # saying it; ROOT_CARD's is held to UNTILED_FREE (unpinned). A refusal records no reading:
    # its figure is the one its words give (reported).
    other = card() != ROOT_CARD
    if plan.returncode == 1 and refusals and other:
        pytest.skip(f"{card()}: the plan is refused: {refusals[0]}")
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
    if tiled and other:
        pytest.skip(f"{card()}: the plan tiles a pass: plan: {tiled[0]}")
    if tiled:
        unpinned(
            plan.stderr,
            f"the plan tiles a pass: plan: {tiled[0]}",
            free_planned.reported / memory.GIB,
        )
    assert passes[0].startswith("encode: untiled, "), planned
    assert passes[1].startswith("decode: untiled, "), planned
    assert listed.is_file() and not (tmp_path / "ours.mkv").exists()
    host_ran = host_available()
    log = upscaled(*job, "-v", numz_padding=True, pin_untiled=False, planned=plan.stderr)
    free_ran = logged_free(log)
    assert free_ran is not None, f"no free memory read in the run's --verbose log: {log[-3000:]}"
    assert_planned_as_read(
        [
            compare_with_planner(
                "--plan", plan.stderr, free_planned, inputs, host_planned, monkeypatch
            )
        ]
    )
    # The run's own passes: the card it planned for may hold less than the --plan call's did.
    ran = plan_lines(log)
    run_passes = [line for line in ran if line.startswith(("encode: ", "decode: "))]
    assert len(run_passes) == 2 and any(line.startswith("pinned: nothing") for line in ran), ran
    run_tiled = [line for line in run_passes if "tiles of" in line]
    if run_tiled and other:
        pytest.skip(f"{card()}: the run's plan tiles a pass: plan: {run_tiled[0]}")
    if run_tiled:
        unpinned(
            log,
            f"the run's plan tiles a pass: plan: {run_tiled[0]}",
            free_ran.reported / memory.GIB,
        )
    # After the run's tiling check: a run that tiles where --plan did not skips or fails as
    # unpinned() says, not on the planner comparison (review of baton 3, 2026-10-31).
    assert_planned_as_read(
        [compare_with_planner("the run", log, free_ran, inputs, host_ran, monkeypatch)]
    )
    same_bits(tmp_path, numz.reference)


def test_plan_and_run_read_the_same_free_memory(tmp_path: Path) -> None:
    # The free memory --plan reads and the run reads, back to back on the same case, are the same
    # figure: both read it once the first pass is done and before any model loads, with a first
    # allocation and a warm-up in place (cli._free_memory), the same to the byte for a new job
    # and a resume (its docstring). A gap is a bug. The run is stopped as soon as it has planned,
    # its models never loaded: its reading is read in its --verbose log (logged_free), the --plan
    # call's in the manifest it leaves, copied before the run. The test cannot make the desktop
    # steady: the other programs' memory on the card (NVML's used, with none of our processes on
    # it) is read before the first call, between the two and after the second; where the readings
    # differ and that memory moved by more than STEADY, the desktop moved, not seedvr2x, and the
    # test skips, saying the figures. (Design's check, baton 3, 2026-10-30.)
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    numz = case("milestone1", "numz", "input_rgb.mkv", 1080, "numz", "m1")
    if not numz.source.exists():
        pytest.skip(f"{numz.source}: absent; this module's docstring says how to make it")
    arguments, environment = command(
        tmp_path,
        numz.source,
        numz.resolution,
        (NUMZ_DIT, NUMZ_VAE),
        numz_padding=True,
        pin_untiled=False,
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
                f" GB: uv run hf download {pull.REPO} {DEFAULT_DIT} {DEFAULT_VAE} {pull.DETECTOR}"
                f" --revision {pull.REVISION}"
            )
    job = case("milestone1", "numz_sharp", "input_rgb.mkv", 1080, "numz_sharp", "m1")
    referenced(tmp_path, job, (DEFAULT_DIT, DEFAULT_VAE), numz_padding=True)
    log = bit_identical(
        tmp_path, job.source, job.resolution, None, job.reference, numz_padding=True
    )
    for name in (DEFAULT_DIT, DEFAULT_VAE, pull.DETECTOR):
        assert f"{name}: in the cache, " in log
        pinned = f"its size and SHA-256 as seedvr2x pins them, {pull.REPO} at {pull.REVISION[:8]}"
        assert f"{name}: {pinned}" in log
    assert "downloading" not in log


@pytest.mark.parametrize(
    ("source", "resolution", "reference"),
    [
        pytest.param("input_rgb.mkv", 1080, "numz_pad1080", id="pad1080"),
        pytest.param("input_rgb.mkv", 720, "numz_pad720", id="pad720"),
        pytest.param("input_rgb_956x530.mkv", 1060, "numz_padcols", id="padcols"),
    ],
)
def test_padding_bit_identical(
    request: pytest.FixtureRequest, tmp_path: Path, source: str, resolution: int, reference: str
) -> None:
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    name = request.node.callspec.id
    job = case("padding", name, source, resolution, reference, "m1")
    models = (NUMZ_DIT, NUMZ_VAE)
    referenced(tmp_path, job, models, numz_padding=False)
    bit_identical(tmp_path, job.source, job.resolution, models, job.reference, numz_padding=False)


@pytest.mark.parametrize(
    ("pins", "reference", "grids"),
    [
        pytest.param(
            ("--decode-tile", "1024"),
            "numz_d1024",
            ["tiled decode: 4 tiles of 1024 px sharing 64, 2 across and 2 down,"],
            id="d1024",
        ),
        pytest.param(
            ("--encode-tile", "1344", "--decode-tile", "1024"),
            "numz_e1344-d1024",
            [
                "tiled encode: 2 tiles of 1344 px sharing 64, 2 across and 1 down,",
                "tiled decode: 4 tiles of 1024 px sharing 64, 2 across and 2 down,",
            ],
            id="e1344-d1024",
        ),
        pytest.param(
            ("--encode-tile", "512", "--decode-tile", "512"),
            "numz_e512-d512",
            [
                "tiled encode: 15 tiles of 512 px sharing 64, 5 across and 3 down,",
                "tiled decode: 15 tiles of 512 px sharing 64, 5 across and 3 down,",
            ],
            id="e512-d512",
        ),
    ],
)
def test_tiled_bit_identical(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    pins: tuple[str, ...],
    reference: str,
    grids: list[str],
) -> None:
    # seedvr2x's tiled passes against numz's own tiling, in numz's padding (the module's
    # docstring): the whole shot under each tile here, numz's one batch there. The debug log
    # says each tiled pass's grid, over the 45 frames padded to 1920x1088, its scratch in system
    # memory at that size (564,019,200 bytes).
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    job = case("tiled", request.node.callspec.id, "input_rgb.mkv", 1080, reference, "m3")
    models = (NUMZ_DIT, NUMZ_VAE)
    referenced(tmp_path, job, models, *pins, numz_padding=True)
    log = bit_identical(
        tmp_path, job.source, job.resolution, models, job.reference, *pins, "-v", numz_padding=True
    )
    # The job's own passes: those of its time estimates' calibration come before, said too
    # (calibration.calibrate, which every run makes: DESIGN.md, Memory planner, Time estimates).
    job_lines = log.split("time estimates: the job's own steps timed once, in", 1)[-1]
    said = [
        line.split("tiling: ", 1)[1] for line in job_lines.splitlines() if "tiling: tiled " in line
    ]
    # At 1080, or at the card's own size where its cases.json gives one, with what each tiled
    # pass's line says there (case).
    grids = list(job.grids) if job.grids is not None else grids
    over = job.over or (
        "over 45 frames padded to 1920x1088; its scratch, 564019200 bytes, in system memory"
    )
    assert len(said) == len(grids), said
    for line, grid in zip(said, grids, strict=True):
        assert line.startswith(grid), said
        assert line.endswith(over), said
    assert not (tmp_path / "ours.mkv.work").exists()
