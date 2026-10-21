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
untiled, as numz ran them; the run then takes that plan, and the
master and frames are numz's bit for bit. A card that can't run untiled can't make numz's untiled
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

The run goes through the CLI in a process of its own, as a user's would: the allocator is set
before torch is imported, which a test process importing torch first wouldn't do. And this
process creates no CUDA context: its memory would be taken from the free memory a run reads for
its plan (cli._free_memory).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from seedvr2x.cli import NUMZ_DRAW, NUMZ_PADDING, NUMZ_TAIL
from seedvr2x.runtime import pull

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
    assert run.returncode == 0, run.stderr[-3000:]
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
    assert "1 shot: 0 cuts detected at 0.5" in log, log[-3000:]


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
# figure, said beside the whole card's or in its place; on a driver that refuses, the one figure
# a refusal gives, or the plan's first line (planner._card_said, planner._there).
REPORTED = (
    r"its driver reports ([\d.]+) GiB free to this process",
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


def unpinned(log: str, why: str) -> None:
    """The unpinned case on a card whose plan can't make numz's untiled master, `why` saying
    it, the refusal or the plan's line of its passes: skipped where the driver reported under
    UNTILED_FREE GiB free to the --plan run whose log this is (reported), failed where it
    reported that or more, or where the log doesn't say."""
    free = reported(log)
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
    m1 = Path(REFERENCE) / "m1"
    if reference == "numz_sharp":
        for path in (m1 / f"{reference}.mkv", m1 / f"{reference}_frames"):
            if not path.exists():
                pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    source = m1 / "input_rgb.mkv"
    bit_identical(tmp_path, source, 1080, (dit, vae), m1 / reference, numz_padding=True)


def test_milestone1_unpinned(tmp_path: Path) -> None:
    # numz-7b with no tile pinned: the passes are the planner's own choice, and the bits go
    # through it. Decided before any run, by the same command with --plan: the first pass and
    # the plan, its cut list written and no master (cli.py); --plan loads no model today. A card
    # where the plan is refused, or tiles a pass, can't make numz's untiled master: the case is
    # skipped where the driver reports under 40 GiB free, saying the refusal, or the plan's line
    # of the pass it tiles (planner.said), and fails where it reports more (unpinned). Else the
    # encode's and the decode's lines say untiled, as numz ran them, and nothing is pinned; the
    # run then takes the same
    # plan, and its bits are numz's. The run resumes the job the --plan call made, so its first
    # pass is read in that call's log (upscaled).
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    m1 = Path(REFERENCE) / "m1"
    job = (tmp_path, m1 / "input_rgb.mkv", 1080, (NUMZ_DIT, NUMZ_VAE))
    arguments, environment = command(*job, numz_padding=True, pin_untiled=False)
    listed = tmp_path / "cuts.txt"
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
    if plan.returncode == 1 and refusals:
        unpinned(plan.stderr, f"the plan is refused: {refusals[0]}")
    assert plan.returncode == 0, plan.stderr[-3000:]
    detected(plan.stderr)
    planned = plan_lines(plan.stderr)
    passes = [line for line in planned if line.startswith(("encode: ", "decode: "))]
    assert len(passes) == 2 and any(line.startswith("pinned: nothing") for line in planned), planned
    tiled = [line for line in passes if "tiles of" in line]
    if tiled:
        unpinned(plan.stderr, f"the plan tiles a pass: plan: {tiled[0]}")
    assert passes[0].startswith("encode: untiled, "), planned
    assert passes[1].startswith("decode: untiled, "), planned
    assert listed.is_file() and not (tmp_path / "ours.mkv").exists()
    log = upscaled(*job, numz_padding=True, pin_untiled=False, planned=plan.stderr)
    assert choices(plan_lines(log)) == choices(planned), (plan_lines(log), planned)
    same_bits(tmp_path, m1 / "numz")


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
    m1 = Path(REFERENCE) / "m1"
    for path in (m1 / "numz_sharp.mkv", m1 / "numz_sharp_frames"):
        if not path.exists():
            pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    log = bit_identical(
        tmp_path, m1 / "input_rgb.mkv", 1080, None, m1 / "numz_sharp", numz_padding=True
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
    tmp_path: Path, source: str, resolution: int, reference: str
) -> None:
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    m1 = Path(REFERENCE) / "m1"
    for path in (m1 / source, m1 / f"{reference}.mkv", m1 / f"{reference}_frames"):
        if not path.exists():
            pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    models = (NUMZ_DIT, NUMZ_VAE)
    bit_identical(tmp_path, m1 / source, resolution, models, m1 / reference, numz_padding=False)


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
    tmp_path: Path, pins: tuple[str, ...], reference: str, grids: list[str]
) -> None:
    # seedvr2x's tiled passes against numz's own tiling, in numz's padding (the module's
    # docstring): the whole shot under each tile here, numz's one batch there. The debug log
    # says each tiled pass's grid, over the 45 frames padded to 1920x1088, its scratch in system
    # memory at that size (564,019,200 bytes).
    assert REFERENCE is not None
    models_present(NUMZ_DIT, NUMZ_VAE)
    source, m3 = Path(REFERENCE) / "m1" / "input_rgb.mkv", Path(REFERENCE) / "m3"
    for path in (source, m3 / f"{reference}.mkv", m3 / f"{reference}_frames"):
        if not path.exists():
            pytest.skip(f"{path}: absent; this module's docstring says how to make it")
    models = (NUMZ_DIT, NUMZ_VAE)
    log = bit_identical(
        tmp_path, source, 1080, models, m3 / reference, *pins, "-v", numz_padding=True
    )
    said = [line.split("tiling: ", 1)[1] for line in log.splitlines() if "tiling: tiled " in line]
    assert len(said) == len(grids), said
    for line, grid in zip(said, grids, strict=True):
        assert line.startswith(grid), said
        assert line.endswith(
            "over 45 frames padded to 1920x1088; its scratch, 564019200 bytes, in system memory"
        ), said
    assert not (tmp_path / "ours.mkv.work").exists()
