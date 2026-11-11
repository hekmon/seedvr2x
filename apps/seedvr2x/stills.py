"""The command's path for a still picture (DESIGN.md, Still images; Repository and apps, step 3):
a file the library's reader identifies by its content as a picture (main._picture) is read, planned,
upscaled by the library's one-picture pass and written as a PNG or a lossless WebP, none of a job's
machinery (shots, windows, segments, manifest, resume) and no ffmpeg. main.py branches here before
its build check.

The module imports no torch at its top (main imports it for the parser's constants, and the
refusals that need no GPU come before anything loads): the picture's run imports it, after
memory.startup chose the allocator, as main._run does. What main reads of the machine
(the memory free, the host's, the DiT's format, the load) is main's, handed
in as `Reads`, so that one function replaces each for a test as it does for a video's run."""

import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seedvr2x.errors import shell_word
from stop import terminable

if TYPE_CHECKING:
    import argparse
    from fractions import Fraction

    import torch

    from seedvr2x.model import Models
    from seedvr2x.passes import ModelPlan
    from seedvr2x.picture import Picture
    from seedvr2x.pull import ModelFiles
    from seedvr2x.record import Record
    from seedvr2x.still_plan import PictureTiles, StillPlan
    from seedvr2x.weights import Architecture

logger = logging.getLogger("cli.stills")

# --format's value for a picture's 8-bit PNG (DESIGN.md, Still images, Output): the picture's
# `png` is 16-bit, all the model made. It is no writer's format (media.writer.FORMATS), a video
# refuses it (main._video_arguments).
PNG8 = "png8"

# What --plan's value is when the option is given alone (main._parser): a picture's plan, with no
# file, a video's cut list needing one. Told by identity, never by its content.
BARE_PLAN = Path("(--plan alone)")

# The options a picture takes, by their dest in the parser. Every other option is in REFUSED, and
# tests/test_still_cli.py walks the parser to fail on one that is in neither, so that a new option
# can't slip through unclassified (DESIGN.md: "Every other is refused, saying why, never ignored").
# --format is taken for png and png8 alone, and not with a .webp output (_bits), --plan alone
# (_check_options).
TAKEN = frozenset(
    {
        "input",
        "output",
        "format",
        "model_dir",
        "dit_model",
        "vae_model",
        "resolution",
        "plan",
        "seed",
        "color_correction",
        "input_sar",
        "encode_tile",
        "decode_tile",
        "swap_blocks",
        "verbose",
    }
)

# The options a picture refuses, by dest: the flag as the user writes it, and why. "Given" is told
# from the parser's default, None for each of these, False for a flag (a store_true).
REFUSED = {
    "cuts": ("--cuts", "a picture has no shots to cut"),
    "cut_threshold": ("--cut-threshold", "a picture has no shots, so no shot detector runs on it"),
    "segment_cmd": (
        "--segment-cmd",
        "a picture is one PNG file, with no video segments for a command to encode: convert the"
        " PNG afterwards",
    ),
    "quality": (
        "--quality",
        "a shot's repeated runs are for video; a picture is encoded alone, once",
    ),
    "min_segment": ("--min-segment", "a picture has no output segments to merge"),
    "input_matrix": (
        "--input-matrix",
        "a picture's decoder gives its RGB, with no YUV matrix to name",
    ),
    "frame_rate": ("--frame-rate", "a picture has no frame rate"),
    # DESIGN.md, Input, "Banded sources": a deband in torch, written from the algorithm's
    # description, comes for the nodes and pictures later.
    "deband": (
        "--deband",
        "a picture is read without ffmpeg, whose deband filter the option runs; a picture's own"
        " deband isn't built yet",
    ),
    "window": ("--window", "a picture is one latent in the DiT, with no windows to cap"),
    "compile": (
        "--compile",
        "the VAE's compilation pays back over a long job's frames, not over one picture",
    ),
    "until": ("--until", "a picture is one unit, with no gap between units to stop in"),
    "accept_env_change": (
        "--accept-env-change",
        "a picture keeps no job to resume, so no environment to compare",
    ),
    "dump_frames": ("--dump-frames", "the picture is the output file itself"),
}


@dataclass(frozen=True)
class Reads:
    """What a picture's run reads of the machine and does with the models, main's own functions
    (main._reads), as the video's run uses them."""

    free: Callable[["torch.device"], int]  # main._free_memory: the GPU's free memory, in bytes
    host: Callable[[], int | None]  # main._host_available: the system memory available, in bytes
    dit_model: Callable[[Path], "Architecture"]  # main._dit_model: the DiT file's format
    # main._load(model_files, device, model plan, reserve, kept): the models, loaded, with the
    # seconds that took; None once a refusal is logged.
    load: Callable[
        ["ModelFiles", "torch.device", "ModelPlan", Mapping[str, int] | int | None, str],
        "tuple[Models, float] | None",
    ]
    refused: Callable[[BaseException], str]  # main._refused: the GPU's memory refused, in words
    out_of_memory: Callable[[], tuple[type[BaseException], ...]]  # main._out_of_memory


def run(args: "argparse.Namespace", kind: str, reads: Reads) -> int:
    """Upscale the picture args.input (of format `kind`, told by its content) to the PNG
    args.output, or with --plan say the picture's plan and stop; the exit status. Every refusal is
    logged here, one line and status 1, as a job's are."""
    import planner
    from media.ffmpeg import MediaError
    from seedvr2x.errors import Refusal

    try:
        with terminable():
            return _run(args, kind, reads)
    except (MediaError, Refusal) as error:
        logger.error("%s", planner.worded(error))
        return 1
    except reads.out_of_memory() as error:
        # The GPU's memory refused before the models loaded, where the run first works on the
        # device (the first allocation and the warm-up the free memory is read with): another
        # program holding the card (main._run).
        logger.error("%s, before the models loaded: nothing was run", reads.refused(error))
        return 1


def _run(args: "argparse.Namespace", kind: str, reads: Reads) -> int:
    from fractions import Fraction

    import planner
    from job import DEFAULT_RESOLUTION, default_smaller
    from seedvr2x import memory, picture, png, pull, still_plan, webp
    from seedvr2x.errors import JobError, Refusal
    from seedvr2x.frames import check_picture_target, factor_said, output_size, target_size
    from seedvr2x.weights import check_models

    # Refused first, before the file is read and before anything else is asked of the machine:
    # no ffmpeg, no torch, no GPU.
    _check_options(args)
    output: Path = args.output
    bits = _bits(args)
    _check_seed(args.seed)
    _check_output(output, args.input)
    name = str(args.input)
    logger.debug("%s: a %s picture by its content", name, kind)
    # The file whole, then the picture: the library takes bytes, never a path (picture.read).
    try:
        data = args.input.read_bytes()
    except OSError as error:
        raise JobError(f"{name}: {error.strerror}") from None
    pic = picture.read(data, name)
    del data
    if _is_webp(output):
        # Refused with the picture read and nothing else asked of the machine: what a WebP can't
        # hold (a source stored at 16 bits, colour chunks with no profile) is known from the file
        # alone, so no model loads for a picture that can't be written there.
        try:
            webp.check(pic.bits, pic.icc, pic.chunks)
        except Refusal as error:
            raise Refusal(f"{name}: {error}") from None
    # Processed as stored, then turned upright for the file (DESIGN.md, Still images,
    # Orientation): the target is the stored picture's, at the sample aspect --input-sar gives,
    # else square pixels.
    height, width = pic.pixels.shape[:2]
    sample_aspect = args.input_sar or Fraction(1)
    resolution = DEFAULT_RESOLUTION if args.resolution is None else args.resolution
    target = target_size(width, height, sample_aspect, resolution)
    check_picture_target(target, resolution)
    out_height, out_width = output_size(target)
    _check_tile_pins(args, target)
    # The file's size is the model's, its sides swapped by the turn for orientations 5 to 8.
    written = (out_height, out_width) if pic.orientation >= 5 else (out_width, out_height)
    # The factor by the short sides, the source's as displayed (DESIGN.md: the apps agree).
    factor = factor_said((round(width * sample_aspect), height), (out_width, out_height))
    logger.info("%s", _read_said(name, pic, written, bits, _is_webp(output), factor))
    if args.resolution is None and (said := default_smaller(name, width, height, sample_aspect)):
        logger.info("%s", said)
    # The model files, as a video's run has them, minus the shot detector, which a picture never
    # runs: not fetched, its file not checked. The header check first, then seedvr2x's own files
    # by their pinned sizes, their SHA-256 read by their load alone, which proves it at its end,
    # before the picture (DESIGN.md, Still images, Model files; Weights, decided on 2026-11-03).
    model_files = pull.resolve(args.model_dir, args.dit_model, args.vae_model, detector=False)
    dit, vae = (None if file.absent else file.path for file in (model_files.dit, model_files.vae))
    check_models(dit, vae, pull.advice(model_files), None)
    pull.check_pinned(model_files, load=True)
    # The allocator, set before torch is imported (seedvr2x/memory.py), where the run first takes
    # the GPU: one refused before, for its picture, its settings or its model files, asks the card
    # nothing.
    memory.startup()
    import torch

    if not torch.cuda.is_available():
        raise JobError("no CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise JobError("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
    device = torch.device("cuda", 0)
    memory.hold(device, first=True)
    free = memory.reading(reads.free(device))
    host = reads.host()
    if host is None:
        logger.debug("the system memory available was not read: the plan counts on none")
    else:
        logger.debug(
            "the plan counts on %.2f GiB (%d bytes) of system memory available",
            host / memory.GIB,
            host,
        )
    assert dit is not None, "a plan made loads its models"
    pins = still_plan.Pins(args.encode_tile, args.decode_tile, args.swap_blocks)
    split = args.color_correction == "split"

    # The plan for this picture alone, as stored, the way the model sees it (still_plan.plan's
    # `picture`; DESIGN.md, Still images, decided on 2026-11-04): a bound of its output in either
    # orientation proved its pins over the portrait frame too, and refused a pin of 0 on a
    # landscape 1080p picture the plan itself took untiled. The bound is its output's long and
    # short sides, even as an output size always is. PROVISIONAL (impl-still,
    # 2026-11-03: DESIGN.md, Still images, "Measured before it ships"): the process's resident
    # system memory is HOST_RESIDENT, a job's measure, plus the picture's own arrays (_process,
    # counted from their sizes); the tiled passes' scratch is the plan's own.
    def planned(dit_model: "Architecture") -> "StillPlan":
        return still_plan.plan(
            (max(out_height, out_width), min(out_height, out_width)),
            free,
            host,
            process=_process(pic.pixels.shape[:2], (out_height, out_width), bits),
            # The picture goes to the GPU at its stored size and is resized there: its copies
            # are counted (still_plan.COPIES).
            input_pixels=height * width,
            dit_model=dit_model,
            split=split,
            pins=pins,
            picture=target,
        )

    plan, smaller = _planned(
        planned, reads.dit_model(dit), dit.name, model_dir=args.model_dir is not None
    )
    logger.info("plan: card: %s", torch.cuda.get_device_name(device))
    for line in still_plan.said(plan, target, smaller, planner.option):
        logger.info("%s", line)
    if args.plan is not None:
        # No upscale, as a video's --plan: the models are not loaded, nothing is written.
        return 0
    tiles = still_plan.tiles(plan, target)
    loaded = reads.load(model_files, device, plan.models, still_plan.reserve_taken(plan), "")
    if loaded is None:
        return 1
    models, _ = loaded
    # The record the file carries (DESIGN.md, The record in each output), made once the models
    # are loaded: each model file's SHA-256 that of the very bytes its load read, a pinned
    # file's or one seedvr2x doesn't pin, numz's (pull.Hashes.proved).
    made = _record(pic, written, resolution, sample_aspect, model_files, args, tiles)
    started = time.monotonic()
    out = _upscaled(models, pic, target, args.seed, tiles, split, reads)
    if out is None:
        return 1
    # Turned upright after the model, in both formats, with no orientation tag written: the turn
    # is a view, so the writers take it as it is (picture.upright says why).
    turned = picture.upright(out, pic.orientation)
    if _is_webp(output):
        data = webp.write(turned, grey=pic.grey, icc=pic.icc, chunks=pic.chunks, xmp=made.xmp())
    else:
        data = png.write(
            turned,
            bits=bits,
            grey=pic.grey,
            icc=pic.icc,
            chunks=pic.chunks,
            xmp=made.xmp(),
            software=made.creator,
        )
    del out, turned
    _write(output, data)
    from seedvr2x.clock import spent

    logger.info(
        "upscaled in %.1f s; wrote %s: %dx%d, %s%s",
        spent(time.monotonic() - started),
        output,
        written[0],
        written[1],
        _made(pic, bits, _is_webp(output)),
        _kept(pic, orientation=False) + _turned(pic),
    )
    return 0


def _planned(
    planned: "Callable[[Architecture], StillPlan]",
    dit_model: "Architecture",
    dit_file: str,
    *,
    model_dir: bool,
) -> "tuple[StillPlan, str | None]":
    """The picture's plan in the format of its DiT file (`planned` makes it for a format;
    `model_dir`: the run was given --model-dir, which the way out then says), and the plan's line
    of the smaller DiT files that swap fewer blocks (a picture is one latent: no window to
    lengthen), None where there is none: as the video's (planner.plan_naming,
    planner.smaller_swaps), a refusal for memory adds the way out of the smaller files of the
    same 7B (pull.smaller_files) that plan the picture, named with their option and size, and
    the file is never changed by itself (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0
    too"). A pin's refusal (phase "pin") is left as it is."""
    import planner
    from seedvr2x import pull
    from seedvr2x.still_plan import PlanRefusal

    try:
        made = planned(dit_model)
    except PlanRefusal as refused:
        if refused.phase in ("", "pin") or "\n" in refused.words(planner.option):
            raise

        def plans(each: "Architecture") -> bool | None:
            try:
                planned(each)
            except PlanRefusal:
                return None
            return True

        found = pull.smaller_planning(dit_file, dit_model, plans)
        if not found:
            raise
        way = planner.smaller_way([each for each, _ in found], model_dir)
        raise refused.reworded(f"{refused.words(planner.option)}; or {way}") from None
    swapped = made.models.swapped
    if not swapped:
        return made, None

    def fewer(each: "Architecture") -> pull.Swaps | None:
        try:
            other = planned(each).models.swapped
        except PlanRefusal:
            return None
        return pull.Swaps(other) if other < swapped else None

    found = pull.smaller_planning(dit_file, dit_model, fewer)
    if not found:
        return made, None
    return made, planner.smaller_line(pull.Swaps(swapped), found)


def _record(
    pic: "Picture",
    written: tuple[int, int],
    resolution: int,
    sample_aspect: "Fraction",
    model_files: "ModelFiles",
    args: "argparse.Namespace",
    tiles: "PictureTiles",
) -> "Record":
    """The picture's record (record.Record): this command's name and version, the source's size as
    its file stores it and the output's as written (written, width and height, turned upright),
    the short side asked (resolution, given or the default) and the sample aspect used (--input-sar
    or square pixels), the model files by the names given and their SHA-256, the seed, the colour
    correction, the passes' tiles (0 untiled)."""
    from importlib.metadata import version

    from seedvr2x.record import Model, Record

    height, width = pic.pixels.shape[:2]
    dit, vae = (
        Model(file.name, model_files.hashes.sha256(file.path))
        for file in (model_files.dit, model_files.vae)
    )
    return Record(
        app="seedvr2x",
        version=version("seedvr2x"),
        source=(width, height),
        output=written,
        resolution=resolution,
        sample_aspect=sample_aspect,
        dit=dit,
        vae=vae,
        seed=args.seed,
        color_correction=args.color_correction,
        encode_tile=0 if tiles.encode is None else tiles.encode.size,
        decode_tile=0 if tiles.decode is None else tiles.decode.size,
    )


def _upscaled(
    models: "Models",
    pic: "Picture",
    target: tuple[int, int],
    seed: int,
    tiles: Any,
    split: bool,
    reads: Reads,
) -> Any:
    """The picture through the models (still.upscale), (H', W', 3) float32 on the CPU; None once
    a refusal is logged: the GPU's memory refused, or NaN or inf in the decode."""
    from seedvr2x import memory, still

    try:
        # A held run warns where torch refused allocations during the pass (memory.watched).
        with memory.watched(models.device, "the picture's passes", "pixels"):
            return still.upscale(
                models,
                pic.pixels,
                target,
                seed,
                encode_tiles=tiles.encode,
                decode_tiles=tiles.decode,
                correct=split,
            )
    except still.NonFinite as error:
        logger.error("%s, so no picture is written", error)
    except reads.out_of_memory() as error:
        # The plan passed the GPU's memory: refused by the driver, or by torch where seedvr2x
        # holds it under a driver that would give system memory instead (seedvr2x/memory.py).
        logger.error("%s; nothing is written", reads.refused(error))
    return None


def _process(size: tuple[int, ...], output: tuple[int, int], bits: int) -> int:
    """The process's own resident system memory for a picture of `size` (height, width) made into
    `output` (height, width) written at `bits`, in bytes: a video run's HOST_RESIDENT (shot_plan.py,
    2.4 GiB over 6 shots) plus the picture's own arrays, counted from their sizes. PROVISIONAL,
    unmeasured (DESIGN.md, Still images, "Measured before it ships"): the input pixels, (H, W, 3)
    float32; the output and the decode's copy of it on the CPU, 2 of (H', W', 3) float32; the
    samples quantised for the PNG, (H', W', 3) at 2 bytes (16 bits) or 1; and the PNG's bytes held
    twice at most, 2 times the raw size, the worst case of zlib on noise (a grey picture's
    are a third of that, counted as colour all the same)."""
    from seedvr2x.shot_plan import HOST_RESIDENT

    height, width = size
    out_height, out_width = output
    samples = out_height * out_width * 3
    raw = samples * (bits // 8)
    return HOST_RESIDENT + height * width * 3 * 4 + 2 * samples * 4 + raw + 2 * raw


def _check_tile_pins(args: "argparse.Namespace", target: tuple[int, int]) -> None:
    """Refuse a tile pin that covers the picture's own padded frame, as a video's planner does
    (shot_plan.plan), in its words: a covering tile is no tiling, and the tiled passes would run it
    untiled without saying so. still_plan.plan, made for the picture alone, refuses the same pin
    at the same frame; this refuses it before the card is read."""
    from seedvr2x import peaks, still_plan
    from seedvr2x.frames import padded_size
    from seedvr2x.passes import Phase, Tiles

    height, width = padded_size(target)
    pins: tuple[tuple[Phase, int | None], ...] = (
        ("encode", args.encode_tile),
        ("decode", args.decode_tile),
    )
    for phase, pin in pins:
        if pin and peaks.covers((height, width), Tiles(pin)):
            raise still_plan.covering(phase, pin, (height, width))


def _check_seed(seed: int) -> None:
    """Refuse a seed out of numz's range (job.SEED_LIMIT), in a picture's words: the picture is a
    shot of one frame at frame 0 (still.upscale doesn't check it), its noise drawn from the seed
    and its encode seeded with ENCODE_SEED_OFFSET more, so the bound is job.check_seed's for a
    shot at frame 0. The words are not the video's, which speak of shots and a source."""
    from job import ENCODE_SEED_OFFSET, SEED_LIMIT
    from seedvr2x.errors import JobError

    highest = SEED_LIMIT - 1 - ENCODE_SEED_OFFSET
    if not 0 <= seed <= highest:
        raise JobError(
            f"--seed {seed}: a picture's noise is drawn from its seed and its encode seeded with"
            f" {ENCODE_SEED_OFFSET:,} more, so the seed is from 0 up to {highest:,}"
        )


def _given(args: "argparse.Namespace", dest: str) -> bool:
    """Whether the option was given: its value is not the parser's default (None, or False for a
    flag)."""
    return getattr(args, dest) not in (None, False)


def _check_options(args: "argparse.Namespace") -> None:
    """Refuse every option a picture has no use for, all at once, each saying why
    (DESIGN.md, Still images, Options taken); --plan with a file, which would be a cut list."""
    from seedvr2x.errors import JobError

    reasons = [f"{flag}: {why}" for dest, (flag, why) in REFUSED.items() if _given(args, dest)]
    if args.plan is not None and args.plan is not BARE_PLAN:
        reasons.append(
            f"--plan {args.plan}: a picture has no cut list to write; --plan alone says its plan"
        )
    if not reasons:
        return
    if len(reasons) == 1:
        raise JobError(f"{reasons[0]} (refused for a picture)")
    listed = "; ".join(f"({number}) {reason}" for number, reason in enumerate(reasons, 1))
    raise JobError(f"{len(reasons)} options refused for a picture: {listed}")


def _is_webp(output: Path) -> bool:
    """Whether -o names a lossless WebP file (the suffix in any case): else it is a PNG, which
    _check_output has held to be one of the two."""
    return output.suffix.lower() == ".webp"


def _bits(args: "argparse.Namespace") -> int:
    """The output's depth: for a PNG, 16, all the model made, for png (the default); 8 for png8,
    rounded to the nearest code; for a .webp, 8, the only depth a lossless WebP holds, and the
    format is the file's, so --format with it is refused (DESIGN.md, Still images, Output). A
    video's format is refused."""
    from seedvr2x.errors import JobError

    if _is_webp(args.output):
        if args.format is not None:
            raise JobError(
                f"--format {args.format}: with a .webp output the format is the file's, a lossless"
                f" WebP at 8 bits; --format is for a .png output (png, 16 bits, or {PNG8}, 8 bits)"
            )
        return 8
    if args.format in (None, "png"):
        return 16
    if args.format == PNG8:
        return 8
    raise JobError(
        f"--format {args.format}: a video's format; a picture is written as a PNG, --format png"
        f" (16 bits, the default) or --format {PNG8} (8 bits)"
    )


def _check_output(output: Path, source: Path) -> None:
    """Refuse an -o a picture can't go to, before any work (DESIGN.md, Still images, Output): a
    .png or .webp file, new, beside a directory that exists; a directory, the source itself or a
    file already there is refused, nothing replaced."""
    from seedvr2x.errors import JobError

    if output.is_dir():
        raise JobError(f"{output}: a directory; a picture goes to a new .png or .webp file")
    if output.suffix.lower() not in (".png", ".webp"):
        # seedvr2x owns no encoder but its PNG and lossless WebP writers (DESIGN.md, Output): the
        # way to the user's format is a converter's, with their names.
        written = output.with_suffix(".png")
        target = output if output.suffix else output.with_suffix(".jpg")
        raise JobError(
            f"{output}: seedvr2x writes a picture as a PNG or a lossless WebP only: write"
            f" {written} (or {output.with_suffix('.webp')}), then convert it, e.g. ffmpeg -i"
            f" {shell_word(str(written))} {shell_word(str(target))}"
        )
    if _same(output, source):
        raise JobError(f"{output}: the source itself; choose another -o")
    if os.path.lexists(output):
        raise JobError(
            f"{output}: a file already there, not replaced; remove it, or choose another -o"
        )
    if not output.parent.is_dir():
        raise JobError(f"{output.parent}: not a directory, so {output.name} can't be written there")


def _same(path: Path, other: Path) -> bool:
    """Whether path and other name one file: the same file when both exist, however each is
    spelled (a link included); else the same place."""
    try:
        return os.path.samefile(path, other)
    except OSError:
        return path.resolve() == other.resolve()


def _write(path: Path, data: bytes) -> None:
    """The file's bytes (PNG or WebP) at path, written whole as a finished file is
    (media.files.write_whole): a partial file beside it, then renamed. Nothing is left on a
    failure, a stop included; a file that has come since the check is not replaced."""
    from media.files import partial_path, write_whole
    from seedvr2x.errors import JobError

    if os.path.lexists(path):
        raise JobError(f"{path}: written to since it was checked, by another program: not replaced")
    try:
        write_whole(path, data)
    except OSError as error:
        raise JobError(f"{path}: not written: {error.strerror}") from error
    except BaseException:
        # Ctrl-C or SIGTERM (terminable) inside the write: write_whole removes its partial file
        # for an OSError alone.
        partial_path(path).unlink(missing_ok=True)
        raise


def _read_said(
    name: str, pic: "Picture", size: tuple[int, int], bits: int, lossless: bool, factor: str
) -> str:
    """The line saying what was read and what is made of it, the upscale factor beside the
    output's size (frames.factor_said)."""
    height, width = pic.pixels.shape[:2]
    made = "a lossless WebP, 8-bit" if lossless else f"a {bits}-bit PNG"
    return (
        f"{name}: a {pic.format} picture, {width}x{height}, {pic.bits}-bit"
        f" {'grey' if pic.grey else 'RGB'}{_kept(pic)}; output {size[0]}x{size[1]} ({factor}),"
        f" {made}"
    )


def _made(pic: "Picture", bits: int, lossless: bool) -> str:
    """What the result line says of the file: its depth and kind, and for a WebP that it is
    lossless, with a grey picture's three equal planes (a WebP has no grey mode)."""
    if lossless:
        return "8-bit lossless WebP, " + ("grey as three equal planes" if pic.grey else "RGB")
    return f"{bits}-bit {'grey' if pic.grey else 'RGB'}"


def _kept(pic: "Picture", *, orientation: bool = True) -> str:
    """What of the picture the output carries besides its pixels, as a clause: ", with its ICC
    profile and colour chunks", or "" when there is none (DESIGN.md, Still images, Colour,
    Metadata). The read line adds the source's orientation (orientation); the output has none
    to carry, _turned says it was turned."""
    carried = ["its ICC profile"] if pic.icc is not None else []
    carried += ["its colour chunks"] if pic.chunks else []
    if orientation and pic.orientation != 1:
        carried.append(f"its orientation, {pic.orientation}")
    return f", with {' and '.join(carried)}" if carried else ""


def _turned(pic: "Picture") -> str:
    """The result line's clause saying the picture was turned upright, whatever the format, and
    nothing at orientation 1 (DESIGN.md, Still images, Orientation)."""
    if pic.orientation == 1:
        return ""
    return f", turned upright from its orientation {pic.orientation}"
