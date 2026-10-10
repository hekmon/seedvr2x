"""Command line: options and logging."""

import argparse
import fcntl
import logging
import math
import os
import re
import shlex
import shutil
import sys
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.cuts import POSSIBLE, THRESHOLD
from seedvr2x.runtime.job import MIN_SEGMENT
from seedvr2x.runtime.pull import DETECTOR, REPO
from seedvr2x.runtime.stop import Stop, Stopped, Terminated, terminable

if TYPE_CHECKING:
    import torch

    from seedvr2x.media.index import FrameIndex
    from seedvr2x.media.source import Declared, FirstPass, Source
    from seedvr2x.runtime.cuts import Cut
    from seedvr2x.runtime.job import JobError, OutputSegment, Shot
    from seedvr2x.runtime.manifest import Manifest
    from seedvr2x.runtime.pull import ModelFile, ModelFiles

# Video file names, other than Matroska's, that -o refuses: an FFV1 master is a .mkv file, and
# anything else names the directory of the output segments, which such a name would only hide.
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts", ".mts", ".mxf", ".nut", ".mpg"}
)

logger = logging.getLogger("seedvr2x")

# numz's padding of the frames before the model, DivisiblePad((16, 16)), zeros up to multiples of
# 16, in place of seedvr2x's (DESIGN.md, Pipeline step 0), for seedvr2x's GPU tests only, never a
# user's: milestone 1's regression holds the output to numz's own with it. It runs the CLI in a
# process of its own, the allocator being set before torch is imported (main), so an environment
# variable reaches it, and no option shows it to users. Read once (_run), passed down as a
# parameter, and recorded in an output directory's settings, so that a job is never resumed in
# the other padding.
NUMZ_PADDING = "SEEDVR2X_TESTS_NUMZ_PADDING"


def main(argv: list[str] | None = None) -> int:
    """Run seedvr2x with argv (sys.argv[1:] when None) and return the exit status."""
    # Hugging Face's telemetry off unless the user set it (DESIGN.md, Weights). huggingface_hub
    # reads the variable once, when imported (constants.py:244-248 in 1.33.0), by runtime/pull.py's
    # lookup in the cache or by diffusers, which the vendored model imports: both after this line,
    # where every entry starts (seedvr2x, seedvr2x verify, python -m seedvr2x). Without it, every
    # request the library makes names torch's version and the AI agent it detects running it, from a
    # registry it fetches from the Hub's /api/agent-harnesses at most once a day and writes to
    # HF_HOME/.agent_harnesses.json (utils/_headers.py:183-189, utils/_detect_agent.py:143-163,
    # 180-203).
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["verify"]:
        return verify(argv[1:])
    parser = argparse.ArgumentParser(
        prog="seedvr2x",
        description="SeedVR2 video upscaler for long runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument("input", type=Path, help="the source, one video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="a .mkv path: one FFV1 master; else a new or empty directory: the output segments"
        " (FFV1 files, or PNG directories) and their manifest, the source cut at its shots,"
        " merged to --min-segment",
    )
    # yuv420p10le by default (DESIGN.md, Output): what every encoder takes, converted here by
    # zscale, exactly, rather than later by sptenc's swscale (16-bit white at 943, not 940). The
    # sizes per hour of 1080p at 24000/1001: DESIGN.md, Output, Master sizes (milestone 5).
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="yuv420p10le",
        help="output format (default: %(default)s): yuv420p10le, FFV1 master in 10-bit YUV 4:2:0"
        " (BT.709 at HD sizes, limited range), what encoders take, converted here exactly by"
        " zscale (white at 940), about 100-135 GiB per hour of 1080p; gbrp16le, FFV1 master in"
        " 16-bit RGB, for precision work, tests and scoring on short samples, 540-690 GiB per hour"
        " of 1080p with colour correction; png, 16-bit PNG",
    )
    # Optional (DESIGN.md, Weights, decided 2026-10-09): without it, seedvr2x's own files come
    # from Hugging Face's cache, pulled from its repo when missing (runtime/pull.py).
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="a directory holding the model files, each read there by its name, the shot"
        f" detector's {DETECTOR} among them unless --cuts is given; nothing is downloaded."
        " Without it, --dit-model and --vae-model name seedvr2x's own files, taken from Hugging"
        f" Face's cache (HF_HOME moves it) with the shot detector's, downloaded into it from {REPO}"
        " at a pinned revision when missing (HF_HUB_OFFLINE keeps the network out). Either way,"
        " seedvr2x's own files are checked by their pinned size and SHA-256",
    )
    # The sharp 7B by default, on the user's eyes: preferred or alike on 67 of 75 windows
    # (DESIGN.md, Weights). A file is recognised by its tensors, not its name (runtime/weights.py).
    parser.add_argument(
        "--dit-model",
        default="seedvr2x_ema_7b_sharp_fp16.safetensors",
        help="the DiT's file: SeedVR2's 7B DiT, regular or sharp, in fp16, recognised by its"
        " tensors; without --model-dir, seedvr2x_ema_7b_fp16.safetensors or the default"
        " (default: %(default)s, the sharp 7B)",
    )
    parser.add_argument(
        "--vae-model",
        default="seedvr2x_ema_vae_fp16.safetensors",
        help="the VAE's file: SeedVR2's VAE in fp16, recognised by its tensors (default:"
        " %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=_positive,
        default=1080,
        help="short side of the output, square pixels at the source's display aspect",
    )
    # Without it, the shot detector runs in the first pass, its probabilities recorded, and the
    # cuts derive from them and --cut-threshold (DESIGN.md, Shot detection).
    parser.add_argument(
        "--cuts",
        type=Path,
        metavar="FILE",
        help="cut list: the first frame of each shot but the first, one frame number per line,"
        " counted from 0, # comments, the fields after the number ignored, as --plan writes it;"
        " in place of the shot detector (default: the cuts TransNetV2 detects, see"
        " --cut-threshold)",
    )
    # The user's proposal, 2026-10-06 (DESIGN.md, Shot detection): erring lower is the cheaper
    # mistake, a miss costing fidelity on four cuts of six, a false cut at most a low-frequency
    # step on a continuous shot (research/docs/cuts.md, What it means for shot detection).
    parser.add_argument(
        "--cut-threshold",
        type=_threshold,
        metavar="P",
        help="the shot detector's threshold, a probability above 0 and at most 1: a cut after the"
        " peak of each run of frames reaching it; lower where cuts go missing, higher where false"
        f" ones show (default: {THRESHOLD}); refused with --cuts, whose list replaces the"
        " detection",
    )
    # PROVISIONAL (implementation, 2026-10-09): DESIGN.md names --plan (Memory planner, Shot
    # detection) without saying where its cut list goes: to a path given, so that --plan works
    # with a one-file output as with a directory, and the list is the user's to edit and keep,
    # never among the job's own files.
    parser.add_argument(
        "--plan",
        type=Path,
        metavar="FILE",
        help="plan the job and stop before the upscale: the first pass, then the cut list"
        f" written to FILE, for editing and --cuts, the possible cuts (runs peaking from {POSSIBLE}"
        " up to the threshold) commented with their probabilities and times; and the counts"
        " printed. With an output directory, the job's manifest and frame index too: the same"
        " command without --plan then runs it, another --cut-threshold or --cuts plans it again,"
        " with no second decode",
    )
    parser.add_argument("--seed", type=int, default=42)
    # split, the colour study's winner, in place of numz's lab (DESIGN.md, Colour correction).
    parser.add_argument(
        "--color-correction",
        choices=("split", "none"),
        default="split",
        help="split: the input's coarse lightness and colour under the model's details, frame by"
        " frame; none: the model's colours, which drift (default: %(default)s)",
    )
    parser.add_argument(
        "--min-segment",
        type=_seconds,
        default=MIN_SEGMENT,
        metavar="SECONDS",
        help="output segments last this long at least, each shorter one merged into its shorter"
        " neighbour, as sptenc's -L (default: %(default)s; 0 keeps every cut)",
    )
    parser.add_argument(
        "--input-matrix",
        choices=sorted(MATRICES),
        help="matrix of a YUV source, when its tag is missing or wrong",
    )
    parser.add_argument(
        "--input-sar",
        type=_ratio,
        metavar="N:D",
        help="sample aspect ratio of the source, when its tag is missing or wrong",
    )
    # An override like the two above (DESIGN.md, Input: exact rational frame rate), for a file
    # whose timestamps contradict the rate it declares; the refusal of such a file says which.
    parser.add_argument(
        "--frame-rate",
        type=_rate,
        metavar="N/D",
        help="frame rate of the source, when its timestamps contradict the one it declares, N/D or"
        " a whole number (24000/1001, 25): its frames taken at that rate, only if every one lies"
        " within half a frame of that rate's timeline",
    )
    parser.add_argument(
        "--window",
        type=int,
        metavar="LATENTS",
        help="cap on the DiT windows, in latents of 4 frames (default: the shot in one window)",
    )
    parser.add_argument(
        "--accept-env-change",
        action="store_true",
        help="resume a job whose environment changed (the GPU, Python, torch or another package,"
        " ffmpeg or its conversions), recorded in its manifest: the output then differs from an"
        " uninterrupted run's",
    )
    parser.add_argument(
        "--dump-frames",
        type=Path,
        metavar="DIR",
        help="also save every output frame as float32 frame_NNNNNN.npy, as ffv1_out.py does",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # numz's allocator, which DESIGN.md keeps (Allocator): set before torch is imported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    # A stop at once, a second Ctrl-C or SIGTERM during the run (runtime/stop.py), SIGTERM during
    # the model files' fetch (stop.terminable), or Ctrl-C before the run: what is kept is what the
    # manifest says, and the same command resumes.
    try:
        return _run(args)
    except KeyboardInterrupt:
        logger.error("stopped at once (Ctrl-C)")
        return 130
    except Terminated:
        logger.error("stopped at once (SIGTERM)")
        return 143
    finally:
        _remove_work()
        _unlock()


def verify(argv: list[str]) -> int:
    """seedvr2x verify: decode a job's output and check every frame against the checksums written
    with it (DESIGN.md, Output, Checksums): an output directory's segments, one of them, or a
    one-file output. Exit status 1 when a frame isn't as written, checksums are missing, or a
    segment isn't finished: the output isn't to be trusted whole yet."""
    parser = argparse.ArgumentParser(
        prog="seedvr2x verify",
        description="Check every frame of a job's output against the checksums written with it.",
    )
    parser.add_argument(
        "output", type=Path, help="an output directory, one of its segments, or a one-file output"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        return _verify(args.output)
    except KeyboardInterrupt:
        logger.error("stopped (Ctrl-C)")
        return 130


@dataclass(frozen=True)
class _Checked:
    """What verify checks: an output segment's file or PNG directory, or a one-file output; the
    pixel format of the frames as it holds them; their size; its checksums, None for a segment not
    finished."""

    path: Path
    pix_fmt: str
    size: tuple[int, int]
    checksums: Path | None
    png: bool = False


def _verify(output: Path) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.checksums import check_frames, read_checksums
    from seedvr2x.media.ffmpeg import MediaError, input_args
    from seedvr2x.runtime.job import JobError

    try:
        ffmpeg.found()
        checked = _checked(output)
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1
    wrong = unfinished = 0
    for item in checked:
        if item.checksums is None:
            unfinished += 1
            logger.warning("%s: unfinished, not checked", item.path)
            continue
        try:
            checksums = read_checksums(item.checksums)
            found = _png_listing(item.path, len(checksums)) if item.png else []
            if not found:
                # PNG frames numbered from 0 (writer.PNGWriter), the first one required.
                pattern = ["-start_number", "0", "-start_number_range", "1"]
                reading = (
                    ["-f", "image2", *pattern, "-i", str(item.path / "%06d.png")]
                    if item.png
                    else input_args(item.path)
                )
                found = check_frames(reading, item.pix_fmt, *item.size, checksums)
        except MediaError as error:
            found = [str(error)]
        if found:
            wrong += 1
            logger.error("%s: %s", item.path, "; ".join(found))
        else:
            logger.info("%s: every frame as written", item.path)
    if wrong or unfinished:
        counts = [f"{wrong} not as written"] if wrong else []
        counts += [f"{unfinished} unfinished"] if unfinished else []
        logger.error("%s: of %d, %s", output, len(checked), " and ".join(counts))
        return 1
    logger.info("%s: %d checked, every frame as written", output, len(checked))
    return 0


def _checked(output: Path) -> list[_Checked]:
    """What verify checks of output: the segments its manifest names, or the one it is, or the
    one-file output it is."""
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.probe import probe
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.manifest import NAME, SUMS, checksums_file, read

    directory, only = (output, None) if output.is_dir() else (output.parent, output.name)
    if not (directory / NAME).is_file():
        if output.is_dir():
            raise JobError(f"{output}: no {NAME}, so not an output directory of seedvr2x")
        # No frame read: verify decodes every one, and its report says what doesn't decode.
        stream = probe(output, first_frame=False)
        if stream.pix_fmt not in ("gbrp16le", "yuv420p10le"):
            raise MediaError(f"{output}: {stream.pix_fmt}, not a master seedvr2x writes")
        sums = output.with_name(f"{output.name}.crc32")
        return [_Checked(output, stream.pix_fmt, (stream.width, stream.height), sums)]
    content = read(directory / NAME)
    try:
        output_format = str(content["output"]["format"])
        width, height = (int(side) for side in content["output"]["size"])
        segments = [(str(entry["name"]), bool(entry["finished"])) for entry in content["segments"]]
    except (KeyError, TypeError, ValueError):
        raise JobError(f"{directory / NAME}: not a manifest seedvr2x wrote") from None
    if only is not None:
        segments = [segment for segment in segments if segment[0] == only]
        if not segments:
            raise JobError(f"{output}: not one of the segments of {directory}")
    png = output_format == "png"
    return [
        _Checked(
            directory / name,
            "rgb48be" if png else output_format,
            (width, height),
            directory / SUMS / checksums_file(name, output_format) if finished else None,
            png,
        )
        for name, finished in segments
    ]


def _png_listing(directory: Path, count: int) -> list[str]:
    """What is wrong with a PNG segment's files, before its frames are decoded: each of its
    `count` frames there, NNNNNN.png from 0, and nothing else."""
    if not directory.is_dir():
        return ["missing"]
    names = {entry.name for entry in directory.iterdir()}
    expected = {f"{index:06d}.png" for index in range(count)}
    missing, foreign = sorted(expected - names), sorted(names - expected)
    found = [f"{len(missing)} PNG missing, from {missing[0]}"] if missing else []
    found += [f"not its frames: {', '.join(foreign[:5])}"] if foreign else []
    return found


def _run(args: argparse.Namespace) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.fingerprint import fingerprint
    from seedvr2x.media.source import counted, declare
    from seedvr2x.runtime import pull
    from seedvr2x.runtime.job import (
        SHARED,
        JobError,
        Shot,
        check_seed,
        check_target,
        output_size,
        read_cuts,
        shots_from_cuts,
        target_size,
    )
    from seedvr2x.runtime.manifest import NAME, planned, read
    from seedvr2x.runtime.weights import ModelError, check_models

    # The build, the source, the target, the cut list and the model files are checked before
    # anything touches the GPU, but for a job resumed, which is first checked against its record
    # (its settings, environment, the GPU's included, and input), before its first pass, which
    # isn't run again when nothing it depends on changed (DESIGN.md, Pause and resume). The job's
    # GPU is checked before its first pass, whose shot detector runs on it.
    prior: _Prior | None = None
    try:
        # The source is the only input (DESIGN.md, Input): a directory is refused before anything
        # is done, the build's check included, so that nothing is made.
        if args.input.is_dir():
            raise JobError(_directory_refused(args.input))
        numz_padding = _numz_padding()
        if args.window is not None and args.window < 2 * SHARED + 1:
            raise JobError(
                f"--window {args.window}: windows share {SHARED} latents with each neighbour, so"
                f" a window needs at least {2 * SHARED + 1}"
            )
        # DESIGN.md, Shot detection: --cut-threshold "is refused with --cuts".
        if args.cuts is not None and args.cut_threshold is not None:
            raise JobError(
                f"--cut-threshold {args.cut_threshold}: refused with --cuts, whose list replaces"
                " the detection"
            )
        png = args.format == "png"
        ffmpeg_version = ffmpeg.check(("png",) if png else (), ("image2",) if png else ())
        conversions = fingerprint()
        logger.info("ffmpeg %s, its conversions' fingerprint %s", ffmpeg_version, conversions[:16])
        declared = declare(args.input, args.input_matrix, args.input_sar, args.frame_rate)
        # The size the frames are resized to, from what the source declares; refused, before the
        # first pass, when too small to pad (job.check_target).
        stream = declared.stream
        target = target_size(stream.width, stream.height, declared.sample_aspect, args.resolution)
        check_target(target, args.resolution)
        cuts = read_cuts(args.cuts) if args.cuts else []
        # The seed against the last shot known before the first pass, the cut list's, so that a
        # seed out of range is refused at once, before the model files and the first pass, whose
        # shot detector imports torch; against every shot after it (check_seed, below).
        # PROVISIONAL (implementation, 2026-10-09; DESIGN.md doesn't order the checks).
        last = cuts[-1] if cuts else 0
        check_seed(args.seed, [Shot(last, last + 1)])
        directory = _output(args)
        if args.plan is not None:
            _plan_file(args, directory)
        # A job resumed: its record read first, and whether its first pass's would be trusted
        # (_trusted), so that the shot detector's file is fetched and checked only when the
        # detector runs: without --cuts, whose list replaces the detection, and in a first pass
        # run (DESIGN.md, Weights: TransNetV2's weights with the shot detector). Nor for a job
        # recorded with a cut list and asked without one, another job, refused below (_prior)
        # before any first pass: its refusal neither fetches nor hashes a file it wouldn't run.
        recorded: dict[str, Any] | None = None
        kept: FrameIndex | None = None
        untrusted = ""
        if directory is not None and (directory / NAME).is_file():
            _lock(directory)
            recorded = read(directory / NAME)
            kept, untrusted = _trusted(directory, recorded, ffmpeg_version, conversions)
        # A job detecting its shots needs the record's probabilities: a plan cut by a list has
        # none, its first pass run again with the shot detector when it is planned again without
        # one (_prior), which gives the detector its frames (PROVISIONAL, implementation,
        # 2026-10-09: the decode is the one way to the frames, a full pass's cost, said in the
        # log). A cut list takes the record with or without them.
        reprobed = args.cuts is None and kept is not None and kept.probabilities is None
        trusted = None if reprobed else kept
        # The detector's file isn't asked for a job recorded with a cut list that has made units,
        # asked without one: another job, which no plan's change takes (_prior).
        detects = (
            args.cuts is None
            and trusted is None
            and (_detecting(recorded) or (recorded is not None and planned(recorded)))
        )
        # The model files, from --model-dir, or seedvr2x's own from Hugging Face's cache, pulled
        # into it when missing, in one fetch; checked by their headers, in a second, then
        # seedvr2x's own by their pinned size and SHA-256 (16.5 GB to read for the DiT), which the
        # manifest's record takes: before the resume's comparison and the first pass, which
        # decodes the whole source (DESIGN.md, Weights). The header check comes first, as
        # DESIGN.md has it (Weights, Recognised by content); its refusal of one of seedvr2x's own
        # files, given in its own role, says how to fetch it again, as the pin's does
        # (pull.advice). SIGTERM, which kills the process at once until the run installs its
        # handlers (runtime/stop.py), unwinds a download as Ctrl-C does, so that the library
        # removes its partial file, up to 16.5 GB that no later download takes up
        # (runtime/pull.py, at its end).
        with terminable():
            model_files = pull.resolve(
                args.model_dir, args.dit_model, args.vae_model, detector=detects
            )
        detector = model_files.detector
        check_models(
            *(model_files.dit.path, model_files.vae.path, pull.advice(model_files)),
            None if detector is None else detector.path,
        )
        pull.check_pinned(model_files)
        if recorded is not None:
            assert directory is not None
            prior = _prior(
                *(args, directory, recorded, trusted, cuts, declared, ffmpeg_version),
                *(conversions, numz_padding, model_files),
            )
            identity = prior.identity
        else:
            identity = _identity(
                *(args, cuts, directory, ffmpeg_version, conversions), numz_padding, model_files
            )
        if prior is not None and prior.known is not None:
            found = prior.known
        else:
            if untrusted:
                # Said once the job is known for the one recorded, never before a refusal.
                logger.warning("%s; the first pass runs again, which makes it", untrusted)
            elif reprobed:
                logger.info(
                    "%s: no shot probabilities in its record, its cuts given by a list: the first"
                    " pass runs again, the shot detector with it",
                    directory,
                )
            # The source's content hashed meanwhile, for the manifest.
            found = _first_pass(declared, directory is not None, model_files, identity.device)
        source = counted(declared, found)
        found_cuts = _source_cuts(args, cuts, source)
        shots = shots_from_cuts([cut.frame for cut in found_cuts.cuts], source.frames)
        logger.info("%s: %s", _plural(len(shots), "shot"), _said(found_cuts))
        check_seed(args.seed, shots)
        segments, paths = _segments(args, source, shots, directory)
        out_height, out_width = output_size(target)
        logger.info(
            "%s; output %dx%d, square pixels",
            _plural(len(segments), "output segment"),
            out_width,
            out_height,
        )
        # No work directory for a plan, which writes nothing but its cut list for one file.
        split = args.color_correction == "split"
        work = _work(args.output) if directory is None and split and args.plan is None else None
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1

    import numpy as np
    import numpy.typing as npt

    from seedvr2x.media.checksums import write_checksums
    from seedvr2x.media.files import make_directories
    from seedvr2x.media.writer import SegmentWriter, Tags, Writer, open_writer
    from seedvr2x.runtime import manifest
    from seedvr2x.runtime.model import load_models
    from seedvr2x.runtime.run import run_job
    from seedvr2x.runtime.shot import CopyError, NonFinite, shot_layout
    from seedvr2x.runtime.units import DiskUnits, Units

    units, record = Units(work), None
    remaining = list(range(len(segments)))  # the segments to write, those not finished
    if directory is not None:
        try:
            if prior is None:
                _lock(directory)
        except JobError as error:
            logger.error("%s", error)
            return 1
        record = manifest.Manifest(
            directory / NAME,
            identity.settings,
            identity.environment,
            source,
            shots,
            [shot_layout(shot.frames, args.window) for shot in shots],
            segments,
            [path.name for path in paths],
            {
                "format": args.format,
                "size": [out_width, out_height],
                "frame_rate": str(source.frame_rate),
            },
        )
        if prior is None and not _empty(directory):
            logger.error("%s: written to since it was checked, by another program", directory)
            return 1
        if prior is not None:
            # The job recorded there, checked whole and taken up before the models load: a plan
            # whose cuts changed planned again (_replan).
            try:
                _resume(record, prior)
            except JobError as error:
                logger.error("%s", error)
                return 1
            remaining = [index for index, done in enumerate(record.finished) if not done]
            if not prior.replanned:
                # The job as recorded: what of it is written before a unit, its first pass run
                # again or its plan taken up in another environment (a plan planned again wrote
                # its own, _replan).
                _taken_up(record, prior)
            if not remaining:
                # Said for --plan too, which still writes the finished job's cut list, its plan.
                logger.info("%s: finished already, its %d segments", directory, len(segments))
                if args.plan is None:
                    return 0
        elif args.plan is not None:
            # A new job's directory made a plan (DESIGN.md, Memory planner): its frame index and
            # manifest, no unit; the same command without --plan runs it.
            _first_write(record, source)
        units = DiskUnits(directory, record)
    if args.plan is not None:
        # No upscale: --plan prints the plan "without running the job" (DESIGN.md, Memory
        # planner), the models not loaded.
        return _write_plan(args, source, found_cuts, len(shots), model_files)
    started = time.monotonic()
    try:
        models = load_models(
            model_files.dit.path, model_files.vae.path, identity.device, model_files.hashes
        )
    except ModelError as error:
        # A model file replaced since its check, during the first pass: refused as the check
        # refuses it, or as changed since its hash.
        logger.error("%s", error)
        return 1
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
    if record is not None and prior is None:
        _first_write(record, source)
    started = time.monotonic()

    tags = Tags.of(stream, source.conversion.matrix_tag)

    def open_segment(path: Path) -> Writer:
        size = (out_width, out_height, source.frame_rate)
        stale = _checksums_path(directory, path, args.format)
        return open_writer(args.format, path, *size, tags, stale=stale)

    def finished(which: int, checksums: list[int]) -> None:
        """A segment whole: its checksums written whole, then the segment recorded (DESIGN.md,
        Output, Checksums)."""
        path = _checksums_path(directory, paths[remaining[which]], args.format)
        make_directories(path.parent)
        write_checksums(path, checksums)
        units.segment_finished(remaining[which])

    # The n-th frame written is the job's frame number(n): the segments left, in order.
    ends = list(accumulate(segments[index].frames for index in remaining))

    def number(written: int) -> int:
        which = bisect_right(ends, written)
        return segments[remaining[which]].start + written - (ends[which - 1] if which else 0)

    outputs = [(paths[index], segments[index].frames) for index in remaining]
    with Stop() as stop:
        try:
            with SegmentWriter(outputs, open_segment, finished) as writer:

                def write(frames: npt.NDArray[np.float32]) -> None:
                    if args.dump_frames is not None:
                        for written, frame in enumerate(frames, writer.written):
                            path = args.dump_frames / f"frame_{number(written):06d}.npy"
                            np.save(path, np.ascontiguousarray(frame))
                    writer.write(frames)

                run_job(
                    *(models, source, shots, segments, target, args.seed, args.window, units),
                    *(write, stop, args.color_correction == "split"),
                    numz_padding=numz_padding,
                )
        except CopyError as error:
            # Derived data, removed (run_job): a resume makes it again.
            again = "" if record is None else "; removed, made again from the input on resuming"
            logger.error("%s%s; stopped: %s", error, again, _kept(record, len(segments)))
            return 1
        except MediaError as error:
            logger.error("%s", error)
            return 1
        except NonFinite as error:
            logger.error("%s, so not recorded; stopped: %s", error, _kept(record, len(segments)))
            return 1
        except Stopped:
            logger.warning(
                "stopped after %s, as asked; %s", stop.unit, _kept(record, len(segments))
            )
            return 130
    if record is not None:
        from seedvr2x.runtime.environment import imported

        # The versions recorded were derived before the models loaded: what the run has
        # imported since must be among them, or a resume wouldn't see it change. imported()
        # leaves out what isn't the run's, pytest or a profiler, as the record does.
        recorded = cast(dict[str, str], identity.environment["packages"])
        unrecorded = sorted(imported() - set(recorded))
        if unrecorded:
            logger.warning("not in the manifest's environment, a seedvr2x bug: %s", unrecorded)
    logger.info(
        "upscaled in %.1f s; wrote %s: %d frames, %s",
        time.monotonic() - started,
        args.output,
        writer.written,
        args.format,
    )
    return 0


def _directory_refused(path: Path) -> str:
    """Why a directory is no input, and what to give instead (DESIGN.md, Input, the user's
    decision of 2026-10-05)."""
    # The source is the only gateway, so that every cut comes from seedvr2x's own detection or a
    # cut list, and every output segment from its rules, never from an outside splitter's joins:
    # of an earlier split's 411 joins, 63 scored under scdet's threshold of 10 and 14 under 4
    # (research/docs/scene-detection.md). The joins given, checked with ffmpeg n9.0.2 on a
    # 2,400-frame FFV1 master at 24000/1001 cut as sptenc's split cuts one (stream copy, the
    # segment muxer, timestamps reset) into 156 Matroska segments: each gives every frame back,
    # identical (framemd5) and in order. ffmpeg's concat demuxer starts each segment where the one
    # before ends by its declared duration, its last timestamp plus its packets' 41 ms, where a
    # frame lasts 41.708: 1 ms early at each of the 155 joins, the last frame 155 ms early. That
    # is not the drift sptenc's ffmpeg/concat.go describes, of encoded segments counting their
    # last frame for 42 ms, late. The frames stay 41 or 42 ms apart, which the first pass
    # accepts, and the output is written at the declared rate, frame by frame, the drift warned
    # of once a frame lies half a frame off that rate's timeline (source._strays); but a default
    # read by ffmpeg (vfr) drops 3 of the 2,400 frames. sptenc's concat, with a duration line
    # per segment and every timestamp snapped to the frame grid (ffmpeg/concat.go), gives the
    # master's timestamps back, every one.
    return (
        f"{path}: a directory; seedvr2x takes one video file, the source, so that every cut comes"
        " from its own detection or a cut list (--cuts), and every output segment from its rules."
        " Give it the file the segments were split from, or join segments split losslessly back"
        " into one file first, with ffmpeg's concat demuxer, which needs no other tool: ffmpeg -f"
        " concat -i list.txt -c copy joined.mkv, list.txt beside them naming each in order, one"
        " line each: file 'seg_000000.mkv'. seedvr2x reads every frame of that join, but its"
        " timestamps drift off the frame grid, and other tools may drop frames. Segments of"
        " sptenc's split can also be joined by sptenc, on the grid:"
        f" sptenc concat {shlex.quote(str(path))} joined.mkv"
    )


def _kept(record: "Manifest | None", segments: int) -> str:
    """What a run stopped before its end keeps, of a job of `segments` output segments."""
    if record is None:
        return "nothing kept: one file can't resume until assembly (milestone 6)"
    return (
        f"{sum(record.finished)} of {segments} segments finished, and the units of the next ones"
        " kept: the same command resumes"
    )


def _work(output: Path) -> Path:
    """The work directory of a one-file output with split, beside it (DESIGN.md, Colour
    correction): its shots' input copies and their checksums, one shot at a time. Made and locked
    from the start, as an output directory is (_lock), so that another run to the same file is
    refused; removed when main returns, however the run ends. What a killed run left there is
    removed; anything else is refused, never deleted."""
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.units import CHECKSUMS, COPY

    work = output.with_name(f"{output.name}.work")
    ours = {COPY, f"{COPY}.partial", CHECKSUMS, f"{CHECKSUMS}.partial"}

    def left(entry: Path) -> bool:
        """Whether entry is a shot's directory, holding nothing but a run's files."""
        return (
            re.fullmatch(r"shot_\d{6}", entry.name) is not None
            and entry.is_dir()
            and not entry.is_symlink()
            and all(f.name in ours and f.is_file() and not f.is_symlink() for f in entry.iterdir())
        )

    refused = f"{work}: not what a run of seedvr2x leaves: remove it, or choose another -o"
    if os.path.lexists(work) and (work.is_symlink() or not work.is_dir()):
        raise JobError(refused)
    _lock(work)
    _WORK.append(work)
    entries = list(work.iterdir())
    if not all(left(entry) for entry in entries):
        _WORK.remove(work)
        raise JobError(refused)
    for entry in entries:
        shutil.rmtree(entry)
    if entries:
        logger.info("%s: left by a run that was killed, emptied", work)
    return work


def _output(args: argparse.Namespace) -> Path | None:
    """The directory the output segments go in, or None for one file. A .mkv path takes the
    source's output whole, one FFV1 master. Else the output is a new or empty directory of
    segments (DESIGN.md, Output). Until assembly (milestone 6), the .mkv path stands for the
    one-file output, and the directory must be new or empty, which keeps another run's files out
    of what sptenc reads (DESIGN.md, Output), or hold a job's manifest, which resume reads back
    (resume.leftovers checks the rest)."""
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.manifest import NAME

    output: Path = args.output
    if output.suffix.lower() in VIDEO_SUFFIXES:
        raise JobError(
            f"{output}: a video file name; an FFV1 master is a .mkv path, and the output segments"
            " go in a directory"
        )
    if output.suffix.lower() == ".mkv":
        if output.is_dir():
            raise JobError(f"{output}: a directory; a .mkv path names the one FFV1 master")
        if args.format == "png":
            raise JobError(f"{output}: PNG output goes to a directory")
        return None
    if output.is_file():
        raise JobError(f"{output}: a file; the output segments need a directory")
    if args.dump_frames is not None and args.dump_frames.resolve().is_relative_to(output.resolve()):
        raise JobError(
            f"--dump-frames {args.dump_frames}: in the output directory, which holds the job's"
            " own files only"
        )
    if output.is_dir() and not _empty(output) and not (output / NAME).is_file():
        raise JobError(
            f"{output}: not empty, and no {NAME} to resume from; the output segments need a new"
            " or empty directory"
        )
    return output


def _segments(
    args: argparse.Namespace,
    source: "Source",
    shots: "Sequence[Shot]",
    directory: Path | None,
) -> "tuple[list[OutputSegment], list[Path]]":
    """The output's segments and each one's path: one file (directory None); else the source's
    shots, merged to --min-segment, and named as sptenc's split names its own (DESIGN.md,
    Output)."""
    from seedvr2x.runtime.job import OutputSegment, merged_segments

    if directory is None:
        return [OutputSegment(args.output.stem, 0, source.frames)], [args.output]
    segments = merged_segments(shots, source.frames, source.frame_rate, args.min_segment)
    suffix = "" if args.format == "png" else ".mkv"
    return segments, [directory / f"{segment.name}{suffix}" for segment in segments]


def _checksums_path(directory: Path | None, output: Path, output_format: str) -> Path:
    """Where the checksums of the output at path `output` go: in the output directory's SUMS, or
    beside the one-file output, its name and .crc32 (DESIGN.md, Output, Checksums)."""
    from seedvr2x.runtime.manifest import SUMS, checksums_file

    if directory is None:
        return output.with_name(f"{output.name}.crc32")
    return directory / SUMS / checksums_file(output.name, output_format)


def _empty(directory: Path) -> bool:
    """Whether directory holds nothing, but what an interrupted first write of the manifest left,
    which the next write replaces: the manifest's partial file, the frame index written whole
    before it (manifest.INDEX), or the index's partial file."""
    from seedvr2x.runtime.manifest import INDEX, NAME

    left = {f"{NAME}.partial", INDEX, f"{INDEX}.partial"}
    return all(entry.name in left for entry in directory.iterdir())


def _write_index(directory: Path, source: "Source") -> None:
    """Write the source's frame index beside the manifest, whole (manifest.INDEX), the shot
    detector's probabilities with it when it ran."""
    from seedvr2x.runtime.manifest import INDEX

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to write")
    source.index.write(directory / INDEX)


def _first_write(record: "Manifest", source: "Source") -> None:
    """A new job's first write in its directory: the frame index, whole, then the manifest naming
    it (manifest.INDEX)."""
    record.path.parent.mkdir(parents=True, exist_ok=True)
    _write_index(record.path.parent, source)
    record.write()


def _first_pass(
    declared: "Declared", hashed: bool, model_files: "ModelFiles", device: "torch.device"
) -> "FirstPass":
    """The first pass over the source, its content hashed meanwhile when `hashed`
    (source.first_pass); with the shot detector when its file was resolved (no --cuts):
    TransNetV2 on the job's device, fed every frame the pass decodes, its memory given back
    before the models load (DESIGN.md, Shot detection; Memory planner, Budget)."""
    from seedvr2x.media.source import first_pass
    from seedvr2x.runtime.detector import Detector

    if model_files.detector is None:
        return first_pass(declared, hashed=hashed)
    with Detector(model_files.detector.path, device, hashes=model_files.hashes) as detector:
        return first_pass(declared, hashed=hashed, detector=detector)


def _settings(
    args: argparse.Namespace, cuts: list[int], numz_padding: bool, model_files: "ModelFiles"
) -> dict[str, object]:
    """The settings a manifest records, those a resume must find again; numz's padding only when
    the tests ask for it (NUMZ_PADDING), so that a user's manifest never names it."""
    from seedvr2x.runtime.manifest import code_sha256

    detects = args.cuts is None
    settings: dict[str, object] = {
        "seedvr2x": version("seedvr2x"),
        "code": code_sha256(),
        "dit_model": _model(model_files.dit, model_files),
        "vae_model": _model(model_files.vae, model_files),
        # The shot detector's file, as the DiT's and the VAE's, when the job detects shots: a
        # resume with another is another job.
        **({"detector_model": _detector_model(model_files)} if detects else {}),
        "resolution": args.resolution,
        "seed": args.seed,
        "color_correction": args.color_correction,
        "window": args.window,
        "format": args.format,
        # How the source is cut (resume.CUT_SETTINGS): the cut list given, or none and the
        # detection's threshold, "a setting the manifest records" (DESIGN.md, Shot detection),
        # as Python writes the float, which reads back exactly; the cuts it gives are the
        # layout's shots, compared after the first pass.
        "cuts": None if detects else cuts,
        **({"cut_threshold": repr(_cut_threshold(args))} if detects else {}),
        "min_segment": str(args.min_segment),
        "input_matrix": args.input_matrix,
        "input_sar": None if args.input_sar is None else str(args.input_sar),
        "frame_rate": None if args.frame_rate is None else str(args.frame_rate),
    }
    if numz_padding:
        settings["numz_padding"] = True
    return settings


def _numz_padding() -> bool:
    """Whether seedvr2x's tests ask for numz's padding (NUMZ_PADDING=1), said when they do;
    refused (JobError) when the variable is set to anything else."""
    from seedvr2x.runtime.job import JobError

    value = os.environ.get(NUMZ_PADDING)
    if value is None:
        return False
    if value != "1":
        raise JobError(f"{NUMZ_PADDING}={value!r}: for seedvr2x's tests only, 1 or unset")
    logger.warning("%s=1: numz's padding, for seedvr2x's tests only", NUMZ_PADDING)
    return True


# The output directories this process holds locked (_lock), until main returns.
_LOCKS: list[int] = []


def _lock(directory: Path) -> None:
    """Lock directory for this process until main returns: one seedvr2x at a time writes an
    output directory, since two would race on the manifest and discard each other's files in
    progress (resume.leftovers). The lock is the kernel's (flock), on the directory itself, so
    no file is added, and it goes with the process, however it ends."""
    from seedvr2x.media.files import make_directories
    from seedvr2x.runtime.job import JobError

    make_directories(directory)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise JobError(f"{directory}: another seedvr2x is writing it") from None
    except OSError as error:
        # A filesystem without locks: logged, the run goes on (AGENTS.md: no silent fallback).
        logger.warning("%s: not locked (%s); let one seedvr2x at a time write it", directory, error)
    _LOCKS.append(descriptor)


# The work directories this process made (_work), removed when main returns.
_WORK: list[Path] = []


def _remove_work() -> None:
    while _WORK:
        shutil.rmtree(_WORK.pop(), ignore_errors=True)


def _unlock() -> None:
    while _LOCKS:
        os.close(_LOCKS.pop())


@dataclass(frozen=True)
class _Identity:
    """What a job's output depends on besides its frames (DESIGN.md, Pause and resume): its
    settings, the models by hash, and the environment, both recorded for an output directory's
    manifest only; and the GPU it runs on."""

    settings: dict[str, object]
    device: "torch.device"
    environment: dict[str, object]


@dataclass(frozen=True)
class _Prior:
    """A job resumed, as checked against its record before its first pass (_prior)."""

    recorded: dict[str, Any]  # its manifest, as written
    identity: _Identity
    changed: list[str]  # its environment's differences, accepted (--accept-env-change)
    known: "FirstPass | None"  # the first pass's record, unless ffmpeg or conversions changed
    # A plan planned again: its cut settings' differences (resume.CUT_SETTINGS); else none.
    replanned: list[str] = field(default_factory=list[str])


def _identity(
    args: argparse.Namespace,
    cuts: list[int],
    directory: Path | None,
    ffmpeg_version: str,
    conversions: str,
    numz_padding: bool,
    model_files: "ModelFiles",
) -> _Identity:
    """The job's identity, refused (JobError) without a CUDA GPU computing in bfloat16; its model
    files are checked before (weights.check_models, pull.check_pinned). The models are hashed
    before any GPU work, each once a run (pull.Hashes)."""
    from seedvr2x.runtime.job import JobError

    settings = _settings(args, cuts, numz_padding, model_files) if directory is not None else {}
    import torch

    if not torch.cuda.is_available():
        raise JobError("no CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise JobError("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
    device = torch.device("cuda", 0)
    if directory is None:
        return _Identity(settings, device, {})
    from seedvr2x.runtime import environment

    return _Identity(settings, device, environment.current(device, ffmpeg_version, conversions))


def _prior(
    args: argparse.Namespace,
    directory: Path,
    recorded: dict[str, Any],
    trusted: "FrameIndex | None",
    cuts: list[int],
    declared: "Declared",
    ffmpeg_version: str,
    conversions: str,
    numz_padding: bool,
    model_files: "ModelFiles",
) -> _Prior:
    """The job recorded in directory, its manifest's content `recorded`, checked against the one
    asked before its first pass: the same settings, environment and source, the source being its
    content (resume.identity), or refused (JobError), but for an environment change accepted
    (--accept-env-change), and for a plan's cuts. The record of the first pass is trusted, and
    the pass isn't run again, with its index `trusted` (_trusted): unless ffmpeg or its
    conversions changed, the same bytes, decoded by the same build, give the same frames, and
    nothing else takes part in the pass's decode (DESIGN.md, Pause and resume); the shot
    detector's probabilities are the record's then, never made again.

    A plan, a job that has made no unit yet (manifest.planned), is planned again when the settings
    asked differ from its own in how its source is cut alone (resume.CUT_SETTINGS): another
    --cut-threshold, a cut list given or another, or none, its cuts derived again from its first
    pass's record (DESIGN.md, Shot detection: another threshold gives its cuts without a second
    decode or detection), the layout and settings rewritten (_replan). PROVISIONAL
    (implementation, 2026-10-09, reported to design), as resume.CUT_SETTINGS says."""
    from seedvr2x.media.scan import Idet
    from seedvr2x.media.source import FirstPass
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import NAME, planned

    path = directory / NAME
    identity = _identity(
        *(args, cuts, directory, ffmpeg_version, conversions), numz_padding, model_files
    )
    asked = {
        "settings": identity.settings,
        "environment": identity.environment,
        "input": _content(declared.path),
    }
    found = resume.differences(resume.identity(recorded), resume.identity(asked))
    changed = [line for line in found if resume.section(line) == "environment"]
    recut = [line for line in found if resume.setting(line) in resume.CUT_SETTINGS]
    other = len(found) > len(changed) + len(recut)
    replanned = recut if recut and not other and planned(recorded) else []
    if other or recut != replanned or (changed and not args.accept_env_change):
        raise _another_job(
            path,
            found,
            only_environment=len(changed) == len(found),
            units_made=bool(recut) and not other and not replanned,
            plan_elsewhere=bool(replanned) and bool(changed),
        )
    entry: dict[str, Any] = recorded["input"]
    if entry["path"] != str(declared.path.resolve()):
        logger.info(
            "%s: the input recorded at %s, moved: the same content", declared.path, entry["path"]
        )
    if trusted is None:
        return _Prior(recorded, identity, changed, None, replanned)
    logger.info(
        "%s: the same input and ffmpeg, the first pass as recorded%s",
        directory,
        "" if trusted.probabilities is None else ", the shot detector's probabilities with it",
    )
    found = FirstPass(
        entry["frames"], entry["sha256"], trusted, Idet.from_record(entry.get("idet"))
    )
    return _Prior(recorded, identity, changed, found, replanned)


def _detecting(recorded: dict[str, Any] | None) -> bool:
    """Whether the job a manifest records detects its shots, its settings naming the shot
    detector's file (_settings); true without a record, a new job detecting unless a cut list is
    given."""
    if recorded is None:
        return True
    settings: Any = recorded.get("settings")
    return isinstance(settings, dict) and "detector_model" in settings


def _trusted(
    directory: Path,
    recorded: dict[str, Any],
    ffmpeg_version: str,
    conversions: str,
) -> "tuple[FrameIndex | None, str]":
    """The frame index of the first pass the manifest `recorded` in directory records, when a
    resume may trust that record (DESIGN.md, Pause and resume): made by the same ffmpeg and
    conversions (manifest.FIRST_PASS), and there as recorded (_recorded_index); else None, the
    first pass then run again, and why, when the index is the reason, for the run to say if the
    pass runs. Read before the model files are, so that a resume trusting it neither fetches nor
    reads the shot detector's file, which it doesn't run, its probabilities there (cli._run);
    whether the job is the one recorded is _prior's to say. Nothing here imports torch, which the
    model files' check comes before (DESIGN.md, Weights, Recognised by content): runtime/resume.py
    does, and isn't read."""
    from seedvr2x.runtime.manifest import FIRST_PASS

    environment: Any = recorded.get("environment")
    entry: Any = recorded.get("input")
    if not isinstance(environment, dict) or not isinstance(entry, dict):
        return None, ""  # not this code's record: _prior refuses the job
    # The environment's fields the first pass depends on, as runtime/environment.py records them.
    current = {"ffmpeg": ffmpeg_version, "conversions": conversions}
    before = cast(dict[str, Any], environment)
    if any(before.get(key) != current[key] for key in FIRST_PASS):
        return None, ""
    # PROVISIONAL (DESIGN.md has the index trusted as the first pass's record is, and says nothing
    # of one missing or damaged): derived data, as an input copy is, never trusted unchecked, nor
    # a reason to refuse the job: the first pass runs again and makes it.
    return _recorded_index(directory, cast(dict[str, Any], entry))


def _recorded_index(directory: Path, entry: dict[str, Any]) -> "tuple[FrameIndex | None, str]":
    """The frame index the input's record names, read from directory and checked against the
    record (manifest.INDEX): its size, SHA-256 and frames; or None, and why it isn't trusted. Its
    shot detector's probabilities, when there, come with it: a record this code wrote has them
    when its job detected its shots, which its settings say (_detecting), or when it is a
    detected plan planned again by a cut list, which keeps them (_replan), so that a later
    threshold needs no second decode."""
    import hashlib

    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime.manifest import INDEX

    path = directory / INDEX
    recorded: Any = entry.get("index")
    if not isinstance(recorded, dict):
        return None, f"{path}: not in the manifest's record of the input"
    named = cast(dict[str, Any], recorded)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None, f"{path}: the frame index missing"
    except OSError as error:
        return None, f"{path}: the frame index not readable: {error}"
    if len(data) != named.get("bytes"):
        return None, f"{path}: {len(data)} bytes, where the manifest recorded {named.get('bytes')}"
    if hashlib.sha256(data).hexdigest() != named.get("sha256"):
        return None, f"{path}: not the frame index the manifest recorded, by its SHA-256"
    try:
        index = FrameIndex.from_bytes(data)
    except ValueError as error:
        return None, f"{path}: not readable as a frame index: {error}"
    if index.frames != entry.get("frames"):
        return None, f"{path}: {index.frames} frames, where the record has {entry.get('frames')}"
    return index, ""


def _content(path: Path) -> dict[str, object]:
    """The source's content, as a resume compares it: its size and SHA-256."""
    from seedvr2x.media.files import sha256

    started = time.monotonic()
    digest = sha256(path)
    logger.info("%s: SHA-256 %s, in %.1f s", path, digest, time.monotonic() - started)
    return {"bytes": path.stat().st_size, "sha256": digest}


def _resume(record: "Manifest", prior: _Prior) -> None:
    """Take up the job recorded beside record, checked before its first pass (_prior): refused
    (JobError) unless the rest is the job asked too, what the first pass found and the layout
    following from it; its directory must be as the manifest says. Then record the environment
    change accepted, and discard what a stop left that the manifest doesn't name
    (resume.leftovers). The manifest is written by the next unit made: a resume stopped
    before one keeps the record as it was, but for the index of a first pass run again and for
    a plan's (_taken_up). A plan planned again (_prior) takes its new cuts and the layout they
    give (_replan)."""
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import planned

    def compared(line: str) -> bool:
        # The environment compared before the first pass; a plan's cuts, accepted there.
        if resume.section(line) == "environment":
            return False
        recut = resume.setting(line) in resume.CUT_SETTINGS
        return not prior.replanned or not (recut or resume.section(line) in ("shots", "segments"))

    found = [
        line for line in resume.differences(prior.recorded, record.content()) if compared(line)
    ]
    if found:
        raise _another_job(record.path, found, only_environment=False)
    if prior.replanned:
        _replan(record, prior)
        return
    resume.adopt(record, prior.recorded)
    if prior.changed:
        change = resume.environment_change(prior.recorded, record.environment)
        record.environment_changes.append(change)
        logger.warning(
            "%s: resumed in another environment, as accepted, which its manifest records %s: %s",
            record.path.parent,
            # A plan's at once (_taken_up).
            "at once, a plan having made no unit in the one before"
            if planned(prior.recorded)
            else "with the next unit made",
            "; ".join(prior.changed),
        )
    discarded = resume.leftovers(record)
    for path in discarded:
        logger.debug("discarded %s", path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    logger.info(
        "resuming %s: %d of %d segments finished, %d of %d shots encoded, %d windows kept; %s"
        " discarded",
        record.path.parent,
        sum(record.finished),
        len(record.finished),
        sum(record.encoded),
        len(record.encoded),
        sum(record.windows_done),
        _plural(len(discarded), "leftover"),
    )


def _taken_up(record: "Manifest", prior: _Prior) -> None:
    """What a job taken up as it is recorded (_resume) writes before any unit: nothing, unless its
    first pass ran again or it is a plan taken up in another environment. The frame index is
    written only with a manifest that names it and records the environment it was made in
    (manifest.INDEX).

    A plan, a job that has made no unit yet (manifest.planned), whose first pass ran again or
    whose environment changed, accepted (--accept-env-change): its index, when made again, then
    its manifest, at once, in this run's environment, the change recorded
    (environment_changes). No unit of it was made in the environment recorded, so nothing is
    left to resume there, and its record then stands for what this run made and found: written
    as recorded, a plan whose pass ran again under another ffmpeg named that ffmpeg's index
    beside the first one's name, and the first ffmpeg then trusted it (a review's finding,
    2026-10-10). PROVISIONAL (implementation, 2026-10-10, for design, as the plan's form is:
    DESIGN.md has the new environment written "with the next unit made").

    A job that has made units, its first pass run again (_prior):

    Its index made in the environment recorded, the recorded one being missing or damaged: at
    once, then the manifest as recorded, naming it, so that the next run trusts the record, that
    of a job finished already too, which makes no unit to write its manifest with. Left to the
    next unit, a finished job kept index bytes its manifest didn't name when the pass gave others
    (a damaged source decodes otherwise from one pass to the next: media/reader.py), and every
    later run warned of them and ran the pass again.

    Made in another environment, accepted (ffmpeg or its conversions changed): with the next unit
    made, whose manifest records that environment (DESIGN.md, Pause and resume), the directory
    keeping the recorded index until then: a resume stopped before a unit "still resumes in its
    old environment", its record whole, and a job finished already, which makes none, keeps what
    it has.

    PROVISIONAL (implementation, 2026-10-10, a question for design: DESIGN.md has a resume stopped
    before a unit leave the manifest as it was, which holds but for the record of an index made
    again in the environment recorded, and for a plan's)."""
    from dataclasses import replace

    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import FIRST_PASS, planned

    directory = record.path.parent
    source = record.source
    if planned(prior.recorded):
        if prior.known is None:
            _write_index(directory, source)
        if prior.known is None or prior.changed:
            record.write()
        return
    if prior.known is not None:
        return
    before: dict[str, Any] = prior.recorded["environment"]
    if any(before.get(key) != record.environment.get(key) for key in FIRST_PASS):
        record.before_write = lambda: _write_index(directory, source)
        return
    _write_index(directory, source)
    as_recorded = replace(
        record,
        environment=before,
        environment_changes=list(prior.recorded.get(resume.CHANGES, [])),
    )
    as_recorded.write()


def _replan(record: "Manifest", prior: _Prior) -> None:
    """Take up a plan with other cuts (_prior): what a stop left of it, found by its own layout
    (resume.recorded_job, resume.leftovers: anything not its own refused, never deleted), is
    discarded, its units' directories with it, none of them the new layout's; its first pass's
    index replaced when the pass ran again (a plan cut by a list, planned again without one, or
    one planned again under another ffmpeg, accepted); then its manifest rewritten with the new
    settings and layout, at once, so that the plan is the new one whether a unit follows or not,
    in this run's environment, a change accepted recorded with it, as a plan taken up is
    (_taken_up). PROVISIONAL, as the rule is (resume.CUT_SETTINGS)."""
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import STATE

    directory = record.path.parent
    planned = resume.recorded_job(record.path, prior.recorded, record.source)
    discarded = resume.leftovers(planned)
    for path in discarded:
        logger.debug("discarded %s", path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    # Only the plan's units' directories are left there, emptied (leftovers): rmdir, which
    # refuses a directory not empty, removes nothing else.
    state = directory / STATE
    if state.is_dir():
        for shot in state.iterdir():
            shot.rmdir()
        state.rmdir()
    record.environment_changes = list(prior.recorded.get(resume.CHANGES, []))
    if prior.changed:
        record.environment_changes.append(
            resume.environment_change(prior.recorded, record.environment)
        )
        logger.warning(
            "%s: planned again in another environment, as accepted, which its manifest records"
            " at once, a plan having made no unit in the one before: %s",
            directory,
            "; ".join(prior.changed),
        )
    if prior.known is None:
        _write_index(directory, record.source)
    record.write()
    logger.info(
        "%s: re-planned, its job having made no unit yet: %s; %s discarded",
        directory,
        "; ".join(prior.replanned),
        _plural(len(discarded), "leftover"),
    )


def _another_job(
    path: Path,
    found: list[str],
    only_environment: bool,
    units_made: bool = False,
    plan_elsewhere: bool = False,
) -> "JobError":
    """A resume refused, each difference listed, `where: recorded -> now`; `units_made` when the
    cuts differ alone, which a plan would take (_prior), but the job has made units;
    `plan_elsewhere` when a plan's cuts differ, which it takes, and its environment too, which
    --accept-env-change takes."""
    from seedvr2x.runtime.job import JobError

    return JobError(
        f"{path}: another job than the one asked, which differs in:\n  "
        + "\n  ".join(found[:20])
        + (f"\n  and {len(found) - 20} more" if len(found) > 20 else "")
        + (
            "\nOnly its environment differs: --accept-env-change resumes it anyway, though"
            " its output then differs from an uninterrupted run's"
            if only_environment
            else ""
        )
        + (
            "\nIts cuts change only while it is a plan, before its first unit, and it has made"
            " some: its cuts are its own; another output directory takes new ones"
            if units_made
            else ""
        )
        + (
            "\nA plan, which has made no unit yet, is planned again with other cuts; its"
            " environment differs too: --accept-env-change takes it up anyway, its first pass"
            " run again where ffmpeg or its conversions changed"
            if plan_elsewhere
            else ""
        )
    )


def _model(file: "ModelFile", model_files: "ModelFiles") -> dict[str, object]:
    """A model as the manifest records it: its file's name, as given, its size and SHA-256, models
    being identified by hash (DESIGN.md, Options kept and dropped). The 7B fp16 DiT is 16.5 GB to
    read, once a run: a file seedvr2x pins was hashed by its check already (pull.Hashes)."""
    digest = model_files.hashes.sha256(file.path)
    return {"name": file.name, "size": file.path.stat().st_size, "sha256": digest}


def _detector_model(model_files: "ModelFiles") -> dict[str, object]:
    """The shot detector as the manifest records it (_model): its file, when the run resolved it;
    else, on a resume whose first pass's record is trusted (_trusted), which runs no detector and
    so neither fetches nor reads its file, the file by its pin: the only one seedvr2x runs in that
    role (pull.check_pinned refuses any other), so the one any run of this code recorded.
    PROVISIONAL (implementation, 2026-10-09; DESIGN.md records the models by hash): settings'
    detector_model, by the pin on such a resume."""
    from seedvr2x.runtime import pull

    if model_files.detector is not None:
        return _model(model_files.detector, model_files)
    pin = pull.PINNED[pull.DETECTOR]
    return {"name": pull.DETECTOR, "size": pin.size, "sha256": pin.sha256}


def _cut_threshold(args: argparse.Namespace) -> float:
    """The detection's threshold: --cut-threshold's, else THRESHOLD."""
    return THRESHOLD if args.cut_threshold is None else args.cut_threshold


@dataclass(frozen=True)
class _Cuts:
    """How a job's source is cut (_source_cuts): its cuts, and the possible ones a detection
    found (--plan's list); the threshold of a detection, as the manifest records it, or the cut
    list given."""

    cuts: "list[Cut]"
    possible: "list[Cut]"
    threshold: str | None  # the detection's, None for a cut list's
    listed: Path | None  # the cut list given, None for a detection's


def _source_cuts(args: argparse.Namespace, listed: list[int], source: "Source") -> _Cuts:
    """The cuts of the job's source: the cut list's, `listed`, as read (--cuts); else detected
    at --cut-threshold from the shot detector's probabilities, its first pass's or its record's
    (DESIGN.md, Shot detection: "the cuts derive from them and the threshold"), with the
    possible ones."""
    from seedvr2x.runtime.cuts import Cut, detected, possible

    if args.cuts is not None:
        return _Cuts([Cut(frame) for frame in listed], [], None, args.cuts)
    probabilities = None if source.index is None else source.index.probabilities
    if probabilities is None:
        # The detector runs in every first pass of a job detecting its shots, and a record
        # without them isn't trusted then (_run).
        raise ValueError(f"{source.path}: no shot probabilities to cut it by: a seedvr2x bug")
    threshold = _cut_threshold(args)
    found = detected(probabilities, threshold), possible(probabilities, threshold)
    return _Cuts(*found, repr(threshold), None)


def _said(found: _Cuts) -> str:
    """How the shots were found, as a run's log says it: "61 cuts detected at 0.3, and 39
    possible cuts from 0.1 up to 0.3, which --plan lists", or "3 cuts from the cut list
    cuts.txt"."""
    if found.threshold is None:
        return f"{_plural(len(found.cuts), 'cut')} from the cut list {found.listed}"
    said = f"{_plural(len(found.cuts), 'cut')} detected at {found.threshold}"
    if float(found.threshold) <= POSSIBLE:
        return said
    possible = _plural(len(found.possible), "possible cut")
    return f"{said}, and {possible} from {POSSIBLE} up to {found.threshold}, which --plan lists"


def _counts(found: _Cuts, shots: int) -> str:
    """The counts --plan prints (DESIGN.md, Shot detection): "61 cuts at 0.3 (62 shots), 39
    possible cuts from 0.1 up to 0.3"; "3 cuts from the cut list cuts.txt (4 shots)"."""
    cuts, shots_said = _plural(len(found.cuts), "cut"), _plural(shots, "shot")
    if found.threshold is None:
        return f"{cuts} from the cut list {found.listed} ({shots_said})"
    said = f"{cuts} at {found.threshold} ({shots_said})"
    if float(found.threshold) <= POSSIBLE:
        # No run from POSSIBLE up peaks under such a threshold (cuts.possible).
        return f"{said}, no possible cuts at a threshold of {POSSIBLE} or under"
    possible = _plural(len(found.possible), "possible cut")
    return f"{said}, {possible} from {POSSIBLE} up to {found.threshold}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _plan_file(args: argparse.Namespace, directory: Path | None) -> None:
    """Refuse a --plan FILE (JobError) that would take the place of what the job reads or writes:
    the source, the cut list given (--cuts), the one-file output or anything in the output
    directory, which holds the job's own files only (resume.leftovers); or that can't be
    written: a directory, in none, or in one that takes no new file. Before the first pass, which
    may take an hour. PROVISIONAL (implementation, 2026-10-09), with the plan's form."""
    import tempfile

    from seedvr2x.runtime.job import JobError

    plan: Path = args.plan
    where = plan.resolve()
    if where == args.input.resolve():
        raise JobError(f"--plan {plan}: the source; it names the cut list's file")
    if args.cuts is not None and where == args.cuts.resolve():
        # The plan of a cut list is that list normalised: written over it, the list's own
        # comments went, the possible cuts it kept commented and their probabilities with them.
        raise JobError(
            f"--plan {plan}: the cut list given (--cuts), which it would replace, its comments"
            " lost; it names another file"
        )
    if directory is None and where == args.output.resolve():
        raise JobError(f"--plan {plan}: the output; it names the cut list's file")
    if directory is not None and where.is_relative_to(directory.resolve()):
        raise JobError(
            f"--plan {plan}: in the output directory, which holds the job's own files only"
        )
    if plan.is_dir():
        raise JobError(f"--plan {plan}: a directory; it names the cut list's file")
    if not plan.parent.is_dir():
        raise JobError(f"--plan {plan}: no directory {plan.parent} to write it in")
    # The list is written beside its place, then renamed into it (files.write_whole): a file made
    # there and removed tells now, not once the first pass has run, whether the directory takes
    # one.
    try:
        with tempfile.NamedTemporaryFile(dir=plan.parent, prefix=f"{plan.name}."):
            pass
    except OSError as error:
        raise JobError(
            f"--plan {plan}: no file can be written in {plan.parent}: {error.strerror}"
        ) from None


def _write_plan(
    args: argparse.Namespace,
    source: "Source",
    found: _Cuts,
    shots: int,
    model_files: "ModelFiles",
) -> int:
    """--plan's end: the cut list written to FILE, whole, an existing one replaced (files.
    write_whole), in the cut list format (cuts.cut_list), and the counts said; 0, or 1 when FILE
    can't be written. PROVISIONAL (implementation, 2026-10-09): the header's wording, and the
    probability and time after each cut, which --cuts ignores (DESIGN.md, Input)."""
    import textwrap

    from seedvr2x.media.files import write_whole
    from seedvr2x.runtime.cuts import cut_list

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to time its cuts by: a seedvr2x bug")
    rate = source.frame_rate
    # No line of the header begins with a digit, nor holds a line break (cuts.cut_list).
    header = [f"seedvr2x cut list of {source.path.name}: {source.frames} frames at {rate} fps"]
    if found.threshold is not None:
        model = _detector_model(model_files)
        header.append(
            f"Shot detector: TransNetV2, {model['name']} (SHA-256 {str(model['sha256'])[:8]}),"
            f" threshold {found.threshold}"
        )
    header += [f"Found: {_counts(found, shots)}", ""]
    detected = found.threshold is not None
    peak = "the shot detector's peak probability, on the frame before it; " if detected else ""
    possible = " A possible cut is a commented line: checking it is a jump to its time in a player."
    lines = (
        "A line per cut: the frame it cuts at, the first frame of a shot, counted from zero;"
        f" {peak}the time of that frame from the source's start, H:MM:SS.mmm, as a player shows it."
        f" seedvr2x reads the frame number alone.{possible if detected else ''}"
    )
    usage = (
        "Uncomment a possible cut to keep it, delete a line to drop a cut"
        if detected
        else "Delete a line to drop a cut, add one to cut at its frame"
    )
    header += textwrap.wrap(lines, 86)
    header += textwrap.wrap(f"{usage}, then run with --cuts FILE, this file.", 86)
    text = cut_list(header, source.index, found.cuts, found.possible)
    try:
        write_whole(args.plan, text.encode())
    except OSError as error:
        logger.error("--plan %s: not written: %s", args.plan, error)
        return 1
    logger.info("%s: the cut list in %s", _counts(found, shots), args.plan)
    return 0


def _threshold(text: str) -> float:
    """--cut-threshold's P: a probability above 0 and at most 1 (DESIGN.md, Shot detection)."""
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0 < value <= 1:  # NaN fails it too
        raise argparse.ArgumentTypeError(
            f"{text!r}: not a threshold; give a probability above 0 and at most 1, such as"
            f" {THRESHOLD}, the default"
        )
    return value


def _positive(text: str) -> int:
    """A whole number above 0."""
    if not re.fullmatch(r"\d+", text) or int(text) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a whole number above 0")
    return int(text)


def _seconds(text: str) -> Fraction:
    """A duration in seconds, 0 or more, exact: 4.99 is 499/100."""
    try:
        seconds = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"{text!r}: not a number of seconds") from None
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: negative")
    return seconds


def _rate(text: str) -> Fraction:
    """A positive frame rate, exact, written N/D or as a whole number, as ffprobe's r_frame_rate
    gives it: 24000/1001, not 23.976."""
    match = re.fullmatch(r"(\d+)(?:/(\d+))?", text)
    if match is None or int(match[1]) == 0 or (match[2] is not None and int(match[2]) == 0):
        raise argparse.ArgumentTypeError(
            f"{text!r}: not a frame rate; write it N/D or as a whole number, exact: 24000/1001 for"
            " 23.976, 30000/1001 for 29.97, 25"
        )
    return Fraction(int(match[1]), int(match[2] or 1))


def _ratio(text: str) -> Fraction:
    """A positive ratio written N:D or N/D, as the sample aspect ratio option takes it."""
    match = re.fullmatch(r"(\d+)[:/](\d+)", text)
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a positive ratio N:D")
    return Fraction(int(match[1]), int(match[2]))
