"""Command line: options and logging."""

import argparse
import fcntl
import logging
import os
import re
import shlex
import shutil
import sys
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.job import MIN_SEGMENT
from seedvr2x.runtime.pull import REPO
from seedvr2x.runtime.stop import Stop, Stopped, Terminated, terminable

if TYPE_CHECKING:
    import torch

    from seedvr2x.media.index import FrameIndex
    from seedvr2x.media.source import Declared, FirstPass, Source
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
        help="a directory holding the model files, each read there by its name; nothing is"
        " downloaded. Without it, --dit-model and --vae-model name seedvr2x's own files, taken"
        f" from Hugging Face's cache (HF_HOME moves it), downloaded into it from {REPO} at a"
        " pinned revision when missing (HF_HUB_OFFLINE keeps the network out). Either way,"
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
    parser.add_argument(
        "--cuts",
        type=Path,
        metavar="FILE",
        help="cut list: the first frame of each shot but the first, one frame number per line,"
        " counted from 0 (default: the whole input is one shot)",
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
    from seedvr2x.media.source import counted, declare, first_pass
    from seedvr2x.runtime import pull
    from seedvr2x.runtime.job import (
        SHARED,
        JobError,
        check_seed,
        check_target,
        output_size,
        read_cuts,
        shots_from_cuts,
        target_size,
    )
    from seedvr2x.runtime.manifest import NAME
    from seedvr2x.runtime.weights import ModelError, check_models

    # The build, the source, the target, the cut list and the model files are checked before
    # anything touches the GPU, but for a job resumed, which is first checked against its record
    # (its settings, environment, the GPU's included, and input), before its first pass, which
    # isn't run again when nothing it depends on changed (DESIGN.md, Pause and resume).
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
        ffmpeg_version = ffmpeg.check(
            ("png",) if args.format == "png" else (),
            ("framehash",) if args.format == "yuv420p10le" else (),
        )
        conversions = fingerprint()
        logger.info("ffmpeg %s, its conversions' fingerprint %s", ffmpeg_version, conversions[:16])
        declared = declare(args.input, args.input_matrix, args.input_sar)
        # The size the frames are resized to, from what the source declares; refused, before the
        # first pass, when too small to pad (job.check_target).
        stream = declared.stream
        target = target_size(stream.width, stream.height, declared.sample_aspect, args.resolution)
        check_target(target, args.resolution)
        cuts = read_cuts(args.cuts) if args.cuts else []
        directory = _output(args)
        # The model files, from --model-dir, or seedvr2x's own from Hugging Face's cache, pulled
        # into it when missing; checked by their headers, in a second, then seedvr2x's own by
        # their pinned size and SHA-256 (16.5 GB to read for the DiT), which the manifest's
        # record takes: before the resume's comparison and the first pass, which decodes the
        # whole source (DESIGN.md, Weights). The header check comes first, as DESIGN.md has it
        # (Weights, Recognised by content); its refusal of one of seedvr2x's own files, given in its
        # own role, says how to fetch it again, as the pin's does (pull.advice).
        # SIGTERM, which kills the process at once until the run installs its handlers
        # (runtime/stop.py), unwinds a download as Ctrl-C does, so that the library removes its
        # partial file, up to 16.5 GB that no later download takes up (runtime/pull.py, at its
        # end).
        with terminable():
            model_files = pull.resolve(args.model_dir, args.dit_model, args.vae_model)
        check_models(model_files.dit.path, model_files.vae.path, pull.advice(model_files))
        pull.check_pinned(model_files)
        if directory is not None and (directory / NAME).is_file():
            _lock(directory)
            prior = _prior(
                *(args, directory, cuts, declared, ffmpeg_version, conversions),
                *(numz_padding, model_files),
            )
        if prior is not None and prior.known is not None:
            found = prior.known
        else:
            # The source's content hashed meanwhile, for the manifest.
            found = first_pass(declared, hashed=directory is not None)
        source = counted(declared, found)
        shots = shots_from_cuts(cuts, source.frames)
        check_seed(args.seed, shots)
        segments, paths = _segments(args, source, shots, directory)
        out_height, out_width = output_size(target)
        logger.info(
            "%d shots, %d output segments; output %dx%d, square pixels",
            len(shots),
            len(segments),
            out_width,
            out_height,
        )
        if prior is not None:
            identity = prior.identity
        else:
            identity = _identity(
                *(args, cuts, directory, ffmpeg_version, conversions), numz_padding, model_files
            )
        work = (
            _work(args.output) if directory is None and args.color_correction == "split" else None
        )
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
                "frame_rate": str(stream.frame_rate),
            },
        )
        if prior is None and not _empty(directory):
            logger.error("%s: written to since it was checked, by another program", directory)
            return 1
        if prior is not None:
            # The job recorded there, checked whole and taken up before the models load.
            try:
                _resume(record, prior)
            except JobError as error:
                logger.error("%s", error)
                return 1
            remaining = [index for index, done in enumerate(record.finished) if not done]
            if prior.known is None:
                # The first pass ran again: its index is written with a manifest naming it.
                _index_made_again(record, prior)
            if not remaining:
                logger.info("%s: finished already, its %d segments", directory, len(segments))
                return 0
        units = DiskUnits(directory, record)
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
        record.path.parent.mkdir(parents=True, exist_ok=True)
        # The frame index first, whole, then the manifest naming it (manifest.INDEX).
        _write_index(record.path.parent, source)
        record.write()
    started = time.monotonic()

    tags = Tags.of(stream, source.conversion.matrix_tag)

    def open_segment(path: Path) -> Writer:
        size = (out_width, out_height, stream.frame_rate)
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
    # accepts, and the output is written at the declared rate, frame by frame; but a default
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
    frame_rate = source.stream.frame_rate
    segments = merged_segments(shots, source.frames, frame_rate, args.min_segment)
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
    """Write the source's frame index beside the manifest, whole (manifest.INDEX)."""
    from seedvr2x.runtime.manifest import INDEX

    if source.index is None:
        raise ValueError(f"{source.path}: no frame index to write")
    source.index.write(directory / INDEX)


def _settings(
    args: argparse.Namespace, cuts: list[int], numz_padding: bool, model_files: "ModelFiles"
) -> dict[str, object]:
    """The settings a manifest records, those a resume must find again; numz's padding only when
    the tests ask for it (NUMZ_PADDING), so that a user's manifest never names it."""
    from seedvr2x.runtime.manifest import code_sha256

    settings: dict[str, object] = {
        "seedvr2x": version("seedvr2x"),
        "code": code_sha256(),
        "dit_model": _model(model_files.dit, model_files),
        "vae_model": _model(model_files.vae, model_files),
        "resolution": args.resolution,
        "seed": args.seed,
        "color_correction": args.color_correction,
        "window": args.window,
        "format": args.format,
        "cuts": cuts,
        "min_segment": str(args.min_segment),
        "input_matrix": args.input_matrix,
        "input_sar": None if args.input_sar is None else str(args.input_sar),
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
    cuts: list[int],
    declared: "Declared",
    ffmpeg_version: str,
    conversions: str,
    numz_padding: bool,
    model_files: "ModelFiles",
) -> _Prior:
    """The job recorded in directory, checked against the one asked before its first pass: the
    same settings, environment and source, the source being its content (resume.identity), or
    refused (JobError), but for an environment change accepted (--accept-env-change). The
    record of the first pass is trusted, and the pass isn't run again, unless ffmpeg or its
    conversions changed: the same bytes, decoded by the same build, give the same frames, and
    nothing else takes part in the pass (DESIGN.md, Pause and resume)."""
    from seedvr2x.media.source import FirstPass
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import NAME, read

    path = directory / NAME
    recorded = read(path)
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
    if len(changed) < len(found) or (changed and not args.accept_env_change):
        raise _another_job(path, found, only_environment=len(changed) == len(found))
    entry: dict[str, Any] = recorded["input"]
    if entry["path"] != str(declared.path.resolve()):
        logger.info(
            "%s: the input recorded at %s, moved: the same content", declared.path, entry["path"]
        )
    before: dict[str, Any] = recorded["environment"]
    if any(before.get(key) != identity.environment.get(key) for key in resume.FIRST_PASS):
        return _Prior(recorded, identity, changed, None)
    index, why = _recorded_index(directory, entry)
    if index is None:
        # PROVISIONAL (DESIGN.md has the index trusted as the first pass's record is, and says
        # nothing of one missing or damaged): derived data, as an input copy is, never trusted
        # unchecked, nor a reason to refuse the job: the first pass runs again and makes it.
        logger.warning("%s; the first pass runs again, which makes it", why)
        return _Prior(recorded, identity, changed, None)
    logger.info("%s: the same input and ffmpeg, the first pass as recorded", directory)
    found = FirstPass(entry["frames"], entry["sha256"], index)
    return _Prior(recorded, identity, changed, found)


def _recorded_index(directory: Path, entry: dict[str, Any]) -> "tuple[FrameIndex | None, str]":
    """The frame index the input's record names, read from directory and checked against the
    record (manifest.INDEX): its size, SHA-256 and frames; or None, and why it isn't trusted."""
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
    before one keeps the record as it was, but for the index of a first pass run again
    (_index_made_again)."""
    from seedvr2x.runtime import resume

    found = [
        line
        for line in resume.differences(prior.recorded, record.content())
        if resume.section(line) != "environment"  # compared before the first pass
    ]
    if found:
        raise _another_job(record.path, found, only_environment=False)
    resume.adopt(record, prior.recorded)
    if prior.changed:
        change = resume.environment_change(prior.recorded, record.environment)
        record.environment_changes.append(change)
        logger.warning(
            "%s: resumed in another environment, as accepted, which its manifest records with"
            " the next unit made: %s",
            record.path.parent,
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
        "resuming %s: %d of %d segments finished, %d of %d shots encoded, %d windows kept; %d"
        " leftovers discarded",
        record.path.parent,
        sum(record.finished),
        len(record.finished),
        sum(record.encoded),
        len(record.encoded),
        sum(record.windows_done),
        len(discarded),
    )


def _index_made_again(record: "Manifest", prior: _Prior) -> None:
    """The frame index of a first pass a resume ran again (_prior), written only with a manifest
    that names it and records the environment it was made in (manifest.INDEX).

    Made in the environment recorded, the recorded index being missing or damaged: at once, then
    the manifest as recorded, naming it, so that the next run trusts the record, that of a job
    finished already too, which makes no unit to write its manifest with. Left to the next unit, a
    finished job kept index bytes its manifest didn't name when the pass gave others (a damaged
    source decodes otherwise from one pass to the next: media/reader.py), and every later run
    warned of them and ran the pass again.

    Made in another environment, accepted (ffmpeg or its conversions changed): with the next unit
    made, whose manifest records that environment (DESIGN.md, Pause and resume), the directory
    keeping the recorded index until then: a resume stopped before a unit "still resumes in its
    old environment", its record whole, and a job finished already, which makes none, keeps what
    it has.

    PROVISIONAL (implementation, 2026-10-10, a question for design: DESIGN.md has a resume stopped
    before a unit leave the manifest as it was, which holds but for the record of an index made
    again in the environment recorded)."""
    from dataclasses import replace

    from seedvr2x.runtime import resume

    directory = record.path.parent
    source = record.source
    before: dict[str, Any] = prior.recorded["environment"]
    if any(before.get(key) != record.environment.get(key) for key in resume.FIRST_PASS):
        record.before_write = lambda: _write_index(directory, source)
        return
    _write_index(directory, source)
    as_recorded = replace(
        record,
        environment=before,
        environment_changes=list(prior.recorded.get(resume.CHANGES, [])),
    )
    as_recorded.write()


def _another_job(path: Path, found: list[str], only_environment: bool) -> "JobError":
    """A resume refused, each difference listed, `where: recorded -> now`."""
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
    )


def _model(file: "ModelFile", model_files: "ModelFiles") -> dict[str, object]:
    """A model as the manifest records it: its file's name, as given, its size and SHA-256, models
    being identified by hash (DESIGN.md, Options kept and dropped). The 7B fp16 DiT is 16.5 GB to
    read, once a run: a file seedvr2x pins was hashed by its check already (pull.Hashes)."""
    digest = model_files.hashes.sha256(file.path)
    return {"name": file.name, "size": file.path.stat().st_size, "sha256": digest}


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


def _ratio(text: str) -> Fraction:
    """A positive ratio written N:D or N/D, as the sample aspect ratio option takes it."""
    match = re.fullmatch(r"(\d+)[:/](\d+)", text)
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a positive ratio N:D")
    return Fraction(int(match[1]), int(match[2]))
