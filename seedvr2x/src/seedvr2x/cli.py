"""Command line: options and logging."""

import argparse
import fcntl
import logging
import os
import re
import shutil
import time
from bisect import bisect_right
from collections.abc import Sequence
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.job import MIN_SEGMENT
from seedvr2x.runtime.stop import Stop, Stopped, Terminated

if TYPE_CHECKING:
    import torch

    from seedvr2x.runtime.job import OutputSegment, Part, Shot
    from seedvr2x.runtime.manifest import Manifest

# Video file names, other than Matroska's, that -o refuses: an FFV1 master is a .mkv file, and
# anything else names the directory of the output segments, which such a name would only hide.
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".webm", ".ts", ".m2ts", ".mts", ".mxf", ".nut", ".mpg"}
)

logger = logging.getLogger("seedvr2x")


def main(argv: list[str] | None = None) -> int:
    """Run seedvr2x with argv (sys.argv[1:] when None) and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="seedvr2x",
        description="SeedVR2 video upscaler for long runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument(
        "input",
        type=Path,
        help="video file, or directory of segments (its .mkv and .mp4 files, in name order: each"
        " join a cut)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="a .mkv path: one FFV1 master (video file input only); else a new or empty"
        " directory: the output segments (FFV1 files, or PNG directories) and their manifest."
        " A directory of segments is mirrored; a video file is cut at its shots, merged to"
        " --min-segment",
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="gbrp16le",
        help="FFV1 master in 16-bit RGB or in 10-bit YUV 4:2:0 (BT.709 at HD sizes, limited range),"
        " or 16-bit PNG (default: %(default)s)",
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="directory of the weights")
    parser.add_argument("--dit-model", required=True, help="DiT file, e.g. 7B fp16 safetensors")
    parser.add_argument("--vae-model", default="ema_vae_fp16.safetensors", help="VAE file")
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
    parser.add_argument(
        "--min-segment",
        type=_seconds,
        default=MIN_SEGMENT,
        metavar="SECONDS",
        help="output segments of a video file last this long at least, each shorter one merged"
        " into its shorter neighbour, as sptenc's -L (default: %(default)s; 0 keeps every cut)",
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
    # A stop at once, a second Ctrl-C or SIGTERM during the run (runtime/stop.py), or Ctrl-C
    # before it: what is kept is what the manifest says, and the same command resumes.
    try:
        return _run(args)
    except KeyboardInterrupt:
        logger.error("stopped at once (Ctrl-C)")
        return 130
    except Terminated:
        logger.error("stopped at once (SIGTERM)")
        return 143
    finally:
        _unlock()


def _run(args: argparse.Namespace) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.source import examine, examine_directory
    from seedvr2x.runtime.job import (
        SHARED,
        JobError,
        check_seed,
        job_shots,
        output_size,
        parts_of,
        read_cuts,
        target_size,
    )

    # The build, the source and the cut list are checked before anything touches the GPU.
    try:
        if args.window is not None and args.window < 2 * SHARED + 1:
            raise JobError(
                f"--window {args.window}: windows share {SHARED} latents with each neighbour, so"
                f" a window needs at least {2 * SHARED + 1}"
            )
        ffmpeg_version = ffmpeg.check(("png",) if args.format == "png" else ())
        logger.info("ffmpeg %s", ffmpeg_version)
        if args.input.is_dir():
            if args.cuts:
                # Until the detector for doubtful joins comes, each join is a cut (DESIGN.md,
                # Input).
                raise JobError("--cuts with a directory: each join of its segments is a cut")
            sources = examine_directory(args.input, args.input_matrix, args.input_sar)
        else:
            sources = [examine(args.input, args.input_matrix, args.input_sar)]
        parts = parts_of(sources)
        cuts = read_cuts(args.cuts) if args.cuts else []
        shots = job_shots(parts, cuts)
        check_seed(args.seed, shots)
        segments, paths, directory = _layout(args, parts, shots)
        for name in (args.dit_model, args.vae_model):
            if not (args.model_dir / name).is_file():
                raise JobError(f"{args.model_dir / name}: no such model file")
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1
    source = sources[0]
    stream = source.stream
    target = target_size(stream.width, stream.height, source.sample_aspect, args.resolution)
    out_height, out_width = output_size(target)
    logger.info(
        "%d shots, %d output segments; output %dx%d, square pixels",
        len(shots),
        len(segments),
        out_width,
        out_height,
    )
    # The manifest's settings, the models' hashes included, before any GPU work.
    settings = _settings(args, cuts) if directory is not None else {}

    import numpy as np
    import numpy.typing as npt
    import torch

    from seedvr2x.media.writer import SegmentWriter, Tags, Writer, open_writer
    from seedvr2x.runtime import manifest
    from seedvr2x.runtime.model import load_models
    from seedvr2x.runtime.run import run_job
    from seedvr2x.runtime.shot import shot_layout
    from seedvr2x.runtime.units import DiskUnits, Units

    if not torch.cuda.is_available():
        logger.error("no CUDA device")
        return 1
    if not torch.cuda.is_bf16_supported():
        logger.error("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
        return 1
    device = torch.device("cuda", 0)
    units, record, resumed = Units(), None, False
    remaining = list(range(len(segments)))  # the segments to write, those not finished
    if directory is not None:
        try:
            _lock(directory)
        except JobError as error:
            logger.error("%s", error)
            return 1
        record = manifest.Manifest(
            directory / manifest.NAME,
            settings,
            _environment(device, ffmpeg_version),
            parts,
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
        resumed = record.path.is_file()
        if not resumed and not _empty(directory):
            logger.error("%s: written to since it was checked, by another program", directory)
            return 1
        if resumed:
            # The job recorded there, checked and taken up before the models load.
            try:
                _resume(record)
            except JobError as error:
                logger.error("%s", error)
                return 1
            remaining = [index for index, done in enumerate(record.finished) if not done]
            if not remaining:
                logger.info("%s: finished already, its %d segments", directory, len(segments))
                return 0
        units = DiskUnits(directory, record)
    started = time.monotonic()
    models = load_models(args.model_dir, args.dit_model, args.vae_model, device)
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
    if record is not None and not resumed:
        record.path.parent.mkdir(parents=True, exist_ok=True)
        record.write()
    started = time.monotonic()

    tags = Tags.of(stream, source.conversion.matrix_tag)

    def open_segment(path: Path) -> Writer:
        return open_writer(args.format, path, out_width, out_height, stream.frame_rate, tags)

    # The n-th frame written is the job's frame number(n): the segments left, in order.
    ends = list(accumulate(segments[index].frames for index in remaining))

    def number(written: int) -> int:
        which = bisect_right(ends, written)
        return segments[remaining[which]].start + written - (ends[which - 1] if which else 0)

    outputs = [(paths[index], segments[index].frames) for index in remaining]
    with Stop() as stop:
        try:
            with SegmentWriter(
                outputs, open_segment, lambda which: units.segment_finished(remaining[which])
            ) as writer:

                def write(frames: npt.NDArray[np.float32]) -> None:
                    if args.dump_frames is not None:
                        for written, frame in enumerate(frames, writer.written):
                            path = args.dump_frames / f"frame_{number(written):06d}.npy"
                            np.save(path, np.ascontiguousarray(frame))
                    writer.write(frames)

                run_job(
                    *(models, parts, shots, segments, target, args.seed, args.window, units),
                    *(write, stop),
                )
        except MediaError as error:
            logger.error("%s", error)
            return 1
        except Stopped:
            if record is None:
                kept = "nothing kept: one file can't resume until assembly (milestone 6)"
            else:
                kept = (
                    f"{sum(record.finished)} of {len(segments)} segments finished, and the"
                    " units of the next ones kept: the same command resumes"
                )
            logger.warning("stopped after %s, as asked; %s", stop.unit, kept)
            return 130
    logger.info(
        "upscaled in %.1f s; wrote %s: %d frames, %s",
        time.monotonic() - started,
        args.output,
        writer.written,
        args.format,
    )
    return 0


def _layout(
    args: argparse.Namespace, parts: "Sequence[Part]", shots: "Sequence[Shot]"
) -> "tuple[list[OutputSegment], list[Path], Path | None]":
    """The output's segments, each one's path, and the directory they go in (None for one file).
    A .mkv path takes a video file's output whole, one FFV1 master. Else the output is a new or
    empty directory of segments, named and cut as sptenc's split: a directory's own, mirrored;
    a video file's shots, merged to --min-segment (DESIGN.md, Output). Until assembly (milestone
    6), the .mkv path stands for the one-file output, and the directory must be new or empty,
    which keeps another run's files out of what sptenc reads (DESIGN.md, Output), or hold a
    job's manifest, which resume reads back (resume.leftovers checks the rest)."""
    from seedvr2x.runtime.job import JobError, OutputSegment, merged_segments, mirrored_segments
    from seedvr2x.runtime.manifest import NAME, STATE

    output: Path = args.output
    total = parts[-1].end
    if output.suffix.lower() in VIDEO_SUFFIXES:
        raise JobError(
            f"{output}: a video file name; an FFV1 master is a .mkv path, and the output segments"
            " go in a directory"
        )
    if output.suffix.lower() == ".mkv":
        if output.is_dir():
            raise JobError(f"{output}: a directory; a .mkv path names the one FFV1 master")
        if args.input.is_dir():
            raise JobError(f"{output}: a directory of segments needs a directory as output")
        if args.format == "png":
            raise JobError(f"{output}: PNG output goes to a directory")
        return [OutputSegment(output.stem, 0, total)], [output], None
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
    if args.input.is_dir():
        segments = mirrored_segments(parts)
    else:
        frame_rate = parts[0].source.stream.frame_rate
        segments = merged_segments(shots, total, frame_rate, args.min_segment)
    suffix = "" if args.format == "png" else ".mkv"
    paths = [output / f"{segment.name}{suffix}" for segment in segments]
    for path in paths:
        # A mirrored segment takes its file's stem, which could be one of seedvr2x's own names.
        if path.name in (NAME, STATE) or path.name.endswith(".partial"):
            raise JobError(f"{path.name}: a name seedvr2x keeps for itself in its output")
    return segments, paths, output


def _empty(directory: Path) -> bool:
    """Whether directory holds nothing, but what an interrupted first write of the manifest left,
    which the next write replaces."""
    from seedvr2x.runtime.manifest import NAME

    return all(entry.name == f"{NAME}.partial" for entry in directory.iterdir())


def _settings(args: argparse.Namespace, cuts: list[int]) -> dict[str, object]:
    """The settings a manifest records, those a resume must find again."""
    from seedvr2x.runtime.manifest import code_sha256

    return {
        "seedvr2x": version("seedvr2x"),
        "code": code_sha256(),
        "dit_model": _model(args.model_dir, args.dit_model),
        "vae_model": _model(args.model_dir, args.vae_model),
        "resolution": args.resolution,
        "seed": args.seed,
        "window": args.window,
        "format": args.format,
        "cuts": cuts,
        "min_segment": None if args.input.is_dir() else str(args.min_segment),
        "input_matrix": args.input_matrix,
        "input_sar": None if args.input_sar is None else str(args.input_sar),
    }


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


def _unlock() -> None:
    while _LOCKS:
        os.close(_LOCKS.pop())


def _resume(record: "Manifest") -> None:
    """Take up the job recorded beside record, refused (JobError) unless it is the one asked, its
    directory as the manifest says; then discard what a stop left that the manifest doesn't
    name (resume.leftovers)."""
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.job import JobError

    recorded = resume.read(record.path)
    found = resume.differences(recorded, record.content())
    if found:
        raise JobError(
            f"{record.path}: another job than the one asked, which differs in:\n  "
            + "\n  ".join(found[:20])
            + (f"\n  and {len(found) - 20} more" if len(found) > 20 else "")
        )
    resume.adopt(record, recorded)
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


def _model(directory: Path, name: str) -> dict[str, object]:
    """A model as the manifest records it: its file's name, size and SHA-256, models being
    identified by hash (DESIGN.md, Options kept and dropped). The 7B fp16 DiT is 16 GB to read."""
    from seedvr2x.runtime.manifest import sha256

    path = directory / name
    started = time.monotonic()
    digest = sha256(path)
    logger.info("%s: SHA-256 %s, in %.1f s", name, digest, time.monotonic() - started)
    return {"name": name, "size": path.stat().st_size, "sha256": digest}


def _environment(device: "torch.device", ffmpeg_version: str) -> dict[str, object]:
    """What the output's bits depend on besides the settings, which a resume must find again to
    stay bit-identical (DESIGN.md, Pause and resume): the stack, the GPU, the attention backend
    and FlashAttention's version, ffmpeg's."""
    import torch

    from seedvr2x.runtime.model import attention_backend

    try:
        flash_attn = version("flash_attn")
    except PackageNotFoundError:
        flash_attn = None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "attention": attention_backend(),
        "flash_attn": flash_attn,
        "ffmpeg": ffmpeg_version,
    }


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
