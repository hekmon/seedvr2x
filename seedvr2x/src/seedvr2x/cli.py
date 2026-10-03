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
from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import version
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.job import MIN_SEGMENT
from seedvr2x.runtime.stop import Stop, Stopped, Terminated

if TYPE_CHECKING:
    import torch

    from seedvr2x.media.source import Declared, FirstPass
    from seedvr2x.runtime.job import JobError, OutputSegment, Part, Shot
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
        "--color-correction",
        choices=("lab", "none"),
        default="lab",
        help="lab: the input's colours back under the model's details, matched over each shot, as"
        " numz's lab; none: the model's colours, which drift (default: %(default)s)",
    )
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
        _remove_work()
        _unlock()


def _run(args: argparse.Namespace) -> int:
    from seedvr2x.media import ffmpeg
    from seedvr2x.media.ffmpeg import MediaError
    from seedvr2x.media.fingerprint import fingerprint
    from seedvr2x.media.source import counted, declare, declare_directory, first_pass
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
    from seedvr2x.runtime.manifest import NAME

    # The build, the source and the cut list are checked before anything touches the GPU, but
    # for a job resumed, which is first checked against its record (its settings, environment,
    # the GPU's included, and inputs), before its first pass, which isn't run again when nothing
    # it depends on changed (DESIGN.md, Pause and resume).
    prior: _Prior | None = None
    try:
        if args.window is not None and args.window < 2 * SHARED + 1:
            raise JobError(
                f"--window {args.window}: windows share {SHARED} latents with each neighbour, so"
                f" a window needs at least {2 * SHARED + 1}"
            )
        ffmpeg_version = ffmpeg.check(("png",) if args.format == "png" else ())
        conversions = fingerprint()
        logger.info("ffmpeg %s, its conversions' fingerprint %s", ffmpeg_version, conversions[:16])
        if args.input.is_dir():
            if args.cuts:
                # Until the detector for doubtful joins comes, each join is a cut (DESIGN.md,
                # Input).
                raise JobError("--cuts with a directory: each join of its segments is a cut")
            declared = declare_directory(args.input, args.input_matrix, args.input_sar)
        else:
            declared = [declare(args.input, args.input_matrix, args.input_sar)]
        cuts = read_cuts(args.cuts) if args.cuts else []
        directory = _output(args)
        if directory is not None and (directory / NAME).is_file():
            _lock(directory)
            prior = _prior(args, directory, cuts, declared, ffmpeg_version, conversions)
        if prior is not None and prior.known is not None:
            passes = prior.known
        else:
            # Each input's content hashed meanwhile, for the manifest.
            passes = [first_pass(each, hashed=directory is not None) for each in declared]
        sources = [counted(each, done) for each, done in zip(declared, passes, strict=True)]
        if args.input.is_dir():
            frames = sum(source.frames for source in sources)
            logger.info("%s: %d segments, %d frames", args.input, len(sources), frames)
        parts = parts_of(sources)
        shots = job_shots(parts, cuts)
        check_seed(args.seed, shots)
        segments, paths = _segments(args, parts, shots, directory)
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
        if prior is not None:
            identity = prior.identity
        else:
            identity = _identity(args, cuts, directory, ffmpeg_version, conversions)
        work = _work(args.output) if directory is None and args.color_correction == "lab" else None
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1

    import numpy as np
    import numpy.typing as npt

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
            if not remaining:
                logger.info("%s: finished already, its %d segments", directory, len(segments))
                return 0
        units = DiskUnits(directory, record)
    started = time.monotonic()
    models = load_models(args.model_dir, args.dit_model, args.vae_model, identity.device)
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
    if record is not None and prior is None:
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
                    *(write, stop, args.color_correction == "lab"),
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


def _kept(record: "Manifest | None", segments: int) -> str:
    """What a run stopped before its end keeps, of a job of `segments` output segments."""
    if record is None:
        return "nothing kept: one file can't resume until assembly (milestone 6)"
    return (
        f"{sum(record.finished)} of {segments} segments finished, and the units of the next ones"
        " kept: the same command resumes"
    )


def _work(output: Path) -> Path:
    """The work directory of a one-file output with lab, beside it (DESIGN.md, Colour correction):
    its shots' input copies and buffers, one shot at a time. Made and locked from the start, as an
    output directory is (_lock), so that another run to the same file is refused; removed when
    main returns, however the run ends. What a killed run left there is removed; anything else
    is refused, never deleted."""
    from seedvr2x.runtime.job import JobError
    from seedvr2x.runtime.units import BUFFER, COPY

    work = output.with_name(f"{output.name}.work")
    ours = {COPY, f"{COPY}.partial", BUFFER}

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
    """The directory the output segments go in, or None for one file. A .mkv path takes a video
    file's output whole, one FFV1 master. Else the output is a new or empty directory of
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
        if args.input.is_dir():
            raise JobError(f"{output}: a directory of segments needs a directory as output")
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
    parts: "Sequence[Part]",
    shots: "Sequence[Shot]",
    directory: Path | None,
) -> "tuple[list[OutputSegment], list[Path]]":
    """The output's segments and each one's path: one file (directory None); else named and cut
    as sptenc's split, a directory's own, mirrored, or a video file's shots, merged to
    --min-segment (DESIGN.md, Output)."""
    from seedvr2x.runtime.job import JobError, OutputSegment, merged_segments, mirrored_segments
    from seedvr2x.runtime.manifest import NAME, STATE

    total = parts[-1].end
    if directory is None:
        return [OutputSegment(args.output.stem, 0, total)], [args.output]
    if args.input.is_dir():
        segments = mirrored_segments(parts)
    else:
        frame_rate = parts[0].source.stream.frame_rate
        segments = merged_segments(shots, total, frame_rate, args.min_segment)
    suffix = "" if args.format == "png" else ".mkv"
    paths = [directory / f"{segment.name}{suffix}" for segment in segments]
    for path in paths:
        # A mirrored segment takes its file's stem, which could be one of seedvr2x's own names.
        if path.name in (NAME, STATE) or path.name.endswith(".partial"):
            raise JobError(f"{path.name}: a name seedvr2x keeps for itself in its output")
    return segments, paths


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
        "color_correction": args.color_correction,
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
    settings, the models by hash, and the environment, both recorded for a directory's manifest
    only; and the GPU it runs on."""

    settings: dict[str, object]
    device: "torch.device"
    environment: dict[str, object]


@dataclass(frozen=True)
class _Prior:
    """A job resumed, as checked against its record before its first pass (_prior)."""

    recorded: dict[str, Any]  # its manifest, as written
    identity: _Identity
    changed: list[str]  # its environment's differences, accepted (--accept-env-change)
    known: "list[FirstPass] | None"  # the first pass's record, unless ffmpeg or conversions changed


def _identity(
    args: argparse.Namespace,
    cuts: list[int],
    directory: Path | None,
    ffmpeg_version: str,
    conversions: str,
) -> _Identity:
    """The job's identity, refused (JobError) without its model files, or without a CUDA GPU
    computing in bfloat16. The models are hashed before any GPU work."""
    from seedvr2x.runtime.job import JobError

    for name in (args.dit_model, args.vae_model):
        if not (args.model_dir / name).is_file():
            raise JobError(f"{args.model_dir / name}: no such model file")
    settings = _settings(args, cuts) if directory is not None else {}
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
    declared: "Sequence[Declared]",
    ffmpeg_version: str,
    conversions: str,
) -> _Prior:
    """The job recorded in directory, checked against the one asked before its first pass: the
    same settings, environment and inputs, an input being its content (resume.identity), or
    refused (JobError), but for an environment change accepted (--accept-env-change). The
    record of the first pass is trusted, and the pass isn't run again, unless ffmpeg or its
    conversions changed: the same bytes, decoded by the same build, give the same frames, and
    nothing else takes part in the pass (DESIGN.md, Pause and resume)."""
    from seedvr2x.media.source import FirstPass
    from seedvr2x.runtime import resume
    from seedvr2x.runtime.manifest import NAME

    path = directory / NAME
    recorded = resume.read(path)
    identity = _identity(args, cuts, directory, ffmpeg_version, conversions)
    asked = {
        "settings": identity.settings,
        "environment": identity.environment,
        "input": [_content(each.path) for each in declared],
    }
    found = resume.differences(resume.identity(recorded), resume.identity(asked))
    changed = [line for line in found if resume.section(line) == "environment"]
    if len(changed) < len(found) or (changed and not args.accept_env_change):
        raise _another_job(path, found, only_environment=len(changed) == len(found))
    inputs: list[dict[str, Any]] = recorded["input"]
    for each, entry in zip(declared, inputs, strict=True):
        if entry["path"] != str(each.path.resolve()):
            logger.info(
                "%s: the input recorded at %s, moved: the same content", each.path, entry["path"]
            )
    before: dict[str, Any] = recorded["environment"]
    if any(before.get(key) != identity.environment.get(key) for key in resume.FIRST_PASS):
        return _Prior(recorded, identity, changed, None)
    logger.info("%s: the same input and ffmpeg, the first pass as recorded", directory)
    return _Prior(
        recorded, identity, changed, [FirstPass(e["frames"], e["sha256"]) for e in inputs]
    )


def _content(path: Path) -> dict[str, object]:
    """An input file's content, as a resume compares it: its size and SHA-256."""
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
    before one keeps the record as it was."""
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


def _model(directory: Path, name: str) -> dict[str, object]:
    """A model as the manifest records it: its file's name, size and SHA-256, models being
    identified by hash (DESIGN.md, Options kept and dropped). The 7B fp16 DiT is 16 GB to read."""
    from seedvr2x.media.files import sha256

    path = directory / name
    started = time.monotonic()
    digest = sha256(path)
    logger.info("%s: SHA-256 %s, in %.1f s", name, digest, time.monotonic() - started)
    return {"name": name, "size": path.stat().st_size, "sha256": digest}


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
