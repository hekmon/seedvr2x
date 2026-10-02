"""Command line: options and logging."""

import argparse
import logging
import os
import re
import time
from collections.abc import Sequence
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS
from seedvr2x.runtime.job import MIN_SEGMENT

if TYPE_CHECKING:
    from seedvr2x.runtime.job import OutputSegment, Part, Shot

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
    return _run(args)


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
        logger.info("ffmpeg %s", ffmpeg.check(("png",) if args.format == "png" else ()))
        if args.input.is_dir():
            if args.cuts:
                # Provisional: a directory's cut list says which joins are cuts (DESIGN.md,
                # Input), which comes with the detector for doubtful joins.
                raise JobError("--cuts with a directory: each join of its segments is a cut")
            sources = examine_directory(args.input, args.input_matrix, args.input_sar)
        else:
            sources = [examine(args.input, args.input_matrix, args.input_sar)]
        parts = parts_of(sources)
        cuts = read_cuts(args.cuts) if args.cuts else []
        shots = job_shots(parts, cuts)
        check_seed(args.seed, shots)
        segments, paths, directory = _layout(args, parts, shots)
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

    import numpy as np
    import numpy.typing as npt
    import torch

    from seedvr2x.media.writer import SegmentWriter, Tags, Writer, open_writer
    from seedvr2x.runtime import manifest
    from seedvr2x.runtime.model import load_models
    from seedvr2x.runtime.run import run_shots

    if not torch.cuda.is_available():
        logger.error("no CUDA device")
        return 1
    if not torch.cuda.is_bf16_supported():
        logger.error("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
        return 1
    device = torch.device("cuda", 0)
    started = time.monotonic()
    models = load_models(args.model_dir, args.dit_model, args.vae_model, device)
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
    record = None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        record = manifest.Manifest(
            directory / manifest.NAME,
            _settings(args, cuts),
            parts,
            shots,
            segments,
            [path.name for path in paths],
            {
                "format": args.format,
                "size": [out_width, out_height],
                "frame_rate": str(stream.frame_rate),
            },
        )
        record.write()
    started = time.monotonic()

    def open_segment(path: Path) -> Writer:
        return open_writer(
            args.format, path, out_width, out_height, stream.frame_rate, Tags.of(stream)
        )

    try:
        outputs = [(path, segment.frames) for path, segment in zip(paths, segments, strict=True)]
        finished = record.segment_finished if record is not None else None
        with SegmentWriter(outputs, open_segment, finished) as writer:

            def write(frames: npt.NDArray[np.float32]) -> None:
                if args.dump_frames is not None:
                    for offset, frame in enumerate(frames, writer.written):
                        path = args.dump_frames / f"frame_{offset:06d}.npy"
                        np.save(path, np.ascontiguousarray(frame))
                writer.write(frames)

            run_shots(models, parts, shots, target, args.seed, write, args.window)
    except MediaError as error:
        logger.error("%s", error)
        return 1
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
    a video file's shots, merged to --min-segment (DESIGN.md, Output).

    Provisional: the .mkv path stands for the one-file output until assembly (milestone 6) joins
    the segments into it; and refusing a directory that holds anything keeps files of another
    run out of the segments sptenc reads, until resume reads the manifest back."""
    from seedvr2x.runtime.job import JobError, OutputSegment, merged_segments, mirrored_segments

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
    if output.is_dir() and any(output.iterdir()):
        raise JobError(f"{output}: not empty; the output segments need a new or empty directory")
    if args.input.is_dir():
        segments = mirrored_segments(parts)
    else:
        frame_rate = parts[0].source.stream.frame_rate
        segments = merged_segments(shots, total, frame_rate, args.min_segment)
    suffix = "" if args.format == "png" else ".mkv"
    return segments, [output / f"{segment.name}{suffix}" for segment in segments], output


def _settings(args: argparse.Namespace, cuts: list[int]) -> dict[str, object]:
    """The settings a manifest records, those a resume must find again."""
    return {
        "seedvr2x": version("seedvr2x"),
        "dit_model": {
            "name": args.dit_model,
            "size": (args.model_dir / args.dit_model).stat().st_size,
        },
        "vae_model": {
            "name": args.vae_model,
            "size": (args.model_dir / args.vae_model).stat().st_size,
        },
        "resolution": args.resolution,
        "seed": args.seed,
        "window": args.window,
        "format": args.format,
        "cuts": cuts,
        "min_segment": None if args.input.is_dir() else str(args.min_segment),
        "input_matrix": args.input_matrix,
        "input_sar": None if args.input_sar is None else str(args.input_sar),
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
