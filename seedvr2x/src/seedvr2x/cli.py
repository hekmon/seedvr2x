"""Command line: options and logging."""

import argparse
import logging
import os
import re
import time
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

from seedvr2x.media.conversion import MATRICES
from seedvr2x.media.writer import FORMATS

logger = logging.getLogger("seedvr2x")


def main(argv: list[str] | None = None) -> int:
    """Run seedvr2x with argv (sys.argv[1:] when None) and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="seedvr2x",
        description="SeedVR2 video upscaler for long runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument("input", type=Path, help="video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="FFV1 master (.mkv), or the directory of the PNG frames",
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
        type=int,
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
    from seedvr2x.media.source import examine
    from seedvr2x.runtime.job import (
        SHARED,
        JobError,
        output_size,
        read_cuts,
        shots_from_cuts,
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
        source = examine(args.input, args.input_matrix, args.input_sar)
        shots = shots_from_cuts(read_cuts(args.cuts) if args.cuts else [], source.frames)
    except (MediaError, JobError) as error:
        logger.error("%s", error)
        return 1
    stream = source.stream
    target = target_size(stream.width, stream.height, source.sample_aspect, args.resolution)
    out_height, out_width = output_size(target)
    logger.info("%d shots; output %dx%d, square pixels", len(shots), out_width, out_height)

    import numpy as np
    import numpy.typing as npt
    import torch

    from seedvr2x.media.writer import Tags, open_writer
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
    started = time.monotonic()
    try:
        with open_writer(
            args.format, args.output, out_width, out_height, stream.frame_rate, Tags.of(stream)
        ) as writer:

            def write(frames: npt.NDArray[np.float32]) -> None:
                if args.dump_frames is not None:
                    for offset, frame in enumerate(frames, writer.written):
                        path = args.dump_frames / f"frame_{offset:06d}.npy"
                        np.save(path, np.ascontiguousarray(frame))
                writer.write(frames)

            run_shots(models, source, shots, target, args.seed, write, args.window)
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


def _ratio(text: str) -> Fraction:
    """A positive ratio written N:D or N/D, as the sample aspect ratio option takes it."""
    match = re.fullmatch(r"(\d+)[:/](\d+)", text)
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a positive ratio N:D")
    return Fraction(int(match[1]), int(match[2]))
