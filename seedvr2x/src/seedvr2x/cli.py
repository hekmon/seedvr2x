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
        description="SeedVR2 video upscaler for long runs. One shot per input (milestone 2).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument("input", type=Path, help="video file, upscaled as one shot")
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
    parser.add_argument("--resolution", type=int, default=1080, help="output short side")
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

    # The build and the source are checked before anything touches the GPU.
    try:
        logger.info("ffmpeg %s", ffmpeg.check(("png",) if args.format == "png" else ()))
        source = examine(args.input, args.input_matrix, args.input_sar)
    except MediaError as error:
        logger.error("%s", error)
        return 1

    import numpy as np
    import torch

    from seedvr2x.media.decode import to_float32
    from seedvr2x.media.writer import Tags, open_writer
    from seedvr2x.runtime.model import load_models, to_input
    from seedvr2x.runtime.shot import upscale_shot

    if not torch.cuda.is_available():
        logger.error("no CUDA device")
        return 1
    if not torch.cuda.is_bf16_supported():
        logger.error("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
        return 1
    device = torch.device("cuda", 0)
    try:
        with source.decoder() as decoder:
            frames = to_float32(decoder.read(source.frames))
    except MediaError as error:
        logger.error("%s: %s", args.input, error)
        return 1
    started = time.monotonic()
    models = load_models(args.model_dir, args.dit_model, args.vae_model, device)
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    started = time.monotonic()
    out = upscale_shot(models, to_input(frames), args.resolution, args.seed, args.window)
    logger.info("upscaled in %.1f s: %s", time.monotonic() - started, tuple(out.shape))
    out_frames = out.numpy()
    _, out_height, out_width, _ = out_frames.shape
    try:
        with open_writer(
            args.format,
            args.output,
            out_width,
            out_height,
            source.stream.frame_rate,
            Tags.of(source.stream),
        ) as writer:
            writer.write(out_frames)
    except MediaError as error:
        logger.error("%s", error)
        return 1
    logger.info("wrote %s: %d frames, %s", args.output, writer.written, args.format)
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(out_frames):
            np.save(args.dump_frames / f"frame_{index:06d}.npy", np.ascontiguousarray(frame))
    return 0


def _ratio(text: str) -> Fraction:
    """A positive ratio written N:D or N/D, as the sample aspect ratio option takes it."""
    match = re.fullmatch(r"(\d+)[:/](\d+)", text)
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        raise argparse.ArgumentTypeError(f"{text!r}: not a positive ratio N:D")
    return Fraction(int(match[1]), int(match[2]))
