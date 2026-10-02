"""Command line: options and logging."""

import argparse
import logging
import os
import time
from importlib.metadata import version
from pathlib import Path

logger = logging.getLogger("seedvr2x")


def main(argv: list[str] | None = None) -> int:
    """Run seedvr2x with argv (sys.argv[1:] when None) and return the exit status."""
    parser = argparse.ArgumentParser(
        prog="seedvr2x",
        description="SeedVR2 video upscaler for long runs. One shot per input (milestone 2).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('seedvr2x')}")
    parser.add_argument("input", type=Path, help="8- or 16-bit RGB video, upscaled as one shot")
    parser.add_argument("-o", "--output", type=Path, required=True, help="FFV1 master (.mkv)")
    parser.add_argument("--model-dir", type=Path, required=True, help="directory of the weights")
    parser.add_argument("--dit-model", required=True, help="DiT file, e.g. 7B fp16 safetensors")
    parser.add_argument("--vae-model", default="ema_vae_fp16.safetensors", help="VAE file")
    parser.add_argument("--resolution", type=int, default=1080, help="output short side")
    parser.add_argument("--seed", type=int, default=42)
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
    import numpy as np
    import torch

    from seedvr2x.media.decode import read_rgb
    from seedvr2x.media.ffv1 import write_gbrp16
    from seedvr2x.media.probe import probe
    from seedvr2x.runtime.model import load_models, to_input
    from seedvr2x.runtime.shot import upscale_shot

    if not torch.cuda.is_available():
        logger.error("no CUDA device")
        return 1
    if not torch.cuda.is_bf16_supported():
        logger.error("the GPU doesn't compute in bfloat16, numz's pipeline dtype")
        return 1
    device = torch.device("cuda", 0)
    stream = probe(args.input)
    frames = read_rgb(args.input, stream)
    logger.info(
        "%s: %d frames %dx%d %s at %s fps",
        args.input,
        len(frames),
        stream.width,
        stream.height,
        stream.pix_fmt,
        stream.frame_rate,
    )
    started = time.monotonic()
    models = load_models(args.model_dir, args.dit_model, args.vae_model, device)
    logger.info(
        "models loaded in %.1f s, attention: %s", time.monotonic() - started, models.attention
    )
    started = time.monotonic()
    out = upscale_shot(models, to_input(frames), args.resolution, args.seed, args.window)
    logger.info("upscaled in %.1f s: %s", time.monotonic() - started, tuple(out.shape))
    out_frames = out.numpy()
    write_gbrp16(args.output, out_frames, stream.frame_rate)
    logger.info("wrote %s", args.output)
    if args.dump_frames is not None:
        args.dump_frames.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(out_frames):
            np.save(args.dump_frames / f"frame_{index:06d}.npy", np.ascontiguousarray(frame))
    return 0
