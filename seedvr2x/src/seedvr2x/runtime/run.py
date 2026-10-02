"""A job's run: its shots upscaled in order, their frames read from the source and written as they
come (DESIGN.md, Pipeline, per shot)."""

import logging
import resource
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x.media.decode import Decoder, to_float32
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.source import Source
from seedvr2x.runtime.job import Shot
from seedvr2x.runtime.model import Models
from seedvr2x.runtime.shot import upscale_shot

logger = logging.getLogger(__name__)

GIB = 1024**3


def run_shots(
    models: Models,
    source: Source,
    shots: Sequence[Shot],
    target: tuple[int, int],
    seed: int,
    write: Callable[[npt.NDArray[np.float32]], None],
    window: int | None = None,
) -> None:
    """Upscale the shots of source, which cover it in order, each with its own seed
    (Shot.seed), its frames resized to target (job.target_size); write(frames) takes the output
    frames as they come, (n, H', W', 3) float32 in [0, 1]."""
    with source.decoder() as decoder:
        for number, shot in enumerate(shots, 1):
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(models.device)
            upscale_shot(
                models,
                _reader(decoder, source.path),
                shot.frames,
                target,
                shot.seed(seed),
                write,
                window,
            )
            logger.info(
                "shot %d/%d: frames %d to %d (%d), seed %d, %.1f s;"
                " VRAM peak %.2f GiB; RAM %.2f GiB, peak %.2f GiB",
                number,
                len(shots),
                shot.start,
                shot.end - 1,
                shot.frames,
                shot.seed(seed),
                time.monotonic() - started,
                torch.cuda.max_memory_allocated(models.device) / GIB,
                _resident() / GIB,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
            )


def _reader(decoder: Decoder, path: Path) -> Callable[[int], npt.NDArray[np.float32]]:
    def read(count: int) -> npt.NDArray[np.float32]:
        frames = decoder.read(count)
        if frames.shape[0] != count:
            raise MediaError(f"{path}: the stream ended after {decoder.decoded} frames")
        return to_float32(frames)

    return read


def _resident() -> int:
    """The process's resident memory now, in bytes (Linux)."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * resource.getpagesize()
