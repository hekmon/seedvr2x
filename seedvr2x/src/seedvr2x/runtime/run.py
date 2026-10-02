"""A job's run: its shots upscaled in order, their frames read from the input files and written as
they come (DESIGN.md, Pipeline, per shot)."""

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
from seedvr2x.runtime.job import Part, Shot
from seedvr2x.runtime.model import Models
from seedvr2x.runtime.shot import upscale_shot

logger = logging.getLogger(__name__)

GIB = 1024**3


def run_shots(
    models: Models,
    parts: Sequence[Part],
    shots: Sequence[Shot],
    target: tuple[int, int],
    seed: int,
    write: Callable[[npt.NDArray[np.float32]], None],
    window: int | None = None,
) -> None:
    """Upscale the shots of a job, which cover its parts in order, no shot spanning two
    (job.job_shots), each with its own seed (Shot.seed), its frames resized to target
    (job.target_size); write(frames) takes the output frames as they come, (n, H', W', 3)
    float32 in [0, 1]."""
    number = 0
    for part in parts:
        part_shots = [shot for shot in shots if part.start <= shot.start < part.end]
        if not part_shots or part_shots[0].start != part.start or part_shots[-1].end != part.end:
            raise ValueError(f"the shots don't cover {part.source.path} alone")
        with part.source.decoder() as decoder:
            for shot in part_shots:
                number += 1
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(models.device)
                upscale_shot(
                    models,
                    _reader(decoder, part.source.path),
                    shot.frames,
                    target,
                    shot.seed(seed),
                    write,
                    window,
                )
                _log_shot(models, number, len(shots), shot, seed, started)


def _log_shot(
    models: Models, number: int, total: int, shot: Shot, seed: int, started: float
) -> None:
    logger.info(
        "shot %d/%d: frames %d to %d (%d), seed %d, %.1f s;"
        " VRAM peak %.2f GiB; RAM %.2f GiB, peak %.2f GiB",
        number,
        total,
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
