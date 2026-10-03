"""A job's run: its shots upscaled in order, their frames read from the input files and written as
they come (DESIGN.md, Pipeline, per shot), unit by unit (DESIGN.md, Pause and resume)."""

import logging
import resource
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Self

import numpy as np
import numpy.typing as npt
import torch

from seedvr2x.media.decode import Decoder, to_float32
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime.job import OutputSegment, Part, Shot
from seedvr2x.runtime.model import Models
from seedvr2x.runtime.shot import (
    decode_shot,
    encode_shot,
    finite,
    merge_windows,
    sample_windows,
    shot_layout,
)
from seedvr2x.runtime.stop import Stop
from seedvr2x.runtime.units import Units

logger = logging.getLogger(__name__)

GIB = 1024**3


def run_job(
    models: Models,
    parts: Sequence[Part],
    shots: Sequence[Shot],
    segments: Sequence[OutputSegment],
    target: tuple[int, int],
    seed: int,
    window: int | None,
    units: Units,
    write: Callable[[npt.NDArray[np.float32]], None],
    stop: Stop | None = None,
) -> None:
    """Upscale the shots of a job, which cover its parts in order, no shot spanning two
    (job.job_shots), each with its own seed (Shot.seed), its frames resized to target
    (job.target_size), its DiT windows capped at `window` latents; write(frames) takes the output
    frames as they come, (n, H', W', 3) float32 in [0, 1], the segments' in turn.

    A shot's units run in order: its encode, then its windows, each kept by `units` as it is
    done, then its decode. With units kept on disk, a segment's shots are all encoded and sampled
    before the first of them is decoded: the segment's decode and write is then a unit of its own,
    and units never interleave. With nothing kept, each shot is decoded as soon as it is sampled,
    as upscale_shot does, and nothing builds up.

    What units kept already is skipped (DESIGN.md, Pause and resume): a finished segment; a
    shot's encode and the windows done; the input frames of what is skipped, read and dropped.
    An unfinished segment's decode and write restart whole, from its shots' windows kept.

    stop, when given, is checked before each unit (Stop.check)."""
    groups = [
        [i for i, shot in enumerate(shots) if s.start <= shot.start < s.end] for s in segments
    ]
    if sorted(i for group in groups for i in group) != list(range(len(shots))):
        raise ValueError("the segments don't hold whole shots")
    with Inputs(parts) as inputs:
        for segment, group in enumerate(groups):
            if units.finished(segment):
                continue
            if units.persistent:
                for index in group:
                    _sample(models, inputs, shots, index, target, seed, window, units, stop)
                _begin(stop, f"segment {segment + 1}/{len(segments)}'s decode and write")
            for index in group:
                if not units.persistent:
                    _sample(models, inputs, shots, index, target, seed, window, units, stop)
                    _begin(stop, f"shot {index + 1}/{len(shots)}'s decode")
                _decode(models, shots, index, target, window, units, write)


def _begin(stop: Stop | None, unit: str) -> None:
    """Before a unit: stop there if asked (Stop.check), else name it for a Ctrl-C's notice."""
    if stop is not None:
        stop.check()
        stop.unit = unit


def _sample(
    models: Models,
    inputs: "Inputs",
    shots: Sequence[Shot],
    index: int,
    target: tuple[int, int],
    seed: int,
    window: int | None,
    units: Units,
    stop: Stop | None,
) -> None:
    """Shot `index`'s encode and windows, those not kept yet."""
    shot = shots[index]
    name = f"shot {index + 1}/{len(shots)}"
    layout = shot_layout(shot.frames, window)
    done = units.windows_done(index)
    if done == len(layout):
        return
    latent = units.latent(index)
    if latent is None:
        _begin(stop, f"{name}'s encode")
        logger.debug("%s: encoding", name)
        started = _started(models)
        latent = encode_shot(models, inputs.reader(shot), shot.frames, target, shot.seed(seed))
        finite(latent, f"{name}'s encode")
        units.save_latent(index, latent)
        logger.info(
            "%s, frames %d to %d (%d), seed %d: encoded in %s",
            name,
            shot.start,
            shot.end - 1,
            shot.frames,
            shot.seed(seed),
            _spent(models, started),
        )
    else:
        latent = latent.to(models.device)
    sampled = sample_windows(models, latent, layout, shot.seed(seed), done)
    for number in range(done, len(layout)):
        _begin(stop, f"{name}'s window {number + 1}/{len(layout)}")
        logger.debug("%s: window %d/%d", name, number + 1, len(layout))
        started = _started(models)
        output = next(sampled)
        finite(output, f"{name}'s window {number + 1}/{len(layout)}")
        units.save_window(index, number, output)
        start, end = layout[number]
        logger.info(
            "%s: window %d/%d, latents %d to %d, in %s",
            name,
            number + 1,
            len(layout),
            start,
            end - 1,
            _spent(models, started),
        )
    del sampled, latent
    units.drop_latent(index)


def _decode(
    models: Models,
    shots: Sequence[Shot],
    index: int,
    target: tuple[int, int],
    window: int | None,
    units: Units,
    write: Callable[[npt.NDArray[np.float32]], None],
) -> None:
    """Shot `index`'s decode, from its windows kept, its frames written as they come."""
    shot = shots[index]
    name = f"shot {index + 1}/{len(shots)}"
    logger.debug("%s: decoding", name)
    started = _started(models)
    merged = merge_windows(units.take_windows(index), shot_layout(shot.frames, window))
    decode_shot(models, merged, shot.frames, target, write, name, shot.start)
    logger.info(
        "%s: decoded in %s; RAM %.2f GiB, peak %.2f GiB",
        name,
        _spent(models, started),
        _resident() / GIB,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
    )


class Inputs:
    """The job's input frames, read in order: a part's decoder is opened at the first frame asked
    of it, and finished once every frame of the part is read, its count checked (Decoder). Frames
    not asked for, those of shots a resume skips, are decoded and dropped: exact, as the first
    pass counts them (provisional: seeking is DESIGN.md's open question, Frame-exact access). Used
    as a context manager, which stops a decoder left open on an exception."""

    def __init__(self, parts: Sequence[Part]) -> None:
        self._parts = parts
        self._part: Part | None = None
        self._decoder: Decoder | None = None
        self._position = 0  # the job's index of the next frame the open decoder gives

    def reader(self, shot: Shot) -> Callable[[int], npt.NDArray[np.float32]]:
        """read(n), giving the next n of shot's frames, (n, H, W, 3) float32 in [0, 1], from its
        first: shots are asked for in order, the frames between them dropped."""
        part = next(part for part in self._parts if part.start <= shot.start < part.end)
        if part is not self._part:
            self._close()
            self._part, self._decoder, self._position = part, part.source.decoder(), part.start
        decoder = self._decoder
        assert decoder is not None
        if self._position > shot.start:
            raise ValueError(f"frame {shot.start} asked, frame {self._position} is next")
        if self._position < shot.start:
            skipped = decoder.skip(shot.start - self._position)
            if skipped != shot.start - self._position:
                raise MediaError(
                    f"{part.source.path}: the stream ended after {decoder.decoded} frames"
                )
            self._position = shot.start

        def read(count: int) -> npt.NDArray[np.float32]:
            frames = decoder.read(count)
            if frames.shape[0] != count:
                raise MediaError(
                    f"{part.source.path}: the stream ended after {decoder.decoded} frames"
                )
            self._position += count
            return to_float32(frames)

        return read

    def _close(self) -> None:
        """Finish the open decoder, its every frame read; else stop it."""
        decoder, part = self._decoder, self._part
        self._decoder, self._part = None, None
        if decoder is None or part is None:
            return
        if self._position == part.end:
            decoder.finish()
        else:
            decoder.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self._close()
        elif self._decoder is not None:
            self._decoder.stop()


def _started(models: Models) -> float:
    torch.cuda.reset_peak_memory_stats(models.device)
    return time.monotonic()


def _spent(models: Models, started: float) -> str:
    """The time since started and the VRAM peak since, as the logs give them."""
    peak = torch.cuda.max_memory_allocated(models.device) / GIB
    return f"{time.monotonic() - started:.1f} s, VRAM peak {peak:.2f} GiB"


def _resident() -> int:
    """The process's resident memory now, in bytes (Linux)."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * resource.getpagesize()
