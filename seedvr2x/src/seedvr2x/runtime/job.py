"""A job's layout, before any GPU work: its shots, their seeds, the output size (DESIGN.md, Input
and Pipeline). Plain Python, no torch: a bad cut list is refused before the models load."""

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

# Latents shared by consecutive DiT windows, M (DESIGN.md, Pipeline step 2): M = 2 cut the boundary
# jump by 80% against independent batches in the stitching study (research/docs/stitching.md).
SHARED = 2


class JobError(ValueError):
    """A job seedvr2x can't lay out: a bad cut list; the message says what."""


@dataclass(frozen=True)
class Shot:
    """Frames [start, end) of the source: the model's unit, between two cuts. Its noise seed is
    the job's plus start (DESIGN.md, Pipeline step 2)."""

    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start

    def seed(self, seed: int) -> int:
        """The shot's seed: the job's seed plus its first frame's index in the source. A shot
        starting at frame 0 gets the plain seed, as numz's single batch does; shots never share a
        noise pattern; and a shot's output depends only on the seed, its frames and its place,
        so a resumed or re-cut job reproduces the untouched shots exactly."""
        return seed + self.start


def read_cuts(path: Path) -> list[int]:
    """A cut list: the first frame of each shot but the first, one frame number per line, counted
    from 0 in the source; blank lines and # comments are ignored.

    Provisional: frame numbers only, until DESIGN.md settles the cut list's format (Open
    questions, scene list format), timestamps included."""
    try:
        text = path.read_text()
    except OSError as error:
        raise JobError(f"{path}: {error.strerror}") from None
    cuts: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        entry = line.split("#", 1)[0].strip()
        if not entry:
            continue
        if not re.fullmatch(r"\d+", entry):
            raise JobError(f"{path}:{number}: {entry!r} is not a frame number")
        cuts.append(int(entry))
    return cuts


def shots_from_cuts(cuts: list[int], frames: int) -> list[Shot]:
    """The shots of a source of `frames` frames cut at `cuts` (read_cuts): a cut is the first
    frame of a shot, so it lies within the source, after frame 0, and they increase."""
    previous = 0
    for cut in cuts:
        if cut == 0:
            raise JobError("a cut at frame 0: the first shot starts there without one")
        if cut <= previous:
            raise JobError(f"cut at frame {cut} after one at {previous}: cuts must increase")
        previous = cut
    if cuts and cuts[-1] >= frames:
        raise JobError(
            f"cut at frame {cuts[-1]}: the source has {frames} frames (0 to {frames - 1})"
        )
    bounds = [0, *cuts, frames]
    return [Shot(start, end) for start, end in pairwise(bounds)]


def target_size(
    width: int, height: int, sample_aspect: Fraction, resolution: int
) -> tuple[int, int]:
    """(height, width) of the picture the input is resized to: the display picture's short side at
    resolution, the long side in proportion, rounded down. The output has square pixels at the
    source's display aspect (DESIGN.md, Colour and shape).

    For square pixels, this is the size NaResize (side mode) gives numz: torchvision's resize to an
    int size, the short side, the long one int(size * long / short), the width short when the
    picture is square (torchvision's _compute_resized_output_size)."""
    display_width = width * sample_aspect
    if display_width <= height:
        return math.floor(resolution * height / display_width), resolution
    return resolution, math.floor(resolution * display_width / height)


def output_size(target: tuple[int, int]) -> tuple[int, int]:
    """(height, width) of the output frames: the target, each side rounded down to an even
    number, as numz crops them (src/core/generation_utils.py:127-136)."""
    height, width = target
    return height - height % 2, width - width % 2
