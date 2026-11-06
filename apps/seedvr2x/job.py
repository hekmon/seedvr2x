"""A job's layout, before any GPU work: its source's shots and their seeds, its output segments
(DESIGN.md, Input, Pipeline and Output); the frame's sizes are frames.py's. Plain Python, no
torch: a bad cut list is refused before the models load."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from seedvr2x.errors import JobError
from seedvr2x.frames import output_size, short_side, target_size
from seedvr2x.passes import ENCODE_SEED_OFFSET
from seedvr2x.shotruns import KEPT_SAMPLE_BYTES as KEPT_SAMPLE_BYTES
from seedvr2x.shotruns import PLACES_BEST_FIRST as PLACES_BEST_FIRST
from seedvr2x.shotruns import QUALITY as QUALITY
from seedvr2x.shotruns import RUN_OFFSETS as RUN_OFFSETS
from seedvr2x.shotruns import Run as Run
from seedvr2x.shotruns import kept_bytes as kept_bytes
from seedvr2x.shotruns import kept_frames as kept_frames
from seedvr2x.shotruns import place as place
from seedvr2x.shotruns import run_of as run_of
from seedvr2x.shotruns import shot_runs as shot_runs
from seedvr2x.shots import Shot

# numz seeds Python's, NumPy's and torch's generators with each seed (common/seed.py), and NumPy
# takes seeds in [0, 2**32). numz seeds again before the VAE encode, with the seed plus
# ENCODE_SEED_OFFSET (seedvr2x/passes.py): seedvr2x/shot_pass.py does the same.
SEED_LIMIT = 2**32

# --resolution's default, the output's short side: numz's (DESIGN.md, Options kept and dropped). The
# parser has no value by default, so that a size given is told from the default (the size line,
# default_smaller), and each path takes this where it reads the option, as --format's is.
DEFAULT_RESOLUTION = 1080


# The runs of a shot, the levels of --quality and the frames each run keeps are the library's
# (seedvr2x.shotruns, which clip.py runs a clip's shots by), imported above and re-exported under
# the names the command's modules and tests use.


@dataclass(frozen=True)
class OutputSegment:
    """Frames [start, end) of the job, the output's unit: a file of their own (FFV1) or a directory
    (PNG), named name (DESIGN.md, Output)."""

    name: str
    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


# sptenc's minimum segment length, its -L default (cmd/sptenc/flags.go, minSegmentLengthDefault):
# short segments cost an encoder a keyframe each and too few frames to amortise it (DESIGN.md,
# Output).
MIN_SEGMENT = Fraction(5)


def min_segment_frames(seconds: Fraction, frame_rate: Fraction) -> int:
    """The fewest frames lasting at least `seconds` at frame_rate: a segment shorter than that is
    too short. sptenc's durationToFrames rounding up (core/scenes.go), exact: 5 s is 125 frames
    at 25 fps, 120 at 24000/1001, 150 at 30000/1001."""
    return math.ceil(seconds * frame_rate)


def merge_short(cuts: Sequence[int], frames: int, min_frames: int) -> list[int]:
    """The cuts left once every segment of a source of `frames` frames cut at `cuts` lasts at
    least min_frames: sptenc's FilterShortScenes (core/scenes.go:92-188 at vmafv1 5790944), in
    frames as it counts them. Repeatedly, the shortest segment too short (the first of equals)
    merges into its shorter neighbour, the left one on a tie; the first and the last into their
    only one. A merge removes the cut between the two. It stops when no segment is too short, or
    no cut is left.

    sptenc takes the last segment's end from the container's duration, rounded to a frame; here
    it is the frames counted, its exact value."""
    if min_frames <= 0 or not cuts:
        return list(cuts)
    kept = list(cuts)
    lengths = [kept[0], *(b - a for a, b in pairwise(kept)), frames - kept[-1]]
    while kept:
        shortest = -1
        for index, length in enumerate(lengths):
            if length < min_frames and (shortest == -1 or length < lengths[shortest]):
                shortest = index
        if shortest == -1:
            break
        # Into the neighbour on the left (shortest - 1) or the right (shortest + 1); the cut
        # between them is the one at the end of the left one.
        if shortest == 0:
            left = 0
        elif shortest == len(lengths) - 1:
            left = shortest - 1
        else:
            left = shortest - 1 if lengths[shortest - 1] <= lengths[shortest + 1] else shortest
        lengths[left : left + 2] = [lengths[left] + lengths[left + 1]]
        del kept[left]
    return kept


def merged_segments(
    shots: Sequence[Shot], frames: int, frame_rate: Fraction, min_seconds: Fraction
) -> list[OutputSegment]:
    """The output segments: the shots' cuts, merged by sptenc's rule to last min_seconds at
    least (merge_short), named as sptenc's split names its own (seg_%06d, ffmpeg/segment.go).
    Shots keep every cut: a segment holds whole shots (DESIGN.md, Output)."""
    kept = merge_short(
        [shot.start for shot in shots[1:]], frames, min_segment_frames(min_seconds, frame_rate)
    )
    bounds = [0, *kept, frames]
    return [
        OutputSegment(f"seg_{index:06d}", start, end)
        for index, (start, end) in enumerate(pairwise(bounds))
    ]


def default_smaller(name: str, width: int, height: int, sample_aspect: Fraction) -> str | None:
    """The line for a picture or a video of `width` x `height` (stored pixels, at `sample_aspect`)
    when --resolution was not given and its default makes the output smaller than the source as
    displayed (DESIGN.md, Still images, Size; decided on 2026-11-03 for a 12 MP photo, 4032x3024,
    which comes out 1440x1080); None where it does not. It names the --resolution that keeps the
    source's size (frames.short_side). A size given says nothing more."""
    keeps = short_side(width, height, sample_aspect)
    if DEFAULT_RESOLUTION >= keeps:
        return None
    rows, columns = output_size(target_size(width, height, sample_aspect, DEFAULT_RESOLUTION))
    return (
        f"{name}: no --resolution given, so the output is {columns}x{rows}, its short side the"
        f" default {DEFAULT_RESOLUTION}, smaller than its source's {round(width * sample_aspect)}x"
        f"{height}: --resolution {keeps} keeps the source's size"
    )


def check_seed(seed: int, shots: Sequence[Shot]) -> None:
    """Refuse a seed some shot's seed would put out of numz's range (SEED_LIMIT): the seed plus
    the shot's first frame, and that plus ENCODE_SEED_OFFSET."""
    highest = shots[-1].seed(seed) + ENCODE_SEED_OFFSET
    if seed < 0 or highest >= SEED_LIMIT:
        raise JobError(
            f"--seed {seed}: each shot is seeded with the seed plus its first frame, and its"
            f" encode with {ENCODE_SEED_OFFSET} more, so the seed must be between 0 and"
            f" {SEED_LIMIT - 1 - ENCODE_SEED_OFFSET - shots[-1].start} for this source"
        )


def read_cuts(path: Path) -> list[int]:
    """A cut list (DESIGN.md, Input): the first frame of each shot but the first, one frame number
    per line, counted from 0 in the source, no timestamps; blank lines and # comments are ignored.
    So are the fields after the frame number, separated by blanks, so that an export carrying
    scores (sptenc's, once it has one) stays readable."""
    try:
        text = path.read_text()
    except OSError as error:
        raise JobError(f"{path}: {error.strerror}") from None
    cuts: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if not re.fullmatch(r"\d+", fields[0]):
            raise JobError(f"{path}:{number}: {fields[0]!r} is not a frame number")
        cuts.append(int(fields[0]))
    return cuts


def shots_from_cuts(cuts: list[int], frames: int) -> list[Shot]:
    """The shots of a source of `frames` frames cut at `cuts` (read_cuts, check_cuts)."""
    check_cuts(cuts, frames)
    bounds = [0, *cuts, frames]
    return [Shot(start, end) for start, end in pairwise(bounds)]


def check_cuts(cuts: Sequence[int], frames: int) -> None:
    """Refuse a cut list that isn't one of a source of `frames` frames: a cut is the first frame
    of a shot, so it lies within the source, after frame 0, and they increase."""
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
