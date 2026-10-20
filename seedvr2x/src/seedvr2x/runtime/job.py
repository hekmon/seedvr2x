"""A job's layout, before any GPU work: its source's shots and their seeds, its output segments,
the output size and the frames' padding (DESIGN.md, Input, Pipeline and Output). Plain Python, no
torch: a bad cut list or target is refused before the models load."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

# numz seeds Python's, NumPy's and torch's generators with each seed (common/seed.py), and NumPy
# takes seeds in [0, 2**32). numz seeds again before the VAE encode, with the seed plus
# ENCODE_SEED_OFFSET (nothing draws then): runtime/shot.py does the same.
SEED_LIMIT = 2**32
ENCODE_SEED_OFFSET = 1_000_000

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


# The runs of a shot --quality offers, and each one's first frame in the shot (DESIGN.md,
# Several runs of a shot, decided on 2026-10-20): the latent grid shifted by 0 to 3 frames with
# 4, by 0 and 2 with 2, the default one run, the grid unshifted.
RUN_OFFSETS = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3)}

# The levels of --quality, each with its runs a shot (DESIGN.md, Several runs of a shot: the user,
# 2026-10-20, a name a user understands, the runs a shot being the code's): standard, the default,
# one run; enhanced two, offered on the user's blind verdict (two runs beat one on 7 of 8 stills);
# maximum four (four beat two on 7 of 8).
QUALITY = {"standard": 1, "enhanced": 2, "maximum": 4}

# A frame's places in a run, best first (DESIGN.md, Several runs of a shot): its first frame, a
# latent of its own (-1), then the last of a group of 4 (3), the one a latent is stamped on in the
# VAE's causal wiring, then 2 and 0, then 1. Measurement's order (the grid study's compose.py,
# pick: "j = 0 (own latent) first, then places 3, 2, 0, 1"; place 2 above place 0 on 5 of 8
# clips, research/docs/numerics.md, Every fourth frame: the latent grid).
PLACES_BEST_FIRST = (-1, 3, 2, 0, 1)


def place(index: int) -> int:
    """The place of a run's frame `index`, counted from the run's first: -1 for its first frame,
    which the VAE encodes alone into a latent, else 0 to 3 in its group of 4 frames, 3 the
    group's last (DESIGN.md, Pipeline step 1)."""
    if index < 0:
        raise ValueError(f"frame {index} of a run")
    return -1 if index == 0 else (index - 1) % 4


def run_of(runs: int, frame: int) -> int:
    """The run a shot's frame `frame`, counted from the shot's first, is taken from with `runs`
    runs a shot: the first frame of that run in the shot, RUN_OFFSETS's. Of the runs that hold the
    frame, those starting at it or before, the one where its place is best (PLACES_BEST_FIRST):
    with 4, the run frame mod 4, where the frame ends its group or is the run's first; with 2,
    the run where its place is 3 or 2, or frame 2's own run, where it is the first. Every run's
    first frame is its own, so a shot with a run s has it give at least that frame. The same
    choice as measurement's composites (compose.py's pick), held to it by
    tests/test_runs.py."""
    if frame < 0:
        raise ValueError(f"frame {frame} of a shot")
    held = [offset for offset in RUN_OFFSETS[runs] if offset <= frame]
    # The first best: no two runs give a frame one place (tests/test_runs.py), but a tie would go
    # to the earlier run, as compose.py's does.
    return min(held, key=lambda offset: PLACES_BEST_FIRST.index(place(frame - offset)))


@dataclass(frozen=True)
class Run:
    """Run `offset` of shot: the shot fed from its frame `offset`, frames [shot.start + offset,
    shot.end) of the source, a whole run of its own, its plan of windows, its padded end and
    its colour correction (DESIGN.md, Several runs of a shot). Its seed is the shot's
    (Shot.seed), as each of measurement's shifted runs took the job's seed, 42, its input a file
    of its own starting at the shifted frame. PROVISIONAL (implementation, 2026-10-20, a
    question for design): the shot's own seed, not the one a shot starting at the run's first
    frame would take, its start in the source: so that the composite is measurement's, and a
    run's output depends on the seed, the shot and the run alone."""

    shot: Shot
    offset: int

    @property
    def start(self) -> int:
        return self.shot.start + self.offset

    @property
    def end(self) -> int:
        return self.shot.end

    @property
    def frames(self) -> int:
        return self.shot.end - self.start

    def seed(self, seed: int) -> int:
        """The run's seed: its shot's (Shot.seed)."""
        return self.shot.seed(seed)


def kept_frames(shot: Shot, runs: int) -> int:
    """The frames of shot its runs but the last give, which wait on disk while its runs are
    decoded in turn (runtime/run.py, Kept): none with one run."""
    last = shot_runs(shot, runs)[-1].offset
    return sum(run_of(runs, frame) != last for frame in range(shot.frames))


def shot_runs(shot: Shot, runs: int) -> list[Run]:
    """The runs of shot with `runs` runs a shot, in order: one for each of RUN_OFFSETS's first
    frames the shot has; a shot too short to have frame s has no run s."""
    return [Run(shot, offset) for offset in RUN_OFFSETS[runs] if offset < shot.frames]


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


# The padding of the resized picture before the model (DESIGN.md, Pipeline step 0; measured in
# research/docs/numerics.md, reflect+black+16): at the bottom, the fewest rows reflected from the
# picture, at least REFLECTED, that bring it to a multiple of MULTIPLE (8 to 23), then BLACK rows
# of zeros; at the right, columns the same way, only when the width isn't a multiple of MULTIPLE.
# All are trimmed after the decode (shot._decoded). numz pads zeros alone, up to the multiple (8
# rows at 1080p, none at 720p), which costs the bottom 16 rows 3-10 dB, while the black rows
# anchor the model's tone: without them the whole frame is worse.
MULTIPLE = 16
REFLECTED = 8
BLACK = 16


def padding(size: tuple[int, int]) -> tuple[int, int]:
    """(rows, columns) reflected from a picture of size (height, width), before the black ones:
    the fewest rows, at least REFLECTED, that bring its height to a multiple of MULTIPLE, 8 to
    23; columns the same way, none when its width is a multiple of MULTIPLE already. 1080 rows
    get 8, 720 and 2160 get 16; 1920 columns none, 1912 get 8."""
    height, width = size
    return _reflected(height), _reflected(width) if width % MULTIPLE else 0


def _reflected(side: int) -> int:
    return (-side - REFLECTED) % MULTIPLE + REFLECTED


def padded_size(size: tuple[int, int]) -> tuple[int, int]:
    """(height, width) of a picture of size (height, width) once padded, as the model takes it:
    its rows reflected (padding) and BLACK more, its columns alike when any are reflected; each a
    multiple of MULTIPLE. 1080x1920 is padded to 1104x1920, 720x1280 to 752x1280, 1060x1912 to
    1088x1936."""
    height, width = size
    rows, columns = padding(size)
    return height + rows + BLACK, width + columns + (BLACK if columns else 0)


def check_target(target: tuple[int, int], resolution: int) -> None:
    """Refuse a target, (height, width), the padding can't be made on: torch reflects fewer rows
    than the picture has (torch.nn.functional.pad, mode reflect), and columns alike, so it takes
    17 rows at least (16 would take 16 reflected) and 16 columns (16 take none; fewer would take
    at least as many as they are). --resolution, the target's short side, takes any whole number
    above 0 (cli.py)."""
    height, width = target
    rows, columns = padding(target)
    if rows >= height or columns >= width:
        raise JobError(
            f"--resolution {resolution}: the frames resized to {width}x{height} are too small"
            f" for the padding, at least {REFLECTED} rows reflected from the picture under it,"
            f" up to a multiple of {MULTIPLE}, and columns alike: it takes 17 rows and 16"
            " columns at least"
        )
