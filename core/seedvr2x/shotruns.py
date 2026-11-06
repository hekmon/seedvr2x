"""The runs of a shot, and the frames each one keeps (DESIGN.md, Several runs of a shot, an option
for quality): the levels of `--quality` and `quality`, each run's first frame in its shot, and the
run each of a shot's frames is taken from. A brick of the library, plain Python and no torch: the
command line plans, records and resumes by it, and clip.py runs a clip's shots by it, so that both
choose the same frames. A module of its own, not shots.py: shots.py is the shot and its windows,
which every pass imports; this is the layer above it that only a job with several runs reads."""

from dataclasses import dataclass

from seedvr2x.shots import Shot

# The runs of a shot --quality offers, and each one's first frame in the shot (DESIGN.md,
# Several runs of a shot, decided on 2026-10-20): the latent grid shifted by 0 to 3 frames with
# 4, by 0 and 2 with 2, the default one run, the grid unshifted.
RUN_OFFSETS: dict[int, tuple[int, ...]] = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3)}

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
    apps/seedvr2x/tests/test_runs.py."""
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


def shot_runs(shot: Shot, runs: int) -> list[Run]:
    """The runs of shot with `runs` runs a shot, in order: one for each of RUN_OFFSETS's first
    frames the shot has; a shot too short to have frame s has no run s."""
    return [Run(shot, offset) for offset in RUN_OFFSETS[runs] if offset < shot.frames]


def taken_from(shot_frames: int, runs: int) -> list[int]:
    """The run each frame of a shot of `shot_frames` frames is taken from (run_of), as the run's
    first frame in the shot, one entry a frame in order. A run's frames are those it is
    taken from: it gives frame `offset` and every later frame whose entry is its offset."""
    return [run_of(runs, frame) for frame in range(shot_frames)]


def kept_frames(shot: Shot, runs: int) -> int:
    """The frames of shot its runs but the last give, which wait on disk while its runs are
    decoded in turn (apps/seedvr2x/run.py, Kept): none with one run."""
    last = shot_runs(shot, runs)[-1].offset
    return sum(run_of(runs, frame) != last for frame in range(shot.frames))


# The bytes of a sample of a frame kept between a shot's runs: the 16 bits the writers quantise a
# frame to (writer.quantise; DESIGN.md, Several runs of a shot).
KEPT_SAMPLE_BYTES = 2


def kept_bytes(shot: Shot, runs: int, size: tuple[int, int]) -> int:
    """The bytes of the frames shot keeps between its runs (kept_frames), (height, width) the
    output's size: 12.4 MB a 1080p frame, 49.8 MB at 4K."""
    height, width = size
    return kept_frames(shot, runs) * height * width * 3 * KEPT_SAMPLE_BYTES


__all__ = [
    "KEPT_SAMPLE_BYTES",
    "PLACES_BEST_FIRST",
    "QUALITY",
    "RUN_OFFSETS",
    "Run",
    "kept_bytes",
    "kept_frames",
    "place",
    "run_of",
    "shot_runs",
    "taken_from",
]
