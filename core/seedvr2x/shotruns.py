"""The runs of a shot, and the frames each one keeps (DESIGN.md, Several runs of a shot, an option
for quality): the levels of `--quality` and `quality`, each run's first frame in its shot, the run
each of a shot's frames is taken from, and the levels that take each frame of a shot alone. A brick
of the library, plain Python and no torch: the command line plans, records and resumes by it, and
clip.py runs a clip's shots by it, so that both choose the same frames. A module of its own, not
shots.py: shots.py is the shot and its windows, which every pass imports; this is the layer above it
that a job's level reads."""

from collections.abc import Sequence
from dataclasses import dataclass

from seedvr2x.shots import Shot

# The runs of a shot --quality offers, and each one's first frame in the shot (DESIGN.md,
# Several runs of a shot, decided on 2026-10-20): the latent grid shifted by 0 to 3 frames with
# 4, by 0 and 2 with 2, the default one run, the grid unshifted.
RUN_OFFSETS: dict[int, tuple[int, ...]] = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3)}

# The levels of --quality, each with its runs a shot (DESIGN.md, Several runs of a shot: the user,
# 2026-10-20, a name a user understands, the runs a shot being the code's): standard, the default,
# one run; enhanced two, offered on the user's blind verdict (two runs beat one on 7 of 8 stills);
# maximum four (four beat two on 7 of 8); deghost one, each of its frames through the pass alone
# (EACH_FRAME).
QUALITY = {"standard": 1, "enhanced": 2, "maximum": 4, "deghost": 1}

# The levels whose run takes each frame of its shot alone (DESIGN.md, Each frame on its own:
# `--quality deghost`, decided on 2026-11-17): a frame upscaled as a shot of one frame is, in a
# latent of its own, its own encode, a window of one latent, its decode and its colour correction
# against its own source frame, every frame with the shot's seed (Shot.seed), so that the noise
# does not change from frame to frame inside a shot. On six shots of an animated episode the DiT
# blended a latent's 4 drawings, a quarter of the later frames' lines in a group's first, none in a
# frame alone; a new noise for each frame flickered where the picture held still, the shot's noise
# brought it back to standard's level (impl-box-7). The output of a shot is bit for bit the shot's
# pass run on each of its frames as a one-frame shot seeded by the shot's seed, frame after frame
# (core/tests/test_clip.py; the command line's, apps/seedvr2x/tests/test_clip_cli.py).
EACH_FRAME = frozenset({"deghost"})

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


def each_frame(level: str) -> bool:
    """Whether `level`, a level of QUALITY, takes each frame of a shot alone (EACH_FRAME)."""
    return level in EACH_FRAME


def frame_layout(frames: int) -> list[tuple[int, int]]:
    """The windows of a shot of `frames` frames whose frames are each taken alone (each_frame), in
    latents: one latent a frame, each a window of its own, [(0, 1), (1, 2), ...]."""
    if frames < 1:
        raise ValueError(f"a shot of {frames} frames")
    return [(frame, frame + 1) for frame in range(frames)]


def planned_shots(shots: Sequence[Shot], level: str) -> list[Shot]:
    """The shots a plan is made for at `level`: the shots themselves, or, where each frame is
    taken alone (each_frame), a shot of one frame, which each of their frames is, so that the
    plan's windows, passes and pins are a frame's (the longest shot decides a plan,
    shot_plan.plan). The shots' own count and lengths still say the job's frames and time."""
    if each_frame(level):
        return [Shot(0, 1)]
    return list(shots)


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
    "EACH_FRAME",
    "KEPT_SAMPLE_BYTES",
    "PLACES_BEST_FIRST",
    "QUALITY",
    "RUN_OFFSETS",
    "Run",
    "each_frame",
    "frame_layout",
    "kept_bytes",
    "kept_frames",
    "place",
    "planned_shots",
    "run_of",
    "shot_runs",
    "taken_from",
]
