"""A shot, the model's unit between two cuts, and its DiT windows (DESIGN.md, Pipeline step 2):
the layout of the windows over a shot's latents and the weights their shared latents are mixed with.
Plain Python, no torch: the planner lays the windows out before torch is imported."""

import math
from dataclasses import dataclass

from seedvr2x.frames import padded_length

# Latents shared by consecutive DiT windows, M (DESIGN.md, Pipeline step 2): M = 2 cut the boundary
# jump by 80% against independent batches in the stitching study (research/docs/stitching.md).
SHARED = 2


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


# window_layout lives here, in plain Python, as frames.padded_length does: the planner's arithmetic
# (shot_plan.py, the command line's resume.py and manifest.py) needs them before torch is imported
# (impl-v010, 2026-10-28: a resume's early reading of the card, main._early_memory, runs before the
# allocator is chosen).
def window_layout(latents: int, window: int, shared: int = SHARED) -> list[tuple[int, int]]:
    """[start, end) of the windows over a shot of `latents` latents: as few windows as `window`
    allows, their lengths balanced (differing by at most 1, the longer first), consecutive ones
    sharing `shared` latents (DESIGN.md, Pipeline step 2)."""
    if latents <= window:
        return [(0, latents)]
    if window < 2 * shared + 1:
        raise ValueError(
            f"windows of {window} latents leave none of their own with {shared} shared"
        )
    count = math.ceil((latents - shared) / (window - shared))
    base, longer = divmod(latents + (count - 1) * shared, count)
    layout: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        length = base + 1 if index < longer else base
        layout.append((start, start + length))
        start += length - shared
    # A latent is in at most two windows: a window is at least as long as the two zones it
    # shares with its neighbours.
    assert layout[-1][1] == latents and base >= 2 * shared
    return layout


def mix_weights(shared: int) -> list[float]:
    """The earlier window's weights on the latents two windows share: cosine, 0.75 then 0.25 for
    2, the curve the stitching study measured (blend_patch.py, weights)."""
    return [0.5 + 0.5 * math.cos(math.pi * i / (shared + 1)) for i in range(1, shared + 1)]


def shot_layout(count: int, window: int | None) -> list[tuple[int, int]]:
    """The DiT windows of a shot of `count` frames (window_layout), capped at `window` latents;
    None runs the shot in one window."""
    latents = (padded_length(count) - 1) // 4 + 1
    return window_layout(latents, window or latents)
