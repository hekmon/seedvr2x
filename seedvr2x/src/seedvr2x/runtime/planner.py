"""The memory planner (DESIGN.md, Memory planner): one plan per job, made after the first pass,
whose shots are its input, and before the models load. No GPU here: what is free comes in as a
figure.

PROVISIONAL (milestone 3's seam): a stand-in until the planner is built. It chooses nothing: the
pins are the plan, each shot in one window without --window, every weight on the GPU, as before
milestone 3; and it refuses the pins whose work isn't built yet, each refusal leaving with the
commit that builds it."""

from dataclasses import dataclass

from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.plan import Plan, Tiles


class PlanError(JobError):
    """A job the planner refuses before the run: a pin it can't take exactly, or a size that
    doesn't fit the memory free."""


@dataclass(frozen=True)
class Pins:
    """The choices the command line fixes (DESIGN.md, Memory planner, Pins), each taken exactly or
    refused, never adjusted; None leaves a choice to the planner."""

    window: int | None = None  # --window: the cap on the DiT windows, in latents
    encode_tile: int | None = None  # --encode-tile: a tile's side in output pixels; 0: untiled
    decode_tile: int | None = None  # --decode-tile, likewise
    swap_blocks: int | None = None  # --swap-blocks: the DiT's blocks swapped


def plan(pins: Pins) -> Plan:
    """The job's plan: the pins as they are, nothing chosen around them yet."""
    return Plan(
        window=pins.window,
        encode_tiles=_tiles("--encode-tile", pins.encode_tile),
        decode_tiles=_tiles("--decode-tile", pins.decode_tile),
        swapped=_swapped(pins.swap_blocks),
    )


def _tiles(option: str, size: int | None) -> Tiles | None:
    """A tile pin's tiles; None without a pin or at 0, an untiled pass."""
    if size:
        # Leaves with the tiled passes' commit (shot.encode_shot, shot.decode_shot).
        raise PlanError(f"{option} {size}: the tiled passes aren't built yet")
    return Tiles(size) if size else None


def _swapped(blocks: int | None) -> int:
    """A --swap-blocks pin's blocks; none without a pin."""
    if blocks is not None and blocks > 36:  # the blocks of the 7B, the DiT v1 runs
        raise PlanError(f"--swap-blocks {blocks}: the DiT has 36 blocks")
    return blocks or 0
