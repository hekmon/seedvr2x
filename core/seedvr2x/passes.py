"""What the library's bricks take of a pass: the tiles of a tiled VAE pass, the phase of the run
a unit is in, the shapes and cache the VAE is compiled with, and what the loaded models follow of
a job's plan (ModelPlan). The library's (DESIGN.md, Repository and apps). Nothing here imports
torch: the command line plans before it does."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The tiles' overlap, in output pixels, fixed (DESIGN.md, Pipeline, per shot, step 3): the colour
# study's tiles after split and the recipes validated per card size ran at 64, and 512-px decode
# tiles at 1080p gained nothing at 128 (research/docs/vram.md, Tiling).
OVERLAP = 64

# The seed the VAE's encode is seeded with is the shot's or the picture's seed plus this: numz seeds
# Python's, NumPy's and torch's generators again before the encode, with the seed plus 1,000,000
# (numz src/core/generation_phases.py:329 at 4490bd1; nothing draws then). Here, not in sampling.py,
# which imports torch: the command line's job.py bounds a seed by it before torch is imported.
ENCODE_SEED_OFFSET = 1_000_000

# A phase of the run, each unit in one (Models.enter): a shot's encode, a window, a decode.
Phase = Literal["encode", "dit", "decode"]


@dataclass(frozen=True)
class Tiles:
    """A tiled VAE pass's tiles: squares of `size` output pixels a side, cut off at the picture's
    edges, each sharing `overlap` pixels with its neighbours. A tile that covers the whole padded
    frame is no tiling (DESIGN.md, Pipeline, per shot, step 3): the planner neither makes nor
    takes one (planner.covers)."""

    size: int
    overlap: int = OVERLAP


@dataclass(frozen=True)
class Shapes:
    """The shapes a compiled VAE is warmed with (planner.compile_shapes), each (rows, columns) in
    pixels of the padded frames, `padded` being their size (frames.padded_size): the encoder's
    input, one per distinct tile of the plan's encode (the whole frame untiled), and the decoder's
    output, likewise for the decode; `latents`, the shots' lengths the slices are warmed at, in
    latents, 1, 2 or planner.COMPILE_LATENTS (3), the shortest of each kind the job runs and the
    calibration's own (planner.COMPILE_CALIBRATED). The decoder's latent is warmed in one memory
    layout, the DiT's, at all of them (compiling.warm; the calibration's decodes take it in that
    layout too, calibration.calibrate)."""

    padded: tuple[int, int]  # the padded frame (frames.padded_size), (rows, columns)
    encode: tuple[tuple[int, int], ...]
    decode: tuple[tuple[int, int], ...]
    latents: tuple[int, ...]


@dataclass
class Compilation:
    """What model.load_models compiles the VAE with, once the plan has compile_vae
    (seedvr2x/compiling.py): the shapes to warm, the job's compile cache (a directory under the
    job's own, manifest.COMPILE_CACHE), and, set by the load, the seconds the compilation took
    in the run's own clock, 0.0 until then and where nothing is compiled."""

    shapes: Shapes
    cache: Path
    seconds: float = 0.0


@dataclass(frozen=True)
class ModelPlan:
    """What the loaded models follow (model.load_models, Models.enter): the DiT's swapped blocks,
    what of it stays on the GPU in each VAE phase, whether the VAE stays there in the DiT's phase,
    and whether the VAE is compiled. The command line's job plan gives it (plan.Plan.model_plan);
    the defaults are no plan's: nothing swapped, nothing sent away, nothing compiled."""

    swapped: int = 0  # the DiT's last blocks swapped, streamed through the two slots
    # What of the DiT is on the GPU during the VAE's encodes and decodes, the rest waiting in
    # its pinned copies (blockswap.Placement.enter). None: nothing moves, the DiT stays as its
    # own phase has it, slots included. 0: nothing of its parameters. A number, up to the blocks
    # that aren't swapped: that many of them, the first ones, with its parameters outside the
    # blocks, the slots freed.
    resident_encode: int | None = None
    resident_decode: int | None = None
    vae_in_dit: bool = True  # False: the VAE's weights leave the GPU for the DiT's phase
    # The VAE compiled (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0",
    # decided on 2026-10-24), its causal convolutions left out of the graph (compiling.py).
    compile_vae: bool = False
