"""A job's plan (DESIGN.md, Memory planner): what the planner chooses once for the job, the same
for every shot, and the run follows: the cap on the DiT windows, the VAE's tiles, the DiT's
swapped blocks, what stays on the GPU in each phase, and what a tiled shot's scratch may take of
system memory. Made by runtime/planner.py; recorded in the manifest (record), and read back from
it by a resume (Plan.recorded). Nothing here imports torch: the command line plans before it
does."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast

from seedvr2x.runtime.job import JobError

# The tiles' overlap, in output pixels, fixed (DESIGN.md, Pipeline, per shot, step 3): the colour
# study's tiles after split and the recipes validated per card size ran at 64, and 512-px decode
# tiles at 1080p gained nothing at 128 (research/docs/vram.md, Tiling).
OVERLAP = 64

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
class Plan:
    """A job's plan. Its first part shapes the output, where a long shot's joins fall and where
    the picture is cut, and a resume takes it as recorded; the rest only says where the weights
    are and where a tiled shot's frames wait, which changes no bit, and a resume plans it again
    (DESIGN.md, Memory planner)."""

    # What shapes the output.
    window: int | None  # the cap on the DiT windows, in latents; None: each shot in one window
    encode_tiles: Tiles | None = None  # None: the encode in one pass, untiled
    decode_tiles: Tiles | None = None  # None: the decode streamed, untiled
    # What doesn't.
    swapped: int = 0  # the DiT's last blocks swapped, streamed through the two slots
    # What of the DiT is on the GPU during the VAE's encodes and decodes, the rest waiting in
    # its pinned copies (blockswap.Placement.enter). None: nothing moves, the DiT stays as its
    # own phase has it, slots included. 0: nothing of its parameters. A number, up to the blocks
    # that aren't swapped: that many of them, the first ones, with its parameters outside the
    # blocks, the slots freed.
    resident_encode: int | None = None
    resident_decode: int | None = None
    vae_in_dit: bool = True  # False: the VAE's weights leave the GPU for the DiT's phase
    # What a tiled shot's scratch may take of system memory, in bytes, a larger one going to its
    # file (tiling.Scratch): the planner's, from the host's memory (DESIGN.md, Memory planner,
    # Host RAM); 0 when no pass is tiled.
    scratch_allowance: int = 0
    # torch.compile (DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0; the
    # DiT, not", decided on 2026-10-24): the VAE compiled with --compile, its causal
    # convolutions left out of the graph (runtime/compiling.py), always as the user set it
    # (planner.plan refuses a job it can't run compiled, decided on 2026-10-29). The DiT is never
    # compiled in v0.1.0:
    # compile_dit stays recorded, false. What the output's bits depend on, so a resume takes it
    # as recorded.
    compile_dit: bool = False
    compile_vae: bool = False

    @classmethod
    def recorded(cls, record: Any) -> "Plan":
        """The plan a manifest records (record), as a resume takes it: what shapes the output
        alone, the cap, the tiles and the compile flags. Where the weights were isn't read, nor
        the scratch's allowance: a resume plans them again, from the memory free that day
        (DESIGN.md, Memory planner, A plan shapes the output a little). Refused (JobError) when
        the record isn't a plan's."""

        def tiles(name: str) -> Tiles | None:
            entry: Any = fields.get(name)
            if entry is None:
                return None
            if isinstance(entry, dict):
                sides = cast(dict[str, Any], entry)
                size: Any = sides.get("size")
                overlap: Any = sides.get("overlap")
                if _whole(size) and _whole(overlap) and 0 <= overlap < size:
                    return Tiles(size, overlap)
            raise JobError(f"not a plan as seedvr2x records one: its {name}, {entry!r}")

        if not isinstance(record, dict):
            raise JobError(f"not a plan as seedvr2x records one: {record!r}")
        fields = cast(dict[str, Any], record)
        window: Any = fields.get("window")
        if window is not None and not (_whole(window) and window >= 1):
            raise JobError(f"not a plan as seedvr2x records one: its window, {window!r}")
        flags: list[bool] = []
        for name in ("compile_dit", "compile_vae"):
            flag: Any = fields.get(name)
            if not isinstance(flag, bool):
                raise JobError(f"not a plan as seedvr2x records one: its {name}, {flag!r}")
            flags.append(flag)
        return cls(
            window,
            tiles("encode_tiles"),
            tiles("decode_tiles"),
            compile_dit=flags[0],
            compile_vae=flags[1],
        )

    def record(self) -> dict[str, Any]:
        """The plan as the manifest records it: what shapes the output, compared on a resume as
        the layout is, and under `memory` what doesn't, for information (resume.UNCOMPARED)."""

        def tiles(tiles: Tiles | None) -> dict[str, int] | None:
            return None if tiles is None else {"size": tiles.size, "overlap": tiles.overlap}

        return {
            "window": self.window,
            "encode_tiles": tiles(self.encode_tiles),
            "decode_tiles": tiles(self.decode_tiles),
            "compile_dit": self.compile_dit,
            "compile_vae": self.compile_vae,
            "memory": {
                "swapped_blocks": self.swapped,
                "resident_encode": self.resident_encode,
                "resident_decode": self.resident_decode,
                "vae_in_dit": self.vae_in_dit,
                "scratch_allowance": self.scratch_allowance,
            },
        }


@dataclass(frozen=True)
class Shapes:
    """The shapes a compiled VAE is warmed with (planner.compile_shapes), each (rows, columns) in
    pixels of the padded frames, `padded` being their size (job.padded_size): the encoder's input,
    one per distinct tile of the plan's encode (the whole frame untiled), and the decoder's
    output, likewise for the decode; `latents`, the shots' lengths the slices are warmed at, in
    latents, 1, 2 or planner.COMPILE_LATENTS (3), the shortest of each kind the job runs."""

    padded: tuple[int, int]  # the padded frame (job.padded_size), (rows, columns)
    encode: tuple[tuple[int, int], ...]
    decode: tuple[tuple[int, int], ...]
    latents: tuple[int, ...]


@dataclass
class Compilation:
    """What model.load_models compiles the VAE with, once the plan has compile_vae
    (runtime/compiling.py): the shapes to warm, the job's compile cache (a directory under the
    job's own, manifest.COMPILE_CACHE), and, set by the load, the seconds the compilation took
    in the run's own clock, 0.0 until then and where nothing is compiled."""

    shapes: Shapes
    cache: Path
    seconds: float = 0.0


def _whole(value: Any) -> TypeGuard[int]:
    """Whether a value read from a record is a whole number: an int, and no bool."""
    return isinstance(value, int) and not isinstance(value, bool)
