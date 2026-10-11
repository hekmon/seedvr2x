"""A job's plan (DESIGN.md, Memory planner): what the planner chooses once for the job, the same
for every shot, and the run follows: the cap on the DiT windows, the VAE's tiles, the DiT's
swapped blocks and what stays on the GPU in each phase. Made by runtime/planner.py; recorded in
the manifest (record). Nothing here imports torch: the command line plans before it does."""

from dataclasses import dataclass
from typing import Any, Literal

# The tiles' overlap, in output pixels, fixed (DESIGN.md, Pipeline, per shot, step 3): the colour
# study's tiles after split and the recipes validated per card size ran at 64, and 512-px decode
# tiles at 1080p gained nothing at 128 (research/docs/vram.md, Tiling).
OVERLAP = 64

# A phase of the run, each unit in one (Models.enter): a shot's encode, a window, a decode.
Phase = Literal["encode", "dit", "decode"]


@dataclass(frozen=True)
class Tiles:
    """A tiled VAE pass's tiles: squares of `size` output pixels a side, cut off at the picture's
    edges, each sharing `overlap` pixels with its neighbours."""

    size: int
    overlap: int = OVERLAP


@dataclass(frozen=True)
class Plan:
    """A job's plan. Its first part shapes the output, where a long shot's joins fall and where
    the picture is cut, and a resume takes it as recorded; the rest only says where the weights
    are, which changes no bit, and a resume plans it again (DESIGN.md, Memory planner)."""

    # What shapes the output.
    window: int | None  # the cap on the DiT windows, in latents; None: each shot in one window
    encode_tiles: Tiles | None = None  # None: the encode in one pass, untiled
    decode_tiles: Tiles | None = None  # None: the decode streamed, untiled
    # What doesn't.
    swapped: int = 0  # the DiT's last blocks swapped, streamed through the two slots
    # The DiT's blocks on the GPU during the VAE's encodes and decodes, the others waiting in
    # their pinned copies (0: every block away); None: the DiT stays as its own phase has it.
    resident_encode: int | None = None
    resident_decode: int | None = None
    vae_in_dit: bool = True  # False: the VAE's weights leave the GPU for the DiT's phase
    # torch.compile, not built with milestone 3 (DESIGN.md, Memory planner): recorded, false, so
    # that building it later reshapes no record.
    compile_dit: bool = False
    compile_vae: bool = False

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
            },
        }
