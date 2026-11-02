"""The VAE tiling on the CPU (runtime/tiling.py): the grid, the scratch, and the tiled encode and
decode held bit for bit to numz's tiled_encode and tiled_decode, the vendored ones, over a stand-in
VAE and, once, over the real one with random weights."""

import contextlib
import gc
import logging
import os
import random
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from seedvr2x.passes import OVERLAP, Tiles
from seedvr2x.runtime import model, tilegrid, tiling
from seedvr2x.runtime.tiling import Tile
from seedvr2x.vendor.common.config import create_object, load_config
from seedvr2x.vendor.core.infer import VideoDiffusionInfer
from seedvr2x.vendor.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

BF16 = torch.bfloat16
CPU = torch.device("cpu")

# The allowance of a scratch in system memory, whatever the shot.
MEMORY = 1 << 40


class Debug:
    """numz's Debug as its tiling uses it: a log, and the tiles it runs, which it records in
    output pixels when these lists are there (attn_video_vae.py:1382-1389, 1539-1551)."""

    def __init__(self) -> None:
        self.encode_tile_boundaries: list[dict[str, int]] = []
        self.decode_tile_boundaries: list[dict[str, int]] = []

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass


def total(terms: Iterator[Tensor]) -> Tensor:
    """The terms added one by one, in order. Elementwise, as everything the stand-in computes: a
    reduction's rounding could follow its input's memory layout, a crop's or a copy's."""
    return reduce(torch.add, terms)


class StandIn(VideoAutoencoderKLWrapper):
    """A VAE of arithmetic, in bfloat16, under numz's own code: slicing_encode, slicing_decode,
    tiled_encode, tiled_decode and their callers, encode and decode, are the vendored class's,
    inherited as they are. Only its constructor is passed by, and _encode and _decode, the
    encoder's and the decoder's pass over one slice, are the stand-in's. They change the shapes
    as the real ones do (8x8 pixels and 4 frames a latent, the first frame alone), depend on
    every value they are given and on its place inside the tile, and carry a state from slice to
    slice, which MemoryState.INITIALIZING resets and MemoryState.DISABLED neither reads nor
    keeps: it is kept where the real convolutions keep theirs (InflatedCausalConv3d.memory),
    which numz's slicing and model.py's clear. So a tile run on another tile's state, a slice out
    of order or a crop one pixel off gives other values."""

    def __init__(self, split_size: int = 4) -> None:
        torch.nn.Module.__init__(self)
        self.spatial_downsample_factor, self.temporal_downsample_factor = 8, 4
        self.use_slicing = False  # as the constructor passed by leaves it, for set_causal_slicing
        self.debug = Debug()
        generator = torch.Generator().manual_seed(0)

        def weights(*shape: int) -> torch.nn.Parameter:
            values = torch.rand(shape, generator=generator) - 0.5
            return torch.nn.Parameter(values.to(BF16), requires_grad=False)

        self.pool = weights(8, 8)  # the encoder's: each pixel of a latent's 8x8
        self.pack = weights(4)  # each frame of a latent's 4
        self.widen = weights(32, 3)  # its channels, from the frames'
        self.narrow = weights(3, 16)  # the decoder's: its channels, from the latents'
        self.unpack = weights(4)
        self.spread = weights(8, 8)
        self.carry = InflatedCausalConv3d(1, 1, 1, inflation_mode="tail")
        torch.nn.Module.to(self, BF16)
        self.set_causal_slicing(split_size=split_size, memory_device="same")
        # Every pass: its kind, its memory state and its input's shape.
        self.calls: list[tuple[str, MemoryState, tuple[int, ...]]] = []
        self.before: Callable[[], None] | None = None  # run before each pass

    def _pass(self, kind: str, state: MemoryState, given: Tensor) -> bool:
        """Record a pass; whether it starts afresh, without the state of the slice before."""
        if self.before is not None:
            self.before()
        self.calls.append((kind, state, tuple(given.shape)))
        if state != MemoryState.ACTIVE:
            self.carry.memory = None
        return state != MemoryState.ACTIVE

    def _finish(self, out: Tensor, state: MemoryState) -> Tensor:
        """The pass's output from `out`: each value with its place inside the tile, squared in
        part, and the state carried in, which the output's last frame then becomes."""
        rows = torch.arange(out.shape[-2]).view(-1, 1)
        columns = torch.arange(out.shape[-1]).view(1, -1)
        # 61, a prime: no tile's stride brings the same values back, as a period of 64 would.
        out = out + (((rows * 37 + columns * 11) % 61) / 64 - 0.5).to(out.dtype)
        out = out + 0.25 * out * out
        if self.carry.memory is not None:
            out = out + 0.5 * self.carry.memory
        if state != MemoryState.DISABLED:
            self.carry.memory = out[:, :, -1:].clone()
        return out

    def _encode(self, x: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        """x (1, 3, t, H, W) to (1, 32, t', H / 8, W / 8): t' is 1 + (t - 1) / 4 afresh, else
        t / 4."""
        fresh = self._pass("encode", memory_state, x)
        assert x.shape[1] == 3 and x.shape[3] % 8 == 0 and x.shape[4] % 8 == 0
        head, rest = x[:, :, : int(fresh)], x[:, :, int(fresh) :]
        assert rest.shape[2] % 4 == 0
        packed = total(rest[:, :, k::4] * self.pack[k] for k in range(4))
        frames = torch.cat((head, packed), dim=2)
        pooled = total(
            frames[..., i::8, j::8] * self.pool[i, j] for i in range(8) for j in range(8)
        )
        wide = total(pooled[:, c : c + 1] * self.widen[:, c].view(1, 32, 1, 1, 1) for c in range(3))
        return self._finish(wide, memory_state)

    def _decode(self, z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        """z (1, 16, t', h, w) to (1, 3, t, 8h, 8w): t is 1 + 4 (t' - 1) afresh, else 4 t'."""
        fresh = self._pass("decode", memory_state, z)
        assert z.shape[1] == 16
        narrow = total(z[:, c : c + 1] * self.narrow[:, c].view(1, 3, 1, 1, 1) for c in range(16))
        head, rest = narrow[:, :, : int(fresh)], narrow[:, :, int(fresh) :]
        phases = self.unpack.repeat(rest.shape[2]).view(1, 1, -1, 1, 1)
        frames = torch.cat((head, rest.repeat_interleave(4, dim=2) * phases), dim=2)
        large = frames.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)
        return self._finish(large * self.spread.repeat(z.shape[3], z.shape[4]), memory_state)


class Failing(RuntimeError):
    """What a stand-in raises when told to fail."""


class Refusing:
    """A store that refuses to be opened, for a pass that must not ask for one."""

    def store(
        self, frames: int, padded: tuple[int, int]
    ) -> contextlib.AbstractContextManager[Tensor]:
        raise Failing("asked for a store")

    def where(self, frames: int, padded: tuple[int, int]) -> str:
        return "nowhere"


def failing_at(vae: StandIn, call: int) -> None:
    """Make the stand-in's pass number `call`, from 0, fail, counted from now."""
    vae.calls.clear()

    def before() -> None:
        if len(vae.calls) == call:
            raise Failing(f"pass {call}")

    vae.before = before


def models_of(vae: Any) -> model.Models:
    """The models as the VAE's passes use them: numz's runner with its configuration, the
    pipeline's dtype, around vae, on the CPU."""
    config: Any = load_config(str(model.VENDOR / "configs_7b" / "main.yaml"))
    vae_yaml = model.VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    config.vae.model = OmegaConf.merge(config.vae.model, load_config(str(vae_yaml)))
    config.vae.dtype = "bfloat16"
    runner: Any = VideoDiffusionInfer(config, cast(Any, Debug()))
    runner.vae = vae
    return cast(model.Models, SimpleNamespace(runner=runner, device=CPU))


@pytest.fixture
def stand_in() -> model.Models:
    return models_of(StandIn())


def vae_of(models: model.Models) -> Any:
    return cast(Any, models).runner.vae


def offload(vae: Any, device: torch.device | None) -> None:
    """numz's tensor_offload_device: set, as its CLI sets it to the CPU by default
    (inference_cli.py:1417, model_configuration.py:1274), or left unset."""
    if device is not None:
        vae.tensor_offload_device = device
    elif "tensor_offload_device" in vars(vae):
        del vae.tensor_offload_device


def numz_encode(
    models: model.Models, video: Tensor, tiles: Tiles, device: torch.device | None = CPU
) -> Tensor:
    """numz's tiled encode of video (C, T, H, W), through its runner: (T', h, w, 16)."""
    runner = cast(Any, models).runner
    offload(runner.vae, device)
    runner.encode_tiled = True
    runner.encode_tile_size = (tiles.size, tiles.size)
    runner.encode_tile_overlap = (tiles.overlap, tiles.overlap)
    with torch.no_grad():
        return runner.vae_encode([video])[0]


def numz_decode(
    models: model.Models, latent: Tensor, tiles: Tiles, device: torch.device | None = CPU
) -> Tensor:
    """numz's tiled decode of latent (T', h, w, 16), through its runner: (C, T, H, W), a single
    frame given its T as the others."""
    runner = cast(Any, models).runner
    offload(runner.vae, device)
    runner.decode_tiled = True
    runner.decode_tile_size = (tiles.size, tiles.size)
    runner.decode_tile_overlap = (tiles.overlap, tiles.overlap)
    with torch.no_grad():
        frames = runner.vae_decode([latent])[0]
    return frames.unsqueeze(1) if frames.dim() == 3 else frames


def bits(tensor: Tensor) -> Tensor:
    """A bfloat16 tensor's 16-bit patterns: -0.0 isn't 0.0 there, and a NaN is itself."""
    assert tensor.dtype == BF16
    return tensor.contiguous().view(torch.int16)


def same(ours: Tensor, theirs: Tensor) -> bool:
    return ours.shape == theirs.shape and torch.equal(bits(ours), bits(theirs))


def video_of(frames: int, padded: tuple[int, int], seed: int = 0) -> Tensor:
    """A shot's transformed input, (3, frames, H, W) bfloat16 in [-1, 1]."""
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand((3, frames, *padded), generator=generator) * 2 - 1).to(BF16)


def latent_of(latents: int, size: tuple[int, int], seed: int = 0) -> Tensor:
    """A shot's latents, (T', h, w, 16) bfloat16."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((latents, *size, 16), generator=generator).to(BF16)


def slices_of(models: model.Models, video: Tensor) -> list[Tensor]:
    """video in the slices the encode takes it in."""
    return list(video.split(model.encode_slices(models, video.shape[1]), dim=1))


# The library's store.


def test_in_memory_store() -> None:
    # What a pass takes of its store (tiling.Store): (3, frames, H, W) bfloat16 on the CPU, zeros
    # each time it is opened, one open at a time, said to be in system memory whatever its size.
    stored = tiling.InMemory()
    assert stored.where(10_000, (2192, 3840)) == "system memory"
    with stored.store(5, (16, 24)) as store:
        assert (store.shape, store.dtype, store.device) == ((3, 5, 16, 24), BF16, CPU)
        assert not bits(store).any()
        store += 1
        with pytest.raises(RuntimeError, match="one store at a time"), stored.store(1, (8, 8)):
            pass
    with stored.store(5, (16, 24)) as again:  # free again, and zeros again
        assert not bits(again).any()
    assert tilegrid.scratch_bytes(5, (16, 24)) == again.numel() * again.element_size()


# The grid.


class Recorder(StandIn):
    """A stand-in whose passes give zeros, of one frame: numz's tiling runs its grid at once, and
    its Debug records the tiles."""

    def _encode(self, x: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        assert x.shape[2] == 1
        return x.new_zeros((1, 32, 1, x.shape[3] // 8, x.shape[4] // 8))

    def _decode(self, z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        assert z.shape[2] == 1
        return z.new_zeros((1, 3, 1, z.shape[3] * 8, z.shape[4] * 8))


def numz_grids(latent_size: tuple[int, int], tiles: Tiles) -> tuple[list[Tile], list[Tile]]:
    """The tiles numz's tiled_encode and tiled_decode run over a picture of latent_size latents,
    in their order, as its Debug records them; none when it runs the untiled pass instead."""
    vae = Recorder()
    rows, columns = latent_size
    size, overlap = (tiles.size, tiles.size), (tiles.overlap, tiles.overlap)
    vae.tiled_encode(torch.zeros((1, 3, 1, rows * 8, columns * 8), dtype=BF16), size, overlap)
    vae.tiled_decode(torch.zeros((1, 16, 1, rows, columns), dtype=BF16), size, overlap)

    def recorded(boundaries: list[dict[str, int]]) -> list[Tile]:
        assert [tile["id"] for tile in boundaries] == list(range(1, len(boundaries) + 1))
        assert all(tile[key] % 8 == 0 for tile in boundaries for key in "yxhw")
        return [
            Tile(t["y"] // 8, (t["y"] + t["h"]) // 8, t["x"] // 8, (t["x"] + t["w"]) // 8)
            for t in boundaries
        ]

    return recorded(vae.debug.encode_tile_boundaries), recorded(vae.debug.decode_tile_boundaries)


def check_grid(latent_size: tuple[int, int], tiles: Tiles) -> list[Tile]:
    """Our grid held to the tiles numz runs, the encode's and the decode's."""
    ours = tiling.grid(latent_size, tiles)
    encode, decode = numz_grids(latent_size, tiles)
    whole = Tile(0, latent_size[0], 0, latent_size[1])
    if ours == [whole]:
        assert encode == decode == []  # numz's untiled pass
    else:
        assert len(ours) > 1 and ours == encode == decode
    return ours


def test_grid_is_numz_over_random_sizes() -> None:
    rng = random.Random(0)
    several = 0
    for _ in range(300):
        side = rng.randint(1, 12)
        tiles = Tiles(side * 8, rng.randint(0, side - 1) * 8)
        ours = check_grid((rng.randint(1, 40), rng.randint(1, 40)), tiles)
        several += len(ours) > 1
    assert several > 200


@pytest.mark.parametrize("side", [2, 5, 9, 16])
def test_grid_is_numz_on_slivers(side: int) -> None:
    # A last row or column of j latents beyond the tiles before it: numz skips it up to the
    # overlap, where the tile before reaches the edge, and runs it from one latent more.
    for overlap in range(side):
        stride = side - overlap
        for full in (1, 2, 3):
            for j in range(-1, overlap + 3):
                length = full * stride + overlap + j
                if length < 1:
                    continue
                tiles = Tiles(side * 8, overlap * 8)
                rows = [(t.top, t.bottom) for t in check_grid((length, 3), tiles) if t.left == 0]
                columns = [(t.left, t.right) for t in check_grid((3, length), tiles) if t.top == 0]
                assert rows == columns and rows[-1][1] == length
                starts = [start for start, _ in rows]
                assert starts == [k * stride for k in range(len(rows))]
                if length > side:
                    # The last tile adds latents of its own, and the one before it stops short.
                    assert rows[-1][1] - rows[-1][0] > overlap and rows[-2][1] < length
                    # numz's own count of the tiles along a side (attn_video_vae.py:1343).
                    assert len(rows) == -(-(length - overlap) // stride)


def test_grid_of_real_frames() -> None:
    # 1080p, padded to 1104x1920, under 1024-px tiles sharing 64: the sliver row DESIGN.md names,
    # 144 px.
    full_hd = check_grid((138, 240), Tiles(1024))
    assert full_hd == [
        Tile(0, 128, 0, 128),
        Tile(0, 128, 120, 240),
        Tile(120, 138, 0, 128),
        Tile(120, 138, 120, 240),
    ]
    assert (full_hd[2].rows * 8, OVERLAP) == (144, 64)
    # A 1280-px tile holds its rows: two tiles, side by side.
    assert check_grid((138, 240), Tiles(1280)) == [Tile(0, 138, 0, 160), Tile(0, 138, 152, 240)]
    # 4K, padded to 2192x3840, under 2048-px tiles; and whole under tiles of its own width.
    assert len(check_grid((274, 480), Tiles(2048))) == 4
    assert check_grid((274, 480), Tiles(3840)) == [Tile(0, 274, 0, 480)]
    # Rows outer, columns inner.
    order = [(t.top, t.left) for t in check_grid((5, 7), Tiles(24, 8))]
    assert order == sorted(order) and len({top for top, _ in order}) == 2


def test_largest() -> None:
    assert tiling.largest((1104, 1920), Tiles(1280)) == 1104 * 1280 == 1_413_120
    assert tiling.largest((1104, 1920), Tiles(1024)) == 1024 * 1024
    assert tiling.largest((1104, 1920), Tiles(2048)) == 1104 * 1920
    assert tiling.largest((2192, 3840), Tiles(2048)) == 2048 * 2048
    assert tiling.largest((2192, 3840), Tiles(512)) == 512 * 512
    rng = random.Random(1)
    for _ in range(200):
        side = rng.randint(1, 12)
        tiles = Tiles(side * 8, rng.randint(0, side - 1) * 8)
        rows, columns = rng.randint(1, 40), rng.randint(1, 40)
        expected = min(side, rows) * min(side, columns) * 64  # the first tile, cut off
        assert tiling.largest((rows * 8, columns * 8), tiles) == expected


@pytest.mark.parametrize(
    "tiles",
    [Tiles(0, 0), Tiles(60, 8), Tiles(64, 60), Tiles(64, 64), Tiles(64, 72), Tiles(64, -8)],
)
def test_bad_tiles(tiles: Tiles) -> None:
    with pytest.raises(ValueError, match="multiples of 8"):
        tiling.grid((10, 10), tiles)
    with pytest.raises(ValueError, match="multiples of 8"):
        tiling.largest((80, 80), tiles)


def test_bad_sizes() -> None:
    with pytest.raises(ValueError):
        tiling.grid((0, 10), Tiles(64))
    for padded in ((100, 96), (96, 100), (0, 96)):
        with pytest.raises(ValueError, match="multiples of 8"):
            tiling.largest(padded, Tiles(64))


@pytest.mark.parametrize("split_size", [4, 8, 16])
def test_slices_follow_the_loops(split_size: int) -> None:
    # model.decode_slices against the loop it describes, and model.encode_slices against its
    # own; the passes' memory states as numz's slicing gives them.
    models = models_of(StandIn(split_size))
    vae = vae_of(models)
    for latents in range(1, 12):
        frames = 1 + 4 * (latents - 1)
        expected = model.decode_slices(models, latents)
        assert sum(expected) == frames
        latent = latent_of(latents, (2, 3))
        vae.calls.clear()
        streamed = list(model.decode_stream(models, latent))
        ours = list(vae.calls)
        assert [s.shape for s in streamed] == [(3, t, 16, 24) for t in expected]
        vae.calls.clear()
        with torch.no_grad():
            reference = cast(Any, models).runner.vae_decode([latent])[0]
        assert vae.calls == ours and len(ours) == len(expected)
        assert same(torch.cat(streamed, dim=1), reference.view(3, frames, 16, 24))
        if len(expected) > 1:
            assert [s for _, s, _ in ours] == [
                MemoryState.INITIALIZING,
                *[MemoryState.ACTIVE] * (len(ours) - 1),
            ]
            # split_size frames a slice, the first frame with the first, the last what is left.
            assert expected[0] == 1 + split_size and 0 < expected[-1] <= split_size
            assert all(t == split_size for t in expected[1:-1])
        else:
            assert ours == [("decode", MemoryState.DISABLED, (1, 16, latents, 2, 3))]

        sizes = model.encode_slices(models, frames)
        video = video_of(frames, (16, 24))
        vae.calls.clear()
        latent = model.encode_stream(models, video.split(sizes, dim=1), frames)
        assert [shape[2] for _, _, shape in vae.calls] == sizes
        assert latent.shape == (latents, 2, 3, 16)
    vae.use_slicing = False
    assert model.decode_slices(models, 9) == [33]
    assert [s.shape[1] for s in model.decode_stream(models, latent_of(9, (2, 3)))] == [33]


# The tiled decode and encode, against numz's.

# (latent rows, latent columns, latents, tile's side, overlap), the last two in output pixels:
# the overlap of 64 and others, none included; a sliver row, a column numz skips; one frame,
# one slice, two, and more.
CASES = [
    (24, 40, 3, 128, 64),
    (20, 30, 6, 96, 64),
    (13, 21, 1, 96, 64),
    (17, 9, 2, 72, 64),
    (10, 14, 3, 64, 0),
    (11, 17, 5, 48, 8),
    (9, 16, 4, 40, 32),
    (7, 19, 3, 64, 16),
    (6, 6, 2, 16, 8),
]


def random_cases(count: int, seed: int) -> list[tuple[int, int, int, int, int]]:
    rng = random.Random(seed)
    cases: list[tuple[int, int, int, int, int]] = []
    while len(cases) < count:
        side = rng.randint(2, 10)
        overlap = rng.randint(0, side - 1)
        rows, columns = rng.randint(1, 22), rng.randint(1, 26)
        if rows > side or columns > side:
            cases.append((rows, columns, rng.choice([1, 2, 3, 3, 4, 7]), side * 8, overlap * 8))
    return cases


@pytest.mark.parametrize("case", [*CASES, *random_cases(8, 2)])
@pytest.mark.parametrize("device", [CPU, None], ids=["offload-cpu", "offload-unset"])
def test_decode_is_numz(
    stand_in: model.Models,
    case: tuple[int, int, int, int, int],
    device: torch.device | None,
) -> None:
    rows, columns, latents, size, overlap = case
    tiles, vae = Tiles(size, overlap), vae_of(stand_in)
    grid = tiling.grid((rows, columns), tiles)
    assert len(grid) > 1
    latent = latent_of(latents, (rows, columns), seed=rows * columns + latents)
    given = latent.clone()
    reference = numz_decode(stand_in, latent, tiles, device)
    theirs = list(vae.calls)
    vae.calls.clear()
    assert vae.carry.memory is None

    scratch = tiling.InMemory()
    slices = list(tiling.decode_tiled(stand_in, latent, tiles, scratch))
    # The same passes, in the same order, on the same crops; one pass over the shot per tile.
    assert vae.calls == theirs and vae.carry.memory is None
    sizes = model.decode_slices(stand_in, latents)
    assert len(theirs) == len(grid) * len(sizes)
    # The slices a streamed decode gives, each its own memory.
    assert [s.shape for s in slices] == [(3, t, rows * 8, columns * 8) for t in sizes]
    assert [s.shape for s in slices] == [s.shape for s in model.decode_stream(stand_in, latent)]
    assert all(s.dtype == BF16 and s.is_contiguous() for s in slices)
    assert same(torch.cat(slices, dim=1), reference)
    assert same(latent, given)
    # Blended, not the untiled decode.
    untiled = torch.cat(list(model.decode_stream(stand_in, latent)), dim=1)
    assert not same(untiled, reference)


@pytest.mark.parametrize("case", [*CASES, *random_cases(8, 3)])
@pytest.mark.parametrize("device", [CPU, None], ids=["offload-cpu", "offload-unset"])
def test_encode_is_numz(
    stand_in: model.Models,
    case: tuple[int, int, int, int, int],
    device: torch.device | None,
) -> None:
    rows, columns, latents, size, overlap = case
    tiles, vae = Tiles(size, overlap), vae_of(stand_in)
    grid = tiling.grid((rows, columns), tiles)
    assert len(grid) > 1
    frames = 1 + 4 * (latents - 1)
    video = video_of(frames, (rows * 8, columns * 8), seed=rows * columns + latents)
    given = video.clone()
    reference = numz_encode(stand_in, video, tiles, device)
    theirs = list(vae.calls)
    vae.calls.clear()
    assert vae.carry.memory is None

    scratch = tiling.InMemory()
    latent = tiling.encode_tiled(stand_in, iter(slices_of(stand_in, video)), frames, tiles, scratch)
    assert vae.calls == theirs and vae.carry.memory is None
    assert len(theirs) == len(grid) * len(model.encode_slices(stand_in, frames))
    # The latent in the memory numz's has: the noise drawn from it depends on it.
    assert latent.shape == (latents, rows, columns, 16) and latent.stride() == reference.stride()
    assert same(latent, reference)
    assert same(video, given)
    untiled = model.encode_stream(stand_in, iter(slices_of(stand_in, video)), frames)
    assert untiled.stride() == latent.stride() and not same(untiled, reference)


@pytest.mark.parametrize("latents", [1, 2, 3, 6])
@pytest.mark.parametrize("size", [(11, 13), (2, 2), (1, 5), (5, 1), (1, 1)])
def test_encode_latent_is_the_whole_batch_encodes(
    stand_in: model.Models, latents: int, size: tuple[int, int]
) -> None:
    # The latent is made slice by slice, from a buffer in system memory, into storage laid out
    # by hand: it has the values and the memory of the latent numz's runner makes of the whole
    # batch at once, untiled and tiled, for one latent, for one slice and for more.
    rows, columns = size
    frames = 1 + 4 * (latents - 1)
    video = video_of(frames, (rows * 8, columns * 8), seed=latents)
    runner = cast(Any, stand_in).runner
    assert not runner.encode_tiled
    with torch.no_grad():
        whole = model.encode(stand_in, video)
    streamed = model.encode_stream(stand_in, iter(slices_of(stand_in, video)), frames)
    assert streamed.shape == whole.shape == (latents, rows, columns, 16)
    assert same(streamed, whole) and streamed.stride() == whole.stride()
    # Channel-major storage, (16, T', h, w), behind a permuted view, on the CPU.
    assert streamed.stride() == (rows * columns, columns, 1, latents * rows * columns)
    assert streamed.permute(3, 0, 1, 2).is_contiguous() and streamed.device == CPU
    assert streamed.untyped_storage().nbytes() == streamed.numel() * 2

    tiles = Tiles(64, 16) if size == (11, 13) else Tiles(16, 8)
    several = len(tiling.grid(size, tiles)) > 1
    assert several == (max(size) > 2)
    whole = numz_encode(stand_in, video, tiles)
    tiled = tiling.encode_tiled(
        stand_in, iter(slices_of(stand_in, video)), frames, tiles, tiling.InMemory()
    )
    assert tiled.stride() == streamed.stride() and tiled.shape == streamed.shape
    assert tiled.device == CPU and same(tiled, streamed) != several
    if several and (rows, latents) == (1, 1):
        # One frame of one row of latents, which no padded frame is: numz's runner gives its
        # tiled latent without the two, (w, 16), a squeeze of its own.
        assert whole.shape == (columns, 16) and same(tiled.view(columns, 16), whole)
    else:
        assert same(tiled, whole) and tiled.stride() == whole.stride()


def test_encode_refuses_moments_that_dont_fill_the_latent(stand_in: model.Models) -> None:
    # The buffer the slices' means go to is sized from the frames: moments of another length or
    # size would leave latents unwritten, or be broadcast into their place.
    vae = vae_of(stand_in)
    encode = vae._encode
    video = video_of(13, (88, 104))
    for cut, match in (
        (lambda m: m[:, :, :0], "2 latents encoded, not 4"),
        (lambda m: m[:, :, :, :1], r"latents 2 on encoded as \(1, 16, 1, 1, 13\) .*within 4"),
        (lambda m: m.float(), r"latents 2 on encoded as .*float32, not .*bfloat16"),
        (lambda m: m.repeat(1, 1, 3, 1, 1), r"latents 2 on encoded as \(1, 16, 3, 11, 13\)"),
    ):

        def other(
            x: Tensor, memory_state: MemoryState = MemoryState.DISABLED, cut: Any = cut
        ) -> Tensor:
            moments = encode(x, memory_state=memory_state)
            return moments if memory_state == MemoryState.INITIALIZING else cut(moments)

        vae._encode = other
        with pytest.raises(RuntimeError, match=match):
            model.encode_stream(stand_in, iter(slices_of(stand_in, video)), 13)
        assert vae.carry.memory is None
    vae._encode = lambda x, memory_state=MemoryState.DISABLED: encode(x, memory_state)[:, :, :0]
    with pytest.raises(RuntimeError, match="0 latents encoded, not 1"):
        model.encode_stream(stand_in, iter(slices_of(stand_in, video[:, :1])), 1)


@pytest.mark.parametrize("latents", [1, 2, 3, 5])
def test_one_tile_is_the_untiled_pass(stand_in: model.Models, latents: int) -> None:
    # The picture within one tile, in both directions or cut off in one: numz runs the untiled
    # pass, and so do we, the store untouched: this one could only refuse.
    untouched = Refusing()
    frames = 1 + 4 * (latents - 1)
    for rows, columns, size in ((6, 9, 72), (9, 6, 72), (9, 9, 72), (4, 5, 1024)):
        tiles = Tiles(size, 8)
        latent = latent_of(latents, (rows, columns))
        streamed = list(model.decode_stream(stand_in, latent))
        slices = list(tiling.decode_tiled(stand_in, latent, tiles, untouched))
        assert len(slices) == len(streamed)
        assert all(same(ours, theirs) for ours, theirs in zip(slices, streamed, strict=True))
        assert same(torch.cat(slices, dim=1), numz_decode(stand_in, latent, tiles))

        video = video_of(frames, (rows * 8, columns * 8))
        ours = tiling.encode_tiled(stand_in, slices_of(stand_in, video), frames, tiles, untouched)
        streamed_latent = model.encode_stream(stand_in, slices_of(stand_in, video), frames)
        assert same(ours, streamed_latent) and ours.stride() == streamed_latent.stride()
        assert same(ours, numz_encode(stand_in, video, tiles))
    # One latent more than the tile: tiled, and the store is asked for.
    with pytest.raises(Failing, match="asked for a store"):
        list(tiling.decode_tiled(stand_in, latent_of(latents, (10, 9)), Tiles(72, 8), untouched))
    with pytest.raises(Failing, match="asked for a store"):
        video = video_of(frames, (72, 80))
        tiling.encode_tiled(stand_in, slices_of(stand_in, video), frames, Tiles(72, 8), untouched)


def test_tiled_passes_are_said(stand_in: model.Models, caplog: pytest.LogCaptureFixture) -> None:
    # One debug line per pass the plan tiles, once its scratch's store is open: its grid, the
    # shot's frames and where they wait (the store's own word: tests/test_scratch.py has the
    # file's); or that the picture is within one tile. The GPU tests read them
    # (test_regression.py, test_resume.py).
    caplog.set_level(logging.DEBUG, logger="seedvr2x.runtime.tiling")
    tiles = Tiles(64, 16)
    latent, video = latent_of(4, (11, 13)), video_of(13, (88, 104))
    assert tilegrid.scratch_bytes(13, (88, 104)) == 713_856
    list(tiling.decode_tiled(stand_in, latent, tiles, tiling.InMemory()))
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, tiles, tiling.InMemory())
    list(tiling.decode_tiled(stand_in, latent, Tiles(1024), Refusing()))
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, Tiles(1024), Refusing())
    grid = "4 tiles of 64 px sharing 16, 2 across and 2 down, over 13 frames padded to 104x88"
    assert [record.getMessage() for record in caplog.records] == [
        f"tiled decode: {grid}; its scratch, 713856 bytes, in system memory",
        f"tiled encode: {grid}; its scratch, 713856 bytes, in system memory",
        "tiled decode: the frames, padded to 104x88, are within one tile of 1024 px: the untiled"
        " pass",
        "tiled encode: the frames, padded to 104x88, are within one tile of 1024 px: the untiled"
        " pass",
    ]
    assert all(record.levelno == logging.DEBUG for record in caplog.records)


@pytest.mark.parametrize("tiled", [True, False], ids=["tiled", "untiled"])
def test_encode_hands_its_moments_on_slice_by_slice(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch, tiled: bool
) -> None:
    # Each slice's moments leave for the sums as they are encoded, and for the untiled pass's
    # buffer. A tile's, kept whole, would stay on the device through the next tile's pass and
    # grow there with the shot, as the untiled pass's would until its end: no pass starts with
    # more than the slice before's still alive.
    vae = vae_of(stand_in)
    encode = vae._encode
    alive: list[weakref.ref[Tensor]] = []
    most: list[int] = []

    def counted(x: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        most.append(sum(ref() is not None for ref in alive))
        moments = encode(x, memory_state=memory_state)
        alive.append(weakref.ref(moments))
        return moments

    monkeypatch.setattr(vae, "_encode", counted)
    video = video_of(21, (88, 104))
    scratch = tiling.InMemory()
    if tiled:
        tiling.encode_tiled(stand_in, iter(slices_of(stand_in, video)), 21, Tiles(64, 16), scratch)
        assert len(most) == 5 * len(tiling.grid((11, 13), Tiles(64, 16)))
    else:
        model.encode_stream(stand_in, iter(slices_of(stand_in, video)), 21)
        assert len(most) == 5
    assert max(most) == 1 and not any(ref() is not None for ref in alive)


@pytest.mark.parametrize(
    ("tiles", "layout"),
    [(None, 1), (Tiles(128), 1), (Tiles(64, 16), 4)],
    ids=["untiled", "one-tile", "four-tiles"],
)
def test_encode_lets_its_first_slice_go(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch, tiles: Tiles | None, layout: int
) -> None:
    # A tiled encode reads the first slice ahead of the others, for the picture's size, and hands
    # it back with them: no reference of its own stays. Within one tile, the untiled pass, the
    # slice goes as the untiled pass lets it go, when the next comes, not at the pass's end; with
    # several, it is kept in the scratch, off the device, before any tile is encoded.
    vae = vae_of(stand_in)
    encode = vae._encode
    video = video_of(13, (88, 104))
    fed: list[weakref.ref[Tensor]] = []
    first: list[bool] = []

    def slices() -> Iterator[Tensor]:
        for piece in video.split(model.encode_slices(stand_in, 13), dim=1):
            made = [piece.clone()]
            fed.append(weakref.ref(made[0]))
            yield made.pop()  # the taker's alone

    def counted(x: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        first.append(fed[0]() is not None)
        return encode(x, memory_state=memory_state)

    monkeypatch.setattr(vae, "_encode", counted)
    if tiles is None:
        model.encode_stream(stand_in, slices(), 13)
    else:
        assert len(tiling.grid((11, 13), tiles)) == layout
        tiling.encode_tiled(stand_in, slices(), 13, tiles, tiling.InMemory())
    assert first == ([True, False, False] if layout == 1 else [False] * (3 * layout))
    assert not any(ref() is not None for ref in fed)


class Tainted(Tensor):
    """A latent, and whatever torch derives from it: a view, a crop, a copy, the decoder's
    input, each a Tainted too, as torch makes of a subclass's what it computes from one."""


def channels_last(z: Tensor) -> bool:
    """Whether torch reads z (1, 16, t, h, w) as channels-last memory, which the layers that take
    it follow (a convolution's algorithm, a padding's or a concatenation's output): a
    concatenation of z with itself is laid out so then, its channels a step of 1 apart."""
    assert z.dim() == 5 and z.shape[1] == 16
    return torch.cat((z, z), dim=2).stride()[1] == 1


def permuted(latent: Tensor) -> Tensor:
    """latent (T', h, w, 16) in channel-major memory, as an encode's."""
    return latent.permute(3, 0, 1, 2).contiguous().permute(1, 2, 3, 0)


@pytest.mark.parametrize("tiled", [True, False], ids=["tiled", "untiled"])
@pytest.mark.parametrize("layout", ["channels-last", "channel-major"])
def test_decode_takes_its_latents_slice_by_slice(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch, tiled: bool, layout: str
) -> None:
    # The latent stays where it is given: the decoder's input is made of one slice's latents at
    # a time, a tile's crop of them, and when a pass starts nothing else made from the latent
    # is alive, the slice before's input neither. Whole on the device, the latent and the
    # decoder's input would grow there with the shot.
    vae = vae_of(stand_in)
    decode = vae._decode
    tiles = Tiles(64, 16)
    plain = latent_of(6, (11, 13))
    plain = plain if layout == "channels-last" else permuted(plain)
    latent = plain.as_subclass(Tainted)
    handed: list[tuple[int, ...]] = []
    others: list[int] = []

    def counted(z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        assert isinstance(z, Tainted)
        handed.append(tuple(z.shape))
        others.append(
            sum(
                type(found) is Tainted and found is not latent and found is not z
                for found in gc.get_objects()
            )
        )
        # The frames aren't the latent's: only what leads to the decoder's input is counted.
        return decode(z.as_subclass(Tensor), memory_state=memory_state)

    monkeypatch.setattr(vae, "_decode", counted)
    if tiled:
        slices = tiling.decode_tiled(stand_in, latent, tiles, tiling.InMemory())
        crops = [(tile.rows, tile.columns) for tile in tiling.grid((11, 13), tiles)]
    else:
        slices = model.decode_stream(stand_in, latent)
        crops = [(11, 13)]
    frames = sum(decoded.shape[1] for decoded in slices)
    assert frames == 21 and model.latent_slices(stand_in, 6) == [2, 1, 1, 1, 1]
    assert handed == [(1, 16, t, *crop) for crop in crops for t in [2, 1, 1, 1, 1]]
    assert others == [0] * len(handed)
    monkeypatch.undo()
    # The same frames as from the latent itself, which the subclass changes nothing to.
    if tiled:
        ours = list(tiling.decode_tiled(stand_in, latent, tiles, tiling.InMemory()))
        theirs = numz_decode(stand_in, plain, tiles)
    else:
        ours = list(model.decode_stream(stand_in, latent))
        with torch.no_grad():
            theirs = cast(Any, stand_in).runner.vae_decode([plain])[0]
    assert same(torch.cat(ours, dim=1).as_subclass(Tensor), theirs)


@pytest.mark.parametrize("tiled", [True, False], ids=["tiled", "untiled"])
@pytest.mark.parametrize("layout", ["channels-last", "channel-major"])
@pytest.mark.parametrize("latents", [1, 2, 3, 6])
def test_decoder_is_handed_what_numz_hands_it(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch, tiled: bool, layout: str, latents: int
) -> None:
    # numz scales the whole batch and hands the decoder slices of it, views but the first; we
    # scale each slice's latents alone. The decoder is handed the same values in the same shape,
    # read by torch in the same memory format, for a latent as the DiT gives it (its channels
    # last) and for one in channel-major memory.
    vae = vae_of(stand_in)
    decode = vae._decode
    tiles = Tiles(64, 16)
    latent = latent_of(latents, (11, 13), seed=latents)
    latent = latent if layout == "channels-last" else permuted(latent)
    handed: list[tuple[Tensor, bool, tuple[int, ...] | None]] = []

    def recorded(z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        # Its strides too where numz hands the decoder a tensor of its own, a pass's first
        # slice, a concatenation, or the untiled latents in one piece; the others are views of
        # the whole batch's, in strides only a whole batch on the device has.
        own = memory_state == MemoryState.INITIALIZING or (
            memory_state == MemoryState.DISABLED and not tiled
        )
        handed.append((z.clone(), channels_last(z), z.stride() if own else None))
        return decode(z, memory_state=memory_state)

    monkeypatch.setattr(vae, "_decode", recorded)
    if tiled:
        numz_decode(stand_in, latent, tiles)
    else:
        with torch.no_grad():
            cast(Any, stand_in).runner.vae_decode([latent])
    theirs = list(handed)
    handed.clear()
    if tiled:
        list(tiling.decode_tiled(stand_in, latent, tiles, tiling.InMemory()))
    else:
        list(model.decode_stream(stand_in, latent))
    assert (
        len(handed)
        == len(theirs)
        == (4 if tiled else 1) * len(model.latent_slices(stand_in, latents))
    )
    for (ours, our_format, our_strides), (their, their_format, their_strides) in zip(
        handed, theirs, strict=True
    ):
        assert same(ours, their) and our_format == their_format and our_strides == their_strides
    assert {read for _, read, _ in handed} == {layout == "channels-last"}
    assert any(strides is not None for _, _, strides in handed) == (not tiled or latents > 2)


def test_encode_refuses_slices_that_dont_follow(stand_in: model.Models) -> None:
    vae = vae_of(stand_in)
    tiles = Tiles(64, 16)
    video = video_of(13, (88, 104))
    good = slices_of(stand_in, video)
    assert [s.shape[1] for s in good] == [5, 4, 4]

    def refused(slices: list[Tensor], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            tiling.encode_tiled(stand_in, iter(slices), 13, tiles, tiling.InMemory())
        # Refused before any pass.
        assert not vae.calls

    # The counts, as model.encode_stream refuses them, in its words.
    refused(list(video.split([4, 5, 4], dim=1)), r"slice 0 of 4 frames, the VAE takes \[5, 4, 4\]")
    refused(list(video.split([5, 8], dim=1)), "slice 1 of 8 frames")
    refused(good[:2], "2 slices, the VAE takes")
    refused([*good, good[-1]], "slice 3 of 4 frames")
    refused([], "0 slices, the VAE takes")
    for slices, match in (
        (list(video.split([4, 5, 4], dim=1)), "slice 0 of 4 frames"),
        (good[:2], "2 slices, the VAE takes"),
    ):
        with pytest.raises(ValueError, match=match):
            model.encode_stream(stand_in, iter(slices), 13)
    vae.calls.clear()
    # A slice of another size, or in another dtype, which the scratch would round.
    refused([good[0], good[1][:, :, :80], good[2]], "slice 1 is")
    refused([good[0], good[1], good[2][:2]], "slice 2 is")
    refused([good[0], good[1].float(), good[2]], "slice 1 is .*float32, not .*bfloat16")
    refused([s.float() for s in good], "slice 0 is .*float32")
    # Frames that aren't multiples of 8, as no padded frame is.
    with pytest.raises(ValueError, match="multiples of 8"):
        odd = video_of(5, (90, 104))
        tiling.encode_tiled(stand_in, [odd], 5, tiles, tiling.InMemory())
    # A latent in another dtype than the VAE's.
    with pytest.raises(ValueError, match=r"latent torch\.float32"):
        list(
            tiling.decode_tiled(stand_in, latent_of(3, (11, 13)).float(), tiles, tiling.InMemory())
        )
    assert not vae.calls


def test_tiled_passes_are_bfloat16() -> None:
    # A VAE in float32, as tests/test_shot.py's: its streamed passes run, its tiled ones are
    # refused, since the store would round what it is given.
    vae = StandIn()
    torch.nn.Module.to(vae, torch.float32)
    models, tiles = models_of(vae), Tiles(64, 16)
    latent = latent_of(3, (11, 13)).float()
    assert sum(s.shape[1] for s in model.decode_stream(models, latent)) == 9
    vae.calls.clear()
    with pytest.raises(ValueError, match=r"a tiled decode takes torch\.bfloat16"):
        list(tiling.decode_tiled(models, latent, tiles, tiling.InMemory()))
    video = video_of(9, (88, 104)).float()
    with pytest.raises(ValueError, match=r"slice 0 is .*float32, not .*bfloat16"):
        tiling.encode_tiled(models, slices_of(models, video), 9, tiles, tiling.InMemory())
    assert not vae.calls


class Short(StandIn):
    """A stand-in whose decoder gives a frame per latent."""

    def _decode(self, z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        return super()._decode(z, memory_state)[:, :, : z.shape[2]]


def test_decode_refuses_slices_of_another_shape() -> None:
    # 2 frames where 5 are due would be added to the first 2 of the scratch's 5, without a word.
    models = models_of(Short())
    with pytest.raises(RuntimeError, match=r"slice 0 decoded as \(3, 2, 64, 64\) .*not \(3, 5, 64"):
        list(tiling.decode_tiled(models, latent_of(3, (11, 13)), Tiles(64, 16), tiling.InMemory()))
    assert vae_of(models).carry.memory is None


# Once through the real convolutions.


@pytest.fixture(scope="module")
def real() -> model.Models:
    """The real VAE, built from the configs with random weights, in bfloat16 on the CPU, in
    numz's runner, with the Debug numz's tiling asks its VAE for."""
    models = models_of(None)
    runner = cast(Any, models).runner
    torch.manual_seed(0)
    vae = create_object(runner.config.vae.model).eval().requires_grad_(False)
    torch.nn.Module.to(vae, BF16)
    vae.set_causal_slicing(**runner.config.vae.slicing)
    vae.set_memory_limit(**runner.config.vae.memory_limit)
    vae.debug = Debug()
    runner.vae = vae
    return models


def test_real_vae_tiled_is_numz(real: model.Models) -> None:
    # 9 frames of 48x64 under 32-px tiles sharing 16: 6 tiles, each in two slices.
    tiles = Tiles(32, 16)
    assert len(tiling.grid((6, 8), tiles)) == 6
    video = video_of(9, (48, 64))
    given = video.clone()
    reference = numz_encode(real, video, tiles)
    # numz crops its tiles from the input itself, and we from its copy in the scratch: the same
    # only if the encoder leaves its input as it was.
    assert same(video, given)
    latent = tiling.encode_tiled(real, slices_of(real, video), 9, tiles, tiling.InMemory())
    assert latent.shape == (3, 6, 8, 16) and latent.stride() == reference.stride()
    assert same(latent, reference)
    assert not same(model.encode_stream(real, slices_of(real, video), 9), reference)

    decoded = numz_decode(real, latent, tiles)
    slices = list(tiling.decode_tiled(real, latent, tiles, tiling.InMemory()))
    assert [s.shape[1] for s in slices] == model.decode_slices(real, 3) == [5, 4]
    assert [s.shape[1] for s in model.decode_stream(real, latent)] == [5, 4]
    assert decoded.shape == (3, 9, 48, 64) and same(torch.cat(slices, dim=1), decoded)
    assert bool(torch.isfinite(decoded.float()).all())


@pytest.mark.parametrize("latents", [1, 2, 3, 4])
def test_real_vae_decode_slices(real: model.Models, latents: int) -> None:
    slices = list(model.decode_stream(real, latent_of(latents, (2, 2))))
    assert [s.shape[1] for s in slices] == model.decode_slices(real, latents)
    assert sum(s.shape[1] for s in slices) == 1 + 4 * (latents - 1)


# On a real card (pytest -m gpu): where a pass keeps what it holds. No test above can tell: their
# device is the CPU, where "on the device" and "in system memory" are one place, and a tensor
# moved to either is itself.

MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
# The files tests/test_placement.py loads: model.load_models takes a DiT with the VAE.
DITS = (
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_fp16.safetensors",
    "seedvr2x_ema_7b_sharp_fp16.safetensors",
)
VAES = ("ema_vae_fp16.safetensors", "seedvr2x_ema_vae_fp16.safetensors")

# Frames of 256x320, 32x40 latents: 40,960 bytes a latent in bfloat16 (16 channels), 81,920 for
# its moments (32). A shot of 13 frames, 4 latents in 3 slices, and one of 141, 36 latents in 35:
# 32 latents more. Tiles of 192 px sharing 64 lay 4 over the picture, 2 rows of 2.
PICTURE = (256, 320)
SHORT, LONG = 13, 141
FOUR = Tiles(192, 64)
# What two measures of one pass may differ by, and still be the same: 256 KiB. The smallest thing
# of a shot's that could be on the device is one tensor of its latent's size, the latent itself,
# the untiled encode's buffer, the tiled encode's sums or the decoder's input whole: 32 latents
# more of it are 32 x 40,960 = 1,310,720 bytes, 1.25 MiB, 5 times this; the moments kept as numz
# keeps them, 2.5 MiB; the latent of the long shot left there after its pass, 36 x 40,960 =
# 1,474,560 bytes, 1.41 MiB.
TOLERANCE = 256 << 10
# The call of a pass, of each tile's in a tiled one, that its later calls are measured against.
# The VAE's causal convolutions (InflatedCausalConv3d, causal_inflation_lib.py) are settled long
# before it: each takes its memory in the pass's first call (INITIALIZING), replaces it in every
# next one (ACTIVE) and loses it when the pass ends; under the memory limit the VAE is loaded
# with (0.5 GiB: slicing_forward) it is a copy of its input's last frames, 2, or 1 where the
# convolution halves time, of one size after every call. So a call starts with no memory at 0,
# and with all of it from 1 on; the pass's own last slice is still held then, the first's 2
# latents at 1 and 1 latent from 2 on. The card showed just that, in the bytes asked of torch's
# allocator: an untiled encode holds 2,457,600 as call 0 starts, its first slice, 320,603,648 at
# 1 and 320,521,728 at each of the 33 after, one latent's moments less (81,920), the same to
# the byte in a 3-call shot. The 6 calls more are a margin nothing asks for.
#
# The figures are taken on those bytes (memory_stats' requested_bytes: "memory requested by
# client code", torch/cuda/memory.py; a running count with its peak: Stat, torch/include/c10/
# core/Allocator.h), not on memory_allocated, which counts the blocks the allocator handed out:
# a cached block goes out whole for less than its size, and the same tensors count for more or
# less by the blocks they land in. Measured on the card (RTX PRO 6000, torch 2.14.1), where the
# requested bytes were the same at every settled call: the allocated ones went from 321,308,160
# to 322,636,288 from call to call of the untiled encode, 786,432 to 2,114,560 over the
# requested, and up to 9,765,504 over in the tiled passes (8,913,536 at the most in two other
# runs of this test, a tiled pass's most 5,898,880 to 9,765,504); the first run's 559,104 bytes
# between the largest allocated figures of a 3-call and a 35-call encode were that rounding, and
# so were the 1,671,168 between the two shots' allocated peaks of the tiled decode (995,573,760
# and 997,244,928). The allocated figures are printed beside the requested ones, which shows
# it. They are compared instead only where the requested ones can't be taken (Use.doubt).
SETTLED = 8


def found(names: tuple[str, ...]) -> Path:
    """The first of names there is in SEEDVR2X_MODEL_DIR; the test is skipped, naming them, when
    there is none."""
    if not MODELS:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    for name in names:
        if (Path(MODELS) / name).is_file():
            return Path(MODELS) / name
    pytest.skip(f"none of {', '.join(names)} in SEEDVR2X_MODEL_DIR")


@dataclass(frozen=True)
class Use:
    """What a pass takes on the device beyond what was there before it, in bytes. By torch's
    count of the blocks its allocator handed out (memory_allocated): at its peak; once it has
    ended, its result still held (left); and as each of the VAE's calls starts, a slice's encode
    or decode, in order, tile after tile in a tiled pass, which is what the pass holds from slice
    to slice (allocated). By its count of the bytes asked of the allocator (memory_stats'
    requested_bytes): as each call starts too (requested) and at its peak (requested_peak).

    doubt is empty when the requested bytes can be taken, or says why they can't: torch doesn't
    give the count (requested is then empty); it was once over the allocated bytes, before the
    pass, at a call or after it, which what is asked for the live tensors can't be, a block being
    no smaller than what it went out for; or reset_peak_memory_stats didn't bring its peak back
    to what was requested then, so that the peak may be an earlier pass's."""

    peak: int
    left: int
    allocated: tuple[int, ...]
    requested: tuple[int, ...]
    requested_peak: int
    doubt: str


def device_use[T](
    models: model.Models, monkeypatch: pytest.MonkeyPatch, run: Callable[[], T]
) -> tuple[T, Use]:
    """What run, one of the VAE's passes, gives, and what it takes on the device (Use)."""
    vae, device = models.runner.vae, models.device
    allocated: list[int] = []
    requested: list[int | None] = []

    def asked(what: str = "current") -> int | None:
        return torch.cuda.memory_stats(device).get(f"requested_bytes.all.{what}")

    def sampled(call: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def wrapped(*args: Any, **kwargs: Any) -> Tensor:
            allocated.append(torch.cuda.memory_allocated(device))
            requested.append(asked())
            return call(*args, **kwargs)

        return wrapped

    with monkeypatch.context() as patch:
        patch.setattr(vae, "_encode", sampled(vae._encode))
        patch.setattr(vae, "_decode", sampled(vae._decode))
        torch.cuda.synchronize(device)
        before, asked_before = torch.cuda.memory_allocated(device), asked()
        torch.cuda.reset_peak_memory_stats(device)
        asked_reset = asked("peak")
        result = run()
        torch.cuda.synchronize(device)
        peak, asked_peak = torch.cuda.max_memory_allocated(device), asked("peak")
        left, asked_left = torch.cuda.memory_allocated(device), asked()
    held = [before, *allocated, left]
    counts = [count for count in (asked_before, *requested, asked_left) if count is not None]
    of_calls: tuple[int, ...] = ()
    at_peak = 0
    if len(counts) != len(held) or asked_reset is None or asked_peak is None:
        doubt = "torch gives no requested_bytes"
    else:
        of_calls = tuple(count - counts[0] for count in counts[1:-1])
        at_peak = asked_peak - counts[0]
        if any(count > there for count, there in zip(counts, held, strict=True)):
            doubt = "the requested ones were over them: no count of what is live"
        elif asked_reset != counts[0]:
            doubt = (
                f"resetting the peaks left the requested one at {asked_reset}, {counts[0]}"
                " being requested then"
            )
        else:
            doubt = ""
    start = tuple(there - before for there in allocated)
    return result, Use(peak - before, left - before, start, of_calls, at_peak, doubt)


def runs(values: tuple[int, ...]) -> str:
    """values in order, one that comes n times in a row said once, as value*n."""
    said: list[list[int]] = []
    for value in values:
        if said and said[-1][0] == value:
            said[-1][1] += 1
        else:
            said.append([value, 1])
    return " ".join(f"{value}" if count == 1 else f"{value}*{count}" for value, count in said)


def by_tile(values: tuple[int, ...], tiles: int) -> list[tuple[int, ...]]:
    """A pass's values, one per call of the VAE, tile by tile: each of `tiles` tiles is as many
    calls, one after the other."""
    calls, rest = divmod(len(values), tiles)
    assert rest == 0, (len(values), tiles)
    return [values[tile * calls : (tile + 1) * calls] for tile in range(tiles)]


def series(values: tuple[int, ...], shapes: list[tuple[int, int]]) -> str:
    """A pass's values (by_tile) in a line (runs), each tile's named with its latents' rows and
    columns when there are several."""
    parts = [runs(part) for part in by_tile(values, len(shapes))]
    if len(shapes) == 1:
        return parts[0]
    return " | ".join(
        f"tile {index}, {rows}x{columns}: {part}"
        for index, ((rows, columns), part) in enumerate(zip(shapes, parts, strict=True))
    )
