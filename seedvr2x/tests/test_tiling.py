"""The VAE tiling on the CPU (runtime/tiling.py): the grid, the scratch, and the tiled encode and
decode held bit for bit to numz's tiled_encode and tiled_decode, the vendored ones, over a stand-in
VAE and, once, over the real one with random weights."""

import errno
import gc
import logging
import mmap
import os
import random
import resource
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from seedvr2x.runtime import model, tilegrid, tiling
from seedvr2x.runtime.plan import OVERLAP, Tiles
from seedvr2x.runtime.tiling import Scratch, ScratchError, Tile
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


def mapped(path: Path) -> bool:
    """Whether this process maps the file at path, removed or not."""
    return any(str(path) in line for line in Path("/proc/self/maps").read_text().splitlines())


def open_on(path: Path) -> list[str]:
    """What this process's descriptors on the file at path, removed or not, are open on
    (/proc/self/fd): nothing once a scratch there and every tensor on it are gone."""
    found: list[str] = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:  # the listing's own, closed since
            continue
        if target.startswith(str(path)):
            found.append(target)
    return found


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


# The scratch.


def test_scratch_needed() -> None:
    # DESIGN.md, Pipeline step 3: 12.7 MB per 1080p frame, 50.5 MB per 4K frame, padded.
    assert Scratch.needed(1, (1104, 1920)) == 3 * 1104 * 1920 * 2 == 12_718_080
    assert Scratch.needed(1, (2192, 3840)) == 50_503_680
    assert Scratch.needed(17, (64, 96)) == 17 * 3 * 64 * 96 * 2
    # The count kept without torch, for the planner, is of the dtype the scratch is made in.
    assert tilegrid.VALUE_BYTES == model.COMPUTE_DTYPE.itemsize == BF16.itemsize
    assert tilegrid.scratch_bytes(17, (64, 96)) == Scratch.needed(17, (64, 96))


def test_scratch_in_memory(tmp_path: Path) -> None:
    needed = Scratch.needed(5, (16, 24))
    scratch = Scratch(tmp_path, needed)
    with scratch.store(5, (16, 24)) as store:
        assert (store.shape, store.dtype, store.device) == ((3, 5, 16, 24), BF16, CPU)
        assert not bits(store).any()
        assert not list(tmp_path.iterdir())
        with pytest.raises(RuntimeError, match="one store at a time"), scratch.store(1, (8, 8)):
            pass
    with scratch.store(1, (8, 8)):  # free again
        pass


def test_scratch_on_disk(tmp_path: Path) -> None:
    directory = tmp_path / "resume" / "shot_000012"  # not there yet
    needed = Scratch.needed(5, (16, 24))
    scratch = Scratch(directory, needed - 1)
    path = directory / tiling.SCRATCH
    with scratch.store(5, (16, 24)) as store:
        assert (store.shape, store.dtype, store.device) == ((3, 5, 16, 24), BF16, CPU)
        assert not bits(store).any()
        # Whole on disk from the start, its blocks allocated.
        assert path.stat().st_size == needed <= path.stat().st_blocks * 512
        assert mapped(path)
        # The descriptor it was allocated through is closed: the mapping's own is the only one,
        # and the mapping is the tensor's to keep, nothing else holding it.
        assert len(open_on(path)) == 1
        gc.collect()
        # In-place adds through the tensor reach the file.
        values = video_of(2, (7, 8))
        store[:, 1:3, 2:9, 3:11] += values
        store[:, 2:4, 4:11, 6:14] += values
        expected = torch.zeros((3, 5, 16, 24), dtype=BF16)
        expected[:, 1:3, 2:9, 3:11] += values
        expected[:, 2:4, 4:11, 6:14] += values
        on_disk = torch.from_numpy(np.fromfile(path, dtype=np.int16)).view(3, 5, 16, 24)
        assert torch.equal(on_disk, bits(expected)) and same(store, expected)
        kept = store[:, 0]
        del store
    # Removed with the store, and unmapped with the last tensor on it, which still reads and
    # writes it until then; the file's blocks go with the mapping and its descriptor.
    assert not path.exists() and mapped(path)
    kept += 1
    assert same(kept, expected[:, 0] + 1)
    del kept
    assert not mapped(path) and not open_on(path)


def test_scratch_goes_on_failure(tmp_path: Path) -> None:
    scratch = Scratch(tmp_path, 0)
    path = tmp_path / tiling.SCRATCH
    with pytest.raises(Failing), scratch.store(2, (8, 8)):
        assert path.exists()
        raise Failing
    assert not path.exists() and not mapped(path) and not open_on(path)
    with scratch.store(2, (8, 8)):  # free again
        assert path.exists()


def test_scratch_allowance_is_its_passs() -> None:
    # What a scratch may take of system memory is given with it, a run's from its plan
    # (planner.plan); one given none, by a test or a script, takes a default of 8 GiB.
    assert Scratch(None, 0).allowance == 0 and Scratch(None, 12_345).allowance == 12_345
    assert Scratch(None).allowance == Scratch(None, None).allowance == tiling.DEFAULT_ALLOWANCE
    assert tiling.DEFAULT_ALLOWANCE == 8 * 2**30
    assert Scratch(None).where(675, (1104, 1920)) == "system memory"
    with (
        pytest.raises(ScratchError, match=f"over the {8 * 2**30} allowed in system memory"),
        Scratch(None).store(676, (1104, 1920)),
    ):
        pass


def test_scratch_replaces_a_killed_runs(tmp_path: Path) -> None:
    path = tmp_path / tiling.SCRATCH
    path.write_bytes(b"\xff" * 4096)
    with Scratch(tmp_path, 0).store(2, (8, 8)) as store:
        assert path.stat().st_size == Scratch.needed(2, (8, 8)) and not bits(store).any()
    assert not path.exists()


@pytest.mark.parametrize("planted", ["link", "file"])
def test_scratch_maps_the_file_it_allocated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: str
) -> None:
    # The scratch's name given to another file between the allocation and the mapping, a link
    # to it or the file itself: the file allocated is mapped and written, by its descriptor,
    # never the other.
    directory = tmp_path / "shot"
    path, other = directory / tiling.SCRATCH, tmp_path / "other"
    needed = Scratch.needed(5, (16, 24))
    theirs = b"\x55" * needed
    other.write_bytes(theirs)
    allocate = os.posix_fallocate
    allocated: list[int] = []

    def swapped(descriptor: int, offset: int, length: int) -> None:
        allocate(descriptor, offset, length)
        allocated.append(os.open(path, os.O_RDONLY))  # the file allocated, still under its name
        path.unlink()
        if planted == "link":
            path.symlink_to(other)
        else:
            other.rename(path)

    monkeypatch.setattr(os, "posix_fallocate", swapped)
    values = video_of(5, (16, 24))
    with Scratch(directory, 0).store(5, (16, 24)) as store:
        monkeypatch.undo()
        there = other if planted == "link" else path
        assert path.is_symlink() == (planted == "link") and there.read_bytes() == theirs
        assert not bits(store).any()
        store += values
        assert there.read_bytes() == theirs
        # The file allocated, read by a descriptor of its own: no name is left to it.
        (descriptor,) = allocated
        status = os.fstat(descriptor)
        assert (status.st_size, status.st_nlink) == (needed, 0)
        written = torch.frombuffer(bytearray(os.pread(descriptor, needed, 0)), dtype=torch.int16)
        assert torch.equal(written, bits(values).flatten()) and same(store, values)
        os.close(descriptor)
        del store
    # The name goes with the store, as the scratch's own would: a link, not what it names.
    assert not path.exists() and not path.is_symlink()
    assert not mapped(path) and not open_on(path)
    assert planted == "file" or other.read_bytes() == theirs


def test_scratch_without_room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    needed = Scratch.needed(2, (8, 8))
    # No directory, and more than the allowance: the bytes and the allowance are named.
    with (
        pytest.raises(ScratchError, match=f"{needed} bytes.*the {needed - 1} allowed"),
        Scratch(None, needed - 1).store(2, (8, 8)),
    ):
        pass

    # A full disk fails at the allocation, naming the file, and leaves none.
    def full(descriptor: int, offset: int, length: int) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(os, "posix_fallocate", full)
    path = tmp_path / tiling.SCRATCH
    scratch = Scratch(tmp_path, 0)
    with pytest.raises(ScratchError, match=f"{path}.*{needed} bytes.*No space left"):
        with scratch.store(2, (8, 8)):
            pass
    assert not path.exists()

    # An allocation interrupted leaves none either.
    def interrupted(descriptor: int, offset: int, length: int) -> None:
        assert path.exists()
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "posix_fallocate", interrupted)
    with pytest.raises(KeyboardInterrupt), scratch.store(2, (8, 8)):
        pass
    assert not path.exists()
    monkeypatch.undo()

    # A file that can't be mapped: refused, and removed, nothing left open on it.
    def unmappable(*args: Any, **kwargs: Any) -> mmap.mmap:
        assert path.stat().st_size == needed
        raise OSError(errno.ENOMEM, os.strerror(errno.ENOMEM))

    monkeypatch.setattr(mmap, "mmap", unmappable)
    with pytest.raises(ScratchError, match=f"{path}.*{needed} bytes, not mapped: .*Cannot alloc"):
        with scratch.store(2, (8, 8)):
            pass
    assert not path.exists() and not open_on(path)
    monkeypatch.undo()
    with scratch.store(2, (8, 8)):  # free again
        pass


def test_scratch_the_system_wont_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A mapping the system itself refuses: 25 GB, with the process's address space held to a
    # gigabyte over what it has (the file left sparse, for the test's sake). A ScratchError, the
    # file removed, and no descriptor left on it: torch.from_file, which mapped it before, kept
    # its own open on the file, removed and allocated whole.
    path = tmp_path / tiling.SCRATCH
    needed = Scratch.needed(4000, (1024, 1024))
    monkeypatch.setattr(
        os, "posix_fallocate", lambda descriptor, offset, length: os.ftruncate(descriptor, length)
    )
    has = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = has + (1 << 30)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    try:
        with pytest.raises(ScratchError, match=f"{needed} bytes, not mapped: .*Cannot alloc"):
            with Scratch(tmp_path, 0).store(4000, (1024, 1024)):
                pass
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    assert not path.exists() and not mapped(path) and not open_on(path)


def test_scratch_that_stays_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A scratch that can't be removed changes no result: a warning, with the reason.
    path = tmp_path / tiling.SCRATCH
    with Scratch(tmp_path, 0).store(2, (8, 8)) as store:
        store += 1

        def denied(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM), str(self))

        monkeypatch.setattr(Path, "unlink", denied)
    monkeypatch.undo()
    assert path.exists()
    assert f"{path}: the shot's scratch, not removed" in caplog.text
    assert "Operation not permitted" in caplog.text


# The slices.


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
    tmp_path: Path,
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

    scratch = Scratch(tmp_path, 0) if device is None else Scratch(None, MEMORY)
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
    assert same(latent, given) and not (tmp_path / tiling.SCRATCH).exists()
    # Blended, not the untiled decode.
    untiled = torch.cat(list(model.decode_stream(stand_in, latent)), dim=1)
    assert not same(untiled, reference)


@pytest.mark.parametrize("case", [*CASES, *random_cases(8, 3)])
@pytest.mark.parametrize("device", [CPU, None], ids=["offload-cpu", "offload-unset"])
def test_encode_is_numz(
    stand_in: model.Models,
    tmp_path: Path,
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

    scratch = Scratch(tmp_path, 0) if device is None else Scratch(None, MEMORY)
    latent = tiling.encode_tiled(stand_in, iter(slices_of(stand_in, video)), frames, tiles, scratch)
    assert vae.calls == theirs and vae.carry.memory is None
    assert len(theirs) == len(grid) * len(model.encode_slices(stand_in, frames))
    # The latent in the memory numz's has: the noise drawn from it depends on it.
    assert latent.shape == (latents, rows, columns, 16) and latent.stride() == reference.stride()
    assert same(latent, reference)
    assert same(video, given) and not (tmp_path / tiling.SCRATCH).exists()
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
        stand_in, iter(slices_of(stand_in, video)), frames, tiles, Scratch(None, MEMORY)
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
    # pass, and so do we, the scratch untouched: this one could only refuse.
    untouched = Scratch(None, 0)
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
    # One latent more than the tile: tiled, and the scratch is asked for.
    with pytest.raises(ScratchError):
        list(tiling.decode_tiled(stand_in, latent_of(latents, (10, 9)), Tiles(72, 8), untouched))
    with pytest.raises(ScratchError):
        video = video_of(frames, (72, 80))
        tiling.encode_tiled(stand_in, slices_of(stand_in, video), frames, Tiles(72, 8), untouched)


def test_tiled_passes_are_said(
    stand_in: model.Models, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # One debug line per pass the plan tiles, once its scratch's store is open: its grid, the
    # shot's frames and where they wait; or that the picture is within one tile. The GPU tests
    # read them (test_regression.py, test_resume.py).
    caplog.set_level(logging.DEBUG, logger="seedvr2x.runtime.tiling")
    tiles = Tiles(64, 16)
    latent, video = latent_of(4, (11, 13)), video_of(13, (88, 104))
    assert Scratch.needed(13, (88, 104)) == 713_856
    list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, MEMORY)))
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, tiles, Scratch(tmp_path, 0))
    list(tiling.decode_tiled(stand_in, latent, Tiles(1024), Scratch(None, 0)))
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, Tiles(1024), Scratch(None, 0))
    grid = "4 tiles of 64 px sharing 16, 2 across and 2 down, over 13 frames padded to 104x88"
    assert [record.getMessage() for record in caplog.records] == [
        f"tiled decode: {grid}; its scratch, 713856 bytes, in system memory",
        f"tiled encode: {grid}; its scratch, 713856 bytes, in {tmp_path / tiling.SCRATCH}",
        "tiled decode: the frames, padded to 104x88, are within one tile of 1024 px: the untiled"
        " pass",
        "tiled encode: the frames, padded to 104x88, are within one tile of 1024 px: the untiled"
        " pass",
    ]
    assert all(record.levelno == logging.DEBUG for record in caplog.records)


def test_scratch_in_memory_is_the_scratch_on_disk(stand_in: model.Models, tmp_path: Path) -> None:
    tiles = Tiles(64, 16)
    latent = latent_of(4, (11, 13))
    path = tmp_path / tiling.SCRATCH
    in_memory = list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(None, MEMORY)))
    assert not list(tmp_path.iterdir())

    # On disk: the file is there from the first tile to the last slice handed on, then gone.
    vae = vae_of(stand_in)
    seen: list[bool] = []
    vae.before = lambda: seen.append(path.exists() and mapped(path))
    decoding = tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, 0))
    assert not path.exists()  # nothing before the first slice is asked for
    on_disk = [next(decoding)]
    assert seen and all(seen) and path.exists()
    assert path.stat().st_size == Scratch.needed(13, (88, 104))
    on_disk += list(decoding)
    assert not path.exists() and not mapped(path)
    assert len(on_disk) == len(in_memory) == 3
    assert all(same(disk, memory) for disk, memory in zip(on_disk, in_memory, strict=True))

    video = video_of(13, (88, 104))
    seen.clear()
    kept = tiling.encode_tiled(
        stand_in, slices_of(stand_in, video), 13, tiles, Scratch(tmp_path, 0)
    )
    assert seen and all(seen) and not path.exists() and not mapped(path)
    vae.before = None
    in_memory_latent = tiling.encode_tiled(
        stand_in, slices_of(stand_in, video), 13, tiles, Scratch(None, MEMORY)
    )
    assert same(kept, in_memory_latent)


def test_decode_closed_early(stand_in: model.Models, tmp_path: Path) -> None:
    path = tmp_path / tiling.SCRATCH
    scratch = Scratch(tmp_path, 0)
    decoding = tiling.decode_tiled(stand_in, latent_of(4, (11, 13)), Tiles(64, 16), scratch)
    first = next(decoding)
    assert path.exists() and mapped(path)
    cast(Any, decoding).close()
    assert not path.exists() and not mapped(path) and not open_on(path)
    # What was handed on is a copy, still whole.
    assert first.shape == (3, 5, 88, 104) and bool(torch.isfinite(first.float()).all())
    assert vae_of(stand_in).carry.memory is None
    with scratch.store(1, (8, 8)):  # free again
        pass


@pytest.mark.parametrize("call", [0, 1, 4, 7])
def test_decode_failing_in_a_tile(stand_in: model.Models, tmp_path: Path, call: int) -> None:
    # A pass that fails, the first of a tile or a later one, in the first tile or another: the
    # scratch goes and the causal memory is cleared.
    vae, path = vae_of(stand_in), tmp_path / tiling.SCRATCH
    scratch = Scratch(tmp_path, 0)
    failing_at(vae, call)
    with pytest.raises(Failing):
        list(tiling.decode_tiled(stand_in, latent_of(4, (11, 13)), Tiles(64, 16), scratch))
    assert len(vae.calls) == call and not path.exists()
    assert vae.carry.memory is None and not mapped(path) and not open_on(path)
    with scratch.store(1, (8, 8)):  # free again
        pass


def test_decode_failing_between_passes(
    stand_in: model.Models, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure of ours, between a tile's passes, as a Ctrl-C raised while the device is polled:
    # the tile's pass is closed, which clears its memory at once, and the scratch goes.
    vae, path = vae_of(stand_in), tmp_path / tiling.SCRATCH
    polls: list[bool] = []

    def synchronize(device: torch.device) -> None:
        polls.append(vae.carry.memory is not None)
        if len(polls) == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr(model, "synchronize", synchronize)
    with pytest.raises(KeyboardInterrupt) as raised:
        list(
            tiling.decode_tiled(
                stand_in, latent_of(4, (11, 13)), Tiles(64, 16), Scratch(tmp_path, 0)
            )
        )
    assert polls[-1]  # in the middle of a tile, its memory kept
    assert vae.carry.memory is None and not path.exists()
    # The mapping goes with the failure's traceback, which holds the frames that held the store.
    del raised
    assert not mapped(path) and not open_on(path)


@pytest.mark.parametrize("call", [0, 3, 5])
def test_encode_failing_in_a_tile(stand_in: model.Models, tmp_path: Path, call: int) -> None:
    vae, path = vae_of(stand_in), tmp_path / tiling.SCRATCH
    scratch = Scratch(tmp_path, 0)
    video = video_of(13, (88, 104))
    failing_at(vae, call)
    with pytest.raises(Failing):
        tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, Tiles(64, 16), scratch)
    assert len(vae.calls) == call and not path.exists()
    assert vae.carry.memory is None and not mapped(path) and not open_on(path)
    with scratch.store(1, (8, 8)):  # free again
        pass


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
    scratch = Scratch(None, MEMORY)
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
        tiling.encode_tiled(stand_in, slices(), 13, tiles, Scratch(None, MEMORY))
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
        slices = tiling.decode_tiled(stand_in, latent, tiles, Scratch(None, MEMORY))
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
        ours = list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(None, MEMORY)))
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
        list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(None, MEMORY)))
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


def test_encode_refuses_slices_that_dont_follow(stand_in: model.Models, tmp_path: Path) -> None:
    vae, path = vae_of(stand_in), tmp_path / tiling.SCRATCH
    tiles = Tiles(64, 16)
    video = video_of(13, (88, 104))
    good = slices_of(stand_in, video)
    assert [s.shape[1] for s in good] == [5, 4, 4]

    def refused(slices: list[Tensor], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            tiling.encode_tiled(stand_in, iter(slices), 13, tiles, Scratch(tmp_path, 0))
        # Refused before any pass, and the scratch gone.
        assert not vae.calls and not path.exists()

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
        tiling.encode_tiled(stand_in, [odd], 5, tiles, Scratch(tmp_path, 0))
    # A latent in another dtype than the VAE's.
    with pytest.raises(ValueError, match=r"latent torch\.float32"):
        list(
            tiling.decode_tiled(
                stand_in, latent_of(3, (11, 13)).float(), tiles, Scratch(tmp_path, 0)
            )
        )
    assert not vae.calls and not path.exists()


def test_tiled_passes_are_bfloat16(tmp_path: Path) -> None:
    # A VAE in float32, as tests/test_shot.py's: its streamed passes run, its tiled ones are
    # refused, since the scratch would round what it is given.
    vae = StandIn()
    torch.nn.Module.to(vae, torch.float32)
    models, tiles, path = models_of(vae), Tiles(64, 16), tmp_path / tiling.SCRATCH
    latent = latent_of(3, (11, 13)).float()
    assert sum(s.shape[1] for s in model.decode_stream(models, latent)) == 9
    vae.calls.clear()
    with pytest.raises(ValueError, match=r"a tiled decode takes torch\.bfloat16"):
        list(tiling.decode_tiled(models, latent, tiles, Scratch(tmp_path, 0)))
    video = video_of(9, (88, 104)).float()
    with pytest.raises(ValueError, match=r"slice 0 is .*float32, not .*bfloat16"):
        tiling.encode_tiled(models, slices_of(models, video), 9, tiles, Scratch(tmp_path, 0))
    assert not vae.calls and not path.exists()


class Short(StandIn):
    """A stand-in whose decoder gives a frame per latent."""

    def _decode(self, z: Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> Tensor:
        return super()._decode(z, memory_state)[:, :, : z.shape[2]]


def test_decode_refuses_slices_of_another_shape(tmp_path: Path) -> None:
    # 2 frames where 5 are due would be added to the first 2 of the scratch's 5, without a word.
    models, path = models_of(Short()), tmp_path / tiling.SCRATCH
    with pytest.raises(RuntimeError, match=r"slice 0 decoded as \(3, 2, 64, 64\) .*not \(3, 5, 64"):
        list(
            tiling.decode_tiled(models, latent_of(3, (11, 13)), Tiles(64, 16), Scratch(tmp_path, 0))
        )
    assert not path.exists() and vae_of(models).carry.memory is None


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


def test_real_vae_tiled_is_numz(real: model.Models, tmp_path: Path) -> None:
    # 9 frames of 48x64 under 32-px tiles sharing 16: 6 tiles, each in two slices.
    tiles = Tiles(32, 16)
    assert len(tiling.grid((6, 8), tiles)) == 6
    video = video_of(9, (48, 64))
    given = video.clone()
    reference = numz_encode(real, video, tiles)
    # numz crops its tiles from the input itself, and we from its copy in the scratch: the same
    # only if the encoder leaves its input as it was.
    assert same(video, given)
    latent = tiling.encode_tiled(real, slices_of(real, video), 9, tiles, Scratch(tmp_path, 0))
    assert latent.shape == (3, 6, 8, 16) and latent.stride() == reference.stride()
    assert same(latent, reference)
    assert not same(model.encode_stream(real, slices_of(real, video), 9), reference)

    decoded = numz_decode(real, latent, tiles)
    slices = list(tiling.decode_tiled(real, latent, tiles, Scratch(tmp_path, 0)))
    assert [s.shape[1] for s in slices] == model.decode_slices(real, 3) == [5, 4]
    assert [s.shape[1] for s in model.decode_stream(real, latent)] == [5, 4]
    assert decoded.shape == (3, 9, 48, 64) and same(torch.cat(slices, dim=1), decoded)
    assert bool(torch.isfinite(decoded.float()).all())
    assert not (tmp_path / tiling.SCRATCH).exists()


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
# requested, and up to 7,338,896 over in the tiled passes; the first run's 559,104 bytes
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


@pytest.mark.gpu
def test_passes_keep_nothing_of_the_shot_on_the_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The four passes over the real VAE on a card, for a short shot and a long one: the encodes'
    # latents are in system memory, in numz's layout, the decodes' slices on the device, and the
    # device holds the same for both shots at the pass's peak and nothing after it: nothing there
    # grows with the shot. The peak alone may not show what a pass gathers after its first slice,
    # the largest (5 frames against 4), when that slice sets the peak; what the device holds as
    # each of the VAE's calls starts hides nothing behind it, and is compared like with like: the
    # same call of both shots, the long shot's later calls with an earlier one of its own
    # (SETTLED), and a tile with an earlier one of its size. The largest of these figures isn't
    # compared between the shots: 35 calls have more occasions for it than 3. They are, like
    # the peak, the bytes asked of torch's allocator where it counts them, the bytes it allocated
    # otherwise (SETTLED): one choice for the run, said in every line and message.
    models = model.load_models(found(DITS), found(VAES), torch.device("cuda:0"))
    device = models.device
    height, width = PICTURE[0] // 8, PICTURE[1] // 8
    layout = tiling.grid((height, width), FOUR)
    assert len(layout) == 4
    scratch = Scratch(tmp_path, 0)  # in its file, mapped

    def fed(frames: int) -> Iterator[Tensor]:
        # The shot's input, made slice by slice: one on the device at a time.
        generator = torch.Generator().manual_seed(frames)
        for size in model.encode_slices(models, frames):
            yield (torch.rand((3, size, *PICTURE), generator=generator) * 2 - 1).to(device, BF16)

    def taken(slices: Iterator[Tensor]) -> list[int]:
        frames: list[int] = []
        for decoded in slices:
            assert decoded.device == device and decoded.dtype == BF16
            assert (decoded.shape[0], *decoded.shape[2:]) == (3, *PICTURE)
            frames.append(decoded.shape[1])
        return frames

    def passes(frames: int) -> list[Use]:
        # The untiled encode, the tiled one, the untiled decode of the first's latent, the tiled.
        latents = 1 + (frames - 1) // 4
        untiled, encode = device_use(
            models, monkeypatch, lambda: model.encode_stream(models, fed(frames), frames)
        )
        tiled, tiled_encode = device_use(
            models,
            monkeypatch,
            lambda: tiling.encode_tiled(models, fed(frames), frames, FOUR, scratch),
        )
        # Channel-major memory seen through a permuted view, numz's.
        numz = torch.empty((16, latents, height, width)).permute(1, 2, 3, 0)
        for latent in (untiled, tiled):
            assert latent.device.type == "cpu" and latent.dtype == BF16
            assert latent.shape == numz.shape and latent.stride() == numz.stride()
        assert not same(untiled, tiled)  # blended
        given, decode = device_use(
            models, monkeypatch, lambda: taken(model.decode_stream(models, untiled))
        )
        tiled_given, tiled_decode = device_use(
            models,
            monkeypatch,
            lambda: taken(tiling.decode_tiled(models, untiled, FOUR, scratch)),
        )
        assert given == tiled_given == model.decode_slices(models, latents) and sum(given) == frames
        assert not (tmp_path / tiling.SCRATCH).exists()
        # One figure per call of the VAE: a slice of the shot, and of each tile in a tiled pass.
        uses = [encode, tiled_encode, decode, tiled_decode]
        encodes, decodes = len(model.encode_slices(models, frames)), len(given)
        calls = [encodes, len(layout) * encodes, decodes, len(layout) * decodes]
        assert [len(use.allocated) for use in uses] == calls
        return uses

    try:
        # Once unmeasured: what a first pass sets up on the device for good is no shot's.
        passes(SHORT)
        short, long = passes(SHORT), passes(LONG)
    finally:
        models.runner.dit = models.runner.vae = None
        gc.collect()
        torch.cuda.empty_cache()
    names = ["encode", "tiled encode", "decode", "tiled decode"]
    whole, tiles = [(height, width)], [(tile.rows, tile.columns) for tile in layout]
    measured = list(zip(names, [whole, tiles, whole, tiles], short, long, strict=True))
    # What the peaks and the calls' figures are compared on, for the whole run.
    doubts = [use.doubt for use in (*short, *long) if use.doubt]
    on, why = ("allocated", f" ({doubts[0]})") if doubts else ("requested", "")

    def figures(use: Use) -> tuple[int, ...]:
        return use.requested if on == "requested" else use.allocated

    def top(use: Use) -> int:
        return use.requested_peak if on == "requested" else use.peak

    # Each pass's figures in a line per shot, said before anything is held of them: pytest shows
    # the output of a test that fails, and of one that passes under -rP or -s.
    for name, shapes, few, many in measured:
        for frames, use in ((SHORT, few), (LONG, many)):
            calls = len(use.allocated) // len(shapes)
            peak = both = "not given by torch"
            apart = ""
            if use.requested:
                peak, both = f"{use.requested_peak}", series(use.requested, shapes)
                slack = [
                    there - count for there, count in zip(use.allocated, use.requested, strict=True)
                ]
                apart = f"; allocated less requested: {min(slack)} to {max(slack)}"
            print(
                f"{name}, {frames} frames, compared on the {on} bytes{why}: requested peak"
                f" {peak}; allocated peak {use.peak}, the two shots' {abs(many.peak - few.peak)}"
                f" apart, and left {use.left}; held as each of the {calls}"
                f" calls{' of a tile' if len(shapes) > 1 else ''} starts, requested: {both};"
                f" allocated: {series(use.allocated, shapes)}{apart}"
            )
    for name, shapes, few, many in measured:
        said = f"{name}: {on} peaks {top(few)} and {top(many)}; allocated "
        if on == "requested":
            said += f"peaks {few.peak} and {many.peak}, "
        said += f"left {few.left} and {many.left}"
        if on == "requested":
            # The peak is this pass's: no less than what was requested as any of its calls
            # started, and reset before it (Use.doubt).
            for use in (few, many):
                most = max(use.requested)
                under = f"a peak of {use.requested_peak}, {most} as a call started: {said}"
                assert use.requested_peak >= most, under
        # The peaks of the two shots, on the requested bytes where they are taken. The allocated
        # peaks aren't compared then: the allocator hands out blocks larger than what is asked,
        # by up to 7 MiB in the tiled passes on the card, where the tiled decode's allocated
        # peaks were 1,671,168 bytes apart and every requested figure the same. The planner's
        # figures, measured at real sizes, are GiB, and were equal to the fourth decimal.
        assert top(few) > TOLERANCE, said  # measured: a pass takes megabytes
        assert abs(few.left) <= TOLERANCE and abs(many.left) <= TOLERANCE, said
        assert abs(top(many) - top(few)) <= TOLERANCE, said
        brief, long_run = by_tile(figures(few), len(shapes)), by_tile(figures(many), len(shapes))
        first: dict[tuple[int, int], int] = {}
        for index, shape in enumerate(shapes):
            starts, later = brief[index], long_run[index]
            said = (
                f"{name}, tile {index} of {len(shapes)}, {shape[0]}x{shape[1]} latents, {on}"
                f" bytes held as each call starts: over {SHORT} frames {runs(starts)}; over"
                f" {LONG} {runs(later)}"
            )
            assert 0 < len(starts) <= len(later) and SETTLED < len(later) - 1, said
            # 1. The same call of both shots, the convolutions' memory and the pass's own slices
            # as far along in both: what a pass puts on the device of the whole shot is in every
            # one, 1,310,720 bytes more for the long shot's 32 latents more.
            for call, start in enumerate(starts):
                assert abs(later[call] - start) <= TOLERANCE, f"call {call} of both shots: {said}"
            # 2. Nothing grows along the long shot: a buffer gathering it on the device takes
            # 40,960 bytes more a call or more, a call being one latent after the first:
            # 286,720 by the 7th call after the settled one, over the tolerance of 262,144, and
            # 26 x 40,960 = 1,064,960 by the last, the 34th, 4 times it; numz's moments twice
            # that.
            for call in range(SETTLED + 1, len(later)):
                against = f"call {call} against call {SETTLED}: {said}"
                assert abs(later[call] - later[SETTLED]) <= TOLERANCE, against
            # 3. Nothing gathers from tile to tile: a tile is measured against the first of its
            # size, whose convolutions keep as much (tiles 1 and 3 against 0 and 2 here, the
            # grid's lower row being 16 latents high, not 24). The smallest thing a tile could
            # leave of the long shot, its 36 latents' means or the decoder's input for them,
            # is 36 x 16 x 16 x 24 x 2 = 442,368 bytes, 1.7 times the tolerance.
            earlier = first.setdefault(shape, index)
            if earlier != index:
                against = (
                    f"call {SETTLED} against tile {earlier}'s, {runs(long_run[earlier])}: {said}"
                )
                assert abs(later[SETTLED] - long_run[earlier][SETTLED]) <= TOLERANCE, against
        # The last tile had an earlier one of its size to be measured against.
        assert len(shapes) == 1 or first[shapes[-1]] < len(shapes) - 1, shapes
