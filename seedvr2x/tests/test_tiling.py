"""The VAE tiling on the CPU (runtime/tiling.py): the grid, the scratch, and the tiled encode and
decode held bit for bit to numz's tiled_encode and tiled_decode, the vendored ones, over a stand-in
VAE and, once, over the real one with random weights."""

import errno
import logging
import os
import random
import weakref
from collections.abc import Callable, Iterator
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from seedvr2x.runtime import model, tiling
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
    # Removed with the store, and unmapped with the last tensor on it.
    assert not path.exists() and mapped(path)
    del kept
    assert not mapped(path)


def test_scratch_goes_on_failure(tmp_path: Path) -> None:
    scratch = Scratch(tmp_path, 0)
    path = tmp_path / tiling.SCRATCH
    with pytest.raises(Failing), scratch.store(2, (8, 8)):
        assert path.exists()
        raise Failing
    assert not path.exists()
    with scratch.store(2, (8, 8)):  # free again
        assert path.exists()


def test_scratch_replaces_a_killed_runs(tmp_path: Path) -> None:
    path = tmp_path / tiling.SCRATCH
    path.write_bytes(b"\xff" * 4096)
    with Scratch(tmp_path, 0).store(2, (8, 8)) as store:
        assert path.stat().st_size == Scratch.needed(2, (8, 8)) and not bits(store).any()
    assert not path.exists()


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

    # A file that can't be mapped: refused, and removed.
    def unmappable(*args: Any, **kwargs: Any) -> Tensor:
        assert path.stat().st_size == needed
        raise RuntimeError("unable to mmap")

    monkeypatch.setattr(torch, "from_file", unmappable)
    with pytest.raises(ScratchError, match=f"{path}.*{needed} bytes, not mapped: unable to mmap"):
        with scratch.store(2, (8, 8)):
            pass
    assert not path.exists()
    monkeypatch.undo()
    with scratch.store(2, (8, 8)):  # free again
        pass


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
    assert not path.exists() and not mapped(path)
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
    assert vae.carry.memory is None
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
    assert not mapped(path)


@pytest.mark.parametrize("call", [0, 3, 5])
def test_encode_failing_in_a_tile(stand_in: model.Models, tmp_path: Path, call: int) -> None:
    vae, path = vae_of(stand_in), tmp_path / tiling.SCRATCH
    scratch = Scratch(tmp_path, 0)
    video = video_of(13, (88, 104))
    failing_at(vae, call)
    with pytest.raises(Failing):
        tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, Tiles(64, 16), scratch)
    assert len(vae.calls) == call and not path.exists()
    assert vae.carry.memory is None
    with scratch.store(1, (8, 8)):  # free again
        pass


def test_encode_hands_its_moments_on_slice_by_slice(
    stand_in: model.Models, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each slice's moments leave for the sums as they are encoded. A tile's, kept whole, would
    # stay on the device through the next tile's pass and grow there with the shot: no pass
    # starts with more than the slice before's still alive.
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
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 21, Tiles(64, 16), scratch)
    assert len(most) == 5 * len(tiling.grid((11, 13), Tiles(64, 16))) and max(most) == 1


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
