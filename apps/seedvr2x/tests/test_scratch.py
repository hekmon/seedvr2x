"""The command line's store for the tiled passes (scratch.py; DESIGN.md, Pipeline step 3):
system memory up to the plan's allowance, else a memory-mapped file in the shot's unit directory,
removed with the store; its errors; and the library's tiling.InMemory held to its in-memory
branch. The passes themselves, over the library's store, are test_tiling.py's: here, the ones
whose point is the file, that a pass which fails, is closed early or is refused leaves none, and
(pytest -m gpu) that the passes over the mapped file keep nothing of the shot on the device."""

import errno
import gc
import logging
import mmap
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_tiling import (
    CASES,
    DITS,
    FOUR,
    LONG,
    MEMORY,
    PICTURE,
    SETTLED,
    SHORT,
    TOLERANCE,
    VAES,
    Failing,
    StandIn,
    Use,
    bits,
    by_tile,
    device_use,
    failing_at,
    found,
    latent_of,
    models_of,
    numz_decode,
    random_cases,
    runs,
    same,
    series,
    slices_of,
    vae_of,
    video_of,
)
from torch import Tensor

from scratch import DEFAULT_ALLOWANCE, Scratch, ScratchError, remove_pending
from seedvr2x import model, tilegrid, tiling
from seedvr2x.passes import Tiles

posix_os: Any = os

BF16 = torch.bfloat16
CPU = torch.device("cpu")


@pytest.fixture
def stand_in() -> model.Models:
    return models_of(StandIn())


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


# The scratch.


def test_scratch_needed_is_the_tile_grids_bytes() -> None:
    # The library's tilegrid.scratch_bytes (seedvr2x/tilegrid.py) is what Scratch.needed counts:
    # one home for the planner's figure and the store's (test_tilegrid.py, the library's, holds
    # the library's half of it).
    for frames, padded in ((1, (1104, 1920)), (17, (64, 96)), (301, (2192, 3840))):
        assert Scratch.needed(frames, padded) == tilegrid.scratch_bytes(frames, padded)


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
    path = directory / tilegrid.SCRATCH
    with scratch.store(5, (16, 24)) as store:
        assert (store.shape, store.dtype, store.device) == ((3, 5, 16, 24), BF16, CPU)
        assert not bits(store).any()
        # Whole on disk from the start, its blocks allocated.
        assert path.stat().st_size == needed <= cast(Any, path.stat()).st_blocks * 512
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
    path = tmp_path / tilegrid.SCRATCH
    with pytest.raises(Failing), scratch.store(2, (8, 8)):
        assert path.exists()
        raise Failing
    assert not path.exists() and not mapped(path) and not open_on(path)
    with scratch.store(2, (8, 8)):  # free again
        assert path.exists()


def test_scratch_allowance_is_its_passs() -> None:
    # What a scratch may take of system memory is given with it, a run's from its plan
    # (shot_plan.plan); one given none, by a test or a script, takes a default of 8 GiB.
    assert Scratch(None, 0).allowance == 0 and Scratch(None, 12_345).allowance == 12_345
    assert Scratch(None).allowance == Scratch(None, None).allowance == DEFAULT_ALLOWANCE
    assert DEFAULT_ALLOWANCE == 8 * 2**30
    assert Scratch(None).where(675, (1104, 1920)) == "system memory"
    with (
        pytest.raises(ScratchError, match=f"over the {8 * 2**30} allowed in system memory"),
        Scratch(None).store(676, (1104, 1920)),
    ):
        pass


def gone(path: Path) -> bool:
    """No file at path once its store has ended: on Windows, which deletes no file still mapped,
    once its pending removal has run (scratch.remove_pending), as the next store's start and the
    segment's end run it; on Linux it is gone as the store ends."""
    if sys.platform == "win32":
        remove_pending()
    return not path.exists()


def test_scratch_replaces_a_killed_runs(tmp_path: Path) -> None:
    path = tmp_path / tilegrid.SCRATCH
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
    path, other = directory / tilegrid.SCRATCH, tmp_path / "other"
    needed = Scratch.needed(5, (16, 24))
    theirs = b"\x55" * needed
    other.write_bytes(theirs)
    allocate = posix_os.posix_fallocate
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
        written = torch.frombuffer(
            bytearray(posix_os.pread(descriptor, needed, 0)), dtype=torch.int16
        )
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
    path = tmp_path / tilegrid.SCRATCH
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
    import resource

    posix_resource: Any = resource
    path = tmp_path / tilegrid.SCRATCH
    needed = Scratch.needed(4000, (1024, 1024))
    monkeypatch.setattr(
        os, "posix_fallocate", lambda descriptor, offset, length: os.ftruncate(descriptor, length)
    )
    has = int(Path("/proc/self/statm").read_text().split()[0]) * posix_os.sysconf("SC_PAGE_SIZE")
    soft, hard = posix_resource.getrlimit(posix_resource.RLIMIT_AS)
    limit = has + (1 << 30)
    if soft != posix_resource.RLIM_INFINITY:
        limit = min(limit, soft)
    posix_resource.setrlimit(posix_resource.RLIMIT_AS, (limit, hard))
    try:
        with pytest.raises(ScratchError, match=f"{needed} bytes, not mapped: .*Cannot alloc"):
            with Scratch(tmp_path, 0).store(4000, (1024, 1024)):
                pass
    finally:
        posix_resource.setrlimit(posix_resource.RLIMIT_AS, (soft, hard))
    assert not path.exists() and not mapped(path) and not open_on(path)


def test_scratch_that_stays_is_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A scratch that can't be removed changes no result: a warning, with the reason.
    path = tmp_path / tilegrid.SCRATCH
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


def test_scratch_in_memory_is_the_scratch_on_disk(stand_in: model.Models, tmp_path: Path) -> None:
    tiles = Tiles(64, 16)
    latent = latent_of(4, (11, 13))
    path = tmp_path / tilegrid.SCRATCH
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
    path = tmp_path / tilegrid.SCRATCH
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
    vae, path = vae_of(stand_in), tmp_path / tilegrid.SCRATCH
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
    # A failure of seedvr2x's own, between a tile's passes, as a Ctrl-C raised while the device is
    # polled: the tile's pass is closed, which clears its memory at once, and the scratch goes.
    vae, path = vae_of(stand_in), tmp_path / tilegrid.SCRATCH
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
    vae, path = vae_of(stand_in), tmp_path / tilegrid.SCRATCH
    scratch = Scratch(tmp_path, 0)
    video = video_of(13, (88, 104))
    failing_at(vae, call)
    with pytest.raises(Failing):
        tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, Tiles(64, 16), scratch)
    assert len(vae.calls) == call and not path.exists()
    assert vae.carry.memory is None and not mapped(path) and not open_on(path)
    with scratch.store(1, (8, 8)):  # free again
        pass


# InMemory, the library's store, against the in-memory branch of the command line's.


@pytest.mark.parametrize("frames", [1, 5])
def test_in_memory_is_the_scratch_in_memory(tmp_path: Path, frames: int) -> None:
    # tiling.InMemory makes the store Scratch makes within its allowance: the same shape, dtype,
    # device and zeros, said to be in system memory, one at a time and refused alike when a
    # second is asked for; nothing on disk.
    padded = (16, 24)
    ours, theirs = tiling.InMemory(), Scratch(tmp_path, Scratch.needed(frames, padded))
    assert ours.where(frames, padded) == theirs.where(frames, padded) == "system memory"
    with ours.store(frames, padded) as one, theirs.store(frames, padded) as other:
        assert (one.shape, one.dtype, one.device) == (other.shape, other.dtype, other.device)
        assert (one.shape, one.dtype, one.device) == ((3, frames, *padded), BF16, CPU)
        assert one.stride() == other.stride() and not bits(one).any() and not bits(other).any()
        for store in (ours, theirs):
            with (
                pytest.raises(RuntimeError, match="a scratch holds one store at a time"),
                store.store(1, (8, 8)),
            ):
                pass
    for store in (ours, theirs):
        with store.store(1, (8, 8)):  # free again
            pass
    assert not list(tmp_path.iterdir())
    # Larger than any allowance, still in system memory: the library's holds no file.
    assert ours.where(10_000, (2192, 3840)) == "system memory"
    assert Scratch(tmp_path, 0).where(10_000, (2192, 3840)) == str(tmp_path / tilegrid.SCRATCH)


def test_a_tiled_pass_says_its_file(
    stand_in: model.Models, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The pass's debug line (tests/test_tiling.py has the memory's) gives the store's own place:
    # its file's path where the scratch went to its file.
    caplog.set_level(logging.DEBUG, logger="seedvr2x.tiling")
    tiles = Tiles(64, 16)
    latent, video = latent_of(4, (11, 13)), video_of(13, (88, 104))
    assert Scratch.needed(13, (88, 104)) == 713_856
    list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, MEMORY)))
    tiling.encode_tiled(stand_in, slices_of(stand_in, video), 13, tiles, Scratch(tmp_path, 0))
    grid = "4 tiles of 64 px sharing 16, 2 across and 2 down, over 13 frames padded to 104x88"
    assert [record.getMessage() for record in caplog.records] == [
        f"tiled decode: {grid}; its scratch, 713856 bytes, in system memory",
        f"tiled encode: {grid}; its scratch, 713856 bytes, in {tmp_path / tilegrid.SCRATCH}",
    ]


@pytest.mark.parametrize("case", [*CASES, *random_cases(8, 2)])
def test_the_passes_on_the_file_are_the_passes_in_memory(
    stand_in: model.Models, tmp_path: Path, case: tuple[int, int, int, int, int]
) -> None:
    # The file changes no value: a tiled decode and a tiled encode over a mapped file give what
    # they give over tiling.InMemory (test_tiling.py holds those to numz's), and leave no file.
    rows, columns, latents, size, overlap = case
    tiles = Tiles(size, overlap)
    frames = 1 + 4 * (latents - 1)
    latent = latent_of(latents, (rows, columns), seed=rows * columns + latents)
    video = video_of(frames, (rows * 8, columns * 8), seed=rows * columns + latents)
    in_memory = list(tiling.decode_tiled(stand_in, latent, tiles, tiling.InMemory()))
    on_disk = list(tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, 0)))
    assert len(on_disk) == len(in_memory)
    assert all(same(disk, memory) for disk, memory in zip(on_disk, in_memory, strict=True))
    assert same(torch.cat(on_disk, dim=1), numz_decode(stand_in, latent, tiles))
    kept = tiling.encode_tiled(
        stand_in, iter(slices_of(stand_in, video)), frames, tiles, Scratch(tmp_path, 0)
    )
    memory = tiling.encode_tiled(
        stand_in, iter(slices_of(stand_in, video)), frames, tiles, tiling.InMemory()
    )
    assert same(kept, memory) and kept.stride() == memory.stride()
    assert not (tmp_path / tilegrid.SCRATCH).exists()


def test_refused_passes_leave_no_file(stand_in: model.Models, tmp_path: Path) -> None:
    # A pass refused before it runs (slices that don't follow the VAE's, a latent in another
    # dtype, frames that aren't multiples of 8) leaves no file, and has made no pass of the VAE.
    vae, path = vae_of(stand_in), tmp_path / tilegrid.SCRATCH
    tiles = Tiles(64, 16)
    video = video_of(13, (88, 104))
    good = slices_of(stand_in, video)
    for slices, match in (
        (good[:2], "2 slices, the VAE takes"),
        ([good[0], good[1][:, :, :80], good[2]], "slice 1 is"),
        ([s.float() for s in good], "slice 0 is .*float32"),
    ):
        with pytest.raises(ValueError, match=match):
            tiling.encode_tiled(stand_in, iter(slices), 13, tiles, Scratch(tmp_path, 0))
        assert not vae.calls and not path.exists()
    with pytest.raises(ValueError, match="multiples of 8"):
        tiling.encode_tiled(stand_in, [video_of(5, (90, 104))], 5, tiles, Scratch(tmp_path, 0))
    with pytest.raises(ValueError, match=r"latent torch\.float32"):
        list(
            tiling.decode_tiled(
                stand_in, latent_of(3, (11, 13)).float(), tiles, Scratch(tmp_path, 0)
            )
        )
    assert not vae.calls and not path.exists()


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
    # A tiled decode hands its frames on after its last tile, when no call of the VAE starts any
    # more: what the device holds then is read as each decoded slice is handed on (taken), and
    # held between the shots and from slice to slice. Without it the whole scratch moved to the
    # device there passed this test, under the pass's peak and gone before the pass's end (a
    # review's mutation, 2026-10-12): 3 x 141 x 256 x 320 x 2 = 69,304,320 bytes for the long
    # shot against 6,389,760 for the short one.
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

    def held_now() -> tuple[int, int | None]:
        torch.cuda.synchronize(device)
        asked = torch.cuda.memory_stats(device).get("requested_bytes.all.current")
        return torch.cuda.memory_allocated(device), asked

    def taken(slices: Iterator[Tensor], handed: list[tuple[int, int | None]]) -> list[int]:
        # handed: what the device holds as each slice is handed on, the slice among it, over
        # what it held before the pass's first call: the allocated bytes and the requested ones.
        frames: list[int] = []
        before, asked_before = held_now()
        for decoded in slices:
            assert decoded.device == device and decoded.dtype == BF16
            assert (decoded.shape[0], *decoded.shape[2:]) == (3, *PICTURE)
            frames.append(decoded.shape[1])
            there, asked = held_now()
            apart = None if asked is None or asked_before is None else asked - asked_before
            handed.append((there - before, apart))
        return frames

    def passes(frames: int) -> tuple[list[Use], list[list[tuple[int, int | None]]]]:
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
        handed: list[list[tuple[int, int | None]]] = [[], []]
        given, decode = device_use(
            models, monkeypatch, lambda: taken(model.decode_stream(models, untiled), handed[0])
        )
        tiled_given, tiled_decode = device_use(
            models,
            monkeypatch,
            lambda: taken(tiling.decode_tiled(models, untiled, FOUR, scratch), handed[1]),
        )
        assert [len(held) for held in handed] == [len(given), len(tiled_given)]
        assert given == tiled_given == model.decode_slices(models, latents) and sum(given) == frames
        assert not (tmp_path / tilegrid.SCRATCH).exists()
        # One figure per call of the VAE: a slice of the shot, and of each tile in a tiled pass.
        uses = [encode, tiled_encode, decode, tiled_decode]
        encodes, decodes = len(model.encode_slices(models, frames)), len(given)
        calls = [encodes, len(layout) * encodes, decodes, len(layout) * decodes]
        assert [len(use.allocated) for use in uses] == calls
        return uses, handed

    try:
        # Once unmeasured: what a first pass sets up on the device for good is no shot's.
        passes(SHORT)
        (short, short_handed), (long, long_handed) = passes(SHORT), passes(LONG)
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
        # by up to 9.8 MB in the tiled passes on the card, where the tiled decode's allocated
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
    # What the device holds as each decoded slice is handed on, on the same bytes as the rest.
    column = 1 if on == "requested" else 0
    decodes = zip(("decode", "tiled decode"), short_handed, long_handed, strict=True)
    for name, few_held, many_held in decodes:
        brief = tuple(cast(int, held[column]) for held in few_held)
        along = tuple(cast(int, held[column]) for held in many_held)
        said = (
            f"{name}, {on} bytes held as each slice is handed on: over {SHORT} frames"
            f" {runs(brief)}; over {LONG} {runs(along)}"
        )
        print(said)
        assert len(brief) > 2 and len(along) > SETTLED + 1, said
        # The slice handed on is there, megabytes of it: 5 frames, then 4.
        assert min(brief) > TOLERANCE and min(along) > TOLERANCE, said
        # 1. The same slice of both shots: the whole shot's frames on the device there are
        # 62,914,560 bytes more for the long shot's 128 frames more.
        for index, held in enumerate(brief):
            assert abs(along[index] - held) <= TOLERANCE, f"slice {index} of both shots: {said}"
        # 2. Nothing gathers as the slices are handed on: a slice kept there is 1,966,080 bytes
        # more at the next one, 7 times the tolerance. The streamed decode against a settled
        # slice, as its calls are; the tiled one, every tile decoded by then, against its
        # second, the first of 4 frames.
        against = SETTLED if name == "decode" else 1
        for index in range(against + 1, len(along)):
            gathered = f"slice {index} against slice {against}: {said}"
            assert abs(along[index] - along[against]) <= TOLERANCE, gathered


def test_on_windows_a_scratch_still_mapped_is_removed_at_the_next_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Windows deletes no file a mapping holds (WinError 32), and the store's caller may still hold
    # a tensor on it as the block ends: said at debug, and removed at the next store's start, by
    # then unmapped, where Linux unlinks it under the mapping (impl-win-3's review, 2026-11-09).
    import scratch as module
    from media import files

    monkeypatch.setattr(module.sys, "platform", "win32")
    # Windows' room reserved (files._reserve) stood in by the file's extension.
    monkeypatch.setattr(files, "_reserve", lambda descriptor, size: os.ftruncate(descriptor, size))
    monkeypatch.setattr(module, "_PENDING", [])
    refusing = [True]
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if refusing[0] and self.name == tilegrid.SCRATCH and self.exists():
            error = PermissionError(13, "The process cannot access the file", str(self))
            error.winerror = 32  # type: ignore[attr-defined]  # Windows' attribute, set by hand
            raise error
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    caplog.set_level(logging.DEBUG, logger="cli.scratch")
    first, second = tmp_path / "shot_0", tmp_path / "shot_1"
    with Scratch(first, allowance=0).store(1, (16, 16)) as store:
        store.fill_(1)
    del store
    gc.collect()
    left = first / tilegrid.SCRATCH
    assert left.exists() and module._PENDING == [left]  # pyright: ignore[reportPrivateUsage]
    assert "still mapped, removed at the next store" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    refusing[0] = False
    with Scratch(second, allowance=0).store(1, (16, 16)):
        assert not left.exists() and module._PENDING == []  # pyright: ignore[reportPrivateUsage]


def test_on_windows_a_scratch_pending_at_a_segments_end_is_removed_before_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # remove_pending, called by units.segment_finished and as main returns, removes what Windows
    # kept mapped once the mapping has gone, so that the shot's directory removal, which ignores
    # errors, finds it empty; a file still refused stays pending, the next start discarding it
    # as a leftover (resume.leftovers; impl-win-4, 2026-11-09).
    import scratch as module

    left = tmp_path / "shot_0" / tilegrid.SCRATCH
    left.parent.mkdir()
    left.write_bytes(b"x")
    monkeypatch.setattr(module, "_PENDING", [left])
    real = Path.unlink
    refusing = [True]

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if refusing[0] and self == left:
            raise PermissionError(13, "The process cannot access the file", str(self))
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    module.remove_pending()
    assert left.exists() and module._PENDING == [left]  # pyright: ignore[reportPrivateUsage]
    refusing[0] = False
    module.remove_pending()
    assert not left.exists() and module._PENDING == []  # pyright: ignore[reportPrivateUsage]
    shutil.rmtree(left.parent)
    assert not left.parent.exists()
    module.remove_pending()  # nothing pending: nothing to do


def test_a_finished_shots_directory_goes_with_its_pending_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A finished segment's last decode can still map its scratch as the segment is recorded, so
    # that the shot's directory stays; once the file goes, the directories it left empty go, the
    # job's resume directory too, and never a file: a unit made again there meanwhile (the
    # assembly's remake of a damaged segment) keeps its own (natively, 2026-11-10: both stayed).
    import scratch as module

    root = tmp_path / "resume"
    shot, other = root / "shot_000004", root / "shot_000009"
    left = shot / tilegrid.SCRATCH
    shot.mkdir(parents=True)
    left.write_bytes(b"x")
    monkeypatch.setattr(module, "_PENDING", [left])
    monkeypatch.setattr(module, "_LEFT", [])
    real = Path.unlink
    refusing = [True]

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if refusing[0] and self == left:
            raise PermissionError(13, "The process cannot access the file", str(self))
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    module.remove_with(shot)
    module.remove_with(root)
    module.remove_pending()
    assert left.exists() and shot.is_dir()
    refusing[0] = False
    (other / "window_0000.pt").parent.mkdir()
    (other / "window_0000.pt").write_bytes(b"remade")
    module.remove_pending()
    # The emptied shot's directory gone; the other's file kept, so its directory and root stay.
    assert not shot.exists() and (other / "window_0000.pt").read_bytes() == b"remade"
    assert module._LEFT == []  # pyright: ignore[reportPrivateUsage]
    (other / "window_0000.pt").unlink()
    other.rmdir()
    module.remove_with(root)
    module.remove_pending()
    assert not root.exists() and module._LEFT == []  # pyright: ignore[reportPrivateUsage]
