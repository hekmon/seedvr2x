"""The shot's pass (seedvr2x/shot_pass.py) with the command line's parts, on the CPU: windows
read back from the units' files (units.py), a tiled pass's scratch where the run tells it
(scratch.py's Scratch: system memory up to its allowance, a file beyond, nothing left after a
failure), and a tiled decode corrected against the shot's input copy (shot.py's Correction) as a
streamed one is. The library's half, whose helpers and stand-ins these use, is
core/tests/test_shot_pass.py.

One test needs a GPU (pytest -m gpu) with the 7B whole on it and a window beside (20 GiB free),
and the weights (SEEDVR2X_MODEL_DIR: a 7B DiT in fp16 and the VAE, as tests/test_placement.py
takes them): a shot's windows through the real DiT, held to a one-window shot's on the device."""

import gc
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch
from posix_only import PROC
from stand_ins import DEVICE, channel_major, cpu_models  # noqa: F401
from test_scratch import gone, mapped
from test_shot_pass import (  # noqa: F401
    LARGE,
    LAYOUT,
    SMALL,
    SMALL_PADDED,
    TILES,
    WHOLE,
    Dit,
    Reads,
    allocated,
    contiguous,
    same_writes,
    shot_frames,
    shot_latent,
    stand_in,
    tiled_slices,
    written_by,
)
from test_tiling import (
    DITS,
    MEMORY,
    VAES,
    Failing,
    failing_at,
    found,
    latent_of,
    same,
)
from test_units import job

import shot
import units
from job import OutputSegment
from media.ffmpeg import MediaError
from scratch import DEFAULT_ALLOWANCE, Scratch, ScratchError
from seedvr2x import model, tiling
from seedvr2x.frames import padded_size
from seedvr2x.passes import Plan, Tiles
from seedvr2x.shot_pass import (
    decode_shot,
    encode_shot,
    sample_windows,
)
from seedvr2x.shots import Shot, shot_layout
from shot import Correction
from units import Units


def test_windows_from_the_units_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The latent as the units give it, the one written for a resume or the one a resume reads
    # back: the windows the encode's own latent gives. And that copy is the one the windows are
    # cut from, no other being made of the latent: on the CPU, where a slice sent to the device
    # is the slice, each window's latent is a view of the copy given.
    latent = channel_major(9)
    dit = Dit(monkeypatch)
    expected = list(sample_windows(DEVICE, latent, LAYOUT, 1234))
    disk = Units(tmp_path, job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)]))
    copies = [disk.save_latent(0, latent), disk.latent(0)]
    assert copies[0] is latent and copies[1] is not latent
    for copy in copies:
        assert copy is not None and copy.device.type == "cpu"
        dit.given.clear()
        windows = list(sample_windows(DEVICE, copy, LAYOUT, 1234))
        for ours, theirs in zip(windows, expected, strict=True):
            assert torch.equal(ours, theirs) and ours.stride() == theirs.stride()
        whole = copy.untyped_storage().data_ptr()
        assert len(dit.given) == len(LAYOUT)
        assert all(cut.untyped_storage().data_ptr() == whole for _, cut in dit.given)


def test_tiled_decode_is_corrected_as_a_streamed_one(
    stand_in: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With split, each slice a tiled decode hands on is corrected against its own frames of the
    # reference and written, as a streamed decode's is (_correct), the last one once the copy is
    # read whole. The shot's input copy is read from memory here, without ffmpeg.
    count = 13
    frames, latent = shot_frames(count), shot_latent(count)
    copies: list[Reads] = []

    @contextmanager
    def copy_reader(
        correction: Correction, count: int
    ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
        copies.append(Reads(frames))
        yield copies[-1]

    monkeypatch.setattr(shot, "_copy_reader", copy_reader)
    correction = Correction(Path("input.mkv"), Path("input.crc32"), SMALL[1], SMALL[0])
    tiled = written_by(stand_in, latent, count, reference=correction, tiles=TILES)
    with monkeypatch.context() as patch:
        patch.setattr(model, "decode_stream", tiled_slices)
        expected = written_by(stand_in, latent, count, reference=correction)
    assert same_writes(tiled, expected)
    assert [written.shape[0] for written in tiled] == [5, 4, 4]
    assert [sum(copy.counts) for copy in copies] == [count, count]
    # Corrected, and tiled: neither the tiled decode alone nor the untiled one corrected.
    assert not same_writes(tiled, written_by(stand_in, latent, count, tiles=TILES))
    assert not same_writes(tiled, written_by(stand_in, latent, count, reference=correction))


@PROC
def test_tiled_passes_keep_their_scratch_where_told(stand_in: model.Models, tmp_path: Path) -> None:  # noqa: F811
    count = 13
    frames, latent = shot_frames(count), shot_latent(count)
    vae = cast(Any, stand_in).runner.vae
    directory = tmp_path / "resume" / "shot_000000"  # not there yet
    path = directory / tiling.SCRATCH
    seen: list[bool] = []
    vae.before = lambda: seen.append(path.is_file() and mapped(path))

    # Within its allowance, the scratch is in system memory: nothing on disk, no directory made.
    # The allowance is the pass's to be given, as a run gives it its plan's; without one, here
    # for the encode, a default of 8 GiB (scratch.py's DEFAULT_ALLOWANCE).
    needed = Scratch.needed(count, padded_size(LARGE))
    assert needed < DEFAULT_ALLOWANCE == 8 * 2**30
    in_memory = encode_shot(
        stand_in, Reads(frames), count, LARGE, 42, tiles=TILES, store=Scratch(directory)
    )
    within: dict[str, Any] = {"tiles": TILES, "store": Scratch(directory, needed)}
    frames_in_memory = written_by(stand_in, latent, count, **within)
    assert len(seen) == 9 * 3 * 2 and not any(seen) and not list(tmp_path.iterdir())

    # Beyond it, here an allowance one byte short, then of 0: the file in the directory given,
    # made with it, there through every tile's pass and every write, gone after; the directory
    # stays, the run's to remove with the shot's units. The same bits either way.
    seen.clear()
    beyond: dict[str, Any] = {"tiles": TILES, "store": Scratch(directory, needed - 1)}
    on_disk = encode_shot(stand_in, Reads(frames), count, LARGE, 42, **beyond)
    assert len(seen) == 9 * 3 and all(seen) and same(on_disk, in_memory)
    assert not list(directory.iterdir()) and not mapped(path)
    seen.clear()
    there: list[bool] = []
    frames_on_disk: list[npt.NDArray[np.float32]] = []

    def write(written: npt.NDArray[np.float32]) -> None:
        there.append(path.is_file())
        frames_on_disk.append(written)

    decode_shot(stand_in, latent, count, LARGE, write, tiles=TILES, store=Scratch(directory, 0))
    assert len(seen) == 9 * 3 and all(seen) and there == [True] * 3
    assert same_writes(frames_on_disk, frames_in_memory)
    assert not list(directory.iterdir()) and not mapped(path)

    # The picture within one tile: the untiled pass, which asks for no scratch, and so runs
    # with nowhere to put one, whatever its allowance.
    seen.clear()
    encode_shot(stand_in, Reads(frames), count, LARGE, 42, tiles=WHOLE, store=Scratch(None, 0))
    written_by(stand_in, latent, count, tiles=WHOLE, store=Scratch(None, 0))
    assert len(seen) == 3 * 2 and not any(seen)

    # With nowhere to put its file, a scratch beyond the allowance is refused, as a file the run
    # can't write (cli._run reports a MediaError and exits 1).
    assert issubclass(ScratchError, MediaError)
    with pytest.raises(ScratchError, match="no directory for its file"):
        encode_shot(stand_in, Reads(frames), count, LARGE, 42, tiles=TILES, store=Scratch(None, 0))
    with pytest.raises(ScratchError, match="no directory for its file"):
        written_by(stand_in, latent, count, tiles=TILES, store=Scratch(None, 0))
    assert vae.carry.memory is None


@PROC
@pytest.mark.parametrize("failure", ["writer", "tile", "correction"])
def test_tiled_decode_failing_leaves_nothing(
    stand_in: model.Models,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    # A tiled decode whose consumer fails, the writer or the correction, or one of whose tiles
    # does: its scratch's file goes at once, while the failure is still held with its traceback,
    # and the VAE's causal memory is cleared.
    count = 13
    latent, vae = shot_latent(count), cast(Any, stand_in).runner.vae
    path = tmp_path / tiling.SCRATCH
    there: list[bool] = []
    options: dict[str, Any] = {"tiles": TILES, "store": Scratch(tmp_path, 0)}

    def write(frames: npt.NDArray[np.float32]) -> None:
        there.append(path.is_file() and mapped(path))
        raise Failing("the writer")

    if failure == "tile":
        failing_at(vae, 14)  # the fifth tile's last slice
    if failure == "correction":

        @contextmanager
        def copy_reader(
            correction: Correction, count: int
        ) -> Generator[Callable[[int], npt.NDArray[np.float32]]]:
            def read(n: int) -> npt.NDArray[np.float32]:
                there.append(path.is_file() and mapped(path))
                raise Failing("the copy")

            yield read

        monkeypatch.setattr(shot, "_copy_reader", copy_reader)
        options["reference"] = Correction(tmp_path / "input.mkv", tmp_path / "input.crc32", 28, 20)
    with pytest.raises(Failing) as raised:
        decode_shot(stand_in, latent, count, LARGE, write, **options)
    assert there == ([] if failure == "tile" else [True])
    assert not path.exists() and vae.carry.memory is None
    assert not list(tmp_path.iterdir())
    del raised
    assert not mapped(path)


def test_tiled_encode_failing_leaves_nothing(stand_in: model.Models, tmp_path: Path) -> None:  # noqa: F811
    # A tiled encode whose source fails while its slices are kept, or one of whose tiles does:
    # its scratch's file goes, and the VAE's causal memory is cleared.
    on_disk: dict[str, Any] = {"tiles": TILES, "store": Scratch(tmp_path, 0)}
    count = 13
    frames, vae = shot_frames(count), cast(Any, stand_in).runner.vae
    path = tmp_path / tiling.SCRATCH
    there: list[bool] = []

    def read(n: int) -> npt.NDArray[np.float32]:
        if n != 5:
            there.append(path.is_file())
            raise Failing("the source")
        return frames[:5]

    with pytest.raises(Failing, match="the source"):
        encode_shot(stand_in, read, count, LARGE, 42, **on_disk)
    assert there == [True] and gone(path)
    failing_at(vae, 4)  # the second tile's second slice
    with pytest.raises(Failing, match="pass 4"):
        encode_shot(stand_in, Reads(frames), count, LARGE, 42, **on_disk)
    assert gone(path) and vae.carry.memory is None
    assert not list(tmp_path.iterdir())


def test_a_window_s_file_allocates_the_window_alone(tmp_path: Path) -> None:
    # A window written to its file from its copy in system memory, and read back for the decode
    # (units.Units): torch allocates nothing to write it, and the tensor alone to read it.
    save = units._save  # pyright: ignore[reportPrivateUsage]
    load = units._load  # pyright: ignore[reportPrivateUsage]
    window, path = contiguous(8), tmp_path / "window_0000.pt"
    assert allocated(lambda: save(window, path)) == (0, 0)
    assert allocated(lambda: load(path)) == (window.nbytes, window.nbytes)


def test_the_tiled_encode_s_sums_are_divided_slice_by_slice(
    stand_in: model.Models,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tiled encode's end before its latent is made (tiling.encode_tiled): the tiles' sums go
    # to the device to be divided by the summed weights a slice at a time, 2 latents then 1
    # (model.latent_slices), and come back to their place: nothing there grows with the shot.
    # The sums given whole, 40 latents', what the tiles' passes would have summed (their part
    # is tests/test_tiling.py's), and the latent's making left out (the test above): torch
    # allocates 3 latents' worth at the most, a slice's copy while the slice before's is let go,
    # and the divisor, on the CPU here, the device's in a run, where the sums taken whole are
    # 40.
    latents, frames = 40, 157
    height, width = SMALL_PADDED[0] // 8, SMALL_PADDED[1] // 8
    sums = contiguous(latents).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    expected = sums.clone()
    one = sums.nbytes // latents
    count = torch.full((height, width), 2.0, dtype=model.COMPUTE_DTYPE)
    given: list[torch.Tensor] = []

    def summed(*arguments: object) -> tuple[torch.Tensor, torch.Tensor]:
        return sums, count

    def made(models: model.Models, mean: torch.Tensor) -> torch.Tensor:
        given.append(mean)
        return mean

    monkeypatch.setattr(tiling, "_means", summed)
    monkeypatch.setattr(model, "to_latent", made)
    sizes = model.encode_slices(stand_in, frames)
    fed = [torch.zeros((3, size, *SMALL_PADDED), dtype=model.COMPUTE_DTYPE) for size in sizes]
    tiles = Tiles(64, 16)
    assert len(tiling.grid((height, width), tiles)) > 1
    most, left = allocated(
        lambda: tiling.encode_tiled(stand_in, iter(fed), frames, tiles, Scratch(None, MEMORY))
    )
    assert len(given) == 1 and given[0] is sums and torch.equal(sums, expected / 2)
    assert model.latent_slices(stand_in, latents)[:2] == [2, 1]
    assert 2 * one <= most <= 3 * one + count.nbytes and left == 0


def test_a_tiled_decode_hands_its_frames_on_slice_by_slice(
    stand_in: model.Models,  # noqa: F811
    tmp_path: Path,
) -> None:
    # A tiled decode's end (tiling.decode_tiled): once its last tile is in, each slice of the
    # sums goes to the device to be divided and handed on, 5 frames at the most, never the
    # scratch whole, which grows with the shot. The scratch in its file, mapped, torch
    # allocating nothing for it: what torch allocates at the most over the pass, on the CPU
    # here, the device's in a run, is then the same to the byte for shots of 21, 81 and 161
    # frames, under 11 frames' worth (a tile's slice as it is weighted and summed, the divisor,
    # a slice handed on). The scratch taken whole to the device after the last tile would be 161
    # frames' more for the longest, which no figure read as a call of the VAE starts can see
    # (tests/test_tiling.py's test on a card): no call starts after it.
    tiles, size = Tiles(64, 16), (11, 13)
    frame = 3 * 88 * 104 * 2
    assert len(tiling.grid(size, tiles)) == 4
    taken: list[int] = []
    for latents in (6, 21, 41):
        latent = latent_of(latents, size)
        frames: list[int] = []

        def decoded(latent: torch.Tensor = latent, frames: list[int] = frames) -> None:
            passed = tiling.decode_tiled(stand_in, latent, tiles, Scratch(tmp_path, 0))
            frames.extend(handed.shape[1] for handed in passed)

        most, left = allocated(decoded)
        assert sum(frames) == 4 * latents - 3 and max(frames) == 5 and left == 0
        assert gone(tmp_path / tiling.SCRATCH)
        taken.append(most)
    assert taken[0] == taken[1] == taken[2] and 5 * frame < taken[0] < 11 * frame


# The shots of the windows' test on a card: 960x540, padded to 960x576, 276,480 bytes a latent in
# bfloat16 (16 channels over 72 x 120 positions). One shot of ONE_WINDOW latents in one window,
# and one of WINDOWED latents, 389 frames, which a cap of ONE_WINDOW lays out in MANY windows of
# that length, sharing 2 latents with each neighbour.
CARD_TARGET = (540, 960)


ONE_WINDOW, MANY = 8, 16


WINDOWED = MANY * (ONE_WINDOW - 2) + 2


# What two measures of the device may differ by, and still be the same, in the bytes asked of
# torch's allocator (tests/test_tiling.py says why those: SETTLED): 1 MiB. The least a run could
# keep there of a shot from window to window is a window's output, 8 x 276,480 = 2,211,840
# bytes, twice this; the shot's noise or its latent whole, 98 x 276,480 = 27,095,040. Where
# torch doesn't give those bytes, the blocks its allocator handed out are compared instead,
# which the same tensors fill by megabytes more or less (up to 9,765,504 bytes over what was
# asked, in the VAE's tiled passes on the card): to 12 MiB then, under the noise or the latent
# alone.
ASKED_TOLERANCE = 1 << 20


BLOCKS_TOLERANCE = 12 << 20


@dataclass(frozen=True)
class OnDevice:
    """What the device holds over a shot's windows as the run makes them (on_the_device), in
    bytes, each figure by torch's count of the blocks its allocator handed out
    (memory_allocated) and by its count of the bytes asked of it (memory_stats'
    requested_bytes; None where torch doesn't give it): as each window's unit starts; at each
    window's peak, its output still there; and once the shot's windows are done. reset: the
    two counts' peaks right after each unit's reset of them, which are what is held then when
    the reset took."""

    starts: tuple[tuple[int, int | None], ...]
    peaks: tuple[tuple[int, int | None], ...]
    left: tuple[int, int | None]
    reset: tuple[tuple[int, int | None], ...]

    def doubt(self) -> str:
        """Why the bytes asked of the allocator can't be compared, the blocks then being; empty
        when they can: torch gives them at every figure, never over the blocks handed out, and
        their peak was brought back to what was asked as each unit started."""
        figures = (*self.starts, *self.peaks, self.left, *self.reset)
        if any(asked is None for _, asked in figures):
            return "torch gives no requested_bytes"
        if any(asked is not None and asked > blocks for blocks, asked in figures):
            return "the requested bytes were over the allocated ones"
        if any(reset[1] != start[1] for reset, start in zip(self.reset, self.starts, strict=True)):
            return "resetting the peaks left the requested one above what was requested then"
        return ""


def on_the_device(
    models: model.Models,
    latents: int,
    cap: int,
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
) -> OnDevice:
    """A shot of `latents` latents at CARD_TARGET through the run's own steps for its windows
    (run._sample over the units a run keeps, units.Units in `directory`, made here, its windows
    capped at `cap` latents), its encode's latent given, random, to the units, in an encode's
    layout, which the steps read back as a resumed run does: what the device holds over them
    (OnDevice). The figures are read where the run's own are: as a unit starts,
    before the run resets the peak it logs (run._started), and as the window's output is given
    to the units, where the run reads that peak."""
    import run

    device = models.device
    started = run._started  # pyright: ignore[reportPrivateUsage]
    sample = run._sample  # pyright: ignore[reportPrivateUsage]
    starts: list[tuple[int, int | None]] = []
    peaks: list[tuple[int, int | None]] = []
    reset: list[tuple[int, int | None]] = []

    def asked(what: str = "current") -> int | None:
        return torch.cuda.memory_stats(device).get(f"requested_bytes.all.{what}")

    def starting(models: model.Models) -> float:
        torch.cuda.synchronize(device)
        starts.append((torch.cuda.memory_allocated(device), asked()))
        at = started(models)
        reset.append((torch.cuda.max_memory_allocated(device), asked("peak")))
        return at

    class Kept(Units):
        def save_window(self, run: int, window: int, sampled: torch.Tensor) -> None:
            torch.cuda.synchronize(device)
            peaks.append((torch.cuda.max_memory_allocated(device), asked("peak")))
            super().save_window(run, window, sampled)

    height, width = padded_size(CARD_TARGET)
    generator = torch.Generator().manual_seed(latents)
    values = torch.randn((16, latents, height // 8, width // 8), generator=generator)
    frames = 4 * latents - 3
    shots = [Shot(0, frames)]
    directory.mkdir()
    kept = Kept(directory, job(directory, shots, [OutputSegment("a", 0, frames)], cap))
    kept.save_latent(0, values.to(model.COMPUTE_DTYPE).permute(1, 2, 3, 0))
    del values
    job_ = (shots, CARD_TARGET, 42, Plan(cap), kept, None, None)
    with monkeypatch.context() as patch:
        patch.setattr(run, "_started", starting)
        sample(0, models, cast(Any, None), *job_)
    torch.cuda.synchronize(device)
    left = (torch.cuda.memory_allocated(device), asked())
    done = kept.windows_done(0)
    assert done == len(shot_layout(frames, cap)) == len(starts) == len(peaks) == len(reset)
    assert not (kept.shot_directory(0) / "latent.pt").exists()
    return OnDevice(tuple(starts), tuple(peaks), left, tuple(reset))


def test_the_card_s_test_reads_the_run_s_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What the windows' test on a card reads the device by (on_the_device), on the CPU, over a
    # stand-in DiT and the device's counts replaced: the run's own steps make every window of
    # the shot, a figure is read as each starts, before the run's reset, and one as each is
    # kept, and the shot's latent is given up. The figures are whatever the counts say: here 3
    # more bytes held each time the device is asked, and a window's pass 100 over that.
    Dit(monkeypatch)
    count = iter(range(0, 10_000, 3))
    held: dict[str, int] = {"now": 0, "peak": 0}
    sample = model.sample

    def now(device: torch.device) -> int:
        held["now"] = next(count)
        return held["now"]

    def sampled(models: object, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        held["peak"] = held["now"] + 100
        return sample(cast(model.Models, models), noise, cond)

    def reset(device: torch.device) -> None:
        held["peak"] = held["now"]

    def stats(device: torch.device) -> dict[str, int]:
        return {
            "requested_bytes.all.current": held["now"],
            "requested_bytes.all.peak": held["peak"],
        }

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", now)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: held["peak"])
    monkeypatch.setattr(torch.cuda, "memory_stats", stats)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(model, "sample", sampled)
    entered: list[str] = []
    models = cast(model.Models, SimpleNamespace(device=torch.device("cpu"), enter=entered.append))
    read = on_the_device(models, WINDOWED, ONE_WINDOW, monkeypatch, tmp_path / "windowed")
    assert entered == ["dit"] * MANY
    assert [start for start, _ in read.starts] == list(range(0, 3 * MANY, 3))
    assert read.starts == read.reset and read.doubt() == ""
    assert read.peaks == tuple((start + 100, start + 100) for start, _ in read.starts)
    assert read.left == (3 * MANY, 3 * MANY)
    layout = shot_layout(4 * WINDOWED - 3, ONE_WINDOW)
    assert [end - start for start, end in layout] == [ONE_WINDOW] * MANY
    alone = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "alone")
    assert len(alone.peaks) == 1


@pytest.mark.gpu
def test_a_shot_s_windows_hold_the_device_as_one_window_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The run's own steps for a shot's windows, over the real DiT on a card: a shot of 98
    # latents in 16 windows of 8 against a shot of 8 latents in one window. Between two windows
    # the device holds what it held before the shot, and each window peaks where the one-window
    # shot's does: nothing of the shot waits there but the window being made, neither its
    # latent, its noise nor a window's output (DESIGN.md, Memory planner, Per-phase peaks). A
    # first one-window shot runs unmeasured: what a process's first window leaves on the device
    # for good, the tables the DiT's blocks cache for a window's size, is no shot's.
    models = model.load_models(found(DITS), found(VAES), torch.device("cuda:0"))
    try:
        on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "first")
        alone = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "alone")
        windowed = on_the_device(models, WINDOWED, ONE_WINDOW, monkeypatch, tmp_path / "windowed")
        again = on_the_device(models, ONE_WINDOW, ONE_WINDOW, monkeypatch, tmp_path / "again")
    finally:
        models.runner.dit = models.runner.vae = None
        gc.collect()
        torch.cuda.empty_cache()
    layout = shot_layout(4 * WINDOWED - 3, ONE_WINDOW)
    assert [end - start for start, end in layout] == [ONE_WINDOW] * MANY
    assert (len(alone.peaks), len(windowed.peaks), len(again.peaks)) == (1, MANY, 1)
    # What the figures are compared on, for the whole run, and to what.
    doubts = [read.doubt() for read in (alone, windowed, again) if read.doubt()]
    on, column, tolerance = ("requested", 1, ASKED_TOLERANCE)
    if doubts:
        on, column, tolerance = (f"allocated ({doubts[0]})", 0, BLOCKS_TOLERANCE)

    def figures(read: tuple[tuple[int, int | None], ...]) -> list[int]:
        return [cast(int, figure[column]) for figure in read]

    # Said before anything is held of them: pytest shows the output of a test that fails, and of
    # one that passes under -rP or -s.
    for name, read in (("one window", alone), ("16 windows", windowed), ("one window", again)):
        print(
            f"{name}, compared on the {on} bytes: held as each window starts"
            f" {figures(read.starts)}, left {figures((read.left,))[0]}; peaks"
            f" {figures(read.peaks)}; allocated: starts {[blocks for blocks, _ in read.starts]},"
            f" peaks {[blocks for blocks, _ in read.peaks]}"
        )
    before, peak = figures(alone.starts)[0], figures(alone.peaks)[0]
    said = f"on the {on} bytes, {before} held before the one-window shot, its peak {peak}"
    # A window takes gigabytes over the weights, at this size.
    assert peak - before > 256 << 20, said
    for name, read in (("one window", alone), ("16 windows", windowed), ("one window", again)):
        # Between two windows, and after the last, what the device held before the shot.
        for window, start in enumerate(figures(read.starts)):
            held = f"{name}: {start} held as window {window} starts: {said}"
            assert abs(start - before) <= tolerance, held
        left = figures((read.left,))[0]
        assert abs(left - before) <= tolerance, f"{name}: {left} left: {said}"
        # Each window's peak, the one-window shot's.
        for window, top in enumerate(figures(read.peaks)):
            reached = f"{name}: window {window} peaks at {top}: {said}"
            assert abs(top - peak) <= tolerance, reached
