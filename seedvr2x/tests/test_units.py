"""The resumable units (DESIGN.md, Pause and resume), on the CPU: kept on disk with their values and
memory layout, the noise drawn again from a latent read back, windows resumed where they stopped,
the manifest recording each unit once its file is whole, and what a finished segment kept
removed; kept in memory, for a one-file output, as copies on the CPU alike; and the copy both
make, its layout kept or refused."""

import json
import os
import pickle
import re
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from seedvr2x.media.source import Source
from seedvr2x.runtime import model, units
from seedvr2x.runtime.job import OutputSegment, Shot
from seedvr2x.runtime.manifest import Manifest
from seedvr2x.runtime.plan import Plan
from seedvr2x.runtime.shot import moved, sample_windows, shot_layout
from seedvr2x.runtime.units import STATE, DiskUnits, Units


def channel_major(latents: int, seed: int = 0) -> Tensor:
    """A latent (T', h, w, 16) in the memory the VAE encode gives it: channel-major, a permuted
    view (model.encode)."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(16, latents, 3, 5, generator=generator).to(torch.bfloat16).permute(1, 2, 3, 0)
    )


def job(tmp_path: Path, shots: list[Shot], segments: list[OutputSegment]) -> Manifest:
    source = SimpleNamespace(
        path=tmp_path / "in.mkv",
        frames=shots[-1].end,
        stream=SimpleNamespace(
            frame_rate=Fraction(25),
            width=64,
            height=48,
            pix_fmt="yuv420p",
            color_primaries="",
            color_transfer="",
        ),
        sample_aspect=Fraction(1),
        conversion=SimpleNamespace(describe=lambda: "RGB"),
        sha256=None,
        index=None,
        idet=None,
    )
    source.path.write_bytes(b"")
    for segment in segments:
        (tmp_path / f"{segment.name}.mkv").write_bytes(b"x" * segment.frames)
    return Manifest(
        tmp_path / "manifest.json",
        {"seed": 42, "window": 5},
        {},
        cast(Source, source),
        shots,
        [shot_layout(shot.frames, 5) for shot in shots],
        segments,
        [f"{segment.name}.mkv" for segment in segments],
        {},
        Plan(5).record(),
    )


@pytest.mark.parametrize("latents", [1, 2, 9])
def test_kept_with_values_and_layout(tmp_path: Path, latents: int) -> None:
    record = job(tmp_path, [Shot(0, 4 * latents - 3)], [OutputSegment("a", 0, 4 * latents - 3)])
    units = DiskUnits(tmp_path, record)
    latent = channel_major(latents)
    units.save_latent(0, latent)
    back = units.latent(0)
    assert back is not None
    assert torch.equal(back, latent) and back.dtype == latent.dtype
    assert back.stride() == latent.stride()
    # The noise drawn from it, which depends on the layout, is drawn again alike.
    noises = []
    for tensor in (latent, back, latent.contiguous()):
        model.set_seed(7)
        noises.append(torch.randn_like(tensor, dtype=torch.bfloat16))
    assert torch.equal(noises[0], noises[1])
    if latents > 1:
        assert not torch.equal(noises[0], noises[2])  # why the layout is kept


class Payload:
    """What a tampered unit file could hold: an object whose unpickling runs code, here making a
    directory."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return os.mkdir, (str(self.marker),)


def test_tampered_unit_runs_no_code(tmp_path: Path) -> None:
    # A resume reads its units as tensors only (torch.load with weights_only=True), so a file
    # put in a unit's place can't run code (DESIGN.md, Pause and resume).
    record = job(tmp_path, [Shot(0, 5)], [OutputSegment("a", 0, 5)])
    units = DiskUnits(tmp_path, record)
    units.save_latent(0, channel_major(2))
    units.save_window(0, 0, channel_major(2))
    marker = tmp_path / "ran"
    for name in ("latent.pt", "window_0000.pt"):
        torch.save(Payload(marker), units.shot_directory(0) / name)
    with pytest.raises(pickle.UnpicklingError):
        units.latent(0)
    with pytest.raises(pickle.UnpicklingError):
        units.take_windows(0)
    assert not marker.exists()
    # The payload is real: read as any pickle, it runs.
    torch.load(units.shot_directory(0) / "latent.pt", weights_only=False)
    assert marker.is_dir()


def stand_in_dit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DiT whose output depends on the window's noise and latent."""
    monkeypatch.setattr(model, "condition", lambda models, noise, latent: latent)
    monkeypatch.setattr(model, "sample", lambda models, noise, cond: noise * 0.5 + cond)


@pytest.mark.parametrize("start", [1, 2])
def test_windows_resumed_as_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: int
) -> None:
    # A shot of 33 frames, 9 latents, windows of 5: (0, 5), (3, 7), (5, 9).
    stand_in_dit(monkeypatch)
    models = cast(model.Models, SimpleNamespace(device=torch.device("cpu")))
    layout = shot_layout(33, 5)
    assert layout == [(0, 5), (3, 7), (5, 9)]
    latent = channel_major(9)
    whole = list(sample_windows(models, latent, layout, 1234))
    record = job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)])
    units = DiskUnits(tmp_path, record)
    units.save_latent(0, latent)
    for window, sampled in enumerate(whole[:start]):
        units.save_window(0, window, sampled)
    # As a resumed run reads them back: in another process, the generators reseeded by others.
    torch.manual_seed(0)
    back = units.latent(0)
    assert back is not None
    rest = list(sample_windows(models, back, layout, 1234, units.windows_done(0)))
    assert len(rest) == len(layout) - start
    for ours, theirs in zip(rest, whole[start:], strict=True):
        assert torch.equal(ours, theirs) and ours.stride() == theirs.stride()


def test_manifest_records_each_unit_once_whole(tmp_path: Path) -> None:
    shots = [Shot(0, 9), Shot(9, 30)]  # 3 latents, one window; 6 latents, two windows
    record = job(tmp_path, shots, [OutputSegment("a", 0, 30)])
    seen: list[tuple[list[bool], list[int], list[str]]] = []
    write = record.write

    def recording() -> None:
        write()
        files = sorted(str(p.relative_to(tmp_path / STATE)) for p in (tmp_path / STATE).rglob("*"))
        seen.append((list(record.encoded), list(record.windows_done), files))

    record.write = recording  # type: ignore[method-assign]
    units = DiskUnits(tmp_path, record)
    for index, shot in enumerate(shots):
        latents = (shot.frames - 1) // 4 + 1
        units.save_latent(index, channel_major(latents))
        for window, (s, e) in enumerate(record.layouts[index]):
            units.save_window(index, window, channel_major(latents)[s:e] + 0)
        units.drop_latent(index)
    s0, s1 = "shot_000000", "shot_000009"
    assert seen == [
        ([True, False], [0, 0], [s0, f"{s0}/latent.pt"]),
        ([True, False], [1, 0], [s0, f"{s0}/latent.pt", f"{s0}/window_0000.pt"]),
        ([True, True], [1, 0], [s0, f"{s0}/window_0000.pt", s1, f"{s1}/latent.pt"]),
        (
            [True, True],
            [1, 1],
            [s0, f"{s0}/window_0000.pt", s1, f"{s1}/latent.pt", f"{s1}/window_0000.pt"],
        ),
        (
            [True, True],
            [1, 2],
            [
                s0,
                f"{s0}/window_0000.pt",
                s1,
                f"{s1}/latent.pt",
                f"{s1}/window_0000.pt",
                f"{s1}/window_0001.pt",
            ],
        ),
    ]
    back = units.take_windows(1)
    latent = channel_major(6)
    assert len(back) == 2
    for window, (s, e) in zip(back, record.layouts[1], strict=True):
        assert torch.equal(window, latent[s:e])
    assert (tmp_path / STATE / s1 / "window_0000.pt").exists()  # until the segment is finished
    units.segment_finished(0)
    assert not (tmp_path / STATE).exists()
    content = json.loads((tmp_path / "manifest.json").read_text())
    assert [(s["finished"], s["bytes"]) for s in content["segments"]] == [(True, 30)]
    assert not list(tmp_path.rglob("*.partial"))


def test_finished_segment_removes_its_shots_only(tmp_path: Path) -> None:
    shots = [Shot(0, 5), Shot(5, 9), Shot(9, 13)]
    record = job(tmp_path, shots, [OutputSegment("a", 0, 9), OutputSegment("b", 9, 13)])
    units = DiskUnits(tmp_path, record)
    for index in range(3):
        units.save_latent(index, channel_major(2))
        units.save_window(index, 0, channel_major(2))
    units.segment_finished(0)
    assert sorted(p.name for p in (tmp_path / STATE).iterdir()) == ["shot_000009"]
    assert record.finished == [True, False] and record.sizes == [9, None]
    units.segment_finished(1)
    assert not (tmp_path / STATE).exists()


def test_in_memory_until_decoded() -> None:
    units = Units()
    latent = channel_major(6)
    units.save_latent(0, latent)
    assert units.latent(0) is latent and units.windows_done(0) == 0
    units.save_window(0, 0, latent[:4])
    with pytest.raises(ValueError, match="window 2 after 1"):
        units.save_window(0, 2, latent[2:])
    units.save_window(0, 1, latent[2:])
    units.drop_latent(0)
    assert units.latent(0) is None and units.windows_done(0) == 2
    assert [w.shape[0] for w in units.take_windows(0)] == [4, 4]
    assert units.windows_done(0) == 0


def test_in_memory_units_are_cpu_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    # A one-file output's units wait on the CPU as an output directory's are read back: the copy
    # the shared helper makes of each (shot.moved), with the values, the dtype and the strides
    # of what was saved, a latent in the encode's channel-major layout and windows as the DiT
    # gives them. On the CPU that copy is the tensor itself: the helper is replaced by one that
    # copies, as it does from a GPU, so that what the units keep is told from what they were
    # given.
    asked: list[tuple[torch.device, str]] = []

    def copied(tensor: Tensor, device: torch.device, what: str) -> Tensor:
        asked.append((device, what))
        return moved(tensor, device, what).clone()

    monkeypatch.setattr(units, "moved", copied)
    kept = Units()
    latent = channel_major(6)
    saved = [latent[:4] + 0, torch.randn(4, 3, 5, 16).to(torch.bfloat16)]
    assert saved[0].stride() == (15, 5, 1, 60) and saved[1].is_contiguous()
    copy = kept.save_latent(0, latent)
    assert copy is kept.latent(0) and copy is not latent
    for window, sampled in enumerate(saved):
        kept.save_window(0, window, sampled)
    cpu = torch.device("cpu")
    assert asked == [
        (cpu, "shot 0's latent"),
        (cpu, "shot 0's window 0"),
        (cpu, "shot 0's window 1"),
    ]
    windows = kept.take_windows(0)
    for back, given in zip([copy, *windows], [latent, *saved], strict=True):
        assert back is not given and back.device == cpu
        assert torch.equal(back, given) and back.dtype == given.dtype == torch.bfloat16
        assert back.stride() == given.stride()


def test_saved_latent_is_the_copy_given_back(tmp_path: Path) -> None:
    # save_latent gives the copy on the CPU that it kept or wrote, for the run to sample the
    # shot's windows from: one copy of a latent in system memory, and no file read back.
    latent = channel_major(9)
    record = job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)])
    disk, in_memory = DiskUnits(tmp_path, record), Units()
    for kept in (in_memory.save_latent(0, latent), disk.save_latent(0, latent)):
        assert kept.device.type == "cpu" and torch.equal(kept, latent)
        assert kept.dtype == latent.dtype and kept.stride() == latent.stride()
    assert in_memory.save_latent(0, latent) is in_memory.latent(0)
    back = disk.latent(0)
    assert back is not None and torch.equal(back, latent) and back.stride() == latent.stride()


def test_moved_keeps_the_layout_or_refuses() -> None:
    # shot.moved, the copy every unit goes through: the tensor itself where it is already; on
    # another device, its strides kept, or refused. torch keeps the strides of a dense tensor,
    # a latent in its channel-major layout, and lays out anew the copy of one that isn't, a
    # slice of that latent along time, dense in the same order of dimensions. The other device
    # here is torch's meta, which holds a tensor's layout without its values: torch lays a copy
    # out by one rule whatever the device, and no test on the CPU can hold a GPU's values.
    latent = channel_major(9)
    cpu, meta = torch.device("cpu"), torch.device("meta")
    assert latent.stride() == (15, 5, 1, 135)
    assert moved(latent, cpu, "a latent") is latent
    assert moved(latent[2:5], cpu, "a slice").stride() == latent.stride()
    there = moved(latent, meta, "a latent")
    assert there.device == meta and there.stride() == latent.stride()
    assert there.shape == latent.shape and there.dtype == latent.dtype
    refusal = "a slice: its copy on meta has strides (15, 5, 1, 45), not its (15, 5, 1, 135)"
    with pytest.raises(RuntimeError, match=re.escape(refusal)):
        moved(latent[2:5], meta, "a slice")
