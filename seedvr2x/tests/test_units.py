"""The resumable units (DESIGN.md, Pause and resume), on the CPU: kept on disk with their values and
memory layout, the noise drawn again from a latent read back, windows resumed where they stopped,
the manifest recording each unit once its file is whole, and what a finished segment kept
removed; each written from a copy on the CPU; and that copy, its layout kept or refused."""

import hashlib
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
from seedvr2x.runtime.units import STATE, Units


def channel_major(latents: int, seed: int = 0) -> Tensor:
    """A latent (T', h, w, 16) in the memory the VAE encode gives it: channel-major, a permuted
    view (model.encode)."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(16, latents, 3, 5, generator=generator).to(torch.bfloat16).permute(1, 2, 3, 0)
    )


def job(
    tmp_path: Path, shots: list[Shot], segments: list[OutputSegment], window: int = 5
) -> Manifest:
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
        {"seed": 42, "window": window},
        {},
        cast(Source, source),
        shots,
        [shot_layout(shot.frames, window) for shot in shots],
        segments,
        [f"{segment.name}.mkv" for segment in segments],
        {},
        Plan(window).record(),
    )


@pytest.mark.parametrize("latents", [1, 2, 9])
def test_kept_with_values_and_layout(tmp_path: Path, latents: int) -> None:
    record = job(tmp_path, [Shot(0, 4 * latents - 3)], [OutputSegment("a", 0, 4 * latents - 3)])
    units = Units(tmp_path, record)
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
    units = Units(tmp_path, record)
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
    units = Units(tmp_path, record)
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
    units = Units(tmp_path, record)
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
    units = Units(tmp_path, record)
    for index in range(3):
        units.save_latent(index, channel_major(2))
        units.save_window(index, 0, channel_major(2))
    units.segment_finished(0)
    assert sorted(p.name for p in (tmp_path / STATE).iterdir()) == ["shot_000009"]
    assert record.finished == [True, False] and record.sizes == [9, None]
    units.segment_finished(1)
    assert not (tmp_path / STATE).exists()


def test_a_segment_is_hashed_when_finished_and_set_back_when_damaged(tmp_path: Path) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-12): the segment's bytes hashed as it is
    # finished (a file's SHA-256, a directory's over its files' names, sizes and bytes in name
    # order); damaged, it is recorded as unfinished with its shots' runs not begun, the other
    # segment untouched.
    shots = [Shot(0, 5), Shot(5, 9), Shot(9, 13)]
    record = job(tmp_path, shots, [OutputSegment("a", 0, 9), OutputSegment("b", 9, 13)])
    units = Units(tmp_path, record)
    for index in range(3):
        units.save_latent(index, channel_major(2))
        units.save_window(index, 0, channel_major(2))
    units.segment_finished(0)
    units.segment_finished(1)
    assert record.hashes == [hashlib.sha256(b"x" * n).hexdigest() for n in (9, 4)]
    record.segment_damaged(0)
    assert (record.finished, record.sizes, record.hashes[0]) == ([False, True], [None, 4], None)
    assert record.hashes[1] is not None
    assert (record.encoded, record.windows_done) == ([False, False, True], [0, 0, 1])
    content = json.loads((tmp_path / "manifest.json").read_text())
    assert [(s["finished"], s["bytes"], s["sha256"]) for s in content["segments"]] == [
        (False, None, None),
        (True, 4, record.hashes[1]),
    ]


def test_a_record_without_hashes_is_taken_up_and_compared_nowhere(tmp_path: Path) -> None:
    # A finished segment's hash is part of its record, compared at no resume (resume.UNCOMPARED):
    # a record with none (older code's, which a code change refuses anyway) is read, not crashed
    # on, and a record with and one without differ in nothing.
    from seedvr2x.runtime import resume

    record = job(tmp_path, [Shot(0, 9)], [OutputSegment("a", 0, 9)])
    Units(tmp_path, record).save_latent(0, channel_major(3))
    Units(tmp_path, record).save_window(0, 0, channel_major(3))
    Units(tmp_path, record).segment_finished(0)
    with_hash = record.content()
    without = json.loads(json.dumps(with_hash))
    del without["segments"][0]["sha256"]
    assert resume.differences(without, with_hash) == []
    again = job(tmp_path, [Shot(0, 9)], [OutputSegment("a", 0, 9)])
    resume.adopt(again, without)
    assert (again.finished, again.hashes) == ([True], [None])


def test_a_directorys_hash_frames_its_files(tmp_path: Path) -> None:
    # Bytes moved from one file to its neighbour change the hash, as do a file renamed and a
    # byte flipped; a file's is its content's SHA-256.
    one, other = tmp_path / "one", tmp_path / "other"
    one.mkdir()
    other.mkdir()
    (one / "a").write_bytes(b"ab")
    (one / "b").write_bytes(b"c")
    (other / "a").write_bytes(b"a")
    (other / "b").write_bytes(b"bc")
    assert units.digest(one) != units.digest(other)
    first = units.digest(one)
    (one / "b").rename(one / "c")
    assert units.digest(one) != first
    (one / "c").rename(one / "b")
    assert units.digest(one) == first
    (one / "b").write_bytes(b"d")
    assert units.digest(one) != first
    big = tmp_path / "big"
    big.write_bytes(b"0123456789" * (units.CHUNK // 5))  # several chunks, the last partial
    assert units.digest(big) == hashlib.sha256(big.read_bytes()).hexdigest()


def test_units_are_written_from_cpu_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A unit's file is written from the copy the shared helper makes of it on the CPU
    # (shot.moved), with the values, the dtype and the strides of what was saved, a latent in
    # the encode's channel-major layout and windows as the DiT gives them, and reads back so. On
    # the CPU that copy is the tensor itself: the helper is replaced by one that copies, as it
    # does from a GPU, so that what the units write and give back is told from what they were
    # given.
    asked: list[tuple[torch.device, str]] = []

    def copied(tensor: Tensor, device: torch.device, what: str) -> Tensor:
        asked.append((device, what))
        return moved(tensor, device, what).clone()

    monkeypatch.setattr(units, "moved", copied)
    record = job(tmp_path, [Shot(0, 21)], [OutputSegment("a", 0, 21)])
    kept = Units(tmp_path, record)
    latent = channel_major(6)
    saved = [latent[:4] + 0, torch.randn(4, 3, 5, 16).to(torch.bfloat16)]
    assert saved[0].stride() == (15, 5, 1, 60) and saved[1].is_contiguous()
    copy = kept.save_latent(0, latent)
    assert copy is not latent
    for window, sampled in enumerate(saved):
        kept.save_window(0, window, sampled)
    cpu = torch.device("cpu")
    shot = tmp_path / STATE / "shot_000000"
    assert asked == [
        (cpu, str(shot / "latent.pt")),
        (cpu, str(shot / "window_0000.pt")),
        (cpu, str(shot / "window_0001.pt")),
    ]
    back = kept.latent(0)
    assert back is not None
    windows = kept.take_windows(0)
    for read, given in zip([copy, back, *windows], [latent, latent, *saved], strict=True):
        assert read is not given and read.device == cpu
        assert torch.equal(read, given) and read.dtype == given.dtype == torch.bfloat16
        assert read.stride() == given.stride()


def test_saved_latent_is_the_copy_given_back(tmp_path: Path) -> None:
    # save_latent gives the copy on the CPU that it wrote, for the run to sample the shot's
    # windows from: one copy of a latent in system memory, and no file read back.
    latent = channel_major(9)
    record = job(tmp_path, [Shot(0, 33)], [OutputSegment("a", 0, 33)])
    disk = Units(tmp_path, record)
    kept = disk.save_latent(0, latent)
    assert kept is latent
    back = disk.latent(0)
    assert back is not None and torch.equal(back, latent) and back.stride() == latent.stride()
    assert back.dtype == latent.dtype and back.device.type == "cpu"


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
