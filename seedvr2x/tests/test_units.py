"""The resumable units (DESIGN.md, Pause and resume), on the CPU: kept on disk with their values and
memory layout, the noise drawn again from a latent read back, windows resumed where they stopped,
the manifest recording each unit once its file is whole, and what a finished segment kept
removed."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from seedvr2x.media.source import Source
from seedvr2x.runtime import model
from seedvr2x.runtime.job import OutputSegment, Shot, parts_of
from seedvr2x.runtime.manifest import Manifest
from seedvr2x.runtime.shot import sample_windows, shot_layout
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
        stream=SimpleNamespace(frame_rate=Fraction(25), width=64, height=48),
        sample_aspect=Fraction(1),
        conversion=SimpleNamespace(describe=lambda: "RGB"),
    )
    source.path.write_bytes(b"")
    for segment in segments:
        (tmp_path / f"{segment.name}.mkv").write_bytes(b"x" * segment.frames)
    return Manifest(
        tmp_path / "manifest.json",
        {"seed": 42, "window": 5},
        {},
        parts_of([cast(Source, source)]),
        shots,
        [shot_layout(shot.frames, 5) for shot in shots],
        segments,
        [f"{segment.name}.mkv" for segment in segments],
        {},
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
