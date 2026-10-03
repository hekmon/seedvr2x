"""The command line's run on the CPU, the model replaced by a stand-in: the input read, the shots
and their seeds, the output written, its segments and its manifest, in every output format.

The stand-in's steps carry each shot's first frame (its seed less the job's) and frame count from
its encode through its windows to its decode, which writes, for each frame, the frame's index in
the job divided by 1000: so each output frame says where it comes from. The model's own output
is the GPU tests' (test_regression.py, test_shots.py)."""

import fcntl
import hashlib
import json
import logging
import math
import os
import platform
import re
import signal
import subprocess
import sys
import zlib
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from fractions import Fraction
from itertools import accumulate, pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import torch

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime.job import output_size
from seedvr2x.runtime.shot import decode_shot, padded_length


def _usable() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _usable(), reason="needs ffmpeg with zscale, scdet and ffv1")

SEED = 42
FRAMES = 9


def stand_in_encode(
    models: object,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
) -> torch.Tensor:
    """A latent (T', 1, 1, 16) in channel-major memory, as the VAE's, saying the shot's first frame
    (channel 0), its frame count (1) and the target's height and width (2, 3)."""
    assert read(count).shape[0] == count
    latent = torch.zeros(16, (padded_length(count) - 1) // 4 + 1, 1, 1)
    latent[0], latent[1] = seed - SEED, count
    latent[2], latent[3] = target
    return latent.permute(1, 2, 3, 0)


def stand_in_windows(
    models: object, latent: torch.Tensor, layout: list[tuple[int, int]], seed: int, start: int = 0
) -> Iterator[torch.Tensor]:
    for s, e in layout[start:]:
        yield latent[s:e] + 0


def stand_in_decode_stream(models: object, latent: torch.Tensor) -> Iterator[torch.Tensor]:
    """The VAE's decode of a stand-in latent (stand_in_encode), in its slices, the first two
    latents then one at a time: frames (3, t, H, W) at the target's size padded to multiples of
    16, as the encoder's input is (model.input_transform), in [-1, 1]: each saying its index in
    the job, divided by 1000, NaN in the padding, which the decode crops; NaN throughout for the
    frames of a latent whose channel 4 is NaN."""
    first, frames = int(latent[0, 0, 0, 0]), int(latent[0, 0, 0, 1])
    height, width = int(latent[0, 0, 0, 2]), int(latent[0, 0, 0, 3])
    padded = (-(-height // 16) * 16, -(-width // 16) * 16)
    # Every latent says so: the shot's own windows, none of another's.
    assert (latent[..., 0] == first).all() and (latent[..., 1] == frames).all()
    assert latent.shape[0] == (padded_length(frames) - 1) // 4 + 1
    latents = latent.shape[0]
    groups = [range(latents)] if latents <= 2 else [range(2), *([k] for k in range(2, latents))]
    for group in groups:
        planes = []
        for k in group:
            for frame in [0] if k == 0 else range(4 * k - 3, 4 * k + 1):
                value = math.nan if latent[k, 0, 0, 4].isnan() else (first + frame) / 500 - 1
                plane = torch.full((3, 1, *padded), math.nan)
                plane[:, :, :height, :width] = value
                planes.append(plane)
        yield torch.cat(planes, dim=1)


# The stand-in's models: the CPU, and the VAE's slicing, which the encode and lab's reference
# follow (model.encode_slices).
STAND_IN_MODELS = SimpleNamespace(
    attention="none",
    device=torch.device("cpu"),
    runner=SimpleNamespace(vae=SimpleNamespace(use_slicing=True, slicing_sample_min_size=4)),
)


def stand_in_model(patch: Callable[[Any, str, Any], None]) -> None:
    """Replace the model by the stand-in, through patch: monkeypatch.setattr, or setattr in a
    process of its own (test_stop.py). The decode around the VAE's is seedvr2x's own."""
    from seedvr2x.runtime import model, run

    patch(torch.cuda, "is_available", lambda: True)
    patch(torch.cuda, "is_bf16_supported", lambda: True)
    patch(torch.cuda, "get_device_name", lambda device: "a stand-in")
    patch(torch.backends.cudnn, "version", lambda: None)
    patch(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    patch(torch.cuda, "max_memory_allocated", lambda device: 0)
    patch(model, "load_models", lambda *a: STAND_IN_MODELS)
    patch(model, "nvidia_driver", lambda: "a stand-in")
    patch(model, "decode_stream", stand_in_decode_stream)
    patch(run, "encode_shot", stand_in_encode)
    patch(run, "sample_windows", stand_in_windows)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_model(monkeypatch.setattr)


def source(path: Path, frames: int = FRAMES, pattern: str = "testsrc2") -> Path:
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"{pattern}=s=64x48:r=25"),
            *("-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)),
        ],
        check=True,
    )
    return path


def upscale(tmp_path: Path, input_path: Path, output: str, *options: str) -> int:
    """seedvr2x run in this process, without colour correction unless options ask for one: the
    stand-in's frames then say which they are (stand_in_decode_stream)."""
    weights = tmp_path / "w.safetensors"
    if not weights.exists():
        weights.write_bytes(b"")
    correction = [] if "--color-correction" in options else ["--color-correction", "none"]
    return cli.main(
        [
            *(str(input_path), "-o", str(tmp_path / output)),
            *("--model-dir", str(tmp_path), "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", str(SEED)),
            *correction,
            *options,
        ]
    )


def indexes(path: Path, pix_fmt: str = "gbrp16le") -> list[int]:
    """Each frame's index in the job, as the stand-in wrote it: index / 1000, quantised."""
    data = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True,
        check=True,
    ).stdout
    if pix_fmt == "rgb48le":
        values = np.frombuffer(data, dtype="<u2").reshape(-1, 96 * 128 * 3)[:, 0]
        return [round(int(v) / 65535 * 1000) for v in values]
    values = np.frombuffer(data, dtype="<u2").reshape(-1, 3 * 96 * 128)[:, 0]
    return [round(int(v) / 65535 * 1000) for v in values]


@pytest.mark.usefixtures("stand_in")
def test_one_master(tmp_path: Path) -> None:
    (tmp_path / "cuts.txt").write_text("3\n4\n")
    dump = tmp_path / "dump"
    cuts = str(tmp_path / "cuts.txt")
    status = upscale(
        tmp_path, source(tmp_path / "in.mkv"), "one.mkv", "--cuts", cuts, "--dump-frames", str(dump)
    )
    assert status == 0
    assert indexes(tmp_path / "one.mkv") == list(range(FRAMES))
    assert sorted(p.name for p in dump.iterdir()) == [f"frame_{i:06d}.npy" for i in range(FRAMES)]
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.usefixtures("stand_in")
def test_segments_merged_with_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shots of 3, 1 and 5 frames; 0.12 s at 25 fps is 3 frames: the 1-frame shot merges into the
    # 3-frame one before it (a tie goes left), giving segments of 4 and 5 frames.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("3\n4\n")
    status = upscale(
        tmp_path, source(tmp_path / "in.mkv"), "out", "--cuts", "cuts.txt", "--min-segment", "0.12"
    )
    assert status == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    assert indexes(out / "seg_000000.mkv") == [0, 1, 2, 3]
    assert indexes(out / "seg_000001.mkv") == [4, 5, 6, 7, 8]
    content = json.loads((out / "manifest.json").read_text())
    assert [(s["start"], s["end"], s["seed"]) for s in content["shots"]] == [
        (0, 3, 42),
        (3, 4, 45),
        (4, 9, 46),
    ]
    assert [(s["name"], s["start"], s["end"], s["finished"]) for s in content["segments"]] == [
        ("seg_000000.mkv", 0, 4, True),
        ("seg_000001.mkv", 4, 9, True),
    ]
    assert content["settings"]["min_segment"] == "3/25"
    assert content["output"] == {"format": "gbrp16le", "size": [128, 96], "frame_rate": "25"}


@pytest.mark.usefixtures("stand_in")
def test_png_segments(tmp_path: Path) -> None:
    (tmp_path / "cuts.txt").write_text("4\n")
    status = upscale(
        tmp_path,
        source(tmp_path / "in.mkv"),
        "png",
        *("--cuts", str(tmp_path / "cuts.txt"), "--min-segment", "0", "--format", "png"),
    )
    assert status == 0
    for name, frames in (("seg_000000", range(4)), ("seg_000001", range(4, 9))):
        pngs = sorted((tmp_path / "png" / name).iterdir())
        assert [p.name for p in pngs] == [f"{i:06d}.png" for i in range(len(frames))]
        assert [indexes(p, "rgb48le")[0] for p in pngs] == list(frames)


@pytest.mark.usefixtures("stand_in")
def test_yuv_segments(tmp_path: Path) -> None:
    status = upscale(
        tmp_path,
        source(tmp_path / "in.mkv"),
        "yuv",
        "--min-segment",
        "0",
        "--format",
        "yuv420p10le",
    )
    assert status == 0
    master = tmp_path / "yuv" / "seg_000000.mkv"
    probe = subprocess.run(
        [
            *(
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-of",
                "csv=p=0",
            ),
            *("-show_entries", "stream=pix_fmt,color_space,nb_read_packets", str(master)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # 128x96 is an SD size: BT.601, tagged smpte170m since the source is untagged (DESIGN.md,
    # Colour and shape).
    assert probe.stdout.strip() == f"yuv420p10le,smpte170m,{FRAMES}"


@pytest.mark.usefixtures("stand_in")
def test_directory_mirrored(tmp_path: Path) -> None:
    split = tmp_path / "split"
    split.mkdir()
    source(split / "b.mkv", 2)
    source(split / "a.mkv", 3)
    assert upscale(tmp_path, split, "mirror") == 0
    out = tmp_path / "mirror"
    assert sorted(p.name for p in out.iterdir()) == ["a.mkv", "b.mkv", "manifest.json"]
    assert indexes(out / "a.mkv") == [0, 1, 2]  # a.mkv comes first, by name
    assert indexes(out / "b.mkv") == [3, 4]
    content = json.loads((out / "manifest.json").read_text())
    assert content["settings"]["min_segment"] is None
    assert [s["seed"] for s in content["shots"]] == [42, 45]


class Steps:
    """The stand-in's steps, each call recorded: "encode S", "window S:K", "decode S", S the
    shot's first frame; each encode's frames, summed up, by S. The call named `stop` raises
    KeyboardInterrupt from inside, after an encode's reads, before a window's output, after a
    decode's first frame, as a stop at once would. The call named `poison` makes NaN: in an
    encode's latent, a window's output, or the frames of a decode's last latent. The decode named
    `damage` flips a byte of the shot's input copy first (damage)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.read: dict[int, str] = {}
        self.stop: str | None = None
        self.poison: str | None = None
        self.damage: str | None = None

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name == self.stop:
            raise KeyboardInterrupt

    def encode(
        self,
        models: object,
        read: Callable[[int], npt.NDArray[np.float32]],
        count: int,
        target: tuple[int, int],
        seed: int,
    ) -> torch.Tensor:
        frames = read(count)
        self.read[seed - SEED] = hashlib.md5(frames.tobytes()).hexdigest()
        self._call(f"encode {seed - SEED}")
        latent = stand_in_encode(models, lambda n: frames[:n], count, target, seed)
        if self.poison == f"encode {seed - SEED}":
            latent[-1, 0, 0, 5] = math.nan
        return latent

    def windows(
        self,
        models: object,
        latent: torch.Tensor,
        layout: list[tuple[int, int]],
        seed: int,
        start: int = 0,
    ) -> Iterator[torch.Tensor]:
        for number, sampled in enumerate(stand_in_windows(models, latent, layout, seed, start)):
            name = f"window {seed - SEED}:{start + number}"
            self._call(name)
            if name == self.poison:
                sampled[-1, 0, 0, 5] = math.nan
            yield sampled

    def decode(
        self,
        models: object,
        merged: torch.Tensor,
        count: int,
        target: tuple[int, int],
        write: Callable[[npt.NDArray[np.float32]], None],
        *names: Any,
    ) -> None:
        """seedvr2x's decode of the shot, the VAE's a stand-in's (stand_in_decode_stream); the
        last latent's frames NaN when it is poisoned."""
        name = f"decode {int(merged[0, 0, 0, 0])}"
        self.calls.append(name)
        if name == self.poison:
            merged = merged.clone()
            merged[-1, 0, 0, 4] = math.nan
        if name == self.damage:
            damage(names[-1].copy, "flipped")

        def write_then_stop(frames: npt.NDArray[np.float32]) -> None:
            # The first frame apart: a stop comes after the decode's first frame, the rest of a
            # slice unwritten.
            write(frames[:1])
            if name == self.stop:
                raise KeyboardInterrupt
            if len(frames) > 1:
                write(frames[1:])

        decode_shot(models, merged, count, target, write_then_stop, *names)  # pyright: ignore[reportArgumentType]


@pytest.fixture
def steps(stand_in: None, monkeypatch: pytest.MonkeyPatch) -> Steps:
    from seedvr2x.runtime import run

    recorded = Steps()
    monkeypatch.setattr(run, "encode_shot", recorded.encode)
    monkeypatch.setattr(run, "sample_windows", recorded.windows)
    monkeypatch.setattr(run, "decode_shot", recorded.decode)
    return recorded


# Shots of 3, 1 and 21 frames (2, 1 and 6 latents; windows of 5: the last in two), segments of 4
# and 21 frames: (0, 3) and (4).
JOB = ("--cuts", "cuts.txt", "--window", "5", "--min-segment", "0.12")
UNINTERRUPTED = [
    *("encode 0", "window 0:0", "encode 3", "window 3:0", "decode 0", "decode 3"),
    *("encode 4", "window 4:0", "window 4:1", "decode 4"),
]


def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("3\n4\n")
    return source(tmp_path / "in.mkv", 25)


def test_units_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # In a directory, a segment's shots are encoded and sampled before its decode, so that its
    # decode and write is a unit of its own; one file decodes each shot as soon as it is sampled.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    assert indexes(out / "seg_000000.mkv") + indexes(out / "seg_000001.mkv") == list(range(25))
    content = json.loads((out / "manifest.json").read_text())
    assert [(s["latents"], s["windows"]) for s in content["shots"]] == [
        (2, [[0, 2]]),
        (1, [[0, 1]]),
        (6, [[0, 4], [2, 6]]),
    ]
    assert [(s["encoded"], s["windows_done"]) for s in content["shots"]] == [
        (True, 1),
        (True, 1),
        (True, 2),
    ]
    assert [s["bytes"] for s in content["segments"]] == [
        (out / name).stat().st_size for name in ("seg_000000.mkv", "seg_000001.mkv")
    ]
    assert content["settings"]["dit_model"] == {
        "name": "w.safetensors",
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert content["environment"]["gpu"] == "a stand-in"
    assert content["environment"]["python"].startswith("CPython 3.")
    assert "torchvision" in content["environment"]["packages"]
    [entry] = content["input"]
    assert entry["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert steps.calls == [
        *("encode 0", "window 0:0", "decode 0", "encode 3", "window 3:0", "decode 3"),
        *("encode 4", "window 4:0", "window 4:1", "decode 4"),
    ]
    assert indexes(tmp_path / "one.mkv") == list(range(25))


@pytest.mark.parametrize("name", ["resume", "manifest.json", "a.partial"])
def test_own_names_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str) -> None:
    # A mirrored PNG segment takes its file's stem: not one of seedvr2x's own names.
    split = tmp_path / "split"
    split.mkdir()
    source(split / f"{name}.mkv", 2)
    assert upscale(tmp_path, split, "out", "--format", "png") == 1
    assert f"{name}: a name seedvr2x keeps for itself" in caplog.text


def stopped(tmp_path: Path, input_path: Path, output: str, *options: str) -> None:
    """Run a job the stand-in stops at once (Steps.stop), as a second Ctrl-C would."""
    assert upscale(tmp_path, input_path, output, *options) == 130


def decoded(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-"],
        capture_output=True,
        check=True,
    ).stdout


@pytest.mark.parametrize(
    ("stop", "resumed"),
    [
        # Inside an encode: the shot before it kept, its own encode again.
        ("encode 3", UNINTERRUPTED[2:]),
        # Inside the first shot of the second segment: the first segment finished, its input
        # frames read again from the shot's first, the four before dropped.
        ("encode 4", UNINTERRUPTED[6:]),
        # Inside a window: the shot's latent and first window kept.
        ("window 4:1", UNINTERRUPTED[8:]),
        # Inside a segment's decode: its shots' windows kept, its decode and write again whole.
        ("decode 0", ["decode 0", "decode 3", *UNINTERRUPTED[6:]]),
        ("decode 4", UNINTERRUPTED[9:]),
    ],
)
@pytest.mark.parametrize("correction", ["none", "lab"])
def test_resumed_as_uninterrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    stop: str,
    resumed: list[str],
    correction: str,
) -> None:
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, "--color-correction", correction)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    read = dict(steps.read)
    steps.calls.clear()
    steps.stop = stop
    stopped(tmp_path, source_path, "out", *options)
    assert steps.calls[-1] == stop
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == resumed
    assert steps.read == read  # every encode read its own shot's frames
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    content = json.loads((out / "manifest.json").read_text())
    assert all(segment["finished"] for segment in content["segments"])
    # Finished already: nothing to do, the models not even loaded.
    from seedvr2x.runtime import model

    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("models loaded"))
    steps.calls.clear()
    (out / "resume" / "shot_000004").mkdir(parents=True)  # as a stop could leave them
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == []
    assert not (out / "resume").exists()


@pytest.mark.parametrize(
    ("poison", "said", "kept", "files", "resumed"),
    [
        # In the encode's latent: the encode made again.
        ("encode 4", "shot 3/3's encode", (False, 0, False), None, UNINTERRUPTED[6:]),
        # In a window's output: the windows before it kept.
        (
            "window 4:1",
            "shot 3/3's window 2/2",
            (True, 1, False),
            ["latent.pt", "window_0000.pt"],
            UNINTERRUPTED[8:],
        ),
        # In a slice of a decode: its segment unfinished, decoded again whole.
        (
            "decode 4",
            "shot 3/3's decode, frames 21 to 24",
            (True, 2, False),
            ["window_0000.pt", "window_0001.pt"],
            UNINTERRUPTED[9:],
        ),
    ],
)
@pytest.mark.parametrize("correction", ["none", "lab"])
def test_non_finite_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    poison: str,
    said: str,
    kept: tuple[bool, int, bool],
    files: list[str] | None,
    resumed: list[str],
    correction: str,
) -> None:
    # NaN or inf in what a unit made stops the run there, naming the unit, which isn't recorded:
    # a resume makes it again, as an uninterrupted run does. With lab, the shot's input copy and
    # its checksums, whole before its latent is checked: discarded unrecorded, else kept.
    options = (*JOB, "--color-correction", correction)
    if correction == "lab":
        files = sorted([*(files or []), "input.crc32", "input.mkv"])
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    steps.poison = poison
    assert upscale(tmp_path, source_path, "out", *options) == 1
    assert f"{said}: " in caplog.text
    assert "NaN or inf, so not recorded; stopped: 1 of 2 segments finished" in caplog.text
    out = tmp_path / "out"
    content = json.loads((out / "manifest.json").read_text())
    shot = content["shots"][2]
    assert (shot["encoded"], shot["windows_done"], content["segments"][1]["finished"]) == kept
    shot_directory = out / "resume" / "shot_000004"
    found = sorted(p.name for p in shot_directory.iterdir()) if shot_directory.exists() else None
    assert found == files
    assert not (out / "seg_000001.mkv").exists()
    steps.calls.clear()
    steps.poison = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == resumed
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def test_non_finite_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # One file keeps nothing: stopped, the master's partial file removed.
    source_path = job(tmp_path, monkeypatch)
    steps.poison = "window 3:0"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 1
    assert "shot 2/3's window 1/1: " in caplog.text and "nothing kept" in caplog.text
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("one")]


LAB = ("--color-correction", "lab")


def test_lab_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # lab unless asked otherwise: a setting, which a resume compares.
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    common = ["--model-dir", str(tmp_path), "--dit-model", "w.safetensors"]
    common += ["--vae-model", "w.safetensors", "--resolution", "96", *JOB]
    assert cli.main([str(source_path), "-o", str(tmp_path / "out"), *common]) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["color_correction"] == "lab"
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--color-correction", "none") == 1
    assert 'settings.color_correction: "lab" -> "none"' in caplog.text and steps.calls == []


def test_lab_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # The frames each encode reads, copied as they come, exactly, and recorded with its latent.
    from seedvr2x.media.decode import Decoder, to_float32
    from seedvr2x.media.ffmpeg import input_args
    from seedvr2x.runtime.shot import COPY_READ

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    from seedvr2x.media.checksums import read_checksums

    shot = tmp_path / "out" / "resume" / "shot_000004"
    names = ["input.crc32", "input.mkv", "latent.pt", "window_0000.pt"]
    assert sorted(p.name for p in shot.iterdir()) == names
    with Decoder(input_args(shot / "input.mkv"), COPY_READ, 64, 48, 21) as decoder:
        planes = decoder.read(21)
    assert hashlib.md5(to_float32(planes).tobytes()).hexdigest() == steps.read[4]
    # Its checksums: of each frame's gbrp16le planes, G, B and R, as written.
    gbr = np.ascontiguousarray(planes.transpose(0, 3, 1, 2)[:, [1, 2, 0]])
    assert read_checksums(shot / "input.crc32") == [zlib.crc32(frame) for frame in gbr]
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["shots"][2]["encoded"]


def test_lab_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # What a kill leaves of lab's is discarded: a decode's buffer, a copy being written. A copy
    # recorded is kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    out = tmp_path / "out"
    shot = out / "resume" / "shot_000004"
    for name in ("decoded.bf16", "input.mkv.partial", "input.crc32.partial"):
        (shot / name).write_bytes(b"left")
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert "3 leftovers discarded" in caplog.text and "made again" not in caplog.text
    assert steps.calls == ["window 4:1", "decode 4"]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    assert not (out / "resume").exists()


def copied(path: Path, count: int) -> str:
    """The frames of a shot's input copy, summed up as an encode's frames are (Steps.read)."""
    from seedvr2x.media.decode import Decoder, to_float32
    from seedvr2x.media.ffmpeg import input_args
    from seedvr2x.runtime.shot import COPY_READ

    with Decoder(input_args(path), COPY_READ, 64, 48, count) as decoder:
        return hashlib.md5(to_float32(decoder.read(count)).tobytes()).hexdigest()


@pytest.mark.parametrize("missing", ["input.mkv", "input.crc32"])
@pytest.mark.parametrize(
    ("stop", "resumed"),
    [
        # Its latent kept: the copy made again, then its windows left.
        ("window 4:1", ["window 4:1", "decode 4"]),
        # Its windows all done: the copy made again before the segment's decode.
        ("decode 4", ["decode 4"]),
    ],
)
def test_lab_copy_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    stop: str,
    resumed: list[str],
    missing: str,
) -> None:
    # A copy recorded but missing, or its checksums, isn't refused: it is derived data, made
    # again from the input, the frames its encode read, the shot's latent and windows kept; the
    # output is then an uninterrupted run's.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    steps.stop = stop
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    out = tmp_path / "out"
    (out / "resume" / "shot_000004" / missing).unlink()
    seen: list[str] = []
    decode = steps.decode

    def looked(*arguments: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert steps.calls == resumed
    assert f"shot 3/3: {missing} missing: its input copy made again from the input" in caplog.text
    assert seen == [steps.read[4]]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def damage(path: Path, how: str) -> None:
    """Damage a shot's input copy or its checksums (Steps.damage, test_lab_copy_damaged): a byte
    flipped in the middle of the copy's third frame, so a slice of it fails its CRC; the copy cut
    short at half its bytes; the copy made again whole with a pixel of frame 7 altered, as ffmpeg
    decodes a damaged slice size without a word; a digit of the first checksum changed; the
    checksums cut short, or short of their last line."""
    from seedvr2x.media.decode import Decoder, to_float32
    from seedvr2x.media.ffmpeg import input_args
    from seedvr2x.media.writer import FFV1Writer, Tags
    from seedvr2x.runtime.shot import COPY_READ

    data = bytearray(path.read_bytes())
    if how in ("cut short", "checksums cut short", "checksums short a line"):
        kept = {
            "cut short": len(data) // 2,
            "checksums cut short": -4,
            "checksums short a line": -9,
        }
        path.write_bytes(data[: kept[how]])
        return
    if how == "checksums altered":
        data[0] = ord("1") if data[0] == ord("0") else ord("0")
        path.write_bytes(data)
        return
    if how == "altered":
        with Decoder(input_args(path), COPY_READ, 64, 48) as decoder:
            frames = decoder.read(1000)
        frames[7, 0, 0, 0] ^= 1
        with FFV1Writer(path, "gbrp16le", 64, 48, Fraction(25), Tags()) as writer:
            writer.write(to_float32(frames))
        return
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pos,size", "-of", "json", str(path)),
        ],
        capture_output=True,
        check=True,
    ).stdout
    packet = json.loads(probed)["packets"][2]
    data[int(packet["pos"]) + int(packet["size"]) // 2] ^= 0xFF
    path.write_bytes(data)


@pytest.mark.parametrize(
    ("how", "said"),
    [
        # Reported by ffmpeg: a slice failing its CRC, the file cut short.
        ("flipped", "slice CRC mismatch"),
        ("cut short", "fails a strict read|it ended after"),
        # Decoded by ffmpeg without a word: caught by the frame's checksum.
        ("altered", r"frame 7: CRC-32 [0-9a-f]{8}, where [0-9a-f]{8} was written"),
        # The checksums damaged, so the copy can't be trusted either.
        ("checksums altered", r"frame 0: CRC-32 [0-9a-f]{8}, where [0-9a-f]{8} was written"),
        ("checksums cut short", r"input\.crc32: cut short, its last line unfinished"),
        ("checksums short a line", r"input\.crc32: 20 checksums, for 21 frames"),
    ],
)
def test_lab_copy_damaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
    said: str,
) -> None:
    # A copy damaged stops the run when its decode reads it, before anything is corrected against
    # it: the copy is removed, named, and the next run makes it again from the input, the shot's
    # latent and windows kept; the output is then an uninterrupted run's.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    out = tmp_path / "out"
    copy = out / "resume" / "shot_000004" / "input.mkv"
    checksums = copy.with_name("input.crc32")
    damage(checksums if how.startswith("checksums") else copy, how)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 1
    assert steps.calls == ["window 4:1", "decode 4"]
    assert f"{copy}: the shot's input copy, not readable whole and intact: " in caplog.text
    assert re.search(said, caplog.text)
    assert (
        "; removed, made again from the input on resuming; stopped: 1 of 2 segments finished"
        in caplog.text
    )
    assert not copy.exists() and not checksums.exists()
    assert not (out / "seg_000001.mkv").exists()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert steps.calls == ["decode 4"]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def test_lab_copy_damaged_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # One file keeps nothing: stopped, the copy named, the work directory and the partial master
    # removed.
    source_path = job(tmp_path, monkeypatch)
    steps.damage = "decode 4"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB) == 1
    copy = tmp_path / "one.mkv.work" / "shot_000002" / "input.mkv"
    assert f"{copy}: the shot's input copy, not readable whole and intact: " in caplog.text
    assert "slice CRC mismatch" in caplog.text and "on resuming" not in caplog.text
    assert "; stopped: nothing kept" in caplog.text
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("one")]


def test_lab_copy_checked_before_its_last_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A shot's last frames are written once its copy is read whole and checked: they may finish
    # the segment, which is then recorded, so a copy failing at the very end of the second pass
    # leaves the segment unfinished, decoded again whole by the next run.
    from seedvr2x.media.decode import Decoder

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    finish = Decoder.finish
    finished: list[Decoder] = []

    def failing(self: Decoder) -> None:
        if self._strict:  # pyright: ignore[reportPrivateUsage]
            finished.append(self)
            if len(finished) == 4:  # shot 2/3's second pass, the last of segment 1
                raise self.failure("a stand-in failure at the end")
        finish(self)

    monkeypatch.setattr(Decoder, "finish", failing)
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 1
    assert "a stand-in failure at the end; removed" in caplog.text
    assert "stopped: 0 of 2 segments finished" in caplog.text
    out = tmp_path / "out"
    content = json.loads((out / "manifest.json").read_text())
    assert [segment["finished"] for segment in content["segments"]] == [False, False]
    assert not (out / "seg_000000.mkv").exists()
    assert not (out / "resume" / "shot_000003" / "input.mkv").exists()
    monkeypatch.setattr(Decoder, "finish", finish)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert steps.calls == ["decode 0", "decode 3", *UNINTERRUPTED[6:]]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


@pytest.mark.parametrize("presses", [1, 2])
def test_lab_copy_again_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    presses: int,
) -> None:
    # Making a copy again is a unit: Ctrl-C once lets it finish, the run stopping before the
    # next unit; twice stops the run at once, the copy unfinished, so missing. Either way, the
    # next run goes on from there.
    from seedvr2x.runtime import run

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    copy = tmp_path / "out" / "resume" / "shot_000004" / "input.mkv"
    copy.unlink()
    reader = run.Inputs.reader

    def pressing(self: run.Inputs, shot: Any) -> Callable[[int], npt.NDArray[np.float32]]:
        read = reader(self, shot)
        pressed: list[bool] = []

        def read_then_press(count: int) -> npt.NDArray[np.float32]:
            frames = read(count)
            if not pressed:  # as a terminal's Ctrl-C would, during the copy's first frames
                pressed.append(True)
                for _ in range(presses):
                    os.kill(os.getpid(), signal.SIGINT)
            return frames

        return read_then_press

    monkeypatch.setattr(run.Inputs, "reader", pressing)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 130
    assert steps.calls == []
    if presses == 1:
        assert "stopped after shot 3/3's input copy, as asked" in caplog.text
        assert copy.exists()
    else:
        assert "stopped at once (Ctrl-C)" in caplog.text
        assert not copy.exists()
    monkeypatch.setattr(run.Inputs, "reader", reader)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    assert ("made again from the input" in caplog.text) == (presses == 2)
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        out = tmp_path / "out" / name
        assert decoded(out) == decoded(tmp_path / "whole" / name)


def test_lab_copy_again_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A copy made again replaces checksums written before, which go first: a stop between the
    # copy and its checksums leaves the copy without any, so made again by the next run, never
    # beside checksums that aren't its own.
    from seedvr2x.runtime import run

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *LAB) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LAB)
    copy = tmp_path / "out" / "resume" / "shot_000004" / "input.mkv"
    checksums = copy.with_name("input.crc32")
    copy.unlink()
    written = run.write_checksums

    def killed(path: Path, values: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "write_checksums", killed)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 130
    assert steps.calls == [] and copy.exists() and not checksums.exists()
    monkeypatch.setattr(run, "write_checksums", written)
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    assert "shot 3/3: input.crc32 missing: its input copy made again" in caplog.text
    assert steps.calls == ["window 4:1", "decode 4"]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        out = tmp_path / "out" / name
        assert decoded(out) == decoded(tmp_path / "whole" / name)


def test_lab_copy_again_from_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A directory's input: the copy made again from its own input file, from its first frame,
    # those of the parts before it never decoded.
    from seedvr2x.media.source import Source

    split = tmp_path / "split"
    split.mkdir()
    for name, frames in (("a.mkv", 3), ("b.mkv", 2), ("c.mkv", 4)):
        source(split / name, frames)
    assert upscale(tmp_path, split, "whole", *LAB) == 0
    steps.stop = "decode 5"
    stopped(tmp_path, split, "out", *LAB)
    copy = tmp_path / "out" / "resume" / "shot_000005" / "input.mkv"
    frames = copied(copy, 4)
    copy.unlink()
    opened: list[str] = []
    seen: list[str] = []
    decoder = Source.decoder
    decode = steps.decode

    def recorded(self: Source) -> object:
        opened.append(self.path.name)
        return decoder(self)

    def looked(*arguments: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments)

    from seedvr2x.runtime import run

    monkeypatch.setattr(Source, "decoder", recorded)
    monkeypatch.setattr(run, "decode_shot", looked)
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, split, "out", *LAB) == 0
    assert steps.calls == ["decode 5"] and opened == ["c.mkv"] and seen == [frames]
    for name in ("a.mkv", "b.mkv", "c.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "whole" / name)


def test_lab_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # One file's copies and buffers go in a work directory beside it, removed at the end, at a stop
    # or an error. A killed run's is removed; anything else there is refused, and kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    one, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    assert upscale(tmp_path, source_path, "out", *JOB, *LAB) == 0
    seen: list[list[str]] = []
    decode = steps.decode

    def looked(*arguments: Any) -> None:
        seen.append(sorted(path.name for path in work.iterdir()))
        decode(*arguments)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB) == 0
    assert seen == [["shot_000000"], ["shot_000001"], ["shot_000002"]]  # one shot at a time
    assert not work.exists()
    out = tmp_path / "out"
    assert decoded(one) == decoded(out / "seg_000000.mkv") + decoded(out / "seg_000001.mkv")
    for stop, poison, status in (("decode 4", None, 130), (None, "decode 4", 1)):
        one.unlink(missing_ok=True)
        steps.stop, steps.poison = stop, poison
        assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB) == status
        assert not work.exists() and not one.exists()
    steps.stop = steps.poison = None
    (work / "shot_000002").mkdir(parents=True)
    for name in ("input.mkv.partial", "input.crc32.partial"):
        (work / "shot_000002" / name).write_bytes(b"left")
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB) == 0
    assert "left by a run that was killed, emptied" in caplog.text and not work.exists()
    work.mkdir()
    (work / "notes.txt").write_text("mine")
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB) == 1
    assert "not what a run of seedvr2x leaves" in caplog.text
    assert (work / "notes.txt").read_text() == "mine"


def test_lab_as_one_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # The two passes give each shot exactly what lab gives its frames in the decode's slices,
    # histograms pooled over the shot: the second maps the values the first counted, and the
    # reference rebuilt from the copy is the encoder's input, its padding cropped as the decode's
    # (target 100 x 133, decoded 112 x 144, written 100 x 132).
    from seedvr2x.media.decode import to_float32
    from seedvr2x.media.source import examine
    from seedvr2x.runtime import colour, model
    from seedvr2x.runtime.job import target_size
    from seedvr2x.runtime.shot import encoder_inputs

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LAB, "--resolution", "100") == 0
    height, width = 100, 132
    planes = np.frombuffer(decoded(tmp_path / "one.mkv"), dtype="<u2")
    written = planes.reshape(25, 3, height, width)[:, [2, 0, 1]]  # G, B, R to R, G, B
    with examine(source_path).decoder() as decoder:
        frames = to_float32(decoder.read(25))
    models = cast(model.Models, STAND_IN_MODELS)
    target = target_size(64, 48, Fraction(1), 100)
    assert target == (100, 133) and output_size(target) == (height, width)
    expected: list[torch.Tensor] = []
    for start, end in ((0, 3), (3, 4), (4, 25)):
        count = end - start
        shot = iter(frames[start:end])

        def read(n: int) -> npt.NDArray[np.float32]:
            return np.stack([next(shot) for _ in range(n)])  # noqa: B023

        reference = torch.cat(list(encoder_inputs(models, read, count, target)), dim=1)
        assert reference.shape[2:] == (112, 144)  # padded to multiples of 16
        reference = reference[:, :count, :height, :width].permute(1, 0, 2, 3)
        values = torch.tensor([(start + frame) / 500 - 1 for frame in range(count)])
        content = values.view(-1, 1, 1, 1).expand(-1, 3, height, width)
        # The decode's slices: 5 frames, then 4, the last cut to the shot's end.
        sizes = [min(5, count), *(min(4, count - frame) for frame in range(5, count, 4))]
        bounds = accumulate(sizes, initial=0)
        slices = [(content[a:b], reference[a:b]) for a, b in pairwise(bounds)]
        histograms = colour.Histograms(models.device)
        for decoded_slice, matched in slices:
            moved = colour.transfer(decoded_slice, matched)
            histograms.add(
                colour.rgb_to_lab(colour.unit_range(moved)),
                colour.rgb_to_lab(colour.unit_range(matched)),
            )
        for decoded_slice, matched in slices:
            moved = colour.transfer(decoded_slice, matched)
            lab = colour.rgb_to_lab(colour.unit_range(moved))
            expected.append(colour.lab_to_rgb(histograms.match(lab)))
    codes = (torch.cat(expected) * 65535).round().numpy()
    assert np.array_equal(codes, written.astype(np.float32))


def test_leftovers_discarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # What a kill leaves, which the manifest doesn't name, is discarded on resume.
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    out = tmp_path / "out"
    state = out / "resume"
    assert sorted(str(p.relative_to(state)) for p in state.rglob("*")) == [
        "shot_000004",
        "shot_000004/latent.pt",
        "shot_000004/window_0000.pt",
    ]
    planted = [
        out / "seg_000001.mkv",  # an unfinished segment's file
        out / "seg_000001.mkv.partial",
        out / "manifest.json.partial",
        state / "shot_000004" / "window_0001.pt.partial",
        state / "shot_000004" / "window_0001.pt",  # not recorded
        state / "shot_000000" / "window_0000.pt",  # a finished segment's
    ]
    for path in planted:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"left")
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    assert not state.exists() and not list(out.rglob("*.partial"))
    assert indexes(out / "seg_000001.mkv") == list(range(4, 25))


def contents(directory: Path) -> dict[str, bytes | None]:
    """Every path under directory, with a file's bytes."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }


def refused(
    tmp_path: Path, source_path: Path, caplog: pytest.LogCaptureFixture, *options: str
) -> str:
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 1
    return caplog.text


def test_another_job_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    out = tmp_path / "out"
    kept = contents(out)
    # Other settings, even with an environment change accepted.
    text = refused(tmp_path, source_path, caplog, *JOB[:3], "6", *JOB[4:])
    assert "another job than the one asked" in text
    assert "settings.window: 5 -> 6" in text
    assert "--accept-env-change" not in text
    other = (*JOB[:3], "6", *JOB[4:], "--accept-env-change")
    assert "settings.window: 5 -> 6" in refused(tmp_path, source_path, caplog, *other)
    # Another model, by its hash.
    (tmp_path / "w.safetensors").write_bytes(b"other")
    weights = tmp_path / "w.safetensors"
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert "settings.dit_model.sha256" in text and "settings.dit_model.size: 0 -> 5" in text
    weights.write_bytes(b"")
    # Another input, by its content, even with an environment change accepted.
    content = source_path.read_bytes()
    source_path.write_bytes(source(tmp_path / "other.mkv", 25, "testsrc").read_bytes())
    text = refused(tmp_path, source_path, caplog, *JOB, "--accept-env-change")
    assert "input[0].sha256" in text
    source_path.write_bytes(content)
    # Other code.
    from seedvr2x.runtime import manifest

    with monkeypatch.context() as patch:
        patch.setattr(manifest, "code_sha256", lambda: "0" * 64)
        assert "settings.code" in refused(tmp_path, source_path, caplog, *JOB)
    # Another GPU, which the user may accept.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "get_device_name", lambda device: "another")
        text = refused(tmp_path, source_path, caplog, *JOB)
    assert 'environment.gpu: "a stand-in" -> "another"' in text
    assert "Only its environment differs: --accept-env-change resumes it anyway" in text
    # And other settings: refused, even with the change accepted.
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "get_device_name", lambda device: "another")
        text = refused(tmp_path, source_path, caplog, *other)
    assert "settings.window: 5 -> 6" in text and "environment.gpu" in text
    assert "Only its environment differs" not in text
    # Another version of a package the run imports, or of Python.
    from seedvr2x.runtime import environment

    versions = environment.versions
    installed = versions()["diffusers"]
    with monkeypatch.context() as patch:
        patch.setattr(environment, "versions", lambda: {**versions(), "diffusers": "0.0.1"})
        text = refused(tmp_path, source_path, caplog, *JOB)
    assert f'environment.packages.diffusers: "{installed}" -> "0.0.1"' in text
    assert "Only its environment differs" in text
    with monkeypatch.context() as patch:
        patch.setattr(platform, "python_version", lambda: "3.99.0")
        assert "environment.python" in refused(tmp_path, source_path, caplog, *JOB)
    # Other conversions: zimg upgraded, which ffmpeg's version doesn't say.
    from seedvr2x.media import fingerprint

    with monkeypatch.context() as patch:
        patch.setattr(fingerprint, "fingerprint", lambda: "0" * 64)
        assert "environment.conversions" in refused(tmp_path, source_path, caplog, *JOB)
    # Another's files, never deleted.
    (out / "notes.txt").write_text("mine")
    assert "not this job's: notes.txt" in refused(tmp_path, source_path, caplog, *JOB)
    (out / "notes.txt").unlink()
    (out / "resume" / "shot_000004" / "notes.txt").write_text("mine")
    assert "shot_000004/notes.txt: not this job's" in refused(tmp_path, source_path, caplog, *JOB)
    (out / "resume" / "shot_000004" / "notes.txt").unlink()
    # Missing what the manifest names.
    (out / "seg_000000.mkv").rename(tmp_path / "moved.mkv")
    assert "finished, the manifest says, but missing" in refused(
        tmp_path, source_path, caplog, *JOB
    )
    (tmp_path / "moved.mkv").rename(out / "seg_000000.mkv")
    window = out / "resume" / "shot_000004" / "window_0000.pt"
    window.rename(tmp_path / "window.pt")
    assert "kept, the manifest says, but missing" in refused(tmp_path, source_path, caplog, *JOB)
    (tmp_path / "window.pt").rename(window)
    # Another manifest version.
    written = (out / "manifest.json").read_bytes()
    (out / "manifest.json").write_text(json.dumps({**json.loads(written), "seedvr2x_manifest": 1}))
    assert "manifest version 1" in refused(tmp_path, source_path, caplog, *JOB)
    (out / "manifest.json").write_bytes(written)
    assert contents(out) == kept  # nothing touched
    # The job asked, at last.
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]


def test_environment_change_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Resumed on another GPU, as the user accepts: the manifest records the change, and takes the
    # new environment, from the next unit made on.
    from seedvr2x.runtime import model

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    recorded = manifest.read_bytes()
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
    monkeypatch.setattr(model, "nvidia_driver", lambda: "another driver")
    # Stopped again before any unit: the record as it was.
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert manifest.read_bytes() == recorded
    # Stopped after one.
    steps.calls.clear()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert steps.calls == ["window 4:1", "decode 4"]
    content = json.loads(manifest.read_text())
    assert content["environment"]["gpu"] == "another"
    [change] = content["environment_changes"]
    assert datetime.fromisoformat(change.pop("accepted")).tzinfo is not None
    assert change == {
        # The driver too, for information.
        "before": {"driver": "a stand-in", "gpu": "a stand-in"},
        "after": {"driver": "another driver", "gpu": "another"},
        # Stopped in the last shot's second window: the first segment finished, the three
        # shots encoded, a window each kept.
        "segments_finished": 1,
        "shots_encoded": 3,
        "windows_done": 3,
    }
    # A third GPU: a second change, after the first.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "a third")
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == 0
    assert steps.calls == ["decode 4"]
    assert indexes(tmp_path / "out" / "seg_000001.mkv") == list(range(4, 25))
    first, second = json.loads(manifest.read_text())["environment_changes"]
    assert first["after"]["gpu"] == "another"
    assert (second["before"], second["after"]) == ({"gpu": "another"}, {"gpu": "a third"})
    assert second["windows_done"] == 4
    # The job is the third GPU's now.
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "a stand-in")
    assert "environment.gpu" in refused(tmp_path, source_path, caplog, *JOB)


def test_input_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # An input is its content: moved and touched, it is the same input, and the manifest says
    # where it is now.
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    moved = source_path.rename(tmp_path / "moved.mkv")
    status = moved.stat()
    os.utime(moved, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    steps.calls.clear()
    steps.stop = None
    caplog.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    assert upscale(tmp_path, moved, "out", *JOB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    assert f"recorded at {source_path.resolve()}, moved: the same content" in caplog.text
    [entry] = json.loads((tmp_path / "out" / "manifest.json").read_text())["input"]
    assert entry["path"] == str(moved.resolve())
    assert entry["modified_ns"] == moved.stat().st_mtime_ns
    assert entry["sha256"] == hashlib.sha256(moved.read_bytes()).hexdigest()


def test_first_pass_not_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A resume trusts the record of the first pass while nothing it depends on changed, ffmpeg
    # and its conversions, and refuses another job before it.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path) -> object:
        scans.append(path)
        return scan(path)

    monkeypatch.setattr(examined, "scan", counted)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    assert len(scans) == 1
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    assert steps.calls[-2:] == ["window 4:0", "window 4:1"]
    refused(tmp_path, source_path, caplog, *JOB[:3], "6", *JOB[4:])
    assert len(scans) == 1
    # Another GPU, accepted: the pass takes no part of it.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
    steps.calls.clear()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert len(scans) == 1 and steps.calls == ["window 4:1", "decode 4"]
    # Another ffmpeg, accepted: the pass runs again.
    check = ffmpeg.check
    monkeypatch.setattr(ffmpeg, "check", lambda *options: "n0.0-another")
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert len(scans) == 2
    # Other conversions alone, the same ffmpeg: that stop made no unit, so the record
    # still has the first.
    from seedvr2x.media import fingerprint

    monkeypatch.setattr(ffmpeg, "check", check)
    monkeypatch.setattr(fingerprint, "fingerprint", lambda: "0" * 64)
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == 0
    assert len(scans) == 3 and steps.calls == ["decode 4"]


def test_probe_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # What the source declares is the job's too: a probe saying otherwise, as another ffmpeg
    # might, once its change is accepted, is another job.
    from seedvr2x.media import source as examined

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    recorded = manifest.read_bytes()
    assert json.loads(recorded)["input"][0]["primaries"] == ""  # untagged
    probe = examined.probe
    monkeypatch.setattr(
        examined, "probe", lambda path: replace(probe(path), color_primaries="bt709")
    )
    assert 'input[0].primaries: "" -> "bt709"' in refused(tmp_path, source_path, caplog, *JOB)
    assert manifest.read_bytes() == recorded


def test_first_pass_compared_after_a_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # After an environment change, accepted, the first pass runs again, and must find what it
    # found.
    from seedvr2x.media import source as examined

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    recorded = manifest.read_bytes()
    scan = examined.scan
    monkeypatch.setattr(examined, "scan", lambda path: replace(scan(path), frames=29))
    monkeypatch.setattr(ffmpeg, "check", lambda *options: "n0.0-another")
    text = refused(tmp_path, source_path, caplog, *JOB, "--accept-env-change")
    assert "input[0].frames: 25 -> 29" in text
    assert manifest.read_bytes() == recorded


def test_package_change_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A package's version changed, as accepted: of the packages, the record names that one.
    from seedvr2x.runtime import environment

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    versions = environment.versions
    installed = versions()["diffusers"]
    monkeypatch.setattr(environment, "versions", lambda: {**versions(), "diffusers": "0.0.1"})
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == 0
    [change] = json.loads((tmp_path / "out" / "manifest.json").read_text())["environment_changes"]
    assert change["before"] == {"packages": {"diffusers": installed}}
    assert change["after"] == {"packages": {"diffusers": "0.0.1"}}


def test_unrecorded_import_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A distribution of the run's imported after its versions are recorded is one a resume
    # wouldn't see change: said, as a bug of seedvr2x's. A profiler's isn't the run's.
    from seedvr2x.runtime import environment, model

    providers = environment._providers()  # pyright: ignore[reportPrivateUsage]
    declared = environment._declared()  # pyright: ignore[reportPrivateUsage]

    def load(*arguments: object) -> object:
        late = {"late_module": ["imported-late"], "profiler_module": ["a-profiler"]}
        monkeypatch.setattr(environment, "_providers", lambda: {**providers, **late})
        monkeypatch.setattr(environment, "_declared", lambda: declared | {"imported-late"})
        for module in late:
            monkeypatch.setitem(sys.modules, module, ModuleType(module))
        return SimpleNamespace(attention="none", device=torch.device("cpu"))

    monkeypatch.setattr(model, "load_models", load)
    assert upscale(tmp_path, source(tmp_path / "in.mkv"), "out") == 0
    assert "not in the manifest's environment, a seedvr2x bug: ['imported-late']" in caplog.text


def test_tools_not_said(tmp_path: Path, steps: Steps, caplog: pytest.LogCaptureFixture) -> None:
    # The process holds pytest and pygments beside the run, recorded by none of its versions:
    # the end of the run says nothing of them.
    pytest.importorskip("pygments")
    assert upscale(tmp_path, source(tmp_path / "in.mkv"), "out") == 0
    assert "not in the manifest's environment" not in caplog.text


def test_driver_recorded_not_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The NVIDIA driver is information: the math kernels ship with torch.
    from seedvr2x.runtime import model

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    assert json.loads(manifest.read_text())["environment"]["driver"] == "a stand-in"
    monkeypatch.setattr(model, "nvidia_driver", lambda: "another")
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    assert json.loads(manifest.read_text())["environment"]["driver"] == "another"


def test_directory_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # The segments finished aren't decoded again; the one resumed is read from its first frame.
    from seedvr2x.media.source import Source

    split = tmp_path / "split"
    split.mkdir()
    for name, frames in (("a.mkv", 3), ("b.mkv", 2), ("c.mkv", 4)):
        source(split / name, frames)
    assert upscale(tmp_path, split, "whole", "--format", "png") == 0
    read = dict(steps.read)
    steps.stop = "decode 5"
    stopped(tmp_path, split, "out", "--format", "png")
    opened: list[str] = []
    decoder = Source.decoder

    def recorded(self: Source) -> object:
        opened.append(self.path.name)
        return decoder(self)

    monkeypatch.setattr(Source, "decoder", recorded)
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, split, "out", "--format", "png") == 0
    assert steps.calls == ["decode 5"] and opened == []
    steps.stop = "encode 5"
    stopped(tmp_path, split, "again", "--format", "png")
    opened.clear()
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, split, "again", "--format", "png") == 0
    assert steps.calls == ["encode 5", "window 5:0", "decode 5"] and opened == ["c.mkv"]
    assert steps.read == read
    for out in ("out", "again"):
        for name in ("a", "b", "c"):
            pngs = sorted((tmp_path / out / name).iterdir())
            assert [p.read_bytes() for p in pngs] == [
                p.read_bytes() for p in sorted((tmp_path / "whole" / name).iterdir())
            ]


def test_directory_moved(tmp_path: Path, steps: Steps) -> None:
    # The segments of a directory moved elsewhere are the same input.
    split = tmp_path / "split"
    split.mkdir()
    for name, frames in (("a.mkv", 3), ("b.mkv", 2)):
        source(split / name, frames)
    steps.stop = "decode 3"
    stopped(tmp_path, split, "out", "--format", "png")
    moved = split.rename(tmp_path / "moved")
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, moved, "out", "--format", "png") == 0
    assert steps.calls == ["decode 3"]


def test_one_writer_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    descriptor = os.open(out, os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX)  # another seedvr2x's
    try:
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
        assert "another seedvr2x is writing it" in caplog.text
    finally:
        os.close(descriptor)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert upscale(tmp_path, source_path, "out", *JOB) == 0  # the lock went with the run
    # A job resumed: locked before its record is read.
    descriptor = os.open(out, os.O_RDONLY)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        caplog.clear()
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
        assert "another seedvr2x is writing it" in caplog.text
    finally:
        os.close(descriptor)


def test_first_manifest_write_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # What a stop during the first manifest write leaves is no job to resume, and no obstacle.
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "manifest.json.partial").write_text("{")
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED
    assert not (tmp_path / "out" / "manifest.json.partial").exists()


def test_dumps_kept_out_of_the_output(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dumps = str(tmp_path / "out" / "frames")
    status = upscale(tmp_path, source(tmp_path / "in.mkv"), "out", "--dump-frames", dumps)
    assert status == 1
    assert "in the output directory, which holds the job's own files only" in caplog.text
