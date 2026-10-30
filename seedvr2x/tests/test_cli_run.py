"""The command line's run on the CPU, the model replaced by a stand-in: the input read, the shots
and their seeds, the output written, its segments and its manifest, in every output format.

The stand-in's steps carry each shot's first frame (its seed less the job's) and frame count from
its encode through its windows to its decode, which writes, for each frame, the frame's index in
the job divided by 1000: so each output frame says where it comes from. The model's own output
is the GPU tests' (test_regression.py, test_shots.py)."""

import ctypes
import errno
import fcntl
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
import warnings
import weakref
import zlib
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import accumulate, pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_probe import BT709_VUI, has_encoder, has_x265, x265_file
from test_rate import mechanism7, retagged, strayed

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import calibration, planner, pull, timing, weights
from seedvr2x.runtime.detector import BATCH, Detector
from seedvr2x.runtime.job import Shot, output_size, padded_size
from seedvr2x.runtime.plan import Plan, Tiles
from seedvr2x.runtime.planner import GIB, Peaks, Pins
from seedvr2x.runtime.pull import Pin
from seedvr2x.runtime.shot import decode_shot, padded_length, shot_layout


def _usable() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

SEED = 42
FRAMES = 9


def stand_in_encode(
    models: object,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
    *,
    numz_padding: bool = False,
    numz_tail: bool = False,
    tiles: Tiles | None = None,
    scratch: Path | None = None,
    allowance: int | None = None,
) -> torch.Tensor:
    """A latent (T', 1, 1, 16) in channel-major memory, as the VAE's, saying the shot's first frame
    (channel 0), its frame count (1), the target's height and width (2, 3), and whether the frames
    were padded as numz pads them (6: cli.NUMZ_PADDING) and the shot's end as numz pads it (7:
    cli.NUMZ_TAIL). Untiled, with no scratch and nothing allowed one: the plan's tiles are the
    tiled passes' own tests'."""
    assert tiles is None and scratch is None and not allowance
    assert read(count).shape[0] == count
    latent = torch.zeros(16, (padded_length(count) - 1) // 4 + 1, 1, 1)
    latent[0], latent[1] = seed - SEED, count
    latent[2], latent[3] = target
    latent[6], latent[7] = numz_padding, numz_tail
    return latent.permute(1, 2, 3, 0)


def stand_in_windows(
    models: object,
    latent: torch.Tensor,
    layout: list[tuple[int, int]],
    seed: int,
    start: int = 0,
    *,
    numz_draw: bool = False,
) -> Iterator[torch.Tensor]:
    for s, e in layout[start:]:
        yield latent[s:e] + 0


def stand_in_decode_stream(models: object, latent: torch.Tensor) -> Iterator[torch.Tensor]:
    """The VAE's decode of a stand-in latent (stand_in_encode), in its slices, the first two
    latents then one at a time: frames (3, t, H, W) at the target's size padded as the encoder's
    input is (model.input_transform: job.padded_size, or numz's multiples of 16), in [-1, 1]: each
    saying its index in the job, divided by 1000, NaN in the padding, which the decode crops; NaN
    throughout for the frames of a latent whose channel 4 is NaN."""
    first, frames = int(latent[0, 0, 0, 0]), int(latent[0, 0, 0, 1])
    height, width = int(latent[0, 0, 0, 2]), int(latent[0, 0, 0, 3])
    padded = padded_size((height, width))
    if latent[0, 0, 0, 6]:
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


# The memory the stand-in's GPU has free and in all, as its driver reports them
# (torch.cuda.mem_get_info), in bytes: the 96 GB card's as numz's process found it (DESIGN.md,
# Memory planner), which holds every plan the stand-in's frames ask for; and the system memory
# the stand-in's host has available (cli._host_available).
FREE, TOTAL = round(94.5 * GIB), 96 * GIB
HOST = 64 * GIB
# The stand-in's frames as the planner counts them: upscale's --resolution 96 of source's 64x48,
# and its padding (job.padded_size), one tile's worth at most.
TARGET = (96, 128)
PADDED = (128, 128)

# The phases the stand-in's models were told, in order (Models.enter), since stand_in_model.
ENTERED: list[str] = []
# What the stand-in was asked before a run's first unit, in order, since stand_in_model: each
# reading of the free memory with the device it was asked for (cli._free_memory), and "models"
# for each load (model.load_models).
ASKED: list[object] = []
# The stand-in's models: the CPU, and the VAE's slicing, which the encode and split's reference
# follow (model.encode_slices).
STAND_IN_MODELS = SimpleNamespace(
    attention="none",
    device=torch.device("cpu"),
    runner=SimpleNamespace(vae=SimpleNamespace(use_slicing=True, slicing_sample_min_size=4)),
    enter=ENTERED.append,
)


def stand_in_probabilities(frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """The stand-in detector's probability of each frame, (n, 27, 48, 3) uint8: its mean component
    value over 255, float32, so that a record's probabilities say which frames reached it."""
    return (frames.mean(axis=(1, 2, 3), dtype=np.float64) / 255).astype(np.float32)


class StandInDetector:
    """The shot detector's stand-in, runtime/detector.py's Detector to the first pass: each
    frame's probability as soon as it is pushed (stand_in_probabilities), none at finish; its file,
    an empty transnetv2.safetensors pinned as such (stand_in_model). `built` lists each one made,
    with the path and device it was given."""

    built: ClassVar[list["StandInDetector"]] = []

    def __init__(
        self, path: Path, device: torch.device, *, batch: int = 4, hashes: object = None
    ) -> None:
        self.path, self.device, self.pushed, self.closed = path, device, 0, False
        StandInDetector.built.append(self)

    @property
    def scored(self) -> int:
        return self.pushed

    def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        assert frames.dtype == np.uint8 and frames.shape[1:] == (27, 48, 3)
        self.pushed += len(frames)
        return stand_in_probabilities(frames)

    def finish(self) -> npt.NDArray[np.float32]:
        return np.empty(0, np.float32)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "StandInDetector":
        return self

    def __exit__(self, *raised: object) -> None:
        self.close()


# The stand-in detector's file: empty, under the shot detector's name, beside the stand-in's
# model files (upscale), pinned as such in place of TransNetV2's weights (stand_in_model).
STAND_IN_DETECTOR = Pin(0, hashlib.sha256(b"").hexdigest(), "detector")
# The shot detector and its pin as seedvr2x has them, which the stand-in replaces.
DETECTOR_PIN = pull.PINNED[pull.DETECTOR]
REAL_DETECTOR = Detector
# The model files' check by their headers, which the stand-in replaces too.
CHECK_MODELS = weights.check_models


def stand_in_model(patch: Callable[[Any, str, Any], None]) -> None:
    """Replace the model by the stand-in, through patch: monkeypatch.setattr, or setattr in a
    process of its own (test_stop.py). The decode around the VAE's is seedvr2x's own. Its files,
    one empty w.safetensors as both, pass for models: the check is test_weights.py's. The shot
    detector is StandInDetector, its file an empty transnetv2.safetensors, pinned so in place of
    TransNetV2's weights, which its pin then checks."""
    from seedvr2x.runtime import detector, memory, model, pull, run, weights

    patch(weights, "check_models", lambda *a: None)
    # The stand-in's GPU has no driver to ask (runtime/memory.py): in a process of its own too,
    # where conftest.py's fixture doesn't reach.
    patch(memory, "paravirtual", lambda: False)
    patch(detector, "Detector", StandInDetector)
    patch(pull, "PINNED", {**pull.PINNED, pull.DETECTOR: STAND_IN_DETECTOR})
    StandInDetector.built.clear()
    ENTERED.clear()
    patch(torch.cuda, "is_available", lambda: True)
    patch(torch.cuda, "is_bf16_supported", lambda: True)
    patch(torch.cuda, "get_device_name", lambda device: "a stand-in")
    patch(torch.backends.cudnn, "version", lambda: None)
    patch(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    patch(torch.cuda, "max_memory_allocated", lambda device: 0)
    # The planner's budget: the free memory's reading as the command line takes it
    # (cli._free_memory), its first tensor and its warm-up on the stand-in's device, the driver's
    # figures the stand-in's; the system memory available a figure of its own, not this
    # machine's; and the attention backend FlashAttention 2, whose fit the plans here are
    # counted by, on a machine without it too (cli._attention).
    free_memory = cli._free_memory  # pyright: ignore[reportPrivateUsage]
    ASKED.clear()

    def read_free(device: torch.device) -> int:
        ASKED.append(("free memory", device))
        return free_memory(STAND_IN_MODELS.device)

    def load(*arguments: object) -> object:
        ASKED.append("models")
        return STAND_IN_MODELS

    patch(torch.cuda, "mem_get_info", lambda device: (FREE, TOTAL))
    patch(cli, "_free_memory", read_free)
    patch(cli, "_host_available", lambda: HOST)
    patch(cli, "_attention", lambda: "flash_attn_2")
    patch(model, "load_models", load)
    patch(model, "nvidia_driver", lambda: "a stand-in")
    patch(model, "decode_stream", stand_in_decode_stream)
    patch(run, "encode_shot", stand_in_encode)
    patch(run, "sample_windows", stand_in_windows)
    # The time estimates' calibration (--plan, --until) on steps that take no time: its own steps
    # would be the stand-in's, made beside the job's (tests/test_timing.py holds it, and
    # tests/test_until.py its runs through the command line).
    patch(calibration, "calibrate", stand_in_calibrate)


def stand_in_calibrate(*arguments: object, **options: object) -> calibration.Calibrated:
    """A calibration whose every kind and term takes no time (timing.Model(0, 0))."""
    models = dict.fromkeys((*timing.KINDS, *timing.TERMS), timing.Model(0.0, 0.0))
    return calibration.Calibrated(timing.Clock(models), 0.0)


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


def upscale(
    tmp_path: Path, input_path: Path, output: str, *options: str, detector: bool = True
) -> int:
    """seedvr2x run in this process, without colour correction unless options ask for one: the
    stand-in's frames then say which they are (stand_in_decode_stream). In gbrp16le unless options
    name a format, not the default yuv420p10le: the tests read the masters' frames as the RGB
    planes written (indexes, stored). The stand-in detector's file beside the models' unless
    `detector` is False."""
    weights = tmp_path / "w.safetensors"
    if not weights.exists():
        weights.write_bytes(b"")
    detector_file = tmp_path / pull.DETECTOR  # the stand-in detector's (stand_in_model)
    if detector and not detector_file.exists():
        detector_file.write_bytes(b"")
    return cli.main(command(tmp_path, input_path, output, *options))


def command(tmp_path: Path, input_path: Path, output: str, *options: str) -> list[str]:
    """upscale's command line."""
    correction = [] if "--color-correction" in options else ["--color-correction", "none"]
    output_format = [] if "--format" in options else ["--format", "gbrp16le"]
    return [
        *(str(input_path), "-o", str(tmp_path / output)),
        *("--model-dir", str(tmp_path), "--dit-model", "w.safetensors"),
        *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", str(SEED)),
        *correction,
        *output_format,
        *options,
    ]


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


def stored(input_args: list[str], pix_fmt: str) -> list[int]:
    """The CRC-32 of each frame of an output at the stand-in's 128x96, decoded as it is stored."""
    from seedvr2x.media.checksums import frame_bytes

    data = subprocess.run(
        ["ffmpeg", "-v", "error", *input_args, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True,
        check=True,
    ).stdout
    size = frame_bytes(pix_fmt, 128, 96)
    return [zlib.crc32(data[k : k + size]) for k in range(0, len(data), size)]


def checksums(path: Path) -> list[int]:
    from seedvr2x.media.checksums import read_checksums

    return read_checksums(path)


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
    # The job's own files were in its work directory, gone once the file is whole.
    assert not (tmp_path / "manifest.json").exists() and not (tmp_path / "one.mkv.work").exists()
    # Its checksums beside it, of the planes it holds; verify checks them.
    master = tmp_path / "one.mkv"
    assert checksums(tmp_path / "one.mkv.crc32") == stored(["-i", str(master)], "gbrp16le")
    assert cli.main(["verify", str(master)]) == 0


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
        "checksums",
        "frame_index.bin",
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    for name in ("seg_000000", "seg_000001"):
        segment = ["-i", str(out / f"{name}.mkv")]
        assert checksums(out / "checksums" / f"{name}.crc32") == stored(segment, "gbrp16le")
    assert cli.main(["verify", str(out)]) == 0
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
def test_png_segments(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "cuts.txt").write_text("4\n")
    status = upscale(
        tmp_path,
        source(tmp_path / "in.mkv"),
        "png",
        *("--cuts", str(tmp_path / "cuts.txt"), "--min-segment", "0", "--format", "png"),
    )
    assert status == 0
    # Each segment's hash, recorded when it was finished (DESIGN.md, Output, Assembly, decided
    # on 2026-10-12): a directory's, over its files in name order, each as name, a zero byte, size
    # in 8 bytes (big endian) and bytes (units.digest).
    recorded = json.loads((tmp_path / "png" / "manifest.json").read_text())["segments"]
    for entry in recorded:
        expected = hashlib.sha256()
        for file in sorted((tmp_path / "png" / entry["name"]).iterdir()):
            expected.update(file.name.encode() + b"\0" + file.stat().st_size.to_bytes(8, "big"))
            expected.update(file.read_bytes())
        assert entry["sha256"] == expected.hexdigest()
    assert len({entry["sha256"] for entry in recorded}) == 2
    for name, frames in (("seg_000000", range(4)), ("seg_000001", range(4, 9))):
        pngs = sorted((tmp_path / "png" / name).iterdir())
        assert [p.name for p in pngs] == [f"{i:06d}.png" for i in range(len(frames))]
        assert [indexes(p, "rgb48le")[0] for p in pngs] == list(frames)
        # Its checksums: of the rgb48be pixels each file holds.
        pattern = ["-start_number", "0", "-i", str(tmp_path / "png" / name / "%06d.png")]
        assert checksums(tmp_path / "png" / "checksums" / f"{name}.crc32") == stored(
            pattern, "rgb48be"
        )
    assert cli.main(["verify", str(tmp_path / "png")]) == 0
    # Its files listed before its frames are decoded: the first missing, or one not its own,
    # named as such.
    first = tmp_path / "png" / "seg_000001" / "000000.png"
    kept = first.read_bytes()
    first.unlink()
    assert cli.main(["verify", str(tmp_path / "png")]) == 1
    assert "seg_000001: 1 PNG missing, from 000000.png" in caplog.text
    first.write_bytes(kept)
    (first.parent / "notes.txt").write_text("mine")
    assert cli.main(["verify", str(tmp_path / "png")]) == 1
    assert "seg_000001: not its frames: notes.txt" in caplog.text


@pytest.mark.usefixtures("stand_in")
def test_yuv_segments(tmp_path: Path) -> None:
    (tmp_path / "cuts.txt").write_text("4\n")
    status = upscale(
        tmp_path,
        source(tmp_path / "in.mkv"),
        "yuv",
        *("--cuts", str(tmp_path / "cuts.txt"), "--min-segment", "0", "--format", "yuv420p10le"),
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
    assert probe.stdout.strip() == "yuv420p10le,smpte170m,4"
    # Its checksums, ffmpeg's: of the yuv420p10le frames it holds.
    for name in ("seg_000000", "seg_000001"):
        sums = tmp_path / "yuv" / "checksums" / f"{name}.crc32"
        assert checksums(sums) == stored(
            ["-i", str(tmp_path / "yuv" / f"{name}.mkv")], "yuv420p10le"
        )
    assert cli.main(["verify", str(tmp_path / "yuv")]) == 0
    # A frame altered, the file valid: verify names it.
    data = bytearray(
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(master),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p10le",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
    )
    from seedvr2x.media.checksums import frame_bytes

    data[2 * frame_bytes("yuv420p10le", 128, 96)] ^= 1
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le"),
            *("-s", "128x96", "-i", "-", "-c:v", "ffv1", "-level", "3", str(master)),
        ],
        input=bytes(data),
        check=True,
    )
    assert cli.main(["verify", str(tmp_path / "yuv")]) == 1
    assert cli.main(["verify", str(master)]) == 1


@pytest.mark.usefixtures("stand_in")
def test_yuv_by_default(tmp_path: Path) -> None:
    # yuv420p10le unless asked otherwise (DESIGN.md, Output), one file or a directory: at 960x720,
    # an HD size, BT.709 and limited range (writer.yuv_matrix), chroma sited left; its checksums,
    # ffmpeg's, of the frames it holds; a setting, which the manifest records.
    from seedvr2x.media.probe import probe

    source_path = source(tmp_path / "in.mkv")
    (tmp_path / "w.safetensors").write_bytes(b"")
    (tmp_path / pull.DETECTOR).write_bytes(b"")  # the stand-in detector's (stand_in_model)
    common = ["--model-dir", str(tmp_path), "--dit-model", "w.safetensors"]
    common += ["--vae-model", "w.safetensors", "--resolution", "720", "--color-correction", "none"]
    for output in ("one.mkv", "out"):
        assert cli.main([str(source_path), "-o", str(tmp_path / output), *common]) == 0
    for master in (tmp_path / "one.mkv", tmp_path / "out" / "seg_000000.mkv"):
        stream = probe(master)
        assert (stream.pix_fmt, stream.width, stream.height) == ("yuv420p10le", 960, 720)
        tags = (stream.color_space, stream.color_range, stream.chroma_location)
        assert tags == ("bt709", "tv", "left")
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["output"]["format"] == content["settings"]["format"] == "yuv420p10le"
    # Its checksums, beside the file and in the directory's: every frame checked against them.
    for output in ("one.mkv", "out"):
        assert cli.main(["verify", str(tmp_path / output)]) == 0


@pytest.mark.usefixtures("stand_in")
def test_tags_of_one_side_carried(tmp_path: Path) -> None:
    # The container declaring the matrix alone over a bitstream tagged BT.709 throughout: the
    # source's primaries and transfer are its first frame's (media/source.py, resolved), which the
    # output declares as they are (DESIGN.md, Colour and shape) and the manifest records with the
    # reading.
    if not has_x265():
        pytest.skip("needs ffmpeg with libx265")
    source_path = x265_file(tmp_path / "in.mkv", "", BT709_VUI, "-colorspace", "bt709")
    assert upscale(tmp_path, source_path, "out") == 0
    master = tmp_path / "out" / "seg_000000.mkv"
    tags = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json"),
            *("-show_entries", "stream=color_primaries,color_transfer", str(master)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(tags.stdout)["streams"] == [
        {"color_primaries": "bt709", "color_transfer": "bt709"}
    ]
    entry = json.loads((tmp_path / "out" / "manifest.json").read_text())["input"]
    assert (entry["primaries"], entry["transfer"]) == ("bt709", "bt709")
    assert entry["read_as"] == "YUV bt709, limited range, chroma left"


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
        self.numz_padding: list[bool] = []  # each encode's, as run_job passed it
        self.numz_draw: list[bool] = []  # each shot's windows', as run_job passed it
        self.numz_tail: list[bool] = []  # each encode's, as run_job passed it
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
        *,
        numz_padding: bool = False,
        numz_tail: bool = False,
        tiles: Tiles | None = None,
        scratch: Path | None = None,
        allowance: int | None = None,
    ) -> torch.Tensor:
        frames = read(count)
        self.read[seed - SEED] = hashlib.md5(frames.tobytes()).hexdigest()
        self.numz_padding.append(numz_padding)
        self.numz_tail.append(numz_tail)
        self._call(f"encode {seed - SEED}")
        latent = stand_in_encode(
            *(models, lambda n: frames[:n], count, target, seed),
            numz_padding=numz_padding,
            numz_tail=numz_tail,
            tiles=tiles,
            scratch=scratch,
            allowance=allowance,
        )
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
        *,
        numz_draw: bool = False,
    ) -> Iterator[torch.Tensor]:
        self.numz_draw.append(numz_draw)
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
        **tiled: Any,
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

        decode_shot(models, merged, count, target, write_then_stop, *names, **tiled)  # pyright: ignore[reportArgumentType]


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
    # A segment's shots are encoded and sampled before its decode, so that its decode and write
    # is a unit of its own; in a .mkv path's work directory as in an output directory, where the
    # three shots are one segment's without --min-segment (5 s by default).
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "checksums",
        "frame_index.bin",
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
    # A job never planned again records no cap of a run's own (manifest.WINDOW_CAP), and each
    # run's windows are the plan's cap's: the record as it was before the field.
    assert all("window_cap" not in shot for shot in content["shots"])
    assert [shot["windows"] for shot in content["shots"]] == [
        [list(window) for window in shot_layout(frames, 5)] for frames in (3, 1, 21)
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
    entry = content["input"]
    assert entry["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert steps.calls == [
        *("encode 0", "window 0:0", "encode 3", "window 3:0"),
        *("encode 4", "window 4:0", "window 4:1", "decode 0", "decode 3", "decode 4"),
    ]
    assert indexes(tmp_path / "one.mkv") == list(range(25))


# JOB's shots, and its plan on the stand-in's GPU: its cap pinned, every weight on the GPU.
JOB_SHOTS = [Shot(0, 3), Shot(3, 4), Shot(4, 25)]


def untimed(recorded: dict[str, Any]) -> dict[str, Any]:
    """A plan's record without the timings the run's calibration adds, for information (every run
    calibrates: DESIGN.md, Memory planner, Time estimates), which it holds for every kind."""
    if "timings" in recorded:
        assert sorted(recorded["timings"]) == sorted([*timing.KINDS, *timing.TERMS])
    return {key: value for key, value in recorded.items() if key != "timings"}


def plan_record(plan: Plan, peaks: Peaks) -> dict[str, Any]:
    """A plan as a manifest records it: its shape, and under memory where the weights are with
    what it is predicted to take (cli._run)."""
    record = plan.record()
    record["memory"].update(peaks.record())
    return record


def free_memory(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    """Have the stand-in's GPU report `free` bytes free from now on."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, TOTAL))


def kept(directory: Path) -> str:
    """What a refusal adds once the job's directory holds the record of its first pass, a plan
    (cli._first_kept; DESIGN.md, Pause and resume, A plan)."""
    return (
        f"; the first pass is kept in {directory}: the same command takes it up, or one changing"
        " what is named here, without decoding the source again"
    )


def plan_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [line for line in caplog.messages if line.startswith("plan: ")]


def without_disk(lines: list[str]) -> list[str]:
    """The plan's lines but the disk's (disk.Disk.line), which says what the test's own disk has
    free: the figure of the machine the test runs on."""
    return [line for line in lines if not line.startswith("plan: disk: ")]


def test_plan_recorded_and_phases_entered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The models are told each unit's phase before it (Models.enter), a segment's decodes after
    # its shots' windows, in a directory as in a .mkv path's work directory, whose three shots
    # are one segment's without --min-segment; the manifest records the plan, which a resume
    # compares but for where the weights are (DESIGN.md, Memory planner).
    caplog.set_level(logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    encode_and_window = ["encode", "dit"]
    assert ENTERED == [
        *(*encode_and_window, *encode_and_window, "decode", "decode"),
        *("encode", "dit", "dit", "decode"),
    ]
    manifest = tmp_path / "out" / "manifest.json"
    content = json.loads(manifest.read_text())
    # The planner's plan for the memory the stand-in reports free, read on its device: the cap
    # pinned, nothing tiled, swapped or moved; the 6-latent shot's two windows of 4 the longest.
    made = planner.plan(FREE, TARGET, JOB_SHOTS, Pins(window=5), HOST)
    assert made[0] == Plan(5)
    assert untimed(content["plan"]) == plan_record(*made)
    assert content["plan"]["memory"] == {
        "swapped_blocks": 0,
        "resident_encode": None,
        "resident_decode": None,
        "vae_in_dit": True,
        "scratch_allowance": 0,
        "free": FREE,
        # A driver that refuses: what it reports is what the plan counts on; the whole card's
        # free memory isn't read. And the attention backend the DiT's peak is predicted for.
        "driver_free": FREE,
        "card_free": None,
        # No reserve on a driver that refuses.
        "reserve": None,
        "attention": "flash_attn_2",
        "budget": FREE - planner.MARGIN,
        "encode_peak": planner.encode_peak(PADDED),
        # A window of the 6-latent shot.
        "dit_peak": planner.dit_peak(PADDED, 4),
        "decode_peak": planner.decode_peak(PADDED, TARGET),
        "bound": "encode",
        # The process's own, and the longest shot as its decode merges its two windows of 4: the
        # windows, 8 latents, the merged latent, 6, and a join's mix, 6, of 8,192 bytes each
        # (planner.host_moments).
        "host": planner.HOST_RESIDENT + 20 * 8_192,
        "host_shot": 20 * 8_192,
        "scratch": 0,
        "longest_window": 4,
    }
    assert list(content).index("plan") == list(content).index("output") + 1
    assert (content["settings"]["encode_tile"], content["settings"]["decode_tile"]) == (None, None)
    ENTERED.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert ENTERED == [*(["encode", "dit"] * 2), "encode", "dit", "dit", *(["decode"] * 3)]
    # --swap-blocks shapes nothing, so is no setting: pinned at what the planner chose, or at
    # anything else, the same job; where the weights were is no part of it, nor the timings. A
    # finished job isn't planned again: its plan stays as recorded.
    content["plan"]["memory"] = {"swapped_blocks": 36, "vae_in_dit": False}
    content["plan"]["timings"] = {"window": 1.5}
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--swap-blocks", "0") == 0
    assert "finished already" in caplog.text and plan_lines(caplog) == []
    assert json.loads(manifest.read_text()) == content
    # A tile pin is a setting, which a resume must find again, though it pins what the planner
    # chose: another job.
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--decode-tile", "0") == 1
    assert "settings.decode_tile: null -> 0" in caplog.text
    # What shapes the output is taken as recorded: another cap there lays the shots out
    # otherwise than the job's units are.
    content["plan"]["window"] = 6
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert "shots[2].windows: 2 recorded -> 1 now" in caplog.text
    # A record that isn't a plan's is refused, the manifest named.
    content["plan"]["window"] = "six"
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert f"{manifest}: not a plan as seedvr2x records one: its window, 'six'" in caplog.text
    # So is a cap no layout exists for, under 5 latents with a shot longer than it (the planner
    # makes one that short for shorter shots alone), and a manifest that records no plan.
    content["plan"]["window"] = 3
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert (
        f"{manifest}: not a plan as seedvr2x records one: its window, 3, with a shot of 6"
        " latents: a cap that a shot is longer than is 5 latents at least"
    ) in caplog.text
    # And tiles of which one covers the whole padded frame, which is no tiling (DESIGN.md,
    # Pipeline, per shot, step 3): the planner makes none, and refuses the pin of one.
    content["plan"]["window"] = 5
    content["plan"]["decode_tiles"] = {"size": 512, "overlap": 64}
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert (
        f"{manifest}: not a plan as seedvr2x records one: its decode_tiles, of 512 px, with"
        " frames padded to 128x128: a tile that covers the whole frame is no tiling, and an"
        " untiled decode records none"
    ) in caplog.text
    del content["plan"]
    manifest.write_text(json.dumps(content))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert f"{manifest}: not a plan as seedvr2x records one: None" in caplog.text


def test_swap_blocks_pin_places_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The pin is the plan's swapped blocks: the loader gets the plan, and places the weights by
    # it (model.load_models); the manifest records it, for information; the units are those of
    # the job without it.
    from seedvr2x.runtime import model

    plans: list[Any] = []

    def load(*arguments: Any) -> Any:
        plans.append(arguments[4])
        return STAND_IN_MODELS

    monkeypatch.setattr(model, "load_models", load)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--swap-blocks", "36") == 0
    # The planner's plan around the pin, on the memory the stand-in reports free: nothing else
    # moves, the DiT's slots staying in the VAE's phases.
    pins = Pins(window=5, swap_blocks=36)
    made = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST)
    assert plans == [made[0]] == [Plan(5, swapped=36)]
    assert steps.calls == UNINTERRUPTED
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["plan"]["memory"]["swapped_blocks"] == 36
    assert untimed(content["plan"]) == plan_record(*made)
    assert untimed(content["plan"]) | {"memory": None} == Plan(5).record() | {"memory": None}


def spilling_driver(monkeypatch: pytest.MonkeyPatch, card: int | None = None) -> list[float]:
    """A driver behind WSL2's device that gives memory the card doesn't have (runtime/memory.py),
    the probe's answer a stand-in's; returns the limits torch is then given, as fractions.
    `card`: what the whole card has free, a stand-in for NVML's reading of it (memory._read);
    None: a machine without NVML, as conftest.py leaves every CPU test, warned of in the run."""
    from seedvr2x.runtime import memory

    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    if card is not None:
        monkeypatch.setattr(memory, "_read", lambda index: card)
    monkeypatch.setattr(
        memory, "probe", lambda: memory.Driver(True, FREE + memory.OVER, FREE, TOTAL)
    )
    # What such a run leaves outside torch's limit and plans under it again, at figures of the
    # tests' own, whatever a card's measure makes them (tests/test_memory.py holds those).
    monkeypatch.setattr(memory, "HELD_BACK", 3 * GIB)
    monkeypatch.setattr(memory, "HELD_BACK_UNREAD", 3 * GIB)
    monkeypatch.setattr(memory, "RESERVE", round(2.3 * GIB))
    limits: list[float] = []
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: limits.append(fraction),
    )
    return limits


def test_a_driver_that_spills_is_planned_under_and_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Where the driver gives memory the card doesn't have: torch's own allocator, chosen before
    # torch is imported; the plan made in the memory free less what is held back; torch limited
    # to that before the models load. The memory free: what the whole card has, 2 GiB less here
    # than the driver reports to the process, another program's. The run itself is the same.
    # And torch is limited a first time before anything of the run works on the GPU: its first
    # pass, whose shot detector runs there, and the warm-up its free memory is read with, which
    # ran before any limit existed.
    from seedvr2x.runtime import memory, model

    caplog.set_level(logging.INFO)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    card = FREE - 2 * GIB
    limits = spilling_driver(monkeypatch, card)
    loads: list[tuple[Any, int]] = []
    monkeypatch.setattr(
        model, "load_models", lambda *a: loads.append((a[4], len(limits))) or STAND_IN_MODELS
    )
    # The limits set by then, at the first pass and at the reading of the free memory.
    held_at: list[tuple[str, int]] = []
    first_pass, read_free = cli._first_pass, cli._free_memory  # pyright: ignore[reportPrivateUsage]

    def passing(*arguments: Any) -> Any:
        held_at.append(("the first pass", len(limits)))
        return first_pass(*arguments)

    def reading(device: torch.device) -> int:
        held_at.append(("the reading", len(limits)))
        return read_free(device)

    monkeypatch.setattr(cli, "_first_pass", passing)
    monkeypatch.setattr(cli, "_free_memory", reading)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native,expandable_segments:True"
    assert held_at == [("the first pass", 1), ("the reading", 1)]
    # What the plan counts on is its budget, the planner's margin counted in what a held run
    # leaves to torch's allocator: the plan of a driver that refuses and reports that margin
    # more.
    planned_in = card - memory.HELD_BACK - memory.RESERVE
    made = planner.plan(planned_in + planner.MARGIN, TARGET, JOB_SHOTS, Pins(window=5), HOST)
    # Held twice: before the first pass, and again once the plan is made, before the models
    # load; the stand-in's card has the same memory free both times.
    assert loads == [(made[0], 2)] and limits == [(card - memory.HELD_BACK) / TOTAL] * 2
    assert steps.calls == UNINTERRUPTED
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    # The record keeps what the plan was made in, and beside it what the driver reported and
    # what the whole card had free.
    recorded = content["plan"]["memory"]
    assert (recorded["free"], recorded["driver_free"], recorded["card_free"]) == (
        planned_in,
        FREE,
        card,
    )
    assert recorded["budget"] == planned_in
    assert memory.LENDS in caplog.text
    # The plan's card lines name each figure, where "92.50 GiB free" alone would be taken for
    # what the plan counts on, and "87.20 GiB free" for what the driver reports; what torch's
    # limit leaves and what the plan leaves again, the reserve, are said apart on the budget's
    # line; and it is the one place a run's log says them: memory's own line is for its debug
    # level.
    assert (
        "plan: card: a stand-in, 92.50 GiB free on the whole card by NVML (its driver reports"
        " 94.50 GiB free to this process): the run is held to the card's own memory, its driver"
        " giving more than the card has"
    ) in caplog.text
    assert (
        "plan: budget: 87.20 GiB, the 92.50 GiB free less torch's limit, 3.00 GiB under it, and the"
        " reserve of 2.30 GiB left to torch's allocator beyond a step's peak; the margin of"
        " 0.60 GiB is in the reserve"
    ) in caplog.text
    assert caplog.text.count("plan: budget: 87.20 GiB") == 1
    assert (
        "torch held to 89.50 GiB on the GPU: the 92.50 GiB free less 3.0, and what it holds;"
        " free on the whole card by NVML (its driver reports 94.50 GiB free to this process)"
    ) in caplog.text
    # The run's limit is said once at the level a user reads: the first is for its debug level.
    assert caplog.text.count("torch held to ") == 1


def test_the_user_s_allocator_stands_on_a_driver_that_spills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    limits = spilling_driver(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync" and len(limits) == 2
    said = "the allocator is the environment's, backend:cudaMallocAsync: torch's limit counts"
    assert said in caplog.text


def test_a_driver_that_refuses_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # No WSL2 device (the tests' own, conftest.py), as on Linux: DESIGN.md's allocator, the whole
    # free memory planned in, torch given no limit, the driver asked nothing.
    from seedvr2x.runtime import memory

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(memory, "probe", lambda: pytest.fail("the driver was asked"))
    limits: list[float] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: limits.append(fraction),
    )
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync" and not limits
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["plan"]["memory"]["free"] == FREE


# The command line in a process of its own, up to where the driver is asked: its status says
# whether torch was imported by then, 7, or not, 8. The model files pass unread.
UP_TO_THE_DRIVER = """
import sys
from seedvr2x import cli
from seedvr2x.runtime import memory, weights

def asked():
    sys.exit(7 if "torch" in sys.modules else 8)

weights.check_models = lambda *a: None
memory.startup = asked
sys.exit(cli.main(sys.argv[1:]))
"""


def test_the_allocator_is_chosen_before_torch_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # torch's library fixes its allocator when it loads, whatever the environment says after:
    # the driver is asked, and the allocator set, before anything imports torch. For a new job;
    # and for one recorded, whose comparison with its record imports the runtime, torch with it:
    # asked later, a resumed job ran on torch's default allocator, never the one chosen.
    source_path = job(tmp_path, monkeypatch)
    args = command(tmp_path, source_path, "out", *JOB)
    for recorded in (False, True):
        if recorded:
            steps.stop = "encode 4"
            stopped(tmp_path, source_path, "out", *JOB)
            assert (tmp_path / "out" / "manifest.json").exists()
        result = subprocess.run(
            [sys.executable, "-c", UP_TO_THE_DRIVER, *args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 8, (recorded, result.stderr)


def test_a_run_refused_before_its_gpu_asks_the_driver_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The driver is asked where a run first takes the GPU: a run refused before, here for a
    # setting, disturbs no card, and sets no allocator.
    from seedvr2x.runtime import memory

    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    monkeypatch.setattr(memory, "probe", lambda: pytest.fail("the driver was asked"))
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--cut-threshold", "0.4") == 1
    assert not any(name in os.environ for name in memory.VARIABLES)


def test_a_run_refused_for_its_model_files_asks_the_driver_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The driver is asked once the run's model files are checked: one refused for them, here a
    # file that is no model, has disturbed no card, and set no allocator. Asked before them, a
    # desktop on that card would blink for a run that then stops on a file.
    from seedvr2x.runtime import memory, weights

    def refuse(*files: object) -> None:
        raise weights.ModelError("w.safetensors: not a SeedVR2 DiT")

    monkeypatch.setattr(weights, "check_models", refuse)
    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    monkeypatch.setattr(memory, "probe", lambda: pytest.fail("the driver was asked"))
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert "w.safetensors: not a SeedVR2 DiT" in caplog.text
    assert not any(name in os.environ for name in memory.VARIABLES) and steps.calls == []


def test_a_plan_on_a_driver_that_spills_is_held_for_its_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # --plan counts in the memory a run would: the same plan as the run's; torch is held for
    # its first pass, as a run's is, and again as it loads the models for the time estimates'
    # calibration (DESIGN.md, Memory planner: --plan "loads them, for the calibration").
    from seedvr2x.runtime import memory

    caplog.set_level(logging.INFO, logger="seedvr2x")
    limits = spilling_driver(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", "plan.txt") == 0
    assert limits == [(FREE - memory.HELD_BACK_UNREAD) / TOTAL] * 2 and steps.calls == []
    assert "torch held to " in caplog.text
    planned_in = FREE - memory.HELD_BACK - memory.RESERVE
    reading = memory.Reading(FREE, None, planned_in, True, memory.HELD_BACK_UNREAD)
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    # And the timings the calibration gave, for information (resume.UNCOMPARED).
    assert sorted(content["plan"].pop("timings")) == sorted([*timing.KINDS, *timing.TERMS])
    assert content["plan"] == plan_record(
        *planner.plan(reading, TARGET, JOB_SHOTS, Pins(window=5), HOST)
    )
    assert content["plan"]["memory"]["free"] == planned_in
    # A machine without NVML (conftest.py): warned of once, as the plan is made, and the plan's
    # card line says which figure it had.
    assert caplog.text.count("the whole card's free memory couldn't be read through NVML") == 1
    assert (
        ", 94.50 GiB free as its driver reports it to this process (the whole card's, NVML's,"
        " wasn't read): the run is held to the card's own memory, its driver giving more than"
        " the card has"
    ) in caplog.text
    assert (
        "the 94.50 GiB free less torch's limit, 3.00 GiB under it, and the reserve of 2.30 GiB left"
        " to torch's allocator"
    ) in caplog.text


def test_resumed_on_a_driver_that_spills_is_held_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # Each start asks the driver, plans in the memory it may count on and holds torch to the
    # card: a job stopped on a driver that refuses and taken up on one that spills keeps its
    # shape, and its memory is planned in less.
    from seedvr2x.runtime import memory

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    assert json.loads(manifest.read_text())["plan"]["memory"]["free"] == FREE
    limits = spilling_driver(monkeypatch)
    # The first start's allocator, left in this process's environment: each start is a process
    # of its own, which finds none.
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native,expandable_segments:True"
    assert limits == [(FREE - memory.HELD_BACK) / TOTAL] * 2
    planned_in = FREE - memory.HELD_BACK - memory.RESERVE
    assert json.loads(manifest.read_text())["plan"]["memory"]["free"] == planned_in
    made = [indexes(tmp_path / "out" / f"seg_{number:06d}.mkv") for number in range(2)]
    assert made[0] + made[1] == list(range(25))


def test_a_held_run_refused_with_4_5_takes_2_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-13, for a resume's plan on 2026-10-21): a held
    # run's plan is made with the reserve of 4.5 GiB, and only where that plan would be refused,
    # with 2.5, said slow in its budget line; a resume's recorded shape too, before the resume
    # stops. The job pinned whole, its window, blocks and passes, on a card 1 GiB short of its
    # need with 4.5 and 1 over it with 2.5: a new job, and one stopped on a driver that refuses,
    # resumed there. The limit set once the plan is made may then lose 2.5 (memory.check).
    from seedvr2x.runtime import memory

    caplog.set_level(logging.INFO, logger="seedvr2x")
    pinned = ("--swap-blocks", "0", "--encode-tile", "0", "--decode-tile", "0")
    pins = Pins(window=5, encode_tile=0, decode_tile=0, swap_blocks=0)
    # The least the job's plan needs: the fewest bytes a driver that refuses plans it in, less
    # the margin, bisected.
    low, high = 0, FREE
    while high - low > 1:
        middle = (low + high) // 2
        try:
            planner.plan(middle, TARGET, JOB_SHOTS, pins, HOST)
            high = middle
        except planner.PlanError:
            low = middle
    need = high - planner.MARGIN
    source_path = job(tmp_path, monkeypatch)
    # Stopped with units made, its plan's shape kept from its record on resuming.
    steps.stop = "encode 3"
    stopped(tmp_path, source_path, "resumed", *JOB, *pinned)
    steps.stop = None
    spilling_driver(monkeypatch)
    # The measured reserve, not the tests' own 2.3, which is under 2.5.
    monkeypatch.setattr(memory, "RESERVE", round(4.5 * GIB))
    free = need - GIB + memory.HELD_BACK_UNREAD + memory.RESERVE
    free_memory(monkeypatch, free)
    # Both refused with 4.5 by the blocks pinned, the resume for its recorded window, the
    # recorded plan's (test_planner.py's test_a_held_plan_refused_takes_the_reserve_of_2_5 has a
    # recorded shape refused as such, unpinned).
    for output, whose in (("out", "--window 5's longest"), ("resumed", "the recorded plan's")):
        refused = f"--swap-blocks 0: a window of 4 latents, {whose}, takes"
        # Each start a process of its own, which finds no allocator set.
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        caplog.clear()
        assert upscale(tmp_path, source_path, output, *JOB, *pinned) == 0
        assert (output == "resumed") == any(
            line.startswith(f"resuming {tmp_path / output}") for line in caplog.messages
        )
        # The budget takes two lines (DESIGN.md, Architecture, What it prints, built on
        # 2026-10-29): its figures, then the reserve of 2.5 and what it costs, the refusal of
        # the plan with 4.5 said at debug only.
        [budget] = [line for line in caplog.messages if line.startswith("plan: budget: ")]
        [reserve] = [line for line in caplog.messages if line.startswith("plan: reserve: ")]
        assert budget.endswith(
            " and the reserve of 2.50 GiB left to torch's allocator beyond a step's peak; the"
            " margin of 0.60 GiB is in the reserve"
        )
        assert refused not in budget + reserve
        assert reserve.startswith("plan: reserve: 2.50 GiB and not 4.50, the plan with 4.50 having")
        assert reserve.endswith(
            "; slow: a step short of room beyond its peak slows down where it would be refused,"
            " and a job that fits with 4.50 runs up to 1.5 times as long with 2.50; memory freed"
            " on the card gives the 4.50 back; --swap-blocks 5 would run with the full reserve, at"
            " full speed"
        )
        content = json.loads((tmp_path / output / "manifest.json").read_text())
        assert content["plan"]["memory"]["reserve"] == memory.RESERVE_TIGHT
        assert content["plan"]["memory"]["free"] == need + GIB
        assert memory._reserve == memory.RESERVE_TIGHT  # pyright: ignore[reportPrivateUsage]
        made = [indexes(tmp_path / output / f"seg_{number:06d}.mkv") for number in range(2)]
        assert made[0] + made[1] == list(range(25))
    # 1 GiB short with 2.5 too: refused, by the figures of the plan with 2.5, both tried, the
    # first pass kept; and with 4.5 to spare nothing is said of 2.5.
    free_memory(monkeypatch, need - GIB + memory.HELD_BACK_UNREAD + memory.RESERVE_TIGHT)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    caplog.clear()
    assert upscale(tmp_path, source_path, "short", *JOB, *pinned) == 1
    [said] = [record.message for record in caplog.records if record.levelname == "ERROR"]
    assert said.endswith(
        "; refused with the reserve of 4.50 GiB that a held run's plan leaves torch's allocator"
        " beyond a step's peak, and with 2.50, the one it takes where that is refused, whose"
        " figures these are" + kept(tmp_path / "short")
    )
    free_memory(monkeypatch, need + memory.HELD_BACK_UNREAD + memory.RESERVE)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    caplog.clear()
    assert upscale(tmp_path, source_path, "room", *JOB, *pinned) == 0
    assert "not 4.50" not in caplog.text and "2.50" not in caplog.text
    content = json.loads((tmp_path / "room" / "manifest.json").read_text())
    assert content["plan"]["memory"]["reserve"] == memory.RESERVE
    # A user's own torch allocator without expandable segments, where a step short of room is
    # refused rather than slowed down: refused with 4.5, as before, saying that torch's default
    # setting would run it with 2.5, slower (design, 2026-10-21).
    free_memory(monkeypatch, free)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    caplog.clear()
    assert upscale(tmp_path, source_path, "native", *JOB, *pinned) == 1
    [said] = [record.message for record in caplog.records if record.levelname == "ERROR"]
    assert said.startswith("--swap-blocks 0: a window of 4 latents") and said.endswith(
        "; torch's own allocator with its expandable segments, a held run's default, which the"
        " allocator's setting in the environment replaces here, would run it with the reserve of"
        " 2.50 GiB, slower" + kept(tmp_path / "native")
    )


def test_a_held_run_warns_where_torch_refused_allocations_in_a_vae_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Allocator (decided on 2026-10-18): a held run warns where torch's count of
    # refused allocations (num_ooms) grows in a VAE unit, its frames maybe not those of a run
    # with more room. The stand-in's count grows by one at every step it makes: each encode and
    # each decode warned of, once, and no window, which isn't a VAE unit; nothing on a driver
    # that refuses, which isn't held.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: {"num_ooms": len(steps.calls)})
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "refusing", *JOB) == 0
    assert "refused" not in caplog.text
    spilling_driver(monkeypatch)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "held", *JOB) == 0
    assert steps.calls == UNINTERRUPTED
    warned = [
        record.message for record in caplog.records if "torch's allocator refused" in record.message
    ]
    assert [line.partition(":")[0] for line in warned] == [
        *("shot 1/3's encode", "shot 2/3's encode", "shot 1/3's decode", "shot 2/3's decode"),
        *("shot 3/3's encode", "shot 3/3's decode"),
    ]
    # Shot 1/3's encode, the run's first step: the count from 0 to 1.
    assert warned[0] == (
        "shot 1/3's encode: torch's allocator refused 1 of its requests during it (0 to 1 in all):"
        " where cuDNN's workspaces are refused it runs other kernels, which give other bits: its"
        " latent, and the frames from it, may not be those of a run with more room on the card"
    )
    assert warned[2].endswith(": its frames may not be those of a run with more room on the card")


def test_gpu_memory_refused_at_the_load_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The weights refused GPU memory as they load, the limit set before them where seedvr2x
    # holds torch: one line, no traceback, nothing run, the first pass kept as a plan.
    from seedvr2x.runtime import model

    spilling_driver(monkeypatch)

    def refused(*arguments: object) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 432.00 MiB.\nMore.")

    monkeypatch.setattr(model, "load_models", refused)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [
        "the GPU's memory was refused (held to the card's own memory: its driver gives more):"
        " CUDA out of memory. Tried to allocate 432.00 MiB, as the models loaded: nothing was"
        " run" + kept(tmp_path / "out")
    ]
    assert all(record.exc_info is None for record in errors) and steps.calls == []


# torch's own words when its limit refuses a request (torch 2.14.1, an RTX 5090 under WSL2,
# 2026-10-12: a unit of a 1080p job, torch limited to 20.98 GiB): one line.
TORCH_REFUSED = (
    "CUDA out of memory. Tried to allocate 456.00 MiB. GPU 0 has a total capacity of 31.82 GiB of"
    " which 9.36 GiB is free. Including non-PyTorch memory, this process has 17179869184.00 GiB"
    " memory in use. 20.98 GiB allowed; Of the allocated memory 19.27 GiB is allocated by PyTorch,"
    " and 1.43 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is"
    " large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. "
    " See documentation for Memory Management "
    " (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)"
)


@pytest.mark.parametrize("place", ["the first pass", "the reading", "the hold"])
def test_gpu_memory_refused_before_the_load_stops_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    place: str,
) -> None:
    # Before the models load a run already works on the GPU: the shot detector in the first
    # pass, the first allocation and warm-up its free memory is read with, then torch's limit
    # where seedvr2x holds it. Memory refused at any of them, another program holding the card:
    # the same one line and status as at the load, no traceback, nothing run; after the first
    # pass, at the reading, that pass kept as a plan (cli._first_kept). The first limit, held
    # before the first pass, and the pass itself keep nothing.
    from seedvr2x.runtime import memory

    def refused(*arguments: object, **options: object) -> None:
        raise torch.OutOfMemoryError(TORCH_REFUSED)

    target = {"the first pass": (cli, "_first_pass"), "the reading": (cli, "_free_memory")}
    monkeypatch.setattr(*target.get(place, (memory, "hold")), refused)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    # On a driver that refuses, the tests' own: the card's memory and what the driver had free
    # of it are said too, which are the card's there.
    assert [record.getMessage() for record in errors] == [
        "the GPU's memory was refused: CUDA out of memory. Tried to allocate 456.00 MiB, 9.36"
        " GiB free of the GPU's 31.82 GiB, torch holding 19.27 GiB allocated and 1.43 GiB more"
        " reserved, 20.98 GiB allowed, before the models loaded: nothing was run"
        + (kept(tmp_path / "out") if place == "the reading" else "")
    ]
    assert all(record.exc_info is None for record in errors) and steps.calls == []


# torch's own words under each allocator, its limit refusing a request of 4 GiB (torch 2.14.1,
# an RTX 5090 under WSL2, 2026-10-12: torch limited to 1% of the card, nothing taken): its own
# allocator's, one line, and cudaMallocAsync's, seven.
NATIVE_REFUSED = (
    "CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 31.82 GiB of"
    " which 30.20 GiB is free. 325.79 MiB allowed; Of the allocated memory 0 bytes is allocated"
    " by PyTorch, and 0 bytes is reserved by PyTorch but unallocated. If reserved but unallocated"
    " memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid"
    " fragmentation.  See documentation for Memory Management "
    " (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)"
)
ASYNC_REFUSED = (
    "Allocation on device 0 would exceed allowed memory. (out of memory)\n"
    "Currently allocated     : 0 bytes\n"
    "Requested               : 4.00 GiB\n"
    "Device limit            : 31.82 GiB\n"
    "Free (according to CUDA): 30.19 GiB\n"
    "PyTorch limit (set by user-supplied memory fraction)\n"
    "                        : 325.79 MiB"
)


def test_the_card_s_figures_stay_in_a_refusal_where_the_driver_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch's refusal names the card's memory and what the driver has free of it. On a driver
    # that refuses a full card they are the card's, and stay in the one line, under either
    # allocator: cudaMallocAsync, DESIGN.md's there, says them on lines of its own, with the
    # request, where its first line alone said no figure at all.
    from seedvr2x.runtime import memory

    refused = cli._refused  # pyright: ignore[reportPrivateUsage]
    assert not memory.spills()
    assert refused(torch.OutOfMemoryError(NATIVE_REFUSED)) == (
        "the GPU's memory was refused: CUDA out of memory. Tried to allocate 4.00 GiB, 30.20 GiB"
        " free of the GPU's 31.82 GiB, torch holding 0 bytes allocated and 0 bytes more reserved,"
        " 325.79 MiB allowed"
    )
    assert refused(torch.OutOfMemoryError(ASYNC_REFUSED)) == (
        "the GPU's memory was refused: Allocation on device 0 would exceed allowed memory. (out"
        " of memory): 4.00 GiB asked, 30.19 GiB free of the GPU's 31.82 GiB, torch holding 0"
        " bytes allocated"
    )
    # On a run held to its card the driver's figure isn't the card's (it read 30.20 GiB free
    # there beside a desktop holding 5.6 GiB of the card): neither is said, as before.
    monkeypatch.setattr(memory, "_spills", True)
    held = "the GPU's memory was refused (held to the card's own memory: its driver gives more): "
    assert refused(torch.OutOfMemoryError(NATIVE_REFUSED)) == (
        f"{held}CUDA out of memory. Tried to allocate 4.00 GiB, torch holding 0 bytes allocated"
        " and 0 bytes more reserved, 325.79 MiB allowed"
    )
    assert refused(torch.OutOfMemoryError(ASYNC_REFUSED)) == (
        f"{held}Allocation on device 0 would exceed allowed memory. (out of memory)"
    )


def test_a_refusal_s_words_are_one_line() -> None:
    # A refusal that isn't torch's allocation message (the per-unit check's, or another) is
    # given by its first line, whole, and no more: the stop stays one line.
    refusal = torch.OutOfMemoryError("the GPU has 1.00 GiB free where 2.00\nand a second line")
    said = cli._refused(refusal)  # pyright: ignore[reportPrivateUsage]
    assert said == "the GPU's memory was refused: the GPU has 1.00 GiB free where 2.00"


def test_no_torch_no_refusal_of_its_memory_to_catch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The allocator is chosen before torch is imported, and the refusals of the GPU's memory
    # are caught from there on: while torch isn't loaded there is none to catch, and the class
    # isn't asked of it (which would import it).
    assert cli._out_of_memory() == (torch.OutOfMemoryError,)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.delitem(sys.modules, "torch")
    assert cli._out_of_memory() == ()  # pyright: ignore[reportPrivateUsage]
    assert "torch" not in sys.modules


@pytest.mark.parametrize("spills", [False, True])
def test_gpu_memory_refused_in_a_run_stops_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    spills: bool,
) -> None:
    # A unit refused GPU memory, by the driver or by torch where seedvr2x holds it: said in one
    # line, no traceback, what is finished kept; the unit isn't recorded. Of torch's own line
    # (TORCH_REFUSED), the request, what torch held and its limit; not its advice, its link, nor
    # its count of the process's memory.
    if spills:
        spilling_driver(monkeypatch)
    entered: list[str] = []

    def enter(phase: str) -> None:
        entered.append(phase)
        if phase == "decode":
            # torch's refusal as it came; and one with no word at all.
            raise torch.OutOfMemoryError(TORCH_REFUSED if spills else "")

    monkeypatch.setattr(STAND_IN_MODELS, "enter", enter)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    held = (
        " (held to the card's own memory: its driver gives more): CUDA out of memory. Tried to"
        " allocate 456.00 MiB, torch holding 19.27 GiB allocated and 1.43 GiB more reserved,"
        " 20.98 GiB allowed"
        if spills
        else ": "
    )
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [
        f"the GPU's memory was refused{held}; stopped: 0 of 2 segments finished, and the units"
        " of the next ones kept: the same command resumes"
    ]
    assert all(record.exc_info is None for record in errors) and entered[-1] == "decode"


def test_host_short_of_memory_for_the_pinned_copies_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The host can't hold the pinned copies of the weights the plan has away from the GPU: the
    # load refuses, naming the RAM they take, logged without a traceback, before any unit, the
    # first pass kept as a plan.
    from seedvr2x.runtime import blockswap, model

    said = (
        "the weights that leave the GPU need 15.19 GiB of host memory, pinned (36 of the DiT's"
        " blocks), and the host has 9.50 GiB available"
    )

    def refused(*arguments: object) -> None:
        raise blockswap.HostMemoryError(said)

    monkeypatch.setattr(model, "load_models", refused)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--swap-blocks", "36") == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [said + kept(tmp_path / "out")]
    assert all(record.exc_info is None for record in errors)
    assert steps.calls == []


@pytest.mark.parametrize("output", ["out", "one.mkv"])
def test_plan_tiles_reach_the_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, output: str
) -> None:
    # A plan's tiles go to each encode and decode with the shot's own directory for their
    # scratch, and what the plan allows it of system memory: among its units, in an output
    # directory or in a .mkv path's work directory, gone once its file is assembled. A plan put
    # in the planner's place, with an allowance of its own, on frames its tiles don't cover
    # (LARGER).
    from seedvr2x.runtime import run

    tiled = Plan(5, Tiles(640), Tiles(512), scratch_allowance=12_345)
    plan = planner.plan
    monkeypatch.setattr(planner, "plan", lambda *job, **options: (tiled, plan(*job, **options)[1]))
    given: list[tuple[str, Tiles | None, Path | None]] = []
    work: list[bool] = []

    def encode(
        *arguments: Any, tiles: Tiles | None, scratch: Path | None, allowance: int, **options: Any
    ) -> Any:
        given.append(("encode", tiles, scratch))
        assert allowance == 12_345
        work.append((tmp_path / "one.mkv.work").is_dir())
        return steps.encode(*arguments, **options)

    def decode(*arguments: Any, tiles: Tiles | None, scratch: Path | None, allowance: int) -> None:
        given.append(("decode", tiles, scratch))
        assert allowance == 12_345
        steps.decode(*arguments)

    monkeypatch.setattr(run, "encode_shot", encode)
    monkeypatch.setattr(run, "decode_shot", decode)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, output, *JOB, *LARGER) == 0
    # A shot's directory is named for its first frame among the units.
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work") / "resume"
    shots = [directory / f"shot_{name:06d}" for name in (0, 3, 4)]
    order = [0, 1, 0, 1, 2, 2]
    kinds = ["encode", "encode", "decode", "decode", "encode", "decode"]
    assert given == [
        (kind, Tiles(640) if kind == "encode" else Tiles(512), shots[index])
        for kind, index in zip(kinds, order, strict=True)
    ]
    assert work == [output == "one.mkv"] * 3
    assert not (tmp_path / "one.mkv.work").exists()
    if output == "out":
        content = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert content["plan"]["encode_tiles"] == {"size": 640, "overlap": 64}
        assert content["plan"]["decode_tiles"] == {"size": 512, "overlap": 64}
        assert content["plan"]["memory"]["scratch_allowance"] == 12_345


def test_plan_said(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every run logs its plan once made, a line each (planner.said), and --plan prints the same,
    # before its counts (DESIGN.md, Memory planner, --plan).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    pins = Pins(window=5)
    made = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST)
    said = planner.said(*made, "a stand-in", TARGET, JOB_SHOTS, pins)
    # The disk's line (DESIGN.md, Memory planner, decided on 2026-10-28), after the scratch's
    # place and before the predicted peaks, its words and figures held by tests/test_disk.py:
    # its free room is this machine's, so the lines are compared without it.
    assert without_disk(plan_lines(caplog)) == said
    disk_at = [line.startswith("plan: disk: ") for line in plan_lines(caplog)].index(True)
    assert plan_lines(caplog)[disk_at + 1].startswith("plan: predicted peaks: ")
    assert said == [
        "plan: card: a stand-in, 94.50 GiB free, a budget of 93.90 GiB: the margin of 0.60 GiB"
        " left to the desktop and other programs; no reserve, the driver refusing memory the card"
        " doesn't have",
        "plan: frames: 128x96 padded to 128x128 for the model, 64 tokens a latent",
        "plan: attention: FlashAttention 2, which is installed",
        "plan: shots: 3, the longest of 21 frames (6 latents)",
        "plan: windows: 5 latents at most (17 frames), the longest run 4 latents, pinned by"
        " --window: 2 of 3 shots in one window, 1 join; a longer window, fewer joins; --window N"
        " caps them",
        "plan: quality: standard, the default, one run a shot; --quality enhanced or maximum runs"
        " 2 or 4 and takes each frame from the one that rebuilds it best, for 2 or 4 times the GPU"
        " time",
        "plan: blocks swapped: none, the window fitting with the DiT whole on the GPU;"
        " --swap-blocks N swaps some",
        "plan: the VAE: on the GPU in the DiT's phase, the window fitting beside it",
        "plan: the DiT in the VAE's passes: in the encode, the DiT's 36 blocks on the GPU; in the"
        " decode, the DiT's 36 blocks on the GPU; as much as fits beside each",
        "plan: system memory: 2.40 GiB in all; nothing pinned; 0.00 GiB for the longest shot at"
        " the most, as its windows are merged",
        "plan: encode: untiled, the whole frame fitting the budget; --encode-tile PX tiles it",
        "plan: decode: untiled, the whole frame fitting the budget; --decode-tile PX tiles it",
        "plan: the VAE's room: 1.25 GiB of the budget left free in each VAE pass, for the large"
        " pieces of memory a pass asks for at once",
        "plan: predicted peaks: encode 18.11 GiB, DiT 15.91 GiB, decode 17.73 GiB; the encode"
        " bounds the plan, its room counted, 74.54 GiB under the budget",
        "plan: time: measured, not guessed: a calibration of the job's own steps once the models"
        " are loaded, its cost said then; it gives the job's time and each unit's time left and"
        " end; --until HH:MM stops before a unit that wouldn't finish by then",
        "plan: pinned: --window 5, taken exactly",
    ]
    caplog.clear()
    plan = tmp_path / "plan.txt"
    with monkeypatch.context():
        # The models load, for the time estimates' calibration (DESIGN.md, Memory planner).
        assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(plan)) == 0
        assert without_disk(plan_lines(caplog)) == said
        counts = next(line for line in caplog.messages if line.endswith(f"the cut list in {plan}"))
        assert caplog.messages.index(said[-1]) < caplog.messages.index(counts)
        # For a .mkv path too, which --plan doesn't write: its cut list, and the job's plan in
        # its work directory.
        caplog.clear()
        steps.calls.clear()
        before = sorted(path.name for path in tmp_path.iterdir())
        listed = tmp_path / "other.txt"
        assert upscale(tmp_path, source_path, "other.mkv", *JOB[:4], "--plan", str(listed)) == 0
        assert without_disk(plan_lines(caplog)) == said and steps.calls == []
        made = sorted(path.name for path in tmp_path.iterdir())
        assert made == sorted([*before, "other.txt", "other.mkv.work"])
        assert listed.read_text() == plan.read_text()


def test_pin_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A pin is taken exactly or refused, never adjusted (DESIGN.md, Memory planner, Pins), the
    # refusal naming what fits, before the models load: no unit made, the first pass kept as a
    # plan, a manifest with no plan's record, which each command after it takes up, its pins
    # changed (DESIGN.md, Pause and resume, A plan).
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--swap-blocks", "37") == 1
    assert "--swap-blocks 37: the DiT has 36 blocks, so from 0 to 36" in caplog.text
    # A tile that covers the whole padded frame is no tiling (DESIGN.md, Pipeline, per shot, step
    # 3): refused, naming 0, the untiled pass, the stand-in's frames being a tile's worth at
    # most; the encode's first. With 0 for both the job runs, as it does without the pins.
    for pins, kind in (
        (("--decode-tile", "512"), "decode"),
        (("--encode-tile", "512", "--decode-tile", "1024"), "encode"),
    ):
        caplog.clear()
        assert upscale(tmp_path, source_path, "out", *JOB, *pins) == 1
        assert (
            f"--{kind}-tile 512: a tile of 512 px covers the whole padded frame, 128x128:"
            f" --{kind}-tile 0 pins an untiled {kind}"
        ) in caplog.text
        assert steps.calls == [] and "models" not in ASKED
        assert json.loads((tmp_path / "out" / "manifest.json").read_text())["plan"] is None
    untiled = ("--encode-tile", "0", "--decode-tile", "0")
    assert upscale(tmp_path, source_path, "untiled.mkv", *JOB[:4], *untiled) == 0
    steps.calls.clear()
    # With no block swapped, the memory that holds a window of 5 latents and not one of 6, the
    # VAE's weights away: --window 6, the longest shot in one window, is refused.
    free_memory(monkeypatch, planner.dit_peak(PADDED, 5, vae_in_dit=False) + planner.MARGIN)
    caplog.clear()
    pinned = ("--cuts", "cuts.txt", "--window", "6", "--swap-blocks", "0")
    assert upscale(tmp_path, source_path, "out", *pinned) == 1
    assert (
        "--window 6: its longest window, 6 latents, takes 16.07 GiB of free VRAM with"
        " --swap-blocks 0, and the driver reports 16.05 GiB free: the longest window that fits is"
        " --window 5"
    ) in caplog.text
    assert steps.calls == []
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["plan"] is None
    # Without the pin, the planner's own cap is that one, and the job runs: the 6-latent shot's
    # two windows are 4 latents long, and no cap is under 5 with a shot longer than it.
    assert upscale(tmp_path, source_path, "out", *pinned[:2], *pinned[4:]) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["plan"]["window"] == 5 and content["plan"]["memory"]["vae_in_dit"] is False
    assert content["plan"]["memory"]["longest_window"] == 4
    assert content["shots"][2]["windows"] == [[0, 4], [2, 6]]


def test_size_and_host_memory_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A size the free memory doesn't hold, and a plan the system memory doesn't, are refused
    # before the models load, as a pin is: exit 1, no unit made, the first pass kept as a plan,
    # its frame index and a manifest with no plan's record (DESIGN.md, Pause and resume, A plan:
    # decided on 2026-10-21, where nothing was written before), which the same command takes up.
    source_path = job(tmp_path, monkeypatch)
    there = sorted(path.name for path in tmp_path.iterdir())
    # The stand-in's frames are too small for a tile: their floor is the untiled encode's, with
    # the room a VAE pass is held to (planner.VAE_ROOM).
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    with monkeypatch.context() as patch:
        free_memory(patch, encode + planner.MARGIN - 1)
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert (
        "an output of 128x96 takes 4.62 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT off the GPU: a peak of 2.77 GiB, 1.25 of room a VAE pass is held to,"
        " and 0.60 of margin), and the driver reports 4.61 GiB free: the largest output that fits"
        " there is --resolution 80, about 106x80"
    ) in caplog.text
    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(cli, "_host_available", lambda: 2 * GIB)
        assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 1
    assert (
        "the plan takes 2.40 GiB of system memory (2.40 for the process) and 2.00 GiB are"
        " available, 0.40 GiB short"
    ) in caplog.text
    assert steps.calls == [] and "models" not in ASKED
    # The stand-in's model files, which upscale puts there, and each job's plan.
    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == sorted([*there, pull.DETECTOR, "w.safetensors", "out", "one.mkv.work"])
    for directory in (tmp_path / "out", tmp_path / "one.mkv.work"):
        assert sorted(path.name for path in directory.iterdir()) == [
            "frame_index.bin",
            "manifest.json",
        ]
        assert json.loads((directory / "manifest.json").read_text())["plan"] is None
    # With the memory back, both run.
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0


def test_one_file_plans_as_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A .mkv path's job keeps its units on disk, in its work directory, as an output
    # directory's does (cli._output): nothing of a shot waits on the GPU either way, and the two
    # plan alike, --plan's as its run's, the planner told nothing of which output it is. JOB's
    # last shot, 6 latents in two windows of 4: the DiT's phase takes a window's peak.
    asked: list[set[str]] = []
    backends: list[str] = []
    peaks: list[int] = []
    plan = planner.plan

    def planned(*arguments: Any, **options: Any) -> Any:
        asked.append(set(options))
        backends.append(options["attention"])
        made = plan(*arguments, **options)
        peaks.append(made[1].dit)
        return made

    monkeypatch.setattr(planner, "plan", planned)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    listed = str(tmp_path / "plan.txt")
    assert upscale(tmp_path, source_path, "other.mkv", *JOB[:4], "--plan", listed) == 0
    # --compile's setting is asked of the planner for every job (planner.plan's compile_vae),
    # which is no output's own: the three alike.
    assert asked == [{"split", "numz_padding", "recorded", "attention", "compile_vae"}] * 3
    # Each is planned for the attention backend its run takes (cli._attention, the stand-in's
    # here).
    assert backends == ["flash_attn_2"] * 3
    assert peaks == [planner.dit_peak(PADDED, 4)] * 3


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("output", ["one.mkv", "out"])
def test_a_run_keeps_a_shot_as_its_units_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    # Between two of a shot's units a run holds the shot as its units' copies alone (run._sample;
    # DESIGN.md, Memory planner, Per-phase peaks), for a .mkv path as for a directory. The latent
    # its windows are sampled from is the copy the units kept or wrote, no other, nor the file
    # read back; and a window's output is let go of once it is kept, before the next window is
    # asked for. The stand-in's device is the CPU, where a unit's copy is the tensor itself: the
    # helper that copies them (shot.moved) is replaced, for the units, by one that makes another
    # tensor, as it does from a GPU.
    from seedvr2x.runtime import run, units

    copies: list[torch.Tensor] = []
    loaded: list[str] = []
    from_a_copy: list[bool] = []
    outputs: list[weakref.ref[torch.Tensor]] = []
    alive: list[bool] = []
    moved, load = units.moved, units._load  # pyright: ignore[reportPrivateUsage]

    def copied(tensor: torch.Tensor, device: torch.device, what: str) -> torch.Tensor:
        copies.append(moved(tensor, device, what).clone())
        return copies[-1]

    def loading(path: Path) -> torch.Tensor:
        loaded.append(path.name)
        return load(path)

    def windows(
        models: object,
        latent: torch.Tensor,
        layout: list[tuple[int, int]],
        seed: int,
        start: int,
        *,
        numz_draw: bool = False,
    ) -> Iterator[torch.Tensor]:
        from_a_copy.append(any(latent is copy for copy in copies))
        made = stand_in_windows(models, latent, layout, seed, start)
        while True:
            # Asked for a window: what was given before is the run's no more.
            alive.append(any(output() is not None for output in outputs))
            given = [window for window in [next(made, None)] if window is not None]
            if not given:
                return
            outputs.append(weakref.ref(given[0]))
            yield given.pop()

    monkeypatch.setattr(units, "moved", copied)
    monkeypatch.setattr(units, "_load", loading)
    monkeypatch.setattr(run, "sample_windows", windows)
    source_path = job(tmp_path, monkeypatch)
    options = JOB[:4] if output == "one.mkv" else JOB
    assert upscale(tmp_path, source_path, output, *options) == 0
    # Three shots, four windows, the last shot's two: each shot's latent copied once, and its
    # windows sampled from that copy; each window copied once.
    assert from_a_copy == [True] * 3 and len(copies) == 3 + 4
    assert len(outputs) == 4 and alive == [False] * 4
    assert all(given() is None for given in outputs)
    # The decode reads each window back, and no latent in a run that encoded it: a .mkv path's
    # job as a directory's, its units in its work directory.
    assert loaded == [f"window_{index:04d}.pt" for index in (0, 0, 0, 1)]
    frames = tmp_path / "one.mkv" if output == "one.mkv" else tmp_path / "out" / "seg_000001.mkv"
    assert indexes(frames) == list(range(25) if output == "one.mkv" else range(4, 25))


def test_free_memory_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The free memory is read once a run, on the job's device, after the first pass, whose shot
    # detector has given its memory back, and before the models load (DESIGN.md, Memory planner,
    # Budget); a resume, which runs no first pass, reads it once too, and a finished job never.
    first_pass = cli._first_pass  # pyright: ignore[reportPrivateUsage]

    def passed(*arguments: Any) -> Any:
        found = first_pass(*arguments)
        ASKED.append("first pass")
        return found

    monkeypatch.setattr(cli, "_first_pass", passed)
    source_path = job(tmp_path, monkeypatch)
    options = JOB[2:]  # no cut list: the shot detector runs
    steps.stop = "window 0:1"
    stopped(tmp_path, source_path, "out", *options)
    reading = ("free memory", torch.device("cuda", 0))
    assert ASKED == ["first pass", reading, "models"]
    [built] = StandInDetector.built
    assert built.closed
    ASKED.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert ASKED == [reading, "models"]
    ASKED.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert ASKED == []


def test_resumed_keeps_its_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A job that has made a unit keeps the cap its windows were laid out by, with all the memory
    # there is: planned whole there, its longest shot would run in one window of 6 latents, as a
    # plan is (test_plan_planned_again_whole); resumed, it keeps its cap of 5 and its two
    # windows, and the manifest takes the day's memory with the next unit made.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    job_options = (*JOB[:2], *JOB[4:])  # JOB without its --window: the cap is the planner's
    options = (*job_options, "--swap-blocks", "0")
    # With no block swapped, the memory that holds a window of 5 latents and not one of 6.
    less = planner.dit_peak(PADDED, 5, vae_in_dit=False) + planner.MARGIN
    free_memory(monkeypatch, less)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    manifest = tmp_path / "out" / "manifest.json"
    started = json.loads(manifest.read_text())
    pins = Pins(swap_blocks=0)
    assert untimed(started["plan"]) == plan_record(
        *planner.plan(less, TARGET, JOB_SHOTS, pins, HOST)
    )
    assert (started["plan"]["window"], started["plan"]["memory"]["vae_in_dit"]) == (5, False)
    layout = [[[0, 2]], [[0, 1]], [[0, 4], [2, 6]]]
    assert [shot["windows"] for shot in started["shots"]] == layout
    assert started["segments"][0]["finished"] and not started["segments"][1]["finished"]
    free_memory(monkeypatch, FREE)
    assert planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST)[0] == Plan(6)
    steps.stop = "window 4:1"
    steps.calls.clear()
    caplog.clear()
    stopped(tmp_path, source_path, "out", *options)
    assert "re-planned" not in caplog.text
    assert (
        "plan: windows: 5 latents at most (17 frames), the longest run 4 latents, kept from the"
        " job's record, so that the output stays bit for bit"
    ) in caplog.text
    assert steps.calls == ["encode 4", "window 4:0", "window 4:1"]
    content = json.loads(manifest.read_text())
    again = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST, recorded=Plan(5))
    assert again[0] == Plan(5) and untimed(content["plan"]) == plan_record(*again)
    assert content["plan"]["memory"]["free"] == FREE != started["plan"]["memory"]["free"]
    assert content["plan"]["memory"]["vae_in_dit"] is True
    assert [shot["windows"] for shot in content["shots"]] == layout
    # Under another --swap-blocks it is the same job still, the pin no setting: taken, the shape
    # as recorded, and the job finishes.
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *job_options, "--swap-blocks", "36") == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    content = json.loads(manifest.read_text())
    pins = Pins(swap_blocks=36)
    again = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST, recorded=Plan(5))
    assert again[0] == Plan(5, swapped=36) and untimed(content["plan"]) == plan_record(*again)
    shape = {
        key: value for key, value in content["plan"].items() if key not in ("memory", "timings")
    }
    assert shape == {
        key: value for key, value in started["plan"].items() if key not in ("memory", "timings")
    }
    assert [shot["windows"] for shot in content["shots"]] == layout
    assert all(segment["finished"] for segment in content["segments"])
    out = tmp_path / "out"
    assert indexes(out / "seg_000000.mkv") + indexes(out / "seg_000001.mkv") == list(range(25))


def test_resumed_with_other_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A job that has made a unit keeps its plan's shape, the cap its windows were laid out by,
    # and plans where the weights are again from the memory free at each start (DESIGN.md,
    # Memory planner, A plan shapes the output a little): with less, blocks are swapped that
    # weren't, and the manifest takes the new memory with the next unit made, never before.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = JOB[:2]  # no --window: each shot in one window, the cap the longest's 6 latents
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    manifest = tmp_path / "out" / "manifest.json"
    started = json.loads(manifest.read_text())
    assert untimed(started["plan"]) == plan_record(
        *planner.plan(FREE, TARGET, JOB_SHOTS, Pins(), HOST)
    )
    assert (started["plan"]["window"], started["plan"]["memory"]["swapped_blocks"]) == (6, 0)
    # The memory that holds the 6-latent window with 20 blocks swapped and the VAE's weights
    # away, and neither with 19 nor with the VAE there.
    less = planner.dit_peak(PADDED, 6, 20, vae_in_dit=False) + planner.MARGIN
    assert planner.dit_peak(PADDED, 6, 19, vae_in_dit=False) + planner.MARGIN > less
    assert planner.dit_peak(PADDED, 6, 20) + planner.MARGIN > less
    free_memory(monkeypatch, less)
    kept = contents(tmp_path / "out")
    caplog.clear()
    stopped(tmp_path, source_path, "out", *options)  # before a unit: the record as it was
    assert contents(tmp_path / "out") == kept
    assert "plan: blocks swapped: 20 of the DiT's 36" in caplog.text
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    content = json.loads(manifest.read_text())
    again = planner.plan(less, TARGET, JOB_SHOTS, Pins(), HOST, recorded=Plan(6))
    assert (again[0].window, again[0].swapped, again[0].vae_in_dit) == (6, 20, False)
    assert untimed(content["plan"]) == plan_record(*again)
    shape = {
        key: value for key, value in content["plan"].items() if key not in ("memory", "timings")
    }
    assert shape == {
        key: value for key, value in started["plan"].items() if key not in ("memory", "timings")
    }
    assert content["plan"]["memory"]["free"] == less != started["plan"]["memory"]["free"]
    assert [shot["windows"] for shot in content["shots"]] == [[[0, 2]], [[0, 1]], [[0, 6]]]
    assert indexes(tmp_path / "out" / "seg_000000.mkv") == list(range(25))


def test_resumed_on_another_attention_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The plan's memory is planned again at each start for the attention backend of that
    # day's run (cli._attention), as for the memory free that day: a job begun with
    # FlashAttention 2 and taken up without it keeps its shape, and where its weights are is
    # counted by SDPA's fit, recorded with the next unit made. Here the memory that holds the
    # 6-latent window with every weight on the GPU by FlashAttention 2's fit: by SDPA's, the
    # VAE's weights leave the DiT's phase.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = JOB[:2]
    exact = planner.dit_peak(PADDED, 6) + planner.MARGIN
    free_memory(monkeypatch, exact)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    manifest = tmp_path / "out" / "manifest.json"
    started = json.loads(manifest.read_text())
    assert untimed(started["plan"]) == plan_record(
        *planner.plan(exact, TARGET, JOB_SHOTS, Pins(), HOST)
    )
    memory = started["plan"]["memory"]
    assert (memory["attention"], memory["swapped_blocks"], memory["vae_in_dit"]) == (
        "flash_attn_2",
        0,
        True,
    )
    monkeypatch.setattr(cli, "_attention", lambda: "sdpa")
    steps.stop = None
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    content = json.loads(manifest.read_text())
    again = planner.plan(exact, TARGET, JOB_SHOTS, Pins(), HOST, recorded=Plan(6), attention="sdpa")
    assert (again[0].window, again[0].swapped, again[0].vae_in_dit) == (6, 0, False)
    assert untimed(content["plan"]) == plan_record(*again)
    assert content["plan"]["memory"]["attention"] == "sdpa"
    shape = {
        key: value for key, value in content["plan"].items() if key not in ("memory", "timings")
    }
    assert shape == {
        key: value for key, value in started["plan"].items() if key not in ("memory", "timings")
    }
    assert "plan: attention: PyTorch's SDPA, FlashAttention 2 not being installed" in caplog.text
    assert indexes(tmp_path / "out" / "seg_000000.mkv") == list(range(25))


# The refusal's third way out, where --accept-env-change was not given (cli._kept_refusal;
# DESIGN.md, Memory planner, A plan shapes the output a little, decided on 2026-10-28).
THIRD_WAY = (
    "or run it again with --accept-env-change, which plans the rest of the job again for the"
    " memory free now, its output from there no longer the uninterrupted run's bit for bit"
)


def test_recorded_plan_no_longer_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A resume whose recorded shape no longer fits the memory free stops, naming the phase and
    # the shortfall (DESIGN.md, Memory planner), nothing changed on disk; the job resumes once
    # the memory is back. Here the untiled encode, a quarter of a GiB short with the DiT away
    # of its peak and the room a VAE pass is held to (planner.VAE_ROOM). Decided on 2026-10-28:
    # the refusal says first that the job is kept and the same command resumes it, then the phase
    # and what it lacks, then how to free it; the figures go under --verbose (a debug line).
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    kept = contents(tmp_path / "out")
    steps.stop = None
    steps.calls.clear()
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    free_memory(monkeypatch, encode + planner.MARGIN - GIB // 4)
    figures = (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT off the GPU, takes 4.62 GiB of free VRAM, 1.25 of them the room a VAE pass is held"
        " to, and the driver reports 4.36 GiB free, 0.26 GiB short"
    )
    words = [
        "the job is kept, nothing is lost, and the same command resumes it",
        "the encode lacks 0.26 GiB of the card's memory",
        f"to free it, {cli.FREE_THE_CARD}",
        THIRD_WAY,
    ]
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    (error,) = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert error.getMessage().split("\n") == words
    # Not the figures, which a default run never says: no debug line is let through.
    assert figures not in caplog.text and "NVML" not in error.getMessage()
    assert contents(tmp_path / "out") == kept and steps.calls == []
    # Under --verbose, which sets the level to debug, the figures are said, in a line of their own.
    caplog.clear()
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    (debug,) = [record for record in caplog.records if record.getMessage() == figures]
    assert debug.levelno == logging.DEBUG
    (error,) = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert error.getMessage().split("\n") == words and figures not in error.getMessage()
    caplog.set_level(logging.WARNING, logger="seedvr2x")
    free_memory(monkeypatch, FREE)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED[6:]


def test_a_recorded_plan_that_falls_short_of_the_systems_memory_is_kept_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # PROVISIONAL (impl-v010, 2026-10-28): DESIGN.md words the shortfall of the card's memory; the
    # system's own (planner, Host RAM) says the job is kept, then its figures in full, nothing
    # about the GPU's programs, whose quitting doesn't cure it.
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB[:4], "--swap-blocks", "36")
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    monkeypatch.setattr(cli, "_host_available", lambda: 1)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 1
    first, second, third = caplog.messages[-1].split("\n")
    assert first == "the job is kept, nothing is lost, and the same command resumes it"
    assert second.startswith("the recorded plan no longer fits the memory free now: the plan takes")
    assert "system memory" in second and "quit the programs" not in caplog.text
    assert third == THIRD_WAY


def long_shots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, int, int]:
    """Two shots of 57 frames, 15 latents each, and a job's two memories: where the DiT's phase
    holds windows of 9 latents (the shots in two windows) with every block swapped, and where it
    holds 7 (three windows). Each latent takes about 1.6 GiB there, the planner's per-token figure
    patched for it, the stand-in's frames being tiny: the encode and the decode fit both."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("57\n")
    monkeypatch.setattr(planner, "PER_TOKEN", planner.PER_TOKEN * 200)

    def holds(latents: int) -> int:
        return planner.dit_peak(PADDED, latents, planner.BLOCKS, False) + planner.MARGIN

    return source(tmp_path / "in.mkv", 114), holds(9), holds(7)


def test_a_resume_short_of_memory_is_planned_again_with_accept_env_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DESIGN.md, Memory planner, A plan shapes the output a little (decided on 2026-10-28): with
    # --accept-env-change a recorded shape that no longer fits is planned again for what is left.
    # A job stopped inside its second shot's windows, its first shot's done under a cap of 9
    # latents, resumed on memory that holds 7: refused without the flag, the refusal naming it;
    # with it, the first shot is kept as made, by its own layout, the second made again from its
    # first window under the new cap, and a stop and a resume after that need no flag.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    out, manifest = tmp_path / "out", tmp_path / "out" / "manifest.json"
    started = json.loads(manifest.read_text())
    assert started["plan"]["window"] == 9
    assert [(s["windows"], s["windows_done"]) for s in started["shots"]] == [
        ([[0, 9], [7, 15]], 2),
        ([[0, 9], [7, 15]], 1),
    ]
    assert all("window_cap" not in shot for shot in started["shots"])
    # Refused without the flag, the third way out named, nothing changed.
    free_memory(monkeypatch, narrow)
    kept = contents(out)
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 1
    (error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    words = error.getMessage().split("\n")
    assert words[1].startswith("the DiT's phase lacks ") and words[-1] == THIRD_WAY
    assert contents(out) == kept and steps.calls == []
    # With it: planned again, said in one warning, the manifest written with the first unit made.
    steps.stop = "window 57:2"
    caplog.clear()
    stopped(tmp_path, source_path, "out", *options, "--accept-env-change")
    assert steps.calls == ["window 57:0", "window 57:1", "window 57:2"]
    (said,) = [m for m in caplog.messages if "planned again for what is left" in m]
    assert "window cap 9 latents -> 7 latents" in said
    assert "kept: 0 segments finished, 2 runs encoded, 2 windows made and kept as they were" in said
    assert "made again, from their first: 1 window of runs begun under a window cap" in said
    # Only the window cap changed: the colour drift of smaller tiles is not claimed.
    assert said.endswith("from here the output is not the uninterrupted run's bit for bit")
    assert "tiles" not in said.split("\n")[-1] and "split" not in said
    # The plan's windows line counts the windows the job runs, the first shot's two under its own
    # cap of 9 and the second's three under 7: 0 of 2 shots in one window and 3 joins, where the
    # new plan's cap alone would count 4 (review, 2026-10-28).
    (line,) = [m for m in caplog.messages if m.startswith("plan: windows: ")]
    assert ": 0 of 2 shots in one window, 3 joins; " in line
    changed = json.loads(manifest.read_text())
    assert changed["plan"]["window"] == 7
    first, second = changed["shots"]
    # The first shot kept as made, by its own layout and cap; the second laid out by the new cap.
    assert (first["windows"], first["window_cap"], first["windows_done"]) == (
        [[0, 9], [7, 15]],
        9,
        2,
    )
    assert (
        second["windows"] == [list(w) for w in shot_layout(57, 7)] and len(second["windows"]) == 3
    )
    assert (second["encoded"], second["windows_done"]) == (True, 2) and "window_cap" not in second
    (entry,) = changed["environment_changes"]
    assert entry["before"] == {"plan": {"window": 9}} and entry["after"] == {"plan": {"window": 7}}
    assert [entry[k] for k in ("segments_finished", "shots_encoded", "windows_done")] == [0, 2, 3]
    assert (entry["windows_kept"], entry["windows_discarded"]) == (2, 1)
    assert not (out / "resume" / "shot_000000" / "window_0002.pt").exists()
    # A second resume takes the new shape as recorded and the mixed layouts as they are: no flag.
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == ["window 57:2", "decode 0", "decode 57"]
    # The next resume counts the same from the layouts it holds, and says the record kept.
    (line,) = [m for m in caplog.messages if m.startswith("plan: windows: ")][-1:]
    assert ": 0 of 2 shots in one window, 3 joins; " in line
    assert indexes(out / "seg_000000.mkv") == list(range(114))
    finished = json.loads(manifest.read_text())
    assert finished["shots"][0]["window_cap"] == 9 and finished["environment_changes"] == [entry]
    # Finished already, nothing made.
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0 and steps.calls == []


def test_a_finished_segment_and_a_latent_are_kept_when_a_shape_is_planned_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The kept units, byte for byte: a finished segment untouched, a shot begun under the old cap
    # made again from its first window, its latent kept (no second encode).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt", "--min-segment", "0")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    out = tmp_path / "out"
    first_segment = (out / "seg_000000.mkv").read_bytes()
    before = json.loads((out / "manifest.json").read_text())
    assert [s["finished"] for s in before["segments"]] == [True, False]
    free_memory(monkeypatch, narrow)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--accept-env-change") == 0
    assert steps.calls == ["window 57:0", "window 57:1", "window 57:2", "decode 57"]
    assert (out / "seg_000000.mkv").read_bytes() == first_segment
    after = json.loads((out / "manifest.json").read_text())
    assert after["segments"][0] == before["segments"][0]
    (entry,) = after["environment_changes"]
    assert [entry[k] for k in ("segments_finished", "shots_encoded", "windows_done")] == [1, 2, 3]
    assert (entry["windows_kept"], entry["windows_discarded"]) == (2, 1)
    assert indexes(out / "seg_000001.mkv") == list(range(57, 114))
    assert cli.main(["verify", str(out)]) == 0


def test_an_environment_change_accepted_with_a_shape_planned_again_is_recorded_with_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
) -> None:
    # Design's ruling of 2026-10-28 (every shape planned again of a job with units made): the
    # environment change and the shape are both written with the next unit made, the
    # environment's entry first, the run's environment with them. A unit made: the stop is at the
    # second window.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    free_memory(monkeypatch, narrow)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options, "--accept-env-change")
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["environment"]["gpu"] == "another" and content["plan"]["window"] == 7
    environment, shape = content["environment_changes"]
    assert environment["before"] == {"gpu": "a stand-in"} and "plan" not in environment["before"]
    assert shape["before"] == {"plan": {"window": 9}} and shape["after"] == {"plan": {"window": 7}}
    assert environment["windows_done"] == shape["windows_done"] == 3


def test_a_shape_planned_again_and_stopped_before_a_unit_leaves_the_job_as_it_was(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Design's ruling of 2026-10-28 (DESIGN.md, Memory planner, A plan shapes the output a
    # little): the new shape is written with the next unit made, as the new environment is. A run
    # stopped before one (Ctrl-C once, before the first unit) leaves the manifest as it was and
    # every window file, the begun ones included; the next start, the memory freed, keeps the
    # recorded shape and so the uninterrupted run's bits. Both entries, the environment's and
    # the shape's, are tried too.
    from seedvr2x.runtime.stop import Stop, Stopped

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    out, manifest = tmp_path / "out", tmp_path / "out" / "manifest.json"
    kept = contents(out)
    assert "resume/shot_000057/window_0000.pt" in kept
    free_memory(monkeypatch, narrow)

    def check(self: Stop, cost: float | None = None, unit: str = "the next unit") -> None:
        raise Stopped

    with monkeypatch.context() as asked:
        asked.setattr(Stop, "check", check)
        for gpu in ("a stand-in", "another"):
            asked.setattr(torch.cuda, "get_device_name", lambda device, gpu=gpu: gpu)
            steps.calls.clear()
            caplog.clear()
            assert upscale(tmp_path, source_path, "out", *options, "--accept-env-change") == 130
            (said,) = [m for m in caplog.messages if "planned again for what is left" in m]
            assert "and written with the next unit made" in said
            assert "window cap 9 latents -> 7 latents" in said
            # Nothing written, nothing discarded: the manifest and every window file as they were.
            assert steps.calls == [] and contents(out) == kept
    # The memory freed overnight: the next start, without the flag, keeps the recorded shape.
    free_memory(monkeypatch, wide)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == ["window 57:1", "decode 0", "decode 57"]
    assert "planned again" not in caplog.text
    finished = json.loads(manifest.read_text())
    assert finished["plan"]["window"] == 9 and finished["environment_changes"] == []
    assert all("window_cap" not in shot for shot in finished["shots"])
    assert indexes(out / "seg_000000.mkv") == list(range(114))


def test_a_shape_planned_again_discards_its_windows_only_as_its_record_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
) -> None:
    # The order that keeps the manifest true (cli._reshape): the begun window made again is
    # still there, named by the manifest, until the first unit's file is about to be kept; the
    # new record is then written, and the window discarded, before that file, which has its name.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    out, manifest = tmp_path / "out", tmp_path / "out" / "manifest.json"
    window = out / "resume" / "shot_000057" / "window_0000.pt"
    old = json.loads(manifest.read_text())
    assert old["shots"][1]["windows_done"] == 1 and window.is_file()
    free_memory(monkeypatch, narrow)
    from seedvr2x.runtime import units

    saved = units._save  # pyright: ignore[reportPrivateUsage]
    seen: list[tuple[int, int, bool]] = []

    def watched(tensor: torch.Tensor, path: Path) -> torch.Tensor:
        if path == window:
            now = json.loads(manifest.read_text())
            # Just before the made-again window is kept: the new record is on disk and the old
            # file gone.
            seen.append((now["plan"]["window"], now["shots"][1]["windows_done"], path.exists()))
        return saved(tensor, path)

    monkeypatch.setattr(units, "_save", watched)
    assert upscale(tmp_path, source_path, "out", *options, "--accept-env-change") == 0
    assert seen == [(7, 0, False)]


def test_plan_looks_at_a_shape_planned_again_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md is silent on --plan here): --plan runs nothing
    # and never destroys made work, so with --accept-env-change on a job short of memory it says
    # what a run would do, in one warning, and rewrites no manifest nor discards any window. Its
    # cut list is written as for any job.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    free_memory(monkeypatch, narrow)
    steps.stop = None
    steps.calls.clear()
    kept = contents(tmp_path / "out")
    caplog.clear()
    listing = tmp_path / "listed.txt"
    status = upscale(
        tmp_path, source_path, "out", *options, "--accept-env-change", "--plan", str(listing)
    )
    assert status == 0 and steps.calls == []
    assert contents(tmp_path / "out") == kept
    cuts = [line.split()[0] for line in listing.read_text().splitlines() if line[:1].isdigit()]
    assert cuts == ["57"]
    (said,) = [m for m in caplog.messages if "a run of the same command with" in m]
    assert said.split("\n")[1:] == [
        "it would keep: 0 segments finished, 2 runs encoded, 2 windows made and kept as they were",
        "it would make again: 1 window of runs begun under a window cap that no longer fits",
        "from here the output is not the uninterrupted run's bit for bit",
        "nothing was written, nothing was discarded",
    ]
    assert (
        "--accept-env-change plans the rest of the job again for it: window cap 9 latents -> 7"
        in said
    )
    # The same command without --plan then does it.
    assert upscale(tmp_path, source_path, "out", *options, "--accept-env-change") == 0
    assert steps.calls == ["window 57:0", "window 57:1", "window 57:2", "decode 0", "decode 57"]


def test_plan_without_the_flag_on_a_job_short_of_memory_says_the_refusal_a_run_gives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Design's ruling of 2026-10-28: --plan with --accept-env-change is as built (it looks, test
    # above); without the flag it gives the refusal a run gives, the flag named in it, adding where
    # its cut list is (written as for any job), and writes nothing of the job.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    steps.calls.clear()
    free_memory(monkeypatch, narrow)
    kept = contents(tmp_path / "out")
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 1
    (run_error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    caplog.clear()
    listing = tmp_path / "listed.txt"
    assert upscale(tmp_path, source_path, "out", *options, "--plan", str(listing)) == 1
    (plan_error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    said = plan_error.getMessage()
    assert said.startswith(run_error.getMessage())
    assert (
        said.removeprefix(run_error.getMessage()) == f"; its cut list is in {listing} all the same"
    )
    assert run_error.getMessage().split("\n")[-1] == THIRD_WAY
    assert "--accept-env-change" in plan_error.getMessage()
    assert contents(tmp_path / "out") == kept and steps.calls == []


def test_the_colour_drift_of_smaller_tiles_is_said_where_a_tile_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Looked at with --plan, the decode's tiles made smaller by the planner (replaced here: the
    # stand-in runs no tiled pass): the sentence on colour is said, only then.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    free_memory(monkeypatch, narrow)
    again = planner.plan_again

    def tiled(*job: Any, **named: Any) -> tuple[Plan, Peaks]:
        made, peaks = again(*job, **named)
        return replace(made, decode_tiles=Tiles(128)), peaks

    monkeypatch.setattr(planner, "plan_again", tiled)
    caplog.clear()
    listing = tmp_path / "listed.txt"
    assert (
        upscale(
            tmp_path, source_path, "out", *options, "--accept-env-change", "--plan", str(listing)
        )
        == 0
    )
    (said,) = [m for m in caplog.messages if "a run of the same command with" in m]
    assert "decode none -> tiles of 128 px" in said
    assert said.split("\n")[-2] == (
        "from here the output is not the uninterrupted run's bit for bit: smaller tiles drift"
        " slightly in colour, which --color-correction split corrects for the most part"
    )


def test_a_shape_the_command_line_pins_is_not_planned_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A pin is a setting: --window kept as given, so that a cap that no longer fits is refused
    # even with --accept-env-change, the refusal saying the shape was planned again and the job
    # is kept, nothing changed on disk.
    source_path, wide, narrow = long_shots(tmp_path, monkeypatch)
    options = ("--cuts", "cuts.txt", "--window", "9")
    free_memory(monkeypatch, wide)
    steps.stop = "window 57:1"
    stopped(tmp_path, source_path, "out", *options)
    kept = contents(tmp_path / "out")
    free_memory(monkeypatch, narrow)
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options, "--accept-env-change") == 1
    (error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    words = error.getMessage().split("\n")
    assert words[0] == "the job is kept, nothing is lost, and the same command resumes it"
    assert words[1].startswith("its shape planned again for the memory free now")
    assert "does not fit" in words[1] and words[-1] == f"to free it, {cli.FREE_THE_CARD}"
    assert contents(tmp_path / "out") == kept and steps.calls == []


def early_card(
    monkeypatch: pytest.MonkeyPatch, free: int | None, used: int = 10 * GIB
) -> list[int]:
    """Have NVML, behind memory.card_state, report `free` bytes free on the whole card and `used`
    held by programs from now on; returns the devices it is asked for, in order. None: a machine
    without it (conftest.py). `used` is large by default: whatever the shape lacks can be freed."""
    from seedvr2x.runtime import memory

    asked: list[int] = []

    def read(index: int) -> tuple[int, int] | str:
        asked.append(index)
        return "no NVML on this machine" if free is None else (free, used)

    monkeypatch.setattr(memory, "_read_all", read)
    return asked


def mark_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have the checks that come after a resume's record is read say so in the log when they run:
    a one-file output's rehearsal (cli._copied), the model files' hashes (pull.check_pinned) and
    the GPU's probe (memory.startup), which the early reading must not move."""
    from seedvr2x.runtime import memory

    log = logging.getLogger("seedvr2x")
    started = memory.startup
    copied, pinned = cli._copied, pull.check_pinned  # pyright: ignore[reportPrivateUsage]

    def rehearsed(*arguments: Any) -> None:
        log.warning("MARK the rehearsal")
        copied(*arguments)

    def hashed(*arguments: Any) -> None:
        log.warning("MARK the hashes")
        pinned(*arguments)

    def probed() -> str:
        log.warning("MARK the probe")
        return started()

    monkeypatch.setattr(cli, "_copied", rehearsed)
    monkeypatch.setattr(pull, "check_pinned", hashed)
    monkeypatch.setattr(memory, "startup", probed)


@pytest.mark.parametrize(("output", "minutes"), [("out", 2), ("one.mkv", 2)])
def test_a_resume_short_of_memory_is_warned_of_before_the_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    output: str,
    minutes: int,
) -> None:
    # DESIGN.md, Memory planner, A plan shapes the output a little (decided on 2026-10-28): as
    # soon as the record is read, before the rehearsal and the hashes, the room is read through
    # NVML alone; where the recorded shape doesn't fit, a warning says so at once, what to free,
    # and for about how long (the models' 2 min, a one-file output's 66 s more: memory's
    # CHECK_*). The check that refuses stays at the plan: the memory moved meanwhile here, and
    # the job goes on.
    source_path = job(tmp_path, monkeypatch)
    options = JOB[:4]
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, output, *options)
    steps.stop = None
    steps.calls.clear()
    from seedvr2x.runtime import memory

    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    # Short by a quarter of a GiB by NVML, this process's CUDA context counted off it.
    asked = early_card(monkeypatch, encode + planner.MARGIN - GIB // 4 + memory.CONTEXT)
    mark_checks(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, output, *options) == 0
    assert asked == [0]
    warned = [m for m in caplog.messages if "does not hold the recorded plan" in m]
    assert warned == [
        "the card's free memory, read now, does not hold the recorded plan: the encode lacks 0.26"
        " GiB of the card's memory; this is an early reading, the check that counts comes once the"
        " files are checked\n"
        f"to free it, {cli.FREE_THE_CARD}; the checks now running take about {minutes} min,"
        " within which to free it"
    ]
    marks = [m for m in caplog.messages if m.startswith("MARK")]
    # A one-file job's resume takes its rehearsal from its record, after the hashes
    # (cli._rehearsed), and counts none in its time.
    assert marks == ["MARK the hashes", "MARK the probe"]
    assert caplog.messages.index(warned[0]) < caplog.messages.index(marks[0])
    # The job goes on from where it stopped (a .mkv path's segments are the whole job's, not
    # --min-segment's).
    assert steps.calls[0] == "encode 4" and steps.calls[-1].startswith("decode")


def test_the_early_warning_given_accept_env_change_says_what_the_plan_then_does(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With --accept-env-change a resume still short at the plan is planned again for the memory
    # free, not refused: the early warning's second line says so (one line), still how to free
    # the card meanwhile and for about how long; its first line is the same.
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB[:4], "--accept-env-change")
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    steps.calls.clear()
    from seedvr2x.runtime import memory

    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    early_card(monkeypatch, encode + planner.MARGIN - GIB // 4 + memory.CONTEXT)
    mark_checks(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    warned = [m for m in caplog.messages if "does not hold the recorded plan" in m]
    assert warned == [
        "the card's free memory, read now, does not hold the recorded plan: the encode lacks 0.26"
        " GiB of the card's memory; this is an early reading, the check that counts comes once the"
        " files are checked\n"
        "the rest of the job is planned again for the memory free if it is still short at the"
        " plan (--accept-env-change), unless it is freed meanwhile, within the checks now running"
        f" (about 2 min): to free it, {cli.FREE_THE_CARD}"
    ]


@pytest.mark.parametrize("held", [False, True])
def test_the_early_warning_s_debug_line_says_nvml_s_reading_and_the_context_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    held: bool,
) -> None:
    # The early reading is NVML's figure less the context this process is yet to make: its debug
    # line says those two, and never that "its driver reports" a figure computed here (DESIGN.md,
    # Memory planner, A resume whose recorded shape no longer fits, 2026-10-29), where the driver
    # refuses and behind WSL2's device alike.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    steps.stop = None
    from seedvr2x.runtime import memory

    real = memory.early

    def early(*arguments: Any) -> Any:
        with monkeypatch.context() as patched:
            patched.setattr(memory, "paravirtual", lambda: held)
            return real(*arguments)

    monkeypatch.setattr(memory, "early", early)
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    nvml = encode + planner.MARGIN - GIB // 4 + memory.CONTEXT + (3 * GIB if held else 0)
    early_card(monkeypatch, nvml)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    [line] = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.DEBUG and r.getMessage().startswith("before the checks")
    ]
    assert re.search(
        r"NVML's \d+\.\d\d GiB free on the whole card, less 0\.75 GiB counted for this process's"
        r" CUDA context to come",
        line,
    ), line
    assert "its driver reports" not in line and "the driver reports" not in line


@pytest.mark.parametrize(
    ("rehearses", "source", "seconds"),
    [
        # The models' hashes alone, whatever the job: 2 min.
        (False, 0, 120),
        (True, 0, 120),
        (False, 1_000_000, 120),
        # Milestone 7's third night, a 7.49 GB source: hashed in 36 s, rehearsed in 66 s.
        (False, 7_490_000_000, 120 + 36),
        (True, 7_490_000_000, 120 + 36 + 66),
        # A source of 26 GB, 3.5 times that one: both scale, the models' time doesn't.
        (False, 26_000_000_000, 120 + 125),
        (True, 26_000_000_000, 120 + 125 + 229),
    ],
)
def test_the_checks_ahead_scale_with_the_source_s_size(
    tmp_path: Path, rehearses: bool, source: int, seconds: int
) -> None:
    # The early warning's minutes (cli._checks_ahead; DESIGN.md, Memory planner, A resume whose
    # recorded shape no longer fits, 2026-10-29): the source's hash and the remux rehearsal, where
    # it runs, by the source's size at the rates of milestone 7's third night, the models' 2 min
    # not. A sparse file stands for the source.
    path = tmp_path / "source.mkv"
    with path.open("wb") as written:
        written.truncate(source)
    args = SimpleNamespace(input=path)
    assert cli._checks_ahead(args, rehearses) == seconds  # pyright: ignore[reportPrivateUsage, reportArgumentType]


@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("cure", [False, True])
def test_the_early_warning_says_what_the_refusal_would_where_no_programs_cure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    accept: bool,
    held: bool,
    cure: bool,
) -> None:
    # DESIGN.md, Memory planner, A resume whose recorded shape no longer fits (2026-10-29): where
    # quitting programs can't free enough, the early warning says what the refusal would, that
    # the job needs a card with more memory free, instead of the advice to free the card and the
    # minutes to do it in. An early reading takes what quitting programs can give back for what
    # they hold of the card by NVML (planner._freeable): short by 0.26 GiB (1.7 held), with 4 GiB
    # held by programs the advice stands; with 0.1 GiB it doesn't. Through the real path.
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB[:4], *(["--accept-env-change"] if accept else []))
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *options)
    steps.stop = None
    steps.calls.clear()
    from seedvr2x.runtime import memory

    real = memory.early
    monkeypatch.setattr(memory, "early", lambda *a: _with_paravirtual(monkeypatch, held, real, *a))
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    nvml = encode + planner.MARGIN - GIB // 4 + memory.CONTEXT + (3 * GIB if held else 0)
    early_card(monkeypatch, nvml, 4 * GIB if cure else GIB // 10)
    mark_checks(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    [warned] = [m for m in caplog.messages if "does not hold the recorded plan" in m]
    first, second = warned.split("\n")
    assert first.endswith("the check that counts comes once the files are checked")
    if cure:
        assert cli.FREE_THE_CARD in second and "min" in second and cli.NOT_FREEABLE not in second
    elif accept:
        assert second == (
            "the rest of the job is planned again for the memory free if it is still short at the"
            f" plan (--accept-env-change); {cli.NOT_FREEABLE}"
        )
    else:
        assert second == cli.NOT_FREEABLE
    if not cure:
        assert "min" not in second and cli.FREE_THE_CARD not in second


def _with_paravirtual(
    monkeypatch: pytest.MonkeyPatch, held: bool, real: Callable[..., Any], *arguments: Any
) -> Any:
    """memory.early as behind WSL2's device (`held`) or where the driver refuses, for this call
    alone, the run's own probe left as the test has it."""
    from seedvr2x.runtime import memory

    with monkeypatch.context() as patched:
        patched.setattr(memory, "paravirtual", lambda: held)
        return real(*arguments)


def test_a_shortfall_no_programs_cure_says_so() -> None:
    # PROVISIONAL (impl-v010, 2026-10-28): where no memory freed on the card gives a held run's
    # shape its memory (planner._freeable), the refusal does not send the user to quit programs.
    error = planner.PlanError(
        "the figures", shortfall=planner.Shortfall("the decode", "1.01", freeable=False)
    )
    assert cli._kept_refusal(error).split("\n") == [  # pyright: ignore[reportPrivateUsage]
        "the job is kept, nothing is lost, and the same command resumes it",
        "the decode lacks 1.01 GiB of the card's memory",
        "quitting programs does not free that much on this card: the job needs a card with more"
        " memory free",
        THIRD_WAY,
    ]


def test_a_resume_warned_of_is_refused_at_its_place_after_the_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The warning is no refusal: the memory still short where the plan is made, after the hashes
    # and the first pass's comparison, the job is refused there, its refusal as ever.
    from seedvr2x.runtime import memory

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    steps.stop = None
    steps.calls.clear()
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    short = encode + planner.MARGIN - GIB // 4
    free_memory(monkeypatch, short)
    early_card(monkeypatch, short + memory.CONTEXT)
    mark_checks(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    messages = caplog.messages
    warned = next(i for i, m in enumerate(messages) if "does not hold the recorded plan" in m)
    hashed = messages.index("MARK the hashes")
    refused_at = next(i for i, m in enumerate(messages) if m.startswith("the job is kept"))
    assert warned < hashed < refused_at and steps.calls == []


def test_nothing_is_said_early_where_the_recorded_shape_fits_or_nothing_is_judged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No early warning where NVML reads room for the shape, where it reads nothing (a debug
    # line says so), for a new job, a plan (planned whole, no shape recorded) and a finished job
    # (which asks the card nothing): the card is read for none of the last three.
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    short = early_card(monkeypatch, 1 * GIB)  # a card far too small, were it read
    listed = tmp_path / "plan.txt"
    # A new job and a plan: nothing recorded to hold to.
    assert upscale(tmp_path, source_path, "planned", *JOB, "--plan", str(listed)) == 0
    assert upscale(tmp_path, source_path, "planned", *JOB) == 0
    assert upscale(tmp_path, source_path, "new", *JOB) == 0
    # A finished job.
    assert upscale(tmp_path, source_path, "new", *JOB) == 0
    assert short == [] and "does not hold the recorded plan" not in caplog.text
    # A resume with room by NVML, then without NVML.
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    steps.stop = None
    roomy = early_card(monkeypatch, FREE)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert roomy == [0] and "does not hold the recorded plan" not in caplog.text
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "again", *JOB)
    steps.stop = None
    early_card(monkeypatch, None)
    caplog.clear()
    assert upscale(tmp_path, source_path, "again", *JOB) == 0
    assert "does not hold the recorded plan" not in caplog.text
    assert "isn't read through NVML: not judged before the checks" in caplog.text


def test_plan_planned_again_whole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan, a job that has made no unit, is planned again whole from the memory free when it
    # is run (DESIGN.md, Memory planner: "A plan recorded with no unit made yet is made again
    # whole"): with less, another cap and so another layout, which is no other job; what a stop
    # left by the plan's own layout is discarded, and the manifest rewritten at once, said.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB[:2], "--swap-blocks", "0")
    assert upscale(tmp_path, source_path, "out", *options, "--plan", str(tmp_path / "p.txt")) == 0
    out = tmp_path / "out"
    planned = json.loads((out / "manifest.json").read_text())
    assert planned["plan"]["window"] == 6
    assert [shot["windows"] for shot in planned["shots"]] == [[[0, 2]], [[0, 1]], [[0, 6]]]
    index = (out / "frame_index.bin").read_bytes()
    # What a stop in shot 4's encode leaves.
    left = out / "resume" / "shot_000004"
    left.mkdir(parents=True)
    (left / "latent.pt.partial").write_bytes(b"left")
    # With no block swapped, the memory that holds a window of 5 latents and not one of 6.
    less = planner.dit_peak(PADDED, 5, vae_in_dit=False) + planner.MARGIN
    free_memory(monkeypatch, less)
    steps.stop = "encode 0"
    caplog.clear()
    stopped(tmp_path, source_path, "out", *options)  # before a unit: rewritten all the same
    said = "re-planned, its job having made no unit yet: plan.window: 6 -> 5; the windows of 1 shot"
    assert f"{out}: {said}; 1 leftover discarded" in caplog.text
    content = json.loads((out / "manifest.json").read_text())
    pins = Pins(swap_blocks=0)
    assert content["plan"] == plan_record(*planner.plan(less, TARGET, JOB_SHOTS, pins, HOST))
    assert content["plan"]["window"] == 5
    assert [shot["windows"] for shot in content["shots"]] == [[[0, 2]], [[0, 1]], [[0, 4], [2, 6]]]
    assert not any(shot["encoded"] or shot["windows_done"] for shot in content["shots"])
    assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
    assert (out / "frame_index.bin").read_bytes() == index
    # Run to its end in that memory: the new layout's windows; the same plan, nothing rewritten
    # before its first unit.
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert "re-planned" not in caplog.text
    assert steps.calls == [
        *("encode 0", "window 0:0", "encode 3", "window 3:0", "encode 4", "window 4:0"),
        *("window 4:1", "decode 0", "decode 3", "decode 4"),
    ]
    assert indexes(out / "seg_000000.mkv") == list(range(25))


def test_plan_without_its_record_planned_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan whose manifest lost its `plan`, deleted by hand, is planned again whole as every
    # plan is, whatever it recorded, and said in one line: no other job, and no KeyError; as a
    # run refused after its first pass leaves one, its plan made, none recorded.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    manifest = tmp_path / "out" / "manifest.json"
    planned = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({k: v for k, v in planned.items() if k != "plan"}))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    [said] = [line for line in caplog.messages if "re-planned" in line]
    assert said == (
        f"{tmp_path / 'out'}: re-planned, its job having made no unit yet: its plan made, none"
        " recorded; 0 leftovers discarded"
    )
    content = json.loads(manifest.read_text())
    # But the timings --plan's calibration recorded, for information: the run calibrates none.
    assert untimed(content["plan"]) == untimed(planned["plan"])
    assert steps.calls == UNINTERRUPTED


# A size a tile of 512 px doesn't cover, for the tests of the tiles: --resolution 528 of source's
# 64x48 is 704x528, padded to 704x560 (job.padded_size), which tiles of 512 px cut 2 x 2 and
# tiles of 640 px 1 x 2. The stand-in's frames otherwise are a tile's worth at most, where every
# tile pin is refused (test_pin_refused).
LARGER = ("--resolution", "528")
LARGER_TARGET = (528, 704)
LARGER_PADDED = (560, 704)
# The tile pins of the tests below, and the plan JOB then records at that size.
TILE_PINS = ("--encode-tile", "640", "--decode-tile", "512")
TILED_PLAN = Plan(5, Tiles(640), Tiles(512)).record()


def tiled_steps(
    monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> list[tuple[str, Tiles | None, Path | None, int]]:
    """Make the stand-in's encode and decode (steps) take a plan's tiles: each keeps a scratch in
    the directory the run gives it, on disk, as a tiled pass beyond its allowance does
    (tiling.Scratch), around the stand-in's own pass, untiled. Returns the list each pass adds
    its kind, its tiles, that directory and the allowance the run gave it to. The tiled passes
    themselves are test_tiling.py's and test_shot.py's."""
    from seedvr2x.runtime import run, tiling

    given: list[tuple[str, Tiles | None, Path | None, int]] = []

    def encode(
        *arguments: Any, tiles: Tiles | None, scratch: Path | None, allowance: int, **options: Any
    ) -> Any:
        given.append(("encode", tiles, scratch, allowance))
        assert scratch is not None and not (scratch / tiling.SCRATCH).exists()
        with tiling.Scratch(scratch, 0).store(1, (8, 8)):
            assert (scratch / tiling.SCRATCH).is_file()
            return steps.encode(*arguments, **options)

    def decode(*arguments: Any, tiles: Tiles | None, scratch: Path | None, allowance: int) -> None:
        given.append(("decode", tiles, scratch, allowance))
        assert scratch is not None and not (scratch / tiling.SCRATCH).exists()
        with tiling.Scratch(scratch, 0).store(1, (8, 8)):
            assert (scratch / tiling.SCRATCH).is_file()
            steps.decode(*arguments)

    monkeypatch.setattr(run, "encode_shot", encode)
    monkeypatch.setattr(run, "decode_shot", decode)
    return given


@pytest.mark.parametrize("output", ["out", "one.mkv"])
@pytest.mark.parametrize("split", [False, True], ids=["none", "split"])
def test_tile_pins_are_the_plan_and_no_scratch_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, output: str, split: bool
) -> None:
    # --encode-tile and --decode-tile are the plan's tiles, which reach each pass, with what
    # the plan allows its scratch of system memory, and are recorded; and of a scratch kept on
    # disk nothing stays once the job is done: the resume directory goes with the job, and a
    # .mkv path's work directory, which holds it, once the file is assembled, with split's
    # copies beside the scratch or without. At a size the tiles don't cover (LARGER).
    given = tiled_steps(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    work = tmp_path / "one.mkv.work"
    options = (*(JOB if output == "out" else JOB[:4]), *LARGER, *TILE_PINS)
    assert upscale(tmp_path, source_path, output, *options, *(SPLIT if split else ())) == 0
    kinds = ["encode", "encode", "decode", "decode", "encode", "decode"]
    if output == "one.mkv":
        # One segment without --min-segment: its three shots sampled before its decode.
        kinds = [*(["encode"] * 3), *(["decode"] * 3)]
    # The planner's plan around the pins, on frames padded to 704x560: the encode's tiles 1 x
    # 2 and the decode's 2 x 2, the scratch of the longest shot's 21 frames, and all of it
    # allowed in the system memory the stand-in's host has; the stand-in keeps a scratch of its
    # own, on disk, for what follows.
    pins = Pins(5, 640, 512)
    made = planner.plan(FREE, LARGER_TARGET, JOB_SHOTS, pins, HOST, split=split)
    kept = planner.scratch(LARGER_PADDED, 21)
    assert made[0] == Plan(5, Tiles(640), Tiles(512), scratch_allowance=kept)
    assert made[1].scratch == kept == 21 * 6 * 560 * 704
    lines = planner.said(*made, "a card", LARGER_TARGET, JOB_SHOTS, pins)
    encode = next(line for line in lines if line.startswith("plan: encode: "))
    decode = next(line for line in lines if line.startswith("plan: decode: "))
    assert encode.startswith("plan: encode: tiles of 640 px overlapping by 64, 2 across, 1 down, ")
    assert decode.startswith("plan: decode: tiles of 512 px overlapping by 64, 2 across, 2 down, ")
    assert [(kind, tiles, allowance) for kind, tiles, _, allowance in given] == [
        (kind, Tiles(640) if kind == "encode" else Tiles(512), kept) for kind in kinds
    ]
    assert not work.exists()
    if output == "out":
        out = tmp_path / "out"
        assert sorted(p.name for p in out.iterdir()) == [
            "checksums",
            "frame_index.bin",
            "manifest.json",
            "seg_000000.mkv",
            "seg_000001.mkv",
        ]
        recorded = json.loads((out / "manifest.json").read_text())["plan"]
        assert untimed(recorded) == plan_record(*made)
        assert (recorded["memory"]["scratch"], recorded["memory"]["scratch_allowance"]) == (
            kept,
            kept,
        )
        assert untimed(recorded) | {"memory": None} == TILED_PLAN | {"memory": None}
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(output)) == (
        ["out"] if output == "out" else ["one.mkv", "one.mkv.crc32", "one.mkv.manifest.json"]
    )


def test_tile_pins_are_the_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The tiles shape the output: a job is resumed with the pins it was made with, and refused
    # with others, or without. The pins are settings, which refuse it: a resume takes the plan's
    # tiles as recorded, whatever is pinned (planner.plan), so the plan itself never differs.
    tiled_steps(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    larger = (*JOB, *LARGER)
    stopped(tmp_path, source_path, "out", *larger, *TILE_PINS)
    steps.stop = None
    text = refused(tmp_path, source_path, caplog, *larger, *TILE_PINS[:2], "--decode-tile", "576")
    assert "settings.decode_tile: 512 -> 576" in text and "plan." not in text
    text = refused(tmp_path, source_path, caplog, *larger)
    assert "settings.encode_tile: 640 -> null" in text
    assert "settings.decode_tile: 512 -> null" in text and "plan." not in text
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *larger, *TILE_PINS) == 0
    assert steps.calls == ["window 4:1", "decode 4"]


def test_tile_pin_beyond_the_measured_sizes_warned_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A pass pinned in tiles larger than the largest measured (planner.TILE_MAX: 2048 px, which
    # the stand-in's frames don't reach, so 576 px here) is taken by its fit, and the log warns
    # of it once per pass, after the plan's lines (planner.beyond_measured): of the encode's 640
    # px, not of the decode's 512; again on the resume, which takes the tiles as recorded; and
    # of nothing while the pins are within the sizes measured.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    tiled_steps(monkeypatch, steps)
    source_path = job(tmp_path, monkeypatch)

    def warned() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING and "sizes measured" in record.getMessage()
        ]

    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *LARGER, *TILE_PINS) == 0
    assert (
        warned() == [] and len(plan_lines(caplog)) == 18
    )  # a line a choice (planner.said), the disk's too
    monkeypatch.setattr(planner, "TILE_MAX", 576)
    said = (
        "the encode in tiles of 640 px: its predicted peak is the fit's, beyond the sizes"
        " measured (up to 576 px), and the pass may take more"
    )
    caplog.clear()
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *LARGER, *TILE_PINS)
    assert warned() == [said]
    assert caplog.messages.index(plan_lines(caplog)[-1]) + 1 == caplog.messages.index(said)
    steps.stop = None
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *LARGER, *TILE_PINS) == 0
    assert warned() == [said]


def test_scratch_left_by_a_kill_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A run killed in a tiled pass leaves its scratch in the shot's directory (tiling.SCRATCH),
    # derived data the manifest never names: the resume discards it, and keeps the shot's units.
    from seedvr2x.runtime import tiling

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    shot = tmp_path / "out" / "resume" / "shot_000004"
    scratch = shot / tiling.SCRATCH
    scratch.write_bytes(b"left")
    caplog.set_level(logging.DEBUG, logger="seedvr2x")
    steps.calls.clear()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    assert f"discarded {scratch}" in caplog.text and "; 1 leftover discarded" in caplog.text
    assert steps.calls == ["window 4:1", "decode 4"]
    assert sorted(p.name for p in shot.iterdir()) == ["window_0000.pt", "window_0001.pt"]
    # In a finished segment's directory too, which goes whole.
    stale = tmp_path / "out" / "resume" / "shot_000000" / tiling.SCRATCH
    stale.parent.mkdir()
    stale.write_bytes(b"left")
    scratch.write_bytes(b"left")
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert not (tmp_path / "out" / "resume").exists()
    assert indexes(tmp_path / "out" / "seg_000001.mkv") == list(range(4, 25))


@pytest.mark.parametrize("output", ["out", "one.mkv"])
def test_scratch_that_cant_be_made_stops_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    output: str,
) -> None:
    # A scratch's file that can't be made, on a full disk, is the user's to fix: the run says so
    # and stops, exit status 1, as for any file it can't write; the units made before are kept,
    # and the same command resumes once there is room, which the message says, as a stop's does,
    # for a .mkv path as for a directory.
    from seedvr2x.runtime import run
    from seedvr2x.runtime.tiling import ScratchError

    source_path = job(tmp_path, monkeypatch)
    said = "scratch.bf16: the shot's scratch, 12718080 bytes: [Errno 28] No space left on device"
    decode = steps.decode

    def full(*arguments: Any, **tiled: Any) -> None:
        if int(arguments[1][0, 0, 0, 0]) == 4:
            raise ScratchError(said)
        decode(*arguments, **tiled)

    monkeypatch.setattr(run, "decode_shot", full)
    options = JOB if output == "out" else JOB[:4]
    assert upscale(tmp_path, source_path, output, *options) == 1
    assert caplog.text.count(said) == 1 and "Traceback" not in caplog.text
    # A .mkv path's units are kept in its work directory, its three shots one segment's.
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work")
    done = "1 of 2" if output == "out" else "0 of 1"
    kept = (
        f"{done} segments finished, and the units of the next ones kept: the same command resumes"
    )
    assert f"{said}; stopped: {kept}" in caplog.messages
    content = json.loads((directory / "manifest.json").read_text())
    finished = [True, False] if output == "out" else [False]
    assert [segment["finished"] for segment in content["segments"]] == finished
    assert [shot["windows_done"] for shot in content["shots"]] == [1, 1, 2]
    monkeypatch.setattr(run, "decode_shot", decode)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, output, *options) == 0
    if output == "one.mkv":
        assert steps.calls == ["decode 0", "decode 3", "decode 4"]
        assert indexes(tmp_path / "one.mkv") == list(range(25)) and not directory.exists()
        return
    assert steps.calls == ["decode 4"]
    assert indexes(tmp_path / "out" / "seg_000001.mkv") == list(range(4, 25))


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
@pytest.mark.parametrize("correction", ["none", "split"])
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
        "checksums",
        "frame_index.bin",
        "manifest.json",
        "seg_000000.mkv",
        "seg_000001.mkv",
    ]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)
    assert contents(out / "checksums") == contents(tmp_path / "whole" / "checksums")
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
    ("stop", "resumed"),
    [
        # After the first segment's unit: its master and checksums kept.
        ("encode 4", UNINTERRUPTED[6:]),
        # Inside the last segment's decode: its master and checksums written again whole.
        ("decode 4", UNINTERRUPTED[9:]),
    ],
)
def test_default_format_resumed_as_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, stop: str, resumed: list[str]
) -> None:
    # The default format, yuv420p10le (no --format, which upscale would add), stopped then
    # resumed: the masters' frames, decoded as stored, and their checksums are the uninterrupted
    # job's, byte for byte.
    from seedvr2x.media.checksums import frame_bytes
    from seedvr2x.media.probe import probe

    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    common = ["--model-dir", str(tmp_path), "--dit-model", "w.safetensors"]
    common += ["--vae-model", "w.safetensors", "--resolution", "96", "--seed", str(SEED)]
    common += [*JOB, "--color-correction", "none"]

    def run(output: str) -> int:
        return cli.main([str(source_path), "-o", str(tmp_path / output), *common])

    assert run("whole") == 0
    steps.calls.clear()
    steps.stop = stop
    assert run("out") == 130
    assert steps.calls[-1] == stop
    steps.calls.clear()
    steps.stop = None
    assert run("out") == 0
    assert steps.calls == resumed
    for name, frames in (("seg_000000.mkv", 4), ("seg_000001.mkv", 21)):
        stored_frames = []
        for output in ("whole", "out"):
            master = tmp_path / output / name
            assert probe(master).pix_fmt == "yuv420p10le"
            stored_frames.append(
                subprocess.run(
                    [
                        *("ffmpeg", "-v", "error", "-i", str(master)),
                        *("-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"),
                    ],
                    capture_output=True,
                    check=True,
                ).stdout
            )
        assert len(stored_frames[0]) == frames * frame_bytes("yuv420p10le", 128, 96)
        assert stored_frames[1] == stored_frames[0]
    sums = contents(tmp_path / "whole" / "checksums")
    assert sorted(sums) == ["seg_000000.crc32", "seg_000001.crc32"]
    assert contents(tmp_path / "out" / "checksums") == sums


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
@pytest.mark.parametrize("correction", ["none", "split"])
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
    # a resume makes it again, as an uninterrupted run does. With split, the shot's input copy and
    # its checksums, whole before its latent is checked: discarded unrecorded, else kept.
    options = (*JOB, "--color-correction", correction)
    if correction == "split":
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
    # A .mkv path keeps what a directory does, in its work directory: stopped, no file yet, the
    # units before kept, and the same command resumes, then assembles.
    source_path = job(tmp_path, monkeypatch)
    steps.poison = "window 3:0"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 1
    assert "shot 2/3's window 1/1: " in caplog.text
    assert "so not recorded; stopped: 0 of 1 segments finished" in caplog.text
    assert "the same command resumes" in caplog.text
    work = tmp_path / "one.mkv.work"
    assert sorted(p.name for p in work.iterdir()) == ["frame_index.bin", "manifest.json", "resume"]
    assert sorted(p.name for p in (work / "resume").iterdir()) == ["shot_000000", "shot_000003"]
    assert not (tmp_path / "one.mkv").exists()
    steps.poison = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert steps.calls == [
        *("window 3:0", "encode 4", "window 4:0", "window 4:1"),
        *("decode 0", "decode 3", "decode 4"),
    ]
    assert indexes(tmp_path / "one.mkv") == list(range(25)) and not work.exists()


SPLIT = ("--color-correction", "split")


def test_split_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # split unless asked otherwise: a setting, which a resume compares.
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    common = ["--model-dir", str(tmp_path), "--dit-model", "w.safetensors"]
    common += ["--vae-model", "w.safetensors", "--resolution", "96", *JOB]
    # upscale's format: the resume below differs by its colour correction alone.
    common += ["--format", "gbrp16le"]
    assert cli.main([str(source_path), "-o", str(tmp_path / "out"), *common]) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["color_correction"] == "split"
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--color-correction", "none") == 1
    assert 'settings.color_correction: "split" -> "none"' in caplog.text and steps.calls == []
    assert "settings.format" not in caplog.text


@pytest.mark.parametrize("split", [True, False])
def test_a_split_job_is_planned_with_its_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, split: bool
) -> None:
    # The command line hands the planner the job's colour correction: a job that corrects its
    # colours is planned with the correction in its decode's phase, a slice of the output
    # corrected beside an untiled decode (planner.decode_peak's split), and one that doesn't,
    # without. The stand-in's decode is untiled, where the term adds to the peak: the plan
    # recorded says which was counted. A tiled decode's peak is the same with and without it.
    source_path = job(tmp_path, monkeypatch)
    options = (*JOB, *SPLIT) if split else JOB
    assert upscale(tmp_path, source_path, "out", *options) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["color_correction"] == ("split" if split else "none")
    assert content["plan"]["decode_tiles"] is None
    plain = planner.decode_peak(PADDED, TARGET)
    corrected = planner.decode_peak(PADDED, TARGET, split=True)
    slice_of_frames = planner.SPLIT_FRAME_4K * planner.SLICE * 96 * 128 / planner.PIXELS_4K
    assert corrected - plain == pytest.approx(slice_of_frames, abs=1) and corrected > plain + 2**20
    assert content["plan"]["memory"]["decode_peak"] == (corrected if split else plain)


def test_split_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # The frames each encode reads, copied as they come, exactly, and recorded with its latent.
    from seedvr2x.media.decode import Decoder, to_float32
    from seedvr2x.media.ffmpeg import input_args
    from seedvr2x.runtime.shot import COPY_READ

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
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


def test_split_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # What a kill leaves of split's is discarded: a copy being written, its checksums. A copy
    # recorded is kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
    out = tmp_path / "out"
    shot = out / "resume" / "shot_000004"
    for name in ("input.mkv.partial", "input.crc32.partial"):
        (shot / name).write_bytes(b"left")
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert "2 leftovers discarded" in caplog.text and "made again" not in caplog.text
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
def test_split_copy_missing(
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
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    steps.stop = stop
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
    out = tmp_path / "out"
    (out / "resume" / "shot_000004" / missing).unlink()
    seen: list[str] = []
    decode = steps.decode

    def looked(*arguments: Any, **tiled: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments, **tiled)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert steps.calls == resumed
    assert f"shot 3/3: {missing} missing: its input copy made again from the input" in caplog.text
    assert seen == [steps.read[4]]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def damage(path: Path, how: str) -> None:
    """Damage a shot's input copy or its checksums (Steps.damage, test_split_copy_damaged): a byte
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
def test_split_copy_damaged(
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
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
    out = tmp_path / "out"
    copy = out / "resume" / "shot_000004" / "input.mkv"
    checksums = copy.with_name("input.crc32")
    damage(checksums if how.startswith("checksums") else copy, how)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 1
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
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert steps.calls == ["decode 4"]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def test_split_copy_damaged_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A .mkv path's copies are among its units, in its work directory: stopped, the copy named
    # and removed, made again by the same command, which then assembles the file.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    steps.damage = "decode 4"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 1
    copy = tmp_path / "one.mkv.work" / "resume" / "shot_000004" / "input.mkv"
    assert f"{copy}: the shot's input copy, not readable whole and intact: " in caplog.text
    assert "slice CRC mismatch" in caplog.text
    assert "; removed, made again from the input on resuming; stopped: 0 of 1 segments" in (
        caplog.text
    )
    assert not copy.exists() and not (tmp_path / "one.mkv").exists()
    steps.damage = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 0
    assert steps.calls == ["decode 0", "decode 3", "decode 4"]
    assert "shot 3/3: input.mkv and input.crc32 missing: its input copy made again" in caplog.text
    assert not (tmp_path / "one.mkv.work").exists()


def test_split_copy_checked_before_its_last_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A shot's last frames are written once its copy is read whole and checked: they may finish
    # the segment, which is then recorded, so a copy failing at the very end of its read leaves
    # the segment unfinished, decoded again whole by the next run.
    from seedvr2x.media.decode import Decoder

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    finish = Decoder.finish
    finished: list[Decoder] = []

    def failing(self: Decoder) -> None:
        if self._strict:  # pyright: ignore[reportPrivateUsage]
            finished.append(self)
            if len(finished) == 2:  # shot 2/3's copy, the last of segment 1
                raise self.failure("a stand-in failure at the end")
        finish(self)

    monkeypatch.setattr(Decoder, "finish", failing)
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 1
    assert "a stand-in failure at the end; removed" in caplog.text
    assert "stopped: 0 of 2 segments finished" in caplog.text
    out = tmp_path / "out"
    content = json.loads((out / "manifest.json").read_text())
    assert [segment["finished"] for segment in content["segments"]] == [False, False]
    assert not (out / "seg_000000.mkv").exists()
    assert not (out / "resume" / "shot_000003" / "input.mkv").exists()
    monkeypatch.setattr(Decoder, "finish", finish)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert steps.calls == ["decode 0", "decode 3", *UNINTERRUPTED[6:]]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


@pytest.mark.parametrize("presses", [1, 2])
def test_split_copy_again_stopped(
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
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
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
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 130
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
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert steps.calls == ["window 4:1", "decode 4"]
    assert ("made again from the input" in caplog.text) == (presses == 2)
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        out = tmp_path / "out" / name
        assert decoded(out) == decoded(tmp_path / "whole" / name)


def test_split_copy_again_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A copy made again replaces checksums written before, which go first: a stop between the
    # copy and its checksums leaves the copy without any, so made again by the next run, never
    # beside checksums that aren't its own.
    from seedvr2x.runtime import run

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB, *SPLIT) == 0
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB, *SPLIT)
    copy = tmp_path / "out" / "resume" / "shot_000004" / "input.mkv"
    checksums = copy.with_name("input.crc32")
    copy.unlink()
    written = run.write_checksums

    def killed(path: Path, values: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "write_checksums", killed)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 130
    assert steps.calls == [] and copy.exists() and not checksums.exists()
    monkeypatch.setattr(run, "write_checksums", written)
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert "shot 3/3: input.crc32 missing: its input copy made again" in caplog.text
    assert steps.calls == ["window 4:1", "decode 4"]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        out = tmp_path / "out" / name
        assert decoded(out) == decoded(tmp_path / "whole" / name)


def test_split_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A .mkv path's copies go among its units, in its work directory beside it, each shot's kept
    # until its segment is finished; the directory is kept at a stop or an error, for the same
    # command to resume, and removed once the file is assembled. Anything else there is refused,
    # and kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    one, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    seen: list[list[str]] = []
    decode = steps.decode

    def looked(*arguments: Any, **tiled: Any) -> None:
        seen.append(sorted(path.name for path in (work / "resume").iterdir()))
        decode(*arguments, **tiled)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 0
    # One segment without --min-segment: its three shots' copies wait for its decode.
    assert seen == [["shot_000000", "shot_000003", "shot_000004"]] * 3
    assert not work.exists()
    out = tmp_path / "out"
    assert decoded(one) == decoded(out / "seg_000000.mkv") + decoded(out / "seg_000001.mkv")
    for stop, poison, status in (("decode 4", None, 130), (None, "decode 4", 1)):
        # A new job to the file: the file and the manifest beside it removed, the same command
        # on a finished file making nothing (DESIGN.md, Output, Assembly, decided on 2026-10-12).
        one.unlink()
        (tmp_path / "one.mkv.manifest.json").unlink()
        steps.stop, steps.poison = stop, poison
        assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == status
        assert not one.exists()
        copies = sorted(str(path.relative_to(work)) for path in work.rglob("input.mkv"))
        assert copies == [f"resume/shot_{start:06d}/input.mkv" for start in (0, 3, 4)]
        steps.stop = steps.poison = None
        assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 0
        assert not work.exists()
        assert decoded(one) == decoded(out / "seg_000000.mkv") + decoded(out / "seg_000001.mkv")
    # Not this job's: refused with a manifest there, and without one; never deleted.
    one.unlink()
    (tmp_path / "one.mkv.manifest.json").unlink()
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT)
    steps.stop = None
    (work / "notes.txt").write_text("mine")
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 1
    assert f"{work}: not this job's: notes.txt" in caplog.text
    (work / "manifest.json").unlink()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 1
    assert f"{work}: not empty, and no manifest.json to resume from; {one}'s job keeps" in (
        caplog.text
    )
    assert (work / "notes.txt").read_text() == "mine"


def test_none_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # With none, nothing is copied (DESIGN.md, Colour correction): no shot of a .mkv path's job
    # gets an input copy in its work directory; each decode goes uncorrected.
    source_path = job(tmp_path, monkeypatch)
    work = tmp_path / "one.mkv.work"
    seen: list[tuple[bool, list[str], object]] = []
    decode = steps.decode

    def looked(*arguments: Any, **tiled: Any) -> None:
        copies = [path.name for path in tmp_path.rglob("input.*")]
        seen.append((work.exists(), copies, arguments[-1]))  # the last, its correction
        decode(*arguments, **tiled)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], "--color-correction", "none") == 0
    assert seen == [(True, [], None)] * 3
    assert indexes(tmp_path / "one.mkv") == list(range(25)) and not work.exists()


def test_split_frame_by_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # Each slice of the decode corrected as it comes out, each frame against its own frame of the
    # reference: the copy's frames through the encoder's transform in float32 (DESIGN.md, Colour
    # correction, Numerics), its padding cropped as the decode's (target 100 x 133, decoded 128 x
    # 160: 12 rows and 11 columns reflected, 16 black; written 100 x 132); the colour at the
    # stages of x2.08 (48 x 64 to 100 x 133: 3). The content split receives is the decode as it
    # comes out (milestone 5: on the same decode), unclamped (DESIGN.md, Beyond numz's lab): the
    # stand-in's frames get a pattern by channel and position, past ±1 in places, the padding
    # still NaN, so that a channel moved, a frame flipped or a value clamped on the way changes
    # the output.
    from seedvr2x.media.decode import to_float32
    from seedvr2x.media.source import examine
    from seedvr2x.runtime import colour, model
    from seedvr2x.runtime.job import target_size
    from seedvr2x.runtime.shot import pad_4n1

    source_path = job(tmp_path, monkeypatch)
    # Uniform noise from -0.5 to 2.5, the same on each frame: the stand-in's frames, -1 to -0.95,
    # then go from -1.5 to 1.55.
    pattern = torch.rand(3, 1, 128, 160, generator=torch.Generator().manual_seed(0)) * 3 - 0.5
    stream = model.decode_stream  # the stand-in's (stand_in_model)

    def patterned(models: object, latent: torch.Tensor) -> Iterator[torch.Tensor]:
        for frames in stream(models, latent):  # pyright: ignore[reportArgumentType]
            yield frames + pattern

    monkeypatch.setattr(model, "decode_stream", patterned)
    # gbrp16le: its frames are read below as stored, three 16-bit planes.
    options = (*JOB[:4], *SPLIT, "--resolution", "100", "--format", "gbrp16le")
    assert upscale(tmp_path, source_path, "one.mkv", *options) == 0
    height, width = 100, 132
    planes = np.frombuffer(decoded(tmp_path / "one.mkv"), dtype="<u2")
    written = planes.reshape(25, 3, height, width)[:, [2, 0, 1]]  # G, B, R to R, G, B
    with examine(source_path).decoder() as decoder:
        frames = to_float32(decoder.read(25))
    target = target_size(64, 48, Fraction(1), 100)
    assert target == (100, 133) and output_size(target) == (height, width)
    stages = colour.colour_stages((48, 64), target)
    assert stages == 3
    transform = model.input_transform(target)
    expected: list[torch.Tensor] = []
    for start, end in ((0, 3), (3, 4), (4, 25)):
        count = end - start
        reference = transform(pad_4n1(torch.from_numpy(frames[start:end])).permute(0, 3, 1, 2))
        assert reference.dtype == torch.float32
        assert reference.shape[2:] == (128, 160) == padded_size(target)
        reference = reference[:, :count, :height, :width].permute(1, 0, 2, 3)
        values = torch.tensor([(start + frame) / 500 - 1 for frame in range(count)])
        content = values.view(-1, 1, 1, 1) + pattern[:, 0, :height, :width]
        assert content.min() < -1 and content.max() > 1
        # The decode's slices: 5 frames, then 4, the last cut to the shot's end.
        sizes = [min(5, count), *(min(4, count - frame) for frame in range(5, count, 4))]
        for a, b in pairwise(accumulate(sizes, initial=0)):
            expected.append(colour.split(content[a:b], reference[a:b], stages))
    codes = (torch.cat(expected) * 65535).round().numpy()
    assert np.array_equal(codes, written.astype(np.float32))


@pytest.mark.usefixtures("stand_in")
def test_split_streams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The decode streams (DESIGN.md, Colour correction): one pass per shot, each slice corrected
    # and written as it comes out, never a file beside the shot's units but its input copy and
    # their checksums; its last frames once the copy is read whole and checked.
    from seedvr2x.media.decode import Decoder
    from seedvr2x.runtime import model, run, shot

    source_path = job(tmp_path, monkeypatch)
    state = tmp_path / "out" / "resume"
    events: list[str] = []
    seen: set[str] = set()

    def look() -> None:
        seen.update(path.name for path in state.rglob("*") if not path.is_dir())

    stream = model.decode_stream  # the stand-in's (stand_in_model)

    def streamed(models: object, latent: torch.Tensor) -> Iterator[torch.Tensor]:
        events.append(f"decode {int(latent[0, 0, 0, 0])}")
        for frames in stream(models, latent):  # pyright: ignore[reportArgumentType]
            look()
            events.append(f"slice {frames.shape[1]}")
            yield frames

    finish = Decoder.finish

    def finished(self: Decoder) -> None:
        if self._strict:  # pyright: ignore[reportPrivateUsage]
            events.append("copy checked")
        finish(self)

    def decode(*arguments: Any, **tiled: Any) -> None:
        models, merged, count, target, write, *names = arguments

        def written(frames: npt.NDArray[np.float32]) -> None:
            look()
            events.append(f"write {frames.shape[0]}")
            write(frames)

        shot.decode_shot(models, merged, count, target, written, *names, **tiled)

    monkeypatch.setattr(model, "decode_stream", streamed)
    monkeypatch.setattr(Decoder, "finish", finished)
    monkeypatch.setattr(run, "decode_shot", decode)
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    assert events == [
        *("decode 0", "slice 5", "copy checked", "write 3"),
        *("decode 3", "slice 1", "copy checked", "write 1"),
        *("decode 4", "slice 5", "write 5", "slice 4", "write 4", "slice 4", "write 4"),
        *("slice 4", "write 4", "slice 4", "copy checked", "write 4"),
    ]
    assert seen == {"input.mkv", "input.crc32", "window_0000.pt", "window_0001.pt"}


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


def altered(path: Path, frame: int) -> None:
    """Make the stand-in's gbrp16le segment at path again whole, a pixel of `frame` altered: a
    valid file, which ffmpeg decodes without a word, its frame not as written."""
    from seedvr2x.media.decode import Decoder, to_float32
    from seedvr2x.media.ffmpeg import input_args
    from seedvr2x.media.writer import FFV1Writer, Tags
    from seedvr2x.runtime.shot import COPY_READ

    with Decoder(input_args(path), COPY_READ, 128, 96) as decoder:
        frames = decoder.read(1000)
    frames[frame, 0, 0, 0] ^= 1
    with FFV1Writer(path, "gbrp16le", 128, 96, Fraction(25), Tags()) as writer:
        writer.write(to_float32(frames))


def test_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # verify decodes each finished segment and checks every frame against its checksums: a
    # segment not finished isn't checked; a frame not as written, or checksums missing, fail it.
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    out = tmp_path / "out"
    caplog.set_level(logging.INFO, logger="seedvr2x")
    # A segment unfinished: the output isn't to be trusted whole yet.
    assert cli.main(["verify", str(out)]) == 1
    assert f"{out / 'seg_000000.mkv'}: every frame as written" in caplog.text
    assert f"{out / 'seg_000001.mkv'}: unfinished, not checked" in caplog.text
    assert f"{out}: of 2, 1 unfinished" in caplog.text
    # One segment alone, its checksums in the directory's.
    assert cli.main(["verify", str(out / "seg_000000.mkv")]) == 0
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    altered(out / "seg_000001.mkv", 2)
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 1
    assert f"{out / 'seg_000001.mkv'}: 1 frames not as written: 2" in caplog.text
    assert f"{out}: of 2, 1 not as written" in caplog.text
    (out / "checksums" / "seg_000000.crc32").unlink()
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 1
    assert f"{out / 'checksums' / 'seg_000000.crc32'}: missing" in caplog.text
    assert f"{out}: of 2, 2 not as written" in caplog.text
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    altered(tmp_path / "one.mkv", 20)
    caplog.clear()
    assert cli.main(["verify", str(tmp_path / "one.mkv")]) == 1
    assert "1 frames not as written: 20" in caplog.text
    (tmp_path / "empty").mkdir()
    assert cli.main(["verify", str(tmp_path / "empty")]) == 1
    assert "empty: no manifest.json, so not an output directory of seedvr2x" in caplog.text
    content = json.loads((out / "manifest.json").read_text())
    del content["segments"]
    (out / "manifest.json").write_text(json.dumps(content))
    assert cli.main(["verify", str(out)]) == 1
    assert "manifest.json: not a manifest seedvr2x wrote" in caplog.text


def test_one_file_checksums_kept_until_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # The checksums of a finished file go only when a new file replaces it: a run to the same
    # file stopped before keeps the old file and its checksums. The old file here is the one a
    # run killed after putting it in place left, its job's work directory beside it: a file with
    # no manifest at all is refused, and the same command on a finished file makes nothing
    # (DESIGN.md, Output, Assembly, decided on 2026-10-12).
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole.mkv", *JOB[:4]) == 0
    master, sums = tmp_path / "one.mkv", tmp_path / "one.mkv.crc32"
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "one.mkv", *JOB[:4])
    master.write_bytes((tmp_path / "whole.mkv").read_bytes())
    sums.write_bytes((tmp_path / "whole.mkv.crc32").read_bytes())
    before = (master.read_bytes(), sums.read_bytes())
    stopped(tmp_path, source_path, "one.mkv", *JOB[:4])
    assert (master.read_bytes(), sums.read_bytes()) == before
    assert cli.main(["verify", str(master)]) == 0


def test_checksums_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A finished segment's checksums are written before it is recorded: missing, the resume is
    # refused, as it is for anything in checksums/ that isn't this job's. An unfinished segment's,
    # and partial files, are discarded.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    steps.stop = "decode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    sums = tmp_path / "out" / "checksums"
    assert sorted(p.name for p in sums.iterdir()) == ["seg_000000.crc32"]
    kept = (sums / "seg_000000.crc32").read_bytes()
    (sums / "seg_000000.crc32").unlink()
    steps.stop = None
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert (
        "seg_000000.crc32: a finished segment's checksums, the manifest says, but missing" in text
    )
    (sums / "seg_000000.crc32").write_bytes(kept)
    (sums / "notes.txt").write_text("mine")
    assert f"{sums / 'notes.txt'}: not this job's" in refused(tmp_path, source_path, caplog, *JOB)
    (sums / "notes.txt").unlink()
    (sums / "seg_000001.crc32").mkdir()
    assert "seg_000001.crc32: not this job's" in refused(tmp_path, source_path, caplog, *JOB)
    (sums / "seg_000001.crc32").rmdir()
    sums.rename(tmp_path / "elsewhere")
    sums.symlink_to(tmp_path / "elsewhere")
    assert f"{sums}: not this job's" in refused(tmp_path, source_path, caplog, *JOB)
    sums.unlink()
    (tmp_path / "elsewhere").rename(sums)
    for name in ("seg_000001.crc32", "seg_000001.crc32.partial", "seg_000000.crc32.partial"):
        (sums / name).write_bytes(b"left")
    caplog.set_level(logging.INFO, logger="seedvr2x")
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert "3 leftovers discarded" in caplog.text and steps.calls == ["decode 4"]
    assert contents(sums) == contents(tmp_path / "whole" / "checksums")


def contents(directory: Path) -> dict[str, bytes | None]:
    """Every path under directory, with a file's bytes."""
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in sorted(directory.rglob("*"))
    }


def refused(
    tmp_path: Path,
    source_path: Path,
    caplog: pytest.LogCaptureFixture,
    *options: str,
    output: str = "out",
) -> str:
    caplog.clear()
    assert upscale(tmp_path, source_path, output, *options) == 1
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
    assert "input.sha256" in text
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
    # Missing what the manifest names (a finished segment missing is no longer refused: made
    # again, test_a_finished_segment_missing_or_changed_is_made_again_at_a_start).
    window = out / "resume" / "shot_000004" / "window_0000.pt"
    window.rename(tmp_path / "window.pt")
    assert "kept, the manifest says, but missing" in refused(tmp_path, source_path, caplog, *JOB)
    (tmp_path / "window.pt").rename(window)
    # Another manifest version: 1, from before resume, or one above this code's, of newer code,
    # refused for it, naming both (manifest.READ).
    written = (out / "manifest.json").read_bytes()
    for version in (1, 4):
        content = {**json.loads(written), "seedvr2x_manifest": version}
        (out / "manifest.json").write_text(json.dumps(content))
        text = refused(tmp_path, source_path, caplog, *JOB)
        assert f"manifest version {version}, where this seedvr2x writes 3" in text
    (out / "manifest.json").write_bytes(written)
    assert contents(out) == kept  # nothing touched
    # The job asked, at last.
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == ["window 4:1", "decode 4"]


def test_numz_padding_for_tests_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # numz's padding (cli.NUMZ_PADDING), which holds the GPU tests' output to numz's: set, every
    # encode gets it, the output is the same (the stand-in's decode crops numz's padding then),
    # and the job's settings record it, so that a resume in the other padding is refused; unset,
    # the settings don't name it.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "ours", *JOB) == 0
    assert steps.numz_padding == [False] * 3
    ours = json.loads((tmp_path / "ours" / "manifest.json").read_text())
    assert "numz_padding" not in ours["settings"]
    monkeypatch.setenv(cli.NUMZ_PADDING, "1")
    steps.numz_padding.clear()
    assert upscale(tmp_path, source_path, "numz", *JOB) == 0
    assert steps.numz_padding == [True] * 3
    assert f"{cli.NUMZ_PADDING}=1: numz's padding, for seedvr2x's tests only" in caplog.text
    numz = json.loads((tmp_path / "numz" / "manifest.json").read_text())
    assert numz["settings"] == {**ours["settings"], "numz_padding": True}
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "numz" / name) == decoded(tmp_path / "ours" / name)
    # Every encode of the other paths too: a .mkv path's, run in its work directory and
    # assembled, and split, each encode copying the frames it reads (run._sample).
    for output, options in (("numz.mkv", JOB[:4]), ("numz_split", (*JOB, *SPLIT))):
        steps.numz_padding.clear()
        assert upscale(tmp_path, source_path, output, *options) == 0
        assert steps.numz_padding == [True] * 3, output
    # A job stopped in one padding isn't resumed in the other.
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "numz_stopped", *JOB)
    monkeypatch.delenv(cli.NUMZ_PADDING)
    assert "settings.numz_padding: true -> null" in refused(
        tmp_path, source_path, caplog, *JOB, output="numz_stopped"
    )
    stopped(tmp_path, source_path, "ours_stopped", *JOB)
    monkeypatch.setenv(cli.NUMZ_PADDING, "1")
    assert "settings.numz_padding: null -> true" in refused(
        tmp_path, source_path, caplog, *JOB, output="ours_stopped"
    )
    # Anything else than 1 is refused, set but empty too.
    for value in ("yes", ""):
        monkeypatch.setenv(cli.NUMZ_PADDING, value)
        text = refused(tmp_path, source_path, caplog, *JOB, output="other")
        assert f"{cli.NUMZ_PADDING}={value!r}: for seedvr2x's tests only, 1 or unset" in text


def test_numz_draw_for_tests_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # numz's draw of the noise, on the GPU (cli.NUMZ_DRAW), which holds the GPU tests' output to
    # numz's references: set, every shot's windows get it, and the job's settings record it, so
    # that a resume in the other draw is refused; unset, the settings don't name it. Apart from
    # numz's padding (cli.NUMZ_PADDING): each its own variable.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "ours", *JOB) == 0
    assert steps.numz_draw == [False] * 3
    ours = json.loads((tmp_path / "ours" / "manifest.json").read_text())
    assert "numz_draw" not in ours["settings"]
    monkeypatch.setenv(cli.NUMZ_DRAW, "1")
    steps.numz_draw.clear()
    assert upscale(tmp_path, source_path, "numz", *JOB) == 0
    assert steps.numz_draw == [True] * 3 and steps.numz_padding[-3:] == [False] * 3
    said = f"{cli.NUMZ_DRAW}=1: numz's draw of the noise, on the GPU, for seedvr2x's tests only"
    assert said in caplog.text
    numz = json.loads((tmp_path / "numz" / "manifest.json").read_text())
    assert numz["settings"] == {**ours["settings"], "numz_draw": True}
    # With numz's padding too, both recorded.
    monkeypatch.setenv(cli.NUMZ_PADDING, "1")
    assert upscale(tmp_path, source_path, "both", *JOB) == 0
    both = json.loads((tmp_path / "both" / "manifest.json").read_text())
    assert both["settings"] == {**ours["settings"], "numz_draw": True, "numz_padding": True}
    monkeypatch.delenv(cli.NUMZ_PADDING)
    # The other paths too: a .mkv path's, run in its work directory and assembled, and split.
    for output, options in (("numz.mkv", JOB[:4]), ("numz_split", (*JOB, *SPLIT))):
        steps.numz_draw.clear()
        assert upscale(tmp_path, source_path, output, *options) == 0
        assert steps.numz_draw == [True] * 3, output
    # A job stopped in one draw isn't resumed in the other.
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "numz_stopped", *JOB)
    monkeypatch.delenv(cli.NUMZ_DRAW)
    assert "settings.numz_draw: true -> null" in refused(
        tmp_path, source_path, caplog, *JOB, output="numz_stopped"
    )
    stopped(tmp_path, source_path, "ours_stopped", *JOB)
    monkeypatch.setenv(cli.NUMZ_DRAW, "1")
    assert "settings.numz_draw: null -> true" in refused(
        tmp_path, source_path, caplog, *JOB, output="ours_stopped"
    )
    # Anything else than 1 is refused, set but empty too.
    for value in ("yes", ""):
        monkeypatch.setenv(cli.NUMZ_DRAW, value)
        text = refused(tmp_path, source_path, caplog, *JOB, output="other")
        assert f"{cli.NUMZ_DRAW}={value!r}: for seedvr2x's tests only, 1 or unset" in text


def test_numz_tail_for_tests_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # numz's padding of a shot's end, mirrored (cli.NUMZ_TAIL), which holds the GPU tests' output
    # to numz's references where a shot is not 4n + 1 frames long: set, every encode gets it, and
    # the job's settings record it, so that a resume in the other tail is refused; unset, the
    # settings don't name it. Apart from numz's padding and draw (cli.NUMZ_PADDING,
    # cli.NUMZ_DRAW): each its own variable.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "ours", *JOB) == 0
    assert steps.numz_tail == [False] * 3
    ours = json.loads((tmp_path / "ours" / "manifest.json").read_text())
    assert "numz_tail" not in ours["settings"]
    monkeypatch.setenv(cli.NUMZ_TAIL, "1")
    steps.numz_tail.clear()
    assert upscale(tmp_path, source_path, "numz", *JOB) == 0
    assert steps.numz_tail == [True] * 3
    assert steps.numz_padding[-3:] == [False] * 3 and steps.numz_draw[-3:] == [False] * 3
    said = f"{cli.NUMZ_TAIL}=1: numz's padding of a shot's end, for seedvr2x's tests only"
    assert said in caplog.text
    numz = json.loads((tmp_path / "numz" / "manifest.json").read_text())
    assert numz["settings"] == {**ours["settings"], "numz_tail": True}
    # With numz's padding and draw too, the three recorded.
    monkeypatch.setenv(cli.NUMZ_PADDING, "1")
    monkeypatch.setenv(cli.NUMZ_DRAW, "1")
    assert upscale(tmp_path, source_path, "all", *JOB) == 0
    every = json.loads((tmp_path / "all" / "manifest.json").read_text())
    assert every["settings"] == {
        **ours["settings"],
        **{"numz_padding": True, "numz_draw": True, "numz_tail": True},
    }
    monkeypatch.delenv(cli.NUMZ_PADDING)
    monkeypatch.delenv(cli.NUMZ_DRAW)
    # The other paths too: a .mkv path's, run in its work directory and assembled, and split.
    for output, options in (("numz.mkv", JOB[:4]), ("numz_split", (*JOB, *SPLIT))):
        steps.numz_tail.clear()
        assert upscale(tmp_path, source_path, output, *options) == 0
        assert steps.numz_tail == [True] * 3, output
    # A job stopped in one tail isn't resumed in the other.
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "numz_stopped", *JOB)
    monkeypatch.delenv(cli.NUMZ_TAIL)
    assert "settings.numz_tail: true -> null" in refused(
        tmp_path, source_path, caplog, *JOB, output="numz_stopped"
    )
    stopped(tmp_path, source_path, "ours_stopped", *JOB)
    monkeypatch.setenv(cli.NUMZ_TAIL, "1")
    assert "settings.numz_tail: null -> true" in refused(
        tmp_path, source_path, caplog, *JOB, output="ours_stopped"
    )
    # Anything else than 1 is refused, set but empty too.
    for value in ("yes", ""):
        monkeypatch.setenv(cli.NUMZ_TAIL, value)
        text = refused(tmp_path, source_path, caplog, *JOB, output="other")
        assert f"{cli.NUMZ_TAIL}={value!r}: for seedvr2x's tests only, 1 or unset" in text


# The model check, which the stand-in replaces (stand_in_model), as the module was imported.
CHECK_MODELS = weights.check_models


def test_models_checked_before_the_resume_compares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The model files are checked by their headers before a resume compares the job with its
    # record, which hashes them (cli._model; 16 GB for the DiT): a 3B as the DiT, refused unhashed.
    from test_weights import V1, as_dtype, write

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_model", lambda *a: pytest.fail("the model files hashed"))
    write(tmp_path / "w.safetensors", as_dtype("3b", "F16"))
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert f"w.safetensors: SeedVR2's 3B DiT in fp16: {V1}" in text


@pytest.mark.usefixtures("stand_in")
def test_model_replaced_before_its_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A model file replaced after its check, during the first pass: load_models' own check
    # refuses it, as the first one would have, logged without a traceback, the first pass kept
    # as a plan, its plan's record with it.
    from seedvr2x.runtime import model

    said = "w.safetensors: SeedVR2's 3B DiT in fp16"

    def replaced(*args: object) -> None:
        raise weights.ModelError(said)

    monkeypatch.setattr(model, "load_models", replaced)
    assert upscale(tmp_path, job(tmp_path, monkeypatch), "out", *JOB) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [said + kept(tmp_path / "out")]
    assert all(record.exc_info is None for record in errors)
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["plan"] is not None


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
    # The job is the third GPU's now. Finished, it is said finished on the first one too, no
    # flag asked, where it was refused as a resume is (design, 2026-10-12): and the first GPU
    # isn't asked its name, a finished job taking none, so that nothing is said of it
    # (test_finished_job_in_another_environment).
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: pytest.fail("asked"))
    finished = manifest.read_bytes()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert "finished already, its 2 segments" in caplog.text
    assert "its environment differs" not in caplog.text
    assert manifest.read_bytes() == finished


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
    entry = json.loads((tmp_path / "out" / "manifest.json").read_text())["input"]
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

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

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
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
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
    assert json.loads(recorded)["input"]["primaries"] == ""  # untagged
    probe = examined.probe
    monkeypatch.setattr(
        examined, "probe", lambda path: replace(probe(path), color_primaries="bt709")
    )
    assert 'input.primaries: "" -> "bt709"' in refused(tmp_path, source_path, caplog, *JOB)
    assert manifest.read_bytes() == recorded


@pytest.mark.parametrize("inputs", ["list", "record"])
def test_version_2_job_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    inputs: str,
) -> None:
    # A job of older code, its manifest at version 2, which this code reads (manifest.READ), and
    # its own code: its input a list of one per input file with its start in the job, as before
    # the source was the only input (a directory of segments was one: DESIGN.md, Input), or the
    # source's record already. Compared, it is another job, refused as one, exit 1, its
    # differences listed, nothing discarded; verify checks its segments, recorded as now.
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    out = tmp_path / "out"
    content = json.loads((out / "manifest.json").read_text())
    content["seedvr2x_manifest"] = 2
    content["settings"]["code"] = "0" * 64
    if inputs == "list":
        entry = content["input"]
        content["input"] = [{**entry, "start": 0, "frames": 3}, {**entry, "start": 3, "frames": 22}]
        content["settings"]["min_segment"] = None
    (out / "manifest.json").write_text(json.dumps(content))
    kept = contents(out)
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert "another job than the one asked" in text and "--accept-env-change" not in text
    assert "settings.code" in text
    assert ('input: [{"bytes": ' in text) == (inputs == "list")
    assert contents(out) == kept
    caplog.set_level(logging.INFO, logger="seedvr2x")
    caplog.clear()
    assert cli.main(["verify", str(out)]) == 1
    assert f"{out / 'seg_000000.mkv'}: every frame as written" in caplog.text
    assert f"{out / 'seg_000001.mkv'}: unfinished, not checked" in caplog.text


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
    monkeypatch.setattr(
        examined, "scan", lambda path, *detector: replace(scan(path, *detector), frames=29)
    )
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    text = refused(tmp_path, source_path, caplog, *JOB, "--accept-env-change")
    assert "input.frames: 25 -> 29" in text
    assert manifest.read_bytes() == recorded


@pytest.mark.usefixtures("stand_in")
def test_frames_left_after_the_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A job ends with the source's end checked (run.Inputs): its decoder finished, ffmpeg done
    # with no error after the frames the first pass counted. A frame more than counted fails it.
    from seedvr2x.media import source as examined

    scan = examined.scan

    def missed(path: Path, *detector: Any) -> object:
        found = scan(path, *detector)
        return replace(found, frames=found.frames - 1)

    monkeypatch.setattr(examined, "scan", missed)
    assert upscale(tmp_path, source(tmp_path / "in.mkv"), "one.mkv") == 1
    assert f"decoding with ffmpeg: frames left after {FRAMES - 1}" in caplog.text
    assert f"{tmp_path / 'in.mkv'} gave at least a frame more than the {FRAMES - 1}" in caplog.text


def understating(monkeypatch: pytest.MonkeyPatch) -> None:
    """From now on, a first pass that misses the source's last frame: its count one short, and
    its frame index a row, so that the record is one a resume trusts. What a source that gives
    a frame more at a read than it gave the first pass looks like to the run."""
    from seedvr2x.media import source as examined

    scan = examined.scan

    def missed(path: Path, *detector: Any) -> object:
        found = scan(path, *detector)
        index = found.index
        assert index is not None and index.probabilities is None
        short = replace(index, pts=index.pts[:-1], crc32=index.crc32[:-1], error=index.error[:-1])
        return replace(found, frames=found.frames - 1, index=short)

    monkeypatch.setattr(examined, "scan", missed)


def frames_left(source_path: Path, counted: int, kept: str) -> str:
    """The one line of a run stopped at the end of a source that gave a frame more than the
    `counted` of its first pass (reader.FramesLeft; design, 2026-10-12): what happened, the way
    out, another -o, then what is `kept`, and never that the same command resumes."""
    return (
        f"decoding with ffmpeg: frames left after {counted}: {source_path} gave at least a frame"
        f" more than the {counted} the first pass counted, and the job's record holds that count,"
        " so the same command would stop here again: run the job to another -o, whose first pass"
        f" counts the frames again; stopped: {kept}"
    )


def errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The run's errors, each message whole."""
    return [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]


def end_checks(monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    """From now on, each check of the source's end (Reader.finish) is recorded among the steps'
    calls, "end checked", where it comes."""
    from seedvr2x.media.reader import Reader

    finish = Reader.finish

    def checked(reader: Reader) -> None:
        steps.calls.append("end checked")
        finish(reader)

    monkeypatch.setattr(Reader, "finish", checked)


@pytest.mark.parametrize(
    ("output", "correction"),
    [("out", "none"), ("one.mkv", "none"), ("out", "split")],
)
def test_end_of_source_checked_before_the_last_shot_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    output: str,
    correction: str,
) -> None:
    # The source's end is checked as its last frame is read (run.Inputs), before anything made
    # of that read is recorded: the last shot's latent, so its windows and its segment. A source
    # giving a frame more than the first pass counted fails the run there, the last shot not
    # encoded and the last segment unfinished, where the check came after the last segment was
    # recorded: a directory's job was then finished, the next run saying so, exit 0, and a .mkv
    # path's next run assembled a file a frame short. The same command reads the shot again and
    # fails again, for as long as the source gives that frame: no file, no "finished already".
    # With split, the shot's input copy, recorded with its latent, isn't kept either.
    # Its one line names the source, says what happened and the way out, another -o, whose
    # first pass counts again, then what is kept; it no longer ends "the same command resumes",
    # which stops there again (design, 2026-10-12).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    understating(monkeypatch)
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work")
    kept = "1 of 2 segments finished, and the units of the next ones kept"
    made = [*("encode 0", "window 0:0", "encode 3", "window 3:0", "decode 0", "decode 3")]
    for calls in (made, []):
        steps.calls.clear()
        caplog.clear()
        options = (*JOB, "--color-correction", correction)
        assert upscale(tmp_path, source_path, output, *options) == 1
        assert errors(caplog) == [frames_left(source_path, 24, kept)]
        assert "the same command resumes" not in caplog.text
        # The read of the last shot's frames fails, its encode given none: nothing kept of it.
        assert steps.calls == calls and sorted(steps.read) == [0, 3]
        content = json.loads((directory / "manifest.json").read_text())
        assert content["input"]["frames"] == 24
        assert [segment["finished"] for segment in content["segments"]] == [True, False]
        assert [shot["encoded"] for shot in content["shots"]] == [True, True, False]
        assert "assembled" not in content and "finished already" not in caplog.text
        assert not (tmp_path / "one.mkv").exists()
        shot = directory / "resume" / "shot_000004"
        assert not (shot / "latent.pt").exists() and not (shot / "input.mkv").exists()


@pytest.mark.parametrize("output", ["out", "one.mkv"])
@pytest.mark.parametrize("more", [False, True])
def test_end_of_source_checked_in_a_job_of_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    output: str,
    more: bool,
) -> None:
    # A source with no cut, one shot and one segment: its one read gives the source's last
    # frame, and the end is checked there (run.Inputs), before the encode, so before the job's
    # only latent is recorded. A source as counted: the job finishes, every frame at its place.
    # A source giving a frame more than the first pass counted: stopped there with the line of
    # test_end_of_source_checked_before_the_last_shot_is_recorded, nothing made: no latent, no
    # segment, no file; and again by the same command.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "cuts.txt").write_text("")
    options = ("--cuts", "cuts.txt", "--window", "5")
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work")
    end_checks(monkeypatch, steps)
    if not more:
        assert upscale(tmp_path, source_path, output, *options) == 0
        made = ["encode 0", "window 0:0", "window 0:1", "decode 0"]
        assert steps.calls == ["end checked", *made] and sorted(steps.read) == [0]
        final = directory / "seg_000000.mkv" if output == "out" else tmp_path / output
        assert indexes(final) == list(range(25)) and errors(caplog) == []
        return
    understating(monkeypatch)
    for _ in range(2):
        steps.calls.clear()
        caplog.clear()
        assert upscale(tmp_path, source_path, output, *options) == 1
        assert steps.calls == ["end checked"] and steps.read == {}
        kept = "0 of 1 segments finished, and the units of the next ones kept"
        assert errors(caplog) == [frames_left(source_path, 24, kept)]
        assert "the same command resumes" not in caplog.text
        content = json.loads((directory / "manifest.json").read_text())
        assert content["input"]["frames"] == 24 and len(content["shots"]) == 1
        assert [segment["finished"] for segment in content["segments"]] == [False]
        assert [shot["encoded"] for shot in content["shots"]] == [False]
        assert "assembled" not in content and "finished already" not in caplog.text
        assert not (tmp_path / "one.mkv").exists()
        assert not (directory / "resume" / "shot_000000" / "latent.pt").exists()


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
        return STAND_IN_MODELS

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


needs_x264 = pytest.mark.skipif(not has_encoder("libx264"), reason="needs ffmpeg with libx264")


def x264_source(path: Path, frames: int = 40) -> Path:
    """An x264 encode with open GOPs and B-frames, a keyframe every 8 frames: what a resume's
    reads seek in (DESIGN.md, Input)."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast"),
            *("-x264-params", "keyint=8:min-keyint=8:scenecut=0:open-gop=1:bframes=3"),
            *("-pix_fmt", "yuv420p", str(path)),
        ],
        check=True,
    )
    return path


def recorded_reads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The ffmpeg command of every read of a source from now on (media/reader.py)."""
    from seedvr2x.media import reader

    commands: list[list[str]] = []
    command = reader.read_command

    def recorded(*arguments: Any, **options: Any) -> list[str]:
        commands.append(command(*arguments, **options))
        return commands[-1]

    monkeypatch.setattr(reader, "read_command", recorded)
    return commands


# Shots of 16 and 24 frames (5 and 7 latents: the second in two windows of 5), a segment each.
SEEKING = ("--cuts", "cuts.txt", "--window", "5", "--min-segment", "0")
SEEKING_RUN = ["encode 0", "window 0:0", "decode 0", "encode 16", "window 16:0", "window 16:1"]
SEEKING_RUN += ["decode 16"]


@needs_x264
@pytest.mark.parametrize("correction", ["none", "split"])
def test_resume_reads_through_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, correction: str
) -> None:
    # A run from the source's first frame reads it once, in order, seeking nowhere. Resumed after
    # its first segment, a job reads its next shot through the frame index: one read, decoding
    # from the keyframe one before the shot's own (frame 8, 0.32 s), its frames selected from the
    # shot's first pts on (640 ms), never from frame 0; the frames each encode reads, and the
    # output, the uninterrupted run's.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("16\n")
    source_path = x264_source(tmp_path / "in.mkv")
    options = (*SEEKING, "--color-correction", correction)
    commands = recorded_reads(monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    assert steps.calls == SEEKING_RUN
    [whole] = commands
    assert "-ss" not in whole and "select=" not in whole[whole.index("-vf") + 1]
    read = dict(steps.read)
    commands.clear()
    steps.calls.clear()
    steps.stop = "encode 16"
    stopped(tmp_path, source_path, "out", *options)
    commands.clear()
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert steps.calls == SEEKING_RUN[3:]
    [seeking] = commands
    assert seeking[seeking.index("-ss") + 1] == "0.320000"
    assert "select=gte(pts\\,640)," in seeking[seeking.index("-vf") + 1]
    assert steps.read == read
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "whole" / name)


@needs_x264
@pytest.mark.parametrize("output", ["out", "one.mkv"])
@pytest.mark.parametrize(
    ("more", "stop"),
    # Inside the last shot, its frames read and its latent not made; and before it, the first
    # shot's units made: the one stop a source with a frame more allows, whose read of the last
    # shot fails before anything in it can stop.
    [(False, "encode 16"), (False, "decode 0"), (True, "decode 0")],
)
def test_end_of_source_checked_on_the_resume_of_the_last_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    output: str,
    more: bool,
    stop: str,
) -> None:
    # A job stopped with its last shot to read, resumed: the resume reads that shot alone, its
    # reader opened mid-file through the frame index (test_resume_reads_through_the_index) and
    # ended at the source's last frame, where the end is checked (run.Inputs), by the read that
    # gives that frame, before the shot's encode. A source as counted: the job finishes, its
    # output the uninterrupted run's. A source giving a frame more than the first pass counted
    # (understating): the resume stops there with the line of
    # test_end_of_source_checked_before_the_last_shot_is_recorded, which names the way out
    # (design, 2026-10-12), the last shot not encoded, the last segment unmade, no file.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("16\n")
    source_path = x264_source(tmp_path / "in.mkv")
    whole = "whole" if output == "out" else "whole.mkv"
    if more:
        understating(monkeypatch)
    else:
        assert upscale(tmp_path, source_path, whole, *SEEKING) == 0
    steps.stop = stop
    stopped(tmp_path, source_path, output, *SEEKING)
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work")
    content = json.loads((directory / "manifest.json").read_text())
    assert [shot["encoded"] for shot in content["shots"]] == [True, False]
    steps.stop = None
    steps.calls.clear()
    steps.read.clear()
    caplog.clear()
    end_checks(monkeypatch, steps)
    commands = recorded_reads(monkeypatch)
    before = ["decode 0"] if stop == "decode 0" else []
    assert upscale(tmp_path, source_path, output, *SEEKING) == (1 if more else 0)
    # One read, from the keyframe one before the shot's own, its frames from the shot's first.
    [seeking] = commands
    assert seeking[seeking.index("-ss") + 1] == "0.320000"
    assert "select=gte(pts\\,640)," in seeking[seeking.index("-vf") + 1]
    if not more:
        assert steps.calls == [*before, "end checked", *SEEKING_RUN[3:]]
        assert sorted(steps.read) == [16] and errors(caplog) == []
        if output == "out":
            for name in ("seg_000000.mkv", "seg_000001.mkv"):
                assert decoded(directory / name) == decoded(tmp_path / whole / name)
            assert contents(directory / "checksums") == contents(tmp_path / whole / "checksums")
        else:
            assert decoded(tmp_path / output) == decoded(tmp_path / whole)
            assert indexes(tmp_path / output) == list(range(40))
        return
    assert steps.calls == [*before, "end checked"] and steps.read == {}
    kept = "1 of 2 segments finished, and the units of the next ones kept"
    assert errors(caplog) == [frames_left(source_path, 39, kept)]
    assert "the same command resumes" not in caplog.text
    content = json.loads((directory / "manifest.json").read_text())
    assert content["input"]["frames"] == 39
    assert [segment["finished"] for segment in content["segments"]] == [True, False]
    assert [shot["encoded"] for shot in content["shots"]] == [True, False]
    assert "assembled" not in content and "finished already" not in caplog.text
    assert not (tmp_path / "one.mkv").exists()
    assert not (directory / "resume" / "shot_000016" / "latent.pt").exists()


@needs_x264
def test_copy_made_again_through_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # split's input copy of a shot, missing on resume, made again from the input through the frame
    # index (run._copy_again): one read from the keyframe before the shot's own; the copy holds
    # the frames the shot's encode read, and the output is the uninterrupted run's.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("16\n")
    source_path = x264_source(tmp_path / "in.mkv")
    assert upscale(tmp_path, source_path, "whole", *SEEKING, *SPLIT) == 0
    steps.stop = "decode 16"
    stopped(tmp_path, source_path, "out", *SEEKING, *SPLIT)
    out = tmp_path / "out"
    (out / "resume" / "shot_000016" / "input.mkv").unlink()
    seen: list[str] = []
    decode = steps.decode

    def looked(*arguments: Any, **tiled: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments, **tiled)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    commands = recorded_reads(monkeypatch)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *SEEKING, *SPLIT) == 0
    assert steps.calls == ["decode 16"]
    assert "shot 2/2: input.mkv missing: its input copy made again from the input" in caplog.text
    [seeking] = commands
    assert seeking[seeking.index("-ss") + 1] == "0.320000"
    assert seen == [steps.read[16]]
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(tmp_path / "whole" / name)


def holed_source(directory: Path, frames: int, absent: tuple[int, ...]) -> tuple[Path, Path]:
    """A source with missing frames, holes.mkv, and the file it was made from, full.mkv, in
    directory (tests/test_grid.py's recipe, at the stand-in's size): `frames` frames of 64x48 at
    24000/1001, an x264 encode with a keyframe every 8 frames and three B-frames that nothing
    references after each I or P frame; holes.mkv its stream copy without the frames `absent`,
    by their number, B-frames all (no multiple of 4), the others' timestamps kept."""
    assert all(frame % 4 for frame in absent)
    full, path = directory / "full.mkv", directory / "holes.mkv"
    x264 = "keyint=8:min-keyint=8:open-gop=1:bframes=3:b-pyramid=none:b-adapt=0:scenecut=0"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=24000/1001"),
            *("-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast"),
            *("-x264-params", x264, "-pix_fmt", "yuv420p", str(full)),
        ],
        check=True,
    )
    drop = "st(0,round(pts*tb*24000/1001));" + "+".join(f"eq(ld(0),{frame})" for frame in absent)
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(full), "-map", "0", "-c", "copy"),
            *("-bsf:v", f"noise=drop='{drop}'", str(path)),
        ],
        check=True,
    )
    return path, full


def repeating(full: Path, path: Path, shown: list[int]) -> Path:
    """A file whose frames really are full's frames `shown`, by number, in that order, one
    after the other at 24000/1001, no place empty: full's decoded pictures, 64x48 yuv420p,
    encoded again without loss (FFV1)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(full), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
        check=True,
    ).stdout
    size = 64 * 48 * 3 // 2
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "64x48"),
            *("-r", "24000/1001", "-i", "-", "-c:v", "ffv1", str(path)),
        ],
        input=b"".join(raw[frame * size : (frame + 1) * size] for frame in shown),
        check=True,
    )
    return path


# The run of a source of 40 places whose frames 17 and 18 are missing: cuts on both sides of the
# hole, and one inside it, at 18; a segment a shot.
HOLED_RUN = [
    *("encode 0", "window 0:0", "decode 0", "encode 12", "window 12:0", "decode 12"),
    *("encode 18", "window 18:0", "decode 18", "encode 24", "window 24:0", "decode 24"),
]
HOLED_SEGMENTS = [f"seg_{number:06d}.mkv" for number in range(4)]
MISSING_SAID = (
    "holes.mkv: 2 frames missing after frame 16: its timestamps step from 0.667 to 0.792 s there,"
    " 3 frames of the rate it declares, 24000/1001, leaving places 17 and 18 empty on that rate's"
    " grid; read as a player shows it, frame 16 held in their place: 40 frames, 2 of them filled"
    " (5.0%), 38 decoded"
)


@needs_x264
def test_missing_frames_filled_through_a_run_and_its_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A source whose frames sit on its rate's grid with two places left empty (DESIGN.md, Input:
    # "Missing frames are filled, not refused") runs as one of as many frames as places: its
    # shots, its segments and its record count 40, the hole's places holding the frame before
    # it. A cut list may name a held place, the user's override: the shot then starts on the
    # held frame's copy. The frames each encode reads, and with split the output, are those of
    # a file where frame 16 really comes three times. Resumed after the hole, a job reads
    # through the index in places, seeking past the hole; resumed on the shot that starts
    # inside it, it reads the held frame for the place; either way the output is the
    # uninterrupted run's, the warning said again, the first pass never run again.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("12\n18\n24\n")
    source_path, full = holed_source(tmp_path, 40, (17, 18))
    shown = [16 if place in (17, 18) else place for place in range(40)]
    repeated = repeating(full, tmp_path / "repeated.mkv", shown)
    options = (*SEEKING, *SPLIT)
    passes = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *options) == 0
    assert steps.calls == HOLED_RUN and len(passes) == 1
    assert f"{tmp_path / MISSING_SAID}" in caplog.text
    assert "holes.mkv: 40 frames (38 decoded, 2 filled), 64x48" in caplog.text
    whole = tmp_path / "whole"
    assert shots_of(whole) == [(0, 12), (12, 18), (18, 24), (24, 40)]
    frame = 96 * 128 * 3 * 2
    assert [len(decoded(whole / name)) // frame for name in HOLED_SEGMENTS] == [12, 6, 6, 16]
    record = json.loads((whole / "manifest.json").read_text())["input"]
    assert record["frames"] == 40
    assert record["filled"] == {
        "decoded": 38,
        "holes": 1,
        "places": 2,
        "first": {"frame": 17, "length": 2},
        "longest": 2,
    }
    read = dict(steps.read)
    # The file whose frames really are repeated: the same frames read by each encode, the same
    # output; and its record the one of before the places, nothing filled.
    steps.read.clear()
    assert upscale(tmp_path, repeated, "repeated", *options) == 0
    assert steps.read == read
    for name in HOLED_SEGMENTS:
        assert decoded(whole / name) == decoded(tmp_path / "repeated" / name)
    record = json.loads((tmp_path / "repeated" / "manifest.json").read_text())["input"]
    assert list(record) == [
        *("path", "bytes", "modified_ns", "sha256", "frames", "frame_rate", "size"),
        *("sample_aspect", "read_as", "pix_fmt", "primaries", "transfer", "index", "idet"),
    ]
    assert record["frames"] == 40
    # Its frame index the file of before the places; the one with the hole has the held rows'.
    headers = [
        (tmp_path / name / "frame_index.bin").read_bytes().split(b"\n")[1]
        for name in ("repeated", "whole")
    ]
    assert b'"version":1' in headers[0] and b'"held"' not in headers[0]
    assert b'"version":2' in headers[1] and b'"held"' in headers[1]
    for stop, output in (("encode 24", "past"), ("encode 18", "inside")):
        steps.stop = stop
        stopped(tmp_path, source_path, output, *options)
        passes.clear()
        caplog.clear()
        steps.calls.clear()
        steps.read.clear()
        steps.stop = None
        commands = recorded_reads(monkeypatch)
        assert upscale(tmp_path, source_path, output, *options) == 0
        assert steps.calls == HOLED_RUN[HOLED_RUN.index(stop) :] and passes == []
        assert f"{tmp_path / output}: the same input and ffmpeg, the first pass as" in caplog.text
        assert f"{tmp_path / MISSING_SAID}" in caplog.text
        assert steps.read == {shot: read[shot] for shot in steps.read}
        seeking = commands[0]
        if stop == "encode 24":
            # Past the hole: frame 24, a keyframe, read from the keyframe before it, frame 16,
            # at 0.667 s, selected from its own pts on, 1,001 ms.
            assert sorted(steps.read) == [24] and len(commands) == 1
            assert seeking[seeking.index("-ss") + 1] == "0.667000"
            assert "select=gte(pts\\,1001)," in seeking[seeking.index("-vf") + 1]
        else:
            # On a held place: the frame held, 16, is read for it, from the keyframe before
            # its own, frame 8, selected from frame 16's pts on, 667 ms.
            assert sorted(steps.read) == [18, 24]
            assert (
                "holes.mkv: frame 18 is a missing frame's place, filled by frame 16, the frame"
                " before it, held: that frame is read for it"
            ) in caplog.text
            assert seeking[seeking.index("-ss") + 1] == "0.334000"
            assert "select=gte(pts\\,667)," in seeking[seeking.index("-vf") + 1]
        for name in HOLED_SEGMENTS:
            assert decoded(tmp_path / output / name) == decoded(whole / name)
        assert contents(tmp_path / output / "checksums") == contents(whole / "checksums")


@pytest.mark.parametrize("how", ["missing", "damaged", "cut short"])
def test_index_derived_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    # The frame index a resume finds missing, or not the one its manifest records, is no reason to
    # refuse the job: the first pass runs again, said, and makes it again, the same bytes; the
    # output is the uninterrupted run's.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

    monkeypatch.setattr(examined, "scan", counted)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    index = tmp_path / "out" / "frame_index.bin"
    kept = index.read_bytes()
    if how == "missing":
        index.unlink()
        said = f"{index}: the frame index missing; the first pass runs again"
    elif how == "damaged":
        index.write_bytes(kept[:100] + bytes([kept[100] ^ 1]) + kept[101:])
        said = f"{index}: not the frame index the manifest recorded, by its SHA-256"
    else:
        index.write_bytes(kept[:-1])
        said = f"{index}: {len(kept) - 1} bytes, where the manifest recorded {len(kept)}"
    scans.clear()
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert said in caplog.text and len(scans) == 1
    assert steps.calls == UNINTERRUPTED[6:]
    assert index.read_bytes() == kept
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "whole" / name)


def another_index(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Path], list[bool]]:
    """The first passes run, and a switch: on, a pass gives another index than the source's own,
    frame 7's CRC-32 changed, as a damaged source does, decoded otherwise from one pass to the
    next (media/reader.py)."""
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan
    otherwise = [False]

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        found = scan(path, *detector)
        if not otherwise[0]:
            return found
        assert found.index is not None
        crc32 = found.index.crc32.copy()
        crc32[7] ^= 1
        return replace(found, index=replace(found.index, crc32=crc32))

    monkeypatch.setattr(examined, "scan", counted)
    return scans, otherwise


def test_index_made_again_named_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A first pass run again, its recorded index missing, that gives another index than the one
    # recorded: the index is written with a manifest naming it, at once, in the environment
    # recorded, the job stopped before a unit. The next run then trusts the record, and runs no
    # first pass; left to the next unit, every run stopped before one warned of an index its
    # manifest didn't name and ran the pass again. A finished job has none made again
    # (test_finished_job_run_again_makes_nothing).
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    out = tmp_path / "out"
    index, manifest = out / "frame_index.bin", out / "manifest.json"
    first, recorded = index.read_bytes(), json.loads(manifest.read_text())
    index.unlink()
    otherwise[0] = True
    scans.clear()
    steps.calls.clear()
    # Another GPU, accepted: the manifest written with the index keeps the environment recorded.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
    caplog.set_level(logging.INFO, logger="seedvr2x")
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert f"{index}: the frame index missing; the first pass runs again" in caplog.text
    assert len(scans) == 1 and steps.calls == ["encode 4"]
    data = index.read_bytes()
    assert data != first and len(data) == len(first)
    written = json.loads(manifest.read_text())
    named = {"name": "frame_index.bin", "bytes": len(data)}
    assert written["input"]["index"] == named | {"sha256": hashlib.sha256(data).hexdigest()}
    assert written["environment"]["gpu"] != "another"
    assert written | {"plan": untimed(written["plan"])} == recorded | {
        "input": written["input"],
        "plan": untimed(recorded["plan"]),
    }
    # The next run reads the record, its index as named: no first pass.
    otherwise[0] = False
    scans.clear()
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == 0
    assert not scans and "the first pass as recorded" in caplog.text
    assert "the first pass runs again" not in caplog.text
    assert index.read_bytes() == data


@pytest.mark.parametrize("stop", [None, "encode 4"])
def test_index_of_another_ffmpeg_written_with_its_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    stop: str | None,
) -> None:
    # A first pass run again under another ffmpeg, accepted, gives an index its manifest's
    # environment didn't make: it is written with the next unit made, whose manifest records that
    # environment, never before. A resume stopped before a unit keeps its recorded index and its
    # manifest as they were, and so does a job finished already, for which no pass runs again at
    # all (cli._finished_already); the first ffmpeg then finds its record whole, no pass run
    # again.
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = stop
    assert upscale(tmp_path, source_path, "out", *JOB) == (0 if stop is None else 130)
    out = tmp_path / "out"
    index, manifest = out / "frame_index.bin", out / "manifest.json"
    first, recorded = index.read_bytes(), manifest.read_bytes()
    check = ffmpeg.check
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    otherwise[0] = True
    scans.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    status = upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert status == (0 if stop is None else 130) and len(scans) == (0 if stop is None else 1)
    assert (index.read_bytes(), manifest.read_bytes()) == (first, recorded)
    # Back on the first ffmpeg, no change to accept: the record as it was, trusted.
    monkeypatch.setattr(ffmpeg, "check", check)
    otherwise[0] = False
    scans.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == (0 if stop is None else 130)
    assert not scans and "the first pass as recorded" in caplog.text
    assert (index.read_bytes(), manifest.read_bytes()) == (first, recorded)
    if stop is None:
        return
    # The other ffmpeg again, to the job's end: the index its pass made is written with the first
    # unit, shot 4's latent, in a manifest naming it and recording that ffmpeg and the change.
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    otherwise[0] = True
    steps.stop = "window 4:0"
    stopped(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    data = index.read_bytes()
    assert data != first and len(data) == len(first) and len(scans) == 1
    written = json.loads(manifest.read_text())
    named = {"name": "frame_index.bin", "bytes": len(data)}
    assert written["input"]["index"] == named | {"sha256": hashlib.sha256(data).hexdigest()}
    assert written["environment"]["ffmpeg"] == "n0.0-another"
    assert len(written["environment_changes"]) == 1
    scans.clear()
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert not scans and "the first pass as recorded" in caplog.text


def written_as(*paths: Path) -> dict[Path, tuple[bytes, int, int] | None]:
    """Each file as it was written: its bytes, modification time and inode, which a file written
    again changes even with the same bytes (files.write_whole renames a new one into its place);
    None for one that isn't there."""
    found: dict[Path, tuple[bytes, int, int] | None] = {}
    for path in paths:
        status = path.stat() if path.exists() else None
        found[path] = status and (path.read_bytes(), status.st_mtime_ns, status.st_ino)
    return found


def spoiled(index: Path, how: str) -> str:
    """Leave a job's frame index intact, or make it missing, damaged (a byte flipped) or cut
    short: why its record is then not trusted, as the run says it, "" for one intact."""
    kept = index.read_bytes()
    if how == "missing":
        index.unlink()
        return f"{index}: the frame index missing"
    if how == "damaged":
        index.write_bytes(kept[:100] + bytes([kept[100] ^ 1]) + kept[101:])
        return f"{index}: not the frame index the manifest recorded, by its SHA-256"
    if how == "cut short":
        index.write_bytes(kept[:-1])
        return f"{index}: {len(kept) - 1} bytes, where the manifest recorded {len(kept)}"
    return ""


@pytest.mark.parametrize("index", ["intact", "missing", "damaged", "cut short"])
@pytest.mark.parametrize("cut", ["by a list", "detected"])
def test_finished_job_run_again_makes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    index: str,
    cut: str,
) -> None:
    # A job run to its end, then run again (DESIGN.md, Pause and resume: "Re-running a finished
    # job doesn't rewrite its manifest, and makes nothing again, its frame index and its plan
    # included: no memory is read"): said finished, exit 0, whether its frame index is there,
    # missing, damaged or cut short. No first pass, no shot detector built nor its file asked of
    # the pull, no plan made nor the GPU's memory read for one, no model loaded, no unit; the
    # manifest not written again, nor the index, by their bytes, modification time and inode: one
    # missing stays missing, a damaged one as it is, said in a warning. What a stop left in
    # resume/ is discarded, as a resume discards it. Another seed asked of it is still another
    # job, refused before any first pass, nothing touched.
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options, segments = (JOB, 2) if cut == "by a list" else (CUT, 4)
    assert upscale(tmp_path, source_path, "out", *options) == 0
    out = tmp_path / "out"
    files = (out / "manifest.json", out / "frame_index.bin")
    untrusted = spoiled(files[1], index)
    kept = written_as(*files)
    assert (kept[files[1]] is None) == (index == "missing")
    (out / "resume" / "shot_000000").mkdir(parents=True)  # as a stop could leave them
    passes, asked = first_passes(monkeypatch), asked_of_the_pull(monkeypatch)
    StandInDetector.built.clear()
    scripted.calls.clear()
    caplog.clear()
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    monkeypatch.setattr(planner, "plan", lambda *a, **k: pytest.fail("a plan made"))
    monkeypatch.setattr(cli, "_free_memory", lambda device: pytest.fail("the GPU's memory read"))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert f"{out}: finished already, its {segments} segments" in caplog.text
    assert passes == [] and StandInDetector.built == [] and asked == [False]
    assert scripted.calls == [] and written_as(*files) == kept
    assert not (out / "resume").exists()
    # What was discarded is said, with its index as without; never that the job resumes.
    assert re.search(rf"{re.escape(str(out))}: \d leftovers? discarded", caplog.text)
    assert "resuming " not in caplog.text
    needs_none = "a finished job needs none, and makes nothing again"
    if index == "intact":
        assert needs_none not in caplog.text
    else:
        assert f"{untrusted}: {needs_none}" in caplog.text
    assert "the first pass runs again" not in caplog.text
    left = contents(out)
    text = refused(tmp_path, source_path, caplog, *options, "--seed", "7")
    assert "another job than the one asked" in text and "settings.seed: 42 -> 7" in text
    assert passes == [] and StandInDetector.built == [] and asked == [False, False]
    assert written_as(*files) == kept and contents(out) == left


@pytest.mark.parametrize("index", ["intact", "missing", "damaged", "another ffmpeg's"])
@pytest.mark.parametrize("cut", ["by a list", "detected"])
def test_file_job_with_every_segment_finished_is_assembled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    index: str,
    cut: str,
) -> None:
    # A .mkv path's job with every segment finished and no file yet isn't finished (DESIGN.md,
    # Pause and resume: "A finished file's job is finished once the file is in place: with every
    # segment made and no file yet, the same command assembles"), whatever became of its frame
    # index, which the assembly times the file by: missing, damaged, or another ffmpeg's,
    # accepted, the first pass runs again, once, with the shot detector for a job that detects
    # its shots, its file asked of the pull, and the file is assembled: exit 0 with the file
    # there, the work directory gone, no model loaded, no unit made. Never a directory's
    # "finished already", exit 0, with no file (test_finished_job_run_again_makes_nothing).
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options, segments = (JOB, 2) if cut == "by a list" else (CUT, 4)
    final, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    # Stopped before its assembly, as tests/test_assembled.py stops one: no room for the file.
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source_path, "one.mkv", *options) == 1
    assert f"not assembled: every segment finished, {segments} in {work}" in caplog.text
    content = json.loads((work / "manifest.json").read_text())
    assert all(segment["finished"] for segment in content["segments"])
    assert "assembled" not in content and not final.exists()
    untrusted, accept = "", ()
    if index == "another ffmpeg's":
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        accept = ("--accept-env-change",)
        # Not finished, its assembly the next unit, made in this environment: refused without
        # the flag as any resume is, where a directory's finished job, which makes nothing, is
        # taken whatever its environment (test_finished_job_in_another_environment).
        left = contents(work)
        text = refused(tmp_path, source_path, caplog, *options, output="one.mkv")
        assert "Only its environment differs: --accept-env-change resumes it anyway" in text
        assert "finished already" not in text and not final.exists() and contents(work) == left
    else:
        untrusted = spoiled(work / "frame_index.bin", index)
    passes, asked = first_passes(monkeypatch), asked_of_the_pull(monkeypatch)
    StandInDetector.built.clear()
    scripted.calls.clear()
    caplog.clear()
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    assert upscale(tmp_path, source_path, "one.mkv", *options, *accept) == 0
    assert final.is_file() and not work.exists()
    assert f"wrote {final}: 25 frames from {segments} segments" in caplog.text
    assert "a finished job needs none" not in caplog.text
    again, detects = index != "intact", index != "intact" and cut == "detected"
    assert len(passes) == again and scripted.calls == []
    assert asked == [detects] and len(StandInDetector.built) == detects
    if untrusted:
        assert f"{untrusted}; the first pass runs again, which makes it" in caplog.text
    if index == "another ffmpeg's":
        # Its manifest, written by the assembly, the next unit made, recorded the change.
        said = f"{work}: resumed in another environment, as accepted, which its manifest records"
        assert f"{said} with the next unit made: environment.ffmpeg: " in caplog.text
    # The file an uninterrupted run assembles: its frames, each saying its place.
    assert indexes(final) == list(range(25))
    assert cli.main(["verify", str(final)]) == 0


def nothing_asked_of_the_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """From here on, a run that asks anything of the GPU fails its test: the driver (the
    question runtime/memory.py puts to it behind WSL2's device, which is there), the card's free
    memory or the driver's version through NVML, either of the driver's libraries loaded, or
    any function of torch.cuda's called, torch.cuda.is_available first, which a machine with no
    CUDA device answers no to."""
    from seedvr2x.runtime import memory, model

    def asked(what: str) -> Callable[..., Any]:
        return lambda *arguments, **options: pytest.fail(f"asked of the GPU: {what}")

    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    for name in ("startup", "probe", "card_free", "reading", "hold", "check"):
        monkeypatch.setattr(memory, name, asked(f"runtime.memory.{name}"))
    monkeypatch.setattr(model, "nvidia_driver", asked("the driver's version, through NVML"))
    monkeypatch.setattr(cli, "_free_memory", asked("the GPU's free memory"))
    load = ctypes.CDLL

    def loading(name: str, *arguments: Any, **options: Any) -> Any:
        if name in (memory.CUDA, memory.NVML, "libnvidia-ml.so.1"):
            pytest.fail(f"asked of the GPU: {name} loaded")
        return load(name, *arguments, **options)

    monkeypatch.setattr(ctypes, "CDLL", loading)
    for name, value in vars(torch.cuda).items():
        if inspect.isroutine(value) and not name.startswith("__"):
            monkeypatch.setattr(torch.cuda, name, asked(f"torch.cuda.{name}"))


@pytest.mark.parametrize("plan", [False, True], ids=["run again", "--plan"])
@pytest.mark.parametrize("index", ["intact", "missing"])
def test_a_finished_job_asks_the_gpu_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    index: str,
    plan: bool,
) -> None:
    # DESIGN.md, Pause and resume: "the finished state is known before anything asks for the
    # GPU"; Allocator: "A finished job run again asks the driver nothing." A finished directory
    # job run again, its frame index there or not, with --plan FILE or without: said finished
    # with nothing asked of the GPU, its driver or NVML, as on a machine with no CUDA device,
    # which answered "no CUDA device", exit 1, until then, and under WSL2 after its driver had
    # been asked for more memory than the card had free.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    out = tmp_path / "out"
    files = (out / "manifest.json", out / "frame_index.bin")
    spoiled(files[1], index)
    kept = written_as(*files)
    nothing_asked_of_the_gpu(monkeypatch)
    steps.calls.clear()
    caplog.clear()
    listed = tmp_path / "plan.txt"
    options = (*JOB, "--plan", str(listed)) if plan else JOB
    # Its cut list is written from its frame index, by --plan, where that is there.
    assert upscale(tmp_path, source_path, "out", *options) == (plan and index == "missing")
    assert f"{out}: finished already, its 2 segments" in caplog.text
    assert "no CUDA device" not in caplog.text and steps.calls == []
    assert listed.is_file() == (plan and index == "intact") and written_as(*files) == kept
    # Another job in that directory is refused as before, and asks no more.
    text = refused(tmp_path, source_path, caplog, *options, "--seed", "7")
    assert "another job than the one asked" in text and "settings.seed: 42 -> 7" in text
    # A job that isn't finished takes the GPU, and is refused where there is none.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(memory_of_the_run(), "startup", lambda: "backend:cudaMallocAsync")
    assert "no CUDA device" in refused(tmp_path, source_path, caplog, *JOB, output="new")


# A job run again in a process of its own, watched from before seedvr2x is imported: every
# routine of torch.cuda's wrapped, each call that isn't made from within another of them written
# with the file and line of its caller, the first frame outside torch's own files; every library
# ctypes loads (its _dlopen, which CDLL, PyDLL and cdll all go through), with its caller's file
# and line; every import of NVML's Python binding (pynvml); and, once the run is done, the
# driver's libraries the process has mapped, by whatever loaded them (/proc/self/maps), each in
# the JSON file named first. The model files pass unread, and the run is taken for one behind
# WSL2's GPU device, where it would ask the driver.
WATCHED_PROCESS = """
import ctypes
import importlib.abc
import inspect
import json
import sys

import torch

calls = []
within = []
loads = []
imports = []


def watched(name, routine):
    def asked(*arguments, **options):
        if not within:
            frame = sys._getframe(1)
            while frame is not None and "/torch/" in frame.f_code.co_filename:
                frame = frame.f_back
            where = ("", 0) if frame is None else (frame.f_code.co_filename, frame.f_lineno)
            calls.append([name, *where])
        within.append(name)
        try:
            return routine(*arguments, **options)
        finally:
            within.pop()

    # What torch's own modules read of a routine as they are imported (__wrapped__).
    asked.__dict__.update(getattr(routine, "__dict__", {}))
    return asked


for name, value in list(vars(torch.cuda).items()):
    if inspect.isroutine(value) and not name.startswith("__"):
        setattr(torch.cuda, name, watched(name, value))
dlopen = ctypes._dlopen


def loading(name, *arguments):
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename == ctypes.__file__:
        frame = frame.f_back
    where = ("", 0) if frame is None else (frame.f_code.co_filename, frame.f_lineno)
    loads.append([str(name), *where])
    return dlopen(name, *arguments)


ctypes._dlopen = loading


class Binding(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "pynvml":
            imports.append(name)
        return None


sys.meta_path.insert(0, Binding())
assert not [name for name in sys.modules if name.split(".")[0] in ("seedvr2x", "pynvml")]
from seedvr2x import cli
from seedvr2x.runtime import memory, weights

weights.check_models = lambda *a: None
# Behind WSL2's GPU device, where a run asks the driver (memory.startup): on any machine, a
# question put to it loads CUDA's library through ctypes, and no device shows to answer it.
memory.paravirtual = lambda: True
status = cli.main(sys.argv[2:])
with open("/proc/self/maps") as maps:
    mapped = sorted({line.split()[-1] for line in maps if "libcuda" in line or "nvidia-ml" in line})
with open(sys.argv[1], "w") as out:
    json.dump({"calls": calls, "loads": loads, "imports": imports, "mapped": mapped}, out)
sys.exit(status)
"""


@pytest.mark.parametrize("output", ["out", "one.mkv"], ids=["finished", "its assembly left"])
def test_a_job_with_every_segment_made_asks_the_gpu_nothing_in_a_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, output: str
) -> None:
    # A job with every segment made run again in a process of its own, as a user runs it, on a
    # machine that shows no CUDA device: a finished one said finished, a .mkv path's with its
    # assembly alone left assembled; and nothing of seedvr2x's asks the GPU anything on its way:
    # no routine of torch.cuda's called, neither of the driver's libraries loaded through ctypes
    # (runtime/memory.py's probe and its card's reading, runtime/model.py's driver version),
    # NVML's Python binding not imported, and NVML's library not mapped by anything. The tests
    # above set their guard in a process that has run a job already: torch reads cuDNN's version
    # through a first call that asks torch.cuda whether a device is there, and keeps it, so that
    # the environment's own reading of it there (environment.current,
    # torch.backends.cudnn.version) passed them unseen (a review, 2026-10-12). What isn't
    # seedvr2x's is told apart by its caller's file: diffusers, which the vendored VAE imports
    # through runtime/model.py, asks torch.cuda.is_available once as it is imported
    # (diffusers/utils/torch_utils.py), which torch answers by CUDA's cudaGetDeviceCount: the
    # driver's library mapped by CUDA's runtime where the machine has one, no context made, no
    # NVML (environment.current). torch's NVML-based check, which would load NVML instead, is
    # left unset.
    source_path = job(tmp_path, monkeypatch)
    # The job is recorded under a cuDNN that no torch answers here, which the run again must
    # take from its record to find no difference.
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 1)
    work = tmp_path / ("one.mkv.work" if output == "one.mkv" else output)
    if output == "one.mkv":
        with monkeypatch.context() as full:
            full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
            assert upscale(tmp_path, source_path, output, *JOB) == 1
        assert not (tmp_path / output).exists()
    else:
        assert upscale(tmp_path, source_path, output, *JOB) == 0
    watched = tmp_path / "watched.json"
    kept = written_as(work / "manifest.json")
    assert json.loads((work / "manifest.json").read_text())["environment"]["cudnn"] == 1
    environment = {
        name: value for name, value in os.environ.items() if name != "PYTORCH_NVML_BASED_CUDA_CHECK"
    }
    result = subprocess.run(
        [
            *(sys.executable, "-c", WATCHED_PROCESS, str(watched)),
            *command(tmp_path, source_path, output, *JOB),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={
            **environment,
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HOME": str(tmp_path / "hf"),
            "HF_HUB_OFFLINE": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    if output == "one.mkv":
        assert (tmp_path / output).is_file() and not work.exists()
        assert f"wrote {tmp_path / output}: 25 frames from 2 segments" in result.stderr
    else:
        assert f"{work}: finished already, its 2 segments" in result.stderr
        assert written_as(work / "manifest.json") == kept
    seen = json.loads(watched.read_text())
    package = str(Path(cli.__file__).resolve().parent)
    made: list[tuple[str, str, int]] = [tuple(call) for call in seen["calls"]]
    assert [call for call in made if call[1].startswith(package)] == []
    driver = [load for load in seen["loads"] if "libcuda" in load[0] or "nvidia-ml" in load[0]]
    assert driver == [] and seen["imports"] == []
    assert [path for path in seen["mapped"] if "nvidia-ml" in path] == []
    # Its environment is the record's where it isn't read, cuDNN's version with the GPU's name.
    assert "its environment differs" not in result.stderr
    # The guard saw the process: the call that isn't seedvr2x's is there, and libraries loaded
    # through ctypes.
    others = {(name, Path(file).parent.name, Path(file).name) for name, file, _ in made}
    assert ("is_available", "utils", "torch_utils.py") in others
    assert seen["loads"] != []


def memory_of_the_run() -> ModuleType:
    from seedvr2x.runtime import memory

    return memory


@pytest.mark.parametrize("index", ["intact", "missing"])
@pytest.mark.parametrize("cut", ["by a list", "detected"])
def test_a_job_with_only_its_assembly_left_asks_the_gpu_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    index: str,
    cut: str,
) -> None:
    # A .mkv path's job with every segment made and no file yet: its assembly is ffmpeg's work,
    # and takes no GPU: the file is assembled with nothing asked of the GPU, its driver or NVML.
    # But where its first pass runs again with the shot detector, its frame index gone and its
    # shots detected: that pass takes the GPU, and the run asks for it as any run does.
    from seedvr2x.runtime import memory

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    options = JOB if cut == "by a list" else CUT
    final, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source_path, "one.mkv", *options) == 1
    assert not final.exists()
    spoiled(work / "frame_index.bin", index)
    takes_the_gpu = index == "missing" and cut == "detected"
    started: list[str] = []
    if takes_the_gpu:
        monkeypatch.setattr(memory, "startup", lambda: started.append("asked") or "backend:x")
    else:
        nothing_asked_of_the_gpu(monkeypatch)
    passes = first_passes(monkeypatch)
    StandInDetector.built.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *options) == 0
    assert final.is_file() and not work.exists() and indexes(final) == list(range(25))
    # It resumes, its assembly left to make, and says so, where a finished job resumes nothing.
    assert f"resuming {work}: " in caplog.text
    assert started == ["asked"] * takes_the_gpu
    assert len(StandInDetector.built) == takes_the_gpu and len(passes) == (index == "missing")


@pytest.mark.parametrize("accept", [(), ("--accept-env-change",)], ids=["unasked", "accepted"])
@pytest.mark.parametrize("change", ["GPU", "stack", "ffmpeg"])
def test_a_job_with_only_its_assembly_left_in_another_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    change: str,
    accept: tuple[str, ...],
) -> None:
    # A .mkv path's job with every segment made and its assembly left, run again in another
    # environment. On another GPU alone: assembled, with the flag or without, nothing asked of
    # the GPU and nothing said: such a job takes no GPU and asks for none, its name no more than
    # the rest, and its manifest keeps the card that made its segments (environment.current).
    # PROVISIONAL, as built: DESIGN.md has that assembly "made in the new environment, refused
    # without the flag as any resume", and has such a job ask the driver nothing and need no
    # GPU; the two can't both hold of another GPU, a question put to design (a review,
    # 2026-10-12). Under another stack or another ffmpeg it is refused without the flag as any
    # resume, nothing made; with it, the difference is said and the file assembled, the first
    # pass run again under another ffmpeg.
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    final, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    with monkeypatch.context() as full:
        full.setattr(os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=4096))
        assert upscale(tmp_path, source_path, "one.mkv", *JOB) == 1
    assert not final.exists()
    kept = written_as(work / "manifest.json")
    recorded = json.loads((work / "manifest.json").read_text())["environment"]
    differs = ""
    if change == "GPU":
        # The machine's card is another one, and asking for its name fails the test.
        assert recorded["gpu"] == "a stand-in"
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another card")
        nothing_asked_of_the_gpu(monkeypatch)
    elif change == "stack":
        monkeypatch.setattr(model, "attention_backend", lambda: "another backend")
        differs = f'environment.attention: "{recorded["attention"]}" -> "another backend"'
    else:
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        differs = f'environment.ffmpeg: "{recorded["ffmpeg"]}" -> "n0.0-another"'
    passes = first_passes(monkeypatch)
    steps.calls.clear()
    caplog.clear()
    status = upscale(tmp_path, source_path, "one.mkv", *JOB, *accept)
    if change != "GPU" and not accept:
        # Refused as any resume in another environment, the flag named, nothing made.
        assert status == 1 and not final.exists() and written_as(work / "manifest.json") == kept
        [said] = errors(caplog)
        assert said.startswith(
            f"{work / 'manifest.json'}: another job than the one asked, which differs in:"
        )
        assert differs in said and "--accept-env-change" in said
        assert passes == [] and steps.calls == []
        return
    assert status == 0 and errors(caplog) == [] and steps.calls == []
    assert final.is_file() and not work.exists() and indexes(final) == list(range(25))
    assert f"resuming {work}: " in caplog.text
    if change == "GPU":
        assert "environment" not in caplog.text and "--accept-env-change" not in caplog.text
        assert passes == []
    else:
        resumed = "resumed in another environment, as accepted, which its manifest records"
        assert f"{work}: {resumed} with the next unit made: {differs}" in caplog.text
        assert len(passes) == (change == "ffmpeg")


def test_a_finished_job_is_held_to_its_model_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Pause and resume: a finished directory job "is still held to its settings, its
    # model files, its source's content and its directory". Its model files are checked and
    # hashed before it is said finished, as any job's before its record is compared, and
    # neither takes a GPU: one missing is refused as a missing model file is, by the header
    # check, exit 1, the job not said finished; another file under its name is another job.
    # Nothing is asked of the GPU for either.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    kept = contents(tmp_path / "out")
    nothing_asked_of_the_gpu(monkeypatch)
    again = command(tmp_path, source_path, "out", *JOB)
    model_file = tmp_path / "w.safetensors"
    model_file.write_bytes(b"another file")
    caplog.clear()
    assert cli.main(again) == 1
    assert "another job than the one asked" in caplog.text
    assert "settings.dit_model.sha256: " in caplog.text and "finished already" not in caplog.text
    model_file.unlink()
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    caplog.clear()
    assert cli.main(again) == 1
    assert errors(caplog) == [f"{model_file}: {weights.MISSING}\n{model_file}: {weights.MISSING}"]
    assert "finished already" not in caplog.text and contents(tmp_path / "out") == kept


@pytest.mark.parametrize("accept", [(), ("--accept-env-change",)], ids=["unasked", "accepted"])
@pytest.mark.parametrize("change", ["GPU", "stack", "ffmpeg"])
def test_finished_job_in_another_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    change: str,
    accept: tuple[str, ...],
) -> None:
    # A finished directory job run again in another environment is said finished whatever the
    # environment, nothing being made (design, 2026-10-12: "no refusal, no --accept-env-change
    # asked"; refused until then, as a resume is, the flag named): the difference said in one
    # line, the same with the flag given and without, then "finished already", exit 0, with no
    # first pass, under another ffmpeg too, whose pass a resume runs again, and nothing written:
    # its manifest keeps the environment that made it, no change recorded, and its index is as
    # it was. Which is what is said, with its trusted index (the stack's change: cli._resume) as
    # without (ffmpeg's: cli._finished_already), never that its manifest records the change "with
    # the next unit made": a finished job makes none, nor that it resumes: it resumes nothing.
    # Another GPU alone isn't said: a finished job asks nothing of the GPU, its name no more
    # than the rest (cli._run; environment.current), on the card that made it or on another.
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    out = tmp_path / "out"
    files = (out / "manifest.json", out / "frame_index.bin")
    kept = written_as(*files)
    passes = first_passes(monkeypatch)
    steps.calls.clear()
    recorded = json.loads(files[0].read_text())["environment"]
    differs = ""
    if change == "GPU":
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: pytest.fail("asked"))
        monkeypatch.setattr(model, "nvidia_driver", lambda: pytest.fail("asked"))
    elif change == "stack":
        monkeypatch.setattr(model, "attention_backend", lambda: "another backend")
        differs = f'environment.attention: "{recorded["attention"]}" -> "another backend"'
    else:
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        differs = f'environment.ffmpeg: "{recorded["ffmpeg"]}" -> "n0.0-another"'
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, *accept) == 0
    said = [record.getMessage() for record in caplog.records]
    stays = "its environment differs from this one and stays its manifest's, nothing being made"
    assert said.count(f"{out}: {stays} in this one: {differs}") == (change != "GPU")
    assert caplog.text.count("its environment differs") == (change != "GPU")
    assert said[-1] == f"{out}: finished already, its 2 segments" and errors(caplog) == []
    assert passes == [] and steps.calls == [] and written_as(*files) == kept
    assert "another job" not in caplog.text and "--accept-env-change" not in caplog.text
    assert "as accepted" not in caplog.text and "resuming " not in caplog.text
    assert "resumed in another environment" not in caplog.text
    assert "the next unit made" not in caplog.text
    assert json.loads(files[0].read_text())["environment_changes"] == []
    # Its environment alone: another setting with it is another job, refused, the flag given
    # or not, and no flag offered for it.
    text = refused(tmp_path, source_path, caplog, *JOB, "--seed", "7", *accept)
    assert "another job than the one asked" in text and "settings.seed: 42 -> 7" in text
    assert differs in text and "Only its environment differs" not in text
    # With --plan FILE, which writes the job's cut list, no flag is asked either (design,
    # 2026-10-12; refused without one until then, in words that promised a resume). Under
    # another stack, as on another GPU, the list is written from the job's frame index, exit 0;
    # under another ffmpeg it isn't, exit 1, that index not being one this ffmpeg made. The
    # manifest is untouched in all three, and no line says the job resumes.
    plan = tmp_path / "plan.txt"
    caplog.clear()
    status = upscale(tmp_path, source_path, "out", *JOB, "--plan", str(plan), *accept)
    assert status == (change == "ffmpeg") and plan.is_file() == (change != "ffmpeg")
    assert caplog.text.count(f"{stays} in this one: {differs}") == (change != "GPU")
    assert f"{out}: finished already, its 2 segments" in caplog.text
    assert "resuming " not in caplog.text and "--accept-env-change" not in caplog.text
    not_written = (
        f"--plan {plan}: not written: {out} is finished already, and a job's cut list is written"
        " from its frame index, which another ffmpeg or other conversions than this run's made"
    )
    assert (not_written in caplog.text) == (change == "ffmpeg")
    assert errors(caplog) == [] or change == "ffmpeg"
    assert passes == [] and steps.calls == [] and written_as(*files) == kept


def test_finished_job_still_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A finished job without its frame index is still held, with no first pass, to what needs no
    # decode: what its source declares and the output that follows, a probe saying otherwise
    # another job, as for a resume (test_probe_compared); and its directory to its manifest,
    # another's file refused, never deleted.
    from seedvr2x.media import source as examined

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    out = tmp_path / "out"
    (out / "frame_index.bin").unlink()
    kept = contents(out)
    passes = first_passes(monkeypatch)
    probe = examined.probe
    with monkeypatch.context() as patch:
        patch.setattr(examined, "probe", lambda path: replace(probe(path), color_primaries="bt709"))
        text = refused(tmp_path, source_path, caplog, *JOB)
    assert "another job than the one asked" in text and 'input.primaries: "" -> "bt709"' in text
    with monkeypatch.context() as patch:
        patch.setattr(examined, "probe", lambda path: replace(probe(path), width=128))
        text = refused(tmp_path, source_path, caplog, *JOB)
    assert "input.size[0]: 64 -> 128" in text and "output.size[0]: 128 -> 256" in text
    (out / "notes.txt").write_text("mine")
    assert "not this job's: notes.txt" in refused(tmp_path, source_path, caplog, *JOB)
    (out / "notes.txt").unlink()
    assert passes == [] and contents(out) == kept
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert passes == [] and contents(out) == kept


def finished_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> tuple[Path, Path, Path]:
    """A finished job, `out`, and the same job finished beside it, `whole`, which stays as the
    uninterrupted run's: the source, `out` and `whole`. The steps' calls are cleared."""
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    steps.calls.clear()
    return source_path, tmp_path / "out", tmp_path / "whole"


def spoil_segment(path: Path, how: str) -> str:
    """Delete the finished segment at path, or cut it short (another size): what a run then finds,
    as it says it."""
    recorded = path.stat().st_size
    if how == "deleted":
        path.unlink()
        return "missing"
    path.write_bytes(path.read_bytes()[:-1])
    return f"{recorded - 1} bytes, where {recorded} were recorded"


@pytest.mark.parametrize("segment", [0, 1])
@pytest.mark.parametrize("how", ["deleted", "truncated"])
def test_a_finished_segment_missing_or_changed_is_made_again_at_a_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
    segment: int,
) -> None:
    # DESIGN.md, Output, Assembly (decided on 2026-10-29): a finished segment found missing or of
    # another size at a start is recorded unfinished and made again, in the warning the assembly
    # gives a segment whose hash changed, where it used to be refused (resume.leftovers). The
    # job was finished, so it loads its models for the one segment: its shots' units only, and
    # the frames of the uninterrupted run's, the manifest and the checksums as they were.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, out, whole = finished_pair(tmp_path, monkeypatch, steps)
    name = f"seg_{segment:06d}.mkv"
    before = json.loads((out / "manifest.json").read_text())["segments"]
    found = spoil_segment(out / name, how)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    frames = ("frames 0 to 3", "frames 4 to 24")[segment]
    assert [m for m in caplog.messages if "its bytes changed" in m] == [
        f"{out / name}: its bytes changed since it was finished ({frames} of the job: {found}):"
        " the disk or another program is to suspect; made again now"
    ]
    assert [r.levelname for r in caplog.records if "its bytes changed" in r.getMessage()] == [
        "WARNING"
    ]
    assert errors(caplog) == [] and "finished already" not in caplog.text
    assert steps.calls == (
        ["encode 0", "window 0:0", "encode 3", "window 3:0", "decode 0", "decode 3"]
        if segment == 0
        else ["encode 4", "window 4:0", "window 4:1", "decode 4"]
    )
    # The record as it was, but for the hash of the file: a Matroska file carries a unique
    # identifier of its own, so two writes of the same frames differ in bytes, not in size.
    after = json.loads((out / "manifest.json").read_text())["segments"]
    assert [{**entry, "sha256": None} for entry in after] == [
        {**entry, "sha256": None} for entry in before
    ]
    assert after[segment]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert decoded(out / name) == decoded(whole / name)
    assert contents(out / "checksums") == contents(whole / "checksums")
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in whole.iterdir())
    # Finished again: the same command makes nothing.
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == [] and "its bytes changed" not in caplog.text


def test_a_finished_segment_changed_is_said_once_and_the_others_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Both segments gone: each named once, both made again, nothing of the job left finished
    # between; the frames are the uninterrupted run's.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, out, whole = finished_pair(tmp_path, monkeypatch, steps)
    spoil_segment(out / "seg_000000.mkv", "deleted")
    spoil_segment(out / "seg_000001.mkv", "truncated")
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert len([m for m in caplog.messages if "its bytes changed" in m]) == 2
    assert steps.calls == UNINTERRUPTED
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(out / name) == decoded(whole / name)


def test_a_finished_segment_changed_with_the_wrong_job_is_refused_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Another job's directory is refused before anything is made again, and said before any
    # segment: nothing written, nothing removed, no warning of a segment made again.
    source_path, out, _ = finished_pair(tmp_path, monkeypatch, steps)
    spoil_segment(out / "seg_000001.mkv", "truncated")
    kept = contents(out)
    text = refused(tmp_path, source_path, caplog, *JOB[:3], "6", *JOB[4:])
    assert "another job than the one asked" in text and "its bytes changed" not in text
    (out / "notes.txt").write_text("mine")
    kept = contents(out)
    assert "not this job's: notes.txt" in refused(tmp_path, source_path, caplog, *JOB)
    assert "its bytes changed" not in caplog.text and contents(out) == kept
    assert steps.calls == []


def test_plan_says_a_finished_segment_would_be_made_again_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # PROVISIONAL (impl-v010, 2026-10-29: DESIGN.md doesn't say what --plan says): the same warning,
    # ending "it would be made again by a run"; the segment's units are in the estimate; the
    # manifest and every file as they were (a cut list is written as for any job); the same
    # command without --plan then makes it.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, out, whole = finished_pair(tmp_path, monkeypatch, steps)
    found = spoil_segment(out / "seg_000001.mkv", "truncated")
    kept = contents(out)
    caplog.clear()
    listing = tmp_path / "listed.txt"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(listing)) == 0
    assert steps.calls == [] and contents(out) == kept and listing.is_file()
    assert [m for m in caplog.messages if "its bytes changed" in m] == [
        f"{out / 'seg_000001.mkv'}: its bytes changed since it was finished (frames 4 to 24 of the"
        f" job: {found}): the disk or another program is to suspect; it would be made again by a"
        " run"
    ]
    (estimate,) = [m for m in caplog.messages if m.startswith("time: about")]
    assert "for 1 encode, " in estimate and "the decode and write of 1 segment" in estimate
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == ["encode 4", "window 4:0", "window 4:1", "decode 4"]
    assert decoded(out / "seg_000001.mkv") == decoded(whole / "seg_000001.mkv")


def test_a_finished_segment_s_missing_checksums_are_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md names a segment missing or of another size alone: its checksums missing, the
    # segment there as recorded, is refused as before, and nothing is made.
    source_path, out, _ = finished_pair(tmp_path, monkeypatch, steps)
    (out / "checksums" / "seg_000001.crc32").unlink()
    kept = contents(out)
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert (
        "seg_000001.crc32: a finished segment's checksums, the manifest says, but missing" in text
    )
    assert contents(out) == kept and steps.calls == []


@pytest.mark.parametrize("how", ["deleted", "truncated"])
@pytest.mark.parametrize("stop", ["encode 4", "window 4:1", "decode 4"])
def test_a_stop_after_a_finished_segment_was_found_changed_leaves_a_start_that_makes_it_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
    stop: str,
) -> None:
    # No state refuses a start: stopped (the stand-in's second Ctrl-C) at the first unit of the
    # segment made again, in the middle of its windows or in its decode, the manifest is as the
    # run left it (written with a unit, not before), and the same command finishes the job, the
    # frames the uninterrupted run's. Before any unit the manifest still says the segment finished
    # and the file is gone or cut, which the next start finds again.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path, out, whole = finished_pair(tmp_path, monkeypatch, steps)
    spoil_segment(out / "seg_000001.mkv", how)
    steps.stop = stop
    before = json.loads((out / "manifest.json").read_text())["segments"]
    stopped(tmp_path, source_path, "out", *JOB)
    after = json.loads((out / "manifest.json").read_text())["segments"]
    if stop == "encode 4":
        assert after == before
    else:
        assert [entry["finished"] for entry in after] == [True, False]
    steps.stop = None
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert errors(caplog) == []
    assert (stop == "encode 4") == ("its bytes changed" in caplog.text)
    assert (
        steps.calls
        == {
            "encode 4": ["encode 4", "window 4:0", "window 4:1", "decode 4"],
            "window 4:1": ["window 4:1", "decode 4"],
            "decode 4": ["decode 4"],
        }[stop]
    )
    assert decoded(out / "seg_000001.mkv") == decoded(whole / "seg_000001.mkv")
    final = json.loads((out / "manifest.json").read_text())["segments"]
    assert [entry["finished"] for entry in final] == [True, True]
    assert [entry["bytes"] for entry in final] == [entry["bytes"] for entry in before]


def test_a_finished_segment_found_changed_resets_its_runs_under_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # manifest.unfinish_changed on a record read: the segment unfinished, no bytes and no hash,
    # its shots' runs not begun, a run made under its own cap (a shape planned again for the
    # memory free) laid out by the plan's window again, the other segment untouched; nothing
    # written; a record it can't read left as it is.
    from seedvr2x.runtime.manifest import WINDOW_CAP, unfinish_changed

    _, out, _ = finished_pair(tmp_path, monkeypatch, steps)
    recorded = json.loads((out / "manifest.json").read_text())
    shot = recorded["shots"][2]
    planned = [[0, 4], [2, 6]]
    assert shot["windows"] == planned and shot["windows_done"] == 2
    shot[WINDOW_CAP], shot["windows"] = 6, [[0, 6]]
    untouched = json.loads(json.dumps(recorded["segments"][0]))
    (out / "seg_000001.mkv").unlink()
    assert unfinish_changed(recorded, out) == [(1, "missing")]
    assert recorded["segments"][0] == untouched
    assert recorded["segments"][1]["finished"] is False
    assert (recorded["segments"][1]["bytes"], recorded["segments"][1]["sha256"]) == (None, None)
    assert (shot["encoded"], shot["windows_done"], WINDOW_CAP in shot) == (False, 0, False)
    assert shot["windows"] == planned
    assert [s["encoded"] for s in recorded["shots"][:2]] == [True, True]
    assert unfinish_changed({"segments": "unreadable"}, out) == []


@needs_x264
def test_finished_job_with_filled_places_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A finished job whose source has missing frames, their places filled (its record's "filled",
    # drawn from its frame index, and its "frames", the places), run again without that index: no
    # pass says again what was filled, and none runs to: both are the record's own, the job said
    # finished, its manifest as it was, never another job for what only a decode finds.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("12\n18\n24\n")
    source_path, _ = holed_source(tmp_path, 40, (17, 18))
    assert upscale(tmp_path, source_path, "out", *SEEKING) == 0
    out = tmp_path / "out"
    files = (out / "manifest.json", out / "frame_index.bin")
    record = json.loads(files[0].read_text())["input"]
    assert record["frames"] == 40 and record["filled"]["places"] == 2
    files[1].unlink()
    kept = written_as(*files)
    passes = first_passes(monkeypatch)
    steps.calls.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *SEEKING) == 0
    assert f"{out}: finished already, its 4 segments" in caplog.text
    assert "another job" not in caplog.text and "frames missing" not in caplog.text
    assert passes == [] and steps.calls == [] and written_as(*files) == kept


def test_frame_index_beside_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The index is written whole before the manifest's first write, which names it in its input
    # record by its size and SHA-256 (manifest.INDEX); a .mkv path's is in its work directory,
    # gone with it once the file is assembled. What an
    # interrupted first write leaves, the index or its partial file, is no obstacle; a partial
    # file a resume finds is discarded.
    from seedvr2x.media.index import FrameIndex

    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("frame_index.bin", "frame_index.bin.partial", "manifest.json.partial"):
        (out / name).write_bytes(b"left")
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    index = out / "frame_index.bin"
    entry = json.loads((out / "manifest.json").read_text())["input"]
    data = index.read_bytes()
    assert entry["index"] == {
        "name": "frame_index.bin",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    found = FrameIndex.read(index)
    assert found.frames == entry["frames"] == 25
    assert found.pts.tolist() == list(range(0, 1000, 40))
    assert not (out / "frame_index.bin.partial").exists()
    (out / "frame_index.bin.partial").write_bytes(b"left")
    caplog.set_level(logging.INFO, logger="seedvr2x")
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert "1 leftover discarded" in caplog.text
    assert not (out / "frame_index.bin.partial").exists()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert not list(tmp_path.glob("*frame_index*"))


def test_a_frame_decoded_otherwise_taken_and_summed_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A first pass whose index holds another CRC-32 for frame 10, as one concealing a damaged
    # frame otherwise would (media/reader.py, _take): the run reads every frame from the first,
    # takes frame 10 as decoded, said at once, and says it again at its end, with what it means
    # for a resume; the output is the run's with the index's own.
    from seedvr2x.media import source as examined

    scan = examined.scan

    def concealed(path: Path, *detector: Any) -> object:
        found = scan(path, *detector)
        assert found.index is not None
        crc32 = found.index.crc32.copy()
        crc32[10] ^= 1
        return replace(found, index=replace(found.index, crc32=crc32))

    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    assert "decoded otherwise" not in caplog.text
    monkeypatch.setattr(examined, "scan", concealed)
    steps.calls.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED
    assert re.search(r"frame 10: CRC-32 [0-9a-f]{8}, where the index has .* taken as", caplog.text)
    said = "1 frame in all decoded otherwise than in the first pass, read from the start and taken"
    assert f"{said} as decoded (10): an output is bit-identical across a resume only" in caplog.text
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "whole" / name)


def test_a_rate_contradicted_refused_by_the_first_pass_never_by_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # 48 fps declared 50/1 (test_rate.retagged), 100 frames, its frames half a frame and more
    # off the declared rate's timeline from frame 12: refused by its first pass (media/source.py,
    # _leaves), naming the rate its timestamps are exactly and what the declared one would cost,
    # nothing made; it was read at 50/1, and warned of by its first run and by each resume. A
    # resume that trusts its first pass's record judges nothing again (source.counted): a job
    # recorded with the judge held off, as one of before the rule, is resumed from its record,
    # no pass run, nothing refused, nothing said of a timeline.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

    monkeypatch.setattr(examined, "scan", counted)
    monkeypatch.chdir(tmp_path)
    source_path = retagged(tmp_path / "retagged.mkv", 50, "round(N*1000/48)", 100)
    (tmp_path / "cuts.txt").write_text("12\n")
    options = ("--cuts", "cuts.txt", "--min-segment", "0.1")
    text = refused(tmp_path, source_path, caplog, *options)
    assert (
        f"{source_path}: 88 of its 100 frames lie half a frame or more from their place on the"
        " timeline of the rate it declares, 50/1, from frame 12, up to 83 ms (4.15 of a frame),"
        " and its output, written at that rate, would be 4.000% shorter than its timestamps' own"
        " time, off the source's other streams: its timestamps follow 48/1 fps (48) exactly"
    ) in text
    assert "\nor give --frame-rate 48/1, which takes its frames at that rate" in text
    assert len(scans) == 1 and not (tmp_path / "out" / "manifest.json").exists()
    with monkeypatch.context() as before_the_rule:
        before_the_rule.setattr(examined, "timing_refusal", lambda *_: "")
        steps.stop = "encode 12"
        stopped(tmp_path, source_path, "out", *options)
    assert len(scans) == 2
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert len(scans) == 2  # the first pass's record trusted
    assert not strayed(caplog) and "timeline" not in caplog.text
    names = ("seg_000000.mkv", "seg_000001.mkv")
    assert indexes(tmp_path / "out" / names[0]) + indexes(tmp_path / "out" / names[1]) == list(
        range(100)
    )
    # With --frame-rate, the rate asked for is the job's, and nothing of the declared one's
    # timeline is said, in the first run or in its resume.
    caplog.clear()
    steps.stop = "encode 12"
    stopped(tmp_path, source_path, "at48", *options, "--frame-rate", "48")
    steps.stop = None
    assert upscale(tmp_path, source_path, "at48", *options, "--frame-rate", "48") == 0
    assert len(scans) == 3 and not strayed(caplog)


def rate_of(path: Path) -> str:
    """The frame rate an output declares, as ffprobe gives it (r_frame_rate)."""
    probed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return probed.strip()


@needs_x264
def test_frame_rate_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # research/docs/seeking.md's mechanism 7 at 64x48 (test_rate.mechanism7): declared 24/1, its
    # 260 frames within 10.8 ms of 24000/1001's timeline. Refused without --frame-rate: its 260
    # frames, which the declared rate's timeline is a quarter of a frame from, get the plain
    # refusal, and 1,001 of them the refusal giving that rate (media/rate.py, twin); with
    # --frame-rate 24000/1001, every frame read, in order, and that
    # rate the job's everywhere (media/source.py, Source.frame_rate): each output segment
    # declares it; the segments' merge rule takes it, 0.5005 s being 12 frames at 24000/1001 and
    # 13 at 24/1, so the 12-frame first shot is a segment of its own; the manifest records it,
    # its settings and output, the input keeping the rate it declares. --frame-rate 24 refused,
    # naming frame 252. A resume with another --frame-rate, or none, is another job, refused
    # before its first pass; with the same, the uninterrupted run's output.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

    monkeypatch.setattr(examined, "scan", counted)
    monkeypatch.chdir(tmp_path)
    source_path = mechanism7(tmp_path / "s9.mkv", size="64x48")
    (tmp_path / "cuts.txt").write_text("12\n")
    options = ("--cuts", "cuts.txt", "--min-segment", "0.5005")
    ntsc = ("--frame-rate", "24000/1001")
    text = refused(tmp_path, source_path, caplog, *options)
    assert "s9.mkv: variable frame rate: frames last from 41 to 62 ms, and only a constant" in text
    assert "its timestamps run at" not in text and "--frame-rate" not in text
    guided = mechanism7(tmp_path / "long.mkv", size="64x48", frames=1001)
    text = refused(tmp_path, guided, caplog, *options)
    assert "long.mkv: its timestamps run at 24000/1001 fps (23.976), within 10.8 ms" in text
    assert "\nor give --frame-rate 24000/1001, which takes its frames at that rate" in text
    assert upscale(tmp_path, source_path, "whole", *options, *ntsc) == 0
    whole = tmp_path / "whole"
    names = ("seg_000000.mkv", "seg_000001.mkv")
    assert sorted(p.name for p in whole.glob("*.mkv")) == list(names)
    assert indexes(whole / names[0]) + indexes(whole / names[1]) == list(range(260))
    assert [rate_of(whole / name) for name in names] == ["24000/1001"] * 2
    content = json.loads((whole / "manifest.json").read_text())
    assert content["settings"]["frame_rate"] == content["output"]["frame_rate"] == "24000/1001"
    assert content["input"]["frame_rate"] == "24"
    assert [(s["start"], s["end"]) for s in content["segments"]] == [(0, 12), (12, 260)]
    text = refused(tmp_path, source_path, caplog, *options, "--frame-rate", "24", output="at24")
    assert "s9.mkv: --frame-rate 24/1: frame 252 lies 21.0 ms after its place on" in text
    steps.stop = "encode 12"
    stopped(tmp_path, source_path, "out", *options, *ntsc)
    scans.clear()
    for other, said in (("24", '"24000/1001" -> "24"'), (None, '"24000/1001" -> null')):
        rate = ("--frame-rate", other) if other else ()
        text = refused(tmp_path, source_path, caplog, *options, *rate)
        assert (
            f"another job than the one asked, which differs in:\n  settings.frame_rate: {said}"
            in text
        )
    assert scans == []
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options, *ntsc) == 0
    assert scans == []  # the first pass's record trusted
    for name in names:
        assert decoded(tmp_path / "out" / name) == decoded(whole / name)


def test_idet_recorded_not_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # idet's counts go in the manifest's input record, for information (manifest._input): a
    # resume trusting the first pass's record says the warning again from them, without a pass;
    # one whose pass runs again (its frame index removed: cli._prior) finds other counts than the
    # recorded ones, which is no other job (resume.UNCOMPARED), and records them.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

    monkeypatch.setattr(examined, "scan", counted)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    manifest = tmp_path / "out" / "manifest.json"
    content = json.loads(manifest.read_text())
    found = json.loads(json.dumps(content["input"]["idet"]))
    assert scans and set(found) == {"repeated", "single", "multiple"}
    assert set(found["multiple"]) == {"tff", "bff", "progressive", "undetermined"}
    combed = {"tff": 20, "bff": 10, "progressive": 0, "undetermined": 0}
    content["input"]["idet"]["multiple"] = combed
    manifest.write_text(json.dumps(content, indent=2) + "\n")
    caplog.clear()
    scans.clear()
    steps.stop = "window 4:1"
    stopped(tmp_path, source_path, "out", *JOB)
    assert scans == []
    said = "ffmpeg's idet finds combed frames in a source declared progressive: 30 of 30 (100%)"
    assert (
        f"{source_path}: {said} interlaced by its multiple-frame detection, 20 top" in caplog.text
    )
    assert json.loads(manifest.read_text())["input"]["idet"]["multiple"] == combed
    (tmp_path / "out" / "frame_index.bin").unlink()
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert len(scans) == 1 and "another job" not in caplog.text
    assert json.loads(manifest.read_text())["input"]["idet"] == found


def extracted(path: Path) -> npt.NDArray[np.uint8]:
    """The frames of path as TransNetV2's official extraction gives them (tests/test_detection.py):
    (n, 27, 48, 3) uint8."""
    data = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(path)),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "pipe:"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(data, np.uint8).reshape(-1, 27, 48, 3)


def asked_of_the_pull(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Whether each run asked pull.resolve for the shot detector's file, in order."""
    asked: list[bool] = []
    resolve = pull.resolve

    def resolving(*arguments: Any, detector: bool = False, **options: Any) -> Any:
        asked.append(detector)
        return resolve(*arguments, detector=detector, **options)

    monkeypatch.setattr(pull, "resolve", resolving)
    return asked


def given_to_the_run(monkeypatch: pytest.MonkeyPatch) -> list[npt.NDArray[np.float32] | None]:
    """The shot probabilities of the source each run's job was given (run.run_job), in order."""
    from seedvr2x.runtime import run

    given: list[npt.NDArray[np.float32] | None] = []
    run_job = run.run_job

    def running(models: Any, source: Any, *arguments: Any, **options: Any) -> Any:
        given.append(None if source.index is None else source.index.probabilities)
        return run_job(models, source, *arguments, **options)

    monkeypatch.setattr(run, "run_job", running)
    return given


def test_shot_probabilities_recorded_then_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # Without --cuts, the shot detector runs in the first pass: its file asked of the pull and
    # checked with the models'; built once, on the job's device, given every frame in order and
    # closed before the models load; each frame's probability in the frame index's file, the
    # manifest's settings naming its file as they name the DiT's, and the threshold, 0.3 by
    # default, no cut list. The job one shot: the stand-in's probabilities, about 0.25 on every
    # frame, reach no threshold of 0.3. A resume trusting the first pass's record reads them
    # back, never asks for the file nor runs the detector again: the job's source holds the
    # recorded probabilities.
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    asked, given = asked_of_the_pull(monkeypatch), given_to_the_run(monkeypatch)
    closed_first: list[bool] = []

    def loading(*arguments: Any) -> Any:
        closed_first.append(all(built.closed for built in StandInDetector.built))
        return STAND_IN_MODELS

    monkeypatch.setattr(model, "load_models", loading)
    options = ("--window", "5")
    steps.stop = "window 0:1"
    stopped(tmp_path, source_path, "out", *options)
    [built] = StandInDetector.built
    assert (built.path, built.device, built.pushed) == (
        tmp_path / pull.DETECTOR,
        torch.device("cuda", 0),
        25,
    )
    assert built.closed and closed_first == [True] and asked == [True]
    expected = stand_in_probabilities(extracted(source_path))
    out = tmp_path / "out"
    index = out / "frame_index.bin"
    recorded = FrameIndex.read(index).probabilities
    assert recorded is not None and recorded.tobytes() == expected.tobytes()
    assert [found.tobytes() for found in given if found is not None] == [expected.tobytes()]
    content = json.loads((out / "manifest.json").read_text())
    assert content["settings"]["detector_model"] == {
        "name": pull.DETECTOR,
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert (content["settings"]["cut_threshold"], content["settings"]["cuts"]) == ("0.3", None)
    assert [(shot["start"], shot["end"]) for shot in content["shots"]] == [(0, 25)]
    kept = index.read_bytes()
    StandInDetector.built.clear()
    given.clear()
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert StandInDetector.built == [] and asked == [True, False]
    assert [found.tobytes() for found in given if found is not None] == [expected.tobytes()]
    said = "the first pass as recorded, the shot detector's probabilities with it"
    assert f"{out}: the same input and ffmpeg, {said}" in caplog.text
    assert index.read_bytes() == kept


def test_cuts_replace_the_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # With --cuts, the list replaces the detection: the shot detector's file neither asked of the
    # pull nor there, the detector never built, the first pass without its output; no
    # probabilities in the index, no detector nor threshold in the settings, the list there.
    from seedvr2x.media import scan
    from seedvr2x.media.index import FrameIndex

    source_path = job(tmp_path, monkeypatch)
    asked = asked_of_the_pull(monkeypatch)
    outputs: list[int | None] = []
    command = scan.command

    def commanded(path: Path, frames: int | None = None) -> list[str]:
        outputs.append(frames)
        return command(path, frames)

    monkeypatch.setattr(scan, "command", commanded)
    assert upscale(tmp_path, source_path, "out", *JOB, detector=False) == 0
    assert not (tmp_path / pull.DETECTOR).exists()
    assert asked == [False] and outputs == [None] and StandInDetector.built == []
    out = tmp_path / "out"
    assert FrameIndex.read(out / "frame_index.bin").probabilities is None
    settings = json.loads((out / "manifest.json").read_text())["settings"]
    assert "detector_model" not in settings and "cut_threshold" not in settings
    assert settings["cuts"] == [3, 4]


@pytest.mark.parametrize("first", ["detected", "cut list"])
def test_detection_or_cut_list_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    first: str,
) -> None:
    # A job detecting its shots resumed with a cut list, or the other way round, even one
    # cutting nowhere, once it has made a unit (its shot's latent, before the stop in its
    # window): another job, refused before its first pass, the detector's file, the threshold
    # and the list in the settings' differences, and why its cuts stay (a plan's change:
    # test_replanned). The detector's file is given by its pin: the refusal neither asks the pull
    # for the file of a detector it wouldn't run, nor warns of the record's index, with or without
    # probabilities, as a first pass to run again.
    source_path = job(tmp_path, monkeypatch)
    asked = asked_of_the_pull(monkeypatch)
    (tmp_path / "none.txt").write_text("")
    cut_list = ("--cuts", "none.txt")
    steps.stop = "window 0:0"
    stopped(tmp_path, source_path, "out", *(cut_list if first == "cut list" else ()))
    built = len(StandInDetector.built)
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *(cut_list if first == "detected" else ())) == 1
    assert "another job than the one asked" in caplog.text
    recorded, now = ("{", "null") if first == "detected" else ("null", "{")
    assert f"settings.detector_model: {recorded}" in caplog.text
    assert f"-> {now}" in caplog.text
    if first == "detected":
        threshold, cuts = '"0.3" -> null', "null -> []"
    else:
        threshold, cuts = 'null -> "0.3"', "[] -> null"
    assert f"settings.cut_threshold: {threshold}" in caplog.text
    assert f"settings.cuts: {cuts}" in caplog.text
    assert "These settings change only while it is a plan, before its first unit" in caplog.text
    assert len(StandInDetector.built) == built  # refused before any first pass
    assert asked == [first == "detected", False]
    assert "the first pass runs again" not in caplog.text
    assert "probabilities" not in caplog.text
    # Its index cut short, its first pass's record no longer trusted: no warning of a first pass
    # to run again either, before the refusal of a job that runs none, nor the detector's file
    # asked for the job recorded with a cut list.
    index = tmp_path / "out" / "frame_index.bin"
    index.write_bytes(index.read_bytes()[:-1])
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *(cut_list if first == "detected" else ())) == 1
    assert "another job than the one asked" in caplog.text
    assert "the first pass runs again" not in caplog.text and "frame_index.bin" not in caplog.text
    assert asked == [first == "detected", False, False]


def test_first_pass_run_again_detects_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The frame index a resume finds damaged: the first pass runs again, the detector with it, its
    # file asked of the pull again, the same probabilities recorded again.
    from seedvr2x.media.index import FrameIndex

    source_path = job(tmp_path, monkeypatch)
    asked = asked_of_the_pull(monkeypatch)
    steps.stop = "encode 0"
    stopped(tmp_path, source_path, "out")
    index = tmp_path / "out" / "frame_index.bin"
    kept = index.read_bytes()
    index.write_bytes(kept[:-1])
    StandInDetector.built.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out") == 0
    assert asked == [True, True] and len(StandInDetector.built) == 1
    assert "the first pass runs again" in caplog.text
    assert index.read_bytes() == kept
    found = FrameIndex.read(index).probabilities
    assert found is not None
    assert found.tobytes() == stand_in_probabilities(extracted(source_path)).tobytes()


def test_real_detector_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # TransNetV2 itself in the CLI's first pass, its weights from SEEDVR2X_MODEL_DIR (skipped
    # without) checked by their pin, run on the CPU in place of the job's GPU: the index file holds
    # runtime/detector.py's probabilities of the official extraction's frames, bit for bit at its
    # batch, the stand-in model doing the rest.
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime import detector

    models = os.environ.get("SEEDVR2X_MODEL_DIR")
    if not models or not (Path(models) / pull.DETECTOR).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {pull.DETECTOR}")
    weights_file = Path(models) / pull.DETECTOR
    (tmp_path / pull.DETECTOR).symlink_to(weights_file)
    monkeypatch.setattr(pull, "PINNED", {**pull.PINNED, pull.DETECTOR: DETECTOR_PIN})
    devices: list[torch.device] = []

    def on_the_cpu(path: Path, device: torch.device, **options: Any) -> Detector:
        devices.append(device)
        return REAL_DETECTOR(path, torch.device("cpu"), **options)

    monkeypatch.setattr(detector, "Detector", on_the_cpu)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out") == 0
    assert devices == [torch.device("cuda", 0)]
    found = FrameIndex.read(tmp_path / "out" / "frame_index.bin").probabilities
    assert found is not None
    with REAL_DETECTOR(weights_file, torch.device("cpu"), batch=BATCH) as reference:
        frames = extracted(source_path)
        expected = np.concatenate((reference.push(frames), reference.finish()))
    assert len(expected) == 25 and found.tobytes() == expected.tobytes()


# The probabilities a scripted detector gives job()'s 25 frames, whatever they show
# (ScriptedDetector): at 0.3, cuts at 5, 15 and 22, after runs peaking on 4 (0.9), on 14 of 14
# and 15 (0.6 both: the first) and on 21 (float32's 0.3, exactly); possible cuts at 10, after a
# run of 9 and 10 peaking at 0.2 on 9, and at 19, after 18's 0.12; a peak on the last frame, 24,
# which cuts nothing. At 0.5, cuts at 5 and 15, 22 a possible cut.
SCRIPT = np.full(25, 0.01, np.float32)
SCRIPT[[3, 4, 9, 10, 14, 15, 18, 21, 24]] = (0.4, 0.9, 0.2, 0.15, 0.6, 0.6, 0.12, 0.3, 0.5)
# The cut list's lines --plan writes of them at 0.3, each cut's time at 25 fps, each probability
# its float32's exact decimals, cut: 0.9's is 0.899999976, 0.12's 0.119999997.
PLANNED = [
    "5 0.899 0:00:00.200",
    "# 10 0.200 0:00:00.400",
    "15 0.600 0:00:00.600",
    "# 19 0.119 0:00:00.760",
    "22 0.300 0:00:00.880",
]
# The options of the jobs cut so: windows of 5 latents, a segment per shot.
CUT = ("--window", "5", "--min-segment", "0")


class ScriptedDetector(StandInDetector):
    """The stand-in detector giving SCRIPT's probabilities in order, frame by frame."""

    def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        first = self.pushed
        super().push(frames)
        return SCRIPT[first : self.pushed].copy()


@pytest.fixture
def scripted(steps: Steps, monkeypatch: pytest.MonkeyPatch) -> Steps:
    """The stand-in's steps (steps), the shot detector ScriptedDetector."""
    from seedvr2x.runtime import detector

    monkeypatch.setattr(detector, "Detector", ScriptedDetector)
    return steps


def first_passes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The source of each first pass from now on (source.scan)."""
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *detector: Any) -> object:
        scans.append(path)
        return scan(path, *detector)

    monkeypatch.setattr(examined, "scan", counted)
    return scans


def cut_lines(path: Path) -> list[str]:
    """The lines of a cut list --plan wrote that are cuts, or possible cuts, commented: a frame,
    a probability for a detection's, a time."""
    cut = r"(# )?\d+( [01]\.\d{3})? \d+:\d\d:\d\d\.\d{3}"
    return [line for line in path.read_text().splitlines() if re.fullmatch(cut, line)]


def shots_of(directory: Path) -> list[tuple[int, int]]:
    content = json.loads((directory / "manifest.json").read_text())
    return [(shot["start"], shot["end"]) for shot in content["shots"]]


def written(directory: Path) -> list[int]:
    """Each frame's index in the job, as the stand-in wrote it, segment after segment."""
    names = sorted(path.name for path in directory.glob("seg_*.mkv"))
    return [index for name in names for index in indexes(directory / name)]


def encoded(steps: Steps) -> list[int]:
    """The first frame of each shot the stand-in encoded."""
    return [int(call.split()[1]) for call in steps.calls if call.startswith("encode ")]


@pytest.mark.parametrize(
    ("threshold", "recorded", "bounds", "said"),
    [
        (None, "0.3", [0, 5, 15, 22, 25], "4 shots: 3 cuts detected at 0.3, and 2 possible cuts"),
        ("0.5", "0.5", [0, 5, 15, 25], "3 shots: 2 cuts detected at 0.5, and 3 possible cuts"),
        (
            "0.15",
            "0.15",
            [0, 5, 10, 15, 22, 25],
            "5 shots: 4 cuts detected at 0.15, and 1 possible",
        ),
        ("1", "1.0", [0, 25], "1 shot: 0 cuts detected at 1.0, and 5 possible cuts"),
    ],
)
def test_cuts_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    threshold: str | None,
    recorded: str,
    bounds: list[int],
    said: str,
) -> None:
    # Without --cuts, the shots come from the detection (DESIGN.md, Shot detection): a cut after
    # the peak of each run reaching --cut-threshold, 0.3 by default, recorded as Python writes the
    # float; the log says how the shots were found; every shot run, every frame written.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    option = () if threshold is None else ("--cut-threshold", threshold)
    assert upscale(tmp_path, source_path, "out", *CUT, *option) == 0
    out = tmp_path / "out"
    assert shots_of(out) == list(pairwise(bounds))
    settings = json.loads((out / "manifest.json").read_text())["settings"]
    assert (settings["cut_threshold"], settings["cuts"]) == (recorded, None)
    assert said in caplog.text
    assert encoded(scripted) == bounds[:-1]
    assert written(out) == list(range(25))


def test_cut_threshold_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With --cuts, whose list replaces the detection (DESIGN.md, Shot detection), refused before
    # anything is made; out of (0, 1], refused, saying the range.
    source_path = job(tmp_path, monkeypatch)
    text = refused(tmp_path, source_path, caplog, *JOB, "--cut-threshold", "0.5")
    said = "--cut-threshold 0.5: refused with --cuts, whose list replaces the detection"
    assert said in text
    assert not (tmp_path / "out").exists() and steps.calls == []
    for given in ("0", "-0.1", "1.01", "nan", "inf", "0.3x", ""):
        with pytest.raises(SystemExit) as exited:
            upscale(tmp_path, source_path, "out", "--cut-threshold", given)
        assert exited.value.code == 2
        error = capsys.readouterr().err
        assert f"argument --cut-threshold: {given!r}: not a threshold; give a probability" in error
        assert "above 0 and at most 1, such as 0.3, the default" in error
    for given, value in (("0.3", 0.3), ("1", 1.0), ("1e-1", 0.1), (" 0.30 ", 0.3)):
        assert cli._threshold(given) == value  # pyright: ignore[reportPrivateUsage]


def test_plan_then_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan FILE with an output directory: the first pass, the shot detector in it, then the cut
    # list in FILE and the counts said; the job's manifest and frame index made, its shots the
    # detected ones, no unit, the models never loaded. The cut list in --cuts' format: a header,
    # then the cuts and the possible ones, commented, in frame order, each with its probability
    # and time. The same command without --plan runs the plan, its first pass as recorded, never
    # run again.
    from seedvr2x.runtime.job import read_cuts

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    # The models load, for the time estimates' calibration (DESIGN.md, Memory planner).
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert len(passes) == 1 and len(StandInDetector.built) == 1 and scripted.calls == []
    out = tmp_path / "out"
    assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
    content = json.loads((out / "manifest.json").read_text())
    assert shots_of(out) == [(0, 5), (5, 15), (15, 22), (22, 25)]
    assert not any(shot["encoded"] or shot["windows_done"] for shot in content["shots"])
    assert not any(segment["finished"] for segment in content["segments"])
    said = "3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3: the cut list in"
    assert f"{said} {plan}" in caplog.text
    assert plan.read_text().splitlines() == [
        "# seedvr2x cut list of in.mkv: 25 frames at 25 fps",
        f"# Shot detector: TransNetV2, {pull.DETECTOR} (SHA-256 e3b0c442), threshold 0.3",
        "# Found: 3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3",
        "#",
        "# A line per cut: the frame it cuts at, the first frame of a shot, counted from zero;",
        "# the shot detector's peak probability, on the frame before it; the time of that frame",
        "# from the source's start, H:MM:SS.mmm, as a player shows it. seedvr2x reads the frame",
        "# number alone. A possible cut is a commented line: checking it is a jump to its time in",
        "# a player.",
        "# Uncomment a possible cut to keep it, delete a line to drop a cut, then run with --cuts",
        "# FILE, this file.",
        *PLANNED,
    ]
    assert read_cuts(plan) == [5, 15, 22]
    # Every "# <digit>" line uncommented, as a search over the file keeps every possible cut: the
    # cuts and the possible cuts, nothing of the header's (its counts begin with a word).
    every = tmp_path / "every.txt"
    every.write_text(re.sub(r"(?m)^# (?=\d)", "", plan.read_text()))
    assert read_cuts(every) == [5, 10, 15, 19, 22]
    passes.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    assert passes == [] and len(StandInDetector.built) == 1
    said = "the first pass as recorded, the shot detector's probabilities with it"
    assert f"{out}: the same input and ffmpeg, {said}" in caplog.text
    assert encoded(scripted) == [0, 5, 15, 22]
    assert written(out) == list(range(25))


@needs_x264
def test_plan_and_run_of_a_source_with_missing_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A source of 27 places whose frames 5 and 6 are missing: the shot detector sees its 25
    # decoded frames, SCRIPT's probabilities theirs, recorded bit for bit; on the grid they are
    # spread so that the cuts fall at their frames' own places (media/grid.py, spread): the cut
    # the detector finds at its frame 5, the first after the hole, is at place 7, never on a held
    # copy of frame 4, and its later ones two places on. The cut list --plan writes counts in
    # places, its frames and its times those of the full file's frames; the run of the plan
    # makes 27 frames, its shots the places'.
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.media.scan import scan
    from seedvr2x.runtime.cuts import time_of, timecode

    caplog.set_level(logging.INFO, logger="seedvr2x")
    monkeypatch.chdir(tmp_path)
    source_path, full = holed_source(tmp_path, 27, (5, 6))
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert len(passes) == 1 and scripted.calls == []
    assert "holes.mkv: 2 frames missing after frame 4: " in caplog.text
    out = tmp_path / "out"
    assert shots_of(out) == [(0, 7), (7, 17), (17, 24), (24, 27)]
    index = FrameIndex.from_bytes((out / "frame_index.bin").read_bytes())
    assert index.frames == 27 and index.held is not None and index.probabilities is not None
    assert np.flatnonzero(index.held).tolist() == [5, 6]
    seen = index.decoded().probabilities
    assert seen is not None and seen.tobytes() == SCRIPT.tobytes()
    # Frame 4, 0.9, held through the hole: its last place takes its probability, the change to
    # the next frame coming after it; its own and the first held take the lower of its and
    # frame 3's, 0.4.
    assert index.probabilities[3:8].tolist() == SCRIPT[[3, 3, 3, 4, 5]].tolist()
    whole = scan(full).index
    assert whole is not None and whole.frames == 27
    # Its header says that its frames count places, and what was filled (cuts.places_note): a
    # list numbered by a tool that counts decoded frames would cut at 5, 15 and 22.
    assert plan.read_text().splitlines()[:-5] == [
        "# seedvr2x cut list of holes.mkv: 27 frames at 24000/1001 fps",
        f"# Shot detector: TransNetV2, {pull.DETECTOR} (SHA-256 e3b0c442), threshold 0.3",
        "# Found: 3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3",
        "#",
        "# A line per cut: the frame it cuts at, the first frame of a shot, counted from zero;",
        "# the shot detector's peak probability, on the frame before it; the time of that frame",
        "# from the source's start, H:MM:SS.mmm, as a player shows it. seedvr2x reads the frame",
        "# number alone. A possible cut is a commented line: checking it is a jump to its time in",
        "# a player.",
        "# Frame numbers count places, the missing frames' filled ones among them: 2 frames",
        "# missing after frame 4, 27 places for 25 frames decoded. A list numbered by a tool that",
        "# counts the decoded frames is off by the holes before each cut, its number lower by the",
        "# frames missing before it.",
        "# Uncomment a possible cut to keep it, delete a line to drop a cut, then run with --cuts",
        "# FILE, this file.",
    ]
    assert cut_lines(plan) == [
        f"7 0.899 {timecode(time_of(whole, 7))}",
        f"# 12 0.200 {timecode(time_of(whole, 12))}",
        f"17 0.600 {timecode(time_of(whole, 17))}",
        f"# 21 0.119 {timecode(time_of(whole, 21))}",
        f"24 0.300 {timecode(time_of(whole, 24))}",
    ]
    assert timecode(time_of(whole, 7)) == "0:00:00.292"
    passes.clear()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    assert passes == [] and "holes.mkv: 2 frames missing after frame 4: " in caplog.text
    assert encoded(scripted) == [0, 7, 17, 24]
    assert written(out) == list(range(27))


def test_replanned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan, a job that has made no unit, is planned again by a run whose settings differ in its
    # cuts alone: another threshold, its cuts derived again from the recorded probabilities, no
    # first pass, a stop's leftovers of the plan's own layout discarded; then a cut list, the
    # plan's own with a possible cut uncommented, likewise, the probabilities kept in the frame
    # index; the layout and settings rewritten, said; run, the new shots.
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime.job import read_cuts

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(first)) == 0
    out = tmp_path / "out"
    index = (out / "frame_index.bin").read_bytes()
    # What a stop in shot 22's encode leaves, which the plan at 0.5 has no shot for.
    left = out / "resume" / "shot_000022"
    left.mkdir(parents=True)
    for name in ("input.mkv.partial", "latent.pt.partial"):
        (left / name).write_bytes(b"left")
    passes.clear()
    caplog.clear()
    options = (*CUT, "--cut-threshold", "0.5", "--plan", str(second))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert passes == [] and len(StandInDetector.built) == 1
    assert shots_of(out) == [(0, 5), (5, 15), (15, 25)]
    content = json.loads((out / "manifest.json").read_text())
    assert content["settings"]["cut_threshold"] == "0.5"
    said = 're-planned, its job having made no unit yet: settings.cut_threshold: "0.3" -> "0.5"'
    assert f"{out}: {said}; 2 leftovers discarded" in caplog.text
    assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
    assert (out / "frame_index.bin").read_bytes() == index
    assert cut_lines(second) == [*PLANNED[:3], "# 19 0.119 0:00:00.760", "# 22 0.300 0:00:00.880"]
    second.write_text(second.read_text().replace("# 10 ", "10 "))
    assert read_cuts(second) == [5, 10, 15]
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *CUT, "--cuts", str(second)) == 0
    assert passes == [] and len(StandInDetector.built) == 1
    for difference in (
        "settings.cuts: null -> [5, 10, 15]",
        'settings.cut_threshold: "0.5" -> null',
        "settings.detector_model: {",
    ):
        assert difference in caplog.text
    content = json.loads((out / "manifest.json").read_text())
    assert (
        "cut_threshold" not in content["settings"] and "detector_model" not in content["settings"]
    )
    assert shots_of(out) == [(0, 5), (5, 10), (10, 15), (15, 25)]
    assert encoded(scripted) == [0, 5, 10, 15]
    recorded = FrameIndex.read(out / "frame_index.bin").probabilities
    assert recorded is not None and recorded.tobytes() == SCRIPT.tobytes()
    assert written(out) == list(range(25))


def test_plan_taken_up_narrowly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan takes any setting its first pass doesn't read (resume.FIRST_PASS_SETTINGS, decided
    # on 2026-10-21; until then its cuts alone): another seed, alone or with another threshold,
    # is planned again from its record, no first pass. The frame rate, which the pass reads,
    # alone or with another threshold, is another job, refused before any first pass, the plan
    # untouched. Once a unit is made, the job's settings are its own: another threshold or a cut
    # list refused, saying why; the same command resumes.
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    plan = str(tmp_path / "plan.txt")
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", plan) == 0
    out = tmp_path / "out"
    kept = contents(out)
    passes.clear()
    text = refused(tmp_path, source_path, caplog, *CUT, "--frame-rate", "50")
    assert "another job than the one asked" in text and 'settings.frame_rate: null -> "50"' in text
    assert "These settings change only" not in text
    text = refused(
        tmp_path, source_path, caplog, *CUT, "--frame-rate", "50", "--cut-threshold", "0.5"
    )
    assert 'settings.frame_rate: null -> "50"' in text
    assert 'settings.cut_threshold: "0.3" -> "0.5"' in text
    assert contents(out) == kept and passes == []
    for seed in (("--seed", "7"), ("--seed", "7", "--cut-threshold", "0.5"), ()):
        assert upscale(tmp_path, source_path, "out", *CUT, *seed, "--plan", plan) == 0
        settings = json.loads((out / "manifest.json").read_text())["settings"]
        assert settings["seed"] == (7 if seed else 42)
    assert passes == []
    # A unit made: shot 0's latent, the stop in shot 5's encode.
    scripted.stop = "encode 5"
    stopped(tmp_path, source_path, "out", *CUT)
    kept = contents(out)
    for other, difference in (
        (("--cut-threshold", "0.5"), 'settings.cut_threshold: "0.3" -> "0.5"'),
        (("--cuts", "cuts.txt"), "settings.cuts: null -> [3, 4]"),
    ):
        text = refused(tmp_path, source_path, caplog, *CUT, *other)
        assert "another job than the one asked" in text and difference in text
        said = "These settings change only while it is a plan, before its first unit, and it has"
        assert (
            f"{said} made some: its settings are its own; another output directory takes new ones"
            in text
        )
    assert contents(out) == kept and passes == []
    scripted.stop = None
    scripted.calls.clear()
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    assert encoded(scripted) == [5, 15, 22]


def test_one_file_plan_kept_in_its_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted: Steps
) -> None:
    # -o x.mkv --plan FILE: FILE written, which replaces the one there, whole, with the
    # playlists of its cuts and of its possible cuts beside it, and the job's plan left in the
    # work directory, its manifest and frame index, as in an output directory: no unit, no file,
    # no checksums; the models never loaded.
    from seedvr2x.runtime.job import read_cuts

    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    (tmp_path / pull.DETECTOR).write_bytes(b"")
    plan = tmp_path / "plan.txt"
    plan.write_text("an older plan")
    before = sorted(os.listdir(tmp_path))
    # The models load, for the time estimates' calibration (DESIGN.md, Memory planner).
    assert upscale(tmp_path, source_path, "one.mkv", *SPLIT, "--plan", str(plan)) == 0
    made = ["one.mkv.work", "plan.txt.cuts.m3u8", "plan.txt.possible.m3u8"]
    assert sorted(os.listdir(tmp_path)) == sorted([*before, *made])
    work = tmp_path / "one.mkv.work"
    assert sorted(os.listdir(work)) == ["frame_index.bin", "manifest.json"]
    assert cut_lines(plan) == PLANNED and read_cuts(plan) == [5, 15, 22]
    assert scripted.calls == []


# The comments --plan's playlist begins with (cuts.playlist), and its entries of SCRIPT's
# possible cuts at 0.3, 10 and 19, in job()'s 25 frames at 25 fps: a second before either is
# before the source's start, 3 seconds after past its last frame, at 0.96 s.
PLAYLIST_HEADER = [
    "#EXTM3U",
    "# seedvr2x: the possible cuts of the cut list beside this file, to check in VLC, which",
    "# plays an entry per possible cut, from 1 s before its time to 3 s after.",
    "# It names the source by its file name alone: it plays from the source's directory.",
]
PLAYLIST = [
    "#EXTINF:-1,possible cut 10 at 0:00:00.400 (0.200)",
    "#EXTVLCOPT:start-time=0.000",
    "#EXTVLCOPT:stop-time=0.960",
    "in.mkv",
    "#EXTINF:-1,possible cut 19 at 0:00:00.760 (0.119)",
    "#EXTVLCOPT:start-time=0.000",
    "#EXTVLCOPT:stop-time=0.960",
    "in.mkv",
]
PLAYLIST_SAID = "for VLC, an entry each, from 1 second before its cut to 3 seconds after"
# The detected cuts' playlist, the second (cuts.OF_DETECTED): its comments, and its entries of
# SCRIPT's cuts at 0.3, 5, 15 and 22.
DETECTED_HEADER = [
    "#EXTM3U",
    "# seedvr2x: the detected cuts of the cut list beside this file, to check in VLC, which",
    "# plays an entry per cut, from 1 s before its time to 3 s after.",
    "# It names the source by its file name alone: it plays from the source's directory.",
]
DETECTED = [
    "#EXTINF:-1,cut 5 at 0:00:00.200 (0.899)",
    "#EXTVLCOPT:start-time=0.000",
    "#EXTVLCOPT:stop-time=0.960",
    "in.mkv",
    "#EXTINF:-1,cut 15 at 0:00:00.600 (0.600)",
    "#EXTVLCOPT:start-time=0.000",
    "#EXTVLCOPT:stop-time=0.960",
    "in.mkv",
    "#EXTINF:-1,cut 22 at 0:00:00.880 (0.300)",
    "#EXTVLCOPT:start-time=0.000",
    "#EXTVLCOPT:stop-time=0.960",
    "in.mkv",
]


def before_the_load(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    """--plan's log, level and message, before its models load for the time estimates'
    calibration (DESIGN.md, Memory planner): its plan and cut list's lines, the time's after."""
    said = [(r.levelname, r.getMessage()) for r in caplog.records]
    loaded = [i for i, (_, message) in enumerate(said) if message.startswith("models loaded in")]
    return said[: loaded[0]] if loaded else said


def test_plan_writes_the_playlists_of_its_cuts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan FILE also writes two playlists beside the list (DESIGN.md, Shot detection; design,
    # 2026-10-12), FILE.cuts.m3u8, an entry per detected cut, and FILE.possible.m3u8, an entry
    # per possible cut, each in frame order and never the other's cuts, and says what to do with
    # each: beside the source, open it in VLC. Planned again at 0.5 with a FILE elsewhere, the
    # cut at 22 a possible one: its entry gone from the one, with the others in the other, and
    # the playlists to move beside the source. At 0.1, no possible cut: no playlist of them, the
    # one from before left, and said; at 1, no detected cut, likewise. A cut list's plan has
    # neither.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    detected, playlist = tmp_path / "plan.txt.cuts.m3u8", tmp_path / "plan.txt.possible.m3u8"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert detected.read_text().splitlines() == [*DETECTED_HEADER, *DETECTED]
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *PLAYLIST]
    assert cut_lines(plan) == PLANNED
    said = [record.getMessage() for record in caplog.records]
    beside = "beside the source, which it names by its file name alone: open it in VLC"
    # Then the time estimate's lines, its models loaded for it.
    first = next(i for i, line in enumerate(said) if line.endswith(f"the cut list in {plan}"))
    assert said[first : first + 3] == [
        f"3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3: the cut list in {plan}",
        f"{detected}: a playlist of the 3 detected cuts {PLAYLIST_SAID}; {beside}",
        f"{playlist}: a playlist of the 2 possible cuts {PLAYLIST_SAID}; {beside}",
    ]
    assert sorted(path.name for path in tmp_path.glob("plan*")) == [
        "plan.txt",
        "plan.txt.cuts.m3u8",
        "plan.txt.possible.m3u8",
    ]
    (tmp_path / "plans").mkdir()
    caplog.clear()
    options = (*CUT, "--cut-threshold", "0.5", "--plan", "plans/again")
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert (tmp_path / "plans" / "again.cuts.m3u8").read_text().splitlines() == [
        *DETECTED_HEADER,
        *DETECTED[:8],
    ]
    assert (tmp_path / "plans" / "again.possible.m3u8").read_text().splitlines() == [
        *PLAYLIST_HEADER,
        *PLAYLIST,
        "#EXTINF:-1,possible cut 22 at 0:00:00.880 (0.300)",
        "#EXTVLCOPT:start-time=0.000",
        "#EXTVLCOPT:stop-time=0.960",
        "in.mkv",
    ]
    move = "it names the source by its file name alone, in.mkv: move it beside the source, then"
    said = [record.getMessage() for record in caplog.records]
    first = next(i for i, line in enumerate(said) if line.startswith("plans/again.cuts.m3u8"))
    assert said[first : first + 2] == [
        f"plans/again.cuts.m3u8: a playlist of the 2 detected cuts {PLAYLIST_SAID}; {move} open"
        " it in VLC",
        f"plans/again.possible.m3u8: a playlist of the 3 possible cuts {PLAYLIST_SAID}; {move}"
        " open it in VLC",
    ]
    assert sorted(os.listdir(tmp_path / "plans")) == [
        "again",
        "again.cuts.m3u8",
        "again.possible.m3u8",
    ]
    caplog.clear()
    options = (*CUT, "--cut-threshold", "0.1", "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *PLAYLIST]
    assert len(detected.read_text().splitlines()) == len(DETECTED_HEADER) + 4 * 5
    left = "a playlist there is an earlier plan's, its entries not this plan's"
    # The plans own last word, before the time estimates lines.
    assert [(r.levelname, r.getMessage()) for r in caplog.records if r.levelno > logging.INFO][
        -1
    ] == (
        "WARNING",
        f"{playlist}: left as it is: this plan has no possible cut, so no playlist of them; {left}",
    )
    caplog.clear()
    options = (*CUT, "--cut-threshold", "1", "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert len(detected.read_text().splitlines()) == len(DETECTED_HEADER) + 4 * 5
    assert len(playlist.read_text().splitlines()) == len(PLAYLIST_HEADER) + 4 * 5
    assert before_the_load(caplog)[-2:] == [
        (
            "WARNING",
            f"{detected}: left as it is: this plan has no detected cut, so no playlist of them;"
            f" {left}",
        ),
        ("INFO", f"{playlist}: a playlist of the 5 possible cuts {PLAYLIST_SAID}; {beside}"),
    ]
    caplog.clear()
    options = ("--cuts", "cuts.txt", "--plan", "listed.txt")
    assert upscale(tmp_path, source_path, "out", *CUT, *options) == 0
    assert sorted(path.name for path in tmp_path.glob("listed*")) == ["listed.txt"]
    assert not [record for record in caplog.records if record.name == "seedvr2x.runtime.cuts"]


def test_plan_file_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A --plan FILE in the place of what the job reads or writes, or that can't be written,
    # refused before the first pass: nothing made.
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    for output, plan, said in (
        ("out", "in.mkv", "the source; it names the cut list's file"),
        ("one.mkv", "one.mkv", "the output; it names the cut list's file"),
        ("one.mkv", "one.mkv.crc32", "the output's checksums; it names the cut list's file"),
        (
            "one.mkv",
            "one.mkv.work/plan.txt",
            f"in the work directory of {tmp_path / 'one.mkv'}, which holds the job's own files"
            " only",
        ),
        ("out", "out/plan.txt", "in the output directory, which holds the job's own files only"),
        ("out", "out", "in the output directory, which holds the job's own files only"),
        ("out", "missing/plan.txt", "no directory missing to write it in"),
        ("out", ".", "a directory; it names the cut list's file"),
    ):
        text = refused(tmp_path, source_path, caplog, "--plan", plan, output=output)
        assert f"--plan {plan}: {said}" in text
    # The cut list given: its plan written over it would lose its comments, the possible cuts it
    # kept commented among them. By its path or by a link to it.
    listed = (tmp_path / "cuts.txt").read_bytes()
    (tmp_path / "link.txt").symlink_to("cuts.txt")
    for plan in ("cuts.txt", "link.txt", str(tmp_path / "cuts.txt")):
        text = refused(tmp_path, source_path, caplog, "--cuts", "cuts.txt", "--plan", plan)
        said = "the cut list given (--cuts), which it would replace, its comments lost; it names"
        assert f"--plan {plan}: {said} another file" in text
    assert (tmp_path / "cuts.txt").read_bytes() == listed
    # A directory that takes no new file: said now, not once the first pass has run.
    if os.geteuid() != 0:  # root writes anywhere
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o555)
        text = refused(tmp_path, source_path, caplog, "--plan", "locked/plan.txt", output="one.mkv")
        assert "--plan locked/plan.txt: no file can be written in locked: Permission denied" in text
    assert passes == [] and not (tmp_path / "out").exists()
    assert not (tmp_path / "one.mkv").exists() and not (tmp_path / "one.mkv.work").exists()


def test_plan_playlist_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The playlists of FILE's cuts and possible cuts go beside it, as FILE.cuts.m3u8 and
    # FILE.possible.m3u8: with either in the place of what the job reads or writes, FILE is
    # refused as for its own place, before the first pass, nothing made: the source; a .mkv
    # path's finished file, its checksums, anything in its work directory; anything in an output
    # directory, or that directory itself, one named as the playlist; a directory. Each place by
    # a link at the playlist's name, the one way there for a .mkv path's, whose names end
    # otherwise. With --cuts, whose plan has neither and no playlist, it isn't.
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    in_directory = "is in the output directory, which holds the job's own files only"
    work = f"is in the work directory of {tmp_path / 'one.mkv'}, which holds the job's own files"
    for name, cuts in (("cuts", "detected cuts"), ("possible", "possible cuts")):
        ending = f"{name}.m3u8"
        (tmp_path / f"a.{ending}").symlink_to("in.mkv")
        (tmp_path / f"b.txt.{ending}").symlink_to("one.mkv")
        (tmp_path / f"c.{ending}").symlink_to("out/c")
        (tmp_path / f"d.cuts.{ending}").mkdir()
        (tmp_path / f"e.{ending}").symlink_to("one.mkv.crc32")
        (tmp_path / f"f.{ending}").symlink_to(f"one.mkv.work/f.{ending}")
        for output, plan, said in (
            ("out", "a", f"a.{ending} beside it, is the source"),
            ("one.mkv", "a", f"a.{ending} beside it, is the source"),
            ("one.mkv", "b.txt", f"b.txt.{ending} beside it, is the output"),
            ("one.mkv", "e", f"e.{ending} beside it, is the output's checksums"),
            ("one.mkv", "f", f"f.{ending} beside it, {work} only"),
            ("out", "c", f"c.{ending} beside it, {in_directory}"),
            (f"p.{ending}", "p", f"p.{ending} beside it, {in_directory}"),
            ("out", "d.cuts", f"d.cuts.{ending} beside it, is a directory"),
            ("one.mkv", "d.cuts", f"d.cuts.{ending} beside it, is a directory"),
        ):
            text = refused(tmp_path, source_path, caplog, "--plan", plan, output=output)
            expected = f"--plan {plan}: the playlist of its {cuts}, {said}; it names another"
            assert f"{expected} file" in text
            assert not (tmp_path / plan).exists()
        assert passes == [] and not (tmp_path / "out").exists()
        assert not (tmp_path / f"p.{ending}").exists()
        assert not (tmp_path / "one.mkv").exists() and not (tmp_path / "one.mkv.work").exists()
        # The other playlist's places are free: these names are this one's alone.
        for path in sorted(tmp_path.glob(f"?.*{ending}")):
            path.rmdir() if path.is_dir() else path.unlink()
    # A link to a .mkv path's own files is no obstacle to a directory's job, nor a directory's to
    # a .mkv path's: each job is held to its own places.
    (tmp_path / "a.cuts.m3u8").symlink_to("in.mkv")
    content = source_path.read_bytes()
    assert upscale(tmp_path, source_path, "out", "--cuts", "cuts.txt", "--plan", "a") == 0
    assert (tmp_path / "a").is_file() and (tmp_path / "a.cuts.m3u8").is_symlink()
    assert source_path.read_bytes() == content


def test_plan_places_are_files_not_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # FILE's place and each playlist's are held to the job's FILES, not to their resolved names
    # alone (cli._same, as the output is held to the source): on a file system that takes a name
    # in any case (macOS's default, a Windows disk under WSL), FILM.MKV is film.mkv under a name
    # no resolved path makes equal, and the cut list renamed into its place would replace the
    # source. Such a disk can't be made here: ANOTHER HARD LINK to the file stands in for it, one
    # file under two names that each resolve to themselves. Refused before the first pass, each
    # with the refusal of the file it is: the source, and a finished -o one.mkv job's file and
    # its checksums; their bytes and inodes untouched, the other name still theirs.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    said = {
        "in.mkv": "the source",
        "one.mkv": "the output",
        "one.mkv.crc32": "the output's checksums",
    }
    kept = {
        name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_ino) for name in said
    }
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    for name, what in said.items():
        # The source is every job's; the finished file and its checksums a .mkv path's own.
        outputs = ("out", "one.mkv") if name == "in.mkv" else ("one.mkv",)
        os.link(tmp_path / name, plan)
        for output in outputs:
            text = refused(tmp_path, source_path, caplog, "--plan", "plan.txt", output=output)
            assert f"--plan plan.txt: {what}; it names the cut list's file" in text
            assert plan.stat().st_ino == kept[name][1]
        plan.unlink()
        for which, cuts in (("cuts", "detected cuts"), ("possible", "possible cuts")):
            playlist = tmp_path / f"plan.txt.{which}.m3u8"
            os.link(tmp_path / name, playlist)
            for output in outputs:
                text = refused(tmp_path, source_path, caplog, "--plan", "plan.txt", output=output)
                expected = f"--plan plan.txt: the playlist of its {cuts}, {playlist.name} beside it"
                assert f"{expected}, is {what}; it names another file" in text
                assert playlist.stat().st_ino == kept[name][1] and not plan.exists()
            playlist.unlink()
    # The cut list given (--cuts) likewise, compared with FILE as a file (design, 2026-10-12):
    # --cuts list.txt --plan LIST.TXT on such a disk would replace the list by its plan, its
    # comments lost. Refused as the list under its own name is (test_plan_file_refused), for a
    # directory's job and a .mkv path's; another file of the same content is no such file.
    listed = tmp_path / "cuts.txt"
    listed.write_text("# mine\n3\n4  # a flash\n")
    content, inode = listed.read_bytes(), listed.stat().st_ino
    os.link(listed, plan)
    for output in ("out", "two.mkv"):
        options = ("--cuts", "cuts.txt", "--plan", "plan.txt")
        text = refused(tmp_path, source_path, caplog, *options, output=output)
        given = "the cut list given (--cuts), which it would replace, its comments lost; it names"
        assert f"--plan plan.txt: {given} another file" in text
        assert (plan.read_bytes(), plan.stat().st_ino) == (content, inode)
    assert (listed.read_bytes(), listed.stat().st_ino) == (content, inode)
    assert passes == [] and not (tmp_path / "out").exists()
    assert not (tmp_path / "one.mkv.work").exists() and not (tmp_path / "two.mkv.work").exists()
    assert kept == {
        name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_ino) for name in said
    }
    # A copy of the list is another file: its plan is written there, the list as it was.
    plan.unlink()
    plan.write_bytes(content)
    assert upscale(tmp_path, source_path, "out", "--cuts", "cuts.txt", "--plan", "plan.txt") == 0
    assert plan.read_bytes() != content and listed.read_bytes() == content


def test_plan_of_a_cut_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With --cuts, the plan of that list: FILE holds it normalised, each cut with its time, no
    # possible cut; the shot detector never built; the counts said.
    from seedvr2x.runtime.job import read_cuts

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "cuts.txt").write_text("# mine\n4 a score\n\n12  # a flash\n")
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, source_path, "out", "--cuts", "cuts.txt", "--plan", str(plan)) == 0
    assert StandInDetector.built == []
    assert f"2 cuts from the cut list cuts.txt (3 shots): the cut list in {plan}" in caplog.text
    assert plan.read_text().splitlines() == [
        "# seedvr2x cut list of in.mkv: 25 frames at 25 fps",
        "# Found: 2 cuts from the cut list cuts.txt (3 shots)",
        "#",
        "# A line per cut: the frame it cuts at, the first frame of a shot, counted from zero;",
        "# the time of that frame from the source's start, H:MM:SS.mmm, as a player shows it.",
        "# seedvr2x reads the frame number alone.",
        "# Delete a line to drop a cut, add one to cut at its frame, then run with --cuts FILE,",
        "# this file.",
        "4 0:00:00.160",
        "12 0:00:00.480",
    ]
    assert read_cuts(plan) == [4, 12]
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]["cuts"] == [
        4,
        12,
    ]


def test_plan_cut_by_a_list_detected_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan cut by a list holds no probabilities: planned again without one, its first pass
    # runs again, the shot detector with it, which said; its record then holds them, the manifest
    # naming the new index, and another threshold needs no decode.
    from seedvr2x.media.index import FrameIndex
    from seedvr2x.runtime.job import read_cuts

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    listed = (*CUT, "--cuts", "cuts.txt", "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *listed) == 0
    out = tmp_path / "out"
    assert FrameIndex.read(out / "frame_index.bin").probabilities is None
    passes.clear()
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert len(passes) == 1 and len(StandInDetector.built) == 1
    said = "no shot probabilities in its record, its cuts given by a list: the first pass runs"
    assert f"{out}: {said} again, the shot detector with it" in caplog.text
    assert 'settings.cut_threshold: null -> "0.3"' in caplog.text
    index = out / "frame_index.bin"
    found = FrameIndex.read(index).probabilities
    assert found is not None and found.tobytes() == SCRIPT.tobytes()
    entry = json.loads((out / "manifest.json").read_text())["input"]["index"]
    assert entry["sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert shots_of(out) == [(0, 5), (5, 15), (15, 22), (22, 25)]
    assert read_cuts(plan) == [5, 15, 22]
    passes.clear()
    # What a stop in shot 22's encode leaves: one leftover, said as one.
    left = out / "resume" / "shot_000022"
    left.mkdir(parents=True)
    (left / "latent.pt.partial").write_bytes(b"left")
    options = (*CUT, "--cut-threshold", "0.5", "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert passes == [] and shots_of(out) == [(0, 5), (5, 15), (15, 25)]
    assert '"0.3" -> "0.5"; 1 leftover discarded' in caplog.text and not (out / "resume").exists()


def test_replanned_in_another_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan planned again under another ffmpeg. Not accepted: refused before any first pass,
    # the refusal naming --accept-env-change, which takes it. Accepted: its first pass runs
    # again, and its manifest is rewritten at once in that environment, the change recorded,
    # naming the index that pass made: a plan has made no unit in the environment before. Written
    # in the environment recorded, it named the other ffmpeg's index under the first one's name,
    # and the first ffmpeg trusted it. The other ffmpeg then finds its record its own, no pass
    # run again; the first one, a job of another environment.
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    out = tmp_path / "out"
    index, manifest = out / "frame_index.bin", out / "manifest.json"
    kept, first = contents(out), index.read_bytes()
    check = ffmpeg.check
    monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    otherwise[0] = True
    scans.clear()
    again = (*CUT, "--cut-threshold", "0.5", "--plan", str(plan))
    text = refused(tmp_path, source_path, caplog, *again)
    assert "another job than the one asked" in text and "environment.ffmpeg" in text
    assert 'settings.cut_threshold: "0.3" -> "0.5"' in text
    said = "A plan, which has made no unit yet, is planned again with other settings; its"
    assert f"{said} environment differs too: --accept-env-change takes it up anyway" in text
    assert "Only its environment differs" not in text
    assert contents(out) == kept and scans == []
    caplog.set_level(logging.INFO, logger="seedvr2x")
    assert upscale(tmp_path, source_path, "out", *again, "--accept-env-change") == 0
    assert len(scans) == 1
    content = json.loads(manifest.read_text())
    data = index.read_bytes()
    assert data != first and content["input"]["index"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert content["environment"]["ffmpeg"] == "n0.0-another"
    [change] = content["environment_changes"]
    assert "ffmpeg" in json.dumps(change)
    assert content["settings"]["cut_threshold"] == "0.5"
    assert shots_of(out) == [(0, 5), (5, 15), (15, 25)]
    said = "planned again in another environment, as accepted, which its manifest records at once"
    assert said in caplog.text
    # The other ffmpeg again, nothing to accept: its record, no first pass.
    scans.clear()
    assert upscale(tmp_path, source_path, "out", *again) == 0 and scans == []
    # The first ffmpeg: a job of another environment, never the one it recorded.
    monkeypatch.setattr(ffmpeg, "check", check)
    otherwise[0] = False
    text = refused(tmp_path, source_path, caplog, *again)
    assert "environment.ffmpeg" in text and "Only its environment differs" in text
    assert scans == []


@pytest.mark.parametrize("how", ["planned", "run", "another GPU"])
def test_plan_taken_up_in_another_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    how: str,
) -> None:
    # A plan taken up as it is in another environment, accepted, by --plan again or by its run,
    # stopped before any unit: its manifest is written at once in that environment, the change
    # recorded; under another ffmpeg, its first pass run again, with the index that pass made,
    # which the manifest names; under another GPU, the pass's record standing, the index as it
    # was. The next start in that environment accepts nothing and runs no first pass.
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    out = tmp_path / "out"
    index, manifest = out / "frame_index.bin", out / "manifest.json"
    first = index.read_bytes()
    if how == "another GPU":
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
    else:
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
    otherwise[0] = True
    scans.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    if how == "run":
        scripted.stop = "encode 0"
        stopped(tmp_path, source_path, "out", *CUT, "--accept-env-change")
        assert scripted.calls == ["encode 0"]
    else:
        options = (*CUT, "--plan", str(plan), "--accept-env-change")
        assert upscale(tmp_path, source_path, "out", *options) == 0
    content = json.loads(manifest.read_text())
    data = index.read_bytes()
    assert content["input"]["index"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert len(content["environment_changes"]) == 1
    assert not any(shot["encoded"] for shot in content["shots"])
    if how == "another GPU":
        assert scans == [] and data == first and content["environment"]["gpu"] == "another"
    else:
        assert len(scans) == 1 and data != first
        assert content["environment"]["ffmpeg"] == "n0.0-another"
    said = "which its manifest records at once, a plan having made no unit in the one before"
    assert f"resumed in another environment, as accepted, {said}" in caplog.text
    scans.clear()
    scripted.stop = None
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert scans == []


def test_plan_of_a_finished_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan on a job finished already: said, and its cut list written all the same, the job's
    # plan, from its record: no first pass, nothing of the job touched, the models not loaded.
    from seedvr2x.runtime import model

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    out = tmp_path / "out"
    kept = contents(out)
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    scripted.calls.clear()
    caplog.clear()
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert f"{out}: finished already, its 4 segments" in caplog.text
    said = "3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3: the cut list in"
    assert f"{said} {plan}" in caplog.text
    assert cut_lines(plan) == PLANNED
    # The playlists of its 3 cuts and 2 possible cuts beside it, as any detected plan's
    # (cuts.write_playlists).
    assert sorted(path.name for path in tmp_path.glob("plan*")) == [
        "plan.txt",
        "plan.txt.cuts.m3u8",
        "plan.txt.possible.m3u8",
    ]
    assert passes == [] and scripted.calls == [] and contents(out) == kept


@pytest.mark.parametrize("index", ["missing", "damaged", "another ffmpeg's"])
def test_plan_of_a_finished_job_without_its_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    index: str,
) -> None:
    # --plan on a finished job whose frame index is missing, damaged, or made by another ffmpeg
    # than the run's, accepted: the cut list is written from the index, which a finished job
    # doesn't make again: said, FILE not written, nor the playlists of its cuts, exit 1,
    # no first pass, nothing of the job touched (PROVISIONAL, cli._finished_already).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    out = tmp_path / "out"
    options: tuple[str, ...] = CUT
    if index == "another ffmpeg's":
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        options = (*CUT, "--accept-env-change")
        why = "another ffmpeg or other conversions than this run's made"
    else:
        spoiled(out / "frame_index.bin", index)
        why = "isn't there as its manifest records it"
    kept = contents(out)
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    StandInDetector.built.clear()
    scripted.calls.clear()
    text = refused(tmp_path, source_path, caplog, *options, "--plan", str(plan))
    assert f"{out}: finished already, its 4 segments" in text
    assert (
        f"--plan {plan}: not written: {out} is finished already, and a job's cut list is written"
        f" from its frame index, which {why} and which a finished job doesn't make again: plan the"
        " source in another output directory (-o) for its cut list"
    ) in text
    assert list(tmp_path.glob("plan*")) == [] and passes == [] and StandInDetector.built == []
    assert scripted.calls == [] and contents(out) == kept


def test_replan_compares_what_its_first_pass_finds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan planned again whose first pass, run again, finds another source than the one
    # recorded, here another number of frames: another job, refused, as a resume's is; only its
    # cuts and the layout they give are a plan's to change.
    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    manifest = tmp_path / "out" / "manifest.json"
    content = json.loads(manifest.read_text())
    assert content["input"]["frames"] == 25
    content["input"]["frames"] = 24
    manifest.write_text(json.dumps(content))
    passes = first_passes(monkeypatch)
    text = refused(tmp_path, source_path, caplog, *CUT, "--cut-threshold", "0.5")
    assert len(passes) == 1  # the index's frames aren't the record's: the pass run again
    assert "another job than the one asked" in text and "input.frames: 24 -> 25" in text
    assert json.loads(manifest.read_text()) == content and scripted.calls == []


@pytest.mark.parametrize("threshold", ["0.1", "0.05"])
def test_no_possible_cuts_at_their_floor_or_under(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
    threshold: str,
) -> None:
    # At a threshold of 0.1 or under, no run from 0.1 up peaks under it: the run's log and the
    # plan's counts say the cuts alone, the plan that there are no possible cuts and why, and its
    # list holds none.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    options = (*CUT, "--cut-threshold", threshold, "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    said = [record.getMessage() for record in caplog.records]
    assert f"6 shots: 5 cuts detected at {threshold}" in said  # the whole message
    counts = f"5 cuts at {threshold} (6 shots), no possible cuts at a threshold of 0.1 or under"
    assert f"{counts}: the cut list in {plan}" in said
    assert f"# Found: {counts}" in plan.read_text().splitlines()
    lines = cut_lines(plan)
    assert [line.split()[0] for line in lines] == ["5", "10", "15", "19", "22"]


def test_plan_not_written_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The cut list can't be written after all, the disk full once the first pass has run: said,
    # and the command fails, where its only output is that list.
    from seedvr2x.media import files

    def full(path: Path, data: bytes) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    monkeypatch.setattr(files, "write_whole", full)
    text = refused(tmp_path, source_path, caplog, "--plan", str(plan), output="one.mkv")
    assert f"--plan {plan}: not written: [Errno 28] No space left on device" in text
    assert not plan.exists()


def test_plan_playlist_not_written_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The cut list written, the disk full for its playlists: said of each, and the command
    # fails, the list there.
    from seedvr2x.media import files

    write_whole = files.write_whole

    def full(path: Path, data: bytes) -> None:
        if path.suffix == ".m3u8":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        write_whole(path, data)

    source_path = job(tmp_path, monkeypatch)
    plan = tmp_path / "plan.txt"
    monkeypatch.setattr(files, "write_whole", full)
    text = refused(tmp_path, source_path, caplog, "--plan", str(plan), output="one.mkv")
    for playlist in (tmp_path / "plan.txt.cuts.m3u8", tmp_path / "plan.txt.possible.m3u8"):
        said = f"its playlist, {playlist}, not written: [Errno 28] No space left on device"
        assert f"--plan {plan}: {said}" in text and not playlist.exists()
    assert cut_lines(plan) == PLANNED


def synthetic_clip(path: Path) -> Path:
    """tests/test_detector.py's synthetic clip, an FFV1 file: four lavfi shots at 25 fps, 280
    frames at 320x180, its cuts at CUTS."""
    from test_detector import SHOTS

    inputs = [arg for source, _ in SHOTS for arg in ("-f", "lavfi", "-i", source)]
    trims = "".join(
        f"[{k}]trim=end_frame={count},setpts=PTS-STARTPTS,format=yuv420p[s{k}];"
        for k, (_, count) in enumerate(SHOTS)
    )
    joined = "".join(f"[s{k}]" for k in range(len(SHOTS)))
    graph = f"{trims}{joined}concat=n={len(SHOTS)}:v=1:a=0[v]"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]"),
            *("-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    return path


def test_real_detection_playlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # TransNetV2 itself, as in test_real_detection_cuts, on the synthetic clip, 280 frames at
    # 25 fps, its last at 11.16 s, planned at thresholds its cuts' peaks stay under (0.963, 0.993
    # and 0.998 here, read from the list), which makes possible cuts of them. At 0.97, the first
    # alone: the possible cuts' playlist's one entry, from a second before its time, 2.92 s, to
    # 3 seconds after, and the other two the detected cuts' playlist's. Planned again at 1,
    # with no second pass: the three possible, in frame order, the last one's stop the last
    # frame's time, and no detected cut, that playlist left. The playlists printed (-rP).
    from test_detector import CUTS

    from seedvr2x.runtime import detector

    models = os.environ.get("SEEDVR2X_MODEL_DIR")
    if not models or not (Path(models) / pull.DETECTOR).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {pull.DETECTOR}")
    (tmp_path / pull.DETECTOR).symlink_to(Path(models) / pull.DETECTOR)
    monkeypatch.setattr(pull, "PINNED", {**pull.PINNED, pull.DETECTOR: DETECTOR_PIN})

    def on_the_cpu(path: Path, device: torch.device, **options: Any) -> Detector:
        return REAL_DETECTOR(path, torch.device("cpu"), **options)

    monkeypatch.setattr(detector, "Detector", on_the_cpu)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    clip = synthetic_clip(tmp_path / "clip.mkv")
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    detected, playlist = tmp_path / "plan.txt.cuts.m3u8", tmp_path / "plan.txt.possible.m3u8"
    times = [("1.920", "5.920"), ("5.000", "9.000"), ("7.480", "11.160")]

    def entries(which: slice, possible: bool = True) -> list[str]:
        # Each possible cut's line in the list, `# <frame> <probability> <time>`, as its entry;
        # or each cut's, `<frame> <probability> <time>`.
        listed = [line for line in cut_lines(plan) if line.startswith("# ") == possible]
        lines = [line.removeprefix("# ").split() for line in listed]
        assert [int(line[0]) for line in lines] == CUTS[which]
        title = "possible cut" if possible else "cut"
        return [
            written
            for (frame, peak, time), (start, stop) in zip(lines, times[which], strict=True)
            for written in (
                f"#EXTINF:-1,{title} {frame} at {time} ({peak})",
                f"#EXTVLCOPT:start-time={start}",
                f"#EXTVLCOPT:stop-time={stop}",
                "clip.mkv",
            )
        ]

    assert upscale(tmp_path, clip, "out", "--cut-threshold", "0.97", "--plan", str(plan)) == 0
    assert "2 cuts at 0.97 (3 shots), 1 possible cut from 0.1 up to 0.97" in caplog.text
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *entries(slice(1))]
    earlier = [*DETECTED_HEADER, *entries(slice(1, 3), possible=False)]
    assert detected.read_text().splitlines() == earlier
    assert f"{detected}: a playlist of the 2 detected cuts for VLC" in caplog.text
    assert f"{playlist}: a playlist of the 1 possible cut for VLC" in caplog.text
    assert upscale(tmp_path, clip, "out", "--cut-threshold", "1", "--plan", str(plan)) == 0
    assert len(passes) == 1
    assert "0 cuts at 1.0 (1 shot), 3 possible cuts from 0.1 up to 1.0" in caplog.text
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *entries(slice(3))]
    assert detected.read_text().splitlines() == earlier
    assert f"{detected}: left as it is: this plan has no detected cut" in caplog.text
    assert [line.split()[3] for line in cut_lines(plan)] == [
        "0:00:02.920",
        "0:00:06.000",
        "0:00:08.480",
    ]
    print("\n".join(record.getMessage() for record in caplog.records))
    print(detected.read_text())
    print(playlist.read_text())


def test_real_detection_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # TransNetV2 itself in the CLI's first pass, its weights from SEEDVR2X_MODEL_DIR (skipped
    # without) checked by their pin, run on the CPU in place of the job's GPU, on the synthetic
    # clip of tests/test_detector.py: its cuts, exactly, at 0.3, in the plan's manifest, the log
    # and the cut list; no possible cut, every other frame scoring under 0.05. The log's lines and
    # the cut list printed (-rP).
    from test_detector import CUTS

    from seedvr2x.runtime import detector
    from seedvr2x.runtime.job import read_cuts

    models = os.environ.get("SEEDVR2X_MODEL_DIR")
    if not models or not (Path(models) / pull.DETECTOR).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {pull.DETECTOR}")
    (tmp_path / pull.DETECTOR).symlink_to(Path(models) / pull.DETECTOR)
    monkeypatch.setattr(pull, "PINNED", {**pull.PINNED, pull.DETECTOR: DETECTOR_PIN})

    def on_the_cpu(path: Path, device: torch.device, **options: Any) -> Detector:
        return REAL_DETECTOR(path, torch.device("cpu"), **options)

    monkeypatch.setattr(detector, "Detector", on_the_cpu)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    clip = synthetic_clip(tmp_path / "clip.mkv")
    plan = tmp_path / "plan.txt"
    assert upscale(tmp_path, clip, "out", "--plan", str(plan)) == 0
    bounds = [0, *CUTS, 280]
    assert shots_of(tmp_path / "out") == list(pairwise(bounds))
    said = "4 shots: 3 cuts detected at 0.3, and 0 possible cuts from 0.1 up to 0.3"
    assert said in caplog.text
    assert "3 cuts at 0.3 (4 shots), 0 possible cuts from 0.1 up to 0.3" in caplog.text
    assert read_cuts(plan) == CUTS
    lines = cut_lines(plan)
    assert [line.split()[2] for line in lines] == ["0:00:02.920", "0:00:06.000", "0:00:08.480"]
    assert all(float(line.split()[1]) > 0.5 for line in lines)
    print("\n".join(record.getMessage() for record in caplog.records))
    print(plan.read_text())


# A run of the command line in a fresh interpreter whose run warns, through Python's warnings,
# twice from one place and once from another (test_python_s_warnings_are_lines_of_seedvr2x_s).
WARNED = """
import sys
import warnings

from seedvr2x import cli


def run(args):
    for _ in range(2):
        warnings.warn("a library's warning", UserWarning)
    warnings.warn("another", FutureWarning)
    return 0


cli._run = run
sys.exit(cli.main(sys.argv[1:]))
"""


def test_python_s_warnings_are_lines_of_seedvr2x_s(tmp_path: Path) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-21): Python's own warnings go
    # through the same lines, "23:05:12 WARNING UserWarning: ...", each on one line, where Python
    # prints two, the second the source's; once a place, as Python's filters have it. Under
    # --verbose, in its format, under Python's name for them, py.warnings.
    for verbose, start in (
        (False, r"\d\d:\d\d:\d\d WARNING "),
        (True, r"\S+ \S+ WARNING py\.warnings: "),
    ):
        options = ["-v"] if verbose else []
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                WARNED,
                str(tmp_path / "in.mkv"),
                "-o",
                str(tmp_path / "out"),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stderr.splitlines()
        assert [line for line in lines if "warning" in line.lower() or "another" in line] == lines
        assert len(lines) == 2, result.stderr
        assert re.fullmatch(f"{start}UserWarning: a library's warning \\(<string>:10\\)", lines[0])
        assert re.fullmatch(f"{start}FutureWarning: another \\(<string>:11\\)", lines[1])


def test_python_s_warnings_are_given_back_on_the_way_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Set in the CLI's entry point and restored on the way out (DESIGN.md, Architecture, What it
    # prints): an in-process run's warnings are seedvr2x's lines during it, and pytest's own
    # capture has them again after it, whether the run returns or raises; verify likewise.
    shown = warnings.showwarning

    def run(args: object) -> int:
        warnings.warn("during the run", UserWarning, stacklevel=1)
        warnings.warn("a warning\n  of two lines", UserWarning, stacklevel=1)
        if getattr(args, "seed", None) == 7:
            raise RuntimeError("a bug")
        return 0

    monkeypatch.setattr(cli, "_run", run)
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        assert cli.main(["in.mkv", "-o", "out"]) == 0
        with pytest.raises(RuntimeError, match="a bug"):
            cli.main(["in.mkv", "-o", "out", "--seed", "7"])
    said = [record for record in caplog.records if record.name == "py.warnings"]
    assert [record.levelname for record in said] == ["WARNING"] * 4
    assert said[1].getMessage().startswith("UserWarning: a warning of two lines (")
    assert re.fullmatch(
        r"UserWarning: during the run \(.*test_cli_run\.py:\d+\)", said[0].getMessage()
    )
    assert warnings.showwarning is shown
    with pytest.warns(UserWarning, match="after the run"):
        warnings.warn("after the run", UserWarning, stacklevel=1)

    def verified(output: Path) -> int:
        warnings.warn("during verify", UserWarning, stacklevel=1)
        return 0

    monkeypatch.setattr(cli, "_verify", verified)
    caplog.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        assert cli.main(["verify", "out"]) == 0
    assert [record.getMessage().partition(" (")[0] for record in caplog.records] == [
        "UserWarning: during verify"
    ]
    assert warnings.showwarning is shown


def test_a_line_starts_with_the_time_of_day_but_under_verbose() -> None:
    # What seedvr2x prints (cli._Lines): the time of day, then the message; a warning's WARNING
    # and an error's ERROR before it; no date, no milliseconds, no logger's name. Under
    # --verbose the format is cli.VERBOSE_FORMAT, every level named, with the logger's.
    def record(level: int) -> logging.LogRecord:
        return logging.LogRecord("seedvr2x", level, "cli.py", 1, "the message", None, None)

    lines = cli._Lines()  # pyright: ignore[reportPrivateUsage]
    time = r"\d\d:\d\d:\d\d"
    assert re.fullmatch(f"{time} the message", lines.format(record(logging.INFO)))
    assert re.fullmatch(f"{time} WARNING the message", lines.format(record(logging.WARNING)))
    assert re.fullmatch(f"{time} ERROR the message", lines.format(record(logging.ERROR)))
    assert re.fullmatch(f"{time} CRITICAL the message", lines.format(record(logging.CRITICAL)))
    verbose = logging.Formatter(cli.VERBOSE_FORMAT)
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO seedvr2x: the message",
        verbose.format(record(logging.INFO)),
    )


def test_a_line_says_when_the_date_or_the_clocks_change_but_not_under_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # time of day alone is ambiguous the night the clocks go back (02:00 to 02:59 come twice), so
    # the date changing and the local zone's offset from UTC changing are each said in a line of
    # their own, before the line logged after the change (cli._Lines: checked at each line
    # logged). A fake clock: the records' own times, in Europe/Paris, from 2026-10-24 23:59 to
    # 2026-10-25 02:30 the second time through, the night the clocks went back at 03:00.
    from datetime import UTC
    from zoneinfo import ZoneInfo

    monkeypatch.setenv("TZ", "Europe/Paris")
    time.tzset()
    try:
        paris = ZoneInfo("Europe/Paris")

        def line(lines: logging.Formatter, when: datetime, text: str) -> str:
            made = logging.LogRecord("seedvr2x", logging.INFO, "cli.py", 1, text, None, None)
            made.created = when.timestamp()
            return lines.format(made)

        lines = cli._Lines()  # pyright: ignore[reportPrivateUsage]
        first = datetime(2026, 10, 24, 23, 59, 58, tzinfo=paris)
        # The first line says nothing: nothing to have changed from.
        assert line(lines, first, "unit 1") == "23:59:58 unit 1"
        assert line(lines, first.replace(second=59), "unit 2") == "23:59:59 unit 2"
        # Past midnight: the date, before the line that notices it, with that line's time.
        after = datetime(2026, 10, 25, 0, 0, 3, tzinfo=paris)
        assert (
            line(lines, after, "unit 3")
            == "00:00:03 the date is now Sun 25 Oct 2026\n00:00:03 unit 3"
        )
        assert line(lines, after.replace(second=4), "unit 4") == "00:00:04 unit 4"
        # 02:30 of the first pass (UTC+2), then of the second (UTC+1, fold=1): the offset said
        # once, the date not again, and the line after it not at all.
        before = datetime(2026, 10, 25, 2, 30, tzinfo=paris, fold=0)
        again = datetime(2026, 10, 25, 2, 30, tzinfo=paris, fold=1)
        assert before.utcoffset() != again.utcoffset()
        assert line(lines, before, "unit 5") == "02:30:00 unit 5"
        assert line(lines, again, "unit 6") == (
            "02:30:00 the clocks changed: the local time is now UTC+01:00 (it was UTC+02:00), the"
            " times of day below in the new one\n02:30:00 unit 6"
        )
        assert line(lines, again.replace(minute=31), "unit 7") == "02:31:00 unit 7"
        # A warning keeps WARNING on its own line, the change's line before it plain.
        night = datetime(2026, 10, 26, 0, 1, tzinfo=paris)
        made = logging.LogRecord("seedvr2x", logging.WARNING, "cli.py", 1, "careful", None, None)
        made.created = night.timestamp()
        assert lines.format(made) == (
            "00:01:00 the date is now Mon 26 Oct 2026\n00:01:00 WARNING careful"
        )
        # The spring: the clocks forward, UTC+1 to UTC+2.
        spring = datetime(2026, 3, 29, 1, 59, tzinfo=paris)
        lines = cli._Lines()  # pyright: ignore[reportPrivateUsage]
        assert line(lines, spring, "a") == "01:59:00 a"
        past = datetime(2026, 3, 29, 3, 1, tzinfo=paris)
        assert past.astimezone(UTC) - spring.astimezone(UTC) == timedelta(minutes=2)
        assert line(lines, past, "b") == (
            "03:01:00 the clocks changed: the local time is now UTC+02:00 (it was UTC+01:00), the"
            " times of day below in the new one\n03:01:00 b"
        )
        # --verbose keeps the old format whole: the date is in every line, nothing added.
        verbose = logging.Formatter(cli.VERBOSE_FORMAT)
        assert line(verbose, after, "x").count("\n") == 0
    finally:
        monkeypatch.undo()
        time.tzset()
