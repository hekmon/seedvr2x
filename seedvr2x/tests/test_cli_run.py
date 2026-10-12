"""The command line's run on the CPU, the model replaced by a stand-in: the input read, the shots
and their seeds, the output written, its segments and its manifest, in every output format.

The stand-in's steps carry each shot's first frame (its seed less the job's) and frame count from
its encode through its windows to its decode, which writes, for each frame, the frame's index in
the job divided by 1000: so each output frame says where it comes from. The model's own output
is the GPU tests' (test_regression.py, test_shots.py)."""

import errno
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
import weakref
import zlib
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
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
from seedvr2x.runtime import planner, pull, weights
from seedvr2x.runtime.detector import BATCH, Detector
from seedvr2x.runtime.job import Shot, output_size, padded_size
from seedvr2x.runtime.plan import Plan, Tiles
from seedvr2x.runtime.planner import GIB, Peaks, Pins
from seedvr2x.runtime.pull import Pin
from seedvr2x.runtime.shot import decode_shot, padded_length


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
    tiles: Tiles | None = None,
    scratch: Path | None = None,
    allowance: int | None = None,
) -> torch.Tensor:
    """A latent (T', 1, 1, 16) in channel-major memory, as the VAE's, saying the shot's first frame
    (channel 0), its frame count (1), the target's height and width (2, 3), and whether the frames
    were padded as numz pads them (6: cli.NUMZ_PADDING). Untiled, with no scratch and nothing
    allowed one: the plan's tiles are the tiled passes' own tests'."""
    assert tiles is None and scratch is None and not allowance
    assert read(count).shape[0] == count
    latent = torch.zeros(16, (padded_length(count) - 1) // 4 + 1, 1, 1)
    latent[0], latent[1] = seed - SEED, count
    latent[2], latent[3] = target
    latent[6] = numz_padding
    return latent.permute(1, 2, 3, 0)


def stand_in_windows(
    models: object, latent: torch.Tensor, layout: list[tuple[int, int]], seed: int, start: int = 0
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
        tiles: Tiles | None = None,
        scratch: Path | None = None,
        allowance: int | None = None,
    ) -> torch.Tensor:
        frames = read(count)
        self.read[seed - SEED] = hashlib.md5(frames.tobytes()).hexdigest()
        self.numz_padding.append(numz_padding)
        self._call(f"encode {seed - SEED}")
        latent = stand_in_encode(
            *(models, lambda n: frames[:n], count, target, seed),
            numz_padding=numz_padding,
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


def plan_record(plan: Plan, peaks: Peaks) -> dict[str, Any]:
    """A plan as a manifest records it: its shape, and under memory where the weights are with
    what it is predicted to take (cli._run)."""
    record = plan.record()
    record["memory"].update(peaks.record())
    return record


def free_memory(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    """Have the stand-in's GPU report `free` bytes free from now on."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, TOTAL))


def plan_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [line for line in caplog.messages if line.startswith("plan: ")]


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
    assert content["plan"] == plan_record(*made)
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
        "attention": "flash_attn_2",
        "budget": FREE - planner.MARGIN,
        "encode_peak": planner.encode_peak(PADDED),
        # A window of the 6-latent shot, over the shot's draw (planner.dit_draw).
        "dit_peak": planner.dit_peak(PADDED, 4),
        "decode_peak": planner.decode_peak(PADDED, TARGET),
        "bound": "encode",
        # The process's own, and the longest shot at the check of its latent: 6 latents of 3.5
        # times their 8,192 bytes (planner.host_moments).
        "host": planner.HOST_RESIDENT + 6 * 28_672,
        "host_shot": 6 * 28_672,
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


def test_until_not_built_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --until is read, and refused until stopping by the clock is built, nothing made: no unit,
    # no manifest.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--until", "7:30") == 1
    assert "--until 07:30: stopping by the clock isn't built yet" in caplog.text
    assert steps.calls == []
    assert not (tmp_path / "out" / "manifest.json").exists()


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
    assert content["plan"] == plan_record(*made)
    assert content["plan"] | {"memory": None} == Plan(5).record() | {"memory": None}


def spilling_driver(monkeypatch: pytest.MonkeyPatch, card: int | None = None) -> list[float]:
    """A driver behind WSL2's device that gives memory the card doesn't have (runtime/memory.py),
    the probe's answer a stand-in's; returns the limits torch is then given, as fractions.
    `card`: what the whole card has free, a stand-in for NVML's reading of it (memory.card_free);
    None: a machine without NVML, as conftest.py leaves every CPU test, warned of in the run."""
    from seedvr2x.runtime import memory

    monkeypatch.setattr(memory, "paravirtual", lambda: True)
    if card is not None:
        monkeypatch.setattr(memory, "card_free", lambda index: card)
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
    from seedvr2x.runtime import memory, model

    caplog.set_level(logging.INFO)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    card = FREE - 2 * GIB
    limits = spilling_driver(monkeypatch, card)
    loads: list[tuple[Any, int]] = []
    monkeypatch.setattr(
        model, "load_models", lambda *a: loads.append((a[4], len(limits))) or STAND_IN_MODELS
    )
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    planned_in = card - memory.HELD_BACK - memory.RESERVE
    made = planner.plan(planned_in, TARGET, JOB_SHOTS, Pins(window=5), HOST)
    assert loads == [(made[0], 1)] and limits == [(card - memory.HELD_BACK) / TOTAL]
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
    assert "the GPU's driver gives memory the card doesn't have" in caplog.text
    # The plan's first line names each figure, where "92.50 GiB free" alone would be taken for
    # what the plan counts on, and "87.20 GiB free" for what the driver reports; and it is the
    # one place a run's log says them: memory's own line is for its debug level.
    assert (
        ", 92.50 GiB free on the whole card by NVML (its driver reports 94.50 GiB free to this"
        " process); the run is held to the card's own memory, 5.30 GiB under that, its driver"
        " giving more than the card has: the plan counts on 87.20 GiB, a budget of 86.60 GiB"
    ) in caplog.text
    assert caplog.text.count("the plan counts on 87.20 GiB") == 1
    assert (
        "torch held to 89.50 GiB on the GPU: the 92.50 GiB free less 3.0, and what it holds;"
        " free on the whole card by NVML (its driver reports 94.50 GiB free to this process)"
    ) in caplog.text


def test_the_user_s_allocator_stands_on_a_driver_that_spills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    limits = spilling_driver(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync" and len(limits) == 1
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
    assert upscale(tmp_path, source_path, "out", *JOB, "--until", "7:30") == 1
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


def test_a_plan_on_a_driver_that_spills_holds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # --plan counts in the memory a run would: the same plan as the run's; torch, which loads
    # no model, is given no limit.
    from seedvr2x.runtime import memory

    caplog.set_level(logging.INFO, logger="seedvr2x")
    limits = spilling_driver(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", "plan.txt") == 0
    assert not limits and steps.calls == []
    planned_in = FREE - memory.HELD_BACK - memory.RESERVE
    reading = memory.Reading(FREE, None, planned_in, True)
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["plan"] == plan_record(
        *planner.plan(reading, TARGET, JOB_SHOTS, Pins(window=5), HOST)
    )
    assert content["plan"]["memory"]["free"] == planned_in
    # A machine without NVML (conftest.py): warned of once, and the plan's first line says
    # which figure it had.
    assert caplog.text.count("the whole card's free memory can't be read through NVML") == 1
    assert (
        ", 94.50 GiB free as its driver reports it to this process (the whole card's, NVML's,"
        " wasn't read); the run is held to the card's own memory, 5.30 GiB under that"
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
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "backend:native"
    assert limits == [(FREE - memory.HELD_BACK) / TOTAL]
    planned_in = FREE - memory.HELD_BACK - memory.RESERVE
    assert json.loads(manifest.read_text())["plan"]["memory"]["free"] == planned_in
    made = [indexes(tmp_path / "out" / f"seg_{number:06d}.mkv") for number in range(2)]
    assert made[0] + made[1] == list(range(25))


def test_gpu_memory_refused_at_the_load_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The weights refused GPU memory as they load, the limit set before them where seedvr2x
    # holds torch: one line, no traceback, nothing run.
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
        " run"
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
    # the same one line and status as at the load, no traceback, nothing run.
    from seedvr2x.runtime import memory

    def refused(*arguments: object, **options: object) -> None:
        raise torch.OutOfMemoryError(TORCH_REFUSED)

    target = {"the first pass": (cli, "_first_pass"), "the reading": (cli, "_free_memory")}
    monkeypatch.setattr(*target.get(place, (memory, "hold")), refused)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [
        "the GPU's memory was refused: CUDA out of memory. Tried to allocate 456.00 MiB, torch"
        " holding 19.27 GiB allocated and 1.43 GiB more reserved, 20.98 GiB allowed, before the"
        " models loaded: nothing was run"
    ]
    assert all(record.exc_info is None for record in errors) and steps.calls == []


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
    # load refuses, naming the RAM they take, logged without a traceback, before any unit.
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
    assert [record.getMessage() for record in errors] == [said]
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
    assert plan_lines(caplog) == said
    assert said == [
        "plan: a stand-in, 94.50 GiB free, a budget of 93.90 GiB",
        "plan: frames padded to 128x128, 64 tokens a latent, attention on FlashAttention 2",
        "plan: 3 shots, the longest of 21 frames (6 latents)",
        "plan: windows of 5 latents at most, the longest run 4 latents: 2 of 3 shots in one"
        " window, 1 join",
        "plan: no block of the DiT swapped, the VAE on the GPU in the DiT's phase; in the encode,"
        " the DiT's 36 blocks on the GPU; in the decode, the DiT's 36 blocks on the GPU",
        "plan: no system memory pinned, nothing leaving the GPU; 0.00 GiB for the longest shot at"
        " the most, as its latent is checked after its encode; 2.40 GiB of system memory in all",
        "plan: encode untiled, decode untiled, each held 1.50 GiB under the budget",
        "plan: predicted peaks: encode 18.11 GiB, DiT 15.91 GiB, decode 17.73 GiB; the encode"
        " bounds the plan, its room counted, 74.29 GiB under the budget",
        "plan: pinned: --window 5",
    ]
    caplog.clear()
    plan = tmp_path / "plan.txt"
    with monkeypatch.context() as patch:
        from seedvr2x.runtime import model

        patch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(plan)) == 0
        assert plan_lines(caplog) == said
        counts = next(line for line in caplog.messages if line.endswith(f"the cut list in {plan}"))
        assert caplog.messages.index(said[-1]) < caplog.messages.index(counts)
        # For a .mkv path too, which --plan doesn't write: its cut list, and the job's plan in
        # its work directory.
        caplog.clear()
        steps.calls.clear()
        before = sorted(path.name for path in tmp_path.iterdir())
        listed = tmp_path / "other.txt"
        assert upscale(tmp_path, source_path, "other.mkv", *JOB[:4], "--plan", str(listed)) == 0
        assert plan_lines(caplog) == said and steps.calls == []
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
    # refusal naming what fits, before the models load: nothing made, no unit, no manifest.
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
        assert not (tmp_path / "out" / "manifest.json").exists()
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
    assert not (tmp_path / "out" / "manifest.json").exists()
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
    # before the models load, as a pin is: exit 1, nothing made, nothing written.
    source_path = job(tmp_path, monkeypatch)
    there = sorted(path.name for path in tmp_path.iterdir())
    # The stand-in's frames are too small for a tile: their floor is the untiled encode's, with
    # the room a VAE pass is held to (planner.VAE_ROOM).
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    with monkeypatch.context() as patch:
        free_memory(patch, encode + planner.MARGIN - 1)
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert (
        "an output of 128x96 takes 4.87 GiB of free VRAM (its encode untiled, too small for a"
        " tile, with the DiT off the GPU: a peak of 2.77 GiB, 1.50 of room a VAE pass is held to,"
        " and 0.60 of margin), and the driver reports 4.86 GiB free: the largest output that fits"
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
    # But the stand-in's model files, which upscale puts there.
    files = sorted(path.name for path in tmp_path.iterdir())
    assert files == sorted([*there, pull.DETECTOR, "w.safetensors"])
    # With the memory back, both run.
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0


def test_one_file_plans_as_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps
) -> None:
    # A .mkv path's job keeps its units on disk, in its work directory, as an output
    # directory's does (cli._output; units.Units' in-memory ones no run of the command line's
    # keeps any more): nothing of a shot waits on the GPU either way, and the two plan alike,
    # --plan's as its run's. JOB's last shot, 6 latents in two windows of 4: the DiT's phase
    # takes a window's peak, the shot's draw being under it. The plan is told which output it
    # is, for what the run holds of a shot in system memory (planner.host_moments): never one
    # file's, whose windows would wait there, since a .mkv path's job writes them to its work
    # directory.
    asked: list[set[str]] = []
    backends: list[str] = []
    one_file: list[bool] = []
    peaks: list[int] = []
    plan = planner.plan

    def planned(*arguments: Any, **options: Any) -> Any:
        asked.append(set(options))
        backends.append(options["attention"])
        one_file.append(options["one_file"])
        made = plan(*arguments, **options)
        peaks.append(made[1].dit)
        return made

    monkeypatch.setattr(planner, "plan", planned)
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    listed = str(tmp_path / "plan.txt")
    assert upscale(tmp_path, source_path, "other.mkv", *JOB[:4], "--plan", listed) == 0
    assert asked == [{"split", "numz_padding", "recorded", "attention", "one_file"}] * 3
    assert one_file == [False, False, False]
    # Each is planned for the attention backend its run takes (cli._attention, the stand-in's
    # here), a one-file output too, which records no environment to read it from.
    assert backends == ["flash_attn_2"] * 3
    window = planner.dit_peak(PADDED, 4)
    assert peaks == [window] * 3 and window > planner.dit_draw(PADDED, 6)


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
        models: object, latent: torch.Tensor, layout: list[tuple[int, int]], seed: int, start: int
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
    assert started["plan"] == plan_record(*planner.plan(less, TARGET, JOB_SHOTS, pins, HOST))
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
    assert "plan: windows of 5 latents at most, the longest run 4 latents" in caplog.text
    assert steps.calls == ["encode 4", "window 4:0", "window 4:1"]
    content = json.loads(manifest.read_text())
    again = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST, recorded=Plan(5))
    assert again[0] == Plan(5) and content["plan"] == plan_record(*again)
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
    assert again[0] == Plan(5, swapped=36) and content["plan"] == plan_record(*again)
    shape = {key: value for key, value in content["plan"].items() if key != "memory"}
    assert shape == {key: value for key, value in started["plan"].items() if key != "memory"}
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
    assert started["plan"] == plan_record(*planner.plan(FREE, TARGET, JOB_SHOTS, Pins(), HOST))
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
    assert "plan: 20 of the DiT's 36 blocks swapped" in caplog.text
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    content = json.loads(manifest.read_text())
    again = planner.plan(less, TARGET, JOB_SHOTS, Pins(), HOST, recorded=Plan(6))
    assert (again[0].window, again[0].swapped, again[0].vae_in_dit) == (6, 20, False)
    assert content["plan"] == plan_record(*again)
    shape = {key: value for key, value in content["plan"].items() if key != "memory"}
    assert shape == {key: value for key, value in started["plan"].items() if key != "memory"}
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
    assert started["plan"] == plan_record(*planner.plan(exact, TARGET, JOB_SHOTS, Pins(), HOST))
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
    assert content["plan"] == plan_record(*again)
    assert content["plan"]["memory"]["attention"] == "sdpa"
    shape = {key: value for key, value in content["plan"].items() if key != "memory"}
    assert shape == {key: value for key, value in started["plan"].items() if key != "memory"}
    assert "tokens a latent, attention on PyTorch's SDPA" in caplog.text
    assert indexes(tmp_path / "out" / "seg_000000.mkv") == list(range(25))


def test_recorded_plan_no_longer_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A resume whose recorded shape no longer fits the memory free stops, naming the phase and
    # the shortfall (DESIGN.md, Memory planner), nothing changed on disk; the job resumes once
    # the memory is back. Here the untiled encode, a quarter of a GiB short with the DiT away
    # of its peak and the room a VAE pass is held to (planner.VAE_ROOM).
    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 4"
    stopped(tmp_path, source_path, "out", *JOB)
    kept = contents(tmp_path / "out")
    steps.stop = None
    steps.calls.clear()
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    free_memory(monkeypatch, encode + planner.MARGIN - GIB // 4)
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert (
        "the recorded plan no longer fits the memory free now: the encode, untiled, with the"
        " DiT off the GPU, takes 4.87 GiB of free VRAM, 1.50 of them the room a VAE pass is held"
        " to, and the driver reports 4.61 GiB free, 0.26 GiB short"
    ) in caplog.text
    assert contents(tmp_path / "out") == kept and steps.calls == []
    free_memory(monkeypatch, FREE)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == UNINTERRUPTED[6:]


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
    # plan is, whatever it recorded, and said in one line: no other job, and no KeyError.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    manifest = tmp_path / "out" / "manifest.json"
    planned = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({k: v for k, v in planned.items() if k != "plan"}))
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    [said] = [line for line in caplog.messages if "re-planned" in line]
    assert said.startswith(
        f"{tmp_path / 'out'}: re-planned, its job having made no unit yet: plan: "
    )
    assert said.endswith("; 0 leftovers discarded")
    content = json.loads(manifest.read_text())
    assert content["plan"] == planned["plan"]
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
    assert "encode in tiles of 640 px, 1 x 2, decode in tiles of 512 px, 2 x 2," in lines[6]
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
        assert recorded == plan_record(*made)
        assert (recorded["memory"]["scratch"], recorded["memory"]["scratch_allowance"]) == (
            kept,
            kept,
        )
        assert recorded | {"memory": None} == TILED_PLAN | {"memory": None}
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(output)) == (
        ["out"] if output == "out" else ["one.mkv", "one.mkv.crc32"]
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
    assert warned() == [] and len(plan_lines(caplog)) == 10
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
        one.unlink()
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
    # file stopped before keeps the old file and its checksums.
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 0
    master, sums = tmp_path / "one.mkv", tmp_path / "one.mkv.crc32"
    before = (master.read_bytes(), sums.read_bytes())
    steps.stop = "decode 4"
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
    # refuses it, as the first one would have, logged without a traceback.
    from seedvr2x.runtime import model

    said = "w.safetensors: SeedVR2's 3B DiT in fp16"

    def replaced(*args: object) -> None:
        raise weights.ModelError(said)

    monkeypatch.setattr(model, "load_models", replaced)
    assert upscale(tmp_path, job(tmp_path, monkeypatch), "out", *JOB) == 1
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert [record.getMessage() for record in errors] == [said]
    assert all(record.exc_info is None for record in errors)
    assert not (tmp_path / "out" / "manifest.json").exists()


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
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    understating(monkeypatch)
    directory = tmp_path / ("out" if output == "out" else "one.mkv.work")
    left = "decoding with ffmpeg: frames left after 24"
    kept = "stopped: 1 of 2 segments finished, and the units of the next ones kept"
    made = [*("encode 0", "window 0:0", "encode 3", "window 3:0", "decode 0", "decode 3")]
    for calls in (made, []):
        steps.calls.clear()
        caplog.clear()
        options = (*JOB, "--color-correction", correction)
        assert upscale(tmp_path, source_path, output, *options) == 1
        assert f"{left}; {kept}" in caplog.text
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
    assert written == recorded | {"input": written["input"]}
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


@pytest.mark.parametrize("change", ["GPU", "ffmpeg"])
def test_finished_job_in_another_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    change: str,
) -> None:
    # A finished job run again in another environment: refused as a resume is, the flag named;
    # accepted, said finished, with no first pass, under another ffmpeg too, whose pass a resume
    # runs again, and nothing written: its manifest keeps the environment that made it, no change
    # recorded, and its index is as it was. Which is what is said, with its trusted index (the
    # GPU's change) as without (ffmpeg's), never that its manifest records the change "with the
    # next unit made": a finished job makes none.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    out = tmp_path / "out"
    files = (out / "manifest.json", out / "frame_index.bin")
    kept = written_as(*files)
    passes = first_passes(monkeypatch)
    steps.calls.clear()
    if change == "GPU":
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "another")
        differs = 'environment.gpu: "a stand-in" -> "another"'
    else:
        recorded = json.loads(files[0].read_text())["environment"]["ffmpeg"]
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        differs = f'environment.ffmpeg: "{recorded}" -> "n0.0-another"'
    text = refused(tmp_path, source_path, caplog, *JOB)
    assert differs in text
    assert "Only its environment differs: --accept-env-change resumes it anyway" in text
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == 0
    assert f"{out}: finished already, its 2 segments" in caplog.text
    assert passes == [] and steps.calls == [] and written_as(*files) == kept
    said = "its environment differs from this one, as accepted, and stays its manifest's"
    assert f"{out}: {said}, nothing being made in this one: {differs}" in caplog.text
    assert "resumed in another environment" not in caplog.text
    assert "the next unit made" not in caplog.text
    assert json.loads(files[0].read_text())["environment_changes"] == []


def test_finished_job_still_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # A finished job without its frame index is still held, with no first pass, to what needs no
    # decode: what its source declares and the output that follows, a probe saying otherwise
    # another job, as for a resume (test_probe_compared); and its directory to its manifest, a
    # segment missing or another's file refused, never deleted.
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
    (out / "seg_000001.mkv").rename(tmp_path / "moved.mkv")
    assert "finished, the manifest says, but missing" in refused(
        tmp_path, source_path, caplog, *JOB
    )
    (tmp_path / "moved.mkv").rename(out / "seg_000001.mkv")
    assert passes == [] and contents(out) == kept
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert passes == [] and contents(out) == kept


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
    assert "; or give --frame-rate 48/1, which takes its frames at that rate" in text
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
    assert "; or give --frame-rate 24000/1001, which takes its frames at that rate" in text
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

    def resolving(*arguments: Any, detector: bool = False) -> Any:
        asked.append(detector)
        return resolve(*arguments, detector=detector)

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
    assert "Its cuts change only while it is a plan, before its first unit" in caplog.text
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
    from seedvr2x.runtime import model
    from seedvr2x.runtime.job import read_cuts

    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    plan = tmp_path / "plan.txt"
    with monkeypatch.context() as patch:
        patch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
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
    lines = plan.read_text().splitlines()
    assert lines[0] == "# seedvr2x cut list of holes.mkv: 27 frames at 24000/1001 fps"
    assert lines[2] == "# Found: 3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3"
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
    # Only a plan's cuts change: another setting, alone or with another threshold, is another
    # job, refused before any first pass, the plan untouched. Once a unit is made, the job's cuts
    # are its own: another threshold or a cut list refused, saying why; the same command resumes.
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(tmp_path / "plan.txt")) == 0
    out = tmp_path / "out"
    kept = contents(out)
    passes.clear()
    text = refused(tmp_path, source_path, caplog, *CUT, "--seed", "7")
    assert "another job than the one asked" in text and "settings.seed: 42 -> 7" in text
    assert "Its cuts change only" not in text
    text = refused(tmp_path, source_path, caplog, *CUT, "--seed", "7", "--cut-threshold", "0.5")
    assert "settings.seed: 42 -> 7" in text and 'settings.cut_threshold: "0.3" -> "0.5"' in text
    assert contents(out) == kept and passes == []
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
        said = "Its cuts change only while it is a plan, before its first unit, and it has made"
        assert f"{said} some: its cuts are its own; another output directory takes new ones" in text
    assert contents(out) == kept and passes == []
    scripted.stop = None
    scripted.calls.clear()
    assert upscale(tmp_path, source_path, "out", *CUT) == 0
    assert encoded(scripted) == [5, 15, 22]


def test_one_file_plan_kept_in_its_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripted: Steps
) -> None:
    # -o x.mkv --plan FILE: FILE written, which replaces the one there, whole, with the
    # playlist of its possible cuts beside it, and the job's plan left in the work directory,
    # its manifest and frame index, as in an output directory: no unit, no file, no
    # checksums; the models never loaded.
    from seedvr2x.runtime import model
    from seedvr2x.runtime.job import read_cuts

    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "w.safetensors").write_bytes(b"")
    (tmp_path / pull.DETECTOR).write_bytes(b"")
    plan = tmp_path / "plan.txt"
    plan.write_text("an older plan")
    before = sorted(os.listdir(tmp_path))
    monkeypatch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
    assert upscale(tmp_path, source_path, "one.mkv", *SPLIT, "--plan", str(plan)) == 0
    assert sorted(os.listdir(tmp_path)) == sorted([*before, "one.mkv.work", "plan.txt.m3u8"])
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


def test_plan_writes_the_playlist_of_its_possible_cuts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan FILE also writes a playlist beside the list, FILE.m3u8, an entry per possible cut
    # (DESIGN.md, Shot detection), in frame order, never a detected cut's, and says what to do
    # with it: beside the source, open it in VLC. Planned again at 0.5 with a FILE elsewhere,
    # the cut at 22 a possible one: its entry with the others, and the playlist to move beside
    # the source. At 0.1, no possible cut: no playlist, the one from before left, and said. A cut
    # list's plan has none.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    plan, playlist = tmp_path / "plan.txt", tmp_path / "plan.txt.m3u8"
    assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(plan)) == 0
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *PLAYLIST]
    assert cut_lines(plan) == PLANNED
    said = [record.getMessage() for record in caplog.records]
    assert said[-2:] == [
        f"3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3: the cut list in {plan}",
        f"{playlist}: a playlist of the 2 possible cuts {PLAYLIST_SAID}; beside the source, which"
        " it names by its file name alone: open it in VLC",
    ]
    assert sorted(path.name for path in tmp_path.glob("plan*")) == ["plan.txt", "plan.txt.m3u8"]
    (tmp_path / "plans").mkdir()
    caplog.clear()
    options = (*CUT, "--cut-threshold", "0.5", "--plan", "plans/again")
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert (tmp_path / "plans" / "again.m3u8").read_text().splitlines() == [
        *PLAYLIST_HEADER,
        *PLAYLIST,
        "#EXTINF:-1,possible cut 22 at 0:00:00.880 (0.300)",
        "#EXTVLCOPT:start-time=0.000",
        "#EXTVLCOPT:stop-time=0.960",
        "in.mkv",
    ]
    assert [record.getMessage() for record in caplog.records][-1] == (
        f"plans/again.m3u8: a playlist of the 3 possible cuts {PLAYLIST_SAID}; it names the"
        " source by its file name alone, in.mkv: move it beside the source, then open it in VLC"
    )
    assert sorted(os.listdir(tmp_path / "plans")) == ["again", "again.m3u8"]
    caplog.clear()
    options = (*CUT, "--cut-threshold", "0.1", "--plan", str(plan))
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *PLAYLIST]
    assert [(r.levelname, r.getMessage()) for r in caplog.records][-1] == (
        "WARNING",
        f"{playlist}: left as it is: this plan has no possible cut, so no playlist; a playlist"
        " there is an earlier plan's, its entries not this plan's",
    )
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
    # The playlist of FILE's possible cuts goes beside it, as FILE.m3u8: in the place of what
    # the job reads or writes, FILE is refused as for its own place, before the first pass,
    # nothing made: the source; a .mkv path's finished file, its checksums, anything in its work
    # directory; anything in an output directory, or that directory itself, one named
    # FILE.m3u8; a directory. Each place by a link at the playlist's name, the one way there for
    # a .mkv path's, whose names end otherwise. With --cuts, whose plan has no possible cut and
    # no playlist, it isn't.
    source_path = job(tmp_path, monkeypatch)
    passes = first_passes(monkeypatch)
    (tmp_path / "a.m3u8").symlink_to("in.mkv")
    (tmp_path / "b.txt.m3u8").symlink_to("one.mkv")
    (tmp_path / "c.m3u8").symlink_to("out/c")
    (tmp_path / "d.cuts.m3u8").mkdir()
    (tmp_path / "e.m3u8").symlink_to("one.mkv.crc32")
    (tmp_path / "f.m3u8").symlink_to("one.mkv.work/f.m3u8")
    in_directory = "is in the output directory, which holds the job's own files only"
    work = f"is in the work directory of {tmp_path / 'one.mkv'}, which holds the job's own files"
    for output, plan, said in (
        ("out", "a", "a.m3u8 beside it, is the source"),
        ("one.mkv", "a", "a.m3u8 beside it, is the source"),
        ("one.mkv", "b.txt", "b.txt.m3u8 beside it, is the output"),
        ("one.mkv", "e", "e.m3u8 beside it, is the output's checksums"),
        ("one.mkv", "f", f"f.m3u8 beside it, {work} only"),
        ("out", "c", f"c.m3u8 beside it, {in_directory}"),
        ("p.m3u8", "p", f"p.m3u8 beside it, {in_directory}"),
        ("out", "d.cuts", "d.cuts.m3u8 beside it, is a directory"),
        ("one.mkv", "d.cuts", "d.cuts.m3u8 beside it, is a directory"),
    ):
        text = refused(tmp_path, source_path, caplog, "--plan", plan, output=output)
        expected = f"--plan {plan}: the playlist of its possible cuts, {said}; it names another"
        assert f"{expected} file" in text
        assert not (tmp_path / plan).exists()
    assert passes == [] and not (tmp_path / "out").exists() and not (tmp_path / "p.m3u8").exists()
    assert not (tmp_path / "one.mkv").exists() and not (tmp_path / "one.mkv.work").exists()
    # A link to a .mkv path's own files is no obstacle to a directory's job, nor a directory's to
    # a .mkv path's: each job is held to its own places.
    content = source_path.read_bytes()
    assert upscale(tmp_path, source_path, "out", "--cuts", "cuts.txt", "--plan", "a") == 0
    assert (tmp_path / "a").is_file() and (tmp_path / "a.m3u8").is_symlink()
    assert source_path.read_bytes() == content


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
    said = "A plan, which has made no unit yet, is planned again with other cuts; its environment"
    assert f"{said} differs too: --accept-env-change takes it up anyway" in text
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
    # The playlist of its 2 possible cuts beside it, as any detected plan's (cuts.write_playlist).
    assert sorted(path.name for path in tmp_path.glob("plan*")) == ["plan.txt", "plan.txt.m3u8"]
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
    # doesn't make again: said, FILE not written, nor the playlist of its possible cuts, exit 1,
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
    # The cut list written, the disk full for its playlist: said, and the command fails, the
    # list there.
    from seedvr2x.media import files

    write_whole = files.write_whole

    def full(path: Path, data: bytes) -> None:
        if path.suffix == ".m3u8":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        write_whole(path, data)

    source_path = job(tmp_path, monkeypatch)
    plan, playlist = tmp_path / "plan.txt", tmp_path / "plan.txt.m3u8"
    monkeypatch.setattr(files, "write_whole", full)
    text = refused(tmp_path, source_path, caplog, "--plan", str(plan), output="one.mkv")
    said = f"its playlist, {playlist}, not written: [Errno 28] No space left on device"
    assert f"--plan {plan}: {said}" in text
    assert cut_lines(plan) == PLANNED and not playlist.exists()


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
    # alone: the playlist's one entry, from a second before its time, 2.92 s, to 3 seconds after.
    # Planned again at 1, with no second pass: the three, in frame order, the last one's stop
    # the last frame's time. The playlist printed (-rP).
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
    plan, playlist = tmp_path / "plan.txt", tmp_path / "plan.txt.m3u8"
    times = [("1.920", "5.920"), ("5.000", "9.000"), ("7.480", "11.160")]

    def entries(count: int) -> list[str]:
        # Each possible cut's line in the list, `# <frame> <probability> <time>`, as its entry.
        lines = [line.split() for line in cut_lines(plan) if line.startswith("# ")]
        assert [int(line[1]) for line in lines] == CUTS[:count]
        return [
            written
            for (_, frame, peak, time), (start, stop) in zip(lines, times, strict=False)
            for written in (
                f"#EXTINF:-1,possible cut {frame} at {time} ({peak})",
                f"#EXTVLCOPT:start-time={start}",
                f"#EXTVLCOPT:stop-time={stop}",
                "clip.mkv",
            )
        ]

    assert upscale(tmp_path, clip, "out", "--cut-threshold", "0.97", "--plan", str(plan)) == 0
    assert "2 cuts at 0.97 (3 shots), 1 possible cut from 0.1 up to 0.97" in caplog.text
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *entries(1)]
    assert f"{playlist}: a playlist of the 1 possible cut for VLC" in caplog.text
    assert upscale(tmp_path, clip, "out", "--cut-threshold", "1", "--plan", str(plan)) == 0
    assert len(passes) == 1
    assert "0 cuts at 1.0 (1 shot), 3 possible cuts from 0.1 up to 1.0" in caplog.text
    assert playlist.read_text().splitlines() == [*PLAYLIST_HEADER, *entries(3)]
    assert [line.split()[3] for line in cut_lines(plan)] == [
        "0:00:02.920",
        "0:00:06.000",
        "0:00:08.480",
    ]
    print("\n".join(record.getMessage() for record in caplog.records))
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
