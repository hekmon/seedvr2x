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
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_probe import BT709_VUI, has_encoder, has_x265, x265_file
from test_rate import mechanism7, retagged, strayed

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import weights
from seedvr2x.runtime.job import output_size, padded_size
from seedvr2x.runtime.shot import decode_shot, padded_length


def _usable() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1"
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
) -> torch.Tensor:
    """A latent (T', 1, 1, 16) in channel-major memory, as the VAE's, saying the shot's first frame
    (channel 0), its frame count (1), the target's height and width (2, 3), and whether the frames
    were padded as numz pads them (6: cli.NUMZ_PADDING)."""
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


# The stand-in's models: the CPU, and the VAE's slicing, which the encode and split's reference
# follow (model.encode_slices).
STAND_IN_MODELS = SimpleNamespace(
    attention="none",
    device=torch.device("cpu"),
    runner=SimpleNamespace(vae=SimpleNamespace(use_slicing=True, slicing_sample_min_size=4)),
)


def stand_in_model(patch: Callable[[Any, str, Any], None]) -> None:
    """Replace the model by the stand-in, through patch: monkeypatch.setattr, or setattr in a
    process of its own (test_stop.py). The decode around the VAE's is seedvr2x's own. Its files,
    one empty w.safetensors as both, pass for models: the check is test_weights.py's."""
    from seedvr2x.runtime import model, run, weights

    patch(weights, "check_models", lambda *a: None)
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
    stand-in's frames then say which they are (stand_in_decode_stream). In gbrp16le unless options
    name a format, not the default yuv420p10le: the tests read the masters' frames as the RGB
    planes written (indexes, stored)."""
    weights = tmp_path / "w.safetensors"
    if not weights.exists():
        weights.write_bytes(b"")
    correction = [] if "--color-correction" in options else ["--color-correction", "none"]
    output_format = [] if "--format" in options else ["--format", "gbrp16le"]
    return cli.main(
        [
            *(str(input_path), "-o", str(tmp_path / output)),
            *("--model-dir", str(tmp_path), "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", str(SEED)),
            *correction,
            *output_format,
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
    assert not (tmp_path / "manifest.json").exists()
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
    ) -> torch.Tensor:
        frames = read(count)
        self.read[seed - SEED] = hashlib.md5(frames.tobytes()).hexdigest()
        self.numz_padding.append(numz_padding)
        self._call(f"encode {seed - SEED}")
        latent = stand_in_encode(
            models, lambda n: frames[:n], count, target, seed, numz_padding=numz_padding
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
        *("encode 0", "window 0:0", "decode 0", "encode 3", "window 3:0", "decode 3"),
        *("encode 4", "window 4:0", "window 4:1", "decode 4"),
    ]
    assert indexes(tmp_path / "one.mkv") == list(range(25))


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
    # One file keeps nothing: stopped, the master's partial file removed.
    source_path = job(tmp_path, monkeypatch)
    steps.poison = "window 3:0"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4]) == 1
    assert "shot 2/3's window 1/1: " in caplog.text and "nothing kept" in caplog.text
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("one")]


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

    def looked(*arguments: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments)

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
    # One file keeps nothing: stopped, the copy named, the work directory and the partial master
    # removed.
    source_path = job(tmp_path, monkeypatch)
    steps.damage = "decode 4"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 1
    copy = tmp_path / "one.mkv.work" / "shot_000002" / "input.mkv"
    assert f"{copy}: the shot's input copy, not readable whole and intact: " in caplog.text
    assert "slice CRC mismatch" in caplog.text and "on resuming" not in caplog.text
    assert "; stopped: nothing kept" in caplog.text
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("one")]


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
    # One file's copies go in a work directory beside it, removed at the end, at a stop or an
    # error. A killed run's is removed; anything else there is refused, and kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    one, work = tmp_path / "one.mkv", tmp_path / "one.mkv.work"
    assert upscale(tmp_path, source_path, "out", *JOB, *SPLIT) == 0
    seen: list[list[str]] = []
    decode = steps.decode

    def looked(*arguments: Any) -> None:
        seen.append(sorted(path.name for path in work.iterdir()))
        decode(*arguments)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 0
    assert seen == [["shot_000000"], ["shot_000001"], ["shot_000002"]]  # one shot at a time
    assert not work.exists()
    out = tmp_path / "out"
    assert decoded(one) == decoded(out / "seg_000000.mkv") + decoded(out / "seg_000001.mkv")
    for stop, poison, status in (("decode 4", None, 130), (None, "decode 4", 1)):
        one.unlink(missing_ok=True)
        steps.stop, steps.poison = stop, poison
        assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == status
        assert not work.exists() and not one.exists()
    steps.stop = steps.poison = None
    (work / "shot_000002").mkdir(parents=True)
    for name in ("input.mkv.partial", "input.crc32.partial"):
        (work / "shot_000002" / name).write_bytes(b"left")
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 0
    assert "left by a run that was killed, emptied" in caplog.text and not work.exists()
    work.mkdir()
    (work / "notes.txt").write_text("mine")
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], *SPLIT) == 1
    assert "not what a run of seedvr2x leaves" in caplog.text
    assert (work / "notes.txt").read_text() == "mine"


def test_none_one_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps) -> None:
    # With none, nothing is copied (DESIGN.md, Colour correction): one file gets no work
    # directory beside it, and no shot an input copy; each decode goes uncorrected.
    source_path = job(tmp_path, monkeypatch)
    work = tmp_path / "one.mkv.work"
    seen: list[tuple[bool, list[str], object]] = []
    decode = steps.decode

    def looked(*arguments: Any) -> None:
        copies = [path.name for path in tmp_path.rglob("input.*")]
        seen.append((work.exists(), copies, arguments[-1]))  # the last, its correction
        decode(*arguments)

    from seedvr2x.runtime import run

    monkeypatch.setattr(run, "decode_shot", looked)
    assert upscale(tmp_path, source_path, "one.mkv", *JOB[:4], "--color-correction", "none") == 0
    assert seen == [(False, [], None)] * 3
    assert indexes(tmp_path / "one.mkv") == list(range(25))


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

    def decode(*arguments: Any) -> None:
        models, merged, count, target, write, *names = arguments

        def written(frames: npt.NDArray[np.float32]) -> None:
            look()
            events.append(f"write {frames.shape[0]}")
            write(frames)

        shot.decode_shot(models, merged, count, target, written, *names)

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
    # The checksums of a one-file output go only when a new file replaces it: a run to the same
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
    # Every encode of the other paths too: one file, each shot decoded as soon as it is sampled
    # (run_job), and split, each encode copying the frames it reads (run._sample).
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
    monkeypatch.setattr(examined, "scan", lambda path: replace(scan(path), frames=29))
    monkeypatch.setattr(ffmpeg, "check", lambda *options: "n0.0-another")
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

    def missed(path: Path) -> object:
        found = scan(path)
        return replace(found, frames=found.frames - 1)

    monkeypatch.setattr(examined, "scan", missed)
    assert upscale(tmp_path, source(tmp_path / "in.mkv"), "one.mkv") == 1
    assert f"decoding with ffmpeg: frames left after {FRAMES - 1}" in caplog.text


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

    def looked(*arguments: Any) -> None:
        seen.append(copied(arguments[-1].copy, arguments[2]))
        decode(*arguments)

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

    def counted(path: Path) -> object:
        scans.append(path)
        return scan(path)

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

    def counted(path: Path) -> object:
        scans.append(path)
        found = scan(path)
        if not otherwise[0]:
            return found
        assert found.index is not None
        crc32 = found.index.crc32.copy()
        crc32[7] ^= 1
        return replace(found, index=replace(found.index, crc32=crc32))

    monkeypatch.setattr(examined, "scan", counted)
    return scans, otherwise


@pytest.mark.parametrize("stop", [None, "encode 4"])
def test_index_made_again_named_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,
    caplog: pytest.LogCaptureFixture,
    stop: str | None,
) -> None:
    # A first pass run again, its recorded index missing, that gives another index than the one
    # recorded: the index is written with a manifest naming it, at once, in the environment
    # recorded, whether the job is finished already, which makes no unit to write its manifest
    # with, or stopped before one. The next run then trusts the record, and runs no first pass;
    # left to the next unit, every run of the finished job warned of an index its manifest didn't
    # name and ran the pass again.
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = stop
    assert upscale(tmp_path, source_path, "out", *JOB) == (0 if stop is None else 130)
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
    assert upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change") == (
        0 if stop is None else 130
    )
    assert f"{index}: the frame index missing; the first pass runs again" in caplog.text
    assert len(scans) == 1 and steps.calls == ([] if stop is None else ["encode 4"])
    assert ("finished already" in caplog.text) == (stop is None)
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
    # environment, never before. A resume stopped before a unit, or a job finished already, which
    # makes none, keeps its recorded index and its manifest as they were, and the first ffmpeg
    # then finds its record whole, no pass run again.
    scans, otherwise = another_index(monkeypatch)
    source_path = job(tmp_path, monkeypatch)
    steps.stop = stop
    assert upscale(tmp_path, source_path, "out", *JOB) == (0 if stop is None else 130)
    out = tmp_path / "out"
    index, manifest = out / "frame_index.bin", out / "manifest.json"
    first, recorded = index.read_bytes(), manifest.read_bytes()
    check = ffmpeg.check
    monkeypatch.setattr(ffmpeg, "check", lambda *options: "n0.0-another")
    otherwise[0] = True
    scans.clear()
    caplog.set_level(logging.INFO, logger="seedvr2x")
    status = upscale(tmp_path, source_path, "out", *JOB, "--accept-env-change")
    assert status == (0 if stop is None else 130) and len(scans) == 1
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
    monkeypatch.setattr(ffmpeg, "check", lambda *options: "n0.0-another")
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


def test_frame_index_beside_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # The index is written whole before the manifest's first write, which names it in its input
    # record by its size and SHA-256 (manifest.INDEX); one file keeps it in memory. What an
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
    assert "1 leftovers discarded" in caplog.text
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

    def concealed(path: Path) -> object:
        found = scan(path)
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


def test_a_drift_off_the_declared_rate_warned_of_again_on_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, caplog: pytest.LogCaptureFixture
) -> None:
    # 48 fps declared 50/1 (test_rate.retagged), 100 frames: read at the rate it declares, its
    # frames half a frame and more off that rate's timeline from frame 12, warned of
    # (media/source.py, _strays). A resume takes the first pass from its record, reads on at
    # the same rate, and says so again, from the recorded index, in the same words: the
    # warning was the first pass's alone, and a job resumed read on without it.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path) -> object:
        scans.append(path)
        return scan(path)

    monkeypatch.setattr(examined, "scan", counted)
    monkeypatch.chdir(tmp_path)
    source_path = retagged(tmp_path / "retagged.mkv", 50, "round(N*1000/48)", 100)
    (tmp_path / "cuts.txt").write_text("12\n")
    options = ("--cuts", "cuts.txt", "--min-segment", "0.1")
    steps.stop = "encode 12"
    stopped(tmp_path, source_path, "out", *options)
    first = strayed(caplog)
    assert len(first) == 1 and len(scans) == 1
    assert first[0].startswith(
        f"{source_path}: 88 of its 100 frames half a frame or more from their place on the"
        " timeline of the rate it declares, 50/1, from frame 12, up to 83 ms (4.15 of a frame):"
    )
    assert "its timestamps follow 48/1 fps (48) exactly" in first[0]
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *options) == 0
    assert len(scans) == 1  # the first pass's record trusted
    assert strayed(caplog) == first
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
    assert len(scans) == 2 and not strayed(caplog)


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
    # 260 frames within 10.8 ms of 24000/1001's timeline. Refused without --frame-rate, the
    # refusal giving that rate; with --frame-rate 24000/1001, every frame read, in order, and that
    # rate the job's everywhere (media/source.py, Source.frame_rate): each output segment
    # declares it; the segments' merge rule takes it, 0.5005 s being 12 frames at 24000/1001 and
    # 13 at 24/1, so the 12-frame first shot is a segment of its own; the manifest records it,
    # its settings and output, the input keeping the rate it declares. --frame-rate 24 refused,
    # naming frame 252. A resume with another --frame-rate, or none, is another job, refused
    # before its first pass; with the same, the uninterrupted run's output.
    from seedvr2x.media import source as examined

    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path) -> object:
        scans.append(path)
        return scan(path)

    monkeypatch.setattr(examined, "scan", counted)
    monkeypatch.chdir(tmp_path)
    source_path = mechanism7(tmp_path / "s9.mkv", size="64x48")
    (tmp_path / "cuts.txt").write_text("12\n")
    options = ("--cuts", "cuts.txt", "--min-segment", "0.5005")
    ntsc = ("--frame-rate", "24000/1001")
    text = refused(tmp_path, source_path, caplog, *options)
    assert "s9.mkv: its timestamps run at 24000/1001 fps (23.976), within 10.8 ms" in text
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
