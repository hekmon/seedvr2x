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
import os
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
import torch

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime.job import output_size
from seedvr2x.runtime.shot import padded_length


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
    (channel 0) and frame count (channel 1)."""
    assert read(count).shape[0] == count
    latent = torch.zeros(16, (padded_length(count) - 1) // 4 + 1, 1, 1)
    latent[0], latent[1] = seed - SEED, count
    return latent.permute(1, 2, 3, 0)


def stand_in_windows(
    models: object, latent: torch.Tensor, layout: list[tuple[int, int]], seed: int, start: int = 0
) -> Iterator[torch.Tensor]:
    for s, e in layout[start:]:
        yield latent[s:e] + 0


def stand_in_decode(
    models: object,
    merged: torch.Tensor,
    count: int,
    target: tuple[int, int],
    write: Callable[[npt.NDArray[np.float32]], None],
) -> None:
    height, width = output_size(target)
    first, frames = int(merged[0, 0, 0, 0]), int(merged[0, 0, 0, 1])
    # Every latent says so: the shot's own windows, none of another's.
    assert (merged[..., 0] == first).all() and (merged[..., 1] == frames).all()
    assert frames == count
    for index in range(first, first + count):
        # As the decode gives them: a view of (C, t, H, W) planes.
        planes = np.full((3, 1, height, width), index / 1000, dtype=np.float32)
        write(planes.transpose(1, 2, 3, 0))


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from seedvr2x.runtime import model, run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "a stand-in")
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(
        model, "load_models", lambda *a: SimpleNamespace(attention="none", device="cpu")
    )
    monkeypatch.setattr(model, "nvidia_driver", lambda: "a stand-in")
    monkeypatch.setattr(run, "encode_shot", stand_in_encode)
    monkeypatch.setattr(run, "sample_windows", stand_in_windows)
    monkeypatch.setattr(run, "decode_shot", stand_in_decode)


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
    weights = tmp_path / "w.safetensors"
    if not weights.exists():
        weights.write_bytes(b"")
    return cli.main(
        [
            *(str(input_path), "-o", str(tmp_path / output)),
            *("--model-dir", str(tmp_path), "--dit-model", "w.safetensors"),
            *("--vae-model", "w.safetensors", "--resolution", "96", "--seed", str(SEED)),
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
    decode's first frame, as a stop at once would."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.read: dict[int, str] = {}
        self.stop: str | None = None

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
        return stand_in_encode(models, lambda n: frames[:n], count, target, seed)

    def windows(
        self,
        models: object,
        latent: torch.Tensor,
        layout: list[tuple[int, int]],
        seed: int,
        start: int = 0,
    ) -> Iterator[torch.Tensor]:
        for number, sampled in enumerate(stand_in_windows(models, latent, layout, seed, start)):
            self._call(f"window {seed - SEED}:{start + number}")
            yield sampled

    def decode(
        self,
        models: object,
        merged: torch.Tensor,
        count: int,
        target: tuple[int, int],
        write: Callable[[npt.NDArray[np.float32]], None],
    ) -> None:
        name = f"decode {int(merged[0, 0, 0, 0])}"
        self.calls.append(name)

        def write_then_stop(frames: npt.NDArray[np.float32]) -> None:
            write(frames)
            if name == self.stop:
                raise KeyboardInterrupt

        stand_in_decode(models, merged, count, target, write_then_stop)


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
def test_resumed_as_uninterrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: Steps, stop: str, resumed: list[str]
) -> None:
    source_path = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source_path, "whole", *JOB) == 0
    read = dict(steps.read)
    steps.calls.clear()
    steps.stop = stop
    stopped(tmp_path, source_path, "out", *JOB)
    assert steps.calls[-1] == stop
    steps.calls.clear()
    steps.stop = None
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
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
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert steps.calls == []
    assert not (out / "resume").exists()


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
