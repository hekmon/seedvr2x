"""The command line's run on the CPU, the model replaced by a stand-in: the input read, the shots
and their seeds, the output written, its segments and its manifest, in every output format.

The stand-in shot writes, for each frame, the frame's index in the job divided by 1000, which its
seed gives (the job's seed plus the shot's first frame): so each output frame says where it comes
from. The model's own output is the GPU tests' (test_regression.py, test_shots.py)."""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime.job import output_size


def _usable() -> bool:
    try:
        ffmpeg.check(("png",))
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _usable(), reason="needs ffmpeg with zscale, scdet and ffv1")

SEED = 42
FRAMES = 9


def stand_in_shot(
    models: object,
    read: Callable[[int], npt.NDArray[np.float32]],
    count: int,
    target: tuple[int, int],
    seed: int,
    write: Callable[[npt.NDArray[np.float32]], None],
    window: int | None = None,
) -> None:
    assert read(count).shape[0] == count
    height, width = output_size(target)
    start = seed - SEED
    for index in range(start, start + count):
        # As the decode gives them: a view of (C, t, H, W) planes.
        planes = np.full((3, 1, height, width), index / 1000, dtype=np.float32)
        write(planes.transpose(1, 2, 3, 0))


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from seedvr2x.runtime import model, run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(
        model, "load_models", lambda *a: SimpleNamespace(attention="none", device=None)
    )
    monkeypatch.setattr(run, "upscale_shot", stand_in_shot)


def source(path: Path, frames: int = FRAMES) -> Path:
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)),
        ],
        check=True,
    )
    return path


def upscale(tmp_path: Path, input_path: Path, output: str, *options: str) -> int:
    (tmp_path / "w.safetensors").write_bytes(b"")
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
    # 128x96 is an SD size: BT.601, provisionally tagged smpte170m (writer.yuv_matrix).
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
