"""A directory of segments as input (sptenc's manual workflow): its files and their order, one
source made of them, each join a cut, the output mirroring them (DESIGN.md, Input).

The ffmpeg tests skip without a build seedvr2x accepts; the GPU test needs what
tests/test_regression.py needs."""

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.source import Source, examine_directory, segment_files
from seedvr2x.media.writer import SegmentWriter, Tags, Writer, count_packets, open_writer
from seedvr2x.runtime.job import (
    JobError,
    OutputSegment,
    Shot,
    job_shots,
    mirrored_segments,
    parts_of,
)


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _usable(), reason="needs ffmpeg with zscale, scdet and ffv1")


def segment(path: Path, frames: int, size: str = "64x48", rate: str = "25") -> Path:
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}"),
            *("-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(path)),
        ],
        check=True,
    )
    return path


def test_segment_files_as_sptenc_reads_them(tmp_path: Path) -> None:
    # .mkv and .mp4 in any case, sorted by name byte for byte; directories and the rest skipped.
    for name in ("seg_000001.mkv", "seg_000000.MKV", "a.mp4", "b.mov", "c.txt", "Z.mkv"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.mkv").mkdir()
    assert [p.name for p in segment_files(tmp_path)] == [
        "Z.mkv",
        "a.mp4",
        "seg_000000.MKV",
        "seg_000001.mkv",
    ]


@needs_ffmpeg
def test_one_source_of_segments(tmp_path: Path) -> None:
    for index, frames in enumerate((3, 2, 4)):
        segment(tmp_path / f"seg_{index:06d}.mkv", frames)
    sources = examine_directory(tmp_path)
    parts = parts_of(sources)
    assert [(p.start, p.end) for p in parts] == [(0, 3), (3, 5), (5, 9)]
    assert job_shots(parts, []) == [Shot(0, 3), Shot(3, 5), Shot(5, 9)]
    assert mirrored_segments(parts) == [
        OutputSegment("seg_000000", 0, 3),
        OutputSegment("seg_000001", 3, 5),
        OutputSegment("seg_000002", 5, 9),
    ]
    # The seeds count from the directory's first frame, as the same frames in one file would.
    assert [shot.seed(42) for shot in job_shots(parts, [])] == [42, 45, 47]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("size", "rate", "what"),
    [("64x48", "24", "frame rate 24, where seg_000000.mkv has 25"), ("32x48", "25", "size 32x48")],
)
def test_segments_must_make_one_source(tmp_path: Path, size: str, rate: str, what: str) -> None:
    segment(tmp_path / "seg_000000.mkv", 3)
    segment(tmp_path / "seg_000001.mkv", 3, size, rate)
    with pytest.raises(MediaError, match=f"seg_000001.mkv: {what}"):
        examine_directory(tmp_path)


def test_empty_directory_refused(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("")
    with pytest.raises(MediaError, match="no segment"):
        examine_directory(tmp_path)


def fake_sources(directory: Path, **frames: int) -> list[Source]:
    """Sources as far as the layout reads them: a path and a frame count."""
    return cast(
        list[Source], [SimpleNamespace(path=directory / n, frames=f) for n, f in frames.items()]
    )


def test_job_shots_cut_at_joins_and_cuts(tmp_path: Path) -> None:
    parts = parts_of(fake_sources(tmp_path, a=10, b=5))
    assert job_shots(parts, []) == [Shot(0, 10), Shot(10, 15)]
    assert job_shots(parts, [4, 10, 12]) == [Shot(0, 4), Shot(4, 10), Shot(10, 12), Shot(12, 15)]
    # The cut list is checked as given, before the joins join it.
    for cuts in ([12, 4], [4, 4], [0], [15]):
        with pytest.raises(JobError):
            job_shots(parts, cuts)


def test_names_must_differ_by_stem(tmp_path: Path) -> None:
    parts = parts_of(fake_sources(tmp_path, **{"seg_1.mkv": 2, "seg_1.mp4": 2}))
    with pytest.raises(JobError, match="two segments named seg_1"):
        mirrored_segments(parts)


@needs_ffmpeg
def test_segment_writer_routes_frames(tmp_path: Path) -> None:
    frames = np.random.default_rng(0).random((5, 16, 32, 3), dtype=np.float32)
    outputs = [(tmp_path / "a.mkv", 3), (tmp_path / "b.mkv", 2)]

    def open_segment(path: Path) -> Writer:
        return open_writer("gbrp16le", path, 32, 16, Fraction(25), Tags())

    with SegmentWriter(outputs, open_segment) as writer:
        for chunk in (frames[:2], frames[2:4], frames[4:]):
            writer.write(chunk)
    assert writer.written == 5
    assert [count_packets(path) for path, _ in outputs] == [3, 2]
    with pytest.raises(RuntimeError, match="stopped after 4 frames, in segment 2 of 2"):
        with SegmentWriter(outputs, open_segment) as writer:
            writer.write(frames[:4])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mkv", "b.mkv"]  # no partial left
    with pytest.raises(ValueError, match="1 frames beyond"):
        with SegmentWriter(outputs, open_segment) as writer:
            writer.write(np.concatenate([frames, frames[:1]]))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--cuts", "cuts.txt"], "--cuts with a directory"),
        (["-o", "out.mkv"], "needs a directory as output"),
        (["-o", "full"], "not empty"),
    ],
)
def test_directory_refusals_before_torch(tmp_path: Path, options: list[str], message: str) -> None:
    source = tmp_path / "in"
    source.mkdir()
    segment(source / "seg_000000.mkv", 3)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "old.mkv").write_bytes(b"")
    (tmp_path / "cuts.txt").write_text("1\n")
    args = [str(source), "-o", "out", "--model-dir", ".", "--dit-model", "x", *options]
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " assert 'torch' not in sys.modules; sys.exit(status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 1, result.stderr
    assert message in result.stderr


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")


@pytest.mark.gpu
@pytest.mark.skipif(
    not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
)
def test_directory_output_mirrors_its_segments(tmp_path: Path) -> None:
    # Milestone 1's 45 frames split losslessly into 3 segments of 10, 2 and 33 frames: the output
    # has the same names and frame counts, and its frames are those of the file cut at the joins,
    # the seeds counting from the directory's first frame.
    assert MODELS is not None and REFERENCE is not None
    source = Path(REFERENCE) / "m1" / "input_rgb.mkv"
    split = tmp_path / "split"
    split.mkdir()
    for index, (start, end) in enumerate(((0, 10), (10, 12), (12, 45))):
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-i", str(source)),
                *("-vf", f"select=between(n\\,{start}\\,{end - 1})", "-fps_mode", "passthrough"),
                *("-c:v", "ffv1", "-pix_fmt", "bgr0", str(split / f"seg_{index:06d}.mkv")),
            ],
            check=True,
        )
    (tmp_path / "cuts.txt").write_text("10\n12\n")
    common = ["--model-dir", MODELS, "--dit-model", "seedvr2_ema_7b_fp16.safetensors"]
    common += ["--resolution", "540", "--window", "5"]
    runs = (
        (split, tmp_path / "mirrored", tmp_path / "dir_frames", []),
        (source, tmp_path / "one.mkv", tmp_path / "file_frames", ["--cuts", "cuts.txt"]),
    )
    for input_path, output, dump, options in runs:
        run = subprocess.run(
            [
                *(sys.executable, "-m", "seedvr2x", str(input_path), "-o", str(output)),
                *common,
                *("--dump-frames", str(dump), *options),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )
        assert run.returncode == 0, run.stderr[-3000:]
        # The versions recorded cover what the run imports, the models loaded too.
        assert "not in the manifest's environment" not in run.stderr
    mirrored = tmp_path / "mirrored"
    names = [f"seg_{i:06d}.mkv" for i in range(3)]
    assert sorted(p.name for p in mirrored.iterdir()) == sorted([*names, "manifest.json"])
    assert [count_packets(mirrored / f"seg_{i:06d}.mkv") for i in range(3)] == [10, 2, 33]
    for index in range(45):
        name = f"frame_{index:06d}.npy"
        assert np.array_equal(
            np.load(tmp_path / "dir_frames" / name), np.load(tmp_path / "file_frames" / name)
        ), index
