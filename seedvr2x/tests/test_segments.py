"""Output segments: sptenc's minimum length merge, their names, the manifest (DESIGN.md, Output).

The merge tests are sptenc's own (core/scenes_test.go at vmafv1 5790944), its boundaries placed on
a 25 fps grid as there: at(s) is frame 25 s, and a duration of s seconds is 25 s frames. The GPU
test needs what tests/test_regression.py needs."""

import json
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from test_weights import default_models

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.source import Source
from seedvr2x.runtime.job import (
    OutputSegment,
    Shot,
    merge_short,
    merged_segments,
    min_segment_frames,
    parts_of,
)
from seedvr2x.runtime.manifest import Manifest, code_sha256


def at(second: int) -> int:
    return second * 25


def merge(cuts: list[int], total_seconds: int, min_seconds: int) -> list[int]:
    return merge_short(
        cuts, at(total_seconds), min_segment_frames(Fraction(min_seconds), Fraction(25))
    )


def test_zero_min_duration() -> None:
    assert merge([at(1), at(2)], 10, 0) == [at(1), at(2)]


def test_no_short_segments() -> None:
    # 5 s segments, a 3 s minimum: no change.
    assert merge([at(5), at(10), at(15)], 20, 3) == [at(5), at(10), at(15)]


def test_single_short_at_start() -> None:
    # 1 s | 9 s, 4 s minimum: the first merges right.
    assert merge([at(1)], 10, 4) == []


def test_single_short_at_end() -> None:
    # 9 s | 1 s, 4 s minimum: the last merges left.
    assert merge([at(9)], 10, 4) == []


def test_single_short_in_middle() -> None:
    # 5 s | 1 s | 4 s, 2 s minimum: the right neighbour is shorter.
    assert merge([at(5), at(6)], 10, 2) == [at(5)]


def test_merge_into_shorter_neighbour() -> None:
    # 6 s | 1 s | 3 s, 2 s minimum: the right neighbour is shorter.
    assert merge([at(6), at(7)], 10, 2) == [at(6)]


def test_equal_neighbours_prefer_left() -> None:
    # 3 s | 1 s | 3 s, 2 s minimum: a tie merges left, removing the cut at 3 s.
    assert merge([at(3), at(4)], 7, 2) == [at(4)]


def test_cluster_of_short_segments() -> None:
    # 1 s | 1 s | 1 s | 5 s | 1 s, 2 s minimum: first right, then left, then the last left.
    assert merge([at(1), at(2), at(3), at(8)], 9, 2) == [at(3)]


def test_total_duration_below_min() -> None:
    assert merge([at(2), at(4)], 5, 10) == []


def test_empty_scenes() -> None:
    assert merge([], 10, 2) == []


def test_merge_creates_new_short_segment() -> None:
    # 3 s | 2 s | 2 s | 3 s, 3 s minimum: the first 2 s one goes right, making 4 s.
    assert merge([at(3), at(5), at(7)], 10, 3) == [at(3), at(7)]


def test_exactly_min_duration() -> None:
    # A segment lasting exactly the minimum is long enough.
    assert merge([at(5)], 10, 5) == [at(5)]


def test_container_rounding() -> None:
    # At 23.976 fps, the 10-frame segment 156-166 between two of 156 frames: a tie, merged
    # left, the cut at 166 kept, whatever the container's millisecond rounding (in frames).
    assert merge_short([156, 166], 322, min_segment_frames(Fraction(5), Fraction(24000, 1001))) == [
        166
    ]


@pytest.mark.parametrize(
    ("seconds", "rate", "frames"),
    [
        (Fraction(5), Fraction(25), 125),  # exactly 125 frames: 125 is not short
        (Fraction(5), Fraction(24000, 1001), 120),  # 119.88 frames: 119 is short, 120 is not
        (Fraction(5), Fraction(30000, 1001), 150),  # 149.85
        (Fraction(0), Fraction(25), 0),
        (Fraction("4.99"), Fraction(24000, 1001), 120),  # 119.64, exact from the decimal
    ],
)
def test_min_segment_frames(seconds: Fraction, rate: Fraction, frames: int) -> None:
    assert min_segment_frames(seconds, rate) == frames


def test_merged_segments_keep_whole_shots() -> None:
    shots = [Shot(0, 10), Shot(10, 12), Shot(12, 30), Shot(30, 45)]
    # 0.5 s at 24000/1001 is 12 frames: the 2-frame shot merges into the 10-frame one before it.
    segments = merged_segments(shots, 45, Fraction(24000, 1001), Fraction(1, 2))
    assert segments == [
        OutputSegment("seg_000000", 0, 12),
        OutputSegment("seg_000001", 12, 30),
        OutputSegment("seg_000002", 30, 45),
    ]
    assert merged_segments(shots, 45, Fraction(24000, 1001), Fraction(0)) == [
        OutputSegment(f"seg_{i:06d}", shot.start, shot.end) for i, shot in enumerate(shots)
    ]
    assert merged_segments(shots, 45, Fraction(24000, 1001), Fraction(5)) == [
        OutputSegment("seg_000000", 0, 45)
    ]


def test_manifest(tmp_path: Path) -> None:
    sources = [
        SimpleNamespace(
            path=tmp_path / name,
            frames=frames,
            stream=SimpleNamespace(
                frame_rate=Fraction(25),
                width=64,
                height=48,
                pix_fmt="yuv420p",
                color_primaries="",
                color_transfer="",
            ),
            sample_aspect=Fraction(1),
            conversion=SimpleNamespace(describe=lambda: "YUV bt709, limited range, chroma left"),
            sha256=f"the content of {name}",
        )
        for name, frames in (("a.mkv", 3), ("b.mkv", 2))
    ]
    for source in sources:
        source.path.write_bytes(b"x" * source.frames)
    parts = parts_of(cast(list[Source], sources))
    record = Manifest(
        tmp_path / "manifest.json",
        {"seed": 42},
        {"gpu": "a GPU"},
        parts,
        [Shot(0, 3), Shot(3, 5)],
        [[(0, 1)], [(0, 1)]],
        [OutputSegment("a", 0, 3), OutputSegment("b", 3, 5)],
        ["a.mkv", "b.mkv"],
        {"format": "gbrp16le"},
    )
    record.write()
    record.shot_encoded(0)
    record.window_done(0, 0)
    record.segment_finished(0, 1234)
    content = json.loads((tmp_path / "manifest.json").read_text())
    assert content["seedvr2x_manifest"] == 2
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha256()) and code_sha256() == code_sha256()
    assert content["environment"] == {"gpu": "a GPU"}
    assert [(s["finished"], s["bytes"]) for s in content["segments"]] == [
        (True, 1234),
        (False, None),
    ]
    assert [s["seed"] for s in content["shots"]] == [42, 45]
    assert [(s["encoded"], s["windows_done"]) for s in content["shots"]] == [(True, 1), (False, 0)]
    assert [(s["latents"], s["windows"]) for s in content["shots"]] == [(1, [[0, 1]])] * 2
    assert [(i["start"], i["frames"], i["bytes"]) for i in content["input"]] == [
        (0, 3, 3),
        (3, 2, 2),
    ]
    assert content["input"][0]["path"] == str((tmp_path / "a.mkv").resolve())
    assert content["input"][1]["sha256"] == "the content of b.mkv"
    assert not (tmp_path / "manifest.json.partial").exists()
    with pytest.raises(ValueError, match="window 2 after 1"):
        record.window_done(0, 2)


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg with zscale, scdet and ffv1")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["-o", "out.mkv", "--format", "png"], "PNG output goes to a directory"),
        (["-o", "file.txt"], "a file; the output segments need a directory"),
        (["-o", "out", "--min-segment", "-1"], "negative"),
        (["-o", "out.mp4"], "a video file name"),
        (["-o", "dir.mkv"], "a .mkv path names the one FFV1 master"),
        (["-o", "out", "--resolution", "0"], "not a whole number above 0"),
        (["-o", "out", "--seed", "-1"], "--seed -1"),
        (["-o", "out", "--seed", "4293967296"], "between 0 and 4293967295"),
    ],
)
def test_output_refusals_before_torch(tmp_path: Path, options: list[str], message: str) -> None:
    source = tmp_path / "in.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "3", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    (tmp_path / "file.txt").write_text("")
    (tmp_path / "dir.mkv").mkdir()
    # Models the check accepts, under the default names: the seed's refusals come after it.
    default_models(tmp_path)
    args = [str(source), "--model-dir", ".", *options]
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " sys.exit(3 if 'torch' in sys.modules else status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode in (1, 2), result.stderr  # 2: argparse's refusal
    assert message in result.stderr


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")


def decoded(path: Path) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
        capture_output=True,
        check=True,
    ).stdout


@pytest.mark.gpu
@pytest.mark.skipif(
    not MODELS or not REFERENCE, reason="needs SEEDVR2X_MODEL_DIR and SEEDVR2X_REFERENCE_DIR"
)
def test_joined_segments_are_the_one_file_output(tmp_path: Path) -> None:
    # Milestone 1's 45 frames, shots of 10, 2, 18 and 15 frames, segments of half a second at
    # least (12 frames): 12, 18 and 15. Joined, they are the one-file output, frame for frame.
    assert MODELS is not None and REFERENCE is not None
    source = Path(REFERENCE) / "m1" / "input_rgb.mkv"
    (tmp_path / "cuts.txt").write_text("10\n12\n30\n")
    common = ["--model-dir", MODELS, "--dit-model", "seedvr2_ema_7b_fp16.safetensors"]
    common += ["--vae-model", "ema_vae_fp16.safetensors"]
    common += ["--resolution", "540", "--window", "5", "--cuts", "cuts.txt"]
    # gbrp16le: the frames are read below as stored, three 16-bit planes.
    common += ["--format", "gbrp16le"]
    for output, options in (("one.mkv", []), ("segments", ["--min-segment", "0.5"])):
        run = subprocess.run(
            [sys.executable, "-m", "seedvr2x", str(source), "-o", output, *common, *options],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )
        assert run.returncode == 0, run.stderr[-3000:]
    directory = tmp_path / "segments"
    names = [f"seg_{i:06d}.mkv" for i in range(3)]
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [*names, "checksums", "manifest.json"]
    )
    content = json.loads((directory / "manifest.json").read_text())
    assert [(s["name"], s["start"], s["end"], s["finished"]) for s in content["segments"]] == [
        ("seg_000000.mkv", 0, 12, True),
        ("seg_000001.mkv", 12, 30, True),
        ("seg_000002.mkv", 30, 45, True),
    ]
    joined = b"".join(decoded(directory / name) for name in names)
    assert joined == decoded(tmp_path / "one.mkv")
    assert np.frombuffer(joined, dtype="<u2").size == 45 * 3 * 540 * 960
    # Every frame as the checksums written with it say (DESIGN.md, Output, Checksums).
    for output in (directory, tmp_path / "one.mkv"):
        verify = [sys.executable, "-m", "seedvr2x", "verify", str(output)]
        checked = subprocess.run(verify, capture_output=True, text=True, check=False)
        assert checked.returncode == 0, checked.stderr[-3000:]
