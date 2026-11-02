"""Output segments: sptenc's minimum length merge, their names, their writer, the manifest
(DESIGN.md, Output).

The merge tests are sptenc's own (core/scenes_test.go at vmafv1 5790944), its boundaries placed on
a 25 fps grid as there: at(s) is frame 25 s, and a duration of s seconds is 25 s frames. The GPU
test needs what tests/test_regression.py needs."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from test_weights import accepted_models

from job import (
    JobError,
    OutputSegment,
    Shot,
    merge_short,
    merged_segments,
    min_segment_frames,
)
from manifest import Manifest, code_sha256, read
from media import ffmpeg
from media.ffmpeg import MediaError
from media.source import Source
from media.writer import SegmentWriter, Tags, Writer, count_packets, open_writer
from plan import Plan


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


def written_manifest(tmp_path: Path) -> Manifest:
    """A job's manifest, written to tmp_path / manifest.json: its source in.mkv there, a stand-in
    for a Source of 5 frames, in two shots and two segments."""
    source = SimpleNamespace(
        path=tmp_path / "in.mkv",
        frames=5,
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
        sha256="the content of in.mkv",
        index=None,
        idet=None,
    )
    source.path.write_bytes(b"x" * 7)
    record = Manifest(
        tmp_path / "manifest.json",
        {"seed": 42},
        {"gpu": "a GPU"},
        cast(Source, source),
        [Shot(0, 3), Shot(3, 5)],
        [[(0, 1)], [(0, 1)]],
        [OutputSegment("a", 0, 3), OutputSegment("b", 3, 5)],
        ["a.mkv", "b.mkv"],
        {"format": "gbrp16le"},
        Plan(None).record(),
    )
    record.write()
    return record


def test_manifest(tmp_path: Path) -> None:
    record = written_manifest(tmp_path)
    record.shot_encoded(0)
    record.window_done(0, 0)
    record.segment_finished(0, 1234, "a" * 64)
    content = json.loads((tmp_path / "manifest.json").read_text())
    assert content["seedvr2x_manifest"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", code_sha256()) and code_sha256() == code_sha256()
    assert content["environment"] == {"gpu": "a GPU"}
    assert [(s["finished"], s["bytes"]) for s in content["segments"]] == [
        (True, 1234),
        (False, None),
    ]
    assert [s["seed"] for s in content["shots"]] == [42, 45]
    assert [(s["encoded"], s["windows_done"]) for s in content["shots"]] == [(True, 1), (False, 0)]
    assert [(s["latents"], s["windows"]) for s in content["shots"]] == [(1, [[0, 1]])] * 2
    # The source, the only input (DESIGN.md, Input): one record, where its file is, its size and
    # content, and its frames.
    entry = content["input"]
    assert (entry["path"], entry["bytes"]) == (str((tmp_path / "in.mkv").resolve()), 7)
    assert (entry["sha256"], entry["frames"]) == ("the content of in.mkv", 5)
    assert "start" not in entry
    assert not (tmp_path / "manifest.json.partial").exists()
    with pytest.raises(ValueError, match="window 2 after 1"):
        record.window_done(0, 2)


def test_manifest_versions(tmp_path: Path) -> None:
    # Version 3 is read, and 2, older code's, so that a resume refuses its job as another job, its
    # differences listed, and verify checks its segments (test_cli_run.py's
    # test_version_2_job_refused); any other version, or none, is refused, naming it: a number as
    # it is, anything else as JSON writes it, none as none.
    path = tmp_path / "manifest.json"
    for version in (2, 3):
        path.write_text(json.dumps({"seedvr2x_manifest": version}))
        assert read(path) == {"seedvr2x_manifest": version}
    for manifest, said in (
        ({"seedvr2x_manifest": 1}, "manifest version 1"),
        ({"seedvr2x_manifest": 4}, "manifest version 4"),
        ({"seedvr2x_manifest": "3"}, 'manifest version "3"'),
        ({"seedvr2x_manifest": None}, "no manifest version"),
        ({}, "no manifest version"),
    ):
        path.write_text(json.dumps(manifest))
        with pytest.raises(JobError) as refused:
            read(path)
        assert str(refused.value) == f"{path}: {said}, where this seedvr2x writes 3"


REPOSITORY = Path(__file__).resolve().parents[3]
# manifest.py as of 4c13da5, the last code to write version 2, its input already the
# source's record; it reads its own version only.
VERSION_2_BLOB = "4187c45028f04ae2509d83483d7a1850dc802686"


# A process on copies of the code alone (python -I: no PYTHON* variable, no working directory on
# its path), the copies first on sys.path: it says where it found both and the hash it computes.
HASH_THERE = """
import sys
from pathlib import Path

sys.path[:0] = sys.argv[2:]
import manifest
import seedvr2x

copies = Path(sys.argv[1])
assert Path(manifest.__file__).is_relative_to(copies), manifest.__file__
assert Path(seedvr2x.__file__).is_relative_to(copies), seedvr2x.__file__
print(manifest.code_sha256())
"""


def copy_the_code(library: Path, app: Path) -> None:
    """The library's package into library/ and the CLI's own files (MODULES, PACKAGES) into app/,
    which may be one directory, as site-packages is for a non-editable install."""
    import manifest
    import seedvr2x

    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(Path(seedvr2x.__file__).parent, library / "seedvr2x", ignore=ignore)
    root = Path(manifest.__file__).parent
    app.mkdir(parents=True, exist_ok=True)
    for module in manifest.MODULES:
        shutil.copy(root / f"{module}.py", app / f"{module}.py")
    for package in manifest.PACKAGES:
        shutil.copytree(root / package, app / package, ignore=ignore)


def hash_there(copies: Path, *path: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-I", "-c", HASH_THERE, str(copies), *map(str, path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_code_hash_holds_wherever_the_package_sits(tmp_path: Path) -> None:
    # settings.code hashes the library's files by their paths inside its package and the CLI's by
    # theirs in the app's root, so that both moved (DESIGN.md, Repository and apps: seedvr2x/src/
    # to core/, then the split) leave a job's record as it was and the job resumes: the same
    # files elsewhere give the same hash. Each in a place of its own, as in a source tree, and
    # both in one directory, as a non-editable install lays them in site-packages.
    apart = tmp_path / "apart"
    copy_the_code(apart / "core", apart / "apps" / "seedvr2x")
    assert hash_there(apart, apart / "apps" / "seedvr2x", apart / "core") == code_sha256()
    together = tmp_path / "site-packages"
    copy_the_code(together, together)
    assert hash_there(together, together) == code_sha256()


def test_older_code_refuses_version_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Older code refuses a manifest of this code's for its version, naming both, rather than fail
    # on its shape (manifest.VERSION). Skipped without git or that blob, as on a copy of core/
    # alone.
    why = "needs git and the repository's history: manifest.py as of 4c13da5"
    if shutil.which("git") is None or not (REPOSITORY / ".git").exists():
        pytest.skip(why)
    shown = subprocess.run(
        ["git", "-C", str(REPOSITORY), "cat-file", "-p", VERSION_2_BLOB],
        capture_output=True,
        check=False,
    )
    if shown.returncode:
        pytest.skip(why)
    # That commit imports the modules by the names of its day, seedvr2x.media.* and
    # seedvr2x.runtime.job: the same modules are the CLI's top-level ones now.
    import job
    import media.files
    import media.source

    for old, new in {
        "seedvr2x.media": media,
        "seedvr2x.media.files": media.files,
        "seedvr2x.media.source": media.source,
        "seedvr2x.runtime.job": job,
    }.items():
        monkeypatch.setitem(sys.modules, old, new)
    older = tmp_path / "manifest_version_2.py"
    older.write_bytes(shown.stdout)
    spec = importlib.util.spec_from_file_location("manifest_version_2", older)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VERSION == 2
    written_manifest(tmp_path)
    with pytest.raises(JobError, match=r"manifest version 3, where this seedvr2x writes 2$"):
        module.read(tmp_path / "manifest.json")


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1")
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


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1")
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["-o", "out.mkv", "--format", "png"], "PNG output goes to a directory"),
        (["-o", "file.txt"], "a file; the output segments need a directory"),
        (["-o", "out", "--min-segment", "-1"], "negative"),
        (["-o", "out.mp4"], "a video file name"),
        (["-o", "dir.mkv"], "a directory; a .mkv path names the finished file"),
        (["-o", "full"], "not empty, and no manifest.json to resume from"),
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
    # An output directory holding another run's files, and no manifest: what sptenc reads must be
    # the job's own (DESIGN.md, Output).
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "old.mkv").write_bytes(b"")
    # Models the check accepts, which no pin refuses: none of these refusals is theirs, each
    # coming before them, the seed's too (cli._run).
    models = accepted_models(tmp_path)
    args = [str(source), "--model-dir", ".", *models, *options]
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; import main as cli; status = cli.main(sys.argv[1:]);"
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


# The installed command, the script beside the interpreter.
COMMAND = str(Path(sys.executable).with_name("seedvr2x"))
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
            [COMMAND, str(source), "-o", output, *common, *options],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )
        assert run.returncode == 0, run.stderr[-3000:]
    directory = tmp_path / "segments"
    names = [f"seg_{i:06d}.mkv" for i in range(3)]
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        [*names, "checksums", "frame_index.bin", "manifest.json"]
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
        verify = [COMMAND, "verify", str(output)]
        checked = subprocess.run(verify, capture_output=True, text=True, check=False)
        assert checked.returncode == 0, checked.stderr[-3000:]
