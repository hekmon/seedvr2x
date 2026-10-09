"""The command line starts: --version, and python -m seedvr2x."""

import argparse
import json
import logging
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_probe import (
    BT2020,
    SDR_TAGS,
    contradicted_file,
    contradiction,
    dolby_vision_file,
    has_x265,
    record_box,
    x265_file,
)

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.media.checksums import write_checksums
from seedvr2x.media.writer import FFV1Writer, Tags


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("seedvr2x 0.")


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "seedvr2x", "--help"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith("usage: seedvr2x")


def test_numz_padding_not_an_option(capsys: pytest.CaptureFixture[str]) -> None:
    # numz's padding is for seedvr2x's tests only (cli.NUMZ_PADDING): never in the help.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = capsys.readouterr().out
    assert cli.NUMZ_PADDING not in shown and "padding" not in shown.lower()


def test_too_small_to_pad_refused_before_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The target is checked with the source, before the first pass and the model files: at
    # --resolution 16, 64x48 is resized to 21x16, whose 16 rows would take 16 reflected; 48x64 to
    # 16x21, whose 21 rows take 11, and 16 columns none, so the model files are checked next.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    from seedvr2x.media import source as sources

    def first_pass(*args: object, **kwargs: object) -> None:
        raise AssertionError("the first pass ran")

    monkeypatch.setattr(sources, "first_pass", first_pass)
    for size, said in (
        ("64x48", "--resolution 16: the frames resized to 21x16 are too small for the padding"),
        ("48x64", "no such model file"),
    ):
        source = tmp_path / f"{size}.mkv"
        subprocess.run(
            [
                *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=25:d=0.2"),
                *("-c:v", "ffv1", "-pix_fmt", "yuv420p", str(source)),
            ],
            check=True,
        )
        caplog.clear()
        args = [str(source), "-o", str(tmp_path / "out.mkv"), "--model-dir", str(tmp_path)]
        assert cli.main([*args, "--resolution", "16"]) == 1
        assert said in caplog.text
        assert ("too small" in caplog.text) == (size == "64x48")


def test_ratio_option() -> None:
    assert cli._ratio("32:27") == Fraction(32, 27)
    assert cli._ratio("4/3") == Fraction(4, 3)
    for text in ("0:1", "16:0", "1.5", "-4:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._ratio(text)


def test_refused_source_stops_before_torch(tmp_path: Path) -> None:
    # The build and the source are checked before the GPU is touched: torch isn't even imported.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    source = tmp_path / "tff.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2"),
            *("-vf", "setfield=tff", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    args = [str(source), "-o", str(tmp_path / "out.mkv"), "--model-dir", ".", "--dit-model", "x"]
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " sys.exit(3 if 'torch' in sys.modules else status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1, result.stderr
    assert "interlaced (field order" in result.stderr


# The command line, exiting 4 should the first pass or the model files' check run, and 3 once torch
# is imported: a failed assert would exit 1, as a refusal does.
EARLY = """
import sys

from seedvr2x import cli
from seedvr2x.media import source
from seedvr2x.runtime import weights


def ran(*args, **kwargs):
    sys.exit(4)


source.first_pass = weights.check_models = ran
status = cli.main(sys.argv[1:])
sys.exit(3 if "torch" in sys.modules else status)
"""


def hdr_file(path: Path, transfer: str) -> Path:
    """A tiny FFV1 file in BT.2020 with that transfer, the frames tagged (test_decode.py)."""
    tags = f"colorspace=bt2020nc:color_primaries=bt2020:color_trc={transfer}:range=tv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2"),
            *("-vf", f"setparams={tags},format=yuv420p10le", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        ("smpte2084", "in.mkv: HDR, transfer smpte2084 (PQ): not supported: the model was trained"),
        ("arib-std-b67", "in.mkv: HDR, transfer arib-std-b67 (HLG): not supported: the model"),
        ("dolby vision", "in.mp4: Dolby Vision profile 5, its base layer Dolby's own"),
        (
            "first frame",
            "in.mkv: 2 reasons: (1) HDR, transfer smpte2084 (PQ) on its first frame, where the",
        ),
    ],
)
def test_hdr_refused_before_the_first_pass(tmp_path: Path, kind: str, said: str) -> None:
    # With the source's other declared refusals (DESIGN.md, Not in the first version): before the
    # first pass, the model files' check and torch, so no model loaded, and nothing written. PQ on
    # the first frame alone, its container saying BT.709, is read by the same probe
    # (media/probe.py); that container contradicts the bitstream's BT.2020 matrix and primaries
    # too, a second reason (test_probe.py, test_hdr_on_the_first_frame).
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    if kind == "first frame" and not has_x265():
        pytest.skip("needs ffmpeg with libx265")
    if kind == "dolby vision":
        source = dolby_vision_file(tmp_path / "in.mp4", record_box(5, 0))
    elif kind == "first frame":
        source = x265_file(tmp_path / "in.mkv", SDR_TAGS, f"{BT2020}:transfer=smpte2084")
    else:
        source = hdr_file(tmp_path / "in.mkv", kind)
    output = tmp_path / "out.mkv"
    files = sorted(tmp_path.rglob("*"))
    args = [str(source), "-o", str(output), "--model-dir", str(tmp_path / "models")]
    result = subprocess.run(
        [sys.executable, "-c", EARLY, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1, result.stderr
    assert said in result.stderr
    assert sorted(tmp_path.rglob("*")) == files


def test_contradicted_tags_refused_before_the_first_pass(tmp_path: Path) -> None:
    # Colour tags its first frame contradicts, a declared refusal like the others (DESIGN.md,
    # Input): before the first pass, the model files' check and torch, whatever the output, and
    # nothing written. A container declaring bt470bg and full range over a BT.709 limited-range
    # bitstream (test_probe.py, test_contradicted_tags_refused).
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    if not has_x265():
        pytest.skip("needs ffmpeg with libx265")
    tags = ("-colorspace:v", "bt470bg", "-color_range:v", "pc")
    source = contradicted_file(tmp_path / "in.mkv", *tags)
    refusal = contradiction(
        "matrix bt470bg against bt709 on its first frame, range pc (full) against tv (limited)",
        "-colorspace:v bt709 -color_range:v tv",
        matrix=True,
    )
    files = sorted(tmp_path.rglob("*"))
    for output in ("out.mkv", "out"):
        args = [str(source), "-o", str(tmp_path / output), "--model-dir", str(tmp_path / "models")]
        result = subprocess.run(
            [sys.executable, "-c", EARLY, *args], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1, result.stderr
        assert f"{source}: {refusal}\n" in result.stderr
        assert sorted(tmp_path.rglob("*")) == files


# EARLY, the build's check and its conversions' fingerprint exiting 4 as well.
FIRST = """
import sys

from seedvr2x import cli
from seedvr2x.media import ffmpeg, fingerprint, source
from seedvr2x.runtime import weights


def ran(*args, **kwargs):
    sys.exit(4)


ffmpeg.check = fingerprint.fingerprint = source.first_pass = weights.check_models = ran
status = cli.main(sys.argv[1:])
sys.exit(3 if "torch" in sys.modules else status)
"""


def test_directory_refused_first(tmp_path: Path) -> None:
    # The source is the only input (DESIGN.md, Input): a directory of segments is refused before
    # anything is done, the build's check included, saying how to join them; nothing is written,
    # neither output, nor an output directory. The directory's name needs quoting for a shell, a
    # space and an apostrophe, so that the sptenc command the message gives is checked quoted.
    segments = tmp_path / "the film's segments"
    segments.mkdir()
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        (segments / name).write_bytes(b"")
    files = sorted(tmp_path.rglob("*"))
    for output in ("out", "out.mkv"):
        args = [str(segments), "-o", str(tmp_path / output), "--model-dir", str(tmp_path)]
        result = subprocess.run(
            [sys.executable, "-c", FIRST, *args], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1, result.stderr
        said = f"{segments}: a directory; seedvr2x takes one video file, the source, so that"
        assert said in result.stderr
        assert "its own detection or a cut list (--cuts), and every output segment" in result.stderr
        joined = "ffmpeg -f concat -i list.txt -c copy joined.mkv, list.txt beside them naming"
        assert joined in result.stderr
        # sptenc's join, on the frame grid, an option: seedvr2x never requires sptenc. The path as
        # shlex.quote gives it, one word for a shell, its apostrophe written '"'"'.
        quoted = f"'{tmp_path}/the film'\"'\"'s segments'"
        sptenc = f"joined by sptenc, on the grid: sptenc concat {quoted} joined.mkv"
        assert sptenc in result.stderr
        assert sorted(tmp_path.rglob("*")) == files


def test_verify_reads_no_frame_to_probe(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # verify decodes every frame of a master itself, so its probe reads none (media/probe.py): a
    # master whose first frames don't decode gets verify's report, not a source's refusal for its
    # first frame. 260 frames at 60 fps, the first 251 zeroed, more than the 250 packets of
    # probe's frame read: ffprobe still finds their pixel format, from the 252nd, in the 5 s it
    # analyses (libavformat/demux.c:2639-2641 at n9.0.2).
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg with zscale, scdet and ffv1")
    master = tmp_path / "one.mkv"
    with FFV1Writer(master, "gbrp16le", 64, 48, Fraction(60), Tags()) as writer:
        writer.write(np.zeros((260, 48, 64, 3), np.float32))
    write_checksums(master.with_name(f"{master.name}.crc32"), writer.checksums)
    assert cli.main(["verify", str(master)]) == 0
    listed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=pos,size", "-of", "json", str(master)),
        ],
        capture_output=True,
        check=True,
    )
    data = bytearray(master.read_bytes())
    for packet in json.loads(listed.stdout)["packets"][:251]:
        start, size = int(packet["pos"]) + 4, int(packet["size"])  # after the block's header
        data[start : start + size] = bytes(size)
    master.write_bytes(data)
    caplog.set_level(logging.INFO)
    assert cli.main(["verify", str(master)]) == 1
    assert f"{master}: 9 frames, where 260 were written" in caplog.text
    assert "no frame decoded" not in caplog.text
