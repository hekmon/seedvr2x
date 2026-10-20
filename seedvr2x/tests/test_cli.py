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
    hevc,
    record_box,
    x265_file,
)

from seedvr2x import cli
from seedvr2x.media import ffmpeg, rate
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
    # numz's padding, and the tail of its padding of a shot's end, are for seedvr2x's tests only
    # (cli.NUMZ_PADDING, cli.NUMZ_TAIL): never in the help.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = capsys.readouterr().out
    assert cli.NUMZ_PADDING not in shown and "padding" not in shown.lower()
    assert cli.NUMZ_TAIL not in shown and "mirror" not in shown.lower()


def test_cuts_help_says_how_its_frames_are_counted(capsys: pytest.CaptureFixture[str]) -> None:
    # A cut list's frame numbers are places (media/grid.py): said where --cuts is read about, a
    # list numbered by a tool that counts decoded frames being off by the holes before each cut.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert (
        "as --plan writes it; frame numbers count the source's frames as seedvr2x counts them, a"
        " missing frame's filled place among them; in place of the shot detector" in shown
    )


def test_plan_help_names_its_two_playlists(capsys: pytest.CaptureFixture[str]) -> None:
    # --plan FILE writes two playlists beside FILE when the shots are detected, under names no
    # option gives (cuts.playlist_path): said where --plan is read about, each with its name, and
    # that a cut list's plan has neither (design, 2026-10-12).
    from seedvr2x.runtime.cuts import PLAYLISTS, playlist_path

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert (
        "commented with their probabilities and times, and two playlists for VLC beside it when"
        " the shots are detected, FILE.cuts.m3u8 of the detected cuts and FILE.possible.m3u8 of"
        " the possible ones (none with --cuts); and the counts printed." in shown
    )
    # The names said are those written.
    for which in PLAYLISTS:
        assert playlist_path(Path("FILE"), which).name in shown


def test_too_small_to_pad_refused_before_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The target is checked with the source, before the first pass and the model files: at
    # --resolution 16, 64x48 is resized to 21x16, whose 16 rows would take 16 reflected; 48x64 to
    # 16x21, whose 21 rows take 11, and 16 columns none, so the model files are checked next.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
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


def test_rate_option() -> None:
    # --frame-rate: N/D or a whole number, exact and positive; a decimal refused, saying how to
    # write it.
    assert cli._rate("24000/1001") == Fraction(24000, 1001)
    assert cli._rate("25") == cli._rate("50/2") == Fraction(25)
    for text in ("23.976", "0", "24/0", "0/1001", "-24", "24:1", "24/1/1", ""):
        with pytest.raises(argparse.ArgumentTypeError, match="write it N/D or as a whole number"):
            cli._rate(text)


def test_refused_source_stops_before_torch(tmp_path: Path) -> None:
    # The build and the source are checked before the GPU is touched: torch isn't even imported.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
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
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
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


# EARLY, the model files' fetch exiting 4 as well.
BEFORE_THE_MODEL_FILES = EARLY.replace(
    "from seedvr2x.runtime import weights", "from seedvr2x.runtime import pull, weights"
).replace("source.first_pass = ", "pull.resolve = source.first_pass = ")


def test_a_line_break_in_the_name_refused_before_the_model_files(tmp_path: Path) -> None:
    # A name ffmpeg prints as it is, a line of the name's own among those the first pass reads
    # (media/scan.py, MAPPING): refused with what the source declares, before the model files are
    # fetched, checked and hashed, where the first pass refused it after them, a 17 GB pull and a
    # 16.5 GB hash later; nothing written. The file is there, a video like another.
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
    assert "pull.resolve = source.first_pass = weights.check_models = ran" in BEFORE_THE_MODEL_FILES
    source = tmp_path / "a\n[info] Stream mapping:.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "3", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    files = sorted(tmp_path.rglob("*"))
    args = [str(source), "-o", str(tmp_path / "out.mkv"), "--model-dir", str(tmp_path / "models")]
    result = subprocess.run(
        [sys.executable, "-c", BEFORE_THE_MODEL_FILES, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert f"{str(source)!r}: a line break in its name, which ffmpeg prints" in result.stderr
    assert sorted(tmp_path.rglob("*")) == files


def test_contradicted_tags_refused_before_the_first_pass(tmp_path: Path) -> None:
    # Colour tags its first frame contradicts, a declared refusal like the others (DESIGN.md,
    # Input): before the first pass, the model files' check and torch, whatever the output, and
    # nothing written. A container declaring bt470bg and full range over a BT.709 limited-range
    # bitstream (test_probe.py, test_contradicted_tags_refused).
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
    if not has_x265():
        pytest.skip("needs ffmpeg with libx265")
    tags = ("-colorspace:v", "bt470bg", "-color_range:v", "pc")
    source = contradicted_file(tmp_path / "in.mkv", *tags)
    refusal = contradiction(
        "matrix bt470bg against bt709 on its first frame, range pc (full) against tv (limited)",
        "-colorspace:v bt709 -color_range:v tv",
        True,
        hevc("matrix_coefficients=5", "video_full_range_flag=1"),
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
    # anything is done, the build's check included, saying how to join them on the frame grid, the
    # advice whole, the one a drifted join's refusal gives (rate.grid_join; what its commands give
    # is tests/test_rate.py's): by ffmpeg alone first, then by sptenc, then the remake where no
    # stream copy holds; nothing is written, neither output, nor an output directory. The
    # directory's name needs quoting for a shell, a space and an apostrophe, so that the commands
    # the message gives are checked quoted; the rate, which the refusal can't know, an example said
    # to be one before each command; and the plain join, the advice until 2026-10-11, said to be
    # refused once it has drifted.
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
        # The path as shlex.quote gives it, one word for a shell, its apostrophe written '"'"'.
        quoted = f"'{tmp_path}/the film'\"'\"'s segments'"
        advice = (
            " Give it the file the segments were split from, or join the segments back into one"
            " file first so that every frame keeps its place on the frame grid,"
            f" {rate.grid_join(quoted)}. A plain join by ffmpeg's concat demuxer, -f concat -c"
            " copy alone, can drift off the grid, up to 1 ms a join, and is refused once a frame"
            " lies half a frame off its place\n"
        )
        assert result.stderr.endswith(advice)
        assert "with ffmpeg alone, each segment listed with its exact duration" in advice
        # The rate the shell's N and D in both commands, the join's and the remake's.
        # The directory after --, so that one whose name begins with a hyphen is no option.
        assert advice.count(f"cd -- {quoted} && N=24000 D=1001 && for f in ./*.mkv; do printf") == 2
        said = f", with N and D set to their frame rate, 24000/1001 here: cd -- {quoted} && N=24000"
        assert advice.count(said) == 2
        assert "ffmpeg -f concat -safe 0 -r $N/$D -i list.txt -map 0:v:0" in advice
        # Each list checked for an apostrophe in a name before ffmpeg reads it.
        assert advice.count("""done > list.txt && ! grep -q "'.*'.*'" list.txt && ffmpeg""") == 2
        assert "-r 24000/1001" not in advice and "echo" not in advice
        # What both commands ask of the segments' names, listed by the shell's ./*.mkv.
        names = (
            "(./*.mkv the segments' names, taken in name order, which must be the order they"
            " play in: numbers with leading zeros, as seg_000012.mkv, seg_10.mkv coming before"
            " seg_2.mkv otherwise; check list.txt, or write it by hand; a name must hold no"
            " apostrophe: the command stops on one, before ffmpeg, and makes no file)"
        )
        assert advice.count(names) == 2
        assert f"../joined.mkv {names}, or with sptenc" in advice
        assert f"../remade.mkv {names}. A plain join by" in advice
        # sptenc's join an option: seedvr2x never requires sptenc.
        assert f"sptenc concat -- {quoted} joined.mkv, which snaps every timestamp" in advice
        assert "-c copy joined.mkv" not in result.stderr and "drift" not in rate.grid_join()
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
        pytest.skip("needs ffmpeg 7.1 or later with zscale, idet and ffv1")
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
