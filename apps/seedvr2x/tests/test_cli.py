"""The command line starts: --version, and the installed seedvr2x command."""

import argparse
import importlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from levels import levels
from library_fixtures import grow, sparse
from posix_only import NAME
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

import main as cli
from media import ffmpeg, rate
from media.checksums import write_checksums
from media.writer import FFV1Writer, Tags


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out.startswith("seedvr2x 0.")


def test_entry_point() -> None:
    # The installed command, the script beside the interpreter.
    result = subprocess.run(
        [Path(sys.executable).with_name("seedvr2x"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("usage: seedvr2x")


def test_every_logger_is_named_for_its_layer() -> None:
    # The CLI's loggers sit under `cli.`, beside the library's `seedvr2x.`, both reaching the root
    # handler main._logging installs (DESIGN.md, Repository and apps): a CLI module's logger is
    # cli.<its import name> (main.py is cli.main, media/reader.py cli.media.reader); a library
    # module's is its own module's name, so that its lines say where they come from.
    import manifest
    import seedvr2x

    named: dict[str, str] = {}
    library = Path(seedvr2x.__file__).resolve().parent
    app = Path(cli.__file__).resolve().parent
    walked = [
        (library, library.parent, "seedvr2x"),
        *((app / p, app, "cli") for p in manifest.PACKAGES),
    ]
    modules = [(Path(app / f"{m}.py"), app, "cli") for m in manifest.MODULES]
    for directory, base, family in walked:
        modules += [(path, base, family) for path in sorted(directory.rglob("*.py"))]
    for path, base, family in modules:
        parts = path.relative_to(base).with_suffix("").parts
        if parts[-1] == "__main__" or {"__pycache__", "vendor", "transnetv2"} & set(parts):
            continue
        dotted = parts[:-1] if parts[-1] == "__init__" else parts
        module = ".".join(dotted)
        found = getattr(importlib.import_module(module), "logger", None)
        if not isinstance(found, logging.Logger):
            continue
        assert found.name == (module if family == "seedvr2x" else f"cli.{module}"), module
        named[module] = found.name
    # The modules with a logger today: a module that loses its own, or one that gains its own
    # under a wrong name, is seen here.
    assert {name for name in named.values() if name.startswith("cli.")} == {
        "cli.main",
        "cli.media.assembly",
        "cli.media.files",
        "cli.media.probe",
        "cli.media.reader",
        "cli.media.scan",
        "cli.media.source",
        "cli.media.writer",
        "cli.calibration",
        "cli.cuts",
        "cli.planner",
        "cli.run",
        "cli.scratch",
        "cli.stop",
        "cli.stills",
        "cli.timing",
    }
    assert all(name.startswith("seedvr2x.") for name in named.values() if "cli" not in name)


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


def test_seed_and_resolution_help_show_their_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --seed said what it seeds (a picture, a video's shots), its default and the range
    # job.check_seed enforces; --resolution its default, as the other options' helps show theirs
    # (next-batch item 14, wording).
    from job import ENCODE_SEED_OFFSET, SEED_LIMIT

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert "square pixels at the source's display aspect (default: 1080)" in shown
    bound = f"{SEED_LIMIT - 1 - ENCODE_SEED_OFFSET:,}"
    assert bound == "4,293,967,295"
    assert (
        "the seed of the model's noise (default: 42): a picture is seeded with it, and each shot"
        " of a video with it plus the shot's first frame; the encode with"
        f" {ENCODE_SEED_OFFSET:,} more, so it is from 0 up to {bound} (a video's, less the first"
        " frame of the source's last shot), refused otherwise, naming that bound for the source"
        in shown
    )


def test_plan_help_names_its_two_playlists(capsys: pytest.CaptureFixture[str]) -> None:
    # --plan FILE writes two playlists beside FILE when the shots are detected, under names no
    # option gives (cuts.playlist_path): said where --plan is read about, each with its name, and
    # that a cut list's plan has neither (design, 2026-10-12).
    from cuts import PLAYLISTS, playlist_path

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


def _option_help(shown: str, option: str) -> str:
    """One option's help as --help prints it, from its flag to the next option's, on one line."""
    after = shown.split(f"\n  {option} ", 1)[1]
    return " ".join(after.split("\n  -", 1)[0].split())


def test_compile_help_says_what_it_gives_and_costs_with_no_figure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --compile's help leaves the figures that drift (minutes, tile sizes, the compiled DiT's dB)
    # to the plan's line and the manual (design, 2026-10-31, baton 4 item b): its four points in
    # words.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = _option_help(capsys.readouterr().out, "--compile")
    assert not re.search(r"\d|\bdB\b|\bmin\b|\bpx\b", text)
    for words in (
        "about a tenth less time on a long job, and a short one loses (the manual says from how"
        " many frames)",
        "paid before the DiT loads, at a new job's start and at each resume (the plan's line says"
        " how long), its cache kept in the job's directory",
        "not plain code's, but the same in every process",
        "a setting, which a resume repeats",
        "only the shapes measured are compiled: a job outside them is refused, never run plain in"
        " its place",
    ):
        assert words in text


def test_quality_help_says_the_pulse_the_measure_and_the_eye_apart(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As the nodes' quality tooltip says it (impl-nodes-7; DESIGN.md, "Every fourth frame is the
    # model's best"): closer to the original, what was measured, and looks sharper, what the eye
    # sees, each said, neither from the other; no longer "softer", which put the eye's word on
    # the measure.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = _option_help(capsys.readouterr().out, "--quality")
    for words in (
        "rebuilds the last of each group closest to the original, so at standard one frame in"
        " four comes out closer to the original than the three before it, and looks sharper",
        "edges that pulse, most on line art, dark scenes and motion",
        "maximum takes every frame from a run where it is rebuilt closest",
        "2 or 4 times the GPU time, the same memory",
    ):
        assert words in text
    assert "softer" not in text


def test_accept_env_change_help_names_both_its_uses(capsys: pytest.CaptureFixture[str]) -> None:
    # The flag also takes a resume whose recorded shape no longer fits the memory free, planned
    # again for what is left (_run, planner.plan_again); its first use's words are kept.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = _option_help(capsys.readouterr().out, "--accept-env-change")
    assert "resume a job whose environment changed (the GPU, Python, torch" in text
    assert (
        "also resume a job whose recorded shape no longer fits the memory free, planned again for"
        " what is left" in text
    )


def test_main_help_names_verify(capsys: pytest.CaptureFixture[str]) -> None:
    # `seedvr2x verify` is dispatched by hand before the parser (main), so the main help says
    # of it what it checks and where its own help is.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert (
        "seedvr2x verify OUTPUT checks every frame of a job's output against the checksums"
        " written with it (seedvr2x verify --help)" in shown
    )


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
    from media import source as sources

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
        "import sys; import main as cli; status = cli.main(sys.argv[1:]);"
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

import main as cli
from media import source
from seedvr2x import weights


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
    "from seedvr2x import weights", "from seedvr2x import pull, weights"
).replace("source.first_pass = ", "pull.resolve = source.first_pass = ")


@NAME
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

import main as cli
from media import ffmpeg, fingerprint, source
from seedvr2x import weights


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
        if sys.platform == "win32":  # a POSIX shell's command there too: Windows' path in it
            quoted = shlex.quote(str(segments))
        advice = (
            " Give it the file the segments were split from, or join the segments back into one"
            " file first so that every frame keeps its place on the frame grid:\n"
            f"{rate.grid_join(quoted)}\n"
            "A plain join by ffmpeg's concat demuxer, -f concat -c copy alone, can drift off the"
            " grid, up to 1 ms a join, and is refused once a frame lies half a frame off its"
            " place\n"
        )
        assert result.stderr.endswith(advice)
        # On Windows the commands said to be a POSIX shell's first (rate.WINDOWS_SHELL).
        shell = rate.WINDOWS_SHELL if sys.platform == "win32" else ""
        assert f"with ffmpeg alone{shell}, each segment listed with its exact duration" in advice
        # The rate the shell's N and D in both commands, the join's and the remake's.
        # The directory after --, so that one whose name begins with a hyphen is no option.
        assert advice.count(f"cd -- {quoted} && N=24000 D=1001 && for f in ./*.mkv; do printf") == 2
        said = (
            f", with N and D set to their frame rate, 24000/1001 here:\ncd -- {quoted} && N=24000"
        )
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
        assert f"../joined.mkv\n{names}\nor with sptenc" in advice
        assert f"../remade.mkv\n{names}\nA plain join by" in advice
        # sptenc's join an option: seedvr2x never requires sptenc. Each command is a line of its
        # own, nothing before it and nothing after (DESIGN.md, What it prints), so that it can be
        # copied whole.
        assert (
            f"which snaps every timestamp to the grid:\nsptenc concat -- {quoted} joined.mkv\nor,"
            in advice
        )
        commands = [
            line for line in advice.splitlines() if line.startswith(("cd -- ", "sptenc concat -- "))
        ]
        assert len(commands) == 3
        assert all(line.endswith(".mkv") for line in commands)
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


# A resume's early look at the card (memory.card_state, cli._early_memory): NVML's reading replaced,
# as the tests of tests/test_cli_run.py replace it, a card too small for the recorded shape.
EARLY_READING = """
import argparse
import logging
import sys
from pathlib import Path

import main as cli
from seedvr2x import memory
import planner

logging.basicConfig(stream=sys.stderr, format="%(levelname)s %(message)s")
memory._read_all = lambda index: (4 << 30, 10 << 30)
assert memory.card_free() == 4 << 30
assert memory.card_state() == (4 << 30, 10 << 30)
recorded = {
    "plan": {
        "window": 8,
        "encode_tiles": None,
        "decode_tiles": None,
        "compile_dit": False,
        "compile_vae": False,
    },
    "shots": [{"start": 0, "end": 300}],
}
args = argparse.Namespace(
    color_correction="split",
    accept_env_change=False,
    input=Path(sys.argv[1]),
    model_dir=Path(sys.argv[2]),
    dit_model=sys.argv[3],
)
cli._early_memory(args, recorded, (1080, 1920), planner.Pins(), rehearses=False)
print(*sorted(name for name in ("torch", "torchvision") if name in sys.modules))
"""


def test_the_early_reading_of_the_card_imports_no_torch(tmp_path: Path) -> None:
    # DESIGN.md, Memory planner, A plan shapes the output a little (decided on 2026-10-28): the
    # room is read through NVML before torch is imported, no seedvr2x CUDA context made on the
    # card: a process that reads the card (memory.card_free, memory.card_state) and judges a
    # recorded shape in the reading (cli._early_memory, which warns here) has loaded no torch.
    # The warning is checked said, so that the path was run to its end. The DiT's file, whose
    # header says the figures of its format, is read there too, without torch: a synthetic 7B in
    # fp16 in --model-dir.
    from test_weights import accepted_models

    source = tmp_path / "source.mkv"
    source.write_bytes(b"x" * 1000)
    dit = accepted_models(tmp_path)[1]
    result = subprocess.run(
        [sys.executable, "-c", EARLY_READING, str(source), str(tmp_path), dit],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "does not hold the recorded plan" in result.stderr
    assert result.stdout.strip() == ""


# A run's entry as a process sees it: the value of TORCH_CPP_LOG_LEVEL at each import of torch,
# recorded by a finder in front of the import system, then the command line run on a source that
# isn't there (refused before any model), then torch imported by hand.
C10_LEVEL = """
import importlib.abc
import os
import sys

seen = []


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch":
            seen.append(os.environ.get("TORCH_CPP_LOG_LEVEL"))
        return None


sys.meta_path.insert(0, Finder())
import main as cli

print("imported", seen)
status = cli.main(sys.argv[1:])
print("status", status, "during", seen)
import torch

print("at torch", seen)
"""


@pytest.mark.parametrize(
    ("flags", "given", "expected"),
    [
        # Default: c10's warnings (the allocator's "recovered from an allocation failure") out.
        ([], None, "ERROR"),
        # --verbose leaves c10 at its own default, warnings shown.
        (["--verbose"], None, None),
        # The user's own setting wins, with or without --verbose.
        ([], "INFO", "INFO"),
        (["--verbose"], "WARNING", "WARNING"),
    ],
)
def test_torch_s_c_plus_plus_lines_are_out_of_the_default_output_but_not_under_verbose(
    tmp_path: Path, flags: list[str], given: str | None, expected: str | None
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-23, built on 2026-10-29): c10
    # reads TORCH_CPP_LOG_LEVEL as torch is imported, and the lines of its log, which no Python
    # warning carries ("[W1023 ... CUDAMallocAsyncAllocator.cpp:377] [cudaMallocAsync]
    # recovered from an allocation failure by trimming the pool and retrying"), are kept out by
    # ERROR. The command line sets it before anything imports torch, and nothing does before:
    # the finder sees the value at the first import, and no import in between.
    environment = {k: v for k, v in os.environ.items() if k != "TORCH_CPP_LOG_LEVEL"}
    if given is not None:
        environment["TORCH_CPP_LOG_LEVEL"] = given
    result = subprocess.run(
        [
            *(sys.executable, "-c", C10_LEVEL, str(tmp_path / "absent.mkv")),
            *("-o", str(tmp_path / "out.mkv"), *flags),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert f"imported {[]}\n" in result.stdout and f"during {[]}\n" in result.stdout
    assert f"at torch {[expected]}\n" in result.stdout


def test_the_line_before_the_masters_are_checked_says_tiny_sizes_and_times_as_they_are(
    tmp_path: Path,
) -> None:
    # "checking the bytes of 1 segment, 0.04 GiB, about 0 s" said nothing (2026-10-29's list from a
    # real job): 41 MiB, under a second. A size from 0.1 GiB stays in GiB, a time from a second
    # "about N s"; with no disk rate, the bytes alone.
    said = cli._checking_said  # pyright: ignore[reportPrivateUsage]
    small = tmp_path / "small.mkv"
    with sparse(small.open("wb")) as handle:
        grow(handle, 41 * 2**20)
    big = tmp_path / "big.mkv"
    with sparse(big.open("wb")) as handle:
        grow(handle, 3 * 2**30)
    rate = 100 * 2**20
    assert said([small], rate) == "checking the bytes of 1 segment, 41 MiB, under a second"
    assert said([small], None) == "checking the bytes of 1 segment, 41 MiB"
    assert said([small, big], rate) == ("checking the bytes of 2 segments, 3.04 GiB, about 31 s")


def test_a_long_read_of_the_source_is_named_before_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # source's SHA-256, read by a resume (cli._content) and on a finished job, is named before it
    # starts when it takes long: what is read and for about how long, from a timed sample of
    # the file (assembly.timed_read). Under five seconds it is no silence worth a line.
    from media import assembly

    source = tmp_path / "source.mkv"
    source.write_bytes(bytes(3 * 2**20))
    levels(caplog, logging.INFO)
    long_read = cli._long_read  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(assembly, "timed_read", lambda path: 124.0)
    assert long_read(source) == ("3 MiB", "about 2 min 4 s")
    cli._content(source)  # pyright: ignore[reportPrivateUsage]
    assert f"{source}: reading its 3 MiB to check its SHA-256, about 2 min 4 s" in caplog.messages
    [read, done] = [m for m in caplog.messages if "SHA-256" in m]
    assert read.endswith("about 2 min 4 s") and ", in " in done
    monkeypatch.setattr(assembly, "timed_read", lambda path: 4.9)
    assert long_read(source) == ("3 MiB", "")
    caplog.clear()
    cli._content(source)  # pyright: ignore[reportPrivateUsage]
    assert [m for m in caplog.messages if "to check its SHA-256" in m] == []

    def unreadable(path: Path) -> float:
        raise OSError("Input/output error")

    monkeypatch.setattr(assembly, "timed_read", unreadable)
    assert long_read(source) == ("3 MiB", "")


def test_the_help_names_pictures_png8_and_the_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    shown = " ".join(capsys.readouterr().out.split())
    assert (
        "the source: one video file, or one picture (PNG, JPEG, WebP, TIFF, BMP or a GIF of one"
        " frame), told by the file's content, never by its name" in shown
    )
    assert "png8, for a picture alone, an 8-bit PNG (a picture's png is 16-bit)" in shown
    assert "output format (default: yuv420p10le)" in shown
    assert "(default: standard)" in shown
    assert "as sptenc's -L (default: 5; 0 keeps every cut)" in shown
    assert (
        "--plan [FILE]" in shown and "for a picture, --plan alone says the picture's plan" in shown
    )


# The help where POSIX's own modules are missing, as on Windows: fcntl (the output directory's
# lock, a video job's) and resource (a video run's memory reading) are imported where a video
# needs them, never as the command starts.
POSIX_ONLY = """
import sys
sys.modules["fcntl"] = None
sys.modules["resource"] = None
import main
sys.exit(main.main(["--help"]))
"""


def test_the_help_needs_no_module_of_posix_alone(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", POSIX_ONLY],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage: seedvr2x ")
