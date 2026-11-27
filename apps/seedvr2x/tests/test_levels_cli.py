"""A level per shot, marked in the cut list (DESIGN.md, A level per shot, marked in the cut list,
decided on 2026-11-24), on the command line, on the CPU, the model replaced by test_cli_run.py's
stand-in: each shot of a job with marks equal frame for frame, masters and --dump-frames, to the
same shot in a job at its level alone, with the same cut list and the tiles pinned in both; the
settings, the record and the plan's lines; --plan FILE's list read back with its marks; --window
with marks, and refused where every shot is at deghost; a stop inside a marked shot and its
resume; a mark changed before any unit and after one. The list's reading is tests/test_job.py's,
its hash and the record's field core's tests/test_record.py's."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from levels import levels
from test_cli_run import (
    Steps,
    decoded,
    job,
    plan_lines,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    stream_tags_of,
    upscale,
)

from job import QUALITY, read_cuts, shot_runs
from seedvr2x import record
from seedvr2x.shots import Shot

# test_cli_run.py's job: 25 frames cut at 3 and 4, shots of 3, 1 and 21 frames, the first
# segment holding the first two shots (--min-segment 0.12), the second the third.
SHOTS = ((0, 3), (3, 4), (4, 25))
# The tiles pinned in every job compared, untiled, so that no plan of other shots tiles a pass
# otherwise (the stand-in's frames are a tile's worth at most: test_cli_run.py's LARGER).
PINNED = ("--encode-tile", "0", "--decode-tile", "0", "--min-segment", "0.12")
# A window cap where a job has windows: the stand-in's longest shot in two windows.
WINDOW = ("--window", "5")
# The frame-for-frame comparison's source and cut list, with no mark: 37 frames, shots of 6, 6
# and 25 frames, every level giving each other frames on the stand-in (up to 4 frames, maximum
# takes each frame from a run it starts, a latent of its own, as a frame alone is), the first
# segment holding the first two shots (--min-segment 0.4, 10 frames at 25 fps), the last shot's
# runs, 22 to 25 frames, 6 or 7 latents, each in two windows at --window 5, the least it takes.
LONGER_FRAMES = 37
LONGER = "6\n12\n"
LONGER_SHOTS = ((0, 6), (6, 12), (12, 37))
LONGER_PINS = ("--encode-tile", "0", "--decode-tile", "0", "--min-segment", "0.4")


def frames_of(directory: Path, frames: int = 25) -> list[str]:
    """The MD5 of each of a job's frames (25, test_cli_run.py's job) as its masters store them,
    segment after segment."""
    data = b"".join(decoded(path) for path in sorted(directory.glob("seg_*.mkv")))
    size, rest = divmod(len(data), frames)
    assert rest == 0 and size
    return [hashlib.md5(data[k : k + size]).hexdigest() for k in range(0, len(data), size)]


def dumped(directory: Path) -> dict[int, bytes]:
    """The float32 frames a job dumped (--dump-frames), by their index in the job."""
    return {
        int(path.stem.removeprefix("frame_")): np.load(path).tobytes()
        for path in sorted(directory.glob("frame_*.npy"))
    }


def windows_of(level: str) -> tuple[str, ...]:
    """--window where a job at `level` alone has windows: refused at deghost."""
    return () if level == "deghost" else WINDOW


@pytest.mark.parametrize(
    ("quality", "listed", "shot_levels"),
    [
        # maximum with a shot marked deghost, which shares its segment with a maximum shot.
        ("maximum", "6 deghost\n12\n", ("maximum", "deghost", "maximum")),
        # deghost with the first and the last shot marked maximum.
        ("deghost", "0 maximum\n6\n12 maximum\n", ("maximum", "deghost", "maximum")),
        # standard with shots marked enhanced and maximum, the first among them.
        (
            "standard",
            "0 enhanced\n6\n12 0.6 0:00:00.480 maximum\n",
            ("enhanced", "standard", "maximum"),
        ),
    ],
    ids=["maximum-deghost", "deghost-first-maximum", "standard-enhanced-maximum"],
)
def test_each_shot_is_the_same_shot_in_a_job_at_its_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    quality: str,
    listed: str,
    shot_levels: tuple[str, ...],
) -> None:
    # DESIGN.md: "Each shot's frames are those of the same shot in a job at its level alone, under
    # the same plan": a job at one level plans other shots (a frame's at deghost, the whole shots
    # elsewhere), so the shots' frames are compared, not the plans, the window and tiles pinned;
    # the last shot's runs merged from two windows each.
    from test_cli_run import source as made

    job(tmp_path, monkeypatch)
    source = made(tmp_path / "longer.mkv", LONGER_FRAMES)
    (tmp_path / "marked.txt").write_text(listed)
    (tmp_path / "longer.txt").write_text(LONGER)
    options = ("--color-correction", "split", *LONGER_PINS)
    dump = tmp_path / "dump"
    marked = ("--cuts", "marked.txt", "--quality", quality, *WINDOW, *options)
    assert upscale(tmp_path, source, "marked", *marked, "--dump-frames", str(dump)) == 0
    got, got_dumped = frames_of(tmp_path / "marked", LONGER_FRAMES), dumped(dump)
    assert sorted(got_dumped) == list(range(LONGER_FRAMES))
    content = json.loads((tmp_path / "marked" / "manifest.json").read_text())
    assert content["settings"]["quality"] == quality
    # The first segment holds the first two shots, of two levels.
    assert [(s["start"], s["end"]) for s in content["segments"]] == [(0, 12), (12, 37)]
    references: dict[str, tuple[list[str], dict[int, bytes]]] = {}
    for level in sorted(set(shot_levels)):
        name, dumps = f"at_{level}", tmp_path / f"dump_{level}"
        alone = ("--cuts", "longer.txt", "--quality", level, *windows_of(level), *options)
        assert upscale(tmp_path, source, name, *alone, "--dump-frames", str(dumps)) == 0
        references[level] = (frames_of(tmp_path / name, LONGER_FRAMES), dumped(dumps))
    # The levels give other frames on the stand-in, so that the comparison holds the marks: the
    # job is not the job at --quality alone, whose shots it shares where unmarked.
    if quality not in references:
        alone = ("--cuts", "longer.txt", "--quality", quality, *windows_of(quality), *options)
        assert upscale(tmp_path, source, f"at_{quality}", *alone) == 0
        references[quality] = (frames_of(tmp_path / f"at_{quality}", LONGER_FRAMES), {})
    for (start, end), level in zip(LONGER_SHOTS, shot_levels, strict=True):
        if level != quality:
            assert got[start:end] != references[quality][0][start:end], (start, level)
    for (start, end), level in zip(LONGER_SHOTS, shot_levels, strict=True):
        frames, floats = references[level]
        assert got[start:end] == frames[start:end], (start, level)
        for frame in range(start, end):
            assert got_dumped[frame] == floats[frame], (frame, level)
    # Each shot's runs at its level (job.shot_runs), and its windows listed but at deghost.
    for entry, (start, end), level in zip(content["shots"], LONGER_SHOTS, shot_levels, strict=True):
        assert 1 + len(entry.get("runs", [])) == len(shot_runs(Shot(start, end), QUALITY[level]))
        assert ("windows" in entry) == (level != "deghost")
    # The last shot's runs, at maximum in every case, each merged from two windows.
    last = content["shots"][2]
    assert [len(run) for run in [last["windows"], *[r["windows"] for r in last["runs"]]]] == [2] * 4


@pytest.mark.usefixtures("stand_in")
def test_the_settings_the_record_and_the_plan_of_a_job_with_marks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The marks among the settings beside the cuts; the record's cut list hash of the plain text
    # with the marks and the levels marked with their counts, --quality kept; the plan's quality
    # line naming each level with its count of shots, and its time at each --quality with this
    # job's as marked.
    levels(caplog, logging.INFO)
    source = job(tmp_path, monkeypatch)
    (tmp_path / "marked.txt").write_text("0 deghost\n3\n4 deghost\n")
    options = ("--cuts", "marked.txt", "--quality", "maximum", *WINDOW, *PINNED)
    assert upscale(tmp_path, source, "plan.mkv", *options, "--plan", "plan.txt") == 0
    lines = plan_lines(caplog)
    assert (
        "plan: quality: by the cut list's marks, maximum for 1 shot (4 runs a shot, each frame"
        " from its best run, --quality's, every shot the list doesn't mark), deghost for 2 shots"
        " (each frame on its own): 3 runs in all, each shot's at its level" in lines
    )
    [windows] = [line for line in lines if line.startswith("plan: windows: ")]
    assert windows.endswith("; the 2 shots at deghost one a frame, of its one latent"), windows
    [times] = [message for message in caplog.messages if message.startswith("time at each")]
    for level in ("standard", "enhanced", "maximum", "deghost"):
        assert f"{level} about " in times
    assert times.count("(this job's)") == 1
    assert ", as marked in the cut list about " in times and times.endswith(" (this job's)")
    caplog.clear()
    assert upscale(tmp_path, source, "out", *options) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    settings = content["settings"]
    assert settings["cuts"] == [3, 4]
    assert settings["cut_levels"] == [[0, "deghost"], [4, "deghost"]]
    assert list(settings).index("cut_levels") == list(settings).index("cuts") + 1
    tags = stream_tags_of(tmp_path / "out" / "seg_000000.mkv")
    assert tags["SEEDVR2X_QUALITY"] == "maximum"
    assert tags["SEEDVR2X_MARKED_LEVELS"] == "deghost:2"
    expected = hashlib.sha256(b"0 deghost\n3\n4 deghost\n").hexdigest()
    assert tags["SEEDVR2X_CUTS_SHA256"] == expected


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize(
    ("quality", "listed", "shots", "quality_line"),
    [
        # Every shot marked deghost under --quality maximum: deghost by the marks, not by
        # --quality, and no shot the list doesn't mark.
        (
            "maximum",
            "0 deghost\n3 deghost\n4 deghost\n",
            "plan: shots: 3, the longest of 21 frames, each frame a latent of its own (each marked"
            " deghost in the cut list)",
            "plan: quality: by the cut list's marks, deghost for 3 shots (each frame on its own):"
            " 3 runs in all, each shot's at its level",
        ),
        # Every shot deghost at --quality deghost, two by --quality and one by its mark.
        (
            "deghost",
            "3 deghost\n4\n",
            "plan: shots: 3, the longest of 21 frames, each frame a latent of its own (--quality"
            " deghost and the cut list's marks)",
            "plan: quality: by the cut list's marks, deghost for 3 shots (each frame on its own,"
            " --quality's, every shot the list doesn't mark): 3 runs in all, each shot's at its"
            " level",
        ),
        # Every shot marked, a mark equal to --quality among them: still a mark, --quality's
        # level no unmarked shot's.
        (
            "standard",
            "0 standard\n3 standard\n4 enhanced\n",
            None,
            "plan: quality: by the cut list's marks, standard for 2 shots (one run a shot),"
            " enhanced for 1 shot (2 runs a shot, each frame from its best run): 4 runs in all,"
            " each shot's at its level",
        ),
    ],
)
def test_the_plans_words_where_every_shot_is_marked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    quality: str,
    listed: str,
    shots: str | None,
    quality_line: str,
) -> None:
    levels(caplog, logging.INFO)
    source = job(tmp_path, monkeypatch)
    (tmp_path / "marked.txt").write_text(listed)
    options = ("--cuts", "marked.txt", "--quality", quality, *PINNED, "--plan", "plan.txt")
    assert upscale(tmp_path, source, "out", *options) == 0
    lines = plan_lines(caplog)
    if shots is not None:
        assert shots in lines, lines
    assert any(line.startswith(quality_line) for line in lines), lines


@pytest.mark.usefixtures("stand_in")
def test_a_list_with_no_mark_gives_the_settings_and_the_hash_of_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No mark: no key, the hash of the numbers alone; the same list marked differs by the key
    # alone, value for value.
    source = job(tmp_path, monkeypatch)
    assert upscale(tmp_path, source, "plain", "--cuts", "cuts.txt", *WINDOW, *PINNED) == 0
    (tmp_path / "marked.txt").write_text("3 maximum\n4\n")
    assert upscale(tmp_path, source, "marked", "--cuts", "marked.txt", *WINDOW, *PINNED) == 0
    plain = json.loads((tmp_path / "plain" / "manifest.json").read_text())["settings"]
    marked = json.loads((tmp_path / "marked" / "manifest.json").read_text())["settings"]
    assert "cut_levels" not in plain
    assert {key: value for key, value in marked.items() if key != "cut_levels"} == plain
    tags = stream_tags_of(tmp_path / "plain" / "seg_000000.mkv")
    assert tags["SEEDVR2X_CUTS_SHA256"] == hashlib.sha256(b"3\n4\n").hexdigest()
    assert tags["SEEDVR2X_CUTS_SHA256"] == record.cuts_sha256([3, 4])
    assert "SEEDVR2X_MARKED_LEVELS" not in tags


@pytest.mark.usefixtures("stand_in")
def test_plan_file_writes_the_marks_back_and_says_how_to_mark_a_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A list given: its marks on their cuts' lines, `<frame> <time> <level>`, the first shot's a
    # line `0 <level>`, read back as given; the header says how to mark a shot, none of its lines
    # beginning with a digit.
    source = job(tmp_path, monkeypatch)
    (tmp_path / "marked.txt").write_text("0 enhanced\n3 0.4 deghost\n4\n")
    plan = tmp_path / "plan.txt"
    options = ("--cuts", "marked.txt", *WINDOW, *PINNED, "--plan", str(plan))
    assert upscale(tmp_path, source, "out", *options) == 0
    text = plan.read_text().splitlines()
    body = [line for line in text if not line.startswith("#")]
    assert body == ["0 enhanced", "3 0:00:00.120 deghost", "4 0:00:00.160"]
    assert read_cuts(plan) == ([3, 4], {0: "enhanced", 3: "deghost"})
    header = " ".join(line.removeprefix("# ") for line in text if line.startswith("#"))
    assert (
        "A level's name (standard, enhanced, maximum, deghost) at the end of a cut's line runs the"
        " shot from that cut at that level, every other shot at --quality's; a line `0 LEVEL`"
        " marks the first shot. A mark changed is another job once a unit is made."
    ) in header
    assert "seedvr2x reads the frame number, and a level's name after it" in header
    assert not [line for line in text if line.removeprefix("# ")[:1].isdigit() and line[:1] == "#"]
    # Planned again from that file: the same marks, the plan as it was.
    assert upscale(tmp_path, source, "out", "--cuts", str(plan), *WINDOW, *PINNED) == 0
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings["cut_levels"] == [[0, "enhanced"], [3, "deghost"]]


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize("plan", [False, True], ids=["run", "plan"])
def test_window_caps_the_shots_with_windows_and_is_refused_where_none_has(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    plan: bool,
) -> None:
    # At deghost with a shot marked maximum, --window caps that shot's windows; every shot at
    # deghost, by --quality or by its mark, refuses it before anything is made.
    levels(caplog, logging.INFO)
    source = job(tmp_path, monkeypatch)
    planning = ("--plan", "plan.txt") if plan else ()
    (tmp_path / "one.txt").write_text("3\n4 maximum\n")
    options = ("--cuts", "one.txt", "--quality", "deghost", *WINDOW, *PINNED, *planning)
    assert upscale(tmp_path, source, "out", *options) == 0
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    third = content["shots"][2]
    assert third["windows"] == [[0, 4], [2, 6]]
    assert [entry.get("windows") for entry in content["shots"][:2]] == [None, None]
    caplog.clear()
    (tmp_path / "none.txt").write_text("0 deghost\n3 deghost\n4 deghost\n")
    before = sorted(path.name for path in tmp_path.iterdir())
    refused = ("--cuts", "none.txt", "--quality", "maximum", *WINDOW, *PINNED, *planning)
    assert upscale(tmp_path, source, "refused", *refused) == 1
    assert (
        "--window: every shot is at deghost, each frame one latent in the DiT, with no windows to"
        " cap; run without --window, or mark some shots at a level with windows in the cut list"
    ) in caplog.text
    assert "first pass" not in caplog.text and "plan: " not in caplog.text
    assert not (tmp_path / "refused").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_a_stop_inside_a_marked_shot_resumes_as_the_uninterrupted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # Stopped at the first frame's window of the third shot, marked deghost, its encode made,
    # after the first segment's two shots of two levels are made: the resume makes the rest, its
    # masters the uninterrupted job's.
    source = job(tmp_path, monkeypatch)
    (tmp_path / "marked.txt").write_text("3 maximum\n4 deghost\n")
    options = ("--cuts", "marked.txt", "--quality", "enhanced", *WINDOW, *PINNED)
    assert upscale(tmp_path, source, "whole", *options, "--color-correction", "split") == 0
    uninterrupted = frames_of(tmp_path / "whole")
    steps.calls.clear()
    steps.stop = "window 4:0"
    stopped(tmp_path, source, "out", *options, "--color-correction", "split")
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [segment["finished"] for segment in content["segments"]] == [True, False]
    assert content["shots"][2]["encoded"] is True
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source, "out", *options, "--color-correction", "split") == 0
    # No encode again: the third shot's 21 frames, each in a window of its own, then its decode.
    assert [call for call in steps.calls if not call.startswith("decode")] == ["window 4:0"] * 21
    assert frames_of(tmp_path / "out") == uninterrupted
    assert not (tmp_path / "out" / "resume").exists()


def test_a_stop_inside_a_segment_of_two_levels_resumes_as_the_uninterrupted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # The first segment's two shots, of 6 frames each, at enhanced and marked deghost: stopped at
    # the second's third frame's window, the first's runs made and the second's encode and two
    # frames' windows: the resume makes the rest of the segment, nothing made again (resume._kept
    # sorting the units by each run's level), its masters the uninterrupted job's.
    from test_cli_run import source as made

    job(tmp_path, monkeypatch)
    source = made(tmp_path / "longer.mkv", LONGER_FRAMES)
    (tmp_path / "marked.txt").write_text("6 deghost\n12\n")
    options = ("--cuts", "marked.txt", "--quality", "enhanced", *WINDOW, *LONGER_PINS)
    assert upscale(tmp_path, source, "whole", *options) == 0
    uninterrupted = frames_of(tmp_path / "whole", LONGER_FRAMES)
    steps.calls.clear()
    steps.stop = "window 6:0#3"
    stopped(tmp_path, source, "out", *options)
    made_first = list(steps.calls)
    assert made_first[:2] == ["encode 0", "window 0:0"] and made_first[-1] == "window 6:0"
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [(s["start"], s["end"]) for s in content["segments"]] == [(0, 12), (12, 37)]
    assert [segment["finished"] for segment in content["segments"]] == [False, False]
    first, second = content["shots"][:2]
    assert first["encoded"] is True and all(run["encoded"] for run in first["runs"])
    assert (second["encoded"], second["windows_done"]) == (True, 2)
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, source, "out", *options) == 0
    # Nothing of the first shot again, the second's four frames left, then the segment's decode.
    resumed = [call for call in steps.calls if not call.startswith(("decode", "encode 12"))]
    assert resumed[:4] == ["window 6:0"] * 4
    assert not any(call.startswith(("encode 0", "encode 6", "window 0:")) for call in resumed)
    assert frames_of(tmp_path / "out", LONGER_FRAMES) == uninterrupted
    assert not (tmp_path / "out" / "resume").exists()


def test_a_mark_changed_is_planned_again_before_a_unit_and_another_job_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DESIGN.md: "A changed mark is a change of the cut list: before a unit is made, the plan is
    # made again from the record, with no decode; after, it is another job".
    from media import source as examined

    levels(caplog, logging.INFO)
    source = job(tmp_path, monkeypatch)
    scans: list[Path] = []
    scan = examined.scan

    def counted(path: Path, *rest: Any, **options: Any) -> object:
        scans.append(path)
        return scan(path, *rest, **options)

    monkeypatch.setattr(examined, "scan", counted)
    (tmp_path / "marked.txt").write_text("3 deghost\n4\n")
    options = ("--cuts", "marked.txt", *WINDOW, *PINNED)
    assert upscale(tmp_path, source, "out", *options, "--plan", "plan.txt") == 0
    assert len(scans) == 1
    # Marked otherwise before a unit: planned again from the record, no first pass again.
    (tmp_path / "marked.txt").write_text("3 maximum\n4\n")
    caplog.clear()
    assert upscale(tmp_path, source, "out", *options, "--plan", "plan.txt") == 0
    assert len(scans) == 1
    assert 'settings.cut_levels[0][1]: "deghost" -> "maximum"' in caplog.text
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings["cut_levels"] == [[3, "maximum"]]
    # A unit made, then marked otherwise again: another job, its line naming the marks' key.
    steps.stop = "window 0:0"
    stopped(tmp_path, source, "out", *options)
    (tmp_path / "marked.txt").write_text("3\n4\n")
    caplog.clear()
    steps.stop = None
    assert upscale(tmp_path, source, "out", *options) == 1
    assert "another job than the one asked" in caplog.text
    assert 'settings.cut_levels: [[3, "maximum"]] -> null' in caplog.text
    assert len(scans) == 1


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize(
    ("quality", "listed", "free", "taken", "way"),
    [
        # The job at standard, which keeps none, its 11-frame shot marked maximum, which keeps 9
        # frames: its mark is what the user lowers, the other shots at their levels.
        (
            "standard",
            "5\n6 maximum\n",
            9,
            "its shot of 11 frames: 0.00 GiB of frames kept between its runs at its level, maximum",
            "mark its shot at a lower level in the cut list (standard none, enhanced keeps"
            " 0.00 GiB)",
        ),
        # The first shot, of 5 frames, marked maximum too, which keeps 4 frames whatever the
        # 11-frame shot's mark: standard for that shot still keeps those.
        (
            "standard",
            "0 maximum\n5\n6 maximum\n",
            9,
            "its shot of 11 frames: 0.00 GiB of frames kept between its runs at its level, maximum",
            "mark its shot at a lower level in the cut list (standard keeps 0.00 GiB, enhanced"
            " keeps 0.00 GiB)",
        ),
        # The shot that keeps the most unmarked, at --quality maximum, the 11-frame shot marked
        # standard: --quality is what the user lowers.
        (
            "maximum",
            "5\n6 standard\n",
            4,
            "its shot of 5 frames: 0.00 GiB of frames kept between its runs at --quality maximum",
            "take a lower --quality, which the shots the list marks don't follow (standard none,"
            " enhanced keeps 0.00 GiB)",
        ),
        # Unmarked at maximum, the first shot marked enhanced, which keeps 4 frames whatever
        # --quality: standard still keeps those.
        (
            "maximum",
            "0 enhanced\n5\n6\n",
            9,
            "its shot of 11 frames: 0.00 GiB of frames kept between its runs at --quality maximum",
            "take a lower --quality, which the shots the list marks don't follow (standard keeps"
            " 0.00 GiB, enhanced keeps 0.00 GiB)",
        ),
    ],
)
def test_the_disk_a_shot_keeps_is_counted_at_its_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    quality: str,
    listed: str,
    free: int,
    taken: str,
    way: str,
) -> None:
    # test_cli_runs.py's disk short of the kept frames (shots of 5, 1 and 11 frames), a byte
    # under what the shot that keeps the most keeps at its level: refused, the way out naming
    # what the user changes, its mark or --quality, each level offered counted per shot at the
    # level each shot would then take (a marked shot keeps its mark whatever --quality).
    from test_cli_run import source as made
    from test_cli_runs import KEPT_BYTES, short_disk
    from test_cli_runs import listed as written

    from seedvr2x import model

    levels(caplog, logging.INFO)
    original = made(tmp_path / "in.mkv", 17)
    cuts = written(tmp_path, "cuts.txt", listed)
    options = ("--cuts", cuts, "--quality", quality)
    with monkeypatch.context() as patch:
        short_disk(patch, free * KEPT_BYTES - 1)
        patch.setattr(model, "load_models", lambda *a: pytest.fail("the models loaded"))
        assert upscale(tmp_path, original, "out", *options) == 1
        said = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR][-1]
        assert f"{taken}; make room there" in said
        assert f"or {way}" in said
    # With the room, it runs.
    assert upscale(tmp_path, original, "out", *options) == 0


@pytest.mark.usefixtures("stand_in")
@pytest.mark.parametrize(
    ("quality", "listed", "alone", "frame_plan"),
    [
        # A shot marked deghost: frames alone timed, the plan not a frame's.
        ("maximum", "3 deghost\n4\n", True, False),
        # Every shot at deghost, one by its mark: the plan a frame's.
        ("deghost", "3\n4\n", True, True),
        ("deghost", "0 maximum\n3\n4\n", True, False),
        ("standard", "3 enhanced\n4\n", False, False),
    ],
)
def test_the_calibration_times_frames_alone_where_a_shot_takes_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality: str,
    listed: str,
    alone: bool,
    frame_plan: bool,
) -> None:
    # The calibration times the frames alone where any shot takes them, and its passes over 5 and
    # 13 frames keep their scratch in memory only where every shot does (calibrate's frame_plan).
    import calibration

    source = job(tmp_path, monkeypatch)
    (tmp_path / "marked.txt").write_text(listed)
    told: list[dict[str, Any]] = []
    calibrate = calibration.calibrate

    def recorded(*arguments: Any, **options: Any) -> Any:
        told.append(options)
        return calibrate(*arguments, **options)

    monkeypatch.setattr(calibration, "calibrate", recorded)
    window = () if frame_plan else WINDOW
    options = ("--cuts", "marked.txt", "--quality", quality, *window, *PINNED)
    assert upscale(tmp_path, source, "out", *options) == 0
    assert [(each["alone"], each["frame_plan"]) for each in told] == [(alone, frame_plan)]


def test_the_quality_line_and_the_compiled_shapes_by_each_shots_level() -> None:
    # The quality line of marked shots: each level with its count, --quality's said, the runs in
    # all, the most a shot keeps (the 21-frame shot at maximum, 16 frames of 1080p at 16 bits,
    # 199,065,600 bytes); a list of one level for every shot planned as that level.
    import planner
    from seedvr2x.passes import Plan, Tiles

    shots = [Shot(0, 3), Shot(3, 4), Shot(4, 25)]
    said = planner._quality_said(  # pyright: ignore[reportPrivateUsage]
        shots, (1080, 1920), "standard", ["enhanced", "standard", "maximum"]
    )
    assert said == (
        "quality: by the cut list's marks, standard for 1 shot (one run a shot, --quality's, every"
        " shot the list doesn't mark), enhanced for 1 shot (2 runs a shot, each frame from its best"
        " run), maximum for 1 shot (4 runs a shot, each frame from its best run): 7 runs in all,"
        " each shot's at its level; 0.19 GiB of frames on disk for the shot that keeps the most"
    )
    alone = planner._quality_said(  # pyright: ignore[reportPrivateUsage]
        shots, (1080, 1920), "deghost", ["deghost", "deghost", "standard"]
    )
    assert alone.endswith(": 3 runs in all, each shot's at its level")
    # Every shot marked, the mark of the first equal to --quality: no shot is --quality's.
    marked = planner._quality_said(  # pyright: ignore[reportPrivateUsage]
        shots, (1080, 1920), "standard", ["standard", "standard", "maximum"], 0
    )
    assert marked.startswith(
        "quality: by the cut list's marks, standard for 2 shots (one run a shot), maximum for 1"
        " shot (4 runs a shot, each frame from its best run): 6 runs in all"
    )
    # The compiled shapes of shots at their levels: each planned shot's runs (planned_runs).
    plan = Plan(5, Tiles(1024), Tiles(1024), compile_vae=True)
    for level in QUALITY:
        assert planner.compile_shapes(plan, (1080, 1920), shots, [QUALITY[level]] * 3) == (
            planner.compile_shapes(plan, (1080, 1920), shots, QUALITY[level])
        )
