"""A run refused after its first pass keeps it as a plan, and a plan takes any setting its first
pass doesn't read (DESIGN.md, Pause and resume, A plan, decided on 2026-10-21), on the CPU, the
model replaced by test_cli_run.py's stand-in: the record written as soon as the pass accepts the
source, before the planner, the disk check and the model load can refuse the job; each refusal
saying the pass is kept; the same command, or one whose settings differ in what the pass doesn't
read, taking it up with no second first pass; the frame rate and a unit made still another job.
And the disk check at every level (cli._room_on_disk): a shot's kept frames and its tiled pass's
scratch file, where the plan spills one, both counted."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from test_cli_run import (
    CUT,
    JOB,
    PADDED,
    UNINTERRUPTED,
    Steps,
    first_passes,
    free_memory,
    indexes,
    job,
    scripted,  # noqa: F401
    source,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    upscale,
)
from test_cli_runs import KEPT_BYTES, listed, short_disk

from seedvr2x import cli
from seedvr2x.runtime import model, planner
from seedvr2x.runtime.blockswap import HostMemoryError
from seedvr2x.runtime.plan import Plan, Tiles

# What every refusal after the first pass adds, the job's directory named (cli._first_kept).
KEPT = (
    "; the first pass is kept in {directory}: the same command takes it up, or one changing what is"
    " named here, without decoding the source again"
)


def errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]


def recorded(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "manifest.json").read_text())


def too_little_memory(patch: pytest.MonkeyPatch) -> None:
    """The stand-in's card short of the untiled encode's floor by a byte: the planner refuses
    every plan (test_cli_run.test_size_and_host_memory_refused)."""
    encode = planner.encode_peak(PADDED, None, 0) + planner.VAE_ROOM
    free_memory(patch, encode + planner.MARGIN - 1)


def test_a_run_refused_by_the_planner_leaves_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Refused by the planner, the run leaves its first pass's record, the frame index and the
    # manifest with no plan's record, no unit; its refusal says the pass is kept. The same
    # command, the memory back, plans it from the record, no second first pass, and runs it to
    # the uninterrupted run's frames.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    with monkeypatch.context() as patch:
        too_little_memory(patch)
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
    [said] = errors(caplog)
    assert said.endswith(KEPT.format(directory=out)), said
    assert "takes 4.62 GiB of free VRAM" in said
    assert sorted(path.name for path in out.iterdir()) == ["frame_index.bin", "manifest.json"]
    content = recorded(out)
    assert content["plan"] is None
    assert not any(shot["encoded"] for shot in content["shots"])
    assert steps.calls == []
    scans = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert scans == []
    assert "the first pass as recorded" in caplog.text
    [replanned] = [line for line in caplog.messages if "re-planned" in line]
    assert replanned == (
        f"{out}: re-planned, its job having made no unit yet: its plan made, none recorded; 0"
        " leftovers discarded"
    )
    assert steps.calls == UNINTERRUPTED
    assert recorded(out)["plan"] is not None
    assert indexes(out / "seg_000000.mkv") + indexes(out / "seg_000001.mkv") == list(range(25))


def test_a_refused_plan_leaves_one_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --plan refused by the planner keeps the first pass as a run does, and writes its cut list
    # all the same, the refusal saying where (DESIGN.md, Memory planner, decided on 2026-10-21):
    # --plan again, the memory back, plans it from the record and writes the same list.
    source_path = job(tmp_path, monkeypatch)
    out, listing = tmp_path / "out", tmp_path / "plan.txt"
    with monkeypatch.context() as patch:
        too_little_memory(patch)
        assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(listing)) == 1
    [said] = errors(caplog)
    assert said.endswith(
        f"; its cut list is in {listing} all the same" + KEPT.format(directory=out)
    ), said
    assert said.startswith("an output of 128x96 takes"), said
    assert recorded(out)["plan"] is None and listing.is_file()
    refused_list = listing.read_text()
    listing.unlink()
    scans = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(listing)) == 0
    assert scans == [] and listing.read_text() == refused_list
    assert recorded(out)["plan"] is not None
    assert steps.calls == []


def test_a_refused_plan_writes_its_playlists_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Its shots detected: the two playlists beside the list, as a plan that runs writes them;
    # and a refusal before the shots are known writes nothing. A refusal by the GPU's memory
    # itself, as the free memory is read, says it too.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    listing = tmp_path / "plan.txt"
    with monkeypatch.context() as patch:
        too_little_memory(patch)
        assert upscale(tmp_path, source_path, "out", *CUT, "--plan", str(listing)) == 1
    [said] = errors(caplog)
    assert f"; its cut list is in {listing} all the same" in said
    assert listing.is_file() and sorted(path.name for path in tmp_path.glob("plan*")) == [
        "plan.txt",
        "plan.txt.cuts.m3u8",
        "plan.txt.possible.m3u8",
    ]
    listed = [message for message in caplog.messages if "the cut list in" in message]
    assert listed == [
        f"3 cuts at 0.3 (4 shots), 2 possible cuts from 0.1 up to 0.3: the cut list in {listing}"
    ]
    for path in tmp_path.glob("plan*"):
        path.unlink()
    caplog.clear()

    def refused(device: object) -> int:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB.")

    with monkeypatch.context() as patch:
        patch.setattr(cli, "_free_memory", refused)
        assert upscale(tmp_path, source_path, "again", *CUT, "--plan", str(listing)) == 1
    [said] = errors(caplog)
    assert ", before the models loaded: nothing was run; its cut list is in" in said
    assert listing.is_file()
    for path in tmp_path.glob("plan*"):
        path.unlink()
    caplog.clear()
    # Refused before its first pass, by a seed out of range: nothing written.
    assert upscale(tmp_path, source_path, "seed", *CUT, "--seed", "-1", "--plan", str(listing)) != 0
    assert not any(tmp_path.glob("plan*"))


def test_a_run_refused_at_the_model_load_leaves_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Refused as its models load, the host short of memory for the pinned copies: the record is
    # the plan's, written with it before the load, and the refusal says the pass is kept.
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"

    def short(*arguments: Any, **options: Any) -> Any:
        raise HostMemoryError("the pinned copies take 15.81 GiB of system memory, 2.00 available")

    with monkeypatch.context() as patch:
        patch.setattr(model, "load_models", short)
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
    [said] = errors(caplog)
    assert said == (
        "the pinned copies take 15.81 GiB of system memory, 2.00 available"
        + KEPT.format(directory=out)
    )
    assert recorded(out)["plan"] is not None
    scans = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB) == 0
    assert scans == [] and steps.calls == UNINTERRUPTED


def test_a_run_refused_before_its_first_pass_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The first pass refusing the source, nothing is kept: no directory made, nothing said kept.
    job(tmp_path, monkeypatch)
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a video")
    assert upscale(tmp_path, broken, "out", *JOB) == 1
    assert not (tmp_path / "out").exists()
    assert not any("the first pass is kept" in line for line in errors(caplog))


# Settings the first pass doesn't read, each changed from JOB's plan (resume.FIRST_PASS_SETTINGS),
# and the field of the manifest's settings that records it, with its new value.
TAKEN = {
    "resolution": (("--resolution", "64"), "resolution", 64),
    "seed": (("--seed", "7"), "seed", 7),
    "quality": (("--quality", "enhanced"), "quality", "enhanced"),
    "window": (("--window", "6"), "window", 6),
    "decode-tile": (("--decode-tile", "0"), "decode_tile", 0),
    "min-segment": (("--min-segment", "0.2"), "min_segment", "1/5"),
    "format": (("--format", "yuv420p10le"), "format", "yuv420p10le"),
    "color-correction": (("--color-correction", "split"), "color_correction", "split"),
    "input-matrix": (("--input-matrix", "bt709"), "input_matrix", "bt709"),
    "input-sar": (("--input-sar", "2:1"), "input_sar", "2"),
    "cuts": (("--cuts", "other.txt"), "cuts", [5]),
}


@pytest.mark.parametrize("setting", list(TAKEN))
def test_a_plan_takes_what_its_first_pass_doesn_t_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    setting: str,
) -> None:
    # A plan asked with another setting its first pass doesn't read is planned again from its
    # record, no second first pass, its manifest rewritten with the new setting at once; the job
    # then runs whole.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    (tmp_path / "other.txt").write_text("5\n")
    out = tmp_path / "out"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    options, field, value = TAKEN[setting]
    asked = dict(zip(JOB[::2], JOB[1::2], strict=True))
    asked.update(zip(options[::2], options[1::2], strict=True))
    scans = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *(x for pair in asked.items() for x in pair)) == 0
    assert scans == []
    [replanned] = [line for line in caplog.messages if "re-planned" in line]
    assert f"settings.{field}: " in replanned, replanned
    assert recorded(out)["settings"][field] == value
    assert steps.calls and all(
        call.split()[0] in ("encode", "window", "decode") for call in steps.calls
    )


def test_a_plan_at_another_frame_rate_is_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # --frame-rate is read by the first pass, which judges the source's timing by it: another
    # rate is another job, refused before any first pass, the plan untouched; the code likewise
    # (resume.FIRST_PASS_SETTINGS).
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    before = (out / "manifest.json").read_bytes()
    scans = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "out", *JOB, "--frame-rate", "30") == 1
    [said] = errors(caplog)
    assert "another job than the one asked" in said and "settings.frame_rate: null -> " in said
    assert scans == [] and (out / "manifest.json").read_bytes() == before
    assert steps.calls == []


def test_a_resume_at_another_level_with_a_unit_made_is_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A job that has made a unit at standard, asked at maximum on a disk that can't hold
    # maximum's frames: another job, refused before its first pass whatever its disk, the disk
    # not judged; its record untouched.
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    steps.stop = "window 3:0"
    assert upscale(tmp_path, source_path, "out", *JOB) == 130
    before = (out / "manifest.json").read_bytes()
    scans = first_passes(monkeypatch)
    caplog.clear()
    with monkeypatch.context() as patch:
        short_disk(patch, 0)
        assert upscale(tmp_path, source_path, "out", *JOB, "--quality", "maximum") == 1
    [said] = errors(caplog)
    assert 'settings.quality: "standard" -> "maximum"' in said
    assert "These settings change only while it is a plan" in said
    assert "free on its disk" not in said
    assert scans == [] and (out / "manifest.json").read_bytes() == before


def test_a_plan_at_another_level_is_planned_again_then_judged_there(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan made at standard, asked at maximum: planned again from its record, no second first
    # pass, then its disk judged at maximum, short of its frames by a byte: refused, the first
    # pass kept; with the room, it runs.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    original = source(tmp_path / "in.mkv", 17)
    monkeypatch.chdir(tmp_path)
    cuts = listed(tmp_path, "cuts.txt", "5\n6\n")
    out = tmp_path / "out"
    assert upscale(tmp_path, original, "out", "--cuts", cuts, "--plan", str(tmp_path / "p")) == 0
    scans = first_passes(monkeypatch)
    caplog.clear()
    with monkeypatch.context() as patch:
        short_disk(patch, 9 * KEPT_BYTES - 1)
        assert upscale(tmp_path, original, "out", "--cuts", cuts, "--quality", "maximum") == 1
    [said] = errors(caplog)
    assert said.startswith(
        f"{out}: 0.00 GiB free on its disk, where the job writes up to 0.00 GiB there at a shot's"
        " peak, its shot of 11 frames: 0.00 GiB of frames kept between its runs at --quality"
        " maximum; make room there, write the output to a disk with room (a new job, its first"
        " pass again) or take a lower level"
        " (standard none, enhanced keeps 0.00 GiB)"
    ), said
    assert said.endswith(KEPT.format(directory=out))
    with monkeypatch.context() as patch:
        short_disk(patch, 9 * KEPT_BYTES)
        assert upscale(tmp_path, original, "out", "--cuts", cuts, "--quality", "maximum") == 0
    assert scans == []
    assert recorded(out)["settings"]["quality"] == "maximum"


def tiled_scratch(patch: pytest.MonkeyPatch, spilled: int) -> list[tuple[int, int]]:
    """Every shot's tiled scratch said to take `spilled` bytes of disk (planner.scratch_file):
    the stand-in's frames are too small for a tile. The frames and the plan's tiles each call
    asked for, recorded."""
    asked: list[tuple[int, int]] = []

    def file(plan: Plan, target: tuple[int, int], frames: int, **options: Any) -> int:
        asked.append((frames, plan.scratch_allowance))
        return spilled

    patch.setattr(planner, "scratch_file", file)
    return asked


@pytest.mark.usefixtures("stand_in")
def test_the_disk_check_counts_a_tiled_pass_s_scratch_file_at_every_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # At standard, which keeps no frame, a scratch file that doesn't fit the disk refuses the
    # job, the figure said, the ways out for it; at maximum, the shot's kept frames and its
    # scratch file are counted at once; --plan warns of either. The first pass is kept.
    caplog.set_level(logging.INFO, logger="seedvr2x")
    original = source(tmp_path / "in.mkv", 17)
    monkeypatch.chdir(tmp_path)
    cuts = listed(tmp_path, "cuts.txt", "5\n6\n")
    spilled = 2**30
    with monkeypatch.context() as patch:
        asked = tiled_scratch(patch, spilled)
        short_disk(patch, spilled - 1)
        assert upscale(tmp_path, original, "one", "--cuts", cuts) == 1
        [said] = errors(caplog)
        allowance = asked[0][1] / 2**30
        assert said == (
            f"{tmp_path / 'one'}: 1.00 GiB free on its disk, where the job writes up to 1.00 GiB"
            " there at a shot's peak, its shot of 5 frames: 1.00 GiB of a tiled pass's scratch"
            f" file, over the {allowance:.2f} GiB of system memory it may take; make room there,"
            " write the output to a disk with room (a new job, its first pass again) or free"
            " system memory, which the scratch"
            " takes first" + KEPT.format(directory=tmp_path / "one")
        )
        assert {frames for frames, _ in asked} == {1, 5, 11}
        caplog.clear()
        short_disk(patch, spilled + 9 * KEPT_BYTES - 1)
        assert upscale(tmp_path, original, "four", "--cuts", cuts, "--quality", "maximum") == 1
        [said] = errors(caplog)
        assert (
            "its shot of 11 frames: 0.00 GiB of frames kept between its runs at --quality" in said
        )
        assert " and 1.00 GiB of a tiled pass's scratch file" in said
        assert "take a lower level (standard none, enhanced keeps 0.00 GiB) or free system" in said
        caplog.clear()
        plan = str(tmp_path / "plan.txt")
        assert (
            upscale(
                tmp_path, original, "four", "--cuts", cuts, "--quality", "maximum", "--plan", plan
            )
            == 0
        )
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(m.endswith("; a run of this plan would be refused") for m in warned)
        short_disk(patch, spilled + 9 * KEPT_BYTES)
        assert upscale(tmp_path, original, "four", "--cuts", cuts, "--quality", "maximum") == 0


def test_scratch_file() -> None:
    # A shot's scratch on disk: none untiled, none within the plan's allowance, else its bytes
    # (planner.scratch_file; tiling.Scratch.where's rule).
    target = (1080, 1920)
    frames = 45
    size = planner.scratch(planner._padded(target, False), frames)  # pyright: ignore[reportPrivateUsage]
    assert planner.scratch_file(Plan(12, scratch_allowance=0), target, frames) == 0
    tiled = Plan(12, decode_tiles=Tiles(1024), scratch_allowance=size)
    assert planner.scratch_file(tiled, target, frames) == 0
    over = Plan(12, encode_tiles=Tiles(1024), scratch_allowance=size - 1)
    assert planner.scratch_file(over, target, frames) == size
    padded = planner.scratch(planner._padded(target, True), frames)  # pyright: ignore[reportPrivateUsage]
    assert (
        planner.scratch_file(
            Plan(12, decode_tiles=Tiles(1024), scratch_allowance=0),
            target,
            frames,
            numz_padding=True,
        )
        == padded
    )


def test_a_hand_edited_level_of_no_name_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A plan's record whose level is a list, hand-edited: refused with the record's error, as
    # another unknown level is, not a TypeError's traceback.
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    content = recorded(out)
    content["settings"]["quality"] = ["maximum"]
    (out / "manifest.json").write_text(json.dumps(content))
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    [said] = errors(caplog)
    assert said == (
        f'{out / "manifest.json"}: quality ["maximum"], where seedvr2x has standard, enhanced and'
        " maximum"
    )


@pytest.mark.parametrize("level", ["enhanced", "maximum"])
def test_a_plan_of_several_runs_a_shot_is_taken_up_by_the_same_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    level: str,
) -> None:
    # Refused by the planner at a level of several runs, the record's runs are each one window,
    # no plan made; JOB's long shot runs in two under the plan the same command then makes, its
    # later runs' windows too (shots[i].runs[j].windows): taken up as a plan, no other job, no
    # second first pass (a review's finding, 2026-10-21).
    caplog.set_level(logging.INFO, logger="seedvr2x")
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    with monkeypatch.context() as patch:
        too_little_memory(patch)
        assert upscale(tmp_path, source_path, "out", *JOB, "--quality", level) == 1
    scans = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB, "--quality", level) == 0, errors(caplog)
    assert scans == []
    content = recorded(out)
    assert content["shots"][2]["windows"] == [[0, 4], [2, 6]]
    assert all(len(run["windows"]) == 2 for run in content["shots"][2]["runs"])


def test_a_seed_out_of_range_keeps_the_first_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A seed the source's last shot puts out of numz's range is known once the shots are: with
    # detection, after the first pass, which is kept, the seed being a setting a plan takes
    # (job.check_seed; resume.FIRST_PASS_SETTINGS). The scripted detector cuts at 5, 15 and 22;
    # the check before the pass knows the first shot alone. The default seed then plans it, no
    # second pass.
    from seedvr2x.runtime.job import ENCODE_SEED_OFFSET, SEED_LIMIT

    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    highest = SEED_LIMIT - 1 - ENCODE_SEED_OFFSET
    assert upscale(tmp_path, source_path, "out", "--seed", str(highest)) == 1
    [said] = errors(caplog)
    assert said.startswith(f"--seed {highest}: each shot is seeded with the seed plus its first")
    assert said.endswith(KEPT.format(directory=out)), said
    scans = first_passes(monkeypatch)
    assert upscale(tmp_path, source_path, "out") == 0
    assert scans == [] and recorded(out)["settings"]["seed"] == 42


@pytest.mark.parametrize("why", ["index lost", "cut list dropped", "ffmpeg changed"])
def test_a_plan_whose_first_pass_runs_again_writes_it_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    why: str,
) -> None:
    # A plan whose first pass must run again writes the new pass at once, as a new job's is, its
    # settings as asked, no plan, in the run's environment (DESIGN.md, Pause and resume, A plan,
    # decided on 2026-10-21): its frame index lost, a cut list dropped (the detector runs in it),
    # ffmpeg changed and accepted. Refused by the planner after that pass, the run says the pass
    # is kept, and the next one takes it up with no first pass of its own.
    from seedvr2x.media import ffmpeg

    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    asked: list[str] = list(JOB)
    if why == "index lost":
        (out / "frame_index.bin").unlink()
    elif why == "cut list dropped":
        asked = asked[2:]
    else:
        monkeypatch.setattr(ffmpeg, "check", lambda *options, **needs: "n0.0-another")
        asked.append("--accept-env-change")
    scans = first_passes(monkeypatch)
    with monkeypatch.context() as patch:
        too_little_memory(patch)
        assert upscale(tmp_path, source_path, "out", *asked) == 1
    [said] = errors(caplog)
    assert said.endswith(KEPT.format(directory=out)), said
    assert len(scans) == 1
    assert (out / "frame_index.bin").is_file()
    content = recorded(out)
    assert content["plan"] is None
    if why == "cut list dropped":
        assert content["settings"]["cuts"] is None and "cut_threshold" in content["settings"]
    if why == "ffmpeg changed":
        assert content["environment"]["ffmpeg"] == "n0.0-another"
        [change] = content["environment_changes"]
        assert "ffmpeg" in json.dumps(change)
    scans.clear()
    caplog.clear()
    taken = [option for option in asked if option != "--accept-env-change"]
    assert upscale(tmp_path, source_path, "out", *taken) == 0, errors(caplog)
    assert scans == [] and recorded(out)["plan"] is not None
    assert len(recorded(out)["environment_changes"]) == (1 if why == "ffmpeg changed" else 0)


def test_a_plan_s_directory_holding_another_s_file_is_refused_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A directory holding anyone else's files is refused before anything is written in it
    # (DESIGN.md, Pause and resume, A plan, decided on 2026-10-21). A plan's directory with a
    # file of another's added, its frame index lost so that its first pass runs again: refused
    # from its record, before that pass (cli._refuse_foreign); one added during the pass,
    # before the pass is written (cli._first_write), by the plan's own layout (resume.
    # leftovers). Each time nothing written there, its record as it was. The file gone, the same
    # command takes the plan up.
    source_path = job(tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert upscale(tmp_path, source_path, "out", *JOB, "--plan", str(tmp_path / "p.txt")) == 0
    (out / "frame_index.bin").unlink()
    (out / "notes.txt").write_text("someone else's\n")
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    scans = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert scans == []
    assert errors(caplog) == [f"{out}: not this job's: notes.txt"]
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before
    (out / "notes.txt").unlink()
    # Come during the pass: judged again before the pass is written.
    from seedvr2x.media import source as examined

    scan = examined.scan

    def meanwhile(path: Path, *detector: Any) -> object:
        (out / "notes.txt").write_text("someone else's\n")
        return scan(path, *detector)

    before.pop("notes.txt")
    caplog.clear()
    with monkeypatch.context() as patch:
        patch.setattr(examined, "scan", meanwhile)
        assert upscale(tmp_path, source_path, "out", *JOB) == 1
    assert errors(caplog) == [f"{out}: not this job's: notes.txt"]
    assert {
        path.name: path.read_bytes() for path in out.iterdir() if path.name != "notes.txt"
    } == before
    assert steps.calls == []
    (out / "notes.txt").unlink()
    caplog.clear()
    assert upscale(tmp_path, source_path, "out", *JOB) == 0, errors(caplog)
    assert (out / "frame_index.bin").is_file() and recorded(out)["plan"] is not None


def test_a_mkv_path_s_work_directory_holding_another_s_file_is_refused_before_its_rehearsal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A .mkv path's job, stopped: its work directory with a file of another's added is refused
    # from its record before the remux rehearsal (cli._copied) and the work directory's cleaning
    # (cli._work), which write there and remove a stopped assembly's leftovers, and before any
    # first pass (design, 2026-10-21); nothing written or removed. An assembly's own work file is
    # seedvr2x's, accepted by its name.
    from seedvr2x.media import assembly

    source_path = job(tmp_path, monkeypatch)
    steps.stop = "encode 3"
    assert upscale(tmp_path, source_path, "one.mkv", *JOB) == 130
    steps.stop = None
    work = tmp_path / "one.mkv.work"
    (work / "notes.txt").write_text("someone else's\n")
    # A segment command's partial file's name, the job recorded with none: another's too.
    from seedvr2x.media.files import command_partial

    stray = command_partial(work / "seg_000001.mkv").name
    (work / stray).write_bytes(b"another's")
    left = sorted(assembly.WORK_FILES)[0]
    (work / left).write_bytes(b"an assembly's")
    before = {path.name: path.read_bytes() for path in work.iterdir() if path.is_file()}
    rehearsed: list[Path] = []
    monkeypatch.setattr(assembly, "preflight", lambda *given: rehearsed.append(given[0]) or "")
    scans = first_passes(monkeypatch)
    caplog.clear()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB) == 1
    assert errors(caplog) == [f"{work}: not this job's: {', '.join(sorted(['notes.txt', stray]))}"]
    assert rehearsed == [] and scans == []
    assert {path.name: path.read_bytes() for path in work.iterdir() if path.is_file()} == before
    (work / "notes.txt").unlink()
    (work / stray).unlink()
    assert upscale(tmp_path, source_path, "one.mkv", *JOB) == 0, errors(caplog)
    # The record holds the passed rehearsal of this input and ffmpeg: a resume trusts it and runs
    # none (DESIGN.md, Output, Assembly, decided on 2026-10-28), the assembly's own leftover removed
    # all the same (cli._work).
    assert rehearsed == [] and (tmp_path / "one.mkv").is_file()
    assert not (work / left).exists()
