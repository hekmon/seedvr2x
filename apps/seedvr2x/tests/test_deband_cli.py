"""The deband on the command line (DESIGN.md, Input, "Banded sources"), which --quality deghost
alone runs since --deband went into the level (DESIGN.md, Each frame on its own; "Dropped as an
option", 2026-11-17): the setting in the job's settings only at that level, a job begun with
--deband at another level another job, the record on the masters and the finished file, the
plan's line, a plan taken up with the other level, a resume, the chains' digest, and the option
gone from the parser and the help. That the model and the correction are given the source
debanded by ffmpeg is tests/test_deghost_cli.py's (a job's frames those of one-frame jobs on
the source ffmpeg debanded). The chain itself is test_deband.py's, whose banded clip this uses;
the jobs run on test_cli_run.py's stand-ins and skip as it does without a build seedvr2x
accepts."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_cli import _option_help  # pyright: ignore[reportPrivateUsage]
from test_cli_run import (
    Steps,
    contents,
    decoded,
    recorded_tags,
    refused,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    stopped,
    stream_tags_of,
    upscale,
)
from test_deband import TAGS, needs_ffmpeg

import main as cli
import resume
from media.conversion import DEBAND

pytestmark = needs_ffmpeg

# The plan's line of the deband (planner._deband_said), whole.
LINE = (
    f"deband: on, ffmpeg's {DEBAND} on every frame the job reads, at 16 bits before"
    " their conversion to RGB (--quality deghost); the shot detection reads the source as it is"
)
# The level that debands, as a job's options give it.
DEGHOST = ("--quality", "deghost")
# test_cli_run.py's JOB without its window: --window is refused at deghost (a frame alone is one
# latent in the DiT), and the same options at every level keep a plan taken up comparable.
JOB = ("--cuts", "cuts.txt", "--min-segment", "0.12")

# test_deband.py's banded clip at the stand-in's size, as many frames as test_cli_run.py's JOB
# cuts: a slow gradient, an 8-bit step every few pixels, tagged; seeded as test_deband.py's.
BANDED = (
    "gradients=s=64x48:c0=0x302818:c1=0x584838:speed=0.02:r=25:d=1:seed=1,"
    f"format=yuv420p,setparams={TAGS}"
)


def run_ffmpeg(*arguments: str) -> None:
    result = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *arguments], check=False)
    assert result.returncode == 0


def clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """JOB's cut list, the banded clip (FFV1, 25 frames), and the same debanded by ffmpeg alone
    into a 16-bit FFV1 file, as test_deband.py's reference."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cuts.txt").write_text("3\n4\n")
    banded, prefiltered = tmp_path / "banded.mkv", tmp_path / "prefiltered.mkv"
    run_ffmpeg("-f", "lavfi", "-i", BANDED, "-frames:v", "25", "-c:v", "ffv1", str(banded))
    run_ffmpeg(
        *("-i", str(banded), "-vf", f"format=yuv420p16le,{DEBAND}:threads=1,setparams={TAGS}"),
        *("-c:v", "ffv1", str(prefiltered)),
    )
    return banded, prefiltered


@pytest.mark.usefixtures("stand_in")
def test_the_setting_the_record_and_the_plan_say_it_only_at_deghost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The settings hold the filter as the reads ran it, the record on each master and on the
    # finished file says it (main._job_record), and the plan has its line, at --quality
    # deghost; a job at another level holds no key, no tag and no line, its record and plan
    # those of before the level.
    banded, _ = clips(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)
    options = (*JOB, "--format", "yuv420p10le")
    assert upscale(tmp_path, banded, "out", *options, *DEGHOST) == 0
    assert LINE in caplog.text
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert (
        content["settings"]["deband"] == "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"
    )
    expected = recorded_tags(content, "64x48", "128x96")
    assert (
        expected["SEEDVR2X_DEBAND"] == "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"
    )
    assert expected["SEEDVR2X_QUALITY"] == "deghost"
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        tags = stream_tags_of(tmp_path / "out" / name)
        assert {key: tags.get(key) for key in expected} == expected, name
    assert upscale(tmp_path, banded, "x.mkv", *options, *DEGHOST) == 0
    beside = json.loads((tmp_path / "x.mkv.manifest.json").read_text())
    tags = stream_tags_of(tmp_path / "x.mkv")
    assert (
        tags.get("SEEDVR2X_DEBAND") == "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"
    )
    assert {key: tags.get(key) for key in recorded_tags(beside, "64x48", "128x96")} == (
        recorded_tags(beside, "64x48", "128x96")
    )
    for level in ("standard", "enhanced", "maximum"):
        caplog.clear()
        assert upscale(tmp_path, banded, level, *options, "--quality", level) == 0
        # The plan's lines alone: the job's paths may hold the word.
        assert "plan: deband:" not in caplog.text and LINE not in caplog.text
        plain = json.loads((tmp_path / level / "manifest.json").read_text())
        assert "deband" not in plain["settings"]
        assert "SEEDVR2X_DEBAND" not in recorded_tags(plain, "64x48", "128x96")
        assert not [
            key for key in stream_tags_of(tmp_path / level / "seg_000000.mkv") if "DEBAND" in key
        ]


def test_a_job_begun_with_deband_at_another_level_is_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A job that --deband began at standard (its manifest holds the key, as the option wrote it):
    # its next shots would be read otherwise than those it made, refused as another job, the
    # setting named with the option gone and the level that debands now, nothing touched; as the
    # retired test modes are (tests/test_cli_run.py).
    banded, _ = clips(tmp_path, monkeypatch)
    steps.stop = "window 4:0"
    stopped(tmp_path, banded, "out", *JOB)
    path = tmp_path / "out" / "manifest.json"
    content = json.loads(path.read_text())
    assert "deband" not in content["settings"]
    content["settings"]["deband"] = DEBAND
    path.write_text(json.dumps(content))
    kept = contents(tmp_path / "out")
    text = refused(tmp_path, banded, caplog, *JOB)
    assert "another job than the one asked" in text
    assert (
        f'settings.deband: "{DEBAND}" -> null (--deband, an option no more: --quality deghost'
        " debands)" in text
    )
    assert contents(tmp_path / "out") == kept


@pytest.mark.parametrize(("before", "after"), [((), DEGHOST), (DEGHOST, ())])
def test_a_resume_at_the_other_level_is_another_job_naming_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    before: tuple[str, ...],
    after: tuple[str, ...],
) -> None:
    # A job that has made units, resumed at deghost or from it: refused, the level and the
    # deband named, nothing touched; the same command as before then resumes it.
    banded, _ = clips(tmp_path, monkeypatch)
    steps.stop = "window 4:0"
    stopped(tmp_path, banded, "out", *JOB, *before)
    kept = contents(tmp_path / "out")
    text = refused(tmp_path, banded, caplog, *JOB, *after)
    assert "another job than the one asked" in text and "settings.deband: " in text
    assert "settings.quality: " in text
    assert contents(tmp_path / "out") == kept
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, banded, "out", *JOB, *before) == 0
    assert steps.calls and "encode 0" not in steps.calls


@pytest.mark.parametrize("deghost", [True, False])
def test_a_plan_is_planned_again_at_the_other_level_its_first_pass_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    deghost: bool,
) -> None:
    # A job that has made no unit (--plan) is taken up at deghost or from it, planned again
    # from its record with no first pass, as any setting the first pass doesn't read
    # (resume.FIRST_PASS_SETTINGS): the cuts and the index are the source's either way. Its
    # masters then are those of the job run at that level from the start, the key written or
    # gone with it.
    banded, _ = clips(tmp_path, monkeypatch)
    split = ("--color-correction", "split")
    level = DEGHOST if deghost else ()
    assert upscale(tmp_path, banded, "reference", *JOB, *split, *level) == 0
    other = () if deghost else DEGHOST
    assert upscale(tmp_path, banded, "out", *JOB, *split, *other, "--plan", "plan.txt") == 0
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert upscale(tmp_path, banded, "out", *JOB, *split, *level) == 0
    assert "the same input and ffmpeg, the first pass as recorded" in caplog.text
    assert "re-planned, its job having made no unit yet: " in caplog.text
    assert "settings.deband: " in caplog.text
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings.get("deband") == (DEBAND if deghost else None)
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "reference" / name)


def test_a_resume_at_deghost_reads_on_as_the_uninterrupted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Stopped inside its last shot's frames, its input copy then deleted, a job at deghost
    # resumed: the copy is made again from the input through the deband, and the masters,
    # corrected against it, are those of the job run uninterrupted.
    banded, _ = clips(tmp_path, monkeypatch)
    options = (*JOB, "--color-correction", "split", *DEGHOST)
    assert upscale(tmp_path, banded, "whole", *options) == 0
    steps.stop = "window 4:0"
    stopped(tmp_path, banded, "out", *options)
    (tmp_path / "out" / "resume" / "shot_000004" / "input.mkv").unlink()
    steps.stop = None
    steps.calls.clear()
    caplog.set_level(logging.INFO)
    assert upscale(tmp_path, banded, "out", *options) == 0
    assert "encode 0" not in steps.calls
    assert "input.mkv missing: its input copy made again from the input" in caplog.text
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "whole" / name)


def content(deband: bool, digest: str) -> dict[str, Any]:
    return {
        "settings": {"seed": 1, **({"deband": DEBAND} if deband else {})},
        "environment": {
            "torch": "2.14.1",
            **({"deband_conversions": digest} if deband else {}),
        },
        "segments": [],
        "shots": [],
    }


def test_the_deband_digest_counts_only_between_two_deband_jobs() -> None:
    # resume.differences and environment_change: the environment's deband_conversions is left
    # out of both sides unless both jobs debanded (as inductor's packages are).
    plain, one, other = content(False, ""), content(True, "a" * 64), content(True, "b" * 64)
    assert resume.differences(one, other) == [
        f'environment.deband_conversions: "{"a" * 64}" -> "{"b" * 64}"'
    ]
    assert resume.differences(one, content(True, "a" * 64)) == []
    # A plain job and a deband one differ by the setting alone, either way.
    assert resume.differences(plain, one) == [f'settings.deband: null -> "{DEBAND}"']
    assert resume.differences(one, plain) == [f'settings.deband: "{DEBAND}" -> null']
    # The accepted change's record.
    change = resume.environment_change(one, other["environment"], other["settings"])
    assert change["before"] == {"deband_conversions": "a" * 64}
    assert change["after"] == {"deband_conversions": "b" * 64}
    for old, new in ((plain, one), (one, plain)):
        change = resume.environment_change(old, new["environment"], new["settings"])
        assert change["before"] == {} and change["after"] == {}
    # A plain job that carries the key (it should not) has it left out against a plain one.
    stray = content(False, "")
    stray["environment"]["deband_conversions"] = "c" * 64
    assert resume.differences(plain, stray) == []


def test_ffmpegs_deband_changing_refuses_a_deband_resume_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A change in the deband chains' digest alone: the resume of a job at deghost, which
    # debands, is refused as an environment change naming the key, that of a job at another level
    # resumes untouched.
    from media import fingerprint

    banded, _ = clips(tmp_path, monkeypatch)
    for options, refusal in (((), False), (DEGHOST, True)):
        out = "deband" if options else "plain"
        steps.stop = "window 4:0"
        stopped(tmp_path, banded, out, *JOB, *options)
        steps.stop = None
        with monkeypatch.context() as patch:
            patch.setattr(fingerprint, "deband_fingerprint", lambda: "0" * 64)
            if refusal:
                text = refused(tmp_path, banded, caplog, *JOB, *options, output=out)
                assert "environment.deband_conversions" in text
                assert "Only its environment differs" in text
            else:
                assert upscale(tmp_path, banded, out, *JOB, *options) == 0


def test_the_option_is_gone_from_the_parser_and_the_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --deband went into --quality deghost (DESIGN.md, Banded sources, "Dropped as an option"):
    # no option, refused by the parser as any unknown one; the help names it nowhere, and says
    # the level's deband in --quality's.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "--deband" not in text
    assert "its source debanded" in _option_help(text, "--quality")
    with pytest.raises(SystemExit) as caught:
        cli.main(["in.mkv", "-o", "out", "--deband"])
    assert caught.value.code == 2
    assert "unrecognized arguments: --deband" in capsys.readouterr().err
