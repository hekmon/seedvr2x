"""--deband on the command line (DESIGN.md, Input, "Banded sources"): the frames the model is given
and its colour correction's reference those of the source debanded by ffmpeg, the setting in the
job's settings only when on, a resume with the other value another job naming the option, the
record on the masters and the finished file, the plan's line and the help. The chain itself is
test_deband.py's, whose banded clip this uses; the jobs run on test_cli_run.py's stand-ins and
skip as it does without a build seedvr2x accepts. A picture's refusal is test_still_cli.py's."""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_cli import _option_help  # pyright: ignore[reportPrivateUsage]
from test_cli_run import (
    JOB,
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

# The plan's line of --deband (planner._deband_said), whole.
LINE = (
    "deband: on, ffmpeg's deband=blur=0 on every frame the job reads, at 16 bits before their"
    " conversion to RGB (--deband); the shot detection reads the source as it is"
)

# test_deband.py's banded clip at the stand-in's size, as many frames as test_cli_run.py's JOB
# cuts: a slow gradient, an 8-bit step every few pixels, tagged.
BANDED = (
    f"gradients=s=64x48:c0=0x302818:c1=0x584838:speed=0.02:r=25:d=1,format=yuv420p,setparams={TAGS}"
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


def test_the_model_and_the_correction_are_given_the_source_debanded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
) -> None:
    # Each shot's frames as its encode reads them (Steps.read), and the masters, whose colours
    # the correction takes from the shot's input copy: a job with --deband is a job without it on
    # the source ffmpeg debanded, shot for shot and frame for frame; a job without it on the
    # banded source is given other frames.
    banded, prefiltered = clips(tmp_path, monkeypatch)
    split = ("--color-correction", "split")
    assert upscale(tmp_path, banded, "debanded", *JOB, *split, "--deband") == 0
    given = dict(steps.read)
    assert sorted(given) == [0, 3, 4]
    steps.read.clear()
    assert upscale(tmp_path, prefiltered, "reference", *JOB, *split) == 0
    assert steps.read == given
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "debanded" / name) == decoded(tmp_path / "reference" / name)
    steps.read.clear()
    assert upscale(tmp_path, banded, "plain", *JOB, *split) == 0
    assert all(steps.read[shot] != given[shot] for shot in given)


@pytest.mark.usefixtures("stand_in")
def test_the_setting_the_record_and_the_plan_say_it_only_when_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The settings hold the filter as the reads ran it, the record on each master and on the
    # finished file says it (main._job_record), and the plan has its line; a job without it
    # holds no key, no tag and no line, its record and plan those of before the option.
    banded, _ = clips(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)
    options = (*JOB, "--format", "yuv420p10le")
    assert upscale(tmp_path, banded, "out", *options, "--deband") == 0
    assert LINE in caplog.text
    content = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert content["settings"]["deband"] == "deband=blur=0"
    expected = recorded_tags(content, "64x48", "128x96")
    assert expected["SEEDVR2X_DEBAND"] == "deband=blur=0"
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        tags = stream_tags_of(tmp_path / "out" / name)
        assert {key: tags.get(key) for key in expected} == expected, name
    assert upscale(tmp_path, banded, "x.mkv", *options, "--deband") == 0
    beside = json.loads((tmp_path / "x.mkv.manifest.json").read_text())
    tags = stream_tags_of(tmp_path / "x.mkv")
    assert tags.get("SEEDVR2X_DEBAND") == "deband=blur=0"
    assert {key: tags.get(key) for key in recorded_tags(beside, "64x48", "128x96")} == (
        recorded_tags(beside, "64x48", "128x96")
    )
    caplog.clear()
    assert upscale(tmp_path, banded, "plain", *options) == 0
    # The plan's lines alone: the job's paths may hold the word.
    assert "plan: deband:" not in caplog.text and LINE not in caplog.text
    plain = json.loads((tmp_path / "plain" / "manifest.json").read_text())
    assert "deband" not in plain["settings"]
    assert "SEEDVR2X_DEBAND" not in recorded_tags(plain, "64x48", "128x96")
    assert not [
        key for key in stream_tags_of(tmp_path / "plain" / "seg_000000.mkv") if "DEBAND" in key
    ]


@pytest.mark.parametrize(
    ("before", "after", "line"),
    [
        ((), ("--deband",), 'settings.deband: null -> "deband=blur=0" (--deband)'),
        (("--deband",), (), 'settings.deband: "deband=blur=0" -> null (--deband)'),
    ],
)
def test_a_resume_with_the_other_value_is_another_job_naming_the_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    before: tuple[str, ...],
    after: tuple[str, ...],
    line: str,
) -> None:
    # A job that has made units, resumed with --deband given or left out, would read its next
    # shots' frames otherwise than the shots it made: refused, the setting named with its
    # option, nothing touched; the same command as before then resumes it.
    banded, _ = clips(tmp_path, monkeypatch)
    steps.stop = "window 4:1"
    stopped(tmp_path, banded, "out", *JOB, *before)
    kept = contents(tmp_path / "out")
    text = refused(tmp_path, banded, caplog, *JOB, *after)
    assert "another job than the one asked" in text and line in text
    assert contents(tmp_path / "out") == kept
    steps.stop = None
    steps.calls.clear()
    assert upscale(tmp_path, banded, "out", *JOB, *before) == 0
    assert steps.calls and "encode 0" not in steps.calls


@pytest.mark.parametrize("deband", [True, False])
def test_a_plan_is_planned_again_with_the_other_value_its_first_pass_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
    deband: bool,
) -> None:
    # A job that has made no unit (--plan) takes --deband given or left out, planned again from
    # its record with no first pass, as any setting the first pass doesn't read
    # (resume.FIRST_PASS_SETTINGS): the cuts and the index are the source's either way. Its
    # frames then are the ones the new value reads, the key written or gone with it.
    banded, prefiltered = clips(tmp_path, monkeypatch)
    split = ("--color-correction", "split")
    flag = ("--deband",) if deband else ()
    assert upscale(tmp_path, prefiltered if deband else banded, "reference", *JOB, *split) == 0
    expected = dict(steps.read)
    steps.read.clear()
    other = () if deband else ("--deband",)
    assert upscale(tmp_path, banded, "out", *JOB, *split, *other, "--plan", "plan.txt") == 0
    assert not steps.read
    caplog.set_level(logging.INFO)
    caplog.clear()
    assert upscale(tmp_path, banded, "out", *JOB, *split, *flag) == 0
    assert "the same input and ffmpeg, the first pass as recorded" in caplog.text
    assert "re-planned, its job having made no unit yet: settings.deband: " in caplog.text
    assert steps.read == expected
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings.get("deband") == (DEBAND if deband else None)
    for name in ("seg_000000.mkv", "seg_000001.mkv"):
        assert decoded(tmp_path / "out" / name) == decoded(tmp_path / "reference" / name)


def test_a_resume_with_it_reads_on_as_the_uninterrupted_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Stopped inside its last shot's windows, its input copy then deleted, a job with --deband
    # resumed with it: the copy is made again from the input through the deband, and the
    # masters, corrected against it, are those of the job run uninterrupted.
    banded, _ = clips(tmp_path, monkeypatch)
    options = (*JOB, "--color-correction", "split", "--deband")
    assert upscale(tmp_path, banded, "whole", *options) == 0
    steps.stop = "window 4:1"
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
    # out of both sides unless both jobs took --deband (as inductor's packages are).
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
    # A change in the deband chains' digest alone: the resume of a job with --deband is refused
    # as an environment change naming the key, that of a job without it resumes untouched.
    from media import fingerprint

    banded, _ = clips(tmp_path, monkeypatch)
    for options, refusal in (((), False), (("--deband",), True)):
        out = "deband" if options else "plain"
        steps.stop = "window 4:1"
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


def test_the_help_says_what_it_does_and_when_with_no_figure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Where it helps and what it may cost, from the study (impl-box-1: dark shaded animation and
    # near-black haze), in words, as --compile's help.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = _option_help(capsys.readouterr().out, "--deband")
    assert not re.search(r"\d", text)
    for words in (
        "smooth the source's banding before the model sees it",
        "ffmpeg's deband on every frame the job reads",
        "the shot detection reads the source as it is",
        "dark shaded animation and near-black haze",
        "draws the steps as contour lines",
        "may take faint lines or fine gradients for banding",
        "never on by default",
        "A setting, which a resume repeats; not for a picture",
    ):
        assert words in text, words
