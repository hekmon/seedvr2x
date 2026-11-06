"""The command line over a dynamic GGUF DiT file (DESIGN.md, Weights, "The dynamic GGUF before
v0.1.0 too"), on the CPU, as test_int8_cli.py has it over an int8 one, the model replaced by
test_cli_run.py's stand-in but for the check of its files: a synthetic 7B in the GGUF layout
(test_weights.py's dynamic_file, as_dtype and write, core/tests/) and a VAE in fp16 in
--model-dir, each checked by its header and said in the log for what it is; the plan made by the
GGUF's figures, its DiT line saying so; the job's record naming the file and the plan's format,
so that the same output asked with another DiT file is another job; a resume judged early by the
GGUF's figures. And the smaller files the plan names where fp16 is short (planner.plan_naming,
planner.smaller_swaps; stills.py's own for a picture, over test_still_cli.py's machine): in a
refusal for memory, only those that plan the job, and in the plan's line where they swap fewer
blocks or run a longer window (a picture, one latent, by the blocks alone); the file never changed
by itself. The library's half is test_weights.py's,
test_blockswap.py's and test_peaks.py's (core/tests/); the planner's, test_planner.py's.

The synthetic files are sparse, 4.76 and 16.48 GB on paper: their SHA-256, which the run reads for
the job's record, is taken over their headers here (hashed), the bytes a sparse file holds."""

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import test_cli_run
from levels import levels
from test_cli_run import (
    CHECK_MODELS,
    GIB,
    JOB,
    JOB_SHOTS,
    TARGET,
    Steps,
    command,
    early_card,
    job,
    plan_lines,
    stand_in,  # noqa: F401
    steps,  # noqa: F401
    without_disk,
)
from test_int8_cli import hashed
from test_still_cli import Machine, log_everything, machine, make_picture  # noqa: F401
from test_still_cli import command as picture_command
from test_weights import as_dtype, dynamic_file, write

import main as cli
import planner
from seedvr2x import files, model, peaks, pull, weights
from seedvr2x.shot_plan import Pins
from seedvr2x.weights import Architecture

DIT_MODEL = cli._dit_model  # pyright: ignore[reportPrivateUsage]
FP16, VAE = "fp16.safetensors", "vae.safetensors"
# The two files, their format and the name the plan's record gives it.
FILES = [
    pytest.param(weights.DIT_7B_DYN, peaks.GGUF_7B, "gguf_dyn", id="7b"),
    pytest.param(weights.DIT_7B_SHARP_DYN, peaks.GGUF_7B_SHARP, "gguf_sharp_dyn", id="sharp"),
]
# The synthetic fp16 file's name is no pinned one, so it does not say which 7B it holds: both 7Bs'
# files are named, each as its model's.
# These tests give --model-dir, so the way out says once that the file goes there.
BOTH_WAY = (
    "a smaller DiT file that plans this job, put in --model-dir: for the sharp 7B, int8"
    " (--dit-model seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF"
    " (--dit-model"
    " seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB); for the 7B, int8 (--dit-model"
    " seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF (--dit-model"
    " seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
)


def models_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory of a VAE and an fp16 DiT, the checks real and the SHA-256 over the headers."""
    models = tmp_path / "models"
    models.mkdir()
    write(models / FP16, as_dtype("7b", "F16"))
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)  # the stand-in load's, the same
    return models


def loads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The DiT files the run loads, in order."""
    loaded: list[Path] = []
    stand_in_load: Any = model.load_models

    def load(dit: Path, *arguments: Any) -> Any:
        loaded.append(dit)
        return stand_in_load(dit, *arguments)

    monkeypatch.setattr(model, "load_models", load)
    return loaded


@pytest.mark.usefixtures("steps")
@pytest.mark.parametrize(("architecture", "dit_model", "name"), FILES)
def test_a_gguf_dit_runs_planned_by_its_figures(
    architecture: Architecture,
    dit_model: Architecture,
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = models_in(tmp_path, monkeypatch)
    gguf = "dit.gguf"
    dynamic_file(models / gguf, architecture)
    loaded = loads(monkeypatch)
    chosen = ("--model-dir", str(models), "--vae-model", VAE)
    asked = command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", gguf)
    assert cli.main(asked) == 0
    assert loaded == [models / gguf]
    assert f"DiT {gguf}: {weights.in_words(architecture)}" in caplog.messages
    assert f"VAE {VAE}: SeedVR2's VAE in fp16" in caplog.messages
    # The plan by the GGUF's figures, as the planner makes it for that job, its DiT line saying so.
    pins = Pins(window=5)
    free = test_cli_run.FREE
    made = planner.plan(free, TARGET, JOB_SHOTS, pins, test_cli_run.HOST, dit_model=dit_model)
    assert without_disk(plan_lines(caplog)) == planner.said(
        *made, "a stand-in", TARGET, JOB_SHOTS, pins
    )
    weights_gib = f"{peaks.weights_bytes(dit_model) / GIB:.2f}"
    buffers_gib = f"{peaks.GGUF_BUFFERS / GIB:.2f}"
    dit_line = [line for line in plan_lines(caplog) if line.startswith("plan: DiT: ")]
    assert dit_line == [
        f"plan: DiT: its weights in seedvr2x's dynamic GGUF, {weights_gib} GiB, under a third of"
        f" fp16's; a window's memory counted as fp16's plus {buffers_gib} GiB for the"
        " dequantisation's buffers; --dit-model names the file"
    ]
    # No smaller file is offered to a GGUF DiT, nor to a plan that swaps nothing.
    assert not [line for line in plan_lines(caplog) if "smaller DiT files" in line]
    fp16_made = planner.plan(free, TARGET, JOB_SHOTS, pins, test_cli_run.HOST)
    record = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert record["plan"]["memory"]["dit_format"] == name
    assert record["plan"]["memory"]["dit_peak"] == made[1].dit
    assert record["settings"]["dit_model"]["name"] == gguf
    assert record["settings"]["dit_model"]["sha256"] == hashed(models / gguf)
    # The same output with the fp16 file: another job, refused, its record left as it was.
    caplog.clear()
    again = command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", FP16)
    assert cli.main(again) == 1
    assert "another job than the one asked" in caplog.text
    assert "settings.dit_model" in caplog.text
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == record
    # And in a directory of its own, the fp16 file runs, planned by fp16's figures, its record
    # saying fp16.
    caplog.clear()
    loaded.clear()
    other = command(tmp_path, source_path, "fp16-out", *JOB, *chosen, "--dit-model", FP16)
    assert cli.main(other) == 0
    assert loaded == [models / FP16]
    fp16_record = json.loads((tmp_path / "fp16-out" / "manifest.json").read_text())
    assert fp16_record["plan"]["memory"]["dit_format"] == "fp16"
    assert fp16_record["plan"]["memory"]["dit_peak"] == fp16_made[1].dit


def test_the_early_judgement_reads_the_gguf_s_format_where_its_file_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A resume's early judgement of the memory (cli._early_dit_model) reads either GGUF file's
    # format from its header, in --model-dir or in Hugging Face's cache for seedvr2x's own files.
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    early = cli._early_dit_model  # pyright: ignore[reportPrivateUsage]
    models = tmp_path / "models"
    models.mkdir()
    held: dict[str, Path] = {}
    monkeypatch.setattr(pull, "cached", lambda name: held.get(name))
    for architecture, dit_model, own in (
        (weights.DIT_7B_DYN, peaks.GGUF_7B, "seedvr2x_ema_7b_dyn.gguf"),
        (weights.DIT_7B_SHARP_DYN, peaks.GGUF_7B_SHARP, "seedvr2x_ema_7b_sharp_dyn.gguf"),
    ):
        path = dynamic_file(models / "mine.gguf", architecture)
        assert early(Namespace(model_dir=models, dit_model="mine.gguf")) == dit_model
        assert own in pull.PINNED
        assert early(Namespace(model_dir=None, dit_model=own)) is None
        held[own] = path
        assert early(Namespace(model_dir=None, dit_model=own)) == dit_model


def test_a_gguf_resume_is_judged_early_by_the_gguf_s_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A GGUF job stopped, then resumed with too little free on the card by NVML: the early
    # judgement, before the checks, plans with the GGUF's architecture, and warns; the plan that
    # counts asks it again with the same. Never the "isn't judged" debug line.
    from seedvr2x import memory

    levels(caplog, logging.DEBUG)
    source_path = job(tmp_path, monkeypatch)
    whole = files.sha256  # the source's, as the first pass took it, which the resume compares
    models = models_in(tmp_path, monkeypatch)
    dynamic_file(models / "dit.gguf", weights.DIT_7B_SHARP_DYN)
    monkeypatch.setattr(
        files, "sha256", lambda path: hashed(path) if Path(path).parent == models else whole(path)
    )
    monkeypatch.setattr(test_cli_run, "digest_of", files.sha256)
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", "dit.gguf")
    steps.stop = "encode 4"
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 130
    steps.stop = None
    asked: list[tuple[int | None, Architecture]] = []
    plan = planner.plan

    def planned(*arguments: Any, **options: Any) -> Any:
        asked.append((arguments[4], options["dit_model"]))
        return plan(*arguments, **options)

    monkeypatch.setattr(planner, "plan", planned)
    padded = test_cli_run.PADDED
    # The plan's last rung takes a VAE pass's room down to peaks.VAE_ROOM_LEAST (patch 03): the
    # card is short of that, as test_cli_run's early-reading tests count it.
    encode = (
        peaks.encode_peak(padded, None, 0, dit_model=peaks.GGUF_7B_SHARP) + peaks.VAE_ROOM_LEAST
    )
    early_card(monkeypatch, encode + planner.MARGIN - GIB // 4 + memory.CONTEXT)
    caplog.clear()
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 0
    assert asked[0] == (None, peaks.GGUF_7B_SHARP)
    assert {dit_model for _, dit_model in asked} == {peaks.GGUF_7B_SHARP}
    assert any("does not hold the recorded plan" in message for message in caplog.messages)
    assert not any("isn't judged before the checks" in message for message in caplog.messages)


@pytest.mark.usefixtures("steps")
def test_a_refusal_for_memory_names_the_smaller_files_that_plan_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A card of 12 GiB and a host of 4: the fp16 DiT's pinned copies don't fit the system memory,
    # int8's and the GGUF's do (test_planner.py holds the figures by the planner). The refusal
    # keeps its words and adds the way out, naming both 7Bs' two files with their
    # sizes; nothing loads; the file is never changed by itself. A GGUF DiT on a host of 2 is
    # refused as before, naming none.
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = models_in(tmp_path, monkeypatch)
    dynamic_file(models / "dit.gguf", weights.DIT_7B_SHARP_DYN)
    loaded = loads(monkeypatch)
    monkeypatch.setattr(test_cli_run, "FREE", 12 * GIB)
    monkeypatch.setattr(cli, "_host_available", lambda: 4 * GIB)
    chosen = ("--model-dir", str(models), "--vae-model", VAE)
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", FP16)) == 1
    refusal = caplog.records[-1].getMessage()
    # Before what the command says of the first pass it keeps (main._run).
    assert f"short; or {BOTH_WAY}; the first pass is kept in " in refusal
    assert "of system memory" in refusal and "pinned" in refusal
    assert loaded == []
    # The same card and host plan the GGUF.
    caplog.clear()
    assert (
        cli.main(command(tmp_path, source_path, "ok", *JOB, *chosen, "--dit-model", "dit.gguf"))
        == 0
    )
    assert loaded == [models / "dit.gguf"]
    # A host too small for any format: the refusal is the plan's own.
    monkeypatch.setattr(cli, "_host_available", lambda: 2 * GIB)
    caplog.clear()
    assert (
        cli.main(command(tmp_path, source_path, "no", *JOB, *chosen, "--dit-model", "dit.gguf"))
        == 1
    )
    assert "smaller DiT file" not in caplog.text
    caplog.clear()
    assert cli.main(command(tmp_path, source_path, "no", *JOB, *chosen, "--dit-model", FP16)) == 1
    assert "smaller DiT file" not in caplog.text


@pytest.mark.usefixtures("steps")
def test_the_plan_names_the_smaller_files_that_plan_the_job_better(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A card of 16 GiB and a host of 8: the fp16 DiT swaps 3 blocks, int8's and the GGUF's none.
    # The plan says so in a line after the swapped blocks, names the files in the way a refusal
    # does not need, and the run keeps the fp16 file; the GGUF's own plan says nothing.
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = models_in(tmp_path, monkeypatch)
    dynamic_file(models / "dit.gguf", weights.DIT_7B_SHARP_DYN)
    loaded = loads(monkeypatch)
    monkeypatch.setattr(test_cli_run, "FREE", 16 * GIB)
    monkeypatch.setattr(cli, "_host_available", lambda: 8 * GIB)
    chosen = ("--model-dir", str(models), "--vae-model", VAE)
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", FP16)) == 0
    assert loaded == [models / FP16]
    lines = plan_lines(caplog)
    at = next(
        number for number, each in enumerate(lines) if each.startswith("plan: blocks swapped")
    )
    assert lines[at].startswith("plan: blocks swapped: 3 of the DiT's 36")
    assert lines[at + 1] == (
        "plan: smaller DiT files plan this job better: this plan swaps 3 blocks with windows of 4"
        " latents; for the sharp 7B (int8 0 blocks, windows of 4; the dynamic GGUF 0 blocks,"
        " windows of 4); for the 7B (int8 0 blocks, windows of 4; the dynamic GGUF 0 blocks,"
        " windows of 4); --dit-model names them"
    )
    caplog.clear()
    assert (
        cli.main(command(tmp_path, source_path, "g", *JOB, *chosen, "--dit-model", "dit.gguf")) == 0
    )
    assert not [line for line in plan_lines(caplog) if "smaller DiT files" in line]
    # --plan says it too (it loads no model).
    caplog.clear()
    listing = str(tmp_path / "plan.txt")
    plan_only = command(tmp_path, source_path, "p", *JOB, *chosen, "--dit-model", FP16)
    assert cli.main([*plan_only, "--plan", listing]) == 0
    assert [line for line in plan_lines(caplog) if "smaller DiT files" in line] == [lines[at + 1]]


def formats_by_name(monkeypatch: pytest.MonkeyPatch, machine: Machine) -> None:  # noqa: F811
    """The stand-in machine's DiT files, empty, each in the format its name says."""
    kinds = {
        "fp16.safetensors": peaks.FP16,
        "int8.safetensors": peaks.INT8,
        "dit.gguf": peaks.GGUF_7B_SHARP,
    }
    for name in kinds:
        (machine.models_dir / name).write_bytes(b"")
    monkeypatch.setattr(cli, "_dit_model", lambda path: kinds[path.name])


def test_a_pictures_refusal_for_memory_names_the_smaller_files_that_plan_it(
    machine: Machine,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The picture's plan (stills.py) as the video's: on a card of 12 GiB and a host of 4, the fp16
    # DiT is refused for the system memory and the refusal names the files that plan the picture,
    # with their sizes; the GGUF DiT runs there, and on a host of 2 is refused naming none.
    formats_by_name(monkeypatch, machine)
    machine.free = 12 * GIB
    monkeypatch.setattr(cli, "_host_available", lambda: 4 * GIB)
    source = make_picture(tmp_path / "in.png")

    def run(dit: str, host: int, name: str) -> int:
        monkeypatch.setattr(cli, "_host_available", lambda: host * GIB)
        asked = picture_command(machine, source, tmp_path / name, "--dit-model", dit)
        return cli.main(asked)

    assert run("fp16.safetensors", 4, "a.png") == 1
    refusal = caplog.records[-1].getMessage()
    assert refusal.endswith(f"; or {BOTH_WAY}"), refusal
    assert machine.loads == [] and not (tmp_path / "a.png").exists()
    # int8's own file is offered the GGUF alone, only where that plans: here int8 plans, so it runs.
    assert run("int8.safetensors", 4, "i.png") == 0
    assert run("dit.gguf", 4, "g.png") == 0
    caplog.clear()
    assert run("dit.gguf", 2, "n.png") == 1
    assert "smaller DiT file" not in caplog.text
    caplog.clear()
    assert run("fp16.safetensors", 2, "m.png") == 1
    assert "smaller DiT file" not in caplog.text


def test_a_pictures_plan_names_the_smaller_files_that_swap_fewer_blocks(
    machine: Machine,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A card of 16 GiB and a host of 8: the fp16 DiT swaps 3 blocks for the picture, int8's and
    # the GGUF's none; the plan says so after the blocks swapped and the run keeps its file.
    formats_by_name(monkeypatch, machine)
    machine.free = 16 * GIB
    monkeypatch.setattr(cli, "_host_available", lambda: 8 * GIB)
    source = make_picture(tmp_path / "in.png")
    asked = picture_command(machine, source, tmp_path / "o.png", "--dit-model", "fp16.safetensors")
    assert cli.main(asked) == 0
    lines = [line for line in caplog.messages if line.startswith("plan: ")]
    at = next(
        number for number, each in enumerate(lines) if each.startswith("plan: blocks swapped")
    )
    assert lines[at].startswith("plan: blocks swapped: 3 of the DiT's 36")
    assert lines[at + 1] == (
        "plan: smaller DiT files plan this job better: this plan swaps 3 blocks; for the sharp 7B"
        " (int8 0 blocks; the dynamic GGUF 0 blocks); for the 7B (int8 0 blocks; the dynamic GGUF"
        " 0 blocks); --dit-model names them"
    )
    assert [Path(load[0]).name for load in machine.loads] == ["fp16.safetensors"]
    # The GGUF's own plan: its line, and no smaller file.
    caplog.clear()
    asked = picture_command(machine, source, tmp_path / "g.png", "--dit-model", "dit.gguf")
    assert cli.main(asked) == 0
    gguf_lines = [line for line in caplog.messages if line.startswith("plan: ")]
    assert any(
        line.startswith("plan: DiT: its weights in seedvr2x's dynamic GGUF") for line in gguf_lines
    )
    assert not [line for line in gguf_lines if "smaller DiT files" in line]


def test_the_help_names_the_int8_and_the_gguf_files_and_what_they_save() -> None:
    # --dit-model's help (main._parser): the three formats, the default, every file named
    # without --model-dir, and what the GGUF saves and costs (DESIGN.md, Weights).
    parser = cli._parser()  # pyright: ignore[reportPrivateUsage]
    option = next(each for each in parser._actions if "--dit-model" in each.option_strings)  # pyright: ignore[reportPrivateUsage]
    text = " ".join((option.help or "").split())
    for name in (
        "seedvr2x_ema_7b_fp16.safetensors",
        "seedvr2x_ema_7b_int8_convrot.safetensors",
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors",
        "seedvr2x_ema_7b_dyn.gguf",
        "seedvr2x_ema_7b_sharp_dyn.gguf",
    ):
        assert name in text and name in pull.PINNED
    assert "int8 files: half the DiT's weights" in text
    assert "dynamic GGUF (under a third of fp16's weights" in text
    assert "no less VRAM where the VAE's floor binds" in text
    assert "(default: %(default)s, the sharp 7B)" in text
    assert option.default == "seedvr2x_ema_7b_sharp_fp16.safetensors"
