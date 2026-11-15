"""The command line over an int8 DiT file (DESIGN.md, Weights, "int8 before v0.1.0 is tagged"), on
the CPU, the model replaced by test_cli_run.py's stand-in but for the check of its files: a
synthetic 7B in the int8 layout and a VAE in fp16 in --model-dir (test_weights.py's int8_file and
write, core/tests/), each checked by its header, the DiT's markers read, and said in the log for
what it is; the plan made by int8's figures, its DiT line saying so; the file loaded the one
given; the job's record naming it, so that the same output asked with the fp16 file is another
job. The library's half is test_int8.py and test_weights.py's (core/tests/).

The synthetic files are sparse, 8.33 and 16.48 GB on paper: their SHA-256, which the run reads for
the job's record, is taken over their headers here (hashed), the bytes a sparse file holds."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import test_cli_run
from levels import levels
from library_fixtures import TF32_WAYS, tf32_readings, tf32_turned_on
from test_cli_run import (
    CHECK_MODELS,
    FREE,
    GIB,
    HOST,
    JOB,
    JOB_SHOTS,
    PADDED,
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
from test_weights import as_dtype, int8_file, write

import main as cli
import planner
from seedvr2x import files, model, peaks, weights
from seedvr2x.shot_plan import Pins
from seedvr2x.weights import Architecture

# The DiT's format by its file's header, as the command line reads it: the stand-in replaces it
# (test_cli_run.stand_in_model), this test takes it back.
DIT_MODEL = cli._dit_model  # pyright: ignore[reportPrivateUsage]
INT8, FP16, VAE = "int8.safetensors", "fp16.safetensors", "vae.safetensors"


def hashed(path: Path) -> str:
    """A stand-in for files.sha256 over a sparse file: the SHA-256 of its first 1 MiB and its
    size, which tell the synthetic files apart."""
    import hashlib

    with Path(path).open("rb") as file:
        head = file.read(1 << 20)
    return hashlib.sha256(head + str(Path(path).stat().st_size).encode()).hexdigest()


@pytest.mark.usefixtures("steps")
def test_an_int8_dit_runs_planned_by_its_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    int8_file(models / INT8)
    write(models / FP16, as_dtype("7b", "F16"))
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)  # the stand-in load's, the same
    loaded: list[Path] = []
    stand_in_load: Any = model.load_models

    def load(dit: Path, *arguments: Any) -> Any:
        loaded.append(dit)
        return stand_in_load(dit, *arguments)

    monkeypatch.setattr(model, "load_models", load)
    chosen = ("--model-dir", str(models), "--vae-model", VAE)
    asked = command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", INT8)
    assert cli.main(asked) == 0
    assert loaded == [models / INT8]
    said = "SeedVR2's 7B DiT in int8, W8A8 with comfy-kitchen's rotation"
    assert f"DiT {INT8}: {said}" in caplog.messages
    assert f"VAE {VAE}: SeedVR2's VAE in fp16" in caplog.messages
    # The plan by int8's figures, as the planner makes it for that job, its DiT's line saying so.
    pins = Pins(window=5)
    made = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST, dit_model=peaks.INT8)
    assert without_disk(plan_lines(caplog)) == planner.said(
        *made, "a stand-in", TARGET, JOB_SHOTS, pins
    )
    dit_line = [line for line in plan_lines(caplog) if line.startswith("plan: DiT: ")]
    assert dit_line == [
        "plan: DiT: its weights in int8, 7.76 GiB, half of fp16's; a window's memory counted as"
        " fp16's plus 0.38 GiB for int8's buffers; --dit-model names the file"
    ]
    fp16_made = planner.plan(FREE, TARGET, JOB_SHOTS, pins, HOST)
    record = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert record["plan"]["memory"]["dit_peak"] == made[1].dit < fp16_made[1].dit
    assert record["settings"]["dit_model"]["name"] == INT8
    assert record["settings"]["dit_model"]["sha256"] == hashed(models / INT8)
    # The same output with the fp16 file: another job, refused, its record left as it was.
    caplog.clear()
    again = command(tmp_path, source_path, "out", *JOB, *chosen, "--dit-model", FP16)
    assert cli.main(again) == 1
    assert "another job than the one asked" in caplog.text
    assert "settings.dit_model" in caplog.text
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == record
    # And in a directory of its own, the fp16 file runs, planned by fp16's figures.
    caplog.clear()
    loaded.clear()
    other = command(tmp_path, source_path, "fp16-out", *JOB, *chosen, "--dit-model", FP16)
    assert cli.main(other) == 0
    assert loaded == [models / FP16]
    assert f"DiT {FP16}: SeedVR2's 7B DiT in fp16" in caplog.messages
    fp16_record = json.loads((tmp_path / "fp16-out" / "manifest.json").read_text())
    assert fp16_record["plan"]["memory"]["dit_peak"] == fp16_made[1].dit


def test_the_early_judgement_reads_the_dit_s_format_where_its_file_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A resume's early judgement of the memory (cli._early_memory) plans by the DiT's format,
    # read from its file's header before the checks (cli._early_dit_model): in --model-dir, or in
    # Hugging Face's cache for one of seedvr2x's own files, found there (pull.cached) or not; no
    # format where the file isn't there, isn't one of seedvr2x's own outside --model-dir, or
    # isn't a DiT seedvr2x runs: the checks then say which.
    from argparse import Namespace

    from seedvr2x import pull

    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    early = cli._early_dit_model  # pyright: ignore[reportPrivateUsage]
    models = tmp_path / "models"
    models.mkdir()
    int8_file(models / INT8)
    write(models / VAE, as_dtype("vae", "F16"))
    assert early(Namespace(model_dir=models, dit_model=INT8)) == peaks.INT8
    assert early(Namespace(model_dir=models, dit_model="missing.safetensors")) is None
    assert early(Namespace(model_dir=models, dit_model=VAE)) is None
    asked: list[str] = []
    held: dict[str, Path] = {}

    def cached(name: str) -> Path | None:
        asked.append(name)
        return held.get(name)

    monkeypatch.setattr(pull, "cached", cached)
    own = "seedvr2x_ema_7b_int8_convrot.safetensors"
    assert own in pull.PINNED
    # Not in the cache: nothing judged.
    assert early(Namespace(model_dir=None, dit_model=own)) is None
    # In the cache: its header read there.
    held[own] = models / INT8
    assert early(Namespace(model_dir=None, dit_model=own)) == peaks.INT8
    assert asked == [own, own]
    # A name that isn't one of seedvr2x's own, without --model-dir: no cache asked.
    assert early(Namespace(model_dir=None, dit_model=INT8)) is None
    assert asked == [own, own]


def test_an_int8_resume_is_judged_early_by_int8_s_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: Steps,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An int8 job stopped, then resumed with too little free on the card by NVML: the early
    # judgement, before the checks, plans with int8's architecture (the planner's first call,
    # without the system memory, unread before torch), and warns; the plan that counts asks it
    # again with int8's too. Never the "isn't judged" debug line.
    from seedvr2x import memory

    levels(caplog, logging.DEBUG)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    int8_file(models / INT8)
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    # The model files' SHA-256 over their headers (hashed), the source's as the first pass took
    # it, which the resume compares.
    whole = files.sha256
    monkeypatch.setattr(
        files, "sha256", lambda path: hashed(path) if Path(path).parent == models else whole(path)
    )
    monkeypatch.setattr(test_cli_run, "digest_of", files.sha256)  # the stand-in load's, the same
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", INT8)
    steps.stop = "encode 4"
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 130
    steps.stop = None
    asked: list[tuple[int | None, Architecture]] = []
    plan = planner.plan

    def planned(*arguments: Any, **options: Any) -> Any:
        asked.append((arguments[4], options["dit_model"]))
        return plan(*arguments, **options)

    monkeypatch.setattr(planner, "plan", planned)
    # The plan's last rung takes a VAE pass's room down to peaks.VAE_ROOM_LEAST (patch 03): the
    # card is short of that, as test_cli_run's early-reading tests count it.
    encode = peaks.encode_peak(PADDED, None, 0, dit_model=peaks.INT8) + peaks.VAE_ROOM_LEAST
    early_card(monkeypatch, encode + planner.MARGIN - GIB // 4 + memory.CONTEXT)
    caplog.clear()
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 0
    assert asked[0] == (None, peaks.INT8)
    assert {dit_model for _, dit_model in asked} == {peaks.INT8}
    assert any("does not hold the recorded plan" in message for message in caplog.messages)
    assert not any("isn't judged before the checks" in message for message in caplog.messages)


# torch's reading of TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (weights.TF32_OVERRIDE; c10/util/env.cpp's
# check_env at v2.14.1): "1" alone forces TF32 in cuBLAS; "0", unset or any other value doesn't.
TF32_VALUES = [None, "0", "1", "", " 1", "1 ", "true", "TRUE", "yes", "01", "2"]


@pytest.mark.usefixtures("steps")
@pytest.mark.parametrize("value", TF32_VALUES)
@pytest.mark.parametrize("plan", [True, False], ids=["plan", "run"])
def test_an_int8_dit_is_refused_where_torch_s_tf32_is_forced_in_cublas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    value: str | None,
    plan: bool,
) -> None:
    # DESIGN.md, Weights, "int8 before v0.1.0 is tagged", TF32: the variable forces TF32 in
    # cuBLAS, and the W8A8 path's rotation is a float32 matrix product. An int8 job is refused
    # where torch reads the variable as set, before its first pass and any model, --plan as the
    # run, naming the variable and the way out in the command line's words; elsewhere it runs.
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    int8_file(models / INT8)
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)
    if value is None:
        monkeypatch.delenv(weights.TF32_OVERRIDE, raising=False)
    else:
        monkeypatch.setenv(weights.TF32_OVERRIDE, value)
    loaded: list[Path] = []
    stand_in_load: Any = model.load_models

    def load(dit: Path, *arguments: Any) -> Any:
        loaded.append(dit)
        return stand_in_load(dit, *arguments)

    monkeypatch.setattr(model, "load_models", load)
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", INT8)
    more = ("--plan", str(tmp_path / "plan.txt")) if plan else ()
    status = cli.main(command(tmp_path, source_path, "out", *JOB, *chosen, *more))
    if value != "1":
        assert status == 0 and "TF32" not in caplog.text
        assert loaded == [models / INT8]  # --plan's calibration loads them too
        return
    assert status == 1 and loaded == []
    refused = [
        message for message in caplog.messages if "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in message
    ]
    assert refused == [
        f"{INT8}: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 in the environment has torch turn TF32 on for"
        " cuBLAS's float32 matrix products, and the int8 DiT's rotation is one: its codes would"
        " move far beyond those seedvr2x measured, so an int8 DiT doesn't run there. Unset"
        " TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, or take an fp16 DiT file (--dit-model)"
    ]
    # Before the first pass, which reads the whole source: nothing of the job is made.
    assert not any("the first pass" in message for message in caplog.messages)
    assert not (tmp_path / "out" / "manifest.json").exists()
    assert not (tmp_path / "plan.txt").exists()


@pytest.mark.usefixtures("steps")
@pytest.mark.parametrize("way", list(TF32_WAYS))
@pytest.mark.parametrize("plan", [True, False], ids=["plan", "run"])
def test_an_int8_dit_is_refused_as_it_loads_where_the_process_turned_tf32_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    way: str,
    plan: bool,
) -> None:
    # DESIGN.md, Weights, TF32: a setting of the process's own (weights.check_tf32_set), in any of
    # torch's forms, the variable unset: an int8 job is refused as its models load, --plan's
    # calibration's load as the run's, nothing loaded and no unit made, in the command line's
    # words; no setting of torch's changed by the run.
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    int8_file(models / INT8)
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)
    monkeypatch.delenv(weights.TF32_OVERRIDE, raising=False)
    loaded: list[Path] = []
    stand_in_load: Any = model.load_models

    def load(dit: Path, *arguments: Any) -> Any:
        loaded.append(dit)
        return stand_in_load(dit, *arguments)

    monkeypatch.setattr(model, "load_models", load)
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", INT8)
    more = ("--plan", str(tmp_path / "plan.txt")) if plan else ()
    with tf32_turned_on(way) as (name, value):
        on = tf32_readings()
        status = cli.main(command(tmp_path, source_path, "out", *JOB, *chosen, *more))
        assert tf32_readings() == on
    assert status == 1 and loaded == []
    refused = [message for message in caplog.messages if "TF32" in message]
    assert len(refused) == 1 and refused[0].startswith(
        f"{INT8}: this process has torch compute CUDA's float32 matrix products in TF32 ({name}"
        f" reads {value}), and the int8 DiT's rotation is one: its codes would move far beyond"
        " those seedvr2x measured, so an int8 DiT doesn't run there. Turn TF32 off where it was"
        " turned on, or take an fp16 DiT file (--dit-model)"
    )
    assert not any("upscaled" in message for message in caplog.messages)
    # Off again, the job runs (into an output of its own: the stand-in's hash of the source is
    # another at each start).
    caplog.clear()
    more = ("--plan", str(tmp_path / "again.txt")) if plan else ()
    assert cli.main(command(tmp_path, source_path, "again", *JOB, *chosen, *more)) == 0
    assert loaded == [models / INT8] and "TF32" not in caplog.text


@pytest.mark.usefixtures("steps")
def test_an_fp16_dit_runs_where_the_process_turned_tf32_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    write(models / FP16, as_dtype("7b", "F16"))
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)
    monkeypatch.delenv(weights.TF32_OVERRIDE, raising=False)
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", FP16)
    with tf32_turned_on("allow_tf32"):
        assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 0
    assert "TF32" not in caplog.text


@pytest.mark.usefixtures("steps")
def test_an_fp16_dit_runs_where_torch_s_tf32_is_forced_in_cublas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Its matrix products are in half precision, which the variable doesn't touch: the job runs.
    levels(caplog, logging.INFO)
    source_path = job(tmp_path, monkeypatch)
    models = tmp_path / "models"
    models.mkdir()
    write(models / FP16, as_dtype("7b", "F16"))
    write(models / VAE, as_dtype("vae", "F16"))
    monkeypatch.setattr(weights, "check_models", CHECK_MODELS)
    monkeypatch.setattr(cli, "_dit_model", DIT_MODEL)
    monkeypatch.setattr(files, "sha256", hashed)
    monkeypatch.setattr(test_cli_run, "digest_of", hashed)
    monkeypatch.setenv(weights.TF32_OVERRIDE, "1")
    chosen = ("--model-dir", str(models), "--vae-model", VAE, "--dit-model", FP16)
    assert cli.main(command(tmp_path, source_path, "out", *JOB, *chosen)) == 0
    assert "TF32" not in caplog.text
