"""The server's service (service.py) on the CPU with the stand-in models (serving.py): a picture's
three steps under the models' locks, pictures at once, the warm-up, the start refused where its
reservation doesn't fit, and the model files' hashes."""

import contextlib
import hashlib
import io
import logging
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from serving import (
    BOUND,
    GIB,
    HOST,
    INPUT,
    Gate,
    StandIn,
    library_file,
    live,
    options,
    png_of,
)
from test_still import pipeline_models

import admission
import answers
import main
import reservation
import service as services
from metrics import Metrics
from seedvr2x import (
    files,
    frames,
    memory,
    model,
    peaks,
    picture,
    pull,
    still,
    still_plan,
    waiting,
    weights,
)
from seedvr2x.errors import Refusal


def barrier_hook(parties: int) -> Callable[[], None]:
    """A `StandIn.fail` hook that passes once `parties` pictures are inside it at the same moment
    (BrokenBarrierError, an error of the picture, after 30 s). It sits in `StandIn.watched`,
    before the models' locks: it shows the pictures admitted together, not that they overlap in
    the models (the test of the passes below does)."""
    barrier = threading.Barrier(parties, timeout=30)

    def hook() -> None:
        barrier.wait()

    return hook


# ---- A picture's steps ----------------------------------------------------------------------


def test_the_three_steps_under_the_locks_give_the_librarys_bits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """service.prepare, then run (encode, DiT pass and decode each under its model's lock), with
    no HTTP: the pixels are still.upscale's on the same models, bit for bit, the file the
    library's."""
    with live(monkeypatch, tmp_path, cpu_models) as server:
        service = server.service
        assert service.made is not None
        data = png_of(40, 24, seed=6)
        prepared = service.prepare(
            data, answers.Settings(48, 5), answers.Formats(answers.PNG, None)
        )
        assert prepared.seed == 5
        assert prepared.format == answers.PNG
        pixels = service._upscaled(prepared)
        expected = still.upscale(
            pipeline_models(cpu_models),
            prepared.picture.pixels,
            prepared.target,
            5,
            encode_tiles=prepared.tiles.encode,
            decode_tiles=prepared.tiles.decode,
            correct=True,
        )
        assert pixels.dtype == np.float32
        assert np.array_equal(pixels, expected)
        made = service.run(prepared)
        assert made.output == (80, 48)
        assert made.data == library_file(server, cpu_models, data, 48, 5)


def test_a_tiled_picture_is_the_librarys_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The plan's tiles are the picture's: both passes pinned to 512 px, the smallest tile, on an
    output of 576 columns, so that they are tiled in system memory; the file is the library's with
    the same tiles."""
    pins = still_plan.Pins(encode_tile=512, decode_tile=512)
    with live(
        monkeypatch, tmp_path, cpu_models, pins=pins, bound=(576, 64), input_pixels=10_000
    ) as server:
        service = server.service
        data = png_of(288, 32, seed=3)
        prepared = service.prepare(
            data, answers.Settings(64, 2), answers.Formats(answers.PNG, None)
        )
        assert prepared.tiles.encode is not None and prepared.tiles.decode is not None
        assert service.run(prepared).data == library_file(server, cpu_models, data, 64, 2)


# ---- Pictures at once -------------------------------------------------------------------------


def test_two_pictures_run_at_once_and_each_is_what_it_is_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, count=2) as server:
        first, second = png_of(40, 24, 1), png_of(36, 28, 2)
        server.machine.most_inside = 0  # the warm-up's two were at once
        alone = [server.upscale(first, 48, 1), server.upscale(second, 48, 2)]
        assert [each[0] for each in alone] == [200, 200]
        assert server.machine.most_inside == 1
        # Both inside their passes at the same moment, or the barrier breaks and a picture fails.
        server.machine.fail = barrier_hook(2)
        with ThreadPoolExecutor(2) as pool:
            together = [
                pool.submit(server.upscale, first, 48, 1),
                pool.submit(server.upscale, second, 48, 2),
            ]
            results = [each.result(120) for each in together]
        assert server.machine.most_inside == 2
        for (status, _, body), (_, _, before) in zip(results, alone, strict=True):
            assert status == 200, body
            assert body == before


class Passes:
    """The three steps of a picture (still.encoded, still.sampled, still.decoded) wrapped inside
    the models' locks, the service calling them there: how many pictures are in the VAE and in the
    DiT at a moment, the most there were, and the overlap of one in the DiT and one in the VAE.

    With `overlap` set, the second entry into the VAE waits (30 s) for a picture to be in the DiT,
    and the first into the DiT waits for that second one to be in the VAE: the pictures can finish
    only if one was in each model at the same moment, and a timeout is a picture that failed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, overlap: bool, slow: float = 0.0):
        self.vae = self.dit = 0
        self.most_vae = self.most_dit = 0
        self.in_vae_during_dit = False
        self.vae_entries = self.dit_entries = 0
        self._lock = threading.Lock()
        self._dit_in = threading.Event()
        self._vae_in = threading.Event()
        self._overlap = overlap
        self._slow = slow
        for name in ("encoded", "decoded"):
            monkeypatch.setattr(still, name, self._vae_pass(getattr(still, name)))
        monkeypatch.setattr(still, "sampled", self._dit_pass(still.sampled))

    def _vae_pass(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.vae_entries += 1
                second = self.vae_entries == 2
            if self._overlap and second:
                assert self._dit_in.wait(30), "no picture was ever in the DiT beside this one"
            with self._lock:
                self.vae += 1
                self.most_vae = max(self.most_vae, self.vae)
                if self.dit:
                    self.in_vae_during_dit = True
                    self._vae_in.set()
            try:
                time.sleep(self._slow)
                return original(*args, **kwargs)
            finally:
                with self._lock:
                    self.vae -= 1

        return wrapper

    def _dit_pass(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.dit += 1
                self.dit_entries += 1
                first = self.dit_entries == 1
                self.most_dit = max(self.most_dit, self.dit)
            self._dit_in.set()
            try:
                if self._overlap and first:
                    assert self._vae_in.wait(30), "no picture was ever in the VAE beside this one"
                return original(*args, **kwargs)
            finally:
                with self._lock:
                    self.dit -= 1

        return wrapper


def test_a_picture_is_in_the_dit_while_another_is_in_the_vae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """Two pictures at once overlap across the models: one inside still.sampled (the DiT's lock)
    while the other is inside still.encoded or still.decoded (the VAE's), at the same moment. The
    wrappers make the pictures finish only if they did (Passes), and each file is the one it
    makes alone."""
    with live(monkeypatch, tmp_path, cpu_models, count=2) as server:
        first, second = png_of(40, 24, 1), png_of(36, 28, 2)
        alone = [server.upscale(first, 48, 1), server.upscale(second, 48, 2)]
        passes = Passes(monkeypatch, overlap=True)
        with ThreadPoolExecutor(2) as pool:
            together = [
                pool.submit(server.upscale, first, 48, 1),
                pool.submit(server.upscale, second, 48, 2),
            ]
            results = [each.result(120) for each in together]
        for (status, _, body), (_, _, before) in zip(results, alone, strict=True):
            assert status == 200, body
            assert body == before
        assert passes.in_vae_during_dit
        # Two pictures, an encode and a decode each, one DiT pass each; never two in a model.
        assert (passes.vae_entries, passes.dit_entries) == (4, 2)
        assert passes.most_vae == 1 and passes.most_dit == 1


def test_two_pictures_are_never_in_the_vae_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The VAE keeps its causal memory in its modules: its lock holds the encodes and decodes of
    all the pictures one at a time, counted inside the wrappers with a pause in them that would
    let a second picture in. The control takes the lock off and sees two."""
    with live(monkeypatch, tmp_path, cpu_models, count=4, queue=8) as server:
        passes = Passes(monkeypatch, overlap=False, slow=0.05)
        data = [png_of(40, 24, seed) for seed in range(4)]
        with ThreadPoolExecutor(4) as pool:
            sent = [pool.submit(server.upscale, each, 48, seed) for seed, each in enumerate(data)]
            statuses = [each.result(120)[0] for each in sent]
        assert statuses == [200] * 4
        assert passes.vae_entries == 8 and passes.dit_entries == 4
        assert passes.most_vae == 1 and passes.most_dit == 1
        # The control: no lock, and the same pictures are in the VAE together.
        server.service._vae = contextlib.nullcontext()  # pyright: ignore[reportAttributeAccessIssue]
        passes.most_vae = 0
        with ThreadPoolExecutor(4) as pool:
            sent = [pool.submit(server.upscale, each, 48, seed) for seed, each in enumerate(data)]
            for each in sent:
                each.result(120)  # whatever they answer: the modules' memory is shared here
        assert passes.most_vae >= 2


def test_one_picture_at_a_time_never_runs_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=8) as server:
        data = png_of(40, 24)
        with ThreadPoolExecutor(3) as pool:
            sent = [pool.submit(server.upscale, data, 48, seed) for seed in (1, 2, 3)]
            statuses = [each.result(120)[0] for each in sent]
        assert statuses == [200, 200, 200]
        assert server.machine.most_inside == 1


# ---- The warm-up ------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2])
def test_the_warm_up_runs_count_pictures_in_landscape_and_in_portrait_through_admission(
    count: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
) -> None:
    seen: list[tuple[bool, services.Prepared]] = []
    entered: list[int] = []
    run = services.Service.run
    enter = admission.Admission.enter

    def spying_run(
        self: services.Service,
        prepared: services.Prepared,
        gone: Callable[[], bool] = lambda: False,
        warm: bool = False,
    ) -> services.Made:
        seen.append((warm, prepared))
        return run(self, prepared, gone, warm)

    def spying_enter(
        self: admission.Admission, need: int, gone: Callable[[], bool] = lambda: False
    ) -> admission.Ticket:
        entered.append(need)
        return enter(self, need, gone)

    monkeypatch.setattr(services.Service, "run", spying_run)
    monkeypatch.setattr(admission.Admission, "enter", spying_enter)
    # The pictures are inside their passes together, or the barrier breaks and the start fails.
    machine = StandIn(cpu_models, fail=barrier_hook(count))
    with live(monkeypatch, tmp_path, cpu_models, count=count, machine=machine) as server:
        # Each orientation, `count` pictures at once: the landscape ones first, then the portrait.
        assert len(seen) == 2 * count == len(entered)
        assert all(warm for warm, _ in seen)
        assert machine.most_inside == count
        landscape, portrait = seen[:count], seen[count:]
        assert len({prepared.seed for _, prepared in landscape}) == count
        assert len({prepared.seed for _, prepared in portrait}) == count
        for _, prepared in landscape:
            height, width = prepared.picture.pixels.shape[:2]
            assert width * height <= INPUT
            # The bound's own ratio, a whole multiple of its smallest picture (4x3 for 128x96:
            # 144x108, 15,552 px of the 16,000), so that the output is the bound itself.
            assert (width, height) == (144, 108)
            rows, columns = frames.output_size(prepared.target)
            assert (columns, rows) == BOUND
        for _, prepared in portrait:
            # The same picture turned, and the bound turned: it pads to other rows and columns,
            # so that it has other tiles (still_plan.tiles).
            height, width = prepared.picture.pixels.shape[:2]
            assert (width, height) == (108, 144)
            rows, columns = frames.output_size(prepared.target)
            assert (columns, rows) == BOUND[::-1]
        # No request made them: the counts a scrape reads are requests'.
        assert server.service.state is services.State.READY
        assert (
            server.service.metrics.text(server.service.gauges).count("upscale_seconds_count 0") == 1
        )


class Steps:
    """The three steps of a picture (still.encoded, still.sampled, still.decoded) taking the time
    each call is given from `seconds` on a clock waiting.monotonic reads, and waiting.named
    recording what it is given: (name, estimate, way). Nothing sleeps."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, seconds: dict[str, list[float]]) -> None:
        self.now = 1000.0
        self.named: list[tuple[str, float | None, str]] = []
        self._seconds = {name: list(each) for name, each in seconds.items()}
        named = waiting.named
        monkeypatch.setattr(waiting, "monotonic", lambda: self.now)

        def recording(name: str, estimate: float | None, way: str = waiting.WAY) -> Any:
            self.named.append((name, estimate, way))
            return named(name, estimate, way)

        monkeypatch.setattr(waiting, "named", recording)
        for name in ("encoded", "sampled", "decoded"):
            monkeypatch.setattr(still, name, self._step(name, getattr(still, name)))

    def _step(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            # The step's time is the clock's move while it runs; the last value repeats.
            each = self._seconds[name]
            self.now += each.pop(0) if len(each) > 1 else each[0]
            return out

        return wrapper


def test_the_warm_ups_longest_time_of_each_step_is_the_estimate_of_every_later_picture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The warm-up runs the landscape and the portrait picture (count 1, so in turn): each step
    keeps the longer of its two times, and a request's step is named with it. The warm-up's own
    steps are named with no estimate, so that no wait in it says anything."""
    steps = Steps(
        monkeypatch,
        {"encoded": [2.0, 5.0], "sampled": [10.0, 7.0], "decoded": [3.0]},
    )
    with live(monkeypatch, tmp_path, cpu_models) as server:
        assert server.service.estimates == {
            services.ENCODE: 5.0,
            services.SAMPLING: 10.0,
            services.DECODE: 3.0,
        }
        warm = list(steps.named)
        assert [(name, estimate) for name, estimate, _ in warm] == [
            (services.ENCODE, None),
            (services.SAMPLING, None),
            (services.DECODE, None),
        ] * 2
        steps.named.clear()
        assert server.upscale(png_of(40, 24))[0] == 200
        assert [(name, estimate) for name, estimate, _ in steps.named] == [
            (services.ENCODE, 5.0),
            (services.SAMPLING, 10.0),
            (services.DECODE, 3.0),
        ]
        # A request's own times change nothing: the estimate is the warm-up's.
        assert server.service.estimates[services.ENCODE] == 5.0
        assert {way for _, _, way in steps.named + warm} == {services.WAY}


def test_a_step_is_named_in_the_thread_that_runs_it_and_the_name_goes_after_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The unit named for a thread's waits is the step's inside it and what it was before after
    it (waiting.named), so that a wait in the models' placement between pictures says nothing."""
    seen: list[str | None] = []

    def seeing(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current = getattr(waiting._named, "unit", None)  # pyright: ignore[reportPrivateUsage]
            seen.append(None if current is None else current.name)
            return original(*args, **kwargs)

        return wrapper

    for name in ("encoded", "sampled", "decoded"):
        monkeypatch.setattr(still, name, seeing(getattr(still, name)))
    with live(monkeypatch, tmp_path, cpu_models) as server:
        assert seen == [services.ENCODE, services.SAMPLING, services.DECODE] * 2
        assert getattr(waiting._named, "unit", None) is None  # pyright: ignore[reportPrivateUsage]
        assert server.service.estimates.keys() == {
            services.ENCODE,
            services.SAMPLING,
            services.DECODE,
        }


def test_stopping_and_failed_win_over_stalled_in_health(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    monkeypatch.setattr(waiting, "stalled", lambda: True)
    with live(monkeypatch, tmp_path, cpu_models, ready=False) as server:
        service = server.service
        # Starting: a probe waits for the warm-up, a stall of the warm-up's steps is not said.
        assert service.health() is services.State.STARTING
        service.state = services.State.READY
        assert service.health() is services.State.STALLED
        service.state = services.State.STOPPING
        assert service.health() is services.State.STOPPING
        service.state = services.State.FAILED
        assert service.health() is services.State.FAILED


# ---- A start refused --------------------------------------------------------------------------


def test_a_reservation_that_does_not_fit_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    machine = StandIn(cpu_models, free_bytes=GIB)
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine) as server:
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
        assert refused.value.most == 0
        assert "no picture fits" in str(refused.value) or "what would fit" in str(refused.value)
        assert machine.reserves == []
        assert server.service.state is services.State.STARTING
        assert server.request("GET", "/health")[0] == 503


@pytest.mark.parametrize("free", [14, 18])
def test_pictures_at_once_on_a_card_that_swaps_are_refused_naming_what_fits(
    free: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """14 GiB swaps blocks for one picture, 18 keeps them all but moves the VAE: neither holds two
    pictures at once, which need the whole DiT and the VAE on the card."""
    machine = StandIn(cpu_models, free_bytes=free * GIB)
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine, count=2) as server:
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
        text = str(refused.value)
        assert refused.value.most == 1
        assert "one picture at once within 128x96 (--max-concurrent)" in text
        assert server.request("GET", "/health")[0] == 503
    # One picture at a time starts on the same card.
    one = StandIn(cpu_models, free_bytes=free * GIB)
    with live(monkeypatch, tmp_path, cpu_models, machine=one, count=1) as server:
        assert server.service.state is services.State.READY


@pytest.mark.parametrize("dit_format", [peaks.INT8, peaks.GGUF_7B_SHARP])
def test_a_start_the_fp16_dit_is_refused_names_the_smaller_files_and_they_start_it(
    dit_format: weights.Architecture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
) -> None:
    """DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too": two pictures at once on 14 GiB
    are refused for the fp16 DiT, and the refusal names the int8 and GGUF files of both 7Bs (the
    stand-in's file name is no pinned one) with their sizes (reservation.smaller); the server
    started on either file's format, the stand-in's, starts on the same card, and the file it was
    given is never changed by itself."""
    machine = StandIn(cpu_models, free_bytes=14 * GIB)
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine, count=2) as server:
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
        assert str(refused.value).endswith(
            "; or a smaller DiT file that plans this bound, put in --model-dir: for the sharp 7B,"
            " int8 (--dit-model"
            " seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF"
            " (--dit-model seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB); for the 7B, int8 (--dit-model"
            " seedvr2x_ema_7b_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF (--dit-model"
            " seedvr2x_ema_7b_dyn.gguf, 4.8 GB)"
        )
        assert server.service.options.dit == "w.safetensors"
        assert machine.reserves == []
    smaller = StandIn(cpu_models, free_bytes=14 * GIB, dit_format=dit_format)
    with live(monkeypatch, tmp_path, cpu_models, machine=smaller, count=2) as server:
        assert server.service.state is services.State.READY
        assert server.service.made is not None
        assert server.service.made.plan.dit_model == dit_format


def test_a_dit_given_by_an_absolute_path_is_known_by_its_file_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The reservation looks the DiT up by its file name, as the command line does (dit.name):
    a pinned file of the sharp 7B given as an absolute path gets the sharp 7B's smaller files
    alone, unmarked, not both 7Bs' as an unknown file does. An absolute path is read where
    --model-dir is given (the directory joined to it), so the way out says to put the file
    there."""
    name = "seedvr2x_ema_7b_sharp_fp16.safetensors"
    directory = tmp_path / "elsewhere"
    directory.mkdir()
    (directory / name).write_bytes(b"")
    monkeypatch.setattr(pull, "check_pinned", lambda *args, **kwargs: None)
    machine = StandIn(cpu_models, free_bytes=14 * GIB)
    with live(
        monkeypatch, tmp_path, cpu_models, ready=False, machine=machine, count=2,
        dit=str(directory / name),
    ) as server:  # fmt: skip
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
    text = str(refused.value)
    assert text.endswith(
        "; or a smaller DiT file that plans this bound, put in --model-dir: int8 (--dit-model"
        " seedvr2x_ema_7b_sharp_int8_convrot.safetensors, 8.3 GB) or the dynamic GGUF"
        " (--dit-model seedvr2x_ema_7b_sharp_dyn.gguf, 4.8 GB)"
    ), text
    assert "for the sharp 7B" not in text and "seedvr2x_ema_7b_dyn.gguf" not in text


def test_the_refusal_of_pictures_at_once_names_no_option_the_server_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    machine = StandIn(cpu_models, free_bytes=14 * GIB)
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine, count=2) as server:
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
        assert "--swap-blocks" not in str(refused.value)
        assert "need the whole DiT on the card" in str(refused.value)


# ---- The model files --------------------------------------------------------------------------


def test_the_model_files_are_read_by_their_load_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    # DESIGN.md, Weights (decided on 2026-11-03): the start reads no model file whole; the
    # load hashes each as it reads it, and the record holds the SHA-256 of the bytes it read,
    # once for the server's life.
    directory = tmp_path / "files"
    directory.mkdir()
    (directory / "dit.safetensors").write_bytes(b"the dit")
    (directory / "vae.safetensors").write_bytes(b"the vae")
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    read: list[Path] = []
    sha256: Callable[[Path], str] = files.sha256

    def counting(path: Path) -> str:
        read.append(path)
        return sha256(path)

    monkeypatch.setattr(files, "sha256", counting)
    given = options(tmp_path, model_dir=directory, dit="dit.safetensors", vae="vae.safetensors")
    service = services.Service(given, StandIn(cpu_models))
    checked = service.check_files()
    assert read == [] and service.record_models is None
    StandIn(cpu_models).load(checked, None, 0)  # pyright: ignore[reportArgumentType]
    assert [checked.hashes.sha256(each) for each in (checked.dit.path, checked.vae.path)] == [
        hashlib.sha256(b"the dit").hexdigest(),
        hashlib.sha256(b"the vae").hexdigest(),
    ]
    assert read == []


def test_the_record_holds_the_hashes_of_the_files_the_server_started_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    # The SHA-256 of the bytes the start's load read, no file read whole besides.
    read: list[Path] = []
    monkeypatch.setattr(files, "sha256", lambda path: read.append(path) or "")
    with live(monkeypatch, tmp_path, cpu_models) as server:
        assert read == []
        assert server.service.record_models is not None
        empty = hashlib.sha256(b"").hexdigest()
        assert [each.sha256 for each in server.service.record_models] == [empty, empty]


# ---- The reservation held ---------------------------------------------------------------------


def test_the_reservation_is_made_again_when_torch_holds_less_and_no_picture_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        machine, service = server.machine, server.service
        assert service.made is not None
        wanted = service.made.reservation
        before = len(machine.reserves)
        # A picture with the reservation whole: torch holds all of it, nothing is made again.
        assert server.upscale(png_of(40, 24), 48, 1)[0] == 200
        assert len(machine.reserves) == before
        # Torch gave a GiB back to the card during the picture: made again as it leaves.
        machine.fail = lambda: setattr(machine, "lost", GIB)
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            assert server.upscale(png_of(40, 24), 48, 2)[0] == 200
        assert machine.reserves[before:] == [wanted]
        assert machine.reserved() == wanted
        assert "the reservation held again" in caplog.text
        # And after the warm-up, directly: the hold made again with nothing running.
        machine.fail = None
        machine.lost = 2 * GIB
        service._held()
        assert machine.reserves[before:] == [wanted, wanted]
        assert machine.reserved() == wanted


def test_a_hold_a_step_short_of_the_reservation_is_whole_and_said_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """torch reserves in steps, so a hold up to memory.MIN_LARGE + ROUND_LARGE under a reservation
    that is no multiple of one is whole: not made again, nothing said (on the RTX 5090, int8, it
    was 0.2 to 2.2 MiB under after each picture, and the line came 12 times in 12 pictures)."""
    with live(monkeypatch, tmp_path, cpu_models) as server:
        machine = server.machine
        before = len(machine.reserves)
        machine.fail = lambda: setattr(machine, "lost", 2 << 20)
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            assert server.upscale(png_of(40, 24), 48, 1)[0] == 200
        assert len(machine.reserves) == before
        assert "the reservation held again" not in caplog.text
        machine.fail = None
        machine.lost = memory.MIN_LARGE + memory.ROUND_LARGE + 1
        server.service._held()
        assert len(machine.reserves) == before + 1


def test_the_warm_up_leaves_the_hold_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    machine = StandIn(cpu_models)
    # Torch gives memory back during the warm-up's first picture; the start ends with it held.
    machine.fail = lambda: setattr(machine, "lost", GIB)
    with live(monkeypatch, tmp_path, cpu_models, machine=machine) as server:
        made = server.service.made
        assert made is not None
        assert len(machine.reserves) >= 2  # the load's, then the warm-up's pictures leaving
        assert machine.reserved() == made.reservation


def test_the_hold_is_not_made_again_while_a_picture_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Another picture running means the allocation of the missing block could starve it
    (Admission.when_idle): the picture that leaves first makes nothing again, the last one does."""
    with live(monkeypatch, tmp_path, cpu_models, count=2) as server:
        machine, service = server.machine, server.service
        assert service.made is not None
        wanted = service.made.reservation
        before = len(machine.reserves)
        gate = Gate()
        machine.fail = gate
        with ThreadPoolExecutor(2) as pool:
            running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert gate.entered.wait(60)
            machine.lost = GIB
            assert server.upscale(png_of(36, 28, 2), 48, 2)[0] == 200
            assert machine.reserves[before:] == []  # one is still running
            assert machine.reserved() == wanted - GIB
            with caplog.at_level(logging.INFO, logger="imgserver.service"):
                gate.release.set()
                assert running.result(60)[0] == 200
        assert machine.reserves[before:] == [wanted]
        assert machine.reserved() == wanted


def test_a_hold_the_card_cannot_give_again_is_a_warning_and_the_server_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        machine = server.machine
        machine.refuse = MemoryError("another program holds it")
        machine.fail = lambda: setattr(machine, "lost", GIB)
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            assert server.upscale(png_of(40, 24), 48, 1)[0] == 200
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name == "imgserver.service"
        ]
        assert len(warnings) == 1
        assert "couldn't give the rest back" in warnings[0].getMessage()
        assert "another program holds it" in warnings[0].getMessage()
        assert "the reservation held again" not in caplog.text
        assert server.request("GET", "/health")[0] == 200
        machine.refuse, machine.fail = None, None
        assert server.upscale(png_of(40, 24), 48, 1)[0] == 200


# ---- A share the pool cannot hold ---------------------------------------------------------------


def test_a_picture_whose_share_passes_the_pool_is_a_500_never_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        gate = Gate(blocking=0)
        server.machine.fail = gate
        admitted = server.service.admitted
        assert admitted is not None
        with monkeypatch.context() as huge:
            huge.setattr(reservation, "share", lambda found: admitted.pool + 1)
            with caplog.at_level(logging.ERROR, logger="imgserver.service"):
                status, _, body = server.upscale(png_of(40, 24), 48, 1)
        assert status == 500
        assert "plan doesn't hold it" in body.decode()
        assert "passes the pool's" in caplog.text and "the plan is wrong" in caplog.text
        assert gate.calls == 0  # the GPU was never asked
        assert admitted.running == 0 and admitted.waiting == 0
        assert server.request("GET", "/health")[0] == 200
        # The share is the plan's again: the same picture is run.
        assert server.upscale(png_of(40, 24), 48, 1)[0] == 200
        assert gate.calls == 1


# ---- What the size of a picture is read with ---------------------------------------------------


def saved(image: Image.Image, format_: str, **given: Any) -> bytes:
    out = io.BytesIO()
    image.save(out, format_, **given)
    return out.getvalue()


@pytest.mark.parametrize(
    ("format_", "given"), [("ICO", {"sizes": [(32, 32)]}), ("PCX", {}), ("PPM", {})]
)
def test_the_stored_size_is_read_in_the_formats_seedvr2x_reads_only(
    format_: str, given: dict[str, Any]
) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    data = saved(Image.fromarray(pixels, "RGB"), format_, **given)
    # Pillow opens it, and the header parser would be exposed to bytes seedvr2x never reads.
    assert Image.open(io.BytesIO(data)).size == (32, 32)
    assert format_ not in picture.FORMATS
    assert services._stored_size(data) is None
    assert services._stored_size(png_of(40, 24)) == (40, 24)


def test_a_picture_in_a_format_seedvr2x_does_not_read_is_a_422_from_the_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (24, 40, 3), dtype=np.uint8)
    data = saved(Image.fromarray(pixels, "RGB"), "PCX")
    with live(monkeypatch, tmp_path, cpu_models) as server:
        gate = Gate(blocking=0)
        server.machine.fail = gate
        status, _, body = server.upscale(data, 48)
        assert status == 422, body
        assert body.strip()
        assert gate.calls == 0


# ---- WebP's side limit, before the GPU ----------------------------------------------------------


def wide_service(server: Any) -> services.Service:
    """The service with the plan of a bound wide enough for an output past WebP's 16,383 px a
    side, 16,400x64, made by the planner on a 96 GiB card as the start makes it (the stand-in's
    models could not run such a picture, and these tests prepare and never run)."""
    service = server.service
    service.made = reservation.make(
        reservation.Asked(
            (16400, 64),
            INPUT,
            1,
            8,
            memory.reading(96 * GIB),
            HOST,
            peaks.FLASH_ATTN_2,
            peaks.FP16,
        )
    )
    return service


def test_an_output_side_over_webps_is_a_png_where_the_client_takes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    from seedvr2x import webp

    with live(monkeypatch, tmp_path, cpu_models) as server:
        gate = Gate(blocking=0)
        server.machine.fail = gate
        service = wide_service(server)
        data = png_of(1025, 4)
        settings = answers.Settings(64, 1)
        # 1025x4 at 64 px short side: a 16400x64 output, past WebP's 16,383.
        png_ok = service.prepare(data, settings, answers.Formats(answers.WEBP, answers.PNG))
        assert png_ok.format == answers.PNG
        assert max(frames.output_size(png_ok.target)) > webp.MAX_SIDE
        with pytest.raises(answers.NotAcceptable) as refused:
            service.prepare(data, settings, answers.Formats(answers.WEBP, None))
        assert str(webp.MAX_SIDE) in str(refused.value) and "WebP" in str(refused.value)
        # A PNG is asked for outright: unchanged. Within the limit a WebP stays one.
        assert service.prepare(data, settings, answers.Formats(answers.PNG, None)).format == (
            answers.PNG
        )
        small = service.prepare(
            png_of(40, 24), answers.Settings(48, 1), answers.Formats(answers.WEBP, answers.PNG)
        )
        assert small.format == answers.WEBP
        assert gate.calls == 0  # nothing ran on the GPU


# ---- A CUDA allocation that fails ---------------------------------------------------------------


def test_an_allocation_cublas_refuses_is_a_500_and_the_server_stays_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        calls: list[int] = []

        def hook() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)")

        server.machine.fail = hook
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            status, _, body = server.upscale(png_of(40, 24), 48, 1)
        assert status == 500
        assert "ran out of the GPU's memory" in body.decode()
        assert "CUBLAS_STATUS_ALLOC_FAILED" in caplog.text
        assert server.service.state is services.State.READY
        assert server.request("GET", "/health")[0] == 200
        assert server.upscale(png_of(40, 24), 48, 1)[0] == 200


def card_without_cuda() -> services.Card:
    """service.Card with torch and no device: `fatal` reads the error and torch's classes."""
    card = object.__new__(services.Card)
    card.torch = torch  # pyright: ignore[reportAttributeAccessIssue]
    return card


@pytest.mark.parametrize(
    "text",
    [
        "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_the_real_card_does_not_call_an_allocation_failure_fatal(text: str) -> None:
    assert card_without_cuda().fatal(RuntimeError(text)) is False


def test_a_cublas_allocation_failure_leaves_the_server_ready_with_the_real_cards_judgement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models) as server:
        real = card_without_cuda()
        monkeypatch.setattr(server.machine, "fatal", real.fatal)
        calls: list[int] = []

        def hook() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)")

        server.machine.fail = hook
        assert server.upscale(png_of(40, 24), 48, 1)[0] == 500
        assert server.service.state is services.State.READY
        assert server.request("GET", "/health")[0] == 200


@pytest.mark.parametrize(
    "text",
    [
        "CUDA error: an illegal memory access was encountered",
        "CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx",
        "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
    ],
)
def test_the_real_card_calls_a_broken_card_fatal(text: str) -> None:
    card = card_without_cuda()
    assert card.fatal(RuntimeError(text)) is True
    assert card.fatal(torch.OutOfMemoryError("CUDA out of memory")) is False
    assert card.fatal(ValueError("a bug")) is False


# ---- The start's refusals and lines ------------------------------------------------------------


def test_a_reservation_the_card_refuses_at_the_start_names_it_and_the_options_to_lower(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    machine = StandIn(cpu_models, refuse=MemoryError("CUDA out of memory"))
    with live(monkeypatch, tmp_path, cpu_models, ready=False, machine=machine) as server:
        with pytest.raises(reservation.ReservationRefused) as refused:
            main.bring_up(server.running)
        made = server.service.made
        assert made is not None
        text = str(refused.value)
        assert f"{reservation.gib(made.reservation)} GiB" in text
        for option in ("--max-size", "--max-input", "--max-concurrent"):
            assert option in text
        assert isinstance(refused.value, Refusal)
        assert machine.reserves == []
        assert server.service.state is services.State.STARTING
        assert server.service.admitted is None


@pytest.mark.parametrize(
    ("count", "said"),
    [
        (1, "one picture at a time"),
        (2, "two on the GPU at a time, one in each model"),
    ],
)
def test_the_reservations_lines_say_how_many_pictures_the_gpu_runs(
    count: int,
    said: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="imgserver.service"):
        with live(monkeypatch, tmp_path, cpu_models, count=count):
            pass
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reservation:")]
    fits = [r.getMessage() for r in caplog.records if r.getMessage().startswith("what fits:")]
    assert len(lines) == 1 and len(fits) == 1
    assert said in lines[0] + fits[0]
    assert "GiB held for torch on the GPU and capped there" in lines[0]
    if count == 2:
        assert "2 pictures in flight" in fits[0]
        assert "one picture in the VAE and one in the DiT" in lines[0]
    else:
        assert "two on the GPU" not in lines[0] + fits[0]


@pytest.mark.parametrize("allocator", [None, memory.SEGMENTS, memory.POOL, memory.PLAIN], ids=str)
def test_the_start_holds_torch_to_the_plans_reserve_of_each_step_and_says_it(
    allocator: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # memory.hold gets the plan's reserve by step (still_plan.reserve_taken): each phase's own
    # beyond its peak under the segments and the pool, the flat memory.RESERVE under torch's own
    # allocator without the segments, none on a driver that refuses; the start's line says the
    # per-step figures only where they are the plan's rule, in the server's words.
    machine = StandIn(cpu_models, allocator=allocator)
    with caplog.at_level(logging.INFO, logger="imgserver.service"):
        with live(monkeypatch, tmp_path, cpu_models, machine=machine) as server:
            made = server.service.made
    assert made is not None
    assert machine.holds == [still_plan.reserve_taken(made.plan)]
    said = [r.getMessage() for r in caplog.records if "the reserve beyond it" in r.getMessage()]
    if allocator is None:
        assert machine.holds == [None] and not said
    elif allocator == memory.PLAIN:
        assert machine.holds == [dict.fromkeys(peaks.PHASES, memory.RESERVE)] and not said
    else:
        held = machine.holds[0]
        assert isinstance(held, dict) and set(held) == set(peaks.PHASES)
        assert all(0 < held[phase] < memory.RESERVE for phase in peaks.PHASES)
        assert len(said) == 1
        for phase, name in (("dit", "DiT"), ("encode", "encode"), ("decode", "decode")):
            assert f"{held[phase] / memory.GIB:.2f} GiB for the {name}" in said[0] or (
                f"{held[phase] / memory.GIB:.2f} for the {name}" in said[0]
            ), (phase, said[0])
        # The reservation is each phase's peak and its reserve, no share added over the peak.
        assert all(each.over == 0.0 for each in still_plan.bound_peaks(made.plan).values())


@pytest.mark.parametrize("dit", [peaks.INT8, peaks.FP16])
def test_the_reservations_line_says_the_tiles_pictures_at_once_take(
    dit: weights.Architecture, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Two pictures within 1920x1080 on the RTX 5090 under WSL2 (2026-11-04: 18.45 GiB counted of
    # 23.45 free on the whole card), the int8 DiT kept whole by the VAE's passes in tiles of 1536
    # px (reservation._made): the line says so. One picture at a time takes the plan's own
    # tiles, and its line says none.
    held = memory.Reading(
        round(30.03 * GIB), round(23.45 * GIB), round(18.45 * GIB), True, left=memory.HELD_BACK
    )
    for count in (2, 1):
        if dit == peaks.FP16 and count == 2:
            continue  # refused (tests/test_reservation.py)
        asked = reservation.Asked(
            (1920, 1080), 1024 * 576, count, 8, held, 64 * GIB, peaks.SDPA, dit
        )
        server = services.Service(
            options(tmp_path, bound=(1920, 1080), input_pixels=1024 * 576, count=count), None
        )
        server.made = reservation.make(asked)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            server._say_reservation()  # pyright: ignore[reportPrivateUsage]
        line = caplog.messages[0]
        tiles = (
            ", the VAE's passes in tiles of 1536 px (untiled where a picture's frame is no"
            " larger), so that the DiT stays whole beside them; the CUDA context"
        )
        assert (tiles in line) == (count == 2), line
        assert "tiles of" not in line or count == 2


# ---- It listens at once -------------------------------------------------------------------------


def test_the_server_listens_before_the_card_is_read_and_bring_up_makes_it_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    from test_still import stand_in_dit

    stand_in_dit(monkeypatch)
    running = main.launch(options(tmp_path), None, ("127.0.0.1", 0), None, Metrics())
    try:
        service = running.service
        assert service.machine is None
        host, port = running.address
        from http.client import HTTPConnection

        def get(path: str) -> tuple[int, bytes]:
            connection = HTTPConnection(host, port, timeout=30)
            try:
                connection.request("GET", path)
                answer = connection.getresponse()
                return answer.status, answer.read()
            finally:
                connection.close()

        assert get("/health") == (503, b"starting\n")
        assert get("/metrics")[0] == 200
        made: list[StandIn] = []

        def factory() -> StandIn:
            # The server answers while its card is made: /health says starting, still.
            assert get("/health")[0] == 503
            made.append(StandIn(cpu_models))
            return made[-1]

        main.bring_up(running, factory)
        assert len(made) == 1
        assert service.machine is made[0]
        service.ready()
        assert get("/health") == (200, b"ready\n")
    finally:
        main.shut(running)


def test_serve_makes_the_card_only_once_it_listens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The real serve: the card's reads are made inside the start, after the socket is listening
    (the factory finds /health answering starting); a card that refuses ends it with 1."""
    import signal
    from http.client import HTTPConnection

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    seen: list[int] = []

    def factory() -> Any:
        connection = HTTPConnection("127.0.0.1", port, timeout=30)
        connection.request("GET", "/health")
        seen.append(connection.getresponse().status)
        connection.close()
        raise Refusal("no CUDA device: stand-in")

    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    code = main.serve(options(tmp_path), None, ("127.0.0.1", port), factory, Metrics())
    assert code == 1
    assert seen == [503]


# ---- The smaller DiT files the start's plan names ----------------------------------------------

FULL_HD = (1920, 1080)
NIGHT_INPUT = 1024 * 576  # 0.6 MP, the warm-up's inputs of the night runs


def plan_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
    free_gib: float,
    **given: Any,
) -> tuple[list[str], services.Service]:
    """The lines the start's plan logs at INFO (the card read, the plan made: no models loaded,
    no warm-up) on a stand-in card with `free_gib` free, one picture at a time within 1920x1080
    from 0.6 MP unless `given` says otherwise; with the service."""
    dit_format = given.pop("dit_format", peaks.FP16)
    machine = StandIn(cpu_models, free_bytes=round(free_gib * GIB), dit_format=dit_format)
    machine.attention_name = peaks.SDPA
    settings: dict[str, Any] = {"bound": FULL_HD, "input_pixels": NIGHT_INPUT, "count": 1}
    settings.update(given)
    with live(
        monkeypatch, tmp_path, cpu_models, ready=False, machine=machine, **settings
    ) as server:
        server.service.check_files()
        with caplog.at_level(logging.INFO, logger="imgserver.service"):
            server.service.plan()
        lines = [r.getMessage() for r in caplog.records if r.name == "imgserver.service"]
        return lines, server.service


def test_the_start_logs_the_smaller_files_that_swap_fewer_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too", the image server's start alike:
    the fp16 plan on a card that makes it swap blocks logs the command line's line, blocks alone
    (a picture is one latent), after the blocks swapped line, at INFO. The stand-in's file name is
    no pinned one, so both 7Bs' files are named. The file the server runs is never changed."""
    lines, service = plan_lines(monkeypatch, tmp_path, cpu_models, caplog, 12)
    blocks = next(at for at, each in enumerate(lines) if each.startswith("plan: blocks swapped"))
    assert lines[blocks + 1] == (
        "plan: smaller DiT files plan this bound better: this plan swaps 16 blocks; for the sharp"
        " 7B (int8 0 blocks; the dynamic GGUF 0 blocks); for the 7B (int8 0 blocks; the dynamic"
        " GGUF 0 blocks); --dit-model names them"
    )
    assert sum("smaller DiT files" in each for each in lines) == 1
    assert service.options.dit == "w.safetensors"
    assert service.made is not None and service.made.plan.dit_model == peaks.FP16


def test_the_start_of_an_int8_dit_names_the_gguf_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lines, _ = plan_lines(monkeypatch, tmp_path, cpu_models, caplog, 10, dit_format=peaks.INT8)
    assert (
        "plan: smaller DiT files plan this bound better: this plan swaps 5 blocks; for the sharp 7B"
        " (the dynamic GGUF 0 blocks); for the 7B (the dynamic GGUF 0 blocks); --dit-model names"
        " them"
    ) in lines


def test_the_start_logs_nothing_of_smaller_files_for_the_smallest_format_or_a_plan_without_swaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The dynamic GGUF is the smallest format, whatever it swaps (6 blocks on 6 GiB here); and a
    plan that swaps none has nothing a smaller file would spare (fp16 whole on 24 GiB)."""
    lines, _ = plan_lines(monkeypatch, tmp_path, cpu_models, caplog, 6, dit_format=peaks.GGUF_7B)
    assert any(each.startswith("plan: blocks swapped: 6 ") for each in lines)
    assert not any("smaller DiT files" in each for each in lines)
    caplog.clear()
    lines, _ = plan_lines(monkeypatch, tmp_path, cpu_models, caplog, 24)
    assert any(each.startswith("plan: blocks swapped: none") for each in lines)
    assert not any("smaller DiT files" in each for each in lines)
