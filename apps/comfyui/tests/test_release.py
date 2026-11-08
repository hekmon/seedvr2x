"""The card freed at the end of a run, on ComfyUI's Cancel and on an error (models.session): the
models back in their pinned copies, torch's cache emptied; models not kept given up with what they
keep on the device, even while the exception is held; and, on the real Models.place and release
over toy models, nothing of them left on the card (test_held.strays)."""

import functools
import gc
import logging
import weakref
from typing import Any

import comfy.model_management as management
import numpy as np
import pytest
import server
import torch
from comfy.utils import ProgressBar
from comfyui import models, plans, words
from standins import GIB, PICTURE, RESOLUTION, StandIn, handle, upscale_image, upscale_video
from test_blockswap import Sim
from test_held import Card, Files, strays

from seedvr2x import model, waiting
from seedvr2x.memory import Reading
from seedvr2x.passes import HELD, ModelPlan
from seedvr2x.pull import ModelFiles

SETTINGS = plans.Settings()


def images(count: int = 1) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(2).random((count, *PICTURE, 3), dtype=np.float32))


def clip_frames(count: int) -> torch.Tensor:
    return images(count)


def freed(machine: StandIn, h: models.Handle) -> None:
    """What every end leaves: the models away from the card, torch emptied."""
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    held = machine.made[0]
    assert not held.on_card  # type: ignore[attr-defined]
    assert management.empties, "torch's cache emptied"


def test_the_card_is_freed_when_a_node_ends(machine: StandIn, tmp_path) -> None:
    h = handle(tmp_path)
    upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    freed(machine, h)
    upscale_video(h, SETTINGS, clip_frames(5), RESOLUTION, "1")
    freed(machine, h)
    assert len(machine.made[0].placed) == 2  # type: ignore[attr-defined]


def cancel_at(update: int) -> None:
    """ComfyUI's Cancel pressed when the bar has been moved `update` times."""
    seen: list[float] = []

    def hook(value: float) -> None:
        seen.append(value)
        if len(seen) == update:
            management.interrupt_current_processing()

    ProgressBar.on_update = hook


@pytest.mark.parametrize("update", [1, 2, 3])
def test_cancel_between_a_pictures_steps_frees_the_card(
    machine: StandIn, tmp_path, update: int
) -> None:
    h = handle(tmp_path)
    cancel_at(update)
    with pytest.raises(management.InterruptProcessingException):
        upscale_image(h, SETTINGS, images(2), RESOLUTION, "1")
    freed(machine, h)
    # The flag was cleared as it raised: the next run goes through.
    out, _ = upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    assert out.shape == (1, 48, 80, 3)


@pytest.mark.parametrize("update", [1, 2, 3, 5])
def test_cancel_between_a_clips_steps_frees_the_card(
    machine: StandIn, tmp_path, update: int
) -> None:
    h = handle(tmp_path)
    cancel_at(update)
    with pytest.raises(management.InterruptProcessingException):
        upscale_video(h, plans.Settings(window=5), clip_frames(33), RESOLUTION, "1")
    freed(machine, h)
    # Stopped between steps: no window or decode ran on after the cancelled step's bar update.
    bar = ProgressBar.created[-1]
    assert len(bar.updates) == update


def test_cancel_between_a_clips_runs_frees_the_card_and_makes_no_run_after(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = handle(tmp_path)
    windows: list[int] = []
    sample = model.sample

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        windows.append(1)
        return sample(*args, **kwargs)

    monkeypatch.setattr(model, "sample", counting)

    # The node's text as the second run starts ("Frames 2 to 33, run 2 of 4: encoding"), said
    # before the bar is moved for it: Cancel as that update is made.
    def hook(value: float) -> None:
        instance = server.PromptServer.instance
        assert instance is not None
        if "run 2 of 4" in instance.texts[-1][0]:
            management.interrupt_current_processing()

    ProgressBar.on_update = hook
    with pytest.raises(management.InterruptProcessingException):
        upscale_video(
            h, plans.Settings(window=5), clip_frames(33), RESOLUTION, "1", quality="maximum"
        )
    freed(machine, h)
    # The first run's 3 windows (33 frames, windows of at most 17): none of the second's.
    assert len(windows) == 3
    # The flag was cleared as it raised: the next run goes through, at any level.
    ProgressBar.on_update = None
    out, _ = upscale_video(h, plans.Settings(), clip_frames(5), RESOLUTION, "1", quality="enhanced")
    assert out.shape == (5, 48, 80, 3)


def test_an_error_in_the_pass_frees_the_card_and_is_not_swallowed(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = handle(tmp_path)

    def boom(*args: object) -> None:
        raise RuntimeError("a window failed")

    monkeypatch.setattr(model, "sample", boom)
    with pytest.raises(RuntimeError, match="a window failed"):
        upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    freed(machine, h)


def test_models_not_kept_are_dropped_on_an_error_too(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = handle(tmp_path, keep=False)
    monkeypatch.setattr(model, "sample", lambda *args: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(ValueError):
        upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    alive = weakref.ref(machine.made[0])
    machine.made.clear()
    gc.collect()
    assert alive() is None


class ToyMachine(StandIn):
    """The stand-in machine over test_held's toy models on its simulated card: the real
    Models.place and Models.release (HELD), the card's buffers as test_held.Card holds them."""

    toy: Files
    card: Card

    def load(self, files, plan, progress=None):  # type: ignore[no-untyped-def,override]
        self.loads.append(plan)
        loaded = self.toy.load(plan)
        self.card.put(loaded)
        self.made.append(loaded)
        return loaded


def toy_planner(reading: Reading, host: int | None, attention: str, arch, held: bool):  # type: ignore[no-untyped-def]
    plan = ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False)
    return models.Planned(None, plan, models.Needs(peak=1 * GIB, margin=0, reading=reading))


@pytest.mark.parametrize("how", ["end", "cancel", "error"])
def test_nothing_of_the_models_is_left_on_the_card_after_a_run(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    toy = ToyMachine(machine.vae_models)
    toy.toy = Files(tmp_path, Sim("random-1"), monkeypatch)
    toy.card = Card(monkeypatch)
    models.use(toy)
    h = handle(tmp_path)
    held: model.Models | None = None
    try:
        with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
            held = run.models
            held.enter("dit")
            held.enter("decode")
            assert strays(held, toy.card), "placed: the buffers are on the card"
            if how == "cancel":
                raise management.InterruptProcessingException()
            if how == "error":
                raise RuntimeError("boom")
    except (management.InterruptProcessingException, RuntimeError):
        assert how != "end"
    assert toy.loads == [HELD] and held is not None
    assert strays(held, toy.card) == []
    # And again: the models placed for a second run, then freed.
    with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
        run.models.enter("dit")
    assert strays(held, toy.card) == []


def test_nothing_waits_on_the_card_to_give_the_models_back_after_an_abandoned_wait(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A wait on the GPU past its bound that Cancel ended (waiting.abandoned): the real
    Models.release sends nothing back (every step of it waits on the device) and torch's cache is
    not emptied, and the next run is refused at its start with nothing read, placed or waited on
    (a card that stopped answering is for restarting ComfyUI)."""
    toy = ToyMachine(machine.vae_models)
    toy.toy = Files(tmp_path, Sim("random-1"), monkeypatch)
    toy.card = Card(monkeypatch)
    models.use(toy)
    h = handle(tmp_path)
    sent: list[str] = []
    send_back = model.Models._send_back  # pyright: ignore[reportPrivateUsage]

    def spy(self: model.Models) -> None:
        sent.append("back")
        send_back(self)

    monkeypatch.setattr(model.Models, "_send_back", spy)
    monkeypatch.setattr(waiting, "_abandoned", set())
    held: model.Models | None = None
    emptied = 0
    with pytest.raises(management.InterruptProcessingException):
        with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
            held = run.models
            held.enter("dit")
            held.enter("decode")
            sent.clear()
            emptied = len(management.empties)
            monkeypatch.setattr(waiting, "_abandoned", {0})
            raise management.InterruptProcessingException()
    assert held is not None
    assert sent == [] and len(management.empties) == emptied
    assert "not sent back to their pinned copies" in caplog.text
    assert strays(held, toy.card), "left as they were: the card's buffers are still there"
    assert held.phase is None and held.model_plan is None
    with pytest.raises(RuntimeError, match="released"):
        held.enter("encode")
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    # The next run is refused before the card is read or the models are placed.
    reads = len(management.empties)
    with pytest.raises(words.Refused, match="restart ComfyUI"):
        with models.session(h, None, toy_planner):  # type: ignore[arg-type]
            pytest.fail("a run went on")
    assert sent == [] and len(management.empties) == reads == emptied
    assert toy.loads == [HELD], "nothing was loaded again either"


def test_models_not_kept_are_gone_while_the_exception_that_ended_the_run_is_held(
    machine: StandIn, tmp_path
) -> None:
    """A traceback keeps the frames of the passes the error went through, and their locals (the
    models): ComfyUI holds the exception until it reports it. The session clears the frames."""
    h = handle(tmp_path, keep=False)
    cancel_at(2)
    with pytest.raises(management.InterruptProcessingException) as held:
        upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    alive = weakref.ref(machine.made[0])
    machine.made.clear()
    gc.collect()
    assert held.value is not None
    assert alive() is None, "the models outlive the run through the exception's frames"


def test_models_not_kept_are_gone_while_a_refusal_raised_from_the_error_is_held(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The node words a non-finite picture and raises it from the library's error: the frames of
    # both tracebacks are cleared.
    h = handle(tmp_path, keep=False)
    real = model.sample
    monkeypatch.setattr(model, "sample", lambda *args: real(*args) * float("nan"))
    with pytest.raises(words.Refused, match="NaN") as held:
        upscale_image(h, SETTINGS, images(), RESOLUTION, "1")
    assert isinstance(held.value.__cause__, BaseException)
    alive = weakref.ref(machine.made[0])
    machine.made.clear()
    gc.collect()
    assert alive() is None


class Rope(torch.nn.Module):
    """A module that memoises on the class, as the vendored rotary embedding does (rope.py:44,
    get_axial_freqs under an lru_cache of 128 entries keyed by the module)."""

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def get_axial_freqs(self, *shape: int) -> torch.Tensor:
        return torch.zeros(shape)


def test_models_not_kept_drop_what_they_keep_on_the_device_besides_parameters(
    machine: StandIn, tmp_path
) -> None:
    Rope.get_axial_freqs.cache_clear()
    real_load = machine.load

    def load(files: ModelFiles, plan: ModelPlan, progress: object = None) -> model.Models:
        loaded = real_load(files, plan, progress)  # type: ignore[arg-type]
        rope = Rope()
        loaded.runner.dit.append(rope)  # type: ignore[attr-defined]
        rope.get_axial_freqs(2, 3)
        return loaded

    machine.load = load  # type: ignore[method-assign]
    upscale_image(handle(tmp_path, keep=False), SETTINGS, images(), RESOLUTION, "1")
    # The cache would keep the dropped DiT's module and its tensors up to 128 shapes.
    assert Rope.get_axial_freqs.cache_info().currsize == 0
    Rope.get_axial_freqs.cache_clear()


# ---- A mark the card itself clears (waiting.settle, models._settle) ----------------------------


def left_release(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, keep: bool = True
) -> tuple[ToyMachine, models.Handle, Sim]:
    """A run Cancelled with work queued on both of the placement's streams, whose release a second
    Cancel left at its first wait (the poll's sleep raises once): the real Models.release over the
    simulated streams, whose waits go through the poll as Cuda.wait's do. The card is marked and
    the work is still queued (the simulation runs it when told: `sim.settle()`)."""
    toy = ToyMachine(machine.vae_models)
    sim = Sim("compute-first")
    toy.toy = Files(tmp_path, sim, monkeypatch)
    toy.card = Card(monkeypatch)
    models.use(toy)
    h = handle(tmp_path, keep=keep)

    pressed: list[int] = []
    drained = sim.wait

    def polled(*streams: Any) -> None:
        if not pressed:
            return drained(*streams)  # before the Cancel: the simulation runs what is queued
        for stream in streams:
            waiting.wait(lambda stream=stream: not stream.queue, 0.01)

    monkeypatch.setattr(sim, "wait", polled)

    def sleep(seconds: float) -> None:
        if pressed == [0]:
            pressed[0] = 1
            raise management.InterruptProcessingException()

    monkeypatch.setattr(waiting, "sleep", sleep)

    def synchronize(device: Any) -> None:
        # The poll on the current stream a run not kept ends with (the stand-in card is the CPU).
        if pressed:
            waiting.wait(lambda: not sim.main.queue, 0.01)

    monkeypatch.setattr(model, "synchronize", synchronize)
    with pytest.raises(management.InterruptProcessingException):
        with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
            run.models.enter("dit")
            run.models.enter("decode")
            sim.kernel("work", lambda: None, [])
            assert sim.sides, "the DiT's prefetch has a side stream"
            sim.sides[0].queue.append(("kernel", "copy", lambda: None, []))
            pressed.append(0)  # the first poll from here on is left, the release's
            raise management.InterruptProcessingException()
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    # Later polls (the release the next run completes) let the simulation run what it has queued,
    # as the device does; the tests count them to show that the query itself sleeps never.
    slept: list[float] = []

    def device_runs(seconds: float) -> None:
        slept.append(seconds)
        sim.settle()

    monkeypatch.setattr(waiting, "sleep", device_runs)
    sim.slept = slept  # type: ignore[attr-defined]
    return toy, h, sim


def test_the_work_a_left_release_waited_for_is_asked_without_waiting_and_all_of_it_counts(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    toy, h, sim = left_release(machine, tmp_path, monkeypatch)
    loads, emptied = len(toy.loads), len(management.empties)
    # The compute stream is drained and the side stream not: the later wait the release never
    # reached counts, so the card is not finished and the run is refused in the new words.
    sim.drain(sim.main)
    slept = len(sim.slept)  # type: ignore[attr-defined]
    for _ in range(2):
        with pytest.raises(words.Refused, match="still busy with the last work") as refused:
            with models.session(h, None, toy_planner):  # type: ignore[arg-type]
                pytest.fail("a run went on")
        assert "restart ComfyUI" in str(refused.value)
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(toy.loads) == loads and len(management.empties) == emptied
    assert len(sim.slept) == slept, "asking the card never waited"  # type: ignore[attr-defined]
    # Once the side stream is through, the next run completes the release and goes on: the models
    # kept, placed again, the mark cleared, said at INFO.
    caplog.set_level(logging.INFO)
    sim.settle()
    held = models.kept(h)
    assert held is not None and held.models is not None
    with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
        assert run.models is held.models
        run.models.enter("dit")
    assert not waiting.abandoned() and models._left_on_card == []  # pyright: ignore[reportPrivateUsage]
    assert toy.loads == [HELD], "nothing read again"
    assert strays(held.models, toy.card) == []
    assert "the card finished the work an earlier run left" in caplog.text


def test_models_not_kept_are_let_go_when_the_card_finishes(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toy, h, sim = left_release(machine, tmp_path, monkeypatch, keep=False)
    alive = weakref.ref(models._left_on_card[0])  # pyright: ignore[reportPrivateUsage]
    # Only the compute stream's poll was left, but the side stream's work counts as well: the
    # release the next run completes would wait for it.
    sim.drain(sim.main)
    with pytest.raises(words.Refused, match="still busy with the last work"):
        with models.session(h, None, toy_planner):  # type: ignore[arg-type]
            pytest.fail("a run went on")
    sim.settle()
    with models.session(h, None, toy_planner):  # type: ignore[arg-type]
        pass
    assert models._left_on_card == [] and not waiting.abandoned()  # pyright: ignore[reportPrivateUsage]
    toy.made.clear()
    gc.collect()
    assert alive() is None, "the models of a run not kept are let go"


def test_a_card_a_cuda_error_failed_is_never_cleared_by_the_query(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toy, h, sim = left_release(machine, tmp_path, monkeypatch)
    sim.settle()
    waiting.fail(0, RuntimeError("CUDA error: an illegal memory access was encountered"))
    loads = len(toy.loads)
    with pytest.raises(words.Refused, match="failed during an earlier seedvr2x run"):
        with models.session(h, None, toy_planner):  # type: ignore[arg-type]
            pytest.fail("a run went on")
    assert waiting.abandoned() and waiting.failed() is not None and len(toy.loads) == loads
    models.reset()


# ---- A wait in the run left by Cancel, past its bound (waiting.pend) ----------------------------


def left_wait(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, keep: bool = True
) -> tuple[ToyMachine, models.Handle, Sim]:
    """A run whose own wait (not a release's) on the compute stream's work is past its bound when
    Cancel leaves it, the side stream's copies and two later compute kernels queued behind: the
    wait's event covers only the first kernel (`_paced`'s block n-2 with n-1 and n behind it).
    The card is marked, the work still queued; the run's end gives the models back (Models.release
    abandoned, or the nodes' lost branch) and records what is queued on every stream."""
    toy = ToyMachine(machine.vae_models)
    sim = Sim("compute-first")
    toy.toy = Files(tmp_path, sim, monkeypatch)
    toy.card = Card(monkeypatch)
    models.use(toy)
    h = handle(tmp_path, keep=keep)
    clock = [0.0]
    monkeypatch.setattr(waiting, "monotonic", lambda: clock[0])
    polls: list[int] = []

    def sleep(seconds: float) -> None:
        # The first poll puts the wait past its bound (60 s); the second is the Cancel.
        clock[0] += 100.0
        polls.append(1)
        if len(polls) > 1:
            raise management.InterruptProcessingException()

    monkeypatch.setattr(waiting, "sleep", sleep)
    monkeypatch.setattr(model, "synchronize", lambda device: None)
    with pytest.raises(management.InterruptProcessingException):
        with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
            run.models.enter("dit")
            first = sim.main
            sim.kernel("n-2", lambda: None, [])
            assert sim.sides, "the DiT's prefetch has a side stream"
            sim.sides[0].queue.append(("kernel", "copy", lambda: None, []))
            sim.kernel("n-1", lambda: None, [])
            sim.kernel("n", lambda: None, [])
            waiting.unit("window", 1.0)
            waiting.wait(lambda: len(first.queue) < 3, 0.01)
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    return toy, h, sim


@pytest.mark.parametrize("stream", ["side", "compute"])
@pytest.mark.parametrize("keep", [True, False])
def test_work_queued_behind_a_left_wait_on_any_stream_keeps_the_card_busy(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch, keep: bool, stream: str
) -> None:
    toy, h, sim = left_wait(machine, tmp_path, monkeypatch, keep)
    monkeypatch.setattr(waiting, "sleep", lambda seconds: sim.settle())
    # The wait's own event is done; what is queued behind it is not, on one stream or the other.
    if stream == "side":
        while sim.step(sim.main):  # the compute stream through, the side stream's copies not
            pass
    else:
        # The wait's own kernel only, two behind it; the side stream through.
        sim.step(sim.main)
        while sim.step(sim.sides[0]):
            pass
    assert sim.sides[0].queue if stream == "side" else sim.main.queue
    loads = len(toy.loads)
    with pytest.raises(words.Refused, match="still busy with the last work"):
        with models.session(h, None, toy_planner):  # type: ignore[arg-type]
            pytest.fail("a run went on")
    assert waiting.abandoned() and len(toy.loads) == loads
    sim.settle()
    with models.session(h, None, toy_planner) as run:  # type: ignore[arg-type]
        run.models.enter("dit")
    assert not waiting.abandoned() and models._left_on_card == []  # pyright: ignore[reportPrivateUsage]


def test_models_not_kept_are_synchronized_in_the_poll_before_they_are_let_go(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, h, sim = left_wait(machine, tmp_path, monkeypatch, keep=False)
    sim.settle()
    order: list[str] = []
    monkeypatch.setattr(model, "synchronize", lambda device: order.append("synchronize"))
    real = model.forget
    monkeypatch.setattr(model, "forget", lambda each: order.append("forget") or real(each))
    with models.session(h, None, toy_planner):  # type: ignore[arg-type]
        pass
    assert order[:2] == ["synchronize", "forget"]


def test_a_cancel_during_the_release_the_next_run_completes_leaves_the_mark_kept(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, h, sim = left_wait(machine, tmp_path, monkeypatch, keep=False)
    sim.settle()
    # The card finished, then stalls again in the settle's synchronize; ComfyUI's Cancel (the
    # check of the settle's polls) leaves it, the card marked and the models still held.
    monkeypatch.setattr(model, "synchronize", lambda device: waiting.wait(lambda: False, 0.01))
    monkeypatch.setattr(waiting, "sleep", lambda seconds: management.interrupt_current_processing())
    with pytest.raises(words.Refused, match="still busy with the last work"):
        with models.session(h, None, toy_planner):  # type: ignore[arg-type]
            pytest.fail("a run went on")
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    models.reset()
