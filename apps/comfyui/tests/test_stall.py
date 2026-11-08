"""A wait on the GPU that lasts, on the nodes (comfyui/progress.py, models.py; seedvr2x/waiting.py;
DESIGN.md, Repository and apps, the nodes' progress, and Allocator, the hang of 2026-11-06): the
units named for the library's waits with the estimate the run has measured, a wait past its bound
said on the node with the way out and on ComfyUI's console, ComfyUI's Cancel leaving it, nothing
given back to the card that waits on it, and nothing left running.

An event that never completes is `waiting.wait(lambda: False, ...)` on the stand-in models, as
apps/seedvr2x/tests/test_stop.py does; the wait's clock is a fake one whose sleep advances it, and
whose steps are where the tests act (the ticker's cue, ComfyUI's Cancel)."""

import gc
import logging
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import comfy.model_management as management
import pytest
import server
from comfyui import models, plans, progress, words
from standins import RESOLUTION, StandIn, handle
from standins import upscale_video as run_video
from test_nodes import pictures
from test_progress import (
    TILES,
    Cue,
    clip_passes,
    lines,
    make,
    node_texts,
    shots_of,
    ticker_threads,
)

from seedvr2x import memory, model, waiting

UNIT_KEYS = ("name", "estimate", "way")


class WaitClock:
    """The wait's readers: monotonic gives `now`, sleep(seconds) advances it, then calls
    `on_sleep`."""

    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


@pytest.fixture(autouse=True)
def world(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> Iterator[WaitClock]:
    """A fake clock in the library's wait, NVML and the stacks' writer replaced, the module's
    state as a fresh process has it (monkeypatch gives each back as it was)."""
    caplog.set_level(logging.INFO)
    found = WaitClock()
    monkeypatch.setattr(waiting, "monotonic", found.monotonic)
    monkeypatch.setattr(waiting, "sleep", found.sleep)
    monkeypatch.setattr(memory, "activity", lambda index=0: None)
    monkeypatch.setattr(waiting, "_stacks", lambda: None)
    monkeypatch.setattr(waiting, "_ended", None)
    monkeypatch.setattr(waiting, "_abandoned", set())
    monkeypatch.setattr(waiting, "_pending", {})
    monkeypatch.setattr(waiting, "_stacks_written", False)
    monkeypatch.setattr(waiting, "_running", {})
    waiting.clear()
    waiting.check(None)
    yield found
    waiting.clear()
    waiting.check(None)


def unit() -> waiting.Unit:
    found: waiting.Unit | None = getattr(waiting._named, "unit", None)  # pyright: ignore[reportPrivateUsage]
    assert found is not None
    return found


def stall_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "seedvr2x.waiting" and "a wait on the GPU has lasted" in r.getMessage()
    ]


# ---- The units named ----------------------------------------------------------------------------


def test_each_event_names_the_next_unit_with_its_estimate_once_a_unit_of_its_kind_is_measured() -> (
    None
):
    passes = clip_passes(shots_of(2, 33), 5, TILES, TILES)
    p, clock, _, _ = make(passes)
    label = "Shot 1 of 2 (frames 1 to 33)"
    p.begin(0)
    # Nothing measured: the unit is named, with no estimate, so that a wait of it says nothing.
    assert unit().name == f"{label}: encoding"
    assert unit().estimate is None and unit().way == progress.WAY
    p.tile("encode", 0, 4)
    clock.t += 2.0
    p.tile("encode", 1, 4)
    # An encode tile is measured: the next one is estimated by its weight at the run's rate.
    assert unit().name == f"{label}: encoding, tile 2 of 4"
    assert unit().estimate == pytest.approx(passes[0].encode[1].weight * p.rate("encode"))
    assert unit().estimate == pytest.approx(
        2.0 * passes[0].encode[1].weight / passes[0].encode[0].weight
    )
    for done in (2, 3, 4):
        clock.t += 2.0
        p.tile("encode", done, 4)
    p.encoded()
    # The first window: no window measured, the encode's rate is another kind's.
    assert unit().name == f"{label}: upscaling, window 1 of 3"
    assert unit().estimate is None
    clock.t += 3.0
    p.window(1)
    assert unit().name == f"{label}: upscaling, window 2 of 3"
    assert unit().estimate == pytest.approx(passes[0].windows[1].weight * p.rate("window"))
    p.window(2)
    p.window(3)
    # The decode's first tile: its kind is not measured.
    assert unit().name == f"{label}: decoding, tile 1 of 4"
    assert unit().estimate is None
    # The next shot's encode is estimated at the encode rate measured in the first.
    p.begin(1)
    assert unit().name == "Shot 2 of 2 (frames 34 to 66): encoding"
    assert unit().estimate == pytest.approx(passes[1].encode[0].weight * p.rate("encode"))


def test_running_sets_cancel_as_the_waits_check_and_forgets_the_unit_at_its_end() -> None:
    p, _, _, _ = make(clip_passes(shots_of(1, 33)))
    assert getattr(waiting._named, "check", None) is None  # pyright: ignore[reportPrivateUsage]
    with p.running(pause=Cue().pause):
        assert (
            getattr(waiting._named, "check", None)  # pyright: ignore[reportPrivateUsage]
            is management.throw_exception_if_processing_interrupted
        )
        p.begin(0)
        assert getattr(waiting._named, "unit", None) is not None  # pyright: ignore[reportPrivateUsage]
    assert getattr(waiting._named, "check", None) is None  # pyright: ignore[reportPrivateUsage]
    assert getattr(waiting._named, "unit", None) is None  # pyright: ignore[reportPrivateUsage]


# ---- A wait that lasts: the text, the console, Cancel -------------------------------------------


def test_a_wait_past_its_bound_is_said_on_the_node_and_in_the_console_and_cancel_leaves_it(
    world: WaitClock, caplog: pytest.LogCaptureFixture
) -> None:
    p, clock, shown, bar = make(clip_passes(shots_of(2, 33), 5), plan="the plan")
    cue = Cue()
    said: dict[str, str] = {}
    steps = {"n": 0}

    def on_sleep() -> None:
        steps["n"] += 1
        if world.now == 30.0:
            cue.fire()
            said["before"] = shown[-1]
        if world.now == 80.0:
            cue.fire()
            said["during"] = shown[-1]
            management.interrupt_current_processing()

    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        clock.t += 1.0
        p.encoded()
        clock.t += 2.0
        p.window(1)
        # Window 2 is next: its estimate is the run's own, a few seconds, the floor 60 s.
        assert unit().estimate is not None and waiting.bound(unit().estimate) == 60.0
        name = unit().name
        world.on_sleep = on_sleep
        updates = list(bar.updates)
        with pytest.raises(management.InterruptProcessingException):
            waiting.wait(lambda: False, 10.0)
    # Under the bound the text said nothing of it; past it, how long and the way out.
    assert "A wait on the GPU" not in said["before"]
    assert lines(said["during"])[:2] == ["the plan", f"{name}"]
    assert (
        "A wait on the GPU has lasted 1 min 20 s, much longer than this step should take: "
        f"{progress.WAY}." in said["during"]
    )
    assert "restart ComfyUI" in said["during"]
    # The library's own line, on ComfyUI's console (a WARNING of seedvr2x.waiting), names the unit
    # and the way out; the bar was never touched by the ticker.
    (console,) = stall_lines(caplog)
    assert console.startswith(f"{name}: a wait on the GPU has lasted 1 min 10 s")
    assert f"seedvr2x goes on waiting: {progress.WAY}" in console
    assert bar.updates == updates
    # Cancel left the wait: ended by the exception, past its bound, so the card is abandoned.
    ended = waiting.ended()
    assert (
        ended is not None
        and ended.unit == name
        and ended.kind is management.InterruptProcessingException
    )
    assert waiting.abandoned()
    assert ticker_threads() == []


def test_the_text_goes_back_to_normal_when_the_wait_ends_by_itself(
    world: WaitClock, caplog: pytest.LogCaptureFixture
) -> None:
    p, clock, shown, _ = make(clip_passes(shots_of(2, 33), 5))
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        clock.t += 1.0
        p.encoded()
        clock.t += 2.0
        p.window(1)

        during: list[str] = []

        def on_sleep() -> None:
            # The line is said at 70 s, after the sleep that reaches it: the tick at 80 has it.
            if world.now == 80.0:
                cue.fire()
                during.append(shown[-1])

        world.on_sleep = on_sleep
        waiting.wait(lambda: world.now >= 100.0, 10.0)
        assert "A wait on the GPU" in during[0], "said while it lasted"
        cue.fire()
        assert "A wait on the GPU" not in shown[-1]
    assert not waiting.abandoned()
    assert any("the wait on the GPU ended after" in r.getMessage() for r in caplog.records)


def test_a_wait_before_any_unit_is_measured_says_nothing(world: WaitClock) -> None:
    p, _, shown, _ = make(clip_passes(shots_of(1, 33), 5))
    cue = Cue()
    with p.running(pause=cue.pause):
        cue.settle()
        p.begin(0)
        world.on_sleep = lambda: cue.fire() if world.now == 600.0 else None
        waiting.wait(lambda: world.now >= 1000.0, 100.0)
        assert "A wait on the GPU" not in shown[-1]
    assert not waiting.stalled()


# ---- Through a node ------------------------------------------------------------------------------


def stalling_sample(
    monkeypatch: pytest.MonkeyPatch, world: WaitClock, cue: Cue, happened: dict[str, Any]
) -> None:
    """model.sample that never completes at its second window, the wait's clock stepping 10 s a
    poll: the ticker is cued at 30 s (under the bound) and at 80 s (past it), where ComfyUI's
    Cancel is pressed as well."""
    real = model.sample
    calls: list[int] = []

    def on_sleep() -> None:
        instance = server.PromptServer.instance
        assert instance is not None
        if world.now == 30.0:
            cue.settle()
            cue.fire()
            happened["before"] = instance.texts[-1][0]
        if world.now == 80.0:
            cue.fire()
            happened["during"] = instance.texts[-1][0]
            happened["empties"] = len(management.empties)
            management.interrupt_current_processing()

    def sample(*args: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            world.on_sleep = on_sleep
            waiting.wait(lambda: False, 10.0)
        return real(*args)

    monkeypatch.setattr(model, "sample", sample)


def test_a_node_stalled_in_a_window_says_so_and_cancel_leaves_it_and_gives_nothing_back(
    machine: StandIn,
    tmp_path,
    world: WaitClock,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cue, happened = Cue(), {}
    monkeypatch.setattr(progress, "_wait", cue.pause)
    stalling_sample(monkeypatch, world, cue, happened)
    h = handle(tmp_path)
    with pytest.raises(management.InterruptProcessingException):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "9")
    assert "A wait on the GPU" not in happened["before"]
    assert "A wait on the GPU has lasted 1 min 20 s" in happened["during"]
    assert "restart ComfyUI" in happened["during"]
    assert happened["during"] in node_texts("9")
    (console,) = stall_lines(caplog)
    assert "upscaling, window 2 of" in console and "restart ComfyUI" in console
    assert waiting.abandoned()
    # Torch's cache was not emptied on the way out, and the kept models stay registered, idle.
    assert len(management.empties) == happened["empties"]
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    assert ticker_threads() == []
    assert getattr(waiting._named, "check", None) is None  # pyright: ignore[reportPrivateUsage]


def test_models_not_kept_are_held_not_dropped_after_an_abandoned_wait(
    machine: StandIn,
    tmp_path,
    world: WaitClock,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cue, happened = Cue(), {}
    monkeypatch.setattr(progress, "_wait", cue.pause)
    stalling_sample(monkeypatch, world, cue, happened)
    with pytest.raises(management.InterruptProcessingException):
        run_video(
            handle(tmp_path, keep=False), plans.Settings(window=5), pictures(33), RESOLUTION, "9"
        )
    # Dropping them would run their finalizers, which wait on the card (the hang's cause).
    alive = weakref.ref(machine.made[0])
    machine.made.clear()
    gc.collect()
    assert alive() is not None and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(management.empties) == happened["empties"]
    assert any(
        "restart ComfyUI" in r.getMessage() for r in caplog.records if r.name != "seedvr2x.waiting"
    )
    models.reset()
    gc.collect()
    assert alive() is None


def stalled_run(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch, *, keep: bool
) -> tuple[models.Handle, dict[str, Any]]:
    cue, happened = Cue(), {}
    monkeypatch.setattr(progress, "_wait", cue.pause)
    stalling_sample(monkeypatch, world, cue, happened)
    h = handle(tmp_path, keep=keep)
    with pytest.raises(management.InterruptProcessingException):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "9")
    return h, happened


@pytest.mark.parametrize("keep", [True, False])
def test_the_run_after_an_abandoned_wait_is_refused_at_once_with_nothing_read(
    machine: StandIn,
    tmp_path,
    world: WaitClock,
    monkeypatch: pytest.MonkeyPatch,
    keep: bool,
) -> None:
    h, happened = stalled_run(machine, tmp_path, world, monkeypatch, keep=keep)
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    loads, emptied = len(machine.loads), len(management.empties)
    placed = len(machine.made[0].placed)  # type: ignore[attr-defined]
    free = machine.free
    reads: list[int] = []
    machine.free = lambda: reads.append(1) or free()  # type: ignore[method-assign]
    for plan_only in (False, True):
        with pytest.raises(words.Refused, match="still busy with the last work") as refused:
            run_video(
                h, plans.Settings(window=5), pictures(5), RESOLUTION, "9", plan_only=plan_only
            )
        assert "restart ComfyUI" in str(refused.value)
    # No read of the card, no load, no place, torch's cache left alone.
    assert reads == [] and len(machine.loads) == loads
    assert len(machine.made[0].placed) == placed  # type: ignore[attr-defined]
    assert len(management.empties) == emptied
    assert happened["empties"] == emptied


def test_unloading_after_an_abandoned_wait_holds_the_models_instead_of_collecting_them(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, _ = stalled_run(machine, tmp_path, world, monkeypatch, keep=True)
    entry = models.kept(h)
    assert entry is not None and entry.models is not None
    alive = weakref.ref(entry.models)
    collected: list[int] = []
    real = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *args: collected.append(1) or real(*args))
    machine.made.clear()
    entry.drop("Unload models", by_comfy=True)
    # Dropped from the registry, but the finalizers of the pinned copies (cudaHostUnregister) wait
    # on the card: the models are held, and no collection ran.
    assert models.kept(h) is None and collected == []
    gc.collect()
    assert alive() is not None
    assert len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    models.reset()
    gc.collect()
    assert alive() is None


def test_unloading_without_an_abandoned_wait_collects_as_before(machine: StandIn, tmp_path) -> None:
    h = handle(tmp_path)
    run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    entry = models.kept(h)
    assert entry is not None and entry.models is not None
    alive = weakref.ref(entry.models)
    machine.made.clear()
    entry.drop("Unload models", by_comfy=True)
    gc.collect()
    assert alive() is None and models._left_on_card == []  # pyright: ignore[reportPrivateUsage]


# ---- The release after a run an exception ended ----------------------------------------------


def cancelled_run(
    machine: StandIn,
    tmp_path,
    world: WaitClock,
    monkeypatch: pytest.MonkeyPatch,
    release: Any,
    *,
    keep: bool = True,
    fail: BaseException | None = None,
    at: int = 2,
) -> tuple[models.Handle, dict[str, Any]]:
    """A run Cancelled between two steps (the flag raised and cleared, as ComfyUI's check does),
    its models' release being `release(happened)`: the library's Models.release is the stand-in's
    here, which waits as the real one does inside waiting.releasing."""
    happened: dict[str, Any] = {}
    real = model.sample
    calls: list[int] = []

    def sample(*args: Any) -> Any:
        calls.append(1)
        if len(calls) == at:
            raise fail or management.InterruptProcessingException()
        return real(*args)

    monkeypatch.setattr(model, "sample", sample)
    h = handle(tmp_path, keep=keep)
    original = machine.load

    def load(*args: Any, **kwargs: Any) -> Any:
        found = original(*args, **kwargs)
        later: list[int] = []

        def over() -> None:
            # The first call is the load's own (models._load_kept), a healthy one.
            if later:
                release(happened)
            later.append(1)

        found.release = over
        return found

    monkeypatch.setattr(machine, "load", load)
    with pytest.raises(type(fail) if fail else management.InterruptProcessingException):
        run_video(h, plans.Settings(window=5), pictures(33), RESOLUTION, "9")
    return h, happened


def never_ending_release(world: WaitClock, press_at: float | None) -> Callable[[dict], None]:
    """A release whose wait never completes; ComfyUI's Cancel is pressed at `press_at` seconds
    of the wait's clock (never for None). The text on the node at its first poll is kept."""

    def release(happened: dict[str, Any]) -> None:
        instance = server.PromptServer.instance
        assert instance is not None
        begun = world.now

        def on_sleep() -> None:
            happened.setdefault("text", instance.texts[-1][0])
            happened["empties"] = len(management.empties)
            if press_at is not None and world.now - begun >= press_at:
                management.interrupt_current_processing()

        world.on_sleep = on_sleep
        with waiting.releasing(0):
            waiting.wait(lambda: False, 10.0)

    return release


def test_a_release_that_never_ends_says_so_and_a_second_cancel_leaves_it_and_marks_the_card(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    h, happened = cancelled_run(
        machine, tmp_path, world, monkeypatch, never_ending_release(world, 30.0)
    )
    # Exactly the words, as the node's text, while the release waited.
    assert happened["text"] == progress.FREEING
    assert progress.FREEING == (
        "freeing the card: waiting for the GPU to finish its last work; Cancel again to leave it,"
        " then run again once the card has finished, or restart ComfyUI"
    )
    # The second Cancel left it: the card is marked whatever the bound (no unit was named), torch's
    # cache was not emptied after it, and the models are held, not collected.
    assert waiting.abandoned()
    assert len(management.empties) == happened["empties"]
    assert node_texts("9")[-1] == progress.STOPPED, "no more 'Cancel again to leave it'"
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    assert len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    assert ticker_threads() == []
    assert getattr(waiting._named, "check", None) is None  # pyright: ignore[reportPrivateUsage]
    # The next run is refused at its start, nothing read.
    loads = len(machine.loads)
    with pytest.raises(words.Refused, match="still busy with the last work"):
        run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    assert len(machine.loads) == loads


def test_a_healthy_release_after_a_cancel_puts_the_text_back_and_marks_nothing(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def release(happened: dict[str, Any]) -> None:
        instance = server.PromptServer.instance
        assert instance is not None
        begun = world.now
        world.on_sleep = lambda: happened.setdefault("text", instance.texts[-1][0])
        with waiting.releasing(0):
            waiting.wait(lambda: world.now - begun >= 20.0, 10.0)

    h, happened = cancelled_run(machine, tmp_path, world, monkeypatch, release)
    texts = node_texts("9")
    assert happened["text"] == progress.FREEING
    assert texts[-1] != progress.FREEING and progress.FREEING in texts
    assert not waiting.abandoned() and models._left_on_card == []  # pyright: ignore[reportPrivateUsage]
    entry = models.kept(h)
    assert entry is not None and not entry.busy
    assert len(management.empties) > 0, "torch's cache emptied as before"
    # The next run goes on.
    run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")


def test_a_release_after_a_run_that_ended_normally_has_no_check_and_no_text(
    machine: StandIn, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    original = machine.load

    def load(*args: Any, **kwargs: Any) -> Any:
        found = original(*args, **kwargs)
        found.release = lambda: seen.append(getattr(waiting._named, "check", None))  # pyright: ignore[reportPrivateUsage]
        return found

    monkeypatch.setattr(machine, "load", load)
    run_video(handle(tmp_path), plans.Settings(window=5), pictures(5), RESOLUTION, "9")
    assert seen and all(each is None for each in seen)
    assert progress.FREEING not in node_texts("9")


def test_models_not_kept_wait_for_the_card_in_the_poll_after_a_cancel_and_a_second_leaves_it(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing after a run of models not kept goes through the poll (the finalizers and torch's
    cache synchronise in C): the device's work is waited for first, shown, and left by a second
    Cancel, the card marked, the models held, the cache not emptied, the next run refused."""
    happened: dict[str, Any] = {}

    def synchronize(device: Any) -> None:
        instance = server.PromptServer.instance
        assert instance is not None
        if instance.texts[-1][0] != progress.FREEING:
            return  # the run's own waits are not the one under test
        begun = world.now
        happened["empties"] = len(management.empties)

        def on_sleep() -> None:
            happened.setdefault("text", instance.texts[-1][0])
            if world.now - begun >= 30.0:
                management.interrupt_current_processing()

        world.on_sleep = on_sleep
        waiting.wait(lambda: False, 10.0)

    monkeypatch.setattr(model, "synchronize", synchronize)
    h, _ = cancelled_run(machine, tmp_path, world, monkeypatch, None, keep=False)
    assert happened["text"] == progress.FREEING
    assert node_texts("9")[-1] == progress.STOPPED
    assert waiting.abandoned() and len(models._left_on_card) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(management.empties) == happened["empties"]
    assert ticker_threads() == []
    with pytest.raises(words.Refused, match="still busy with the last work"):
        run_video(h, plans.Settings(window=5), pictures(5), RESOLUTION, "9")


@pytest.mark.parametrize("keep", [True, False])
def test_an_error_before_any_progress_event_leaves_no_freeing_text_after_a_healthy_release(
    machine: StandIn, tmp_path, world: WaitClock, monkeypatch: pytest.MonkeyPatch, keep: bool
) -> None:
    def release(happened: dict[str, Any]) -> None:
        with waiting.releasing(0):
            waiting.wait(lambda: world.now >= 10.0, 10.0)

    # The first window raises: no Progress event has been told before the plan's text only.
    cancelled_run(
        machine,
        tmp_path,
        world,
        monkeypatch,
        release,
        keep=keep,
        fail=RuntimeError("boom"),
        at=1,
    )
    texts = node_texts("9")
    assert progress.FREEING in texts and texts[-1] != progress.FREEING
    assert not waiting.abandoned()
