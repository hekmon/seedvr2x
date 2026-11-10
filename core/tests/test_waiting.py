"""The timed wait on the GPU (seedvr2x/waiting.py; DESIGN.md, Allocator, the hang of 2026-11-06),
on the CPU: the line said once past four times the unit's estimate and never under it, the floor,
the waits that say nothing (no estimate, no unit, a wait that ends in time), NVML's reading in
the line, the stacks with the process's first line alone, a stop or an error answered through the
wait and recorded, the running waits read from another thread, the unit per thread, an app's check
called at each poll of its thread's waits and ending one by what it raises, a wait past its bound
that an exception ended abandoned until one completes, torch's flags read without a CUDA call, and
the module importing no torch. Time is a fake clock whose sleep advances it: nothing sleeps."""

import logging
import os
import signal
import subprocess
import sys
import threading
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from seedvr2x import memory, model, waiting
from seedvr2x.memory import Activity

posix_signal: Any = signal

REAL_STACKS = waiting._stacks  # pyright: ignore[reportPrivateUsage]
UNIT = "segment 2/5's decode and write"
GIB = 2**30


class Clock:
    """The wait's readers: monotonic gives `now`, sleep(seconds) advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += 1
        self.now += seconds

    def until(self, end: float) -> Callable[[], bool]:
        """A done() that is true once the clock has reached `end`."""
        return lambda: self.now >= end


class World:
    """What the wait reads and writes besides its clock: NVML's answer (None for no reading),
    the card index asked, and the calls to the stacks' writer."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.activity: Activity | None = Activity(busy=6, state=8, used=int(27.18 * GIB))
        self.asked: list[int] = []
        self.stacks = 0


@pytest.fixture(autouse=True)
def world(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> Iterator[World]:
    """A fake clock in the wait's readers, memory.activity and the stacks' writer replaced, and
    the module's state (the last wait ended, the stacks written, the waits running, the unit of
    the calling thread) as a fresh process has it."""
    caplog.set_level(logging.DEBUG, logger="seedvr2x.waiting")
    found = Clock()
    state = World(found)

    def activity(index: int = 0) -> Activity | None:
        state.asked.append(index)
        return state.activity

    def stacks() -> None:
        state.stacks += 1

    monkeypatch.setattr(waiting, "monotonic", found.monotonic)
    monkeypatch.setattr(waiting, "sleep", found.sleep)
    monkeypatch.setattr(memory, "activity", activity)
    monkeypatch.setattr(waiting, "_stacks", stacks)
    monkeypatch.setattr(waiting, "_ended", None)
    monkeypatch.setattr(waiting, "_abandoned", set())
    monkeypatch.setattr(waiting, "_stacks_written", False)
    monkeypatch.setattr(waiting, "_running", {})
    monkeypatch.setattr(waiting, "_units", {})
    waiting.clear()
    waiting.check(None)
    yield state
    waiting.clear()
    waiting.check(None)


def warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_line_once_at_the_bound_and_never_under_it(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    fake = world.clock
    waiting.unit(UNIT, 30.0)
    assert waiting.bound(30.0) == 120.0
    # Polled to 119.9 s: nothing said.
    waiting.wait(fake.until(119.9), 0.1)
    assert not caplog.records and world.stacks == 0
    # Past the bound, polled on to ten times it: one line, then the end's.
    fake.now = 0.0
    waiting.wait(fake.until(1200.0), 10.0, index=1)
    assert warnings(caplog) == [
        f"{UNIT}: a wait on the GPU has lasted 2 min 10 s, the whole unit estimated at 30.0 s;"
        " CUDA has reported no error; NVML reads the card 6% busy at P8, 27.18 GiB held by its"
        f" programs; seedvr2x goes on waiting: {waiting.WAY}; every thread's stack follows",
        f"{UNIT}: the wait on the GPU ended after 20 min 0 s",
    ]
    assert [r.levelno for r in caplog.records] == [logging.WARNING] * 2
    assert world.asked == [1] and world.stacks == 1
    assert waiting.ended() is None


def test_the_way_out_is_the_apps(world: World, caplog: pytest.LogCaptureFixture) -> None:
    waiting.unit(UNIT, 30.0, "press Ctrl-C twice")
    waiting.wait(world.clock.until(130.0), 10.0)
    assert (
        "goes on waiting: press Ctrl-C twice; every thread's stack follows" in warnings(caplog)[0]
    )


def test_the_floor_and_no_estimate() -> None:
    assert waiting.bound(2.0) == waiting.FLOOR == 60.0
    assert waiting.bound(14.0) == 60.0 and waiting.bound(16.0) == 64.0
    assert waiting.bound(None) is None


def test_two_seconds_estimated_say_nothing_at_eight(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    waiting.unit(UNIT, 2.0)
    fake = world.clock
    waiting.wait(fake.until(59.0), 1.0)
    assert not caplog.records
    fake.now = 0.0
    waiting.wait(fake.until(70.0), 1.0)
    assert "has lasted 1 min 1 s, the whole unit estimated at 2.0 s" in warnings(caplog)[0]


@pytest.mark.parametrize("named", ["none", "no estimate"])
def test_no_estimate_or_no_unit_says_nothing_after_hours(
    world: World, caplog: pytest.LogCaptureFixture, named: str
) -> None:
    if named == "no estimate":
        waiting.unit(UNIT, None)
    fake = world.clock
    waiting.wait(fake.until(5 * 3600.0), 60.0)
    assert not caplog.records and world.asked == [] and world.stacks == 0
    assert waiting.ended() is None


def test_a_wait_that_ends_under_the_bound_says_nothing(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    waiting.unit(UNIT, 30.0)
    waiting.wait(world.clock.until(5.0), 1.0)
    assert not caplog.records and world.asked == []


def test_nvml_giving_none(world: World, caplog: pytest.LogCaptureFixture) -> None:
    waiting.unit(UNIT, 30.0)
    world.activity = None
    waiting.wait(world.clock.until(130.0), 10.0)
    assert (
        "CUDA has reported no error; NVML gives no reading of the card; seedvr2x"
        in (warnings(caplog)[0])
    )


def test_an_unknown_performance_state(world: World, caplog: pytest.LogCaptureFixture) -> None:
    waiting.unit(UNIT, 30.0)
    world.activity = Activity(busy=100, state=32, used=GIB)
    waiting.wait(world.clock.until(130.0), 10.0)
    assert (
        "NVML reads the card 100% busy at an unknown performance state, 1.00 GiB held"
        in (warnings(caplog)[0])
    )


def test_the_stacks_come_with_the_process_s_first_line_alone(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    waiting.unit(UNIT, 30.0)
    fake = world.clock
    waiting.wait(fake.until(130.0), 10.0)
    fake.now = 0.0
    waiting.wait(fake.until(130.0), 10.0)
    said = [m for m in warnings(caplog) if "has lasted" in m]
    assert len(said) == 2 and world.stacks == 1
    assert said[0].endswith("; every thread's stack follows")
    assert not said[1].endswith("follows") and said[1].endswith(waiting.WAY)


def test_the_real_stacks_are_written_to_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # faulthandler writes to a file descriptor: a real file stands in for stderr.
    with (tmp_path / "stderr").open("w+") as file:
        monkeypatch.setattr(sys, "stderr", file)
        REAL_STACKS()
        file.flush()
        file.seek(0)
        text = file.read()
    assert "Current thread" in text or "Thread 0x" in text
    assert "test_the_real_stacks_are_written_to_stderr" in text


class Stop(BaseException):
    """A stop, as a signal's handler raises one."""


@pytest.mark.parametrize("error", [KeyboardInterrupt(), Stop("stop"), RuntimeError("CUDA error")])
def test_a_stop_is_answered_and_recorded(world: World, error: BaseException) -> None:
    waiting.unit(UNIT, 30.0)
    fake = world.clock

    def done() -> bool:
        if fake.now >= 45.0:
            raise error
        return False

    with pytest.raises(type(error)) as raised:
        waiting.wait(done, 5.0)
    assert raised.value is error
    ended = waiting.ended()
    assert ended is not None and ended.unit == UNIT and ended.seconds == 45.0
    # Known by the very exception, which the record doesn't keep alive (its frames would).
    assert waiting.ended_by(error) == ended and ended.kind is type(error)
    assert waiting.ended_by(type(error)()) is None
    # The wait is no longer running.
    assert waiting.longest() is None and not waiting.stalled()


def test_a_wait_with_no_unit_ended_by_a_stop_is_recorded_nameless(world: World) -> None:
    fake = world.clock

    def done() -> bool:
        if fake.now >= 3.0:
            raise Stop
        return False

    with pytest.raises(Stop):
        waiting.wait(done, 1.0)
    ended = waiting.ended()
    assert ended is not None and ended.unit is None and ended.seconds == 3.0


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="sends SIGUSR2 with os.kill; Windows has no SIGUSR2 (native, 2026-11-10)",
)
def test_a_signal_between_two_polls_ends_the_wait_after_the_line(
    world: World, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A handler that raises inside sleep, as stop.py's and the server's do: the line was said
    # first (the poll that passed the bound), the exception propagates and is recorded.
    def handler(number: int, frame: Any) -> None:
        raise Stop("signal")

    before = signal.signal(posix_signal.SIGUSR2, handler)
    waiting.unit(UNIT, 30.0)
    fake = world.clock
    plain = fake.sleep
    fired: list[float] = []

    def sleep(seconds: float) -> None:
        if fake.now > 120.0:
            fired.append(fake.now)
            os.kill(os.getpid(), posix_signal.SIGUSR2)
        plain(seconds)

    monkeypatch.setattr(waiting, "sleep", sleep)
    try:
        with pytest.raises(Stop):
            waiting.wait(lambda: False, 10.0)
    finally:
        signal.signal(posix_signal.SIGUSR2, before)
    assert fired == [130.0]
    said = warnings(caplog)
    assert len(said) == 1 and "a wait on the GPU has lasted 2 min 10 s" in said[0]
    ended = waiting.ended()
    assert ended is not None and ended.unit == UNIT and ended.seconds == 130.0
    assert ended.kind is Stop and not any("ended after" in m for m in said)


def test_the_longest_wait_and_the_stall_seen_from_another_thread(world: World) -> None:
    # The CLI's stop line and the server's /health and /metrics read these while a wait runs.
    fake = world.clock
    arrived, go = threading.Event(), threading.Event()
    finished = threading.Event()
    problems: list[BaseException] = []

    def done() -> bool:
        arrived.set()
        assert go.wait(5.0)
        go.clear()
        return finished.is_set()

    def run() -> None:
        try:
            with waiting.named(UNIT, 30.0):
                waiting.wait(done, 0.0)
        except BaseException as error:
            problems.append(error)

    def step(now: float) -> None:
        fake.now = now
        go.set()
        assert arrived.wait(5.0)
        arrived.clear()

    assert waiting.longest() is None and not waiting.stalled()
    thread = threading.Thread(target=run)
    thread.start()
    assert arrived.wait(5.0)
    arrived.clear()
    assert waiting.longest() == 0.0 and not waiting.stalled()
    step(50.0)
    assert waiting.longest() == 50.0 and not waiting.stalled()
    step(130.0)
    assert waiting.longest() == 130.0 and waiting.stalled()
    finished.set()
    go.set()
    thread.join(5.0)
    assert not thread.is_alive() and not problems
    assert waiting.longest() is None and not waiting.stalled()


def test_units_are_per_thread(world: World, caplog: pytest.LogCaptureFixture) -> None:
    waiting.unit(UNIT, 30.0)
    fake = world.clock
    seen: list[BaseException] = []

    def run() -> None:
        try:
            waiting.wait(fake.until(10_000.0), 100.0)
        except BaseException as error:
            seen.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(5.0)
    assert not seen and not caplog.records
    # The main thread's unit is its own still.
    fake.now = 0.0
    waiting.wait(fake.until(130.0), 10.0)
    assert "has lasted 2 min 10 s" in warnings(caplog)[0]


def test_another_thread_reads_a_threads_unit_and_whether_it_waits(world: World) -> None:
    # What a watcher of the main thread reads (stop.Watch): the unit named in that thread, kept
    # beside the thread-local one, given back by named() and cleared by clear(); inside a wait
    # or not.
    ident = threading.get_ident()
    assert waiting.unit_of(ident) is None and not waiting.waiting_in(ident)
    waiting.unit(UNIT, 30.0)
    seen: list[object] = []
    thread = threading.Thread(target=lambda: seen.append(waiting.unit_of(ident)))
    thread.start()
    thread.join(5.0)
    assert seen == [waiting.Unit(UNIT, 30.0)]
    with waiting.named("inner", 5.0):
        assert waiting.unit_of(ident) == waiting.Unit("inner", 5.0)
    assert waiting.unit_of(ident) == waiting.Unit(UNIT, 30.0)

    def inside() -> bool:
        assert waiting.waiting_in(ident)
        return True

    waiting.wait(inside, 0.1)
    assert not waiting.waiting_in(ident)
    waiting.clear()
    assert waiting.unit_of(ident) is None


def test_a_thread_held_in_one_call_is_said_by_its_watcher(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    unit = waiting.Unit(UNIT, 30.0, "press Ctrl-C twice")
    waiting.held(unit, 130.0, "run.py:210 in decode", "SIGTERM twice ends it", index=1)
    waiting.held(unit, 140.0, "run.py:210 in decode")
    said = warnings(caplog)
    assert said == [
        f"{UNIT}: the main thread has been held in one call for 2 min 10 s, at run.py:210 in"
        " decode, the whole unit estimated at 30.0 s; no error has been raised; NVML reads the"
        " card 6% busy at P8, 27.18 GiB held by its programs; seedvr2x goes on waiting: press"
        " Ctrl-C twice; SIGTERM twice ends it; every thread's stack follows",
        f"{UNIT}: the main thread has been held in one call for 2 min 20 s, at run.py:210 in"
        " decode, the whole unit estimated at 30.0 s; no error has been raised; NVML reads the"
        " card 6% busy at P8, 27.18 GiB held by its programs; seedvr2x goes on waiting: press"
        " Ctrl-C twice",
    ]
    # The stacks once for the process, shared with a wait's line.
    assert world.asked == [1, 0] and world.stacks == 1


def test_named_gives_back_the_unit_before(world: World, caplog: pytest.LogCaptureFixture) -> None:
    fake = world.clock
    waiting.unit("outer", 30.0)
    with waiting.named("inner", 100.0):
        waiting.wait(fake.until(130.0), 10.0)
        assert not caplog.records  # 400 s is the inner's bound
        with pytest.raises(Stop), waiting.named("raising", None):
            raise Stop
    fake.now = 0.0
    waiting.wait(fake.until(130.0), 10.0)
    assert warnings(caplog)[0].startswith("outer: a wait on the GPU has lasted")
    # And none where there was none.
    waiting.clear()
    with waiting.named("only", 30.0):
        pass
    caplog.clear()
    fake.now = 0.0
    waiting.wait(fake.until(10_000.0), 100.0)
    assert not caplog.records


class Cancel(BaseException):
    """An app's cancel, as the check raises one."""


def test_the_check_is_called_at_each_poll_after_the_sleep_and_a_wait_done_at_once_never_calls_it(
    world: World,
) -> None:
    fake, calls = world.clock, []
    waiting.check(lambda: calls.append(fake.now))
    waiting.wait(fake.until(0.0), 1.0)
    assert calls == []
    waiting.wait(fake.until(3.0), 1.0)
    assert calls == [1.0, 2.0, 3.0]


def test_what_the_check_raises_ends_the_wait_recorded_past_its_bound_or_not(world: World) -> None:
    fake = world.clock

    def hook() -> None:
        if fake.now >= 50.0:
            raise Cancel

    waiting.check(hook)
    waiting.unit(UNIT, 30.0)  # bound 120 s: the cancel comes under it
    with pytest.raises(Cancel):
        waiting.wait(lambda: False, 10.0)
    ended = waiting.ended()
    assert ended is not None and ended.unit == UNIT and ended.seconds == 50.0
    assert ended.kind is Cancel
    assert not waiting.abandoned(), "under its bound: a cancel of a slow wait, not a stall"
    assert waiting.longest() is None


def test_a_wait_past_its_bound_that_the_check_ended_is_abandoned_until_a_release_completes(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    fake = world.clock

    def hook() -> None:
        if fake.now >= 200.0:
            raise Cancel

    waiting.check(hook)
    waiting.unit(UNIT, 30.0)
    with pytest.raises(Cancel):
        waiting.wait(lambda: False, 10.0)
    assert len(warnings(caplog)) == 1, "the line was said at 130 s"
    assert waiting.abandoned()
    # A wait that completes no longer clears it; a release of the card whose waits all completed
    # does.
    waiting.check(None)
    fake.now = 0.0
    waiting.wait(fake.until(5.0), 1.0)
    assert waiting.abandoned()
    with waiting.releasing():
        fake.now = 0.0
        waiting.wait(fake.until(5.0), 1.0)
    assert not waiting.abandoned()


def test_an_error_from_the_query_past_the_bound_abandons_it_too(world: World) -> None:
    fake = world.clock

    def done() -> bool:
        if fake.now >= 200.0:
            raise RuntimeError("CUDA error")
        return False

    waiting.unit(UNIT, 30.0)
    with pytest.raises(RuntimeError):
        waiting.wait(done, 10.0)
    assert waiting.abandoned()


def test_the_check_is_the_calling_threads_and_checking_gives_the_one_before_back(
    world: World,
) -> None:
    fake, outer, inner = world.clock, [], []
    seen: list[BaseException] = []
    waiting.check(lambda: outer.append(1))

    def run() -> None:
        try:
            waiting.wait(fake.until(3.0), 1.0)
        except BaseException as error:
            seen.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(5.0)
    assert not seen and outer == [], "another thread's waits call the check of their own"
    fake.now = 0.0
    with waiting.checking(lambda: inner.append(1)):
        waiting.wait(fake.until(2.0), 1.0)
    assert inner == [1, 1] and outer == []
    fake.now = 0.0
    waiting.wait(fake.until(2.0), 1.0)
    assert outer == [1, 1]


def test_cuda_is_used_where_torch_initialised_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # This environment has no GPU (CUDA_VISIBLE_DEVICES is empty): torch's flag is false.
    assert not torch.cuda.is_initialized() and not waiting.cuda_used()
    for answer in (True, False):
        stand_in = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_initialized=lambda a=answer: a)
        )
        monkeypatch.setitem(sys.modules, "torch", stand_in)
        assert waiting.cuda_used() is answer
    monkeypatch.delitem(sys.modules, "torch")
    assert not waiting.cuda_used()


@pytest.mark.parametrize(
    ("error", "failed"),
    [
        (RuntimeError("CUDA error: an illegal memory access was encountered"), True),
        (RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"), True),
        (RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"), True),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"), False),
        (RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"), False),
        (ValueError("CUDA error"), False),
        (RuntimeError("something else"), False),
        (KeyboardInterrupt(), False),
    ],
)
def test_which_errors_are_cuda_s(error: BaseException, failed: bool) -> None:
    assert waiting.cuda_failed(error) is failed


def test_out_of_memory_is_no_failed_card() -> None:
    assert not waiting.cuda_failed(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate"))
    # Torch's out of memory is a RuntimeError of CUDA error's family in some builds' words.
    assert not waiting.cuda_failed(torch.OutOfMemoryError("CUDA error: out of memory"))


def test_torch_s_accelerator_error_is_a_failed_card() -> None:
    error_type = getattr(torch, "AcceleratorError", None)
    if error_type is None:
        pytest.skip("this torch has no AcceleratorError")
    assert waiting.cuda_failed(error_type("any message"))


def test_no_torch_no_failed_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch")
    assert not waiting.cuda_failed(RuntimeError("CUDA error: an illegal memory access"))


def test_a_synchronize_on_the_cpu_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*arguments: Any) -> None:
        raise AssertionError("waited")

    monkeypatch.setattr(waiting, "wait", fail)
    model.synchronize(torch.device("cpu"))


def test_importing_it_imports_no_torch() -> None:
    code = "import seedvr2x.waiting, sys; assert 'torch' not in sys.modules, 'torch imported'"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_line_that_fails_never_ends_the_wait(
    world: World, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The GPU reproduction of 2026-11-06: a replaced reading of the card raised in the line, and
    # the exception ended the run. The line's failure is said at debug, the wait goes on.
    caplog.set_level(logging.DEBUG, logger="seedvr2x.waiting")

    def broken(index: int) -> None:
        raise TypeError("unexpected keyword argument 'busy'")

    monkeypatch.setattr(memory, "activity", broken)
    waiting.unit(UNIT, 30.0)
    calls = iter([False] * 30 + [True])
    waiting.wait(lambda: next(calls), 10.0)
    said = warnings(caplog)
    assert "NVML gives no reading of the card" in said[0]
    assert any("can't be read" in r.message for r in caplog.records if r.levelno == logging.DEBUG)

    def failing(current: waiting.Unit, seconds: float, index: int) -> None:
        raise RuntimeError("the line itself")

    monkeypatch.setattr(waiting, "_say", failing)
    calls = iter([False] * 30 + [True])
    waiting.wait(lambda: next(calls), 10.0)
    assert any("can't be said" in r.message for r in caplog.records)


# ---- Abandoned per card, and by a release --------------------------------------------------------


def stall(world: World, index: int) -> None:
    """A wait on card `index` past its bound that a check ends."""
    fake = world.clock
    fake.now = 0.0

    def hook() -> None:
        if fake.now >= 200.0:
            raise Cancel

    waiting.check(hook)
    waiting.unit(UNIT, 30.0)
    with pytest.raises(Cancel):
        waiting.wait(lambda: False, 10.0, index)
    waiting.check(None)
    waiting.clear()


def test_the_mark_is_per_card_and_a_release_clears_its_own_card_only(world: World) -> None:
    stall(world, 1)
    assert waiting.abandoned(1) and not waiting.abandoned(0) and not waiting.abandoned()
    stall(world, 0)
    assert waiting.abandoned(0)
    with waiting.releasing(0):
        world.clock.now = 0.0
        waiting.wait(world.clock.until(5.0), 1.0, 0)
    assert not waiting.abandoned(0) and waiting.abandoned(1), "the other card is untouched"
    with waiting.releasing(1):
        pass
    assert not waiting.abandoned(1)


def test_a_wait_ended_inside_a_release_marks_the_card_with_no_unit_and_under_any_bound(
    world: World,
) -> None:
    fake = world.clock

    def hook() -> None:
        if fake.now >= 20.0:
            raise Cancel

    waiting.check(hook)
    assert getattr(waiting._named, "unit", None) is None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(Cancel):
        with waiting.releasing(2):
            waiting.wait(lambda: False, 10.0, 0)
    assert waiting.abandoned(2), "the release's card"
    assert not waiting.abandoned(0), "a wait outside its bound marks nothing of its own"
    # Outside a release the same wait marks nothing.
    fake.now = 0.0
    with pytest.raises(Cancel):
        waiting.wait(lambda: False, 10.0, 3)
    assert not waiting.abandoned(3)


def test_a_completed_wait_clears_nothing_and_a_release_that_raised_does_not_clear(
    world: World,
) -> None:
    stall(world, 0)
    world.clock.now = 0.0
    waiting.wait(world.clock.until(5.0), 1.0)
    assert waiting.abandoned()
    with pytest.raises(ValueError):
        with waiting.releasing(0):
            raise ValueError("not a wait")
    assert waiting.abandoned()
    with waiting.releasing(0):
        pass
    assert not waiting.abandoned()


def test_a_wait_that_raised_and_was_caught_in_a_release_keeps_the_mark(world: World) -> None:
    fake = world.clock
    stall(world, 0)
    fake.now = 0.0
    with waiting.releasing(0):
        waiting.check(lambda: (_ for _ in ()).throw(Cancel()))
        try:
            waiting.wait(lambda: False, 10.0)
        except Cancel:
            pass
        waiting.check(None)
        fake.now = 0.0
        waiting.wait(fake.until(5.0), 1.0)
    assert waiting.abandoned(0), "the block ended cleanly, but one of its waits did not complete"
    with waiting.releasing(0):
        fake.now = 0.0
        waiting.wait(fake.until(5.0), 1.0)
    assert not waiting.abandoned(0)


# ---- A card a CUDA error failed ---------------------------------------------------------------

ILLEGAL = "CUDA error: an illegal memory access was encountered"


def test_a_cuda_error_from_the_query_marks_the_card_failed_whatever_the_bound(world: World) -> None:
    # No unit, no bound: the error alone marks its card, with its text's first line, and only it.
    def done() -> bool:
        raise RuntimeError(f"{ILLEGAL}\nCUDA kernel errors might be asynchronously reported")

    with pytest.raises(RuntimeError):
        waiting.wait(done, 0.01, index=1)
    assert waiting.failed(1) == ILLEGAL and waiting.abandoned(1)
    assert waiting.failed(0) is None and not waiting.abandoned(0)


def test_out_of_memory_and_other_errors_mark_nothing(world: World) -> None:
    torch = pytest.importorskip("torch")
    for error in (torch.OutOfMemoryError("CUDA out of memory"), ValueError("not CUDA's")):

        def done(raised: BaseException = error) -> bool:
            raise raised

        with pytest.raises(type(error)):
            waiting.wait(done, 0.01)
    assert waiting.failed(0) is None and not waiting.abandoned(0)
    with pytest.raises(ValueError, match="not a CUDA error"):
        waiting.fail(0, ValueError("no"))


def test_a_failed_card_stays_marked_through_a_release_that_completes(world: World) -> None:
    fake = world.clock
    waiting.fail(0, RuntimeError(ILLEGAL))
    waiting.fail(0, RuntimeError("CUDA error: unspecified launch failure"))
    assert waiting.failed(0) == ILLEGAL, "the first error's text is kept"
    with waiting.releasing():
        fake.now = 0.0
        waiting.wait(fake.until(2.0), 1.0)
    assert waiting.abandoned() and waiting.failed(0) == ILLEGAL


# ---- A mark the card itself clears (settle) ---------------------------------------------------


class Work:
    """Work queued on a card: a query that says whether it finished, and counts how often it was
    asked. Nothing in it waits."""

    def __init__(self) -> None:
        self.finished = False
        self.asked = 0

    def query(self) -> bool:
        self.asked += 1
        return self.finished


def abandon_in_release(world: World, work: Work, extra: Work | None = None) -> None:
    """A release of card 0 left by a second Cancel at its wait for `work`, having recorded `extra`
    (the later waits' work) first (expect)."""
    fake = world.clock

    def hook() -> None:
        if fake.now >= 20.0:
            raise Cancel

    waiting.check(hook)
    with pytest.raises(Cancel):
        with waiting.releasing(0):
            if extra is not None:
                waiting.expect(extra.query)
            waiting.wait(work.query, 10.0)
    waiting.check(None)
    assert waiting.abandoned()


def test_settle_clears_the_mark_once_all_the_work_is_done_asked_without_waiting(
    world: World,
) -> None:
    stuck, later = Work(), Work()
    abandon_in_release(world, stuck, later)
    slept = world.clock.now
    assert not waiting.settle(), "nothing finished"
    stuck.finished = True
    assert not waiting.settle(), "the work the release had yet to wait for counts"
    assert waiting.abandoned()
    later.finished = True
    assert waiting.settle() and not waiting.abandoned()
    assert world.clock.now == slept, "the queries never slept"
    assert waiting.settle(), "a card not abandoned is settled"


def test_settle_clears_after_a_wait_past_its_bound_ended_outside_a_release(world: World) -> None:
    work = Work()
    fake = world.clock
    fake.now = 0.0

    def hook() -> None:
        if fake.now >= 200.0:
            raise Cancel

    waiting.check(hook)
    waiting.unit(UNIT, 30.0)
    with pytest.raises(Cancel):
        waiting.wait(work.query, 10.0)
    waiting.check(None)
    assert waiting.abandoned() and not waiting.settle()
    work.finished = True
    assert waiting.settle() and not waiting.abandoned()


def test_settle_never_clears_a_failed_card_whatever_the_work_says(world: World) -> None:
    work = Work()
    work.finished = True
    abandon_in_release(world, Work(), work)
    waiting.fail(0, RuntimeError(ILLEGAL))
    assert not waiting.settle() and waiting.abandoned() and waiting.failed(0) == ILLEGAL


def test_settle_with_nothing_recorded_or_a_query_that_raises_keeps_the_mark(world: World) -> None:
    waiting._abandoned.add(3)  # pyright: ignore[reportPrivateUsage]
    assert not waiting.settle(3), "nothing to ask about: not known to be finished"

    def raises() -> bool:
        raise RuntimeError(ILLEGAL)

    abandon_in_release(world, Work(), None)
    waiting._pending[0] = [raises]  # pyright: ignore[reportPrivateUsage]
    assert not waiting.settle()
    assert waiting.failed(0) == ILLEGAL, "a CUDA error from the query marks the card failed"


def test_a_release_that_completes_forgets_the_work_it_had_kept(world: World) -> None:
    abandon_in_release(world, Work())
    with waiting.releasing(0):
        pass
    assert not waiting.abandoned()
    assert 0 not in waiting._pending  # pyright: ignore[reportPrivateUsage]
    waiting.expect(Work().query)  # outside a release: nothing
    assert 0 not in waiting._pending  # pyright: ignore[reportPrivateUsage]


def test_pend_adds_queries_to_an_abandoned_card_and_settle_needs_all_of_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(waiting, "_abandoned", {0})
    monkeypatch.setattr(waiting, "_pending", {0: [lambda: True]})
    monkeypatch.setattr(waiting, "_failed", {})
    later = [False]
    waiting.pend(0, lambda: later[0])
    assert waiting.settle(0) is False and waiting.abandoned(0)
    later[0] = True
    assert waiting.settle(0) is True and not waiting.abandoned(0)


def test_pend_records_nothing_for_a_card_marked_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(waiting, "_abandoned", set())
    monkeypatch.setattr(waiting, "_pending", {})
    monkeypatch.setattr(waiting, "_failed", {})
    waiting.fail(0, RuntimeError("CUDA error: an illegal memory access was encountered"))
    waiting.pend(0, lambda: True)
    assert waiting._pending == {}  # pyright: ignore[reportPrivateUsage]
    assert waiting.settle(0) is False
