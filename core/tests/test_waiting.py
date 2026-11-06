"""The timed wait on the GPU (seedvr2x/waiting.py; DESIGN.md, Allocator, the hang of 2026-11-06),
on the CPU: the line said once past four times the unit's estimate and never under it, the floor,
the waits that say nothing (no estimate, no unit, a wait that ends in time), NVML's reading in
the line, the stacks with the process's first line alone, a stop or an error answered through the
wait and recorded, the running waits read from another thread, the unit per thread, torch's flags
read without a CUDA call, and the module importing no torch. Time is a fake clock whose sleep
advances it: nothing sleeps."""

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
    monkeypatch.setattr(waiting, "_stacks_written", False)
    monkeypatch.setattr(waiting, "_running", {})
    waiting.clear()
    yield state
    waiting.clear()


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


def test_a_signal_between_two_polls_ends_the_wait_after_the_line(
    world: World, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A handler that raises inside sleep, as stop.py's and the server's do: the line was said
    # first (the poll that passed the bound), the exception propagates and is recorded.
    def handler(number: int, frame: Any) -> None:
        raise Stop("signal")

    before = signal.signal(signal.SIGUSR2, handler)
    waiting.unit(UNIT, 30.0)
    fake = world.clock
    plain = fake.sleep
    fired: list[float] = []

    def sleep(seconds: float) -> None:
        if fake.now > 120.0:
            fired.append(fake.now)
            os.kill(os.getpid(), signal.SIGUSR2)
        plain(seconds)

    monkeypatch.setattr(waiting, "sleep", sleep)
    try:
        with pytest.raises(Stop):
            waiting.wait(lambda: False, 10.0)
    finally:
        signal.signal(signal.SIGUSR2, before)
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
