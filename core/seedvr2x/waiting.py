"""Waits on the GPU, timed (DESIGN.md, Allocator, the hang of 2026-11-06): a wait on the work
queued on the device, polled so that a signal's handler runs between two polls (stop.py's, the
image server's), that says once when it has lasted longer than several times its unit's
estimate, and goes on waiting. Not a timeout: a run can't tell a GPU whose driver stopped without
a word from a slow one (a card shared with a game), and the user decides. On place-study's night
of 2026-11-06 a decode's wait polled for 34 minutes with nothing said, the card 6% busy, an NVIDIA
driver error in Windows' log 3.5 s into it and CUDA in the guest never told.

The app names the unit in progress, its estimate and the way out in its own words (unit, named):
the CLI before each unit, from its calibration (run.py), the image server for each step of a
picture, from its warm-up. A wait with no unit named, or a unit with no estimate (the
calibration's own, an input copy, a picture of the CLI's, the server's warm-up, ComfyUI's nodes),
says nothing. The waits running are kept per thread, for the server's /health and /metrics, and
the last one a stop or an error ended, for the stop's line (the CLI's main); reading them makes
no CUDA call.

Importing it imports no torch."""

import faulthandler
import logging
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from seedvr2x import clock

logger = logging.getLogger(__name__)

# How many times its unit's estimate a wait lasts before it is said, and the least it lasts.
# A wait is a part of its unit, so its unit's whole estimate is over the wait's own. The slowest
# units measured against their estimates came at about 1.5 times them: a single unit missing by
# up to 22 to 31% (DESIGN.md, Memory planner, Time estimates, RTX 5090, 2026-10-23), milestone
# 7's resumes calibrated at 0.72 and 0.73 s a frame against the first night's 0.53 (1.36 times,
# 2026-10-24 to 27), October's full card at 1.4 to 1.5 times (research/docs/small-cards.md). Four
# times leaves 2.7 times over the slowest seen. The floor keeps the short units, an encode of a
# few seconds or a phase's change, clear of a desktop's or WSL2's pauses; a line said wrongly
# costs a warning and the stacks, the run going on (design, 2026-11-06).
TIMES = 4.0
FLOOR = 60.0

# The way out in the library's words, for an app that gives none.
WAY = "a stop keeps every unit done, and the same command resumes"


@dataclass(frozen=True)
class Unit:
    """The unit a thread's waits are part of: its name as the app says it ("segment 2/5's decode
    and write"), its estimate in the run's own seconds (clock.py), None without one, and the way
    out in the app's words."""

    name: str
    estimate: float | None
    way: str = WAY


@dataclass(frozen=True)
class Ended:
    """A wait an exception ended (a stop's, or an error's): its unit's name, None without one,
    how long it had lasted, in the run's own seconds, and the exception's identity (id), by which
    a stop's line tells its own from an earlier one's (ended_by). Not the exception itself: its
    traceback would keep the run's frames, the models among them, alive for the process's life
    (the nodes' runs must leave nothing behind: DESIGN.md, Repository and apps)."""

    unit: str | None
    seconds: float
    error: int
    kind: type[BaseException]


@dataclass
class _Running:
    began: float
    unit: Unit | None
    past: bool = False


# The readers, replaced by the tests.
monotonic: Callable[[], float] = time.monotonic
sleep: Callable[[float], None] = time.sleep

_named = threading.local()
_lock = threading.Lock()
_running: dict[int, _Running] = {}
_ended: Ended | None = None
_stacks_written = False


def bound(estimate: float | None) -> float | None:
    """The run's seconds after which a wait of a unit estimated at `estimate` is said, None
    without an estimate."""
    return None if estimate is None else max(FLOOR, TIMES * estimate)


def unit(name: str, estimate: float | None, way: str = WAY) -> None:
    """Name the unit the calling thread's waits are part of from now on (Unit)."""
    _named.unit = Unit(name, estimate, way)


def clear() -> None:
    """No unit named for the calling thread's waits from now on: they say nothing."""
    _named.unit = None


@contextmanager
def named(name: str, estimate: float | None, way: str = WAY) -> Generator[None]:
    """The calling thread's waits part of that unit inside the with block, the unit named before
    given back after."""
    before: Unit | None = getattr(_named, "unit", None)
    unit(name, estimate, way)
    try:
        yield
    finally:
        _named.unit = before


def wait(done: Callable[[], bool], poll: float, index: int = 0) -> None:
    """Return once done() is true, called every `poll` seconds, a signal's handler running
    between two calls. Past its unit's bound (bound), the wait is said once at WARNING, with NVML's
    reading of card `index` (CUDA's numbering) and the way out, every thread's stack after the
    first such line of the process, and it goes on; its end is then said too. done is an
    event's query: it raises where CUDA reports an error."""
    global _ended
    current: Unit | None = getattr(_named, "unit", None)
    limit = None if current is None else bound(current.estimate)
    began = monotonic()
    key = threading.get_ident()
    state = _Running(began, current)
    with _lock:
        _running[key] = state
    try:
        while not done():
            sleep(poll)
            if limit is not None and not state.past and monotonic() - began > limit:
                state.past = True
                assert current is not None
                try:
                    _say(current, monotonic() - began, index)
                except Exception as error:
                    # Never the run's failure: the wait goes on, whatever the line met.
                    logger.debug("the long wait's line can't be said: %r", error)
    except BaseException as error:
        name = None if current is None else current.name
        _ended = Ended(name, monotonic() - began, id(error), type(error))
        raise
    finally:
        with _lock:
            _running.pop(key, None)
    if state.past:
        assert current is not None
        logger.warning(
            "%s: the wait on the GPU ended after %s",
            current.name,
            clock.spoken(clock.spent(monotonic() - began)),
        )


def longest() -> float | None:
    """How long the longest wait running has lasted, in the run's own seconds, None without
    one."""
    now = monotonic()
    with _lock:
        return max((now - each.began for each in _running.values()), default=None)


def stalled() -> bool:
    """Whether a wait running has passed its unit's bound and was said."""
    with _lock:
        return any(each.past for each in _running.values())


def ended() -> Ended | None:
    """The last wait an exception ended, None if none did."""
    return _ended


def ended_by(error: BaseException) -> Ended | None:
    """The wait `error` ended, None where it ended none (raised outside a wait). `error` alive
    while it is asked, so that its identity is no other's."""
    found = _ended
    if found is None or found.error != id(error) or found.kind is not type(error):
        return None
    return found


def cuda_used() -> bool:
    """Whether this process has used CUDA: torch imported and its CUDA initialised, read from
    torch's own flag, no CUDA call made."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    return bool(torch.cuda.is_initialized())


def cuda_failed(error: BaseException) -> bool:
    """Whether error is a CUDA error other than out of memory, after which the card's state is
    untrusted and its work may never end: torch's AcceleratorError, or a RuntimeError CUDA,
    cuBLAS or cuDNN reported. cuBLAS's and cuDNN's refused allocations are out of memory
    (CUBLAS_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED)."""
    torch = sys.modules.get("torch")
    if torch is None or isinstance(error, torch.OutOfMemoryError):
        return False
    accelerator = getattr(torch, "AcceleratorError", None)
    if accelerator is not None and isinstance(error, accelerator):
        return True
    text = str(error)
    if "_ALLOC_FAILED" in text:
        return False
    return isinstance(error, RuntimeError) and any(
        word in text for word in ("CUDA error", "CUBLAS_STATUS", "cuDNN error")
    )


def _say(current: Unit, seconds: float, index: int) -> None:
    global _stacks_written
    # Imported here: memory imports nothing of torch at its import either, but this module is
    # model.py's and blockswap.py's, and memory's NVML reading is needed only now.
    from seedvr2x import memory

    estimate = current.estimate
    assert estimate is not None
    try:
        found = memory.activity(index)
    except Exception as error:
        # Never raises by its contract; a replacement of it might (research/scripts' own).
        logger.debug("the card's activity can't be read: %r", error)
        found = None
    if found is None:
        reading = "NVML gives no reading of the card"
    else:
        state = "an unknown performance state" if found.state == 32 else f"P{found.state}"
        reading = (
            f"NVML reads the card {found.busy}% busy at {state}, {found.used / 2**30:.2f} GiB"
            " held by its programs"
        )
    stacks = not _stacks_written
    _stacks_written = True
    logger.warning(
        "%s: a wait on the GPU has lasted %s, the whole unit estimated at %s; CUDA has reported"
        " no error; %s; seedvr2x goes on waiting: %s%s",
        current.name,
        clock.spoken(clock.spent(seconds)),
        _seconds(clock.wall(estimate)),
        reading,
        current.way,
        "; every thread's stack follows" if stacks else "",
    )
    if stacks:
        _stacks()


def _stacks() -> None:
    """Every thread's Python stack, on stderr, after the lines already logged."""
    for handler in logging.getLogger().handlers:
        with suppress(Exception):
            handler.flush()
    try:
        sys.stderr.flush()
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except (AttributeError, ValueError, OSError) as error:
        # stderr with no descriptor (a test's capture, a closed pipe): the line stands alone.
        logger.debug("the threads' stacks can't be written to stderr (%s)", error)


def _seconds(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 60 else clock.spoken(seconds)
