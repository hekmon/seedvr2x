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
calibration's own, an input copy, a picture of the CLI's, the server's warm-up, the nodes'
before their first unit is measured), says nothing. The waits running are kept per thread, for the
server's /health and /metrics, and the last one a stop or an error ended, for the stop's line
(the CLI's main); reading them makes no CUDA call.

An app may also set a check, a callable the poll calls once a poll for the calling thread's waits
(check, checking), which may raise and so end the wait: the nodes set their Cancel's, so that a wait
that never ends can be left. A wait past its bound that an exception ended, or any wait inside a
release (releasing) that one ended, whatever its bound, marks its card abandoned (abandoned): the
card may never answer again, and whoever gives the models back (Models.release) leaves alone what
would wait on it. The mark is per card, by CUDA's index, and only a release on that card whose
waits all completed clears it, no other wait: a card that answered one wait may not answer the
next, and a process that ran on several cards keeps each one's state apart. The work the
abandoned waits were waiting for is kept as events' queries (wait, expect), which never wait: an
app asks settle, without waiting, whether it has all finished, and gives the models back then, a
release that completes clearing the mark, so that a second Cancel pressed in impatience on a
healthy card costs no restart.

A CUDA error other than torch's OutOfMemoryError (cuda_failed: a "CUDA error: out of memory" a
kernel met is one) marks its card too, failed (fail, failed), whether a wait met it or the caller
did, and no release clears that mark: the card's state is untrusted from then on, and under
cudaMallocAsync a tensor's free is itself a call to CUDA, which reports the error again inside
torch's deleter, where it cannot raise and so aborts the process (the nodes on Windows,
2026-11-07: "Fatal Python error: Aborted" at a free of the VAE's causal memory, no message before
it). So whoever would free on that card leaves it as it is, as far as seedvr2x's code goes: the
library's passes keep what they hold (model.unless_failed), and an app keeps the error, whose frames
hold the rest, for the process's life; the process is for restarting.

Importing it imports no torch."""

import faulthandler
import logging
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field

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
class _Release:
    index: int
    ended: bool = False  # a wait in the block was ended by an exception, caught or not
    pending: list[Callable[[], bool]] = field(default_factory=lambda: [])  # see expect


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
# The unit each thread's waits are part of, by thread (get_ident), beside the thread-local one: a
# thread that watches the main thread (the CLI's stop.Watch) reads the main thread's unit, which
# only its own thread-local can't give.
_units: dict[int, Unit] = {}
_running: dict[int, _Running] = {}
_ended: Ended | None = None
_stacks_written = False
# The cards (CUDA's indices) on which a wait was ended by an exception past its bound, or inside a
# release, and no release has completed since.
_abandoned: set[int] = set()
# For each abandoned card, the queries (an event's, which never waits) of the work the abandoned
# wait or the release was waiting for, until a release completes or settle finds it all done.
_pending: dict[int, list[Callable[[], bool]]] = {}
# The cards (CUDA's indices) a CUDA error other than out of memory was met on, with its text's first
# line; never cleared (fail).
_failed: dict[int, str] = {}


def bound(estimate: float | None) -> float | None:
    """The run's seconds after which a wait of a unit estimated at `estimate` is said, None
    without an estimate."""
    return None if estimate is None else max(FLOOR, TIMES * estimate)


def unit(name: str, estimate: float | None, way: str = WAY) -> None:
    """Name the unit the calling thread's waits are part of from now on (Unit)."""
    _set(Unit(name, estimate, way))


def clear() -> None:
    """No unit named for the calling thread's waits from now on: they say nothing."""
    _set(None)


def _set(current: Unit | None) -> None:
    _named.unit = current
    key = threading.get_ident()
    with _lock:
        if current is None:
            _units.pop(key, None)
        else:
            _units[key] = current


def unit_of(thread: int) -> Unit | None:
    """The unit named for thread `thread`'s waits (threading.get_ident), None for none. For
    another thread's reading: held (below) is said on its behalf."""
    with _lock:
        return _units.get(thread)


def waiting_in(thread: int) -> bool:
    """Whether thread `thread` (threading.get_ident) is inside a wait (wait) now: it then says its
    own long wait, and no other thread need say that it is held."""
    with _lock:
        return thread in _running


def check(hook: Callable[[], None] | None) -> None:
    """Call `hook` at each poll of the calling thread's waits from now on, None for none. It may
    raise: the wait then ends with that exception, as a stop's does (wait). It is called after the
    sleep, so a wait that is done at once never calls it; it runs on the calling thread, which
    is the one that waits, and no other thread's waits call it."""
    _named.check = hook


@contextmanager
def checking(hook: Callable[[], None]) -> Generator[None]:
    """The calling thread's waits call `hook` at each poll inside the with block, the check
    before given back after."""
    before: Callable[[], None] | None = getattr(_named, "check", None)
    check(hook)
    try:
        yield
    finally:
        _named.check = before


@contextmanager
def releasing(index: int = 0) -> Generator[None]:
    """The calling thread's waits inside the with block are a release of card `index` (CUDA's
    numbering): a wait there that an exception ended marks the card abandoned whatever its bound
    (the user's second Cancel decides, as a repeated signal does in the CLI), and the block
    ending without any such wait clears the mark: every wait of the release completed, so the
    card answers. A wait that raised and was caught inside the block counts: the mark stays. No
    unit or estimate is needed. Nested blocks are the innermost's. The mark and its clear are the
    process's for that card, not the thread's: the nodes' release runs on ComfyUI's one worker
    thread."""
    before: _Release | None = getattr(_named, "releasing", None)
    block = _Release(index)
    _named.releasing = block
    try:
        yield
    finally:
        _named.releasing = before
        if block.ended:
            with _lock:
                _pending.setdefault(index, []).extend(block.pending)
    # Only reached where the block ended without an exception. A failed card stays marked (fail).
    if not block.ended:
        with _lock:
            if index not in _failed:
                _abandoned.discard(index)
                _pending.pop(index, None)


def expect(*queries: Callable[[], bool]) -> None:
    """Inside a release (releasing), the work the rest of the release will wait for, as queries
    that say whether it has finished (an event's `query`, which never waits and holds no frame):
    kept for the card if the release ends in a wait an exception ended, so that settle can tell
    later whether all of it has finished. A wait ended before the release reached its later
    waits would otherwise leave those unknown: the compute stream's wait comes first and the
    copy stream's after. Nothing outside a release."""
    block: _Release | None = getattr(_named, "releasing", None)
    if block is not None:
        block.pending.extend(queries)


def pend(index: int, *queries: Callable[[], bool]) -> None:
    """Keep for card `index` (CUDA's numbering), already abandoned, queries (an event's `query`,
    which never waits) of work that an abandoned wait did not cover: a wait past its bound that
    an exception ended keeps only its own event, but the work queued behind it (the next blocks
    on the compute stream, the copies on the side stream) is what a later release would wait for
    too, so settle must see all of it finished. Nothing for a card marked failed (fail): it is
    never queried, and an event recorded there would report the error again."""
    with _lock:
        if index not in _failed:
            _pending.setdefault(index, []).extend(queries)


@contextmanager
def named(name: str, estimate: float | None, way: str = WAY) -> Generator[None]:
    """The calling thread's waits part of that unit inside the with block, the unit named before
    given back after."""
    before: Unit | None = getattr(_named, "unit", None)
    unit(name, estimate, way)
    try:
        yield
    finally:
        _set(before)


def wait(done: Callable[[], bool], poll: float, index: int = 0) -> None:
    """Return once done() is true, called every `poll` seconds, a signal's handler running
    between two calls. Past its unit's bound (bound), the wait is said once at WARNING, with NVML's
    reading of card `index` (CUDA's numbering) and the way out, every thread's stack after the
    first such line of the process, and it goes on; its end is then said too. done is an
    event's query: it raises where CUDA reports an error, which marks card `index` failed (fail)
    where it is one other than out of memory (cuda_failed). The thread's check (check) is called
    after each sleep, and what it raises ends the wait."""
    global _ended
    current: Unit | None = getattr(_named, "unit", None)
    hook: Callable[[], None] | None = getattr(_named, "check", None)
    release: _Release | None = getattr(_named, "releasing", None)
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
            if hook is not None:
                hook()
    except BaseException as error:
        name = None if current is None else current.name
        _ended = Ended(name, monotonic() - began, id(error), type(error))
        with _lock:
            if state.past:
                _abandoned.add(index)
                _pending.setdefault(index, []).append(done)
            if release is not None:
                release.ended = True
                _abandoned.add(release.index)
                release.pending.append(done)
        if cuda_failed(error):
            fail(index, error)
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


def abandoned(index: int = 0) -> bool:
    """Whether card `index` (CUDA's numbering) was abandoned: a wait on it that had passed its
    bound was ended by an exception (a Cancel, a stop, an error), or a wait inside its release
    (releasing) was, whatever its bound, and no release of it has completed since. The card has
    not answered, and may never. A caller that gives work back to the card (Models.release, a
    cache's emptying) leaves alone what would wait on it. Cleared by a release of that card whose
    waits all completed, by no other wait, and never by another card's."""
    with _lock:
        return index in _abandoned


def settle(index: int = 0) -> bool:
    """Whether card `index` (CUDA's numbering) can be given back its releases: it was abandoned,
    and all the work its abandoned waits were waiting for has finished, asked without waiting (the
    events' queries, kept by wait and expect), or it was not abandoned at all. Then the mark is
    cleared, so that the caller's release runs as a healthy one and its waits are immediate; a
    release that then fails marks the card again. False, the mark kept, while any of that work is
    pending, where nothing was recorded to ask about, and always for a card marked failed (fail):
    a CUDA error is never cleared by a query. A query that raises a CUDA error marks the card
    failed. DESIGN.md, Repository and apps, the nodes' Progress: a mark the card itself clears."""
    with _lock:
        if index in _failed:
            return False
        if index not in _abandoned:
            return True
        queries = list(_pending.get(index, ()))
    if not queries:
        return False
    try:
        finished = all([query() for query in queries])
    except Exception as error:
        if cuda_failed(error):
            fail(index, error)
        else:
            logger.debug("the abandoned work can't be asked about: %r", error)
        return False
    if not finished:
        return False
    with _lock:
        if index in _failed:
            return False
        _abandoned.discard(index)
        _pending.pop(index, None)
    return True


def fail(index: int, error: BaseException) -> None:
    """Mark card `index` (CUDA's numbering) failed by `error`, a CUDA error other than out of
    memory (cuda_failed; ValueError for another): abandoned too (abandoned), so that nothing waits
    on it or gives work back to it, and never cleared, a release included. The first error's text is
    kept (failed); the error itself is not, its frames being its holder's (the module's
    docstring)."""
    if not cuda_failed(error):
        raise ValueError(f"not a CUDA error other than out of memory: {error!r}")
    text = str(error).strip().splitlines()
    with _lock:
        _abandoned.add(index)
        _failed.setdefault(index, text[0] if text else type(error).__name__)


def failed(index: int = 0) -> str | None:
    """The first line of the CUDA error that marked card `index` failed (fail), None where none
    did. Nothing on that card may be freed or waited on again in the process: under
    cudaMallocAsync a free reports the error again and aborts the process (the module's
    docstring)."""
    with _lock:
        return _failed.get(index)


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


def _reading(index: int) -> str:
    # Imported here: memory imports nothing of torch at its import either, but this module is
    # model.py's and blockswap.py's, and memory's NVML reading is needed only now.
    from seedvr2x import memory

    try:
        found = memory.activity(index)
    except Exception as error:
        # Never raises by its contract; a replacement of it might (research/scripts' own).
        logger.debug("the card's activity can't be read: %r", error)
        found = None
    if found is None:
        return "NVML gives no reading of the card"
    state = "an unknown performance state" if found.state == 32 else f"P{found.state}"
    return (
        f"NVML reads the card {found.busy}% busy at {state}, {found.used / 2**30:.2f} GiB"
        " held by its programs"
    )


def held(current: Unit, seconds: float, where: str, also: str = "", index: int = 0) -> None:
    """Say once, at WARNING, that a thread has been held in one call for `seconds` (the run's own
    seconds) at `where` (the frame, as the watcher names it), outside any wait (wait says its own
    long wait): its unit as `current` names it, whose estimate is needed, NVML's reading of card
    `index`, the unit's way out and `also`, the app's own words for what a watcher can do, and
    every thread's stack after the process's first such line, a wait's included. The watcher's
    (stop.Watch) call: a call in C that CUDA holds (a sync on a stalled card) is no wait that
    polls, and no Python code runs in that thread to say so."""
    global _stacks_written
    estimate = current.estimate
    assert estimate is not None
    reading = _reading(index)
    stacks = not _stacks_written
    _stacks_written = True
    logger.warning(
        "%s: the main thread has been held in one call for %s, at %s, the whole unit estimated at"
        " %s; no error has been raised; %s; seedvr2x goes on waiting: %s%s%s",
        current.name,
        clock.spoken(clock.spent(seconds)),
        where,
        _seconds(clock.wall(estimate)),
        reading,
        current.way,
        f"; {also}" if also else "",
        "; every thread's stack follows" if stacks else "",
    )
    if stacks:
        _stacks()


def _say(current: Unit, seconds: float, index: int) -> None:
    global _stacks_written
    estimate = current.estimate
    assert estimate is not None
    reading = _reading(index)
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
