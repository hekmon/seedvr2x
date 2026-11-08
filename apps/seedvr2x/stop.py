"""Stopping a run (DESIGN.md, Pause and resume): Ctrl-C once lets the unit in progress finish, and
the run stops before the next one; Ctrl-C again stops it at once, as SIGTERM does. Either way the
process exits, which frees the GPU, and the units kept let the same command resume.

ffmpeg's processes are started out of the terminal's reach (their own process group), so that a
Ctrl-C reaches seedvr2x alone: ffmpeg would otherwise end its output early, and the unit in
progress with it. Python only handles a signal between two bytecodes, so the run waits for the
GPU polling (model.synchronize), not blocked in a copy for a whole decode slice or window.

--until's time stops a run too, by the clock: before a unit that wouldn't finish in time, by its
estimate (timing.py), never inside one. The estimate is in the run's own clock, steady,
and is held against --until's time as the wall clock shows it (clock.py: One clock)."""

import contextlib
import logging
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from datetime import time as day_time
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, ClassVar, Self

from seedvr2x import clock, waiting
from seedvr2x.clock import wall

logger = logging.getLogger("cli.stop")

# What signal.signal takes and gives back.
Handler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


# What a unit's estimate is multiplied by before it is held against --until's time (DESIGN.md,
# Pause and resume, Stopping: "the run stops when now plus 1.15 times the unit's estimate passes
# it").
MARGIN = 1.15


class Stopped(Exception):
    """The run stopped between two units, as asked; every unit done is kept."""


class Deadline(Stopped):
    """The run stopped before a unit that wouldn't finish by --until's time (Stop.check); every
    unit done is kept. `unit` names the one not begun, and `cost` is its estimate, in the wall
    clock's seconds (Stop.check), None without one."""

    def __init__(self, unit: str, cost: float | None) -> None:
        super().__init__(unit, cost)
        self.unit, self.cost = unit, cost


class Terminated(BaseException):
    """The run stopped at once on SIGTERM: unwound as KeyboardInterrupt is, by what catches
    BaseException only."""


class Stop:
    """The stop asked of a run, while it runs: a context manager, which installs the handlers of
    SIGINT and SIGTERM and gives Python's back on exit. The run calls check() before each unit,
    and names it in `unit` for the notice a Ctrl-C prints. `deadline` is --until's time
    (stop.deadline), aware; None without one."""

    # The Stop in force, for Watch's reading of whether a Ctrl-C is a first one.
    current: ClassVar["Stop | None"] = None

    def __init__(self, deadline: datetime | None = None) -> None:
        self.asked = False
        self.unit = "the unit in progress"
        self.deadline = deadline
        self._previous: dict[int, Handler] = {}

    def check(self, cost: float | None = None, unit: str = "the next unit") -> None:
        """Raise Stopped if a stop was asked: the unit before is done and kept. With a deadline,
        raise Deadline if the unit to come, named `unit` and estimated at `cost` seconds of the
        run's clock, would end after it: now plus MARGIN times its estimate, said in the wall
        clock's seconds (clock.wall), which `until` is read by. A unit nothing estimates (None) is
        held to the time alone, so stopped once the deadline is past."""
        if self.asked:
            raise Stopped
        if self.deadline is not None:
            said = None if cost is None else wall(cost)
            if late(now(), said, self.deadline):
                raise Deadline(unit, said)

    def _handle(self, number: int, frame: FrameType | None) -> None:
        if number == signal.SIGTERM:
            _terminated(number, frame)
        if self.asked:
            answered()
            raise KeyboardInterrupt
        self.asked = True
        answered()
        # A handler may run in the middle of a write to stderr, which a logging call would then
        # enter again: written straight to the descriptor instead, after the time of day, as
        # every line seedvr2x prints starts (main._Lines).
        when = time.strftime("%H:%M:%S")
        notice = f"\n{when} stopping after {self.unit}; Ctrl-C again stops at once\n"
        # stderr may be a pipe whose reader the same Ctrl-C ended (| tee): the stop goes on.
        with contextlib.suppress(OSError):
            os.write(2, notice.encode())

    def __enter__(self) -> Self:
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)
        Stop.current = self
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        Stop.current = None
        for number, handler in self._previous.items():
            signal.signal(number, handler)


@contextlib.contextmanager
def terminable() -> Generator[None]:
    """SIGTERM raised as Terminated inside the with block, as Stop raises it during the run,
    Python's handler given back on exit; Ctrl-C left to Python, as KeyboardInterrupt. For what runs
    before the run and must unwind on either, its finally clauses with it: the model files'
    download (seedvr2x/pull.py), whose partial file the library removes then. Stop doesn't fit
    there: its first Ctrl-C lets the unit in progress finish, which no download checks for."""
    previous = signal.signal(signal.SIGTERM, _terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def late(at: datetime, cost: float | None, deadline: datetime) -> bool:
    """Whether a unit begun at `at`, estimated at `cost` seconds (None: nothing estimates it),
    would end after deadline by --until's rule: at plus MARGIN times its estimate."""
    return at + timedelta(seconds=MARGIN * (cost or 0.0)) > deadline


def deadline(shown: day_time, at: datetime) -> datetime:
    """--until's time (DESIGN.md, Pause and resume, Stopping): the next time after `at`, aware,
    that the local clock shows `shown`, a time of day, as a full date in the local zone. Here and
    not with the estimates (timing.py), which import torch: the command sets it as it
    starts, before a first pass that may last past it.

    By the zone's own rules (the system's, through astimezone): an hour the clock skips in spring
    isn't shown that day, so the time is the day after's; one it shows twice in autumn counts
    twice, the first that is still to come taken."""
    today = at.astimezone().date()
    instants: list[datetime] = []
    # Two days ahead: the day after's when today's is past, and one more for a skipped hour.
    for days in range(3):
        for fold in (0, 1):
            naive = datetime.combine(today + timedelta(days=days), shown).replace(fold=fold)
            instant = naive.astimezone()
            # A skipped time comes back from astimezone moved by the hour skipped: not shown.
            if instant.time() == shown and instant > at:
                instants.append(instant)
    return min(instants)


def now() -> datetime:
    """The time it is, aware, in the local zone: what a job's end is said in (timing.Progress) and
    --until's time held against. The wall clock's, as --until's time is one of its own."""
    return datetime.now().astimezone()


def _terminated(number: int, frame: FrameType | None) -> None:
    answered()
    raise Terminated


# A stop at once that needs no main thread (DESIGN.md, Allocator, "Out of the stop's reach").
# Python runs a signal's handler on the main thread only, between two bytecodes, so a main thread
# held in one call in C (a sync on a stalled card, in the vendored DiT or a driver call) answers
# nothing: place-study's run R3 (impl-stop, 2026-11-06) left SIGTERM unanswered until SIGKILL. The
# C-level handler runs anyway and writes the signal's number to the wakeup descriptor
# (signal.set_wakeup_fd), which Watch's thread reads. POSIX only: Windows' set_wakeup_fd takes a
# socket, and its twin is a later baton's, so this one guard is all there is to it.
WATCHED = os.name == "posix"

# What Watch reads from the pipe besides the signals' numbers (all under 65): a handler on the
# main thread writes ACK when it runs, which puts the answer in the pipe's own order with the
# signals it answers, so that the watcher needs no clock to tell an unanswered signal from one
# the main thread answered before the watcher read its byte; CLOSE ends the thread.
ACK = 0xFF
CLOSE = 0

# How often the main thread's frame is sampled, which is the select's timeout on the pipe:
# a sampling period, never a grace period before a stop (the user's repeated signal decides).
SAMPLE = 1.0

# How long Watch lets its own line and cleanup take before going on without them: a logging
# handler's lock held by the main thread, in a write to a pipe nobody reads, would hold the
# stop for ever otherwise. About what a normal cleanup takes (_end: a few ms), a hundred times.
CLEANUP = 5.0

# The descriptor the main thread's handlers write ACK to, None where no Watch runs.
_ack: int | None = None
# Who ends the process, "main" (main._end) or "watch" (Watch), whichever asks first: exactly once.
_decision = threading.Lock()
_ender: str | None = None


def answered() -> None:
    """Tell the watcher that a handler of SIGINT or SIGTERM has run on the main thread: every
    signal read before this byte is answered. Called first by each such handler (Stop._handle,
    _terminated, interrupted). Nothing where no Watch runs."""
    descriptor = _ack
    if descriptor is not None:
        with contextlib.suppress(OSError):
            os.write(descriptor, bytes([ACK]))


def interrupted(number: int, frame: FrameType | None) -> None:
    """Python's own handler of Ctrl-C, KeyboardInterrupt, answering to the watcher first: what
    Watch installs where the process runs with it (before Stop's)."""
    answered()
    signal.default_int_handler(number, frame)


def ending() -> None:
    """The main thread is about to end the process (main._end): it does, or, where the watcher
    already is, waits here for the watcher's os._exit. Called once the main thread's own work
    of ending is to start, so that the work is done exactly once, whichever thread gets there
    first."""
    global _ender
    with _decision:
        if _ender in (None, "main"):
            _ender = "main"
            return
    threading.Event().wait()


def _bounded(work: Callable[[], None], seconds: float) -> bool:
    """Run `work` on a daemon thread of its own, waiting `seconds` for it; whether it ended."""
    done = threading.Event()

    def run() -> None:
        try:
            work()
        except Exception as error:
            logger.debug("the stop watcher's step failed: %r", error)
        finally:
            done.set()

    threading.Thread(target=run, name="stop-watch-step", daemon=True).start()
    return done.wait(seconds)


def _signature(frame: FrameType | None) -> tuple[tuple[object, int], ...]:
    """A thread's stack as a call holds it: each frame's code and instruction, innermost first.
    A thread held in one call has the same signature at every sample; one that polls or loops
    changes it."""
    found: list[tuple[object, int]] = []
    while frame is not None:
        found.append((frame.f_code, frame.f_lasti))
        frame = frame.f_back
    return tuple(found)


def _where(frame: FrameType | None) -> str:
    if frame is None:
        return "no Python frame"
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno} in {frame.f_code.co_name}"


class Watch:
    """A thread that needs no main thread: it reads the signals as they come (the wakeup
    descriptor), counts those the main thread has not answered, and samples the main thread's
    frame. A context manager, around the command's whole run (main.main with owns_process); with
    `WATCHED` false, off the main thread, or where another owner has the wakeup descriptor, it
    does nothing and says why at debug.

    A stop at once is SIGTERM, or Ctrl-C when a first one is unanswered, a Stop has been asked,
    or no Stop is in force (Python's KeyboardInterrupt). One asked while another stop at once is
    still unanswered ends the process from the watcher: `finish` (the work of ending that needs
    no main thread: main._flush) is run, one ERROR line says the signal, the main thread's
    frame and how long it had been held, and `leave(status)` follows (os._exit; 143 for SIGTERM,
    130 for Ctrl-C), exactly once with main._end (ending). No clock decides it.

    A main thread held in the same frame for longer than its unit's bound (waiting.bound), and
    not inside a wait (which says its own), gets the stall's line once per hold (waiting.held),
    `also` being the CLI's words for what the repeated signal does."""

    def __init__(
        self,
        finish: Callable[[], None],
        also: str = "",
        *,
        leave: Callable[[int], object] = os._exit,
        every: float = SAMPLE,
        thread: int | None = None,
    ) -> None:
        self.finish, self.also, self.leave, self.every = finish, also, leave, every
        # The thread watched (threading.get_ident), the main thread's by default.
        self.thread = (threading.main_thread().ident or 0) if thread is None else thread
        self.active = False
        self.write = -1  # the pipe's write end while active, the signals' and the main thread's
        self._read = -1
        self._closed = False
        self._thread: threading.Thread | None = None
        self._wrapped = False
        # The unanswered stops: whether any is a stop at once, and the signals in their order.
        self._hard = False
        self._pending: list[int] = []
        self._signature: tuple[tuple[object, int], ...] | None = None
        self._since = 0.0
        self._said = False
        self._seen = ""

    def __enter__(self) -> Self:
        global _ack, _ender
        if not WATCHED:
            logger.debug("no stop watcher: this platform has no wakeup descriptor for it")
            return self
        if threading.current_thread() is not threading.main_thread():
            logger.debug("no stop watcher: not on the main thread, where the descriptor is set")
            return self
        read, write = os.pipe()
        os.set_blocking(read, False)
        os.set_blocking(write, False)
        previous = signal.set_wakeup_fd(write, warn_on_full_buffer=False)
        if previous != -1:
            signal.set_wakeup_fd(previous)
            os.close(read)
            os.close(write)
            logger.debug("no stop watcher: the wakeup descriptor is another's (%d)", previous)
            return self
        self._read, self.write = read, write
        with _decision:
            _ender = None
        self._closed = False
        if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            signal.signal(signal.SIGINT, interrupted)
            self._wrapped = True
        _ack = write
        self._since = waiting.monotonic()
        self.active = True
        self._thread = threading.Thread(target=self._watch, name="stop-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        global _ack
        if not self.active:
            return
        with _decision:
            leaving = _ender == "watch"
            self._closed = True
        if leaving:
            # The watcher is ending the process: nothing of the main thread's goes on beside it.
            threading.Event().wait()
        _ack = None
        signal.set_wakeup_fd(-1)
        if self._wrapped and signal.getsignal(signal.SIGINT) is interrupted:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        os.write(self.write, bytes([CLOSE]))
        assert self._thread is not None
        self._thread.join(CLEANUP)
        os.close(self._read)
        os.close(self.write)
        self.active = False

    def _watch(self) -> None:
        last = waiting.monotonic()
        try:
            while True:
                found = waiting.monotonic()
                timeout = max(0.0, self.every - (found - last))
                ready = select.select([self._read], [], [], timeout)[0]
                if ready:
                    try:
                        data = os.read(self._read, 4096)
                    except BlockingIOError:
                        data = b""
                    for byte in data:
                        if byte == CLOSE:
                            return
                        if byte == ACK:
                            self._hard, self._pending = False, []
                        elif byte in (signal.SIGINT, signal.SIGTERM):
                            self._arrived(byte)
                found = waiting.monotonic()
                if found - last >= self.every:
                    last = found
                    self._sample(found)
        except Exception as error:
            # The stop is then the main thread's alone, as before: said, for a user who wonders.
            logger.warning(
                "the stop watcher failed (%r): a stop at once can't reach a main thread held in"
                " one call",
                error,
            )

    def _arrived(self, number: int) -> None:
        if self._hard:
            self._leave(number)
            return
        current = Stop.current
        self._hard = (
            number == signal.SIGTERM or current is None or current.asked or bool(self._pending)
        )
        self._pending.append(number)

    def _sample(self, now: float) -> None:
        frame = sys._current_frames().get(self.thread)  # pyright: ignore[reportPrivateUsage]
        signature = _signature(frame)
        if signature != self._signature:
            self._signature, self._since, self._said = signature, now, False
            self._seen = _where(frame)
            return
        if self._said:
            return
        current = waiting.unit_of(self.thread)
        if current is None or waiting.waiting_in(self.thread):
            return
        limit = waiting.bound(current.estimate)
        seconds = now - self._since
        if limit is None or seconds <= limit:
            return
        self._said = True
        _bounded(
            lambda: waiting.held(current, seconds, self._seen, self.also),
            CLEANUP,
        )

    def _leave(self, number: int) -> None:
        global _ender
        with _decision:
            if self._closed or _ender is not None:
                return
            _ender = "watch"
        name = "SIGTERM" if number == signal.SIGTERM else "Ctrl-C"
        status = 143 if number == signal.SIGTERM else 130
        frame = sys._current_frames().get(self.thread)  # pyright: ignore[reportPrivateUsage]
        where = _where(frame)
        held = ""
        if _signature(frame) == self._signature:
            seconds = waiting.monotonic() - self._since
            if seconds >= 1:
                held = f" for {clock.spoken(clock.spent(seconds))}"
        current = waiting.unit_of(self.thread)
        unit = f" ({current.name})" if current is not None else ""
        text = (
            f"stopped at once ({name}), asked again before the main thread answered the first:"
            f" it had been held in one call{held}, at {where}; the process ends from another"
            f" thread without unwinding it, every unit done kept and the unit in progress{unit}"
            " made again by the same command"
        )
        if not _bounded(lambda: logger.error(text), CLEANUP):
            with contextlib.suppress(OSError):
                os.write(2, f"{time.strftime('%H:%M:%S')} ERROR {text}\n".encode())
        _bounded(self.finish, CLEANUP)
        self.leave(status)
        # Not reached by os._exit: a replaced leave (the tests') returns, and ends nothing.
        with _decision:
            _ender = None
