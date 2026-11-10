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
import socket
import sys
import threading
import time
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from datetime import time as day_time
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, ClassVar, NoReturn, Self

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


# The signals of a stop at once besides a second Ctrl-C: SIGTERM, and on Windows Ctrl-Break
# (SIGBREAK, which Python's C runtime raises for CTRL_BREAK_EVENT), SIGTERM's twin there: a
# console's processes get no SIGTERM, and Ctrl-Break is the keyboard's other stop.
if sys.platform == "win32":
    AT_ONCE: tuple[int, ...] = (signal.SIGTERM, signal.SIGBREAK)
else:
    AT_ONCE = (signal.SIGTERM,)
# Ctrl-Break's own signal, which Closing raises for a console's close: SIGBREAK on Windows, and
# SIGTERM, its twin, elsewhere, where the tests hold Closing.
BREAK = AT_ONCE[-1]


def named(number: int) -> str:
    """A stop's signal as the lines say it: Ctrl-C, SIGTERM, Ctrl-Break, or what closed the
    console (Closing) where a stop at once came from there."""
    if number == signal.SIGINT:
        return "Ctrl-C"
    if _cause is not None:
        return _cause
    return "SIGTERM" if number == signal.SIGTERM else "Ctrl-Break"


class Terminated(BaseException):
    """The run stopped at once on SIGTERM, or on Windows on Ctrl-Break or its console's close:
    unwound as KeyboardInterrupt is, by what catches BaseException only. `how` names it, as
    the stop's line says it."""

    def __init__(self, how: str = "SIGTERM") -> None:
        super().__init__(how)
        self.how = how


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
        if number in AT_ONCE:
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
        for number in (signal.SIGINT, *AT_ONCE):
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
    """SIGTERM (and Windows' Ctrl-Break, AT_ONCE) raised as Terminated inside the with block, as
    Stop raises it during the run, Python's handlers given back on exit; Ctrl-C left to Python,
    as KeyboardInterrupt. For what runs
    before the run and must unwind on either, its finally clauses with it: the model files'
    download (seedvr2x/pull.py), whose partial file the library removes then. Stop doesn't fit
    there: its first Ctrl-C lets the unit in progress finish, which no download checks for."""
    previous = {number: signal.signal(number, _terminated) for number in AT_ONCE}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


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
    raise Terminated(named(number))


# A stop at once that needs no main thread (DESIGN.md, Allocator, "Out of the stop's reach").
# Python runs a signal's handler on the main thread only, between two bytecodes, so a main thread
# held in one call in C (a sync on a stalled card, in the vendored DiT or a driver call) answers
# nothing: place-study's run R3 (impl-stop, 2026-11-06) left SIGTERM unanswered until SIGKILL. The
# C-level handler runs anyway and writes the signal's number to the wakeup descriptor
# (signal.set_wakeup_fd), which Watch's thread reads: a pipe on POSIX, a socket pair on Windows,
# where the C runtime's handler writes the byte from the console's own thread while the main
# thread sits in C (impl-win-2's measure, 2026-11-09: Ctrl-C's and Ctrl-Break's bytes read within
# 0.06 s of the event, the main thread in kernel32's Sleep). The one guard, which the tests turn
# off.
WATCHED = os.name in ("posix", "nt")

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

# What the main thread's handlers write ACK with (Watch.put), None where no Watch runs.
_ack: Callable[[bytes], object] | None = None
# Who ends the process, "main" (main._end) or "watch" (Watch), whichever asks first: exactly once.
_decision = threading.Lock()
_ender: str | None = None


def answered() -> None:
    """Tell the watcher that a handler of SIGINT or SIGTERM has run on the main thread: every
    signal read before this byte is answered. Called first by each such handler (Stop._handle,
    _terminated, interrupted). Nothing where no Watch runs."""
    put = _ack
    if put is not None:
        with contextlib.suppress(OSError):
            put(bytes([ACK]))


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


def end_process(status: int) -> NoReturn:
    """os._exit(status), POSIX's _exit: the process ended with no teardown of Python's or of its
    libraries'. On Windows TerminateProcess on itself: os._exit there is ExitProcess, which runs
    every library's detach, CUDA's among them, and once a console's close was handled with
    torch's libraries loaded it ended the process with 0 whatever its status (natively,
    2026-11-09: os._exit(143) gave 0, TerminateProcess 143; without torch both 143)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess(ctypes.c_void_p(kernel32.GetCurrentProcess()), status)
    os._exit(status)


def closed() -> bool:
    """A console's close, its user's logoff or the system's shutdown was handled (Closing):
    the process is ending, and ends by end_process, never by Python's own exit, which ended it
    with 0 there (main.main)."""
    return _cause is not None


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
    frame and how long it had been held, and `leave(status)` follows (end_process; 143 for SIGTERM,
    130 for Ctrl-C), exactly once with main._end (ending). No clock decides it.

    A main thread held in the same frame for longer than its unit's bound (waiting.bound), and
    not inside a wait (which says its own), gets the stall's line once per hold (waiting.held),
    `also` being the CLI's words for what the repeated signal does."""

    def __init__(
        self,
        finish: Callable[[], None],
        also: str = "",
        *,
        leave: Callable[[int], object] = end_process,
        every: float = SAMPLE,
        thread: int | None = None,
    ) -> None:
        self.finish, self.also, self.leave, self.every = finish, also, leave, every
        # The thread watched (threading.get_ident), the main thread's by default.
        self.thread = (threading.main_thread().ident or 0) if thread is None else thread
        self.active = False
        self.write = -1  # the wakeup's write end while active, the signals' and the main thread's
        self._read = -1
        self._closed = False
        self._thread: threading.Thread | None = None
        self._wrapped = False
        # Windows' wakeup ends, read and write (a socket pair); none on POSIX, a pipe's there.
        self._sockets: tuple[socket.socket, ...] = ()
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
        if sys.platform == "win32":
            # Windows' set_wakeup_fd takes a socket's descriptor (signal's documentation), which
            # select reads as well.
            try:
                self._sockets = socket.socketpair()
            except OSError as error:
                # An emulated loopback connection (socket's documentation), which a firewall may
                # refuse: the stop is then the main thread's alone, as before the watcher.
                logger.debug("no stop watcher: its socket pair can't be made (%r)", error)
                return self
            for end in self._sockets:
                end.setblocking(False)
            read, write = self._sockets[0].fileno(), self._sockets[1].fileno()
        else:
            read, write = os.pipe()
            os.set_blocking(read, False)
            os.set_blocking(write, False)
        previous = signal.set_wakeup_fd(write, warn_on_full_buffer=False)
        if previous != -1:
            signal.set_wakeup_fd(previous)
            self._close(read, write)
            logger.debug("no stop watcher: the wakeup descriptor is another's (%d)", previous)
            return self
        self._read, self.write = read, write
        with _decision:
            _ender = None
        self._closed = False
        if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            signal.signal(signal.SIGINT, interrupted)
            self._wrapped = True
        _ack = self.put
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
        self.put(bytes([CLOSE]))
        assert self._thread is not None
        self._thread.join(CLEANUP)
        self._close(self._read, self.write)
        self.active = False

    def put(self, data: bytes) -> None:
        """Write `data` to the wakeup descriptor, as the C handler writes a signal's byte: ACK,
        CLOSE, or a signal's number (the tests')."""
        if self._sockets:
            self._sockets[1].send(data)
        else:
            os.write(self.write, data)

    def _take(self) -> bytes:
        """What is waiting on the read end, nothing if select's readiness was spurious."""
        try:
            if self._sockets:
                return self._sockets[0].recv(4096)
            return os.read(self._read, 4096)
        except BlockingIOError:
            return b""

    def _close(self, read: int, write: int) -> None:
        if self._sockets:
            for end in self._sockets:
                end.close()
            self._sockets = ()
        else:
            os.close(read)
            os.close(write)

    def _watch(self) -> None:
        last = waiting.monotonic()
        try:
            while True:
                found = waiting.monotonic()
                timeout = max(0.0, self.every - (found - last))
                ready = select.select([self._read], [], [], timeout)[0]
                if ready:
                    for byte in self._take():
                        if byte == CLOSE:
                            return
                        if byte == ACK:
                            self._hard, self._pending = False, []
                        elif byte == signal.SIGINT or byte in AT_ONCE:
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
        self._hard = number in AT_ONCE or current is None or current.asked or bool(self._pending)
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
        name = named(number)
        status = 143 if number in AT_ONCE else 130
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


# The console's events that end its processes (wincon.h): its window closed, the user logging
# off, the system shutting down. Python's C runtime raises SIGBREAK for each (Closing), and its
# handler returns at once, after which Windows ends the process: 3 ms after the window's close,
# before Python's handler had run (natively, 2026-11-09). Windows ends it anyway once its time is
# up (CLOSE_WAIT). The last two reach no process
# that has loaded user32 or gdi32 (SetConsoleCtrlHandler's documentation), as torch's does: they
# are handled for a run that hasn't, the window's close being the one a run meets.
CLOSINGS = {2: "the console closed", 5: "the user logging off", 6: "the system shutting down"}

# How long the main thread is given to unwind a console's close as its stop at once, before the
# process is ended from the handler's thread, as Watch ends a stop asked again; then how long
# that ending's own line and cleanup are given: 4 s in all at the most, within the 5 s Windows
# leaves a closed console's process (HandlerRoutine's documentation; natively on 2026-11-09, a
# process whose handler held was ended 5.00 to 5.01 s after the window's close, and the main
# thread ran the SIGBREAK the handler raised 0.04 s after it).
CLOSE_WAIT = 2.0
CLOSE_FINISH = 1.5

# What closed the console, once it has (named): the stop's line says it.
_cause: str | None = None
# The console handlers made, kept for the process's life: one Windows may still be running.
_handlers: list[object] = []


class Closing:
    """Windows' console events, a context manager around the command's whole run (main.main
    with owns_process): a console's close, its user's logoff or the system's shutdown is a stop
    at once, as Ctrl-Break, SIGTERM's twin there, and Ctrl-Break itself is one (Terminated)
    wherever no stop of the run's own is in force, as SIGTERM's default ends a POSIX process.

    Windows ends the process once the handler of such an event returns, or past its time: the
    handler raises SIGBREAK for the main thread, which unwinds as for Ctrl-Break, and waits for
    it CLOSE_WAIT seconds at most; past them, with nothing ending the process yet, it ends it as
    Watch does a stop asked again: one ERROR line, `finish` (main._flush) within CLOSE_FINISH
    seconds, then `leave(143)`, exactly once with main._end (ending). Nothing elsewhere: POSIX
    has SIGHUP for a terminal's close, which seedvr2x leaves to its default."""

    def __init__(
        self,
        finish: Callable[[], None],
        *,
        leave: Callable[[int], object] = end_process,
        wait: float = CLOSE_WAIT,
        hold: Callable[[], object] = lambda: threading.Event().wait(),
    ) -> None:
        # hold: what the handler does while the process ends otherwise, for ever (the tests'
        # returns).
        self.finish, self.leave, self.wait, self.hold = finish, leave, wait, hold
        self.active = False
        self._done = False
        self._closing = False
        self._break: Handler = None

    def __enter__(self) -> Self:
        if sys.platform != "win32":
            return self
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        routine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
        kernel32.SetConsoleCtrlHandler.argtypes = (routine, ctypes.c_int)
        handler = routine(self.handle)
        if not kernel32.SetConsoleCtrlHandler(handler, 1):
            logger.debug(
                "no handler of the console's close: SetConsoleCtrlHandler failed (WinError %d)",
                ctypes.get_last_error(),
            )
            return self
        _handlers.append(handler)
        self._unregister = lambda: kernel32.SetConsoleCtrlHandler(handler, 0)
        self._break = signal.getsignal(BREAK)
        if self._break is signal.SIG_DFL:
            signal.signal(BREAK, _terminated)
        self.active = True
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # The main thread is past the run: a close from here on waits for its own end.
        self._done = True
        if not self.active or sys.platform != "win32":
            return
        # Not while a close is handled: SetConsoleCtrlHandler's removal waits for the handler,
        # which holds, so the main thread waited there until Windows ended the process 5 s after
        # the close, with its own status, 0xC000013A, not 143 (natively, 2026-11-09). The process
        # is ending: the handler stays.
        if not self._closing:
            self._unregister()
        if signal.getsignal(BREAK) is _terminated:
            signal.signal(BREAK, self._break)
        self.active = False

    def handle(self, event: int) -> int:
        """The console's handler: Ctrl-C and Ctrl-Break passed on to the C runtime's, which
        raises them to Python's handlers (0); one of CLOSINGS answered here, never returning
        while the process may still end well (1, once ended from here by a replaced leave)."""
        global _cause, _ender
        if event not in CLOSINGS:
            return 0
        self._closing = True
        _cause = CLOSINGS[event]
        answering = callable(signal.getsignal(BREAK))
        if answering and not self._done:
            signal.raise_signal(BREAK)
            deadline = time.monotonic() + self.wait
            while time.monotonic() < deadline and _ender is None and not self._done:
                time.sleep(0.02)
        with _decision:
            ending_already = _ender is not None or self._done or not answering
            if not ending_already:
                _ender = "watch"
        if ending_already:
            # The main thread ends the process (main._end, or its exit), or has its stops
            # ignored as it does: Windows ends it past its time otherwise.
            self.hold()
            return 1
        text = (
            f"stopped at once ({_cause}): the main thread had not ended the run {self.wait:g} s"
            " later, and Windows ends the process soon after; it ends from another thread"
            " without unwinding it, every unit done kept and the unit in progress made again by"
            " the same command"
        )
        if not _bounded(lambda: logger.error(text), CLOSE_FINISH / 3):
            with contextlib.suppress(OSError):
                os.write(2, f"{time.strftime('%H:%M:%S')} ERROR {text}\n".encode())
        _bounded(self.finish, CLOSE_FINISH)
        self.leave(143)
        with _decision:
            _ender = None
        return 1
