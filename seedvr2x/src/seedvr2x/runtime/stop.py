"""Stopping a run (DESIGN.md, Pause and resume): Ctrl-C once lets the unit in progress finish, and
the run stops before the next one; Ctrl-C again stops it at once, as SIGTERM does. Either way the
process exits, which frees the GPU, and the units kept let the same command resume.

ffmpeg's processes are started out of the terminal's reach (their own process group), so that a
Ctrl-C reaches seedvr2x alone: ffmpeg would otherwise end its output early, and the unit in
progress with it. Python only handles a signal between two bytecodes, so the run waits for the
GPU polling (model.synchronize), not blocked in a copy for a whole decode slice or window.

--until's time stops a run too, by the clock: before a unit that wouldn't finish in time, by its
estimate (runtime/timing.py), never inside one. The estimate is in the run's own clock, steady,
and is held against --until's time as the wall clock shows it (clock.py: One clock)."""

import contextlib
import os
import signal
import time
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from datetime import time as day_time
from types import FrameType, TracebackType
from typing import Any, Self

from seedvr2x.clock import wall

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
            raise KeyboardInterrupt
        self.asked = True
        # A handler may run in the middle of a write to stderr, which a logging call would then
        # enter again: written straight to the descriptor instead, after the time of day, as
        # every line seedvr2x prints starts (cli._Lines).
        when = time.strftime("%H:%M:%S")
        notice = f"\n{when} stopping after {self.unit}; Ctrl-C again stops at once\n"
        # stderr may be a pipe whose reader the same Ctrl-C ended (| tee): the stop goes on.
        with contextlib.suppress(OSError):
            os.write(2, notice.encode())

    def __enter__(self) -> Self:
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)


@contextlib.contextmanager
def terminable() -> Generator[None]:
    """SIGTERM raised as Terminated inside the with block, as Stop raises it during the run,
    Python's handler given back on exit; Ctrl-C left to Python, as KeyboardInterrupt. For what runs
    before the run and must unwind on either, its finally clauses with it: the model files'
    download (runtime/pull.py), whose partial file the library removes then. Stop doesn't fit
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
    not with the estimates (runtime/timing.py), which import torch: the command sets it as it
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
    raise Terminated
