"""Stopping a run (DESIGN.md, Pause and resume): Ctrl-C once lets the unit in progress finish, and
the run stops before the next one; Ctrl-C again stops it at once, as SIGTERM does. Either way the
process exits, which frees the GPU, and the units kept let the same command resume.

ffmpeg's processes are started out of the terminal's reach (their own process group), so that a
Ctrl-C reaches seedvr2x alone: ffmpeg would otherwise end its output early, and the unit in
progress with it. Python only handles a signal between two bytecodes, so the run waits for the
GPU polling (model.synchronize), not blocked in a copy for a whole decode slice or window."""

import contextlib
import os
import signal
from collections.abc import Callable
from types import FrameType, TracebackType
from typing import Any, Self

# What signal.signal takes and gives back.
Handler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


class Stopped(Exception):
    """The run stopped between two units, as asked; every unit done is kept."""


class Terminated(BaseException):
    """The run stopped at once on SIGTERM: unwound as KeyboardInterrupt is, by what catches
    BaseException only."""


class Stop:
    """The stop asked of a run, while it runs: a context manager, which installs the handlers of
    SIGINT and SIGTERM and gives Python's back on exit. The run calls check() before each unit,
    and names it in `unit` for the notice a Ctrl-C prints."""

    def __init__(self) -> None:
        self.asked = False
        self.unit = "the unit in progress"
        self._previous: dict[int, Handler] = {}

    def check(self) -> None:
        """Raise Stopped if a stop was asked: the unit before is done and kept."""
        if self.asked:
            raise Stopped

    def _handle(self, number: int, frame: FrameType | None) -> None:
        if number == signal.SIGTERM:
            raise Terminated
        if self.asked:
            raise KeyboardInterrupt
        self.asked = True
        # A handler may run in the middle of a write to stderr, which a logging call would then
        # enter again: written straight to the descriptor instead.
        notice = f"\nseedvr2x: stopping after {self.unit}; Ctrl-C again stops at once\n"
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
