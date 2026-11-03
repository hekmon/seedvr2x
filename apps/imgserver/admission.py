"""Which pictures run on the GPU, and which wait (DESIGN.md, The image server, Running): a picture
starts when its predicted share of the reservation fits in what is left of it, at most
--max-concurrent at once; else it waits, up to --queue, in the order it came. Plain Python, no
torch: the tests hold it on its own."""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


class Full(Exception):
    """The queue holds --queue pictures already: answered 503, with Retry-After."""


class Stopping(Exception):
    """The server is stopping: a waiting picture is answered 503, and no new one waits."""


class Gone(Exception):
    """The client closed its connection while its picture waited: the picture is dropped."""


@dataclass(eq=False)
class Ticket:
    """A picture admitted or waiting: its share of the pool, in bytes, and when it came, in
    time.monotonic()'s seconds."""

    need: int
    came: float = field(default_factory=time.monotonic)
    admitted: float | None = None

    @property
    def waited(self) -> float:
        """The seconds it waited before it was admitted (0.0 until then)."""
        return 0.0 if self.admitted is None else self.admitted - self.came


class Admission:
    """The pictures running and waiting. `pool` is what the reservation leaves the pictures
    above the models, in bytes; `most` the pictures at once (--max-concurrent); `queue` the
    pictures that may wait (--queue). First come, first served: a picture never starts before
    one that came earlier, so that a large one isn't passed over for ever by smaller ones."""

    def __init__(self, pool: int, most: int, queue: int, poll: float = 0.5) -> None:
        if most < 1 or queue < 0 or pool < 0:
            raise ValueError(f"pool {pool}, most {most}, queue {queue}")
        self.pool, self.most, self.queue, self.poll = pool, most, queue, poll
        self._lock = threading.Condition()
        self._running: list[Ticket] = []
        self._waiting: deque[Ticket] = deque()
        self._stopping = False

    @property
    def running(self) -> int:
        with self._lock:
            return len(self._running)

    @property
    def waiting(self) -> int:
        with self._lock:
            return len(self._waiting)

    def _fits(self, ticket: Ticket) -> bool:
        used = sum(each.need for each in self._running)
        return len(self._running) < self.most and used + ticket.need <= self.pool

    def enter(self, need: int, gone: Callable[[], bool] = lambda: False) -> Ticket:
        """Admit a picture whose share of the pool is `need` bytes, waiting for its turn: the
        ticket to give back to leave. Full when --queue pictures wait already and this one would
        wait too; Stopping when the server stops before it starts; Gone when `gone()`, asked every
        `poll` seconds while it waits, says its client has left. ValueError for a need over the
        whole pool, which no wait would admit (the server's bound keeps every picture within it)."""
        if need > self.pool:
            raise ValueError(f"a picture's share of {need} bytes, over the pool's {self.pool}")
        ticket = Ticket(need)
        with self._lock:
            if self._stopping:
                raise Stopping
            if not self._waiting and self._fits(ticket):
                return self._admit(ticket)
            if len(self._waiting) >= self.queue:
                raise Full
            self._waiting.append(ticket)
            try:
                while True:
                    if self._stopping:
                        raise Stopping
                    if self._waiting[0] is ticket and self._fits(ticket):
                        self._waiting.popleft()
                        return self._admit(ticket)
                    self._lock.wait(self.poll)
                    if gone():
                        raise Gone
            except BaseException:
                if ticket in self._waiting:
                    self._waiting.remove(ticket)
                    self._lock.notify_all()
                raise

    def _admit(self, ticket: Ticket) -> Ticket:
        ticket.admitted = time.monotonic()
        self._running.append(ticket)
        # The next in line may fit beside it.
        self._lock.notify_all()
        return ticket

    def leave(self, ticket: Ticket) -> None:
        """A picture done, its share given back to the pool."""
        with self._lock:
            self._running.remove(ticket)
            self._lock.notify_all()

    def when_idle(self, work: Callable[[], None]) -> bool:
        """`work` run under the admission's lock where no picture runs, none admitted meanwhile:
        True where it ran. For what must not overlap a picture on the GPU (the server's hold of
        its reservation made again)."""
        with self._lock:
            if self._running:
                return False
            work()
            return True

    def stop(self) -> None:
        """No picture waits any longer: those waiting are answered Stopping, and so is any new
        one. The pictures running finish."""
        with self._lock:
            self._stopping = True
            self._lock.notify_all()

    def idle(self, timeout: float | None = None) -> bool:
        """Wait until no picture runs or waits: True when that came within `timeout` seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while self._running or self._waiting:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._lock.wait(left if left is None else min(left, self.poll))
            return True
