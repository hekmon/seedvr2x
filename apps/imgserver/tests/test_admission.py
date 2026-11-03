"""Which pictures run and which wait (admission.py; DESIGN.md, The image server, Running), on
threads with a short `poll`: admitted at once where the picture fits, at most `most` at once,
the pool bounding the sum of the shares, first come first served (a large picture that waits is
never passed by a later small one), Full past `queue` waiting, stop() answering the waiters
Stopping and refusing new ones while the running finish, a client gone giving up its place, and
the ticket's wait. Nothing sleeps for long: events and a poll of 10 ms."""

import threading
import time
from collections.abc import Callable

import pytest

from admission import Admission, Full, Gone, Stopping, Ticket

POLL = 0.01
LONG = 5.0


class Entrant:
    """A picture entering on a thread of its own: `outcome` is its ticket or the exception."""

    def __init__(
        self, admission: Admission, need: int, gone: Callable[[], bool] = lambda: False
    ) -> None:
        self.outcome: Ticket | BaseException | None = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(admission, need, gone), daemon=True)
        self.thread.start()

    def _run(self, admission: Admission, need: int, gone: Callable[[], bool]) -> None:
        try:
            self.outcome = admission.enter(need, gone)
        except BaseException as error:
            self.outcome = error
        self.done.set()

    def wait(self) -> bool:
        return self.done.wait(LONG)

    @property
    def ticket(self) -> Ticket:
        assert self.wait()
        assert isinstance(self.outcome, Ticket), self.outcome
        return self.outcome

    def stays_waiting(self, seconds: float = 0.15) -> bool:
        return not self.done.wait(seconds)


def until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + LONG
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)


def make(pool: int = 100, most: int = 4, queue: int = 8) -> Admission:
    return Admission(pool, most, queue, POLL)


def test_a_picture_that_fits_is_admitted_at_once() -> None:
    admission = make()
    ticket = admission.enter(60)
    assert admission.running == 1
    assert admission.waiting == 0
    assert ticket.need == 60
    assert ticket.admitted is not None
    assert ticket.waited < 0.05
    admission.leave(ticket)
    assert admission.running == 0
    assert admission.idle(0)


def test_a_need_of_the_whole_pool_is_admitted() -> None:
    admission = make(pool=100)
    admission.leave(admission.enter(100))


def test_a_need_over_the_pool_is_a_value_error() -> None:
    admission = make(pool=100)
    with pytest.raises(ValueError, match="101"):
        admission.enter(101)
    assert admission.running == 0 and admission.waiting == 0


@pytest.mark.parametrize("pool, most, queue", [(-1, 1, 0), (10, 0, 0), (10, 1, -1)])
def test_impossible_limits_are_refused(pool: int, most: int, queue: int) -> None:
    with pytest.raises(ValueError):
        Admission(pool, most, queue)


def test_at_most_that_many_run_at_once() -> None:
    admission = make(pool=1000, most=2)
    first, second = admission.enter(1), admission.enter(1)
    third = Entrant(admission, 1)
    until(lambda: admission.waiting == 1)
    assert third.stays_waiting()
    assert admission.running == 2
    admission.leave(first)
    assert third.wait()
    assert admission.running == 2
    admission.leave(second)
    admission.leave(third.ticket)


def test_the_pool_bounds_the_sum_of_the_needs() -> None:
    admission = make(pool=100, most=8)
    small = [admission.enter(20), admission.enter(20)]
    large = Entrant(admission, 70)  # 40 + 70 > 100
    until(lambda: admission.waiting == 1)
    assert large.stays_waiting()
    admission.leave(small[0])
    ticket = large.ticket  # 20 + 70 <= 100: it fits now
    assert admission.running == 2
    admission.leave(small[1])
    admission.leave(ticket)


def test_a_large_picture_waits_while_small_ones_run_and_starts_when_they_leave() -> None:
    admission = make(pool=100, most=8)
    small = [admission.enter(30), admission.enter(30)]
    large = Entrant(admission, 80)
    until(lambda: admission.waiting == 1)
    assert large.stays_waiting()
    admission.leave(small[0])
    assert large.stays_waiting()  # 30 + 80 > 100
    admission.leave(small[1])
    ticket = large.ticket
    assert ticket.waited > 0.1
    admission.leave(ticket)


def test_first_come_first_served() -> None:
    # A later small picture never starts before an earlier large one that waits, though it would
    # fit beside the running one: otherwise smaller pictures would pass the large one for ever.
    admission = make(pool=100, most=8)
    running = admission.enter(50)
    large = Entrant(admission, 80)
    until(lambda: admission.waiting == 1)
    small = Entrant(admission, 10)
    until(lambda: admission.waiting == 2)
    assert small.stays_waiting()
    assert large.stays_waiting(0.05)
    admission.leave(running)
    first = large.ticket
    # The small one is admitted beside the large one (80 + 10 <= 100), after it.
    assert small.wait()
    assert small.ticket.admitted is not None and first.admitted is not None
    assert first.admitted <= small.ticket.admitted
    admission.leave(first)
    admission.leave(small.ticket)


def test_waiting_pictures_start_in_the_order_they_came() -> None:
    admission = make(pool=10, most=1)
    running = admission.enter(10)
    order: list[int] = []
    entrants: list[Entrant] = []
    for each in range(4):
        entrants.append(Entrant(admission, 10))
        until(lambda each=each: admission.waiting == each + 1)
    for position, entrant in enumerate(entrants):
        admission.leave(running)
        running = entrant.ticket
        order.append(position)
        assert admission.waiting == len(entrants) - position - 1
    admission.leave(running)
    assert order == [0, 1, 2, 3]
    stamps = [entrant.ticket.admitted or 0.0 for entrant in entrants]
    assert stamps == sorted(stamps)


def test_full_past_the_queue() -> None:
    admission = make(pool=10, most=1, queue=2)
    running = admission.enter(10)
    waiters = [Entrant(admission, 10), Entrant(admission, 10)]
    until(lambda: admission.waiting == 2)
    with pytest.raises(Full):
        admission.enter(10)
    assert admission.waiting == 2
    admission.leave(running)
    for waiter in waiters:
        admission.leave(waiter.ticket)


def test_with_no_queue_a_picture_that_would_wait_is_full() -> None:
    admission = make(pool=10, most=1, queue=0)
    running = admission.enter(10)
    with pytest.raises(Full):
        admission.enter(1)
    assert admission.waiting == 0
    admission.leave(running)
    admission.leave(admission.enter(10))


def test_with_no_queue_a_picture_that_fits_is_admitted() -> None:
    admission = make(pool=10, most=2, queue=0)
    first, second = admission.enter(5), admission.enter(5)
    assert admission.running == 2
    admission.leave(first)
    admission.leave(second)


def test_stop_answers_every_waiter_stopping_and_refuses_new_ones() -> None:
    admission = make(pool=10, most=1)
    running = admission.enter(10)
    waiters = [Entrant(admission, 10) for _ in range(3)]
    until(lambda: admission.waiting == 3)
    admission.stop()
    for waiter in waiters:
        assert waiter.wait()
        assert isinstance(waiter.outcome, Stopping)
    assert admission.waiting == 0
    with pytest.raises(Stopping):
        admission.enter(1)
    # The picture running is unaffected: it finishes and gives its share back.
    assert admission.running == 1
    assert not admission.idle(0.05)
    admission.leave(running)
    assert admission.idle(0)


def test_stop_refuses_a_picture_that_would_fit() -> None:
    admission = make()
    admission.stop()
    with pytest.raises(Stopping):
        admission.enter(1)


def test_a_client_gone_gives_up_its_place() -> None:
    admission = make(pool=10, most=1)
    running = admission.enter(10)
    leaves = threading.Event()
    first = Entrant(admission, 10, leaves.is_set)
    until(lambda: admission.waiting == 1)
    second = Entrant(admission, 10)
    until(lambda: admission.waiting == 2)
    leaves.set()
    assert first.wait()
    assert isinstance(first.outcome, Gone)
    assert admission.waiting == 1
    admission.leave(running)
    # The next in line is admitted, not blocked behind the one that left.
    admission.leave(second.ticket)
    assert admission.idle(0)


def test_gone_is_asked_every_poll_while_waiting_and_not_when_admitted() -> None:
    admission = make(pool=10, most=1)
    asked: list[int] = []
    # Admitted at once: never asked.
    admission.leave(admission.enter(10, lambda: bool(asked.append(1))))
    assert asked == []
    running = admission.enter(10)
    waiter = Entrant(admission, 10, lambda: bool(asked.append(1)))
    until(lambda: len(asked) >= 3)
    admission.leave(running)
    admission.leave(waiter.ticket)


def test_a_client_gone_in_front_of_a_fitting_picture_lets_it_in() -> None:
    # A large picture at the head that left no longer holds a smaller one back.
    admission = make(pool=100, most=8)
    running = admission.enter(60)
    leaves = threading.Event()
    large = Entrant(admission, 90, leaves.is_set)
    until(lambda: admission.waiting == 1)
    small = Entrant(admission, 30)
    until(lambda: admission.waiting == 2)
    assert small.stays_waiting(0.05)
    leaves.set()
    assert isinstance(small.ticket, Ticket)
    assert isinstance(large.outcome, Gone)
    admission.leave(running)
    admission.leave(small.ticket)


def test_idle_waits_for_the_running_and_the_waiting() -> None:
    admission = make(pool=10, most=1)
    assert admission.idle(0)
    running = admission.enter(10)
    assert not admission.idle(0.03)
    threading.Timer(0.05, admission.leave, args=(running,)).start()
    assert admission.idle(LONG)


def test_idle_counts_the_waiting_too() -> None:
    admission = make(pool=10, most=1)
    running = admission.enter(10)
    waiter = Entrant(admission, 10)
    until(lambda: admission.waiting == 1)
    admission.leave(running)
    # The waiter is admitted and runs: not idle until it leaves.
    ticket = waiter.ticket
    assert not admission.idle(0.03)
    admission.leave(ticket)
    assert admission.idle(0)


def test_when_idle_runs_the_work_with_nothing_running() -> None:
    admission = make()
    ran: list[str] = []
    assert admission.when_idle(lambda: ran.append("done")) is True
    assert ran == ["done"]


def test_when_idle_does_not_run_while_a_picture_runs() -> None:
    admission = make()
    ticket = admission.enter(10)
    ran: list[str] = []
    assert admission.when_idle(lambda: ran.append("done")) is False
    assert ran == []
    admission.leave(ticket)
    assert admission.when_idle(lambda: ran.append("done")) is True
    assert ran == ["done"]


def test_when_idle_runs_under_the_lock_so_that_no_picture_is_admitted_meanwhile() -> None:
    admission = make()
    inside, release = threading.Event(), threading.Event()

    def work() -> None:
        inside.set()
        assert release.wait(LONG)

    runner = threading.Thread(target=admission.when_idle, args=(work,), daemon=True)
    runner.start()
    assert inside.wait(LONG)
    # A picture that comes now is not admitted until the work is done, though the pool is free
    # (it blocks on the lock the work holds).
    entrant = Entrant(admission, 10)
    assert entrant.stays_waiting()
    release.set()
    runner.join(LONG)
    assert entrant.ticket.need == 10


def test_when_idle_work_that_raises_leaves_the_admission_usable() -> None:
    admission = make()

    def work() -> None:
        raise RuntimeError("a hold that failed")

    with pytest.raises(RuntimeError, match="a hold that failed"):
        admission.when_idle(work)
    admission.leave(admission.enter(10))
    assert admission.idle(0)


def test_the_ticket_waited_is_zero_until_admitted() -> None:
    ticket = Ticket(5)
    assert ticket.waited == 0.0
    ticket.admitted = ticket.came + 1.5
    assert ticket.waited == pytest.approx(1.5)


def test_a_waiting_picture_reports_how_long_it_waited() -> None:
    admission = make(pool=10, most=1)
    running = admission.enter(10)
    waiter = Entrant(admission, 10)
    until(lambda: admission.waiting == 1)
    time.sleep(0.12)
    admission.leave(running)
    assert 0.1 <= waiter.ticket.waited < LONG
    admission.leave(waiter.ticket)


def test_a_pool_is_never_over_committed_under_load() -> None:
    # Twenty pictures of 1 to 5 units on a pool of 10 and at most 3 at once: at no moment do the
    # running ones' needs pass the pool or their number the limit.
    admission = make(pool=10, most=3, queue=64)
    lock = threading.Lock()
    state = {"used": 0, "count": 0, "worst": 0, "most": 0}
    errors: list[BaseException] = []

    def picture(need: int) -> None:
        try:
            ticket = admission.enter(need)
            with lock:
                state["used"] += need
                state["count"] += 1
                state["worst"] = max(state["worst"], state["used"])
                state["most"] = max(state["most"], state["count"])
            time.sleep(0.003)
            with lock:
                state["used"] -= need
                state["count"] -= 1
            admission.leave(ticket)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=picture, args=(1 + each % 5,)) for each in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(LONG)
    assert errors == []
    assert state["worst"] <= 10
    assert state["most"] <= 3
    assert admission.idle(0)
