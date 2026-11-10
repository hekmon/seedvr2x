"""Windows' stops at once besides a second Ctrl-C (stop.py): Ctrl-Break, SIGTERM's twin there,
and a console's close, its user's logoff or the system's shutdown (stop.Closing), each said by
its own name and ending with 143, as SIGTERM does on POSIX. On Linux the handler is driven
directly, BREAK being SIGTERM here; the native cases send the events to a console of its own."""

import logging
import signal
import sys
from collections.abc import Generator
from pathlib import Path
from types import FrameType

import pytest
from test_stop import Leaving, until

import main
import stop as stop_module
from stop import AT_ONCE, BREAK, CLOSINGS, Closing, Terminated, Watch, named

CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT = 0, 1, 2


@pytest.fixture(autouse=True)
def fresh() -> Generator[None]:
    """The stop's module state as a process starts: no cause said, nobody ending it, BREAK's
    handler given back."""
    before = signal.getsignal(BREAK)
    yield
    signal.signal(BREAK, before)
    stop_module._cause = None  # pyright: ignore[reportPrivateUsage]
    stop_module._ender = None  # pyright: ignore[reportPrivateUsage]


def test_ctrl_break_is_sigterms_twin_on_windows_alone() -> None:
    if sys.platform == "win32":
        assert (signal.SIGTERM, signal.SIGBREAK) == AT_ONCE and signal.SIGBREAK == BREAK
    else:
        assert (signal.SIGTERM,) == AT_ONCE and signal.SIGTERM == BREAK
    assert (named(signal.SIGINT), named(signal.SIGTERM)) == ("Ctrl-C", "SIGTERM")
    assert Terminated().how == "SIGTERM" and Terminated("Ctrl-Break").how == "Ctrl-Break"


@pytest.mark.parametrize("how", ["SIGTERM", "Ctrl-Break", "the console closed"])
def test_a_stop_at_once_is_said_by_its_own_name_with_143(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    def run(args: object) -> int:
        raise Terminated(how)

    monkeypatch.setattr(main, "_run", run)
    with caplog.at_level(logging.ERROR):
        assert main.main(["in.mkv", "-o", "out.mkv"]) == 143
    assert f"stopped at once ({how})" in caplog.text


class Recorded:
    """What Closing's handler did: the finish, the leave's statuses, the holds."""

    def __init__(self) -> None:
        self.finished, self.left, self.held = 0, list[int](), 0

    def closing(self, wait: float = 0.05) -> Closing:
        return Closing(self.finish, leave=self.left.append, wait=wait, hold=self.hold)

    def finish(self) -> None:
        self.finished += 1

    def hold(self) -> None:
        self.held += 1


def test_no_console_handler_off_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("Windows registers it: test_on_windows_* below")
    before = signal.getsignal(BREAK)
    with Recorded().closing() as closing:
        assert not closing.active and signal.getsignal(BREAK) is before


@pytest.mark.parametrize("event", [CTRL_C_EVENT, CTRL_BREAK_EVENT])
def test_ctrl_c_and_ctrl_break_are_passed_on_to_pythons_handlers(event: int) -> None:
    recorded = Recorded()
    assert recorded.closing().handle(event) == 0
    assert (recorded.finished, recorded.left, recorded.held) == (0, [], 0)
    assert stop_module._cause is None  # pyright: ignore[reportPrivateUsage]


def test_a_close_the_main_thread_ends_is_left_to_it() -> None:
    raised: list[int] = []

    def answer(number: int, frame: FrameType | None) -> None:
        raised.append(number)
        stop_module.ending()  # as main._end does once the run has unwound

    signal.signal(BREAK, answer)
    recorded = Recorded()
    assert recorded.closing(wait=5.0).handle(CTRL_CLOSE_EVENT) == 1
    assert raised == [BREAK]
    assert (recorded.finished, recorded.left, recorded.held) == (0, [], 1)
    assert named(BREAK) == "the console closed"


@pytest.mark.parametrize("event", sorted(CLOSINGS))
def test_a_close_the_main_thread_doesnt_end_in_time_is_ended_from_the_handler(
    caplog: pytest.LogCaptureFixture, event: int
) -> None:
    raised: list[int] = []
    signal.signal(BREAK, lambda number, frame: raised.append(number))
    recorded = Recorded()
    with caplog.at_level(logging.ERROR, logger="cli.stop"):
        assert recorded.closing().handle(event) == 1
    assert raised == [BREAK]
    assert (recorded.finished, recorded.left, recorded.held) == (1, [143], 0)
    assert f"stopped at once ({CLOSINGS[event]}): the main thread had not ended the run" in (
        caplog.text
    )
    assert stop_module._ender is None  # pyright: ignore[reportPrivateUsage]


def test_a_close_while_the_stops_are_ignored_waits_for_the_main_threads_end() -> None:
    # main._ignore_stops: the main thread is ending the process already.
    signal.signal(BREAK, signal.SIG_IGN)
    recorded = Recorded()
    assert recorded.closing().handle(CTRL_CLOSE_EVENT) == 1
    assert (recorded.finished, recorded.left, recorded.held) == (0, [], 1)


def test_a_close_once_the_run_is_over_waits_for_the_process_to_end() -> None:
    raised: list[int] = []
    signal.signal(BREAK, lambda number, frame: raised.append(number))
    recorded = Recorded()
    closing = recorded.closing()
    with closing:
        pass
    assert closing.handle(CTRL_CLOSE_EVENT) == 1
    assert raised == [] and (recorded.finished, recorded.left, recorded.held) == (0, [], 1)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' console handler")
def test_on_windows_ctrl_break_is_terminated_wherever_no_stop_is_in_force() -> None:
    assert sys.platform == "win32"  # for the type checker
    before = signal.getsignal(signal.SIGBREAK)
    with Recorded().closing() as closing:
        assert closing.active
        if before is signal.SIG_DFL:
            assert signal.getsignal(signal.SIGBREAK) is stop_module._terminated  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(Terminated, match="Ctrl-Break"):
                signal.raise_signal(signal.SIGBREAK)
    assert signal.getsignal(signal.SIGBREAK) is before


def test_closing_is_around_the_commands_run() -> None:
    source = Path(main.__file__).read_text()
    assert "stopping.Watch(_flush, WATCH_ALSO), stopping.Closing(_flush)" in source


# SIGBREAK's number on Windows (signal.h), which the C runtime's handler writes to the wakeup.
SIGBREAK = 21


def test_the_watcher_ends_a_ctrl_break_asked_again_with_143(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Windows' AT_ONCE on any platform: the watcher reads Ctrl-Break's byte as SIGTERM's twin.
    monkeypatch.setattr(stop_module, "AT_ONCE", (signal.SIGTERM, SIGBREAK))
    leaving = Leaving()
    with Watch(leaving.finish, leave=leaving.leave) as watch:
        assert watch.active
        watch.put(bytes([SIGBREAK]))
        until(lambda: watch._pending == [SIGBREAK])  # pyright: ignore[reportPrivateUsage]
        assert leaving.statuses == []
        watch.put(bytes([SIGBREAK]))
        until(lambda: leaving.statuses != [])
    assert leaving.statuses == [143] and leaving.finished == 1
    assert "stopped at once (Ctrl-Break), asked again before the main thread answered" in (
        caplog.text
    )


def test_the_watchers_wakeup_is_a_socket_on_windows_and_a_pipe_elsewhere() -> None:
    leaving = Leaving()
    with Watch(leaving.finish, leave=leaving.leave) as watch:
        assert watch.active
        sockets = watch._sockets  # pyright: ignore[reportPrivateUsage]
        assert (len(sockets) == 2) == (sys.platform == "win32")
        assert signal.set_wakeup_fd(watch.write, warn_on_full_buffer=False) == watch.write
    assert signal.set_wakeup_fd(-1) == -1


def test_verify_stopped_by_ctrl_break_says_so_with_143(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Closing makes Windows' Ctrl-Break a Terminated wherever no stop of the run's is in force:
    # seedvr2x verify says it as it says a Ctrl-C, not as a traceback.
    def checking(output: Path) -> int:
        raise Terminated("Ctrl-Break")

    monkeypatch.setattr(main, "_verify", checking)
    with caplog.at_level(logging.ERROR):
        assert main.main(["verify", "out.mkv"]) == 143
    assert "stopped (Ctrl-Break)" in caplog.text


@pytest.mark.parametrize("closed", [True, False])
def test_the_handler_stays_registered_once_a_close_is_handled(
    monkeypatch: pytest.MonkeyPatch, closed: bool
) -> None:
    # Removing a console handler waits for it while it runs, and it holds: a main thread that
    # unwound a close and removed it waited until Windows ended the process, with 0xC000013A
    # (natively, 2026-11-09). Windows faked: the removal recorded.
    monkeypatch.setattr(stop_module.sys, "platform", "win32")
    removed: list[bool] = []
    signal.signal(BREAK, lambda number, frame: stop_module.ending())
    closing = Recorded().closing(wait=5.0)
    closing.active = True
    closing._unregister = lambda: removed.append(True)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    if closed:
        assert closing.handle(CTRL_CLOSE_EVENT) == 1
    closing.__exit__(None, None, None)
    assert removed == ([] if closed else [True])


@pytest.mark.parametrize("closed", [True, False])
def test_a_run_a_close_ended_ends_by_end_process_with_its_status(
    monkeypatch: pytest.MonkeyPatch, closed: bool
) -> None:
    # Python's own exit, with torch's libraries loaded, ended a closed console's process with 0
    # whatever its status, where TerminateProcess (stop.end_process) keeps it (natively,
    # 2026-11-09): main ends such a run through _end; any other returns as before.
    ended: list[int] = []

    def run(args: object) -> int:
        if closed:
            stop_module._cause = CLOSINGS[CTRL_CLOSE_EVENT]  # pyright: ignore[reportPrivateUsage]
        return 143

    def end(status: int) -> None:
        ended.append(status)

    monkeypatch.setattr(main, "_run", run)
    monkeypatch.setattr(main, "_end", end)
    monkeypatch.setattr(main, "_ignore_stops", lambda: None)
    monkeypatch.setattr(main, "HAS_SIGUSR1", False)  # no faulthandler left in pytest's process
    assert main.main(["in.mkv", "-o", "out.mkv"], owns_process=True) == 143
    assert ended == ([143] if closed else [])


@pytest.mark.parametrize(
    ("raised", "status", "said"),
    [
        (Terminated("the console closed"), 143, "stopped at once (the console closed)"),
        (RuntimeError("boom"), 1, "the run failed as the console closed: RuntimeError('boom')"),
    ],
)
def test_a_run_a_close_ended_with_an_exception_ends_by_end_process(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raised: BaseException,
    status: int,
    said: str,
) -> None:
    # An exception out of _main after a close (a Ctrl-Break the run's stops didn't take, an
    # error raised as the console closed) ends through _end too, with its status, never by
    # Python's exit (the review of 2026-11-10).
    ended: list[int] = []

    def run(args: object) -> int:
        stop_module._cause = CLOSINGS[CTRL_CLOSE_EVENT]  # pyright: ignore[reportPrivateUsage]
        raise raised

    def end(status: int) -> None:
        ended.append(status)

    monkeypatch.setattr(main, "_run", run)
    monkeypatch.setattr(main, "_main", lambda argv, owns_process: run(argv))
    monkeypatch.setattr(main, "_end", end)
    monkeypatch.setattr(main, "_ignore_stops", lambda: None)
    monkeypatch.setattr(main, "HAS_SIGUSR1", False)
    with caplog.at_level(logging.ERROR):
        assert main.main(["in.mkv", "-o", "out.mkv"], owns_process=True) == status
    assert ended == [status] and said in caplog.text
