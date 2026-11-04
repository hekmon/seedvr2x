"""Stopping the server (DESIGN.md, The image server, Stopping): no new request, the picture running
finished and answered, the waiting ones answered 503, then the exit. In-process through main.shut,
and out of process through SIGTERM to main.serve, which owns the signals (a process of its own
because only a main thread may take them, and a second signal ends the process).

And the server that a CUDA error failed (DESIGN.md, /health): the picture that failed answered 500,
the waiting ones 503, /health failed meanwhile, then main.serve shuts the server and returns
main.FAILED_EXIT: in-process (main.serve in a thread, the signals it takes left alone) and as a
process, whose exit code is the container's."""

import http.client
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import serving
from serving import Gate, Live, StandIn, live, png_of, until
from test_still import stand_in_dit

import main
import service as services
from metrics import Metrics
from seedvr2x import model, weights

TESTS = Path(__file__).resolve().parent

# main.serve on the stand-in machine of serving.py, as a process: the CPU models of the library's
# tests, the driver asked nothing, and a pass that takes `argv[4]` seconds, announced as RUNNING on
# stderr, for every picture after the warm-up's first (the old tests' habit; it sleeps in the
# warm-up too); with `argv[6]` "cuda", the first request after the warm-up's two pictures fails
# with a CUDA error instead. The server's log goes to stderr as
# bare messages, `listening on http://HOST:PORT` among them.
DRIVER = r"""
import logging
import sys
import time
from pathlib import Path

for each in sys.argv[1:4]:
    sys.path.insert(0, each)

import pytest
import serving
import stand_ins
from test_still import stand_in_dit

import main
from metrics import Metrics
from seedvr2x import weights

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
patches = pytest.MonkeyPatch()
serving.quiet_driver(patches)
patches.setattr(weights, "check_models", lambda *args: None)
stand_in_dit(patches)
machine = serving.StandIn(stand_ins.cpu_models._get_wrapped_function()())
calls = []


def slow():
    calls.append(1)
    if sys.argv[6] == "cuda":
        # The warm-up's two pictures (landscape and portrait) pass; the first request fails.
        if len(calls) > 2:
            print("RUNNING", file=sys.stderr, flush=True)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
    elif len(calls) > 1:
        print("RUNNING", file=sys.stderr, flush=True)
        time.sleep(float(sys.argv[4]))


machine.fail = slow
options = serving.options(Path(sys.argv[5]))
sys.exit(main.serve(options, None, ("127.0.0.1", 0), lambda: machine, Metrics()))
"""


class Process:
    """The driver's process and its stderr, read line by line in a thread of its own."""

    def __init__(self, seconds: float, tmp_path: Path, mode: str = "slow") -> None:
        core_tests = TESTS.parents[2] / "core" / "tests"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                DRIVER,
                str(core_tests),
                str(TESTS.parent),
                str(TESTS),
                str(seconds),
                str(tmp_path),
                mode,
            ],
            stderr=subprocess.PIPE,
            text=True,
            env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        self.lines: list[str] = []
        self._new: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.lines.append(line.rstrip("\n"))
            self._new.put(line.rstrip("\n"))

    def wait_for(self, text: str, timeout: float = 180.0) -> str:
        """The first line from now on holding `text`; fails after `timeout` seconds or if the
        process ended without it."""
        while True:
            try:
                line = self._new.get(timeout=timeout)
            except queue.Empty:
                raise AssertionError(
                    f"no line with {text!r} in {timeout} s: {self.lines}"
                ) from None
            if text in line:
                return line

    def address(self) -> tuple[str, int]:
        line = self.wait_for("listening on http://")
        host, _, rest = line.split("http://", 1)[1].partition(":")
        return host, int(rest.split(",", 1)[0])


@pytest.fixture
def driver(tmp_path: Path) -> Iterator[Callable[..., Process]]:
    """Starts the driver with a pass of the given seconds; every process left is killed."""
    started: list[Process] = []

    def start(seconds: float, mode: str = "slow") -> Process:
        started.append(Process(seconds, tmp_path, mode))
        return started[-1]

    yield start
    for each in started:
        if each.process.poll() is None:
            each.process.kill()
        each.process.wait()


def post(address: tuple[str, int], data: bytes) -> int:
    connection = http.client.HTTPConnection(*address, timeout=300)
    try:
        connection.request("POST", "/upscale", data, {"Seedvr2x-Resolution": "48"})
        return connection.getresponse().status
    finally:
        connection.close()


# ---- In-process -------------------------------------------------------------------------------


def test_shut_finishes_the_picture_running_and_turns_the_waiting_one_away(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    with live(monkeypatch, tmp_path, cpu_models, count=1, queue=2) as server:
        gate = Gate()
        server.machine.fail = gate
        admitted = server.service.admitted
        assert admitted is not None
        address = server.running.address
        with ThreadPoolExecutor(3) as pool:
            running = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert gate.entered.wait(60)
            waiting = pool.submit(server.upscale, png_of(40, 24, 2), 48, 2)
            until(lambda: admitted.waiting == 1, "the second picture never waited")
            shutting = threading.Thread(target=main.shut, args=(server.running,))
            shutting.start()
            # The waiting picture is answered 503 at once, the running one still holds the stop.
            status, headers, body = waiting.result(60)
            assert status == 503
            assert "stopping" in body.decode()
            assert "retry-after" in headers
            assert server.service.state is services.State.STOPPING
            assert shutting.is_alive()
            gate.release.set()
            assert running.result(60)[0] == 200
        shutting.join(60)
        assert not shutting.is_alive(), "shut didn't return"
        assert gate.calls == 1
        with pytest.raises(ConnectionRefusedError):
            http.client.HTTPConnection(*address, timeout=5).request("GET", "/health")


def test_a_picture_sent_while_stopping_is_turned_away(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    """The service stopped, its socket still open: 503 stopping, and /health says so."""
    with live(monkeypatch, tmp_path, cpu_models) as server:
        server.service.stop()
        status, headers, body = server.upscale(png_of(40, 24), 48)
        assert status == 503
        assert "stopping" in body.decode()
        assert "retry-after" in headers
        status, _, body = server.request("GET", "/health")
        assert (status, body.strip()) == (503, b"stopping")


# ---- SIGTERM ----------------------------------------------------------------------------------


def test_sigterm_finishes_the_picture_running_and_exits_0(
    driver: Callable[..., Process],
) -> None:
    process = driver(4.0)
    address = process.address()
    process.wait_for("ready: /health says ready")
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(post, address, png_of(40, 24))
        process.wait_for("RUNNING")
        process.process.send_signal(signal.SIGTERM)
        assert running.result(120) == 200
    assert process.process.wait(120) == 0
    process.wait_for("stopped", 10)


def test_a_second_sigterm_while_stopping_ends_the_process_at_once(
    driver: Callable[..., Process],
) -> None:
    process = driver(120.0)
    address = process.address()
    process.wait_for("ready: /health says ready")
    with ThreadPoolExecutor(1) as pool:
        pool.submit(post, address, png_of(40, 24))
        process.wait_for("RUNNING")
        process.process.send_signal(signal.SIGTERM)
        process.wait_for("stopping")
        assert process.process.poll() is None, "the picture running should hold the stop"
        process.process.send_signal(signal.SIGTERM)
        assert process.process.wait(30) == 128 + signal.SIGTERM
    assert any("asked to stop again" in line for line in process.lines)


# ---- A CUDA error --------------------------------------------------------------------------------


def serving_in_a_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models, **given: int
) -> tuple[Live, threading.Thread, list[int]]:
    """main.serve on the stand-in machine in a thread of its own, its signals left alone (only a
    main thread may take them): the server it launched, ready, and the code it returns."""
    monkeypatch.setattr(weights, "check_models", lambda *args: None)
    stand_in_dit(monkeypatch)
    machine = StandIn(cpu_models)
    made: list[main.Running] = []
    launch = main.launch

    def recording(*args: object) -> main.Running:
        made.append(launch(*args))  # type: ignore[arg-type]
        return made[-1]

    monkeypatch.setattr(main, "launch", recording)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    codes: list[int] = []
    options = serving.options(tmp_path, **given)
    thread = threading.Thread(
        target=lambda: codes.append(
            main.serve(options, None, ("127.0.0.1", 0), lambda: machine, Metrics())
        ),
        name="serve",
    )
    thread.start()
    until(lambda: bool(made), "the server never listened")
    server = Live(made[0], machine, thread, [])
    until(lambda: server.service.state is services.State.READY, "the server never got ready", 120)
    return server, thread, codes


def test_a_cuda_error_answers_the_pictures_then_ends_the_server_with_the_failed_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_models: model.Models,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three pictures: A running, B running and failing with a CUDA error, C waiting. B is
    answered 500, C 503, /health says failed and no new picture is taken while A is still
    running, which holds the server up; once A is answered, serve shuts it and returns 3."""
    caplog.set_level("INFO")
    server, thread, codes = serving_in_a_thread(monkeypatch, tmp_path, cpu_models, count=2, queue=2)
    release_a, release_b = threading.Event(), threading.Event()
    entered = [threading.Event(), threading.Event()]
    calls: list[int] = []
    lock = threading.Lock()

    def hook() -> None:
        with lock:
            calls.append(1)
            mine = len(calls)
        entered[mine - 1].set()
        if mine == 1:
            assert release_a.wait(60)
        else:
            assert release_b.wait(60)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    server.machine.fail = hook
    admitted = server.service.admitted
    assert admitted is not None
    try:
        with ThreadPoolExecutor(4) as pool:
            a = pool.submit(server.upscale, png_of(40, 24, 1), 48, 1)
            assert entered[0].wait(60)
            b = pool.submit(server.upscale, png_of(40, 24, 2), 48, 2)
            assert entered[1].wait(60)
            c = pool.submit(server.upscale, png_of(40, 24, 3), 48, 3)
            until(lambda: admitted.waiting == 1, "the third picture never waited")
            release_b.set()
            status, _, body = b.result(60)
            assert status == 500 and "CUDA error" in body.decode()
            status, headers, body = c.result(60)
            assert status == 503 and "failed" in body.decode()
            assert "retry-after" not in headers
            assert server.service.failed.is_set()
            # A still runs: the server is up, and says it has failed.
            assert thread.is_alive()
            status, _, body = server.request("GET", "/health")
            assert (status, body.strip()) == (503, b"failed")
            status, _, body = server.upscale(png_of(40, 24, 4), 48, 4)
            assert status == 503 and "failed" in body.decode()
            assert thread.is_alive()
            release_a.set()
            assert a.result(60)[0] == 200
        thread.join(60)
        assert not thread.is_alive(), "serve didn't return"
    finally:
        release_a.set()
        release_b.set()
        thread.join(60)
    assert codes == [main.FAILED_EXIT] == [3]
    assert "illegal memory access" in caplog.text
    assert f"exits with code {main.FAILED_EXIT}" in caplog.text
    with pytest.raises(ConnectionRefusedError):
        http.client.HTTPConnection(*server.running.address, timeout=5).request("GET", "/health")


def test_a_failed_server_with_no_picture_running_ends_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_models: model.Models
) -> None:
    server, thread, codes = serving_in_a_thread(monkeypatch, tmp_path, cpu_models)

    def broken() -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    server.machine.fail = broken
    assert server.upscale(png_of(40, 24), 48)[0] == 500
    thread.join(60)
    assert not thread.is_alive()
    assert codes == [main.FAILED_EXIT]


def test_a_cuda_error_exits_the_process_with_the_failed_code(
    driver: Callable[..., Process],
) -> None:
    process = driver(0.0, "cuda")
    address = process.address()
    process.wait_for("ready: /health says ready")
    assert post(address, png_of(40, 24)) == 500
    assert process.process.wait(120) == main.FAILED_EXIT == 3
    process.wait_for("stopped", 10)
    text = "\n".join(process.lines)
    assert "illegal memory access" in text
    assert "the GPU's state is untrusted" in text
    assert "exits with code 3" in text
