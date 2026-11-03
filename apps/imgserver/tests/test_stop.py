"""Stopping the server (DESIGN.md, The image server, Stopping): no new request, the picture running
finished and answered, the waiting ones answered 503, then the exit. In-process through main.shut,
and out of process through SIGTERM to main.serve, which owns the signals (a process of its own
because only a main thread may take them, and a second signal ends the process)."""

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
from serving import Gate, live, png_of, until

import main
import service as services
from seedvr2x import model

TESTS = Path(__file__).resolve().parent

# main.serve on the stand-in machine of serving.py, as a process: the CPU models of the library's
# tests, the driver asked nothing, and a pass that takes `argv[5]` seconds, announced as RUNNING on
# stderr, for every picture after the warm-up's (the first). The server's log goes to stderr as
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
    if len(calls) > 1:
        print("RUNNING", file=sys.stderr, flush=True)
        time.sleep(float(sys.argv[4]))


machine.fail = slow
options = serving.options(Path(sys.argv[5]))
sys.exit(main.serve(options, None, ("127.0.0.1", 0), lambda: machine, Metrics()))
"""


class Process:
    """The driver's process and its stderr, read line by line in a thread of its own."""

    def __init__(self, seconds: float, tmp_path: Path) -> None:
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
def driver(tmp_path: Path) -> Iterator[Callable[[float], Process]]:
    """Starts the driver with a pass of the given seconds; every process left is killed."""
    started: list[Process] = []

    def start(seconds: float) -> Process:
        started.append(Process(seconds, tmp_path))
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
    driver: Callable[[float], Process],
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
    driver: Callable[[float], Process],
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
