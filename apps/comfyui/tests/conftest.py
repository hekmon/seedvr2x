"""The nodes' tests: the library's autouse fixtures (core/tests/library_fixtures.py, on this suite's
pythonpath), the stand-in models (core/tests/stand_ins.py), and the stand-in for ComfyUI's few
modules (tests/comfy_stand_in), reset for every test with the nodes' own state."""

import threading
from collections.abc import Iterator

import comfy.model_management as management
import pytest
import server
from comfy.utils import ProgressBar
from comfyui import models, progress
from library_fixtures import no_card_failed, no_driver_asked, one_to_one_clock  # noqa: F401
from stand_ins import cpu_models  # noqa: F401
from standins import machine, not_one_short_line  # noqa: F401


@pytest.fixture(autouse=True)
def comfy_as_new() -> Iterator[server.PromptServer]:
    """ComfyUI's stand-in, the nodes' registry and their machine as new; the server's recorded
    texts and the bars' records fresh."""
    management.reset()
    models.reset()
    models.use(None)
    ProgressBar.created.clear()
    ProgressBar.on_update = None
    server.PromptServer.instance = server.PromptServer()
    yield server.PromptServer.instance
    models.reset()
    models.use(None)
    server.PromptServer.instance = None


@pytest.fixture(autouse=True)
def texts_are_one_short_line(comfy_as_new: server.PromptServer) -> Iterator[None]:
    """Every text any test's nodes sent to a node's text box is one status line of at most 40
    characters with no name (the user, 2026-11-08: ComfyUI squeezes the box, and a longer text
    must be scrolled): the check is on the server's record at the end of each test, so that no
    path is missed. tests/test_lines.py holds the lines' forms."""
    yield
    # None, or no attribute at all where a test runs a node without the server (as ComfyUI has
    # none until its server is made).
    instance = getattr(server.PromptServer, "instance", None)
    if instance is None:
        return
    bad = not_one_short_line([text for text, _ in instance.texts])
    assert not bad, f"node texts over {progress.LIMIT} characters, on several lines or named: {bad}"


@pytest.fixture(autouse=True)
def no_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text's ticker (progress.Ticker) waits for its stop and says nothing between events:
    a stand-in run is slower than the second it ticks at, and the tests that count a node's texts
    would count its seconds too. The tests of the ticker and of the stall give their own pause
    (tests/test_progress.py's Cue)."""

    def never(stopped: threading.Event, seconds: float) -> None:
        stopped.wait()

    monkeypatch.setattr(progress, "_wait", never)
