"""The nodes' tests: the library's autouse fixtures (core/tests/library_fixtures.py, on this suite's
pythonpath), the stand-in models (core/tests/stand_ins.py), and the stand-in for ComfyUI's few
modules (tests/comfy_stand_in), reset for every test with the nodes' own state."""

from collections.abc import Iterator

import comfy.model_management as management
import pytest
import server
from comfy.utils import ProgressBar
from comfyui import models
from library_fixtures import no_driver_asked, one_to_one_clock  # noqa: F401
from stand_ins import cpu_models  # noqa: F401
from standins import machine  # noqa: F401


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
