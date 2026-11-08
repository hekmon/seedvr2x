"""seedvr2x's lines in ComfyUI's console (console.py): "[seedvr2x] " before each line of the
library's loggers and the nodes', through the root logger's handlers as ComfyUI sets them (a
handler of the message alone), other loggers' lines untouched."""

import io
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from comfyui import console

HERE = Path(__file__).resolve().parent


@pytest.fixture
def comfy_console() -> Iterator[io.StringIO]:
    """A root handler as ComfyUI's setup_logger makes one (v0.44.0, app/logger.py: the message
    alone), at INFO, written to a string."""
    written = io.StringIO()
    handler = logging.StreamHandler(written)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield written
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def test_the_nodes_and_the_librarys_lines_are_prefixed_and_others_are_not(
    comfy_console: io.StringIO,
) -> None:
    # The package's import installed it (comfyui/__init__.py); the tests import it as ComfyUI does.
    logging.getLogger("seedvr2x_comfyui.models").info("keep in RAM between runs: auto -> kept")
    logging.getLogger("seedvr2x.model").info("attention: PyTorch's SDPA")
    logging.getLogger("seedvr2x.vendor.models.dit_7b.nadit").warning("a vendored line")
    logging.getLogger("another_node").info("another node's line")
    logging.getLogger("seedvr2xfoo").info("no child of seedvr2x's")
    logging.getLogger().info("the root's own")
    assert comfy_console.getvalue().splitlines() == [
        "[seedvr2x] keep in RAM between runs: auto -> kept",
        "[seedvr2x] attention: PyTorch's SDPA",
        "[seedvr2x] a vendored line",
        "another node's line",
        "no child of seedvr2x's",
        "the root's own",
    ]


def test_each_line_of_a_message_is_prefixed_and_its_arguments_made(
    comfy_console: io.StringIO,
) -> None:
    logging.getLogger("seedvr2x_comfyui.nodes").info("plan: %s\n%s", "window 21", "decode tiles")
    assert comfy_console.getvalue().splitlines() == [
        "[seedvr2x] plan: window 21",
        "[seedvr2x] decode tiles",
    ]


def test_the_root_handlers_levels_and_the_loggers_levels_still_hold(
    comfy_console: io.StringIO,
) -> None:
    # A debug line of the library is below the logger's level (the root's INFO): never made; a
    # handler at WARNING takes no INFO line.
    logging.getLogger("seedvr2x.memory").debug("a debug line")
    strict = io.StringIO()
    handler = logging.StreamHandler(strict)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    try:
        logging.getLogger("seedvr2x.memory").info("an info line")
        logging.getLogger("seedvr2x.memory").warning("a warning")
    finally:
        logging.getLogger().removeHandler(handler)
    assert comfy_console.getvalue().splitlines() == [
        "[seedvr2x] an info line",
        "[seedvr2x] a warning",
    ]
    assert strict.getvalue().splitlines() == ["[seedvr2x] a warning"]


def test_a_traceback_is_kept_after_the_prefixed_line(comfy_console: io.StringIO) -> None:
    try:
        raise ValueError("the card failed")
    except ValueError:
        logging.getLogger("seedvr2x_comfyui.models").exception("the run failed")
    lines = comfy_console.getvalue().splitlines()
    assert lines[0] == "[seedvr2x] the run failed"
    assert lines[1] == "Traceback (most recent call last):"
    assert lines[-1] == "ValueError: the card failed"


def test_install_is_once_however_often_it_is_asked() -> None:
    console.install()
    console.install()
    for name in console.LOGGERS:
        found = logging.getLogger(name)
        assert sum(isinstance(each, console.Prefixing) for each in found.handlers) == 1
        # The loggers propagate as before: the record itself is prefixed on its way.
        assert found.propagate is True


def test_comfyuis_console_shows_the_nodes_import_line_prefixed(tmp_path: Path) -> None:
    # A process of its own, as ComfyUI starts: its root handler set first (the message alone, at
    # INFO), then the nodes' folder imported, whose one line at import is said prefixed, and a
    # line of the library's after it too.
    code = (
        "import logging, sys\n"
        "handler = logging.StreamHandler(sys.stdout)\n"
        "handler.setFormatter(logging.Formatter('%(message)s'))\n"
        "logging.getLogger().addHandler(handler)\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        "import comfyui\n"
        "logging.getLogger('seedvr2x.pull').info('a library line')\n"
        "logging.getLogger('comfy').info('ComfyUI line')\n"
    )
    paths = [str(HERE.parent.parent), str(HERE.parent.parent.parent / "core" / "tests")]
    paths.append(str(HERE / "comfy_stand_in"))
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env=environment,
    )
    lines = done.stdout.splitlines()
    assert lines[0].startswith("[seedvr2x] seedvr2x's nodes hold a clip whole in RAM"), lines
    assert lines[-2:] == ["[seedvr2x] a library line", "ComfyUI line"]
