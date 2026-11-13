"""seedvr2x's lines in ComfyUI's console (console.py): "[seedvr2x] " before each line of the
library's loggers and the nodes', through the root logger's handlers as ComfyUI sets them (a
handler of the message alone), other loggers' lines untouched; a text of several lines said a
line a record, each with its level (ComfyUI's formatter puts the level before a record's first
line alone); the library's checks of the model files at DEBUG there."""

import io
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from comfyui import console
from standins import Lines

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
    logging.getLogger("seedvr2x.model").info("models loaded in 1.0 s")
    logging.getLogger("seedvr2x.vendor.models.dit_7b.nadit").warning("a vendored line")
    logging.getLogger("another_node").info("another node's line")
    logging.getLogger("seedvr2xfoo").info("no child of seedvr2x's")
    logging.getLogger().info("the root's own")
    assert comfy_console.getvalue().splitlines() == [
        "[seedvr2x] keep in RAM between runs: auto -> kept",
        "[seedvr2x] models loaded in 1.0 s",
        "[seedvr2x] a vendored line",
        "another node's line",
        "no child of seedvr2x's",
        "the root's own",
    ]


def test_each_line_of_a_message_is_prefixed_and_its_arguments_made(
    comfy_console: io.StringIO, records_are_one_line: Lines
) -> None:
    # A record of several lines (the nodes say none: console.say) still reads as seedvr2x's in
    # every line, a record from a module that logs its own way.
    logging.getLogger("seedvr2x_comfyui.nodes").info("plan: %s\n%s", "window 21", "decode tiles")
    assert comfy_console.getvalue().splitlines() == [
        "[seedvr2x] plan: window 21",
        "[seedvr2x] decode tiles",
    ]
    assert records_are_one_line.found == ["[seedvr2x] plan: window 21\n[seedvr2x] decode tiles"]
    records_are_one_line.found.clear()


@pytest.fixture
def leveled_console() -> Iterator[io.StringIO]:
    """A root handler as ComfyUI's setup_logger makes one, its level before the message (v0.44.0,
    app/logger.py's ColoredFormatter: "[INFO] " then the message, the colours left out here), at
    INFO, written to a string."""
    written = io.StringIO()
    handler = logging.StreamHandler(written)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield written
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def test_a_text_of_several_lines_is_said_a_line_a_record_each_with_its_level(
    leveled_console: io.StringIO,
) -> None:
    nodes = logging.getLogger("seedvr2x_comfyui.nodes")
    records: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = records.append  # type: ignore[method-assign]
    nodes.addHandler(keep)
    try:
        console.say(nodes, logging.INFO, "plan: window 21\ndecode tiles\nhost memory")
        console.say(nodes, logging.WARNING, "refused: the card is short\nYou can choose int8.")
    finally:
        nodes.removeHandler(keep)
    assert leveled_console.getvalue().splitlines() == [
        "[INFO] [seedvr2x] plan: window 21",
        "[INFO] [seedvr2x] decode tiles",
        "[INFO] [seedvr2x] host memory",
        "[WARNING] [seedvr2x] refused: the card is short",
        "[WARNING] [seedvr2x] You can choose int8.",
    ]
    # One line a record, each holding the whole text for a handler that reads it whole.
    assert len(records) == 5 and all("\n" not in r.getMessage() for r in records)
    whole = {r.__dict__["seedvr2x_text"] for r in records[:3]}
    assert whole == {"plan: window 21\ndecode tiles\nhost memory"}
    # Each record names the line that said it, not console.say.
    assert all(r.funcName != "say" for r in records)


def test_the_librarys_checks_of_the_model_files_are_debug_in_comfyuis_console(
    leveled_console: io.StringIO,
) -> None:
    detail = io.StringIO()
    handler = logging.StreamHandler(detail)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    try:
        pull = logging.getLogger("seedvr2x.pull")
        pull.info("hekmon/seedvr2x at c14a2bc4, in Hugging Face's cache, a directory")
        pull.info("seedvr2x_ema_vae_fp16.safetensors: its size as seedvr2x pins it, 0.5 GB")
        logging.getLogger("seedvr2x.weights").info("VAE seedvr2x_ema_vae_fp16.safetensors: VAE")
        pull.warning("a warning of the pull's")
        logging.getLogger("seedvr2x.model").info("models loaded in 1.0 s")
        logging.getLogger("seedvr2x_comfyui.models").info("VAE, 0.5 GB, in Hugging Face's cache")
    finally:
        logging.getLogger().removeHandler(handler)
    # ComfyUI's console, at INFO: the checks left out, the rest as it was.
    assert leveled_console.getvalue().splitlines() == [
        "[WARNING] [seedvr2x] a warning of the pull's",
        "[INFO] [seedvr2x] models loaded in 1.0 s",
        "[INFO] [seedvr2x] VAE, 0.5 GB, in Hugging Face's cache",
    ]
    # A handler at DEBUG (ComfyUI's `--verbose DEBUG`, or a file log at DEBUG) has them, at DEBUG.
    assert detail.getvalue().splitlines()[:3] == [
        "[DEBUG] [seedvr2x] hekmon/seedvr2x at c14a2bc4, in Hugging Face's cache, a directory",
        "[DEBUG] [seedvr2x] seedvr2x_ema_vae_fp16.safetensors: its size as seedvr2x pins it,"
        " 0.5 GB",
        "[DEBUG] [seedvr2x] VAE seedvr2x_ema_vae_fp16.safetensors: VAE",
    ]
    assert console.QUIET == ("seedvr2x.pull", "seedvr2x.weights")


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
    # A process of its own, as ComfyUI starts: its root handler set first (the message alone, the
    # handler at INFO as setup_logger sets its stream handler), then the nodes' folder imported,
    # whose one line at import is said prefixed, and a line of the library's after it too.
    code = (
        "import logging, sys\n"
        "handler = logging.StreamHandler(sys.stdout)\n"
        "handler.setFormatter(logging.Formatter('%(message)s'))\n"
        "handler.setLevel(logging.INFO)\n"
        "logging.getLogger().addHandler(handler)\n"
        "logging.getLogger().setLevel(logging.INFO)\n"
        "import comfyui\n"
        "logging.getLogger('seedvr2x.model').info('a library line')\n"
        "logging.getLogger('seedvr2x.pull').info('a check of a model file')\n"
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
    assert lines[0].startswith(
        "[seedvr2x] seedvr2x's nodes read a video from its file shot by shot"
    ), lines
    # The check of a model file is DEBUG there, below the console's INFO.
    assert lines[-2:] == ["[seedvr2x] a library line", "ComfyUI line"]
