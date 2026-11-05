"""The autouse fixtures the library's tests need, and the CLI's tests with them
(apps/seedvr2x/tests/conftest.py imports them from here: core/tests is on its pythonpath).

For every test: the GPU's driver is asked nothing unless the test is a GPU one.

seedvr2x asks the driver whether it gives memory the card doesn't have, where a run first takes
the GPU and where it can (seedvr2x/memory.py: behind WSL2's GPU device). A CPU test that runs the
command line that far on such a machine would ask the real card, which another process may be
using: for the tests not marked gpu, WSL2's device isn't there, nothing was found before, and
the allocator's setting a run leaves in the environment goes with the test. A test's child
process doesn't inherit this: one that runs the stand-in model has its own
(test_cli_run.stand_in_model), and no other gets as far as the GPU.

Where a run holds itself to the card, it reads the whole card's free memory through NVML
(memory._read), and a CPU test that stands in for such a driver would read the machine's
real card: for the tests not marked gpu the machine has no NVML, its library's name one that
no loader finds, so that such a test goes by its stand-in's figure, with the warning a machine
without NVML gets. tests/test_memory.py gives the reading a stand-in of its own.

What a test says it measured is in the wall clock's seconds by the ratio of the run's clock to the
wall's (clock.py): the machine's own would drift a long session's expectations (WSL2's
runs 10% away), so every test has a ratio of 1.0, frozen clocks, unless it sets another
(tests/test_clock.py)."""

from collections.abc import Iterator

import pytest

from seedvr2x import clock, memory
from seedvr2x.errors import Refusal, Setting


def cli_say(setting: Setting) -> str:
    """The command line's words for a setting: its option, "--" and the setting's name with its
    underscores as hyphens, then the value (apps/seedvr2x/planner.py, `option`, is the command
    line's own; this is the tests' copy, so that a library test holds a refusal's text as the
    command line says it without importing the command line)."""
    return "--" + setting.name.replace("_", "-") + (f" {setting.value}" if setting.value else "")


def cli_words(error: Refusal) -> str:
    """A refusal's message as the command line words it (Refusal.words with cli_say)."""
    return error.words(cli_say)


@pytest.fixture(autouse=True)
def no_driver_asked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(memory, "paravirtual", lambda: False)
        monkeypatch.setattr(memory, "NVML", "libnvidia-ml.so.none-in-a-cpu-test")
        for name in memory.VARIABLES:
            monkeypatch.delenv(name, raising=False)
    state = (("_spills", False), ("_native", False), ("_limit", None), ("_footing", None))
    held = (("_segments", False), ("_plan_s_limit", None), ("_base", None), ("_kept", None))
    for name, value in (*state, *held, ("_planned", None), ("_left", memory.HELD_BACK)):
        monkeypatch.setattr(memory, name, value)
    yield


@pytest.fixture(autouse=True)
def one_to_one_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clock, "pace", clock.Pace(lambda: 0.0, lambda: 0.0))
