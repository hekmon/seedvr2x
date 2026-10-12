"""For every test: the GPU's driver is asked nothing unless the test is a GPU one.

seedvr2x asks the driver whether it gives memory the card doesn't have, where a run first takes
the GPU and where it can (runtime/memory.py: behind WSL2's GPU device). A CPU test that runs the
command line that far on such a machine would ask the real card, which another process may be
using: for the tests not marked gpu, WSL2's device isn't there, nothing was found before, and
the allocator's setting a run leaves in the environment goes with the test. A test's child
process doesn't inherit this: one that runs the stand-in model has its own
(test_cli_run.stand_in_model), and no other gets as far as the GPU.

Where a run holds itself to the card, it reads the whole card's free memory through NVML
(memory.card_free), and a CPU test that stands in for such a driver would read the machine's
real card: for the tests not marked gpu the machine has no NVML, its library's name one that
no loader finds, so that such a test goes by its stand-in's figure, with the warning a machine
without NVML gets. tests/test_memory.py gives the reading a stand-in of its own."""

from collections.abc import Iterator

import pytest

from seedvr2x.runtime import memory


@pytest.fixture(autouse=True)
def no_driver_asked(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(memory, "paravirtual", lambda: False)
        monkeypatch.setattr(memory, "NVML", "libnvidia-ml.so.none-in-a-cpu-test")
        for name in memory.VARIABLES:
            monkeypatch.delenv(name, raising=False)
    state = (("_spills", False), ("_native", False), ("_limit", None))
    for name, value in (*state, ("_unread", False), ("_left", memory.HELD_BACK)):
        monkeypatch.setattr(memory, name, value)
    yield
