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

from seedvr2x import clock, memory, peaks, waiting
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
        monkeypatch.setattr(memory, "_torch_loaded", lambda: False)  # as a run's start finds it
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


@pytest.fixture(autouse=True)
def no_card_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No card marked failed or abandoned before a test (waiting.fail, waiting.abandoned): the
    marks are the process's, and a test that raises a CUDA error on purpose marks card 0, which
    the stand-ins' CPU device is (model.card_index), for every test after it. Nor any reason for
    an unread card said yet (memory._unread), said once a process at INFO."""
    monkeypatch.setattr(waiting, "_failed", {})
    monkeypatch.setattr(waiting, "_abandoned", set())
    monkeypatch.setattr(waiting, "_pending", {})
    monkeypatch.setattr(memory, "_unread_said", set())


@pytest.fixture
def plain_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DiT's fit of the plain attention, before seedvr2x's lean one (patch of 2026-11-06):
    127.16 KiB a token and 2.726 MiB a window, FlashAttention 2's fit, which these runs
    measured (SDPA's plain fit was 4.69% more a token and 0.141 GiB; FlashAttention 2 went on
    2026-11-08 and the tests keep the figures of the run they held, SDPA's lean fit being the
    shipped one). For the tests that hold plans or peaks a run measured on the
    plain code (numz's recipes, the first matrices, the windowed shots): the planner's arithmetic
    is held to those runs at the fit of the code they ran, the lean fit to its own measures
    (LEAN_WINDOWS)."""
    for name in ("PER_TOKEN", "INT8_PER_TOKEN", "GGUF_PER_TOKEN"):
        monkeypatch.setattr(peaks, name, 127.16 * 1024)
    for name in ("PER_WINDOW", "INT8_PER_WINDOW", "GGUF_PER_WINDOW"):
        monkeypatch.setattr(peaks, name, 2.726 * 1024**2)
    monkeypatch.setattr(peaks, "DIT_ATTENTION", 0)


# The DiT's windows on the lean attention, each the allocated peak of one forward in bytes (the
# figures of research/docs/small-cards.md, The lean attention's peak, in bytes:
# hold_probe.py, an RTX 5090, 2026-11-06, one sampling step, the sharp 7B fp16), in
# seedvr2x's padding: the padded frame's rows (1104 for 1080p, 2192 for 4K), the window's
# latents, the DiT's blocks swapped, whether the VAE's weights were on the GPU (all were).
# FlashAttention 2 and SDPA peaked at the same figure at every one.
LEAN_WINDOWS: list[tuple[int, int, int, bool, int]] = [
    (1104, 1, 0, True, 17_677_632_986),
    (1104, 1, 36, True, 2_271_102_426),
    (1104, 3, 0, True, 18_700_522_394),
    (1104, 3, 36, True, 3_293_991_834),
    (1104, 5, 0, True, 19_739_284_754),
    (1104, 5, 36, True, 4_332_754_194),
    (1104, 7, 0, True, 20_808_704_114),
    (1104, 7, 36, True, 5_402_173_554),
    (1104, 9, 36, True, 6_441_170_634),
    (1104, 11, 36, True, 7_510_887_714),
    (1104, 13, 36, True, 8_518_121_154),
    (1104, 15, 36, True, 9_587_384_994),
    (1104, 17, 36, True, 10_594_619_154),
    (1104, 19, 36, True, 11_665_064_514),
    (2192, 1, 0, True, 19_214_845_794),
    (2192, 1, 36, True, 3_808_315_234),
    (2192, 2, 36, True, 5_974_921_634),
    (2192, 3, 36, True, 8_141_528_034),
    (2192, 4, 36, True, 10_308_134_434),
]
