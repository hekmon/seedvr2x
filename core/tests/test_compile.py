"""--compile, the VAE compiled: the library's side (seedvr2x/compiling.py, peaks.py's range;
DESIGN.md, Memory planner, "torch.compile: the VAE, opt-in, in v0.1.0; the DiT, not", decided on
2026-10-24): the shapes warmed, inductor's deterministic mode kept on through every trace, the
causal convolutions left out of the graph, the order of the load, the cache in the job's
directory. On the CPU: nothing is compiled on a GPU here; a tiny module is compiled on the CPU
where a C++ compiler is there.

The planner's compiled passes, the job's plan, record and resume, and the plan's words are
test_compile_cli.py's (apps/seedvr2x/tests/)."""

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from seedvr2x import compiling, peaks
from seedvr2x.model import COMPUTE_DTYPE, Models
from seedvr2x.passes import Shapes, Tiles
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState


def test_the_range_is_the_1080p_frame_as_a_whole_not_its_area() -> None:
    # The more conservative reading of "the 1080p frame whole" (PROVISIONAL, planner.COMPILE_
    # ENCODE_WHOLE): rows and columns each no larger than the padded 1080p frame's. A portrait
    # frame of the same area is another shape, and so is 1440p.
    inside = peaks.in_compile_range
    assert inside((1104, 1920), "encode", None) and inside((752, 1280), "encode", None)
    assert not inside((1920, 1104), "encode", None) and not inside((1456, 2576), "encode", None)
    assert inside((1920, 1104), "encode", Tiles(1856))
    assert not inside((2192, 3840), "encode", Tiles(2112))
    assert inside((2192, 3840), "decode", Tiles(2048)) and inside(
        (2192, 3840), "decode", Tiles(512)
    )
    assert not inside((2192, 3840), "decode", Tiles(2112))
    assert not inside((1104, 1920), "decode", None)
    assert inside((1104, 1920), "dit", None)


# ----------------------------------------------------------------------------- keep_deterministic


@pytest.fixture
def inductor_state() -> Iterator[Any]:
    """torch's use_deterministic_algorithms and inductor's flag as they were, put back."""
    config: Any = pytest.importorskip("torch._inductor.config")
    function, flag, mode = (
        torch.use_deterministic_algorithms,
        config.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    config.deterministic = False
    yield config
    torch.use_deterministic_algorithms = function
    config.deterministic = flag
    torch.use_deterministic_algorithms(mode)


def test_inductors_deterministic_mode_stays_on_through_torchs_resets(inductor_state: Any) -> None:
    # torch 2.14.1 sets it again at each trace's end, through torch.use_deterministic_algorithms
    # (_dynamo/convert_frame.py:375, with the state before the trace), whose call sets inductor's
    # flag too (torch/__init__.py:1852). Kept: every such call leaves it on, torch's eager mode as
    # asked.
    torch.use_deterministic_algorithms(False)
    assert inductor_state.deterministic is False  # torch's own: the flag follows the call
    restore = compiling.keep_deterministic()
    assert inductor_state.deterministic is True
    torch.use_deterministic_algorithms(False)  # dynamo's reset after a trace
    assert inductor_state.deterministic is True
    assert not torch.are_deterministic_algorithms_enabled()  # the eager operations unchanged
    torch.use_deterministic_algorithms(False, warn_only=True)
    assert inductor_state.deterministic is True
    # Idempotent: a second call wraps nothing more, and its restore does nothing.
    wrapped = torch.use_deterministic_algorithms
    again = compiling.keep_deterministic()
    assert torch.use_deterministic_algorithms is wrapped
    again()
    assert torch.use_deterministic_algorithms is wrapped
    restore()
    assert torch.use_deterministic_algorithms is not wrapped
    torch.use_deterministic_algorithms(False)
    assert inductor_state.deterministic is False


# ----------------------------------------------------------------------------- the exclusion


def tiny_convolution() -> InflatedCausalConv3d:
    return InflatedCausalConv3d(2, 2, kernel_size=1, inflation_mode="tail")


def test_the_causal_convolutions_are_what_leaves_the_graph() -> None:
    # The predicate takes the VAE's causal convolutions and nothing else, a plain Conv3d among the
    # rest; exclusion wraps their forward on the instance and no other module's, and the class's
    # is untouched (numz's attribute, _dynamo_disable, is read by no one: bug 15).
    tree = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        tiny_convolution(),
        torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1), tiny_convolution()),
    )
    found = [compiling.is_causal_convolution(module) for module in tree.modules()]
    assert (
        found.count(True) == 2
        and sum(isinstance(m, InflatedCausalConv3d) for m in tree.modules()) == 2
    )
    assert not compiling.is_causal_convolution(torch.nn.Conv3d(2, 2, 1))
    original = InflatedCausalConv3d.forward
    assert compiling.exclude_causal_convolutions(tree) == 2
    wrapped = [name for name, m in tree.named_modules() if "forward" in vars(m)]
    assert wrapped == ["1", "2.1"]
    assert InflatedCausalConv3d.forward is original


# A tiny module around a causal convolution, compiled on the CPU in a process of its own: a
# compilation imports jinja2, networkx and others into the process (environment.import_inductor),
# which tests of the environment, run after in this one, must not find there. argv: whether the
# convolution is excluded (1 or 0). Prints the values of torch.compiler.is_compiling() seen
# inside the convolution's forward, and how far the compiled output is from plain code's.
COMPILED = """
import json
import sys

import torch
import torch._inductor.config as inductor_config

from seedvr2x import compiling
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

inductor_config.compile_threads = 1


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = torch.nn.Linear(4, 4)
        self.conv = InflatedCausalConv3d(2, 2, kernel_size=1, inflation_mode="tail")

    def forward(self, x):
        y = torch.relu(self.pre(x)).reshape(1, 2, 1, 2, 4)
        return self.conv(y, MemoryState.DISABLED) * 2


torch.manual_seed(0)
module = Tiny().eval()
x = torch.randn(4, 4)
expected = module(x)
seen = []
probed = module.conv.forward


def probe(*arguments, **options):
    seen.append(torch.compiler.is_compiling())
    return probed(*arguments, **options)


module.conv.forward = probe
if sys.argv[1] == "1":
    assert compiling.exclude_causal_convolutions(module) == 1
with torch.no_grad():
    got = torch.compile(module, **compiling.SETTINGS)(x)
compiling.shutdown_workers()
print(json.dumps({"seen": seen, "far": float((got - expected).abs().max())}))
"""


@pytest.mark.parametrize("excluded", [True, False], ids=["excluded", "traced"])
def test_the_causal_convolutions_run_as_plain_code_inside_a_compiled_module(
    tmp_path: Path, excluded: bool
) -> None:
    # Compiled on the CPU, a tiny module around a causal convolution: its forward, wrapped by
    # exclusion, runs eagerly (torch.compiler.is_compiling() is False inside it, where dynamo
    # would have traced it, as it does without the exclusion), and the compiled module gives plain
    # code's values. Skipped without a C++ compiler.
    if shutil.which("g++") is None and shutil.which("c++") is None:
        pytest.skip("no C++ compiler for inductor on the CPU")
    result = subprocess.run(
        [sys.executable, "-c", COMPILED, "1" if excluded else "0"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "cache")},
    )
    found = json.loads(result.stdout.strip().splitlines()[-1])
    assert found["seen"] == ([False] if excluded else [True])
    assert found["far"] < 1e-5


# ----------------------------------------------------------------------------- workers and cache


def test_the_compile_workers_are_shut_through_torchs_own_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch 2.14.1: torch._inductor.async_compile.shutdown_compile_workers, the one its atexit
    # hook calls; it exists, and is the one called.
    module: Any = pytest.importorskip("torch._inductor.async_compile")
    assert callable(module.shutdown_compile_workers)
    called: list[str] = []
    config: Any = pytest.importorskip("torch._inductor.config")
    monkeypatch.setattr(config, "compile_threads", 8)  # put back by monkeypatch
    monkeypatch.setattr(module, "shutdown_compile_workers", lambda: called.append("shut"))
    compiling.shutdown_workers()
    assert called == ["shut"]
    # Any compile after it runs in this process, no pool started again (the review of
    # 2026-10-29, LOW 5): torch asks for a pool when compile_threads is over 1.
    assert config.compile_threads == 1
    monkeypatch.undo()
    monkeypatch.setattr(config, "compile_threads", 8)
    compiling.shutdown_workers()  # no pool started: nothing to do
    assert config.compile_threads == 1


def test_the_compile_threads_are_one_after_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, undo_global_state: None
) -> None:
    # A compile after the warm-up must not bring the workers back (compile_vae's own shutdown is
    # replaced by the Recorder's, so the real one is called here, with torch's shut replaced).
    config: Any = pytest.importorskip("torch._inductor.config")
    monkeypatch.setattr(config, "compile_threads", 8)
    module: Any = pytest.importorskip("torch._inductor.async_compile")
    monkeypatch.setattr(module, "shutdown_compile_workers", lambda: None)
    compiling.shutdown_workers()
    from torch._inductor.async_compile import get_compile_threads  # pyright: ignore

    assert get_compile_threads() == 1


@pytest.fixture
def frames(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """torch._dynamo's count of frames compiled as the test sets it (put back), and the
    compilation module's warm-up mark cleared."""
    utils: Any = pytest.importorskip("torch._dynamo.utils")
    monkeypatch.setitem(utils.counters["frames"], "total", 10)
    monkeypatch.setattr(compiling, "_warmed", [None])
    yield utils.counters["frames"]


def test_a_compilation_after_the_warm_up_is_said_at_warning_naming_the_pass(
    frames: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=compiling.__name__)
    # No warm-up (the VAE not compiled): nothing said, whatever dynamo counts.
    frames["total"] = 99
    assert compiling.report_late("shot 0's encode") == 0 and not caplog.records
    # The warm-up ends at 10 frames compiled.
    frames["total"] = 10
    compiling._warmed[0] = compiling.frames_compiled()  # pyright: ignore[reportPrivateUsage]
    assert compiling.report_late("shot 0's encode") == 0 and not caplog.records
    frames["total"] = 12
    assert compiling.report_late("shot 1's decode") == 2
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert record.getMessage().startswith(
        "shot 1's decode: the compiler compiled, or tried to compile, 2 more graphs"
    )
    # Said once: the next pass compiles nothing more, the one after one frame.
    caplog.clear()
    assert compiling.report_late("shot 2's encode") == 0 and not caplog.records
    frames["total"] = 13
    assert compiling.report_late("shot 2's decode") == 1
    assert "compiled, or tried to compile, 1 more graph after" in caplog.text


def test_the_cache_is_the_jobs_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "users"))
    cache = tmp_path / "job" / "compile-cache"
    compiling.use_cache(cache)
    assert cache.is_dir()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache)
    # torch's own reading of it, which exports the directory for Triton as well.
    from torch._inductor.runtime.cache_dir_utils import cache_dir  # pyright: ignore

    assert cache_dir() == str(cache)


# ----------------------------------------------------------------------------- the warm calls


class FakeVAE(torch.nn.Module):
    """The VAE as the run's own passes (model.encode_sliced, model.decode_sliced) call it, with
    the slicing of the config; _encode and _decode record the shape and memory state of each
    call. Its encoder and decoder are plain modules, compiled by nothing here."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=COMPUTE_DTYPE))
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()
        self.use_slicing = True
        self.slicing_sample_min_size = 4
        self.slicing_latent_min_size = 1
        self.temporal_downsample_factor = 4
        self.calls: list[tuple[str, tuple[int, ...], str]] = []
        # The decoder's inputs with the layout the compiler's guards tell apart: whether the
        # channels are the innermost dimension in memory (the DiT's layout), else channel-major.
        self.decoded: list[tuple[tuple[int, ...], str, bool]] = []

    def _encode(self, x: torch.Tensor, memory_state: MemoryState) -> torch.Tensor:
        self.calls.append(("encode", tuple(x.shape), memory_state.name))
        return torch.zeros(1, 32, 1, 1, 1, dtype=x.dtype)

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> list[torch.Tensor]:
        self.calls.append(("decode", tuple(z.shape), memory_state.name))
        self.decoded.append((tuple(z.shape), memory_state.name, z.stride(1) == 1))
        return [torch.zeros(1, dtype=z.dtype)]


def fake_models(vae: FakeVAE | None = None) -> Models:
    config = OmegaConf.create({"vae": {"scaling_factor": 0.9152}})
    runner = SimpleNamespace(vae=vae or FakeVAE(), config=config)
    return Models(runner, torch.zeros(1), torch.zeros(1), torch.device("cpu"))


def test_every_shape_and_slice_kind_is_warmed() -> None:
    # The kinds as the run meets them: a shot of 1 latent is one slice of 1 frame, of 2 one of 5
    # frames (neither with a memory), of 3 a first slice of 5 frames and a next one of 4; the
    # decode's the first 2 latents then 1 (DESIGN.md: "5 then 4 a slice"; "2 latents then 1").
    fake = FakeVAE()
    shapes = Shapes((128, 192), ((64, 128), (128, 128)), ((64, 64),), (1, 2, 3))
    compiling.warm(fake_models(fake), shapes)
    disabled, initial, active = "DISABLED", "INITIALIZING", "ACTIVE"
    expected: list[tuple[str, tuple[int, ...], str]] = []
    for rows, columns in shapes.encode:
        expected += [("encode", (1, 3, 1, rows, columns), disabled)]
        expected += [("encode", (1, 3, 5, rows, columns), disabled)]
        expected += [("encode", (1, 3, 5, rows, columns), initial)]
        expected += [("encode", (1, 3, 4, rows, columns), active)]
    # The shapes of a shot of 1 latent come first, then 2, then 3: the encode's order within.
    ordered = sorted(
        (call for call in fake.calls if call[0] == "encode"), key=lambda call: call[1:2]
    )
    assert sorted(call for call in fake.calls if call[0] == "encode") == sorted(expected)
    assert ordered
    decodes = [call for call in fake.calls if call[0] == "decode"]
    # 1 latent: one call of 1; 2 latents: one of 2; 3: the first 2 latents (a first slice of 1 +
    # 1) then 1; a latent 8 px per side: the tile's 64x64 is 8x8 latents.
    assert decodes == [
        ("decode", (1, 16, 1, 8, 8), "DISABLED"),
        ("decode", (1, 16, 2, 8, 8), "DISABLED"),
        ("decode", (1, 16, 2, 8, 8), "INITIALIZING"),
        ("decode", (1, 16, 1, 8, 8), "ACTIVE"),
    ]


def test_only_the_kinds_the_job_has_are_warmed() -> None:
    fake = FakeVAE()
    shapes = Shapes((128, 128), ((128, 128),), ((64, 64),), (3,))
    compiling.warm(fake_models(fake), shapes)
    assert [call[1:] for call in fake.calls] == [
        ((1, 3, 5, 128, 128), "INITIALIZING"),
        ((1, 3, 4, 128, 128), "ACTIVE"),
        ((1, 16, 2, 8, 8), "INITIALIZING"),
        ((1, 16, 1, 8, 8), "ACTIVE"),
    ]


def test_the_warm_up_latent_is_dense_channels_last() -> None:
    # What the warm-up's latent is to the decoder: the DiT's (T', h, w, 16), contiguous, its
    # channel stride 1, where the encode's (model.to_latent) is a permuted view of (16, T', h, w).
    dense = compiling._latent(3, 4, 4)  # pyright: ignore[reportPrivateUsage]
    channel_major = torch.zeros((16, 3, 4, 4), dtype=COMPUTE_DTYPE).permute(1, 2, 3, 0)
    assert dense.shape == channel_major.shape == (3, 4, 4, 16)
    assert dense.is_contiguous() and not channel_major.is_contiguous()
    assert dense.stride(3) == 1 and channel_major.stride(3) > 1


@pytest.fixture
def undo_global_state(inductor_state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "unset")
    for name in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "fail_on_recompile_limit_hit",
        "suppress_errors",
    ):
        monkeypatch.setattr(torch._dynamo.config, name, getattr(torch._dynamo.config, name))


def test_the_vae_leaving_the_card_comes_for_the_warm_calls_and_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A plan with the VAE's weights away from the card in the DiT's phase has them in their
    # pinned copy at the load: brought to the device for the warm calls as Models.enter brings
    # them, and sent away after the device's work, the allocator trimmed (blockswap.Cuda).
    events: list[str] = []

    class Part:
        device = None

        def enter(self, backend: Any) -> None:
            events.append("enter")
            self.device = "on"

        def leave(self) -> None:
            events.append("leave")
            self.device = None

    class Backend:
        def __init__(self, device: Any) -> None:
            pass

        def compute(self) -> str:
            return "stream"

        def wait(self, *streams: Any) -> None:
            events.append("wait")

        def trim(self) -> None:
            events.append("trim")

    monkeypatch.setattr(compiling.blockswap, "Cuda", Backend)
    models = fake_models()
    models.vae = Part()  # pyright: ignore[reportAttributeAccessIssue]
    models.device = SimpleNamespace(type="cuda")  # pyright: ignore[reportAttributeAccessIssue]
    with compiling._on_the_device(models):  # pyright: ignore[reportPrivateUsage]
        events.append("work")
    assert events == ["enter", "work", "wait", "leave", "trim"]
    # Weights that never leave need nothing.
    events.clear()
    models.vae = None
    with compiling._on_the_device(models):  # pyright: ignore[reportPrivateUsage]
        events.append("work")
    assert events == ["work"]
