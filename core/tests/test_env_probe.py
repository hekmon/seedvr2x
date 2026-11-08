"""The environment probe (seedvr2x/probe.py), on the CPU: the fingerprint's parts, read from
stand-ins for torch's and the driver's readings; the cache (written, read, skipping the checks on
a match, run again for each part changed, a missing check added, a file damaged ignored and
written again, the write atomic); each check run on the CPU device against the CPU, to exercise
its code path (what is equal, trivially, passes); a kernel made wrong on purpose, refused with its
facts and the refusal cached; the platform's cache directory for each platform. `pytest -m gpu`:
the real checks on CUDA for each DiT layout, printing every error against its tolerance (the
PROVISIONAL bounds of matmul and attention are refined from those figures)."""

import ctypes
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from seedvr2x import int8, kquant, memory, probe, weights
from seedvr2x.probe import ProbeRefusal

CPU = torch.device("cpu")
PARTS = {
    "seedvr2x": "0.1.0",
    "code": "c0de",
    "torch": "2.13.0+cu130",
    "cuda": "13.0",
    "driver": "580.82.07",
    "capability": "12.0",
    "allocator": "cudaMallocAsync",
    "gpu": "A Stand-in GPU",
    "alloc_conf": "unset",
    "flash_attn": "none",
}


@pytest.fixture(autouse=True)
def info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="seedvr2x.probe")


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "probe.json"


@pytest.fixture
def stand_in_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The fingerprint replaced by PARTS' copy, which a test changes a part of."""
    parts = dict(PARTS)
    monkeypatch.setattr(probe, "fingerprint", lambda device: dict(parts))
    return parts


class Ran:
    """probe._run recorded: the kernels each check that ran makes results for."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.names: list[str] = []
        original = probe._run  # pyright: ignore[reportPrivateUsage]

        def run(names: tuple[str, ...], call: Callable[[], list[probe.Result]]) -> Any:
            self.names.extend(names)
            return original(names, call)

        monkeypatch.setattr(probe, "_run", run)


def sdpa(architecture: weights.Architecture = weights.DIT_7B, path: Path | None = None) -> Any:
    return probe.check(CPU, architecture, "sdpa", path=path)


def test_importing_the_probe_imports_no_torch() -> None:
    code = "import sys, seedvr2x.probe; sys.exit('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --- the fingerprint ---


class CudaReadings:
    """torch's readings of a card, replaced."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "A Stand-in GPU")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
        monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        monkeypatch.setattr(probe, "driver_version", lambda: "580.82.07")
        monkeypatch.setattr(probe, "library_code_sha256", lambda: "c0de")
        monkeypatch.setattr(probe, "_version", lambda name: {"seedvr2x-core": "0.1.0"}.get(name))


def test_the_fingerprint_s_parts_on_a_card(monkeypatch: pytest.MonkeyPatch) -> None:
    CudaReadings(monkeypatch)
    for name in memory.VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert probe.fingerprint(torch.device("cuda:0")) == PARTS
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setattr(probe, "_version", lambda name: "2.8.0" if name == "flash-attn" else None)
    parts = probe.fingerprint(torch.device("cuda:0"))
    assert parts["alloc_conf"] == (
        "PYTORCH_CUDA_ALLOC_CONF=backend:native;PYTORCH_ALLOC_CONF=expandable_segments:True"
    )
    assert parts["flash_attn"] == "2.8.0" and parts["seedvr2x"] == probe.UNKNOWN


def test_a_part_that_cannot_be_read_is_unknown_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    CudaReadings(monkeypatch)
    monkeypatch.setattr(probe, "driver_version", lambda: None)
    parts = probe.fingerprint(torch.device("cuda:0"))
    assert parts["driver"] == "unknown"
    assert "the environment's driver is unknown" in caplog.text


def test_the_fingerprint_without_a_card() -> None:
    parts = probe.fingerprint(CPU)
    assert parts["gpu"] == "cpu" and parts["driver"] == parts["capability"] == "none"
    assert parts["allocator"] == "none" and parts["torch"] == torch.__version__
    assert parts["cuda"] == (torch.version.cuda or "none")


def test_the_library_s_code_is_hashed_by_its_files() -> None:
    import hashlib

    package = Path(probe.__file__).resolve().parent
    keys = {
        f"seedvr2x/{path.relative_to(package).as_posix()}": path
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode() + b"\0" + keys[key].read_bytes())
    assert probe.library_code_sha256() == digest.hexdigest()


def test_the_digest_is_stable_and_follows_each_part() -> None:
    assert probe.digest(PARTS) == probe.digest(dict(reversed(PARTS.items())))
    assert len(probe.digest(PARTS)) == 64
    digests = {probe.digest({**PARTS, name: PARTS[name] + "x"}) for name in PARTS}
    assert len(digests) == len(PARTS) and probe.digest(PARTS) not in digests


class NVML:
    """libnvidia-ml / nvml.dll as driver_version calls it."""

    def __init__(self, init: int = 0, query: int = 0) -> None:
        self.init, self.query, self.shut = init, query, False

    def nvmlInit_v2(self) -> int:
        return self.init

    def nvmlSystemGetDriverVersion(
        self, version: ctypes.Array[ctypes.c_char], length: ctypes.c_uint
    ) -> int:
        version.value = b"581.15"
        return self.query

    def nvmlShutdown(self) -> int:
        self.shut = True
        return 0


def test_the_driver_is_read_from_nvml_on_linux_and_windows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked: list[str] = []

    def loader(libraries: dict[str, Any]) -> Callable[[str], Any]:
        def load(name: str) -> Any:
            asked.append(name)
            if name not in libraries:
                raise OSError(f"{name}: cannot open shared object file")
            return libraries[name]

        return load

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(memory, "NVML", "libnvidia-ml.so.1")
    nvml = NVML()
    monkeypatch.setattr(ctypes, "CDLL", loader({"libnvidia-ml.so.1": nvml}))
    assert probe.driver_version() == "581.15" and nvml.shut and asked == ["libnvidia-ml.so.1"]
    # Windows: nvml.dll first, then the NVSMI directory's.
    monkeypatch.setattr(sys, "platform", "win32")
    asked.clear()
    nvsmi = NVML()
    monkeypatch.setattr(ctypes, "CDLL", loader({probe.NVML_WINDOWS[1]: nvsmi}))
    assert probe.driver_version() == "581.15" and asked == list(probe.NVML_WINDOWS)
    # An initialisation or a query that fails is unknown, logged, with the library shut where it
    # was initialised.
    for nvml in (NVML(init=9), NVML(query=3)):
        monkeypatch.setattr(ctypes, "CDLL", lambda name, nvml=nvml: nvml)
        assert probe.driver_version() is None
        assert nvml.shut == (nvml.init == 0)
    monkeypatch.setattr(ctypes, "CDLL", loader({}))
    assert probe.driver_version() is None
    assert caplog.text.count("the NVIDIA driver's version is unknown") == 3
    # macOS has no NVIDIA driver: nothing asked, nothing said.
    monkeypatch.setattr(sys, "platform", "darwin")
    caplog.clear()
    asked.clear()
    assert probe.driver_version() is None and not asked and not caplog.text


# --- the cache's place ---


def test_the_cache_directory_of_each_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert probe.cache_directory() == tmp_path / "home" / ".cache" / "seedvr2x"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert probe.cache_directory() == tmp_path / "xdg" / "seedvr2x"
    # The XDG specification has a relative path ignored.
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    assert probe.cache_directory() == tmp_path / "home" / ".cache" / "seedvr2x"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert probe.cache_directory() == tmp_path / "home" / "Library" / "Caches" / "seedvr2x"
    monkeypatch.setattr(sys, "platform", "win32")
    assert probe.cache_directory() == tmp_path / "home" / "AppData" / "Local" / "seedvr2x"
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert probe.cache_directory() == tmp_path / "local" / "seedvr2x"
    assert probe.cache_path() == tmp_path / "local" / "seedvr2x" / "probe.json"


# --- the checks, on the CPU ---


@pytest.mark.parametrize(
    ("architecture", "names"),
    [
        (weights.DIT_7B, ("matmul bfloat16", "attention sdpa")),
        (
            weights.DIT_7B_INT8,
            ("matmul bfloat16", "attention sdpa", "int8 product", "int8 codes", "int8 linear"),
        ),
        (
            weights.DIT_7B_DYN,
            (
                "matmul bfloat16",
                "attention sdpa",
                "dequantise Q3_K",
                "dequantise Q4_K",
                "dequantise Q5_K",
            ),
        ),
    ],
    ids=["float16", "int8", "gguf"],
)
def test_each_layout_s_checks_run_and_pass_on_the_cpu(
    architecture: weights.Architecture, names: tuple[str, ...], cache: Path
) -> None:
    assert probe.kernels(architecture, "sdpa") == names
    probed = sdpa(architecture, cache)
    assert [result.kernel for result in probed.results] == list(names)
    assert not probed.cached and probed.path == cache
    for result in probed.results:
        assert result.passed and result.error is not None and result.error <= result.tolerance
    # The exact checks are exact: the same code on both sides.
    exact = {r.kernel: r for r in probed.results if r.tolerance == 0.0 or "codes" in r.kernel}
    assert all(result.error == 0.0 for result in exact.values())


def test_a_dynamic_file_that_holds_one_type_checks_that_type_alone() -> None:
    only = weights.Architecture(
        "a stand-in", 1, "x", None, layout=weights.GGUF, types=("33333333", "33333333")
    )
    assert probe.kernels(only, "sdpa")[2:] == ("dequantise Q3_K",)
    assert probe.kernels(weights.DIT_3B, "flash_attn_2") == (
        "matmul bfloat16",
        "attention flash_attn_2",
    )


def test_an_attention_mode_that_does_not_exist_fails_the_check_and_is_cached(cache: Path) -> None:
    # The vendored attention raises a ValueError for another mode: a failed check, not a crash.
    with pytest.raises(ProbeRefusal) as refusal:
        probe.check(CPU, weights.DIT_7B, "sage", path=cache)
    assert refusal.value.kernel == "attention sage" and refusal.value.error is None
    assert "does not run" in str(refusal.value) and "attention_mode" in str(refusal.value)


# --- the cache ---


def test_a_first_run_checks_and_writes_the_cache_and_a_second_is_vouched_for(
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ran = Ran(monkeypatch)
    first = sdpa(path=cache)
    assert not first.cached and ran.names == ["matmul bfloat16", "attention sdpa"]
    assert f"kept in {cache}" in caplog.text and "checked matmul bfloat16" in caplog.text
    document = json.loads(cache.read_text())
    assert document["format"] == probe.FORMAT
    environment = document["environments"][probe.digest(PARTS)]
    assert environment["fingerprint"] == PARTS
    assert set(environment["checks"]) == {"matmul bfloat16", "attention sdpa"}
    one = environment["checks"]["matmul bfloat16"]
    assert set(one) == {"error", "tolerance", "unit", "passed", "at", "detail"}
    assert one["passed"] is True and one["at"].endswith("Z")
    # The place for the allocator's margin: there, null, and nothing measures it.
    assert environment["allocator_margin"] is None and first.allocator_margin is None
    ran.names.clear()
    caplog.clear()
    second = sdpa(path=cache)
    assert second.cached and ran.names == [] and second.results == first.results
    assert second.digest == probe.digest(PARTS) and second.allocator_margin is None
    assert "the cache vouches for matmul bfloat16, attention sdpa" in caplog.text
    assert str(cache) in caplog.text
    # No temporary file left beside the cache.
    assert [path.name for path in cache.parent.iterdir()] == ["probe.json"]


@pytest.mark.parametrize("part", list(PARTS))
def test_each_part_changed_runs_the_checks_again_and_keeps_the_other_environment(
    part: str,
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
) -> None:
    ran = Ran(monkeypatch)
    before = probe.digest(PARTS)
    sdpa(path=cache)
    ran.names.clear()
    stand_in_environment[part] += "-changed"
    again = sdpa(path=cache)
    assert not again.cached and ran.names == ["matmul bfloat16", "attention sdpa"]
    assert again.digest != before
    assert set(json.loads(cache.read_text())["environments"]) == {before, again.digest}
    # Back to the first environment: the cache still knows it.
    stand_in_environment[part] = PARTS[part]
    ran.names.clear()
    assert sdpa(path=cache).cached and ran.names == []


def test_a_check_the_cache_lacks_is_run_and_added(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    ran = Ran(monkeypatch)
    sdpa(path=cache)
    ran.names.clear()
    probed = sdpa(weights.DIT_7B_INT8, cache)
    assert ran.names == ["int8 product", "int8 codes", "int8 linear"] and not probed.cached
    checks = json.loads(cache.read_text())["environments"][probe.digest(PARTS)]["checks"]
    assert len(checks) == 5
    # The first layout is vouched for still, and so is the second now.
    ran.names.clear()
    assert sdpa(path=cache).cached and sdpa(weights.DIT_7B_INT8, cache).cached
    # Another attention kernel is another check.
    probe.check(CPU, weights.DIT_7B, "sdpa", path=cache)
    assert ran.names == []


def test_the_margin_a_future_measure_wrote_is_kept_when_checks_are_added(
    stand_in_environment: dict[str, str], cache: Path
) -> None:
    sdpa(path=cache)
    document = json.loads(cache.read_text())
    document["environments"][probe.digest(PARTS)]["allocator_margin"] = 0.5
    cache.write_text(json.dumps(document))
    assert sdpa(path=cache).allocator_margin == 0.5
    assert sdpa(weights.DIT_7B_INT8, cache).allocator_margin == 0.5
    stored = json.loads(cache.read_text())["environments"][probe.digest(PARTS)]
    assert stored["allocator_margin"] == 0.5


@pytest.mark.parametrize(("margin", "over"), [(0.1, False), (0.33, False), (0.5, True)])
def test_a_cached_margin_counts_in_the_plan_where_it_is_over_the_rules_share(
    stand_in_environment: dict[str, str], cache: Path, margin: float, over: bool
) -> None:
    # The reading side of the allocator's margin (the measurement is not built): the cache's
    # value rides on the reading (Reading.margin) and peaks.reserve counts the larger of it and
    # the rule's share, here the segments' DiT share 0.33. Written into the cache by hand.
    from dataclasses import replace

    from seedvr2x import peaks

    sdpa(path=cache)
    document = json.loads(cache.read_text())
    document["environments"][probe.digest(PARTS)]["allocator_margin"] = margin
    cache.write_text(json.dumps(document))
    found = sdpa(path=cache).allocator_margin
    assert found == margin
    held = memory.Reading(30 * peaks.GIB, None, 24 * peaks.GIB, True, 3 * peaks.GIB)
    counted = replace(held, margin=found)
    working = 10 * peaks.GIB
    rule = peaks.reserve(peaks.DIT_PHASE, working, held.allocator)
    taken = peaks.reserve(peaks.DIT_PHASE, working, counted.allocator, counted.margin)
    assert rule == round(0.33 * working)
    assert (taken > rule) is over and taken == max(rule, round(margin * working))


def damaged_results(document: dict[str, Any]) -> str:
    """One check's result with a field of the wrong type."""
    environment = next(iter(document["environments"].values()))
    environment["checks"]["matmul bfloat16"]["passed"] = "yes"
    return json.dumps(document)


@pytest.mark.parametrize(
    ("damage", "said"),
    [
        (lambda document: "{ not json", "is not JSON"),
        (lambda document: json.dumps([1, 2]), "of another format"),
        (lambda document: json.dumps({**document, "format": 0}), "of another format"),
        (lambda document: json.dumps({**document, "environments": []}), "holds no environments"),
        (damaged_results, "is damaged"),
    ],
    ids=["text", "list", "format", "environments", "result"],
)
def test_a_damaged_cache_is_ignored_logged_and_written_again(
    damage: Callable[[dict[str, Any]], str],
    said: str,
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ran = Ran(monkeypatch)
    sdpa(path=cache)
    ran.names.clear()
    cache.write_text(damage(json.loads(cache.read_text())))
    caplog.clear()
    assert not sdpa(path=cache).cached
    assert said in caplog.text and str(cache) in caplog.text
    assert ran.names == ["matmul bfloat16", "attention sdpa"]
    assert json.loads(cache.read_text())["environments"][probe.digest(PARTS)]
    ran.names.clear()
    assert sdpa(path=cache).cached


def test_an_unreadable_cache_is_ignored_and_logged(
    stand_in_environment: dict[str, str], cache: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"\xff\xfe\x00 not utf-8 \x80")
    assert not sdpa(path=cache).cached
    assert "is unreadable" in caplog.text
    assert sdpa(path=cache).cached


def test_the_write_is_atomic(
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdpa(path=cache)
    before = cache.read_bytes()

    def broken(source: Any, target: Any) -> None:
        # The new file is complete beside the old, which is untouched.
        assert json.loads(Path(source).read_text())["environments"]
        assert cache.read_bytes() == before
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    # Another layout, so that something is written.
    probed = sdpa(weights.DIT_7B_INT8, cache)
    assert not probed.cached and "could not be written: disk full" in caplog.text
    assert cache.read_bytes() == before
    assert [path.name for path in cache.parent.iterdir()] == ["probe.json"]


def test_a_directory_that_cannot_be_made_is_logged_and_the_checks_stand(
    stand_in_environment: dict[str, str], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "a file"
    blocked.write_text("")
    probed = sdpa(path=blocked / "probe.json")
    assert not probed.cached and all(result.passed for result in probed.results)
    assert "could not be written" in caplog.text


def test_the_oldest_environments_are_dropped(
    stand_in_environment: dict[str, str], cache: Path
) -> None:
    for number in range(probe.ENVIRONMENTS + 2):
        stand_in_environment["gpu"] = f"gpu {number}"
        sdpa(path=cache)
    kept = json.loads(cache.read_text())["environments"]
    assert len(kept) == probe.ENVIRONMENTS
    assert [entry["fingerprint"]["gpu"] for entry in kept.values()][-1] == (
        f"gpu {probe.ENVIRONMENTS + 1}"
    )
    assert "gpu 0" not in {entry["fingerprint"]["gpu"] for entry in kept.values()}


# --- a kernel gone wrong ---


def refused(architecture: weights.Architecture, cache: Path, cached: bool = False) -> ProbeRefusal:
    with pytest.raises(ProbeRefusal) as refusal:
        sdpa(architecture, cache)
    assert refusal.value.cached is cached
    return refusal.value


def test_a_wrong_matrix_product_is_refused_with_its_facts_and_the_refusal_is_cached(
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = torch.Tensor.__matmul__

    def wrong(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        out = original(self, other)
        return out * 1.1 if self.dtype == torch.bfloat16 else out

    monkeypatch.setattr(torch.Tensor, "__matmul__", wrong)
    refusal = refused(weights.DIT_7B, cache)
    assert refusal.kind == probe.KIND == "probe" and isinstance(refusal, ValueError)
    assert refusal.kernel == "matmul bfloat16" and refusal.tolerance == 1.0
    assert refusal.error is not None and refusal.error > 1.0
    assert refusal.fingerprint == PARTS and refusal.path == cache
    message = str(refusal)
    assert "matmul bfloat16 gives the wrong numbers on A Stand-in GPU" in message
    assert "torch 2.13.0+cu130" in message and "driver 580.82.07" in message
    assert f"{refusal.error:.4g}" in message and "1 allowed" in message
    assert (
        str(cache) in message and "delete that file" in message and "measured before" not in message
    )
    assert "--" not in message  # no option of any application
    assert (
        refusal.reason in message
        and "A Stand-in GPU" in refusal.reason
        and cache.name in refusal.way
    )
    # Cached: a second run, the fault gone, refuses at once without running anything.
    monkeypatch.undo()
    monkeypatch.setattr(probe, "fingerprint", lambda device: dict(PARTS))
    ran = Ran(monkeypatch)
    caplog.clear()
    again = refused(weights.DIT_7B, cache, cached=True)
    assert ran.names == [] and again.result == refusal.result
    assert f"It was measured before, at {refusal.result.at} UTC" in str(again)
    assert str(cache) in str(again) and "holds a failure" in caplog.text
    # And another layout needing that kernel refuses the same.
    assert refused(weights.DIT_7B_INT8, cache, cached=True).kernel == "matmul bfloat16"
    # The way out the message names: the file deleted, the kernel right now, the checks pass.
    cache.unlink()
    assert not sdpa(path=cache).cached


def test_a_wrong_attention_is_refused(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    original = torch.nn.functional.scaled_dot_product_attention
    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        lambda *args, **kwargs: original(*args, **kwargs) * 1.5,  # pyright: ignore[reportUnknownLambdaType]
    )
    refusal = refused(weights.DIT_7B, cache)
    assert refusal.kernel == "attention sdpa" and refusal.error is not None
    assert refusal.tolerance == probe.ATTENTION_RELATIVE and refusal.error > refusal.tolerance


def first_call_wrong(
    monkeypatch: pytest.MonkeyPatch, module: Any, name: str, wrong: Callable[[Any], Any]
) -> None:
    """module.name's first call, the device's, gives wrong; the others, the CPU's, are right."""
    original = getattr(module, name)
    calls: list[int] = []

    def replaced(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        out = original(*args, **kwargs)
        return wrong(out) if len(calls) == 1 else out

    monkeypatch.setattr(module, name, replaced)


def test_a_wrong_integer_product_is_refused(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    first_call_wrong(monkeypatch, int8, "product", lambda out: out + 1)
    refusal = refused(weights.DIT_7B_INT8, cache)
    assert refusal.kernel == "int8 product" and refusal.tolerance == 0.0
    assert refusal.error == 8 * 1536 and "elements apart" in str(refusal)


def test_wrong_codes_are_refused(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    def wrong(out: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        codes, scales = out
        return codes.clamp(-60, 60), scales

    first_call_wrong(monkeypatch, int8, "quantize", wrong)
    refusal = refused(weights.DIT_7B_INT8, cache)
    assert refusal.kernel == "int8 codes" and refusal.error is not None
    assert (
        refusal.error > refusal.tolerance == 2 + 4
    )  # 8 rows: 2 at the least, 300: 460,800 // 100,000


def test_a_wrong_int8_output_is_refused(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    first_call_wrong(monkeypatch, int8, "linear", lambda out: out * 1.1)
    refusal = refused(weights.DIT_7B_INT8, cache)
    assert refusal.kernel == "int8 linear" and refusal.error is not None and refusal.error > 1.0


@pytest.mark.parametrize("qtype", ["Q3_K", "Q4_K", "Q5_K"])
def test_a_wrong_dequantisation_is_refused_by_a_single_value(
    qtype: str, stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    def wrong(out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out.view(-1)[5] += 1
        return out

    # The first call of the type under test: let the others' pass.
    original = kquant.dequantise
    seen: list[str] = []

    def replaced(
        raw: torch.Tensor, kind: str, shape: Any, dtype: torch.dtype = torch.float32
    ) -> Any:
        out = original(raw, kind, shape, dtype)
        if kind == qtype and not seen:
            seen.append(kind)
            return wrong(out)
        return out

    monkeypatch.setattr(kquant, "dequantise", replaced)
    refusal = refused(weights.DIT_7B_DYN, cache)
    assert refusal.kernel == f"dequantise {qtype}" and refusal.error == 1.0
    assert refusal.tolerance == 0.0


def test_a_kernel_that_raises_is_a_failed_check_with_its_message_cached(
    stand_in_environment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken(*args: Any) -> Any:
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(int8, "product", broken)
    refusal = refused(weights.DIT_7B_INT8, cache)
    assert refusal.kernel == "int8 product" and refusal.error is None
    assert "does not run on A Stand-in GPU" in str(refusal)
    assert "RuntimeError: no kernel image is available" in str(refusal)
    stored = json.loads(cache.read_text())["environments"][probe.digest(PARTS)]["checks"]
    assert [stored[name]["passed"] for name in ("int8 product", "int8 codes", "int8 linear")] == [
        False
    ] * 3
    assert (
        stored["int8 product"]["error"] is None
        and "no kernel image" in stored["int8 product"]["detail"]
    )
    assert "raised" in caplog.text
    # Read back from the file: the same refusal, said measured before.
    monkeypatch.undo()
    monkeypatch.setattr(probe, "fingerprint", lambda device: dict(PARTS))
    assert refused(weights.DIT_7B_INT8, cache, cached=True).result.error is None


def test_an_out_of_memory_error_says_nothing_of_the_kernel_and_is_not_cached(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    def full(*args: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(int8, "product", full)
    with pytest.raises(torch.OutOfMemoryError):
        sdpa(weights.DIT_7B_INT8, cache)
    assert not cache.exists()


def test_a_cuda_error_says_the_card_failed_and_is_not_cached(
    stand_in_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, cache: Path
) -> None:
    # A CUDA error other than out of memory says nothing of the kernel either, but that the card
    # failed: cached as a failed check, it would refuse the environment for good.
    def failed(*args: Any) -> Any:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(int8, "product", failed)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        sdpa(weights.DIT_7B_INT8, cache)
    assert not cache.exists()


# --- the real card ---


@pytest.mark.gpu
@pytest.mark.parametrize(
    "architecture",
    [weights.DIT_7B, weights.DIT_7B_INT8, weights.DIT_7B_DYN],
    ids=["fp16", "int8", "gguf"],
)
def test_the_checks_on_cuda_pass_for_each_layout(
    architecture: weights.Architecture, tmp_path: Path
) -> None:
    # The real kernels, each error against its tolerance, printed: the PROVISIONAL bounds of the
    # matrix product and the attention (probe.py) are refined from these figures on each card.
    from seedvr2x import model

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda:0")
    path = tmp_path / "probe.json"
    probed = probe.check(device, architecture, model.attention_backend(), path=path)
    for result in probed.results:
        print(result.said())
        assert result.passed
    print(json.dumps(probed.fingerprint, indent=2))
    assert probe.check(device, architecture, model.attention_backend(), path=path).cached
    if model.attention_backend() == "sdpa":
        return
    # FlashAttention 2 installed: PyTorch's SDPA is checked too, the other kernel of the pair.
    assert all(r.passed for r in probe.check(device, architecture, "sdpa", path=path).results)
