"""The vendored model imports and builds without numz's runtime, on the CPU; the NVIDIA
driver's version is read from NVML."""

import ctypes
import logging
import subprocess
import sys

import pytest

# A fresh interpreter: importing must print nothing and create no CUDA context, where numz's
# runtime printed banners, patched libraries and probed the GPU at import (provenance.md).
IMPORT_ALL = """
import importlib, pathlib
import torch
import seedvr2x.vendor as vendor

root = pathlib.Path(vendor.__file__).parent
for path in sorted(root.rglob("*.py")):
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    importlib.import_module(".".join(("seedvr2x.vendor", *parts)))
assert not torch.cuda.is_initialized(), "a CUDA context was created at import"
"""

# The models built from the vendored configs on the meta device: the config loader resolves
# numz's registry keys to the vendored classes, and the constructors run.
BUILD = """
import pathlib
import torch
from omegaconf import OmegaConf
import seedvr2x.vendor
from seedvr2x.vendor.common.config import create_object, load_config
from seedvr2x.vendor.models.dit_7b.attention import FlashAttentionVarlen

VENDOR = pathlib.Path(seedvr2x.vendor.__file__).parent
for size in ("7b", "3b"):
    config = load_config(str(VENDOR / f"configs_{size}" / "main.yaml"))
    with torch.device("meta"):
        dit = create_object(config.dit.model)
    assert type(dit).__module__ == f"seedvr2x.vendor.models.dit_{size}.nadit", type(dit)
    if size == "7b":
        attention = [m for m in dit.modules() if isinstance(m, FlashAttentionVarlen)]
        assert len(attention) == 36, len(attention)
vae_yaml = VENDOR / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
config.vae.model = OmegaConf.merge(config.vae.model, load_config(str(vae_yaml)))
with torch.device("meta"):
    vae = create_object(config.vae.model)
assert type(vae).__name__ == "VideoAutoencoderKLWrapper"
"""


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_vendor_imports_quietly() -> None:
    result = run(IMPORT_ALL)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_models_build_from_configs() -> None:
    result = run(BUILD)
    assert result.returncode == 0, result.stderr


class NVML:
    """libnvidia-ml as nvidia_driver calls it, each call answering `status`."""

    def __init__(self, init: int = 0, query: int = 0) -> None:
        self.init, self.query, self.shut = init, query, False

    def nvmlInit_v2(self) -> int:
        return self.init

    def nvmlSystemGetDriverVersion(
        self, version: ctypes.Array[ctypes.c_char], length: ctypes.c_uint
    ) -> int:
        assert length.value == len(version) == 80
        version.value = b"580.82.07"
        return self.query

    def nvmlShutdown(self) -> int:
        self.shut = True
        return 0


def test_nvidia_driver(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # NVML stood in for: no driver is asked anything.
    from seedvr2x.runtime import model

    for nvml, version in ((NVML(), "580.82.07"), (NVML(query=3), None)):
        monkeypatch.setattr(ctypes, "CDLL", lambda name, nvml=nvml: nvml)
        assert model.nvidia_driver() == version
        assert nvml.shut  # shut down after its initialisation, whatever the answer
    nvml = NVML(init=9)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: nvml)
    assert model.nvidia_driver() is None and not nvml.shut

    def missing(name: str) -> object:
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    assert model.nvidia_driver() is None
    assert caplog.text.count("the NVIDIA driver's version is unknown") == 3


def sampler_bar() -> object:
    """The progress bar the vendored sampler builds for a window, as its own method builds it."""
    from types import SimpleNamespace

    from seedvr2x.vendor.common.diffusion.samplers.base import Sampler

    sampler = SimpleNamespace(timesteps=range(4), return_endpoint=True)
    return Sampler.get_progress_bar(sampler)  # pyright: ignore[reportArgumentType]


def test_the_samplers_bars_are_out_of_the_default_output_but_not_a_download_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DESIGN.md, Architecture, What it prints (decided on 2026-10-28, built on 2026-10-29): the
    # vendored sampler's tqdm, one bar a window (848 lines in a night), is off but under
    # --verbose; vendor/ is never edited, so model.sampler_bars replaces the name its module
    # builds the bar with. A download's bar (huggingface_hub's own tqdm) keeps its own.
    from huggingface_hub.utils import tqdm as download_bar

    from seedvr2x.runtime import model
    from seedvr2x.vendor.common.diffusion.samplers import base

    monkeypatch.setattr(base, "tqdm", base.tqdm)  # restored after the test
    model.sampler_bars(False)
    quiet = sampler_bar()
    assert quiet.disable is True  # pyright: ignore[reportAttributeAccessIssue]
    model.sampler_bars(True)
    shown = sampler_bar()
    assert shown.disable is False  # pyright: ignore[reportAttributeAccessIssue]
    assert shown.desc == "SimpleNamespace"  # pyright: ignore[reportAttributeAccessIssue]
    shown.close()  # pyright: ignore[reportAttributeAccessIssue]
    # Silenced from outside: tqdm itself is untouched, and a download's bar is shown.
    model.sampler_bars(False)
    import tqdm

    assert tqdm.tqdm.__module__ == "tqdm.std"
    downloading = download_bar(total=1, name="test")
    try:
        assert downloading.disable is False
    finally:
        downloading.close()


def test_the_models_load_sets_the_samplers_bars_by_the_log_level(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: pytest.TempPathFactory,
) -> None:
    # model.load_models decides on every load: shown at the debug level (--verbose), off else.
    # The file is missing, which it finds after that decision.
    from pathlib import Path

    import torch

    from seedvr2x.runtime import model
    from seedvr2x.runtime.weights import ModelError

    seen: list[bool] = []
    monkeypatch.setattr(model, "sampler_bars", seen.append)
    absent = Path(str(tmp_path)) / "absent.safetensors"
    for level, shown in ((logging.DEBUG, True), (logging.INFO, False)):
        caplog.set_level(level, logger="seedvr2x")
        with pytest.raises((ModelError, OSError)):
            model.load_models(absent, absent, torch.device("cpu"))
        assert seen[-1] is shown
    assert len(seen) == 2
