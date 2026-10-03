"""The vendored model imports and builds without numz's runtime, on the CPU; the NVIDIA
driver's version is read from NVML."""

import ctypes
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
