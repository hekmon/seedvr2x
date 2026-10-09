"""The versions a resume compares (runtime/environment.py), derived from what the run imports and
from the libraries torch loads, on the CPU."""

import importlib
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest
import torch

from seedvr2x.runtime import environment


def test_versions_derived() -> None:
    found = environment.versions()
    # What the model computes with besides torch, which a hand-kept list once missed.
    assert {"torch", "torchvision", "diffusers", "rotary-embedding-torch", "einops"} <= found.keys()
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) for name in found)  # PEP 503
    assert all(found.values())


def test_torch_runtime_libraries() -> None:
    # Loaded by torch without being imported, so found from its requirements: through the extras
    # it asks (cuda-toolkit[cublas,…]), those shipping a shared library other than Python's
    # extension modules. They are the NVIDIA wheels torch requires, and triton (DESIGN.md).
    if torch.version.cuda is None:
        pytest.skip("a CPU build of torch loads no CUDA library")
    libraries = environment._libraries("torch")  # pyright: ignore[reportPrivateUsage]
    required = environment._required(metadata.requires("torch") or [])  # pyright: ignore[reportPrivateUsage]
    nvidia = {name for name in required if name.startswith("nvidia-")}
    assert {"nvidia-cublas", "nvidia-cudnn-cu13"} <= nvidia
    assert libraries == nvidia | {"triton"}
    # Extension modules only: cuda-bindings, which torch imports, is recorded as imported;
    # MarkupSafe (jinja2's speed-ups), which no run imports, isn't recorded.
    found = environment.versions()
    assert libraries | {"cuda-bindings"} <= found.keys()
    assert "markupsafe" not in found


def test_tools_not_recorded() -> None:
    # Imported here beside the run, yet no part of it: pytest, and pygments, which httpx imports
    # for its command line when a dev install has it. A resume mustn't depend on them.
    pytest.importorskip("pygments")
    environment.versions()
    providers = metadata.packages_distributions()
    loaded = {
        environment._name(distribution)  # pyright: ignore[reportPrivateUsage]
        for module in list(sys.modules)
        for distribution in providers.get(module.partition(".")[0], ())
    }
    assert {"pytest", "pygments"} <= loaded
    assert not {"pytest", "pygments"} & environment.versions().keys()


def test_start_hook_not_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    # setuptools' hook, which site imports at start unless SETUPTOOLS_USE_DISTUTILS=stdlib, is
    # none of the run's: the run imports no setuptools, and the record mustn't depend on that
    # variable.
    for module in [name for name in sys.modules if name.partition(".")[0] == "setuptools"]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, "_distutils_hack", ModuleType("_distutils_hack"))
    assert "setuptools" in metadata.packages_distributions()["_distutils_hack"]
    assert "setuptools" not in environment.imported()


# In a process of its own, whose HF_HOME is the test's: importing hf_xet writes a log under it.
DOWNLOAD_ONLY = """
import sys

import hf_xet

from seedvr2x.runtime import environment

assert hf_xet.__name__ in sys.modules
assert "hf-xet" in environment._declared()
assert "hf-xet" not in environment.imported()
found = environment.versions()
assert "hf-xet" not in found and "huggingface-hub" in found
"""


def test_download_only_not_recorded(tmp_path: Path) -> None:
    # hf_xet, which huggingface_hub imports for a Xet file's download alone, is none of the run's:
    # a job started with a download and resumed with the files in the cache record the same
    # environment. It is within seedvr2x's requirements, through huggingface-hub's.
    environment_variables = {
        name: value for name, value in os.environ.items() if name != "HF_XET_CACHE"
    }
    result = subprocess.run(
        [sys.executable, "-c", DOWNLOAD_ONLY],
        capture_output=True,
        text=True,
        check=False,
        env={**environment_variables, "HF_HOME": str(tmp_path / "hf")},
    )
    assert result.returncode == 0, result.stderr


def stand_in_distribution(site: Path, name: str, module: str, *requires: str) -> None:
    """Installs a distribution of one empty module into site."""
    (site / module).mkdir()
    (site / module / "__init__.py").write_text("")
    info = site / f"{module}-1.0.dist-info"
    info.mkdir()
    fields = [f"Name: {name}", "Version: 1.0", *(f"Requires-Dist: {each}" for each in requires)]
    (info / "METADATA").write_text("\n".join(["Metadata-Version: 2.1", *fields, ""]))
    (info / "RECORD").write_text(f"{module}/__init__.py,,\n")


@pytest.fixture
def site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """A directory on sys.path for stand-in distributions, which the environment's caches read
    once the test has installed them, and forget after."""
    environment.versions()  # seedvr2x's own imports first: no stand-in can take their place
    monkeypatch.syspath_prepend(tmp_path)
    yield tmp_path
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", None) or "").startswith(str(tmp_path)):
            del sys.modules[name]
    environment._providers.cache_clear()  # pyright: ignore[reportPrivateUsage]
    environment._declared.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_flash_attention_recorded(site: Path) -> None:
    # Not declared, since PyPI only has its source, yet the run's when installed: recorded. A
    # profiler, imported as well, isn't.
    try:
        metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        stand_in_distribution(site, "flash-attn", "flash_attn", "torch", "einops")
    stand_in_distribution(site, "a-profiler", "a_profiler")
    environment._providers.cache_clear()  # pyright: ignore[reportPrivateUsage]
    environment._declared.cache_clear()  # pyright: ignore[reportPrivateUsage]
    for module in ("flash_attn", "a_profiler"):
        importlib.import_module(module)
    found = environment.versions()
    assert "flash-attn" in found
    assert "a-profiler" not in found


def test_stray_files_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A file in the package whose name is no module's is taken for no module.
    package = tmp_path / "seedvr2x"
    (package / "runtime").mkdir(parents=True)
    (package / "runtime" / "model.orig.py").write_text("raise SystemExit('imported')\n")
    monkeypatch.setattr(environment, "PACKAGE", package)
    assert environment.versions()
