"""The versions a resume compares (runtime/environment.py), derived from what the run imports and
from the libraries torch loads, on the CPU."""

import re
from pathlib import Path

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
    # it asks (cuda-toolkit[cublas,…]), those shipping a shared library.
    if torch.version.cuda is None:
        pytest.skip("a CPU build of torch loads no CUDA library")
    libraries = environment._libraries("torch")  # pyright: ignore[reportPrivateUsage]
    for library in ("nvidia-cublas", "nvidia-cudnn"):
        assert any(name.startswith(library) for name in libraries), library
    assert not {"cuda-toolkit", "sympy"} & libraries  # neither ships a library
    # And recorded.
    assert any(name.startswith("nvidia-cublas") for name in environment.versions())


def test_stray_files_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A file in the package whose name is no module's is taken for no module.
    package = tmp_path / "seedvr2x"
    (package / "runtime").mkdir(parents=True)
    (package / "runtime" / "model.orig.py").write_text("raise SystemExit('imported')\n")
    monkeypatch.setattr(environment, "PACKAGE", package)
    assert environment.versions()
