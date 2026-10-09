"""The model check (DESIGN.md, Weights: recognised by content): a model file is known by its
tensors, from its safetensors header, read without torch, never by its name. v1 accepts SeedVR2's
7B DiT, regular or sharp, and its VAE, every tensor F16; anything else is refused before the first
pass, the message saying what the file is.

The files are synthetic: a header written from the vendored model's tensors, or from a variant of
them as the real files measured hold theirs, then the file extended to its full size without its
data, a sparse file. The real files are checked too, header only, when SEEDVR2X_MODEL_DIR holds
them."""

import functools
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from seedvr2x import cli
from seedvr2x.media import ffmpeg
from seedvr2x.runtime import weights
from seedvr2x.runtime.weights import DIT_3B, DIT_7B, VAE, Architecture, ModelError, TensorInfo

# A model's tensors, by name: (dtype, shape).
Tensors = dict[str, tuple[str, tuple[int, ...]]]

# Bytes per value of the dtypes these files hold, safetensors' sizes.
BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1, "I64": 8, "F64": 8}

# What every refusal of a readable file ends with.
V1 = (
    "seedvr2x v1 runs SeedVR2's 7B DiT, regular or sharp, in fp16, with its VAE in fp16; phase 2"
    " brings the two 7Bs' smaller files"
)


@functools.cache
def vendored(model: str) -> dict[str, tuple[int, ...]]:
    """The shapes of the vendored model's tensors, by name: its state_dict, built on the meta
    device with tests/test_model.py's BUILD calls; "7b" and "3b" the DiTs of configs_7b and
    configs_3b, "vae" configs_7b's VAE merged with its architecture's config, as
    runtime/model.py's load_models merges it. Built once per session, as a module-scoped fixture
    would be, and taken by the CLI's tests of other modules too (accepted_models)."""
    import torch
    from omegaconf import OmegaConf

    import seedvr2x.vendor
    from seedvr2x.vendor.common.config import create_object, load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    config: Any = load_config(str(vendor / f"configs_{'3b' if model == '3b' else '7b'}/main.yaml"))
    model_config: Any = config.dit.model
    if model == "vae":
        vae_config: Any = load_config(
            str(vendor / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
        )
        vae_config.spatial_downsample_factor = vae_config.get("spatial_downsample_factor", 8)
        vae_config.temporal_downsample_factor = vae_config.get("temporal_downsample_factor", 4)
        model_config = OmegaConf.merge(config.vae.model, vae_config)
    with torch.device("meta"):
        built = create_object(model_config)
    return {name: tuple(tensor.shape) for name, tensor in built.state_dict().items()}


def as_dtype(model: str, dtype: str) -> Tensors:
    """The vendored model's tensors, every one in dtype."""
    return {name: (dtype, shape) for name, shape in vendored(model).items()}


def write(path: Path, tensors: Tensors, metadata: Mapping[str, str] | None = None) -> Path:
    """A safetensors file of these tensors, their data contiguous in the order given: its header,
    with __metadata__ when given, then nothing but its full length, by truncate: a sparse file,
    which costs no disk (the 7B in fp16, 16.48 GB on paper); the check reads the header alone."""
    header: dict[str, object] = {} if metadata is None else {"__metadata__": dict(metadata)}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        end = offset + math.prod(shape) * BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header).encode()
    with path.open("wb") as file:
        file.write(len(raw).to_bytes(8, "little") + raw)
        file.truncate(8 + len(raw) + offset)
    return path


def accepted_models(directory: Path) -> list[str]:
    """SeedVR2's 7B DiT and its VAE in fp16, synthetic (write), in directory under numz's names,
    which seedvr2x doesn't pin (runtime/pull.py): models the header check accepts and no pin
    refuses, for the CLI's tests running it for real in a process of their own, which imports no
    torch. Returns the options naming them."""
    dit, vae = "seedvr2_ema_7b_sharp_fp16.safetensors", "ema_vae_fp16.safetensors"
    write(directory / dit, as_dtype("7b", "F16"))
    write(directory / vae, as_dtype("vae", "F16"))
    return ["--dit-model", dit, "--vae-model", vae]


# The 288 matrices of the 7B's blocks, 8 per block, which phase 2's files quantise (DESIGN.md,
# Weights, Phase 2).
MATRIX = re.compile(r"blocks\.\d+\.(attn\.proj_(qkv|out)\.(vid|txt)|mlp\.(vid|txt)\.proj_(in|out))")


def phase2(layout: str) -> Tensors:
    """The 7B in a layout of seedvr2x's phase-2 files, comfy-kitchen's, as their headers hold it:
    each block matrix's weight quantised, its scales and its comfy_quant marker (U8) beside it;
    the 840 other tensors F16."""
    tensors: Tensors = {}
    for name, shape in vendored("7b").items():
        layer = name.removesuffix(".weight")
        if not (name.endswith(".weight") and MATRIX.fullmatch(layer)):
            tensors[name] = ("F16", shape)
            continue
        out, inputs = shape
        if layout == "fp8_scaled":
            tensors[name] = ("F8_E4M3", shape)
            tensors[f"{layer}.weight_scale"] = ("F32", ())
            tensors[f"{layer}.comfy_quant"] = ("U8", (27,))
        elif layout == "int8_convrot":
            tensors[name] = ("I8", shape)
            tensors[f"{layer}.weight_scale"] = ("F32", (out, 1))
            tensors[f"{layer}.comfy_quant"] = ("U8", (72,))
        else:  # nvfp4: two 4-bit values per byte, a scale every 16 values
            tensors[name] = ("U8", (out, inputs // 2))
            tensors[f"{layer}.weight_scale"] = ("F8_E4M3", (out, inputs // 16))
            tensors[f"{layer}.weight_scale_2"] = ("F32", ())
            tensors[f"{layer}.comfy_quant"] = ("U8", (19,))
    return tensors


def numz_fp8() -> Tensors:
    """numz's 7B fp8 file, "mixed_block35_fp16": blocks 0-34 and the 12 tensors outside the
    blocks in F8_E4M3, block 35's 31 in F16."""
    return {
        name: ("F16" if name.startswith("blocks.35.") else "F8_E4M3", shape)
        for name, shape in vendored("7b").items()
    }


def one_f32() -> Tensors:
    """The 7B in F16 but for one tensor in F32, which a check of a sample would miss."""
    tensors = as_dtype("7b", "F16")
    tensors["blocks.17.attn.norm_q.vid.weight"] = ("F32", (128,))
    return tensors


def reshaped() -> Tensors:
    """The 7B's tensors under the same names in F16, one of another shape: txt_in.weight taking
    4096 values where SeedVR2's text embeddings have 5120."""
    tensors = as_dtype("7b", "F16")
    tensors["txt_in.weight"] = ("F16", (3072, 4096))
    return tensors


def refused(path: Path, role: weights.Role) -> str:
    with pytest.raises(ModelError) as error:
        weights.check(path, role)
    return str(error.value)


@pytest.mark.parametrize(("model", "architecture"), [("7b", DIT_7B), ("3b", DIT_3B), ("vae", VAE)])
def test_pinned_digests(model: str, architecture: Architecture) -> None:
    # Each pin is the vendored model's inventory: its tensors' sorted lines `name [shape]`.
    shapes = vendored(model)
    lines = sorted(f"{name} {list(shape)}" for name, shape in shapes.items())
    assert len(lines) == architecture.tensors
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == architecture.digest
    infos = {name: TensorInfo("F16", shape) for name, shape in shapes.items()}
    assert weights.inventory(infos) == architecture.digest
    if model != "vae":
        # The DiTs by their block count and width, as a quantised file is named after them.
        blocks = {name.split(".")[1] for name in shapes if name.startswith("blocks.")}
        assert weights.DITS[(len(blocks), shapes["txt_in.weight"][0])] is architecture


@pytest.mark.parametrize(
    ("name", "metadata"),
    [
        # numz's files, no metadata; seedvr2x's, with theirs: the regular and the sharp 7B have
        # the same tensors.
        ("seedvr2_ema_7b_fp16.safetensors", None),
        ("seedvr2_ema_7b_sharp_fp16.safetensors", None),
        ("seedvr2x_ema_7b_fp16.safetensors", {"format": "pt", "license": "apache-2.0"}),
        ("seedvr2x_ema_7b_sharp_fp16.safetensors", {"format": "pt", "change": "rounded to fp16"}),
    ],
)
def test_7b_accepted(tmp_path: Path, name: str, metadata: dict[str, str] | None) -> None:
    path = write(tmp_path / name, as_dtype("7b", "F16"), metadata)
    found = weights.check(path, "dit")
    assert found is DIT_7B and found.config == "configs_7b"
    assert weights.describe(path) == "SeedVR2's 7B DiT in fp16"


@pytest.mark.parametrize(
    ("name", "metadata"),
    [("ema_vae_fp16.safetensors", None), ("seedvr2x_ema_vae_fp16.safetensors", {"format": "pt"})],
)
def test_vae_accepted(tmp_path: Path, name: str, metadata: dict[str, str] | None) -> None:
    path = write(tmp_path / name, as_dtype("vae", "F16"), metadata)
    assert weights.check(path, "vae") is VAE
    assert weights.describe(path) == "SeedVR2's VAE in fp16"


def test_renamed_files_recognised_by_content(tmp_path: Path) -> None:
    # The 7B under a 3B's name is the 7B, built from its config; the 3B under the default DiT's
    # name is refused as the 3B, whatever its metadata says.
    seven = write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("7b", "F16"))
    assert weights.check(seven, "dit").config == "configs_7b"
    three = tmp_path / "seedvr2x_ema_7b_sharp_fp16.safetensors"
    write(three, as_dtype("3b", "F16"), {"source": "seedvr2_ema_7b_sharp.pth"})
    assert refused(three, "dit") == f"{three}: SeedVR2's 3B DiT in fp16: {V1}"


def test_load_models_by_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # load_models builds the DiT of what its file is, on the meta device here, the weights not
    # loaded: the 7B under a 3B's name builds the 7B, where numz would build the 3B.
    import torch

    from seedvr2x.runtime import model

    loaded: list[str] = []
    monkeypatch.setattr(model, "_load_weights", lambda module, path, *a: loaded.append(path.name))
    write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("7b", "F16"))
    write(tmp_path / "ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    models = model.load_models(
        tmp_path / "seedvr2_ema_3b_fp16.safetensors",
        tmp_path / "ema_vae_fp16.safetensors",
        torch.device("cpu"),
    )
    assert type(models.runner.dit).__module__ == "seedvr2x.vendor.models.dit_7b.nadit"
    assert loaded == ["ema_vae_fp16.safetensors", "seedvr2_ema_3b_fp16.safetensors"]
    # The 3B under the 7B's name, refused before anything is built; and a VAE in bf16, with the
    # 7B as the DiT.
    loaded.clear()
    write(tmp_path / "seedvr2_ema_7b_fp16.safetensors", as_dtype("3b", "F16"))
    write(tmp_path / "ema_vae_bf16.safetensors", as_dtype("vae", "BF16"))
    for dit, vae, what in (
        ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors", "3B DiT in fp16"),
        ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_bf16.safetensors", "VAE in bf16"),
    ):
        with pytest.raises(ModelError, match=f"SeedVR2's {what}"):
            model.load_models(tmp_path / dit, tmp_path / vae, torch.device("cpu"))
    assert loaded == []


QUANTISED = "SeedVR2's 7B DiT quantised, its block matrices in {} with scale tensors beside them,"
QUANTISED += " in comfy-kitchen's layout ({})"


@pytest.mark.parametrize(
    ("name", "tensors", "role", "what"),
    [
        (
            "seedvr2_ema_7b_bf16.safetensors",
            lambda: as_dtype("7b", "BF16"),
            "dit",
            "SeedVR2's 7B DiT in bf16",
        ),
        (
            "seedvr2_ema_7b.safetensors",
            lambda: as_dtype("7b", "F32"),
            "dit",
            "SeedVR2's 7B DiT in fp32",
        ),
        (
            "seedvr2_ema_7b_fp16.safetensors",
            one_f32,
            "dit",
            "SeedVR2's 7B DiT in mixed precision (1,127 tensors F16, 1 F32)",
        ),
        (
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            numz_fp8,
            "dit",
            "SeedVR2's 7B DiT in fp8, cast without scales (1,097 tensors F8_E4M3, 31 F16)",
        ),
        (
            "seedvr2_ema_3b_fp16.safetensors",
            lambda: as_dtype("3b", "F16"),
            "dit",
            "SeedVR2's 3B DiT in fp16",
        ),
        (
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            lambda: as_dtype("3b", "F8_E4M3"),
            "dit",
            "SeedVR2's 3B DiT in fp8, cast without scales (635 tensors F8_E4M3)",
        ),
        (
            "seedvr2x_ema_7b_fp8_scaled.safetensors",
            lambda: phase2("fp8_scaled"),
            "dit",
            QUANTISED.format("fp8", "1,704 tensors: 840 F16, 288 F32, 288 F8_E4M3, 288 U8"),
        ),
        (
            "seedvr2x_ema_7b_int8_convrot.safetensors",
            lambda: phase2("int8_convrot"),
            "dit",
            QUANTISED.format("INT8", "1,704 tensors: 840 F16, 288 F32, 288 I8, 288 U8"),
        ),
        (
            "seedvr2x_ema_7b_sharp_nvfp4.safetensors",
            lambda: phase2("nvfp4"),
            "dit",
            QUANTISED.format(
                "4-bit values packed two per byte",
                "1,992 tensors: 840 F16, 576 U8, 288 F32, 288 F8_E4M3",
            ),
        ),
        (
            "seedvr2_ema_7b_fp16.safetensors",
            reshaped,
            "dit",
            "SeedVR2's 7B DiT by its blocks and width, with other tensors than its own (1,128"
            " tensors: 1,128 F16)",
        ),
        (
            "ema_vae_fp16.safetensors",
            lambda: as_dtype("vae", "F16"),
            "dit",
            "SeedVR2's VAE in fp16, given as the DiT (--dit-model)",
        ),
        (
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            lambda: as_dtype("7b", "F16"),
            "vae",
            "SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model)",
        ),
        (
            "seedvr2x_ema_7b_sharp_nvfp4.safetensors",
            lambda: phase2("nvfp4"),
            "vae",
            "SeedVR2's 7B DiT quantised, its block matrices in 4-bit values packed two per byte"
            " with scale tensors beside them, in comfy-kitchen's layout (1,992 tensors: 840 F16,"
            " 576 U8, 288 F32, 288 F8_E4M3), given as the VAE (--vae-model)",
        ),
        (
            "transnetv2.safetensors",
            lambda: (
                {f"layer{k}.weight": ("F32", (8,)) for k in range(84)}
                | {f"layer{k}.step": ("I64", ()) for k in range(6)}
            ),
            "vae",
            "a safetensors file of 90 tensors (84 F32, 6 I64), neither SeedVR2's DiT nor its VAE",
        ),
    ],
    ids=[
        "7b-bf16",
        "7b-fp32",
        "7b-one-f32",
        "numz-7b-fp8",
        "3b-fp16",
        "3b-fp8",
        "fp8-scaled",
        "int8",
        "nvfp4",
        "7b-a-shape-changed",
        "vae-as-dit",
        "7b-as-vae",
        "nvfp4-as-vae",
        "other-safetensors",
    ],
)
def test_refused_saying_what(
    tmp_path: Path, name: str, tensors: Callable[[], Tensors], role: weights.Role, what: str
) -> None:
    path = write(tmp_path / name, tensors())
    assert refused(path, role) == f"{path}: {what}: {V1}"


def test_other_formats(tmp_path: Path) -> None:
    # Known by their first bytes: GGUF's signature, a zip (torch.save's format), a pickle (its
    # earlier one); random bytes are neither. GGUF's version, then its tensor count: the 3B's
    # 635 = 0x027B puts safetensors' brace at byte 8.
    for name, tensors in (("seedvr2_ema_7b-Q4_K_M.gguf", 1128), ("seedvr2_ema_3b-Q8_0.gguf", 635)):
        header = b"GGUF" + (3).to_bytes(4, "little") + tensors.to_bytes(8, "little") + bytes(8)
        (tmp_path / name).write_bytes(header)
    assert (tmp_path / "seedvr2_ema_3b-Q8_0.gguf").read_bytes()[8:9] == b"{"
    with zipfile.ZipFile(tmp_path / "seedvr2_ema_7b.pth", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"weights": 1}, protocol=2))
    (tmp_path / "ema_vae.pth").write_bytes(pickle.dumps({"weights": 1}, protocol=2))
    noise = random.Random(0).randbytes(4096)
    assert noise[8:9] != b"{" and noise[:1] not in (b"G", b"P", b"\x80")
    (tmp_path / "noise.safetensors").write_bytes(noise)
    checkpoint = (
        "a PyTorch checkpoint ({}), as ByteDance's fp32 masters (.pth) are; seedvr2x reads"
        " safetensors files, which the repository's models/ scripts make from them"
    )
    gguf = (
        "a GGUF file (quantised weights, as numz's Q4_K_M and seedvr2x's own Q8_0, Q4_K and"
        " dynamic files)"
    )
    expected = {
        "seedvr2_ema_7b-Q4_K_M.gguf": gguf,
        "seedvr2_ema_3b-Q8_0.gguf": gguf,
        "seedvr2_ema_7b.pth": checkpoint.format("zip"),
        "ema_vae.pth": checkpoint.format("pickle"),
        "noise.safetensors": "neither safetensors, GGUF nor a PyTorch checkpoint",
    }
    for name, what in expected.items():
        path = tmp_path / name
        assert weights.describe(path) == what
        for role in ("dit", "vae"):
            assert refused(path, role) == f"{path}: {what}: {V1}"


def test_missing(tmp_path: Path) -> None:
    # Nothing there, or something that isn't a file to read: a directory, a FIFO (never opened,
    # which would wait for a writer).
    (tmp_path / "a directory").mkdir()
    os.mkfifo(tmp_path / "a fifo")
    for name, what in (
        ("x.safetensors", "no such model file"),
        ("a directory", "not a regular file"),
        ("a fifo", "not a regular file"),
    ):
        path = tmp_path / name
        assert refused(path, "dit") == f"{path}: {what}"
        with pytest.raises(ModelError, match=what):
            weights.describe(path)


def test_cut_short(tmp_path: Path) -> None:
    # A download stopped: the data cut, then the header itself.
    path = write(tmp_path / "seedvr2x_ema_7b_sharp_fp16.safetensors", as_dtype("7b", "F16"))
    with path.open("rb") as file:
        header = 8 + int.from_bytes(file.read(8), "little")
    full = path.stat().st_size
    assert full == header + 16_479_216_464  # the data of numz's and seedvr2x's 7B fp16 files
    os.truncate(path, header + 1_000_000)
    assert refused(path, "dit") == (
        f"{path}: cut short, {header + 1_000_000:,} of {full:,} bytes; fetch it again"
    )
    os.truncate(path, 1_000)
    assert refused(path, "dit") == (
        f"{path}: cut short inside its header, 1,000 bytes of at least {header:,}; fetch it again"
    )


def test_empty(tmp_path: Path) -> None:
    # What a download failed before its first byte leaves (wget -O): refused as a file cut short
    # is, not as a file of another kind, so without V1's tail, in either role; describe too.
    path = tmp_path / "seedvr2x_ema_7b_sharp_fp16.safetensors"
    path.write_bytes(b"")
    said = f"{path}: empty (0 bytes); fetch it again"
    for role in ("dit", "vae"):
        assert refused(path, role) == said
    with pytest.raises(ModelError) as error:
        weights.describe(path)
    assert str(error.value) == said


def test_too_short_to_tell(tmp_path: Path) -> None:
    # 1 to 8 bytes: too few for a safetensors header's length and its brace, which tell the
    # format, and for any model file: what a download cut short leaves, refused as one, its size
    # said, without V1's tail, in either role; describe too. "GGUF" and a zip's signature alone
    # are such files, not a GGUF file or a checkpoint. From 9 bytes on, the start tells.
    path = tmp_path / "seedvr2x_ema_7b_sharp_fp16.safetensors"
    for start in (b"\x01", b"\x80\x02", b"GGUF", b"PK\x03\x04", b"GGUF\x03\x00\x00", bytes(8)):
        path.write_bytes(start)
        size = f"{len(start)} byte{'s' if len(start) > 1 else ''}"
        said = f"{path}: cut short, {size}, too few for any model file; fetch it again"
        for role in ("dit", "vae"):
            assert refused(path, role) == said
        with pytest.raises(ModelError) as error:
            weights.describe(path)
        assert str(error.value) == said
    path.write_bytes(b"GGUF" + bytes(5))
    assert weights.describe(path).startswith("a GGUF file")
    path.write_bytes(random.Random(0).randbytes(9))
    assert weights.describe(path) == "neither safetensors, GGUF nor a PyTorch checkpoint"
    path.write_bytes((1_000).to_bytes(8, "little") + b"{")
    assert refused(path, "dit") == (
        f"{path}: cut short inside its header, 9 bytes of at least 1,008; fetch it again"
    )


def raw(path: Path, header: bytes | object, data: int, length: int | None = None) -> Path:
    """A safetensors file of this header, JSON unless bytes, its length said as length when given,
    followed by data bytes of zeros."""
    text = header if isinstance(header, bytes) else json.dumps(header).encode()
    said = len(text) if length is None else length
    path.write_bytes(said.to_bytes(8, "little") + text + bytes(data))
    return path


def u8(*spans: tuple[int, int]) -> dict[str, object]:
    """Tensors of U8 values, one per span of the data, named a, b...."""
    return {
        chr(ord("a") + k): {"dtype": "U8", "shape": [end - begin], "data_offsets": [begin, end]}
        for k, (begin, end) in enumerate(spans)
    }


@pytest.mark.parametrize(
    ("header", "data", "what"),
    [
        (b'{"a": {"dtype": "U8", ', 0, "its header isn't JSON: Expecting property name"),
        (u8((0, 2), (3, 5)), 5, "b: its data at 3, after a gap from 2"),
        (u8((0, 2), (1, 3)), 3, "b: its data at 1, inside the data before it, up to 2"),
        (u8((0, 2)), 3, "1 byte after its tensors' data"),
        (
            {"a": {"dtype": "F16", "shape": [3], "data_offsets": [0, 4]}},
            4,
            "a: 4 bytes of data for 3 values of F16",
        ),
        ({"a": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 0]}}, 0, "a: its shape"),
        ({"a": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}}, 1, "a: its shape"),
        ({"a": {"dtype": "U8", "shape": [1]}}, 1, "a: its data_offsets aren't [begin, end]"),
        ({"a": {"shape": [1], "data_offsets": [0, 1]}}, 1, "a: no dtype"),
        ({"__metadata__": {"k": 1}, **u8((0, 1))}, 1, "its __metadata__ doesn't map strings"),
        ({"a": [1]}, 0, "a: not a tensor's entry"),
        (b'{"\xff": 1}', 0, "its header isn't UTF-8"),
        # Headers safetensors 0.8.0 refuses at once, which escaped the check as other exceptions
        # (ValueError, RecursionError, UnicodeEncodeError) or held numbers past 64 bits.
        (
            b'{"a": {"dtype": "U8", "shape": [' + b"1" * 5000 + b'], "data_offsets": [0, 1]}}',
            1,
            "its header isn't JSON: a number too long",
        ),
        (b'{"a": ' + b"[" * 200_000, 0, "its header isn't JSON: nested too deep"),
        # 250 tensors, as the VAE has: their digest would be taken.
        (
            {
                ("\ud800" if k == 0 else f"t{k}"): {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [k, k + 1],
                }
                for k in range(250)
            },
            250,
            "\\ud800: its name isn't UTF-8",
        ),
        (
            {"a": {"dtype": "U8", "shape": [2**40] * 3, "data_offsets": [0, 1]}},
            1,
            "a: its shape overflows 64 bits",
        ),
        (
            {"a": {"dtype": "U8", "shape": [2**32, 2**32], "data_offsets": [0, 1]}},
            1,
            "a: its shape overflows 64 bits",
        ),
        (
            {"a": {"dtype": "U8", "shape": [2**64], "data_offsets": [0, 1]}},
            1,
            "a: its shape isn't whole numbers below 2^64",
        ),
        (
            {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 2**64]}},
            1,
            "a: its data_offsets aren't [begin, end]",
        ),
        # Sizes that agree: the file's size they make, 4,301 digits, was too long to write in
        # the message that it is cut short (ValueError).
        (
            {"a": {"dtype": "U8", "shape": [10**4300 - 1], "data_offsets": [0, 10**4300 - 1]}},
            1,
            "a: its shape isn't whole numbers below 2^64",
        ),
    ],
    ids=[
        "bad-json",
        "gap",
        "overlap",
        "bytes-after",
        "size",
        "negative-shape",
        "bool-shape",
        "no-offsets",
        "no-dtype",
        "metadata",
        "entry",
        "utf-8",
        "5000-digits",
        "nested",
        "surrogate",
        "overflow",
        "overflow-at-2^64",
        "dimension-2^64",
        "offset-2^64",
        "4300-digits",
    ],
)
def test_damaged(tmp_path: Path, header: bytes | object, data: int, what: str) -> None:
    path = raw(tmp_path / "x.safetensors", header, data)
    assert refused(path, "dit").startswith(f"{path}: a damaged safetensors file: {what}")


def test_many_dimensions_refused_at_once(tmp_path: Path) -> None:
    # 200,000 dimensions of 2^63-1, which math.prod took over a minute to multiply: refused at
    # the second, past 2^64.
    header = {"a": {"dtype": "U8", "shape": [2**63 - 1] * 200_000, "data_offsets": [0, 1]}}
    path = raw(tmp_path / "x.safetensors", header, 1)
    started = time.monotonic()
    assert refused(path, "dit") == (
        f"{path}: a damaged safetensors file: a: its shape overflows 64 bits"
    )
    assert time.monotonic() - started < 5


def test_block_index_of_any_length(tmp_path: Path) -> None:
    # safetensors takes any name: a block index of 5,000 digits, which int() refuses (over
    # 4,300), read as written.
    header = {f"blocks.{'1' * 5000}.weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = raw(tmp_path / "x.safetensors", header, 1)
    assert weights.describe(path) == (
        "a safetensors file of 1 tensor (1 U8), neither SeedVR2's DiT nor its VAE"
    )


def test_header_length_limits(tmp_path: Path) -> None:
    # safetensors' own: a header of 1 to 100,000,000 bytes.
    for length in (0, 100_000_001):
        path = raw(tmp_path / "x.safetensors", b"{}", 0, length)
        assert refused(path, "dit") == (
            f"{path}: a damaged safetensors file: a header of {length:,} bytes, where safetensors"
            " takes 1 to 100,000,000"
        )
    # 100,000,000 taken, the header read whole (a sparse file: the brace, then zeros), then
    # refused for what it holds.
    path = tmp_path / "y.safetensors"
    with path.open("wb") as file:
        file.write((100_000_000).to_bytes(8, "little") + b"{")
        file.truncate(8 + 100_000_000)
    assert refused(path, "dit").startswith(
        f"{path}: a damaged safetensors file: its header isn't JSON: Expecting property name"
    )


def test_unknown_dtype_read(tmp_path: Path) -> None:
    # A dtype the read doesn't size fails no read, and no check passes it.
    header = {"a": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}
    path = raw(tmp_path / "x.safetensors", header, 2)
    what = "a safetensors file of 1 tensor (1 F4), neither SeedVR2's DiT nor its VAE"
    assert weights.describe(path) == what
    assert refused(path, "vae") == f"{path}: {what}: {V1}"


def test_safetensors_told_first(tmp_path: Path) -> None:
    # A header of 640 bytes, padded with spaces as safetensors' format allows: its length begins
    # as a pickle of protocol 2 does, 0x80 0x02, and the file is safetensors all the same.
    header = json.dumps(u8((0, 1))).encode()
    path = raw(tmp_path / "x.safetensors", header + b" " * (640 - len(header)), 1)
    with path.open("rb") as file:
        assert file.read(2) == b"\x80\x02"
    what = "a safetensors file of 1 tensor (1 U8), neither SeedVR2's DiT nor its VAE"
    assert weights.describe(path) == what


def test_check_models(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    accepted_models(tmp_path)
    dit, vae = "seedvr2_ema_7b_sharp_fp16.safetensors", "ema_vae_fp16.safetensors"
    with caplog.at_level(logging.INFO):
        weights.check_models(tmp_path / dit, tmp_path / vae)
    assert f"DiT {dit}: SeedVR2's 7B DiT in fp16" in caplog.messages
    assert f"VAE {vae}: SeedVR2's VAE in fp16" in caplog.messages
    # Both refused at once, each saying what it is.
    with pytest.raises(ModelError) as error:
        weights.check_models(tmp_path / vae, tmp_path / dit)
    assert str(error.value).splitlines() == [
        f"{tmp_path / vae}: SeedVR2's VAE in fp16, given as the DiT (--dit-model): {V1}",
        f"{tmp_path / dit}: SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model): {V1}",
    ]


# In a fresh interpreter: the header read takes neither torch nor the safetensors library.
NO_TORCH = """
import sys
from pathlib import Path
from seedvr2x.runtime import weights
assert weights.check(Path(sys.argv[1]), "dit") is weights.DIT_7B
print(weights.describe(Path(sys.argv[2])))
assert "torch" not in sys.modules and "safetensors" not in sys.modules, "imported"
"""


def test_header_read_without_torch(tmp_path: Path) -> None:
    seven = write(tmp_path / "seedvr2x_ema_7b_fp16.safetensors", as_dtype("7b", "F16"))
    three = write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("3b", "F16"))
    result = subprocess.run(
        [sys.executable, "-c", NO_TORCH, str(seven), str(three)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "SeedVR2's 3B DiT in fp16\n"


def _usable() -> bool:
    try:
        ffmpeg.check()
    except ffmpeg.MediaError:
        return False
    return True


@pytest.mark.skipif(not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1")
def test_cli_refuses_before_the_first_pass(tmp_path: Path) -> None:
    # A 3B as the DiT stops the run in its checks, before the first pass (which logs the frames
    # it counted) and before torch is imported.
    source = tmp_path / "in.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "3", "-c:v", "ffv1", str(source)),
        ],
        check=True,
    )
    write(tmp_path / "seedvr2x_ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("3b", "F16"))
    args = [str(source), "-o", "out.mkv", "--model-dir", "."]
    args += ["--dit-model", "seedvr2_ema_3b_fp16.safetensors"]
    # Exit 3 if torch was imported: a failed assert would exit 1, as the refusal does.
    code = (
        "import sys; from seedvr2x import cli; status = cli.main(sys.argv[1:]);"
        " sys.exit(3 if 'torch' in sys.modules else status)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 1, result.stderr
    assert f"seedvr2_ema_3b_fp16.safetensors: SeedVR2's 3B DiT in fp16: {V1}" in result.stderr
    assert "frames, 64x48" not in result.stderr
    assert "VAE seedvr2x_ema_vae_fp16.safetensors: SeedVR2's VAE in fp16" in result.stderr


def test_defaults_in_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(["--help"])
    assert exited.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "(default: seedvr2x_ema_7b_sharp_fp16.safetensors, the sharp 7B)" in text
    assert "(default: seedvr2x_ema_vae_fp16.safetensors)" in text
    assert "SeedVR2's 7B DiT, regular or sharp, in fp16, recognised by its tensors" in text


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")

# The real files, by name, as a role: None when accepted, else the words of their refusal.
GIVEN_AS_VAE = "SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model)"
GIVEN_AS_DIT = "SeedVR2's VAE in fp16, given as the DiT (--dit-model)"
PHASE_2 = "SeedVR2's 7B DiT quantised, its block matrices in {} with scale tensors beside them"
GGUF = "a GGUF file"
OTHER = "a safetensors file of {}, neither SeedVR2's DiT nor its VAE"
REAL: list[tuple[str, weights.Role, str | None]] = [
    # numz's
    ("seedvr2_ema_7b_fp16.safetensors", "dit", None),
    ("seedvr2_ema_7b_fp16.safetensors", "vae", GIVEN_AS_VAE),
    ("seedvr2_ema_7b_sharp_fp16.safetensors", "dit", None),
    ("ema_vae_fp16.safetensors", "vae", None),
    ("ema_vae_fp16.safetensors", "dit", GIVEN_AS_DIT),
    ("seedvr2_ema_3b_fp16.safetensors", "dit", "SeedVR2's 3B DiT in fp16:"),
    (
        "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "dit",
        "SeedVR2's 3B DiT in fp8, cast without scales (635 tensors F8_E4M3):",
    ),
    (
        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "dit",
        "SeedVR2's 7B DiT in fp8, cast without scales (1,097 tensors F8_E4M3, 31 F16):",
    ),
    ("seedvr2_ema_7b-Q4_K_M.gguf", "dit", GGUF),
    # 635 tensors, 0x027B: the brace at byte 8, after GGUF's signature.
    ("seedvr2_ema_3b-Q4_K_M.gguf", "dit", GGUF),
    ("seedvr2_ema_3b-Q8_0.gguf", "dit", GGUF),
    ("seedvr2_ema_7b.pth", "dit", "a PyTorch checkpoint (zip)"),
    ("ema_vae.pth", "vae", "a PyTorch checkpoint (zip)"),
    # seedvr2x's
    ("seedvr2x_ema_7b_fp16.safetensors", "dit", None),
    ("seedvr2x_ema_7b_fp16.safetensors", "vae", GIVEN_AS_VAE),
    ("seedvr2x_ema_7b_sharp_fp16.safetensors", "dit", None),
    ("seedvr2x_ema_7b_sharp_fp16.safetensors", "vae", GIVEN_AS_VAE),
    ("seedvr2x_ema_vae_fp16.safetensors", "vae", None),
    ("seedvr2x_ema_vae_fp16.safetensors", "dit", GIVEN_AS_DIT),
    *(
        (f"seedvr2x_ema_7b{sharp}_{layout}.safetensors", "dit", PHASE_2.format(matrices))
        for sharp in ("", "_sharp")
        for layout, matrices in (
            ("fp8_scaled", "fp8"),
            ("int8_convrot", "INT8"),
            ("nvfp4", "4-bit values packed two per byte"),
        )
    ),
    *(
        (f"seedvr2x_ema_7b{sharp}_{kind}.gguf", "dit", GGUF)
        for sharp in ("", "_sharp")
        for kind in ("Q8_0", "Q4_K", "Q4_K_imatrix", "dyn")
    ),
    ("transnetv2.safetensors", "dit", OTHER.format("90 tensors (84 F32, 6 I64)")),
    *(
        (
            f"seedvr2_ema_7b{sharp}_fp16.imatrix.safetensors",
            "dit",
            OTHER.format("576 tensors (288 F64, 288 I64)"),
        )
        for sharp in ("", "_sharp")
    ),
]


@pytest.mark.parametrize(
    ("name", "role", "refusal"), REAL, ids=[f"{name}-{role}" for name, role, _ in REAL]
)
def test_real_files(name: str, role: weights.Role, refusal: str | None) -> None:
    if not MODELS:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    path = Path(MODELS) / name
    if not path.is_file():
        pytest.skip(f"{name}: not in SEEDVR2X_MODEL_DIR")
    if refusal is None:
        assert weights.check(path, role) is weights.ACCEPTED[role]
    else:
        message = refused(path, role)
        assert message.startswith(f"{path}: ") and refusal in message and message.endswith(V1)
