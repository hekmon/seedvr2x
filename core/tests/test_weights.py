"""The model check (DESIGN.md, Weights: recognised by content): a model file is known by its
tensors, from its safetensors header, read without torch, never by its name. v1 accepts SeedVR2's
7B DiT, regular or sharp, every tensor F16 or in seedvr2x's int8 layout, each of its quantised
layers marked so, and its VAE, every tensor F16, and TransNetV2 as the shot detector, every tensor
F32 but its BatchNorm counters, I64; anything else is refused before the first pass, the message
saying what the file is, and a model given in another's role saying so.

The files are synthetic: a header written from the vendored model's tensors, or from a variant of
them as the real files measured hold theirs, then the file extended to its full size without its
data, a sparse file, but for an int8 file's markers, written where its header puts them. The real
files are checked too, header only, when SEEDVR2X_MODEL_DIR holds them.

The command line over the check (a run refused before the first pass, the defaults in --help)
is test_weights_cli.py's (apps/seedvr2x/tests/)."""

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
from library_fixtures import cli_words, grow, sparse
from test_gguf_file import Spec, body, entry, write_gguf

from seedvr2x import gguf_file, weights
from seedvr2x.weights import (
    DIT_3B,
    DIT_7B,
    DIT_7B_DYN,
    DIT_7B_INT8,
    DIT_7B_SHARP_DYN,
    VAE,
    Architecture,
    ModelError,
    TensorInfo,
)

posix_os: Any = os

# A model's tensors, by name: (dtype, shape).
Tensors = dict[str, tuple[str, tuple[int, ...]]]

# Bytes per value of the dtypes these files hold, safetensors' sizes.
BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1, "I64": 8, "F64": 8}

# What every refusal of a readable file given as the DiT or the VAE ends with, and as the shot
# detector.
V1 = (
    "seedvr2x v1 runs SeedVR2's 7B DiT, regular or sharp, in fp16, in int8 (seedvr2x's int8"
    " files, W8A8 with comfy-kitchen's rotation) or in seedvr2x's dynamic GGUF files (its block"
    " matrices in Q3_K, Q4_K and Q5_K), with its VAE in fp16; phase 2 brings the two 7Bs' other"
    " smaller files"
)
SHOTS = "seedvr2x detects shots with TransNetV2 in fp32, as its own transnetv2.safetensors holds it"
TAILS: dict[weights.Role, str] = {"dit": V1, "vae": V1, "detector": SHOTS}
# What a safetensors file of other tensors is not.
NONE_OF_THEM = "neither SeedVR2's DiT, its VAE nor TransNetV2"


@functools.cache
def vendored(model: str) -> dict[str, tuple[int, ...]]:
    """The shapes of the vendored model's tensors, by name: its state_dict, built on the meta
    device with tests/test_model.py's BUILD calls; "7b" and "3b" the DiTs of configs_7b and
    configs_3b, "vae" configs_7b's VAE merged with its architecture's config, as
    seedvr2x/model.py's load_models merges it. Built once per session, as a module-scoped fixture
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


@functools.cache
def transnetv2() -> Tensors:
    """TransNetV2's tensors as the vendored model's state_dict gives them, built on the meta
    device, by name: F32, its BatchNorm counters I64, as transnetv2.safetensors holds them."""
    import torch

    from seedvr2x.transnetv2.transnetv2_pytorch import TransNetV2

    with torch.device("meta"):
        built = TransNetV2()
    dtypes = {torch.float32: "F32", torch.int64: "I64"}
    return {name: (dtypes[t.dtype], tuple(t.shape)) for name, t in built.state_dict().items()}


def transnetv2_with(
    name: str, dtype: str | None = None, shape: tuple[int, ...] | None = None
) -> Tensors:
    """TransNetV2's tensors, the one named name in another dtype or shape."""
    tensors = dict(transnetv2())
    found, size = tensors[name]
    tensors[name] = (dtype or found, shape or size)
    return tensors


# The metadata of seedvr2x's transnetv2.safetensors, as models/transnetv2_weights.py writes it,
# in part: the check reads none of it.
TRANSNETV2_METADATA = {"format": "pt", "license": "mit"}


def write(
    path: Path,
    tensors: Tensors,
    metadata: Mapping[str, str] | None = None,
    data: Mapping[str, bytes] | None = None,
) -> Path:
    """A safetensors file of these tensors, their data contiguous in the order given: its header,
    with __metadata__ when given, then nothing but its full length, by truncate: a sparse file
    (marked so on NTFS, library_fixtures.sparse), which costs no disk (the 7B in fp16, 16.48 GB
    on paper); the check reads the header alone, and an int8 file's markers, whose bytes `data`
    gives by name, written in their places."""
    header: dict[str, object] = {} if metadata is None else {"__metadata__": dict(metadata)}
    offset = 0
    starts: dict[str, int] = {}
    for name, (dtype, shape) in tensors.items():
        end = offset + math.prod(shape) * BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        starts[name] = offset
        offset = end
    raw = json.dumps(header).encode()
    with sparse(path.open("wb")) as file:
        file.write(len(raw).to_bytes(8, "little") + raw)
        grow(file, 8 + len(raw) + offset)
        for name, content in (data or {}).items():
            file.seek(8 + len(raw) + starts[name])
            file.write(content)
    return path


def accepted_models(directory: Path) -> list[str]:
    """SeedVR2's 7B DiT and its VAE in fp16, synthetic (write), in directory under numz's names,
    which seedvr2x doesn't pin (seedvr2x/pull.py): models the header check accepts and no pin
    refuses, for the CLI's tests running it for real in a process of their own, which imports no
    torch. Returns the options naming them."""
    dit, vae = "seedvr2_ema_7b_sharp_fp16.safetensors", "ema_vae_fp16.safetensors"
    write(directory / dit, as_dtype("7b", "F16"))
    write(directory / vae, as_dtype("vae", "F16"))
    return ["--dit-model", dit, "--vae-model", vae]


# The 288 matrices of the 7B's blocks, 8 per block, which phase 2's files quantise (DESIGN.md,
# Weights, Phase 2): written here, the library's own (weights.MATRIX) held to it.
MATRIX = re.compile(r"blocks\.\d+\.(attn\.proj_(qkv|out)\.(vid|txt)|mlp\.(vid|txt)\.proj_(in|out))")
# The int8 files' marker, as models/seedvr2_int8.py writes it: 72 bytes.
INT8_MARKER = json.dumps(
    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
).encode()


def int8_file(path: Path, marker: bytes = INT8_MARKER, tensors: Tensors | None = None) -> Path:
    """The 7B in the int8 layout (phase2), synthetic, each marker's bytes `marker` written: a
    file the check accepts with the default marker. Its sparse data is 8.33 GB on paper."""
    held = phase2("int8_convrot") if tensors is None else tensors
    markers = {name: marker for name in held if name.endswith(".comfy_quant")}
    return write(path, held, {"format": "pt"}, markers)


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
    return cli_words(error.value)


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


def test_int8_digest() -> None:
    # The int8 files' architecture: the vendored 7B's tensors, each of its 288 block matrices'
    # weight with a scale a row, (rows, 1), and its marker of 72 bytes beside it, 1,704 in all;
    # both files' headers at c14a2bc4 give the same digest (weights.py says so). The library's
    # MATRIX is the test's own, which names exactly 288 of the 7B's weights.
    lines = []
    for name, shape in vendored("7b").items():
        lines.append(f"{name} {list(shape)}")
        layer = name.removesuffix(".weight")
        if name.endswith(".weight") and MATRIX.fullmatch(layer):
            lines += [f"{layer}.weight_scale {[shape[0], 1]}", f"{layer}.comfy_quant [72]"]
    assert len(lines) == DIT_7B_INT8.tensors == 1704
    assert hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest() == DIT_7B_INT8.digest
    assert weights.MATRIX.pattern == MATRIX.pattern
    assert len(INT8_MARKER) == 72 and json.loads(INT8_MARKER) == weights.MARKER
    assert DIT_7B_INT8.config == DIT_7B.config == "configs_7b"
    assert weights.ACCEPTED["dit"] == (DIT_7B, DIT_7B_INT8, DIT_7B_DYN, DIT_7B_SHARP_DYN)


def test_int8_bytes_are_its_headers() -> None:
    # A block's and the other parameters' bytes as the synthetic int8 file's header gives them,
    # its markers and the RoPE buffers left out (no parameters): weights.py's figures, which the
    # planner and the placement count by. The real headers gave the same (weights.py).
    tensors = phase2("int8_convrot")
    blocks = dict.fromkeys(range(36), 0)
    other = 0
    for name, (dtype, shape) in tensors.items():
        if name.endswith(".comfy_quant") or ".rope." in name:
            continue
        size = math.prod(shape) * BYTES[dtype]
        match = re.match(r"blocks\.(\d+)\.", name)
        if match is None:
            other += size
        else:
            blocks[int(match[1])] += size
    assert tuple(blocks.values()) == DIT_7B_INT8.blocks and other == DIT_7B_INT8.other
    assert f"{sum(DIT_7B_INT8.blocks) / 2**30:.2f}" == "7.61"
    assert f"{(sum(DIT_7B_INT8.blocks) + DIT_7B_INT8.other) / 2**30:.2f}" == "7.76"


def test_transnetv2_pinned() -> None:
    # TransNetV2's pin is the vendored model's inventory, as for the others; and the tensors of
    # seedvr2x's file, read from its header at c14a2bc4: 90, 84 F32 and its 6 BatchNorm counters
    # I64, 7,618,056 values.
    tensors = transnetv2()
    infos = {name: TensorInfo(dtype, shape) for name, (dtype, shape) in tensors.items()}
    assert len(infos) == weights.TRANSNETV2.tensors == 90
    assert weights.inventory(infos) == weights.TRANSNETV2.digest
    counters = sorted(name for name in tensors if name.endswith(weights.COUNTER))
    assert counters == [
        f"SDDCNN.{i}.DDCNN.{j}.bn.num_batches_tracked" for i in range(3) for j in (0, 1)
    ]
    assert sorted(name for name, (dtype, _) in tensors.items() if dtype == "I64") == counters
    assert sum(math.prod(shape) for _, shape in tensors.values()) == 7_618_056
    assert tensors["fc1.weight"] == ("F32", (1024, 4864))
    assert tensors["SDDCNN.0.DDCNN.0.Conv3D_1.layers.0.weight"] == ("F32", (32, 3, 1, 3, 3))


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
    "name",
    [
        "seedvr2x_ema_7b_int8_convrot.safetensors",
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors",
        "renamed_fp16.safetensors",
    ],
)
def test_int8_accepted(tmp_path: Path, name: str) -> None:
    # seedvr2x's int8 files, whatever their name, each marker read where the header puts it.
    path = int8_file(tmp_path / name)
    found = weights.check(path, "dit")
    assert found is DIT_7B_INT8 and found.config == "configs_7b"
    said = "SeedVR2's 7B DiT in int8, W8A8 with comfy-kitchen's rotation"
    assert weights.describe(path) == weights.in_words(found) == said
    assert weights.in_words(DIT_7B) == "SeedVR2's 7B DiT in fp16"
    # The dtypes it runs them in, by name.
    assert [
        DIT_7B_INT8.dtype_of(name)
        for name in (
            "blocks.3.mlp.vid.proj_in.weight",
            "blocks.3.mlp.vid.proj_in.weight_scale",
            "blocks.3.mlp.vid.proj_in.comfy_quant",
            "blocks.3.mlp.vid.proj_in.bias",
            "blocks.3.ada.vid.attn_gate",
            "vid_in.proj.weight",
        )
    ] == ["I8", "F32", "U8", "F16", "F16", "F16"]


def test_int8_marked_otherwise_refused(tmp_path: Path) -> None:
    # A marker of the same 72 bytes saying another rotation, or not JSON, refused, naming the
    # layer and what it says; the other markers right.
    other = INT8_MARKER.replace(b"256", b"128")
    assert len(other) == 72
    path = int8_file(tmp_path / "seedvr2x_ema_7b_int8_convrot.safetensors", other)
    expected = json.dumps(weights.MARKER)
    assert refused(path, "dit") == (
        f"{path}: SeedVR2's 7B DiT in comfy-kitchen's INT8 layout, its blocks.0.attn.proj_out.txt"
        f" marked {other.decode()}, not {expected}: {V1}"
    )
    tensors = phase2("int8_convrot")
    markers = {name: INT8_MARKER for name in tensors if name.endswith(".comfy_quant")}
    markers["blocks.7.mlp.vid.proj_out.comfy_quant"] = b"\xff" * 72
    write(path, tensors, None, markers)
    message = refused(path, "dit")
    assert message.startswith(
        f"{path}: SeedVR2's 7B DiT in comfy-kitchen's INT8 layout, its blocks.7.mlp.vid.proj_out"
        " marked \\xff"
    )
    # Its scales in fp16, the int8 file's tensors otherwise: what it is, refused.
    halved = {
        name: ("F16" if name.endswith(".weight_scale") else dtype, shape)
        for name, (dtype, shape) in tensors.items()
    }
    write(path, halved)
    assert refused(path, "dit") == (
        f"{path}: SeedVR2's 7B DiT in comfy-kitchen's INT8 layout, in other dtypes (1,128 F16, 288"
        f" I8, 288 U8): {V1}"
    )


@pytest.mark.parametrize(
    ("name", "metadata"),
    [("ema_vae_fp16.safetensors", None), ("seedvr2x_ema_vae_fp16.safetensors", {"format": "pt"})],
)
def test_vae_accepted(tmp_path: Path, name: str, metadata: dict[str, str] | None) -> None:
    path = write(tmp_path / name, as_dtype("vae", "F16"), metadata)
    assert weights.check(path, "vae") is VAE
    assert weights.describe(path) == "SeedVR2's VAE in fp16"


@pytest.mark.parametrize("metadata", [None, TRANSNETV2_METADATA], ids=["bare", "seedvr2x"])
def test_transnetv2_accepted(tmp_path: Path, metadata: dict[str, str] | None) -> None:
    # As the shot detector, in its own role only, whatever its name; its counters said nowhere.
    path = write(tmp_path / "transnetv2.safetensors", transnetv2(), metadata)
    assert weights.check(path, "detector") is weights.TRANSNETV2
    assert weights.describe(path) == "TransNetV2 in fp32"
    renamed = write(tmp_path / "seedvr2x_ema_vae_fp16.safetensors", transnetv2(), metadata)
    assert weights.check(renamed, "detector") is weights.TRANSNETV2


# seedvr2x's dynamic GGUF files (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too"): the
# test's own reading of each architecture's `types`, apart from the library's (weights.GGUF_ORDER,
# GGUF_TYPES): a block's 8 matrices in this order, each digit a ggml type.
MATRIX_ORDER = (
    "attn.proj_qkv.vid",
    "attn.proj_qkv.txt",
    "attn.proj_out.vid",
    "attn.proj_out.txt",
    "mlp.vid.proj_in",
    "mlp.vid.proj_out",
    "mlp.txt.proj_in",
    "mlp.txt.proj_out",
)
GGML = {"3": "Q3_K", "4": "Q4_K", "5": "Q5_K"}
DYNAMIC = [DIT_7B_DYN, DIT_7B_SHARP_DYN]
DYNAMIC_IDS = ["7b", "sharp"]


def gguf_specs(
    architecture: Architecture | None = None, matrices: str | None = None, model: str = "7b"
) -> list[Spec]:
    """The vendored model's tensors as a GGUF file lists them: F16, but for the block matrices,
    those of `architecture` in its types (the test's own reading of `types`), or all `matrices`
    when given, a ggml type's name."""
    specs: list[Spec] = []
    for name, shape in vendored(model).items():
        layer = name.removesuffix(".weight")
        kind = "F16"
        if name.endswith(".weight") and MATRIX.fullmatch(layer):
            if matrices is not None:
                kind = matrices
            elif architecture is not None:
                block, _, module = layer.removeprefix("blocks.").partition(".")
                kind = GGML[architecture.types[int(block)][MATRIX_ORDER.index(module)]]
        specs.append((name, kind, shape))
    return specs


def dynamic_file(path: Path, architecture: Architecture) -> Path:
    """The 7B in the dynamic GGUF layout of this architecture, synthetic: a sparse file, 4.76 GB
    on paper, its header as the real files' (version 3, the default alignment)."""
    return write_gguf(path, gguf_specs(architecture), [entry("general.architecture", 8, "seedvr")])


def test_dynamic_pins() -> None:
    # The types pinned: 36 blocks of 8 digits, 3, 4 or 5, the files' counts (17 Q3_K, 203 and 204
    # Q4_K, 68 and 67 Q5_K), the two files differing on 5 matrices; their names and shapes DIT_7B's.
    for architecture, counts in ((DIT_7B_DYN, (17, 203, 68)), (DIT_7B_SHARP_DYN, (17, 204, 67))):
        assert len(architecture.types) == 36 and all(len(each) == 8 for each in architecture.types)
        digits = "".join(architecture.types)
        assert (digits.count("3"), digits.count("4"), digits.count("5")) == counts
        assert set(digits) == {"3", "4", "5"} and len(digits) == 288
        assert architecture.tensors == DIT_7B.tensors == 1128
        assert architecture.digest == DIT_7B.digest and architecture.config == "configs_7b"
        assert architecture.layout == weights.GGUF == "gguf"
        assert architecture.matrix_types() == ["Q3_K", "Q4_K", "Q5_K"]
        assert len(architecture.blocks) == 36
    regular, sharp = "".join(DIT_7B_DYN.types), "".join(DIT_7B_SHARP_DYN.types)
    assert sum(1 for a, b in zip(regular, sharp, strict=True) if a != b) == 5
    assert weights.GGUF_ORDER == MATRIX_ORDER and weights.GGUF_TYPES == GGML
    assert DIT_7B_DYN.name == "SeedVR2's 7B DiT"
    assert DIT_7B_SHARP_DYN.name == "SeedVR2's sharp 7B DiT"


@pytest.mark.parametrize("architecture", DYNAMIC, ids=DYNAMIC_IDS)
def test_dynamic_bytes_are_its_headers(architecture: Architecture) -> None:
    # A block's and the other parameters' bytes as a file of these types holds them (ggml's block
    # sizes: Q3_K 110 bytes to 256 values, Q4_K 144, Q5_K 176, F16 2 a value), the RoPE buffers
    # left out as the others': weights.py's figures, which the real files' headers gave too
    # (test_dynamic_real_headers).
    blocks = dict.fromkeys(range(36), 0)
    other = 0
    for name, kind, shape in gguf_specs(architecture):
        if ".rope." in name:
            continue
        values, size = gguf_file.BLOCKS[kind]
        match = re.match(r"blocks\.(\d+)\.", name)
        if match is None:
            other += math.prod(shape) // values * size
        else:
            blocks[int(match[1])] += math.prod(shape) // values * size
    assert tuple(blocks.values()) == architecture.blocks and other == architecture.other
    assert len(set(architecture.blocks)) > 1  # they differ from block to block
    # Under a third of the fp16 file's, a swap moves less than that.
    assert sum(architecture.blocks) < sum(DIT_7B.blocks) / 3
    assert sum(architecture.blocks) == (
        4_590_563_328 if architecture is DIT_7B_DYN else 4_590_637_056
    )


@pytest.mark.parametrize("architecture", DYNAMIC, ids=DYNAMIC_IDS)
def test_dynamic_dtype_of(architecture: Architecture) -> None:
    # Each tensor's pinned type, by name, equal to the test's own reading for the 1,128: the
    # matrices' weights in their types, every other tensor, a matrix's bias included, F16.
    for name, kind, _ in gguf_specs(architecture):
        assert architecture.dtype_of(name) == kind, name
    assert architecture.dtype_of("blocks.0.mlp.vid.proj_in.bias") == "F16"
    assert architecture.dtype_of("vid_in.proj.weight") == "F16"
    assert (
        architecture.dtype_of("blocks.35.attn.proj_qkv.vid.weight")
        == (GGML[architecture.types[35][0]])
    )
    # The safetensors architectures know no GGUF type.
    assert DIT_7B.dtype_of("blocks.0.attn.proj_qkv.vid.weight") == "F16"
    assert DIT_7B_INT8.dtype_of("blocks.0.attn.proj_qkv.vid.weight") == "I8"


@pytest.mark.parametrize("architecture", DYNAMIC, ids=DYNAMIC_IDS)
@pytest.mark.parametrize("name", ["seedvr2x_ema_7b_dyn.gguf", "renamed.gguf", "x.safetensors"])
def test_dynamic_accepted(tmp_path: Path, architecture: Architecture, name: str) -> None:
    # Known by its tensors, names, shapes and the ggml type of each, whatever the file is called:
    # the regular 7B's file the regular's, the sharp 7B's the sharp's.
    path = dynamic_file(tmp_path / name, architecture)
    found = weights.check(path, "dit")
    assert found is architecture and found.config == "configs_7b"
    said = (
        f"{architecture.name} in seedvr2x's dynamic GGUF, its block matrices in Q3_K, Q4_K and Q5_K"
    )
    assert weights.describe(path) == weights.in_words(found) == said
    # Read as any model file, its tensors' offsets where the data is.
    tensors = weights._read(path)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tensors, dict) and len(tensors) == 1128
    header = gguf_file.read(path)
    assert [info.at for info in tensors.values()] == [each.offset for each in header.tensors]
    assert tensors["blocks.0.attn.proj_qkv.vid.weight"].dtype == GGML[architecture.types[0][0]]
    assert tensors["blocks.0.attn.proj_qkv.vid.weight"].shape == (9216, 3072)
    # Given as the VAE or as the shot detector, it says what it is.
    assert refused(path, "vae") == f"{path}: {said}, given as the VAE (--vae-model): {V1}"
    assert refused(path, "detector").startswith(f"{path}: {said}, given as the shot detector")


def test_dynamic_files_told_apart(tmp_path: Path) -> None:
    # The two share every name and shape and differ on 5 matrices' types: a file of the regular's
    # types but one of the sharp's is neither.
    regular, sharp = "".join(DIT_7B_DYN.types), "".join(DIT_7B_SHARP_DYN.types)
    index = next(i for i in range(288) if regular[i] != sharp[i])
    specs = gguf_specs(DIT_7B_DYN)
    matrices = [
        i for i, (name, _, _) in enumerate(specs) if MATRIX.fullmatch(name[: -len(".weight")])
    ]
    assert len(matrices) == 288
    name, _, shape = specs[matrices[index]]
    assert name.startswith(f"blocks.{index // 8}.")
    specs[matrices[index]] = (name, GGML[sharp[index]], shape)
    path = write_gguf(tmp_path / "x.gguf", specs)
    what = refused(path, "dit")
    assert "not one of seedvr2x's dynamic files" in what and weights.describe(path) in what


def test_dynamic_check_models(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dit = dynamic_file(tmp_path / "seedvr2x_ema_7b_sharp_dyn.gguf", DIT_7B_SHARP_DYN)
    vae = write(tmp_path / "ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    with caplog.at_level(logging.INFO):
        weights.check_models(dit, vae)
    said = "SeedVR2's sharp 7B DiT in seedvr2x's dynamic GGUF, its block matrices in Q3_K, Q4_K"
    assert f"DiT {dit.name}: {said} and Q5_K" in caplog.messages


def test_other_ggml_types_refused(tmp_path: Path) -> None:
    # A GGUF file of the 7B's own tensors in other types is not one of seedvr2x's: said by what it
    # is, by content, before torch. numz's Q4_K_M and seedvr2x's static Q4_K and Q8_0 files are
    # such (phase 2 brings them), and so is a file of fp16 in GGUF.
    base = gguf_specs(DIT_7B_DYN)
    first = next(i for i, (name, kind, _) in enumerate(base) if kind == "Q4_K")
    changed = list(base)
    changed[first] = (base[first][0], "Q5_K", base[first][2])
    cases = {
        "one-matrix-changed": (changed, "Q4_K (202), Q5_K (69) and Q3_K (17)"),
        "all-q4_k": (gguf_specs(matrices="Q4_K"), "Q4_K (288)"),
        "all-q8_0": (gguf_specs(matrices="Q8_0"), "Q8_0 (288)"),
        "all-q6_k": (gguf_specs(matrices="Q6_K"), "Q6_K (288)"),
        "all-f16": (gguf_specs(), "F16 (288)"),
    }
    for name, (specs, kinds) in cases.items():
        path = write_gguf(tmp_path / f"{name}.gguf", specs)
        what = (
            f"a GGUF file of SeedVR2's 7B DiT, its block matrices in {kinds},"
            " not one of seedvr2x's dynamic files"
        )
        assert weights.describe(path) == what
        assert refused(path, "dit") == f"{path}: {what}: {V1}"
        assert refused(path, "vae") == f"{path}: {what}, given as the VAE (--vae-model): {V1}"
        assert refused(path, "detector") == f"{path}: {what}, given as the shot detector: {SHOTS}"


def test_gguf_of_other_tensors_refused(tmp_path: Path) -> None:
    # The 3B's tensors in Q8_0 (numz's), a DiT's names with one shape changed, tensors of no model.
    quantised: list[Spec] = [
        (
            name,
            "Q8_0"
            if name.startswith("blocks.") and name.endswith(".weight") and len(shape) == 2
            else "F16",
            shape,
        )
        for name, shape in vendored("3b").items()
    ]
    three = write_gguf(tmp_path / "three.gguf", quantised)
    what = weights.describe(three)
    assert what.startswith("a GGUF file of SeedVR2's 3B DiT, its block matrices in Q8_0 (")
    assert "dynamic" not in what
    assert refused(three, "dit") == f"{three}: {what}: {V1}"
    reshaped = [
        (name, kind, (3072, 4096) if name == "txt_in.weight" else shape)
        for name, kind, shape in gguf_specs(DIT_7B_DYN)
    ]
    path = write_gguf(tmp_path / "reshaped.gguf", reshaped)
    assert weights.describe(path) == (
        "a GGUF file of SeedVR2's 7B DiT by its blocks and width, with other tensors than its own"
        " (1,128 tensors: 840 F16, 203 Q4_K, 68 Q5_K, 17 Q3_K)"
    )
    path = write_gguf(tmp_path / "other.gguf", [("x.weight", "F32", (8,)), ("y", "Q4_0", (2, 32))])
    what = f"a GGUF file of 2 tensors (1 F32, 1 Q4_0), {NONE_OF_THEM}"
    assert weights.describe(path) == what
    for role in ("dit", "vae", "detector"):
        assert refused(path, role) == f"{path}: {what}: {TAILS[role]}"
    path = write_gguf(tmp_path / "none.gguf", [])
    assert weights.describe(path) == f"a GGUF file of no tensors, {NONE_OF_THEM}"
    # 1,128 F16 tensors under DIT_7B's names are DIT_7B's in safetensors, not in GGUF.
    path = write_gguf(tmp_path / "f16.gguf", gguf_specs())
    assert weights.describe(path).endswith("not one of seedvr2x's dynamic files")


def test_gguf_versions_and_types_refused(tmp_path: Path) -> None:
    # A GGUF file seedvr2x doesn't read is said so, as what it is, with V1's tail.
    specs = gguf_specs(DIT_7B_DYN)
    for version in (2, 1):
        path = write_gguf(tmp_path / f"v{version}.gguf", specs[:3], version=version)
        what = f"a GGUF file of version {version}, where seedvr2x reads 3"
        assert weights.describe(path) == what
        assert refused(path, "dit") == f"{path}: {what}: {V1}"
    # An id ggml removed (4, Q4_2): the second tensor's type is the u32 before its offset, 12
    # bytes from the end of the header's body.
    path = tmp_path / "t.gguf"
    raw = bytearray(write_gguf(path, specs[:2]).read_bytes())
    end = len(body(specs[:2]))
    raw[end - 12 : end - 8] = (4).to_bytes(4, "little")
    path.write_bytes(bytes(raw))
    what = f"a GGUF file where {specs[1][0]} is of an unknown ggml type 4"
    assert refused(path, "dit") == f"{path}: {what}: {V1}"


def test_gguf_cut_or_damaged_refused(tmp_path: Path) -> None:
    # The data cut: the size said, the download fetched again. The header cut inside. An offset
    # inside another tensor's data: a damaged file.
    path = dynamic_file(tmp_path / "seedvr2x_ema_7b_dyn.gguf", DIT_7B_DYN)
    header = gguf_file.read(path)
    full = path.stat().st_size
    os.truncate(path, header.data_start + 1_000_000)
    assert refused(path, "dit") == (
        f"{path}: cut short, {header.data_start + 1_000_000:,} of {full:,} bytes; fetch it again"
    )
    os.truncate(path, 5_000)
    said = refused(path, "dit")
    assert said.startswith(f"{path}: cut short inside its header, ") and said.endswith(
        "fetch it again"
    )
    specs = gguf_specs(DIT_7B_DYN)[:3]
    path = write_gguf(tmp_path / "overlap.gguf", specs, offsets=[0, 32, 64])
    assert refused(path, "dit").startswith(f"{path}: a damaged GGUF file: ")
    with pytest.raises(ModelError, match="a damaged GGUF file"):
        weights.describe(path)


def test_gguf_header_read_without_torch(tmp_path: Path) -> None:
    path = dynamic_file(tmp_path / "seedvr2x_ema_7b_dyn.gguf", DIT_7B_DYN)
    code = (
        "import sys\nfrom pathlib import Path\nfrom seedvr2x import weights\n"
        "assert weights.check(Path(sys.argv[1]), 'dit') is weights.DIT_7B_DYN\n"
        "assert 'torch' not in sys.modules and 'gguf' not in sys.modules, 'imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


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

    from seedvr2x import model

    loaded: list[str] = []

    def load_weights(
        module: object, path: Path, *args: object, **more: object
    ) -> dict[str, object]:
        loaded.append(path.name)
        return {}  # no part away from the device (model._load_weights)

    monkeypatch.setattr(model, "_load_weights", load_weights)
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


def test_load_models_builds_the_int8_dit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An int8 file: the 7B built from its config, each of its 288 block matrices an Int8Linear
    # (seedvr2x/int8.py) before its weights are read, its markers left unread; on the meta device
    # here, the weights not loaded. The order of the model's parameters as the fp16 7B's, the
    # first one the same (model.sample's autocast goes by it).
    import torch

    from seedvr2x import int8, model

    loaded: list[tuple[str, object, object]] = []

    def load_weights(module: Any, path: Path, *args: Any, **more: Any) -> dict[str, object]:
        loaded.append((path.name, module, args[3] if len(args) > 3 else None))
        return {}

    monkeypatch.setattr(model, "_load_weights", load_weights)
    dit = int8_file(tmp_path / "seedvr2x_ema_7b_int8_convrot.safetensors")
    write(tmp_path / "ema_vae_fp16.safetensors", as_dtype("vae", "F16"))
    models = model.load_models(dit, tmp_path / "ema_vae_fp16.safetensors", torch.device("cpu"))
    assert [name for name, *_ in loaded] == [
        "ema_vae_fp16.safetensors",
        "seedvr2x_ema_7b_int8_convrot.safetensors",
    ]
    assert loaded[0][2] is None and loaded[1][2] is model._marker  # pyright: ignore[reportPrivateUsage]
    built = models.runner.dit
    assert built is loaded[1][1]
    swapped = [name for name, m in built.named_modules() if isinstance(m, int8.Int8Linear)]
    assert len(swapped) == 288 and all(MATRIX.fullmatch(name) for name in swapped)
    # The file's tensors but the markers, by name and shape; the quantised ones in their dtypes,
    # as built (the others take the file's at the load).
    state = {name: tensor for name, tensor in built.state_dict().items()}
    held = phase2("int8_convrot")
    assert {name: tuple(tensor.shape) for name, tensor in state.items()} == {
        name: shape for name, (_, shape) in held.items() if not name.endswith(".comfy_quant")
    }
    dtypes = {"F32": torch.float32, "I8": torch.int8}
    for name, (dtype, _) in held.items():
        if dtype in dtypes:
            assert state[name].dtype == dtypes[dtype], name
    fp16 = list(vendored("7b"))
    parameters = [name for name, _ in built.named_parameters()]
    assert parameters[0] == "vid_in.proj.weight"
    assert [name for name in parameters if not name.endswith(".weight_scale")] == [
        name for name in fp16 if name in parameters
    ]


QUANTISED = "SeedVR2's 7B DiT quantised, its block matrices in {} with scale tensors beside them,"
QUANTISED += " in comfy-kitchen's layout ({})"
INT8_SAID = "SeedVR2's 7B DiT in int8, W8A8 with comfy-kitchen's rotation"


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
            "vae",
            f"{INT8_SAID}, given as the VAE (--vae-model)",
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
            f"a safetensors file of 90 tensors (84 F32, 6 I64), {NONE_OF_THEM}",
        ),
        # TransNetV2 given as another model, and the others given as the shot detector.
        (
            "transnetv2.safetensors",
            transnetv2,
            "dit",
            "TransNetV2 in fp32, given as the DiT (--dit-model)",
        ),
        (
            "transnetv2.safetensors",
            transnetv2,
            "vae",
            "TransNetV2 in fp32, given as the VAE (--vae-model)",
        ),
        (
            "seedvr2x_ema_7b_sharp_fp16.safetensors",
            lambda: as_dtype("7b", "F16"),
            "detector",
            "SeedVR2's 7B DiT in fp16, given as the shot detector",
        ),
        (
            "seedvr2x_ema_vae_fp16.safetensors",
            lambda: as_dtype("vae", "F16"),
            "detector",
            "SeedVR2's VAE in fp16, given as the shot detector",
        ),
        (
            "seedvr2_ema_3b_fp16.safetensors",
            lambda: as_dtype("3b", "F16"),
            "detector",
            "SeedVR2's 3B DiT in fp16, given as the shot detector",
        ),
        (
            "seedvr2x_ema_7b_int8_convrot.safetensors",
            lambda: phase2("int8_convrot"),
            "detector",
            f"{INT8_SAID}, given as the shot detector",
        ),
        # TransNetV2 in other dtypes, or with a tensor of another shape.
        (
            "transnetv2.safetensors",
            lambda: {
                name: ("I64" if dtype == "I64" else "F16", shape)
                for name, (dtype, shape) in transnetv2().items()
            },
            "detector",
            "TransNetV2 in fp16",
        ),
        (
            "transnetv2.safetensors",
            lambda: transnetv2_with("fc1.weight", dtype="F16"),
            "detector",
            "TransNetV2 in mixed precision (83 tensors F32, 1 F16)",
        ),
        (
            "transnetv2.safetensors",
            lambda: {name: ("F32", shape) for name, (_, shape) in transnetv2().items()},
            "detector",
            "TransNetV2 in fp32, its BatchNorm counters 6 F32 where PyTorch keeps them I64",
        ),
        (
            "transnetv2.safetensors",
            lambda: transnetv2_with("fc1.weight", shape=(1024, 4096)),
            "detector",
            f"a safetensors file of 90 tensors (84 F32, 6 I64), {NONE_OF_THEM}",
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
        "int8-as-vae",
        "nvfp4",
        "7b-a-shape-changed",
        "vae-as-dit",
        "7b-as-vae",
        "nvfp4-as-vae",
        "other-safetensors",
        "transnetv2-as-dit",
        "transnetv2-as-vae",
        "7b-as-detector",
        "vae-as-detector",
        "3b-as-detector",
        "int8-as-detector",
        "transnetv2-fp16",
        "transnetv2-one-f16",
        "transnetv2-counters-f32",
        "transnetv2-a-shape-changed",
    ],
)
def test_refused_saying_what(
    tmp_path: Path, name: str, tensors: Callable[[], Tensors], role: weights.Role, what: str
) -> None:
    path = write(tmp_path / name, tensors())
    assert refused(path, role) == f"{path}: {what}: {TAILS[role]}"


def test_other_formats(tmp_path: Path) -> None:
    # Known by their first bytes: a zip (torch.save's format), a pickle (its earlier one); random
    # bytes are neither. A GGUF file is told by its signature first and read as one (below): its
    # version, then its tensor count, the 3B's 635 = 0x027B putting safetensors' brace at byte 8.
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
    expected = {
        "seedvr2_ema_7b.pth": checkpoint.format("zip"),
        "ema_vae.pth": checkpoint.format("pickle"),
        "noise.safetensors": "neither safetensors, GGUF nor a PyTorch checkpoint",
    }
    for name, what in expected.items():
        path = tmp_path / name
        assert weights.describe(path) == what
        for role in ("dit", "vae", "detector"):
            assert refused(path, role) == f"{path}: {what}: {TAILS[role]}"
    # The two GGUF stubs claim tensors their 24 bytes cannot hold: a file cut short, said so, with
    # the advice of a download cut short and none of V1's tail.
    for name, tensors in (("seedvr2_ema_7b-Q4_K_M.gguf", 1128), ("seedvr2_ema_3b-Q8_0.gguf", 635)):
        path = tmp_path / name
        said = (
            f"{path}: cut short inside its header, {tensors:,} tensors claimed and 0 bytes left;"
            f" {weights.AGAIN}"
        )
        for role in ("dit", "vae", "detector"):
            assert refused(path, role) == said
        with pytest.raises(ModelError) as error:
            weights.describe(path)
        assert str(error.value) == said


def test_missing(tmp_path: Path) -> None:
    # Nothing there, or something that isn't a file to read: a directory, a FIFO (never opened,
    # which would wait for a writer).
    (tmp_path / "a directory").mkdir()
    cases = [("x.safetensors", "no such model file"), ("a directory", "not a regular file")]
    if sys.platform != "win32":  # Windows has no os.mkfifo (native, 2026-11-10)
        posix_os.mkfifo(tmp_path / "a fifo")
        cases.append(("a fifo", "not a regular file"))
    for name, what in cases:
        path = tmp_path / name
        assert refused(path, "dit") == f"{path}: {what}"
        with pytest.raises(ModelError, match=what):
            weights.describe(path)


def test_a_link_the_system_cant_follow_is_unreadable_never_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Natively on Windows a symbolic link made under WSL can't be followed: pathlib's is_file
    # raises "The system cannot access the file" (WinError 1920) rather than answer, which ended a
    # picture's plan in a traceback (impl-win-2, 2026-11-09; reproduced natively 2026-11-09).
    target = tmp_path / "elsewhere.safetensors"
    target.write_bytes(b"")
    link = tmp_path / "seedvr2x_ema_7b_sharp_fp16.safetensors"
    link.symlink_to(target)
    plain = tmp_path / "seedvr2x_ema_vae_fp16.safetensors"
    plain.write_bytes(b"")
    real = Path.is_file

    # Its options passed on as given: is_file takes follow_symlinks from Python 3.13 alone.
    def inaccessible(self: Path, **options: Any) -> bool:
        if self.name.startswith("seedvr2x_"):
            raise OSError(22, "The system cannot access the file", str(self))
        return real(self, **options)

    monkeypatch.setattr(Path, "is_file", inaccessible)
    assert refused(link, "dit") == (
        f"{link}: The system cannot access the file; a symbolic link this system can't follow (one"
        " made under WSL, on Windows): the file itself, or a hard link to it, will do"
    )
    assert refused(plain, "vae") == f"{plain}: The system cannot access the file"
    # Windows' own error for a WSL link, which its Python takes for no symbolic link.
    wsl = tmp_path / "seedvr2x_ema_7b_fp16.safetensors"
    wsl.write_bytes(b"")

    def unfollowable(self: Path, *, follow_symlinks: bool = True) -> bool:
        error = OSError(22, "The system cannot access the file", str(self))
        error.winerror = weights.CANT_ACCESS_FILE  # type: ignore[attr-defined]  # Windows' own
        raise error

    monkeypatch.setattr(Path, "is_file", unfollowable)
    assert refused(wsl, "dit").endswith("the file itself, or a hard link to it, will do")
    with pytest.raises(ModelError) as error:
        weights.check(link, "dit")
    assert type(error.value).__name__ == "ModelUnreadable"


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
    for role in ("dit", "vae", "detector"):
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
        for role in ("dit", "vae", "detector"):
            assert refused(path, role) == said
        with pytest.raises(ModelError) as error:
            weights.describe(path)
        assert str(error.value) == said
    # From 9 bytes on a GGUF file is read as one: its version, here 0.
    path.write_bytes(b"GGUF" + bytes(5))
    assert weights.describe(path) == "a GGUF file of version 0, where seedvr2x reads 3"
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
    assert weights.describe(path) == f"a safetensors file of 1 tensor (1 U8), {NONE_OF_THEM}"


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
    with sparse(path.open("wb")) as file:
        file.write((100_000_000).to_bytes(8, "little") + b"{")
        grow(file, 8 + 100_000_000)
    assert refused(path, "dit").startswith(
        f"{path}: a damaged safetensors file: its header isn't JSON: Expecting property name"
    )


def test_unknown_dtype_read(tmp_path: Path) -> None:
    # A dtype the read doesn't size fails no read, and no check passes it.
    header = {"a": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}
    path = raw(tmp_path / "x.safetensors", header, 2)
    what = f"a safetensors file of 1 tensor (1 F4), {NONE_OF_THEM}"
    assert weights.describe(path) == what
    assert refused(path, "vae") == f"{path}: {what}: {V1}"


def test_safetensors_told_first(tmp_path: Path) -> None:
    # A header of 640 bytes, padded with spaces as safetensors' format allows: its length begins
    # as a pickle of protocol 2 does, 0x80 0x02, and the file is safetensors all the same.
    header = json.dumps(u8((0, 1))).encode()
    path = raw(tmp_path / "x.safetensors", header + b" " * (640 - len(header)), 1)
    with path.open("rb") as file:
        assert file.read(2) == b"\x80\x02"
    what = f"a safetensors file of 1 tensor (1 U8), {NONE_OF_THEM}"
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
    assert cli_words(error.value).splitlines() == [
        f"{tmp_path / vae}: SeedVR2's VAE in fp16, given as the DiT (--dit-model): {V1}",
        f"{tmp_path / dit}: SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model): {V1}",
    ]


def test_check_models_with_the_detector(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # The shot detector's file checked with the others when given, logged with what it is; every
    # file refused at once, the detector's last, the advice for its role in place of "fetch it
    # again".
    accepted_models(tmp_path)
    dit, vae = "seedvr2_ema_7b_sharp_fp16.safetensors", "ema_vae_fp16.safetensors"
    detector = write(tmp_path / "transnetv2.safetensors", transnetv2())
    with caplog.at_level(logging.INFO):
        weights.check_models(tmp_path / dit, tmp_path / vae, None, detector)
    assert "Shot detector transnetv2.safetensors: TransNetV2 in fp32" in caplog.messages
    cut = tmp_path / "cut" / "transnetv2.safetensors"
    cut.parent.mkdir()
    cut.write_bytes(detector.read_bytes()[:1000])
    advice: dict[weights.Role, str] = {"detector": "the detector's advice"}
    with pytest.raises(ModelError) as error:
        weights.check_models(tmp_path / vae, tmp_path / vae, advice, cut)
    header = 8 + int.from_bytes(detector.read_bytes()[:8], "little")
    assert cli_words(error.value).splitlines() == [
        f"{tmp_path / vae}: SeedVR2's VAE in fp16, given as the DiT (--dit-model): {V1}",
        f"{cut}: cut short inside its header, 1,000 bytes of at least {header:,}; the detector's"
        " advice",
    ]
    with pytest.raises(ModelError) as error:
        weights.check_models(tmp_path / dit, tmp_path / vae, None, tmp_path / dit)
    assert cli_words(error.value) == (
        f"{tmp_path / dit}: SeedVR2's 7B DiT in fp16, given as the shot detector: {SHOTS}"
    )


# In a fresh interpreter: the header read takes neither torch nor the safetensors library.
NO_TORCH = """
import sys
from pathlib import Path
from seedvr2x import weights
assert weights.check(Path(sys.argv[1]), "dit") is weights.DIT_7B
assert weights.check(Path(sys.argv[3]), "detector") is weights.TRANSNETV2
print(weights.describe(Path(sys.argv[2])))
assert "torch" not in sys.modules and "safetensors" not in sys.modules, "imported"
"""


def test_header_read_without_torch(tmp_path: Path) -> None:
    seven = write(tmp_path / "seedvr2x_ema_7b_fp16.safetensors", as_dtype("7b", "F16"))
    three = write(tmp_path / "seedvr2_ema_3b_fp16.safetensors", as_dtype("3b", "F16"))
    detector = write(tmp_path / "transnetv2.safetensors", transnetv2())
    result = subprocess.run(
        [sys.executable, "-c", NO_TORCH, str(seven), str(three), str(detector)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "SeedVR2's 3B DiT in fp16\n"


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")

# The real files, by name, as a role: None when accepted, else the words of their refusal.
GIVEN_AS_VAE = "SeedVR2's 7B DiT in fp16, given as the VAE (--vae-model)"
GIVEN_AS_DIT = "SeedVR2's VAE in fp16, given as the DiT (--dit-model)"
PHASE_2 = "SeedVR2's 7B DiT quantised, its block matrices in {} with scale tensors beside them"
GGUF = "a GGUF file"
OTHER = f"a safetensors file of {{}}, {NONE_OF_THEM}"
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
            ("nvfp4", "4-bit values packed two per byte"),
        )
    ),
    # seedvr2x's int8 files, each marker read.
    ("seedvr2x_ema_7b_int8_convrot.safetensors", "dit", None),
    ("seedvr2x_ema_7b_sharp_int8_convrot.safetensors", "dit", None),
    (
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors",
        "vae",
        "SeedVR2's 7B DiT in int8, W8A8 with comfy-kitchen's rotation, given as the VAE",
    ),
    *(
        (f"seedvr2x_ema_7b{sharp}_{kind}.gguf", "dit", GGUF)
        for sharp in ("", "_sharp")
        for kind in ("Q8_0", "Q4_K", "Q4_K_imatrix")
    ),
    # seedvr2x's dynamic files, read by their header.
    ("seedvr2x_ema_7b_dyn.gguf", "dit", None),
    ("seedvr2x_ema_7b_sharp_dyn.gguf", "dit", None),
    (
        "seedvr2x_ema_7b_sharp_dyn.gguf",
        "vae",
        "SeedVR2's sharp 7B DiT in seedvr2x's dynamic GGUF, its block matrices in Q3_K, Q4_K and"
        " Q5_K, given as the VAE",
    ),
    ("transnetv2.safetensors", "detector", None),
    ("transnetv2.safetensors", "dit", "TransNetV2 in fp32, given as the DiT (--dit-model)"),
    ("transnetv2.safetensors", "vae", "TransNetV2 in fp32, given as the VAE (--vae-model)"),
    ("seedvr2x_ema_7b_sharp_fp16.safetensors", "detector", "7B DiT in fp16, given as the shot"),
    ("seedvr2x_ema_vae_fp16.safetensors", "detector", "VAE in fp16, given as the shot detector"),
    ("seedvr2_ema_7b-Q4_K_M.gguf", "detector", GGUF),
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
        assert weights.check(path, role) in weights.ACCEPTED[role]
    else:
        message = refused(path, role)
        assert message.startswith(f"{path}: ") and refusal in message
        assert message.endswith(TAILS[role])


REAL_DYNAMIC = [
    ("seedvr2x_ema_7b_dyn.gguf", DIT_7B_DYN),
    ("seedvr2x_ema_7b_sharp_dyn.gguf", DIT_7B_SHARP_DYN),
]


@pytest.mark.parametrize(("name", "architecture"), REAL_DYNAMIC, ids=DYNAMIC_IDS)
def test_dynamic_real_headers(name: str, architecture: Architecture) -> None:
    # The pins are the files' own headers: every tensor's type, each block's bytes and the bytes
    # outside the blocks, read here once more from the header and held to weights.py's literals.
    if not MODELS or not (Path(MODELS) / name).is_file():
        pytest.skip(f"{name}: needs SEEDVR2X_MODEL_DIR")
    path = Path(MODELS) / name
    assert weights.check(path, "dit") is architecture
    header = gguf_file.read(path)
    assert len(header.tensors) == architecture.tensors
    blocks = dict.fromkeys(range(36), 0)
    other = 0
    digits = ["" for _ in range(36)]
    byname = {each.name: each for each in header.tensors}
    for each in header.tensors:
        layer = each.name.removesuffix(".weight")
        if each.name.endswith(".weight") and MATRIX.fullmatch(layer):
            assert each.type in ("Q3_K", "Q4_K", "Q5_K")
        else:
            assert each.type == "F16", each.name
        if ".rope." in each.name:
            continue
        match = re.match(r"blocks\.(\d+)\.", each.name)
        if match is None:
            other += each.nbytes
        else:
            blocks[int(match[1])] += each.nbytes
    for index in range(36):
        digits[index] = "".join(
            {v: k for k, v in GGML.items()}[byname[f"blocks.{index}.{module}.weight"].type]
            for module in MATRIX_ORDER
        )
    assert tuple(digits) == architecture.types
    assert tuple(blocks.values()) == architecture.blocks and other == architecture.other
