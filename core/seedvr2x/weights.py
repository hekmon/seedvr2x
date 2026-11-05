"""Model files, recognised by their tensors, not by their names (DESIGN.md, Weights): a file's
header is read and checked before anything else runs, without torch, so that a wrong file is
refused in a second, saying what it is, rather than failing after the first pass or running
unchecked.

v1 runs SeedVR2's 7B DiT, the regular or the sharp one, every tensor in fp16, in seedvr2x's int8
files (INT8 W8A8, comfy-kitchen's layout with its rotation: DESIGN.md, Weights, "int8 before
v0.1.0 is tagged") or in its dynamic GGUF files (a ggml type a block matrix: DESIGN.md, Weights,
"The dynamic GGUF before v0.1.0 too"), with its VAE in fp16, and detects shots with TransNetV2 in
fp32 (DESIGN.md, Input, Shot detection). A safetensors file is known by its header, read without
the safetensors library: the header's length, then the header, never the tensors' data but for an
int8 file's markers, 72 bytes for each of its 288 quantised layers, which say the layout its values
are in. A GGUF file's header is read by seedvr2x/gguf_file.py, none of its data."""

import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal, cast

from seedvr2x import gguf_file
from seedvr2x.errors import (
    Message,
    ModelDamaged,
    ModelError,
    ModelMissing,
    ModelsRefused,
    ModelUnknown,
    ModelUnreadable,
    Say,
    Setting,
    aside,
    worded,
)
from seedvr2x.gguf_file import GgufError

logger = logging.getLogger(__name__)

Role = Literal["dit", "vae", "detector"]


# The device buffers a DiT's swapped blocks stream through, one computing, one being filled
# (seedvr2x/blockswap.py; DESIGN.md, BlockSwap, rewritten): with a block's bytes, what a swapped
# block frees on the GPU, and what one or two don't.
SLOTS = 2


@dataclass(frozen=True)
class Architecture:
    """A model seedvr2x knows by its tensors: their count and their inventory's digest
    (inventory); the vendored config directory a DiT is built from, None for another model; and
    the dtype seedvr2x runs it in, every tensor's but a BatchNorm counter's (COUNTER), I64, and
    in a quantised layout (`layout`) but its quantised layers' (dtype_of)."""

    name: str
    tensors: int
    digest: str
    config: str | None
    dtype: str = "F16"
    # A DiT's blocks, each one's parameters in bytes as its file holds them, in the model's
    # order, and its parameters outside the blocks: what the placement moves between the GPU and
    # the host (seedvr2x/blockswap.py) and the planner counts, by the model's shapes
    # (tests/test_blockswap.py holds them to the vendored configs, tests/test_weights.py to the
    # files' headers). No block, and 0, for another model.
    blocks: tuple[int, ...] = ()
    other: int = 0
    # How the file holds the block matrices (MATRIX), None where every tensor is in dtype: INT8,
    # comfy-kitchen's INT8 layout with its rotation (seedvr2x/int8.py).
    layout: str | None = None
    # In the GGUF layout, each block's 8 matrices' ggml types, as a string of GGUF_ORDER's 8
    # digits a block (3, 4 or 5: GGUF_TYPES): the pin of the types, which the inventory's names and
    # shapes don't tell. Empty for another layout.
    types: tuple[str, ...] = ()

    def dtype_of(self, name: str) -> str:
        """The dtype seedvr2x runs the tensor of this name in: in the INT8 layout, a block
        matrix's weight I8, its scale F32 and its marker U8; in the GGUF layout, a block matrix's
        weight in the ggml type `types` pins for it (Q3_K, Q4_K or Q5_K); a BatchNorm counter
        I64; any other tensor the architecture's dtype."""
        if name.endswith(COUNTER):
            return "I64"
        layer, _, leaf = name.rpartition(".")
        if self.layout == INT8 and MATRIX.fullmatch(layer) and leaf in INT8_LEAVES:
            return INT8_LEAVES[leaf]
        if self.layout == GGUF and leaf == "weight" and MATRIX.fullmatch(layer):
            index, _, module = layer.removeprefix("blocks.").partition(".")
            return GGUF_TYPES[self.types[int(index)][GGUF_ORDER.index(module)]]
        return self.dtype

    def matrix_types(self) -> list[str]:
        """The ggml types of a GGUF layout's block matrices, each once, in name order."""
        return sorted({GGUF_TYPES[each] for block in self.types for each in block})


# The 288 attention and MLP matrices of the 7B's blocks, 8 a block, by their modules' names: the
# matrices phase 2's files quantise (DESIGN.md, Weights, Phase 2; models/seedvr2_fp8.py's BLOCK),
# and the int8 files' (seedvr2x/int8.py swaps them).
MATRIX = re.compile(r"blocks\.\d+\.(attn\.proj_(qkv|out)\.(vid|txt)|mlp\.(vid|txt)\.proj_(in|out))")
# comfy-kitchen's INT8 layout with its rotation, as seedvr2x's int8 files hold it
# (models/seedvr2_int8.py; models/FORMATS.md, INT8): each block matrix's `<layer>.weight` in int8,
# its rows rotated per GROUP columns, `<layer>.weight_scale` (rows, 1) in float32, and
# `<layer>.comfy_quant`, the marker, MARKER's bytes as U8; its bias, if any, F16 as every other
# tensor.
INT8 = "int8"
INT8_LEAVES = {"weight": "I8", "weight_scale": "F32", "comfy_quant": "U8"}
GROUP = 256
MARK = ".comfy_quant"
MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
# How such a file's values are said: what it is, and how it multiplies.
INT8_SAID = "in int8, W8A8 with comfy-kitchen's rotation"
GGUF_SAID = "in seedvr2x's dynamic GGUF"


# seedvr2x's dynamic GGUF files (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too"): the
# 1,128 tensors of the fp16 7B under the same names and shapes, each of the 288 block matrices'
# `<layer>.weight` quantised by ggml to a type of its own, every other tensor F16. GGUF_ORDER is
# the order of a block's 8 matrices in Architecture.types, GGUF_TYPES its digits' ggml types.
GGUF = "gguf"
GGUF_ORDER = (
    "attn.proj_qkv.vid",
    "attn.proj_qkv.txt",
    "attn.proj_out.vid",
    "attn.proj_out.txt",
    "mlp.vid.proj_in",
    "mlp.vid.proj_out",
    "mlp.txt.proj_in",
    "mlp.txt.proj_out",
)
GGUF_TYPES = {"3": "Q3_K", "4": "Q4_K", "5": "Q5_K"}


# A BatchNorm layer's count of the batches it trained on, num_batches_tracked, I64 in PyTorch's
# state_dict, which no inference reads: TransNetV2's file holds its six as the converter saved
# them (models/transnetv2_weights.py), and a precision is said of the other tensors.
COUNTER = ".num_batches_tracked"

# Each digest is inventory()'s, over the model's tensors' names and shapes: what the vendored
# model's state_dict gives, built on the meta device, from which tests/test_weights.py derives
# each; and what the real files' headers give: numz's 7B, sharp 7B and 7B fp8 files and seedvr2x's
# 7B and sharp 7B fp16 ones the 7B's, numz's 3B fp16 and fp8 files the 3B's, numz's and seedvr2x's
# fp16 VAE the VAE's, seedvr2x's transnetv2.safetensors TransNetV2's (read on 2026-10-09 from
# the file pinned at hekmon/seedvr2x's c14a2bc4: 84 tensors F32 and the 6 counters I64).
# The 7B: 36 blocks of 453,133,312 bytes in fp16 (0.4220 GiB: 226,566,656 values in 30
# parameters), 15.19 of its 15.35 GiB; the other 0.155 GiB are emb_in, txt_in, vid_in and vid_out.
DIT_7B = Architecture(
    "SeedVR2's 7B DiT",
    1128,
    "960321767ecf7a1aeee79faefb0dcfb00acf66d205170ab813fff5bd3d1f3b2f",
    "configs_7b",
    blocks=(453_133_312,) * 36,
    other=166_416_512,
)
# The 7B in seedvr2x's int8 files (DESIGN.md, Weights, "int8 before v0.1.0 is tagged"): the same
# 1,128 tensors, each of the 288 block matrices' weight in int8, with its scale and its marker
# beside it, 1,704 in all, the digest the vendored model's names and shapes give with those 576
# added (tests/test_weights.py) and both files' headers at hekmon/seedvr2x's c14a2bc4 (read on
# 2026-11-02 by range requests, not downloaded). Its 36 blocks of 226,862,080 bytes (0.2113 GiB:
# the matrices' 226,492,416 values in int8, their 55,296 rows' scales in float32, the other 22
# parameters in fp16), 7.61 of its 7.76 GiB; the 0.155 GiB outside the blocks the fp16 file's.
# The headers say so: each block's tensors 226,862,100 bytes with its RoPE buffer's 20, which
# isn't a parameter, and 166,416,512 bytes outside the blocks; the markers 20,736 bytes.
DIT_7B_INT8 = Architecture(
    "SeedVR2's 7B DiT",
    1704,
    "b67ac67e0f4dfe0d0cb777f26b3f4022829ada407897f29710c2eed29e7e8dde",
    "configs_7b",
    blocks=(226_862_080,) * 36,
    other=166_416_512,
    layout=INT8,
)
# The 7B in seedvr2x's dynamic GGUF file (seedvr2x_ema_7b_dyn.gguf), the regular 7B's: GGUF
# version 3, alignment 32, the 1,128 tensors of DIT_7B under its digest, 840 F16 and the 288 block
# matrices Q3_K (17), Q4_K (203) or Q5_K (68), `types` giving each one's; read from the file's own
# header (4,757,055,264 bytes) on 2026-11-04 by seedvr2x/gguf_file.py, and agreeing with gguf-py
# 0.19.0's GGUFReader. Each of its 36 blocks' bytes as the file holds them, the matrices' quantised
# blocks and the 22 other parameters in F16 (they differ from block to block and file to file),
# 4,590,563,328 bytes in all (4.28 GiB), the RoPE buffers (20 bytes a block, no parameters) left
# out as in the others; and 166,416,512 bytes outside the blocks, the fp16 file's.
DIT_7B_DYN = Architecture(
    "SeedVR2's 7B DiT",
    1128,
    "960321767ecf7a1aeee79faefb0dcfb00acf66d205170ab813fff5bd3d1f3b2f",
    "configs_7b",
    blocks=(
        113_689_600,
        124_896_256,
        123_716_608,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        128_730_112,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        124_896_256,
        117_449_728,
    ),
    other=166_416_512,
    layout=GGUF,
    types=(
        "44543343",
        "44554443",
        "44544443",
        "44554434",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44544444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554443",
        "44534433",
    ),
)

# The same in the sharp 7B's file (seedvr2x_ema_7b_sharp_dyn.gguf, 4,757,129,024 bytes): the
# same tensors and the same figures but for the types of 5 matrices (17 Q3_K, 204 Q4_K, 67 Q5_K),
# read from its header the same day; its blocks 4,590,637,056 bytes in all. The two are told apart
# by the types they pin, as the regular and the sharp 7B in fp16 cannot be: they have the same
# tensors.
DIT_7B_SHARP_DYN = Architecture(
    "SeedVR2's sharp 7B DiT",
    1128,
    "960321767ecf7a1aeee79faefb0dcfb00acf66d205170ab813fff5bd3d1f3b2f",
    "configs_7b",
    blocks=(
        112_509_952,
        121_136_128,
        123_716_608,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        124_896_256,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        124_896_256,
        128_730_112,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        129_909_760,
        124_896_256,
        117_449_728,
    ),
    other=166_416_512,
    layout=GGUF,
    types=(
        "44443343",
        "34554443",
        "44544443",
        "44554434",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554443",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554443",
        "44544444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554444",
        "44554443",
        "44534433",
    ),
)

# The 3B: 32 blocks, the first 10 of 317,266,944 bytes (0.295 GiB, their text and video halves
# apart), the other 22 of 158,633,472 (0.148 GiB, shared), 6.21 of its 6.32 GiB.
DIT_3B = Architecture(
    "SeedVR2's 3B DiT",
    635,
    "cc68bcef364ddcfa3b51f09a62e171dc9f69bb7743588c43d82aad94456801ac",
    "configs_3b",
    blocks=(317_266_944,) * 10 + (158_633_472,) * 22,
    other=120_345_728,
)
VAE = Architecture(
    "SeedVR2's VAE", 250, "090ead68b30f9e815c50f1bdd1ed9c522f669461d42888ed56cbf0e57fb863e1", None
)
TRANSNETV2 = Architecture(
    "TransNetV2",
    90,
    "007f515b2f0977ca26cdec87b41da10511f34c4c6138f60cb9547e63ccee1504",
    None,
    "F32",
)
ARCHITECTURES = (DIT_7B, DIT_7B_INT8, DIT_7B_DYN, DIT_7B_SHARP_DYN, DIT_3B, VAE, TRANSNETV2)

# What v1 runs in each role (DESIGN.md, Weights): the 7B DiT, the regular and the sharp one alike
# (one architecture, the same tensors), every tensor in F16, in the INT8 layout or, in the dynamic
# GGUF files (one architecture each, told by their types), F16 but for the block matrices; the VAE,
# every tensor in F16; and TransNetV2, the shot detector, as its official weights are converted,
# every tensor F32 but its counters (DESIGN.md, Input, Shot detection).
ACCEPTED: dict[Role, tuple[Architecture, ...]] = {
    "dit": (DIT_7B, DIT_7B_INT8, DIT_7B_DYN, DIT_7B_SHARP_DYN),
    "vae": (VAE,),
    "detector": (TRANSNETV2,),
}
# The end of every refusal of a readable file given as the DiT or the VAE, the 3B's included:
# phase 2 brings the two 7Bs' other smaller files, and no 3B (DESIGN.md, Weights, Phase 2).
V1 = (
    "seedvr2x v1 runs SeedVR2's 7B DiT, regular or sharp, in fp16, in int8 (seedvr2x's int8"
    " files, W8A8 with comfy-kitchen's rotation) or in seedvr2x's dynamic GGUF files (its block"
    " matrices in Q3_K, Q4_K and Q5_K), with its VAE in fp16; phase 2 brings the two 7Bs' other"
    " smaller files"
)
# The same, of a file given as the shot detector.
SHOTS = "seedvr2x detects shots with TransNetV2 in fp32, as its own transnetv2.safetensors holds it"
# What a safetensors file of other tensors is not.
NONE_OF_THEM = "neither SeedVR2's DiT, its VAE nor TransNetV2"


def given_as(role: Role, say: Say) -> str:
    """How a refusal names the role a file was given as, when the file is a model of another
    role: "given as the DiT (--dit-model)", the option after it only where an application words
    the settings."""
    if role == "dit":
        return f"given as the DiT{aside(say, f' ({say(Setting("dit_model"))})')}"
    if role == "vae":
        return f"given as the VAE{aside(say, f' ({say(Setting("vae_model"))})')}"
    return "given as the shot detector"


# The end of a refusal of a file missing data (_read): for one of seedvr2x's own files in its own
# role, check_models says where from instead (seedvr2x/pull.py, advice).
AGAIN = "fetch it again"
# A file that isn't there (_read) was never fetched there: check_models' advice says so, without
# its "again"; and says what it is for the role whose file no option names, the shot detector's,
# which --model-dir holds under seedvr2x's own name for it (pull.DETECTOR).
MISSING = "no such model file"
FETCH = "fetch it"
UNNAMED: dict[Role, Callable[[Say], str]] = {
    "detector": lambda say: (
        "the shot detector's weights, TransNetV2's, which a run that detects its shots reads in"
        f" {say(Setting('model_dir'))}"
        + aside(
            say,
            f" ({say(Setting('cuts'))}, a cut list, runs without them)",
            " (a cut list runs without them)",
        )
    ),
}


@dataclass(frozen=True)
class TensorInfo:
    """A tensor as a safetensors header declares it: its dtype, by safetensors' name (F16, BF16,
    F8_E4M3...), and its shape; and read from a file, where its data starts, from the file's
    start, in bytes (the INT8 layout's markers are read there: _unmarked), which no comparison of
    two tensors counts. A GGUF file's tensors are given the same way, the dtype ggml's type name
    (F16, Q4_K...), the shape PyTorch's (GGUF lists dimensions the other way round)."""

    dtype: str
    shape: tuple[int, ...]
    at: int | None = field(default=None, compare=False)


class GgufTensors(dict[str, TensorInfo]):
    """The tensors of a GGUF file (_read), told from a safetensors file's, whose dtypes can read
    the same (F16): a GGUF file of 1,128 F16 tensors isn't DIT_7B."""


# Bytes per value of the dtypes safetensors names, those whose data sizes are checked; any other
# name fails no read, and no check passes it.
DTYPE_BYTES = {
    **dict.fromkeys(("BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3"), 1),
    **dict.fromkeys(("I16", "U16", "F16", "BF16"), 2),
    **dict.fromkeys(("I32", "U32", "F32"), 4),
    **dict.fromkeys(("F64", "I64", "U64"), 8),
}
# safetensors' own limit on a header's length: the pinned version (0.8.0) refuses a header of
# 100,000,001 bytes as too large, and parses one of 100,000,000.
HEADER_LIMIT = 100_000_000
# safetensors' bound on a header's numbers (0.8.0): it reads each dimension and data offset as a
# 64-bit unsigned integer, refusing 2^64 or more, and refuses a shape whose product, multiplied
# in order, reaches 2^64 ("overflow computing buffer size"). Bounded so, no product takes long
# (math.prod of 200,000 dimensions of 2^63-1 takes over a minute) and no number in a message is
# too long for Python to write (over 4,300 digits: ValueError).
NUMBER_LIMIT = 2**64
# The floating-point dtypes of 16 bits or more, and the names of a precision held by one of them.
FLOATS = frozenset({"F16", "BF16", "F32", "F64"})
PRECISIONS = {"F16": "fp16", "BF16": "bf16", "F32": "fp32"}
# SeedVR2's DiTs by their block count and width, txt_in.weight's first dimension: the vendored
# configs_7b's 36 blocks of 3072 and configs_3b's 32 of 2560.
DITS = {(36, 3072): DIT_7B, (32, 2560): DIT_3B}
BLOCK = re.compile(r"blocks\.(\d+)\.")
# What a quantised matrix's dtype holds, but for fp8 (F8_*): int8 values, or two 4-bit E2M1
# values per byte, as NVFP4 packs them in comfy-kitchen's layout (DESIGN.md, Weights, Phase 2).
MATRICES = {"I8": "INT8", "U8": "4-bit values packed two per byte"}


def check(path: Path, role: Role) -> Architecture:
    """The architecture of the model file at path, given as the DiT, the VAE or the shot detector
    (role), known by its tensors, not its name: SeedVR2's 7B DiT as the DiT, the regular or the
    sharp one, whoever made the file, every tensor F16 (DIT_7B) or in the INT8 layout, each of
    its quantised layers marked as such (DIT_7B_INT8), and its VAE as the VAE, every tensor F16;
    TransNetV2 as the shot detector, every tensor F32 but its BatchNorm counters (COUNTER), I64.
    Anything else raises ModelError, saying what the file is (describe)."""
    tail = SHOTS if role == "detector" else V1
    try:
        read = _read(path)
    except ModelError as error:
        error.role = role  # the role is the caller's, not the file's
        raise
    if isinstance(read, str):
        raise ModelUnknown(f"{path}: {read}: {tail}", role=role, file=path.name, found=read)
    found = _architecture(read)
    if found in ACCEPTED[role] and _as_run(read, found):
        assert found is not None
        unmarked = _unmarked(path, read) if found.layout == INT8 else None
        if unmarked is None:
            return found
        raise ModelUnknown(
            f"{path}: {found.name} in comfy-kitchen's INT8 layout, {unmarked}: {tail}",
            role=role,
            file=path.name,
            found=found.name,
        )
    owner = _role(read, found)
    what = _what(read, found)
    if owner is not None and owner != role:
        settings = (Setting(f"{role}_model"),) if role != "detector" else ()

        def message(say: Say) -> str:
            return f"{path}: {what}, {given_as(role, say)}: {tail}"

        raise ModelUnknown(message, role=role, file=path.name, found=what, settings=settings)
    raise ModelUnknown(f"{path}: {what}: {tail}", role=role, file=path.name, found=what)


def in_words(architecture: Architecture) -> str:
    """What a file of this architecture, in the dtypes seedvr2x runs it in, is, in words: "SeedVR2's
    7B DiT in fp16", "SeedVR2's 7B DiT in int8, W8A8 with comfy-kitchen's rotation", "SeedVR2's
    sharp 7B DiT in seedvr2x's dynamic GGUF, its block matrices in Q3_K, Q4_K and Q5_K"."""
    if architecture.layout == INT8:
        return f"{architecture.name} {INT8_SAID}"
    if architecture.layout == GGUF:
        return f"{architecture.name} {GGUF_SAID}, {_matrices(architecture.matrix_types())}"
    return f"{architecture.name} in {PRECISIONS[architecture.dtype]}"


def check_models(
    dit: Path | None,
    vae: Path | None,
    advice: Mapping[Role, Message] | None = None,
    detector: Path | None = None,
) -> None:
    """Check the DiT file at dit, the VAE file at vae and, given, the shot detector's file at
    detector (check), in --model-dir or in Hugging Face's cache (seedvr2x/pull.py), each accepted
    one logged with what it is; ModelError says what is wrong with each one refused, and how to
    fetch it again when advice has it for its role (one of seedvr2x's own files, given in its own
    role: pull.advice), in place of AGAIN; how to fetch it, without "again", when it isn't there
    (MISSING), and what it is when no option named it (UNNAMED). The CLI calls it before the
    first pass, which can take tens of minutes on a film. A DiT or a VAE given as None isn't
    checked: one of seedvr2x's own that a job with every segment made, which loads no model,
    holds to its pin, absent from Hugging Face's cache (pull.ModelFile.absent).

    The CPU tests' stand-in replaces this function (tests/test_cli_run.py, stand_in_model): for
    tests only, never a way around the check for users."""
    refused: list[tuple[Role, ModelError]] = []
    files: list[tuple[str, Role, Path | None]] = [("DiT", "dit", dit), ("VAE", "vae", vae)]
    if detector is not None:
        files.append(("Shot detector", "detector", detector))
    for label, role, path in files:
        if path is None:
            continue
        try:
            found = check(path, role)
        except ModelError as error:
            refused.append((role, error))
            continue
        logger.info("%s %s: %s", label, path.name, in_words(found))
    if not refused:
        return
    hows = {role: worded(how) for role, how in (advice or {}).items()}

    def line(role: Role, error: ModelError, say: Say) -> str:
        said = error.words(say)
        missing = isinstance(error, ModelMissing)
        if missing and role in UNNAMED:
            said = f"{said}: {UNNAMED[role](say)}"
        if role in hows:
            how = hows[role](say)
            how = how.replace(AGAIN, FETCH, 1) if missing else how
            said = f"{said.removesuffix(f'; {AGAIN}')}; {how}"
        return said

    def message(say: Say) -> str:
        return "\n".join(line(role, error, say) for role, error in refused)

    raise ModelsRefused([error for _, error in refused], message)


def describe(path: Path) -> str:
    """What the model file at path is, in words: SeedVR2's 7B or 3B DiT, its VAE or TransNetV2,
    and its precision; another DiT and how it is quantised; another safetensors file; a GGUF file;
    a PyTorch checkpoint; or neither. Raises ModelError when it is missing, not a regular file,
    empty, cut short or damaged."""
    read = _read(path)
    return read if isinstance(read, str) else _what(read, _architecture(read))


def inventory(tensors: Mapping[str, TensorInfo]) -> str:
    """The digest of a model's tensors, by name and shape: the SHA-256, in hex, of the sorted lines
    `name [shape]`, the shape as Python prints a list ([9216, 3072]; [] for a scalar), joined by
    newlines."""
    lines = sorted(f"{name} {list(info.shape)}" for name, info in tensors.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _architecture(tensors: Mapping[str, TensorInfo]) -> Architecture | None:
    """The architecture whose tensors these are, exactly, by name and shape; else None. A GGUF
    file's are one of the GGUF layout's, whose types they must be as well (the dynamic files share
    DIT_7B's names and shapes and differ by the ggml type of some matrices: DIT_7B_DYN's types
    are not DIT_7B_SHARP_DYN's, nor those of a file quantised otherwise); a safetensors file's
    never are."""
    gguf = isinstance(tensors, GgufTensors)
    candidates = [
        each
        for each in ARCHITECTURES
        if each.tensors == len(tensors) and (each.layout == GGUF) == gguf
    ]
    if not candidates:
        return None
    digest = inventory(tensors)
    named = [each for each in candidates if each.digest == digest]
    if gguf:
        return next((each for each in named if _as_run(tensors, each)), None)
    return next(iter(named), None)


def _as_run(tensors: Mapping[str, TensorInfo], architecture: Architecture) -> bool:
    """Whether an architecture's tensors are in the dtypes seedvr2x runs it in (dtype_of): its
    dtype, but a BatchNorm's counters (COUNTER), I64, and in the INT8 layout its quantised
    layers' tensors."""
    return all(info.dtype == architecture.dtype_of(name) for name, info in tensors.items())


def _unmarked(path: Path, tensors: Mapping[str, TensorInfo]) -> str | None:
    """What is wrong with the markers of a file of the INT8 layout's tensors, in words; None when
    each says MARKER, the layout its layer's values are in, read where the header puts it
    (TensorInfo.at): 288 reads of 72 bytes for the 7B. A marker of the same length can say
    another rotation (a convrot_groupsize of 128), whose values the multiply would take for
    others."""
    markers = sorted((name, info) for name, info in tensors.items() if name.endswith(MARK))
    try:
        with path.open("rb") as file:
            for name, info in markers:
                assert info.at is not None, "read from a file"
                file.seek(info.at)
                raw = file.read(info.shape[0])
                try:
                    content: object = json.loads(raw.decode())
                except ValueError:
                    content = None
                if content != MARKER:
                    shown = raw.decode(errors="backslashreplace")
                    return f"its {name.removesuffix(MARK)} marked {shown}, not {json.dumps(MARKER)}"
    except OSError as error:
        raise ModelUnreadable(f"{path}: {error.strerror}", file=path.name) from None
    return None


def _role(tensors: Mapping[str, TensorInfo], found: Architecture | None) -> Role | None:
    """The role of the model these tensors are, found the architecture they are exactly, if any:
    a DiT's, SeedVR2's or one named as SeedVR2's names its tensors (_dit), the VAE's, or the shot
    detector's; None for any other file."""
    if (found is not None and found.config is not None) or (
        found is None and _dit(tensors) is not None
    ):
        return "dit"
    if found is VAE:
        return "vae"
    if found is TRANSNETV2:
        return "detector"
    return None


def _read(path: Path) -> dict[str, TensorInfo] | str:
    """The tensors of the safetensors file at path, as its header declares them, validated as the
    safetensors library validates a file (_parse); of the GGUF file at path (_gguf: GgufTensors);
    or what a file of another format is, in words (_other). Reads the fewest bytes: the header's
    length (8), then the header, never the data. Raises ModelError when the file is missing, not a
    regular file, empty, cut short or damaged."""
    if not path.is_file():
        if os.path.lexists(path):
            raise ModelUnreadable(f"{path}: not a regular file", file=path.name)
        raise ModelMissing(f"{path}: {MISSING}", file=path.name)
    try:
        with path.open("rb") as file:
            size = os.fstat(file.fileno()).st_size
            # Empty: what a download failed before its first byte leaves, not a file of another
            # kind, so refused as one cut short is, without V1's tail. wget -O truncates its
            # file at once (GNU Wget's manual, -O), and leaves it empty on a 404 (Wget 1.21.4).
            if size == 0:
                raise ModelDamaged(
                    f"{path}: empty (0 bytes); {AGAIN}", file=path.name, figure="size", found=0
                )
            # 1 to 8 bytes: too few for a safetensors header's length and the brace that tell the
            # format (below), and for any model file's header (GGUF's 24 bytes, ggml's
            # docs/gguf.md; a zip's local file header 30, APPNOTE.TXT 4.3.7): what a download cut
            # short leaves, refused as one, its size said, without V1's tail. "GGUF" or "PK" 3 4
            # alone is such a file, not a GGUF file or a checkpoint.
            if size < 9:
                raise ModelDamaged(
                    f"{path}: cut short, {_count(size, 'byte')}, too few for any model file;"
                    f" {AGAIN}",
                    file=path.name,
                    figure="size",
                    found=size,
                )
            # The header's length, then the header, which begins with its JSON object's brace,
            # as safetensors' format requires. GGUF's signature is told first: read as a header's
            # length, it is at least 0x46554747 = 1,179,993,927 bytes, over the limit, so no
            # safetensors file begins with it, where a GGUF file's ninth byte, the low byte of
            # its tensor count, can be the brace (numz's 3B GGUF files: 635 tensors, 0x027B).
            # Then the brace, before the others' signatures: a header's length can begin as a
            # pickle's (0x80 0x02) or a zip's ("PK" 3 4: 67,324,752 bytes, within the limit)
            # does.
            start = file.read(9)
            if start.startswith(b"GGUF"):
                return _gguf(path, file, size)
            if start[8:] != b"{":
                return _other(start)
            length = int.from_bytes(start[:8], "little")
            if length == 0 or length > HEADER_LIMIT:
                raise _damaged(
                    path,
                    f"a header of {length:,} bytes, where safetensors takes 1 to {HEADER_LIMIT:,}",
                )
            if 8 + length > size:
                raise ModelDamaged(
                    f"{path}: cut short inside its header, {size:,} bytes of at least"
                    f" {8 + length:,}; {AGAIN}",
                    file=path.name,
                    figure="size",
                    expected=8 + length,
                    found=size,
                )
            header = start[8:] + file.read(length - 1)
    except OSError as error:
        raise ModelUnreadable(f"{path}: {error.strerror}", file=path.name) from None
    tensors, data = _parse(path, header)
    end = 8 + length + data
    if size < end:
        raise ModelDamaged(
            f"{path}: cut short, {size:,} of {end:,} bytes; {AGAIN}",
            file=path.name,
            figure="size",
            expected=end,
            found=size,
        )
    if size > end:
        raise _damaged(path, f"{_count(size - end, 'byte')} after its tensors' data")
    return tensors


def _gguf(path: Path, file: BinaryIO, size: int) -> GgufTensors | str:
    """The tensors of the GGUF file open as file, size bytes long, as its header declares them
    (seedvr2x/gguf_file.py, which checks its offsets and its size), each dtype ggml's type name and
    each `at` where its bytes begin; or what the file is, in words, when it is a GGUF file
    seedvr2x doesn't read (a version other than 3, a tensor of a ggml type it doesn't know). Raises
    ModelError when the header is cut short (the file is fetched again) or damaged."""
    try:
        header = gguf_file.header_of(file, size)
    except GgufError as error:
        if error.kind == "cut":
            raise ModelDamaged(f"{path}: {error}; {AGAIN}", file=path.name) from None
        if error.kind in ("version", "type"):
            return str(error)
        raise ModelDamaged(f"{path}: a damaged GGUF file: {error}", file=path.name) from None
    return GgufTensors(
        {each.name: TensorInfo(each.type, each.shape, each.offset) for each in header.tensors}
    )


def _other(start: bytes) -> str:
    """What a file that isn't safetensors or GGUF is, by its first bytes (start): a zip's, "PK"
    3 4, torch.save's format from PyTorch 1.6; or a pickle's protocol 2 to 5 (pickle's PROTO
    opcode, 0x80, then the protocol), torch.save's format before it."""
    if start.startswith(b"PK\x03\x04"):
        kind = "zip"
    elif start[:1] == b"\x80" and start[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        kind = "pickle"
    else:
        return "neither safetensors, GGUF nor a PyTorch checkpoint"
    return (
        f"a PyTorch checkpoint ({kind}), as ByteDance's fp32 masters (.pth) are; seedvr2x reads"
        " safetensors files, which the repository's models/ scripts make from them"
    )


def _parse(path: Path, header: bytes) -> tuple[dict[str, TensorInfo], int]:
    """The tensors a safetensors header declares, by name, and the length of their data, D,
    validated as the safetensors library validates them (measured on 0.8.0): a JSON object, in
    UTF-8; its __metadata__, if any, mapping strings to strings; every other entry a tensor's, its
    name UTF-8, its dtype a string, its shape whole numbers below NUMBER_LIMIT and their product
    too, its data_offsets [begin, end], whole numbers below NUMBER_LIMIT, end not before begin,
    the size of its values for a dtype of DTYPE_BYTES; and the offsets, sorted, tiling [0, D)
    with no gap or overlap."""
    try:
        content: object = json.loads(header.decode())
    except UnicodeDecodeError:
        raise _damaged(path, "its header isn't UTF-8") from None
    except json.JSONDecodeError as error:
        raise _damaged(path, f"its header isn't JSON: {error}") from None
    except ValueError:
        # An integer of over 4,300 digits, which Python doesn't convert (int_max_str_digits);
        # safetensors refuses it too ("number out of range").
        raise _damaged(path, "its header isn't JSON: a number too long") from None
    except RecursionError:
        # Arrays or objects nested deeper than Python's parser goes, thousands of levels;
        # safetensors refuses from 128 levels on ("recursion limit exceeded").
        raise _damaged(path, "its header isn't JSON: nested too deep") from None
    if not isinstance(content, dict):
        raise _damaged(path, "its header isn't a JSON object")
    entries = cast(dict[str, object], content)
    metadata = entries.pop("__metadata__", None)
    if metadata is not None and not (
        isinstance(metadata, dict)
        and all(isinstance(value, str) for value in cast(dict[str, object], metadata).values())
    ):
        raise _damaged(path, "its __metadata__ doesn't map strings to strings")
    tensors: dict[str, TensorInfo] = {}
    spans: list[tuple[int, int, str]] = []
    for name, entry in entries.items():
        # A lone surrogate (JSON's "\ud800") has no UTF-8, in which the inventory's digest
        # writes the names; safetensors refuses it as it parses the header.
        try:
            name.encode()
        except UnicodeEncodeError:
            shown = name.encode(errors="backslashreplace").decode()
            raise _damaged(path, f"{shown}: its name isn't UTF-8") from None
        if not isinstance(entry, dict):
            raise _damaged(path, f"{name}: not a tensor's entry")
        fields = cast(dict[str, object], entry)
        dtype = fields.get("dtype")
        shape = _whole_numbers(fields.get("shape"))
        offsets = _whole_numbers(fields.get("data_offsets"))
        if not isinstance(dtype, str):
            raise _damaged(path, f"{name}: no dtype")
        if shape is None:
            raise _damaged(path, f"{name}: its shape isn't whole numbers below 2^64")
        if offsets is None or len(offsets) != 2 or offsets[1] < offsets[0]:
            raise _damaged(path, f"{name}: its data_offsets aren't [begin, end]")
        begin, end = offsets
        values = 1
        for dimension in shape:
            values *= dimension
            if values >= NUMBER_LIMIT:
                raise _damaged(path, f"{name}: its shape overflows 64 bits")
        if dtype in DTYPE_BYTES and end - begin != values * DTYPE_BYTES[dtype]:
            raise _damaged(
                path,
                f"{name}: {_count(end - begin, 'byte')} of data for {_count(values, 'value')}"
                f" of {dtype}",
            )
        tensors[name] = TensorInfo(dtype, tuple(shape), 8 + len(header) + begin)
        spans.append((begin, end, name))
    position = 0
    for begin, end, name in sorted(spans):
        if begin > position:
            raise _damaged(path, f"{name}: its data at {begin:,}, after a gap from {position:,}")
        if begin < position:
            raise _damaged(
                path,
                f"{name}: its data at {begin:,}, inside the data before it, up to {position:,}",
            )
        position = end
    return tensors, position


def _whole_numbers(value: object) -> list[int] | None:
    """value as a list of whole numbers below NUMBER_LIMIT, as JSON gives them; else None."""
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if not all(type(item) is int and 0 <= item < NUMBER_LIMIT for item in items):
        return None
    return cast(list[int], items)


def _damaged(path: Path, what: str) -> ModelError:
    return ModelDamaged(f"{path}: a damaged safetensors file: {what}", file=path.name)


def _what(tensors: Mapping[str, TensorInfo], found: Architecture | None) -> str:
    """What a safetensors file of these tensors is, in words; found, the architecture they are
    exactly, if any."""
    dtypes = Counter(info.dtype for info in tensors.values())
    if isinstance(tensors, GgufTensors):
        return _what_gguf(tensors, found, dtypes)
    if found is not None and found.layout == INT8:
        if _as_run(tensors, found):
            return in_words(found)
        return f"{found.name} in comfy-kitchen's INT8 layout, in other dtypes ({_counts(dtypes)})"
    if found is not None:
        # The precision of the values, a BatchNorm's counters (COUNTER) apart, said only when
        # they aren't I64, as PyTorch keeps them.
        values = Counter(info.dtype for name, info in tensors.items() if not name.endswith(COUNTER))
        counters = Counter(info.dtype for name, info in tensors.items() if name.endswith(COUNTER))
        said = f"{found.name} {_precision(values)}"
        if counters.keys() - {"I64"}:
            said += f", its BatchNorm counters {_counts(counters)} where PyTorch keeps them I64"
        return said
    dit = _dit(tensors)
    if dit is not None:
        return _other_dit(tensors, *dit, dtypes)
    if not tensors:
        return f"a safetensors file of no tensors, {NONE_OF_THEM}"
    tensor_count = _count(len(tensors), "tensor")
    return f"a safetensors file of {tensor_count} ({_counts(dtypes)}), {NONE_OF_THEM}"


def _what_gguf(
    tensors: Mapping[str, TensorInfo], found: Architecture | None, dtypes: Counter[str]
) -> str:
    """What a GGUF file of these tensors is, in words; found, the dynamic architecture they are
    exactly, names, shapes and types, if any. A DiT of SeedVR2's tensors says how its block
    matrices are quantised, by type and count, and that it is none of seedvr2x's files."""
    if found is not None:
        return in_words(found)
    dit = _dit(tensors)
    if dit is None:
        if not tensors:
            return f"a GGUF file of no tensors, {NONE_OF_THEM}"
        return (
            f"a GGUF file of {_count(len(tensors), 'tensor')} ({_counts(dtypes)}), {NONE_OF_THEM}"
        )
    known = DITS.get(dit)
    if known is None or inventory(tensors) != known.digest:
        name = known.name if known else f"a DiT of {dit[0]} blocks, {dit[1]} wide"
        other = (
            f"{name} by its blocks and width, with other tensors than its own" if known else name
        )
        return f"a GGUF file of {other} ({_count(len(tensors), 'tensor')}: {_counts(dtypes)})"
    # SeedVR2's own tensors, by name and shape: the block matrices' types are what differs from
    # seedvr2x's files.
    matrices = Counter(
        info.dtype
        for each, info in tensors.items()
        if BLOCK.match(each) and each.endswith(".weight") and len(info.shape) == 2
    )
    ranked = sorted(matrices.items(), key=lambda item: (-item[1], item[0]))
    said = _matrices([f"{dtype} ({count:,})" for dtype, count in ranked])
    return f"a GGUF file of {known.name}, {said}" + (
        ", not one of seedvr2x's dynamic files" if known is DIT_7B else ""
    )


def _matrices(kinds: list[str]) -> str:
    """ "its block matrices in Q4_K", "... in Q3_K, Q4_K and Q5_K": the kinds, each said once."""
    head, last = kinds[:-1], kinds[-1]
    return (
        f"its block matrices in {', '.join(head)} and {last}"
        if head
        else f"its block matrices in {last}"
    )


def _precision(dtypes: Counter[str]) -> str:
    """The precision of a file holding exactly an architecture's tensors, in these dtypes."""
    single = next(iter(dtypes)) if len(dtypes) == 1 else None
    if single is not None and single in PRECISIONS:
        return f"in {PRECISIONS[single]}"
    fp8 = {dtype for dtype in dtypes if dtype.startswith("F8_")}
    if fp8 and dtypes.keys() - fp8 <= FLOATS:
        # No room for a scale beside them: an architecture's tensors exactly, as numz's fp8 files
        # hold them (DESIGN.md, Weights, Phase 2).
        return f"in fp8, cast without scales ({_counts(dtypes, 'tensor')})"
    if single is not None:
        return f"with every tensor {single}"
    return f"in mixed precision ({_counts(dtypes, 'tensor')})"


def _dit(tensors: Mapping[str, TensorInfo]) -> tuple[int, int] | None:
    """A DiT's block count and width, txt_in.weight's first dimension, when the tensors are a
    DiT's, as SeedVR2's name them: txt_in.weight, vid_in.proj.weight and blocks.<i>. ones."""
    text = tensors.get("txt_in.weight")
    # The indices as written: int() refuses one of over 4,300 digits (ValueError), which a
    # header may hold.
    blocks = {match[1] for name in tensors if (match := BLOCK.match(name))}
    if text is None or not text.shape or "vid_in.proj.weight" not in tensors or not blocks:
        return None
    return len(blocks), text.shape[0]


def _other_dit(
    tensors: Mapping[str, TensorInfo], blocks: int, width: int, dtypes: Counter[str]
) -> str:
    """What a DiT is that isn't one of the architectures exactly: SeedVR2's by its block count
    and width, and how its blocks' matrices are quantised, if they are."""
    known = DITS.get((blocks, width))
    name = known.name if known is not None else f"a DiT of {blocks} blocks, {width} wide"
    other = f"{name} by its blocks and width, with other tensors than its own" if known else name
    counts = f"{_count(len(tensors), 'tensor')}: {_counts(dtypes)}"
    # The blocks' matrices, quantised: their 2-D weights in another dtype than a float of 16 bits
    # or more.
    quantised = {
        each: info.dtype
        for each, info in tensors.items()
        if BLOCK.match(each)
        and each.endswith(".weight")
        and len(info.shape) == 2
        and info.dtype not in FLOATS
    }
    if not quantised:
        return f"{other} ({counts})"
    kinds = sorted(
        {
            "fp8" if dtype.startswith("F8_") else MATRICES.get(dtype, dtype)
            for dtype in quantised.values()
        }
    )
    # Their scales beside them, as <layer>.weight_scale (and NVFP4's weight_scale_2) in
    # seedvr2x's phase-2 files, comfy-kitchen's layout.
    layers = {each.removesuffix(".weight") for each in quantised}
    scaled = any(
        layer in layers and "scale" in leaf
        for layer, _, leaf in (each.rpartition(".") for each in tensors)
    )
    comfy = any(each.endswith(".comfy_quant") for each in tensors)
    return (
        f"{name} quantised, its block matrices in {' and '.join(kinds)}"
        + (" with scale tensors beside them" if scaled else " without scales")
        + (", in comfy-kitchen's layout" if comfy else "")
        + f" ({counts})"
    )


def _counts(dtypes: Counter[str], noun: str = "") -> str:
    """Tensor counts by dtype, the largest first, the noun after the first count: with "tensor",
    "1,097 tensors F8_E4M3, 31 F16"."""
    ranked = sorted(dtypes.items(), key=lambda item: (-item[1], item[0]))
    words = [f"{count:,} {dtype}" for dtype, count in ranked]
    if noun:
        words[0] = f"{_count(ranked[0][1], noun)} {ranked[0][0]}"
    return ", ".join(words)


def _count(number: int, noun: str) -> str:
    """number, with thousands separators, and the noun, plural but for 1."""
    return f"{number:,} {noun}{'' if number == 1 else 's'}"
