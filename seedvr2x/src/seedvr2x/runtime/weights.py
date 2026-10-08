"""Model files, recognised by their tensors, not by their names (DESIGN.md, Weights): a file's
header is read and checked before anything else runs, without torch, so that a wrong file is
refused in a second, saying what it is, rather than failing after the first pass or running
unchecked.

v1 runs SeedVR2's 7B DiT, the regular or the sharp one, with its VAE, every tensor in fp16. A
safetensors file is known by its header alone, read without the safetensors library: the header's
length, then the header, never the data."""

import hashlib
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from seedvr2x.runtime.job import JobError

logger = logging.getLogger(__name__)

Role = Literal["dit", "vae"]


class ModelError(JobError):
    """A model file seedvr2x refuses: the message says what the file is, or what is wrong with
    it."""


@dataclass(frozen=True)
class Architecture:
    """A model seedvr2x knows by its tensors: their count and their inventory's digest
    (inventory), and the vendored config directory a DiT is built from, None for the VAE."""

    name: str
    tensors: int
    digest: str
    config: str | None


# Each digest is inventory()'s, over the model's tensors' names and shapes: what the vendored
# model's state_dict gives, built on the meta device, from which tests/test_weights.py derives
# each; and what the real files' headers give: numz's 7B, sharp 7B and 7B fp8 files and seedvr2x's
# 7B and sharp 7B fp16 ones the 7B's, numz's 3B fp16 and fp8 files the 3B's, numz's and seedvr2x's
# fp16 VAE the VAE's.
DIT_7B = Architecture(
    "SeedVR2's 7B DiT",
    1128,
    "960321767ecf7a1aeee79faefb0dcfb00acf66d205170ab813fff5bd3d1f3b2f",
    "configs_7b",
)
DIT_3B = Architecture(
    "SeedVR2's 3B DiT",
    635,
    "cc68bcef364ddcfa3b51f09a62e171dc9f69bb7743588c43d82aad94456801ac",
    "configs_3b",
)
VAE = Architecture(
    "SeedVR2's VAE", 250, "090ead68b30f9e815c50f1bdd1ed9c522f669461d42888ed56cbf0e57fb863e1", None
)
ARCHITECTURES = (DIT_7B, DIT_3B, VAE)

# What v1 runs in each role (DESIGN.md, Weights): the 7B DiT, the regular and the sharp one alike
# (one architecture, the same tensors), and the VAE, every tensor in F16.
ACCEPTED: dict[Role, Architecture] = {"dit": DIT_7B, "vae": VAE}
V1 = (
    "seedvr2x v1 runs SeedVR2's 7B DiT, regular or sharp, in fp16, with its VAE in fp16; phase 2"
    " brings the other models"
)


@dataclass(frozen=True)
class TensorInfo:
    """A tensor as a safetensors header declares it: its dtype, by safetensors' name (F16, BF16,
    F8_E4M3...), and its shape."""

    dtype: str
    shape: tuple[int, ...]


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
    """The architecture of the model file at path, given as the DiT or as the VAE (role), known by
    its tensors, not its name: SeedVR2's 7B DiT as the DiT, the regular or the sharp one, whoever
    made the file, and its VAE as the VAE, every tensor F16. Anything else raises ModelError,
    saying what the file is (describe)."""
    read = _read(path)
    if isinstance(read, str):
        raise ModelError(f"{path}: {read}: {V1}")
    found, accepted = _architecture(read), ACCEPTED[role]
    if found is accepted and all(info.dtype == "F16" for info in read.values()):
        return accepted
    given = ""
    if role == "dit" and found is VAE:
        given = ", given as the DiT (--dit-model)"
    elif role == "vae" and (found in DITS.values() or _dit(read) is not None):
        given = ", given as the VAE (--vae-model)"
    raise ModelError(f"{path}: {_what(read, found)}{given}: {V1}")


def check_models(model_dir: Path, dit: str, vae: str) -> None:
    """Check the DiT file dit and the VAE file vae of model_dir (check), each accepted one logged
    with what it is; ModelError says what is wrong with each one refused. The CLI calls it before
    the first pass, which can take tens of minutes on a film.

    The CPU tests' stand-in replaces this function (tests/test_cli_run.py, stand_in_model): for
    tests only, never a way around the check for users."""
    refusals: list[str] = []
    files: tuple[tuple[str, Role, str], ...] = (("DiT", "dit", dit), ("VAE", "vae", vae))
    for label, role, name in files:
        try:
            found = check(model_dir / name, role)
        except ModelError as error:
            refusals.append(str(error))
            continue
        logger.info("%s %s: %s in fp16", label, name, found.name)
    if refusals:
        raise ModelError("\n".join(refusals))


def describe(path: Path) -> str:
    """What the model file at path is, in words: SeedVR2's 7B or 3B DiT or its VAE, and its
    precision; another DiT and how it is quantised; another safetensors file; a GGUF file; a
    PyTorch checkpoint; or neither. Raises ModelError when it is missing, not a regular file, cut
    short or damaged."""
    read = _read(path)
    return read if isinstance(read, str) else _what(read, _architecture(read))


def inventory(tensors: Mapping[str, TensorInfo]) -> str:
    """The digest of a model's tensors, by name and shape: the SHA-256, in hex, of the sorted lines
    `name [shape]`, the shape as Python prints a list ([9216, 3072]; [] for a scalar), joined by
    newlines."""
    lines = sorted(f"{name} {list(info.shape)}" for name, info in tensors.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _architecture(tensors: Mapping[str, TensorInfo]) -> Architecture | None:
    """The architecture whose tensors these are, exactly, by name and shape; else None."""
    candidates = [each for each in ARCHITECTURES if each.tensors == len(tensors)]
    if not candidates:
        return None
    digest = inventory(tensors)
    return next((each for each in candidates if each.digest == digest), None)


def _read(path: Path) -> dict[str, TensorInfo] | str:
    """The tensors of the safetensors file at path, as its header declares them, validated as the
    safetensors library validates a file (_parse); or what a file of another format is, in words
    (_other). Reads the fewest bytes: the header's length (8), then the header, never the data.
    Raises ModelError when the file is missing, not a regular file, cut short or damaged."""
    if not path.is_file():
        if os.path.lexists(path):
            raise ModelError(f"{path}: not a regular file")
        raise ModelError(f"{path}: no such model file")
    try:
        with path.open("rb") as file:
            size = os.fstat(file.fileno()).st_size
            # The header's length, then the header, which begins with its JSON object's brace,
            # as safetensors' format requires. GGUF's signature is told first: read as a header's
            # length, it is at least 0x46554747 = 1,179,993,927 bytes, over the limit, so no
            # safetensors file begins with it, where a GGUF file's ninth byte, the low byte of
            # its tensor count, can be the brace (numz's 3B GGUF files: 635 tensors, 0x027B).
            # Then the brace, before the others' signatures: a header's length can begin as a
            # pickle's (0x80 0x02) or a zip's ("PK" 3 4: 67,324,752 bytes, within the limit)
            # does.
            start = file.read(9)
            if start.startswith(b"GGUF") or start[8:] != b"{":
                return _other(start, size)
            length = int.from_bytes(start[:8], "little")
            if length == 0 or length > HEADER_LIMIT:
                raise _damaged(
                    path,
                    f"a header of {length:,} bytes, where safetensors takes 1 to {HEADER_LIMIT:,}",
                )
            if 8 + length > size:
                raise ModelError(
                    f"{path}: cut short inside its header, {size:,} bytes of at least"
                    f" {8 + length:,}; fetch it again"
                )
            header = start[8:] + file.read(length - 1)
    except OSError as error:
        raise ModelError(f"{path}: {error.strerror}") from None
    tensors, data = _parse(path, header)
    end = 8 + length + data
    if size < end:
        raise ModelError(f"{path}: cut short, {size:,} of {end:,} bytes; fetch it again")
    if size > end:
        raise _damaged(path, f"{_count(size - end, 'byte')} after its tensors' data")
    return tensors


def _other(start: bytes, size: int) -> str:
    """What a file that isn't safetensors is, by its first bytes (start): GGUF's signature,
    "GGUF" (ggml's docs/gguf.md); a zip's, "PK" 3 4, torch.save's format from PyTorch 1.6; or a
    pickle's protocol 2 to 5 (pickle's PROTO opcode, 0x80, then the protocol), torch.save's
    format before it."""
    if size == 0:
        return "an empty file"
    if start.startswith(b"GGUF"):
        return (
            "a GGUF file (quantised weights, as numz's Q4_K_M and seedvr2x's own Q8_0, Q4_K and"
            " dynamic files)"
        )
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
        tensors[name] = TensorInfo(dtype, tuple(shape))
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
    return ModelError(f"{path}: a damaged safetensors file: {what}")


def _what(tensors: Mapping[str, TensorInfo], found: Architecture | None) -> str:
    """What a safetensors file of these tensors is, in words; found, the architecture they are
    exactly, if any."""
    dtypes = Counter(info.dtype for info in tensors.values())
    if found is not None:
        return f"{found.name} {_precision(dtypes)}"
    dit = _dit(tensors)
    if dit is not None:
        return _other_dit(tensors, *dit, dtypes)
    if not tensors:
        return "a safetensors file of no tensors, neither SeedVR2's DiT nor its VAE"
    return (
        f"a safetensors file of {_count(len(tensors), 'tensor')} ({_counts(dtypes)}), neither"
        " SeedVR2's DiT nor its VAE"
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
