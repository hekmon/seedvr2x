"""Model files, recognised by their tensors, not by their names (DESIGN.md, Weights): a file's
header is read and checked before anything else runs, without torch, so that a wrong file is
refused in a second, saying what it is, rather than failing after the first pass or running
unchecked.

v1 runs SeedVR2's 7B DiT, the regular or the sharp one, with its VAE, every tensor in fp16, and
detects shots with TransNetV2 in fp32 (DESIGN.md, Input, Shot detection). A safetensors file is
known by its header alone, read without the safetensors library: the header's length, then the
header, never the data."""

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

Role = Literal["dit", "vae", "detector"]


class ModelError(JobError):
    """A model file seedvr2x refuses: the message says what the file is, or what is wrong with
    it."""


@dataclass(frozen=True)
class Architecture:
    """A model seedvr2x knows by its tensors: their count and their inventory's digest
    (inventory); the vendored config directory a DiT is built from, None for another model; and
    the dtype seedvr2x runs it in, every tensor's but a BatchNorm counter's (COUNTER), I64."""

    name: str
    tensors: int
    digest: str
    config: str | None
    dtype: str = "F16"


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
TRANSNETV2 = Architecture(
    "TransNetV2",
    90,
    "007f515b2f0977ca26cdec87b41da10511f34c4c6138f60cb9547e63ccee1504",
    None,
    "F32",
)
ARCHITECTURES = (DIT_7B, DIT_3B, VAE, TRANSNETV2)

# What v1 runs in each role (DESIGN.md, Weights): the 7B DiT, the regular and the sharp one alike
# (one architecture, the same tensors), and the VAE, every tensor in F16; and TransNetV2, the shot
# detector, as its official weights are converted, every tensor F32 but its counters (DESIGN.md,
# Input, Shot detection).
ACCEPTED: dict[Role, Architecture] = {"dit": DIT_7B, "vae": VAE, "detector": TRANSNETV2}
# The end of every refusal of a readable file given as the DiT or the VAE, the 3B's included:
# phase 2 brings the two 7Bs' smaller files, and no 3B (DESIGN.md, Weights, Phase 2).
V1 = (
    "seedvr2x v1 runs SeedVR2's 7B DiT, regular or sharp, in fp16, with its VAE in fp16; phase 2"
    " brings the two 7Bs' smaller files"
)
# The same, of a file given as the shot detector.
SHOTS = "seedvr2x detects shots with TransNetV2 in fp32, as its own transnetv2.safetensors holds it"
# What a safetensors file of other tensors is not.
NONE_OF_THEM = "neither SeedVR2's DiT, its VAE nor TransNetV2"
# How each role's refusal names it, when the file is a model of another role.
GIVEN: dict[Role, str] = {
    "dit": "given as the DiT (--dit-model)",
    "vae": "given as the VAE (--vae-model)",
    "detector": "given as the shot detector",
}
# The end of a refusal of a file missing data (_read): for one of seedvr2x's own files in its own
# role, check_models says where from instead (runtime/pull.py, advice).
AGAIN = "fetch it again"
# A file that isn't there (_read) was never fetched there: check_models' advice says so, without
# its "again"; and says what it is for the role whose file no option names, the shot detector's,
# which --model-dir holds under seedvr2x's own name for it (pull.DETECTOR).
MISSING = "no such model file"
FETCH = "fetch it"
UNNAMED: dict[Role, str] = {
    "detector": (
        "the shot detector's weights, TransNetV2's, which a run that detects its shots reads in"
        " --model-dir (--cuts, a cut list, runs without them)"
    ),
}


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
    """The architecture of the model file at path, given as the DiT, the VAE or the shot detector
    (role), known by its tensors, not its name: SeedVR2's 7B DiT as the DiT, the regular or the
    sharp one, whoever made the file, and its VAE as the VAE, every tensor F16; TransNetV2 as the
    shot detector, every tensor F32 but its BatchNorm counters (COUNTER), I64. Anything else
    raises ModelError, saying what the file is (describe)."""
    tail = SHOTS if role == "detector" else V1
    read = _read(path)
    if isinstance(read, str):
        raise ModelError(f"{path}: {read}: {tail}")
    found, accepted = _architecture(read), ACCEPTED[role]
    if found is accepted and _as_run(read, accepted):
        return accepted
    owner = _role(read, found)
    given = f", {GIVEN[role]}" if owner is not None and owner != role else ""
    raise ModelError(f"{path}: {_what(read, found)}{given}: {tail}")


def check_models(
    dit: Path, vae: Path, advice: Mapping[Role, str] | None = None, detector: Path | None = None
) -> None:
    """Check the DiT file at dit, the VAE file at vae and, given, the shot detector's file at
    detector (check), in --model-dir or in Hugging Face's cache (runtime/pull.py), each accepted
    one logged with what it is; ModelError says what is wrong with each one refused, and how to
    fetch it again when advice has it for its role (one of seedvr2x's own files, given in its own
    role: pull.advice), in place of AGAIN; how to fetch it, without "again", when it isn't there
    (MISSING), and what it is when no option named it (UNNAMED). The CLI calls it before the
    first pass, which can take tens of minutes on a film.

    The CPU tests' stand-in replaces this function (tests/test_cli_run.py, stand_in_model): for
    tests only, never a way around the check for users."""
    refusals: list[str] = []
    files: list[tuple[str, Role, Path]] = [("DiT", "dit", dit), ("VAE", "vae", vae)]
    if detector is not None:
        files.append(("Shot detector", "detector", detector))
    for label, role, path in files:
        try:
            found = check(path, role)
        except ModelError as error:
            said = str(error)
            missing = said.endswith(MISSING)
            if missing and role in UNNAMED:
                said = f"{said}: {UNNAMED[role]}"
            if advice is not None and role in advice:
                how = advice[role].replace(AGAIN, FETCH, 1) if missing else advice[role]
                said = f"{said.removesuffix(f'; {AGAIN}')}; {how}"
            refusals.append(said)
            continue
        logger.info("%s %s: %s in %s", label, path.name, found.name, PRECISIONS[found.dtype])
    if refusals:
        raise ModelError("\n".join(refusals))


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
    """The architecture whose tensors these are, exactly, by name and shape; else None."""
    candidates = [each for each in ARCHITECTURES if each.tensors == len(tensors)]
    if not candidates:
        return None
    digest = inventory(tensors)
    return next((each for each in candidates if each.digest == digest), None)


def _as_run(tensors: Mapping[str, TensorInfo], architecture: Architecture) -> bool:
    """Whether an architecture's tensors are in the dtypes seedvr2x runs it in: its dtype, but a
    BatchNorm's counters (COUNTER), I64."""
    return all(
        info.dtype == ("I64" if name.endswith(COUNTER) else architecture.dtype)
        for name, info in tensors.items()
    )


def _role(tensors: Mapping[str, TensorInfo], found: Architecture | None) -> Role | None:
    """The role of the model these tensors are, found the architecture they are exactly, if any:
    a DiT's, SeedVR2's or one named as SeedVR2's names its tensors (_dit), the VAE's, or the shot
    detector's; None for any other file."""
    if found in DITS.values() or (found is None and _dit(tensors) is not None):
        return "dit"
    if found is VAE:
        return "vae"
    if found is TRANSNETV2:
        return "detector"
    return None


def _read(path: Path) -> dict[str, TensorInfo] | str:
    """The tensors of the safetensors file at path, as its header declares them, validated as the
    safetensors library validates a file (_parse); or what a file of another format is, in words
    (_other). Reads the fewest bytes: the header's length (8), then the header, never the data.
    Raises ModelError when the file is missing, not a regular file, empty, cut short or
    damaged."""
    if not path.is_file():
        if os.path.lexists(path):
            raise ModelError(f"{path}: not a regular file")
        raise ModelError(f"{path}: {MISSING}")
    try:
        with path.open("rb") as file:
            size = os.fstat(file.fileno()).st_size
            # Empty: what a download failed before its first byte leaves, not a file of another
            # kind, so refused as one cut short is, without V1's tail. wget -O truncates its
            # file at once (GNU Wget's manual, -O), and leaves it empty on a 404 (Wget 1.21.4).
            if size == 0:
                raise ModelError(f"{path}: empty (0 bytes); {AGAIN}")
            # 1 to 8 bytes: too few for a safetensors header's length and the brace that tell the
            # format (below), and for any model file's header (GGUF's 24 bytes, ggml's
            # docs/gguf.md; a zip's local file header 30, APPNOTE.TXT 4.3.7): what a download cut
            # short leaves, refused as one, its size said, without V1's tail. "GGUF" or "PK" 3 4
            # alone is such a file, not a GGUF file or a checkpoint.
            if size < 9:
                raise ModelError(
                    f"{path}: cut short, {_count(size, 'byte')}, too few for any model file;"
                    f" {AGAIN}"
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
            if start.startswith(b"GGUF") or start[8:] != b"{":
                return _other(start)
            length = int.from_bytes(start[:8], "little")
            if length == 0 or length > HEADER_LIMIT:
                raise _damaged(
                    path,
                    f"a header of {length:,} bytes, where safetensors takes 1 to {HEADER_LIMIT:,}",
                )
            if 8 + length > size:
                raise ModelError(
                    f"{path}: cut short inside its header, {size:,} bytes of at least"
                    f" {8 + length:,}; {AGAIN}"
                )
            header = start[8:] + file.read(length - 1)
    except OSError as error:
        raise ModelError(f"{path}: {error.strerror}") from None
    tensors, data = _parse(path, header)
    end = 8 + length + data
    if size < end:
        raise ModelError(f"{path}: cut short, {size:,} of {end:,} bytes; {AGAIN}")
    if size > end:
        raise _damaged(path, f"{_count(size - end, 'byte')} after its tensors' data")
    return tensors


def _other(start: bytes) -> str:
    """What a file that isn't safetensors is, by its first bytes (start): GGUF's signature,
    "GGUF" (ggml's docs/gguf.md); a zip's, "PK" 3 4, torch.save's format from PyTorch 1.6; or a
    pickle's protocol 2 to 5 (pickle's PROTO opcode, 0x80, then the protocol), torch.save's
    format before it."""
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
