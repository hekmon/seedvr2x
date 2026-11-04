"""A GGUF file's header, read without torch and without its tensors' data (DESIGN.md, Weights,
"The dynamic GGUF before v0.1.0 too"): the metadata, each tensor's name, ggml type, shape and where
its bytes lie, and the checks that tell a whole file from a cut or damaged one.

Written from GGUF's format specification (ggml's docs/gguf.md, llama.cpp, MIT), not adapted from
another reader. The format, version 3, all little-endian:

    magic "GGUF" | version u32 | tensor count u64 | metadata count u64
    metadata: key (string) | value type u32 | value, repeated
    tensor infos: name (string) | dimensions u32 | each dimension u64 | ggml type u32 | offset u64
    padding to the alignment (general.alignment, a u32 in the metadata, else 32), then the data

a string being a u64 length then that many bytes of UTF-8, an array its element type u32, a count
u64 and the elements; the offsets in bytes from the start of the data, each a multiple of the
alignment. GGUF lists dimensions the way ggml does, the fastest-varying first: a PyTorch tensor of
shape (9216, 3072) has dimensions [3072, 9216], so the shapes here are the dimensions reversed.

Everything claimed is bounded before it is read or allocated: a count or a length the file cannot
hold is refused (GgufError), never trusted."""

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from seedvr2x.errors import Refusal

MAGIC = b"GGUF"
# The only version read: 3 is the current one (it added big-endian files); version 2 has the same
# layout but for what ggml no longer writes, version 1 has 32-bit counts.
VERSION = 3
# The alignment of the data and of each tensor in it when the metadata names none (docs/gguf.md,
# general.alignment).
ALIGNMENT = 32
# ggml's most dimensions a tensor has (ggml.h, GGML_MAX_DIMS).
MAX_DIMENSIONS = 4
# A header's most bytes, safetensors' limit (weights.HEADER_LIMIT) kept: the dynamic files' are
# 74,272, and a language model's, with its vocabulary, a few megabytes.
HEADER_LIMIT = 100_000_000
# The most values the metadata's arrays may hold in all, so that bytes of 1 each, which a header
# of HEADER_LIMIT can claim, never become that many Python objects.
VALUE_LIMIT = 10_000_000
# How deep arrays may nest in the metadata (the format allows an array of arrays).
DEPTH_LIMIT = 16
# The fewest bytes of a tensor's info (a name's empty length, the dimension count, the type, the
# offset: 8 + 4 + 4 + 8) and of a metadata entry (a key's empty length and the value type), which
# bound the counts a file of a given size can hold.
TENSOR_INFO_MIN = 24
ENTRY_MIN = 12

# GGUF's metadata value types (docs/gguf.md, gguf_metadata_value_type) that hold a fixed number of
# bytes, by id: their struct format and size. 7 is a bool, one byte, 0 or 1.
FIXED = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
STRING = 8
ARRAY = 9

# ggml's tensor types by id (ggml.h, enum ggml_type): name, values in a block, bytes in a block
# (ggml-common.h's block structs; a plain type's block is one value). As llama.cpp's gguf-py
# (0.19.0) lists them in GGML_QUANT_SIZES, to which tests/test_gguf_file.py compares every one
# when it is installed. Ids left out: those ggml removed (4, 5, 31 to 33, 36 to 38), Q8_1, an
# activation type no file holds, whose size gguf-py gives as 40 where the block struct of ggml.h
# has 36 (two half floats and 32 bytes), and the newest (NVFP4, Q1_0), which no file of seedvr2x's
# holds: a file with one is refused as of an unknown type, not read on a guess.
TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}
# A type's block, by name: values and bytes.
BLOCKS = {name: (values, size) for name, values, size in TYPES.values()}

# ggml reads sizes and offsets as signed 64-bit numbers: nothing from 2^63 on.
LIMIT = 2**63

# A metadata value: a number, a bool, a string, or an array of them (and of arrays).
Value = int | float | bool | str | list["Value"]


class GgufError(Refusal):
    """A GGUF header seedvr2x refuses, in words that name no file (the caller adds it). `kind`:
    "cut" for a file that stops inside what it claims (a download cut short: fetch it again),
    "version" for a version other than 3 (or big-endian), "type" for a tensor of a ggml type
    unknown here, "magic" for no GGUF signature, "damaged" for anything else wrong."""


@dataclass(frozen=True)
class GgufTensor:
    """A tensor of a GGUF file: its name, its ggml type by name (F16, Q4_K...), its PyTorch shape
    (GGUF's dimensions reversed), where its bytes begin, in bytes from the start of the file, and
    how many bytes they take."""

    name: str
    type: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GgufHeader:
    """What a GGUF file's header holds: its version, its metadata by key (in file order), the
    alignment of its data in bytes, where the data starts, in bytes from the start of the file, and
    its tensors in file order. Each tensor's bytes are inside the file, aligned, and apart."""

    version: int
    metadata: dict[str, Value]
    alignment: int
    data_start: int
    tensors: tuple[GgufTensor, ...]


def read(path: Path) -> GgufHeader:
    """The header of the GGUF file at path (header_of). Raises GgufError, or OSError."""
    with path.open("rb") as file:
        return header_of(file, os.fstat(file.fileno()).st_size)


def header_of(file: BinaryIO, size: int) -> GgufHeader:
    """The header of a GGUF file open for reading, size bytes long, read from its start; checked
    against size as the file holds tensors' data (see the module). Reads the header's bytes and
    none of the data."""
    file.seek(0)
    cursor = _Cursor(file, size)
    if cursor.take(4, "its signature") != MAGIC:
        raise GgufError("not a GGUF file: no GGUF signature", kind="magic")
    raw = cursor.take(4, "its version")
    version = int.from_bytes(raw, "little")
    if version != VERSION:
        if int.from_bytes(raw, "big") == VERSION:
            raise GgufError(
                "a GGUF file in big-endian byte order, which seedvr2x doesn't read", kind="version"
            )
        raise GgufError(
            f"a GGUF file of version {version}, where seedvr2x reads {VERSION}", kind="version"
        )
    tensor_count = cursor.u64("its tensor count")
    entry_count = cursor.u64("its metadata count")
    cursor.fits(tensor_count, TENSOR_INFO_MIN, "tensors")
    cursor.fits(entry_count, ENTRY_MIN, "metadata entries")
    metadata: dict[str, Value] = {}
    for _ in range(entry_count):
        key = cursor.string("a metadata key")
        if key in metadata:
            raise GgufError(f"the metadata key {key} twice", kind="damaged")
        metadata[key] = cursor.value(f"the metadata {key}")
    alignment = metadata.get("general.alignment", ALIGNMENT)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise GgufError(
            f"its general.alignment is {alignment!r}, not a power of two", kind="damaged"
        )
    infos: list[GgufTensor] = []
    names: set[str] = set()
    for _ in range(tensor_count):
        info = _tensor_info(cursor)
        if info.name in names:
            raise GgufError(f"the tensor {info.name} twice", kind="damaged")
        names.add(info.name)
        infos.append(info)
    data_start = -(-cursor.position // alignment) * alignment
    if data_start > size:
        raise GgufError(
            f"cut short inside its header's padding, {size:,} bytes where its data starts at"
            f" {data_start:,}",
            kind="cut",
        )
    tensors = _placed(infos, data_start, alignment, size)
    return GgufHeader(version, metadata, alignment, data_start, tensors)


def _tensor_info(cursor: "_Cursor") -> GgufTensor:
    """The next tensor's info: its offset is relative to the data, for _placed to make absolute."""
    name = cursor.string("a tensor's name")
    dimensions = cursor.u32(f"the dimension count of {name}")
    if dimensions > MAX_DIMENSIONS:
        raise GgufError(
            f"{name}: {dimensions:,} dimensions, where ggml has at most {MAX_DIMENSIONS}",
            kind="damaged",
        )
    shape = tuple(reversed([cursor.u64(f"a dimension of {name}") for _ in range(dimensions)]))
    code = cursor.u32(f"the type of {name}")
    offset = cursor.u64(f"the offset of {name}")
    if code not in TYPES:
        raise GgufError(f"a GGUF file where {name} is of an unknown ggml type {code}", kind="type")
    kind, values, size = TYPES[code]
    count = math.prod(shape)
    if count >= LIMIT or offset >= LIMIT:
        raise GgufError(f"{name}: a size or an offset over 2^63", kind="damaged")
    # The last dimension in the file's order, the first listed (ne0), is the row ggml quantises a
    # block at a time.
    if shape and shape[-1] % values:
        raise GgufError(
            f"{name}: rows of {shape[-1]:,} values, which {kind} can't hold, {values} to a block",
            kind="damaged",
        )
    return GgufTensor(name, kind, shape, offset, count // values * size)


def _placed(
    infos: list[GgufTensor], start: int, alignment: int, size: int
) -> tuple[GgufTensor, ...]:
    """The tensors with their offsets made absolute (from the start of the file), after the
    checks: each offset a multiple of the alignment, no two overlapping, all within the file, and
    nothing in the file after the last one's padding."""
    position = 0
    end = 0
    for info in sorted(infos, key=lambda each: each.offset):
        if info.offset % alignment:
            raise GgufError(
                f"{info.name}: its data at {info.offset:,}, not a multiple of the alignment,"
                f" {alignment}",
                kind="damaged",
            )
        if info.offset < position:
            raise GgufError(
                f"{info.name}: its data at {info.offset:,}, inside the data before it, up to"
                f" {position:,}",
                kind="damaged",
            )
        position = info.offset + info.nbytes
        end = max(end, position)
    if start + end > size:
        raise GgufError(
            f"cut short, {size:,} of {start + end:,} bytes",
            kind="cut",
        )
    padded = -(-end // alignment) * alignment
    if size > start + padded:
        extra = size - start - padded
        raise GgufError(
            f"{extra:,} byte{'s' if extra != 1 else ''} after its last tensor's data and padding",
            kind="damaged",
        )
    return tuple(
        GgufTensor(each.name, each.type, each.shape, start + each.offset, each.nbytes)
        for each in infos
    )


class _Cursor:
    """A reader over the header: every read is bounded by the file's size and HEADER_LIMIT before
    it is made, so that a length the file cannot hold is refused, never allocated."""

    def __init__(self, file: BinaryIO, size: int) -> None:
        self.file = file
        self.size = size
        self.position = 0
        self.values = 0

    def take(self, count: int, what: str) -> bytes:
        if self.position + count > HEADER_LIMIT:
            raise GgufError(
                f"{what} takes {count:,} bytes at {self.position:,}, over the {HEADER_LIMIT:,}"
                " bytes a header may take",
                kind="damaged",
            )
        if self.position + count > self.size:
            raise GgufError(
                f"cut short inside its header, {what} takes {count:,} bytes at {self.position:,},"
                f" {max(self.size - self.position, 0):,} left of {self.size:,}",
                kind="cut",
            )
        data = self.file.read(count)
        if len(data) != count:
            raise GgufError(f"cut short inside its header, {what} at {self.position:,}", kind="cut")
        self.position += count
        return data

    def fits(self, count: int, each: int, what: str) -> None:
        """Refuse a claim of count things of at least `each` bytes that the header cannot hold."""
        if count * each > HEADER_LIMIT - self.position:
            raise GgufError(
                f"{count:,} {what}, more than a header of {HEADER_LIMIT:,} bytes holds",
                kind="damaged",
            )
        if count * each > self.size - self.position:
            raise GgufError(
                f"cut short inside its header, {count:,} {what} claimed and"
                f" {max(self.size - self.position, 0):,} bytes left",
                kind="cut",
            )

    def u32(self, what: str) -> int:
        return int.from_bytes(self.take(4, what), "little")

    def u64(self, what: str) -> int:
        return int.from_bytes(self.take(8, what), "little")

    def string(self, what: str) -> str:
        """A string: its length, then its bytes in UTF-8."""
        length = self.u64(f"the length of {what}")
        raw = self.take(length, what)
        try:
            return raw.decode()
        except UnicodeDecodeError:
            raise GgufError(f"{what} isn't UTF-8", kind="damaged") from None

    def value(self, what: str, depth: int = 0) -> Value:
        """The next metadata value, its type first."""
        code = self.u32(f"the type of {what}")
        return self.typed(code, what, depth)

    def typed(self, code: int, what: str, depth: int) -> Value:
        if code in FIXED:
            fmt, size = FIXED[code]
            return struct.unpack(f"<{fmt}", self.take(size, what))[0]
        if code == STRING:
            # A value's text is never used for a decision: one that isn't UTF-8 is kept readable
            # rather than refused, where a key or a tensor's name is refused.
            length = self.u64(f"the length of {what}")
            return self.take(length, what).decode(errors="replace")
        if code != ARRAY:
            raise GgufError(f"{what} is of an unknown value type {code}", kind="damaged")
        if depth >= DEPTH_LIMIT:
            raise GgufError(f"{what}: arrays nested over {DEPTH_LIMIT} deep", kind="damaged")
        element = self.u32(f"the element type of {what}")
        count = self.u64(f"the length of {what}")
        if element not in FIXED and element not in (STRING, ARRAY):
            raise GgufError(
                f"{what} is an array of an unknown value type {element}", kind="damaged"
            )
        # Each element takes at least its size: 8 for a string's length, 12 for an array's.
        smallest = FIXED[element][1] if element in FIXED else 8 if element == STRING else 12
        self.fits(count, smallest, f"elements of {what}")
        self.values += count
        if self.values > VALUE_LIMIT:
            raise GgufError(
                f"{what}: over {VALUE_LIMIT:,} values in the metadata's arrays", kind="damaged"
            )
        if element in FIXED:
            fmt, size = FIXED[element]
            return list(struct.unpack(f"<{count}{fmt}", self.take(count * size, what)))
        return [self.typed(element, what, depth + 1) for _ in range(count)]
