"""seedvr2x/gguf_file.py, a GGUF file's header read without torch or the tensors' data (DESIGN.md,
Weights): held to GGUF files written here by hand with struct, from the format's specification
(ggml's docs/gguf.md), never from the module's own code; to llama.cpp's own reader, gguf-py's
GGUFReader, where it is installed (the tests' reference, not a dependency); and, where
SEEDVR2X_MODEL_DIR holds seedvr2x's dynamic files, to their real headers.

The files are sparse: a header, then the tensors' data as a file's full length by truncate. How the
model check (weights.py) uses it is test_weights.py's."""

import itertools
import math
import os
import struct
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from seedvr2x import gguf_file
from seedvr2x.gguf_file import ALIGNMENT, TYPES, GgufError, GgufHeader, GgufTensor

# A tensor to write: its name, its ggml type by name, its PyTorch shape.
Spec = tuple[str, str, tuple[int, ...]]

IDS = {name: code for code, (name, _, _) in TYPES.items()}
FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
STRING, ARRAY = 8, 9


def text(value: str) -> bytes:
    """A GGUF string: its length as a u64, then its bytes."""
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def payload(code: int, value: object) -> bytes:
    """A metadata value of type code, without its type: a number, a string, or an array, given as
    (the element's type, the items), whose items are payloads of that type."""
    if code in FORMATS:
        return struct.pack(f"<{FORMATS[code]}", value)
    if code == STRING:
        assert isinstance(value, str)
        return text(value)
    assert code == ARRAY and isinstance(value, tuple)
    element, items = value
    return struct.pack("<IQ", element, len(items)) + b"".join(payload(element, i) for i in items)


def entry(key: str, code: int, value: object) -> bytes:
    """A metadata entry: its key, its value's type, its value."""
    return text(key) + struct.pack("<I", code) + payload(code, value)


def aligned(number: int, alignment: int) -> int:
    return -(-number // alignment) * alignment


def placed(specs: Sequence[Spec], alignment: int = ALIGNMENT) -> tuple[list[int], int]:
    """Where each tensor's data goes in the data section, in order, each at a multiple of the
    alignment, and the section's length, the last tensor's padding included."""
    offsets: list[int] = []
    end = 0
    for _, kind, shape in specs:
        values, size = gguf_file.BLOCKS[kind]
        offsets.append(end)
        end = aligned(end + math.prod(shape) // values * size, alignment)
    return offsets, end


def body(
    specs: Sequence[Spec],
    entries: Sequence[bytes] = (),
    *,
    alignment: int = ALIGNMENT,
    version: int = 3,
    offsets: Sequence[int] | None = None,
    tensors: int | None = None,
    metadata: int | None = None,
) -> bytes:
    """A GGUF header for these tensors, its metadata entries (entry) and the offsets given or each
    tensor's in turn, without its padding; the counts it claims given when they are to be wrong.
    The dimensions are listed the way GGUF lists them, the shape reversed."""
    where = placed(specs, alignment)[0] if offsets is None else offsets
    raw = b"GGUF" + struct.pack(
        "<IQQ",
        version,
        len(specs) if tensors is None else tensors,
        len(entries) if metadata is None else metadata,
    )
    raw += b"".join(entries)
    for (name, kind, shape), offset in zip(specs, where, strict=True):
        raw += text(name) + struct.pack("<I", len(shape))
        raw += b"".join(struct.pack("<Q", each) for each in reversed(shape))
        raw += struct.pack("<IQ", IDS[kind], offset)
    return raw


def header(
    specs: Sequence[Spec],
    entries: Sequence[bytes] = (),
    *,
    alignment: int = ALIGNMENT,
    **claims: Any,
) -> bytes:
    """body() and the padding to the alignment that ends the header."""
    raw = body(specs, entries, alignment=alignment, **claims)
    return raw + bytes(aligned(len(raw), alignment) - len(raw))


def write_gguf(
    path: Path,
    specs: Sequence[Spec],
    entries: Sequence[bytes] = (),
    *,
    alignment: int = ALIGNMENT,
    **claims: Any,
) -> Path:
    """A GGUF file of these tensors: its header, then nothing to its full length, by truncate (a
    sparse file). `claims` go to header()."""
    raw = header(specs, entries, alignment=alignment, **claims)
    with path.open("wb") as file:
        file.write(raw)
        file.truncate(len(raw) + placed(specs, alignment)[1])
    return path


# Tensors of every kind of shape the files hold: a scalar-like one-value row, a vector, a matrix of
# quantised rows, one of 3 dimensions.
SMALL: list[Spec] = [
    ("norm.weight", "F32", (64,)),
    ("a.weight", "F16", (4, 8)),
    ("m.weight", "Q4_K", (3, 512)),
    ("n.weight", "Q3_K", (2, 256)),
    ("o.weight", "Q5_K", (5, 256)),
    ("q.weight", "Q8_0", (2, 64)),
    ("c.weight", "F16", (2, 3, 5)),
]


def test_tensors_of_every_size(tmp_path: Path) -> None:
    # Each tensor's bytes are its values over its type's block, times the block's bytes, its shape
    # PyTorch's (the file lists the dimensions the other way round), its offset absolute.
    path = write_gguf(tmp_path / "x.gguf", SMALL)
    found = gguf_file.read(path)
    assert [t.name for t in found.tensors] == [name for name, _, _ in SMALL]
    assert [t.type for t in found.tensors] == [kind for _, kind, _ in SMALL]
    assert [t.shape for t in found.tensors] == [shape for _, _, shape in SMALL]
    assert [t.nbytes for t in found.tensors] == [
        256,
        64,
        3 * 2 * 144,
        2 * 110,
        5 * 176,
        2 * 2 * 34,
        60,
    ]
    offsets = placed(SMALL)[0]
    assert [t.offset - found.data_start for t in found.tensors] == offsets
    assert found.data_start == len(header(SMALL)) and found.data_start % ALIGNMENT == 0
    assert found.version == 3 and found.alignment == ALIGNMENT and found.metadata == {}
    assert path.stat().st_size == found.data_start + placed(SMALL)[1]
    assert isinstance(found, GgufHeader) and isinstance(found.tensors[0], GgufTensor)


@pytest.mark.parametrize("code", sorted(TYPES))
def test_every_type_sized(tmp_path: Path, code: int) -> None:
    # A row of two blocks of each type the module knows: 2 * the block's bytes.
    kind, values, size = TYPES[code]
    path = write_gguf(tmp_path / "x.gguf", [("t", kind, (3, 2 * values))])
    (tensor,) = gguf_file.read(path).tensors
    assert tensor.type == kind and tensor.nbytes == 3 * 2 * size


def test_known_sizes() -> None:
    # The sizes the dynamic files hold, from ggml-common.h's block structs: Q3_K is 110 bytes of 256
    # values (2 + 64 + 32 + 12), Q4_K 144 (2 + 2 + 12 + 128), Q5_K 176 (2 + 2 + 12 + 32 + 128).
    assert TYPES[11] == ("Q3_K", 256, 2 + 256 // 4 + 256 // 8 + 12)
    assert TYPES[12] == ("Q4_K", 256, 2 + 2 + 12 + 256 // 2)
    assert TYPES[13] == ("Q5_K", 256, 2 + 2 + 12 + 256 // 8 + 256 // 2)
    assert TYPES[14] == ("Q6_K", 256, 128 + 64 + 16 + 2) and TYPES[8] == ("Q8_0", 32, 34)
    assert TYPES[2] == ("Q4_0", 32, 18) and TYPES[1] == ("F16", 1, 2) and TYPES[0] == ("F32", 1, 4)
    assert TYPES[30] == ("BF16", 1, 2) and TYPES[28] == ("F64", 1, 8)
    assert len({name for name, _, _ in TYPES.values()}) == len(TYPES)


def test_types_equal_gguf_py() -> None:
    # llama.cpp's gguf-py (0.19.0) lists the same ids, names and sizes, but for the ids left out
    # (the module says which and why).
    constants = pytest.importorskip("gguf.constants")
    sizes = constants.GGML_QUANT_SIZES
    for code, (name, values, size) in TYPES.items():
        kind = constants.GGMLQuantizationType(code)
        assert kind.name == name and sizes[kind] == (values, size), name
    left_out = {int(kind) for kind in sizes} - set(TYPES)
    assert left_out == {9, 40, 41}  # Q8_1, NVFP4, Q1_0


def test_tensors_in_file_order_whatever_their_offsets(tmp_path: Path) -> None:
    # The list follows the file's tensor infos; their data may be in another order: the third
    # tensor's first (864 bytes), then the first's (256), then the second's (64).
    specs = SMALL[:3]
    offsets = [864, 1120, 0]
    path = write_gguf(tmp_path / "x.gguf", specs, offsets=offsets)
    found = gguf_file.read(path)
    assert [t.name for t in found.tensors] == ["norm.weight", "a.weight", "m.weight"]
    assert [t.offset - found.data_start for t in found.tensors] == offsets


@pytest.mark.parametrize("alignment", [16, 32, 64, 128, 4096])
def test_alignment_named_by_the_metadata(tmp_path: Path, alignment: int) -> None:
    # general.alignment, a u32 in the metadata, sets the padding of the header and of each tensor;
    # without it, 32.
    entries = [entry("general.alignment", 4, alignment)]
    path = write_gguf(tmp_path / "x.gguf", SMALL, entries, alignment=alignment)
    found = gguf_file.read(path)
    assert found.alignment == alignment and found.metadata == {"general.alignment": alignment}
    assert found.data_start == aligned(len(header(SMALL, entries, alignment=alignment)), alignment)
    assert all((t.offset - found.data_start) % alignment == 0 for t in found.tensors)
    assert found.tensors[-1].offset + found.tensors[-1].nbytes <= path.stat().st_size


def test_header_ending_on_the_alignment_has_no_padding(tmp_path: Path) -> None:
    # A header whose length is a multiple of the alignment is followed by the data at once.
    specs: list[Spec] = [("t", "F16", (4,))]
    fill = 128 - len(body(specs, [entry("k", 4, 1), entry("p", STRING, "")]))
    entries = [entry("k", 4, 1), entry("p", STRING, "x" * fill)]
    assert len(body(specs, entries)) == 128
    path = write_gguf(tmp_path / "x.gguf", specs, entries)
    assert gguf_file.read(path).data_start == 128 and path.stat().st_size == 128 + 32


# Every metadata value type of the format (docs/gguf.md), as a key, a type and a value.
VALUES: list[tuple[str, int, object, object]] = [
    ("u8", 0, 255, 255),
    ("i8", 1, -128, -128),
    ("u16", 2, 65535, 65535),
    ("i16", 3, -32768, -32768),
    ("u32", 4, 2**32 - 1, 2**32 - 1),
    ("i32", 5, -(2**31), -(2**31)),
    ("f32", 6, 0.5, 0.5),
    ("bool", 7, True, True),
    ("false", 7, False, False),
    ("text", STRING, "seedvr2x", "seedvr2x"),
    ("unicode", STRING, "dynamique, シード", "dynamique, シード"),
    ("empty", STRING, "", ""),
    ("u64", 10, 2**64 - 1, 2**64 - 1),
    ("i64", 11, -(2**63), -(2**63)),
    ("f64", 12, 1e300, 1e300),
    ("ints", ARRAY, (4, [1, 2, 3]), [1, 2, 3]),
    ("floats", ARRAY, (6, [0.25, -2.0]), [0.25, -2.0]),
    ("bytes", ARRAY, (0, list(range(256))), list(range(256))),
    ("bools", ARRAY, (7, [True, False]), [True, False]),
    ("strings", ARRAY, (STRING, ["a", "", "bc"]), ["a", "", "bc"]),
    ("none", ARRAY, (4, []), []),
    ("nested", ARRAY, (ARRAY, [(4, [1]), (STRING, ["x", "y"]), (4, [])]), [[1], ["x", "y"], []]),
    ("deeper", ARRAY, (ARRAY, [(ARRAY, [(0, [7])])]), [[[7]]]),
]


def test_every_metadata_type(tmp_path: Path) -> None:
    entries = [entry(key, code, value) for key, code, value, _ in VALUES]
    path = write_gguf(tmp_path / "x.gguf", SMALL, entries)
    found = gguf_file.read(path)
    assert found.metadata == {key: expected for key, _, _, expected in VALUES}
    assert list(found.metadata) == [key for key, *_ in VALUES]  # the file's order
    assert type(found.metadata["bool"]) is bool and type(found.metadata["f32"]) is float
    assert [t.name for t in found.tensors] == [name for name, _, _ in SMALL]


def test_metadata_text_is_kept_readable_when_not_utf8(tmp_path: Path) -> None:
    # A value's text decides nothing, so one that isn't UTF-8 is read with a replacement character;
    # a key's or a tensor's name, which do, are refused (below).
    raw = struct.pack("<Q", 3) + b"a\xffb"
    entries = [text("note") + struct.pack("<I", STRING) + raw]
    found = gguf_file.read(write_gguf(tmp_path / "x.gguf", [], entries))
    assert found.metadata == {"note": "a�b"}


def test_no_tensors(tmp_path: Path) -> None:
    found = gguf_file.read(write_gguf(tmp_path / "x.gguf", []))
    assert found.tensors == () and found.data_start == 32 and found.metadata == {}


def refusal(path: Path) -> GgufError:
    with pytest.raises(GgufError) as error:
        gguf_file.read(path)
    return error.value


def test_not_gguf(tmp_path: Path) -> None:
    path = tmp_path / "x.gguf"
    path.write_bytes(b"GGUX" + bytes(40))
    assert refusal(path).kind == "magic"


@pytest.mark.parametrize("version", [0, 1, 2, 4, 1_000_000])
def test_other_versions(tmp_path: Path, version: int) -> None:
    path = write_gguf(tmp_path / "x.gguf", SMALL, version=version)
    error = refusal(path)
    assert error.kind == "version"
    assert str(error) == f"a GGUF file of version {version}, where seedvr2x reads 3"


def test_big_endian(tmp_path: Path) -> None:
    path = tmp_path / "x.gguf"
    path.write_bytes(b"GGUF" + struct.pack(">IQQ", 3, 0, 0) + bytes(8))
    error = refusal(path)
    assert error.kind == "version" and "big-endian" in str(error)


def test_unknown_type(tmp_path: Path) -> None:
    # An id ggml removed (4, Q4_2), one it doesn't know yet, and one gguf-py knows that this table
    # leaves out (Q1_0, 41, newer than the files seedvr2x reads).
    for code in (4, 31, 40, 41, 99, 2**32 - 1):
        raw = bytearray(header(SMALL[:2]))
        # The second tensor's type field: the u32 before its offset, the last 12 bytes of the
        # header's body, before the padding.
        end = len(body(SMALL[:2]))
        raw[end - 12 : end - 8] = struct.pack("<I", code)
        path = tmp_path / "x.gguf"
        path.write_bytes(bytes(raw) + bytes(placed(SMALL[:2])[1]))
        error = refusal(path)
        assert error.kind == "type"
        assert str(error) == f"a GGUF file where a.weight is of an unknown ggml type {code}"


def test_every_prefix_is_cut_short(tmp_path: Path) -> None:
    # A file cut anywhere before its last tensor's data ends is refused as cut short, never read
    # past its end: every length from 0 to the data's end, the padding after the last tensor
    # being optional.
    entries = [entry(key, code, value) for key, code, value, _ in VALUES[:12]]
    path = write_gguf(tmp_path / "x.gguf", SMALL, entries)
    raw = path.read_bytes()
    found = gguf_file.read(path)
    last = found.tensors[-1].offset + found.tensors[-1].nbytes
    for length in range(last):
        with pytest.raises(GgufError) as error:
            gguf_file.header_of(BytesIO(raw[:length]), length)
        assert error.value.kind == "cut", (length, error.value)
    assert gguf_file.header_of(BytesIO(raw[:last]), last).tensors == found.tensors


def test_cut_data_said(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "x.gguf", SMALL)
    found = gguf_file.read(path)
    os.truncate(path, found.data_start + 100)
    error = refusal(path)
    assert error.kind == "cut"
    last = found.tensors[-1].offset + found.tensors[-1].nbytes
    assert str(error) == f"cut short, {found.data_start + 100:,} of {last:,} bytes"


def test_bytes_after_the_data(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "x.gguf", SMALL)
    found = gguf_file.read(path)
    last = found.tensors[-1].offset + found.tensors[-1].nbytes
    padding = aligned(last, ALIGNMENT) - last
    with path.open("ab") as file:
        file.write(b"x")
    error = refusal(path)
    assert error.kind == "damaged"
    assert str(error) == "1 byte after its last tensor's data and padding"
    os.truncate(path, last + padding)  # the padding is allowed, and the file may omit it
    assert gguf_file.read(path).tensors == found.tensors


def entries_with(key: str, code: int, value: object) -> list[bytes]:
    return [entry(key, code, value)]


def test_header_claims_the_file_cannot_hold(tmp_path: Path) -> None:
    # Counts and lengths are checked against the file before anything is read or allocated: a claim
    # of 2^60 tensors, entries, bytes or elements is refused at once, in a few bytes' file.
    path = tmp_path / "x.gguf"
    huge = 2**60
    cases: dict[str, tuple[bytes, str]] = {
        "tensors": (header([], tensors=huge), "damaged"),
        "entries": (header([], metadata=huge), "damaged"),
        "a few too many tensors": (header([], tensors=10), "cut"),
        "a few too many entries": (header([], metadata=10), "cut"),
        "a key of 2^60 bytes": (header([], [struct.pack("<Q", huge)], metadata=1), "damaged"),
        "a key of 10 GB": (header([], [struct.pack("<Q", 10 * 2**30)], metadata=1), "damaged"),
        "a key of 1 MB in a short file": (
            header([], [struct.pack("<Q", 2**20)], metadata=1),
            "cut",
        ),
        "a string of 2^62": (
            header([], [text("k") + struct.pack("<IQ", STRING, 2**62)]),
            "damaged",
        ),
        "an array of 2^60": (
            header([], [text("k") + struct.pack("<IIQ", ARRAY, 4, huge)], metadata=1),
            "damaged",
        ),
        "an array of strings of 2^30": (
            header([], [text("k") + struct.pack("<IIQ", ARRAY, STRING, 2**30)], metadata=1),
            "damaged",
        ),
        "an array of arrays of 1000": (
            header([], [text("k") + struct.pack("<IIQ", ARRAY, ARRAY, 1000)], metadata=1),
            "cut",
        ),
    }
    for what, (raw, kind) in cases.items():
        # With room after the claim for the smallest thing it could be, so that only the claim
        # is wrong.
        path.write_bytes(raw + bytes(64))
        assert refusal(path).kind == kind, what


def test_tensor_info_claims(tmp_path: Path) -> None:
    path = tmp_path / "x.gguf"
    # A name of 2^60 bytes.
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + struct.pack("<Q", 2**60) + b"t" + bytes(64)
    )
    assert refusal(path).kind == "damaged"
    # Five dimensions, where ggml has at most 4; a dimension of 2^63; a product over 2^63.
    for dimensions, what in (
        ([1, 1, 1, 1, 1], "5 dimensions, where ggml has at most 4"),
        ([2**63], "a size or an offset over 2^63"),
        ([2**40, 2**40], "a size or an offset over 2^63"),
    ):
        raw = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + text("t"))
        raw += struct.pack("<I", len(dimensions))
        raw += b"".join(struct.pack("<Q", each) for each in dimensions)
        raw += struct.pack("<IQ", 1, 0)
        raw += bytes(aligned(len(raw), 32) - len(raw))
        path.write_bytes(bytes(raw))
        error = refusal(path)
        assert error.kind == "damaged" and what in str(error), what


def test_rows_must_hold_whole_blocks(tmp_path: Path) -> None:
    # ggml quantises a row (the first dimension GGUF lists) a block at a time.
    path = write_gguf(tmp_path / "x.gguf", [("w", "Q4_K", (4, 300))], offsets=[0])
    error = refusal(path)
    assert error.kind == "damaged"
    assert str(error) == "w: rows of 300 values, which Q4_K can't hold, 256 to a block"


def test_misaligned_and_overlapping_offsets(tmp_path: Path) -> None:
    specs = SMALL[:3]
    offsets = placed(specs)[0]
    error = refusal(write_gguf(tmp_path / "x.gguf", specs, offsets=[0, offsets[1] + 1, offsets[2]]))
    assert error.kind == "damaged" and "a.weight: its data at 257, not a multiple" in str(error)
    # The third tensor starting inside the second.
    error = refusal(
        write_gguf(tmp_path / "y.gguf", specs, offsets=[0, offsets[1], offsets[1] + 32])
    )
    assert error.kind == "damaged" and "inside the data before it, up to" in str(error)
    # Two tensors at one offset.
    error = refusal(write_gguf(tmp_path / "z.gguf", specs, offsets=[0, 0, offsets[2]]))
    assert error.kind == "damaged" and "inside the data before it" in str(error)
    # A tensor that ends past the file.
    error = refusal(write_gguf(tmp_path / "w.gguf", specs, offsets=[0, offsets[1], 10_000_000]))
    assert error.kind == "cut"


def test_repeats_and_bad_alignments(tmp_path: Path) -> None:
    error = refusal(write_gguf(tmp_path / "x.gguf", [], [entry("k", 4, 1), entry("k", 4, 2)]))
    assert error.kind == "damaged" and str(error) == "the metadata key k twice"
    error = refusal(write_gguf(tmp_path / "y.gguf", [SMALL[0], SMALL[0]], offsets=[0, 256]))
    assert error.kind == "damaged" and str(error) == "the tensor norm.weight twice"
    for code, value in ((4, 0), (4, 48), (4, 3), (STRING, "32"), (10, 2**40 + 1)):
        entries = entries_with("general.alignment", code, value)
        error = refusal(write_gguf(tmp_path / "z.gguf", [], entries))
        assert error.kind == "damaged" and "general.alignment" in str(error), (code, value)


def test_names_that_are_not_utf8(tmp_path: Path) -> None:
    bad = struct.pack("<Q", 2) + b"\xff\xfe"
    path = tmp_path / "x.gguf"
    path.write_bytes(header([], [bad + struct.pack("<II", 4, 1)], metadata=1))
    assert refusal(path).kind == "damaged"
    raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + bad + struct.pack("<IIIQ", 1, 4, 0, 0)
    path.write_bytes(raw + bytes(aligned(len(raw), 32) - len(raw)) + bytes(64))
    error = refusal(path)
    assert error.kind == "damaged" and "a tensor's name isn't UTF-8" in str(error)


def test_unknown_value_types_and_nesting(tmp_path: Path) -> None:
    path = tmp_path / "x.gguf"
    path.write_bytes(header([], [text("k") + struct.pack("<II", 13, 0)], metadata=1))
    assert "unknown value type 13" in str(refusal(path))
    path.write_bytes(header([], [text("k") + struct.pack("<IIQ", ARRAY, 13, 0)], metadata=1))
    assert "an array of an unknown value type 13" in str(refusal(path))
    # Arrays nested over the limit, each holding one array: refused, not a RecursionError.
    inner = (4, [1])
    for _ in range(gguf_file.DEPTH_LIMIT + 5):
        inner = (ARRAY, [inner])  # type: ignore[assignment]
    path.write_bytes(header([], [entry("k", ARRAY, inner)]))
    error = refusal(path)
    assert error.kind == "damaged" and "nested over 16 deep" in str(error)
    ok = (4, [1])
    for _ in range(gguf_file.DEPTH_LIMIT - 1):
        ok = (ARRAY, [ok])  # type: ignore[assignment]
    path.write_bytes(header([], [entry("k", ARRAY, ok)]))
    assert "k" in gguf_file.read(path).metadata


def test_metadata_values_are_bounded(tmp_path: Path) -> None:
    # Bytes of 1 each become Python ints: over VALUE_LIMIT of them are refused, though the file
    # holds them, and the header's own limit stands.
    count = gguf_file.VALUE_LIMIT + 1
    raw = header([], [text("k") + struct.pack("<IIQ", ARRAY, 0, count)], metadata=1)
    path = tmp_path / "x.gguf"
    with path.open("wb") as file:
        file.write(raw)
        file.truncate(len(raw) + count)
    error = refusal(path)
    assert error.kind == "damaged" and "over 10,000,000 values" in str(error)


def test_many_tensors_read_fast(tmp_path: Path) -> None:
    # Thousands of tensors, as a language model's file has: bounded work, every one found.
    specs: list[Spec] = [(f"blk.{i}.w", "Q4_0", (1, 32)) for i in range(5000)]
    found = gguf_file.read(write_gguf(tmp_path / "x.gguf", specs))
    assert len(found.tensors) == 5000 and found.tensors[-1].name == "blk.4999.w"


# gguf-py, llama.cpp's own reader, as the reference for what a file holds.


def same_as_gguf_py(path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    mine = gguf_file.read(path)
    reader = gguf.GGUFReader(str(path))
    assert reader.data_offset == mine.data_start
    assert reader.alignment == mine.alignment
    assert len(reader.tensors) == len(mine.tensors)
    for theirs, each in zip(reader.tensors, mine.tensors, strict=True):
        assert theirs.name == each.name
        assert theirs.tensor_type.name == each.type
        assert tuple(int(d) for d in reversed(theirs.shape)) == each.shape
        assert int(theirs.data_offset) == each.offset
        assert int(theirs.n_bytes) == each.nbytes


def test_equal_to_gguf_py_on_the_synthetic_files(tmp_path: Path) -> None:
    entries = [entry(key, code, value) for key, code, value, _ in VALUES[:12]]
    same_as_gguf_py(write_gguf(tmp_path / "a.gguf", SMALL, entries))
    entries = [entry("general.alignment", 4, 64)]
    same_as_gguf_py(write_gguf(tmp_path / "b.gguf", SMALL, entries, alignment=64))
    every = [(f"t{code}", name, (2, values)) for code, (name, values, _) in TYPES.items()]
    same_as_gguf_py(write_gguf(tmp_path / "c.gguf", every))


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
DYNAMIC = {
    "seedvr2x_ema_7b_dyn.gguf": (4_757_055_264, {"Q3_K": 17, "Q4_K": 203, "Q5_K": 68}),
    "seedvr2x_ema_7b_sharp_dyn.gguf": (4_757_129_024, {"Q3_K": 17, "Q4_K": 204, "Q5_K": 67}),
}


@pytest.mark.parametrize("name", sorted(DYNAMIC))
def test_the_real_headers(name: str) -> None:
    # seedvr2x's dynamic files: GGUF version 3, the default alignment, 1,128 tensors, 840 F16 and
    # the 288 matrices in Q3_K, Q4_K and Q5_K; the data ends where the file does; and gguf-py reads
    # the same header.
    if not MODELS or not (Path(MODELS) / name).is_file():
        pytest.skip(f"{name}: needs SEEDVR2X_MODEL_DIR")
    path = Path(MODELS) / name
    size, quantised = DYNAMIC[name]
    found = gguf_file.read(path)
    assert path.stat().st_size == size
    assert (
        found.version == 3 and found.alignment == 32 and "general.alignment" not in found.metadata
    )
    assert len(found.tensors) == 1128
    kinds = {kind: sum(1 for t in found.tensors if t.type == kind) for kind in ("F16", *quantised)}
    assert kinds == {"F16": 840, **quantised}
    last = max(found.tensors, key=lambda each: each.offset)
    assert last.offset + last.nbytes == size
    assert found.metadata["general.architecture"] == "seedvr"
    same_as_gguf_py(path)


def test_each_prefix_of_a_real_header_is_cut_short() -> None:
    # A real header, cut at a few places across it.
    if not MODELS or not (Path(MODELS) / "seedvr2x_ema_7b_dyn.gguf").is_file():
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    path = Path(MODELS) / "seedvr2x_ema_7b_dyn.gguf"
    found = gguf_file.read(path)
    with path.open("rb") as file:
        raw = file.read(found.data_start)
    for length in itertools.chain(range(0, 200), range(200, found.data_start, 997)):
        with pytest.raises(GgufError) as error:
            gguf_file.header_of(BytesIO(raw[:length]), length)
        assert error.value.kind == "cut"
