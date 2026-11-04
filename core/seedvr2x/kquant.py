"""The dynamic GGUF's block matrices (DESIGN.md, Weights, "The dynamic GGUF before v0.1.0 too";
models/FORMATS.md, "Dynamic GGUF"): a type per matrix among ggml's K-quants Q3_K, Q4_K and Q5_K,
dequantised on the device just before each multiply and freed after it.

What the files hold. Each of the 288 attention and MLP matrices of the 7B's blocks
(weights.MATRIX) is `<layer>.weight`, `out` rows of `in` values, in one ggml type. ggml stores a
row's `in` values contiguously, in blocks of 256 (QK_K): a row is in / 256 blocks of the type's
bytes, and the tensor, as GGUFLinear's weight holds it, uint8 (out, in / 256 * block bytes), the
file's own bytes row by row. A block, little-endian (ggml-common.h at llama.cpp v0.8.0,
block_q3_K, block_q4_K, block_q5_K):
- Q3_K, 110 bytes: hmask[32] (the quants' high bit), qs[64] (their low 2 bits), scales[12] (16
  scales of 6 bits), d (float16); 3.4375 bits a value;
- Q4_K, 144 bytes: d, dmin (float16), scales[12] (8 scales and 8 minimums of 6 bits), qs[128]
  (4 bits); 4.5 bits a value;
- Q5_K, 176 bytes: d, dmin (float16), scales[12], qh[32] (the quants' high bit), qs[128] (their
  low 4 bits); 5.5 bits a value.

The arithmetic is ggml's own, in float32, written from its C (llama.cpp v0.8.0, MIT,
commit 8642541, the tag's; NOTICE), ggml/src/ggml-quants.c: dequantize_row_q3_K,
dequantize_row_q4_K, dequantize_row_q5_K and get_scale_min_k4 (the 6-bit scales of Q4_K and Q5_K).
The fp16 d and dmin are widened to float32 (exact); the scales, minimums and quants are
integers, converted to float32 exactly; then
- Q3_K: y = (d * (scale - 32)) * q, q in [-4, 3];
- Q4_K: y = (d * scale) * q - (dmin * minimum), q in [0, 15];
- Q5_K: y = (d * scale) * q - (dmin * minimum), q in [0, 31];
each multiply and the subtraction a float32 rounding of its own, in ggml's order. They are written
as separate torch operations on purpose: one kernel could contract `a * b - c` into a fused
multiply-add (nvcc does by default) and round once, another bit; separate kernels cannot. The
tests hold the result bit for bit to the `gguf` package's (llama.cpp's own gguf-py, quants.py,
numpy's float32, no fused operation) on synthetic blocks and on the dynamic files' matrices
(tests/test_kquant.py), on the CPU and on CUDA. A build of ggml's C may contract the same
expression (GCC's -ffp-contract=fast with FMA hardware): float32's last bit at most, under the
rounding to bfloat16 that follows (below).

At run time (GGUFLinear) the matrix is dequantised straight into the dtype autocast gives a linear
at that call (bfloat16 under the pipeline's, model.sample), BLOCKS blocks at a time: each chunk in
float32, then rounded into the output, allocated once, so that no float32 matrix is ever whole;
then multiplied as an nn.Linear would. What a call holds beside the quantised bytes the block
already holds: the matrix in the dtype (the 7B's largest, 3072 x 12288 values, 75.5 MB in
bfloat16, 151 MB in float32), and the dequantisation's temporaries, bounded by BLOCKS whatever the
matrix: one chunk's float32 values, BLOCKS x 256 x 4 bytes = 16.8 MB (none when the dtype is
float32, which writes the output itself), and the step's integer and float32 intermediates.
Measured on the CPU by torch's memory profiler (2026-11-04, the largest matrix, bfloat16): the
temporaries peak at 43.5 MB for Q4_K, 48.1 for Q3_K and 51.9 for Q5_K beside the output, 27 to
35 MB in float32. On CUDA the planner's term is measured (peaks.GGUF_BUFFERS, RTX 5090,
2026-11-04, tests/test_kquant.py): the chunk's temporaries are freed before the output is
allocated, so beside it the call holds the bfloat16 matrix and 24,576 bytes. The bits are the
same whatever BLOCKS is (tests/test_kquant.py)."""

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from seedvr2x import weights

# The values of a block, ggml's QK_K, for every K-quant.
VALUES = 256
# The bytes of a block by ggml type (ggml-common.h at llama.cpp v0.8.0, sizeof of block_q3_K,
# block_q4_K, block_q5_K; the file's own sizes, gguf-py's GGML_QUANT_SIZES).
BLOCK_BYTES = {"Q3_K": 110, "Q4_K": 144, "Q5_K": 176}
# The blocks dequantised at a time (impl-gguf, 2026-11-04; DESIGN.md gives none): 16,384 blocks,
# 4.2 M values, whose temporaries peak at 52 MB on the CPU (the module's docstring), under the 7B's
# largest matrix in bfloat16 (75.5 MB). On the RTX 5090 the CUDA measure saw the chunk's
# temporaries freed before the output was allocated (peaks.GGUF_BUFFERS). The DiT ran 10-15%
# slower than float16 at 1080p on that card, 54-58 s against 47-50 s.
BLOCKS = 16384


def _half(raw: Tensor, at: int) -> Tensor:
    """The float16 at bytes at, at + 1 of each block of raw (n, bytes), little-endian, widened to
    float32 (exact, subnormals and zero included): (n, 1). A copy first, so that the view's
    storage offset is aligned whatever the slice's."""
    return raw[:, at : at + 2].contiguous().view(torch.float16).to(torch.float32)


def _scales_and_minimums(packed: Tensor) -> tuple[Tensor, Tensor]:
    """The 8 scales and 8 minimums of 6 bits that the 12 bytes of Q4_K's and Q5_K's blocks hold,
    packed (n, 12) int16 -> two (n, 8) int16 in [0, 63]: ggml's get_scale_min_k4, for j in
    0..7. j < 4: scale q[j] & 63, minimum q[j + 4] & 63; j >= 4: scale (q[j + 4] & 15) | ((q[j - 4]
    >> 6) << 4), minimum (q[j + 4] >> 4) | ((q[j] >> 6) << 4)."""
    low, middle, high = packed[:, 0:4], packed[:, 4:8], packed[:, 8:12]
    scales = torch.cat([low & 63, (high & 15) | ((low >> 6) << 4)], dim=1)
    minimums = torch.cat([middle & 63, (high >> 4) | ((middle >> 6) << 4)], dim=1)
    return scales, minimums


def _q3_k(raw: Tensor, out: Tensor) -> None:
    """dequantize_row_q3_K on raw (n, 110) uint8, into out (n, 256) float32. The 256 values of a
    block are 2 halves of 4 shifts of 32 values, and the 16 scales are those 16 runs of 16: the
    run of half h, shift j, values 0-15 or 16-31 has scale h * 8 + j * 2 + (0 or 1); its quants
    are the 2 bits (qs >> 2j) & 3, less 4 where bit h * 4 + j of hmask is clear."""
    n = raw.shape[0]
    high = raw[:, 0:32].to(torch.int16).reshape(n, 1, 1, 32)
    quants = raw[:, 32:96].to(torch.int16).reshape(n, 2, 1, 32)
    packed = raw[:, 96:108].to(torch.int16)
    bits = torch.arange(8, device=raw.device, dtype=torch.int16).reshape(2, 4, 1)
    shifts = 2 * torch.arange(4, device=raw.device, dtype=torch.int16).reshape(4, 1)
    # The low 4 bits of scales 0-7 are bytes 0-7's low nibbles, of 8-15 their high nibbles; the
    # high 2 bits of scales 4 j + i are bits 2 j, 2 j + 1 of byte 8 + i.
    low = torch.cat([packed[:, 0:8] & 15, packed[:, 0:8] >> 4], dim=1)
    top = ((packed[:, 8:12].reshape(n, 1, 4) >> shifts) & 3).reshape(n, 16)
    scale = (low | (top << 4)) - 32
    step = _half(raw, 108) * scale.to(torch.float32)
    value = ((quants >> shifts) & 3) + 4 * ((high >> bits) & 1) - 4
    torch.mul(
        step.reshape(n, 16, 1), value.reshape(n, 16, 16).to(torch.float32), out=out.view(n, 16, 16)
    )


def _q4_k(raw: Tensor, out: Tensor) -> None:
    """dequantize_row_q4_K on raw (n, 144) uint8, into out (n, 256) float32: 4 steps of 64 values,
    each the low nibbles of 32 bytes of qs under scale and minimum 2c, then their high nibbles
    under 2c + 1."""
    n = raw.shape[0]
    scales, minimums = _scales_and_minimums(raw[:, 4:16].to(torch.int16))
    step = _half(raw, 0) * scales.to(torch.float32)
    least = _half(raw, 2) * minimums.to(torch.float32)
    nibbles = 4 * torch.arange(2, device=raw.device, dtype=torch.int16).reshape(2, 1)
    value = (raw[:, 16:144].to(torch.int16).reshape(n, 4, 1, 32) >> nibbles) & 15
    target = out.view(n, 4, 2, 32)
    # Two roundings, never fused (the module's docstring).
    torch.mul(step.reshape(n, 4, 2, 1), value.to(torch.float32), out=target)
    target.sub_(least.reshape(n, 4, 2, 1))


def _q5_k(raw: Tensor, out: Tensor) -> None:
    """dequantize_row_q5_K on raw (n, 176) uint8, into out (n, 256) float32: Q4_K's steps, each
    quant's 5th bit the bit 2c (low nibbles) or 2c + 1 (high nibbles) of qh's byte l."""
    n = raw.shape[0]
    scales, minimums = _scales_and_minimums(raw[:, 4:16].to(torch.int16))
    step = _half(raw, 0) * scales.to(torch.float32)
    least = _half(raw, 2) * minimums.to(torch.float32)
    nibbles = 4 * torch.arange(2, device=raw.device, dtype=torch.int16).reshape(2, 1)
    bits = torch.arange(8, device=raw.device, dtype=torch.int16).reshape(4, 2, 1)
    high = (raw[:, 16:48].to(torch.int16).reshape(n, 1, 1, 32) >> bits) & 1
    value = ((raw[:, 48:176].to(torch.int16).reshape(n, 4, 1, 32) >> nibbles) & 15) + (high << 4)
    target = out.view(n, 4, 2, 32)
    torch.mul(step.reshape(n, 4, 2, 1), value.to(torch.float32), out=target)
    target.sub_(least.reshape(n, 4, 2, 1))


_BLOCKS = {"Q3_K": _q3_k, "Q4_K": _q4_k, "Q5_K": _q5_k}


@torch.no_grad()
def dequantise(
    raw: Tensor,
    qtype: str,
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """raw, whole blocks of ggml type qtype ("Q3_K", "Q4_K" or "Q5_K"; ValueError for another),
    as the file holds them, uint8 in any shape, contiguous or not: the matrix of `shape`, in
    dtype, on raw's device. Each BLOCKS blocks are dequantised in float32, equal bit for bit to
    ggml's dequantisation (the module's docstring), then rounded to dtype (to nearest, ties to
    even, as Tensor.to does) into the output, allocated once: no float32 matrix is ever whole
    unless dtype is float32. The blocks are in row order, a row's blocks one after the other.
    ValueError when raw isn't uint8, isn't a whole number of blocks, or holds other than shape's
    values."""
    if qtype not in BLOCK_BYTES:
        raise ValueError(f"{qtype!r}: not a type seedvr2x dequantises ({', '.join(BLOCK_BYTES)})")
    if raw.dtype != torch.uint8:
        raise ValueError(f"{raw.dtype}: the quantised bytes are uint8")
    size = BLOCK_BYTES[qtype]
    if raw.numel() % size:
        raise ValueError(f"{raw.numel()} bytes: not a whole number of {qtype}'s {size}-byte blocks")
    blocks = raw.numel() // size
    values = 1
    for side in shape:
        values *= side
    if blocks * VALUES != values:
        raise ValueError(
            f"{blocks} {qtype} blocks hold {blocks * VALUES} values, not {tuple(shape)}"
        )
    flat = raw.reshape(blocks, size)
    out = torch.empty((blocks, VALUES), dtype=dtype, device=raw.device)
    # One chunk's float32 values, reused, when the output isn't float32 itself.
    chunk = None
    if dtype != torch.float32:
        chunk = torch.empty((min(BLOCKS, blocks), VALUES), dtype=torch.float32, device=raw.device)
    step = _BLOCKS[qtype]
    for start in range(0, blocks, BLOCKS):
        part = flat[start : start + BLOCKS]
        target = out[start : start + BLOCKS]
        if chunk is None:
            step(part, target)
        else:
            step(part, chunk[: part.shape[0]])
            target.copy_(chunk[: part.shape[0]])
    return out.reshape(shape)


@torch.no_grad()
def linear(
    x: Tensor, weight: Tensor, bias: Tensor | None, qtype: str, dtype: torch.dtype
) -> Tensor:
    """x (..., K), any floating dtype, through the matrix weight (N, K / 256 * block bytes) uint8
    of ggml type qtype and bias (N,) or None: (..., N) in dtype, the model's. The matrix is
    dequantised straight into dtype, a chunk of float32 at a time (dequantise), so that the
    multiply holds the dtype's matrix alone; x and the bias are rounded to dtype, as autocast
    rounds a linear's arguments (GGUFLinear)."""
    matrix = dequantise(weight, qtype, (weight.shape[0], x.shape[-1]), dtype)
    return torch.nn.functional.linear(x.to(dtype), matrix, None if bias is None else bias.to(dtype))


class GGUFLinear(torch.nn.Module):
    """One of the DiT's block matrices as the dynamic GGUF holds it, in an nn.Linear's place: its
    parameters the file's own, `weight` (out, in / 256 * block bytes) uint8, the bytes of ggml's
    blocks row by row, and `bias` (out,) as the file holds it (float16), or None. Parameters, as
    the nn.Linear's were, so that the placement moves them as it moves any (seedvr2x/blockswap.py:
    only parameters move), never requiring a gradient, which an integer tensor can't. The ggml type
    is a plain attribute, `qtype`: a block on the card holds the quantised bytes (the module's
    docstring).

    Its forward runs linear in the dtype autocast gives a linear at that call, bfloat16 under the
    pipeline's (model.sample), or the input's when autocast is off, with autocast off inside. That
    is the dtype autocast gives an nn.Linear holding float16: its input, weight and bias are all
    cast to the autocast dtype, torch's lower-precision list (linear on CPU and CUDA), not the
    weight's own; here the weight is the dequantised float32 values rounded once to that dtype,
    the bias the file's float16 widened to it."""

    weight: torch.nn.Parameter
    bias: torch.nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qtype: str,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if qtype not in BLOCK_BYTES:
            raise ValueError(
                f"{qtype!r}: not a type seedvr2x dequantises ({', '.join(BLOCK_BYTES)})"
            )
        if in_features % VALUES:
            raise ValueError(f"{in_features} inputs: not a multiple of the blocks' {VALUES}")
        self.in_features, self.out_features, self.qtype = in_features, out_features, qtype

        def parameter(shape: tuple[int, ...], dtype: torch.dtype) -> torch.nn.Parameter:
            empty = torch.empty(shape, dtype=dtype, device=device)
            return torch.nn.Parameter(empty, requires_grad=False)

        row = in_features // VALUES * BLOCK_BYTES[qtype]
        self.weight = parameter((out_features, row), torch.uint8)
        if bias:
            self.bias = parameter((out_features,), torch.float16)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" bias={self.bias is not None}, ggml {self.qtype} dequantised per call"
        )

    def forward(self, x: Tensor) -> Tensor:
        kind = x.device.type
        autocast = torch.is_autocast_enabled(kind)
        dtype = torch.get_autocast_dtype(kind) if autocast else x.dtype
        with torch.autocast(kind, enabled=False):
            return linear(x, self.weight, self.bias, self.qtype, dtype)


def swap(dit: torch.nn.Module, types: Mapping[str, str]) -> list[str]:
    """Put a GGUFLinear in the place of each of the DiT's block matrices (weights.MATRIX), its
    nn.Linear's shape, bias and device (the meta device, before the load), of the ggml type that
    types gives its name (the module's name, `blocks.0.attn.proj_qkv.vid`; other names are
    ignored), each in its parent's place, so that the model's order of parameters stays as it was.
    Returns the names of the modules swapped, in the model's order: 288 for the 7B. ValueError,
    the model untouched, for a matrix whose type is missing or isn't one seedvr2x dequantises."""
    named: Any = dit.named_modules()  # pyright: ignore[reportUnknownVariableType]
    modules = cast("list[tuple[str, torch.nn.Module]]", list(named))
    linears = [
        (name, module)
        for name, module in modules
        if isinstance(module, torch.nn.Linear) and weights.MATRIX.fullmatch(name)
    ]
    for name, _ in linears:
        if name not in types:
            raise ValueError(f"{name}: no ggml type given")
        if types[name] not in BLOCK_BYTES:
            raise ValueError(
                f"{name}: {types[name]!r} is not a type seedvr2x dequantises"
                f" ({', '.join(BLOCK_BYTES)})"
            )
    for name, old in linears:
        parent, _, child = name.rpartition(".")
        bias = cast("Tensor | None", old.bias)
        new = GGUFLinear(
            old.in_features, old.out_features, types[name], bias is not None, old.weight.device
        )
        setattr(dit.get_submodule(parent), child, new)
    return [name for name, _ in linears]
