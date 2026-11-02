"""INT8 W8A8 for the DiT's block matrices, as seedvr2x's int8 files hold them (DESIGN.md, Weights,
"int8 before v0.1.0 is tagged"; models/FORMATS.md, INT8): comfy-kitchen's INT8 layout with its
rotation. Each of the 288 attention and MLP matrices of the 7B's blocks (weights.MATRIX) is
`<layer>.weight`, int8 (out, in), each 256 of its input columns multiplied by H, the regular
Hadamard matrix (hadamard), and `<layer>.weight_scale`, one float32 scale a row, (out, 1).

At run time each row of the input, a token, is rotated by the same H, so that (x H)(W H)^T = x W^T
needs no rotation back, and quantized to int8 with a scale of its own; the two int8 matrices are
multiplied with int32 accumulation (torch._int_mm, cuBLASLt's int8 multiply on CUDA), and the
product is rescaled by the token's and the row's scales, the bias added, in the model's dtype.
No comfy-kitchen runs at run time: seedvr2x's own linear adds no dependency and keeps every
buffer known to the planner (peaks.INT8_BUFFERS); whether comfy-kitchen's fused kernel is worth a
dependency is measured on the 96 GB card (DESIGN.md, Weights). The arithmetic is comfy-kitchen's
own CUDA int8_linear at v0.2.37 (Comfy-Org, Apache-2.0; NOTICE), the version and path the files
were validated with on the GPU (models/VALIDATION.md: "int8_linear (cuda)"), read in its source:
the fused quantizer (backends/cuda/ops/int8_linear.cu, quantize_int8_rowwise_convrot64_kernel)
and the dequantization (the same file's dequantize_int8_linear_kernel, and the CUTLASS epilogue
of cutlass_gemm_int8.cu, the same sums):
- the rotation in float32, from the input as the model's dtype holds it;
- a token's scale: its rotated values' largest magnitude, the dtype's largest finite value at
  most, times 1/127, in float32, 1e-30 at the least;
- each rotated value rounded to the model's dtype, divided by the scale rounded to it, the
  quotient rounded to it again, then to the nearest integer, ties to even, clamped to [-128, 127];
- the int32 product converted to float32, times the token's scale, times the row's, plus the bias
  rounded to the model's dtype, then rounded to that dtype. seedvr2x rounds each of the three
  steps in float32; the kernel's last multiply and the add may be one fused multiply-add
  (`value = ... * weight_scale; value += bias`, which nvcc contracts by default, as the CUTLASS
  epilogue may), rounded once: float32's last bit at most, before the rounding to the model's
  dtype.

comfy-kitchen's pure-Python backends (eager, Triton: the only ones its py3 wheel holds) round
otherwise: the rotation in the model's dtype, and the scale from the rotated values so rounded,
divided by 127 rather than multiplied by its inverse (backends/eager/quantization.py at v0.2.37:
int8_linear, quantize_int8_rowwise). The files were validated with the CUDA kernel's arithmetic,
which this follows.

The rotation here is a float32 product with H; the kernel's is a fast Hadamard transform in four
float32 stages of four terms, each stage's sums rounded. H's entries are 1/16 or -1/16, so every
term is exact, but the two sum them in another order. Measured on synthetic activations shaped
as the DiT's (2026-11-02: 256 tokens a matrix, each channel's magnitude drawn between 0.2 and
8.2 and 8 channels multiplied by 30, GELU's output into the MLP's output projection; against an
emulation of the kernel's transform), the rotated values are equal for 97% of them into the
K = 3,072 matrices and 55% into the MLP's output projection (K = 12,288), the tokens' scales for
93-98% and 67%, the codes for 100% and 99.9999%, and the outputs bit for bit for 99.9998-100%
and 99.81%, 1e-4 apart relative where they differ: nothing against W8A8's own error, about
0.6% (models/VALIDATION.md, Results: the int8 kernel's median per layer, 0.59%). The 96 GB card
compares the model's own activations with the kernel itself. The product stays: a transform
written in torch's operations is four passes over the values, each its own kernel and buffer,
where the product is one.

A linear takes ROWS of its input's rows at a time, the output allocated once: what the W8A8 path
holds beside the output, the input in float32 and rotated, the codes, the int32 product and its
float32 rescaling, is bounded by ROWS rows whatever the window, and a transposed copy of the
weight beside them (peaks.INT8_BUFFERS counts it all). On the CPU, every step works on a row alone
and the integer product is exact, so the rows taken at a time change no bit (tests/test_int8.py).
On CUDA, cuBLAS may choose another kernel for a last chunk of fewer rows, the rotation's float32
sums then in another order: still the same bits for the same window each time it runs, cuBLAS's
results being the same run to run on the same card and toolkit (its documentation, "Results
reproducibility"), so that a resume stays bit for bit."""

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from seedvr2x import weights

# The columns each rotation takes, comfy-kitchen's convrot_groupsize in the files' markers
# (weights.MARKER): 256, 4^4.
GROUP = weights.GROUP
# The rows of a linear's input taken at a time. PROVISIONAL (impl-int8, 2026-11-02; DESIGN.md
# gives none): 4,096 tokens, about half a latent of 1080p (8,280 tokens), so that the 7B's
# matrices take at most 0.375 GiB beside their output (peaks.W8A8_ROWS, a literal there since
# this module imports torch; tests/test_int8.py holds the two equal). The multiply's speed at that
# size is unmeasured: the 96 GB card's first int8 runs measure it.
ROWS = 4096
# torch._int_mm on CUDA takes a first matrix of more than 16 rows (aten's _int_mm_out_cuda,
# native/cuda/Blas.cpp at v2.14.1): seedvr2x follows torch's own rule, a first matrix of 16 rows
# or fewer multiplied as this many, the rows added zeros, any other as it is. comfy-kitchen's
# eager backend rounds every first matrix on CUDA up to a multiple of 32 rows, 32 at least
# (backends/eager/quantization.py at v0.2.37, _int8_matmul_accumulate): the same products.
LEAST_ROWS = 32
# torch._int_mm on CUDA takes a second matrix of a multiple of 8 columns (the same check): a
# linear's outputs, which Int8Linear holds to it.
OUTPUTS_MULTIPLE = 8

# H for each device it was asked on, float32 (hadamard).
_HADAMARD: dict[torch.device, Tensor] = {}
# torch's int8 multiply, int32 out: an aten operator torch documents nowhere, typed here.
_int_mm: Callable[[Tensor, Tensor], Tensor] = torch._int_mm  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]


def hadamard(device: torch.device | str = "cpu") -> Tensor:
    """H, the regular Hadamard matrix of GROUP rows, normalised: H4 (x) H4 (x) H4 (x) H4 / 16, H4
    the regular 4x4 one, (GROUP, GROUP) float32 on device, every entry 1/16 or -1/16. Symmetric
    and its own inverse. comfy-kitchen's (comfy_kitchen/tensor/int8_utils.py, _build_hadamard), as
    the files' script rebuilds it (models/seedvr2_int8.py, hadamard): made in float64 and cast,
    exact."""
    device = torch.device(device)
    if device not in _HADAMARD:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64
        )
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < GROUP:
            h = torch.kron(h, h4)
        _HADAMARD[device] = (h / GROUP**0.5).to(device, torch.float32)
    return _HADAMARD[device]


def rotate(values: Tensor) -> Tensor:
    """values (rows, K) float32, each GROUP columns times H (hadamard): (rows, K) float32, a new
    tensor. ValueError when K isn't a multiple of GROUP."""
    rows, features = values.shape
    if features % GROUP:
        raise ValueError(f"{features} columns: not a multiple of the rotation's {GROUP}")
    grouped = values.reshape(rows, features // GROUP, GROUP)
    return (grouped @ hadamard(values.device)).reshape(rows, features)


def quantize(rotated: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """The int8 codes of rotated (rows, K) float32, the rotated input, and each row's scale, as
    comfy-kitchen's quantizer gives them (the module's docstring): (rows, K) int8 in [-128, 127],
    and (rows, 1) float32, the scale its product is rescaled by. dtype is the model's, the input
    rounded to it. rotated is overwritten when dtype is float32.

    In float16, which the pipeline never computes in (bfloat16 under its autocast, model.sample),
    a row of zeros has its scale, 1e-30, rounded to 0: its codes are 0 / 0, undefined, but not its
    output: their product times the float32 scale, 1e-30, is far under float16's smallest value,
    and the row's output is its bias, as with codes of zero."""
    low, high = torch.aminmax(rotated, dim=1, keepdim=True)
    largest = torch.maximum(high, low.neg())
    # 1 / 127 in float32, the scalar taken in the tensor's float32 as C's (1.0f / 127.0f) is.
    scale = largest.clamp_(max=torch.finfo(dtype).max).mul_(1 / 127).clamp_(min=1e-30)
    codes = rotated.to(dtype)
    codes.div_(scale.to(dtype)).round_().clamp_(-128, 127)
    return codes.to(torch.int8), scale


def transposed(weight: Tensor) -> Tensor:
    """weight (N, K) int8, as the file and the model hold it, transposed into a contiguous copy,
    (K, N) int8: product's second matrix, made once a linear's call. torch._int_mm is given it
    contiguous, as comfy-kitchen's eager int8 linear gives it (weight.T.contiguous()) and torchao's
    safe_int_mm makes it, whose comment has cuBLAS give a wrong result silently for the transposed
    view; torch's own _int_mm_out_cuda says as much ("cuBLAS is currently broken for some
    combination of transposed inputs", native/cuda/Blas.cpp at v2.14.1). The copy stays until the
    96 GB card has run the view against it. Up to 36 MiB for the 7B's MLP matrices, counted in
    peaks.INT8_BUFFERS."""
    return weight.t().contiguous()


def product(codes: Tensor, weight_t: Tensor) -> Tensor:
    """codes (rows, K) int8 times weight_t (K, N) int8, contiguous (transposed): (rows, N) int32,
    every sum exact (at most 12288 x 128 x 128, under 2^31, for the 7B). torch._int_mm: cuBLASLt's
    int8 multiply on CUDA; on the CPU, which only the tests use, torch's own kernel, the same
    integer product, exact in int32 (tests/test_int8.py holds it to an int64 one)."""
    rows = codes.shape[0]
    if codes.is_cuda and rows <= 16:
        padded = codes.new_zeros((LEAST_ROWS, codes.shape[1]))
        padded[:rows] = codes
        return _int_mm(padded, weight_t)[:rows]
    return _int_mm(codes, weight_t)


@torch.no_grad()
def linear(
    x: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None, dtype: torch.dtype
) -> Tensor:
    """x (..., K), any floating dtype, through the W8A8 linear of weight (N, K) int8, scale (N, 1)
    float32, the row's, and bias (N,) or None: (..., N) in dtype, the model's (the module's
    docstring). x is rounded to dtype first, as autocast rounds a linear's input, and taken ROWS
    rows at a time. Inference only, without gradients: its last operation writes into the
    output's rows (out=), which autograd refuses, and the codes have no gradient anyway."""
    features = x.shape[-1]
    rows = x.reshape(-1, features)
    out = torch.empty((rows.shape[0], weight.shape[0]), dtype=dtype, device=x.device)
    row_scale = scale.reshape(1, -1).to(torch.float32)
    added = None if bias is None else bias.to(dtype).to(torch.float32).reshape(1, -1)
    weight_t = transposed(weight)
    for start in range(0, rows.shape[0], ROWS):
        part = rows[start : start + ROWS]
        codes, token_scale = quantize(rotate(part.to(dtype).to(torch.float32)), dtype)
        accumulated = product(codes, weight_t)
        del codes
        # The int32 product times the token's scale in one pass, converted to float32 on the way
        # in, as torch's type promotion takes it: the bits of a conversion, then a multiply.
        result = torch.mul(accumulated, token_scale)
        del accumulated
        # The last operation written into the output's rows, rounded to dtype there: the bits of
        # the operation in float32, then a copy.
        target = out[start : start + part.shape[0]]
        if added is None:
            torch.mul(result, row_scale, out=target)
        else:
            torch.add(result.mul_(row_scale), added, out=target)
        # Gone before the next rows' buffers are made, which it would otherwise sit beside.
        del result
    return out.reshape(*x.shape[:-1], weight.shape[0])


class Int8Linear(torch.nn.Module):
    """One of the DiT's block matrices in comfy-kitchen's INT8 layout, in an nn.Linear's place:
    its parameters the file's own, `weight` (out, in) int8 and `weight_scale` (out, 1) float32,
    and `bias` (out,) as the file holds it (float16), or None. Parameters, as the nn.Linear's
    were, so that the placement moves them as it moves any (seedvr2x/blockswap.py: only
    parameters move), never requiring a gradient, which an integer tensor can't.

    Its forward runs linear in the dtype autocast gives a linear at that call, bfloat16 under the
    pipeline's (model.sample), or the input's when autocast is off, with autocast off inside."""

    weight: torch.nn.Parameter
    weight_scale: torch.nn.Parameter
    bias: torch.nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % GROUP:
            raise ValueError(f"{in_features} inputs: not a multiple of the rotation's {GROUP}")
        if out_features % OUTPUTS_MULTIPLE:
            raise ValueError(
                f"{out_features} outputs: not a multiple of {OUTPUTS_MULTIPLE}, which torch's int8"
                " multiply takes on CUDA"
            )
        self.in_features, self.out_features = in_features, out_features

        def parameter(shape: tuple[int, ...], dtype: torch.dtype) -> torch.nn.Parameter:
            empty = torch.empty(shape, dtype=dtype, device=device)
            return torch.nn.Parameter(empty, requires_grad=False)

        self.weight = parameter((out_features, in_features), torch.int8)
        self.weight_scale = parameter((out_features, 1), torch.float32)
        if bias:
            self.bias = parameter((out_features,), torch.float16)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" bias={self.bias is not None}, int8 W8A8 rotated per {GROUP} columns"
        )

    def forward(self, x: Tensor) -> Tensor:
        kind = x.device.type
        autocast = torch.is_autocast_enabled(kind)
        dtype = torch.get_autocast_dtype(kind) if autocast else x.dtype
        with torch.autocast(kind, enabled=False):
            return linear(x, self.weight, self.weight_scale, self.bias, dtype)


def swap(dit: torch.nn.Module) -> list[str]:
    """Put an Int8Linear in the place of each of the DiT's block matrices (weights.MATRIX), its
    nn.Linear's shape, bias and device (the meta device, before the load), each in its parent's
    place, so that the model's order of parameters stays as it was. Returns the names of the
    modules swapped, in the model's order: 288 for the 7B."""
    named: Any = dit.named_modules()  # pyright: ignore[reportUnknownVariableType]
    modules = cast("list[tuple[str, torch.nn.Module]]", list(named))
    linears = [
        (name, module)
        for name, module in modules
        if isinstance(module, torch.nn.Linear) and weights.MATRIX.fullmatch(name)
    ]
    for name, old in linears:
        parent, _, child = name.rpartition(".")
        bias = cast("Tensor | None", old.bias)
        new = Int8Linear(old.in_features, old.out_features, bias is not None, old.weight.device)
        setattr(dit.get_submodule(parent), child, new)
    return [name for name, _ in linears]
