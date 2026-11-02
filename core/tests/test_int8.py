"""The W8A8 linear of seedvr2x's int8 files (seedvr2x/int8.py) on the CPU, against the same
arithmetic written here apart: the rotation in float64, the quantization in NumPy's float32, its
roundings to bfloat16 by their bits, the integer product in int64, the rescaling in float64.
Equal where the arithmetic is integer or exact (the rotation of values whose sums float32 holds,
the codes, the scales, the product), within bfloat16's rounding after (the output). Then its rows
taken at a time, which change no bit on the CPU; its module under autocast, as the DiT runs it;
the 7B's 288 block matrices swapped on the meta device; and the figures the memory model takes
from it (seedvr2x/peaks.py), its buffers measured by torch's memory profiler.

On the GPU (pytest -m gpu, skipped without CUDA), the path on CUDA, cuBLASLt's int8 multiply
among it, against the CPU's, for the 7B's four shapes and first matrices of 1 to 4,101 rows: the
integer product exact, the codes and the outputs the CPU's but where the rotation's float32 sums
round otherwise, and a call's memory within the output and peaks.INT8_BUFFERS. And opt-in, where
comfy-kitchen 0.2.37 is installed (it is no dependency: `uv pip install --no-deps
comfy-kitchen==0.2.37`, its cp312-abi3 wheel, which Python 3.13 takes), the same activations
through its own CUDA int8_linear, the arithmetic int8.py follows, codes and outputs compared:
skipped without it. tests/test_blockswap.py holds the real int8 file's blocks swapped to the
bits of the DiT loaded whole."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor

from seedvr2x import int8, peaks, weights

GROUP = 256


def h64() -> npt.NDArray[np.float64]:
    """The regular Hadamard matrix of 256 rows, normalised, in float64, made here: the 4x4 one
    of comfy-kitchen's int8_utils.py, H4 (x) H4 (x) H4 (x) H4 / 16."""
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    h = h4
    while h.shape[0] < GROUP:
        h = np.kron(h, h4)
    return h / 16


def to_bf16(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """float32 values rounded to bfloat16's 8 significant bits, to nearest, ties to even, by
    their bits (finite values)."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def reference(
    x: Tensor, weight: Tensor, scale: Tensor, bias: Tensor | None, dtype: torch.dtype
) -> dict[str, Any]:
    """Each step of the W8A8 linear, written apart from the module: the rotated input (float32,
    from the float64 rotation), the codes and the scales, the int64 product, and the output in
    float64."""
    rows = x.to(dtype).to(torch.float32).numpy().astype(np.float64).reshape(-1, x.shape[-1])
    rotated = (rows.reshape(rows.shape[0], -1, GROUP) @ h64()).reshape(rows.shape)
    rotated = rotated.astype(np.float32)
    largest = np.abs(rotated).max(axis=1, keepdims=True)
    limit = np.float32(torch.finfo(dtype).max)
    inverse = np.float32(1) / np.float32(127)
    scales = np.maximum(np.minimum(largest, limit) * inverse, np.float32(1e-30))
    rounding = to_bf16 if dtype == torch.bfloat16 else (lambda v: v.astype(np.float32))
    quotient = rounding(rounding(rotated) / rounding(scales))
    codes = np.clip(np.rint(quotient), -128, 127).astype(np.int8)
    product = codes.astype(np.int64) @ weight.numpy().astype(np.int64).T
    out = product.astype(np.float64) * scales.astype(np.float64)
    out = out * scale.numpy().astype(np.float64).reshape(1, -1)
    if bias is not None:
        out = out + bias.to(dtype).to(torch.float32).numpy().astype(np.float64)
    return {"rotated": rotated, "codes": codes, "scales": scales, "product": product, "out": out}


def layer(rows: int, inputs: int, outputs: int, seed: int, grid: bool) -> tuple[Tensor, ...]:
    """An input (rows, inputs) and an int8 layer of that shape, by a seed: weights over the
    whole int8 range, a scale a row, a float16 bias. With grid, the input's values are on a
    grid of 1/64 within 4, whose rotated sums float32 holds exactly; else normal ones."""
    generator = torch.Generator().manual_seed(seed)
    if grid:
        x = torch.randint(-256, 257, (rows, inputs), generator=generator).float() / 64
    else:
        x = torch.randn(rows, inputs, generator=generator) * 3
    weight = torch.randint(-128, 128, (outputs, inputs), generator=generator, dtype=torch.int8)
    scale = torch.rand(outputs, 1, generator=generator) * 0.01 + 1e-4
    bias = (torch.randn(outputs, generator=generator) * 0.5).to(torch.float16)
    return x, weight, scale, bias


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("grid", [True, False], ids=["grid", "normal"])
def test_the_w8a8_linear_is_its_arithmetic(dtype: torch.dtype, grid: bool) -> None:
    x, weight, scale, bias = layer(37, 768, 96, seed=1, grid=grid)
    expected = reference(x, weight, scale, bias, dtype)
    rows = x.to(dtype).to(torch.float32)
    rotated = int8.rotate(rows)
    if grid:
        # Every sum exact: the rotation bit for bit.
        assert np.array_equal(rotated.numpy(), expected["rotated"])
    else:
        # Sums of values that span float32's 24 bits round, here or in float64: a few of
        # float32's ulps of the row's values; the steps after taken from the reference's.
        reach = float(np.abs(expected["rotated"]).max())
        np.testing.assert_allclose(rotated.numpy(), expected["rotated"], rtol=0, atol=reach * 2e-6)
        rotated = torch.from_numpy(expected["rotated"].copy())
    codes, scales = int8.quantize(rotated.clone(), dtype)
    assert codes.dtype == torch.int8 and scales.dtype == torch.float32
    assert np.array_equal(codes.numpy(), expected["codes"])
    assert np.array_equal(scales.numpy(), expected["scales"])
    product = int8.product(codes, int8.transposed(weight))
    assert product.dtype == torch.int32
    assert np.array_equal(product.numpy().astype(np.int64), expected["product"])
    out = int8.linear(x, weight, scale, bias, dtype)
    assert out.dtype == dtype and out.shape == (37, 96)
    # Rescaled in float32, rounded once to dtype: within its rounding of the float64 output, and
    # of float32's of the terms before the bias, which can cancel.
    ulp = 2.0**-8 if dtype == torch.bfloat16 else 2.0**-22
    reach = float(np.abs(expected["out"]).max())
    np.testing.assert_allclose(
        out.to(torch.float64).numpy(), expected["out"], rtol=ulp, atol=reach * 2.0**-20
    )
    if grid and dtype == torch.bfloat16:
        # Its output the reference's rounded to bfloat16 but for a value a float32 step tips
        # over a rounding boundary.
        rounded = torch.from_numpy(expected["out"]).to(torch.float32).to(dtype)
        assert (out != rounded).float().mean() < 0.01


def test_the_integer_product_is_exact() -> None:
    # torch's int8 multiply on the CPU, int32 out, against int64: the extremes of the 7B's widest
    # input (12,288 x 128 x 128 = 201,326,592, under 2^31) and random codes.
    lowest = torch.full((20, 12288), -128, dtype=torch.int8)
    weight = torch.full((24, 12288), -128, dtype=torch.int8)
    assert int(int8.product(lowest, int8.transposed(weight)).max()) == 12288 * 128 * 128
    generator = torch.Generator().manual_seed(2)
    codes = torch.randint(-128, 128, (19, 12288), generator=generator, dtype=torch.int8)
    weight = torch.randint(-128, 128, (40, 12288), generator=generator, dtype=torch.int8)
    expected = codes.long() @ weight.long().T
    assert torch.equal(int8.product(codes, int8.transposed(weight)).long(), expected)
    # A single row, taken as it is on the CPU, nothing padded: CUDA's multiply takes it padded to
    # int8.LEAST_ROWS rows (the GPU test below).
    assert torch.equal(int8.product(codes[:1], int8.transposed(weight)).long(), expected[:1])


def test_rows_taken_at_a_time_change_no_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every step works on a row alone: 70 rows at once, or 8 at a time and a last of 6, or one
    # at a time, give the same bits; a 3-D input as its rows.
    x, weight, scale, bias = layer(70, 512, 64, seed=3, grid=False)
    whole = int8.linear(x, weight, scale, bias, torch.bfloat16)
    for rows in (8, 1):
        monkeypatch.setattr(int8, "ROWS", rows)
        assert torch.equal(int8.linear(x, weight, scale, bias, torch.bfloat16), whole)
    batched = int8.linear(x.reshape(2, 35, 512), weight, scale, bias, torch.bfloat16)
    assert batched.shape == (2, 35, 64) and torch.equal(batched.reshape(70, 64), whole)
    # No bias.
    plain = int8.linear(x, weight, scale, None, torch.bfloat16)
    expected = reference(x, weight, scale, None, torch.bfloat16)["out"]
    np.testing.assert_allclose(plain.double().numpy(), expected, rtol=2.0**-8, atol=1e-6)


def test_the_module_runs_in_autocast_s_dtype() -> None:
    # As the DiT runs it, under the pipeline's bfloat16 autocast, an input in float32: the
    # output in bfloat16, the same bits as linear in bfloat16; with autocast off, the input's
    # dtype. Its parameters never require a gradient; its weight's bytes the file's.
    x, weight, scale, bias = layer(21, 512, 48, seed=4, grid=False)
    module = int8.Int8Linear(512, 48)
    assert [name for name, _ in module.named_parameters()] == ["weight", "weight_scale", "bias"]
    assert not any(parameter.requires_grad for parameter in module.parameters())
    module.load_state_dict({"weight": weight, "weight_scale": scale, "bias": bias}, assign=True)
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        out = module(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, int8.linear(x, weight, scale, bias, torch.bfloat16))
    with torch.no_grad():
        assert module(x).dtype == torch.float32
    unbiased = int8.Int8Linear(512, 48, bias=False)
    assert unbiased.bias is None and "bias=False" in repr(unbiased)
    with pytest.raises(ValueError, match="not a multiple of the rotation's 256"):
        int8.Int8Linear(300, 48)
    # torch's int8 multiply on CUDA takes a multiple of 8 outputs: another is refused when the
    # module is built, not at its first call.
    with pytest.raises(ValueError, match=r"^50 outputs: not a multiple of 8, which torch's int8"):
        int8.Int8Linear(512, 50)


def test_h_is_comfy_kitchen_s_rotation() -> None:
    # H made here in float64, its entries 1/16 or -1/16, symmetric and its own inverse; rotating
    # by it twice gives the values back.
    h = int8.hadamard("cpu")
    assert h.dtype == torch.float32 and h.shape == (GROUP, GROUP)
    assert np.array_equal(h.double().numpy(), h64())
    assert torch.equal(h, h.T) and torch.equal(h.abs(), torch.full_like(h, 1 / 16))
    assert torch.equal(h.double() @ h.double(), torch.eye(GROUP, dtype=torch.float64))
    assert int8.hadamard(torch.device("cpu")) is h
    values = torch.randint(-64, 65, (3, 512)).float()
    assert torch.equal(int8.rotate(int8.rotate(values)), values)
    with pytest.raises(ValueError, match="not a multiple of the rotation's 256"):
        int8.rotate(torch.zeros(2, 300))


def test_the_7b_s_block_matrices_are_swapped() -> None:
    # Built on the meta device: the 288 nn.Linear of the blocks' attention and MLP (weights.MATRIX)
    # each an Int8Linear of its shape and bias, in its place, nothing else touched; their shapes,
    # (inputs, outputs), the memory model's (peaks.W8A8_MATRICES).
    from seedvr2x.model import VENDOR
    from seedvr2x.vendor.common.config import create_object, load_config

    config: Any = load_config(str(VENDOR / "configs_7b" / "main.yaml"))
    with torch.device("meta"):
        dit = create_object(config.dit.model)
    before = {name: type(module) for name, module in dit.named_modules()}
    linears = {
        name: (module.in_features, module.out_features, module.bias is not None)
        for name, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear) and weights.MATRIX.fullmatch(name)
    }
    swapped = int8.swap(dit)
    assert swapped == list(linears) and len(swapped) == 288
    after = dict(dit.named_modules())
    for name, kind in before.items():
        module = after[name]
        if name in linears:
            assert isinstance(module, int8.Int8Linear)
            assert (module.in_features, module.out_features, module.bias is not None) == (
                linears[name]
            )
            assert module.weight.device.type == "meta" and module.weight.dtype == torch.int8
        else:
            assert type(module) is kind, name
    shapes = {(inputs, outputs) for inputs, outputs, _ in linears.values()}
    assert shapes == set(peaks.W8A8_MATRICES)


def test_the_memory_model_s_figures_are_the_module_s() -> None:
    # The rows taken at a time and the rotation's group, literals in torch-free modules.
    assert peaks.W8A8_ROWS == int8.ROWS == 4096
    assert int8.GROUP == weights.GROUP == GROUP
    assert peaks.W8A8_ROTATION == int8.hadamard("cpu").nbytes
    # 96 KiB a row at the most, the MLP's two matrices: 0.375 GiB at 4,096 rows; the transposed
    # weight of the largest, 36 MiB, and H.
    assert max(peaks.w8a8_row_bytes(*shape) for shape in peaks.W8A8_MATRICES) == 96 * 1024
    weight = torch.zeros(3072, 12288, dtype=torch.int8)
    assert (
        int8.transposed(weight).nbytes == 36 * 1024**2 and int8.transposed(weight).is_contiguous()
    )
    assert peaks.INT8_BUFFERS == 4096 * 96 * 1024 + 36 * 1024**2 + 1024**2 + 256 * 1024


def allocated(run: Callable[[], object]) -> int:
    """The most bytes torch allocated in system memory while run() ran, over what was allocated
    before it, by torch's memory profiler: every allocation and release of its CPU allocator, in
    order (the CLI's test_shot.py measures so)."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, profile_memory=True) as profiled:
        result = run()
    assert profiled.profiler is not None and profiled.profiler.kineto_results is not None
    # torch's stubs don't give a memory event's bytes.
    recorded: list[Any] = list(profiled.profiler.kineto_results.events())
    events = [event for event in recorded if event.name() == "[memory]"]
    most = left = 0
    for event in sorted(events, key=lambda event: event.start_ns()):
        left += event.nbytes()
        most = max(most, left)
    del result
    return most


def cpu_row_bytes(inputs: int, outputs: int) -> int:
    """What the W8A8 path holds beside a linear's output for a row of its input on the CPU, in
    bytes: CUDA's (peaks.w8a8_row_bytes), or the int32 product, torch's float32 copy of it for the
    multiply by the token's scale and that multiply's result, 12 bytes an output, where it is the
    more: torch's temporaries for the CPU alone (peaks.INT8_BUFFERS' comment)."""
    return max(peaks.w8a8_row_bytes(inputs, outputs), 12 * outputs)


@pytest.mark.parametrize(("inputs", "outputs"), [(1024, 256), (256, 1024), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("biased", [True, False], ids=["bias", "no-bias"])
def test_the_buffers_are_the_memory_model_s(
    monkeypatch: pytest.MonkeyPatch, inputs: int, outputs: int, dtype: torch.dtype, biased: bool
) -> None:
    # What the W8A8 path holds beside its output, measured on the CPU, its input of 200 rows
    # taken 64 at a time: the rows' buffers at their most (cpu_row_bytes, peaks.w8a8_row_bytes'
    # three moments with the CPU's temporaries), the transposed weight, and the small ones within
    # W8A8_SMALL; what peaks.INT8_BUFFERS counts for the 7B's shapes at 4,096 rows on CUDA.
    monkeypatch.setattr(int8, "ROWS", 64)
    x, weight, scale, bias = layer(200, inputs, outputs, seed=5, grid=False)
    x = x.to(dtype)
    int8.hadamard("cpu")
    added = bias if biased else None
    most = allocated(lambda: int8.linear(x, weight, scale, added, torch.bfloat16))
    output = 200 * outputs * 2
    counted = 64 * cpu_row_bytes(inputs, outputs) + inputs * outputs
    assert counted <= most - output <= counted + peaks.W8A8_SMALL


# The first matrices' rows the GPU tests take: CUDA's multiply padded (16 rows and fewer, int8.
# LEAST_ROWS), just over its 16, a few more, a whole ROWS and a last chunk of 5 after one.
GPU_ROWS = (1, 16, 17, 33, 4096, 4101)


def cuda() -> torch.device:
    """The first CUDA device; the test is skipped without one."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda:0")


def activations(rows: int, inputs: int, outputs: int, seed: int) -> tuple[Tensor, ...]:
    """A layer of the 7B's shape (layer) and its input in bfloat16, as the pipeline's autocast
    gives a linear its input, a few channels 50 times the others' range, as a DiT's activations
    have."""
    x, weight, scale, bias = layer(rows, inputs, outputs, seed=seed, grid=False)
    x[:, :4] *= 50
    return x.to(torch.bfloat16), weight, scale, bias


def bf16_ulp(values: Tensor) -> Tensor:
    """bfloat16's spacing at each value's magnitude, float64: 2^(e - 7) for a magnitude in
    [2^e, 2^(e + 1)), the smallest normal's spacing below it."""
    magnitude = values.double().abs().clamp(min=torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


def codes_agree(codes: Tensor, other: Tensor) -> None:
    """Fails unless two sides' codes, int8 of one shape, are the same for 99.999% of them, two
    allowed apart at the least: a first matrix of a few rows holds too few codes for a share,
    and one code apart is the rare rounding the module's docstring measures (the codes 99.9999%
    the kernel's into the MLP's output projection)."""
    apart = int((codes.cpu() != other.cpu()).sum())
    assert apart <= max(2, codes.numel() // 100_000), (apart, codes.numel())


def made(run: Callable[[], Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """run's output, and the codes and the scales int8.linear quantized for it, its chunks of
    ROWS concatenated on the CPU: what its output was made from, which the codes of the input
    rotated whole may not be on CUDA, where cuBLAS may take another kernel for a last chunk of
    fewer rows (the module's docstring)."""
    quantized: list[tuple[Tensor, Tensor]] = []
    quantize = int8.quantize

    def recorded(rotated: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        codes, scales = quantize(rotated, dtype)
        quantized.append((codes.cpu(), scales.cpu()))
        return codes, scales

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(int8, "quantize", recorded)
        out = run()
    codes = torch.cat([codes for codes, _ in quantized])
    scales = torch.cat([scales for _, scales in quantized])
    return out, (codes, scales)


def explained(
    out: Tensor,
    other: Tensor,
    sides: list[tuple[Tensor, Tensor]],
    weight: Tensor,
    scale: Tensor,
    same_bits: bool = True,
) -> None:
    """Fails unless two outputs of one linear in bfloat16, (rows, N) on the CPU, are apart by no
    more than the codes and the scales each was made from, sides' (codes, scales) on the CPU,
    make them apart: their float64 rescalings' difference, products exact in float64 (integers
    under 2^53), within one bfloat16 spacing of the larger and float32's roundings of the terms.
    And with same_bits, bit for bit on every row whose codes and scale are the same on both
    sides: seedvr2x's float32 rescaling is the same arithmetic on either device. Not against
    comfy-kitchen's kernel, whose epilogue may fuse the last multiply and the add (int8.py's
    docstring): one bfloat16 spacing there too."""
    rows = scale.reshape(1, -1).double()
    rescaled = [
        (codes.double() @ weight.double().T) * scales.double() * rows for codes, scales in sides
    ]
    apart = (out.double() - other.double()).abs()
    larger = torch.maximum(out.double().abs(), other.double().abs())
    terms = torch.maximum(rescaled[0].abs(), rescaled[1].abs()) + larger
    allowed = (rescaled[0] - rescaled[1]).abs() + bf16_ulp(larger) + terms * 2.0**-20
    assert bool((apart <= allowed).all()), float((apart - allowed).max())
    if same_bits:
        (codes, scales), (other_codes, other_scales) = sides
        same = (codes == other_codes).all(dim=1) & (scales == other_scales).reshape(-1)
        assert torch.equal(out[same], other[same])


@pytest.mark.gpu
@pytest.mark.parametrize("rows", GPU_ROWS)
@pytest.mark.parametrize(("inputs", "outputs"), peaks.W8A8_MATRICES)
def test_the_w8a8_linear_on_cuda_is_the_cpu_s(rows: int, inputs: int, outputs: int) -> None:
    # The path on CUDA against the CPU's, the 7B's four shapes: cuBLASLt's int32 product exact,
    # the integer product computed apart in float64; the codes the CPU's for 99.999% of them at
    # least, the rotation's float32 sums rounding otherwise there (int8.py's docstring); the
    # outputs the CPU's but for what the codes and the scales that differ make, within one
    # bfloat16 spacing (explained). And a call's memory: what torch's allocator holds at its
    # most beside the input, the weight and what the warm-up left (H, cuBLAS's workspace), the
    # output and peaks.INT8_BUFFERS at the most, printed for the card's record (the PROVISIONAL
    # figure measured on CUDA).
    device = cuda()
    x, weight, scale, bias = activations(rows, inputs, outputs, seed=7)
    on = (x.to(device), weight.to(device), scale.to(device), bias.to(device))
    codes, _ = int8.quantize(int8.rotate(on[0].float()), torch.bfloat16)
    product = int8.product(codes, int8.transposed(on[1]))
    exact = codes.cpu().double() @ weight.double().T
    assert product.dtype == torch.int32 and torch.equal(product.cpu().double(), exact)
    cpu_codes, _ = int8.quantize(int8.rotate(x.float()), torch.bfloat16)
    codes_agree(codes, cpu_codes)
    int8.linear(*on, torch.bfloat16)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    out = int8.linear(*on, torch.bfloat16)
    torch.cuda.synchronize(device)
    held = torch.cuda.max_memory_allocated(device) - before - out.nbytes
    print(f"{rows} x {inputs} -> {outputs}: {held / 2**30:.4f} GiB beside the output")
    assert held <= peaks.INT8_BUFFERS, (held, peaks.INT8_BUFFERS)
    # Each side's output beside the codes and the scales its own chunks were made from.
    on_cuda, cuda_side = made(lambda: int8.linear(*on, torch.bfloat16))
    assert torch.equal(on_cuda, out)
    expected, cpu_side = made(lambda: int8.linear(x, weight, scale, bias, torch.bfloat16))
    codes_agree(cuda_side[0], cpu_side[0])
    explained(out.cpu(), expected, [cuda_side, cpu_side], weight, scale)


# comfy-kitchen's version whose CUDA int8_linear the files were validated with and int8.py
# follows (models/VALIDATION.md; NOTICE).
KITCHEN = "0.2.37"


def kitchen() -> Any:
    """comfy-kitchen's CUDA backend, its compiled extension loaded; the test is skipped without
    comfy-kitchen at KITCHEN, which seedvr2x doesn't depend on, or without its extension."""
    from importlib import metadata

    try:
        version = metadata.version("comfy-kitchen")
    except metadata.PackageNotFoundError:
        pytest.skip(f"needs comfy-kitchen {KITCHEN}, opt-in: not installed")
    if version != KITCHEN:
        pytest.skip(f"needs comfy-kitchen {KITCHEN}, opt-in: {version} installed")
    from comfy_kitchen.backends import cuda as backend  # pyright: ignore[reportMissingImports]

    if not getattr(backend, "_EXT_AVAILABLE", False):
        pytest.skip(f"comfy-kitchen {KITCHEN}'s CUDA extension isn't loaded")
    return backend


@pytest.mark.gpu
@pytest.mark.parametrize("rows", (2, *GPU_ROWS[1:]))
@pytest.mark.parametrize(("inputs", "outputs"), peaks.W8A8_MATRICES)
def test_the_w8a8_linear_is_comfy_kitchen_s_cuda_int8_linear(
    rows: int, inputs: int, outputs: int
) -> None:
    # The arithmetic int8.py follows, held to comfy-kitchen's own: the same activations through
    # its CUDA int8_linear with its rotation (its fused quantizer, quantize_int8_rowwise_convrot64,
    # then its two-row kernel or the CUTLASS GEMM with its epilogue: cuda/__init__.py at v0.2.37,
    # int8_linear) and through seedvr2x's on CUDA. The codes, its quantizer's, the same for
    # 99.999% of them at least; the outputs apart by no more than the codes and the scales that
    # differ make (explained: the kernel's transform rounds its sums otherwise, its epilogue may
    # fuse a multiply and the add; int8.py's docstring). Not one row: int8_linear takes it to a
    # kernel of its own, int8_linear_m1, quantizing inside it, whose codes its quantizer's call
    # here doesn't give.
    device = cuda()
    backend = kitchen()
    x, weight, scale, bias = activations(rows, inputs, outputs, seed=8)
    on = (x.to(device), weight.to(device), scale.to(device), bias.to(device))
    theirs_codes, theirs_scales = backend.quantize_int8_rowwise_convrot64(on[0], int8.GROUP)
    theirs = backend.int8_linear(
        on[0],
        on[1],
        on[2].reshape(-1),
        on[3],
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=int8.GROUP,
    )
    ours, our_side = made(lambda: int8.linear(*on, torch.bfloat16))
    assert theirs.dtype == ours.dtype == torch.bfloat16 and theirs.shape == ours.shape
    codes_agree(theirs_codes, our_side[0])
    sides = [(theirs_codes.cpu(), theirs_scales.cpu()), our_side]
    explained(theirs.cpu(), ours.cpu(), sides, weight, scale, same_bits=False)
