"""The dynamic GGUF's K-quant dequantisation and linear (seedvr2x/kquant.py) on the CPU, against
the `gguf` package's (llama.cpp's own gguf-py, quants.py: numpy, float32, no fused operation),
bit for bit: synthetic blocks of each type (random bytes, the fp16 d and dmin of several
magnitudes, subnormal and zero among them, every 6-bit scale value in every position), and, where
SEEDVR2X_MODEL_DIR holds seedvr2x's two dynamic files, the first and the last matrix of each type
in each. The `gguf` package is the CLI's environment's dev dependency, never the library's: its
tests skip without it. Then the module: the linear under autocast, as an nn.Linear would get
the same weight and bias, in float32 without it; the 7B's 288 block matrices swapped on the meta
device.

On the GPU (pytest -m gpu, skipped without CUDA), the dequantisation on CUDA against the CPU's,
bit for bit, and a linear's memory beside its output, printed for the planner's term and held to
peaks.GGUF_BUFFERS."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from seedvr2x import kquant, peaks, weights

TYPES = tuple(kquant.BLOCK_BYTES)
# Where each type's fp16 d (and dmin) sit in a block, and its scales (ggml-common.h).
HALVES = {"Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2)}
# float16 values of several magnitudes: zero, subnormals (2^-24, the largest, 1023 x 2^-24), the
# smallest normal, small, 1 and large, both signs.
MAGNITUDES = torch.tensor(
    [0.0, 2.0**-24, 1023 * 2.0**-24, 2.0**-14, 3e-4, 0.0123, 0.5, 1.0, 7.3, 1000.0, 65504.0],
    dtype=torch.float16,
)
DIRECTORY = os.environ.get("SEEDVR2X_MODEL_DIR")
FILES = ("seedvr2x_ema_7b_dyn.gguf", "seedvr2x_ema_7b_sharp_dyn.gguf")


def gguf_package() -> Any:
    """The gguf package, or the test skipped saying why."""
    return pytest.importorskip(
        "gguf",
        reason="the gguf package (llama.cpp's gguf-py) is the reference: the CLI's dev group",
    )


def reference(raw: Tensor | np.ndarray[Any, Any], qtype: str) -> np.ndarray[Any, Any]:
    """gguf-py's float32 dequantisation of raw, uint8, blocks in the last axis."""
    gguf = gguf_package()
    data = raw.numpy() if isinstance(raw, Tensor) else raw
    kind = getattr(gguf.GGMLQuantizationType, qtype)
    return np.asarray(gguf.quants.dequantize(data, kind), dtype=np.float32)


def blocks(qtype: str, count: int, seed: int) -> Tensor:
    """count blocks (count, bytes) uint8 of qtype: random bytes, the fp16 fields of each block
    one of MAGNITUDES with a random sign."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(
        0, 256, (count, kquant.BLOCK_BYTES[qtype]), generator=generator, dtype=torch.uint8
    )
    for at in HALVES[qtype]:
        picked = MAGNITUDES[torch.randint(0, len(MAGNITUDES), (count,), generator=generator)]
        signs = torch.where(torch.rand(count, generator=generator) < 0.5, -1.0, 1.0)
        raw[:, at : at + 2] = (picked * signs.to(torch.float16)).view(torch.uint8).reshape(-1, 2)
    return raw


def same_bits(a: Tensor | np.ndarray[Any, Any], b: Tensor | np.ndarray[Any, Any]) -> bool:
    """Equal bit for bit, float32: a NaN or a signed zero too."""
    x = a.numpy() if isinstance(a, Tensor) else a
    y = b.numpy() if isinstance(b, Tensor) else b
    return x.shape == y.shape and np.array_equal(x.view(np.int32), y.view(np.int32))


def scale_values(raw: Tensor, qtype: str) -> set[int]:
    """Which of the 6-bit values appear in every one of a block's scale positions, unpacked here
    from the layouts (ggml-common.h): Q3_K's 16 scales, Q4_K's and Q5_K's 8 scales and 8 minimums.
    Returns the values present in all positions at once."""
    data = raw.numpy()
    if qtype == "Q3_K":
        packed = data[:, 96:108].astype(np.int16)
        low = np.concatenate([packed[:, :8] & 15, packed[:, :8] >> 4], axis=1)
        top = np.concatenate([(packed[:, 8:] >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1)
        positions = [low | (top << 4)]
    else:
        packed = data[:, 4:16].astype(np.int16)
        low, middle, high = packed[:, :4], packed[:, 4:8], packed[:, 8:]
        positions = [
            np.concatenate([low & 63, (high & 15) | ((low >> 6) << 4)], axis=1),
            np.concatenate([middle & 63, (high >> 4) | ((middle >> 6) << 4)], axis=1),
        ]
    present = set(range(64))
    for table in positions:
        for column in range(table.shape[1]):
            present &= {int(value) for value in np.unique(table[:, column])}
    return present


@pytest.mark.parametrize("qtype", TYPES)
def test_dequantisation_is_ggml_s_on_synthetic_blocks(qtype: str) -> None:
    count = 20000
    raw = blocks(qtype, count, seed=7)
    assert scale_values(raw, qtype) == set(range(64))
    out = kquant.dequantise(raw, qtype, (count * 256,))
    assert out.dtype == torch.float32 and out.shape == (count * 256,)
    assert same_bits(out.reshape(count, 256), reference(raw, qtype))


@pytest.mark.parametrize("qtype", TYPES)
def test_blocks_taken_at_a_time_and_rows_change_no_bit(
    qtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A matrix of 6 rows of 3 blocks: the shape and the step (BLOCKS) are the layout's, not the
    # arithmetic's.
    raw = blocks(qtype, 18, seed=3)
    whole = kquant.dequantise(raw, qtype, (6, 768))
    monkeypatch.setattr(kquant, "BLOCKS", 5)
    assert same_bits(kquant.dequantise(raw, qtype, (6, 768)), whole)
    monkeypatch.setattr(kquant, "BLOCKS", 1)
    rows = kquant.dequantise(raw.reshape(6, -1), qtype, (6, 768))
    assert same_bits(rows, whole)
    # Not contiguous: a view with a stride is read as its values are.
    spread = torch.zeros(18, kquant.BLOCK_BYTES[qtype] * 2, dtype=torch.uint8)
    spread[:, ::2] = raw
    assert same_bits(kquant.dequantise(spread[:, ::2], qtype, (6, 768)), whole)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("qtype", TYPES)
def test_a_dtype_is_the_float32_matrix_rounded_to_it(
    qtype: str, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 23 blocks: more than the steps below, with a last short one. Rounded chunk by chunk or whole,
    # the same bits as the float32 matrix's .to(dtype); float32 itself is the float32 matrix.
    raw = blocks(qtype, 23, seed=9)
    whole = kquant.dequantise(raw, qtype, (23, 256))
    assert whole.dtype == torch.float32
    assert same_bits(kquant.dequantise(raw, qtype, (23, 256), torch.float32), whole)
    expected = whole.to(dtype)
    for step in (1, 5, kquant.BLOCKS):
        monkeypatch.setattr(kquant, "BLOCKS", step)
        out = kquant.dequantise(raw, qtype, (23, 256), dtype)
        assert out.dtype == dtype and out.shape == (23, 256)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), step


def test_a_block_of_zeros_and_of_the_largest_scale() -> None:
    # d = 0: every value is zero (Q4_K and Q5_K: less the minimum, also zero), whatever the bytes.
    for qtype in TYPES:
        raw = torch.randint(0, 256, (4, kquant.BLOCK_BYTES[qtype]), dtype=torch.uint8)
        for at in HALVES[qtype]:
            raw[:, at : at + 2] = 0
        assert not kquant.dequantise(raw, qtype, (4, 256)).any()


def test_other_types_and_other_bytes_are_refused() -> None:
    raw = torch.zeros(144, dtype=torch.uint8)
    for other in ("Q6_K", "Q8_0", "F16", "q4_k", ""):
        with pytest.raises(ValueError, match="not a type seedvr2x dequantises"):
            kquant.dequantise(raw, other, (256,))
        with pytest.raises(ValueError, match="not a type seedvr2x dequantises"):
            kquant.GGUFLinear(256, 8, other)
    with pytest.raises(ValueError, match="uint8"):
        kquant.dequantise(raw.to(torch.int8), "Q4_K", (256,))
    with pytest.raises(ValueError, match="not a whole number of Q4_K's 144-byte blocks"):
        kquant.dequantise(raw[:143], "Q4_K", (256,))
    with pytest.raises(ValueError, match=r"1 Q4_K blocks hold 256 values, not \(2, 256\)"):
        kquant.dequantise(raw, "Q4_K", (2, 256))


def model_files() -> list[tuple[str, Path]]:
    """The dynamic files SEEDVR2X_MODEL_DIR holds, as (name, path)."""
    if not DIRECTORY:
        return []
    return [(name, Path(DIRECTORY) / name) for name in FILES if (Path(DIRECTORY) / name).is_file()]


def real_matrices(path: Path) -> list[Any]:
    """The first and the last matrix of each of the three types in a file, in file order, as
    gguf-py's reader gives them (`data` (out, row bytes) uint8, `tensor_type`, `name`)."""
    gguf = gguf_package()
    reader = gguf.GGUFReader(str(path))
    chosen: list[Any] = []
    for qtype in TYPES:
        of_type = [
            tensor
            for tensor in reader.tensors
            if tensor.tensor_type.name == qtype and weights.MATRIX.fullmatch(tensor.name[:-7])
        ]
        assert of_type, f"{path.name}: no {qtype} matrix"
        chosen += [of_type[0], of_type[-1]]
    return chosen


@pytest.mark.parametrize("name", FILES)
def test_dequantisation_is_ggml_s_on_the_dynamic_files_matrices(name: str) -> None:
    gguf_package()
    found = dict(model_files())
    if name not in found:
        pytest.skip(f"SEEDVR2X_MODEL_DIR does not hold {name}")
    for tensor in real_matrices(found[name]):
        raw = torch.from_numpy(np.array(tensor.data))
        shape = (raw.shape[0], raw.shape[1] // kquant.BLOCK_BYTES[tensor.tensor_type.name] * 256)
        out = kquant.dequantise(raw, tensor.tensor_type.name, shape)
        assert same_bits(out, reference(raw, tensor.tensor_type.name)), tensor.name


def layer(qtype: str, inputs: int, outputs: int, seed: int) -> tuple[Tensor, Tensor]:
    """A matrix's bytes (outputs, inputs / 256 * block bytes), and a float16 bias."""
    raw = blocks(qtype, outputs * (inputs // 256), seed).reshape(outputs, -1)
    generator = torch.Generator().manual_seed(seed + 100)
    return raw, (torch.randn(outputs, generator=generator) * 0.5).to(torch.float16)


@pytest.mark.parametrize("qtype", TYPES)
def test_the_module_runs_in_autocast_s_dtype(qtype: str) -> None:
    # As the DiT runs it, under the pipeline's bfloat16 autocast, an input in float32: the
    # output in bfloat16, the weight dequantised in float32 and rounded to bfloat16, the bias
    # float16 widened to it; and what autocast gives an nn.Linear holding that same float32 weight
    # and float16 bias, which casts input, weight and bias alike.
    inputs, outputs = 512, 24
    raw, bias = layer(qtype, inputs, outputs, seed=5)
    matrix = kquant.dequantise(raw, qtype, (outputs, inputs))
    # The random blocks' values are large, but bfloat16 holds float32's range.
    assert torch.isfinite(matrix.bfloat16()).all()
    x = torch.randn(2, 3, inputs, generator=torch.Generator().manual_seed(1))
    module = kquant.GGUFLinear(inputs, outputs, qtype)
    assert [name for name, _ in module.named_parameters()] == ["weight", "bias"]
    assert not any(parameter.requires_grad for parameter in module.parameters())
    assert module.weight.dtype == torch.uint8 and module.weight.shape == raw.shape
    assert module.bias is not None and module.bias.dtype == torch.float16
    assert module.qtype == qtype and not list(module.buffers())
    assert set(module.state_dict()) == {"weight", "bias"}
    module.load_state_dict({"weight": raw, "bias": bias}, assign=True)
    expected = torch.nn.functional.linear(x.bfloat16(), matrix.bfloat16(), bias.bfloat16())
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        out = module(x)
        linear = torch.nn.Linear(inputs, outputs)
        linear.weight = torch.nn.Parameter(matrix, requires_grad=False)
        linear.bias = torch.nn.Parameter(bias, requires_grad=False)
        autocast_out = linear(x)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 3, outputs)
    assert torch.equal(out, expected)
    assert torch.equal(out, autocast_out)
    assert torch.equal(out, kquant.linear(x, raw, bias, qtype, torch.bfloat16))


@pytest.mark.parametrize("qtype", TYPES)
def test_the_module_without_autocast_runs_in_the_input_s_dtype(qtype: str) -> None:
    inputs, outputs = 256, 16
    raw, bias = layer(qtype, inputs, outputs, seed=6)
    matrix = kquant.dequantise(raw, qtype, (outputs, inputs))
    x = torch.randn(5, inputs, generator=torch.Generator().manual_seed(2))
    module = kquant.GGUFLinear(inputs, outputs, qtype)
    module.load_state_dict({"weight": raw, "bias": bias}, assign=True)
    with torch.no_grad():
        out = module(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.nn.functional.linear(x, matrix, bias.float()))
    # Bias None: no bias parameter, no addition.
    unbiased = kquant.GGUFLinear(inputs, outputs, qtype, bias=False)
    assert unbiased.bias is None and "bias=False" in repr(unbiased) and qtype in repr(unbiased)
    unbiased.load_state_dict({"weight": raw}, assign=True)
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        out = unbiased(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch.nn.functional.linear(x.bfloat16(), matrix.bfloat16()))
    with pytest.raises(ValueError, match="300 inputs: not a multiple of the blocks' 256"):
        kquant.GGUFLinear(300, 16, qtype)


class Toy(torch.nn.Module):
    """A DiT of two blocks with the 7B's block matrices' names (weights.MATRIX), 256 inputs and
    512 outputs, and linears that are not block matrices beside them."""

    def __init__(self) -> None:
        super().__init__()

        def linear() -> torch.nn.Linear:
            return torch.nn.Linear(256, 512)

        def pair() -> torch.nn.ModuleDict:
            return torch.nn.ModuleDict({"vid": linear(), "txt": linear()})

        self.emb_in = torch.nn.Linear(256, 512)
        self.blocks = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            block.attn = torch.nn.ModuleDict({"proj_qkv": pair(), "proj_out": pair()})
            block.mlp = torch.nn.ModuleDict(
                {
                    "vid": torch.nn.ModuleDict({"proj_in": linear(), "proj_out": linear()}),
                    "txt": torch.nn.ModuleDict({"proj_in": linear(), "proj_out": linear()}),
                }
            )
            block.norm = torch.nn.Linear(256, 512)
            self.blocks.append(block)


def test_the_matrices_are_swapped_with_their_types_and_the_order_kept() -> None:
    with torch.device("meta"):
        dit = Toy()
    names = [name for name, module in dit.named_modules() if weights.MATRIX.fullmatch(name)]
    assert len(names) == 16 and names[0] == "blocks.0.attn.proj_qkv.vid"
    types = {name: TYPES[index % 3] for index, name in enumerate(names)}
    types["emb_in"] = "Q4_K"  # not a block matrix: ignored
    types["blocks.0.norm"] = "nonsense"
    before = [name for name, _ in dit.named_parameters()]
    kinds = {name: type(module) for name, module in dit.named_modules()}
    swapped = kquant.swap(dit, types)
    assert swapped == names
    assert [name for name, _ in dit.named_parameters()] == before
    for name, module in dit.named_modules():
        if name in types and name in names:
            assert isinstance(module, kquant.GGUFLinear) and module.qtype == types[name]
            assert (module.in_features, module.out_features) == (256, 512)
            assert module.weight.device.type == "meta" and module.weight.dtype == torch.uint8
            assert module.weight.shape == (512, kquant.BLOCK_BYTES[types[name]])
            assert module.bias is not None and module.bias.dtype == torch.float16
        else:
            assert type(module) is kinds[name], name
    assert isinstance(dit.emb_in, torch.nn.Linear)


def test_a_swap_with_a_type_missing_or_unknown_leaves_the_model_as_it_was() -> None:
    with torch.device("meta"):
        dit = Toy()
    names = [name for name, module in dit.named_modules() if weights.MATRIX.fullmatch(name)]
    types = dict.fromkeys(names, "Q4_K")
    missing = {name: kind for name, kind in types.items() if name != names[-1]}
    with pytest.raises(ValueError, match=f"^{names[-1]}: no ggml type given"):
        kquant.swap(dit, missing)
    with pytest.raises(ValueError, match="'Q6_K' is not a type seedvr2x dequantises"):
        kquant.swap(dit, {**types, names[3]: "Q6_K"})
    assert not any(isinstance(module, kquant.GGUFLinear) for module in dit.modules())


def test_the_7b_s_block_matrices_are_swapped() -> None:
    # Built on the meta device: the 288 nn.Linear of the blocks' attention and MLP
    # (weights.MATRIX), each a GGUFLinear of its shape and bias, in its place, nothing else
    # touched.
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
    types = {name: TYPES[index % 3] for index, name in enumerate(linears)}
    swapped = kquant.swap(dit, types)
    assert swapped == list(linears) and len(swapped) == 288
    after = dict(dit.named_modules())
    for name, kind in before.items():
        module = after[name]
        if name in linears:
            assert isinstance(module, kquant.GGUFLinear) and module.qtype == types[name]
            assert (module.in_features, module.out_features, module.bias is not None) == (
                linears[name]
            )
            assert module.weight.device.type == "meta" and module.weight.dtype == torch.uint8
        else:
            assert type(module) is kind, name
    assert max(inputs * outputs for inputs, outputs, _ in linears.values()) == 3072 * 12288


# ---- On the GPU ----------------------------------------------------------------------------


def cuda() -> torch.device:
    """The first CUDA device; the test is skipped without one."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda:0")


@pytest.mark.gpu
@pytest.mark.parametrize("qtype", TYPES)
def test_dequantisation_on_cuda_is_the_cpu_s_on_synthetic_blocks(qtype: str) -> None:
    device = cuda()
    count = 20000
    raw = blocks(qtype, count, seed=11)
    expected = kquant.dequantise(raw, qtype, (count, 256))
    out = kquant.dequantise(raw.to(device), qtype, (count, 256))
    assert out.device.type == "cuda" and out.dtype == torch.float32
    assert same_bits(out.cpu(), expected)
    # And gguf-py's, where it is installed: the arithmetic itself, not only the CPU's.
    if pytest.importorskip("gguf", reason="the reference"):
        assert same_bits(out.cpu(), reference(raw, qtype))


@pytest.mark.gpu
@pytest.mark.parametrize("name", FILES)
def test_dequantisation_on_cuda_is_the_cpu_s_on_the_dynamic_files_matrices(name: str) -> None:
    device = cuda()
    found = dict(model_files())
    if name not in found:
        pytest.skip(f"SEEDVR2X_MODEL_DIR does not hold {name}")
    for tensor in real_matrices(found[name]):
        qtype = tensor.tensor_type.name
        raw = torch.from_numpy(np.array(tensor.data))
        shape = (raw.shape[0], raw.shape[1] // kquant.BLOCK_BYTES[qtype] * 256)
        out = kquant.dequantise(raw.to(device), qtype, shape)
        assert same_bits(out.cpu(), kquant.dequantise(raw, qtype, shape)), tensor.name


@pytest.mark.gpu
@pytest.mark.parametrize("qtype", TYPES)
def test_a_linear_s_memory_beside_its_output_is_printed(
    qtype: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The 7B's largest matrix, 3072 x 12288, one latent's tokens of 1080p (8,280) in bfloat16:
    # the peak above what the call starts with (the matrix's bytes and the input) less its
    # output, for the planner's term (the module's docstring: the matrix in bfloat16,
    # 75.5 MB, plus a chunk's temporaries, 52 MB on the CPU at most).
    device = cuda()
    inputs, outputs, rows = 3072, 12288, 8280
    raw = blocks(qtype, outputs * (inputs // 256), seed=13).reshape(outputs, -1).to(device)
    bias = torch.zeros(outputs, dtype=torch.float16, device=device)
    x = torch.randn(rows, inputs, device=device).bfloat16()
    module = kquant.GGUFLinear(inputs, outputs, qtype, device=device)
    module.load_state_dict({"weight": raw, "bias": bias}, assign=True)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        module(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        out = module(x)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    beside = peak - start - out.numel() * out.element_size()
    with capsys.disabled():
        print(
            f"\n{qtype}: {outputs} x {inputs} matrix, {rows} rows bfloat16: peak {peak - start:,}"
            f" bytes above the start, {beside:,} beside the output ({beside / 1e6:.0f} MB; the"
            f" float32 matrix {outputs * inputs * 4 / 1e6:.0f} MB, bfloat16's"
            f" {outputs * inputs * 2 / 1e6:.0f} MB)"
        )
    assert beside >= outputs * inputs * 2
    # The planner's term is what the RTX 5090 measured (2026-11-04): the matrix in bfloat16 and
    # 24,576 bytes beside the output, the chunk's temporaries freed before the output is allocated.
    # <=, not ==, in case another allocator granularity moves the figure by a few bytes.
    assert beside <= peaks.GGUF_BUFFERS, (qtype, beside, peaks.GGUF_BUFFERS)
