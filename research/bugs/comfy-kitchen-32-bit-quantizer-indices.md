# comfy-kitchen: the CUDA fp8 and NVFP4 quantizers index in 32 bits

| | |
|---|---|
| Project | comfy-kitchen (Comfy-Org, Apache-2.0), its CUDA backend: `backends/cuda/ops/per_tensor_quantize.cu`, `quantize_nvfp4.cu` (not numz) |
| Severity | wrong output: from 2^32 values in one tensor, fp8's output is mostly never written (an all-NaN video in our runs), NVFP4's is silently wrong |
| Status | measured on the GPU, and read in the code; **not filed yet** |
| Version | comfy-kitchen 0.2.37 (tag `v0.2.37`, `be003b7`), torch 2.14.1+cu130, an RTX PRO 6000 Blackwell (sm_120) |
| Workaround | keep each quantized tensor under 2^32 values, or quantize it in row chunks at the scale of the whole tensor |

## Summary

The CUDA kernels that quantize a tensor at run time, a layer's input in a W8A8 or W4A4 multiply
(`QuantizedTensor.from_float`), index in 32 bits. From 2^32 values on:

- **fp8, one scale per tensor** (`quantize_per_tensor_fp8`, `TensorCoreFP8Layout`): the kernel
  takes the value count as a `uint32_t`, so it sees the count modulo 2^32, and writes that many
  values only: of a tensor of 2^32 + r values, the first r. The other 2^32 bytes of its output,
  allocated with `torch.empty`, keep whatever the memory held. In fp8 E4M3 some of those bytes
  are NaN, and one NaN in a layer's input makes the next quantization's per-tensor scale (its
  absolute maximum) NaN, then every value after it: a SeedVR2 7B run at 3840×2160 came out NaN
  on every frame.
- **NVFP4** (`quantize_nvfp4`, `TensorCoreNVFP4Layout`), when the rows and columns are multiples
  of 16 (its aligned path): the load offset is computed in 32 bits, so from value 2^32 on the
  kernel quantizes the values 2^32 earlier, and writes them, with their block scales, in the
  right place: finite, wrong, without an error. Its padded (misaligned) path computes the offsets
  in 64 bits and is right.

int8's quantizer (`quantize_int8_rowwise_convrot`, `int8_linear.cu`) offsets its rows in 64 bits:
the same bytes chunked or not at 4,777,574,400 values. Multiplies that quantize no input (W8A16,
W4A16) aren't concerned, nor are a model's weights.

Read in the code, not run: fp8's dequantizer and its stochastic rounding take the same `uint32_t`
count and index, and NVFP4's dequantizer offsets its stores in 32 bits.

## Reproduction

Measured with 12,288 columns (the SeedVR2 7B's widest linear input) of bf16 random values:
comfy-kitchen's quantization of the whole tensor against the same quantizer run on chunks of rows
under 2^30 values, at the whole tensor's scale. For fp8, a freed block of the output's size
filled with 0x55 came first, which the caching allocator hands to the output, so that the bytes
the kernel never writes read 0x55.

| Values | Rows | fp8: bytes that differ, from value | NVFP4: bytes that differ, from value |
|---|---|---|---|
| 2^32 − 65,536 | 349,520 | | 0 |
| 2^32 − 4,096 | 349,525 | 0 | |
| 2^32 + 8,192 | 349,526 | 4,271,944,933, from 8,192 | |
| 2^32 + 131,072 | 349,536 | | 65,256, from 2^32 |
| 4,459,069,440 | 362,880 | 4,270,303,325, from 164,102,144 | 81,703,327, from 2^32 |
| 4,777,574,400 | 388,800 | 4,271,934,215, from 482,607,104 | 240,280,102, from 2^32 |
| 4,777,586,688 | 388,801 (padded to 388,816) | | 0 |

- fp8: every byte that differs reads 0x55, and they start at value r = count − 2^32: the last
  2^32 bytes were never written. The 0.5% of those that match are bytes whose right value is
  0x55 itself.
- NVFP4: from value 2^32 on, the data and the block scales equal those from value 0; before it,
  every byte matches. With rows that aren't a multiple of 16, the padded path, nothing differs.

The same, condensed into comfy-kitchen's own functions (not run in this exact form; about 30 GB
of GPU memory):

```python
import torch
import comfy_kitchen as ck

# fp8: the whole tensor against chunks of rows at the same scale (TensorCoreFP8Layout's formula)
rows, cols = 349_526, 12_288  # 2**32 + 8,192 values; 349,525 rows (2**32 - 4,096) come out right
x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
scale = (x.abs().amax() / 448).float()
ref = torch.cat([ck.quantize_per_tensor_fp8(c, scale).view(torch.uint8) for c in x.split(65_536)])
torch.full((rows, cols), 0x55, dtype=torch.uint8, device="cuda")  # freed: the output's block
out = ck.quantize_per_tensor_fp8(x, scale).view(torch.uint8).flatten()
ref = ref.flatten()
r = out.numel() - 2**32  # 8,192
print(torch.equal(out[:r], ref[:r]), int((out[r:] != ref[r:]).sum()))  # True 4271944933
del x, ref, out

# NVFP4, aligned path: from value 2**32 on, the first values again
x = torch.randn(349_536, cols, device="cuda", dtype=torch.bfloat16)  # rows a multiple of 16
q, _ = ck.quantize_nvfp4(x, (x.abs().amax() / (448 * 6)).float())
q = q.flatten()  # two values per byte: value 2**32 is byte 2**31
print(torch.equal(q[2**31:], q[: q.numel() - 2**31]))  # True
```

## Code path (comfy-kitchen v0.2.37)

- `backends/cuda/__init__.py:561-584`, `quantize_per_tensor_fp8`: the output is
  `torch.empty(x.shape, dtype=torch.uint8)`, and `numel = x.numel()` is passed on.
- `backends/cuda/ops/per_tensor_quantize.cu:170-201`, `launch_quantize_fp8_kernel(..., int64_t
  numel, ...)`: the blocks come from the 64-bit count (16 values per thread, 128 threads per
  block), then the kernel gets `numel` for its `const uint32_t size`, the count modulo 2^32.
- `:39-71`, `quantize_fp8_tensor_kernel`: `uint32_t idx = blockIdx.x * blockDim.x +
  threadIdx.x; idx *= kE4M3Alignment;` (16), then `if (idx >= size) return;`. The threads below
  the truncated count write the first r values; those past value 2^32 wrap to an index below it
  and write the same values again; every other thread returns. The last 2^32 values are never
  written.
- `:74-119`, `dequantize_fp8_tensor_kernel`, and `:122-161`, `stochastic_round_fp8_kernel`: the
  same `uint32_t size` and index.
- `quantize_nvfp4.cu`: `kValsPerThread` is a `constexpr unsigned int` (`:31`), and the thread
  index an `unsigned int` (`:77`). The aligned path loads `input + kValsPerThread * idx`
  (`:102-105`), a product in 32 bits that wraps at value 2^32. The store, `store_fp4x4(OType*,
  size_t idx, ...)` at `output[2*idx]` (`float_utils.cuh:140-151`), and the block scale's offset,
  `static_cast<size_t>(idx) * kValsPerThread` (`:153`), are 64-bit, so the wrapped values land
  in the right place. The misaligned path computes in `size_t` (`:82-100`).
- `quantize_nvfp4.cu:217-296`, `dequantize_nvfp4_kernel`: it writes at `output + kValsPerThread *
  idx`, 32 bits again.
- Not concerned: int8 (`int8_linear.cu:198` and `:303`, `const int64_t row_offset =
  static_cast<int64_t>(row) * K`), and MXFP8 (`quantize_mxfp8.cu:50`,
  `static_cast<size_t>(idx) * kMXFP8ValsPerThread`).

## Impact

- A W8A8 or W4A4 multiply quantizes each linear layer's input. In the SeedVR2 7B the widest is
  the MLP's output projection, 12,288 values per token, which reaches 2^32 from 349,526 video
  tokens in one pass: at 3840×2160 from 41 frames (45 frames: 388,800 tokens; 37: 324,000), at
  1080p from 169. fp8 W8A8 then gives an all-NaN video, NVFP4 W4A4 a wrong one without a word.
- Any caller that quantizes a tensor of 2^32 values or more through these kernels meets it;
  comfy-kitchen is ComfyUI's kernel library. Video models at high resolution get there first.
- Weights stay far below the limit (the 7B's largest tensor holds 56.6 million values), and
  int8 runs right.

## Workaround

- Keep every quantized tensor under 2^32 values: for the 7B at 3840×2160, at most 37 frames per
  pass.
- Or quantize a larger tensor in chunks of rows, each through comfy-kitchen's own quantizer at the
  scale its layout computes for the whole tensor (the absolute maximum over the whole tensor,
  divided by 448, or by 448 × 6 for NVFP4): comfy-kitchen's values bit for bit wherever its
  kernel is right. seedvr2x's validation runs do this above 2^31 − 1 values
  ([ck_patch.py](../../models/gpu/ck_patch.py),
  [VALIDATION.md](../../models/VALIDATION.md#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices)), and
  seedvr2x's phase-2 runtime will too ([DESIGN.md](../../seedvr2x/DESIGN.md#weights)).

## Possible fix

64-bit counts and offsets: `int64_t` or `size_t` for `size` and `idx` in
`per_tensor_quantize.cu`'s three kernels, and `static_cast<size_t>(idx) * kValsPerThread` for
NVFP4's aligned load and its dequantizer's store, as its misaligned path and MXFP8 already do; or
a grid-stride loop over a `size_t` index. A test: the reproduction above, the whole tensor
against its chunks, equal byte for byte at 2^32 + 8,192 values for fp8 and at 2^32 + 131,072
(rows a multiple of 16) for NVFP4.

## Before filing

- Run the condensed reproduction as written, on comfy-kitchen's current main; check whether a
  release after 0.2.37 fixed it.
- Search comfy-kitchen's issues.
- Attach the table, the snippet and the kernel lines above.
