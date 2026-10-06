# Formats: what each one does to SeedVR2's weights

> Status: **measured on the CPU**, weights only, against ByteDance's fp32 masters:
> [`formats_study.py`](formats_study.py), [`numz_check.py`](numz_check.py), and each build
> script's own checks ([`seedvr2_fp8.py`](seedvr2_fp8.py), [`seedvr2_gguf.py`](seedvr2_gguf.py),
> [`seedvr2_int8.py`](seedvr2_int8.py), [`seedvr2_nvfp4.py`](seedvr2_nvfp4.py)), with
> [`ck_check.py`](ck_check.py) and [`numz_gguf_check.py`](numz_gguf_check.py) loading the files in
> their runtimes. What each file does to the video is measured by GPU runs to come: until then its
> quality is "to be measured". [DESIGN.md](../seedvr2x/DESIGN.md#weights) (Weights) cites these
> figures.

In short, on the 7B's 288 attention and MLP matrices of the blocks, 99% of its weights, the error
per weight being ||W_hat - W|| / ||W|| against the fp32 master:

- **fp8 needs a scale.** Cast with no scale, as numz's and Comfy-Org's fp8 files are, a typical
  matrix loses about as much as with one, 2.78% against 2.65%; but 43 of the 288 lose more than
  3%, and the worst 16.7%: those whose weights are so small that most of them fall below 2^-6,
  fp8's smallest normal value, where it keeps fewer bits. One scale per tensor, max|W| / 448,
  brings every matrix to 2.49-2.67%.
- **One scale per tensor is enough.** One per output row gains 0.2% in the median and 0.9% at
  best, is worse on some matrices, and isn't a layout comfy-kitchen multiplies.
- **Where numz's fp8 loses:** the text branch of the last blocks (blocks 30-35, 70-100% of their
  weights subnormal, up to 16.7%), the timestep embedding (`emb_in.proj_hid`, 15.5%, all its
  weights subnormal), the output layer (`vid_out.proj`, 7.7%) and biases (up to 16.8%). numz's
  Q4_K_M keeps the embedding and the output layer in 16 bits: the likely reason it measured closer
  to the source than fp8 ([models.md](../research/docs/models.md)).
- **Every format, typical (worst):** float16 0.02% (0.02%); fp8 with a scale per tensor 2.65%
  (2.67%); int8 with a scale per row 1.01% (1.78%), 0.86% (1.08%) with comfy-kitchen's rotation;
  GGUF Q8_0 0.56% (0.64%); Q4_K 7.35% (7.95%); NVFP4 9.45% (10.1%) with comfy-kitchen's scales,
  8.80% (8.94%) with each block's scale chosen among 8 to minimise its error, as in our file.
- **numz's 3B is ByteDance's first 3B.** ByteDance replaced the 3B's weights on 2025-06-22
  ("update ckpt"); numz's 3B fp16 file is the earlier master rounded to the nearest float16, and
  its fp8 file that master cast straight to fp8, every one of 635 tensors; against the current
  master they agree on 4.9% and 96% of the values.
- **Our phase-2 files** (the 7B and the sharp 7B): fp8 8.33 GB, Q4_K 4.76 GB, Q8_0 8.84 GB, INT8
  8.33 GB, NVFP4 4.76 GB, each loading in its runtime, made the same on every run.

## Method

- **The tensors:** the 288 attention and MLP matrices of the blocks (`attn.proj_qkv`,
  `attn.proj_out`, `mlp.proj_in`, `mlp.proj_out`, each for the video and the text stream, in 36
  blocks: 8,153,726,976 of the 8,239,608,232 weights). The 840 others (6 matrices outside the
  blocks, biases, norms, modulation, RoPE's frequencies) stay in 16 bits in every file of ours,
  as in numz's Q4_K_M.
- **The error of a matrix:** ||W_hat - W|| / ||W||, W the fp32 master's, W_hat what the format
  stores, decoded. Per format, the median over the 288 matrices ("typical"), the worst, and the
  pooled error over all their weights at once.
- **Simulated formats** ([`formats_study.py`](formats_study.py)): float16 (nearest, ties to even);
  fp8 E4M3 with no scale, one per tensor (max|W| / 448) or one per output row; int8 with one scale
  per row (max / 127); GGUF Q8_0 (gguf-py's quantizer); NVFP4 (E2M1 values, an E4M3 scale per 16
  values along the input, a float32 scale per tensor, max|W| / (448 x 6)): each block's scale as
  comfy-kitchen sets it, from the block's max, or chosen among that one and the 7 E4M3 values
  below it to minimise the block's squared error. And numz's Q4_K_M file as it is.
- **Our files:** each build script measures its own file the same way, decoded.

## fp8: no scale, one per tensor, one per row

| fp8 E4M3 | Median | Pooled | Worst | Matrices over 3% |
|---|---|---|---|---|
| no scale (numz's, Comfy-Org's) | 2.78% | 2.80% | 16.7% | 43 (6 over 5%, 2 over 10%) |
| one scale per tensor | 2.65% | 2.65% | 2.67% | 0 |
| one scale per output row | 2.64% | 2.64% | 2.65% | 0 |

- Without a scale, a median of 36% of each matrix's nonzero weights fall in E4M3's subnormal
  range (below 2^-6 = 0.0156), up to 100%; with one per tensor, 0.06%, up to 0.34%.
- Per row against per tensor, matrix by matrix: from 0.991 to 1.004 times the error.

## Where a cast without scale loses

The worst block matrices, then the worst other tensors, cast with no scale as numz's fp8 file
stores them:

| Tensor | No scale | One scale per tensor | max\|W\| | Subnormal with no scale |
|---|---|---|---|---|
| `blocks.35.mlp.txt.proj_out.weight` | 16.7% | 2.49% | 0.0058 | 100% |
| `blocks.34.mlp.txt.proj_out.weight` | 11.3% | 2.66% | 0.622 | 99.6% |
| `blocks.35.mlp.txt.proj_in.weight` | 8.30% | 2.49% | 0.0116 | 100% |
| `blocks.35.attn.proj_out.txt.weight` | 8.30% | 2.49% | 0.0116 | 100% |
| `blocks.33.mlp.txt.proj_out.weight` | 7.11% | 2.64% | 0.397 | 94.3% |
| `blocks.34.mlp.txt.proj_in.weight` | 6.88% | 2.64% | 0.211 | 96.4% |
| `blocks.35.mlp.txt.proj_out.bias` | 16.8% | 2.52% | 0.0058 | |
| `emb_in.proj_hid.weight` (the timestep embedding) | 15.5% | 2.65% | 0.153 | 100% |
| `vid_out.proj.weight` (the output layer) | 7.66% | 2.65% | 0.0753 | 95.4% |
| `emb_in.proj_out.weight` | 3.86% | 2.65% | 0.734 | 74.0% |

- The last blocks' text branch carries tiny weights (block 35's text MLP: max 0.0058): numz's
  "mixed" 7B fp8 file keeps block 35 in 16 bits, where a cast without scale would put all of
  them in the subnormal range.
- Biases: median 3.1%, up to 16.8%; modulation tables up to 4.9%; norms up to 3.4%; RoPE's
  frequencies 3.1% (the same in every block).

## Every format on the 7B

| Format | Median | Pooled | Worst | Bits per weight |
|---|---|---|---|---|
| float16 | 0.021% | 0.021% | 0.021% | 16 |
| fp8, one scale per tensor | 2.65% | 2.65% | 2.67% | 8 |
| int8, one scale per row | 1.01% | 1.07% | 1.78% | 8 |
| int8, comfy-kitchen's rotation | 0.86% | – | 1.08% | 8 |
| GGUF Q8_0 | 0.56% | 0.56% | 0.64% | 8.5 |
| GGUF Q4_K (numz's Q4_K_M, ours) | 7.35% | 7.39% | 7.95% | 4.5 |
| NVFP4, comfy-kitchen's scales | 9.45% | 9.44% | 10.1% | 4.5 |
| NVFP4, each block's scale the best of 8 | 8.80% | 8.80% | 8.94% | 4.5 |

- **fp8 is not the precise 8-bit format:** 3 mantissa bits, about 2.7% per weight. int8 with a
  scale per row is 2.6 times closer; Q8_0, an int8 with a scale every 32 weights, 4.7 times. fp8's
  virtue is its multiply on RTX 40 and later, not its precision.
- **At 4 bits, Q4_K is closer than NVFP4 as comfy-kitchen quantizes it** (7.35% against 9.45%):
  Q4_K's scale and minimum per 32 weights, searched to minimise the error, against NVFP4's scale
  per 16 from the block's max with one mantissa bit per value. Choosing each NVFP4 block's scale
  to minimise its error narrows the gap (8.80%); the layout is unchanged, so comfy-kitchen loads
  such a file as it is: ours is one ([`seedvr2_nvfp4.py`](seedvr2_nvfp4.py)).

## numz's files against ByteDance's masters

[`numz_check.py`](numz_check.py), every tensor of numz's file against the master rounded to the
file's dtype (float16: nearest, ties to even; fp8: the master cast straight to E4M3):

| numz's file | Master | Equal to the master's rounding | Equal to numz's fp16 cast again |
|---|---|---|---|
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16` (the control) | 7B (`eb0c428`) | 100%: 1,097 fp8 tensors, 31 fp16 | 99.72% |
| `seedvr2_ema_3b_fp16` | 3B, current (`37255ff`, 2025-06-22) | 4.9% | |
| `seedvr2_ema_3b_fp16` | 3B, first (`e2bc8d4`, 2025-06-11 to 06-20) | **100%**, 635 tensors | |
| `seedvr2_ema_3b_fp8_e4m3fn` | 3B, current | 96.0% | 99.76% |
| `seedvr2_ema_3b_fp8_e4m3fn` | 3B, first | **100%**, 635 tensors | 99.76% |

- The control finds what measurement found: numz's 7B fp8 is the master cast straight to fp8, a
  cast of its fp16 file differing on 0.28% of the values.
- numz's 3B files are made the same way, but from the 3B's first weights, which ByteDance
  replaced on 2025-06-22 (`seedvr2_ema_3b.pth` 91627bba… then 6bcc5ac5…, same size; the 7B's
  never changed). Every measurement of the 3B so far (models.md) ran those first weights.

## Our phase-2 files

From ByteDance's fp32 masters (`ByteDance-Seed/SeedVR2-7B` at `eb0c428`), the 288 block matrices
quantized, every other tensor our fp16 file's, byte for byte:

| File | Bytes | SHA-256 | Error: median (worst) | Checks |
|---|---|---|---|---|
| `seedvr2x_ema_7b_fp8_scaled.safetensors` | 8,325,670,552 | `3222ce3e…` | 2.648% (2.669%) | read back; comfy-kitchen decodes all 288 bit for bit, its multiply exact |
| `seedvr2x_ema_7b_sharp_fp8_scaled.safetensors` | 8,325,670,560 | `6cdf191b…` | 2.648% (2.668%) | read back; comfy-kitchen decodes all 288 bit for bit, its multiply exact |
| `seedvr2x_ema_7b_Q4_K.gguf` | 4,758,307,552 | `7f4642d0…` | 7.345% (7.948%), numz's the same, ours lower on 165 of 288 | ggml = gguf-py; numz's loader |
| `seedvr2x_ema_7b_sharp_Q4_K.gguf` | 4,758,307,584 | `a5e423a5…` | 7.347% (7.942%), numz's the same, ours lower on 187 | ggml = gguf-py; numz's loader |
| `seedvr2x_ema_7b_Q8_0.gguf` | 8,835,171,040 | `67ea572a…` | 0.558% (0.641%) | ggml = gguf-py; numz's loader |
| `seedvr2x_ema_7b_sharp_Q8_0.gguf` | 8,835,171,072 | `03fad523…` | 0.558% (0.639%) | ggml = gguf-py; numz's loader |
| `seedvr2x_ema_7b_int8_convrot.safetensors` | 8,333,648,992 | `7eb2c784…` | 0.855% (1.075%) | read back; comfy-kitchen's own quantizer gives the same codes on 99.9994%; comfy-kitchen decodes all 288 as we do (to float32's precision), its int8 multiply 0.74% from the float one on a layer (the activations' own rounding) |
| `seedvr2x_ema_7b_sharp_int8_convrot.safetensors` | 8,333,649,000 | `d83aeaa4…` | 0.855% (1.077%) | read back; comfy-kitchen's own quantizer gives the same codes on 99.9994%; comfy-kitchen decodes all 288 as we do (to float32's precision), its int8 multiply 0.74% from the float one on a layer (the activations' own rounding) |
| `seedvr2x_ema_7b_nvfp4.safetensors` | 4,758,446,416 | `9cd14359…` | 8.802% (8.941%), comfy-kitchen's own quantizer 9.451% (10.087%), ours lower on all 288 | read back; comfy-kitchen's own quantizer gives the search's first candidate, every code and scale; comfy-kitchen decodes all 288 bit for bit, its multiply with a float input exact |
| `seedvr2x_ema_7b_sharp_nvfp4.safetensors` | 4,758,446,424 | `d0a1d5a4…` | 8.803% (8.951%), comfy-kitchen's own quantizer 9.452% (10.142%), ours lower on all 288 | read back; comfy-kitchen's own quantizer gives the search's first candidate, every code and scale; comfy-kitchen decodes all 288 bit for bit, its multiply with a float input exact |

- **fp8:** comfy-kitchen's FP8 layout, `<layer>.weight` (E4M3, W x (1/s) rounded to nearest, ties
  to even, clamped to +-448) and `<layer>.weight_scale` (float32, s = max|W| / 448), marked by
  `<layer>.comfy_quant` = {"format": "float8_e4m3fn"} as Comfy-Org's SeedVR2 files mark theirs.
- **GGUF:** quantized by ggml's own `ggml_quantize_chunk` (llama.cpp `abeada3`, built from source
  without native CPU flags, no importance matrix) straight from float32, written by gguf-py in
  city96's conventions (`general.architecture` "seedvr", shapes reversed). Q4_K is the type of
  every quantized tensor in numz's Q4_K_M file. ggml's and gguf-py's decoders agree bit for bit
  on every tensor. numz's own loader reads every file: names, shapes, 288 `GGUFQuantizedLinear`,
  its float32 decoding equal to gguf-py's; its forward decodes in float16, which rounds Q4_K's
  values twice (equal to one rounding on 29% of them, up to 0.17% of the tensor's max away): a
  runtime matter, worth a float32 decode in seedvr2x's.
- **INT8:** comfy-kitchen's INT8 layout with its rotation: each 256 columns of W times H, the
  regular Hadamard H4 (x) H4 (x) H4 (x) H4 / 16 (deterministic: nothing to store), one float32
  scale per row (max / 127), values rounded to nearest, ties to even; marked by
  {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}. At run time
  comfy-kitchen rotates each input the same way, inside its int8 linear, before quantizing it per
  token: (x H)(W H)^T = x W^T, so nothing is rotated back.
- **NVFP4:** comfy-kitchen's NVFP4 layout, `<layer>.weight` (uint8 [N, K/2], two E2M1 codes per
  byte, the even value's in the high nibble, the sign in bit 3), `<layer>.weight_scale` (E4M3, one
  per 16 values along the input, in cuBLAS's 128x4 tiles) and `<layer>.weight_scale_2` (float32,
  s2 = max|W| / (448 x 6)); marked by {"format": "nvfp4"} as Comfy-Org's SeedVR2 files mark theirs.
  Each block's scale is searched: comfy-kitchen takes the block's max / 6 / s2, rounded to the
  nearest E4M3; we try that one and the 7 E4M3 values below it, each with the block's values
  rounded to the nearest E2M1 (ties to even, saturated at +-6), and keep the one with the smallest
  squared error, the first of equals. The first candidate is comfy-kitchen's own quantizer's, so
  no matrix can come out worse. comfy-kitchen decodes any block scale, the max's or not: it loads
  the file as it is, and the search costs nothing at run time.

## How each format runs

The users' table, as the card gives it once the files are published, quality "to be measured"
until the GPU runs. "W8A8": weights and activations in 8 bits, the multiply in 8 bits;
"W8A16": weights stored in 8 bits, widened to 16 for the multiply (memory only).

| Format | Size (7B) | How it multiplies | Faster on | Error per weight | Quality |
|---|---|---|---|---|---|
| float16 | 16.5 GB | 16-bit | every GPU, the reference | 0.02% | the reference |
| fp8, a scale per tensor | 8.3 GB | W8A8 from RTX 40 (Ada, sm_89); W8A16 before | RTX 40, 50 | 2.65% | to be measured |
| int8, rotated | 8.3 GB | W8A8 from RTX 20 (Turing, sm_75) | RTX 20 to 50 | 0.86% | to be measured |
| GGUF Q8_0 | 8.8 GB | W8A16 | none: memory only | 0.56% | to be measured |
| GGUF Q4_K | 4.8 GB | W4A16 | none: memory only, a little slower | 7.35% | to be measured |
| NVFP4, searched scales | 4.8 GB | W4A4 on Blackwell (sm_100, sm_120); W4A16 before | RTX 50 | 8.80% | to be measured |

- Speed: only the DiT gets faster, about a fifth of a 1080p job (the VAE, in 16 bits, is the
  rest): a DiT twice as fast shortens a job by about 10%. The main gain is memory. No speed is
  measured here: the GPU box has a power-cap fault, so this column states native support only,
  until a healthy card measures it.
- An 8- or 4-bit multiply rounds each layer's input too: a second loss the error per weight
  doesn't show, which only the GPU runs measure.

## Reproduce

```bash
M="--masters /path/to/bytedance --cache /path/to/cache"
uv run models/numz_check.py --numz /path/to/numz $M          # numz's 3B (both masters) and the 7B control
uv run models/formats_study.py /path/to/seedvr2_ema_7b.pth formats.json \
  --numz-q4 /path/to/seedvr2_ema_7b-Q4_K_M.gguf               # every format on the 7B
uv run models/seedvr2_fp8.py $M --fp16 models/dist --out phase2
uv run models/seedvr2_gguf.py $M --numz /path/to/numz --out phase2
uv run models/seedvr2_int8.py $M --fp16 models/dist --out phase2
uv run models/seedvr2_nvfp4.py $M --fp16 models/dist --out phase2
CUDA_VISIBLE_DEVICES= uv run models/ck_check.py phase2/*fp8_scaled.safetensors phase2/*int8_convrot.safetensors \
  phase2/*nvfp4.safetensors
cd /path/to/numz && CUDA_VISIBLE_DEVICES= .venv/bin/python /path/to/models/numz_gguf_check.py . phase2/*.gguf
```
