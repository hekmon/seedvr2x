# Formats: what each one does to SeedVR2's weights

> Status: **measured on the CPU**, weights only, against ByteDance's fp32 masters:
> [`formats_study.py`](formats_study.py), [`numz_check.py`](numz_check.py), and each build
> script's own checks ([`seedvr2_fp8.py`](seedvr2_fp8.py), [`seedvr2_gguf.py`](seedvr2_gguf.py),
> [`seedvr2_int8.py`](seedvr2_int8.py), [`seedvr2_nvfp4.py`](seedvr2_nvfp4.py),
> [`seedvr2_gguf_dyn.py`](seedvr2_gguf_dyn.py)), with [`ck_check.py`](ck_check.py) and
> [`numz_gguf_check.py`](numz_gguf_check.py) loading the files in their runtimes. What each file
> does to the video was measured by GPU runs: [VALIDATION.md](VALIDATION.md).
> [DESIGN.md](../seedvr2x/DESIGN.md#weights) (Weights) cites these figures.

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
- **An importance matrix helps where the activations are large:** made with one (from GPU runs
  of the fp16 model), Q4_K is closer to the master than ours on all 288 matrices once each
  weight's error is weighted by the activations it meets (median 6.64% against 7.15%), though
  not by the plain error (7.55% against 7.35%); one type per matrix within the same bytes goes
  further (6.31%). On video, the 7B's dynamic file is the closest 4 GB file to the fp16 model
  ([VALIDATION.md](VALIDATION.md)).
- **Our phase-2 files** (the 7B and the sharp 7B): fp8 8.33 GB, Q4_K 4.76 GB, Q8_0 8.84 GB, INT8
  8.33 GB, NVFP4 4.76 GB, the dynamic GGUF and its control 4.76 GB, each loading in its runtime,
  made the same on every run.

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
  never changed). Every measurement of the 3B in models.md ran those first weights; on video, the
  current ones run beside them in [VALIDATION.md](VALIDATION.md#the-3b-at-1080p).

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
- **The dynamic GGUF** and its control, made with an importance matrix: [their own
  section](#dynamic-gguf-a-type-per-matrix-chosen-with-an-importance-matrix).

## Dynamic GGUF: a type per matrix, chosen with an importance matrix

> Status: **built and measured** (2026-10-07): the importance matrices from GPU runs of each
> model's fp16 file, the files on the CPU, then judged like the others on the GPU
> ([VALIDATION.md](VALIDATION.md)).

[`seedvr2_gguf_dyn.py`](seedvr2_gguf_dyn.py) makes, from the fp32 master and an importance matrix,
`seedvr2x_ema_7b_dyn.gguf` (one ggml type per block matrix, within our uniform Q4_K's bytes) and
`seedvr2x_ema_7b_Q4_K_imatrix.gguf` (every block matrix Q4_K with the same importance, our Q4_K's
size: the control that tells the importance's effect from the mix's); with `--model sharp`, the
sharp 7B's, from its own importance.

- **The importance matrix** ([`gpu/imatrix_hook.py`](gpu/imatrix_hook.py), llama.cpp's method):
  numz's own runs of the model's fp16 file; a forward pre-hook on each of the 288 block matrices
  adds, at every DiT forward, the input's sum of squares per input channel and its token count;
  the importance of channel k is the mean of x_k^2. Calibration: 4 clips, one per kind (anime,
  flat cel, live action, dark), none of tier 1's nor of the 7B's 4K shots, so that the evaluation
  never sees its calibration: colour_clips.py's d1 at a quarter of the size of four 4K shots, 45
  frames at 1080p, seed 42, the reference runs' settings. The hooks only read: numz's 7B gives
  the same output, bit for bit, with and without them (on the CPU; on the GPU, a control run's
  decode has the unhooked run's SHA-256).
- **The quantizer:** ggml's own `ggml_quantize_chunk` with the importance (llama.cpp `abeada3`,
  as our static files). Q3_K, Q4_K and Q5_K weigh each weight's squared error by
  imp_k sqrt(sigma^2 + w^2) in their search, Q6_K by imp_k; Q8_0 ignores it. A flat importance is
  not "no importance": ggml then runs its importance-aware search instead of its reference one;
  on the 7B, Q4_K so made is closer to the master than ours on all 288 matrices (median 7.283%
  against 7.345%).
- **The measure:** per matrix and type, besides the plain error, the error weighted by the
  activations, e = sqrt(sum_k imp_k ||dW[:, k]||^2 / sum_k imp_k ||W[:, k]||^2): with uncorrelated
  channels, the layer's output error relative to its output on the calibration. For a file made
  with an importance the plain error misleads: the importance moves error from the input channels
  the activations barely use to those they use most, and the plain error counts both the same.
- **The choice:** among Q3_K, Q4_K, Q5_K, Q6_K and Q8_0 (numz's loader also decodes Q2_K, Q4_0,
  Q4_1, Q5_0, Q5_1 and BF16: Q2_K is too coarse to trust a per-layer measure with; the four legacy
  types are never on a matrix's best trade-off curve), the types minimising the sum of e^2 within
  our uniform Q4_K's bytes for the 288 matrices, less 1 MiB for the metadata (the file is no
  larger than our Q4_K): a greedy over each matrix's lower convex hull of (bytes, e^2), the most
  error saved per byte first, ties broken by name. The last block's text attention output and MLP
  take the cheapest type: nothing reads their output (NaDiT keeps only the video tokens after the
  blocks; zeroed, numz's output is unchanged bit for bit).
- **Limits:** per-layer statistics: every layer's relative error weighs the same whatever it does
  to the video; errors taken as additive, channels as uncorrelated; 4 calibration clips. No
  end-to-end sensitivity: the GPU runs judge the file, against its control.
- **Checks, as for the static files:** ggml's and gguf-py's decoders agree on every tensor; each
  matrix's errors against the master beside our static Q4_K's (which the reference quantization
  reproduces byte for byte); the second quantization of each chosen matrix equal to the first;
  read back, with no path or time in the metadata; the size; numz's loader
  (`numz_gguf_check.py`).

| File | Bytes | SHA-256 | Error: median (worst) | Weighted: median (worst) | Q3_K / Q4_K / Q5_K | Quality ([VALIDATION.md](VALIDATION.md)) |
|---|---|---|---|---|---|---|
| `seedvr2x_ema_7b_dyn.gguf` | 4,757,055,264 | `f8c0c50d…` | 7.550% (18.26%) | 6.310% (11.67%) | 17 / 203 / 68 | the closest 4 GB file to the fp16 model (44.2 dB); past the calibrated line on the cartoon clip's DISTS; at 4K 44.3 dB, closer than another seed on all 6 shots and past the line nowhere |
| `seedvr2x_ema_7b_Q4_K_imatrix.gguf` | 4,758,308,480 | `2d9f7e60…` | 7.548% (8.831%) | 6.641% (7.370%) | 0 / 288 / 0 | 43.7 dB; past the line on live action's detail and the cartoon clip's DISTS; not run at 4K |
| `seedvr2x_ema_7b_Q4_K.gguf` (ours, static) | 4,758,307,552 | `7f4642d0…` | 7.345% (7.948%) | 7.154% (8.987%) | 0 / 288 / 0 | 42.9 dB; softer on some sources, within the line; not run at 4K |
| `seedvr2x_ema_7b_sharp_dyn.gguf` | 4,757,129,024 | `90ba80c8…` | 7.545% (18.31%) | 6.346% (12.05%) | 17 / 204 / 67 | 42.5 dB; past the line on cartoon and an anime clip's flicker and colour; at 4K 43.3 dB, closer than its control and Q4_K on all 5 shots and past the line nowhere: the card's 4 GB pick |
| `seedvr2x_ema_7b_sharp_Q4_K_imatrix.gguf` | 4,758,308,512 | `486a7da9…` | 7.537% (8.867%) | 6.651% (7.410%) | 0 / 288 / 0 | 42.1 dB; past the line on the cartoon clip's DISTS only; at 4K 42.8 dB, between the dynamic file and Q4_K on all 5 shots, past the line nowhere |
| `seedvr2x_ema_7b_sharp_Q4_K.gguf` (ours, static) | 4,758,307,584 | `a5e423a5…` | 7.347% (7.942%) | 7.161% (8.416%) | 0 / 288 / 0 | 41.4 dB; past the line on cartoon, live action and anime; at 4K 42.0 dB, adding detail to digital live action, past the line |

- **Weighted, the importance helps on every matrix:** the Q4_K made with it is closer than our
  static Q4_K on all 288 (median 6.64% against 7.15%), though further by the plain error on all
  but 20 (7.55% against 7.35%). The dynamic file goes further: 6.31% median, below the static
  Q4_K on 271 of 288; its sum of e^2 over the matrices whose output is read is 24% below the
  static Q4_K's and 11% below its control's (the sharp's: 22% and 10%).
- **Its worst errors sit where the measure says they cost least** (the 7B's): the 17 Q3_K
  matrices (the 3 of block 35 nothing reads, the text MLP's output projection of blocks 0 to 10
  but 3 and of block 34, block 3's text MLP input, block 0's video MLP) hold the worst plain
  error (18.3%) and weighted one (11.7%, block 35's; 11.4% the worst of a matrix whose output is
  read, block 0's video MLP input). The Q5_K bytes go to every video attention output projection
  and 32 of the 36 text ones. The sharp's choice differs on 5 matrices: block 0's video attention
  output projection in Q4_K (the 7B's Q5_K), block 1's video QKV in Q3_K (Q4_K), the text MLP's
  output projection of blocks 9 and 10 in Q4_K (Q3_K) and of block 16 in Q3_K (Q4_K).
- **The importance files** (`seedvr2_ema_7b_fp16.imatrix.safetensors`, 12,454,888 bytes,
  `f2283e03…`; the sharp's, 12,454,944 bytes, `2c0dedc3…`) are `seedvr2_gguf_dyn.py`'s pinned
  inputs, uploaded with the files made from them (`hekmon/seedvr2x` at `c14a2bc4`): without
  `--imatrix` the script takes them from there, size and SHA-256 checked, and they key its pinned
  outputs. Their own metadata hold the fp16 file by name and size and each run's
  settings, numz's arguments with each path cut to its base name, with no path and no time, in a
  fixed layout: the same runs merged give the same bytes. The GGUF files' metadata name the
  importance file by its name and SHA-256 and the calibration clips by name, with no path: the
  same importance file, under its name, gives the same bytes wherever it is read from, and a
  second build gave them.
- 1.25 MB smaller than our Q4_K (the 7B's; the sharp's 1.18 MB), the dynamic file loads in
  numz's GGUF loader as it is: numz decodes every K-quant type.

## How each format runs

The users' table, as the card gives it, the quality from the GPU runs
([VALIDATION.md](VALIDATION.md): tier 1, 8 clips at 1080p, each file against its own model's fp16
output at the same seed, judged against how much the fp16 model's own seeds differ; "as close as
another seed": no score past 2.7 times the spread of 3 seeds or [its
floor](VALIDATION.md#the-guards-and-two-rules), a line a further seed of the fp16 model crosses on
one score in twenty). "W8A8": weights and activations in 8 bits, the multiply in 8 bits; "W8A16":
weights stored in 8 bits, widened to 16 for the multiply (memory only).

| Format | Size (7B) | How it multiplies | Multiply rate against 16 bits | Error per weight | Quality |
|---|---|---|---|---|---|
| float16 | 16.5 GB | 16-bit: bf16 accumulated in fp32 (fp16 on RTX 20) | 1x, the reference | 0.02% | the reference |
| fp8, a scale per tensor | 8.3 GB | W8A8 from RTX 40 (Ada, sm_89); W8A16 before | 2x on RTX 40 and 50 (RTX 50: 2.3-2.9x measured by others) and on workstation Ada and Blackwell cards; 1x before RTX 40 | 2.65% | as close as another seed or closer, W8A8 and W8A16, both models |
| int8, rotated | 8.3 GB | W8A8 from RTX 20 (Turing, sm_75) | 4x on GeForce RTX 20 to 50; 2x on workstation cards | 0.86% | as close as another seed or closer |
| GGUF Q8_0 | 8.8 GB | W8A16 | 1x: memory only | 0.56% | as close as another seed or closer: the closest of all |
| GGUF Q4_K | 4.8 GB | W4A16 | 1x: memory only, a little slower | 7.35%; weighted 7.15% | the 7B: softer on some sources, within the line; the sharp 7B: further than a seed on cartoon, live action and anime |
| GGUF Q4_K, with an importance matrix | 4.8 GB | W4A16 | 1x: memory only, a little slower | 7.55%; weighted 6.64% | further than a seed on the cartoon clip's DISTS (both models), live action's fine detail (the 7B) |
| GGUF, a type per matrix (dynamic) | 4.8 GB | W4A16 (3 to 5 bits per matrix) | 1x: memory only, a little slower | 7.55%; weighted 6.31% | the 7B: the closest 4 GB file, further than a seed on the cartoon clip's DISTS only; the sharp 7B: further on cartoon and anime |
| NVFP4, searched scales | 4.8 GB | W4A4 on Blackwell (sm_100, sm_120); W4A16 before | 8x on RTX 50; 4x on workstation Blackwell cards; 1x before Blackwell | 8.80% | W4A4: clearly worse on DISTS (cartoon, anime), both models; W4A16: softer on live action (the 7B), further than a seed on anime and cartoon (the sharp 7B) |

- The rate column gives the peak rate of the multiply the DiT's matrices use, by NVIDIA's own
  tables, against the 16-bit multiply on the same card: Turing (TU102), Ampere (GA102), Ada
  (AD102) and RTX Blackwell (GB202, GeForce and PRO) whitepapers, each card family's dense rates.
  - GeForce cards run a multiply accumulated in fp32 at half rate when its inputs are 16-bit
    floats or fp8, but integer and 4-bit multiplies at full rate. Workstation cards (Quadro RTX
    6000/8000, RTX A6000, RTX 6000 Ada, RTX PRO 6000) and the TITAN RTX halve nothing.
  - So int8's multiply is 4 times the 16-bit one on a GeForce card, twice fp8's on an RTX 40. On a
    workstation card both 8-bit multiplies are twice the 16-bit one.
  - RTX 20 has no bf16 tensor cores: its 16-bit reference is fp16.
  - RTX 50's fp8 at 2x is NVIDIA's table. cuBLAS, which runs it, uses Blackwell's block-scaled
    instruction, which GeForce doesn't halve: others measured 2.3-2.9x the bf16 rate.
  - fp8 and NVFP4 accumulate in fp32: cuBLAS offers them no other way.
- A peak is not a job's speed:
  - each 8- or 4-bit multiply first rounds its input, an extra pass over the activations;
  - attention stays in 16 bits;
  - the DiT is about a fifth of a 1080p job (the VAE, in 16 bits, is the rest).

  By our estimate, matrices multiplied 3 to 4 times faster make the DiT about twice as fast, and
  a job about 10% shorter. The main gain is memory. No speed is measured here: the GPU box has a
  power-cap fault, until a healthy card measures it.
- An 8- or 4-bit multiply rounds each layer's input too: a second loss the error per weight
  doesn't show. The GPU runs measured it on real activations: a layer's output moves by 2.09% in
  fp8 W8A8 against 1.50% in W8A16, by 7.18% in NVFP4 W4A4 against 4.93% in W4A16, by 0.76% in
  int8 (medians, [VALIDATION.md](VALIDATION.md#per-layer-on-real-activations-the-smoke-test-the-7b)).
- **comfy-kitchen 0.2.37 rounds a layer's input wrongly once it holds 2^32 values or more**
  (its CUDA quantizers index in 32 bits): fp8 W8A8 then gives an all-NaN video, NVFP4 W4A4 a
  silently wrong one; int8, W8A16 and W4A16 are unaffected. In the 7B only the MLP's output
  projection gets there, from 349,526 video tokens in one forward: at 3840×2160, a batch of 41
  frames or more. [`gpu/ck_patch.py`](gpu/ck_patch.py) quantizes such an input in row chunks at
  the scale comfy-kitchen gives the whole tensor, comfy-kitchen's values bit for bit; a runtime
  calling comfy-kitchen must do the same, or stay below the limit
  ([VALIDATION.md](VALIDATION.md#fp8-w8a8-at-4k-comfy-kitchens-32-bit-indices); the report for
  comfy-kitchen, not filed yet:
  [research/bugs](../research/bugs/comfy-kitchen-32-bit-quantizer-indices.md)).

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
uv run models/seedvr2_gguf_dyn.py $M --static phase2/seedvr2x_ema_7b_Q4_K.gguf \
  --out phase2-dyn   # the importance: ours, fetched from hekmon/seedvr2x at c14a2bc4 into --cache and checked
uv run models/seedvr2_gguf_dyn.py --model sharp $M --static phase2/seedvr2x_ema_7b_sharp_Q4_K.gguf --out phase2-dyn
CUDA_VISIBLE_DEVICES= uv run models/ck_check.py phase2/*fp8_scaled.safetensors phase2/*int8_convrot.safetensors \
  phase2/*nvfp4.safetensors
cd /path/to/numz && CUDA_VISIBLE_DEVICES= .venv/bin/python /path/to/models/numz_gguf_check.py . phase2/*.gguf \
  phase2-dyn/*.gguf
```
