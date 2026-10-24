# torch.compile on seedvr2x: what it gives, what it costs, what it breaks

> Status: **measured on an RTX 5090 only** (2026-10-23 and 24) for the open question
> "torch.compile, undecided" of [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). seedvr2x at
> main 7220e88, its code unchanged since (the commits after it touch docs only); its own runtime
> imported by research harnesses that install the compiler at import time, nothing of seedvr2x
> edited. One GPU: GeForce RTX 5090 32 GB under WSL2 (a desktop on it: 27.2 to 30.2 GiB free),
> torch 2.14.1+cu130, triton 3.8.0, PyTorch's SDPA attention (FlashAttention 2 not installed:
> what most machines run, and the case where attention is compiled with the rest), torch's
> allocator with expandable segments as seedvr2x sets it there, the 7B fp16 DiT (the sharp one in
> whole runs) and the fp16 VAE. Times on this card are healthy (no throttling; its clocks may
> run above stock): every time below is its own, and its ratios. The parts that
> need more than 32 GB (the VAE untiled at 1080p and on 4K's large tiles, the DiT's windows above
> 22 latents, the bits there) were to run on the 96 GB card, which failed on 2026-10-23 before
> running any of them: [what is left](#whats-left).
>
> numz's own figures ([vram.md](vram.md#torchcompile)) are numz's flow; the first study of
> 2026-10-10 on seedvr2x 0db0a72 (the throttled 96 GB card, never reported) is superseded here,
> but for the [mechanism](#the-vae-three-ways-to-compile-it) it found.

In short:

- **The DiT, compiled block by block, runs its windows 20 to 25% faster from 2 latents (12% at
  1), over seedvr2x's own BlockSwap too; in the harness, for plain code's memory up to 19 latents
  at 1080p, but in a job a new length's first call took 4.9 GiB more.** Each new window length
  costs a compilation, 40 to 80 s at 1080p and 75 to 150 s at 4K, 9 to 13 s from a warm cache. Compiling
  the whole DiT, numz's way, fails over seedvr2x's swap; dynamic shapes cost 19 minutes and run
  slower than plain code.
- **The VAE is the larger lever, 80% of a 1080p job on this card: compiled, its passes run 15 to
  23% faster** (19 to 26% against the same process's plain passes), but each way of compiling it costs memory the planner doesn't know: with its
  causal convolutions out of the graph (numz's intent), 0.9 to 2.2 GiB more per pass; with them
  in the graph (the faster way), 0.5 to 2.3 GiB more once compiled and 1.6 to 8.2 GiB more on the
  first call of each tile's shape (up to 1.6 times plain's peak), which a 32 GB card's own plans
  don't hold.
- **Compiled output is not plain output, but it is 14 to 17.5 dB closer to it than another seed:**
  on 8 full-reference clips, the compiled DiT's frames are 51.1 to 61.3 dB (PSNR-Y) from the plain
  ones, another seed's 35.6 to 45.3 dB; against the ground truth, compiled moves PSNR-Y by 0.07 dB
  at most, another seed by up to 0.67.
- **The VAE's `excl` compile passes design's checks for v0.1.0:** bit for bit from one cold
  process to another and through a stop and resume, a whole job planned at 1.1 times plain's
  peaks fitting and its units 10% faster than plain code's own plan, its compile workers 4.4 GiB
  of host memory (none in a single thread, for 20% more compile time).
- **Two processes compiling cold give different bits, unless inductor's deterministic mode is
  really on.** Triton picks some kernels' block sizes by timing them, and a reduction's block size
  changes its sums; torch 2.14.1 turns the mode that prevents it back off at every trace.
  Kept on, cold processes and warm caches agree bit for bit, at no measured cost.
- **A long job gains, over the 12 real cut lists, 10.5 to 15.1% of its DiT time at the 5090's
  1080p cap** (modelled from the measured costs), 2 to 3% of the job (the DiT a fifth of it); the VAE compiled would add 13 to 17%.
  **A short one loses:** a 45-s excerpt compiled spent about four times plain code's DiT time,
  every new window length 68 to 85 s of compilation.
- **What it breaks:** the planner's peaks (a compiled window of 20 latents or more ran out of
  memory at its first call where plain code fits; the VAE's first calls), the resume rule without
  the deterministic mode, the environment's record (compiling imports three packages after it);
  the calibration takes every compilation of its shapes and the job's units none.

## How it was measured

The harnesses import seedvr2x's runtime, load the models as its command line does, and run its
own code paths:

- [`scripts/compile_dit.py`](../scripts/compile_dit.py): one window's step as the pipeline makes
  it (seedvr2x's noise draw, condition and Euler step, `model.sample`), at a size and a length in
  latents, with seedvr2x's BlockSwap (`--swap N`, a Plan with N blocks swapped, the hooks of
  `runtime/blockswap.py`); plain and compiled steps in turn in one process, so that times compare
  as ratios of neighbours. Per step: GPU time, torch's peak allocated, the compilations (frames,
  seconds, recompilations and graph breaks with their reasons, cache entries per code object) and
  the output's SHA-256.
- [`scripts/compile_vae.py`](../scripts/compile_vae.py): seedvr2x's encode and decode of a clip's
  frames, untiled or tiled (`runtime/tiling.py`), the latent kept in system memory as the pipeline
  keeps it; passes plain (A) and compiled (B) in turn, the first A a warm-up (the first plain pass
  also sets up cuDNN). Per pass: time per frame, the peak above what is allocated before it (the
  VAE's 0.47 GiB of weights), per tile, the compiling pass's own peak, what is left allocated, the
  compilations, the latent's and the frames' SHA-256 and their distance from plain.
- [`scripts/compile_run.py`](../scripts/compile_run.py): seedvr2x's own command line with the
  compiler installed right after the models load (`model.load_models`), every unit recorded
  (`run._started` / `run._made`: wall time, peak, compilations, seedvr2x's own estimate), the
  calibration's steps too, and `--record-flags` setting the plan's `compile_dit` /
  `compile_vae` record. With nothing compiled it is a no-op: the same output, bit for bit, as
  seedvr2x alone (checked: identical CRC-32 files).
- [`scripts/compile_bits.py`](../scripts/compile_bits.py): runs' outputs against each other and
  against the clips' ground truth.
- [`scripts/compile_layouts.py`](../scripts/compile_layouts.py): a job's gain from real cut lists
  (12 sources, 580,642 frames, 6,829 shots, detected at seedvr2x's 0.3: the script's own `cuts`
  from the shot detector's probabilities, checked against seedvr2x's `runtime/cuts.py` and
  `runtime/shot.py` at 7220e88, identical on 12 of 12 sources and 4,326 length and cap pairs) and
  a cost model of measured figures.

Peaks: torch's peak allocated memory, the whole process's in the DiT's tables (the weights the
plan keeps on the GPU included), above what is allocated before the pass in the VAE's (its 0.47
GiB of weights left out). Ratios: the median of each compiled step's or pass's time over its
plain neighbours', in one process.

Compile settings throughout: inductor, mode `default`, static shapes (dynamic shapes measured
apart), torch's recompile limit raised to 64 (128 for the VAE, numz's) unless said, a cold
Inductor and Triton cache per process unless said "warm" (the same cache directories again).

## The DiT

### Which code to compile

seedvr2x's DiT has 36 transformer blocks of one class. Two ways:

- **`whole`** (numz's `--compile_dit`): `torch.compile` on the DiT module. Without BlockSwap it
  runs (0.92 to 0.99 of plain at 540p, the same graph breaks as below). **Over seedvr2x's swap it
  fails**: the swap's hooks, which record CUDA events on its side stream before and after each
  swapped block, are traced into the compiled code, and dynamo raises ("An event was recorded on
  a stream where a graph in…"), 1080p, 35 blocks swapped. numz's compiled DiT over numz's own swap
  ran but gained nothing ([vram.md](vram.md#torchcompile)), its graph broken at every block.
- **`blocks`**: each block's own `forward` compiled (the class's `forward` compiled once and bound
  to each block), the rest of the DiT plain. BlockSwap's hooks run in the module's call, outside
  the compiled code, so the swap is untouched: over 20 swapped blocks at 540p, the same
  compilations and graph breaks as unswapped, the plain output swapped equal to the plain output
  unswapped bit for bit. **One compiled graph serves the 36 blocks**: the block's code object holds
  one cache entry per window length, not 36 (torch 2.14 lifts each block's parameters as the
  graph's inputs), which is also why the swap's parameters, rebound to the slots' memory
  (`Part.bind`), recompile nothing.

Inside a block the graph breaks where seedvr2x's DiT reads tensors' values into Python: the window
partition (`na.py` unflatten, a data-dependent `.item()`), the modulation's `repeat_interleave`
by the text's length, the rotary frequencies (`@torch._dynamo.disable`), and FlashAttention 2's
call where it is installed (`@torch._dynamo.disable`); SDPA is traced. About 45 to 90 frames are
compiled per length (75 at 540p, 57 to 93 at 1080p), 23 to 54 graph breaks.

### Speed and compilation, by window length

1080p (padded 1104x1920, 8,280 tokens a latent), 35 of 36 blocks swapped (the 5090's 1080p plan
with 27.2 GiB free), SDPA, `blocks`. Plain and compiled: the median of two steps each, one process.
The compilation: the first compiled call less a steady compiled step, cold (a new cache) and warm
(a second process on the first one's cache).

| Latents | Plain s | Compiled s | Ratio | Cold compile s | Warm compile s | Peak plain / compiled GiB |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.955 | 0.844 | 0.884 | 40 | 9.8 | 3.09 / 3.18 |
| 2 | 1.599 | 1.287 | 0.805 | 46 | 8.7 | 4.23 / 4.23 |
| 3 | 2.272 | 1.813 | 0.798 | 60 | 9.5 | 5.37 / 5.37 |
| 5 | 3.582 | 2.824 | 0.788 | 58 | 9.3 | 7.47 / 7.56 |
| 8 | 5.597 | 4.341 | 0.776 | 81 | 12.0 | 10.80 / 10.80 |
| 13 | 8.756 | 6.726 | 0.768 | 79 | 10.8 | 15.96 / 16.05 |
| 19 | 12.725 | 9.770 | 0.768 | 79 | – | 22.35 / 22.44 |

With all 36 blocks swapped (the plan with 30 GiB free), 19 latents: 15.16 s plain, 11.43
compiled (0.754), 98 s of compilation, peaks 21.93 / 22.02 GiB. With inductor's deterministic
mode kept on ([Bits](#bits)), 5 and 13 latents: ratios 0.760 to 0.785, cold compilations 68 and
77 s, warm 12.7 and 10.5 s (0.758 and 0.768): the same.

At 4K (2160x3840), 33 blocks swapped (the 5090's 4K plan, cap 5 latents): ratios 0.768, 0.781,
0.771 and 0.754 at 1, 2, 3 and 5 latents (plain 2.99 to 13.24 s), cold compilations 75, 105, 152
and 148 s, peaks 7.26 to 24.53 GiB plain, compiled the same within 0.1 but at 1 latent (8.02 for
7.26).

The gain grows with the window and settles at 22 to 25% from 8 latents. The compilation is CPU
work (Inductor's and Triton's, with up to 32 worker processes: they held up to 16.2 GiB of host memory
by their summed resident sizes, beside the main process), and grows with the window too.

### The memory, and where it jumps

In the harness, up to 19 latents at 1080p, a compiled window takes plain code's memory: its first
call 0.09 GiB more at most, its steady steps none (cold, or warm with the deterministic mode;
without it a warm process ran out of memory at 19). **From 20 latents its first call doesn't fit where plain code
does:** at 1080p with every block swapped, plain code peaks at 22.98 GiB at 20 latents and 25.08
at 22, while the compiled first call ran out of memory at both, the card having 28.2 GiB free by
NVML (torch's own figure 30.2; the harness, torch's allocator bounded by the card alone): over 5
GiB more at 20 latents, over 3.1 at 22. seedvr2x's own compiled run of a 90-s excerpt, planned at
22 latents, ran out of memory in its calibration's 22-latent window (29.47 GiB allocated and 2.09
asked, 29.57 allowed by seedvr2x's hold), where its plain run peaked at 24.61. And a job planned
near its budget met it below 19 latents: a new length's first call in the middle of a compiled job
(below, [a short job](#a-short-job-measured)) went from 4.02 GiB allocated to 29.34, where the
shot's 19-latent windows had peaked at 24.42. The step from 19 to 20 latents in the harness is not a slope: the cause is not isolated (FlashAttention
2, outside the graph, untested; the windows' attention layout changes with the window's length,
`window.py`).

Without the deterministic mode, a warm cache's first call took more than a cold one's at some
lengths (10.78 GiB at 5 latents for 7.47 plain); with it, none did.

### Recompilations and torch's limit

Each new window length is a recompilation of the block's graph and of the small functions around
it. torch's default limit, 8 per code object (`torch._dynamo.config.recompile_limit`), is reached
silently: past it, the code runs plain, with a warning. Lengths 1 to 12 in turn (270x480, the
default limit): the modulation's and the projections' functions stop being compiled from the
second length (the modulation held 8 entries after the first: video and text, dtypes, guards),
the windowed attention's from the fifth, the block itself from the ninth: from 9 latents on, nothing is
compiled, and windows run at plain speed. A new length cost 16 to 18 s of compilation at 2 to 4
latents, 2.7 to 2.8 s at 5 to 8 (only the block left to compile), 0 after. seedvr2x's own run with the
default limit hit it twice in its calibration alone. **A compiled seedvr2x must raise the limit
well above the number of lengths it compiles**, and say so: 64 was not reached by jobs of about
ten lengths.

### Dynamic shapes

`dynamic=True`, the promise of one compilation for every length: at 1080p over 35 swapped
blocks, the first compiled call at 5 latents took 1,163 s, its steps then ran at 1.05 of plain
(slower), and 8 latents compiled again (stopped after 9 minutes). At 270x480, every length 1 to 6
and 8 compiled again (49 to 71 s each for 1 to 4), the windowed attention hit the limit at 5, and
steps ran at 1.0 to 1.29 of plain. The window partition's Python integers (`window.py`) make
every length a new graph anyway. **Static shapes, one compilation per length.**

## The VAE: three ways to compile it

seedvr2x's VAE runs its causal 3D convolutions through `memory_limit_conv`
(`vendor/models/video_vae_v3/modules/causal_inflation_lib.py`), which computes each call's size
as a tensor and branches on its value: a data-dependent branch that breaks a compiled graph at
every convolution call (62 convolutions, a few calls each per slice). Three variants
(`compile_vae.py --variant`), each the encoder and the decoder compiled with numz's settings:

- **`numz`**: the code as it stands, numz's `--compile_vae`. Dynamo compiles hundreds of frames
  per pass and hits the recompile limit (128) on `memory_limit_conv` (281 graph breaks and 327
  graphs on a small random VAE, the CPU check).
- **`excl`**: numz's intent ([bug 15](../bugs/15-compile-vae-exclusion-dead-code.md)): every
  `InflatedCausalConv3d.forward` behind `torch.compiler.disable`, the convolutions plain, the norms,
  activations and arithmetic between them compiled: 500 to 800 compilations per run, 250 to 330
  graph breaks.
- **`int`**: the convolutions in the graph: `memory_limit_conv`'s size arithmetic done on Python
  integers (float32 rounding kept), the same splits and the same one-output path. Checked on the
  CPU against the original, bit for bit, on 756 cases of a real convolution (splits along H and
  W, caches, every memory state, above and below `ONE_OUTPUT_FROM`) and 30,000 size cases
  (`compile_vae.py selftest`): 0 differences. Compiled, the encoder and the decoder are 4 graphs
  with no break (fullgraph holds), 16 to 32 compilations a run (each tile shape and slice kind).

1080p (21 frames, 5 then 4 a slice), a clip's frames; peaks above what is allocated before a pass
(the VAE's weights), GiB; time per frame, the plain passes' median → the compiled ones', and in
parentheses the median of each compiled pass's ratio to its plain neighbours (the two can differ,
plain passes drifting within a process).

| Variant | Tiles enc / dec px | Enc peak plain / compiled / first call | Dec peak plain / compiled / first call | Enc s/frame plain → compiled | Dec s/frame plain → compiled | Compile s |
|---|---|---|---|---|---|---:|
| excl | untiled / 1088 | 19.38 / 21.37 / 21.37 | 19.65 / 20.55 / 20.55 | 0.202 → 0.162 (0.81) | 0.444 → 0.374 (0.84) | 224 |
| excl | 1856 / 1088 | 19.08 / 21.00 / 21.00 | 19.65 / 20.55 / 20.55 | 0.215 → 0.170 (0.79) | 0.442 → 0.376 (0.85) | 258 |
| excl | 1024 / 1024 | 10.51 / 11.46 / 11.46 | 17.45 / 18.52 / 18.52 | 0.230 → 0.193 (0.86) | 0.437 → 0.437 (0.95) | 210 |
| int | 512 / 512 | 3.55 / 4.21 / 5.16 | 5.89 / 8.23 / 8.75 | 0.275 → 0.237 (0.85) | 0.563 → 0.474 (0.84) | 260 |
| int | 640 / 640 | 4.95 / 6.03 / 7.51 | 8.09 / 9.91 / 12.63 | 0.248 → 0.207 (0.84) | 0.514 → 0.440 (0.86) | 312 |
| int | 768 / 768 | 7.11 / 8.11 / 10.24 | 11.10 / 12.48 / 17.22 | 0.204 → 0.165 (0.81) | 0.432 → 0.347 (0.80) | 344 |
| int | 896 / 896 | 8.59 / 9.11 / 12.02 | 14.10 / 15.81 / 22.27 | 0.244 → 0.172 (0.74) | 0.494 → 0.385 (0.84) | 339 |
| int | 1024 / 1024, 1856 / 1088, untiled / 1088 | 10.51, 19.08, 19.38 / out of memory | 17.45, 19.65 / out of memory | – | – | – |
| numz | untiled / 1088 | 19.38 / out of memory | – | – | – | – |

At 4K (2160x3840, the 5090's 4K plan: encode tiles 1536, decode 1152), `excl`: encode 21.53 /
23.75 GiB, decode 21.99 / 22.85, 0.781 → 0.651 s a frame (0.83) and 1.738 → 1.454 (0.83), 362 s
of compilation; `int` out of memory.

- **`int` is the fast way**: at 768 px 0.81 and 0.80 of the same process's plain passes, its
  passes 0.165 and 0.347 s a frame where plain code took 0.215 and 0.442 on the 32 GB plan's
  larger tiles in another process (−23% and −21%; plain times move by up to 15% from one process
  to another): plain code is no slower at 768 px than at 1856 or 1088 (0.204 and 0.432). Its cost
  is **its first call at each tile's shape**: 3.1 GiB over plain in the encode and 6.1 in the
  decode at 768 px (1.6 to 8.2 across 512 to 896 px), its later calls 1.0 and 1.4 (0.5 to 2.3). That first
  call is not the compiler's tuning: the same with a warm cache, with the deterministic mode (no
  benchmarking of reductions), without Triton's pointwise autotuning (then 20% slower), without
  inductor's layout optimisation. `excl` has no such first call (its convolutions run plain). The
  cause is not isolated: it is the measure numz's VAE's doubling ([bug
  14](../bugs/14-compile-vae-doubles-memory.md)) stood for, set as numz meant.
- **`excl` costs 0.9 to 2.2 GiB more than plain on every call**, for 5 to 21% less time.
- Nothing stays allocated after a compiled pass (0.000 GiB above the weights in every run, plain or
  compiled): the compiled VAE leaves nothing behind for the DiT's phase, where numz's left 5.5 GiB.
- **A fit the planner can use** (`int` with the deterministic mode, 512 to 896 px, and `excl`, 1024
  px to untiled and 4K's 1536 and 1152): the compiled pass's peak above the weights against the
  same tile's plain peak, the planner's own table (`MEASURED` in `runtime/planner.py`). `int`'s
  first call 1.45, 1.52, 1.44 and 1.40 times the plain encode, 1.49, 1.56, 1.55 and 1.58 times the
  plain decode, its later calls 1.06 to 1.22 and 1.12 to 1.40: **a compiled `int` pass is held to
  1.6 times its plain peak** (the decode's ratio grows with the tile, to be measured beyond 896
  px). `excl`'s calls, the first among them, 1.09 to 1.10 times the plain encode and 1.04 to 1.06
  times the plain decode: **1.1 times**. Over 21 frames, as the planner's table; one clip.
- Small tiles cost plain code time too (0.275 and 0.563 s a frame at 512 px, 0.215 and 0.442 at
  the plan's), so `int` gains on the plan's own tiles only from about 768 px: at 640 px it is
  the plan's speed.
- Neither fits a 32 GB card's own plans: the 5090's 1080p plan encodes untiled 0.16 GiB under its
  budget, its 4K plan 0.06.

## Bits

Compiled kernels round otherwise than plain code: fused operations keep intermediates in
registers, reductions sum in another order. Compiled output is never plain output, bit for bit.

### How far, against another seed

seedvr2x's own runs (`compile_run.py`) on the 8 full-reference clips (45 frames each, 960x540
upscaled to 1920x1080, one shot, one window of 12 latents, the sharp 7B, `split`, gbrp16le
masters), on the 5090's plan: plain with seed 42 and 43, and the DiT compiled (`blocks`, the
deterministic mode on) with seed 42. `compile_bits.py`: PSNR-Y between two runs' frames, the mean
absolute difference in 8-bit levels, and each run's PSNR-Y against the ground truth.

| Clip | Seed 42 / 43: PSNR-Y dB, mean levels | Compiled / plain: PSNR-Y dB, mean levels | Ground truth PSNR-Y: plain 42, plain 43, compiled 42 |
|---|---|---|---|
| anime-bright | 35.61, 1.49 | 51.14, 0.30 | 24.68, 24.85, 24.66 |
| anime-clean | 37.00, 2.04 | 53.46, 0.31 | 28.51, 29.18, 28.57 |
| anime-dark | 45.29, 0.99 | 59.66, 0.21 | 36.91, 37.21, 36.93 |
| anime-grain | 40.17, 1.75 | 56.77, 0.26 | 30.38, 30.68, 30.34 |
| anime-sky | 45.06, 1.05 | 61.30, 0.19 | 41.12, 41.15, 41.11 |
| cartoon-bright | 40.62, 0.97 | 54.68, 0.24 | 28.65, 28.69, 28.61 |
| live-slow | 40.75, 1.25 | 55.99, 0.23 | 30.28, 30.43, 30.27 |
| live-vfx | 43.06, 0.98 | 60.56, 0.16 | 37.67, 37.29, 37.66 |

The compiled run is 14.1 to 17.5 dB closer to the plain one than another seed is, its mean
difference a sixth of a seed's; but not everywhere small: its largest sample difference reaches 22
to 126 levels (a seed's 66 to 255), the diffusion step carrying a rounding where the picture is
unstable. Against the ground truth, it moves PSNR-Y by −0.05 to +0.07 dB from plain code, where
another seed moves it −0.38 to +0.67.

The VAE alone, compiled against plain on the same latent (`compile_vae.py`, 1080p and 4K, any
variant): the frames 59.0 to 61.8 dB apart, 0.06 to 0.12 levels on average, 2 to 9 levels at
most; the latent 0.10 to 0.14 at most.

### From one process to another

**Two processes compiling the same model cold give different bits** (the DiT at 540p and at
1080p, with or without the environment variable below; the VAE at 1080p, in its `excl` form, the
one pair run without the mode, no `int` pair; with Python's hash seed fixed, at one length of
two), while a process repeats its own compiled output exactly and a
warm cache reproduces the process that filled it. The generated kernels are the same code; the
difference is Triton's runtime autotuning: it times a few block sizes for each kernel on its
first call and keeps the fastest, a choice that moves with the timing's noise (22 of 68 kernels
differed between two processes at 540p: 21 pointwise ones, which still differ with the mode below
on and leave the bits alone, and the reductions that changed the sums, a persistent reduction's
block of 1 or 8 rows and a looped one's of 4096 or 1024), and for a reduction the block size is
the order of its sums. Inductor's deterministic mode (`TORCHINDUCTOR_DETERMINISTIC=1`) exists for this: reductions
get one fixed configuration, no timing. **In torch 2.14.1 it is undone at the first trace**:
dynamo ends every trace with `torch.use_deterministic_algorithms(<the prior global state>)`
(`torch/_dynamo/convert_frame.py:375`), and that call sets inductor's flag too
(`torch/__init__.py:1902`), so the environment variable's `True` is reset to `False` before
inductor compiles anything (every generated kernel's metadata said `deterministic: False`).

Kept on (`compile_dit.keep_deterministic`: the flag set again after each of those calls; eager
code untouched), **cold processes, a warm cache and every repeat agree bit for bit**: the DiT at
540p (one length) and at 1080p over 35 swapped blocks (5 and 13 latents, 2 cold processes and a
warm one), the VAE at 1080p in 768-px tiles (2 cold, 1 warm), and seedvr2x's own compiled runs
of 2 clips, each in two processes whose caches two cold compilations had filled; and a whole
compiled job of a 45-s excerpt (14 shots), its plan pinned (windows of 13 latents, decode tiles of
1216 px, 29 blocks swapped): the first segment (157 frames) that a second process made, its cache
cold, equals the first process's bit for bit. Unpinned, two processes of the same job planned
other decode tiles (1280 and 1216 px, the card's free memory having moved by 0.23 GiB between
them): the plans differed, so that pair doesn't test it; a resume takes its record's tiles. Its cost: none measured (the DiT's ratios
0.76 to 0.79 as without it, the VAE's compiled passes 0.165 and 0.347 s a frame as without it).

## What a job gains

The DiT's windows per length on the 12 sources' cut lists (seedvr2x's balanced layout,
`compile_layouts.py`), costed with the figures above: 0.083 ms a token plain (1080p, the steady
lengths), the gain per length measured (11.6% at 1 latent to 23.2% from 13), a cold compilation of
48 + 2 x L seconds, a warm one of 9 + 0.15 x L, at the 5090's 1080p cap of 19 latents.
`every` compiles each length a job meets, `pay` only those whose windows save more than their
compilation; both against the plain DiT's time.

| Strategy | One process | Three processes (a job over three nights) | torch's limit at 8 |
|---|---:|---:|---:|
| every | +10.5% | +7.2% | +9.4% |
| pay | +15.1% | +12.2% | +14.6% |
| dynamic shapes | −17.5% | −23.9% | −17.5% |

The dynamic row is its best case, one 1,163-s compilation for every length (measured, lengths
compiled again); the limit's column counts lengths, not the smaller functions that stop compiling
earlier ([above](#recompilations-and-torchs-limit)). The other layouts (the same table,
`compile_layouts.py`): full windows with a short last one, every +9.8% and pay +15.3%; full
windows with the last slid back, every −1.6%, pay +4.7%, and −16.3% plain for the latents it
computes twice: no layout makes compiling pay more than seedvr2x's balanced one, whose lengths
it doesn't reduce.

A length pays from 28 windows at 19 latents (cold; 4 warm) to 194 at 2 (35 warm). On a real job
the DiT is a fifth of the time: seedvr2x's plain run of a 90-s excerpt (one run, at cap 22 where
the table takes 19) (2,158 frames, 20 shots, at
1080p on the 5090's plan, 22 latents at most, every block swapped) spent 25.6% of its units'
time in encodes, 20.2% in windows and 54.2% in decodes. **So the DiT compiled gains 2 to 3% of
a 1080p job on this card**; `--quality enhanced` and `maximum` run every unit 2 and 4 times, the
DiT's share unchanged, each length's compilation paid once for 2 or 4 times its windows.
**The VAE compiled would gain 13 to 17%** (`excl` at the plan's tiles: −20% of the encodes and
−15% of the decodes; `int` at 768 px against the plan's tiles: −23% and −21%), for a few minutes
of compilation a process against the many hours a long job runs (that excerpt's 2,158 frames took
32 minutes of units: an hour of 1080p video, about 21 hours on this card).

### A short job, measured

seedvr2x's own runs of a 45-s excerpt of the same source (1,079 frames, 14 shots, 540p to 1080p,
the window pinned to 19 latents, the DiT's first call at 22 not fitting), plain and with the DiT
compiled (`blocks`, the deterministic mode, cold caches):

| | Plain | Compiled |
|---|---:|---:|
| Units, encodes / windows / decodes, s | 282 / 235 / 616 | 281 / 1,029 / 591 |
| of which compiling, in the windows | 0 | 822 |
| The calibration's steps, s (of which compiling) | 60 (0) | 406 (286), two processes |
| The window's estimate against its time: median, largest | −0.3%, +8.7% | +0.3%, +1,380% |

The windows' own work took 12% less, but the job met about ten lengths its calibration hadn't
compiled, 68 to 85 s each, and a stop and resume compiled them again in the new process (its cache
fresh, as the test meant): **the compiled short job took about four times the plain one's DiT
time** (1,029 / 235 = 4.4 as run; (1,029 − 79) / 235 = 4.0 without the 79 s the resumed process
spent compiling again the one length, 14 latents, that the stopped one had compiled).
A unit that compiled took 68 to 85 s more than its own work, up to 15 times its estimate for a short window; the encodes' and decodes' estimates
held as without compile (within 5%). A first run of the same compiled job ran out of memory in
the middle, at a new length's first call (shot 13's third window: 75 compilations, 29.34 GiB
reached, 27.70 allocated and 3.41 asked against 29.48 allowed, where the shot's two 19-latent
windows peaked at 24.42): seedvr2x stopped there, 3 segments of 4 made, and resumed by the same
command.

## What compile breaks

- **The memory planner's peaks.** The compiled DiT's first call from 20 latents (SDPA, 1080p), a
  new length's first call in the middle of a job planned near its budget, and the `int` VAE's
  first call at every tile shape need more than plain code; the planner knows none of them, and a
  plan packed to its budget runs out of memory (seedvr2x's compiled calibration,
  above; the `int` VAE in seedvr2x's 1080p plan: 26.13 GiB allocated against 19.87 plain at the
  calibration's first encode). `excl` would need its 0.9 to 2.2 GiB added to every pass.
- **The resume rule, unless the deterministic mode is kept on.** A resume is another process: its
  compiled kernels would be timed again, and its frames would differ from those the stopped run
  made (the rule: a resumed job's output equals an uninterrupted one's). With the mode kept on,
  a cold cache gives the same bits; a cache kept in the job's work directory would also save the
  compilations. Shown end to end for the VAE's `excl` compile ([the checks](#the-vaes-checks-for-v010)):
  stopped by `--until` and resumed with a fresh cache, the same frames. Not for the DiT: the one
  DiT-compiled job stopped by `--until` and resumed (the 45-s excerpt, its plan pinned, a fresh
  cache) ran out of memory in the resumed process, its
  first compiled window having left 13.24 GiB allocated where every other run's windows started
  with 2.8 to 4.5 (the size of some 26 swapped blocks' weights); seen once, the cause not
  isolated. The plan's record holds `compile_dit` and `compile_vae`, compared on resume
  (`runtime/resume.py`): a job started plain and resumed compiled is another job, refused; a job
  recorded compiled resumes as recorded.
- **The environment's record.** Compiling imports `jinja2`, `markupsafe` and `networkx` (Inductor's
  code generation) after seedvr2x has recorded the packages a job ran with: seedvr2x then warns of
  packages "not in the manifest's environment, a seedvr2x bug", and a resume could see a changed
  environment. A compiled seedvr2x would import them before the record.
- **The time estimates' calibration.** It runs the job's shapes before the job: compiled, it pays
  every compilation of them (seedvr2x's 1080p smoke: 207 compilations, 135 s; its calibration 183 s
  against 49), and the units after it compiled nothing; a length the calibration doesn't run
  would compile in the job, a unit slower than its estimate.
- **Host memory.** Inductor's worker processes held up to 16.2 GiB (summed resident sizes) while
  compiling, beside a process that pins 15.8 GiB for a fully swapped DiT and keeps 8.2 GiB of
  tiled frames: on a machine with 40 GiB for it, close to its limit. The planner's host budget
  doesn't count them.

## The VAE's checks for v0.1.0

Design's three checks of 2026-10-24, before the VAE's `excl` compile is built (opt-in, the
deterministic mode kept on), on the same RTX 5090, each a process of its own:

- **Bits, from one process to another: they hold.** `compile_vae.py`, `excl` at 1856 / 1088 px:
  two cold processes, a warm one and a cold one compiling in a single thread give the same latent
  and frames, bit for bit (SHA-256 982c49ed… and 8c620bce…), the compiled passes 0.83 to 0.88 of
  plain. seedvr2x's own runs (`compile_run.py --vae excl --keep-deterministic`) of 2 full-reference
  clips, the plan pinned (encode tiles 1856, decode 1088, 29 blocks swapped), in two processes with
  cold caches and a third on a warm one: the same frames (frame MD5s equal; the files' bytes
  differ: their container's metadata).
- **A whole job, planned for the compiled peak: it fits, it gains 10% of its units, and it
  resumes bit for bit.** The 45-s excerpt (1,079 frames, 14 shots), windows of 13 latents and 29
  blocks swapped pinned in every run; `compile_run.py --vae-peak-factor 1.1` makes the planner
  judge every VAE pass at 1.1 times its plain peak, and it chose an untiled encode and decode
  tiles of 1088 px, where plain code's own plan, on the same card a few minutes later, took 1216:

  | Run | Decode tiles px | Units s: encodes / windows / decodes, all | Calibration s (compiling) | The run s |
  |---|---:|---|---:|---:|
  | `excl` compiled, planned at 1.1 | 1088 | 216 / 207 / 458, 880 | 276 (238) | 1,320 |
  | plain, the same tiles | 1088 | 254 / 206 / 535, 995 | 44 (0) | 1,212 |
  | plain, its own plan | 1216 | 252 / 206 / 523, 980 | 43 (0) | 1,188 |

  The compiled run's units took 11.6% less than plain code's at its tiles and 10.2% less than
  plain code's own plan (encodes −14%, decodes −12%): the gain a user gets, its compilation (238
  s, in the calibration) paid back from about 2,500 frames (some 100 s of 1080p); this 45-s job
  took 11% longer in all, an hour of 1080p about 2.2 hours less. No unit compiled outside the
  calibration (0.2 s), nothing ran out of memory. The same compiled job stopped by `--until`
  (1 segment of 4 made, in shot 9) and resumed by a new process with a fresh cache made the same
  1,079 frames as the uninterrupted one, bit for bit (frame MD5s; the files' sizes equal too).
- **The compiler's host memory: 4.4 GiB, or none in a single thread.** `compile_mem.py` around
  `compile_vae.py`'s cold run: with torch's default (32 workers), 37 processes, their tree's summed
  RSS 27.7 GiB but its PSS 15.5, the workers' RSS 16.4 GiB and their PSS 4.4 (forked from the
  compiling process, they share most of its pages); with `TORCHINDUCTOR_COMPILE_THREADS=1`, 3
  processes, 0.38 GiB beside the main one, the tree 12.0 GiB, for 299 s of compilation against 241
  and 249 (+20 to +24%), the bits the same.

## What's left

Not measured, the 96 GB card having failed on 2026-10-23 before running them:

- the VAE untiled at 1080p (its compiled peaks, numz's doubling as seedvr2x would meet it) and on
  4K's large tiles (1536 to 2048 px), all three variants; a fit of `int`'s first call over tile
  sizes larger than a 32 GB card holds;
- the DiT's windows above 22 latents at 1080p, compiled: whether the jump from 20 latents is
  SDPA's (FlashAttention 2, outside the graph, untested), and the gain at the long windows a large
  card runs (76 latents and more);
- the bits at those shapes.

Not isolated: the cause of the `int` VAE's first-call peak, of the DiT's jump at 20 latents and of
the memory a compiled window left in a resumed process. Not shown: a compiled job stopped and
resumed to its end with the DiT compiled, its output against the same job uninterrupted (on the
5090, at a shape that leaves the first calls room); the VAE's, shown.

## Reproduce

```bash
# seedvr2x's venv python, run from the repo root; one GPU job at a time. MODELS: a directory with
# seedvr2_ema_7b_fp16.safetensors (and _sharp_), ema_vae_fp16.safetensors, transnetv2.safetensors.
# A fresh TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR for a cold compilation; the same again: warm.

# DiT: speed, compilation and peaks per length, 1080p, 35 blocks swapped, deterministic mode kept
python research/scripts/compile_dit.py run --model-dir MODELS --size 1080 --swap 35 --variant blocks \
    --recompile-limit 64 --keep-deterministic inductor --lengths 1,2,3,5,8,13,19 \
    --seq plain,blocks,plain,blocks --out dit-1080.json
python research/scripts/compile_dit.py table dit-1080.json
# the same with --variant whole (fails over the swap), --dynamic, --size 2160x3840 --swap 33
# torch's recompile limit: lengths 1 to 12 at 270x480, the default limit
python research/scripts/compile_dit.py limit --model-dir MODELS --out limit.json

# VAE: a variant against plain, tiled as a plan tiles, the deterministic mode kept
python research/scripts/compile_vae.py run --model-dir MODELS --clip CLIP.gt.mkv --size 1080x1920 \
    --frames 21 --variant int --keep-deterministic --passes AABAB --encode-tile 768 \
    --decode-tile 768 --out vae-int-768.json
python research/scripts/compile_vae.py table vae-*.json
CUDA_VISIBLE_DEVICES= python research/scripts/compile_vae.py selftest   # the int patch, CPU

# seedvr2x's own run, the DiT compiled, units recorded
python research/scripts/compile_run.py --dit blocks --keep-deterministic --recompile-limit 64 \
    --record-flags --units-json run.jsonl -- CLIP.d1.lr.mkv -o OUT --model-dir MODELS \
    --dit-model seedvr2_ema_7b_sharp_fp16.safetensors --vae-model ema_vae_fp16.safetensors \
    --resolution 1080 --format gbrp16le --cuts EMPTY.txt --seed 42

# a whole job with the VAE compiled (excl), its passes planned at 1.1 times plain's peaks, and
# the host memory of the process and its compile workers
python research/scripts/compile_mem.py mem.json -- python research/scripts/compile_run.py \
    --vae excl --keep-deterministic --vae-peak-factor 1.1 --record-flags --units-json job.jsonl \
    -- SOURCE.mkv -o OUT.mkv --model-dir MODELS --resolution 1080 --window 13 --swap-blocks 29

# the outputs against each other and the ground truth, CPU
python research/scripts/compile_bits.py --clips anime-bright,anime-clean --gt-dir CLIPS \
    --run 'p42=runs/p42/{clip}' --run 'p43=runs/p43/{clip}' --run 'd42=runs/d42/{clip}' \
    --pair p42:p43 --pair p42:d42 --out bits.json

# a job's gain from cut lists (cuts-0.3.json: compile_layouts.py cuts), CPU
python research/scripts/compile_layouts.py tables cuts-0.3.json --caps 19 --t-token-ms 0.083 \
    --gain 1:0.116,2:0.195,3:0.2,5:0.212,8:0.224,13:0.232,19:0.232 --compile-cold 48,2.0 \
    --compile-warm 9,0.15 --recompile-limit 0 --processes 1 --dit-share 0.2 --no-details
```
