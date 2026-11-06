# Small cards: where a 4K picture's seconds go, what a step needs beyond its peak under expandable segments, what the VAE's decode and the DiT's step hold

> Status: **measured** (2026-11-05, and the night to 11-06). One card: an RTX 5090 (32 GB)
> under WSL2, torch 2.14.1+cu130, seedvr2x at `6967542` (its own runtime, the 7B fp16 and the
> VAE; attention on PyTorch's SDPA unless a column says FlashAttention 2). By day, beside a
> desktop: seedvr2x's runs counted on the card less the 4 GiB left to the desktop; at night
> (its own section), the whole card with the desktop idle. Through
> [`scripts/small_card.py`](../scripts/small_card.py) (the emulation),
> [`scripts/picture_phases.py`](../scripts/picture_phases.py) (a picture's phases),
> [`scripts/dit_need.py`](../scripts/dit_need.py) and
> [`scripts/dit_lean.py`](../scripts/dit_lean.py) (what a DiT step holds, and a leaner one),
> [`scripts/vae_profile.py`](../scripts/vae_profile.py) (where a VAE pass's time goes),
> [`scripts/slice_probe.py`](../scripts/slice_probe.py) (the decode cut finer in time),
> [`scripts/hold_probe.py`](../scripts/hold_probe.py) (the scans of [hold.md](hold.md)),
> [`scripts/comfy_nodes.py`](../scripts/comfy_nodes.py) and
> [`scripts/comfy_wrap.py`](../scripts/comfy_wrap.py) (ComfyUI v0.44.0, node by node),
> [`scripts/tile_times.py`](../scripts/tile_times.py) (a VAE pass's time at each tile side) and
> [`scripts/tile_cost.py`](../scripts/tile_cost.py) (a tiled output against the untiled one).
> ComfyUI is GPL-3.0: its code is read and described here in other words, with `file:line`, and
> none is copied; C below is its `comfy/ldm/seedvr/vae.py`, and the vendored VAE's
> `causal_inflation_lib.py` and `attn_video_vae.py` are L and A.

## Why

On an 8 GB card, seedvr2x's plan refused the 4K picture and the 1080p clip that ComfyUI
v0.44.0 ran (comfy-study, 2026-11-04), and a 4K picture took 14 s in seedvr2x where ComfyUI's
took 5 s. This asks, of seedvr2x's own runtime, where the 14 s go; what a step needs beyond its
allocated peak once torch's allocator uses expandable segments (hold.md put it at 4.5 GiB under
the plain allocator, sized on its largest step); what the VAE's decode holds at the small tiles
a small card takes; and what one 4K latent's DiT step holds, line by line.

## Method

**One emulation.** `small_card.py ballast` is a process of its own, with a CUDA context of its
own and no torch. F0 is the card's free memory by NVML before it, T the free memory an N GB
card leaves a program (N - 0.65 GiB: 0.35 hidden by the driver and 0.3 held by a desktop, the
figure of comfy-study's emulated cards). It holds F0 - T - keep through the driver API, which
NVML counts as another program's, so that the card has T + keep free (keep 4.0 GiB: the
desktop's share, checked on a 16 GB emulation, where the card read T + 4.00 GiB free). Where
the card has less than keep - 0.25 GiB free for two readings in a row, it frees what it holds,
stops the measured process and logs an alarm. `small_card.py run` starts the measured process
(seedvr2x's command or a research script) with seedvr2x's reading of the card less keep, and
torch held under T less seedvr2x's 0.5 GiB. `--card whole` holds nothing and only watches: the
measures of this document are of that kind, with 24 to 26 GiB counted and no alarm in any.

**Phases.** `picture_phases.py` runs seedvr2x's command as a user's, then wraps the library's
steps: each is timed between two CUDA synchronisations, with torch's allocated and reserved
peaks. `still.upscale` runs three times on the models the command loaded: the first (cold) is
the command's own picture, models just loaded; the others (warm) are a server's next pictures,
and each output's SHA-256 is recorded to show that the reps give the same bits.

**Scans.** `hold_probe.py` as in hold.md: torch's limit at the step's allocated peak plus a
margin, going down; where the step runs, where it keeps its usual speed (within 5% of the roomy
rung), and down to where it keeps the roomy run's bits. Expandable segments, the DiT off the GPU
for VAE passes, 36 blocks swapped and the VAE away for DiT steps.

**Allocation history.** `dit_need.py` runs one DiT window three times (a warm-up, one with
torch's allocation history recorded, one plain for its time) and groups what is live at the
allocated peak by the source line that allocated it. `vae_profile.py` runs a tiled decode or
encode under torch's profiler. `slice_probe.py` runs `hold_probe.py` with the decode's slices
or its tail cut finer.

## The 4K picture's time, phase by phase

seedvr2x, whole card, 1920x1080 to 3840x2160, in seconds. Warm is the median of reps 2 and 3,
cold is rep 1:

| phase | SDPA, the plan (5 blocks swapped, tiles 2048) | FA2, the same pins | FA2, its plan (7 swapped, 2048) | FA2, tiles 1024 (5 swapped) | ComfyUI sharp fp16, warm |
|---|---|---|---|---|---|
| moves between phases (Models.enter) | 0.69 | 0.45 | 0.71 | 0.13 | - |
| resize on the GPU | 0.004 | 0.008 | 0.005 | 0.004 | 0.19 (Pillow, cached) |
| encode | 1.82 | 1.52 | 1.77 | 1.35 | 1.33-1.42 (cold; cached warm) |
| DiT step (noise 0.035 apart) | 3.26 | 2.60 | 3.09 | 2.85 | 1.68 (KSampler) |
| decode | 3.60 | 3.10 | 3.50 | 2.73 | 1.20 (512 px tiles) |
| colour correction (split) | 0.006 | 0.006 | 0.006 | 0.006 | off in the template |
| upscale, warm / cold | 9.48 / 11.17 | 7.81 / 9.38 | 9.17 / 11.70 | 7.17 / 8.78 | 3.86 prompt (encode cached) |
| PNG, 16-bit (41.1 MB) | 1.80 | 2.35 | 1.75 | 2.04 | 8-bit SaveImage 0.92 (11.3 MB) |

- **The 14 s of comfy-study** is the cold upscale (11.2) and the 16-bit PNG (1.8) and its write:
  a command's picture is always cold. ComfyUI's 5 s was a second prompt of the same picture, its
  resize and encode cached: a new picture there costs about 1.3 + 1.7 + 1.2 + 0.9 = 5.1 s.
  Dynamic VRAM changes nothing warm (3.76 s with it off).
- **Where the 9 s go**, warm against ComfyUI's new picture: the decode +2.4 s, the DiT step
  +1.6, the PNG +0.9 (16-bit against 8-bit), the moves +0.7, the encode +0.5. Cold adds 1.6 to
  2.5 s of first calls (cuDNN and cuBLAS set-up on the first encode and DiT step).
- **The decode** (`vae_profile.py`, one 4K latent) is GPU-bound, the kernels busy 94 to 98% of
  the wall. In tiles of 1024 px it takes 2.97 s: convolutions 1.58 (sm80 implicit-GEMM bf16
  kernels, and 0.24 s of NCHW to NHWC conversions around them), elementwise 0.58, GroupNorm
  0.36, copies 0.32, pads 0.28, cats 0.28. By tile: 2048 px 3.62 s, 1024 px 2.97, 512 px 3.32:
  the largest tile is the slowest. ComfyUI's decoder, from its code (`comfy/ldm/seedvr/vae.py`,
  read 2026-11-05, [vae-compare](#where-the-two-vaes-differ)): a one-frame chunk is convolved as
  one frame with its kernel summed over time (C:692-743), where numz's VAE repeats the first
  frame twice and convolves three (`extend_head`, `causal_inflation_lib.py:455-470`), 3 times
  the work for a picture; GroupNorm, SiLU and padding are one fp16 kernel writing the
  convolution's input (C:746-765); the layout is channels-last fp16 against NCDHW bf16; the
  residual is added in place (C:1067-1084). Its one-frame sum and its fp16 change the bits.
- **The DiT step**: FA2 takes 0.6 to 0.7 s less than SDPA. The vendored SDPA path runs its 100
  attention windows one by one, each with a host copy of `cu_seqlens`; ComfyUI batches equal
  windows, and that batching, tried here at run time, gives other bits and is no faster (2.92 s
  against 2.94). The lean attention of the DiT section also speeds the step: 2.48 to 2.61 s
  against 2.89 to 3.35 for the same latent, both backends, the same bits (`dit_need.py`, 36
  blocks swapped).
- **The moves**: with tiles of 2048 px the plan takes DiT blocks off the GPU for the VAE passes
  and back (0.3 s each way); with tiles of 1024 px nothing moves (0.13 s in all).
- **The PNG**: zlib level 6 with the adaptive filter, 1.80 s for 41.1 MB; level 1 1.26 s for
  41.5 MB (+1%); level 3 1.48 s; filter none 0.90 s for 48.3 MB (+18%). 8-bit at level 6 takes
  2.30 s (11.8 MB; the adaptive filter's choice is slower on 8 bits), at level 1 0.45 s
  (12.7 MB, +8%).

**What follows.** Four changes were measured, none of them in the VAE's arithmetic. The fastest
tile that fits, not the largest: on the whole card 1024 px gives 7.0 to 7.3 s warm against the
plan's 9.0 to 9.5 (-2.2 s), the frames differing only by the tiles' seams (the colour correction
already corrects their drift); a plan would need a time per tile side, the decode's by megapixel
and tile, measured here at 512 to 2048 px. zlib level 1 for the PNG: -0.5 s, +1% size, no pixel
changed. The lean attention of the DiT section: -0.3 to -0.7 s, bits kept. A warm process for
pictures (the image server already is one): -1.6 to -2.5 s of first calls. Together about 7.2 s
warm plus 1.3 s of PNG, against ComfyUI's 5.1 s with an 8-bit save. What is left is the VAE's
own implementation (decode 2.7 s against 1.2): the per-frame convolution of a picture and the
fused fp16 kernels change the bits; the in-place and fused-free parts (items 3 to 5 of the
reading) keep them, unmeasured for time.

## The room over a step's peak under expandable segments

`hold_probe.py` scans on the whole card, expandable segments, torch's limit at the step's
allocated peak plus the margin (GiB): where the step runs, where it keeps its usual speed
(within 5% of the roomy rung), and down to where it keeps the roomy run's bits. DiT steps: 36
blocks swapped, the VAE away, SDPA; VAE passes: the DiT off the GPU.

| step | allocated peak | runs down to | usual speed from | same bits down to |
|---|---|---|---|---|
| window, 1 latent, 1080p | 2.20 | 0.25 (refused at 0) | 0.25 | 0.25 |
| window, 3 latents | 4.48 | 0.25 | 0.25 | 0.25 |
| window, 5 latents | 6.58 | 0.25 | 0.5 (0.25: x1.49) | 0.25 |
| window, 7 latents | 8.77 | 0.25 | 0.75 (0.5: x1.48) | 0.25 |
| the picture's step, one 4K latent | 5.53 | 0.25 | 0.5 (x1.05; 0.25: x1.51) | 0.25 |
| decode, 1080p, 5 latents, 512 px | 6.39 | 0 (0.28 effective) | 1.25 (1.0: x1.07) | 0.5 (0.25: other) |
| decode, 768 px | 11.60 | 0 | 1.0-1.5 | 0.5 (0.25: other) |
| decode, 1024 px | 17.95 | 0.5 (refused at 0.25) | 2.5 | 0.5 |
| encode, 1080p, 33 frames, 512 px | 4.02 | 0 (other bits) | 1.0 | 0.25 |
| encode, 1024 px | 10.97 | 0 (other bits) | 2.0 | 0.25 |
| decode, 4K, 1 latent (the picture), 512 px | 2.35 | 0.25 | 0.25 | 0.25 (0: other) |
| decode, 4K, 1024 px | 5.25 | 0.25 | 0.5 | 0.25 |
| encode, 4K, 1 frame, 512 / 1024 px | 1.51 / 4.02 | 0.25 | 0.25 / 0.5 | 0.25 |

**Cause.** Under expandable segments the need beyond a peak is no longer the plain allocator's
fragmented cache (hold.md: 2.6 GiB for 4K's 3 latents, the figure the 4.5 was sized on,
measured on 2026-10-13 without them). It is a slope, and its knee follows the step's own size:
about a tenth of a DiT step's peak (0.25 at 1 latent, 0.75 at 7, 0.5 for the 4K latent), as
hold.md's own expandable table (0.4 to 2.5 for peaks of 2.7 to 23.3) already suggested. A VAE
pass needs 1.0 to 1.25 GiB at the small tiles a small card takes, more at large tiles (2.0 to
2.5 at 1024 px). The bits: DiT steps keep them down to 0.25 (no cuDNN in them); VAE passes of
several frames lose them at 0.25 (cuDNN's workspaces refused), one-frame passes at 0.

**What follows.** A reserve of max(0.6, 0.1 x P) GiB beyond a DiT step's predicted peak P, on a
held run under expandable segments, would leave every step measured its usual speed and its bits
(the 0.6 is the planner's margin already, the 0.1 x P the knee measured), where the plan today
counts on 4.5 (2.5 tightened). A VAE pass keeps its own room of 1.25 GiB (`VAE_ROOM`, unchanged:
usual speed at 512 to 768 px, the bits' 0.5 inside it) with no reserve on top; tiles of 1024 px
and more would need max(1.25, 0.2 x P) for their usual speed. The 0.5 GiB left outside torch
stays. It costs nothing measured in bits (every step keeps them at its room) and nothing in time
at that room. What it leaves out is the cushion the 4.5 also gave against a desktop that grows:
the re-hold check (DESIGN.md, Allocator) covers growth up to the reserve, so that with 0.6 to
0.9 GiB a desktop growing more stops the run at its next unit, where today it is held again
(resumable).

## The VAE decode at small tiles

`hold_probe.py`'s decode and encode steps (the runtime's own tiled passes, the latent and the
scratch in system memory), the DiT off the GPU, expandable segments. Allocated peak in GiB, the
VAE's weights and torch's 0.03 included (0.50 allocated before each step); in brackets, the
pass's seconds:

| step | 512 px | 640 | 768 | 1024 |
|---|---|---|---|---|
| 1080p decode, 1 latent (1 frame) | 2.35 | | | |
| 1080p decode, 5 latents (first slice 2 latents, 5 frames, then 4 a slice) | 6.39 | | | |
| 1080p decode, 19 latents (the clip) | 6.39 (43 s) | 8.59 (40 s) | 11.60 (40 s) | 17.95 (42 s) |
| the same, the decoder's tail a frame at a time | 4.70 (48 s) | 6.57, on 5 latents (below) | 9.24 (44 s) | 14.16 (44 s) |
| `peaks.py`'s figure (the plan's, with the VAE room 1.25 apart) | 6.42 | | | |
| 4K decode, 1 latent (the picture) | 2.35 (4.1 s) | 3.01 (3.5) | 4.10 (3.4) | 5.25 (3.1) |
| 1080p encode, 73 frames | 4.02 (20 s) | | 7.58 (18) | 10.97 (18) |
| 4K encode, 1 frame | 1.51 (1.9) | | | 4.02 (1.5) |

- **What decides the 1080p clip's floor** is the decode's slice, not the clip. A slice is 4
  frames (5 for the first) taken through every block of the decoder at once (vendored
  `attn_video_vae.py`, `Decoder3D`; `model.latent_slices`); 5 latents and 19 peak alike, 1
  latent (1 frame) at 2.35. `peaks.py`'s 6.42 for 512 px is the measured 6.39 (within 0.03).
  With the plan's VAE room (1.25) and its reserve that is the 7.67 GiB of the refusal.
- **ComfyUI on the same clip and tiles** (its 3B template; the VAE is the same file): one tiled
  decode call for all 73 frames in 61.4 s, peak 1.24 GiB allocated (0.03 before: its VAE weights
  stay on the host until the call), flat in the clip's length, against seedvr2x's 6.39 in 43 s.
  From its code (`comfy/ldm/seedvr/vae.py`): each tile decodes all its frames, 2 latents then 1
  per call as seedvr2x does, but after the last temporal upsampler its decoder runs one frame at
  a time with the causal cache between frames (the full-resolution 128-channel blocks, the
  largest tensors, C:427-455, 1214-1223), writes each decoded frame into one preallocated output
  (C:1304-1333), fuses GroupNorm, SiLU and padding into one fp16 kernel and adds the residuals
  in place. Its encode of the clip took 27.9 s (2.03 GiB) against seedvr2x's 19.7.
- **Fewer frames at once** (`slice_probe.py`, run-time patches): decoding the first latent alone
  (then one a slice) saves nothing (6.38: the 4-frame slices peak as high) and changes the bits.
  Running the decoder's tail (up blocks 2 and 3, the output norm and convolution) a frame at a
  time, the causal convolutions' memory carried from frame to frame, takes 6.39 to 4.70 at
  512 px, 11.60 to 9.24 at 768 and 17.95 to 14.16 at 1024 (-20 to -27%), in about the same time
  (+3 to +11%). The frames are the reference's at 768 and 1024 px and not at 512 px (cause not
  found: cuDNN choosing another algorithm for the smaller shapes is the likely one, unproven).
  Tiles of 576 px (5 latents: 5.82 against 7.95) give other bits too, and tiles of 640 px (6.57
  against 8.59) the reference's.
- **Does it fit 8 GB.** An 8 GB card has 6.68 GiB free by NVML after the context, torch held to
  6.18. Today the clip needs 6.42 + 1.25 + 4.5 (or 2.5): no. With the room of the previous
  section and the tail, 4.70 + 1.25 = 5.95 fits at 512 px, where its bits are not today's.
  ComfyUI's level (1.24) would need its per-frame pieces before the tail too, and its fused fp16
  kernels (other bits).

**What follows.** The decoder's tail a frame at a time in the vendored VAE (a marked change) is
bit-exact where measured at 640, 768 and 1024 px and not at 512 or 576 px, the difference to be
understood first; the planner's decode figures would be measured again for it (0.71 to 0.79 of
today's above the weights).

### Where the two VAEs differ

Read 2026-11-05 (C, and the vendored VAE as N, with L and A its two files), by effect on the
memory and the time of a decode:

1. **A single frame is convolved as one frame, not as three** (picture and clip start). N
   repeats the first frame twice to give the causal 3x3x3 convolution its history
   (`extend_head`, L:455-470), so that every convolution of a one-frame decode reads 3 frames
   (203 MB at full resolution in a 512 px tile) and does 3 times the useful work. C sums the
   kernel over time when there is no history (C:692-743). Not bit-identical.
2. **GroupNorm, SiLU and padding are one kernel** writing into the convolution's buffer
   (C:746-765), fp16 only (C:48-50); N has four full tensors live in turn (A:300-348,
   L:399-443). Changes the bits.
3. **The residual is added in the epilogue or in place** (C:688-689, 1067-1098), where N makes a
   sum and a quotient (A:345). Same values.
4. **The upsampler writes its pixel-shuffle phases one at a time** into the convolution's buffer
   (C:934-995), where N makes the expanded, the shuffled and the padded tensors (A:105-163).
   Same values.
5. **After the last temporal upsampler, one frame at a time** (C:427-455, 1214), where N takes a
   whole slice through every block (A:968-1021, 1268-1290). Same values by C's own comment
   (C:427-433); measured here, the same bits at 640 px and up.
6. **Speed**: the three-fold convolution of item 1, the channels-last fp16 layout, the fused
   kernel of item 2. Not the cause: cuDNN's benchmark mode (off by default in ComfyUI) and the
   fp16-accumulate convolution (opt-in).

Both hold one tile at a time on the GPU, both use SDPA in the mid-block attention, and both
split large convolutions by a 0.5 GiB limit.

## What one 4K latent's DiT step holds

One 4K latent (3840x2192 padded, 32,880 tokens), every block swapped, the VAE off the GPU, as
the 8 GB picture's plan has it (`dit_need.py`; allocated GiB; resident before the step 1.06: two
block slots, the DiT's other parameters, buffers, text embeddings):

| the DiT step | SDPA peak | FA2 peak | bits | warm step, s (SDPA / FA2) |
|---|---|---|---|---|
| as it is | 5.53 | 5.31 | reference | 3.25-3.35 / 2.89-3.11 |
| lean: the projection's q,k,v dropped once windowed, q and k cast to bf16 before the window concatenation, SDPA's outputs written into one tensor | 3.91 | 3.91 | equal, both | 2.76-2.94 / 3.12-3.14 |
| + the norm, RoPE and cast of q and k 4,096 tokens at a time | 3.35 | 3.35 | equal, both | 3.00-3.04 / 2.69-2.76 |
| + the MLP's GELU in place on proj_in's output | **3.08** | **3.08** | **equal, both** | 2.56-2.61 / 2.48-2.59 |
| (rejected) + the MLP in chunks of 8,192 tokens | 3.08 | 3.08 | other bits | |
| (rejected) SDPA's equal windows batched, ComfyUI's way | 6.05 | | other bits | 2.87-2.92 |

- **What the 4.47 GiB over the resident weights is made of** at the step's peak, inside the
  attention (a torch allocation history, grouped by source line): the projection's q,k,v 0.56
  (dead once windowed, kept by a local name), its windowed copy 0.56, q, k and v concatenated
  per window with the text 1.11 (q and k in float32), RoPE's float32 q and k 0.75, the norms'
  outputs 0.38, the attention's bf16 casts of q and k 0.44, SDPA's 100 window outputs and their
  concatenation 0.44, the residual 0.19. q and k are float32 from the norm on: the RMS norm's
  float32 weight promotes them (vendored `dit_7b/normalization.py:97`), so RoPE and the
  concatenation run in float32 and the attention casts after. The MLP (GELU, 4x: 0.75 GiB a
  tensor at 4K) is the next peak.
- **Why the bits stay**: a cast and a copy commute bit for bit; RMS norm, RoPE and the cast act
  on each token and head alone; GELU in place is the same kernel on the same values. Checked on
  the step's output (SHA-256 of the latent, seeds 42 and 43) under both backends at 4K, and
  under SDPA on a 1080p window of 5 latents (41,400 tokens: 6.58 to 3.57 GiB). The two rejected
  ones change the matrix multiplies' shapes or the SDPA kernel's batch.
- **ComfyUI's DiT phase** for the same picture: 2.39 GiB allocated at its peak with 0.50 before,
  so about 1.9 GiB of activations (torch's count; comfy-aimdo pages the weights outside it),
  against 4.47 as it is and 2.02 lean here. It lowers its peak by the means the lean step uses
  and by two more: the MLP in chunks of 8,192 tokens into one output
  (`comfy/ldm/seedvr/model.py: 774-800`; the rejected row above, other bits here) and a partial
  RoPE in place, 4,096 tokens at a time (model.py:330-340). On 8 GB (comfy-study) its card went
  down to 0.11 GiB free: dynamic VRAM fills what the activations leave with weights, and evicts
  under a 256 MB headroom.
- **The planner's figure**: 5.64 GiB for this step under SDPA (5.53 measured), its `PER_TOKEN`
  term 127.16 KiB x 1.0469 a token; lean measures 2.02 GiB over the resident weights, about
  64 KiB a token at 4K (63.6 KiB at 1080p, 5 latents, against 139.9 as it is: 0.45 to 0.46 of
  today's at both sizes; the plans below take `PER_TOKEN` = 59,900 bytes, 0.46 x 127.16 KiB).

**What follows.** The lean attention and the in-place GELU in the vendored 7B would be a marked
change that moves no bit (the milestone-1 regression would show it on the whole pipeline), and
the planner's per-token term would be set again for it (59,900 bytes in the plans below). The
picture's need on 8 GB goes from 5.64 to about 3.4 GiB with the planner's margin, under the
3.68 GiB the plan counts on there with the tightened reserve; with the room of the second
section, under what it counts on with none.

## Under cudaMallocAsync

ComfyUI's default on its cu130 builds. The same scans, torch's limit held a further 2.5 GiB
under the card's figure, for the pool:

| step | allocated peak | runs, usual speed and same bits down to | reserved past torch's limit at a margin of 0 |
|---|---|---|---|
| window, 1 / 3 / 5 / 7 latents, 1080p | 2.20 / 4.48 / 6.58 / 8.77 | 0 (every rung) | 0.36 / 0.61 / 1.04 / 1.45 |
| the picture's step, one 4K latent | 5.53 | 0 | 0.85 |
| decode, 1080p, 5 latents, 512 / 768 / 1024 px | 6.39 / 11.60 / 17.95 | 0 | 1.83 / 2.34 / 4.58 (4.74 at 0.25) |
| encode, 1080p, 33 frames, 512 / 1024 px | 4.02 / 10.97 | 0 | 0.92 / 2.65 |
| decode, 4K, 1 latent, 512 / 1024 px | 2.35 / 5.25 | 0 | 0.71 / 1.06 |
| encode, 4K, 1 frame, 512 / 1024 px | 1.51 / 4.02 | 0 | 0.31 / 0.89 |

- **Torch's limit binds what is allocated, never the pool.** No step slowed or changed its bits
  at any rung, because none ever ran short; the pool held 0.14 x P (DiT windows) to 0.30 x P
  (decodes) beyond the allocated peak, outside torch's limit, on the card. `memory_stats` gives
  the peaks under it (`max_memory_allocated` and `max_memory_reserved` both filled; NVML's card
  figure moved with the reserved one). In one roomy reference (the 1024 px decode, 23.2 GiB
  reserved for 17.95 allocated) the card went to 3.79 GiB free for a moment: the pool went past
  a limit set 2.5 GiB under the card's figure.
- **What the room must be.** The room beyond a step's peak P has to be on the card, not under
  torch's limit: max(0.6, 0.2 x P) for a DiT step, about 0.3 x P for a VAE pass (1.9 GiB at
  512 px, 3.5 at 768, 5.4 at 1024 for a 1080p decode), where the plain allocator and expandable
  segments need nothing comparable. seedvr2x's own limit cannot hold the pool, so on a lending
  driver the card's figure, checked between units, is the only bound. (Two measures of the pool:
  the scan's is how far it goes past a limit at the peak itself, over 5 latents (1.83 and
  2.34 GiB at 512 and 768 px); the clip's runs below, reserved over allocated with room, over 19
  latents (1.96 and 2.93).)
- **The same 8 GB card.** The 1080p clip's decode at 512 px takes 8.2-8.3 GiB of card under it
  (6.39 allocated; 8.22 reserved at the peak over 5 latents, 8.34 over the clip's 19), more than
  an 8 GB card has; with the tail a frame at a time 6.50 (4.70 allocated), still over the 6.18
  torch would be held to. The picture's lean step (3.08) with 0.2 x P fits; its 512 px decode
  (2.35 + 0.71) fits.
- **The clip's decode and the tail, under it.** The allocated peaks are the same to the MiB (the
  plain decode 6.39 at 512 px and 11.60 at 768; with the tail 4.70 and 9.24), the frames the
  same (the tail again other bits at 512 px, the same at 768), and the pool reserves 1.96 /
  2.93 GiB more (plain) and 1.80 / 2.29 (tail): 8.34 / 14.53 and 6.50 / 11.53 GiB reserved.

## Emulated cards and the proposals

### The plans and runs on 16, 12 and 8 GB

One emulation (`small_card.py`: T = N - 0.65 GiB before the context; 6.68 to 6.69 GiB free after
it on 8 GB, 10.69 to 10.71 on 12, 14.68 to 14.69 on 16), SDPA, the sharp 7B in each format
(fp16, int8, dynamic GGUF). The plans (`--plan`):

| card | picture (3840x2160): fp16 / int8 / GGUF | clip (73 frames to 1080p): fp16 / int8 / GGUF |
|---|---|---|
| 16 GB | plans (4.5): 27 / 17 / 0 blocks swapped, tiles 1728 / 1280 | plans (4.5): windows 7 / 7 / 8 latents, 35 / 32 / 35 swapped, tiles 768 / 576 |
| 12 GB | fp16 plans only with the 2.5 reserve (32 swapped, 1344 / 1088); int8 36 swapped and GGUF 31, 1024 / 768, with 4.5 | all three plan only with 2.5: windows 5 / 5 / 6, tiles 640 / 512 (the decode's floor, 7.67 in a budget of 7.69) |
| 8 GB | refused: needs 5.64 / 5.60 / 5.11 GiB, counts on 3.68 | refused: needs 7.67 (decode at 512 px), counts on 3.68 |

The runs (seconds; "upscaled" is the command's own, the models loaded; a clip's phases are its
units'):

| card | file | picture: upscaled (load), blocks swapped, tiles | clip: upscaled = encode + windows + decode, windows, tiles, largest unit peak |
|---|---|---|---|
| 16 GB | fp16 | 12.1 (78), 27, 1728/1280 | 92.8 = 19.8 + 22.2 + 50.9, 7 latents, 768/576, 9.19 GiB |
| | int8 | 13.4 (64), 17 | 87.7 = 18.5 + 20.4 + 48.9, 7, 9.19 |
| | GGUF | 12.8 (35), none | 86.6 = 17.8 + 21.4 + 47.2, 8, 9.41 |
| 12 GB | fp16 | 12.0 (117), 36, 1024/768 (reserve 4.5: 10.69 free; its `--plan` at 10.56 took 2.5, 32 swapped, 1344/1088) | 93.5 = 17.9 + 23.5 + 52.1, 5, 640/512, 7.42 (reserve 2.5) |
| | int8 | 13.2 (62), 36 | 92.7 = 18.5 + 24.1 + 50.2, 5, 7.00 (2.5) |
| | GGUF | 13.4 (45), 31 | 94.5 = 18.4 + 25.2 + 50.4, 6, 7.45 (2.5) |
| 8 GB | all | refused by the plan | refused by the plan |

Every clip unit's VRAM peak (the run's own log) was at or under its plan's prediction (the
decodes equal to it, the windows 0.11 to 0.49 GiB under); the card's least free was 6.6 to
9.4 GiB beside a desktop kept at 4 GiB. On 16 and 12 GB the three files run alike: the clip is
the VAE's (encode 18 s, decode 47 to 52 s of 87 to 95), the DiT 20 to 25 s; the picture 12 to
13 s cold whatever the file. The files differ in load time (GGUF 27 to 45 s, int8 42 to 64, fp16
78 to 117, from a Windows drive under WSL2) and host memory, not in what fits.

- **impl-gguf's 12 GB refusal** (DESIGN.md, Repository and apps, small cards) against
  comfy-study's 12 GB runs comes from the emulation. impl-gguf's ballast (in the process, after
  its context, the desktop counted as 0.3 GiB) left 10.24 GiB free; comfy-study's and this one's
  leave 10.69. The clip's 1080p decode at 512 px needs 7.67 GiB with the 2.5 reserve and 0.5
  held back: 10.69 - 3.0 = 7.69 plans (0.02 to spare), 10.24 - 3.0 = 7.24 is refused 0.43 short.
  No real difference between the files: at 12 GB the decode's floor decides, which no DiT file
  moves, so all three plan or refuse together. Which emulation is a real 12 GB card under
  Windows is not measured (no such card here); the 0.02 GiB margin says a real 12 GB card is on
  the edge either way.

### What each proposal gives

seedvr2x's own plans and runs with the proposals set at run time (`small_card.py --set/--lean/
--tail-frames`), SDPA, sharp 7B fp16, under expandable segments. "Room" is the reserve of the
second section, taken as one figure of 1.0 GiB (max(0.6, 0.1 x P) for every peak under 10 GiB);
"lean" the DiT section (the planner's per-token term 0.46 of today's, 59,900 bytes); "tail" the
decoder's tail a frame at a time (decode figures 0.79 of today's above the weights); "floor" a
reserve of 0.6 and the VAE's room at its bits' 0.5, lean, tail at its 512 px 0.72.

| card | today | room | room + lean | room + lean + tail | floor |
|---|---|---|---|---|---|
| 8 GB picture | refused (5.64 for 3.68) | refused (5.64 for a budget of 5.19) | **runs: 11.9 s cold**, 32 swapped, 960 / 704 px, card least free 5.40 | - | - |
| 8 GB clip | refused (7.67 for 3.68) | refused (decode 7.67) | refused | refused (6.43 for 5.19) | **runs: 101.4 s**, windows of 8, 512 / 512 px, every unit under its prediction; the decode's frames not today's (tail at 512 px) |
| 12 GB picture | 12.0 s, 36 swapped (or 32 with 2.5: the desktop's 0.13 GiB decides), 1024 / 768 | 28 swapped, 1664 / 1280 (plan) | 23 swapped (plan) | - | - |
| 12 GB clip | 93.5 s, reserve 2.5, windows of 5, 640 / 512 | windows of 7, 704 / 512 (plan) | windows of 11 (plan) | **94.9 s**, windows of 11 (2 for 5), decode 640 px (its bits kept) | - |
| 16 GB picture | 12.1 s, 27 swapped, 1728 / 1280 | 19 swapped, 2048 / 1728 (plan) | 13 swapped (plan) | - | - |
| 16 GB clip | 92.8 s, windows of 7, 768 / 576 | windows of 11, 1024 / 768 (plan) | windows of 19, the shot whole (plan) | windows of 19, decode 896 px (plan) | - |

- **What each costs.** Room: no bit and no time at the room measured, and the cushion the 4.5
  gave a desktop that grows (a run stops at its next unit, resumable, where it would be held
  again today). Lean: nothing (the same bits, the step 0.3 to 0.7 s faster). Tail: the frames at
  512 and 576 px tiles (bit-identical at 640 and up) and +3 to +11% on the decode. Floor: the
  VAE's room at 0.5 makes a 512 px decode about 1.2 times as long (11.86 s against 9.4 to 9.7)
  and stands on the bits' last measured rung.
- **Under cudaMallocAsync** (the table of that section) the pool needs 0.14 to 0.30 x P more on
  the card: the 8 GB picture still fits lean, the 8 GB clip does not.

## The night of 2026-11-05: a really full card, longer lean windows, tile sides, the tiles' cost

The same card, seedvr2x at `6967542` with this repository's scripts, the sharp 7B fp16, SDPA,
and the lean attention with the in-place GELU (`dit_lean.py`) unless a line says otherwise. The
scans of longer lean windows, the tile sides and the tiled arms ran by day under the desktop
rule; the whole jobs, the untiled passes and the largest tile sides ran on the whole card from
00:02 to 04:24 with the desktop idle (its window manager holding 1.9 GiB, outside every figure
below), the emulated cards left exactly T: `small_card.py ballast --keep 0 --alarm 0.3`, which
stops everything under 0.3 GiB free on the card. No alarm in the night.

### Whole jobs on a really full card

Each job planned and run on an emulated card left exactly T, then the same plan (its blocks,
window and tiles pinned) on the whole card, its twin, for the bits and the time. The reserve:
4.5 or 2.5 (`RESERVE` and `RESERVE_TIGHT` both set to it), the measured room (`RESERVE` max(0.6,
0.1 x P) for the DiT's predicted peak P, `VAE_ROOM` making a VAE pass's max(1.25, 0.2 x P),
both from the plan's own predicted peaks, planned again until the plan held the rule, the room
only ever growing), and the floor (0.6 and 0.5, the decoder's tail a frame at a time, the plan's
decode peaks x 0.72, as above). Both times are the command's `upscaled in`: the 4K picture's
holds its two upscales on the models loaded (the first cold; 16.2 to 18.2 s the two, the twins'
15.1 to 16.9) and the 16-bit PNG, encoded and written (2.5 to 3.1 s); the 73-frame clip's holds
its units: the encode, the windows, the decode and the frames' write.

| job | card | reserve (DiT / VAE pass) | plan: swapped, window, tiles | s | twin s | the card's least free | bits |
|---|---|---|---|---|---|---|---|
| 4K picture | 16 | 4.5 | 22, -, 1728 / 1280 | 19.7 | 18.5 | 5.19 | the twin's |
| | 16 | 2.5 | 17, -, 2048 / 1600 | 21.0 | 19.2 | 3.08 | the twin's |
| | 16 | room, 0.6 / 2.57 | 12, -, 2048 / 1728 | 21.3 | 19.9 | 0.97 | the twin's |
| | 12 | 2.5 | 26, -, 1344 / 1088 | 19.2 | 18.1 | 2.89 | the twin's |
| | 12 | room, 0.6 / 1.76 | 22, -, 1728 / 1280 | 19.7 | 18.1 | 1.19 | the twin's |
| | 8 | room, 0.6 / 1.25 | 31, -, 1088 / 832 | 19.0 | 17.7 | 1.00 | the twin's |
| 1080p clip | 16 | 4.5 | 30, 11, 768 / 576 | 77.6 | 77.1 | 4.15 | the twin's |
| | 16 | 2.5 | 34, 19, 896 / 704 | 72.9 | 71.9 | 0.84 | the twin's |
| | 16 | room, 1.33 / 2.56 | 32, 19, 1024 / 704 | 73.9 | 71.8 | 1.43 | the twin's |
| | 12 | 2.5 | 35, 11, 640 / 512 | 77.8 | 77.0 | 2.26 | the twin's |
| | 12 | room, 0.94 / 1.72 | 31, 11, 768 / 576 | 78.2 | 76.1 | 1.02 | the twin's |
| | 8 | floor, 0.6 / 1.1, tail | 36, 8, 512 / 512 | 89.5 | 87.3 | 0.70 | the twin's |

- **A really full card costs a job 1 to 9%, whatever its reserve**, and no bit. The passes
  themselves keep their time: the picture's warm encode, DiT step and decode within 0.1 to 0.2 s
  of the twin's, the clip's encode and decode within 1 to 3%. What differs is the moves between
  phases: the emulated cards swap blocks, which enter for the DiT and leave for the decode (0.15
  to 0.44 s each for the picture, 1.2 to 2.2 s a clip), where the twin, on a card with room for
  the whole DiT beside each pass, moves nothing.
- **One pass slowed**: the 16 GB clip's window of 19 latents at the measured room, 13.3 s
  against 11.6 for the twin (+15%; +3% at 2.5). Its room was the plan's 1.33 GiB over a
  predicted peak of 12.47; the window's measured peak is 10.40, so it had about 3.4 GiB in fact,
  at the edge of what the next section finds such a window needs. The windows of the 12 GB
  clip at its room and of the 8 GB floor clip ran +6.7% and +4.6% against their twins, the
  other three rows -0.7 to +3.4% (each as the command times a window, its phase change in it).
- **A tighter plan is no faster.** The measured room swaps 12 blocks for the 16 GB picture where
  4.5 swaps 22, and its larger tiles take back what the blocks save.
- **On 8 GB both run**: the 4K picture (7.4 s warm, 9 s cold), the clip in 89.5 s (101.4 by day
  with the desktop's 4 GiB kept).

**October's 1.4 to 1.5 times.** hold.md's whole jobs again, on the whole card really full, the
shipped code (no lean attention), SDPA:

| job | reserve | plan | s | frames |
|---|---|---|---|---|
| 66 frames in two shots at 1080p, windows of 9 latents, tiles 1024 / 832 pinned | 4.5 | 7 blocks swapped | 69.4 | one digest for all four |
| | 2.5 | none swapped | 69.4, 69.5 | |
| | 2.5 | 7 swapped, pinned | 68.2 | |
| 121 frames at 1080p, unpinned | 4.5 | 26 swapped, windows of 17, encode untiled, decode 1152 | 121.2, 121.5 | one digest |
| | 2.5 | 22 swapped, windows of 17, encode untiled, decode 1280 | 132.2, 132.7 | another (other tiles) |

The 2.5 slows no pass. The 121-frame job's 1.09 times (hold.md: 216 s against 152, 1.42) is its
decode alone, 81.2 s in 1280-px tiles against 70.9 in 1152 (the encode 27.3 against 26.5, the
windows 23.7 both): a plan with more room takes larger decode tiles, and at 1080p a decode slows
past about 1150 px (below). October's further slowdown is not reproduced on this code.

**A hang.** The 66-frame job's second run at 4.5 stopped in its second shot's decode (6 tiles of
832 px, 33 frames, 17.9 s for the first shot's) and never moved: the GPU at 6%, the card 4.25
GiB free, the main thread in a timed sleep, CUDA's threads polling, both ffmpeg processes
waiting on their pipes. SIGTERM was logged 34 minutes after the last line, and the process was
still there 20 s later: SIGKILL (exit 137). Its first run, three minutes before, took 69.4 s;
nothing else hung that night. Its cause is not found (no Python stack was taken).

### Lean windows beyond 7 latents

`slice_probe.py --lean --gelu-inplace` (the lean attention installed in `hold_probe.py`'s
process, the allocator's settings set first), 1080p windows, 36 blocks swapped, the VAE away,
expandable segments, by day; each window once with room, then one forward a rung down from the
roomy figure, the cache emptied before each. The room: torch's limit over the allocated peak P.

| window | P (reserved at the peak) | s a forward with room | usual speed down to | under it | the reference's bits down to |
|---|---|---|---|---|---|
| 7 latents, lean | 4.56 (5.22) | 4.7 | 0.8 (10.7 s at 1.0, 6.3 at 0.9) | 15-16 s at 0.7-0.5, 12 at 0.4, 29-31 at 0.3-0.1 | 0.1 |
| 11, lean | 6.53 (7.64) | 7.5 | 1.3 | 19-35 s from 1.2 to 0.5, 31-39 under | 0.1 |
| 19, lean | 10.40 (12.27) | 13.1 | 3.0 | 16.5-17.2 s (x1.25) at 2.75-2.25, 30-40 at 2.0, 45 at 1.0, 52-65 under | 0.2 (refused at 0.1) |
| 11, plain | 13.06 (14.02) | 9.0 | 1.0 | 21-25 s from 0.75 to 0.25 | 0.25 |

- **max(0.6, 0.1 x P) keeps a lean window's bits, not its speed**: at the rule's room (0.6,
  0.65, 1.04 GiB) the three ran 3.2 to 3.4, 2.7 and 3.4 times as long as with room.
- **The room follows the window's work, not its peak.** A lean window needs what a plain one of
  its length does: 0.8 GiB at 7 latents against 0.75 plain (the table of the day above), 1.3 at
  11 against 1.0, 3.0 at 19 where hold.md's plain windows of 16 and 17 latents needed 2.5 and 2.
  The reserved-over-allocated gap at the peak is 0.65, 1.11 and 1.87 GiB lean, 0.96 plain at 11.
- **max(0.6, 0.3 x P)** on the lean P covers every lean window measured (1.37, 1.96 and 3.12 GiB
  asked against 0.8, 1.3 and 3.0 needed).

### A VAE pass's time at each tile side

`tile_times.py`: the models loaded once, the DiT off the GPU, torch held as a run holds it; at
each side the planner tries (multiples of 64 px from 512 to the largest under the frame's longer
side), a warm-up on a short shot of the same tile and slice shapes, then three passes, the
median kept. By day, the card less 4 GiB; the clip's largest sides on the whole card at night.

| side | 4K picture, one frame: encode s (peak GiB) | decode s (peak) | 1080p clip, 73 frames: encode s (peak) | decode s (peak) |
|---|---|---|---|---|
| 512 | 1.76 (1.51) | 3.35 (2.35) | 19.47 (4.02) | 40.84 (6.39) |
| 640 | 1.57 (2.07) | 3.02 (3.01) | 16.75 (5.42) | 37.41 (8.59) |
| 768 | 1.55 (2.76) | 3.04 (4.10) | 17.06 (7.58) | 37.24 (11.60) |
| 896 | 1.55 (3.19) | 3.04 (4.39) | 17.54 (9.06) | 38.10 (14.60) |
| 1024 | 1.44 (4.02) | 2.95 (5.25) | 17.62 (10.97) | 33.73 (17.95), night |
| 1152 | 1.37 (4.95) | 2.94 (6.06) | 17.12 (13.18) | 33.77 (21.54), night |
| 1280 | 1.39 (4.95) | 3.01 (6.74) | 17.00 (14.05) | 38.74 (23.66), night |
| 1472 | 1.55 (5.69) | 3.03 (7.82) | 17.03 (16.08) | 50.54 (26.77), night |
| 1600 | 1.58 (6.63) | 3.28 (8.78) | 15.56 (16.92), night | refused |
| 1856 | 1.70 (8.10) | 3.46 (10.79) | 15.90 (19.55), night | - |
| 2048 | 1.83 (8.84) | 3.52 (12.57) | - | - |

(Every side is in the run's file; by day the clip's encode from 1664 px and its decode from 960
ran short of room beside the desktop, 2 to 4.5 times as long, and were run again at night.)

- **A trough, not a slope.** The 4K decode is 2.9 to 3.1 s from 640 to 1472 px, slower at 512
  (45 tiles, each overlap decoded twice) and from 1536 up (3.52 at 2048); the clip's decode
  fastest from 960 to 1152 px (33.7 to 34.8 s), slower at 1280 (38.7) and beyond. The encodes
  vary less: 4K 1.34 to 1.55 s from 832 to 1408, the clip 15.6 to 18 s from 576 up.
- **The first pass at a side costs more** (its convolutions' engines chosen): the 4K decode's
  warm-up took 3.43, 4.93 and 6.64 s at 512, 1024 and 2048 px.
- **The moves each side causes**, from the whole card's plan with both passes pinned to the
  side: the 4K picture swaps no block at any side, the whole DiT staying on the GPU beside both
  passes up to 1152 px, then leaving 1 block for the decode at 1216 px, 8 at 1472, 15 at 2048
  (up to 8 for the encode); the clip swaps 6 and runs one window of 19 latents at every side,
  up to 1152 px, the decode keeping 30 of the DiT's blocks on the GPU to 640 px, 24 at 768, 9
  at 1024, 1 at 1152 (the encode 30 to 832, 20 at 1152); at 1216 and 1280 px it swaps none and
  keeps 3 and 1 beside the decode; from 1344 px the decode is refused (25.96 GiB asked).

### What the tiles cost the picture

`tile_cost.py`: each output against the untiled one, both after `split`: the colour study's
drift (colour.md, VAE tiles at 1080p: the largest |mean luma difference| of a 60x60 block, and
over the flattest third of blocks, here the untiled output's), PSNR-Y over the frame, inside and
outside the tiles' overlap zones widened by 16 px, and maps of where the differences sit.

The untiled passes: the 4K picture's encode untiled peaks 14.7 GiB over the models (1.85 s
warm), its decode 23.7 GiB over them (3.47 to 3.51 s warm), where the plan, which takes no
untiled decode past the 1080p frame measured, would count 29.0 (its one-frame ratio of 0.21 x
138.10); it ran outside the plan (`picture_phases.py --then`, 36 blocks swapped), the same bits
twice. The clip's untiled encode peaks 19.85 GiB (15.4 s); its untiled decode does not fit the
card (27.64 GiB allocated when refused; 27.23 with the decoder's tail a frame at a time, which
ran in 81 s with torch's reserve at its limit), so the clip's reference is the untiled encode
with 1280-px decode tiles.

| encode / decode tiles | 4K picture: drift (flat third), PSNR-Y dB | 1080p clip, against U / 1280 |
|---|---|---|
| U / 2048 | 0.16 (0.04), 53.7 | - |
| U / 1024 | 0.24 (0.07), 48.6 | 0.06 (0.04), 50.8 |
| U / 768 | 0.30 (0.09), 46.0 | 0.14 (0.08), 46.8 |
| U / 512 | 0.33 (0.15), 44.4 | 0.13 (0.08), 46.7 |
| 2048 / U | 0.25 (0.08), 44.6 | - |
| 1024 / U (clip: / 1280) | 0.40 (0.13), 39.8 | 0.10 (0.06), 47.4 |
| 768 / U | 0.61 (0.11), 38.3 | 0.20 (0.11), 42.7 |
| 512 / U | 0.61 (0.14), 37.0 | 0.24 (0.16), 40.1 |
| both 2048, 1024, 768, 512 | 44.7, 39.8, 38.2, 36.9 | -, 45.9, 42.0, 39.7 |

- **The drift stays a fraction of a level** after `split` (0.06 to 0.61 in the worst block),
  the colour study's 1080p range.
- **The encode's tiles cost more than the decode's**, 7 to 9 dB of PSNR-Y on the picture, 3 to
  7 on the clip: the DiT redraws from a latent the encode's tiles moved, where the decode's
  tiles only move the pixels they make.
- **The differences are not at the seams.** The overlap bands hold about their share of the
  squared error (0.10 to 0.45 of it in 0.07 to 0.38 of the pixels: 0.5 to 1.2 times their share
  at every side but the picture's 2048-px tiles, decode, encode or both, 1.8 to 2.0 times); the
  flattest segment's step across a zone (colour.md's measure) stays under 0.21 of a level, the
  largest over all segments 0.3 to 1.5, the encode's tiles at 512 and 768 px the most. The maps
  show the differences spread over the picture, strongest in its detail.
- No ground truth here: a difference from the untiled output is neither a loss nor a gain by
  itself; the colour study's judgement (its eyes found the tiles invisible after `split`) is the
  one made against a GT.

## Not measured

- **No real small card, no Windows-native run.** Every 8, 12 and 16 GB figure above is the RTX
  5090 under WSL2 with a ballast process holding the rest of the card: by day beside a desktop
  kept at 4 GiB, at night on the whole card with the desktop idle (its 1.9 GiB outside every
  figure). Only the memory is emulated: the card's compute, bandwidth and clocks are the 5090's,
  so a real small card is slower than these times. Which emulation a real 12 GB card matches is
  open.
- **Timings by day.** Runs of one configuration are 5 to 15% apart by day; at night, with the
  desktop idle, 0.1 to 0.4% (three jobs run twice each).
- **ComfyUI's sharp fp16 clip and its DiT phase on 8 GB.** The clip takes 24.5 GiB, beyond the
  desktop rule by day (its 3B template, with the same VAE, was measured instead); in the DiT
  phase comfy-aimdo reads the card itself, which a ballast by day cannot hide.
- **The lean attention through the whole pipeline.** The milestone-1 regression has not run with
  it, and its planner term (`PER_TOKEN` = 59,900 bytes) is today's scaled by 0.46, not refitted.
- **Why the tail's frames move at 512 and 576 px only.** The cause is not found (another cuDNN
  algorithm for the smaller shapes is the likely one, unproven).
- **ComfyUI's other bit-keeping VAE items.** Items 3 and 4 of the comparison above, and item 5
  beyond the decoder's tail a frame at a time: read, not built.
- **The lean windows' room on a real job past 19 latents**, and under cudaMallocAsync: the
  night's scans stop at 19 latents and at torch's own allocator.
- **The hang's cause.** One run of the night's hung in its decode with the GPU idle (the night's
  section); no Python stack was taken, and it did not come back.
- **The tiles against a ground truth.** The night's tile costs are against the untiled output;
  only the colour study judged tiles against a GT, at 1080p.
- **Three brought-up scripts.** `vae_room.py`, `tile_peaks.py` and `one_output_peaks.py` were
  brought up to the library and checked statically, not run on a GPU; `hold_probe.py`, brought
  up too, ran the scans above.
