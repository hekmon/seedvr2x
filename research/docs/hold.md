# The hold on a driver that spills: what torch's allocator needs beyond a peak, what a run holds outside torch's count, what the driver's figure counts

> Status: **measured** for the open question "The hold on a driver that gives memory the card
> doesn't have" of [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure), on one card: an RTX 5090
> (32 GB) under WSL2, Windows driver 616.92, torch 2.14.1+cu130, attention on PyTorch's SDPA,
> the 7B fp16 and the VAE, seedvr2x's own runtime (2026-10-12 and 13). Through
> [`scripts/hold_probe.py`](../scripts/hold_probe.py) (one process, one way of holding, a
> sequence of steps), [`scripts/card_watch.py`](../scripts/card_watch.py) (the card second by
> second, with Windows' own counters) and
> [`scripts/neighbour_probe.py`](../scripts/neighbour_probe.py) (a second program on the card).
> The card is a desktop's: its other programs held 3.2 to 4.6 GiB of it through the measures.

## Why

Windows' driver gives system memory where the card has none left, under WSL2 too, and a step
that passes the card slows down, then fails where nothing can catch it. seedvr2x holds itself
to the card there ([DESIGN.md](../../seedvr2x/DESIGN.md#allocator)): torch's own allocator
under a limit, the free memory less 0.5 GiB, and a plan that counts on 4.5 GiB less again,
"what torch's own allocator needs beyond a step's allocated peak". That one figure was sized on
the largest step measured; this asks what the need follows, step by step, so that a plan could
leave each step its own room.

## Method

**A step and its ladder.** `hold_probe.py` loads the models as a plan would have them (the DiT
whole, or its 36 blocks swapped; `--without vae` or `--without dit` takes one model's weights
off the GPU), limits torch `--under` GiB under the free memory (the smaller of what the driver
reports to the process and of what the whole card has by NVML), and runs steps: a DiT window of
L latents (random latents, seeds 42 and 43), a VAE encode of N frames or a decode of F latent
frames in tiles of T pixels, through the runtime's own functions. `r:STEP:FROM:BY:FINE` is a
ladder: the step once under the hold, for its allocated peak, then under that peak and a margin
going down from FROM GiB by BY, the cache emptied before each rung, to the first refusal (an
`OutOfMemoryError` torch raises at its limit, which is clean: the rungs after it are right);
then by FINE from the last rung that ran; then `--past` rungs more under the refused one,
whatever they do. `s:STEP:FROM:TO:BY` runs every rung whatever the others did. A step's need
is the smallest margin that ran with every rung above it. Each step's digest is compared with a
reference another process made (two processes that agree, held 2 to 4 GiB under the card).

**The card beside it.** `card_watch.py watch` reads the whole card by NVML every second and,
under WSL2, Windows' own counters through `typeperf.exe`: `GPU Adapter Memory` (dedicated,
shared, committed) and `GPU Process Memory` per process. nvidia-smi names no process there.
A Linux process's memory is in none of the per-process counters, so their sum is the desktop's
own: what the card's figure moved by, less what that sum moved by, is the run's. `hold_probe.py
--trace` writes both free figures and torch's count, read together, through the load and every
step.

**Nothing passes the card.** Every step runs under torch's limit, set under the card's own
figure; a ladder goes down, towards a refusal by torch.

## What torch's own allocator needs beyond a step's peak

torch's own allocator takes the driver's memory in segments, splits them for smaller requests
and gives a segment back only when it is whole again. Under a limit, a step whose cache is in
pieces is refused although its allocated peak is far under the limit: the room it needs beyond
that peak is what the plan must leave. The tables give it per configuration: a step, and what
else was on the card.

### The plain allocator

| window: size | step | tokens | beside it | allocated peak GiB | ran at / refused at (GiB over the peak), per process | past the refusal | s (usual) |
|---|---|---|---|---|---|---|---|
| 1080 | 1 | 8,280 | 36 blocks swapped | 2.690 | 0.2 / 0.15 |  | 1.8 |
| 1080 | 1 | 8,280 | 36 blocks swapped, the VAE away | 2.206 | 0.25 / 0.2 |  | 1.8 |
| 1080 | 1 | 8,280 | the models whole | 17.040 | 0.2 / 0.1; 0.2 / 0.1 |  | 0.9-0.9 |
| 1080 | 3 | 24,840 | 36 blocks swapped | 4.967 | 0.45 / 0.4 |  | 4.0 |
| 2160 | 1 | 32,880 | 36 blocks swapped | 6.016 | 0.6 / 0.5 |  | 5.5 |
| 2160 | 1 | 32,880 | 36 blocks swapped, the VAE away | 5.534 | 0.6 / 0.5 |  | 5.3 |
| 1080 | 5 | 41,400 | 36 blocks swapped | 7.070 | 0.7 / 0.6 |  | 5.6 |
| 1080 | 6 | 49,680 | 36 blocks swapped | 8.209 | 0.6 / 0.5 |  | 6.3 |
| 1080 | 6 | 49,680 | 36 blocks swapped, the VAE away | 7.725 | 0.6 / 0.5 |  | 6.8 |
| 1080 | 6 | 49,680 | the models whole | 22.558 | 0.6 / 0.5; 0.6 / 0.5 |  | 4.2-4.2 |
| 2160 | 2 | 65,760 | 36 blocks swapped | 10.484 | 0.7 / 0.6; 0.7 / 0.6 |  | 8.3-10.1 |
| 2160 | 2 | 65,760 | 36 blocks swapped, the VAE away | 10.001 | 0.7 / 0.6 |  | 8.3 |
| 1080 | 8 | 66,240 | 36 blocks swapped | 10.399-10.400 | 0.8 / 0.7; 0.8 / 0.6 |  | 6.1-7.7 |
| 1080 | 8 | 66,240 | the models whole | 24.750 | 0.8 / 0.7; 0.8 / 0.7 |  | 5.5-5.6 |
| 1080 | 10 | 82,800 | 36 blocks swapped | 12.409 | 1 / 0.9 |  | 9.9 |
| 2160 | 3 | 98,640 | 36 blocks swapped | 14.952 | 2.6 / 2.5 | ran at 2.4, 2.3, 2.2 | 12.1 |
| 2160 | 3 | 98,640 | 36 blocks swapped, the VAE away | 14.469 | 2.6 / 2.5 | ran at 2.4, 2.3, 2.2 | 12.6 |
| 1080 | 12 | 99,360 | 36 blocks swapped | 14.599-14.602 | 1.2 / 1.1; 1.2 / 1.1; 1.2 / 0.9 |  | 9.0-14.6 |
| 1080 | 12 | 99,360 | 36 blocks swapped, the VAE away | 14.116 | 1.2 / 1.1 |  | 12.8 |
| 1080 | 14 | 115,920 | 36 blocks swapped | 16.613 | 1.4 / 1.3 |  | 15.3 |
| 1080 | 14 | 115,920 | 36 blocks swapped, the VAE away | 16.129 | 1.4 / 1.3 |  | 15.9 |
| 2160 | 4 | 131,520 | 36 blocks swapped | 19.425-19.426 | 1.9 / 1.8; 1.9 / 1.8 |  | 15.4-17.4 |
| 2160 | 4 | 131,520 | 36 blocks swapped, the VAE away | 18.943 | 2 / 1.9 |  | 15.9 |
| 1080 | 16 | 132,480 | 36 blocks swapped | 18.803-18.806 | 3 / 2.9; 3 / 2.9 |  | 15.3-17.3 |
| 1080 | 16 | 132,480 | 36 blocks swapped, the VAE away | 18.319 | 3 / 2.9 |  | 15.9 |
| 1080 | 17 | 140,760 | 36 blocks swapped | 19.766-19.768 | 3.4 / 3.3; 3.4 / 3.3; 3.6 / 3.3 |  | 12.0-20.4 |
| 1080 | 17 | 140,760 | 36 blocks swapped, the VAE away | 19.284 | 2.6 / 2.5 |  | 17.3 |
| 2160 | 5 | 164,400 | 36 blocks swapped | 23.293 | - / 2.4; 4.5 / 4.4; 4.6 / 4.4 |  | 13.1-13.6 |
| 2160 | 5 | 164,400 | 36 blocks swapped, the VAE away | | first run refused under 26.407 GiB (22.72 allocated then) | | |
| 1080 | 21 | 173,880 | 36 blocks swapped | 23.967 | 3.3 / 3 |  | 15.2 |
| 1080 | 21 | 173,880 | 36 blocks swapped, the VAE away | | first run refused under 26.407 GiB (22.411 allocated then) | | |

| encode: size | step | largest tile (rows x columns), grid | beside it | allocated peak GiB | ran at / refused at (GiB over the peak), per process | past the refusal | s (usual) |
|---|---|---|---|---|---|---|---|
| 1080 | e33/1088 | 1088x1088, 2x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 13.348 | 1.8 / 1.7 (OTHER BITS at 2, 1.9, 1.8: the reference's down to 2.5) |  | 10.5 |
| 1080 | e33/1088 | 1088x1088, 2x2 | the DiT off the GPU | 12.325 | 1.8 / 1.7 (OTHER BITS at 2, 1.9, 1.8: the reference's down to 2.5); 1.8 / 1.7 (OTHER BITS at 2, 1.9, 1.8: the reference's down to 2.1) |  | 10.7-12.1 |
| 2160 | e17/1088 | 1088x1088, 3x4 | the DiT off the GPU | 12.325 | 1.5 / 1.4 |  | 20.6 |
| 1080 | e33/1344 | 1104x1344, 1x2 | the DiT off the GPU | 14.741 | 2.9 / 2.8 |  | 8.4 |
| 1080 | e33/1600 | 1104x1600, 1x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 17.961 | 2.9 / 2.8 |  | 9.1 |
| 1080 | e33/1600 | 1104x1600, 1x2 | the DiT off the GPU | 16.938 | 2.9 / 2.8 |  | 9.6 |
| 1080 | e33/1792 | 1104x1792, 1x2 | the DiT off the GPU | 18.911 | 3.3 / 3.2 |  | 10.3 |
| 1080 | e33/1856 | 1104x1856, 1x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 20.590-20.591 | 3.4 / 3.3; 3.5 / 3; 3.4 / 3.3 |  | 6.8-10.2 |
| 1080 | e33/1856 | 1104x1856, 1x2 | the DiT off the GPU | 19.568 | 5.2 / 5.1; 5.2 / 5.1 |  | 10.4-10.9 |
| 1080 | e33/1920 | 1104x1920, 1x1 | 36 blocks swapped (the DiT's other parts on the GPU) | 20.888 | 3.8 / 3.7; 4 / 3.5; 3.8 / 3.7 |  | 6.5-9.3 |
| 1080 | e33/1920 | 1104x1920, 1x1 | the DiT off the GPU | 19.866 | 3.8 / 3.7 |  | 9.3 |
| 2160 | e17/1472 | 1472x1472, 2x3 | 36 blocks swapped (the DiT's other parts on the GPU) | 21.279 | 5 / 4.5; 4.8 / 4.7 |  | 14.4-14.5 |
| 2160 | e17/1472 | 1472x1472, 2x3 | the DiT off the GPU | 20.254 | 4.4 / 4.3 |  | 22.1 |

| decode: size | step | largest tile (rows x columns), grid | beside it | allocated peak GiB | ran at / refused at (GiB over the peak), per process | past the refusal | s (usual) |
|---|---|---|---|---|---|---|---|
| 1080 | d9/832 | 832x832, 2x3 | 36 blocks swapped (the DiT's other parts on the GPU) | 13.842 | 2.5 / 2.4 |  | 19.7 |
| 1080 | d9/832 | 832x832, 2x3 | the DiT off the GPU | 12.819 | 2.7 / 2.6 |  | 19.5 |
| 2160 | d5/832 | 832x832, 3x5 | the DiT off the GPU | 12.819 | 3 / 2.9 |  | 41.3 |
| 1080 | d9/960 | 960x960, 2x3 | the DiT off the GPU | 16.193 | 4.1 / 4; 4.1 / 4 |  | 20.8-21.3 |
| 1080 | d9/1024 | 1024x1024, 2x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 18.988 | 4.1 / 4 |  | 20.9 |
| 1080 | d9/1024 | 1024x1024, 2x2 | the DiT off the GPU | 17.965 | 4.1 / 4 |  | 21.0 |
| 1080 | d9/1088 | 1088x1088, 2x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 21.188 | 5 / 4.9; 5 / 4.5; 5 / 4.9 |  | 15.7-23.6 |
| 1080 | d9/1088 | 1088x1088, 2x2 | the DiT off the GPU | 20.164 | 5 / 4.9 |  | 23.7 |
| 2160 | d5/1088 | 1088x1088, 3x4 | 36 blocks swapped (the DiT's other parts on the GPU) | 21.193 | 5.5 / 5; 5.1 / 5 |  | 33.1-33.5 |
| 2160 | d5/1088 | 1088x1088, 3x4 | the DiT off the GPU | 20.165 | - / 5.5; 5.9 / 5.8 |  | 46.4-47.6 |
| 1080 | d9/1152 | 1104x1152, 1x2 | 36 blocks swapped (the DiT's other parts on the GPU) | 22.576 | - / 5; 5.3 / 5.2 |  | 14.9-15.0 |

The 1080p decode in tiles of 1088 px (d9/1088) was judged against the digest it gave in its
own processes, where other passes' ladders had been refused before it, not against the
reference's ([A pass once refused](#a-pass-once-refused-and-the-bits-after-it)).

No step was slower at its smallest margin. Five things keep these figures from being a fit
by the kind of step:

- **It is no function of the tokens.** At 1080p a window needs 0.1 GiB a latent up to 14
  latents, then 3.0 at 16 and 3.4 at 17; at 4K, 98,640 tokens (3 latents) need 2.6 GiB where
  99,360 tokens need 1.2 at 1080p (12 latents), and 131,520 (4 latents) need 1.9 where 132,480
  need 3.0 (16 latents). The DiT's attention layouts step with a window's length in time
  (`planner.attention_layouts`), and no count of them follows the need either.
- **It is no threshold.** 4K's window of 3 latents was refused under its peak and 2.5 GiB, and
  ran under 2.4, 2.3 and 2.2. A scan of every margin from 3.0 GiB down to 0.3 by 0.1: it ran
  at 3.0 to 2.6 and at 2.4 to 1.6, and was refused at 2.5 alone, then at 1.5 and under: one
  hole, at the same margin in three processes, with the VAE's weights on the GPU or away.
- **What else is on the card moves it by over 1 GiB.** An encode at 1080p in tiles of 1856 px
  (1 x 2 tiles, 19.05 GiB of working memory) ran under its peak and 3.55 GiB with the DiT's
  parts on the GPU as 36 swapped blocks leave them, and was refused under its peak and 4.62
  with the DiT off the GPU, six times of six in three processes, first step of its process or
  not; the untiled encode the other way round (refused under 3.29 with the DiT's parts there,
  run under 3.76 without them). By their ladders: the encode in tiles of 1856 px needs 3.4
  GiB with the DiT's parts there and 5.2 with the DiT off the GPU, two processes each; the
  untiled one, larger, 3.8 either way. A window of 17 latents at 1080p needs 3.4 GiB with the
  VAE's 0.47 GiB of weights on the GPU and 2.6 with them away; most other steps need the same
  either way.
- **A pass short of room may run with other bits.** The encode in tiles of 1088 px (2 x 2)
  gave the reference's bits down to its peak and 2.1 GiB, other bits at 2.0, 1.9 and 1.8, in
  about the same time, and was refused at 1.7: three processes, the DiT's parts on the GPU or
  not. Every other step's rungs gave the reference's bits down to the last that ran. So a need
  is read as the smallest margin with the reference's bits, and "it ran" is no proof.
- **It is deterministic**, so none of this is noise: two processes give the same ladder to the
  rung, and the same allocated bytes when refused.

### The upper bound, and what a plan would count on under it

An upper bound over every configuration of the two nights (76 ladders, 53 configurations), each
need the smallest margin that ran with the reference's bits above the largest refused:

| kind | bound | through | the most it stands over a measured need |
|---|---|---|---|
| a window | 0.025 GiB + 28.55 KiB a token | 1 latent at 1080p with the VAE away (0.25 GiB), 5 latents at 4K (4.5) | 1.78 GiB |
| an encode | - 1.194 GiB + 3.12 GiB a megapixel of its largest tile | tiles of 1088 px (2.5), tiles of 1856 px with the DiT off the GPU (5.2) | 1.80 GiB |
| a decode | - 1.084 GiB + 5.90 GiB a megapixel | tiles of 832 px at 4K (3.0), tiles of 1088 px at 4K with the DiT off the GPU (5.9) | 1.12 GiB |

What a plan would count on of this card under it, each step left its own room (the planner on
the CPU), against the one figure of 4.5 GiB for every step:

| the card's free memory | job | a room by the kind of step | one figure of 4.5 GiB |
|---|---|---|---|
| 27.07 GiB | 1080p, 121 frames | 22.17 GiB: windows of 17 latents, 30 blocks swapped, encode in tiles of 1856 px, decode 1088 | 22.07: the same plan |
| | 1080p, 66 frames in two shots, windows of 9, tiles of 1024 and 832 px pinned | 24.52: 4 blocks swapped | 22.07: 10 swapped |
| | 4K, its smallest window (5 latents) | refused: it counts on 22.07 GiB for 22.87 | refused alike |
| 28.05 GiB | 1080p, 121 frames | 23.06: 17 latents, 28 swapped, encode untiled, decode 1088 | 23.05: the same plan |
| | 1080p, 66 frames in two shots, pinned | 25.50: no block swapped | 23.05: 8 swapped |
| | 4K, its smallest window | 23.05: 5 latents, 36 swapped, encode 1472, decode 1088 | the same |

So a room by the kind of step gives a plan of long windows nothing to speak of (0.01 to 0.10
GiB), a job of short windows 2.5 GiB, and the largest step nothing: the largest steps are the
points that make the bound. And the bound holds over what was run alone: the last ladder of
the night, a 4K decode in tiles of 1088 px with the DiT off the GPU, needed 5.9 GiB where 5.1
with the DiT's parts there, and moved the decode's bound; a VAE pass is left 6.0 today.

### Expandable segments

`PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True` makes each of the
allocator's pools one range of addresses that grows and shrinks by mapping. It works under
WSL2, torch's limit holds under it as before, and it turns the cliff into a slope:

| window: size | step | tokens | beside it | allocated peak GiB | ran down to / refused at | at its usual speed from | s there / s at the least margin |
|---|---|---|---|---|---|---|---|
| 1080 | 1 | 8,280 | 36 blocks swapped | 2.668 | 0.1 / 0.05 | 0.4 | 1.7 / 2.3 |
| 1080 | 6 | 49,680 | 36 blocks swapped | 8.187 | 0.1 / 0 | 0.7 | 7.4 / 27.3 |
| 2160 | 2 | 65,760 | 36 blocks swapped | 10.463 | 0.1 / 0 | 1 | 11.5 / 38.9 |
| 1080 | 12 | 99,360 | 36 blocks swapped | 14.577 | 0.1 / 0 | 1.7 | 11.1 / 46.3 |
| 1080 | 14 | 115,920 | 36 blocks swapped | 16.589 | 0.1 / 0 | 1.5 | 13.5 / 44.8 |
| 2160 | 4 | 131,520 | 36 blocks swapped | 19.398 | 0.1 / 0 | 2.5 | 14.0 / 64.6 |
| 1080 | 16 | 132,480 | 36 blocks swapped | 18.778 | 0.1 / 0 | 2.5 | 14.1 / 41.3 |
| 1080 | 17 | 140,760 | 36 blocks swapped | 19.740 | 0.1 / 0 | 2 | 15.2 / 51.3 |
| 2160 | 5 | 164,400 | 36 blocks swapped | 23.269 | 0.1 / 0 | 2.3 (the hold allowed 2.8 GiB at the most) | 20.1 / 65.9 |
| 2160 | 5 | 164,400 | 36 blocks swapped, the VAE away | 22.802 | 0.1 / 0 | 2.3 | 20.7 / 90.3 |
| 1080 | 21 | 173,880 | 36 blocks swapped | 23.941 | 0.1 / 0 | 1.2 (the hold allowed 2.8 GiB at the most) | 66.0 / 94.3 |

| encode: size | step | largest tile (rows x columns), grid | beside it | allocated peak GiB | ran down to / refused at | at its usual speed from | s there / s at the least margin |
|---|---|---|---|---|---|---|---|
| 1080 | e33/1088 | 1088x1088, 2x2 | the DiT off the GPU | 12.306 | 0 / - (OTHER BITS at 0: the reference's down to 0.5) | 2 | 11.3 / 49.0 |
| 1080 | e33/1792 | 1104x1792, 1x2 | the DiT off the GPU | 18.892 | 0 / - (OTHER BITS at 0: the reference's down to 0.5) | 3.5 | 14.6 / 62.3 |
| 1080 | e33/1920 | 1104x1920, 1x1 | the DiT off the GPU | 19.849 | 0.3 / 0.2 (OTHER BITS at 0.4, 0.3: the reference's down to 0.5) | 3.5 | 18.8 / 59.0 |

| decode: size | step | largest tile (rows x columns), grid | beside it | allocated peak GiB | ran down to / refused at | at its usual speed from | s there / s at the least margin |
|---|---|---|---|---|---|---|---|
| 1080 | d9/1024 | 1024x1024, 2x2 | the DiT off the GPU | 17.949 | 0.3 / 0.2 (OTHER BITS at 0.4, 0.3: the reference's down to 0.5) | not reached on its ladder (24.9 s with 9.2 GiB, all the hold allowed) | 37.7 / 95.1 |
| 1080 | d9/1088 | 1088x1088, 2x2 | the DiT off the GPU | 20.147 | 0.4 / 0.3 (OTHER BITS at 0.4: the reference's down to 0.5) | not reached on its ladder (27.0 s with 7.0 GiB, all the hold allowed) | 37.8 / 100.3 |

- A step short of room **runs, slower**: every window ran under its allocated peak and 0.1
  GiB, 3 to 4 times slower there, and was refused under the peak itself; at its usual speed
  from about the margin the plain allocator needs to run at all.
- 4K's window of 5 latents, which the plain allocator runs under its peak and 4.5 GiB, ran
  beside a desktop holding 3.2 to 4.3 GiB of the card: with the VAE's weights away, 18 s a
  forward with 4.3 GiB over its peak, 21 with 2.8, 22 with 2.3, 59 with 1.8 and 79 to 90 from
  1.3 down to 0.1.
- A window of 21 latents at 1080p, which the plain allocator could not run beside that
  desktop, ran with all the hold allowed, 2.8 GiB over its peak, and down to 0.1: 40 to 52 s
  a forward at the most room, 56 to 94 under it, where 16 latents take 14: the card had no
  room for its usual speed.
- The times are a rung's: one forward from an emptied cache. A second forward, the cache
  kept, is faster (4K's window of 5: 14 s for 22).
- A VAE pass wants more room for its usual speed than a window: an encode in tiles of 1088 px
  2.0 GiB, in tiles of 1792 px 3.5; a decode takes 3.0 to 4.7 GiB beyond its peak when it is
  allowed them, and 1.4 to 1.5 times as long with 4.0.
- The bits are the plain allocator's, and the night before's, down to 0.5 GiB over a pass's
  peak. Under that a pass may still run, with other bits: at 0.4 and 0.3 GiB (the untiled
  encode, the decode in tiles of 1024 px), at the peak itself (the encodes in tiles).

Whole jobs, seedvr2x's command line holding itself as it does, each pair on one plan (the
setting given in the environment, or by default):

| job | allocator | frames | upscaled in | torch held at its checks, at the most | the card's least free |
|---|---|---|---|---|---|
| 66 frames at 1080p in two shots, windows of 9 latents, tiles of 1024 and 832 px | plain | the reference's | 80.9 s; 89.5 s on another plan | 25.24; 26.50 GiB | 956; 1,165 MiB |
| | expandable segments | the same | 83.5 s; 95.7 and 99.5 s | 22.11; 23.38 | 4,271; 4,438 and 3,914 MiB |
| 121 frames at 1080p, windows of 17, encode untiled, decode in tiles of 1088 px | plain | | 151.5 s | 27.08 | 570 MiB |
| | expandable segments | the same | 149.1 s | 24.43 | 2,212 MiB |

A job of six shots in a row (18 units) ran as it does under the plain allocator, torch's count
going down again between units. And with 2.5 GiB planned under torch's limit in place of 4.5,
4K's smallest window (5 latents, 17 frames) is planned beside a desktop holding 3.2 GiB of
the card, where 4.5 refuses it, and runs: four runs, the same frames, which are those of a run
under the plain allocator with over 5 GiB of room the night before (the 17 frames' CRC32); in
134 to 139 s where that run took 65.6. A job that fitted already then runs 1.4 to 1.5 times
as long (216 s for 152 over 121 frames, 307 for 207 in six shots), its plan packed tighter:
fewer swapped blocks, larger tiles, every pass squeezed; and with other frames than its run
under 4.5, the plan being another.

### A pass once refused, and the bits after it

The decode at 1080p in tiles of 1088 px gave one digest in every process where no pass had
been refused before it (five processes over the two nights, under both allocators), and another
in three processes where other passes' ladders had gone down to their refusals first: from
its very first run there, with 5.9 to 7.0 GiB of room. One refused rung of a neighbouring
decode didn't do it. It fits torch keeping, for a convolution's shape, the engine it fell
back to when the first one's workspace was refused, for the life of the process; that wasn't
verified in torch's source. A reference is therefore made in a process where nothing was
refused, and a run is judged by its frames.

## What a run holds of the card outside torch's count, and the desktop's own movement

**The desktop alone.** 56.5 minutes with nothing of ours on the card, at night (3,391 readings
by NVML): its used memory stayed between 4,378 and 4,474 MiB, 4,378 at the start and 4,384 at
the end; 118 steps of 1 MiB or more (125 an hour), as many up as down, 1.4 MiB at the median;
one of 95 MiB up, given back within the minute. Within 1, 5 or 15 minutes it rose by 96 MiB at
the most and fell by 89 to 91. Over the whole night by Windows' per-process counters, the runs
going on beside (they count none of a WSL process's memory): a burst of 55 to 114 MiB given
back in 10 to 25 s about every 35 minutes, and two falls that stayed, of 152 and 250 MiB;
within 5 minutes up by 116 MiB at the most, down by 261. With its user at work the evening
before, the same desktop had moved by 0.44 to 0.58 GiB within a 4-minute job.

**The run's own.** From the reading a plan would be made at: what the card lost, less what
torch reserved, less what the desktop's processes moved by. It is 0.013 to 0.014 GiB with 36
blocks swapped and 0.001 to 0.002 with nothing pinned, at every step, under both allocators.
Where the raw figure (card less torch's count) read 0.020 to 0.065 GiB, the desktop had grown
by 7 to 67 MiB; where it read 0.10 to 0.23 GiB under zero, the desktop had fallen by 152 to
250. The 0.019 to 0.102 GiB read the night before were the desktop's.

## What the driver's free figure counts near a full card

Torch held 0.25 GiB under the card's own figure, windows run one after the other with no
cache emptied, so that torch reserves up to its limit; both free figures and torch's count
read about 4 times a second (4.3 on average), and kept where torch's count had not moved
around the reading.
Alone (limit 26.82 GiB; reserved 26.62 at the most, the card down to 0.77 GiB free) and beside
a second program holding 3 GiB and working on it (limit 23.53; the card down to 0.81). Every
window ran, with the reference's bits.

| | the rest of the card (all but torch's count) | the driver's figure + torch's count, while the card has room | near the full card | the driver's figure less the card's there |
|---|---|---|---|---|
| alone | 4.35, then 4.05 GiB | 30.05 to 30.07 GiB (card free 27.05 down to 9.69) | 28.53 to 28.56 (card free 2.42 to 1.91) | + 1.14 to + 1.17 |
| beside a program holding 3 GiB | 7.64 to 7.66, then 7.33 | 30.05 to 30.07 (22.78 down to 7.50) | 24.89 to 24.95 (5.73 to 0.82) | + 1.14 to + 1.18, then + 0.78 to + 0.85 |
| the ladders' runs earlier that night, a desktop of 4.4 to 4.7 GiB | 4.90 to 5.26 | 30.05 to 30.07 (down to 5.11) | 28.31 to 28.56 (5.11 down to 0.63) | + 1.9 to + 2.3 |
| a job's checks the evening before, a desktop of 5.2 to 5.9 GiB | | | | + 2.3 to + 2.8 |
| a job's checks the night before, a desktop of 1.0 GiB | | | | - 0.33 to - 0.83 |

While the card has more than about 5 to 7 GiB free, the driver offers the process a fixed
budget, 30.06 GiB here, less the process's own use, whatever anything else holds. Under that
it offers about what the card has free and, with it, the part of the Windows programs' memory
over some 2.4 to 3.1 GiB; a second WSL program's memory is counted whole. And Windows does
take that part back: the desktop's processes lost 455 MiB of dedicated memory at once as the
card went down to 0.77 GiB free, never passed, and 330 MiB more in the run beside the second
program. So near a full card the driver's figure counts memory it would push off the card: no
figure to plan in, nor to check a unit on.

## The check before a unit, judged by what the next unit needs

Computed from the logs of 33 command-line runs of the two nights (127 checks), nothing built.
The check before a unit stops the run when the card has less free than torch may still take
under its limit. Judged by the next unit alone, it would stop when the card has less free than
`min(limit, P + room) - reserved`, P the unit's predicted peak. That rule can only give back:
it never stops a run the present check passes. With the room the plan already leaves (4.5 GiB
for a window, 6.0 for a VAE pass) it gives back the plan's own slack under its budget, 0.03 to
0.43 GiB on the evening's jobs, and 4 of the 5 stops of the two nights would have passed; the
fifth, a 2 GiB program arriving, rightly still stops. It is not safe alone: torch keeps what
it has used (it stood at 0.58 to 0.999 of its limit at every check after the first), and in 12
of 93 units it took more than that rule counted. It would need torch's limit lowered for the
unit, or the cache emptied first; and from the plan a peak per unit, where the plan has three
maxima.

## Caveats

- One card, one driver, one version of torch, SDPA; a desktop that held 3.2 to 4.6 GiB of the card:
  a window of 21 latents at 1080p and 4K's of 5 could not be run under the plain allocator
  (last night's figures for them, beside a lighter desktop, are in the tables).
- The times are a ladder's: the cache emptied before each rung and two sampler threads in the
  process, 30 to 40% over a forward in a job.
- "At its usual speed from" is read on rungs 0.5 GiB apart.
- Not measured: FlashAttention 2; a driver that refuses (the allocator there is
  cudaMallocAsync); tiles under 832 px; other versions of torch.

## Reproduce

```sh
# seedvr2x's venv python, one process at a time; MODELS holds the 7B fp16 and the VAE
P="python research/scripts/hold_probe.py --dit $MODELS/dit.safetensors --vae $MODELS/vae.safetensors --allocator native"
# references, far under the limit: the first process writes, the second is compared
$P --placement 36 --under 4.0 --trim --steps 1,6,8,12,14,16 --reference ref.json --write-reference ref.json --out ref-a.jsonl
$P --placement 36 --under 4.0 --trim --steps 1,6,8,12,14,16 --reference ref.json --out ref-b.jsonl
# ladders: 1080p windows, the VAE's weights on the GPU, then away; 4K with --size 2160
$P --placement 36 --under 0.5 --trim --past 3 --steps r:6:1.0:0.1,r:12:2.2:0.5:0.1,r:17:4.6:0.5:0.1 --reference ref.json --out w.jsonl --trace w.trace.jsonl
$P --placement 36 --without vae --under 0.5 --trim --past 3 --steps r:12:2.2:0.5:0.1 --reference ref.json --out wv.jsonl
# VAE passes, the DiT off the GPU, then its parts on it (--placement 36)
$P --without dit --under 0.5 --trim --past 3 --steps r:e33/1856:6.0:0.5:0.1,r:d9/1024:6.5:0.5:0.1 --reference ref.json --out v.jsonl
# a scan: every rung whatever the others did
$P --size 2160 --placement 36 --under 0.5 --trim --steps s:3:3.0:0.3:0.1 --reference ref.json --out scan.jsonl
# the same under expandable segments
$P --conf expandable_segments:True --placement 36 --under 0.5 --trim --steps r:17:3.5:0.5:0.1 --reference ref.json --out xw.jsonl
# the card filled to its limit, traced, alone and beside a second program
$P --placement 36 --under 0.25 --steps 1,6,12,16,17,17,12,6,1 --reference ref.json --out c.jsonl --trace c.trace.jsonl --trace-every 0.03
python research/scripts/neighbour_probe.py holder 3 900 active &
# the card second by second, with Windows' counters under WSL2; then its summary
python research/scripts/card_watch.py watch --out card.jsonl --windows win.csv --names names.csv
python research/scripts/card_watch.py summary card.jsonl --since 02:00:00 --until 03:00:00
```
