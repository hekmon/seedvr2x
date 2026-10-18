# The VAE's room: what a VAE pass needs between its peak and the free memory, and what cures it

> Status: **measured** (2026-10-13; the floor by tile size and its cause, 2026-10-16; under
> expandable segments on a card that gives memory it doesn't have, 2026-10-17; the large tiles
> under the one-output cure and the tiles left, 2026-10-17; whole runs in large tiles under the
> cure, 2026-10-17) for the
> open question "The VAE's room" of
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). seedvr2x's planner holds every VAE pass 1.5 GiB
> further under its budget than a DiT window ([DESIGN.md](../../seedvr2x/DESIGN.md#memory-planner),
> Budget), because passes packed to the 0.6 GiB margin ran out of memory on 2026-10-11, at one
> large request, the card full while the allocator's pool held more than was asked. This is the
> study of that room: seedvr2x's own command line on cards emulated by
> [`scripts/vram_cap.py`](../scripts/vram_cap.py), the planner's room set in the research driver,
> never in the product, by [`scripts/vae_room.py`](../scripts/vae_room.py). One GPU (RTX PRO 6000
> Blackwell 96 GB, torch 2.14.1+cu130, cuDNN 9.24, flash-attn 2.8.3, `cudaMallocAsync`), 7B fp16,
> the colour correction on (`split`), seedvr2x as it stood before the changes this study leads to.
> One section on a second GPU (RTX 5090 under WSL2, torch's own allocator with expandable
> segments, the room set by torch's limit), by [`scripts/held_floor.py`](../scripts/held_floor.py).

In short:

- **The frames move with the memory's pressure.** A VAE pass left too little memory beyond its
  peak still runs, slower, with other frames than the same pass with room to spare, and not the
  same ones twice. Shown on one convolution: left too little memory, it runs 8.5 times slower
  and gives other bits, without a word, and gave the same other bits again with the memory free,
  in the same process (three calls): another cuDNN engine taken, as torch's source and the
  cuDNN frontend's log showed since ([the convolution alone](#one-convolution-alone),
  [the cause](#the-cause-in-torchs-source)). A decode in 640-px tiles keeps its frames
  with 0.90 GiB left and loses them with 0.80; one in 832-px tiles keeps them with 1.08 and loses
  them with 0.89; one in tiles of 1104 × 1152 keeps them with 0.94
  ([the floors](#the-floor-is-the-frames)). **The room's floor is where the frames
  hold, not where the run fails.**
- **At room 0 the large passes fail, every time:** a 1080p decode in 1600-px tiles 3 times of 3
  and a 4K decode in 1664-px tiles 2 of 2 (and alone), at the concatenation of a split
  convolution's pieces, 3.37 and 5.28 GiB asked in one piece while the pool held 4.3 and 6.0 GiB
  more than was allocated and the card was full. The small ones run (and move their frames).
- **Of four cures, one holds on both:** the pieces written into one output allocated before the
  first of them. Emptying the pool on entering the phase (which seedvr2x does already), before
  the request or before every slice doesn't save the 4K decode. The cure moves no bit. It takes
  0.80 to 2.85 GiB off the large passes' peaks and puts 0.04 to 0.50 on the small ones', so it is
  kept for outputs of 2.5 GiB or more.
- **By tile size, a pass alone**
  ([the floor by tile size](#the-floor-by-tile-size-and-its-cause-2026-10-16)): a decode in
  tiles of 640 to 832 px keeps its frames with 0.70 to 0.82 GiB left and moves them with 0.1
  less, an encode its latent with 0.72 at the most; from 1,024 px a decode is refused at a split
  convolution's concatenation (0.60 to 1.00 GiB left) before its frames move, and keeps them
  above it, with 1.10 at the most. **The cause, read in torch's source:** a convolution whose
  workspace the allocator refuses runs another cuDNN engine, taken without a word, and keeps it
  for its shape for the life of the process. torch counts no refused allocation under
  `cudaMallocAsync` (its counters 0 on every pass); the cuDNN frontend's log tells it (20 or 40
  plans more in every pass whose frames moved, none in any that kept them).
- **Under the one-output cure the large tiles' frames move where the code without it refused them
  first**
  ([the large tiles under the cure](#the-large-tiles-under-the-one-output-cure-and-the-tiles-left-2026-10-17)).
  Every 1080p and 4K decode tile it takes that ran (26 sizes, from 1,104 × 1,216 to 4K's 2,048 px)
  and 1080p's untiled decode kept their frames with 0.68 to 0.97 GiB left and moved them with 0.1
  less; none was refused. Its 4K encode squares that ran (1,664 to 2,048 px) kept their latent with
  0.50 to 0.60. For the same tile a pass under the cure asks 0.20 to 4.97 GiB less free memory, its
  peak lower. Not run under the cure: 4K's untiled encode (the 96 GB card's own 4K plan encodes
  untiled) and the other output sizes' passes it takes (1440p's tiles cut off at its 1,472 rows,
  its untiled decode and encode, among them). On the code without it, the tiles left: decodes of
  768 to 960 px keep their frames with 0.72 to 0.81 GiB, the 1,088-px one is refused first (kept
  with 0.90), and the encodes run keep their latent with 0.47 to 0.87.
- **In six tiles, whole runs keep their frames where the passes alone do, within 0.2 GiB either
  way** ([whole runs](#whole-runs-in-large-tiles-under-the-one-output-cure-2026-10-17)): six decode
  tiles on the cure's code (1080p's 1,104 × 1,472, 1,104 × 1,792, its untiled decode and its 832-px
  tile; 4K's 1,600 and 2,048 px), the encode, the windows and the decode in one process, kept their
  frames with 0.77 to 1.08 GiB left, beside passes alone that kept them with 0.79 (832's, alone on
  6e1be76) to 0.97: 4K's 1,600-px tile with 0.2 less than alone, 1080p's untiled decode with 0.2
  more. Near that floor a whole run doesn't repeat: five of the twelve cards run twice gave two
  outcomes, where every pass alone's pair agreed.
- **Under expandable segments, where the driver gives memory the card doesn't have**
  ([on an RTX 5090](#under-expandable-segments-on-an-rtx-5090-2026-10-17)): the same passes keep
  their frames with 0.40 GiB beyond their peak and lose them with 0.35 (decodes in 832- and
  1,024-px tiles, which keep them with 0.79 and 0.75 under the 96 GB card's `cudaMallocAsync`),
  and whole runs as the passes alone. torch's own allocator counts every refused workspace there:
  `num_ooms` 0 across every pass that kept its frames, 19 to 21 across every one that lost them.
- **For seedvr2x:** that cure for large outputs; the room stays until its floor is measured by
  the frames, tile size by tile size ([What follows](#what-follows-for-seedvr2x)).

## Method

**The launcher.** `vae_room.py run` is one seedvr2x run in the launcher's own process. In order:
the emulated card (`vram_cap.py`'s ballast, a driver-level allocation held to the end, so that
the card is really full where a card of N GB would be); the planner's constants, set in the
process before the plan is made (`planner.VAE_ROOM`, and `ENCODE_EXTRA` for the last section);
a cure, by a patch of the process; the probes; then seedvr2x's command line, unchanged. Nothing
of seedvr2x is edited: the code under study is one frozen copy, its tree hash checked before
every run. Every run is one hold of the GPU's lock, alone on the card.

**The plans.** The planner's own, for the matrix's cases
([`scripts/planner_matrix.py`](../scripts/planner_matrix.py): a shot of two windows at the cap,
of one window and a latent, or of one window, cut from one 1080p source), on cards emulated at
12 to 48 GB. With the room at 0 the planner packs each VAE pass to the budget, the free memory
less 0.6 GiB: the largest tile that fits, then as many of the DiT's blocks beside it as fit.

**A pass alone.** A run stopped by its own stop before its decode leaves its latent and its
windows; resumed, it runs the decode alone, in a new process. And a run stopped after its
encode ran that pass alone. Used where said: a pass alone doesn't always fail as it does in the
whole run.

**The probes,** in the run's process, none of which allocates on the GPU:

- per unit (a shot's encode, a window, a decode) and per slice of a VAE pass (the VAE takes a
  shot 4 frames at a time, a decode a latent at a time): what torch has allocated and what its
  pool holds (`memory_allocated`, `memory_reserved`: under `cudaMallocAsync` torch reads both
  from the driver's memory pool, its used and its reserved bytes), their peaks, the driver's
  free memory (`cudaMemGetInfo`) and the card's by NVML, nvidia-smi's figure, sampled every 20 ms
  by a thread of its own;
- every concatenation of 0.25 GiB or more in the VAE's causal convolutions, by call site and
  size: how many, the least memory the driver had free before one, and the most torch had
  allocated once one was given: the site where that is the pass's peak is where the pass peaks;
- a request refused: its call site, its size, the traceback's frames, torch's own message,
  `memory_stats`, and the card's free memory over the second before.

What `cudaMallocAsync` doesn't tell: the largest free block of its pool. What it holds beyond
what is allocated is `reserved - allocated`; torch's own retry on a refused request ("recovered
from an allocation failure by trimming the pool and retrying") is counted from the log.

**The cures,** each a patch of the launcher's process:

| Cure | What it does |
|---|---|
| `enter` | the pool emptied (`torch.cuda.synchronize`, then `empty_cache`) when a unit enters a VAE phase |
| `request` | the pool emptied before each concatenation of a split convolution's pieces of 1 GiB or more |
| `slice` | the pool emptied before every slice of a VAE pass |
| `output` | a split convolution's pieces written into one output allocated before the first of them is computed, where the vendored code concatenates them once all are made |

The first, the second and the fourth are DESIGN.md's; the third was added when the first two
didn't hold.

**The convolution that asks.** The VAE's causal convolutions are loaded with a memory limit of
0.5 GiB (`conv_max_mem`): one whose padded input is that large is cut along the rows in pieces
under the limit (`InflatedCausalConv3d.memory_limit_conv`), each piece convolved, and the
pieces' outputs concatenated, `torch.cat(x, split_dim)`. That concatenation is the request that
fails: a whole frame's or tile's output for one slice, asked for in one piece while the pieces,
as large together, are still there. Its size is the output's: 128 channels × 4 frames × the
tile's pixels × 2 bytes in an encode (2.02 GiB for 1104 × 1920), 256 channels in a decode (2.43
GiB for a tile of 1104 × 1152, 3.37 for 1104 × 1600, 5.28 for 1664 × 1664).

## Results

Peaks, predictions and what is left are in GiB (2³⁰ bytes); "left" is the free memory the plan
was made in less the pass's peak; "retries" counts torch's "recovered from an allocation
failure" lines in the run's log; "frames off the reference" counts the frames whose checksum
differs from the reference run's, the same shape pinned with room to spare (the bare card, the
room as shipped, uncured), where one was made. The tables are `vae_room.py`'s records through a
script, one row a run.

### At room 0

The plans at room 0, uncured, whole runs (the three that failed on 2026-10-11, then the others run):

| run | card | room | cure | encode | predicted / peak / left | decode | predicted / peak / left | decode s | retries | result | md5 | frames off the reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a-1080-24-r0-1 | 24 | 0.0 | none | untiled, DiT as in its phase | 21.78 / 20.871 / 1.64 | 1152 px, 0 blocks | 21.56 / 21.567 / 0.94 | 429.8 | 420 | exit 0 | cb022a78 | 0 of 149 |
| a-1080-32-r0-1 | 32 | 0.0 | none | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 (refused) / 0.92 | 167.0 | 257 | REFUSED in the decode, slice 18, at memory_limit_conv:162: 3.37 asked, 26.21 allocated, pool 30.53, card free 5 MiB |  | no frames |
| a-2160-48-r0-1 | 48 | 0.0 | none | 2048 px, 16 blocks | 45.59 / 45.185 / 1.32 | 1664 px, 0 blocks | 45.82 / 43.089 (refused) / 3.42 | 28.2 | 85 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.55 allocated, pool 46.53, card free 7 MiB |  | no frames |
| a-1080-32-r0-2 | 32 | 0.0 | none | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 (refused) / 0.92 | 148.3 | 257 | REFUSED in the decode, slice 16, at memory_limit_conv:162: 3.37 asked, 26.21 allocated, pool 30.53, card free 5 MiB |  | no frames |
| a-2160-48-r0-2 | 48 | 0.0 | none | 2048 px, 16 blocks | 45.59 / 45.185 / 1.32 | 1664 px, 0 blocks | 45.82 / 43.089 (refused) / 3.42 | 28.2 | 82 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.55 allocated, pool 46.53, card free 7 MiB |  | no frames |
| a-1080-32-r0-3 | 32 | 0.0 | none | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 (refused) / 0.92 | 241.6 | 313 | REFUSED in the decode, slice 26, at memory_limit_conv:162: 3.37 asked, 26.21 allocated, pool 30.53, card free 5 MiB |  | no frames |
| c-1080-16-n-r0-1 | 16 | 0.0 | none | 1152 px, 0 blocks | 13.60 / 13.207 / 1.30 | 832 px, DiT as in its phase | 13.82 / 13.825 / 0.68 | 266.4 | 448 | exit 0 | d939ad83 | 80 of 85 |
| c-1080-12-n-r0-1 | 12 | 0.0 | none | 896 px, 0 blocks | 9.48 / 9.082 / 1.43 | 640 px, DiT as in its phase | 9.61 / 9.613 / 0.90 | 157.0 | 248 | exit 0 | 9b003b65 | 0 of 53 |
| c-2160-24-of-r0-1 | 24 | 0.0 | output from 2.5 GiB | 1472 px, DiT as in its phase | 21.66 / 21.258 / 1.25 | 1088 px, DiT as in its phase | 21.17 / 21.173 / 1.34 | 210.8 | 134 | exit 0 | 1c50d0d5 | no reference |
| c-1080-12-n-r0-2 | 12 | 0.0 | none | 896 px, 0 blocks | 9.48 / 9.082 / 1.43 | 640 px, DiT as in its phase | 9.61 / 9.613 / 0.90 | 157.2 | 261 | exit 0 | 9b003b65 | 0 of 53 |

- The 24 GB plan's two passes are the ones that failed on 2026-10-11, to the hundredth: its
  untiled encode peaks at 20.87 GiB, where the first failure stood at 18.85 allocated and 2.02
  asked. It runs today. What changed since: no pass keeps anything of the shot on the GPU, and the
  free memory is read after the libraries have loaded (0.29 GiB that the earlier readings didn't
  show).
- The two that fail, fail at the pass's peak: the pieces of a split convolution and their
  concatenation, both whole. The 1080p one passed that moment 15 to 25 times before failing,
  never at the same slice; the 4K one fails the first time it reaches it. Run alone, in a new
  process, the 1080p decode ran once; the 4K one failed as in the whole run.
- An encode's predicted peak holds 0.4 GiB that a measured tile doesn't need (0.39 to 0.40 under
  its prediction for tiles of 896, 1104 × 1152, 1472 and 2048 px): at room 0 an encode is left
  1.0 GiB or more, and none failed.

### The cures

The cures on the two plans that fail (`-s-`: the decode alone, by a resume; `output from 2.5 GiB`: for outputs that large alone):

| run | card | room | cure | encode | predicted / peak / left | decode | predicted / peak / left | decode s | retries | result | md5 | frames off the reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b-1080-32-s-none-1 | 32 | 0.0 | none |  |  | 1600 px, 0 blocks | 29.58 / 29.583 / 0.93 | 325.2 | 137 | exit 0 | 20a890f7 | compared with (packed) |
| b-1080-32-s-output-1 | 32 | 0.0 | output |  |  | 1600 px, 0 blocks | 29.58 / 28.782 / 1.73 | 324.5 | 65 | exit 0 | 20a890f7 | 0 of 113 |
| b-1080-32-output-1 | 32 | 0.0 | output | untiled, 21 blocks | 29.80 / 29.188 / 1.32 | 1600 px, 0 blocks | 29.58 / 28.789 / 1.72 | 324.7 | 235 | exit 0 | 20a890f7 | 0 of 113 |
| b-1080-32-slice-1 | 32 | 0.0 | slice | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 / 0.92 | 327.6 | 309 | exit 0 | 20a890f7 | 0 of 113 |
| b-2160-48-s-none-1 | 48 | 0.0 | none |  |  | 1664 px, 0 blocks | 45.82 / 43.085 (refused) / 3.42 | 27.7 | 4 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.54 allocated, pool 46.53, card free 9 MiB |  | no frames |
| b-2160-48-output-1 | 48 | 0.0 | output | 2048 px, 16 blocks | 45.59 / 43.757 / 2.75 | 1664 px, 0 blocks | 45.82 / 42.981 / 3.53 | 496.3 | 88 | exit 0 | 8360c2bd | compared with (packed) |
| b-2160-48-s-slice-1 | 48 | 0.0 | slice |  |  | 1664 px, 0 blocks | 45.82 / 43.085 (refused) / 3.42 | 28.1 | 4 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.54 allocated, pool 46.53, card free 9 MiB |  | no frames |
| b-2160-48-s-request-1 | 48 | 0.0 | request |  |  | 1664 px, 0 blocks | 45.82 / 43.085 (refused) / 3.42 | 28.2 | 2 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.54 allocated, pool 46.53, card free 9 MiB |  | no frames |
| b-2160-48-s-enter-1 | 48 | 0.0 | enter |  |  | 1664 px, 0 blocks | 45.82 / 43.085 (refused) / 3.42 | 28.0 | 4 | REFUSED in the decode, slice 2, at memory_limit_conv:162: 5.28 asked, 40.54 allocated, pool 46.53, card free 9 MiB |  | no frames |
| b-1080-32-request-1 | 32 | 0.0 | request | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 / 0.92 | 328.4 | 212 | exit 0 | 20a890f7 | 0 of 113 |
| b-1080-32-enter-1 | 32 | 0.0 | enter | untiled, 21 blocks | 29.80 / 28.890 / 1.62 | 1600 px, 0 blocks | 29.58 / 29.590 (refused) / 0.92 | 167.1 | 281 | REFUSED in the decode, slice 18, at memory_limit_conv:162: 3.37 asked, 26.21 allocated, pool 30.53, card free 5 MiB |  | no frames |
| b-1080-32-output-2 | 32 | 0.0 | output | untiled, 21 blocks | 29.80 / 29.188 / 1.32 | 1600 px, 0 blocks | 29.58 / 28.789 / 1.72 | 324.5 | 261 | exit 0 | 20a890f7 | 0 of 113 |
| b-2160-48-s-of-1 | 48 | 0.0 | output from 2.5 GiB |  |  | 1664 px, 0 blocks | 45.82 / 42.810 / 3.70 | 496.3 | 33 | exit 0 | 8360c2bd | 0 of 41 |

- These two shapes have no reference with room to spare: their rows are compared with a packed
  run of their own (b-1080-32-s-none-1, b-2160-48-output-1, both at room 0), which says that a
  cure moves no bit against that run, not that its frames are a bare card's.
- `enter` is what seedvr2x's placement does already when weights leave: the pool holds 0.6 GiB
  as the failing decodes start. With it or without, they fail.
- `request` and `slice` each saved the 1080p plan once, and not the 4K decode: its first slice at
  full size asks 5.28 GiB with 40.5 allocated on a card of 46.5, and the pool, emptied or not,
  can't give it in one piece.
- `output` saves both, with the frames of the concatenation where it ran (the 1080p decode
  alone; the 4K one has no uncured run to compare). It costs no time.
- `output` moves the peaks. The whole output is held through its pieces' convolutions, where
  the concatenation held the pieces made so far: a small pass peaks higher (the convolution of
  the last piece is its peak), a large one lower (the concatenation's two copies were).

Cure `output` everywhere on the small-tile plans (and with `slice`; room 0, and 0.5):

| run | card | room | cure | encode | predicted / peak / left | decode | predicted / peak / left | decode s | retries | result | md5 | frames off the reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c-1080-12-o-r0-1 | 12 | 0.0 | output | 896 px, 0 blocks | 9.48 / 9.339 / 1.17 | 640 px, DiT as in its phase | 9.61 / 9.875 / 0.63 | 185.2 | 221 | exit 0 | c6f05dba | 48 of 53 |
| c-1080-16-o-r0-1 | 16 | 0.0 | output | 1152 px, 0 blocks | 13.60 / 13.511 / 1.00 | 832 px, DiT as in its phase | 13.82 / 13.900 (refused) / 0.61 | 15.5 | 132 | REFUSED in the decode, slice 4, at decode slice: 169.41 MiB asked, 13.73 allocated, pool 14.53, card free 5 MiB |  | no frames |
| c-1080-16-o-r0.5-1 | 16 | 0.5 | output | 1088 px, 0 blocks | 12.73 / 12.612 / 1.90 | 832 px, 0 blocks | 12.82 / 13.324 / 1.19 | 252.5 | 129 | exit 0 | 952e3e77 | no reference |
| c-1080-16-os-r0-1 | 16 | 0.0 | output+slice | 1152 px, 0 blocks | 13.60 / 13.511 / 1.00 | 832 px, DiT as in its phase | 13.82 / 13.900 (refused) / 0.61 | 20.5 | 126 | REFUSED in the decode, slice 5, at decode slice: 169.41 MiB asked, 13.73 allocated, pool 14.53, card free 5 MiB |  | no frames |
| c-1080-12-o-r0-2 | 12 | 0.0 | output | 896 px, 0 blocks | 9.48 / 9.339 / 1.17 | 640 px, DiT as in its phase | 9.61 / 9.875 / 0.63 | 187.2 | 237 | exit 0 | f13bb1e9 | 48 of 53 |

  Under `output` everywhere the small tiles stand over their measured table values, by 0.27 GiB
  (640 px), 0.50 (832 px) and 0.04 (1104 × 1152), and a plan packed on them is refused or moves
  its frames. So the cure is kept for outputs of 2.5 GiB or more (`--output-floor 2.5`), between
  the 2.43 GiB that gains nothing and the 3.37 that fails without it: PROVISIONAL, two points.

### The floor is the frames

**The 12 GB plan's shape** (encode in 896-px tiles, decode in 640-px tiles beside the DiT's 2 slots, window 8; 53 frames):

| run | card | cure | GiB left to the decode's usual peak | frames off the reference | decode s | decode's peak | retries | md5 |
|---|---|---|---|---|---|---|---|---|
| r-1080-12-ref (the reference) | bare | none | | | 156.0 | | 0 | 9b003b65 |
| c-1080-12-o-r0-1 | 12 | output | 0.63 | 48 of 53 | 185.2 | 9.875 | 221 | c6f05dba |
| c-1080-12-n-r0-1 | 12 | none | 0.90 | 0 of 53 | 157.0 | 9.613 | 248 | 9b003b65 |
| r-1080-12-o-14 | 14 | output | 2.63 | 0 of 53 | 156.3 | 9.875 | 0 | 9b003b65 |
| c-1080-12-o-r0-2 | 12 | output | 0.63 | 48 of 53 | 187.2 | 9.875 | 237 | f13bb1e9 |
| c-1080-12-n-r0-2 | 12 | none | 0.90 | 0 of 53 | 157.2 | 9.613 | 261 | 9b003b65 |
| c-1080-12-n-11.8-1 | 11.8 | none | 0.70 | 48 of 53 | 180.3 | 9.613 | 307 | 3a89a77e |
| c-1080-12-n-11.7-1 | 11.7 | none | 0.60 | 48 of 53 | 175.4 | 9.337 | 376 | e113218d |
| c-1080-12-n-11.6-1 | 11.6 | none | 0.50 | 48 of 53 | 188.0 | 9.337 | 372 | 5d8f413f |
| c-1080-12-n-11.9-1 | 11.9 | none | 0.80 | 24 of 53 | 160.2 | 9.613 | 282 | 6918a713 |
| c-1080-12-n-11.8-2 | 11.8 | none | 0.70 | 48 of 53 | 173.7 | 9.613 | 298 | 4ab5a383 |

**The 16 GB plan's shape** (encode in 1152-px tiles, decode in 832-px tiles beside the slots, window 12; 85 frames):

| run | card | cure | GiB left to the decode's usual peak | frames off the reference | decode s | decode's peak | retries | md5 |
|---|---|---|---|---|---|---|---|---|
| r-1080-16-ref (the reference) | bare | none | | | 251.9 | | 0 | 1c983cb7 |
| c-1080-16-n-r0-1 | 16 | none | 0.68 | 80 of 85 | 266.4 | 13.825 | 448 | d939ad83 |
| c-1080-16-n-16.2-1 | 16.2 | none | 0.89 | 80 of 85 | 267.1 | 13.567 | 408 | 721165bf |
| c-1080-16-n-16.4-1 | 16.4 | none | 1.08 | 0 of 85 | 254.0 | 13.825 | 364 | 1c983cb7 |

**The 24 GB plan's shape** (untiled encode, decode in 1152-px tiles, window 20; 149 frames):

| run | card | cure | GiB left to the decode's usual peak | frames off the reference | decode s | decode's peak | retries | md5 |
|---|---|---|---|---|---|---|---|---|
| r-1080-24-ref (the reference) | bare | none | | | 427.7 | | 0 | cb022a78 |
| a-1080-24-r0-1 | 24 | none | 0.94 | 0 of 149 | 429.8 | 21.567 | 420 | cb022a78 |

**A pass alone, packed, against the reference's units** (`dec`: the decode alone on a card of 12 GB from the reference's latent and windows; `enc`: the encode alone packed, the rest on the bare card), **and the product as it is** (`p-`: room 1.5 on 12 GB against its own shape's reference):

| run | card | room | cure | encode | predicted / peak / left | decode | predicted / peak / left | decode s | retries | result | md5 | frames off the reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x-1080-12-dec-o-12-1 | 12 | 0.0 | output |  |  | 640 px, DiT as in its phase | 9.61 / 9.871 / 0.64 | 177.1 | 155 | exit 0 | d5cd5aea | 48 of 53 |
| x-1080-12-dec-o-12-2 | 12 | 0.0 | output |  |  | 640 px, DiT as in its phase | 9.61 / 9.871 / 0.64 | 177.2 | 155 | exit 0 | d5cd5aea | 48 of 53 |
| x-1080-12-dec-n-12-1 | 12 | 0.0 | none |  |  | 640 px, DiT as in its phase | 9.61 / 9.609 / 0.90 | 157.1 | 152 | exit 0 | 9b003b65 | 0 of 53 |
| r-1080-12-w5-ref | bare | 1.5 | none | 896 px, DiT as in its phase | 24.83 / 24.429 / 69.78 | 640 px, DiT as in its phase | 23.96 / 23.961 / 70.25 | 156.1 | 0 | exit 0 | a4cfc1ac | no reference |
| x-1080-12-enc-11.2 | bare | 1.5 | none |  |  | 640 px, DiT as in its phase | 23.96 / 23.961 / 70.25 | 156.0 | 0 | exit 0 | a4cfc1ac | 0 of 53 |
| p-1080-12-r1.5-1 | 12 | 1.5 | none | 768 px, 0 blocks | 8.00 / 7.605 / 2.90 | 576 px, 0 blocks | 7.97 / 7.973 / 2.54 | 160.6 | 45 | exit 0 | 6b77e4a7 | 0 of 53 |
| r-1080-12-p-ref | bare | 1.5 | none | 768 px, DiT as in its phase | 23.35 / 22.953 / 71.26 | 576 px, DiT as in its phase | 23.32 / 23.320 / 70.89 | 161.7 | 0 | exit 0 | 6b77e4a7 | no reference |

- The 12 GB plan at room 0, uncured, gives the reference's frames (0.90 GiB left to its decode),
  and so does the 24 GB plan (0.94 left); the 16 GB plan at room 0 runs with 80 frames of 85 off
  (0.68 left). No one floor fits the three shapes: the 832-px decode lost 80 frames of 85 with
  0.89 GiB left, where the 640-px one kept its frames with 0.90; and a rung kept was seen one to
  three times, where one command gave two results (the next point). Whether it moves with the
  tile isn't shown.
- A packed whole run doesn't give the same frames twice (the same command, 32 and 48 frames of
  53 apart), and its decode is 2 to 20% slower. The decode alone, packed, from the reference's
  own latent and windows, loses its frames too, and gave the same other frames twice. torch's
  retries don't tell it: 248 in a run whose frames hold, 307 in one whose frames don't.
- Cure `output` moves no bit: on a card with room to spare its frames are the reference's.
- An encode packed alone (896-px tiles, 0.63 GiB left), its job then finished on the bare card,
  kept its frames, once.

### One convolution alone

`vae_room.py conv`: a 3 × 3 × 3 convolution of 256 channels in bfloat16 over one piece of a
decode's input (1 × 256 × 6 × 160 × 640, 0.293 GiB), cuDNN 9.24, `benchmark` and
`deterministic` off as seedvr2x leaves them, run with the memory free, then with less and less
of it left, three times a step:

| process | GiB left | output, three runs | seconds | GiB allocated over the input at the most |
|---|---|---|---|---|
| 1 | the card's memory free | the first bits, twice | 0.049 | 0.687 |
| 1 | 2.0 | the first bits | 0.049 to 0.050 | 0.687 |
| 1 | 0.7 | the first bits | 0.049 to 0.050 | 0.687 |
| 1 | 0.5 | OTHER BITS | 0.418 to 0.463 | 0.195 |
| 1 | 0.3 | OTHER BITS | 0.418 | 0.195 |
| 1 | free again, after the steps short of memory | OTHER BITS | 0.418 | 0.195 |
| 2 | the card's memory free | the first bits, twice | 0.049 | 0.687 |
| 2 | 4 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 3 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 2 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 1.5 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 1.2 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 1.0 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 0.9 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 0.8 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 0.7 | the first bits | 0.049 to 0.050 | 0.687 |
| 2 | 0.6 | OTHER BITS | 0.418 to 0.470 | 0.195 |
| 2 | 0.5 | OTHER BITS | 0.418 | 0.195 |
| 2 | 0.4 | OTHER BITS | 0.418 | 0.195 |
| 2 | 0.3 | OTHER BITS | 0.418 | 0.195 |
| 2 | 0.2 | OTHER BITS | 0.418 | 0.195 |
| 2 | 0.1 | refused (200.00 MiB asked) | | |

(Process 2 is a second run of the probe, with fewer steps, for its last line.)

No log line, no error: short of memory, the convolution gives other bits, 8.5 times slower, and
with the memory free again, in the same process, it gave the same other bits three times of
three. The reading, inferred from the bits, the time and the memory (no log names an engine, and
"kept" is three calls in one process): torch tries cuDNN's engines in turn, swallows a refused
workspace and keeps the engine it fell back to for that convolution's shape. In a pass, the
frames move from the first slice short of room on: a packed decode's first slice (5 frames)
keeps the reference's frames and all the later ones differ, and the pass is slower to its end.
(Shown since, in torch's source and the cuDNN frontend's log: [the cause](#the-cause-in-torchs-source).)

### The floor by tile size, and its cause (2026-10-16)

The frames' floor of each VAE pass by its tile, on seedvr2x at 6e1be76 (one frozen copy, its tree
checked before every run), by a ladder of emulated cards a tile
([`scripts/floor.py`](../scripts/floor.py)); the cause, read in torch 2.14.1's source; and what a
run can see of it.

#### Method as run

- **A pass alone, the same plan on every rung.** A rung is one run of seedvr2x's own command line
  by `vae_room.py run`, in its own process, one hold of the GPU's lock, the planner's room at
  −50 GiB in that process (no plan refused, none moved) and the DiT's phase unjudged
  (`--dit-unjudged`: a pass alone runs no window), so that every rung of a ladder runs its
  pinned shape's plan on less and less free memory: 36 blocks swapped, the DiT's slots and other
  parameters (about 1 GiB) on the GPU beside the pass.
- **Two shapes,** two clips of the matrix's 1080p source: 53 frames at 1080p, window 8, for the
  decode's squares up to 1,088 px and the tiles cut off at 1080p's 1,104 rows; 17 frames at 4K,
  window 5, for the squares from 1,152 px; an empty cut list (one shot, no shot detector).
- **The units, made once a shape:** the shape on the bare card, stopped before its decode. A
  decode rung is seedvr2x's resume of a copy of them whose manifest says decode tile T (the pin a
  resume repeats, and the shape it takes): seedvr2x's code unchanged, the decode is seedvr2x's
  in tile T. An encode rung is a new job in tile T, stopped after its encode, its latent compared
  bit for bit.
- **A tile's reference** is its own rung on the bare card, on the same code.
- **The room.** A rung's card is the reference's peak, the target room and the card's offset
  (card less free memory: 1.489 to 1.492 GiB, measured on the rungs); its room ("usual") is the
  free memory its plan was made in (the run's own plan, which the record keeps) less the
  reference's peak: the "left" of the sections above, what the ladder steps on.
- **The frames.** Kept: every frame's CRC-32 (seedvr2x's checksums) the reference's; moved: how
  many differ, from which, and the worst PSNR-Y; an encode kept only with its latent bit for
  bit. Refused: out of memory, or another memory error.
- **The ladder:** the reference, then 1.1 GiB (or near the floors found for its kind), steps of
  0.2, 0.2, 0.4 then 0.8 GiB one way, then the bracket halved to 0.1 GiB; the least room kept and
  the most lost then run again on the same cards (a card lost on any run counts as lost); 12
  rungs at the most. A first round (decode squares of 640, 1024, 1408, 1792 and 2048 px, the
  1,104-row tiles of 1152, 1536 and 1856, encodes of 512, 1024, 1472 and 2048), then a second,
  then the rest in 64-px steps; the queue was stopped after its 120th rung, the tiles the
  one-output cure takes to be laddered with it.
- **What the run could see,** on every rung, the signals that don't depend on the clock: torch's
  allocator counters at the pass's start and end (its count of refused allocations); the cuDNN
  frontend's log on (`CUDNN_FRONTEND_LOG_INFO=1`, `CUDNN_FRONTEND_LOG_FILE=stderr`), which names
  every execution plan built, its engine and its workspace; torch's "recovered from an
  allocation failure" lines. The pass's time is recorded but not tried as a signal on this card:
  power-capped, its speed drifts from run to run (up to 9% back to back, 40 to 60% between
  sessions: [benchmarking.md](benchmarking.md#caveats)); it is to be tried on another card. A rung
  took 170 to 190 s (a 1080p decode), 216 s (a 4K one), 88 s (a 1080p encode), for information.

#### The floors

| ladder | tile | ref peak GiB | ref pass s | ref plans | least room kept (runs) | most room lost (runs) | done |
|---|---|---|---|---|---|---|---|
| dec1080-640 | 640x640 | 9.609 | 156.0 | 266 | 0.701 (2: 1: kept, 2: kept) | 0.600 (2: 1: moved, 2: moved) | yes |
| enc1080-512 | 512x512 | 5.040 | 72.5 | 152 | 0.499 (2: 1: kept, 2: kept) | 0.399 (2: 1: moved, 2: moved) | yes |
| dec1080-1024 | 1024x1024 | 18.971 | 157.3 | 308 | 0.751 (2: 1: kept, 2: kept) | 0.651 (2: 1: refused, 2: refused) | yes |
| dec2160-1408 | 1408x1408 | 34.089 | 202.2 | 387 | 0.980 (2: 1: kept, 2: kept) | 0.880 (2: 1: refused, 2: refused) | yes |
| dec2160-1792 | 1792x1792 | 53.999 | 203.0 | 332 | 0.959 (2: 1: kept, 2: kept) | 0.859 (2: 1: refused, 2: refused) | yes |
| dec2160-2048 | 2048x2048 | 70.004 | 203.9 | 364 | 0.700 (2: 1: kept, 2: kept) | 0.601 (2: 1: refused, 2: refused) | yes |
| dec1080-1152 | 1104x1152 | 22.557 | 152.2 | 187 | 0.770 (2: 1: kept, 2: kept) | 0.671 (2: 1: refused, 2: refused) | yes |
| dec1080-1536 | 1104x1536 | 29.463 | 151.3 | 178 | 0.860 (2: 1: kept, 2: kept) | 0.761 (2: 1: refused, 2: refused) | yes |
| dec1080-1856 | 1104x1856 | 35.168 | 154.2 | 166 | 0.970 (2: 1: kept, 2: kept) | 0.869 (2: 1: refused, 2: refused) | yes |
| enc1080-1024 | 1024x1024 | 11.996 | 68.5 | 175 | 0.650 (2: 1: kept, 2: kept) | 0.551 (2: 1: moved, 2: moved) | yes |
| enc2160-1472 | 1472x1472 | 21.258 | 88.3 | 212 | 0.720 (2: 1: kept, 2: kept) | 0.620 (2: 1: moved, 2: moved) | yes |
| enc2160-2048 | 2048x2048 | 39.277 | 92.2 | 225 | 0.570 (2: 1: kept, 2: kept) | 0.470 (2: 1: refused, 2: refused) | yes |
| dec1080-832 | 832x832 | 13.819 | 157.1 | 283 | 0.791 (2: 1: kept, 2: kept) | 0.690 (2: 1: moved, 2: moved) | yes |
| dec2160-1216 | 1216x1216 | 25.922 | 203.4 | 351 | 0.880 (2: 1: kept, 2: kept) | 0.780 (2: 1: refused, 2: refused) | yes |
| dec2160-1600 | 1600x1600 | 43.479 | 202.0 | 379 | 0.989 (2: 1: kept, 2: kept) | 0.890 (2: 1: refused, 2: refused) | yes |
| dec2160-1920 | 1920x1920 | 61.686 | 205.9 | 309 | 1.100 (2: 1: kept, 2: kept) | 1.000 (2: 1: refused, 2: refused) | yes |
| dec1080-1344 | 1104x1344 | 26.024 | 152.5 | 186 | 0.821 (2: 1: kept, 2: kept) | 0.721 (2: 1: refused, 2: refused) | yes |
| dec1080-1728 | 1104x1728 | 32.907 | 152.9 | 166 | 0.921 (2: 1: kept, 2: kept) | 0.821 (2: 1: refused, 2: refused) | yes |
| dec1080-704 | 704x704 | 11.309 | 151.6 | 294 | 0.819 (2: 1: kept, 2: kept) | 0.719 (2: 1: moved, 2: moved) | yes |
| dec1080-768 | 768x768 | 12.624 | 156.3 | 290 | 0.720 (1: 1: kept) | 0.621 (1: 1: moved) | no |

"Least room kept": the least room on which every run kept the frames, above every loss; "most
room lost": the most on which a run moved its frames or was refused; each with its runs.
Not run here: dec1080-896, dec1080-960, dec1080-1088, dec2160-1152, dec2160-1280, dec2160-1344, dec2160-1472, dec2160-1536, dec2160-1664, dec2160-1728, dec2160-1856, dec2160-1984, dec1080-1216, dec1080-1280, dec1080-1408, dec1080-1472, dec1080-1600, dec1080-1664, dec1080-1792

- **From 1,024 px a decode is refused before its frames move; smaller passes move them.** Decodes
  in tiles of 640 to 832 px kept their frames with 0.70 to 0.82 GiB left and moved them with 0.1
  less: 48 frames of 53 in 640- and 832-px tiles (from the sixth, the same other frames on both
  runs), 8 in 704-px tiles (from the 46th, twice), 7 and 15 in 768-px tiles (from the first, once
  each); worst PSNR-Y 72.8 to 89.8 dB. Every decode tile of 1,024 px or more was refused before any frame moved, at
  the concatenation of a split convolution's pieces, 2 KiB a pixel of the tile asked in one piece
  (2.00 GiB at 1,024 px, 8.00 at 2,048) while the pool held more than that beyond what was
  allocated: the cure found above isn't in this code. There the floor is the refusal's (0.60 to
  1.00 GiB left), and the frames hold above it. An encode moved its latent instead (in 512- and
  1,024-px tiles, and 4K's 1,472), but in 4K's 2,048-px tiles, refused first.
- **These are main's floors.** The ladders ran seedvr2x at 6e1be76, without the held cure that
  writes a split convolution's pieces into one output. For the tiles that cure takes (decode
  squares from 1,152 px, 1,104-row tiles from 1,104 × 1,216, encode squares from 1,664, and
  1080p's untiled decode) they stay main's record; their floor under the cure came since from
  ladders run with it, and the tiles not run here from ladders of their own
  ([below](#the-large-tiles-under-the-one-output-cure-and-the-tiles-left-2026-10-17)).
- **A pass alone keeps its frames closer than a whole run.** The 640-px decode alone kept them
  with 0.70 GiB left and the 832-px one with 0.79, where the whole runs of the same shapes (on
  the code of 2026-10-12) lost them with 0.70 to 0.80 and with 0.89
  ([the floor is the frames](#the-floor-is-the-frames)): 0.2 to 0.3 GiB more. On the cure's code,
  the six tiles' whole runs, in the passes alone's own shapes, kept their frames within 0.2 GiB of
  them, either way
  ([below](#whole-runs-in-large-tiles-under-the-one-output-cure-2026-10-17)).

The floor against the tile (the least room kept; the tile's area; the largest workspace its
reference's plans asked):

| ladder | tile area Mpx | reference's largest workspace GiB | least kept | most lost |
|---|---|---|---|---|
| dec1080-640 | 0.410 | 1.221 | 0.701 | 0.600 (moved) |
| enc1080-512 | 0.262 | 0.754 | 0.499 | 0.399 (moved) |
| dec1080-1024 | 1.049 | 3.125 | 0.751 | 0.651 (refused) |
| dec2160-1408 | 1.982 | 5.908 | 0.980 | 0.880 (refused) |
| dec2160-1792 | 3.211 | 9.570 | 0.959 | 0.859 (refused) |
| dec2160-2048 | 4.194 | 12.500 | 0.700 | 0.601 (refused) |
| dec1080-1152 | 1.272 | 3.790 | 0.770 | 0.671 (refused) |
| dec1080-1536 | 1.696 | 5.054 | 0.860 | 0.761 (refused) |
| dec1080-1856 | 2.049 | 6.107 | 0.970 | 0.869 (refused) |
| enc1080-1024 | 1.049 | 1.360 | 0.650 | 0.551 (moved) |
| enc2160-1472 | 2.167 | 2.810 | 0.720 | 0.620 (moved) |
| enc2160-2048 | 4.194 | 3.966 | 0.570 | 0.470 (refused) |
| dec1080-832 | 0.692 | 2.063 | 0.791 | 0.690 (moved) |
| dec2160-1216 | 1.479 | 4.407 | 0.880 | 0.780 (refused) |
| dec2160-1600 | 2.560 | 7.629 | 0.989 | 0.890 (refused) |
| dec2160-1920 | 3.686 | 10.986 | 1.100 | 1.000 (refused) |
| dec1080-1344 | 1.484 | 4.422 | 0.821 | 0.721 (refused) |
| dec1080-1728 | 1.908 | 5.685 | 0.921 | 0.821 (refused) |
| dec1080-704 | 0.496 | 1.477 | 0.819 | 0.719 (moved) |
| dec1080-768 | 0.590 | 1.758 | 0.720 | 0.621 (moved) |

- dec: every bracketed ladder kept its frames with 1.100 GiB (dec2160-1920) or less, over 16 ladders
- dec, least kept against the area: 0.764 + 0.0527 x, worst miss 0.284 GiB over 16 ladders
- dec, least kept against the workspace: 0.764 + 0.0177 x, worst miss 0.284 GiB over 16 ladders
- dec1080: every bracketed ladder kept its frames with 0.970 GiB (dec1080-1856) or less, over 10 ladders
- dec1080, least kept against the area: 0.674 + 0.1187 x, worst miss 0.086 GiB over 10 ladders
- dec1080, least kept against the workspace: 0.674 + 0.0398 x, worst miss 0.086 GiB over 10 ladders
- dec2160: every bracketed ladder kept its frames with 1.100 GiB (dec2160-1920) or less, over 6 ladders
- dec2160, least kept against the area: 1.013 + -0.0275 x, worst miss 0.197 GiB over 6 ladders
- dec2160, least kept against the workspace: 1.013 + -0.0092 x, worst miss 0.197 GiB over 6 ladders
- enc: every bracketed ladder kept its frames with 0.720 GiB (enc2160-1472) or less, over 4 ladders
- enc, least kept against the area: 0.591 + 0.0097 x, worst miss 0.108 GiB over 4 ladders
- enc, least kept against the workspace: 0.566 + 0.0196 x, worst miss 0.099 GiB over 4 ladders
- enc1080: every bracketed ladder kept its frames with 0.650 GiB (enc1080-1024) or less, over 2 ladders
- enc1080, least kept against the area: 0.448 + 0.1925 x, worst miss 0.000 GiB over 2 ladders
- enc1080, least kept against the workspace: 0.310 + 0.2500 x, worst miss 0.000 GiB over 2 ladders
- enc2160: every bracketed ladder kept its frames with 0.720 GiB (enc2160-1472) or less, over 2 ladders
- enc2160, least kept against the area: 0.881 + -0.0742 x, worst miss 0.000 GiB over 2 ladders
- enc2160, least kept against the workspace: 1.086 + -0.1302 x, worst miss 0.000 GiB over 2 ladders

For 1080p's decodes a line holds roughly, the frames' floor of the small tiles and the refusal's
of the large ones alike: 0.67 + 0.12 GiB a megapixel of tile, missing by up to 0.09 GiB over its
ten tiles (640 px to 1,104 × 1,856; the 704-px tile, kept at 0.82, its worst). 4K's six don't
follow one (a worst miss of 0.20 GiB): 0.88 GiB at 1,216 px up to 1.10 at 1,920, and 0.70 at
2,048. What bounds them all is a step: every decode kept its frames with 1.10 GiB or less, every
encode its latent with 0.72 or less. The largest workspace a reference's plans asked (up to 12.5
GiB) was given at its moment in every rung that kept its frames: the floor isn't that workspace.

#### The cause, in torch's source

torch 2.14.1's source at its tag (file and line), with its cuDNN frontend (1.26.0, the
submodule's commit `35fd7b0`); cuDNN 9.24 on the card (torch's `backends.cudnn.version()`
92400).

1. **The path.** A bfloat16 convolution, 2-D or 3-D, goes to cuDNN only through its v8 API, the
   frontend: `use_cudnn` takes cuDNN for bfloat16 only where the v8 API is on
   ([Convolution.cpp:423-427](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/Convolution.cpp#L423-L427)),
   and the forward goes to `run_single_conv` unless `TORCH_CUDNN_V8_API_DISABLED` is set
   ([Conv_v8.cpp:1484-1509](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1484-L1509),
   [ConvUtils.h:96-105](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ConvUtils.h#L96-L105)).
   The dimensions only change the descriptors.
2. **Heuristics, not benchmarks.** `benchmark` and `deterministic` come from torch's global
   context ([Convolution.cpp:1244-1249](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/Convolution.cpp#L1244-L1249)),
   both off by default ([Context.h:478, 496](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/Context.h#L478-L496));
   seedvr2x sets them only inside its shot detector's forward. So a convolution takes
   `run_single_conv`'s heuristic branch
   ([Conv_v8.cpp:1155-1283](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1155-L1283)):
   its key (the shapes, strides, padding, dtype, layout, `deterministic`, TF32, and the three
   pointers' alignment) looked up in a cache of plans (1168-1188); missing, the heuristic's top
   configuration (1192-1219), then all its configurations but that one (1220-1255), then cuDNN's
   fallback list (1256-1281).
3. **The workspace, asked of torch's allocator.** A plan runs in `run_conv_plan`, which asks the
   process's allocator for the plan's workspace before it executes
   ([Conv_v8.cpp:382-384](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L382-L384)),
   the convolution's output already allocated
   ([ConvShared.cpp:256-259](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/ConvShared.cpp#L256-L259)).
4. **A refusal, swallowed.** `try_configs` builds each configuration's plan and runs it; an
   `OutOfMemoryError` (or a cuDNN error) is caught and ignored, and the next configuration
   tried, without a word
   ([Conv_v8.cpp:1053-1084](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1053-L1084)).
   A cached plan whose workspace is refused starts the search again from the top (1180-1188).
5. **The plan that ran, kept.** The first plan that runs is put in the cache (1075), a cache by
   thread, never freed (349-362), of 10,000 keys by default
   ([257-285](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L257-L285)):
   every later call of the same key runs it, with room or without, for the life of the process.
6. **`deterministic` doesn't stop it:** it only takes the nondeterministic configurations out
   of the lists ([Conv_v8.cpp:140-168](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cudnn/Conv_v8.cpp#L140-L168)):
   another engine is still taken where the first's workspace is refused.
7. **What `cudaMallocAsync` does and counts** (the allocator seedvr2x runs on a driver that
   refuses), in `mallocAsync`
   ([CUDAMallocAsyncAllocator.cpp:355-413](https://github.com/pytorch/pytorch/blob/v2.14.1/c10/cuda/CUDAMallocAsyncAllocator.cpp#L355-L413)):
   a request the driver refuses is tried once more after the stream is synchronised and the pool
   trimmed; given then, it says "[cudaMallocAsync] recovered from an allocation failure by
   trimming the pool and retrying" (377-379); refused again, it raises `OutOfMemoryError`,
   without a log line and without a count. Its `memory_stats` holds only the allocated, active
   and reserved bytes, "many stat types" left at 0 (733-828), `num_ooms` and
   `num_alloc_retries` among them; it can't take an out-of-memory observer (692-697).
8. **What torch's own allocator does and counts** (seedvr2x's on a driver that gives memory the
   card doesn't have, its segments expandable), in `malloc`
   ([CUDACachingAllocator.cpp:1721-1953](https://github.com/pytorch/pytorch/blob/v2.14.1/c10/cuda/CUDACachingAllocator.cpp#L1721-L1953)):
   a request over torch's limit, or that the driver refuses, goes through its retries (the
   cache freed, `num_alloc_retries` + 1 at 3853), and refused at the end counts `num_ooms` + 1
   (1857) and calls the attached out-of-memory observers (1900-1907) before it raises.

**On the card**, with the frontend's log on:

- **One convolution** (`conv_probe.py`: S1's convolution alone, 1 × 256 × 6 × 160 × 640 in
  bfloat16): with the memory free, one plan built, engine 22, a workspace of 527,826,960 bytes;
  with 0.5 GiB left, that plan again (the cached plan's workspace refused, the search begun
  again), 16 more of engine 22 with the same workspace and 2 of engine 23 (947,308,719 bytes),
  then engine 28 with no workspace, which ran: other bits, in 0.46 s where it takes 0.05 with
  room; the next call, and two with the memory free again, built no plan and gave the same other
  bits (0.42 s).
- **In a pass,** the rungs whose frames moved built 20 plans more than their reference in one
  slice (40 in two, where two convolutions fell back), the same 20 as the convolution's: the usual plan's tag again (engine 22, a 0.664 GiB
  workspace), 16 more of engine 22, 2 of engine 23 (1.189 GiB), and engine 28, no workspace; no
  plan in any later slice. The rungs whose frames held built exactly their reference's plans.
  The 1,024-px rungs refused at the concatenation had fallen back the same way before it
  (engine 28 at a slice before the refusal, or in it).
- **The convolution that fell back,** by the tensors the frontend logged before its search: in
  640-px tiles a 3 × 3 × 3 convolution of 256 channels over one piece of a tile at full
  resolution, a 4-frame slice's (input 1 × 256 × 6 × 217 × 642, output 1 × 256 × 4 × 215 × 640),
  whose usual plan asks a 0.664 GiB workspace; in 1,024-px tiles its like (output
  1 × 256 × 4 × 149 × 1024, 0.738 GiB); in the encodes, the encoder's 3 × 3 × 3 convolution of
  128 channels at full resolution (output 1 × 128 × 4 × 512 × 512 in 512-px tiles, a workspace of
  0.63 GiB; in 1,024-px tiles two of them, a piece of an inner tile and one of an edge tile's: two
  fallbacks, 40 plans). Where 48 frames of 53 moved (640 and 832 px): a tile's first slice
  holds 5 frames, its convolutions other shapes with plans of their own, made with room, and the
  frames they give, the first five (an encode's first two latents), held; every later frame takes
  a part of some tile run after the fallback, and moved. In 704- and 768-px tiles fewer frames
  moved (8, from the 46th; 7 to 15, from the first): the convolution and the moment of the
  fallback differ, and weren't read.

Shown so: a workspace refused, torch's next engines tried without a word until one ran without a
workspace, and that engine kept for the convolution's shape; the frames move from that slice on.
Still inferred: that the refused request was the workspace (no line says it: the plan rebuilt, then
the next ones with the same or larger workspaces, then one with none that runs, is what a refused
workspace makes in that code, where nothing else turns on the memory left); why engine 28 gives
other bits (cuDNN's own, closed: another order of the sums, read from the bits). That `num_ooms`
counts these refusals under torch's own allocator, its source's here, was measured since on
another card: one a refused workspace
([under expandable segments](#under-expandable-segments-on-an-rtx-5090-2026-10-17)).

#### What the run can see

Every rung against its reference rung, by the signals that don't depend on the clock: torch's
count of refused allocations across the pass (`num_ooms`, `num_alloc_retries` and
`num_oom_rejections` moved, and what else of its counters moved), the plans the cuDNN frontend
built more than the reference's (cuDNN's log, not torch's count; and where: the slice, the
driver's free memory before it, the first plan built again and the last, the one that ran),
torch's "recovered" lines:

| rung | outcome | room usual | frames differing | torch's refused allocations (counters moved) | cuDNN's plans built - the reference's | where the extra plans were built | torch's recovered lines |
|---|---|---|---|---|---|---|---|
| dec1080-640-c11.799-1 | kept | 0.701 | 0 | 0 (none) | 0 | - | 201 |
| enc1080-512-c7.029-1 | kept | 0.499 | 0 | 0 (none) | 0 | - | 124 |
| dec1080-640-c11.599-1 | moved | 0.500 | 48 (from 5) | 0 (none) | 20 | slice 3 (+20, 0.296 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.664 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 219 |
| dec1080-640-c11.699-1 | moved | 0.600 | 48 (from 5) | 0 (none) | 20 | slice 6 (+20, 0.052 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.664 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 209 |
| dec1080-640-c11.799-2 | kept | 0.701 | 0 | 0 (none) | 0 | - | 201 |
| dec1080-640-c11.699-2 | moved | 0.600 | 48 (from 5) | 0 (none) | 20 | slice 6 (+20, 0.052 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.664 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 209 |
| dec1080-1024-c21.212-1 | kept | 0.751 | 0 | 0 (none) | 0 | - | 152 |
| dec1080-1024-c21.212-2 | kept | 0.751 | 0 | 0 (none) | 0 | - | 152 |
| dec2160-1408-c36.560-1 | kept | 0.980 | 0 | 0 (none) | 0 | - | 22 |
| dec2160-1408-c36.560-2 | kept | 0.980 | 0 | 0 (none) | 0 | - | 22 |
| dec2160-1792-c56.549-1 | kept | 1.060 | 0 | 0 (none) | 0 | - | 23 |
| dec2160-1792-c56.449-1 | kept | 0.959 | 0 | 0 (none) | 0 | - | 24 |
| dec2160-1792-c56.449-2 | kept | 0.959 | 0 | 0 (none) | 0 | - | 24 |
| dec2160-2048-c72.394-1 | kept | 0.900 | 0 | 0 (none) | 0 | - | 17 |
| dec2160-2048-c72.194-1 | kept | 0.700 | 0 | 0 (none) | 0 | - | 17 |
| dec2160-2048-c72.194-2 | kept | 0.700 | 0 | 0 (none) | 0 | - | 17 |
| dec1080-1152-c24.917-1 | kept | 0.870 | 0 | 0 (none) | 0 | - | 54 |
| dec1080-1152-c24.817-1 | kept | 0.770 | 0 | 0 (none) | 0 | - | 55 |
| dec1080-1152-c24.817-2 | kept | 0.770 | 0 | 0 (none) | 0 | - | 55 |
| dec1080-1536-c31.813-1 | kept | 0.860 | 0 | 0 (none) | 0 | - | 40 |
| dec1080-1536-c31.813-2 | kept | 0.860 | 0 | 0 (none) | 0 | - | 40 |
| dec1080-1856-c37.727-1 | kept | 1.070 | 0 | 0 (none) | 0 | - | 35 |
| dec1080-1856-c37.627-1 | kept | 0.970 | 0 | 0 (none) | 0 | - | 36 |
| dec1080-1856-c37.627-2 | kept | 0.970 | 0 | 0 (none) | 0 | - | 36 |
| enc1080-512-c6.829-1 | moved | 0.300 | 12 (from 2) | 0 (none) | 20 | slice 7 (+20, 0.027 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.629 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 7 |
| enc1080-512-c6.929-1 | moved | 0.399 | 12 (from 2) | 0 (none) | 20 | slice 8 (+20, 0.032 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.629 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 7 |
| enc1080-512-c7.029-2 | kept | 0.499 | 0 | 0 (none) | 0 | - | 124 |
| enc1080-512-c6.929-2 | moved | 0.399 | 12 (from 2) | 0 (none) | 20 | slice 8 (+20, 0.032 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.629 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 7 |
| enc1080-1024-c14.036-1 | moved | 0.551 | 4 (from 10) | 0 (none) | 40 | slice 10 (+20, 1.390 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.632 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB); slice 26 (+20, 0.171 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.790 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 161 |
| enc1080-1024-c14.235-1 | kept | 0.750 | 0 | 0 (none) | 0 | - | 112 |
| enc1080-1024-c14.135-1 | kept | 0.650 | 0 | 0 (none) | 0 | - | 134 |
| enc1080-1024-c14.135-2 | kept | 0.650 | 0 | 0 (none) | 0 | - | 134 |
| enc1080-1024-c14.036-2 | moved | 0.551 | 4 (from 10) | 0 (none) | 40 | slice 10 (+20, 1.390 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.632 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB); slice 26 (+20, 0.171 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.790 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 161 |
| enc2160-1472-c23.368-1 | moved | 0.620 | 3 (from 2) | 0 (none) | 20 | slice 3 (+20, 1.847 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.753 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 46 |
| enc2160-1472-c23.568-1 | kept | 0.820 | 0 | 0 (none) | 0 | - | 52 |
| enc2160-1472-c23.468-1 | kept | 0.720 | 0 | 0 (none) | 0 | - | 60 |
| enc2160-1472-c23.468-2 | kept | 0.720 | 0 | 0 (none) | 0 | - | 60 |
| enc2160-1472-c23.368-2 | moved | 0.620 | 3 (from 2) | 0 (none) | 20 | slice 3 (+20, 1.847 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.753 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 46 |
| enc2160-2048-c41.437-1 | kept | 0.669 | 0 | 0 (none) | 0 | - | 16 |
| enc2160-2048-c41.337-1 | kept | 0.570 | 0 | 0 (none) | 0 | - | 17 |
| enc2160-2048-c41.337-2 | kept | 0.570 | 0 | 0 (none) | 0 | - | 17 |
| dec1080-832-c16.200-1 | kept | 0.891 | 0 | 0 (none) | 0 | - | 187 |
| dec1080-832-c16.000-1 | moved | 0.690 | 48 (from 5) | 0 (none) | 20 | slice 8 (+20, 0.321 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.422 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 210 |
| dec1080-832-c16.100-1 | kept | 0.791 | 0 | 0 (none) | 0 | - | 194 |
| dec1080-832-c16.100-2 | kept | 0.791 | 0 | 0 (none) | 0 | - | 194 |
| dec1080-832-c16.000-2 | moved | 0.690 | 48 (from 5) | 0 (none) | 20 | slice 8 (+20, 0.321 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.422 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 210 |
| dec2160-1216-c28.292-1 | kept | 0.880 | 0 | 0 (none) | 0 | - | 48 |
| dec2160-1216-c28.292-2 | kept | 0.880 | 0 | 0 (none) | 0 | - | 48 |
| dec2160-1600-c46.059-1 | kept | 1.091 | 0 | 0 (none) | 0 | - | 20 |
| dec2160-1600-c45.959-1 | kept | 0.989 | 0 | 0 (none) | 0 | - | 20 |
| dec2160-1600-c45.959-2 | kept | 0.989 | 0 | 0 (none) | 0 | - | 20 |
| dec2160-1920-c64.277-1 | kept | 1.100 | 0 | 0 (none) | 0 | - | 28 |
| dec2160-1920-c64.277-2 | kept | 1.100 | 0 | 0 (none) | 0 | - | 28 |
| dec1080-1344-c28.435-1 | kept | 0.920 | 0 | 0 (none) | 0 | - | 44 |
| dec1080-1344-c28.335-1 | kept | 0.821 | 0 | 0 (none) | 0 | - | 51 |
| dec1080-1344-c28.335-2 | kept | 0.821 | 0 | 0 (none) | 0 | - | 51 |
| dec1080-1728-c35.317-1 | kept | 0.921 | 0 | 0 (none) | 0 | - | 39 |
| dec1080-1728-c35.317-2 | kept | 0.921 | 0 | 0 (none) | 0 | - | 39 |
| dec1080-704-c13.719-1 | kept | 0.920 | 0 | 0 (none) | 0 | - | 100 |
| dec1080-704-c13.519-1 | moved | 0.719 | 8 (from 45) | 0 (none) | 20 | slice 25 (+20, 0.404 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.804 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 123 |
| dec1080-704-c13.619-1 | kept | 0.819 | 0 | 0 (none) | 0 | - | 102 |
| dec1080-704-c13.619-2 | kept | 0.819 | 0 | 0 (none) | 0 | - | 102 |
| dec1080-704-c13.519-2 | moved | 0.719 | 8 (from 45) | 0 (none) | 20 | slice 25 (+20, 0.404 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.804 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 123 |
| dec1080-768-c15.035-1 | kept | 0.920 | 0 | 0 (none) | 0 | - | 65 |
| dec1080-768-c14.835-1 | kept | 0.720 | 0 | 0 (none) | 0 | - | 92 |
| dec1080-768-c14.635-1 | moved | 0.521 | 15 (from 0) | 0 (none) | 20 | slice 1 (+20, 11.648 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.851 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 144 |
| dec1080-768-c14.735-1 | moved | 0.621 | 7 (from 0) | 0 (none) | 20 | slice 14 (+20, 0.152 GiB free before it): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.851 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) | 95 |

The refused rungs, apart:

| rung | room usual | where | asked GiB | allocated / pool GiB | torch's counters moved | cuDNN's plans built - the reference's | extra plans before the refusal |
|---|---|---|---|---|---|---|---|
| dec1080-1024-c21.012-1 | 0.549 | memory_limit_conv:162 | 2.00 | 16.92 / 19.53 | none | -195 | slice 2 (+19, 2.085 GiB free before it): ConvFwd_eng22 x16, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=3_k13=0_k14=2_k18=0_k23=0 (0.738 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) |
| dec1080-1024-c21.112-1 | 0.651 | memory_limit_conv:162 | 2.00 | 16.92 / 19.62 | none | -194 | slice 13 (+20): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.738 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) |
| dec1080-1024-c21.112-2 | 0.651 | memory_limit_conv:162 | 2.00 | 16.92 / 19.62 | none | -194 | slice 13 (+20): ConvFwd_eng22 x17, ConvFwd_eng23 x2, ConvFwd_eng28 x1; first ConvFwd_eng22_k2=0_k13=1_k14=2_k18=0_k23=0 (0.738 GiB), last ConvFwd_eng28_k2=2_k3=0 (0.000 GiB) |
| dec2160-1408-c36.360-1 | 0.781 | memory_limit_conv:162 | 3.78 | 30.31 / 34.88 | none | -301 | none |
| dec2160-1408-c36.460-1 | 0.880 | memory_limit_conv:162 | 3.78 | 30.31 / 35.00 | none | -301 | none |
| dec2160-1408-c36.460-2 | 0.880 | memory_limit_conv:162 | 3.78 | 30.31 / 35.00 | none | -301 | none |
| dec2160-1792-c56.349-1 | 0.859 | memory_limit_conv:162 | 6.12 | 47.87 / 54.88 | none | -255 | none |
| dec2160-1792-c56.349-2 | 0.859 | memory_limit_conv:162 | 6.12 | 47.87 / 54.88 | none | -255 | none |
| dec2160-2048-c71.994-1 | 0.499 | memory_limit_conv:162 | 8.00 | 62.00 / 70.53 | none | -280 | none |
| dec2160-2048-c72.094-1 | 0.601 | memory_limit_conv:162 | 8.00 | 62.00 / 70.62 | none | -280 | none |
| dec2160-2048-c72.094-2 | 0.601 | memory_limit_conv:162 | 8.00 | 62.00 / 70.62 | none | -280 | none |
| dec1080-1152-c24.717-1 | 0.671 | memory_limit_conv:162 | 2.43 | 20.12 / 23.25 | none | -90 | none |
| dec1080-1152-c24.717-2 | 0.671 | memory_limit_conv:162 | 2.43 | 20.12 / 23.25 | none | -90 | none |
| dec1080-1536-c31.613-1 | 0.659 | memory_limit_conv:162 | 3.23 | 26.23 / 30.12 | none | -94 | none |
| dec1080-1536-c31.713-1 | 0.761 | memory_limit_conv:162 | 3.23 | 26.22 / 30.25 | none | -76 | none |
| dec1080-1536-c31.713-2 | 0.761 | memory_limit_conv:162 | 3.23 | 26.22 / 30.25 | none | -76 | none |
| dec1080-1856-c37.527-1 | 0.869 | memory_limit_conv:162 | 3.91 | 31.25 / 36.06 | none | -60 | none |
| dec1080-1856-c37.527-2 | 0.869 | memory_limit_conv:162 | 3.91 | 31.25 / 36.06 | none | -60 | none |
| enc2160-2048-c41.237-1 | 0.470 | memory_limit_conv:162 | 4.00 | 35.27 / 39.75 | none | -152 | none |
| enc2160-2048-c41.237-2 | 0.470 | memory_limit_conv:162 | 4.00 | 35.27 / 39.75 | none | -152 | none |
| dec2160-1216-c28.093-1 | 0.680 | memory_limit_conv:162 | 2.82 | 23.10 / 26.62 | none | -262 | none |
| dec2160-1216-c28.193-1 | 0.780 | memory_limit_conv:162 | 2.82 | 23.10 / 26.72 | none | -262 | none |
| dec2160-1216-c28.193-2 | 0.780 | memory_limit_conv:162 | 2.82 | 23.10 / 26.72 | none | -262 | none |
| dec2160-1600-c45.859-1 | 0.890 | memory_limit_conv:162 | 4.88 | 38.60 / 44.38 | none | -295 | none |
| dec2160-1600-c45.859-2 | 0.890 | memory_limit_conv:162 | 4.88 | 38.60 / 44.38 | none | -295 | none |
| dec2160-1920-c64.077-1 | 0.901 | memory_limit_conv:162 | 7.03 | 54.63 / 62.59 | none | -205 | none |
| dec2160-1920-c64.177-1 | 1.000 | memory_limit_conv:162 | 7.03 | 54.63 / 62.69 | none | -205 | none |
| dec2160-1920-c64.177-2 | 1.000 | memory_limit_conv:162 | 7.03 | 54.63 / 62.69 | none | -205 | none |
| dec1080-1344-c28.235-1 | 0.721 | memory_limit_conv:162 | 2.83 | 23.19 / 26.75 | none | -86 | none |
| dec1080-1344-c28.235-2 | 0.721 | memory_limit_conv:162 | 2.83 | 23.19 / 26.75 | none | -86 | none |
| dec1080-1728-c35.117-1 | 0.720 | memory_limit_conv:162 | 3.64 | 29.26 / 33.66 | none | -64 | none |
| dec1080-1728-c35.217-1 | 0.821 | memory_limit_conv:162 | 3.64 | 29.26 / 33.75 | none | -64 | none |
| dec1080-1728-c35.217-2 | 0.821 | memory_limit_conv:162 | 3.64 | 29.26 / 33.75 | none | -64 | none |

What separates the moved rungs from the kept ones (a higher figure saying moved):

- torch's count of refused allocations across the pass (num_ooms + num_alloc_retries + num_oom_rejections): kept 0 to 0, moved 0 to 0: does NOT separate; over 0: 0 true positives, 0 false; 51 true negatives, 16 false
- the cuDNN frontend's plans built in the pass over the reference's (cuDNN's log): kept 0 to 0, moved 20 to 40: SEPARATES, over 10 said moved: 16 true positives, 0 false; 51 true negatives, 0 false
- torch's "recovered from an allocation failure" lines in the pass: kept 16 to 201, moved 7 to 219: does NOT separate; over 104: 10 true positives, 12 false; 39 true negatives, 6 false
- the pass's time: not a signal here (its seconds are in every rung's record, for information): a power-capped card's speed drifts from run to run

So, on this card and its allocator:

- **torch's count of refused allocations says nothing on this allocator.** Under `cudaMallocAsync`
  `num_ooms`, `num_alloc_retries` and `num_oom_rejections` stayed at 0 across every pass, those
  whose frames moved as those that kept them, and an out-of-memory observer can't be attached:
  as its source says, it counts no refusal (above, 7). Under torch's own allocator `num_ooms`
  across a unit counts each refused workspace at no cost: measured on an RTX 5090 under WSL2, 0
  across every pass that kept its frames, 19 to 21 across every one that lost them
  ([under expandable segments](#under-expandable-segments-on-an-rtx-5090-2026-10-17)).
- **The cuDNN frontend's log tells it without a miss:** 20 or 40 plans more than the reference's
  (one or two convolutions fallen back) in every pass whose frames moved, none in any that kept
  them. It is cuDNN's log, not a count of torch's: a run can use it only with the log written
  and read for each pass, and a count to compare with, or the fallback's own marks there (a run
  of plans built for one shape, a plan built again for a shape planned already).
- **torch's "recovered" lines don't separate:** they count the requests a trim saved, not the
  refusals that move an engine (7 in the 512-px encodes that moved, 124 in those that kept their
  latent).
- **The pass's time** wasn't tried here: on this power-capped card it drifts from run to run;
  it stays in the records, and is to be tried on another card.

#### Every rung

"room" is the free memory the plan was made in less the rung's own peak (a refused rung's stops
at its refusal), "room usual" less its reference's, as above; the plans and the seconds are the
pass's, its reference's beside them.

| rung | card | free | peak | room | room usual | outcome | frames differing | worst PSNR | plans (ref) | recovered | pass s (ref) | exit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dec1080-640-ref | bare | 94.212 | 9.609 | 84.603 | - | reference | - | - | 266 (-) | 0 | 156.0 (-) | 0 |
| dec1080-640-c11.799-1 | 11.799 | 10.310 | 9.609 | 0.701 | 0.701 | kept | 0 of 53 | - | 266 (266) | 201 | 157.3 (156.0) | 0 |
| enc1080-512-ref | bare | 94.212 | 5.040 | 89.173 | - | reference | - | - | 152 (-) | 0 | 72.5 (-) | 130 |
| enc1080-512-c7.029-1 | 7.029 | 5.538 | 5.040 | 0.499 | 0.499 | kept | 0 of 14 | - | 152 (152) | 124 | 73.1 (72.5) | 130 |
| dec1080-640-c11.599-1 | 11.599 | 10.109 | 9.609 | 0.500 | 0.500 | moved | 48 of 53 | 73.13 | 286 (266) | 219 | 174.6 (156.0) | 0 |
| dec1080-640-c11.699-1 | 11.699 | 10.208 | 9.609 | 0.600 | 0.600 | moved | 48 of 53 | 73.14 | 286 (266) | 209 | 173.3 (156.0) | 0 |
| dec1080-640-c11.799-2 | 11.799 | 10.310 | 9.609 | 0.701 | 0.701 | kept | 0 of 53 | - | 266 (266) | 201 | 157.3 (156.0) | 0 |
| dec1080-640-c11.699-2 | 11.699 | 10.208 | 9.609 | 0.600 | 0.600 | moved | 48 of 53 | 73.14 | 286 (266) | 209 | 173.3 (156.0) | 0 |
| dec1080-1024-ref | bare | 94.212 | 18.971 | 75.241 | - | reference | - | - | 308 (-) | 0 | 157.3 (-) | 0 |
| dec1080-1024-c21.212-1 | 21.212 | 19.722 | 18.971 | 0.751 | 0.751 | kept | 0 of 53 | - | 308 (308) | 152 | 159.1 (157.3) | 0 |
| dec1080-1024-c21.012-1 | 21.012 | 19.520 | 18.923 | 0.597 | 0.549 | refused | - | - | 113 (308) | 21 | 16.9 (157.3) | 1 |
| dec2160-1408-ref | bare | 94.212 | 34.089 | 60.123 | - | reference | - | - | 387 (-) | 0 | 202.2 (-) | 0 |
| dec1080-1024-c21.112-1 | 21.112 | 19.622 | 18.971 | 0.651 | 0.651 | refused | - | - | 114 (308) | 108 | 70.3 (157.3) | 1 |
| dec1080-1024-c21.212-2 | 21.212 | 19.722 | 18.971 | 0.751 | 0.751 | kept | 0 of 53 | - | 308 (308) | 152 | 159.2 (157.3) | 0 |
| dec1080-1024-c21.112-2 | 21.112 | 19.622 | 18.971 | 0.651 | 0.651 | refused | - | - | 114 (308) | 108 | 70.4 (157.3) | 1 |
| dec2160-1408-c36.360-1 | 36.360 | 34.870 | 32.193 | 2.678 | 0.781 | refused | - | - | 86 (387) | 3 | 19.7 (202.2) | 1 |
| dec2160-1408-c36.560-1 | 36.560 | 35.070 | 34.089 | 0.980 | 0.980 | kept | 0 of 17 | - | 387 (387) | 22 | 202.4 (202.2) | 0 |
| dec2160-1408-c36.460-1 | 36.460 | 34.970 | 32.193 | 2.777 | 0.880 | refused | - | - | 86 (387) | 3 | 19.5 (202.2) | 1 |
| dec2160-1408-c36.560-2 | 36.560 | 35.070 | 34.089 | 0.980 | 0.980 | kept | 0 of 17 | - | 387 (387) | 22 | 202.2 (202.2) | 0 |
| dec2160-1408-c36.460-2 | 36.460 | 34.970 | 32.193 | 2.777 | 0.880 | refused | - | - | 86 (387) | 3 | 19.6 (202.2) | 1 |
| dec2160-1792-ref | bare | 94.212 | 53.999 | 40.213 | - | reference | - | - | 332 (-) | 0 | 203.0 (-) | 0 |
| dec2160-1792-c56.349-1 | 56.349 | 54.859 | 50.850 | 4.008 | 0.859 | refused | - | - | 77 (332) | 3 | 32.3 (203.0) | 1 |
| dec2160-1792-c56.549-1 | 56.549 | 55.060 | 53.999 | 1.060 | 1.060 | kept | 0 of 17 | - | 332 (332) | 23 | 204.0 (203.0) | 0 |
| dec2160-1792-c56.449-1 | 56.449 | 54.958 | 53.999 | 0.959 | 0.959 | kept | 0 of 17 | - | 332 (332) | 24 | 204.2 (203.0) | 0 |
| dec2160-1792-c56.449-2 | 56.449 | 54.958 | 53.999 | 0.959 | 0.959 | kept | 0 of 17 | - | 332 (332) | 24 | 204.3 (203.0) | 0 |
| dec2160-1792-c56.349-2 | 56.349 | 54.859 | 50.850 | 4.008 | 0.859 | refused | - | - | 77 (332) | 3 | 32.5 (203.0) | 1 |
| dec2160-2048-ref | bare | 94.212 | 70.004 | 24.208 | - | reference | - | - | 364 (-) | 0 | 203.9 (-) | 0 |
| dec2160-2048-c72.394-1 | 72.394 | 70.904 | 70.004 | 0.900 | 0.900 | kept | 0 of 17 | - | 364 (364) | 17 | 205.0 (203.9) | 0 |
| dec2160-2048-c72.194-1 | 72.194 | 70.704 | 70.004 | 0.700 | 0.700 | kept | 0 of 17 | - | 364 (364) | 17 | 204.8 (203.9) | 0 |
| dec2160-2048-c71.994-1 | 71.994 | 70.503 | 65.771 | 4.732 | 0.499 | refused | - | - | 84 (364) | 3 | 42.9 (203.9) | 1 |
| dec2160-2048-c72.094-1 | 72.094 | 70.605 | 65.771 | 4.833 | 0.601 | refused | - | - | 84 (364) | 3 | 42.9 (203.9) | 1 |
| dec2160-2048-c72.194-2 | 72.194 | 70.704 | 70.004 | 0.700 | 0.700 | kept | 0 of 17 | - | 364 (364) | 17 | 204.3 (203.9) | 0 |
| dec2160-2048-c72.094-2 | 72.094 | 70.605 | 65.771 | 4.833 | 0.601 | refused | - | - | 84 (364) | 3 | 42.8 (203.9) | 1 |
| dec1080-1152-ref | bare | 94.212 | 22.557 | 71.655 | - | reference | - | - | 187 (-) | 0 | 152.2 (-) | 0 |
| dec1080-1152-c24.917-1 | 24.917 | 23.427 | 22.557 | 0.870 | 0.870 | kept | 0 of 53 | - | 187 (187) | 54 | 153.2 (152.2) | 0 |
| dec1080-1152-c24.717-1 | 24.717 | 23.228 | 22.557 | 0.671 | 0.671 | refused | - | - | 97 (187) | 33 | 52.7 (152.2) | 1 |
| dec1080-1152-c24.817-1 | 24.817 | 23.327 | 22.557 | 0.770 | 0.770 | kept | 0 of 53 | - | 187 (187) | 55 | 153.2 (152.2) | 0 |
| dec1080-1152-c24.817-2 | 24.817 | 23.327 | 22.557 | 0.770 | 0.770 | kept | 0 of 53 | - | 187 (187) | 55 | 153.2 (152.2) | 0 |
| dec1080-1152-c24.717-2 | 24.717 | 23.228 | 22.557 | 0.671 | 0.671 | refused | - | - | 97 (187) | 33 | 52.6 (152.2) | 1 |
| dec1080-1536-ref | bare | 94.212 | 29.463 | 64.749 | - | reference | - | - | 178 (-) | 0 | 151.3 (-) | 0 |
| dec1080-1536-c31.813-1 | 31.813 | 30.323 | 29.463 | 0.860 | 0.860 | kept | 0 of 53 | - | 178 (178) | 40 | 152.2 (151.3) | 0 |
| dec1080-1536-c31.613-1 | 31.613 | 30.122 | 28.166 | 1.957 | 0.659 | refused | - | - | 84 (178) | 3 | 16.8 (151.3) | 1 |
| dec1080-1536-c31.713-1 | 31.713 | 30.224 | 29.463 | 0.761 | 0.761 | refused | - | - | 102 (178) | 7 | 25.7 (151.3) | 1 |
| dec1080-1536-c31.813-2 | 31.813 | 30.323 | 29.463 | 0.860 | 0.860 | kept | 0 of 53 | - | 178 (178) | 40 | 152.2 (151.3) | 0 |
| dec1080-1536-c31.713-2 | 31.713 | 30.224 | 29.463 | 0.761 | 0.761 | refused | - | - | 102 (178) | 7 | 25.7 (151.3) | 1 |
| dec1080-1856-ref | bare | 94.212 | 35.168 | 59.044 | - | reference | - | - | 166 (-) | 0 | 154.2 (-) | 0 |
| dec1080-1856-c37.527-1 | 37.527 | 36.036 | 35.168 | 0.869 | 0.869 | refused | - | - | 106 (166) | 16 | 64.0 (154.2) | 1 |
| dec1080-1856-c37.727-1 | 37.727 | 36.237 | 35.168 | 1.070 | 1.070 | kept | 0 of 53 | - | 166 (166) | 35 | 155.2 (154.2) | 0 |
| dec1080-1856-c37.627-1 | 37.627 | 36.138 | 35.168 | 0.970 | 0.970 | kept | 0 of 53 | - | 166 (166) | 36 | 155.3 (154.2) | 0 |
| dec1080-1856-c37.627-2 | 37.627 | 36.138 | 35.168 | 0.970 | 0.970 | kept | 0 of 53 | - | 166 (166) | 36 | 155.3 (154.2) | 0 |
| dec1080-1856-c37.527-2 | 37.527 | 36.036 | 35.168 | 0.869 | 0.869 | refused | - | - | 106 (166) | 16 | 64.1 (154.2) | 1 |
| enc1080-512-c6.829-1 | 6.829 | 5.339 | 5.040 | 0.300 | 0.300 | moved | 12 of 14 | 67.51 | 172 (152) | 7 | 157.4 (72.5) | 130 |
| enc1080-512-c6.929-1 | 6.929 | 5.439 | 5.040 | 0.399 | 0.399 | moved | 12 of 14 | 67.57 | 172 (152) | 7 | 156.3 (72.5) | 130 |
| enc1080-512-c7.029-2 | 7.029 | 5.538 | 5.040 | 0.499 | 0.499 | kept | 0 of 14 | - | 152 (152) | 124 | 73.0 (72.5) | 130 |
| enc1080-512-c6.929-2 | 6.929 | 5.439 | 5.040 | 0.399 | 0.399 | moved | 12 of 14 | 67.57 | 172 (152) | 7 | 156.1 (72.5) | 130 |
| enc1080-1024-ref | bare | 94.212 | 11.996 | 82.217 | - | reference | - | - | 175 (-) | 0 | 68.5 (-) | 130 |
| enc1080-1024-c14.036-1 | 14.036 | 12.546 | 11.996 | 0.551 | 0.551 | moved | 4 of 14 | 67.66 | 215 (175) | 161 | 73.2 (68.5) | 130 |
| enc1080-1024-c14.235-1 | 14.235 | 12.745 | 11.996 | 0.750 | 0.750 | kept | 0 of 14 | - | 175 (175) | 112 | 69.3 (68.5) | 130 |
| enc1080-1024-c14.135-1 | 14.135 | 12.646 | 11.996 | 0.650 | 0.650 | kept | 0 of 14 | - | 175 (175) | 134 | 69.4 (68.5) | 130 |
| enc1080-1024-c14.135-2 | 14.135 | 12.646 | 11.996 | 0.650 | 0.650 | kept | 0 of 14 | - | 175 (175) | 134 | 69.3 (68.5) | 130 |
| enc1080-1024-c14.036-2 | 14.036 | 12.546 | 11.996 | 0.551 | 0.551 | moved | 4 of 14 | 67.66 | 215 (175) | 161 | 73.0 (68.5) | 130 |
| enc2160-1472-ref | bare | 94.212 | 21.258 | 72.954 | - | reference | - | - | 212 (-) | 0 | 88.3 (-) | 130 |
| enc2160-1472-c23.368-1 | 23.368 | 21.878 | 21.258 | 0.620 | 0.620 | moved | 3 of 5 | 69.17 | 232 (212) | 46 | 93.6 (88.3) | 130 |
| enc2160-1472-c23.568-1 | 23.568 | 22.077 | 21.258 | 0.820 | 0.820 | kept | 0 of 5 | - | 212 (212) | 52 | 88.9 (88.3) | 130 |
| enc2160-1472-c23.468-1 | 23.468 | 21.978 | 21.258 | 0.720 | 0.720 | kept | 0 of 5 | - | 212 (212) | 60 | 89.0 (88.3) | 130 |
| enc2160-1472-c23.468-2 | 23.468 | 21.978 | 21.258 | 0.720 | 0.720 | kept | 0 of 5 | - | 212 (212) | 60 | 89.1 (88.3) | 130 |
| enc2160-1472-c23.368-2 | 23.368 | 21.878 | 21.258 | 0.620 | 0.620 | moved | 3 of 5 | 69.17 | 232 (212) | 46 | 93.6 (88.3) | 130 |
| enc2160-2048-ref | bare | 94.212 | 39.277 | 54.935 | - | reference | - | - | 225 (-) | 0 | 92.2 (-) | 130 |
| enc2160-2048-c41.437-1 | 41.437 | 39.946 | 39.277 | 0.669 | 0.669 | kept | 0 of 5 | - | 225 (225) | 16 | 92.6 (92.2) | 130 |
| enc2160-2048-c41.237-1 | 41.237 | 39.747 | 39.277 | 0.470 | 0.470 | refused | - | - | 73 (225) | 4 | 28.5 (92.2) | 1 |
| enc2160-2048-c41.337-1 | 41.337 | 39.847 | 39.277 | 0.570 | 0.570 | kept | 0 of 5 | - | 225 (225) | 17 | 92.6 (92.2) | 130 |
| enc2160-2048-c41.337-2 | 41.337 | 39.847 | 39.277 | 0.570 | 0.570 | kept | 0 of 5 | - | 225 (225) | 17 | 92.6 (92.2) | 130 |
| enc2160-2048-c41.237-2 | 41.237 | 39.747 | 39.277 | 0.470 | 0.470 | refused | - | - | 73 (225) | 4 | 28.5 (92.2) | 1 |
| dec1080-832-ref | bare | 94.212 | 13.819 | 80.393 | - | reference | - | - | 283 (-) | 0 | 157.1 (-) | 0 |
| dec1080-832-c16.200-1 | 16.200 | 14.710 | 13.819 | 0.891 | 0.891 | kept | 0 of 53 | - | 283 (283) | 187 | 158.6 (157.1) | 0 |
| dec1080-832-c16.000-1 | 16.000 | 14.509 | 13.819 | 0.690 | 0.690 | moved | 48 of 53 | 72.83 | 303 (283) | 210 | 164.3 (157.1) | 0 |
| dec1080-832-c16.100-1 | 16.100 | 14.611 | 13.819 | 0.791 | 0.791 | kept | 0 of 53 | - | 283 (283) | 194 | 158.8 (157.1) | 0 |
| dec1080-832-c16.100-2 | 16.100 | 14.611 | 13.819 | 0.791 | 0.791 | kept | 0 of 53 | - | 283 (283) | 194 | 159.0 (157.1) | 0 |
| dec1080-832-c16.000-2 | 16.000 | 14.509 | 13.819 | 0.690 | 0.690 | moved | 48 of 53 | 72.83 | 303 (283) | 210 | 164.8 (157.1) | 0 |
| dec2160-1216-ref | bare | 94.212 | 25.922 | 68.290 | - | reference | - | - | 351 (-) | 0 | 203.4 (-) | 0 |
| dec2160-1216-c28.292-1 | 28.292 | 26.802 | 25.922 | 0.880 | 0.880 | kept | 0 of 17 | - | 351 (351) | 48 | 204.2 (203.4) | 0 |
| dec2160-1216-c28.093-1 | 28.093 | 26.603 | 25.342 | 1.261 | 0.680 | refused | - | - | 89 (351) | 7 | 14.8 (203.4) | 1 |
| dec2160-1216-c28.193-1 | 28.193 | 26.702 | 25.342 | 1.361 | 0.780 | refused | - | - | 89 (351) | 7 | 14.8 (203.4) | 1 |
| dec2160-1216-c28.292-2 | 28.292 | 26.802 | 25.922 | 0.880 | 0.880 | kept | 0 of 17 | - | 351 (351) | 48 | 204.3 (203.4) | 0 |
| dec2160-1216-c28.193-2 | 28.193 | 26.702 | 25.342 | 1.361 | 0.780 | refused | - | - | 89 (351) | 7 | 14.9 (203.4) | 1 |
| dec2160-1600-ref | bare | 94.212 | 43.479 | 50.733 | - | reference | - | - | 379 (-) | 0 | 202.0 (-) | 0 |
| dec2160-1600-c45.859-1 | 45.859 | 44.368 | 40.927 | 3.441 | 0.890 | refused | - | - | 84 (379) | 2 | 25.6 (202.0) | 1 |
| dec2160-1600-c46.059-1 | 46.059 | 44.570 | 43.479 | 1.091 | 1.091 | kept | 0 of 17 | - | 379 (379) | 20 | 202.6 (202.0) | 0 |
| dec2160-1600-c45.959-1 | 45.959 | 44.468 | 43.479 | 0.989 | 0.989 | kept | 0 of 17 | - | 379 (379) | 20 | 202.8 (202.0) | 0 |
| dec2160-1600-c45.959-2 | 45.959 | 44.468 | 43.479 | 0.989 | 0.989 | kept | 0 of 17 | - | 379 (379) | 20 | 202.8 (202.0) | 0 |
| dec2160-1600-c45.859-2 | 45.859 | 44.368 | 40.927 | 3.441 | 0.890 | refused | - | - | 84 (379) | 2 | 25.7 (202.0) | 1 |
| dec2160-1920-ref | bare | 94.212 | 61.686 | 32.526 | - | reference | - | - | 309 (-) | 0 | 205.9 (-) | 0 |
| dec2160-1920-c64.077-1 | 64.077 | 62.587 | 61.686 | 0.901 | 0.901 | refused | - | - | 104 (309) | 7 | 58.1 (205.9) | 1 |
| dec2160-1920-c64.277-1 | 64.277 | 62.786 | 61.686 | 1.100 | 1.100 | kept | 0 of 17 | - | 309 (309) | 28 | 207.0 (205.9) | 0 |
| dec2160-1920-c64.177-1 | 64.177 | 62.687 | 61.686 | 1.000 | 1.000 | refused | - | - | 104 (309) | 7 | 57.8 (205.9) | 1 |
| dec2160-1920-c64.277-2 | 64.277 | 62.786 | 61.686 | 1.100 | 1.100 | kept | 0 of 17 | - | 309 (309) | 28 | 207.3 (205.9) | 0 |
| dec2160-1920-c64.177-2 | 64.177 | 62.687 | 61.686 | 1.000 | 1.000 | refused | - | - | 104 (309) | 7 | 58.1 (205.9) | 1 |
| dec1080-1344-ref | bare | 94.212 | 26.024 | 68.188 | - | reference | - | - | 186 (-) | 0 | 152.5 (-) | 0 |
| dec1080-1344-c28.435-1 | 28.435 | 26.945 | 26.024 | 0.920 | 0.920 | kept | 0 of 53 | - | 186 (186) | 44 | 153.3 (152.5) | 0 |
| dec1080-1344-c28.235-1 | 28.235 | 26.745 | 26.024 | 0.721 | 0.721 | refused | - | - | 100 (186) | 24 | 53.9 (152.5) | 1 |
| dec1080-1344-c28.335-1 | 28.335 | 26.845 | 26.024 | 0.821 | 0.821 | kept | 0 of 53 | - | 186 (186) | 51 | 153.5 (152.5) | 0 |
| dec1080-1344-c28.335-2 | 28.335 | 26.845 | 26.024 | 0.821 | 0.821 | kept | 0 of 53 | - | 186 (186) | 51 | 153.5 (152.5) | 0 |
| dec1080-1344-c28.235-2 | 28.235 | 26.745 | 26.024 | 0.721 | 0.721 | refused | - | - | 100 (186) | 24 | 53.8 (152.5) | 1 |
| dec1080-1728-ref | bare | 94.212 | 32.907 | 61.306 | - | reference | - | - | 166 (-) | 0 | 152.9 (-) | 0 |
| dec1080-1728-c35.317-1 | 35.317 | 33.827 | 32.907 | 0.921 | 0.921 | kept | 0 of 53 | - | 166 (166) | 39 | 154.0 (152.9) | 0 |
| dec1080-1728-c35.117-1 | 35.117 | 33.626 | 32.907 | 0.720 | 0.720 | refused | - | - | 102 (166) | 7 | 29.1 (152.9) | 1 |
| dec1080-1728-c35.217-1 | 35.217 | 33.728 | 32.907 | 0.821 | 0.821 | refused | - | - | 102 (166) | 13 | 49.4 (152.9) | 1 |
| dec1080-1728-c35.317-2 | 35.317 | 33.827 | 32.907 | 0.921 | 0.921 | kept | 0 of 53 | - | 166 (166) | 39 | 154.0 (152.9) | 0 |
| dec1080-1728-c35.217-2 | 35.217 | 33.728 | 32.907 | 0.821 | 0.821 | refused | - | - | 102 (166) | 13 | 49.4 (152.9) | 1 |
| dec1080-704-ref | bare | 94.212 | 11.309 | 82.903 | - | reference | - | - | 294 (-) | 0 | 151.6 (-) | 0 |
| dec1080-704-c13.719-1 | 13.719 | 12.230 | 11.309 | 0.920 | 0.920 | kept | 0 of 53 | - | 294 (294) | 100 | 152.5 (151.6) | 0 |
| dec1080-704-c13.519-1 | 13.519 | 12.029 | 11.309 | 0.719 | 0.719 | moved | 8 of 53 | 77.50 | 314 (294) | 123 | 153.7 (151.6) | 0 |
| dec1080-704-c13.619-1 | 13.619 | 12.128 | 11.309 | 0.819 | 0.819 | kept | 0 of 53 | - | 294 (294) | 102 | 152.3 (151.6) | 0 |
| dec1080-704-c13.619-2 | 13.619 | 12.128 | 11.309 | 0.819 | 0.819 | kept | 0 of 53 | - | 294 (294) | 102 | 152.2 (151.6) | 0 |
| dec1080-704-c13.519-2 | 13.519 | 12.029 | 11.309 | 0.719 | 0.719 | moved | 8 of 53 | 77.50 | 314 (294) | 123 | 153.6 (151.6) | 0 |
| dec1080-768-ref | bare | 94.212 | 12.624 | 81.588 | - | reference | - | - | 290 (-) | 0 | 156.3 (-) | 0 |
| dec1080-768-c15.035-1 | 15.035 | 13.544 | 12.624 | 0.920 | 0.920 | kept | 0 of 53 | - | 290 (290) | 65 | 156.9 (156.3) | 0 |
| dec1080-768-c14.835-1 | 14.835 | 13.345 | 12.624 | 0.720 | 0.720 | kept | 0 of 53 | - | 290 (290) | 92 | 157.2 (156.3) | 0 |
| dec1080-768-c14.635-1 | 14.635 | 13.146 | 12.356 | 0.789 | 0.521 | moved | 15 of 53 | 78.53 | 310 (290) | 144 | 159.5 (156.3) | 0 |
| dec1080-768-c14.735-1 | 14.735 | 13.245 | 12.611 | 0.634 | 0.621 | moved | 7 of 53 | 89.80 | 310 (290) | 95 | 157.6 (156.3) | 0 |

### Under expandable segments, on an RTX 5090 (2026-10-17)

The same floor for two of those tiles on a card whose driver gives memory the card doesn't have:
an RTX 5090 under WSL2 (driver 616.92, torch 2.14.1+cu130, cuDNN 9.24, SDPA), where seedvr2x holds
itself to the card under torch's own allocator with expandable segments
([DESIGN.md](../../seedvr2x/DESIGN.md#allocator)). No ballast is refused there, so no smaller card
can be emulated: the room is set by torch's limit
([`scripts/held_floor.py`](../scripts/held_floor.py), `vae_room.py run --limit`). seedvr2x at
6e1be76, its code unchanged; each card's frames judged against that card's own reference only.

#### Method as run

- **The 1080p shape and its units, as above:** clip-53 (the matrix's source, 53 frames from its
  48th, cut by planner_matrix.py's own trim), window 8, encode in 768-px tiles, 36 blocks
  swapped, one shot, run once under seedvr2x's own limit and stopped before its decode. A rung
  is seedvr2x's resume of a copy whose manifest says decode tile T, in a process of its own, the
  planner's room at −50 GiB and the DiT's phase unjudged: one plan on every rung, the DiT's 2
  slots on the GPU beside the pass. The allocator: `backend:native,expandable_segments:True`,
  seedvr2x's own default there.
- **The room:** once seedvr2x's hold has set torch's limit (the plan made, the models not
  loaded), torch's limit set to the tile's reference's decode peak (torch's max allocated in the
  unit) and the room, `memory._limit` with it, never above seedvr2x's own (the card's free
  memory less 0.5 GiB: 23.98 to 24.03 GiB): the card was never passed, 4.57 GiB free at the
  least beside a pass so held. The reference is the same resume under seedvr2x's own limit, 10.2
  GiB over the 832-px peak and 5.1 over the 1,024-px one.
- **The frames:** every frame's CRC-32 against the reference's, the worst PSNR-Y where they
  differ; refused: out of memory.
- **The ladder:** 1.0 GiB, down by 0.2 while the frames hold, then by 0.1; the least room kept
  and the most lost run again; then 0.35 once each.
- **What the run could see,** the clock aside: the plans cuDNN's frontend built in the pass
  (its log on), torch's counters across it (`num_ooms`, `num_alloc_retries`,
  `num_oom_rejections`, `num_sync_all_streams`, `num_device_alloc`, `num_device_free`). The
  pass's time is recorded, for information. The desktop held 6.3 to 7.2 GiB of the card
  (nvidia-smi before and after every run), no other compute process.

#### The floors

The references are the 96 GB card's to the MiB: a decode peak of 13.819 GiB and 283 plans built
in 832-px tiles, 18.971 GiB and 308 plans in 1,024-px tiles, the same engines (22, 58 and 23).

| tile | least room kept (runs) | most room lost (runs) | the 96 GB card's (`cudaMallocAsync`): least kept / most lost |
|---|---|---|---|
| 832 × 832 | 0.40 GiB (2: kept, kept) | 0.35 (1: moved); 0.30 (2: moved, moved); 0.20 (1: moved) | 0.791 / 0.690 (moved) |
| 1024 × 1024 | 0.40 GiB (2: kept, kept) | 0.35 (1: refused); 0.30 (2: refused, refused); 0.20 (1: refused) | 0.751 / 0.651 (refused) |

Decode in 832-px tiles (reference: 27.1 s, its peak reserved 16.256 GiB, counters 0):

| room GiB | outcome | frames differing (of 53) | worst PSNR-Y dB | plans built (over the reference's) | `num_ooms` | `num_alloc_retries` | pass s (times the reference's) |
|---|---|---|---|---|---|---|---|
| 1.00 | kept | 0 | - | 283 (+0) | 0 | 83 | 37.3 (1.38) |
| 0.80 | kept | 0 | - | 283 (+0) | 0 | 143 | 41.1 (1.52) |
| 0.60 | kept | 0 | - | 283 (+0) | 0 | 155 | 42.5 (1.57) |
| 0.40 | kept, kept | 0, 0 | - | 283 (+0) | 0, 0 | 192, 192 | 41.9, 41.4 (1.55, 1.53) |
| 0.35 | moved | 48 from the 6th | 71.34 | 303 (+20) | 20 | 220 | 42.7 (1.58) |
| 0.30 | moved, moved | 48, 48 from the 6th | 71.34, 71.34 | 303 (+20) | 20, 20 | 208, 208 | 43.2, 42.1 (1.59, 1.55) |
| 0.20 | moved | 48 from the 6th | 71.16 | 302 (+19) | 19 | 240 | 44.9 (1.66) |

Decode in 1,024-px tiles (reference: 28.7 s, its peak reserved 22.857 GiB, `num_ooms` 0,
`num_alloc_retries` 4):

| room GiB | outcome | frames differing (of 53) | plans built | `num_ooms` | `num_alloc_retries` | pass s (times the reference's) |
|---|---|---|---|---|---|---|
| 1.00 | kept | 0 | 308 (+0) | 0 | 85 | 40.1 (1.40) |
| 0.80 | kept | 0 | 308 (+0) | 0 | 97 | 42.2 (1.47) |
| 0.60 | kept | 0 | 308 (+0) | 0 | 111 | 42.3 (1.47) |
| 0.40 | kept, kept | 0, 0 | 308 (+0) | 0, 0 | 146, 146 | 45.9, 45.3 (1.60, 1.58) |
| 0.35 | refused, slice 8 | - | 114, engine 28 among them | 21 | 76 | 13.0 to the refusal |
| 0.30 | refused, refused, slice 2 | - | 100, engine 28 among them | 20, 20 | 33, 33 | 4.6, 4.2 |
| 0.20 | refused, slice 2 | - | 100, engine 28 among them | 20 | 34 | 4.8 |

- **One mechanism on both cards.** The same convolution's workspace refused (its top plan's,
  of engine 22: 0.422 GiB in 832-px tiles, 0.738 in 1,024-px ones, as on the 96 GB card), then
  16 more plans of that engine and 2 of engine 23 (0.755 and 1.320 GiB) refused, then engine 28,
  with no workspace, run: in 832-px tiles 48 of 53 frames moved from the 6th, the first slice's
  5 kept (fallen back in slice 7 at 0.35 and 0.30, the same other frames each time; in slice 2
  at 0.20, others). In 1,024-px tiles the pass fell back too, then was refused in the same slice
  at the concatenation of a split convolution's pieces (`memory_limit_conv:162`, 2.00 GiB asked
  with 16.92 allocated), the 96 GB card's site.
- **The floor 0.35 to 0.39 GiB lower than the 96 GB card's.** At the refusals torch's own
  allocator held 0.348 GiB (slice 2) and 0.413 (slice 8) beyond what was allocated and couldn't
  give them back (torch's message), where `cudaMallocAsync`'s pool held 2.6 to 2.7 GiB more at
  the 96 GB card's refusals in the same tile. The 2.00 GiB request was over the limit by 0.19
  MiB at 0.30 GiB of room, 8.99 MiB at 0.35, 102.59 MiB at 0.20.
- **The pass's time, for information only:** under the segments every pass at a room ran 1.38 to
  1.66 times as long as its reference from 1.0 GiB down, kept or moved, torch giving its cache back
  to make room (`num_alloc_retries`); not tried as a signal (deferred on this card, 2026-10-16).

#### What the run can see: torch's own allocator counts the refusals

| across the pass | 14 runs that kept their frames | 5 that moved them | 4 refused |
|---|---|---|---|
| torch's `num_ooms` | 0 in every one | 19 to 20 | 20 to 21 |
| cuDNN's plans built over the reference's | 0 | +19 to +20, engine 28 among them | engine 28 built in each, before the refusal |
| torch's `num_alloc_retries` | 83 to 192 | 208 to 240 | 33 to 76 |

(The 23 runs at a room: the rungs above and the whole runs below.) `num_ooms` counts each refused
workspace exactly: 19 for the convolution's top plan, 16 more of its engine and 2 of engine 23;
one more where the plan cached for that shape was refused first; one more for the run's own
refusal. It stood at 0 in the four references and in every pass that kept its frames: under
torch's own allocator a run can see a fallback at no cost, where `cudaMallocAsync` counts
nothing ([above](#what-the-run-can-see)). `num_alloc_retries` doesn't separate: it counts the
squeeze, rising as the room falls in passes that keep their frames, and 4 in the 1,024-px
reference itself. `num_oom_rejections` stayed at 0.

#### Whole runs

The encode, the windows and the decode in one process, torch's limit the pass alone's reference
peak and the room from the plan on. Their references give the passes alone's frames, CRC-32 for
CRC-32 (decode peaks 13.824 and 18.976 GiB). In 832-px tiles (the planner's room as shipped):
kept with 0.40 GiB twice (`num_ooms` 0), moved with 0.30 (48 frames, 71.16 dB, `num_ooms` 20).
In 1,024-px tiles the planner refused the pinned tile at its room of 1.5 GiB beside that
desktop, 0.01 GiB short ("the largest tile that fits is --decode-tile 960"), so they ran planned
at −50 GiB as the rungs are: kept with 0.40 twice (`num_ooms` 0). On this allocator a whole run
lost nothing a pass alone kept, at 0.1 GiB.

So under expandable segments on that card a VAE pass keeps its frames with 0.40 GiB beyond its
peak in tiles of 832 and 1,024 px, where a held run's plan leaves it RESERVE's 4.5 GiB and the
room's 1.5 beyond its predicted peak (runtime/memory.py, RESERVE): 6.0 GiB.

### The large tiles under the one-output cure, and the tiles left (2026-10-17)

The ladders the floor by tile size left: the 1080p and 4K tiles the one-output cure takes and
1080p's untiled decode, run with the cure; and, on 6e1be76 as above, the tiles it doesn't take that
hadn't run, those the planner gives the cards of 12 to 48 GB. The same driver
([`scripts/floor.py`](../scripts/floor.py), a ladder's code given by its name), one hold of the
GPU's lock a rung: 347 rungs in 57 ladders, none failed. Not run: under the cure, 4K's untiled
encode (the 96 GB card's own 4K plan encodes untiled, an output of 8.0 GiB a slice of four frames)
and the other output sizes' passes it takes (1440p's tiles cut off at its 1,472 rows, its untiled
decode and encode, among them); on 6e1be76, the decodes in 512- and 576-px tiles.

#### Method as run

- **Two codes, a ladder's by its name.** A ladder named as above (`dec1080-896`) ran seedvr2x at
  6e1be76, as the first ladders did. One named with `-a4` (`dec2160-1216-a4`) ran seedvr2x as
  changed after 6e1be76 on 2026-10-17, the one-output cure in it: a split convolution's pieces
  written into one output allocated first, where a slice of four frames makes that output 2.5 GiB or
  more, the path chosen once a pass. That code was another frozen copy with its own environment,
  its tree checked before every rung as the first copy's was.
- **What ran under the cure:** the decode squares from 1,152 px (4K, 15 tiles), the tiles cut off at
  1080p's rows from 1,104 × 1,216 (11), 1080p's untiled decode, and the encode squares from 1,664 px
  (4K, 7). The tiles among these that the first ladders ran on 6e1be76 ran again under the cure;
  their first ladders stay as 6e1be76's record.
- **What ran on 6e1be76:** the decodes in 768-px tiles (the first ladders' unfinished one, its
  deciding rungs run again), 896, 960 and 1,088 px; 1080p's untiled encode; and the encodes the
  planner gives the cards of 12 to 48 GB: 1080p's squares of 768 to 1,088 px and its tiles of
  1,104 × 1,152 to 1,104 × 1,792, 4K's squares of 1,152 to 1,600 px. 4K's squares of 768 to 1,088 px
  are 1080p's own shapes and weren't run apart.
- **Each code its own units and references.** The shapes are those above (clip-53 at 1080p,
  clip-17 at 4K), their units made once on each code. On the cure's code the noise is drawn on the
  CPU, so its frames are its own. Each tile's reference is its own bare-card rung on its code. The
  room, the frames, the ladder and the signals: as above.
- **The untiled passes as ladders:** `--decode-tile 0` (the units' copy saying decode tile 0 and no
  decode tiles) and `--encode-tile 0`. Every untiled rung recorded two things. Whether the untiled
  pass is what ran: its own plan had no tiles for the pass, and the pass ran one stream's 13 slices,
  as many as a tiled pass runs a tile. And the pass's peak above what the GPU held once its phase was
  entered, against the planner's untiled fit of 1,104 × 1,920 (`planner._fit`: 19.853 GiB for the
  encode, 34.927 for the decode, on both codes).
- **The order:** first the tiles the planner gives the cards of 12 to 48 GB at a room of 1.0 GiB.
  These were the cure's decodes by the least card that takes them (1,104 × 1,216 at 26 GB to 4K's
  1,600 px at 46 to 48), its encodes, the two untiled passes, then 6e1be76's decodes and encodes.
  Then the cure's decodes that no such card takes (4K's 1,664 to 2,048 px).
- **The run:** from 03:59 to 18:57 UTC. The queue was interrupted once (06:36 UTC): the rung then
  running (4K's 1,280-px decode, 0.67 GiB of room) was stopped by SIGTERM 220 s into its pass,
  recorded by vae_room.py (exit 143) but not by the driver. It ran again once the queue was started
  again (07:21 UTC), as that card's second run; nothing was overwritten, and nothing else was lost.
  No other compute process was on the card at any rung (nvidia-smi before each). The card's offset
  was 1.489 to 1.491 GiB. A rung took 25 to 255 s, for information.

#### The floors

| ladder | code | tile | ref peak GiB | ref pass s | ref plans | least room kept (runs) | most room lost (runs) | done |
|---|---|---|---|---|---|---|---|---|
| dec1080-1216-a4 | a4 | 1104x1216 | 23.763 | 151.9 | 189 | 0.801 (2: 1: kept, 2: kept) | 0.699 (2: 1: moved, 2: moved) | yes |
| dec2160-1152-a4 | a4 | 1152x1152 | 23.507 | 203.2 | 346 | 0.750 (2: 1: kept, 2: kept) | 0.650 (2: 1: moved, 2: moved) | yes |
| dec1080-1280-a4 | a4 | 1104x1280 | 24.679 | 152.4 | 180 | 0.830 (2: 1: kept, 2: kept) | 0.730 (2: 1: moved, 2: moved) | yes |
| dec1080-1344-a4 | a4 | 1104x1344 | 25.701 | 152.3 | 186 | 0.941 (2: 1: kept, 2: kept) | 0.841 (2: 1: moved, 2: moved) | yes |
| dec2160-1216-a4 | a4 | 1216x1216 | 25.659 | 203.2 | 351 | 0.680 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| dec1080-1408-a4 | a4 | 1104x1408 | 26.731 | 153.6 | 175 | 0.851 (2: 1: kept, 2: kept) | 0.751 (2: 1: moved, 2: moved) | yes |
| dec1080-1472-a4 | a4 | 1104x1472 | 27.787 | 152.0 | 172 | 0.861 (2: 1: kept, 2: kept) | 0.759 (2: 1: moved, 2: moved) | yes |
| dec2160-1280-a4 | a4 | 1280x1280 | 27.917 | 203.1 | 329 | 0.869 (2: 1: kept, 2: kept) | 0.770 (2: 1: moved, 2: moved) | yes |
| dec1080-1536-a4 | a4 | 1104x1536 | 28.820 | 150.1 | 178 | 0.870 (2: 1: kept, 2: kept) | 0.771 (2: 1: moved, 2: moved) | yes |
| dec1080-1600-a4 | a4 | 1104x1600 | 29.781 | 148.2 | 183 | 0.680 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| dec1080-1664-a4 | a4 | 1104x1664 | 30.793 | 150.2 | 174 | 0.860 (2: 1: kept, 2: kept) | 0.761 (2: 1: moved, 2: moved) | yes |
| dec2160-1344-a4 | a4 | 1344x1344 | 30.359 | 201.4 | 363 | 0.771 (2: 1: kept, 2: kept) | 0.671 (2: 1: moved, 2: moved) | yes |
| dec1080-1728-a4 | a4 | 1104x1728 | 31.776 | 152.4 | 166 | 0.960 (2: 1: kept, 2: kept) | 0.858 (2: 1: moved, 2: moved) | yes |
| dec1080-1792-a4 | a4 | 1104x1792 | 32.802 | 150.7 | 162 | 0.971 (2: 1: kept, 2: kept) | 0.869 (2: 1: moved, 2: moved) | yes |
| dec2160-1408-a4 | a4 | 1408x1408 | 32.835 | 201.5 | 387 | 0.790 (2: 1: kept, 2: kept) | 0.690 (2: 1: moved, 2: moved) | yes |
| dec1080-1856-a4 | a4 | 1104x1856 | 33.901 | 153.9 | 166 | 0.782 (2: 1: kept, 2: kept) | 0.680 (2: 1: moved, 2: moved) | yes |
| dec2160-1472-a4 | a4 | 1472x1472 | 35.496 | 201.0 | 382 | 0.880 (2: 1: kept, 2: kept) | 0.781 (2: 1: moved, 2: moved) | yes |
| dec2160-1536-a4 | a4 | 1536x1536 | 38.265 | 202.1 | 367 | 0.680 (2: 1: kept, 2: kept) | 0.578 (2: 1: moved, 2: moved) | yes |
| dec2160-1600-a4 | a4 | 1600x1600 | 41.170 | 200.8 | 379 | 0.970 (2: 1: kept, 2: kept) | 0.870 (2: 1: moved, 2: moved) | yes |
| enc2160-1664-a4 | a4 | 1664x1664 | 26.398 | 89.7 | 205 | 0.500 (2: 1: kept, 2: kept) | 0.400 (2: 1: moved, 2: moved) | yes |
| enc2160-1728-a4 | a4 | 1728x1728 | 28.242 | 89.9 | 202 | 0.550 (2: 1: kept, 2: kept) | 0.451 (2: 1: moved, 2: moved) | yes |
| enc2160-1792-a4 | a4 | 1792x1792 | 29.957 | 90.3 | 195 | 0.579 (2: 1: kept, 2: kept) | 0.479 (2: 1: moved, 2: moved) | yes |
| enc2160-1856-a4 | a4 | 1856x1856 | 31.979 | 87.8 | 200 | 0.591 (2: 1: kept, 2: kept) | 0.489 (2: 1: moved, 2: moved) | yes |
| enc2160-1920-a4 | a4 | 1920x1920 | 33.812 | 79.8 | 191 | 0.601 (2: 1: kept, 2: kept) | 0.501 (2: 1: moved, 2: moved) | yes |
| enc2160-1984-a4 | a4 | 1984x1984 | 35.804 | 84.2 | 228 | 0.509 (2: 1: kept, 2: kept) | 0.410 (2: 1: moved, 2: moved) | yes |
| enc2160-2048-a4 | a4 | 2048x2048 | 37.848 | 90.2 | 225 | 0.500 (2: 1: kept, 2: kept) | 0.399 (2: 1: moved, 2: moved) | yes |
| dec1080-untiled-a4 | a4 | untiled (1104x1920) | 34.952 | 149.0 | 100 | 0.879 (2: 1: kept, 2: kept) | 0.780 (2: 1: moved, 2: moved) | yes |
| enc1080-untiled | base | untiled (1104x1920) | 20.871 | 54.4 | 59 | 0.661 (2: 1: kept, 2: kept) | 0.561 (2: 1: refused, 2: refused) | yes |
| dec1080-768 | base | 768x768 | 12.624 | 156.3 | 290 | 0.720 (2: 1 (history): kept, 2: kept) | 0.621 (2: 1 (history): moved, 2: moved) | yes |
| dec1080-896 | base | 896x896 | 15.621 | 134.2 | 274 | 0.810 (2: 1: kept, 2: kept) | 0.710 (2: 1: moved, 2: moved) | yes |
| dec1080-960 | base | 960x960 | 17.196 | 151.2 | 270 | 0.809 (2: 1: kept, 2: kept) | 0.709 (2: 1: moved, 2: moved) | yes |
| dec1080-1088 | base | 1088x1088 | 21.169 | 159.7 | 329 | 0.901 (2: 1: kept, 2: kept) | 0.799 (2: 1: refused, 2: refused) | yes |
| enc1080-768 | base | 768x768 | 8.604 | 60.8 | 159 | 0.670 (2: 1: kept, 2: kept) | 0.571 (2: 1: moved, 2: moved) | yes |
| enc1080-896 | base | 896x896 | 10.081 | 68.8 | 161 | 0.580 (2: 1: kept, 2: kept) | 0.481 (2: 1: moved, 2: moved) | yes |
| enc1080-960 | base | 960x960 | 11.338 | 70.8 | 163 | 0.470 (2: 1: kept, 2: kept) | 0.370 (2: 1: moved, 2: moved) | yes |
| enc1080-1088 | base | 1088x1088 | 13.328 | 70.2 | 178 | 0.550 (2: 1: kept, 2: kept) | 0.451 (2: 1: moved, 2: moved) | yes |
| enc1080-1152 | base | 1104x1152 | 14.206 | 66.7 | 102 | 0.750 (2: 1: kept, 2: kept) | 0.651 (2: 1: moved, 2: moved) | yes |
| enc1080-1280 | base | 1104x1280 | 15.068 | 66.8 | 102 | 0.558 (2: 1: kept, 2: kept) | 0.459 (2: 1: moved, 2: moved) | yes |
| enc1080-1408 | base | 1104x1408 | 16.422 | 67.1 | 100 | 0.659 (2: 1: kept, 2: kept) | 0.560 (2: 1: moved, 2: moved) | yes |
| enc1080-1472 | base | 1104x1472 | 17.099 | 66.9 | 97 | 0.660 (2: 1: kept, 2: kept) | 0.560 (2: 1: moved, 2: moved) | yes |
| enc1080-1600 | base | 1104x1600 | 17.944 | 66.5 | 100 | 0.870 (2: 1: kept, 2: kept) | 0.770 (2: 1: moved, 2: moved) | yes |
| enc1080-1728 | base | 1104x1728 | 19.257 | 66.4 | 96 | 0.681 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| enc1080-1792 | base | 1104x1792 | 19.914 | 62.9 | 96 | 0.691 (2: 1: kept, 2: kept) | 0.589 (2: 1: moved, 2: moved) | yes |
| enc2160-1152 | base | 1152x1152 | 14.757 | 81.9 | 196 | 0.590 (2: 1: kept, 2: kept) | 0.491 (2: 1: moved, 2: moved) | yes |
| enc2160-1216 | base | 1216x1216 | 15.673 | 87.7 | 190 | 0.591 (2: 1: kept, 2: kept) | 0.490 (2: 1: moved, 2: moved) | yes |
| enc2160-1280 | base | 1280x1280 | 17.194 | 88.6 | 185 | 0.580 (2: 1: kept, 2: kept) | 0.479 (2: 1: moved, 2: moved) | yes |
| enc2160-1344 | base | 1344x1344 | 18.312 | 88.1 | 220 | 0.680 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| enc2160-1408 | base | 1408x1408 | 19.978 | 88.2 | 213 | 0.680 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| enc2160-1536 | base | 1536x1536 | 23.018 | 88.8 | 211 | 0.589 (2: 1: kept, 2: kept) | 0.489 (2: 1: refused, 2: refused) | yes |
| enc2160-1600 | base | 1600x1600 | 24.731 | 88.7 | 204 | 0.681 (2: 1: kept, 2: kept) | 0.581 (2: 1: refused, 2: refused) | yes |
| dec2160-1664-a4 | a4 | 1664x1664 | 43.976 | 154.1 | 388 | 0.781 (2: 1: kept, 2: kept) | 0.681 (2: 1: moved, 2: moved) | yes |
| dec2160-1728-a4 | a4 | 1728x1728 | 47.202 | 164.7 | 343 | 0.879 (2: 1: kept, 2: kept) | 0.781 (2: 1: moved, 2: moved) | yes |
| dec2160-1792-a4 | a4 | 1792x1792 | 50.449 | 173.5 | 332 | 0.781 (2: 1: kept, 2: kept) | 0.679 (2: 1: moved, 2: moved) | yes |
| dec2160-1856-a4 | a4 | 1856x1856 | 54.000 | 166.4 | 349 | 0.780 (2: 1: kept, 2: kept) | 0.681 (2: 1: moved, 2: moved) | yes |
| dec2160-1920-a4 | a4 | 1920x1920 | 57.208 | 174.5 | 309 | 0.680 (2: 1: kept, 2: kept) | 0.580 (2: 1: moved, 2: moved) | yes |
| dec2160-1984-a4 | a4 | 1984x1984 | 60.770 | 174.5 | 380 | 0.770 (2: 1: kept, 2: kept) | 0.670 (2: 1: moved, 2: moved) | yes |
| dec2160-2048-a4 | a4 | 2048x2048 | 64.863 | 181.3 | 364 | 0.870 (2: 1: kept, 2: kept) | 0.769 (2: 1: moved, 2: moved) | yes |

"base": seedvr2x at 6e1be76; "a4": with the cure. The 768-px decode's first rungs are the first
ladders' (its reference too), marked "(history)" below.

- **Under the cure, the large tiles' frames move where 6e1be76 refused them first.** All 27 decode
  ladders under the cure moved their frames; none was refused. They kept them with 0.68 to 0.97 GiB
  left and moved them with 0.1 less:
  - 8 or 12 of 4K's 17 frames moved, from the 6th or the 10th: the first slice's five held;
  - 8 to 48 of 1080p's 53 moved, from the 6th to the 46th;
  - worst PSNR-Y 70.9 to 76.6 dB at 4K, 71.2 to 73.8 at 1080p.

  On 6e1be76 every one of these tiles that ran was refused at the concatenation of a split
  convolution's pieces, with 0.60 to 1.00 GiB left, before any frame moved. The 7 encodes under the
  cure kept their latent with 0.50 to 0.60 GiB, and with 0.1 less moved 2 or 3 of its 5 latent
  frames (66.3 to 69.2 dB).
- **A tile on both codes** (the table below):
  - under the cure the reference's peak is 0.26 to 5.14 GiB lower, the concatenation's second copy
    gone (0.26 at 1,216 px, 5.14 at 2,048);
  - the free memory a pass asks to keep its frames (its peak and its room) is 0.20 to 4.97 GiB less;
  - the room alone is lower in 7 tiles of 11 and higher in 4 (4K's 2,048-px decode by 0.17,
    1,104 × 1,344 by 0.12, × 1,728 by 0.04, × 1,536 by 0.01).
- **6e1be76's tiles left.**
  - Decodes in 768-, 896- and 960-px tiles kept their frames with 0.72, 0.81 and 0.81 GiB and moved
    them with 0.1 less (7, 44 and 48 of 53 frames).
  - In 1,088-px tiles the decode was refused at the concatenation (2.26 GiB asked) with 0.80 left,
    before its frames moved; it kept them above, with 0.90.
  - The 768-px ladder's deciding rungs, run again, gave the first ladders' outcomes and frames.
  - Encodes kept their latent with 0.47 to 0.67 GiB in 1080p's squares, 0.56 to 0.87 in its
    1,104-row tiles (1,104 × 1,600 the most) and 0.58 to 0.68 in 4K's squares of 1,152 to 1,600 px.
    With 0.1 less they moved it: 5 to 12 of 14 latent frames at 1080p, 3 of 5 at 4K, 66.5 to
    73.2 dB.
  - The exception is 4K's 1,536- and 1,600-px encodes: refused at the concatenation (2.25 and 2.44
    GiB asked) with 0.49 and 0.58 left.
- **The untiled passes ran untiled** on every rung that ran to its end (the table below).
  - 1080p's decode under the cure kept its frames with 0.88 GiB and moved them with 0.78 (36 of 53).
    Its pass peaked at 33.430 GiB above what the GPU held, 1.50 under the planner's fit (34.927).
  - 1080p's encode on 6e1be76 kept its latent with 0.66 and was refused with 0.56 and 0.46, at the
    concatenation (2.02 GiB asked) in its third slice, after a fallback in that slice (20 plans
    more). Its pass peaked at 19.350, 0.50 under the fit (19.853).
  - The refused rungs stopped in their third slice of 13 ("ran untiled" false: their pass didn't
    end); their plan was untiled.
  - The phase's peaks, models included, were 2.20 and 0.91 GiB under the plan's predictions.

The tiles on both codes (the cure's, then 6e1be76's: the first ladders' for these tiles):

| tile | code | ref peak | least kept / most lost | free asked | code | ref peak | least kept / most lost | free asked |
|---|---|---|---|---|---|---|---|---|
| dec1080-1344 (1104x1344) | a4 | 25.701 | 0.941 / 0.841 (moved) | 26.642 | base | 26.024 | 0.821 / 0.721 (refused) | 26.845 |
| dec2160-1216 (1216x1216) | a4 | 25.659 | 0.680 / 0.580 (moved) | 26.339 | base | 25.922 | 0.880 / 0.780 (refused) | 26.802 |
| dec1080-1536 (1104x1536) | a4 | 28.820 | 0.870 / 0.771 (moved) | 29.691 | base | 29.463 | 0.860 / 0.761 (refused) | 30.323 |
| dec1080-1728 (1104x1728) | a4 | 31.776 | 0.960 / 0.858 (moved) | 32.736 | base | 32.907 | 0.921 / 0.821 (refused) | 33.827 |
| dec2160-1408 (1408x1408) | a4 | 32.835 | 0.790 / 0.690 (moved) | 33.624 | base | 34.089 | 0.980 / 0.880 (refused) | 35.070 |
| dec1080-1856 (1104x1856) | a4 | 33.901 | 0.782 / 0.680 (moved) | 34.683 | base | 35.168 | 0.970 / 0.869 (refused) | 36.138 |
| dec2160-1600 (1600x1600) | a4 | 41.170 | 0.970 / 0.870 (moved) | 42.140 | base | 43.479 | 0.989 / 0.890 (refused) | 44.468 |
| enc2160-2048 (2048x2048) | a4 | 37.848 | 0.500 / 0.399 (moved) | 38.349 | base | 39.277 | 0.570 / 0.470 (refused) | 39.847 |
| dec2160-1792 (1792x1792) | a4 | 50.449 | 0.781 / 0.679 (moved) | 51.230 | base | 53.999 | 0.959 / 0.859 (refused) | 54.958 |
| dec2160-1920 (1920x1920) | a4 | 57.208 | 0.680 / 0.580 (moved) | 57.888 | base | 61.686 | 1.100 / 1.000 (refused) | 62.786 |
| dec2160-2048 (2048x2048) | a4 | 64.863 | 0.870 / 0.769 (moved) | 65.734 | base | 70.004 | 0.700 / 0.601 (refused) | 70.704 |

The untiled passes (GiB; "peak" the phase's, models included, against the plan's prediction; "pass peak" above what the GPU held once the phase was entered, against the planner's fit):

| rung | outcome | plan's tiles | slices (one stream) | ran untiled | peak | predicted (plan) | pass peak | fit | over the fit |
|---|---|---|---|---|---|---|---|---|---|
| dec1080-untiled-a4-ref | reference | None | 13 (13) | True | 34.952 | 37.151 | 33.430 | 34.927 | -1.496 |
| dec1080-untiled-a4-c37.322-1 | kept | None | 13 (13) | True | 34.952 | 37.151 | 33.430 | 34.927 | -1.496 |
| dec1080-untiled-a4-c37.122-1 | moved | None | 13 (13) | True | 34.855 | 37.151 | 33.334 | 34.927 | -1.593 |
| dec1080-untiled-a4-c37.222-1 | moved | None | 13 (13) | True | 34.952 | 37.151 | 33.430 | 34.927 | -1.496 |
| dec1080-untiled-a4-c37.322-2 | kept | None | 13 (13) | True | 34.952 | 37.151 | 33.430 | 34.927 | -1.496 |
| dec1080-untiled-a4-c37.222-2 | moved | None | 13 (13) | True | 34.952 | 37.151 | 33.430 | 34.927 | -1.496 |
| enc1080-untiled-ref | reference | None | 13 (13) | True | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |
| enc1080-untiled-c23.022-1 | kept | None | 13 (13) | True | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |
| enc1080-untiled-c22.822-1 | refused | None | 2 (13) | False | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |
| enc1080-untiled-c22.922-1 | refused | None | 2 (13) | False | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |
| enc1080-untiled-c23.022-2 | kept | None | 13 (13) | True | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |
| enc1080-untiled-c22.922-2 | refused | None | 2 (13) | False | 20.871 | 21.777 | 19.350 | 19.853 | -0.503 |

#### What the run can see

What separates the moved rungs from the kept ones, every rung of the tail (a higher figure saying moved):

- torch's count of refused allocations across the pass (num_ooms + num_alloc_retries + num_oom_rejections): kept 0 to 0, moved 0 to 0: does NOT separate; over 0: 0 true positives, 0 false; 147 true negatives, 133 false
- the cuDNN frontend's plans built in the pass over the reference's (cuDNN's log): kept 0 to 0, moved 19 to 40: SEPARATES, over 9.5 said moved: 133 true positives, 0 false; 147 true negatives, 0 false
- torch's "recovered from an allocation failure" lines in the pass: kept 15 to 166, moved 21 to 166: does NOT separate; over 93.5: 36 true positives, 41 false; 106 true negatives, 97 false
- the pass's time: not a signal here (its seconds are in every rung's record, for information): a power-capped card's speed drifts from run to run

The codes apart: under the cure the plans built separate 87 moved rungs from 91 kept, on 6e1be76 46 from 56, neither with a miss; torch's count is 0 on both.

Over the 280 rungs that kept or moved their frames (147 kept, 133 moved; 11 refused apart, all on
6e1be76):

- **The cuDNN frontend's plans built tell it without a miss again,** on both codes: 19 to 40 more
  than the reference's in every pass whose frames moved, none in any that kept them. Each
  convolution that fell back built the same plans: engine 22 sixteen or seventeen times, engine 23
  twice, then engine 28 with no workspace, which ran. One convolution fell back (19 or 20 plans
  more), or two (38 to 40).
- **torch's count of refused allocations stayed at 0** across every pass under `cudaMallocAsync`.
- **The "recovered" lines don't separate.**
- **The refused rungs:** all at `memory_limit_conv:162`, the concatenation, on 6e1be76. Five of them
  (1080p's untiled encode, 4K's 1,536-px encode) fell back in the slice refused, before the refusal.

#### The floor against the tile

- dec: every bracketed ladder kept its frames with 0.901 GiB (dec1080-1088) or less, over 4 ladders
- dec, least kept against the area: 0.556 + 0.2908 x, worst miss 0.021 GiB over 4 ladders
- dec, least kept against the workspace: 0.556 + 0.0976 x, worst miss 0.021 GiB over 4 ladders
- dec1080: every bracketed ladder kept its frames with 0.901 GiB (dec1080-1088) or less, over 4 ladders
- dec1080, least kept against the area: 0.556 + 0.2908 x, worst miss 0.021 GiB over 4 ladders
- dec1080, least kept against the workspace: 0.556 + 0.0976 x, worst miss 0.021 GiB over 4 ladders
- dec-a4: every bracketed ladder kept its frames with 0.971 GiB (dec1080-1792-a4) or less, over 27 ladders
- dec-a4, least kept against the area: 0.853 + -0.0137 x, worst miss 0.153 GiB over 27 ladders
- dec-a4, least kept against the workspace: 0.853 + -0.0046 x, worst miss 0.153 GiB over 27 ladders
- dec1080-a4: every bracketed ladder kept its frames with 0.971 GiB (dec1080-1792-a4) or less, over 12 ladders
- dec1080-a4, least kept against the area: 0.772 + 0.0493 x, worst miss 0.179 GiB over 12 ladders
- dec1080-a4, least kept against the workspace: 0.772 + 0.0166 x, worst miss 0.179 GiB over 12 ladders
- dec2160-a4: every bracketed ladder kept its frames with 0.970 GiB (dec2160-1600-a4) or less, over 15 ladders
- dec2160-a4, least kept against the area: 0.775 + 0.0077 x, worst miss 0.175 GiB over 15 ladders
- dec2160-a4, least kept against the workspace: 0.775 + 0.0026 x, worst miss 0.175 GiB over 15 ladders
- enc: every bracketed ladder kept its frames with 0.870 GiB (enc1080-1600) or less, over 19 ladders
- enc, least kept against the area: 0.551 + 0.0570 x, worst miss 0.218 GiB over 19 ladders
- enc, least kept against the workspace: 0.547 + 0.0455 x, worst miss 0.218 GiB over 19 ladders
- enc1080: every bracketed ladder kept its frames with 0.870 GiB (enc1080-1600) or less, over 12 ladders
- enc1080, least kept against the area: 0.513 + 0.0958 x, worst miss 0.187 GiB over 12 ladders
- enc1080, least kept against the workspace: 0.506 + 0.0775 x, worst miss 0.186 GiB over 12 ladders
- enc2160: every bracketed ladder kept its frames with 0.681 GiB (enc2160-1600) or less, over 7 ladders
- enc2160, least kept against the area: 0.527 + 0.0533 x, worst miss 0.064 GiB over 7 ladders
- enc2160, least kept against the workspace: 0.527 + 0.0411 x, worst miss 0.064 GiB over 7 ladders
- enc-a4: every bracketed ladder kept its frames with 0.601 GiB (enc2160-1920-a4) or less, over 7 ladders
- enc-a4, least kept against the area: 0.586 + -0.0111 x, worst miss 0.056 GiB over 7 ladders
- enc-a4, least kept against the workspace: 0.599 + -0.0133 x, worst miss 0.053 GiB over 7 ladders
- enc2160-a4: every bracketed ladder kept its frames with 0.601 GiB (enc2160-1920-a4) or less, over 7 ladders
- enc2160-a4, least kept against the area: 0.586 + -0.0111 x, worst miss 0.056 GiB over 7 ladders
- enc2160-a4, least kept against the workspace: 0.599 + -0.0133 x, worst miss 0.053 GiB over 7 ladders

- **Under the cure no line holds; a step bounds the floors.** The 27 decodes kept their frames with
  0.68 to 0.97 GiB whatever the tile's area: a line, 0.853 − 0.014 GiB a megapixel, misses by up to
  0.15. Every decode under the cure kept its frames with 0.97 GiB or less, and every encode its
  latent with 0.60 or less.
- **On 6e1be76:** the tail's four decodes keep to a line (0.556 + 0.291 GiB a megapixel, missing by
  0.02; the first ladders' ten 1080p decodes gave 0.674 + 0.119, above). Its 19 encodes don't (0.47
  to 0.87 GiB, a line missing by 0.22). Every encode measured on 6e1be76 kept its latent with
  0.87 GiB or less.

#### Every rung

Each ladder's rungs in their order: the reference's peak (GiB), then each rung's room (usual) and
outcome, the frames (an encode's latent frames) that differ, and the plans built over the
reference's.

| ladder | rungs |
|---|---|
| dec1080-1216-a4 | ref 23.763; 1.10 kept (+0); 0.90 kept (+0); 0.70 moved 40/53 (+40); 0.80 kept (+0); 0.80 kept (+0); 0.70 moved 40/53 (+40) |
| dec2160-1152-a4 | ref 23.507; 0.85 kept (+0); 0.65 moved 12/17 (+40); 0.75 kept (+0); 0.75 kept (+0); 0.65 moved 12/17 (+40) |
| dec1080-1280-a4 | ref 24.679; 0.83 kept (+0); 0.63 moved 48/53 (+38); 0.73 moved 48/53 (+19); 0.83 kept (+0); 0.73 moved 48/53 (+19) |
| dec1080-1344-a4 | ref 25.701; 0.84 moved 32/53 (+20); 1.04 kept (+0); 0.94 kept (+0); 0.94 kept (+0); 0.84 moved 32/53 (+20) |
| dec2160-1216-a4 | ref 25.659; 0.88 kept (+0); 0.68 kept (+0); 0.48 moved 12/17 (+38); 0.58 moved 12/17 (+39); 0.68 kept (+0); 0.58 moved 12/17 (+39) |
| dec1080-1408-a4 | ref 26.731; 0.85 kept (+0); 0.65 moved 36/53 (+20); 0.75 moved 16/53 (+20); 0.85 kept (+0); 0.75 moved 16/53 (+20) |
| dec1080-1472-a4 | ref 27.787; 0.86 kept (+0); 0.66 moved 44/53 (+20); 0.76 moved 8/53 (+20); 0.86 kept (+0); 0.76 moved 8/53 (+20) |
| dec2160-1280-a4 | ref 27.917; 0.87 kept (+0); 0.67 moved 12/17 (+40); 0.77 moved 12/17 (+40); 0.87 kept (+0); 0.77 moved 12/17 (+40) |
| dec1080-1536-a4 | ref 28.820; 0.87 kept (+0); 0.67 moved 44/53 (+20); 0.77 moved 40/53 (+20); 0.87 kept (+0); 0.77 moved 40/53 (+20) |
| dec1080-1600-a4 | ref 29.781; 0.88 kept (+0); 0.68 kept (+0); 0.48 moved 48/53 (+38); 0.58 moved 48/53 (+39); 0.68 kept (+0); 0.58 moved 48/53 (+39) |
| dec1080-1664-a4 | ref 30.793; 0.86 kept (+0); 0.66 moved 48/53 (+40); 0.76 moved 40/53 (+20); 0.86 kept (+0); 0.76 moved 40/53 (+20) |
| dec2160-1344-a4 | ref 30.359; 0.87 kept (+0); 0.67 moved 8/17 (+40); 0.77 kept (+0); 0.77 kept (+0); 0.67 moved 8/17 (+40) |
| dec1080-1728-a4 | ref 31.776; 0.86 moved 36/53 (+20); 1.06 kept (+0); 0.96 kept (+0); 0.96 kept (+0); 0.86 moved 36/53 (+20) |
| dec1080-1792-a4 | ref 32.802; 0.87 moved 28/53 (+20); 1.07 kept (+0); 0.97 kept (+0); 0.97 kept (+0); 0.87 moved 28/53 (+20) |
| dec2160-1408-a4 | ref 32.835; 0.89 kept (+0); 0.69 moved 12/17 (+39); 0.79 kept (+0); 0.79 kept (+0); 0.69 moved 12/17 (+39) |
| dec1080-1856-a4 | ref 33.901; 0.88 kept (+0); 0.68 moved 40/53 (+20); 0.78 kept (+0); 0.78 kept (+0); 0.68 moved 40/53 (+20) |
| dec2160-1472-a4 | ref 35.496; 0.88 kept (+0); 0.68 moved 12/17 (+20); 0.78 moved 12/17 (+20); 0.88 kept (+0); 0.78 moved 12/17 (+20) |
| dec2160-1536-a4 | ref 38.265; 0.88 kept (+0); 0.68 kept (+0); 0.48 moved 12/17 (+38); 0.58 moved 12/17 (+40); 0.68 kept (+0); 0.58 moved 12/17 (+40) |
| dec2160-1600-a4 | ref 41.170; 0.87 moved 12/17 (+20); 1.07 kept (+0); 0.97 kept (+0); 0.97 kept (+0); 0.87 moved 12/17 (+20) |
| enc2160-1664-a4 | ref 26.398; 1.10 kept (+0); 0.90 kept (+0); 0.70 kept (+0); 0.30 moved 3/5 (+40); 0.50 kept (+0); 0.40 moved 3/5 (+40); 0.50 kept (+0); 0.40 moved 3/5 (+40) |
| enc2160-1728-a4 | ref 28.242; 0.55 kept (+0); 0.35 moved 3/5 (+40); 0.45 moved 3/5 (+40); 0.55 kept (+0); 0.45 moved 3/5 (+40) |
| enc2160-1792-a4 | ref 29.957; 0.58 kept (+0); 0.38 moved 3/5 (+40); 0.48 moved 3/5 (+40); 0.58 kept (+0); 0.48 moved 3/5 (+40) |
| enc2160-1856-a4 | ref 31.979; 0.59 kept (+0); 0.39 moved 3/5 (+20); 0.49 moved 3/5 (+20); 0.59 kept (+0); 0.49 moved 3/5 (+20) |
| enc2160-1920-a4 | ref 33.812; 0.60 kept (+0); 0.40 moved 3/5 (+20); 0.50 moved 3/5 (+20); 0.60 kept (+0); 0.50 moved 3/5 (+20) |
| enc2160-1984-a4 | ref 35.804; 0.61 kept (+0); 0.41 moved 2/5 (+40); 0.51 kept (+0); 0.51 kept (+0); 0.41 moved 2/5 (+40) |
| enc2160-2048-a4 | ref 37.848; 0.60 kept (+0); 0.40 moved 2/5 (+20); 0.50 kept (+0); 0.50 kept (+0); 0.40 moved 2/5 (+20) |
| dec1080-untiled-a4 | ref 34.952; 0.88 kept (+0); 0.68 moved 48/53 (+39); 0.78 moved 36/53 (+20); 0.88 kept (+0); 0.78 moved 36/53 (+20) |
| enc1080-untiled | ref 20.871; 0.66 kept (+0); 0.46 refused (+20); 0.56 refused (+20); 0.66 kept (+0); 0.56 refused (+20) |
| dec1080-768 | ref 12.624; 0.92 kept (+0) (history); 0.72 kept (+0) (history); 0.52 moved 15/53 (+20) (history); 0.62 moved 7/53 (+20) (history); 0.72 kept (+0); 0.62 moved 7/53 (+20) |
| dec1080-896 | ref 15.621; 0.91 kept (+0); 0.71 moved 44/53 (+20); 0.81 kept (+0); 0.81 kept (+0); 0.71 moved 44/53 (+20) |
| dec1080-960 | ref 17.196; 0.91 kept (+0); 0.71 moved 48/53 (+20); 0.81 kept (+0); 0.81 kept (+0); 0.71 moved 48/53 (+20) |
| dec1080-1088 | ref 21.169; 0.90 kept (+0); 0.70 refused (-214); 0.80 refused (-214); 0.90 kept (+0); 0.80 refused (-214) |
| enc1080-768 | ref 8.604; 0.67 kept (+0); 0.47 moved 12/14 (+20); 0.57 moved 5/14 (+20); 0.67 kept (+0); 0.57 moved 5/14 (+20) |
| enc1080-896 | ref 10.081; 0.68 kept (+0); 0.48 moved 5/14 (+20); 0.58 kept (+0); 0.58 kept (+0); 0.48 moved 5/14 (+20) |
| enc1080-960 | ref 11.338; 0.67 kept (+0); 0.47 kept (+0); 0.27 moved 12/14 (+20); 0.37 moved 12/14 (+20); 0.47 kept (+0); 0.37 moved 12/14 (+20) |
| enc1080-1088 | ref 13.328; 0.65 kept (+0); 0.45 moved 11/14 (+20); 0.55 kept (+0); 0.55 kept (+0); 0.45 moved 11/14 (+20) |
| enc1080-1152 | ref 14.206; 0.65 moved 9/14 (+20); 0.85 kept (+0); 0.75 kept (+0); 0.75 kept (+0); 0.65 moved 9/14 (+20) |
| enc1080-1280 | ref 15.068; 0.66 kept (+0); 0.46 moved 11/14 (+20); 0.56 kept (+0); 0.56 kept (+0); 0.46 moved 11/14 (+20) |
| enc1080-1408 | ref 16.422; 0.66 kept (+0); 0.46 moved 11/14 (+20); 0.56 moved 11/14 (+20); 0.66 kept (+0); 0.56 moved 11/14 (+20) |
| enc1080-1472 | ref 17.099; 0.66 kept (+0); 0.46 moved 11/14 (+20); 0.56 moved 11/14 (+20); 0.66 kept (+0); 0.56 moved 11/14 (+20) |
| enc1080-1600 | ref 17.944; 0.67 moved 11/14 (+20); 0.87 kept (+0); 0.77 moved 5/14 (+20); 0.87 kept (+0); 0.77 moved 5/14 (+20) |
| enc1080-1728 | ref 19.257; 0.68 kept (+0); 0.48 moved 11/14 (+20); 0.58 moved 11/14 (+20); 0.68 kept (+0); 0.58 moved 11/14 (+20) |
| enc1080-1792 | ref 19.914; 0.69 kept (+0); 0.49 moved 11/14 (+20); 0.59 moved 8/14 (+20); 0.69 kept (+0); 0.59 moved 8/14 (+20) |
| enc2160-1152 | ref 14.757; 0.69 kept (+0); 0.49 moved 3/5 (+20); 0.59 kept (+0); 0.59 kept (+0); 0.49 moved 3/5 (+20) |
| enc2160-1216 | ref 15.673; 0.69 kept (+0); 0.49 moved 3/5 (+20); 0.59 kept (+0); 0.59 kept (+0); 0.49 moved 3/5 (+20) |
| enc2160-1280 | ref 17.194; 0.68 kept (+0); 0.48 moved 3/5 (+20); 0.58 kept (+0); 0.58 kept (+0); 0.48 moved 3/5 (+20) |
| enc2160-1344 | ref 18.312; 0.68 kept (+0); 0.48 moved 3/5 (+20); 0.58 moved 3/5 (+20); 0.68 kept (+0); 0.58 moved 3/5 (+20) |
| enc2160-1408 | ref 19.978; 0.68 kept (+0); 0.48 moved 3/5 (+20); 0.58 moved 3/5 (+20); 0.68 kept (+0); 0.58 moved 3/5 (+20) |
| enc2160-1536 | ref 23.018; 0.69 kept (+0); 0.49 refused (-128); 0.59 kept (+0); 0.59 kept (+0); 0.49 refused (-128) |
| enc2160-1600 | ref 24.731; 0.68 kept (+0); 0.48 refused (-144); 0.58 refused (-144); 0.68 kept (+0); 0.58 refused (-144) |
| dec2160-1664-a4 | ref 43.976; 0.88 kept (+0); 0.68 moved 12/17 (+40); 0.78 kept (+0); 0.78 kept (+0); 0.68 moved 12/17 (+40) |
| dec2160-1728-a4 | ref 47.202; 0.88 kept (+0); 0.68 moved 12/17 (+38); 0.78 moved 12/17 (+38); 0.88 kept (+0); 0.78 moved 12/17 (+38) |
| dec2160-1792-a4 | ref 50.449; 0.88 kept (+0); 0.68 moved 12/17 (+38); 0.78 kept (+0); 0.78 kept (+0); 0.68 moved 12/17 (+38) |
| dec2160-1856-a4 | ref 54.000; 0.88 kept (+0); 0.68 moved 12/17 (+20); 0.78 kept (+0); 0.78 kept (+0); 0.68 moved 12/17 (+20) |
| dec2160-1920-a4 | ref 57.208; 0.88 kept (+0); 0.68 kept (+0); 0.48 moved 12/17 (+38); 0.58 moved 12/17 (+39); 0.68 kept (+0); 0.58 moved 12/17 (+39) |
| dec2160-1984-a4 | ref 60.770; 0.87 kept (+0); 0.67 moved 12/17 (+19); 0.77 kept (+0); 0.77 kept (+0); 0.67 moved 12/17 (+19) |
| dec2160-2048-a4 | ref 64.863; 0.87 kept (+0); 0.67 moved 12/17 (+19); 0.77 moved 12/17 (+19); 0.87 kept (+0); 0.77 moved 12/17 (+19) |

### Whole runs in large tiles under the one-output cure (2026-10-17)

The floors by tile size above are a pass alone's: a process of its own that enters its decode with
the models alone, the allocator's pool holding 1.53 GiB. A whole run enters it after its encode and
its windows (S1's, [the floor is the frames](#the-floor-is-the-frames), ran on 2026-10-12's code in
the plans' own shapes). Here, whole runs of six decode tiles on the cure's code, each beside the
floor of the same decode alone
([above](#the-large-tiles-under-the-one-output-cure-and-the-tiles-left-2026-10-17)):
1080p's tiles of 1,104 × 1,472 and 1,104 × 1,792 and its untiled decode, 4K's tiles of 1,600 and
2,048 px, all taken by the cure, and 1080p's 832-px tile, which the cure doesn't take. The rungs
ran under the run's own driver, one hold of the GPU's lock a rung: 34 runs in 6 ladders, none
failed. [`scripts/whole_floor.py`](../scripts/whole_floor.py) is that driver as committed, its places
made arguments and its two probes folded in: it reads the run's records as they are and gives from
them, byte for byte, the tables this section was taken from; and on the card it ran a reference of
1,104 × 1,472's whole job and one rung with 1.20 GiB left, which kept its frames: both gave the
run's reference's 53 frames, CRC-32 for CRC-32. Not run: whole runs of the other tiles, of encodes
at their floor (the encode tiles stayed the shapes' small ones), and on 6e1be76.

#### Method as run

- **A rung is a whole job:** seedvr2x's command line by `vae_room.py run` with neither its stop nor
  a seed, the encode, the DiT's windows and the decode in one process, under the ballast. The job is
  the pass alone's own shape (clip-53 at 1080p, window 8, encode tiles of 768 px; clip-17 at 4K,
  window 5, encode tiles of 1,408 px; 36 blocks swapped; one shot), the decode tile pinned, the
  planner's room at −50 GiB and the DiT's phase unjudged as the passes alone ran. So every rung ran
  its reference's plan, checked on each: the DiT's two slots and its other parameters beside both
  VAE passes, the VAE kept in the DiT's phase. The code: the cure's, the frozen copy the tail's
  ladders named `-a4` ran, its tree checked before every rung, nothing written in it.
- **Two probes more,** in the rung's process: the md5 of the shot's latent once its encode kept it,
  and of each window the DiT gave once kept, read back on the CPU from seedvr2x's files (a finished
  job removes them). Every rung's latent and windows were its reference's, and so were its encode's
  and its windows' cuDNN plans built: the frames that moved were the decode's.
- **The reference:** the same whole job on the bare card. On the cure's code each gave the pass
  alone's reference's frames, CRC-32 for CRC-32 (5 tiles), its decode peak 0.004 to 0.005 GiB over
  the pass alone's (what the GPU held entering the decode, 1.526 GiB against 1.521; the decode's own
  peak above it the same to the MiB). The 832-px tile's pass alone is the first ladders' on
  6e1be76, whose noise is drawn on the GPU: its frames differ (53 of 53), as they should; its peak
  doesn't (13.824 GiB, 13.819 alone).
- **The room:** "usual", the free memory the rung's plan was made in less the whole run reference's
  decode peak (torch's max allocated in the decode unit); beside it the same free memory less the
  pass alone's reference peak, 0.004 to 0.005 GiB more, so that both floors stand on one measure.
- **The ladder:** from the pass alone's least room kept (the table's fourth column), up by 0.1 GiB
  while the frames moved, down while they held, then the least room kept
  and the most lost each run a second time on the same card; a card lost on any run counts as lost.
- **The run:** from 19:30 to 21:52 UTC, two queues, the second (the last four tiles) waiting for
  the first's end. No other compute process was on the card at any rung (nvidia-smi before each).
  The card's offset was 1.489 to 1.491 GiB. A whole rung took 192 to 279 s at 1080p and 196 to
  379 s at 4K, for information.

#### The floors

Room usual, GiB, with each deciding card's runs; the pass alone's from the ladders above (their
deciding cards twice each, both runs alike):

| tile | whole run: least room kept | most room lost | pass alone: least kept | most lost | least kept, whole − alone | moved: frames, worst PSNR-Y | decode plans over the reference's: lost / kept |
|---|---|---|---|---|---|---|---|
| 1,104 × 1,472 | 0.960 (kept, kept) | 0.860 (moved, moved) | 0.861 | 0.759 | +0.10 | 32 and 44 of 53; 72.39, 73.16 dB | +40, +20 / +0 |
| 4K 1,600 | 0.772 (kept, kept) | 0.673 (refused, moved) | 0.970 | 0.870 | −0.20 | 12 of 17; 72.57 dB | +40 / +0 |
| 1,104 × 1,792 | 1.072 (kept, kept) | 0.972 (kept, moved) | 0.971 | 0.869 | +0.10 | 16 and 32 of 53; 72.50, 71.26 dB | +20, +40 / +0 |
| 1080p untiled | 1.078 (kept, kept) | 0.979 (moved, kept) | 0.879 | 0.780 | +0.20 | 28 and 48 of 53; 72.30, 70.97 dB | +20, +38 / +0 |
| 832 (no cure; alone on 6e1be76) | 0.890 (kept, kept) | 0.791 (moved, kept) | 0.791 | 0.690 | +0.10 | 48 of 53; 72.57 dB | +20 / +0 |
| 4K 2,048 | 0.870 (kept, kept) | 0.770 (moved, kept) | 0.870 | 0.769 | 0.00 | 12 of 17; 74.15 dB | +19 / +0 |

- **In these six tiles a whole run's floor is the pass alone's within 0.2 GiB, either way.**
  1,104 × 1,472, 1,104 × 1,792 and the 832-px tile need 0.1 GiB more, 1080p's untiled decode 0.2,
  4K's 2,048 px none, and 4K's 1,600 px 0.2 less: the whole run kept its frames with 0.872 and
  0.772 where the pass alone moved them with 0.870 (12 of 17, twice). Every whole run kept its
  frames with 1.08 GiB or less (1080p's untiled decode) and lost them, on some run, with 0.98 or
  less.
- **The 832-px tile:** S1's whole runs on 2026-10-12's code kept their frames with 1.08 and lost
  them with 0.89 ([the floor is the frames](#the-floor-is-the-frames)), in the 16 GB plan's shape
  (85 frames, window 12, encode tiles of 1,152 px), not this one. In the pass alone's shape on the
  cure's code the whole run kept them with 0.89 twice and lost them in one run of two with 0.79.

#### Two runs at one card

The deciding cards ran twice each, 12 cards. Five gave two outcomes:

| tile | room | first run | second run |
|---|---|---|---|
| 4K 1,600 | 0.673 | refused (the decode's fifth slice) | moved 12 of 17, 72.57 dB, +40 plans |
| 1,104 × 1,792 | 0.972 | kept | moved 16 of 53, 72.50 dB, +20 |
| 1080p untiled | 0.979 | moved 28 of 53, 72.30 dB, +20 | kept |
| 832 | 0.791 | moved 48 of 53, 72.57 dB, +20 | kept |
| 4K 2,048 | 0.770 | moved 12 of 17, 74.15 dB, +19 | kept |

And at 1,104 × 1,472 with 0.860 both runs moved, other frames each time: 32 of 53 from the 22nd
(+40 plans: two convolutions fell back), then 44 from the 10th (+20). The tail's passes alone ran
114 cards twice, each pair alike, frames included; whole runs don't repeat near their floor. S1 saw
it too: a packed whole run doesn't give the same frames twice
([the floor is the frames](#the-floor-is-the-frames)).

#### The pool at the decode's start

A whole run enters its decode with the pool still holding what its encode and windows reserved:
11.8 to 12.2 GiB at 1080p and 26.6 to 27.5 at 4K, of which 1.53 is allocated (the models' parts on
the GPU); a pass alone enters it with 1.53 reserved. At one card the plan's free memory is the same
in every run, split between the pool and the driver's free memory. That split varies from run to
run, by up to 0.72 GiB at one card (4K's 1,600 px at 0.772, kept both times), so the reserve's size
alone doesn't decide; but in each of the four pairs that split kept and lost, the run that lost
entered its decode with the more in the pool, by 0.03 to 0.41 GiB, and as much less free at the
driver. Read, not shown: where what the pool kept of the earlier phases lies decides where the
decode's large requests find room, and so whether a convolution's workspace is refused; the room as
measured (the free memory less the peak) doesn't see it.

#### What the run can see

- **The cuDNN frontend's plans built tell it in whole runs too:** 19 to 40 more than the
  reference's in the decode of every run whose frames moved (9 runs), none in any of the 18 that
  kept them, the encode's and the windows' the reference's in every rung.
- **torch's count of refused allocations** stayed at 0 in every unit of every run
  (`cudaMallocAsync`), and the "recovered" lines don't separate (decodes kept 33 to 210, moved 33
  to 230).
- **The refusal** (4K's 1,600 px, 0.673): in the decode's fifth slice (a tile's first, of five
  frames: 4K's clip runs four slices a tile), at the concatenation the cure leaves to a convolution
  whose four-frame output is under its 2.5 GiB (`memory_limit_conv:225` on that code): 3.05 GiB
  asked, over the floor that four-frame output is judged by, with 37.94 GiB allocated and 1.19 MiB
  free on the card.
- **The plans' predicted decode peaks** stand 0.00 (832 px) to 5.17 GiB (4K's 2,048 px) over the
  measured ones, the cure's tiles still predicted by the table measured without it: a plan leaves
  those passes that much more than its room.

#### Every whole-run rung

Each ladder's rungs in their order: the reference's decode peak (GiB), then each rung's room
(usual) and outcome, the frames that differ, and the decode's plans built over the reference's.

| ladder | rungs |
|---|---|
| whole1080-1472 | ref 27.791; 0.86 moved 32/53 (+40); 0.96 kept (+0); 0.96 kept (+0); 0.86 moved 44/53 (+20) |
| whole2160-1600 | ref 41.174; 0.97 kept (+0); 0.87 kept (+0); 0.77 kept (+0); 0.67 refused (slice 5); 0.77 kept (+0); 0.67 moved 12/17 (+40) |
| whole1080-1792 | ref 32.806; 0.97 kept (+0); 0.87 moved 32/53 (+40); 0.97 moved 16/53 (+20); 1.07 kept (+0); 1.07 kept (+0) |
| whole1080-untiled | ref 34.956; 0.88 moved 48/53 (+38); 0.98 moved 28/53 (+20); 1.08 kept (+0); 1.08 kept (+0); 0.98 kept (+0) |
| whole1080-832 | ref 13.824; 0.79 moved 48/53 (+20); 0.89 kept (+0); 0.89 kept (+0); 0.79 kept (+0) |
| whole2160-2048 | ref 64.868; 0.87 kept (+0); 0.77 moved 12/17 (+19); 0.87 kept (+0); 0.77 kept (+0) |

### What follows for seedvr2x

- **The room can't be 0.** At 0.6 GiB left a pass's convolutions lose their workspaces: the run
  ends, slower, with frames that are neither the reference's nor the same from run to run. The
  1.5 GiB of today leave 2.2 GiB or more to every VAE pass of the matrix.
- **Its floor is to measure by the frames,** tile size by tile size: a ladder of cards under a
  pinned shape, each run's checksums against the shape's reference. Three points here: between
  0.80 and 0.90 GiB left for 640-px tiles, between 0.89 and 1.08 for 832-px tiles, at 0.94 or
  under for tiles of 1104 × 1152.
- **The cure for large outputs** removes the failures of the large passes and lowers their
  peaks; the tiles' table, measured before it, then stands over the truth for them. Built into
  seedvr2x, the planner's nine-case matrix ran at the room as it is, the small passes on their
  table values to the hundredth, the large ones 0.29 to 5.35 GiB lower than before it (a 1080p
  decode in 1472-px tiles by 0.52, a 4K one in 2048-px tiles by 5.35).
- **The 0.4 GiB of an encode** is part of what keeps an encode's frames today: to be judged with
  the floor, not before.
- **The floor by tile size** (2026-10-16 and 17, passes alone, on this card and allocator): see
  [its section](#the-floor-by-tile-size-and-its-cause-2026-10-16) and
  [the large tiles under the cure](#the-large-tiles-under-the-one-output-cure-and-the-tiles-left-2026-10-17).
  Every pass alone measured on this card kept its frames with 1.10 GiB or less beyond its peak.
  With the cure, every 1080p and 4K decode it takes that ran did with 0.97 or less and moved them
  below, refused nowhere, and its 4K encode squares did with 0.60; 4K's untiled encode and the
  other output sizes' passes it takes didn't run. Without it, the decodes it doesn't take that ran
  (squares of 640 to 1,088 px, 1,104 × 1,152) did with 0.90 or less (from 1,024 px refused first;
  the 512- and 576-px decodes didn't run), the encodes run with 0.87. A VAE pass is left at least
  the room and the budget's 0.6 GiB beyond its predicted peak: 2.1 GiB with the room at 1.5.
- **Whole runs** ([six tiles on the cure's code](#whole-runs-in-large-tiles-under-the-one-output-cure-2026-10-17)) kept their
  frames with 1.08 GiB or less beyond their decode's peak, within 0.2 of the passes alone either
  way, and lost them, on some run, with 0.98 or less; near that floor two runs at one card don't
  always agree. The plans predicted those decodes' peaks 0.00 to 5.17 GiB over what they measured.
- **A run can't see a fallback for free on `cudaMallocAsync`:** torch counts no refusal there;
  the cuDNN frontend's log does. Under torch's own allocator, a held run's, `num_ooms` across a
  unit counts each refused workspace: 0 across every pass that kept its frames, 19 to 21 across
  every one that lost them ([on an RTX 5090](#under-expandable-segments-on-an-rtx-5090-2026-10-17)),
  a check a held run could make at no cost. The pass's time is untried as a signal: on the
  power-capped card it drifts from run to run, and on the RTX 5090 it is deferred.
- **On a held run** (a driver that gives memory the card doesn't have, expandable segments) the
  two tiles measured keep their frames with 0.40 GiB beyond their peak, a fifteenth of the 6.0 GiB
  a plan leaves a pass there.

## Caveats

- **Emulated cards,** as in [planner-limits.md](planner-limits.md#caveats): a ballast on a 96 GB
  card gives the memory a smaller card leaves, not its context, its driver or its kernels.
- **One GPU, one driver, one software stack.** The pool is the driver's
  (`cudaMallocAsync`): another driver's may hold on to less, or to more.
- **Few repeats.** A plan ran one to three times. Where a pass fails it doesn't always fail at
  the same slice, and one that failed in the whole run ran alone: a run that passes once proves
  little, a plan that passes three times a little more.
- **One source,** an animated film's shot, one seed. The VAE's memory doesn't depend on the
  picture; the allocator's pool is fed the same sizes whatever the content.
- **A driver that gives memory the card doesn't have** (seedvr2x holds itself to the card there,
  under torch's own allocator) was measured for two decode tiles only, 832 and 1,024 px at
  1080p, on one RTX 5090 beside one desktop, a room run once or twice, by torch's limit rather
  than a smaller card's free memory; the encodes, the other tiles and torch's own allocator
  without the segments, not.
- **Passes alone** (the floor by tile size): a whole run with the same room left lost frames a
  pass alone kept (640-px tiles: 48 of 53 moved with 0.70 GiB left in a whole run, none alone),
  and a pass alone in a new process starts with no plan cached. Under expandable segments on the
  RTX 5090 the whole runs kept and lost their frames where the passes alone did (832 and 1,024
  px, at 0.1 GiB). On the 96 GB card, on the cure's code, the whole runs of six tiles kept their
  frames within 0.2 GiB of the passes alone, either way ([whole runs](#whole-runs-in-large-tiles-under-the-one-output-cure-2026-10-17));
  other tiles' whole runs, encodes at their floor in a whole run, and whole runs on 6e1be76, not
  run.
- **The large tiles' frames' floor was measured under the cure** for 1080p's and 4K's tiles and
  1080p's untiled decode, passes alone: on the code without it, from 1,024 px a decode is refused
  first, at the concatenation the cure removes, and those ladders are that code's record.
- **Two codes, two sets of units:** under the cure the units were made on its own code, whose
  noise is drawn on the CPU, so its frames aren't the other code's. Each tile is judged against its
  own code's reference; between codes only the rooms and the peaks compare, and the free memory a
  pass asks (its peak and its room) is the fair figure.
- **A rung ran once or twice,** the deciding ones twice, the 768-px decode's in the tail. The 38
  rungs of the first ladders that ran twice agreed each time, in outcome and frames (the moved ones
  with the same other frames), and so did the 114 cards the tail ran twice (the same outcome, the
  same other frames or latent, a refusal in the same slice). One rung of the tail was
  stopped when the queue was interrupted, recorded by vae_room.py and not by the driver, and ran
  again as that card's second run. The whole runs' twelve cards run twice didn't all agree: five
  gave two outcomes and a sixth moved other frames each time.
- **Not run:** under the cure, 4K's untiled encode and the other output sizes' passes it takes
  (1440p's cut-off tiles and untiled passes among them); on 6e1be76, the decodes in 512- and
  576-px tiles; and, apart, 4K's encode squares of 768 to 1,088 px, 1080p's own shapes. Whole runs:
  only the six tiles measured, on the cure's code.

## Reproduce

```bash
# seedvr2x's venv python; vram_cap.py and planner_matrix.py beside vae_room.py; one GPU job at a
# time (queue takes the lock for each run). CLIP-N.mkv: the matrix's clips, N frames of a 1080p
# source (planner_matrix.py makes them).

# what the planner makes of the matrix's cases under a room, no GPU
python vae_room.py plan --room 0 --case 1080:24:2w --case 1080:32:w1 --case 2160:48:w1

# one run: a card emulated at 32 GB, the room at 0, cure output
python vae_room.py run --name NAME --records RECORDS --work WORK --vram-cap vram_cap.py \
    --card 32 --room 0 --cure output -- CLIP-113.mkv --resolution 1080 --cut-threshold 1 \
    --model-dir MODELS

# a queue of runs, one hold of the lock each; then the tables
python vae_room.py queue QUEUE.txt --records RECORDS --lock LOCK --copy SEEDVR2X --want TREE
python vae_room.py table RECORDS

# the frames' floor by tile size: a shape's units once, then the ladders (one lock hold a rung),
# then the tables
python floor.py --dir DIR --python PYTHON --copy SEEDVR2X --want TREE --models MODELS \
    --clips CLIPS --lock LOCK seed 1080
python floor.py --dir DIR --python PYTHON --copy SEEDVR2X --want TREE --models MODELS \
    --clips CLIPS --lock LOCK queue
python floor.py --dir DIR table
python floor.py --dir DIR detector

# the tail: the tiles the one-output cure takes on a second code (its ladders named -a4, their
# units made on it), the base code's tiles left, the untiled passes; the first study read only
python floor.py --dir DIR2 --python PYTHON --copy SEEDVR2X --want TREE \
    --code a4 PYTHON_A4 SEEDVR2X_A4 TREE_A4 --history DIR --models MODELS --clips CLIPS \
    --lock LOCK queue --order tail
python floor.py --dir DIR2 --history DIR table --order tail --compact
python floor.py --dir DIR2 --history DIR detector

# whole runs beside the passes alone: the encode, the windows and the decode in one process, the
# decode tile pinned (one lock hold a rung); each ladder's pass alone read from floor.py's studies,
# read only; then the tables
LADDERS=whole1080-1472=dec1080-1472-a4,whole2160-1600=dec2160-1600-a4,whole1080-1792=dec1080-1792-a4
LADDERS=$LADDERS,whole1080-untiled=dec1080-untiled-a4,whole1080-832=dec1080-832
LADDERS=$LADDERS,whole2160-2048=dec2160-2048-a4
python whole_floor.py --dir DIR3 --python PYTHON_A4 --copy SEEDVR2X_A4 --want TREE_A4 \
    --models MODELS --clips CLIPS --lock LOCK --alone DIR2 --alone DIR --ladders $LADDERS queue
python whole_floor.py --dir DIR3 --alone DIR2 --alone DIR --ladders $LADDERS table
python whole_floor.py --dir DIR3 --alone DIR2 --alone DIR --ladders $LADDERS table --compact
# the tables from rungs' files written elsewhere (several, or a glob; the night's as they are)
python whole_floor.py --dir DIR4 --rungs 'DIR3/*rungs.jsonl' --alone DIR2 --alone DIR \
    --ladders $LADDERS table

# where the driver gives memory the card doesn't have (WSL2): the floor by torch's limit, under
# torch's own allocator with expandable segments, vae_room.py beside held_floor.py; the shape's
# units once, the ladders and whole runs (one lock hold a run), then the tables
python held_floor.py --dir DIR --python PYTHON --clip CLIP-53.mkv --lock LOCK --tree TREE seed
python held_floor.py --dir DIR --python PYTHON --clip CLIP-53.mkv --lock LOCK --tree TREE \
    ladder 832
python held_floor.py --dir DIR --python PYTHON --clip CLIP-53.mkv --lock LOCK --tree TREE \
    whole 832 0.4 0.3
python held_floor.py --dir DIR table

# what cudaMallocAsync lets a run see of a refused request; one convolution short of memory
python alloc_probe.py
CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=stderr python conv_probe.py
```
