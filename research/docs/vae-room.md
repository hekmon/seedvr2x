# The VAE's room: what a VAE pass needs between its peak and the free memory, and what cures it

> Status: **measured** (2026-10-13) for the open question "The VAE's room" of
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure). seedvr2x's planner holds every VAE pass 1.5 GiB
> further under its budget than a DiT window ([DESIGN.md](../../seedvr2x/DESIGN.md#memory-planner),
> Budget), because passes packed to the 0.6 GiB margin ran out of memory on 2026-10-11, at one
> large request, the card full while the allocator's pool held more than was asked. This is the
> study of that room: seedvr2x's own command line on cards emulated by
> [`scripts/vram_cap.py`](../scripts/vram_cap.py), the planner's room set in the research driver,
> never in the product, by [`scripts/vae_room.py`](../scripts/vae_room.py). One GPU (RTX PRO 6000
> Blackwell 96 GB, torch 2.14.1+cu130, cuDNN 9.24, flash-attn 2.8.3, `cudaMallocAsync`), 7B fp16,
> the colour correction on (`split`), seedvr2x as it stood before the changes this study leads to.

In short:

- **The frames move with the memory's pressure.** A VAE pass left too little memory beyond its
  peak still runs, slower, with other frames than the same pass with room to spare, and not the
  same ones twice. Shown on one convolution: left too little memory, it runs 8.5 times slower
  and gives other bits, without a word, and gave the same other bits again with the memory free,
  in the same process (three calls); another cuDNN engine taken is the reading, inferred from
  the bits, the time and the memory, no log naming an engine
  ([the convolution alone](#one-convolution-alone)). A decode in 640-px tiles keeps its frames
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
  under torch's own allocator) was not measured here.

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
```
