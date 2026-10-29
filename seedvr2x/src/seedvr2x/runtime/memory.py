"""The GPU memory seedvr2x can really use: whether the driver refuses what the card can't hold
(DESIGN.md, Memory planner, Allocator).

A CUDA driver on Linux refuses an allocation the card has no room for, and everything above it
counts on that: torch's allocators give their slack back when a request fails, and a plan that
doesn't fit ends in an OutOfMemoryError. Windows' driver gives system memory in the card's place
instead ("system memory fallback", driver 536.40 and later; its setting, "CUDA - Sysmem
Fallback Policy", 546.01 and later, is documented per Windows program: NVIDIA's article 5490),
and under WSL2, whose GPU is Windows', it does so whatever that setting says (asked with the
setting at "Prefer No Sysmem Fallback" and at "Prefer Sysmem Fallback": given both times). A
step that passes the card there runs 2 to 6 times slower, its bits still right in every run
that ended; one that passes it further kills the process, CUDA's "sticky" out-of-memory error
raised where no Python code can catch it; and the work after an over-committed step isn't to
be trusted: once it gave other bits, once tensors of wrong sizes and an "unknown error" (RTX
5090, driver 616.92, 2026-10-11; research/scripts/spill_probe.py, hold_probe.py).

So seedvr2x asks the driver, once a run, where the run first takes the GPU and before torch is
imported (startup): probe requests a little more memory than is free. Refused, the driver is one
that refuses, and nothing changes. Given, it is one that spills, as is, unasked, one that
reports more free to the process than the whole card has; and seedvr2x holds itself to the
card's memory by its own means:
- torch's own allocator in cudaMallocAsync's place (startup), its segments expandable
  (EXPANDABLE): it counts every byte it has from the driver, its cache included, where
  cudaMallocAsync's pool keeps memory torch's limit doesn't see (2.0 to 2.6 GiB of it at 11
  latents at 1080p); and with expandable segments a step short of room under that limit
  slows down where torch's allocator otherwise refuses it;
- that allocator limited (hold) to the memory free less HELD_BACK: what passes it is refused
  by torch, an OutOfMemoryError that Python catches, in under a second, and the work after it
  is right;
- the planner planning in what is left less RESERVE (usable), which the plan leaves that
  allocator beyond what a step allocates: the plan's budget, the planner's own margin counted in
  it; less RESERVE_TIGHT only where a plan so made would be refused (tightened);
- the memory free read again before every unit (check): when what torch may still take no
  longer fits in it, other programs holding the rest of the card, the run stops; under
  expandable segments torch is held again to what the card has, within the reserve the plan
  took, and the run goes on.

The memory free is, for the limit and the plan, the smaller of two readings: what the driver
reports free to the process (CUDA's figure, torch's mem_get_info) and what the whole card has
free, read through NVML (card_free), the driver's own library, nvidia-smi's, which answers
under WSL2; for the check, the whole card's (why: check). NVML is asked again wherever it fails
(_twice). The plan, the limit and the checks stand on one footing, the plan's (_footing; DESIGN.md,
Allocator, decided on 2026-10-13): where NVML doesn't answer the plan's reading, the driver's
figure alone for the three, under a larger margin (HELD_BACK_UNREAD), and a warning; where it
does, the card's for the three, a limit whose reading NVML fails being set from the card's figure
the plan was made in, and a check it fails at skipped with a warning, never judged on the
driver's figure (hold, check).

Why two: the free memory such a driver reports is the process's own while the card has room: a
process read 30.20 GiB free of 31.82 whether another held 8.6 GiB of the card or not, idle or at
work, and all day beside a desktop that nvidia-smi showed using 2.7 to 4.5 GiB; nvidia-smi's
figure is the whole card's. As the card fills the figure falls to what is really free: a run
holding 15.8 GiB beside that other program read 3.89 GiB (nvidia-smi: 3.91 free), and the other
program 10.3 GiB less than before. So a plan made in that figure alone beside another program
counted on memory the card doesn't have, and the check was what stopped it, at its first unit
(research/scripts/neighbour_probe.py). At rest on that card (2026-10-12), beside a desktop
holding 2.73 GiB of it and the driver's own 0.41: a process with a bare context was reported
30.20 GiB free where the whole card had 28.22, NVML's figure and nvidia-smi's; the 1.98 GiB
between the two stayed, to the byte, as the process took 1 then 5 GiB.

Measured on the RTX 5090 under WSL2 (research/scripts/hold_probe.py, 2026-10-11, 30.20 GiB
reported free to the process, the figure the limit was then taken under, the card's own not
read yet; the models whole on the card, windows at 1080p: 17.0 GiB allocated at 1 latent, 22.6
at 6, 24.7 at 8, 27.9 at 11):
- torch's own allocator held 3.0 GiB under the free memory: 1, 6 and 8 latents ran at their
  usual speed; 9, 10, 11 and 12 were refused in 0.2 to 0.6 s; 6 and 1 ran again after them,
  right and as fast; the card never had under 1.73 GiB free. A 1080p VAE decode past the card,
  the DiT's 36 blocks swapped: refused in 3.2 s, the card down to 1.59 GiB free, and the
  6-latent window after it right. Held 2.0 GiB under: the same, the card down to 0.16 GiB free.
- held 0.6 GiB under: the card went to 0 free; 6 latents took 7 to 9 s for 5, then 28 to 37
  after the refusals; and after the VAE decode's refusal the next window failed (wrong tensor
  sizes, then "CUDA error: unknown error").
- cudaMallocAsync with torch's limit 0.6 GiB under, or none: 11 latents ran in 16 to 55 s for
  8.5, and 12 killed the process, 4 runs of 4 ("Sticky error detected", SIGABRT). With the
  limit 4.0 GiB under, 11 and 12 were refused in 0.3 s, the pool keeping the card at 0 free.
- every step that ended gave the reference's bits, the slow ones too.

Measured again against the card's own figure (the same probe, 2026-10-12; a desktop holding 1.6
to 1.8 GiB of the card, 2.7 before the first process; torch's own allocator; 326 steps of
eleven processes, 37 of them refused. The card's figure was the smaller of the two at each of
the 26 readings a plan or a limit was taken under; the driver's was sampled under it during 34
of the steps. The 1080p windows' references were made with the DiT's 36 blocks swapped and
torch held 4 GiB under the card, in two processes that agreed, and with the day before's. Of
the 289 steps that ended, 240 gave the bits of a reference another process had made and 31
those of an earlier step of their own process; 18, each step's first run, were held to
nothing, the probe then writing a step's digest before it compared: 4K's windows of 1 to 4
latents ran once each, and 1080p's of 17 and of 21 latents in one process each):
- the models whole, torch held 1.0, 0.5 and 0.25 GiB under the card's free memory, each in two
  processes: windows of 1, 6 and 8 latents at 1080p ran at their usual speed with the
  reference's bits, 9 too under 0.5 and 0.25; 12 were refused in 0.2 to 0.4 s, and 6 and 1 ran
  right after; the card never had under 0.58 GiB free;
- in nine of the eleven processes the run held 0.016 GiB of the card outside torch's count at
  the most, 0.003 with the models whole; in the two others (1080p windows of 8 to 21 latents,
  36 blocks swapped, held 0.25 GiB under the card's figure), 24 of the 38 steps that ended
  read 0.019 to 0.102 GiB during the step and 30 read 0.019 to 0.058 after it, the desktop's
  own memory having grown 14 and 37 MiB under them by nvidia-smi, the desktop's or their own:
  the 2.1 to 2.8 GiB of the day before were the desktop's (HELD_BACK);
- torch's allocator needed 0.2 to 0.8 GiB beyond the allocated peak of a 1080p window of 1 to
  8 latents, 3.6 of 17 and 4.5 of 4K's 5; and 3.4 to 5.3 beyond a VAE pass's (RESERVE);
- unheld, that allocator keeps what it has used: with 36 blocks swapped it held 19.2 GiB for a
  window of 8 latents that allocates 10.4, after windows of 1 and 6 (14.1 with its cache
  emptied before the window, held 4 GiB under the card's figure); at 12 the card had 0.2 GiB
  left (nvidia-smi), and the desktop's own memory on it fell from 2.73 GiB to 1.57, and stood
  at 1.81 an hour and a quarter later.

Nothing here imports torch at import: the allocator is chosen before torch is (cli._run), whose
library fixes it when it loads.
"""

import ctypes
import logging
import os
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

GIB = 1 << 30

# WSL2's GPU device, through which a Linux process reaches Windows' driver: where it is, the
# driver is Windows', and may give system memory for the card's.
PARAVIRTUAL = "/dev/dxg"
# What the probe asks beyond the free memory: enough that the card can't hold the request by
# chance (the driver's own rounding), little enough that a spilling driver commits next to
# nothing of system memory for it.
OVER = 256 << 20
# The driver's two libraries, as the dynamic loader finds them: CUDA's, which the probe asks and
# which names the run's device, and NVML's, which reads the whole card (card_free).
CUDA, NVML = "libcuda.so.1", "libnvidia-ml.so.1"
# CUDA's result codes (cuda.h): CUDA_SUCCESS, CUDA_ERROR_OUT_OF_MEMORY. NVML's success is 0 too
# (nvml.h: NVML_SUCCESS).
SUCCESS, OUT_OF_MEMORY = 0, 2
# What seedvr2x leaves of the free memory on a driver that spills, where the whole card's is
# read (card_free): to what the run itself holds on the card outside torch's count, from the
# reading its limit is set on, and to the rest of the card, a desktop's own memory moving under
# the run. 0.5 GiB, design's figure (DESIGN.md, Allocator, decided on 2026-10-12): "a desktop
# that wakes at night mustn't stop a job nobody will start again before morning". It was 0.2,
# the measure's own, under which a job's checks ran with 0.16 to 0.20 GiB to spare, and 0.10
# once, its desktop having taken 90 MiB during the load (the same RTX 5090, 2026-10-12, the
# check judged on the card's figure).
# The measure, against the card's own figure (research/scripts/hold_probe.py, an RTX 5090 under
# WSL2, driver 616.92, 2026-10-12, a desktop holding 1.6 to 1.8 GiB of the card; 326 steps of
# eleven processes, 37 of them refused, the card's figure the smaller of the two at each of the
# 26 readings a plan or a limit was taken under): in nine of the eleven processes the run held
# 0.016 GiB of the card outside torch's count at the most, windows and VAE passes alike, at
# 1080p and at 4K, 36 blocks swapped, and 0.003 with the models whole, through windows of 1 to
# 9 latents at 1080p. In the two others, 1080p windows of 8 to 21 latents with 36 blocks
# swapped, held 0.25 GiB under the card's figure, 24 of the 38 steps that ended read 0.019 to
# 0.102 GiB during the step and 30 read 0.019 to 0.058 after it, the desktop's own memory
# having grown 14 and 37 MiB under them by nvidia-smi: the card's figure can't tell the run's
# memory from the desktop's.
# Held 1.0, 0.5 and 0.25 GiB under the card's figure alike, each in two processes: what fits
# ran at its usual speed with the reference's bits, 12 latents were refused in 0.2 to 0.4 s
# and the windows after them were right, the card never under 0.58 GiB free (0.27 at the
# least in the passes run under 0.25 with 36 blocks swapped). It is the margin under the
# card's figure, read: the limit is set that far under the smaller of the two figures (hold),
# and each unit checked against the card's alone, the driver's standing up to 0.83 GiB under
# it near the full card with nothing behind it (check). What it leaves to another program's
# growth is that 0.5 GiB less what the run holds outside torch's count: a desktop that takes
# more during a run stops it at its next unit, or under expandable segments has torch held
# again there, within RESERVE (check). Run under it that evening on the same
# card, a desktop holding 5.2 to 5.8 GiB of it: a 1080p job's checks had 0.47 to 0.50 GiB to
# spare in two runs whose desktop stood still, and 0.73 to 1.05 in one whose desktop's own
# memory fell by 0.44 GiB through the run (nvidia-smi), its last check 0.30 under the one
# before it; a second program taking 2 GiB during the job's first unit stopped it at its
# second, 1.48 GiB short, the card never under 0.9 GiB free. PROVISIONAL (design, 2026-10-12):
# one card, one driver, one desktop.
HELD_BACK = round(0.5 * GIB)
# The same where the whole card's free memory isn't read (NVML silent), and the driver's figure
# is all there is: that figure leaves the other programs' memory out until the card fills, and
# stood 1.98 GiB over the card's at rest (the module's docstring), so that HELD_BACK under it
# would be a limit over what the card has. Sized against the driver's figure alone, the day
# before the card's was read (research/scripts/hold_probe.py, the same RTX 5090, 2026-10-11, a
# desktop holding 2.7 to 4.5 GiB of it): once a run held 22 GiB or more, the driver reported 2.1
# to 2.8 GiB less free than torch's own count explains (6 latents at 1080p: 30.16 GiB gone from
# the figure with 27.82 held by torch; a 720p VAE decode: 2.6; the command line at 432p with no
# block swapped: 2.1), and 0.14 GiB under that (12 or 36 blocks swapped, a window of 1 latent).
# Held 2.0 GiB under, the figure went down to 0.16 GiB at the least; held 0.6 under, the card
# went to 0 free and the work after a refusal failed. 3 GiB: the largest measured and a little;
# held so, the figure never went under 1.59 GiB. PROVISIONAL (implementation, 2026-10-11): one
# card, one driver (616.92), one desktop, 1080p and under.
HELD_BACK_UNREAD = 3 * GIB
# What a held run's plan leaves torch's own allocator beyond the peak a step allocates. Without
# expandable segments (EXPANDABLE), what that allocator needs from the driver there, held as
# above: it keeps the blocks it has used, splits them for smaller requests, and gives back at
# its limit only those that are whole again. Measured as the smallest limit a step ran under,
# its cache emptied first, less its allocated peak, going down to the first refusal, by 0.1 GiB
# but where a figure says 0.2 or 0.3 (research/scripts/hold_probe.py's ladders, an RTX 5090
# under WSL2, 2026-10-12; every run that ended gave the reference's bits, and none was slower
# at its smallest limit):
# - a window at 1080p: 0.2 GiB at 1 latent, 0.6 at 6 and 0.8 at 8, the models whole (17.04,
#   22.56 and 24.75 GiB allocated; each in two processes); with the DiT's 36 blocks swapped,
#   0.8 again at 8 (by 0.2), 1.2 at 12 and 3.6 at 17 (14.60 and 19.77 allocated; by 0.3, once);
# - a window of 5 latents at 4K, 36 blocks swapped (23.29 allocated): 4.5, in two processes;
# - a VAE pass, 36 blocks swapped, each in two processes: over 33 frames at 1080p, 3.4 for an
#   encode in tiles of 1856 px (20.59 allocated), 3.8 untiled (20.89), 5.0 for a decode in
#   tiles of 1088 px (21.19), 5.3 in tiles of 1152 (22.58), and in tiles of 1280 refused with
#   28.76 GiB allowed (24.88 allocated by then); over 17 frames at 4K, 4.8 for an encode in
#   tiles of 1472 px (21.28) and 5.1 for a decode in tiles of 1088 (21.19).
# It grows with what the step works in and is no constant; this one covers the plans a 32 GB card
# makes. The planner holds every VAE pass 1.25 GiB under its budget besides (planner.VAE_ROOM; 1.5
# when this was measured), and a window not: so a plan counts on the limit less the larger of a
# window's need, 4.5 GiB, and of a VAE pass's less that room, 5.3 - 1.25 = 4.05. The planner's own
# margin (planner.MARGIN, 0.6 GiB) is counted IN it, not on top: it measures the same need where a
# driver refuses, and with both the planner refused a window of 5 latents at 4K, the smallest,
# beside that card's usual desktop, by 0.27 GiB (counted on the CPU, never run so on the card;
# design, 2026-10-12; DESIGN.md, Allocator): on a held run the budget is what the plan counts on
# (planner._Job.margin). The day before it was 2.3, from one window of 8 latents: with the margin
# and the room that left a VAE pass 4.4 GiB beyond its peak, under what a decode in tiles of 1088 px
# needs. PROVISIONAL (implementation, 2026-10-12): one card; a larger card's larger windows and
# tiles need more, unmeasured; it follows planner.VAE_ROOM, to be raised by what that room loses;
# whether a held run should take an allocator that needs less (max_split_size_mb at 128 MiB ran the
# decodes in less, 2.4 to 3.7 times slower) is design's to decide, as is a room by the kind of step
# in this one figure's place (DESIGN.md, To measure).
# Measured again on 2026-10-13 (the same probe and card, a desktop holding 3.2 to 4.6 GiB of it):
# that need is no threshold and follows no size a plan knows. 4K's window of 3 latents was refused
# under its peak and 2.5 GiB, and ran under 2.4, 2.3 and 2.2; 98,640 tokens needed 2.6 GiB at 4K
# where 99,360 needed 1.2 at 1080p, and 131,520 needed 1.9 where 132,480 needed 3.0; and an encode
# in tiles of 1856 px at 1080p needed 3.4 GiB with the DiT's parts on the GPU as 36 swapped blocks
# leave them, and 5.2 with the DiT off it, where the untiled encode, larger, needed 3.8 either way.
# Hence expandable segments by default (EXPANDABLE), under which a step with less than this figure
# slows down and runs, down to 0.1 GiB over its peak; the figure is kept: with it every window
# measured there ran at its usual speed (2.5 GiB at the most), and a VAE pass, left 6.0 with the
# planner's room then (1.5; 5.75 with 1.25), too (a decode takes 3.0 to 4.7 GiB beyond its peak when
# it is allowed them). Whether less should be planned under that allocator is design's to decide.
# Under those segments it is also what a check that finds the card short may take from the
# limit, other programs having grown, before the run stops (check; DESIGN.md, Allocator,
# decided on 2026-10-16): a step is then left less beyond its peak, never under it.
RESERVE = round(4.5 * GIB)
# The reserve a held run's plan takes where the plan made with RESERVE would be refused, a new
# job's as a resume's recorded shape (planner.plan; DESIGN.md, Allocator, decided on 2026-10-13
# with the user's go, for a resume's plan on 2026-10-21), and nowhere else: under expandable
# segments a step left less than its need beyond its peak slows down and runs (EXPANDABLE). Run
# so on the RTX 5090 under WSL2 (2026-10-13, a desktop holding 3.2 GiB of the card): 4K's
# smallest window planned and ran with it four times, with the frames of a run that had room to
# spare, in 134 to 139 s for 65.6; but every other plan packed tighter, and a job that fitted
# with RESERVE ran 1.4 to 1.5 times as long with this one; a 1080p job whose plan with it had the
# budget of a roomy run's, 1.08 times (2026-10-22). So it is the plan's last resort, said slow
# in the plan's budget line, up to 1.5 times as long, by how tightly the plan packs
# (planner._card_said; DESIGN.md, Allocator, decided on 2026-10-22). The VAE's room
# (planner.VAE_ROOM) is inside the budget either way, never touched: a VAE pass is still left
# that room and more beyond its predicted peak, its frames' floor under the segments, 0.40 GiB,
# under it (EXPANDABLE). And it
# is what a check may take from the limit before the run stops (check, hold's `reserve`).
RESERVE_TIGHT = round(2.5 * GIB)
# What this process's own CUDA context and libraries will hold of the card once torch has made them,
# in bytes, for the reading a resume takes through NVML before any of that exists (early): the plan
# is made in the free memory read with the context, the warm-up and the first allocation already on
# the card (cli._free_memory), so the earlier reading, taken without them, counts them off.
# PROVISIONAL (impl-v010, 2026-10-28: DESIGN.md gives no figure, and the code none for a context
# alone): 1.23 GiB, the most the card's use rose through a first pass, "the process's context and
# libraries in it" (hold, the same RTX 5090 under WSL2, nvidia-smi, 2026-10-12). It is the larger
# figure of those the code records (0.24 GiB for the first allocation, 0.29 with the warm-up on the
# 96 GB card), so that the early reading errs towards warning of a shape the later check would
# accept, never towards saying "fits" where it refuses. To be measured on the RTX 5090 at the
# release gate (design, 2026-10-28): until then the figure is the most the code has seen, and the
# warning it feeds says it is an early reading (cli._early_memory).
CONTEXT = round(1.23 * GIB)
# The allocator's setting by the driver: DESIGN.md's default (Allocator), and torch's own
# allocator where the driver spills.
ASYNC, NATIVE = "backend:cudaMallocAsync", "backend:native"
# torch's own allocator as a held run takes it by default: its segments expandable
# (PYTORCH_CUDA_ALLOC_CONF's expandable_segments: each of the allocator's pools one range of
# addresses, grown and given back by mapping, where its segments are otherwise allocations of
# the driver's that it gives back only whole). Without them a step short of room is refused,
# its cache in pieces that fit no request, and what it needs to run follows nothing a plan
# knows (RESERVE). With them a step short of room slows down, and runs. Measured by
# research/scripts/hold_probe.py's ladders (an RTX 5090 under WSL2, driver 616.92, torch
# 2.14.1, 2026-10-13, a desktop holding 3.2 to 4.6 GiB of the card; every digest compared with
# a reference's):
# - windows of 1 to 17 latents at 1080p and of 2, 4 and 5 at 4K (36 blocks swapped) ran under
#   their allocated peak and 0.1 GiB, with the reference's bits, and were refused under the
#   peak itself, in 0.5 to 6.6 s; under the plain allocator those at 1080p needed 0.2 to 3.4
#   GiB, and 4K's of 5 latents 4.5, which the card didn't have that night;
# - they ran at their usual speed from 0.7 GiB over the peak at 6 latents, 1.7 at 12, 1.5 at
#   14, 2.5 at 16 and 2.0 at 17 (1080p), 1.0 at 2 and 2.5 at 4 (4K), and 3 to 4 times slower
#   at the least margin (14 s a forward of 16 latents, 41 there); 4K's window of 5, the VAE's
#   weights away, took 22 s a forward from an emptied cache with 4.3 GiB over its peak (14 s
#   the next one), 21 with 2.8, 22 with 2.3, 59 with 1.8 and 79 to 90 from 1.3 down to 0.1;
# - a VAE pass at 1080p, the DiT off the GPU: an encode in tiles of 1088 px ran at its usual
#   speed from 2.0 GiB over its peak, in tiles of 1792 px from 3.5 (14.6 s; 28 at 3.0),
#   untiled in 18.8 s at 3.5 where 15.1 with 7.3; a decode in tiles of 1024 px in 37.7 s at
#   4.0 where 24.9 with 9.2, in tiles of 1088 px in 37.8 s where 27.0 with 7.0; each two to
#   four times slower from 1.0 down. They gave the reference's bits down to 0.5 GiB over the
#   peak, and OTHER bits where they still ran under that: at 0.4 and 0.3 (the untiled encode,
#   the decode in tiles of 1024 px), at the peak itself (the encodes in tiles), as torch's
#   plain allocator does in the last 0.3 GiB above its refusal (the encode in tiles of 1088
#   px: other bits at its peak and 2.0 to 1.8 GiB, refused at 1.7): a pass is never to be
#   left that little, and RESERVE with the planner's room leaves it twelve times that;
# - torch's limit holds as before, and its count is the card's: 0.002 to 0.027 GiB of the
#   card outside it;
# - a job of 66 frames at 1080p, planned alike under both: the same frames; 83.5 s where
#   80.9; torch holding 22.1 GiB at the most at its checks where 25.2, the card never under
#   4.27 GiB free where 0.96.
# PROVISIONAL (implementation, 2026-10-13): one card, one driver, one version of torch; not
# measured on a driver that refuses, where the default stays cudaMallocAsync (ASYNC).
EXPANDABLE = f"{NATIVE},expandable_segments:True"
# The variables torch reads that setting from. seedvr2x sets the first, and only when the user
# has set neither: with both set, torch 2.14 takes the first and never reads the second, so a
# default written over a user's PYTORCH_ALLOC_CONF would silence it.
VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


@dataclass(frozen=True)
class Driver:
    """What the probe found: whether the driver gives memory the card doesn't have (spills), and
    its figures, in bytes: the request (asked), the memory the driver reported free before it
    (free) and the card's (total); and, where the driver wasn't asked (asked 0), what the whole
    card had free by NVML, less than it reported (card)."""

    spills: bool
    asked: int
    free: int
    total: int
    card: int | None = None

    def said(self) -> str:
        """The finding in words, for the log."""
        figures = f"{self.asked / GIB:.2f} GiB asked with {self.free / GIB:.2f} free"
        if not self.spills:
            return f"the GPU's driver refuses what the card can't hold ({figures}: refused)"
        figures += ": given"
        if self.card is not None:
            figures = (
                f"it reports {self.free / GIB:.2f} GiB free to this process where the whole card"
                f" has {self.card / GIB:.2f} by NVML, other programs' memory left out: not asked"
                " for more, which would push theirs off the card"
            )
        return (
            f"the GPU's driver gives memory the card doesn't have ({figures}): system"
            " memory in its place, as Windows' driver does, under WSL2 too whatever NVIDIA's"
            ' "CUDA - Sysmem Fallback Policy" says. Nothing there refuses a run that passes the'
            " card: it would slow down, then fail without an error to catch, what it made last"
            " not to be trusted"
        )


@dataclass(frozen=True)
class Reading:
    """The GPU's free memory as a plan is made in it, in bytes, each figure by its name, for
    whoever words the plan and its refusals (runtime/planner.py) and for the manifest:
    - reported: what the driver reports free to this process (cli._free_memory);
    - card: what the whole card has free by NVML, other programs' memory counted (card_free);
      None where it isn't read: on a driver that refuses, whose figure is the card's, and
      where NVML doesn't answer;
    - counted: what the plan may count its peaks in (usable): `reported` on a driver that
      refuses, less on one that spills;
    - held: whether the run holds itself to the card, its driver giving memory the card doesn't
      have (spills);
    - left: on a held run, what torch's limit leaves of the free memory (hold: HELD_BACK, or
      HELD_BACK_UNREAD where the card's figure isn't read), the rest of what `counted` is
      under it being the reserve; 0 on a driver that refuses;
    - tight: on a held run, whether the plan is made with RESERVE_TIGHT, the plan made with
      RESERVE refused (tightened);
    - expandable: on a held run, whether torch's own allocator takes it with expandable segments
      (EXPANDABLE, startup), under which a step short of room slows down and runs where it is
      otherwise refused: the only allocator a plan takes RESERVE_TIGHT under (planner.plan;
      DESIGN.md, Allocator: "With a reserve of 2.5 GiB under those segments")."""

    reported: int
    card: int | None
    counted: int
    held: bool
    left: int = 0
    tight: bool = False
    expandable: bool = True

    @property
    def reserve(self) -> int:
        """On a held run, what the plan leaves torch's allocator beyond a step's peak, in bytes:
        RESERVE, or RESERVE_TIGHT where the reading is `tight`; meaningless on a driver that
        refuses, which leaves none."""
        return RESERVE_TIGHT if self.tight else RESERVE


def tightened(reading: Reading) -> Reading:
    """`reading`, a held run's, its plan counting on RESERVE_TIGHT less than torch's limit
    where RESERVE: what the planner plans in where the plan made in `reading` is refused, a new
    job's or a resume's recorded shape (planner.plan; DESIGN.md, Allocator). The free memory and
    what torch's limit leaves under it are the reading's: only the reserve moves."""
    taken = reading.reported if reading.card is None else min(reading.reported, reading.card)
    counted = max(taken - reading.left - RESERVE_TIGHT, 0)
    return replace(reading, counted=counted, tight=True)


def reading(free: int) -> Reading:
    """The reading a plan is made in, of `free` bytes the driver reports free to this process
    (cli._free_memory): on a driver that refuses, all of it, the card's own figure not read; on
    one that spills, the smaller of it and of what the whole card has free now (_twice: NVML
    asked again where it fails; the run's device is CUDA's first, the one the probe asks), less
    what torch is held back from (hold: HELD_BACK) and RESERVE; where NVML fails twice, the
    driver's figure alone less HELD_BACK_UNREAD and RESERVE, with a warning. The planner's
    budget is `counted` less its own margin on a driver that refuses, and `counted` itself on a
    held run, RESERVE holding that margin (runtime/planner.py); its plan's first line names the
    figures: here they are said at debug level only, with what each of the two constants is
    left to.

    The figure taken is the run's footing (_footing): torch's limit for the run's units is set on
    it, and its units are checked on it (hold, check), so that the plan stays RESERVE under the
    limit, and the warning says what is done. Until 2026-10-13 NVML failing here once was taken
    for the run's answer, while the limit and the checks read the card again: the plan then
    counted on the driver's figure less 7.5 GiB under a limit set 0.5 under the card's, and the
    warning said the run was held 3.0 GiB under the driver's figure, which it wasn't (design's
    check, 2026-10-13; DESIGN.md, Allocator: "NVML failing at the plan's reading is asked again
    as well ... the plan, the limit and the checks stand on one footing")."""
    if not _spills:
        return Reading(free, None, free, False)
    global _planned, _footing
    read = _twice(0)
    card = read if isinstance(read, int) else None
    if card is None:
        logger.warning(
            "the whole card's free memory couldn't be read through NVML as the plan was made,"
            " asked twice (%s): what the driver reports free to this process is taken alone,"
            " which leaves other programs' memory out while the card has room: the plan counts"
            " on that figure, torch is held %.1f GiB under it and the run's units are checked on"
            " it",
            read,
            HELD_BACK_UNREAD / GIB,
        )
    _footing, _planned = (DRIVER if card is None else CARD), card
    taken = free if card is None else min(free, card)
    counted = max(taken - _held_back(card) - RESERVE, 0)
    logger.debug(
        "the plan counts on %.2f GiB of the GPU's memory, of the %.2f GiB free %s: %.1f GiB are"
        " left to what torch doesn't count on the card, %.1f to what the plan leaves torch's"
        " allocator beyond a peak",
        counted / GIB,
        taken / GIB,
        _named(free, card),
        _held_back(card) / GIB,
        RESERVE / GIB,
    )
    return Reading(free, card, counted, True, _held_back(card), expandable=_native and _segments)


def early(card: int) -> Reading:
    """The reading a plan would be made in for a resume, from `card`, what the whole card has
    free by NVML (card_free), taken before torch is imported and so before any CUDA context of
    this process is on the card: CONTEXT less, which the later reading has already counted. The
    allocator is not chosen yet (startup) and the driver not asked: behind WSL2's device the run
    is taken for held, as startup takes a driver that doesn't answer, the free memory the
    driver reports to the process counted as the card's (the smaller of the two figures, as
    reading takes it), torch's own allocator with expandable segments unless the environment's
    setting says otherwise (_allocator, as startup reads it); elsewhere the driver refuses and
    its figure is the card's. Nothing is remembered and nothing is logged: the module's state is
    startup's, and a run's own reading comes later (DESIGN.md, Memory planner, A plan shapes the
    output a little, decided on 2026-10-28: "the room is read through NVML ... NVML only")."""
    free = max(card - CONTEXT, 0)
    if not paravirtual():
        return Reading(free, None, free, False)
    given = next((os.environ[name] for name in VARIABLES if os.environ.get(name)), None)
    native, segments = _allocator(EXPANDABLE if given is None else given)
    counted = max(free - HELD_BACK - RESERVE, 0)
    return Reading(free, free, counted, True, HELD_BACK, expandable=native and segments)


# What startup found, for the run that follows: whether the run holds itself to the card,
# whether its allocator is torch's own, and whether that allocator's segments are expandable,
# under which a check that finds the card short holds torch again (check); and, once hold has
# set it, torch's limit in bytes.
_spills = False
_native = False
_segments = False
_limit: int | None = None
# The limit set for the run's units once its plan is made (hold), in bytes: what a check that
# holds torch again counts the limit's loss from, against RESERVE (check); None before it.
_plan_s_limit: int | None = None
# The figure the checks are judged on, the card's or on the driver's footing the driver's, as the
# limit in force was set (hold, or check where it holds torch again), and what torch held then, in
# bytes: what the memory outside torch's allocator grew since is counted from them (check).
_base: tuple[int, int] | None = None
# What hold left of the free memory outside torch's limit, in bytes, for the check's words.
_left = HELD_BACK
# The reserve the run's plan took (Reading.reserve), in bytes, as hold was given it once the plan
# was made: what a check may take from the limit before the run stops (check).
_reserve = RESERVE
# The figure a held run's plan, its limit and the checks before its units go by, one footing for
# the three (DESIGN.md, Allocator, decided on 2026-10-13): CARD, the whole card's free memory by
# NVML, or DRIVER, what the driver reports free to this process alone. Set by the plan's reading
# (reading) and by each limit (hold): the first, before the first pass, by its own reading; the
# run's, once the plan is made, on the plan's footing; None before either (startup).
CARD, DRIVER = "card", "driver"
_footing: str | None = None
# The whole card's free memory as the run's plan was made in it (reading), in bytes; None where
# it wasn't read then, or no plan made yet: what the limit is set from where NVML, having given it
# to the plan, fails as the limit is set (hold).
_planned: int | None = None


class _Memory(ctypes.Structure):
    """NVML's nvmlMemory_v2_t (nvml.h), in bytes: the card's memory in all, what its driver
    keeps for itself (reserved), what is unallocated (free) and what programs hold (used).
    version: the structure's size and its version, 2, as NVML_STRUCT_VERSION packs them."""

    _fields_ = (
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


def startup() -> str:
    """Ask the driver whether it refuses, once a run, before torch is imported, and set the
    allocator the run takes by default (VARIABLES, which the user's own setting overrides):
    EXPANDABLE where the driver spills, ASYNC elsewhere; returned. Called where a run first takes
    the GPU (cli._run, its model files checked), so that a run refused before it, for its
    source, its settings or its model files, asks nothing: the request disturbs the card (probe).
    Nor does a job recorded with every segment made, which takes no GPU: it isn't called for one.

    The driver is asked only where it can spill, behind WSL2's device: a Linux driver refuses,
    and isn't disturbed. What was found is said in the log, a warning where it spills. No
    answer there (probe) is taken for a driver that spills: it is Windows', which gave what
    the card didn't have every time it was asked, and a run held for nothing loses memory, a
    run not held where it should be loses its frames."""
    global _spills, _native, _segments, _limit, _left, _footing, _planned, _plan_s_limit, _base
    global _reserve
    _spills, _limit, _left, _footing, _planned = False, None, HELD_BACK, None, None
    _plan_s_limit, _base, _reserve = None, None, RESERVE
    if paravirtual():
        found = probe()
        if found is None:
            _spills = True
            logger.warning(
                "the GPU is Windows', reached through WSL2, and its driver didn't say whether"
                " it refuses what the card can't hold: taken for one that gives system memory"
                " in its place, as that driver did whenever it was asked"
            )
        else:
            _spills = found.spills
            logger.log(logging.WARNING if _spills else logging.INFO, "%s", found.said())
    # An empty setting is no setting, and goes from the environment. torch takes the first of
    # the two variables that is set, empty or not: left there, an empty PYTORCH_CUDA_ALLOC_CONF
    # would hide a PYTORCH_ALLOC_CONF that names an allocator, and itself names none, which is
    # torch's own (torch 2.14.1, each case read in a process of its own: tests/test_memory.py).
    # The other variable's setting then stands, or the default.
    for name in VARIABLES:
        if os.environ.get(name) == "":
            del os.environ[name]
    given = next((os.environ[name] for name in VARIABLES if name in os.environ), None)
    chosen = EXPANDABLE if _spills else ASYNC
    if given is None:
        os.environ[VARIABLES[0]] = chosen
    # torch's own allocator is its default too, when the environment's setting names none.
    _native, _segments = _allocator(chosen if given is None else given)
    if _spills:
        # The environment's allocator stands, and is said next where it isn't the default.
        named = (
            "torch's own allocator, its segments expandable, limited"
            if _segments
            else "torch's own allocator without expandable segments, limited"
            if _native
            else "torch limited"
        )
        # The tight reserve said where a plan can take it: under the segments (planner.plan).
        tight = (
            f", {RESERVE_TIGHT / GIB:.1f} where that plan would be refused, slower"
            if _native and _segments and RESERVE_TIGHT < RESERVE
            else ""
        )
        logger.warning(
            "seedvr2x holds itself to the card there: %s to the"
            " memory free less %.1f GiB, and plans in %.1f GiB less%s; a run that passes the"
            " limit is refused. The memory free is the smaller of two figures: what the driver"
            " reports free to this process, which leaves other programs' memory out while the"
            " card has room, and what the whole card has free, read through NVML; where NVML"
            " doesn't answer, the driver's figure less %.1f GiB",
            named,
            HELD_BACK / GIB,
            RESERVE / GIB,
            tight,
            HELD_BACK_UNREAD / GIB,
        )
        if not _native:
            # The user's own allocator stands; said, since the hold then holds less.
            logger.warning(
                "the allocator is the environment's, %s: torch's limit counts what torch has"
                " allocated there, not what that allocator keeps beyond it (2 GiB and more),"
                " so the card can still be passed; unset it for torch's own allocator",
                given,
            )
        elif not _segments:
            # The user's own setting of torch's allocator stands, without the segments a held
            # run takes by default (EXPANDABLE): said, since a step short of room is then
            # refused where they slow it down and run it (DESIGN.md, Allocator, decided on
            # 2026-10-13: "a user's own backend:native without the segments is said").
            logger.warning(
                "the allocator is the environment's, %s: torch's own without expandable"
                " segments, under which a step short of room is refused where they would slow it"
                " down and run it; add expandable_segments:True to it, or unset it, for a held"
                " run's default, %s",
                given,
                EXPANDABLE,
            )
    return chosen


def _allocator(setting: str) -> tuple[bool, bool]:
    """How torch reads an allocator's setting (VARIABLES) as it is loaded: whether it takes its
    own allocator, and whether that allocator's segments are expandable. Its tokens are what
    lies between commas, colons and brackets, blanks dropped; an entry is a key, a colon and a
    value, a bracketed list one value (roundup_power2_divisions:[256:1,512:2]). The first
    backend stands: cudaMallocAsync where it names that, to the letter, torch's own allocator
    otherwise; the last expandable_segments stands, True to the letter. So torch 2.14.1 read
    them, a setting at a time in a process of its own (tests/test_memory.py, TORCH_READS:
    "backend: native" its own allocator, "backend:cudaMallocAsync,backend:native"
    cudaMallocAsync), and on an RTX 5090 at a first allocation (2026-10-16): "backend: native,
    expandable_segments:True" its own allocator with the segments, "expandable_segments:True,
    expandable_segments:False" without them, "False,True" with them. A setting torch then
    can't run stops it at that first allocation, before any unit: a backend it doesn't know
    ("backend:cudamallocasync"), a second one other than the first under its own allocator
    ("backend:native,backend:cudaMallocAsync"), "true" or "1" for the segments, a list that
    isn't of numbers; and a key or a value missing at a colon ("backend:") as it is imported.
    Until 2026-10-16 the backend was read by a substring, so that "backend: native" passed for
    another allocator, the run then unchecked (check), and "backend:cudaMallocAsync,backend:
    native" for torch's own (a review)."""
    tokens: list[str] = []
    word = ""
    for char in setting:
        if char in ",:[]":
            tokens += [word, char] if word else [char]
            word = ""
        elif not char.isspace():
            word += char
    tokens += [word] if word else []
    backend, segments, depth = None, False, 0
    for position, token in enumerate(tokens):
        depth += {"[": 1, "]": -1}.get(token, 0)
        if depth or token != ":" or not 0 < position < len(tokens) - 1:
            continue
        key, value = tokens[position - 1], tokens[position + 1]
        if key == "backend" and backend is None:
            backend = value
        elif key == "expandable_segments":
            segments = value == "True"
    native = backend != ASYNC.partition(":")[2]
    return native, native and segments


@contextmanager
def watched(device: Any, unit: str, output: str) -> Generator[None]:
    """Around a VAE unit on `device`, `unit` naming it (a shot's encode, a decode), `output`
    what it gives: on a held run under torch's own allocator, warn where torch's count of the
    allocations it refused (memory_stats' num_ooms) grew across the unit, which then ended
    (DESIGN.md, Allocator, decided on 2026-10-18): measured on the RTX 5090 under WSL2 under the
    expandable segments, a VAE pass alone (2026-10-17), the count stayed 0 on every pass that
    kept its frames and grew by 19 to 38 on every one that lost them, cuDNN's workspaces refused,
    which cuDNN runs on without, other bits from other kernels and nothing said. A count that
    grows says the unit ran short of room, its output maybe not that of a run with more room;
    one that doesn't says nothing either way. Nothing elsewhere: a driver that refuses isn't
    held, and under cudaMallocAsync, or an allocator of the environment's, torch counts no
    refused allocation (DESIGN.md, Memory planner). Nothing said of a unit that raises: its
    error is the run's."""
    if not (_spills and _native):
        yield
        return
    import torch

    before = int(torch.cuda.memory_stats(device).get("num_ooms", 0))
    yield
    after = int(torch.cuda.memory_stats(device).get("num_ooms", 0))
    if after > before:
        # PROVISIONAL (implementation, 2026-10-21): the words, and a warning for each unit.
        logger.warning(
            "%s: torch's allocator refused %d of its requests during it (%d to %d in all):"
            " where cuDNN's workspaces are refused it runs other kernels, which give other bits:"
            " its %s may not be those of a run with more room on the card",
            unit,
            after - before,
            before,
            after,
            output,
        )


def spills() -> bool:
    """Whether the run holds itself to the card: its driver was found to give memory the card
    doesn't have, or is Windows' and gave no answer (startup)."""
    return _spills


def usable(free: int) -> int:
    """The memory a plan may count its peaks in, of `free` bytes the driver reports free to this
    process: the reading's `counted` (reading), for who wants the number alone."""
    return reading(free).counted


def hold(device: Any, first: bool = False, reserve: int | None = None) -> int | None:
    """On a driver that spills, limit torch on `device` (a torch.device) to what it holds now
    and the memory free now less HELD_BACK, the memory free being the smaller of what the
    driver reports to this process and of what the whole card has (card_free); less
    HELD_BACK_UNREAD where the card's figure isn't read, the driver's alone standing over what
    the card has. More is refused, an OutOfMemoryError. Returns the limit, in bytes; None,
    nothing done, on a driver that refuses.

    Called twice a run (cli._run). A first time (`first`) before anything of the run works on
    the GPU: the shot detector in the first pass, 0.53 GiB at its peak, and the reading's
    warm-up ran before any limit existed, the card's use up 1.23 GiB at the most through a
    first pass, the process's context and libraries in it (nvidia-smi, the same RTX 5090,
    2026-10-12). Then once the plan is made and before the models load, from the memory free
    then, which is the limit the run's units are checked against, and the one a check that
    holds torch again counts what the limit has lost from (check), within `reserve`, the one the
    plan took (Reading.reserve: RESERVE, or RESERVE_TIGHT where a plan with RESERVE was refused;
    None, RESERVE).
    The first is said at debug level, the second being the run's.

    The figure each limit is set under is the run's footing, which it sets (_footing; DESIGN.md,
    Allocator: "the plan, the limit and the checks stand on one footing"): the first's, its own
    reading of the card, NVML asked twice where it fails (_twice), the driver's figure where it
    fails twice; the second's, the plan's (reading). Where the plan went by the driver's figure,
    NVML isn't asked: the limit is set HELD_BACK_UNREAD under that figure, which the plan counted
    on, as its warning says. Where the plan went by the card's, the card is read again, twice
    where NVML fails, and where it fails twice the limit is set from the card's figure the plan
    was made in (_planned), with a warning, never under the driver's figure alone: the plan,
    made HELD_BACK and RESERVE under the card's figure, would stand closer to a limit set
    HELD_BACK_UNREAD under the driver's than RESERVE asks, by what the driver's figure is over
    the card's less the difference of the two constants (1.98 GiB over: 3.98 GiB under the
    limit for 4.5; a review, 2026-10-12). With no plan's figure of the card either (a limit set
    again with no plan made, which a run never does), the driver's figure, with a warning: the
    run's units are then checked on it (check)."""
    global _limit, _left, _footing, _plan_s_limit, _base, _reserve
    if not _spills:
        return None
    import torch

    reported, total = torch.cuda.mem_get_info(device)
    card = _limit_s_card(_index(device), first)
    _footing = DRIVER if card is None else CARD
    free = reported if card is None else min(reported, card)
    _left = _held_back(card)
    reserved = torch.cuda.memory_reserved(device)
    limit = reserved + max(free - _left, 0)
    fraction = min(limit / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device)  # pyright: ignore[reportUnknownMemberType]
    _limit = limit
    _plan_s_limit = None if first else limit
    if not first:
        _reserve = RESERVE if reserve is None else reserve
    _base = (reported if card is None else card, reserved)
    logger.log(
        logging.DEBUG if first else logging.INFO,
        "torch held to %.2f GiB on the GPU%s: the %.2f GiB free less %.1f, and what it holds;"
        " free %s",
        limit / GIB,
        ", a first time, before the first pass" if first else "",
        free / GIB,
        _left / GIB,
        _named(reported, card),
    )
    return limit


def check(device: Any) -> None:
    """On a driver that spills, once torch is held: refuse (torch.OutOfMemoryError, as torch
    refuses what passes its limit) when what torch may still take under its limit no longer fits
    in what the whole card has free (card_free); in what the driver reports free to this
    process, where the card's figure was never read in the run. Under expandable segments, on
    the card's figure, hold torch again to what the card has instead, within the plan's reserve
    (below).
    Called before every unit (model.Models.enter).

    The limit is worth the memory free when it was set, less HELD_BACK. Other programs taking
    the card's memory since, torch at its limit would pass the card with nothing to refuse it.
    So the run stops, its units kept.

    Under expandable segments (startup), where the card's figure is read, a check that finds the
    card short holds torch again and the run goes on (DESIGN.md, Allocator, decided on 2026-10-16):
    to what it holds and the card's free memory less HELD_BACK, said in a warning, as long as what
    the limit has lost since the one set once the plan was made (hold) stays within the reserve that
    plan leaves torch's allocator beyond a step's peak, RESERVE, or RESERVE_TIGHT where the plan
    took it (_reserve; DESIGN.md, Allocator, decided on 2026-10-21: a re-hold bounded by the reserve
    its plan took, 4.5 there cutting into the VAE's room). A step is then left that much less beyond
    its peak, a step short of room slowing down under the segments where it would be refused
    (EXPANDABLE), and the VAE's room, inside the plan's budget, is never touched. It holds torch
    again only where the card has HELD_BACK free at that check, the new limit at or above what torch
    holds: under it the limit would stand under torch's own reservation, every later check would
    find the run's room under nothing and pass, the bound gone, while torch gives none of its cache
    back between the units of a phase (a review, 2026-10-16: held again to 24.75 GiB with 25 held, a
    program taking 4.75 more went unsaid). The run stops there instead. But a check that finds the
    card short gives torch's cache back first (empty_cache) and judges the card again (DESIGN.md,
    Allocator, decided on 2026-10-18): what torch holds and doesn't use goes back to the card, so
    that a growth larger than torch's room, within the plan's reserve, finds the HELD_BACK free and
    is held again, not stopped; the stop and the re-hold then say what the cache gave back. A
    desktop in use had stopped 3 runs of 6 under the segments and none of 4 under the plain
    allocator, the memory outside torch having grown 0.50 to 0.59 GiB past its level at the plan:
    the segments leave room on the card, and the desktop took it and kept it, where the plain
    allocator filled the card and the desktop's growth was given back (an RTX 5090 under WSL2,
    2026-10-16; DESIGN.md, Allocator). Past that bound, under torch's allocator without the
    segments, or on the driver's figure, the run stops. A check never raises the limit again, the
    card having more free than when torch was held (PROVISIONAL, implementation, 2026-10-16: design
    decided the re-hold, not whether memory given back is taken again). Every stop and every re-hold
    says what the memory outside torch's allocator grew since the limit in force was set, against
    what that limit left it (HELD_BACK), counted from the figure judged and what torch held then
    (_base): the check itself goes by the limit, which may have been set under the driver's figure,
    the smaller then, the card left that much more (a review, 2026-10-16: counted from the limit,
    the growth said fell short by that much). Run so on the same card (2026-10-16, a job of 66
    frames at 1080p, its window, tiles and swapped blocks pinned, a second program taking memory
    once the plan's limit was set): under the segments, with 1.0 and 2.0 GiB, torch was held again
    1.00 and 2.00 GiB lower at the first unit's check and the run went on, upscaled in 60.2 and 60.1
    s (59.7 to 60.7 with no program, earlier that day) with the frames of a run with none; with 5.0
    GiB it stopped there, and the same command resumed to those frames; with 0.6 GiB nothing was
    done, the memory outside torch grown 0.47 GiB by that check, the desktop's own fallen by 0.14,
    every check 0.03 to 0.04 GiB to spare. Under the plain allocator 0.6 GiB stopped it there, 0.61
    grown.

    On the card's figure, where it is read, and not on the smaller of the two as the limit was
    set: near the full card the driver's figure is no longer the process's own, and isn't the
    card's either. Measured on the RTX 5090 (driver 616.92, 2026-10-12, a desktop holding 947
    to 1,037 MiB of the card by nvidia-smi): from a run's first encode on, the driver reported
    0.33 to 0.83 GiB less free to the process than the whole card had, at every reading, the
    run alone or beside a program holding 8 GiB (0.13 GiB less to 0.12 more, the same night,
    beside a desktop holding 1,663 MiB: there the two had passed for one). Judged on the
    smaller of the two, six runs of six stopped at their second check, with nothing else on
    the card having moved: at 1080p and at 4K, 4 to 35 blocks swapped, alone and beside that
    program. Judged on the card's, the same job at 1080p ran every unit at its usual speed and
    gave the reference's frames: its second window of 9 latents was entered with 0.000 GiB
    reported free to the process, 0.348 free on the card and torch 0.251 under its limit, and
    ran in its usual 7.0 s. And what torch takes from the driver there is the card's: three
    takes of 0.38 to 0.50 GiB, the last leaving 0.000 GiB reported free to the process and
    0.215 on the card, whose figure fell by each of them; written and read as fast as what
    torch already held (1.8 to 2.0 ms a pass), each took 10 to 58 ms to get. What the driver's
    figure counts there, a share kept for the other programs perhaps, wasn't found.

    Measured on the same card the day before (2026-10-11, the command line at
    432p, 30.20 GiB reported free at the start; the limit and the check on the driver's figure
    alone, which left the other programs' memory out until the card filled):
    - alone, the figure stayed 2.85 GiB above what torch could still take with 12 or 36 blocks
      swapped, and 0.77 at the least with none (torch holding 22 GiB);
    - beside a program holding 8.6 GiB and working on it, a run with no block swapped loaded
      its 15.8 GiB, then read 3.89 GiB free where it could still take 11.36, and stopped at its
      first unit; the card had 28.6 GiB of its 31.8 in use at the most (nvidia-smi), never more;
    - beside the same program, a run with 36 blocks swapped, 8 GiB at its most, read the card
      as if alone and finished, its frames right: nothing passed the card there.

    The figure judged is the run's footing (_footing), the one its plan and its limit went by
    (reading, hold; DESIGN.md, Allocator, decided on 2026-10-13: "the plan, the limit and the
    checks stand on one footing"). On the card's, a check NVML fails at is never judged on the
    driver's figure, the very figure that stopped those six runs: NVML is asked again, once
    (_twice), and if it fails again that check is skipped, with a warning, the unit running
    under torch's limit alone (design, 2026-10-12). On the driver's, where NVML didn't answer
    the plan's reading, the limit set HELD_BACK_UNREAD under that figure, the check is judged on
    that figure, NVML not asked: until 2026-10-13 a run whose NVML answered its first limit
    alone had every check skipped, its limit set under the driver's figure, and nothing held it
    to the card (design's check).

    The two figures are said rounded apart, what is free down and what the run may take up, so
    that a refusal by a few bytes never gives one figure twice ("1.00 GiB free ... where the
    run may still take 1.00"); what the memory outside torch's allocator grew, and what the
    limit has lost, up as well: a check short by a byte says 0.51 GiB grown, past the 0.5.

    Nothing is checked under an allocator of the environment's, whose own slack torch doesn't
    count (startup)."""
    global _limit, _base
    if not _native or _limit is None:
        return
    import torch

    found = _figures(device)
    if found is None:
        return
    reported, total, card = found
    free = reported if card is None else card
    reserved = torch.cuda.memory_reserved(device)
    room = _limit - reserved
    if free >= room:
        return
    # The card short: torch's cache given back first, and the card judged again (DESIGN.md,
    # Allocator, decided on 2026-10-18), what torch holds and doesn't use then the card's again.
    # The comparison above doesn't move by it, the card gaining what torch's room does; what it
    # moves is the card's free memory, which a re-hold needs HELD_BACK of, so that a growth
    # larger than torch's room, within the plan's reserve, is held again, not stopped. Judged
    # again, the other programs may have moved too.
    torch.cuda.empty_cache()
    held = reserved
    reserved = torch.cuda.memory_reserved(device)
    given = max(held - reserved, 0)
    found = _figures(device, again=True)
    if found is None:
        # NVML failing at the second reading, the first having found the card short: judged on
        # that reading and what the cache gave back to the card, never skipped, the card known
        # short (a review's finding, 2026-10-21).
        free += given
    else:
        reported, total, card = found
        free = reported if card is None else card
    room = _limit - reserved
    if free >= room:
        logger.debug(
            "the card short of what the run may still take before this unit, torch's cache given"
            " back (%.2f GiB), and judged again: %.2f GiB free %s, the run may still take %.2f",
            given / GIB,
            free / GIB,
            _named(reported, card, smaller=False),
            room / GIB,
        )
        return
    # What the memory outside torch's allocator grew since the limit in force was set, other
    # programs' or the run's own: the figure judged and what torch held then (_base), against
    # the same two now. Counted from the limit (_left + room - free), it fell short by what the
    # card had over the driver's figure where the limit was set under that one (a review).
    grown = _left + room - free if _base is None else sum(_base) - free - reserved
    named = _named(reported, card, smaller=False)
    short = f"the GPU has {_down(free)} GiB free {named} where the run may still take {_up(room)}"
    if round(given / GIB, 2):
        # Said where it gave anything to the hundredth: the figures are those judged after it.
        short += f", torch's cache of {given / GIB:.2f} GiB given back first"
    bound = ""
    if card is not None and _segments and _plan_s_limit is not None:
        limit = reserved + free - _left
        lost = _plan_s_limit - limit
        if free < _left:
            # Held again there, torch would stand under what it holds, every later check pass
            # unjudged and the bound go (a review, 2026-10-16): the run stops (PROVISIONAL).
            bound = (
                f": the card has less than the {_left / GIB:.1f} GiB left to the rest of it, where"
                " torch held again would stand under what it holds"
            )
        elif lost > _reserve:
            bound = (
                f": torch held again to what it holds and the card's free memory less"
                f" {_left / GIB:.1f} would stand {_up(lost)} GiB under the limit set once the plan"
                f" was made, past the {_reserve / GIB:.1f} GiB that plan leaves torch's allocator"
                " beyond a step's peak"
            )
        else:
            fraction = min(limit / total, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, device)  # pyright: ignore[reportUnknownMemberType]
            _limit, _base = limit, (free, reserved)
            logger.warning(
                "%s: the memory outside torch's allocator, other programs' or the run's own, grew"
                " %s GiB on the card since torch's limit was set, past the %.1f GiB left to it;"
                " torch held again to %.2f GiB, what it holds and the card's free memory less"
                " %.1f: %s GiB under the limit set once the plan was made, within the %.1f GiB"
                " that plan leaves torch's allocator beyond a step's peak, and the run goes on,"
                " a step left that much less beyond its peak",
                short,
                _up(grown),
                _left / GIB,
                limit / GIB,
                _left / GIB,
                _up(lost),
                _reserve / GIB,
            )
            return
    where = "on the card" if card is not None else "by the driver's figure"
    raise torch.OutOfMemoryError(
        f"{short}: other programs hold the rest of the card's memory, or the run's own work took"
        f" more than the {_left / GIB:.1f} GiB left to it outside torch's allocator; the memory"
        f" outside that allocator grew {_up(grown)} GiB {where} since torch's limit was set"
        f"{bound}"
    )


def _figures(device: Any, again: bool = False) -> tuple[int, int, int | None] | None:
    """What a check judges (check): what the driver reports free to this process and the card's
    total (mem_get_info), and on the card's footing what the whole card has free (_twice; None
    on the driver's footing); None where NVML fails twice on the card's footing, said in a
    warning, the unit not checked; at debug level for the check's second reading (`again`),
    which then judges on its first."""
    import torch

    reported, total = torch.cuda.mem_get_info(device)
    if _footing != CARD:
        return reported, total, None
    read = _twice(_index(device))
    if not isinstance(read, int) and again:
        logger.debug(
            "the whole card's free memory couldn't be read through NVML again, its cache given"
            " back, asked twice (%s): judged on the first reading and what the cache gave back",
            read,
        )
        return None
    if not isinstance(read, int):
        # A warning at every check so skipped, each being a unit that nothing but torch's
        # limit holds. PROVISIONAL (implementation, 2026-10-12): design decided the second
        # reading and the warning, not how often a run whose NVML has gone for good repeats it.
        logger.warning(
            "the whole card's free memory couldn't be read through NVML before this unit,"
            " asked twice (%s): the unit isn't checked against the card, and runs under"
            " torch's limit alone; what the driver reports free to this process isn't taken"
            " in its place, which stands under the card's near a full card and would stop the"
            " run for nothing",
            read,
        )
        return None
    return reported, total, read


def _twice(index: int) -> int | str:
    """The whole card's free memory behind CUDA's device `index`, in bytes, as a held run reads it
    for its plan, its limits and its checks (reading, hold, check): read (_read), and read once
    more where NVML fails, said at debug level where the second reading answers (DESIGN.md,
    Allocator: NVML failing at a check "is asked again"; at the plan's reading as well, decided
    on 2026-10-13). Where it fails twice, both failures in words, "A; then B", for the caller's
    warning, which says what is done without the card's figure."""
    first = _read(index)
    if isinstance(first, int):
        return first
    second = _read(index)
    if isinstance(second, int):
        logger.debug(
            "the whole card's free memory wasn't read through NVML at first (%s): read again",
            first,
        )
        return second
    return f"{first}; then {second}"


def _limit_s_card(index: int, first: bool) -> int | None:
    """The whole card's free memory a limit is set under (hold), in bytes; None where it is set
    under the driver's figure alone. NVML isn't asked for the run's limit where the plan went by
    the driver's figure (_footing): the limit goes by it too. Else the card is read, twice where
    NVML fails (_twice); where it fails twice: for the first limit, before the first pass, None;
    for the run's, the card's figure the plan was made in (_planned), or None where no plan had
    it; each saying which figure the limit is set under, a warning but for the first limit,
    whose own line is said at debug level (hold)."""
    if not first and _footing == DRIVER:
        return None
    read = _twice(index)
    if isinstance(read, int):
        return read
    if first:
        # At debug level, as that limit's own line: the plan's reading asks again, and warns of
        # what the run does where NVML fails there too (reading).
        logger.debug(
            "the whole card's free memory couldn't be read through NVML as torch was held a first"
            " time, asked twice (%s): the first pass is held %.1f GiB under what the driver"
            " reports free to this process, which leaves other programs' memory out while the"
            " card has room",
            read,
            HELD_BACK_UNREAD / GIB,
        )
        return None
    if _planned is None:
        logger.warning(
            "as torch's limit was set, the whole card's free memory couldn't be read through NVML,"
            " asked twice (%s), nor when the plan was made: the limit is set %.1f GiB under what"
            " the driver reports free to this process, and the run's units are checked on that"
            " figure",
            read,
            HELD_BACK_UNREAD / GIB,
        )
        return None
    logger.warning(
        "as torch's limit was set, the whole card's free memory couldn't be read through NVML,"
        " asked twice (%s): the limit is set from the %.2f GiB the card had free when the plan"
        " was made, not from what the driver reports free to this process alone",
        read,
        _planned / GIB,
    )
    return _planned


def _down(count: int) -> str:
    """Bytes in GiB, to 2 decimals, rounded down: what is free (check)."""
    return f"{count * 100 // GIB / 100:.2f}"


def _up(count: int) -> str:
    """Bytes in GiB, to 2 decimals, rounded up: what the run may still take (check)."""
    return f"{-(-count * 100 // GIB) / 100:.2f}"


def _held_back(card: int | None) -> int:
    """What a held run leaves of the free memory its plan or its limit took, outside torch's
    limit, in bytes: HELD_BACK where the whole card's figure was read (`card`), whichever of the
    two figures is the smaller; HELD_BACK_UNREAD under the driver's figure alone, where it
    wasn't."""
    return HELD_BACK_UNREAD if card is None else HELD_BACK


def _named(reported: int, card: int | None, smaller: bool = True) -> str:
    """Which of the two figures a line has just given as "N GiB free", in words, with the other
    figure: the smaller of the two, as the plan and the limit take it (the plan's line, the
    limit's); or, with `smaller` False, the card's wherever it is read (the check's)."""
    if card is None:
        return "as its driver reports it to this process (the whole card's, NVML's, wasn't read)"
    if card <= reported or not smaller:
        return (
            f"on the whole card by NVML (its driver reports {reported / GIB:.2f} GiB free to this"
            " process)"
        )
    return (
        "as its driver reports it to this process (the whole card has"
        f" {card / GIB:.2f} GiB free by NVML)"
    )


def _index(device: Any) -> int:
    """CUDA's index of `device`, a torch.device or what makes one. One that names no index (its
    index None, whatever torch's annotation says) is CUDA's current device, which seedvr2x never
    moves from the first: 0, without asking torch, whose answer would make a context."""
    import torch

    return torch.device(device).index or 0


def card_free(index: int = 0) -> int | None:
    """The memory free on the whole card behind CUDA's device `index`, in bytes, whichever
    programs hold the rest: NVML's figure, the one nvidia-smi prints. None when NVML doesn't
    say: no library, a call that fails, a card it doesn't know; the reason at debug level.
    Never raises. One reading, nothing remembered (research/scripts/hold_probe.py reads the card
    so): a held run reads it through _twice, which asks again where NVML fails, and says what
    the run does without the card's figure (reading, hold, check).

    The card is CUDA's, found in NVML by its identifier (cuDeviceGetUuid, whose 16 bytes NVML
    writes "GPU-" and 8-4-4-4-12 hexadecimal digits), never by its index: CUDA numbers the
    devices CUDA_VISIBLE_DEVICES leaves it, fastest first unless CUDA_DEVICE_ORDER says
    otherwise (NVIDIA's CUDA documentation, Environment variables), NVML all of them by PCI
    address. CUDA's calls here need no context and make none.

    NVML is asked for its nvmlMemory_v2_t, which counts what the driver keeps of the card
    (reserved) apart from what programs hold, as nvidia-smi prints them; its first structure,
    nvmlMemory_t, counts both as used. Measured on an RTX 5090 under WSL2 (driver 616.92,
    2026-10-12), a process's bare context on the card: 32,607 MiB in all, 420 reserved,
    3,289.12 used and 28,897.88 free, nvidia-smi's four figures; by the first structure the
    same total and the same free memory, with 3,709.12 used. The free memory is the same in
    both there; the second is taken as the one that names the driver's share. A reading took
    10 ms (9 to 23 over three runs of 200), NVML initialised and shut down in it and no
    descriptor left open; a process's first 0.1 s, the two libraries loading in it."""
    found = _read(index)
    if isinstance(found, int):
        return found
    logger.debug("the whole card's free memory can't be read through NVML (%s)", found)
    return None


def _read(index: int) -> int | str:
    """card_free's reading: the bytes free on the card behind CUDA's device `index`, or why
    NVML gave none, in words. NVML is loaded first, so that a machine without it asks CUDA
    nothing; initialised only once CUDA has named the card, and shut down whatever follows."""
    try:
        nvml, cuda = ctypes.CDLL(NVML), ctypes.CDLL(CUDA)
        device, identifier = ctypes.c_int(), (ctypes.c_ubyte * 16)()
        handle = ctypes.c_void_p()
        memory = _Memory(version=ctypes.sizeof(_Memory) | 2 << 24)
        steps = (
            ("cuInit", lambda: cuda.cuInit(0)),
            ("cuDeviceGet", lambda: cuda.cuDeviceGet(ctypes.byref(device), index)),
            ("cuDeviceGetUuid", lambda: cuda.cuDeviceGetUuid_v2(ctypes.byref(identifier), device)),
            ("nvmlInit", lambda: nvml.nvmlInit_v2()),
        )
        for name, step in steps:
            status: int = step()
            if status != SUCCESS:
                return f"{name}: {status}"
        try:
            named = f"GPU-{uuid.UUID(bytes=bytes(identifier))}"
            status = nvml.nvmlDeviceGetHandleByUUID(named.encode(), ctypes.byref(handle))
            if status != SUCCESS:
                return f"nvmlDeviceGetHandleByUUID, {named}: {status}"
            status = nvml.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(memory))
            if status != SUCCESS:
                return f"nvmlDeviceGetMemoryInfo: {status}"
            return int(memory.free)
        finally:
            nvml.nvmlShutdown()
    except (OSError, AttributeError) as error:
        # No such library, or one without a function asked of it (ctypes: AttributeError).
        return str(error)


def paravirtual() -> bool:
    """Whether the GPU is reached through WSL2's device: Windows' driver under a Linux process."""
    return os.path.exists(PARAVIRTUAL)


def probe(index: int = 0, library: str = CUDA) -> Driver | None:
    """Ask the driver of GPU `index` for OVER more than it has free, in a context of the
    probe's own, and free it at once: Driver says whether it was given. None, with a warning,
    when the driver can't be asked (no CUDA library, no such device) or answers neither yes
    nor no: behind WSL2's device, startup then takes it for one that spills.

    A driver that refuses is asked nothing it can give. One that spills commits the card's free
    memory and OVER of system memory for an instant, untouched: a desktop on that card may
    blink. Measured on an RTX 5090 under WSL2 (driver 616.92, 2026-10-11): 0.85 and 0.99 s, the
    probe's own context made and destroyed in it; 30.45 GiB asked with 30.20 free, given, with
    the policy at "Prefer No Sysmem Fallback" and at "Prefer Sysmem Fallback" alike.

    Called before torch creates its own context, so that nothing of this process's is on the
    card yet; another process freeing memory between the reading and the request could make
    a refusing driver give: a request given counts only if the card has no more free after it
    than before it, and is asked once more otherwise.

    The driver isn't asked where its own figure answers: where it reports more free to this
    process than the whole card has by NVML (card_free), by more than OVER, it leaves the other
    programs' memory out of what it offers, and is taken for one that spills (Driver.card). A
    request there takes the card from them: beside a program holding 8.6 GiB, 30.45 GiB asked
    with 20.7 really free put 28.9 GiB of the card in use for an instant (nvidia-smi), the
    other program's and the desktop's memory left the card and came back over the next minute,
    0.30 GiB of it after the run had set its limit, and the run was stopped at its first unit
    (check; the same RTX 5090, 2026-10-12)."""
    try:
        cuda = ctypes.CDLL(library)
    except OSError as error:
        logger.warning("the GPU's driver can't be asked whether it refuses: %s", error)
        return None
    device, context = ctypes.c_int(), ctypes.c_void_p()
    steps = (
        ("cuInit", lambda: cuda.cuInit(0)),
        ("cuDeviceGet", lambda: cuda.cuDeviceGet(ctypes.byref(device), index)),
        ("cuCtxCreate", lambda: cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)),
    )
    for name, step in steps:
        status: int = step()
        if status != SUCCESS:
            logger.warning(
                "the GPU's driver can't be asked whether it refuses: %s: %d", name, status
            )
            return None
    try:
        told = _told(cuda, index)
        if told is not None:
            return told
        for _ in range(2):
            answer, found = _ask(cuda)
            if answer:
                return found
        logger.warning(
            "the GPU's driver gave more than was free twice, the card having room each time:"
            " another process is freeing memory there, and whether the driver refuses isn't known"
        )
        return None
    finally:
        cuda.cuCtxDestroy_v2(context)


def _told(cuda: ctypes.CDLL, index: int) -> Driver | None:
    """What the driver's own figure says of it, in the current context, without a request: one
    that spills, when it reports more free to this process than the whole card behind CUDA's
    device `index` has by NVML, by more than OVER (probe); None where NVML doesn't answer or
    the two agree that far, and the driver is then asked."""
    card = _read(index)
    if not isinstance(card, int):
        return None  # said where the run first plans in the driver's figure alone (card_free)
    now, total = ctypes.c_size_t(), ctypes.c_size_t()
    status: int = cuda.cuMemGetInfo_v2(ctypes.byref(now), ctypes.byref(total))
    if status != SUCCESS or now.value <= card + OVER:
        return None
    return Driver(True, 0, now.value, total.value, card)


def _ask(cuda: ctypes.CDLL) -> tuple[bool, Driver | None]:
    """One request in the current context, OVER more than is free, freed at once: whether it
    answers the question, and the answer. Refused: a driver that refuses. Given, the card
    having no more free once it is given back than before it (within OVER): one that spills.
    Given, the card having OVER more free after than before: it had room after all, another
    process having freed some meanwhile: no answer. A failure of the driver's calls: an
    answer, None."""

    def free() -> tuple[int, int] | None:
        now, total = ctypes.c_size_t(), ctypes.c_size_t()
        status: int = cuda.cuMemGetInfo_v2(ctypes.byref(now), ctypes.byref(total))
        if status != SUCCESS:
            logger.warning(
                "the GPU's driver can't be asked whether it refuses: cuMemGetInfo: %d", status
            )
            return None
        return now.value, total.value

    before = free()
    if before is None:
        return True, None
    asked = before[0] + OVER
    pointer = ctypes.c_uint64()
    status: int = cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(asked))
    if status == OUT_OF_MEMORY:
        return True, Driver(False, asked, *before)
    if status != SUCCESS:
        logger.warning("the GPU's driver can't be asked whether it refuses: cuMemAlloc: %d", status)
        return True, None
    cuda.cuMemFree_v2(pointer)
    after = free()
    if after is None:
        return True, None
    if after[0] >= asked:
        return False, None
    return True, Driver(True, asked, *before)
