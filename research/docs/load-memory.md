# The load's peak in system memory: the weights' file, mapped

> Status: **measured** (2026-10-12) for the open question "The load's peak in system memory" of
> [DESIGN.md](../../seedvr2x/DESIGN.md#to-measure): a job's resident memory peaked at 16 to
> 17 GiB while its weights loaded, where its plan counted 9 to 11. Through
> [`scripts/load_peak.py`](../scripts/load_peak.py), on seedvr2x's own loader
> (`runtime/model.py`, `runtime/blockswap.py`, at `0393413`) with seedvr2x's sharp 7B in fp16
> (15.35 GiB) and its VAE (0.47 GiB), torch 2.14.1+cu130, safetensors 0.8.0. Two machines: an
> RTX 5090 desktop under WSL2 (kernel 6.6, 39 GiB given to WSL), the weights on an NVMe drive of
> Windows', reached through WSL's 9p file bridge; and an RTX PRO 6000 Blackwell 96 GB under Linux
> (377 GiB), the weights on xfs on NVMe. Three runs of every row unless its count says otherwise.

In short:

- **The peak is the DiT's file, mapped.** safetensors gives a file's tensors as views of a
  private mapping of the file; the loader copies each tensor bound for the GPU from it, and a
  page read stays in the process until the mapping goes, at the end of the file's read. With
  every weight bound for the GPU the process peaks at 15.81 to 16.46 GiB resident, 15.00 to
  15.65 of it file-backed, and holds 1.18 once loaded (16.53 and 1.25 on the other machine).
- **It is the same whatever the plan pins,** which is why a plan's count never met it: the
  pinned copies, read straight from the file since 2026-10-11, take the mapped pages' place. 14
  blocks and the VAE pinned (6.38 GiB): 15.88 to 16.69 GiB, 6.86 anonymous and 8.94 to 9.74
  file-backed; the 36 blocks pinned (15.19): 16.11 to 16.28, nearly all anonymous. Nothing of it
  reads as locked: `Locked`, `VmLck` and `VmPin` stay at 0 on both machines, CUDA's registration
  of the pinned copies showing in none of them.
- **Read at the file's offsets into one buffer of the largest tensor's size, the peak goes:**
  1.08 GiB with nothing pinned (1.14 on the other machine), 7.38 to 7.44 with 6.38 pinned, the
  pinned copies and the process. Every tensor has the same SHA-256 as today's load, on both
  machines, 1,450 tensors a run.
- **It costs no time where a file reads slowly, and a fraction of a second where it reads
  fast:** through 9p, 70.5 to 77.3 s against 86.2 to 92.9 mapped; from xfs on NVMe, 3.4 to 3.6 s
  against 3.0 with the page cache cold, and 1.9 to 2.6 against 2.1 to 2.2 warm.
- **Giving the mapped pages back as they are read (`madvise`) removes the process's peak too,**
  but not the page cache's: under WSL the kernel still holds the file's 15.4 to 15.5 GiB until
  the mapping goes, where a read through 9p leaves none in it (0.1 to 0.2 GiB).
- **A group held to 16 GiB loads today's way all the same** where the plan fits it: the mapped
  pages are the file's, clean, and the kernel takes them back ([below](#a-host-held-to-16-gib)).
- **Through 9p, reads at once nearly halve the load:** 40.6 to 45.4 s in four pieces for 70.5 to
  77.3, two giving as much. On NVMe four through one descriptor cost a second with the cache
  cold, and none through a descriptor each ([below](#reading-in-pieces)).
- **Written into seedvr2x's loader,** the read gives the same tensors and the same frames with a
  load's peak of 1.07 GiB and nothing pinned, in 74.7 to 75.4 s on the desktop, and in 42.2 to
  43.3 s with its large tensors read in four pieces ([below](#the-loader-rewritten)).

## Method

`load_peak.py load` loads the two models as a job does (`memory.startup`, `memory.hold`,
`model.load_models`), with no plan or with a plan that pins weights in system memory, prints a
mark with the wall clock at each step, and then the SHA-256 of every parameter and buffer of the
two models, read back from wherever it is. `load_peak.py watch` runs it as a child and samples
it from outside: `/proc/PID/status` every 50 ms (`VmRSS`, `VmHWM`, `RssAnon`, `RssFile`,
`RssShmem`, `VmLck`, `VmPin`), the host's `/proc/meminfo` and the control group's memory files
with it, and `/proc/PID/smaps_rollup` once a second (`Locked`). The peak is the kernel's own
high-water mark, `VmHWM`; what it is made of is the sample with the most resident memory, which
a load of three seconds misses by up to 0.45 GiB (0.01 on the desktop's slower loads). Reading
`smaps_rollup` walks the process's pages, up to 100 ms a read with 16 GiB resident: sampled
every 50 ms it took most of a core, and the first run, made so, was set aside.

Three plans, each a `Plan` given to `load_models`: none (every weight bound for the GPU); 6
blocks swapped, 25 and 22 blocks kept on the GPU in the VAE's phases and the VAE away in the
DiT's, which pins 14 blocks and the VAE, 6.38 GiB (the plan a 1080p job got on the 32 GB card
with 29 GiB of it free); and the 36 blocks swapped, 15.19 GiB pinned.

`--reader` replaces the loader's two reads for the run (`safetensors.torch.load_file` for a
model with no part pinned, `blockswap.read` for one with parts), the product untouched. On the
desktop each run is one hold of the card, with whatever else the desktop was doing; on the other
machine each row ran with the two files dropped from the page cache first
(`POSIX_FADV_DONTNEED`, checked with `mincore`: 0 of 15.35 GiB held) and with them read into it
first. Under WSL the state needs no setting: a file read through 9p isn't kept in Linux's page
cache, and a mapped one leaves it when its mapping goes (0.000 of 15.35 GiB held before and
after every run, 134 readings).

## What the peak is made of

| machine | plan | runs | peak resident (GiB) | at the sampled peak: anonymous | file-backed | shared | resident once loaded |
|---|---|---|---|---|---|---|---|
| RTX 5090, WSL2, 9p | nothing pinned (the DiT whole on the GPU) | 3 | 15.81 to 16.46 | 0.48 | 15.00 to 15.65 | 0.33 | 1.18 |
| RTX 5090, WSL2, 9p | 14 blocks and the VAE pinned (6 blocks swapped) | 3 | 15.88 to 16.69 | 6.86 | 8.94 to 9.74 | 0.08 | 7.30 to 7.31 |
| RTX 5090, WSL2, 9p | the 36 blocks pinned (all swapped) | 3 | 16.11 to 16.28 | 15.68 to 15.86 | 0.30 to 0.32 | 0.11 | 15.88 to 16.16 |
| 96 GB card, xfs, cache cold | nothing pinned (the DiT whole on the GPU) | 3 | 16.53 | 0.49 | 15.46 to 15.63 | 0.34 | 1.25 |
| 96 GB card, xfs, cache cold | 14 blocks and the VAE pinned (6 blocks swapped) | 3 | 16.75 to 16.76 | 6.87 | 9.47 to 9.72 | 0.09 | 7.38 |
| 96 GB card, xfs, cache cold | the 36 blocks pinned (all swapped) | 3 | 16.35 | 15.87 | 0.36 | 0.12 | 16.23 |
| 96 GB card, xfs, cache warm | nothing pinned (the DiT whole on the GPU) | 3 | 16.52 to 16.53 | 0.49 | 15.35 to 15.61 | 0.34 | 1.25 |
| 96 GB card, xfs, cache warm | 14 blocks and the VAE pinned (6 blocks swapped) | 3 | 16.75 to 16.76 | 6.87 | 9.36 to 9.61 | 0.09 | 7.38 |
| 96 GB card, xfs, cache warm | the 36 blocks pinned (all swapped) | 3 | 16.35 | 15.87 | 0.36 | 0.12 | 16.23 |

The file-backed part is the DiT's file as far as it was mapped and read, and the process's
libraries (0.3 GiB); the shared part is there before the first weight is read. On the desktop
two of today's loads peaked lower than the others, one with 14 blocks pinned and one with the
36: WSL dropped its page cache while they ran, and the pages the process had mapped went with
it, the file's and its libraries'. One run's steps on the desktop, nothing pinned:

| step | s | peak resident | anonymous | file-backed | at its end: resident | file-backed |
|---|---|---|---|---|---|---|
| Python and torch start, CUDA's context | 4.1 | 0.81 | 0.44 | 0.30 | 0.81 | 0.30 |
| the VAE is built (no weight yet) | 0.2 | 0.83 | 0.45 | 0.30 | 0.83 | 0.30 |
| the VAE's file is read (0.47 GiB) | 3.2 | 1.33 | 0.46 | 0.77 | 0.89 | 0.30 |
| the DiT is built | 0.1 | 0.89 | 0.48 | 0.30 | 0.89 | 0.30 |
| the DiT's file is read (15.35 GiB) | 87.5 | 16.46 | 0.48 | 15.65 | 1.11 | 0.30 |
| the load ends | 0.1 | 1.18 | 0.54 | 0.30 | 1.18 | 0.30 |
| loaded (one second idle) | 1.0 | 1.18 | 0.54 | 0.30 | 1.18 | 0.30 |

A quarter, half and three quarters into the DiT's read of 87 s, the process's file-backed memory
is 4.17, 8.05 and 12.06 GiB, its libraries' 0.30 in it; 15.65 at the read's end, and 0.30 a
sample later: the file's pages are given back in one go, when the mapping goes at the end of the
file's read. Until then nothing gives one back, though each was only read once, for its copy.

With 14 blocks and the VAE pinned, the pinned copies (anonymous) and the mapped blocks
(file-backed) add up to the same. The copies are whole in the process before the first byte is
read, page-locking faulting every page in (6.86 GiB anonymous a quarter into the read, 6.86 at
its end):

| step | s | peak resident | anonymous | file-backed | at its end: resident | file-backed |
|---|---|---|---|---|---|---|
| Python and torch start, CUDA's context | 4.1 | 0.81 | 0.44 | 0.30 | 0.81 | 0.30 |
| the VAE is built (no weight yet) | 0.2 | 0.83 | 0.45 | 0.30 | 0.83 | 0.30 |
| the VAE's file is read (0.47 GiB) | 3.0 | 1.78 | 0.94 | 0.76 | 1.33 | 0.30 |
| the DiT is built | 0.1 | 1.33 | 0.95 | 0.30 | 1.33 | 0.30 |
| the DiT's file is read (15.35 GiB) | 82.0 | 16.68 | 6.86 | 9.74 | 7.24 | 0.30 |
| the load ends | 0.1 | 7.29 | 6.90 | 0.30 | 7.29 | 0.30 |
| loaded (one second idle) | 1.0 | 7.30 | 6.92 | 0.30 | 7.30 | 0.30 |

## The same tensors, read another way

Each way loads the same bytes: a digest per tensor, 1,450 a run, against today's first load.
Nothing pinned, the desktop (9p):

| the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|
| as loaded today (mapped) | 3 | 15.81 to 16.46 | 1.18 | 14.6 to 15.7 | 86.2 to 92.9 | 90.6 | 0, 0, 0 |
| mapped, pages given back (`MADV_DONTNEED`) | 3 | 1.06 to 1.07 | 0.93 | 15.4 to 15.5 | 84.0 to 90.8 | 88.5 | 0, 0, 0 |
| the same, and dropped from the page cache | 3 | 1.06 | 0.93 | 0.2 to 0.3 | 80.1 to 84.4 | 81.6 | 0, 0, 0 |
| safetensors' own `backend="pread"` | 3 | 1.31 | 1.18 | 0.3 | 68.7 to 74.4 | 74.1 | 0, 0, 0 |
| read at its offsets, one buffer | 3 | 1.08 | 0.94 | 0.1 to 0.2 | 70.5 to 77.3 | 73.9 | 0, 0, 0 |
| the same, the buffer page-locked | 3 | 1.06 | 0.93 | 0.1 to 0.2 | 69.5 to 74.7 | 73.7 | 0, 0, 0 |
| read, and dropped from the page cache | 3 | 1.07 to 1.08 | 0.94 to 0.95 | 0.1 | 71.1 to 73.4 | 72.1 | 0, 0, 0 |

The other machine (xfs on NVMe), the page cache cold:

| the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|
| as loaded today (mapped) | 3 | 16.53 | 1.25 | 16.1 | 3.0 | 3.0 | 0, 0, 0 |
| mapped, pages given back (`MADV_DONTNEED`) | 3 | 1.12 | 1.00 | 15.8 | 2.4 to 2.6 | 2.5 | 0, 0, 0 |
| the same, and dropped from the page cache | 3 | 1.12 | 1.00 | 0.2 to 0.6 | 2.4 to 2.5 | 2.5 | 0, 0, 0 |
| safetensors' own `backend="pread"` | 3 | 1.37 | 1.25 | 16.1 | 3.1 to 3.3 | 3.1 | 0, 0, 0 |
| read at its offsets, one buffer | 3 | 1.14 | 1.01 | 15.8 to 15.9 | 3.4 to 3.6 | 3.5 | 0, 0, 0 |
| the same, the buffer page-locked | 3 | 1.12 | 1.00 | 15.8 to 16.0 | 3.1 to 3.2 | 3.1 | 0, 0, 0 |
| read, and dropped from the page cache | 3 | 1.14 | 1.01 | 0.1 | 3.5 to 3.6 | 3.5 | 0, 0, 0 |

and warm:

| the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|
| as loaded today (mapped) | 3 | 16.52 to 16.53 | 1.25 | 0.3 | 2.1 to 2.2 | 2.2 | 0, 0, 0 |
| mapped, pages given back (`MADV_DONTNEED`) | 3 | 1.12 | 1.00 | 0.0 to 0.1 | 1.7 to 1.8 | 1.7 | 0, 0, 0 |
| the same, and dropped from the page cache | 3 | 1.12 | 1.00 | 0.0 | 2.2 | 2.2 | 0, 0, 0 |
| safetensors' own `backend="pread"` | 3 | 1.37 | 1.25 | 0.3 to 0.4 | 1.6 to 2.2 | 1.6 | 0, 0, 0 |
| read at its offsets, one buffer | 3 | 1.14 | 1.01 | 0.0 to 0.1 | 1.9 to 2.6 | 1.9 | 0, 0, 0 |
| the same, the buffer page-locked | 3 | 1.12 | 1.00 | 0.0 to 0.2 | 1.4 to 2.0 | 1.5 | 0, 0, 0 |
| read, and dropped from the page cache | 3 | 1.14 | 1.01 | 0.0 | 1.8 to 2.7 | 1.9 | 0, 0, 0 |

With 14 blocks and the VAE pinned (the desktop; then the other machine, cold and warm):

| the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|
| as loaded today (mapped) | 3 | 15.88 to 16.69 | 7.30 to 7.31 | 8.6 to 9.6 | 84.2 to 87.3 | 84.3 | 0, 0, 0 |
| mapped, pages given back (`MADV_DONTNEED`) | 3 | 7.43 | 7.30 to 7.31 | 9.5 to 10.0 | 85.1 to 90.5 | 87.3 | 0, 0, 0 |
| safetensors' own `backend="pread"` | 3 | 7.45 | 7.32 | 0.1 to 0.4 | 80.0 to 80.7 | 80.1 | 0, 0, 0 |
| read at its offsets, one buffer | 3 | 7.38 to 7.44 | 7.26 to 7.31 | 0.1 to 0.2 | 75.5 to 80.6 | 76.8 | 0, 0, 0 |

| cache | the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|
| cold | as loaded today (mapped) | 3 | 16.75 to 16.76 | 7.38 | 15.8 to 15.9 | 3.5 | 3.5 | 0, 0, 0 |
| cold | mapped, pages given back (`MADV_DONTNEED`) | 3 | 7.50 | 7.38 | 15.8 to 15.9 | 3.3 to 3.5 | 3.4 | 0, 0, 0 |
| cold | safetensors' own `backend="pread"` | 3 | 7.52 | 7.38 | 15.8 to 15.9 | 6.3 to 6.9 | 6.7 | 0, 0, 0 |
| cold | read at its offsets, one buffer | 3 | 7.50 | 7.38 | 15.8 to 15.9 | 4.0 to 4.2 | 4.1 | 0, 0, 0 |
| cold | the same, the buffer page-locked | 3 | 7.50 | 7.38 | 15.8 to 15.9 | 3.8 to 3.9 | 3.9 | 0, 0, 0 |
| warm | as loaded today (mapped) | 3 | 16.75 to 16.76 | 7.38 | 0.0 to 0.2 | 1.9 to 2.4 | 2.4 | 0, 0, 0 |
| warm | mapped, pages given back (`MADV_DONTNEED`) | 3 | 7.50 | 7.38 | 0.0 to 0.4 | 1.9 to 2.4 | 2.0 | 0, 0, 0 |
| warm | safetensors' own `backend="pread"` | 3 | 7.52 | 7.38 | 0.0 to 0.2 | 4.5 to 7.0 | 4.6 | 0, 0, 0 |
| warm | read at its offsets, one buffer | 3 | 7.50 | 7.38 | 0.0 to 0.3 | 2.4 to 2.9 | 2.4 | 0, 0, 0 |
| warm | the same, the buffer page-locked | 3 | 7.50 | 7.38 | 0.0 to 0.3 | 2.1 to 2.7 | 2.1 | 0, 0, 0 |

With the 36 blocks pinned nearly every byte goes straight to a pinned copy already, and the ways
don't differ (the desktop: 16.11 to 16.28 GiB and 80.7 to 87.9 s as loaded today, 16.07 to 16.26
and 85.7 to 87.2 read).

- **Mapped, with each tensor's pages given back once it is copied** (`madvise` `MADV_DONTNEED`
  on its span of safetensors' mapping): the process's peak goes, and on xfs it is the fastest
  way. The pages stay in the page cache, which is where the kernel keeps a file on xfs whichever
  way it is read; under WSL it is 15 GiB the kernel holds for the length of the load that a read
  doesn't. It reaches into another library's mapping by its address.
- **safetensors' own `backend="pread"`** removes the peak in one argument for a model loaded
  whole, and leaves 1.18 GiB resident once loaded for 0.94. With parts it reads a tensor's bytes
  when the tensor is asked for, where the mapped file only gave a view: the loader's first pass
  over the file, for the parts' layout, has to take shapes from the header instead. Tried so
  here, the parts read as today and the other tensors through its `get_tensor`, a cold load on
  NVMe took 6.3 to 6.9 s for 3.5.
- **Read at the file's own offsets** (`pread`), as the loader reads a part into its pinned copy
  since 2026-10-11: a tensor bound for the GPU goes into one buffer of the largest tensor's size
  (108 MiB for the 7B) and is copied to the GPU from it before the next is read. No mapping, no
  safetensors at the load. With the buffer page-locked a cold load on NVMe takes 3.1 to 3.2 s
  for 3.4 to 3.6, and a load that pins nothing then depends on CUDA's registration.
- **Dropping each span from the page cache once read** (`POSIX_FADV_DONTNEED`) keeps the host's
  cache from growing where the file system has one, for the next load's time: it would read the
  disk again.

## A host held to 16 GiB

The desktop's WSL lets a user set a limit on a group of processes (`systemd-run --user --scope
-p MemoryMax=16G -p MemorySwapMax=0`, cgroup v2); the other machine doesn't: no user manager
runs there, and the session's control group isn't the user's to divide. Each run is the sampler
and the load inside such a scope; the group's figures are the kernel's (`memory.peak`,
`memory.events`).

| plan | the file's tensors | runs | exit | the limit met (times) | the group's oom_kill | the group's memory at the most (GiB) | the process's peak resident | load (s) | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|---|
| nothing pinned (the DiT whole on the GPU) | as loaded today (mapped) | 3 | 0, 0, 0 | 2074, 2015, 2073 | 0, 0, 0 | 16.00 | 16.21 to 16.22 | 89.0 to 92.0 | 0, 0, 0 |
| nothing pinned (the DiT whole on the GPU) | read at its offsets, one buffer | 3 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0.78 to 0.81 | 1.07 to 1.08 | 72.9 to 75.1 | 0, 0, 0 |
| nothing pinned (the DiT whole on the GPU) | the loader rewritten | 1 | 0 | 0 | 0 | 0.78 | 1.07 | 74.0 | 0 |
| 14 blocks and the VAE pinned (6 blocks swapped) | as loaded today (mapped) | 3 | 0, 0, 0 | 726, 3776, 3842 | 0, 0, 0 | 16.00 | 16.23 | 86.2 to 89.0 | 0, 0, 0 |
| 14 blocks and the VAE pinned (6 blocks swapped) | read at its offsets, one buffer | 3 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 7.15 to 7.21 | 7.43 to 7.44 | 78.2 to 82.0 | 0, 0, 0 |
| 14 blocks and the VAE pinned (6 blocks swapped) | the loader rewritten | 1 | 0 | 0 | 0 | 7.17 | 7.43 | 77.0 | 0 |
| nothing pinned (the DiT whole on the GPU) | the same, large tensors in 4 pieces at once | 1 | 0 | 0 | 0 | 0.78 | 1.07 | 43.4 | 0 |
| 14 blocks and the VAE pinned (6 blocks swapped) | the same, large tensors in 4 pieces at once | 1 | 0 | 0 | 0 | 7.16 | 7.44 | 45.9 | 0 |
| the 36 blocks pinned (all swapped) | as loaded today (mapped) | 2 | -9, -9 | 1472, 1488 | 1, 1 | 16.00 | 16.14 to 16.25 | 90.1 to 99.4 | -, - |
| the 36 blocks pinned (all swapped) | read at its offsets, one buffer | 3 | 0, -9, -9 | 9, 438, 1062 | 0, 1, 1 | 15.99 to 16.00 | 16.07 to 16.26 | 86.6 to 93.2 | 0, -, - |

With nothing pinned, or 6.38 GiB, and loaded as today, the group sits at its limit for the
length of the load and the load ends in its usual time: a mapped page is the file's own, clean,
and the kernel takes it back as soon as it is asked for memory, the limit met thousands of times
a load and no process killed. Read, the group never comes near it. Under smaller limits, one run
each, to find where a load stops fitting:

| the group's limit | plan | the file's tensors | runs | exit | the limit met (times) | the group's oom_kill | the group's memory at the most (GiB) | the process's peak resident | load (s) | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 GiB | 14 blocks and the VAE pinned (6 blocks swapped) | as loaded today (mapped) | 1 | 0 | 36654 | 0 | 12.00 | 12.25 | 86.3 | 0 |
| 12 GiB | 14 blocks and the VAE pinned (6 blocks swapped) | read at its offsets, one buffer | 1 | 0 | 0 | 0 | 7.17 | 7.44 | 78.3 | 0 |
| 8 GiB | 14 blocks and the VAE pinned (6 blocks swapped) | as loaded today (mapped) | 1 | 0 | 64619 | 0 | 8.00 | 8.26 | 86.9 | 0 |
| 8 GiB | 14 blocks and the VAE pinned (6 blocks swapped) | read at its offsets, one buffer | 1 | 0 | 0 | 0 | 7.17 | 7.44 | 76.8 | 0 |
| 4 GiB | nothing pinned (the DiT whole on the GPU) | as loaded today (mapped) | 1 | 0 | 100294 | 0 | 4.00 | 4.27 | 84.9 | 0 |
| 4 GiB | nothing pinned (the DiT whole on the GPU) | read at its offsets, one buffer | 1 | 0 | 0 | 0 | 0.79 | 1.08 | 74.0 | 0 |
| 2 GiB | nothing pinned (the DiT whole on the GPU) | as loaded today (mapped) | 1 | 0 | 116659 | 0 | 2.00 | 2.28 | 92.5 | 0 |
| 2 GiB | nothing pinned (the DiT whole on the GPU) | read at its offsets, one buffer | 1 | 0 | 0 | 0 | 0.79 | 1.08 | 72.2 | 0 |

A load fits any limit that holds what it leaves, the process and its pinned copies, and takes no
longer for it. What doesn't fit is the anonymous part. With the 36 blocks pinned, 15.19 GiB and
the process pass 16 GiB by themselves, whichever way the file is read, and there the kernel
refuses nothing: the load ends, the group at its limit, and the kernel's OOM killer takes the
process at one of its next allocations, in this probe's own digests after the load (the rows
above: exit -9, `oom_kill` 1, four runs; a first run loaded as today ended "Terminated" before
the sampler could say more, the system's journal naming the OOM killer; and one of the three
read passed). The page-locked copies can't be taken back. In 17 GiB both ways load:

| plan | the file's tensors | runs | exit | the limit met (times) | the group's oom_kill | the group's memory at the most (GiB) | the process's peak resident | load (s) | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|---|
| the 36 blocks pinned (all swapped) | as loaded today (mapped) | 1 | 0 | 0 | 0 | 16.42 | 16.11 | 85.9 | 0 |
| the 36 blocks pinned (all swapped) | read at its offsets, one buffer | 1 | 0 | 0 | 0 | 16.24 | 16.06 | 88.7 | 0 |

seedvr2x's plan counts the pinned copies before the load, and refuses such a job in words.

## Reading in pieces

WSL's file bridge serves one read of the DiT's file at 227 to 250 MB/s from an NVMe drive, in
requests of 64 KiB (`msize=65536`). `--threads N` reads each tensor's span of 4 MiB or more in N
pieces at once, each by `pread` at its own offset, through the one descriptor. The desktop,
nothing pinned (393 to 438 MB/s in 4 pieces):

| the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|
| read at its offsets, one buffer | 3 | 1.08 | 0.94 | 0.1 to 0.2 | 70.5 to 77.3 | 73.9 | 0, 0, 0 |
| read in 2 pieces at once | 1 | 1.08 | 0.94 | 0.1 | 43.2 | 43.2 | 0 |
| read in 4 pieces at once | 3 | 1.07 to 1.08 | 0.94 | 0.1 | 40.6 to 45.4 | 43.1 | 0, 0, 0 |
| read in 8 pieces at once | 2 | 1.07 | 0.94 | 0.1 | 40.6 to 41.1 | 40.8 | 0, 0 |
| read in 16 pieces at once | 1 | 1.08 | 0.94 | 0.1 | 42.9 | 42.9 | 0 |

The other machine, where one stream reads at 5.1 to 5.4 GB/s with the page cache cold:

| cache | the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|
| cold | read at its offsets, one buffer | 3 | 1.14 | 1.01 | 15.8 to 15.9 | 3.4 to 3.6 | 3.5 | 0, 0, 0 |
| cold | read in 4 pieces at once | 3 | 1.14 | 1.01 | 15.8 to 15.9 | 4.5 to 4.6 | 4.6 | 0, 0, 0 |
| warm | read at its offsets, one buffer | 3 | 1.14 | 1.01 | 0.0 to 0.1 | 1.9 to 2.6 | 1.9 | 0, 0, 0 |
| warm | read in 4 pieces at once | 3 | 1.14 | 1.01 | 0.0 to 0.1 | 1.6 to 1.9 | 1.7 | 0, 0, 0 |
| cold, 36 pinned | read at its offsets, one buffer | 3 | 16.33 | 16.20 | 15.8 | 4.5 to 4.8 | 4.7 | 0, 0, 0 |
| cold, 36 pinned | read in 4 pieces at once | 3 | 16.33 | 16.20 | 15.8 to 15.9 | 5.6 to 5.9 | 5.8 | 0, 0, 0 |
| warm, 36 pinned | read at its offsets, one buffer | 3 | 16.33 | 16.20 | 0.0 to 0.1 | 3.1 to 3.7 | 3.1 | 0, 0, 0 |
| warm, 36 pinned | read in 4 pieces at once | 3 | 16.33 | 16.20 | 0.0 to 0.1 | 3.0 to 3.3 | 3.0 | 0, 0, 0 |

There four pieces through one descriptor cost a second on a cold file. The kernel keeps its
read-ahead's state by open file, and four readers of one descriptor break each other's run:
`load_peak.py read` makes the same reads into one buffer, no torch and no GPU, through the one
descriptor or through one a thread. The other machine:

| pieces at once | descriptors | page cache | runs | seconds | MB/s |
|---|---|---|---|---|---|
| 1 | - | cold | 3 | 2.47 to 2.55 | 6455 to 6660 |
| 2 | one each | cold | 3 | 2.65 to 2.70 | 6115 to 6215 |
| 4 | one for all | cold | 3 | 3.31 to 3.38 | 4875 to 4973 |
| 4 | one each | cold | 3 | 2.49 to 2.55 | 6453 to 6612 |
| 8 | one each | cold | 3 | 1.85 to 1.87 | 8804 to 8926 |
| 1 | - | warm | 3 | 0.91 to 0.95 | 17399 to 18157 |
| 4 | one for all | warm | 3 | 0.50 to 0.54 | 30659 to 32887 |
| 4 | one each | warm | 3 | 0.54 to 0.57 | 29100 to 30798 |
| 8 | one each | warm | 3 | 0.37 to 0.41 | 40255 to 44984 |

and the desktop, through 9p, where Linux's page cache never holds the file and two reads at once
already give all the bridge has:

| pieces at once | descriptors | page cache | runs | seconds | MB/s |
|---|---|---|---|---|---|
| 1 | - | cold | 3 | 65.1 to 65.2 | 253 |
| 2 | one each | cold | 3 | 34.2 to 34.4 | 479 to 481 |
| 4 | one for all | cold | 3 | 34.2 to 34.3 | 480 to 482 |
| 4 | one each | cold | 3 | 34.1 to 34.4 | 479 to 483 |
| 8 | one each | cold | 3 | 34.3 to 34.6 | 477 to 481 |

## A whole job against its plan's count

Whole jobs sampled the same way from their first line to their last (66 frames in two shots, on
the desktop): what the plan counts of system memory (its sixth line, and its manifest's
`plan.memory.host`), and what the process holds at the load and after it. The planner pins what
the card's free memory asks for, which moved with the desktop from one run to the next. The last
rows are the same jobs on the loader rewritten (below): their frames are the same, by the md5 of
every frame. One more run of the 432p job, and one of the last row's, were stopped after their
load by seedvr2x's own check of the card, the desktop having taken more of it meanwhile; each
was run again.

| job | runs | the plan counts (GiB) | its manifest's plan.memory.host | of it pinned | peak resident at the load | anonymous | file-backed | the most resident once loaded, to the job's end | less the plan's pinned copies, shot and scratch | models loaded in (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 432p, the planner's own plan | 3 | 6.21 | 6.20 | 3.80 | 16.02 to 16.55 | 4.47 | 11.44 to 11.97 | 6.14 | 2.33 | 86.2 to 87.6 |
| 1080p in tiles, the planner's own plan | 3 | 12.15 to 12.57 | 12.15 to 12.57 | 9.33 to 9.75 | 16.99 to 17.00 | 10.00 to 10.42 | 6.49 to 6.90 | 12.17 to 12.59 | 2.42 to 2.43 | 87.9 to 90.5 |
| 1080p in tiles, `--swap-blocks 36` | 3 | 18.01 | 18.01 | 15.19 | 16.56 to 16.57 | 15.87 | 0.58 to 0.59 | 18.06 to 18.09 | 2.45 to 2.48 | 80.7 to 82.6 |
| the loader rewritten: 432p | 1 | 6.21 | 6.20 | 3.80 | 5.07 | 4.58 | 0.41 | 6.11 | 2.30 | 75.7 |
| the loader rewritten: 1080p in tiles | 1 | 12.99 | 13.00 | 10.17 | 11.41 | 10.92 | 0.41 | 12.95 | 2.36 | 79.2 |
| the loader rewritten: 1080p, `--swap-blocks 36` | 1 | 18.01 | 18.01 | 15.19 | 16.47 | 15.97 | 0.41 | 18.04 | 2.43 | 82.3 |
| with the read in pieces: 1080p in tiles | 1 | 12.15 | 12.15 | 9.33 | 10.61 | 10.11 | 0.41 | 12.17 | 2.42 | 46.1 |
| with the read in pieces: 1080p, `--swap-blocks 36` | 1 | 18.01 | 18.01 | 15.19 | 16.47 | 15.97 | 0.41 | 18.04 | 2.43 | 49.8 |

## The loader rewritten

The read above written into seedvr2x's loader itself (every tensor read at the file's offsets, a
part's into its pinned copy and any other through one buffer; no mapping), and then with a
tensor of 4 MiB or more read in 4 pieces at once, each through a descriptor of its own: two
changes proposed to seedvr2x, each measured as today's loader is, its tensors' digests against
today's. The desktop:

| plan | the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|
| nothing pinned | as loaded today (mapped) | 3 | 15.81 to 16.46 | 1.18 | 14.6 to 15.7 | 86.2 to 92.9 | 90.6 | 0, 0, 0 |
| nothing pinned | the loader rewritten | 3 | 1.07 | 0.94 | 0.1 to 0.3 | 74.7 to 75.4 | 75.4 | 0, 0, 0 |
| nothing pinned | the same, large tensors in 4 pieces at once | 3 | 1.06 to 1.07 | 0.94 | 0.1 to 0.9 | 42.2 to 43.3 | 43.2 | 0, 0, 0 |
| 14 blocks and the VAE pinned | as loaded today (mapped) | 3 | 15.88 to 16.69 | 7.30 to 7.31 | 8.6 to 9.6 | 84.2 to 87.3 | 84.3 | 0, 0, 0 |
| 14 blocks and the VAE pinned | the loader rewritten | 3 | 7.43 to 7.44 | 7.31 to 7.41 | 0.1 | 76.2 to 79.9 | 76.4 | 0, 0, 0 |
| 14 blocks and the VAE pinned | the same, large tensors in 4 pieces at once | 3 | 7.44 | 7.31 | 0.1 | 44.4 to 45.2 | 44.9 | 0, 0, 0 |
| the 36 blocks pinned | as loaded today (mapped) | 3 | 16.11 to 16.28 | 15.88 to 16.16 | 0.6 to 0.8 | 80.7 to 87.9 | 83.8 | 0, 0, 0 |
| the 36 blocks pinned | the loader rewritten | 3 | 16.19 to 16.25 | 16.07 to 16.13 | 0.1 | 80.7 to 84.3 | 83.3 | 0, 0, 0 |
| the 36 blocks pinned | the same, large tensors in 4 pieces at once | 3 | 16.25 | 16.13 | 0.1 | 50.4 to 51.9 | 50.7 | 0, 0, 0 |

The other machine (the loader rewritten: an earlier cut of the same change):

| plan | cache | the file's tensors | runs | peak resident (GiB) | resident once loaded | the host's page cache grows by | load (s) | median | tensors unlike today's |
|---|---|---|---|---|---|---|---|---|---|
| nothing pinned | cold | as loaded today (mapped) | 3 | 16.53 | 1.25 | 16.1 | 3.0 | 3.0 | 0, 0, 0 |
| nothing pinned | cold | the loader rewritten | 3 | 1.13 | 1.01 | 15.9 | 3.6 to 3.8 | 3.7 | 0, 0, 0 |
| nothing pinned | cold | the same, large tensors in 4 pieces at once | 3 | 1.13 to 1.14 | 1.01 | 15.8 to 16.0 | 3.7 to 4.0 | 3.9 | 0, 0, 0 |
| nothing pinned | warm | as loaded today (mapped) | 3 | 16.52 to 16.53 | 1.25 | 0.3 | 2.1 to 2.2 | 2.2 | 0, 0, 0 |
| nothing pinned | warm | the loader rewritten | 3 | 1.13 | 1.01 | 0.0 to 0.3 | 1.9 to 2.4 | 2.0 | 0, 0, 0 |
| nothing pinned | warm | the same, large tensors in 4 pieces at once | 3 | 1.13 | 1.01 | 0.0 to 0.1 | 1.7 | 1.7 | 0, 0, 0 |
| 14 blocks and the VAE pinned | cold | as loaded today (mapped) | 3 | 16.75 to 16.76 | 7.38 | 15.8 to 15.9 | 3.5 | 3.5 | 0, 0, 0 |
| 14 blocks and the VAE pinned | cold | the loader rewritten | 3 | 7.51 | 7.38 | 15.8 to 15.9 | 3.9 to 4.3 | 4.2 | 0, 0, 0 |
| 14 blocks and the VAE pinned | cold | the same, large tensors in 4 pieces at once | 3 | 7.51 | 7.38 | 15.8 to 15.9 | 4.2 to 5.6 | 4.4 | 0, 0, 0 |
| 14 blocks and the VAE pinned | warm | as loaded today (mapped) | 3 | 16.75 to 16.76 | 7.38 | 0.0 to 0.2 | 1.9 to 2.4 | 2.4 | 0, 0, 0 |
| 14 blocks and the VAE pinned | warm | the loader rewritten | 3 | 7.51 | 7.38 | 0.0 to 0.1 | 2.6 to 3.6 | 2.6 | 0, 0, 0 |
| 14 blocks and the VAE pinned | warm | the same, large tensors in 4 pieces at once | 3 | 7.51 | 7.38 | 0.0 to 0.2 | 2.1 to 2.4 | 2.2 | 0, 0, 0 |
| the 36 blocks pinned | cold | as loaded today (mapped) | 3 | 16.35 | 16.23 | 15.9 | 4.6 to 4.9 | 4.8 | 0, 0, 0 |
| the 36 blocks pinned | cold | the loader rewritten | 3 | 16.32 | 16.20 | 15.8 to 16.0 | 4.8 to 4.9 | 4.9 | 0, 0, 0 |
| the 36 blocks pinned | cold | the same, large tensors in 4 pieces at once | 3 | 16.32 | 16.20 | 15.9 | 4.9 to 6.4 | 5.1 | 0, 0, 0 |
| the 36 blocks pinned | warm | as loaded today (mapped) | 3 | 16.35 | 16.23 | 0.1 to 0.2 | 3.2 to 3.7 | 3.2 | 0, 0, 0 |
| the 36 blocks pinned | warm | the loader rewritten | 3 | 16.32 | 16.20 | 0.0 to 0.2 | 3.1 to 3.5 | 3.4 | 0, 0, 0 |
| the 36 blocks pinned | warm | the same, large tensors in 4 pieces at once | 3 | 16.32 | 16.20 | 0.0 to 0.1 | 2.7 to 3.0 | 2.9 | 0, 0, 0 |

## For seedvr2x

- The mapped load's peak was never a need: a load that leaves 7.30 to 7.31 GiB resident fits a
  group held to 8, and one that leaves 1.18 a group held to 2. It is what the process shows
  while it loads, 16 to 17 GiB whatever the plan, and what a plan can't count (6.21, 12.15 to
  12.57 and 18.01 GiB counted by the three jobs above); under WSL it is also 15 GiB of the
  virtual machine's memory taken as page cache for the length of the load, which a read through
  the file bridge doesn't take.
- Read at the file's offsets, a load peaks at what it leaves and the buffer a tensor passes
  through: a plan's count of the process and its pinned copies then bounds the load as it bounds
  the shots.
- Nothing pinned shows as locked or pinned in `/proc`: the pinned copies are counted as
  anonymous memory, by the kernel as by this study.
- The load's time is the file system's: 86.2 to 92.9 s for 15.35 GiB through WSL's file bridge,
  3.0 s from NVMe. Through the bridge, reading in four pieces at once takes 40.6 to 45.4 s; a
  job's SHA-256 of the same file, before its load, reads it once more, one stream.

## Reproduce

```bash
# One load, sampled, with its digests (seedvr2x's venv python, under the GPU's lock):
python scripts/load_peak.py watch --out today.tsv -- \
    python scripts/load_peak.py load --dit DIT.safetensors --vae VAE.safetensors \
        --whole --digests today.json > today.log
python scripts/load_peak.py table today.tsv today.log
# Another reader, the same tensors:
python scripts/load_peak.py watch --out pread.tsv -- \
    python scripts/load_peak.py load --dit DIT.safetensors --vae VAE.safetensors \
        --whole --reader pread --digests pread.json > pread.log
python scripts/load_peak.py same today.json pread.json
# Inside a group held to 16 GiB:
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
    python scripts/load_peak.py watch --out held.tsv -- python scripts/load_peak.py load ...
# How much of a file the page cache holds, and a cold start:
python scripts/load_peak.py cache --drop DIT.safetensors
# The file's reads alone, four at once through a descriptor each; a directory's runs as a table:
python scripts/load_peak.py read DIT.safetensors --threads 4 --own --times 3
python scripts/load_peak.py rows RUNS/
```
