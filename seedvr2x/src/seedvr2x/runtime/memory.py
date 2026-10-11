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
that refuses, and nothing changes. Given, it is one that spills, and seedvr2x holds itself to
the card's memory by its own means:
- torch's own allocator in cudaMallocAsync's place (startup): it counts every byte it has from
  the driver, its cache included, where cudaMallocAsync's pool keeps memory torch's limit doesn't
  see (2.0 to 2.6 GiB of it at 11 latents at 1080p);
- that allocator limited (hold) to the memory the driver reports free less HELD_BACK: what
  passes it is refused by torch, an OutOfMemoryError that Python catches, in under a second,
  and the work after it is right;
- the planner planning in what is left less RESERVE (usable), which that allocator needs
  beyond what a step allocates;
- the card's free memory read again before every unit (check): the run stops when what torch
  may still take no longer fits in it, other programs holding the rest of the card.

The free memory such a driver reports is the process's own while the card has room: a process
read 30.20 GiB free of 31.82 whether another held 8.6 GiB of the card or not, idle or at work,
and all day beside a desktop that nvidia-smi showed using 2.7 to 4.5 GiB; nvidia-smi's figure
is the whole card's. As the card fills the figure falls to what is really free: a run holding
15.8 GiB beside that other program read 3.89 GiB (nvidia-smi: 3.91 free), and the other program
10.3 GiB less than before. So a plan made beside another program counts on memory the card
doesn't have, and the check is what stops it, at its first unit
(research/scripts/neighbour_probe.py).

Measured on the RTX 5090 under WSL2 (research/scripts/hold_probe.py, 2026-10-11, 30.20 GiB free,
the models whole on the card, windows at 1080p: 17.0 GiB allocated at 1 latent, 22.6 at 6, 24.7
at 8, 27.9 at 11):
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

Nothing here imports torch at import: the allocator is chosen before torch is (cli._run), whose
library fixes it when it loads.
"""

import ctypes
import logging
import os
from dataclasses import dataclass
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
# CUDA's result codes (cuda.h): CUDA_SUCCESS, CUDA_ERROR_OUT_OF_MEMORY.
SUCCESS, OUT_OF_MEMORY = 0, 2
# What seedvr2x leaves of the free memory on a driver that spills, beyond the planner's own
# margin: once a run holds 22 GiB or more, the driver reports 2.1 to 2.8 GiB less free than
# torch's own count explains (6 latents at 1080p: 30.16 GiB gone from the figure with 27.82
# held by torch; a 720p VAE decode: 2.6; the command line at 432p with no block swapped: 2.1),
# and 0.14 GiB under that (12 or 36 blocks swapped, a window of 1 latent): most likely the other
# programs' memory, which the figure leaves out until the card fills (the module's docstring;
# the desktop held 2.7 to 4.5 GiB that day), with what CUDA's libraries take. Held 2.0 GiB
# under, the figure went down to 0.16 GiB at the least. 3 GiB: the largest measured and a
# little; held so, it never went under 1.59 GiB. PROVISIONAL (implementation, 2026-10-11): one
# card, one driver (616.92), one desktop, 1080p and under.
HELD_BACK = 3 * GIB
# What torch's own allocator needs from the driver beyond the peak a step allocates, held as
# above: it keeps freed blocks of the sizes it has used, and gives them back only when a
# request fails. 8 latents at 1080p, the models whole: 24.75 GiB allocated, 26.98 held, under a
# limit of 27.08; 9 latents were refused there. So a plan counts on the limit less 2.3 GiB (the
# planner's own margin comes on top). PROVISIONAL (implementation, 2026-10-11): one measured
# point; a 720p VAE decode held 24.14 GiB for 17.95 allocated with room to spare, not at a limit.
RESERVE = round(2.3 * GIB)
# The allocator's setting by the driver: DESIGN.md's default (Allocator), and torch's own
# allocator where the driver spills.
ASYNC, NATIVE = "backend:cudaMallocAsync", "backend:native"
# The variables torch reads that setting from. seedvr2x sets the first, and only when the user
# has set neither: with both set, torch 2.14 takes the first and never reads the second, so a
# default written over a user's PYTORCH_ALLOC_CONF would silence it.
VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


@dataclass(frozen=True)
class Driver:
    """What the probe found: whether the driver gave memory the card didn't have (spills), and
    its figures, in bytes: the request (asked), the memory free before it (free) and the
    card's (total)."""

    spills: bool
    asked: int
    free: int
    total: int

    def said(self) -> str:
        """The finding in words, for the log."""
        figures = f"{self.asked / GIB:.2f} GiB asked with {self.free / GIB:.2f} free"
        if not self.spills:
            return f"the GPU's driver refuses what the card can't hold ({figures}: refused)"
        return (
            f"the GPU's driver gives memory the card doesn't have ({figures}: given): system"
            " memory in its place, as Windows' driver does, under WSL2 too whatever NVIDIA's"
            ' "CUDA - Sysmem Fallback Policy" says. Nothing there refuses a run that passes the'
            " card: it would slow down, then fail without an error to catch, what it made last"
            " not to be trusted"
        )


# What startup found, for the run that follows: whether the run holds itself to the card and
# whether its allocator is torch's own; and, once hold has set it, torch's limit in bytes.
_spills = False
_native = False
_limit: int | None = None


def startup() -> str:
    """Ask the driver whether it refuses, once a run, before torch is imported, and set the
    allocator the run takes by default (VARIABLES, which the user's own setting overrides):
    NATIVE where the driver spills, ASYNC elsewhere; returned. Called where a run first takes
    the GPU (cli._run, its model files checked), so that a run refused before it, for its
    source, its settings or its model files, asks nothing: the request disturbs the card (probe).

    The driver is asked only where it can spill, behind WSL2's device: a Linux driver refuses,
    and isn't disturbed. What was found is said in the log, a warning where it spills. No
    answer there (probe) is taken for a driver that spills: it is Windows', which gave what
    the card didn't have every time it was asked, and a run held for nothing loses memory, a
    run not held where it should be loses its frames."""
    global _spills, _native, _limit
    _spills, _limit = False, None
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
    given = next((os.environ[name] for name in VARIABLES if name in os.environ), None)
    chosen = NATIVE if _spills else ASYNC
    if given is None:
        os.environ[VARIABLES[0]] = chosen
    # torch's own allocator is its default too, when the environment's setting names none.
    _native = NATIVE in (chosen if given is None else given) or "backend:" not in (given or chosen)
    if _spills:
        logger.warning(
            "seedvr2x holds itself to the card there: torch's own allocator, limited to the"
            " memory free less %.1f GiB, and plans in %.1f GiB less; a run that passes the"
            " limit is refused",
            HELD_BACK / GIB,
            RESERVE / GIB,
        )
        if not _native:
            # The user's own allocator stands; said, since the hold then holds less.
            logger.warning(
                "the allocator is the environment's, %s: torch's limit counts what torch has"
                " allocated there, not what that allocator keeps beyond it (2 GiB and more),"
                " so the card can still be passed; unset it for torch's own allocator",
                given,
            )
    return chosen


def spills() -> bool:
    """Whether the run holds itself to the card: its driver was found to give memory the card
    doesn't have, or is Windows' and gave no answer (startup)."""
    return _spills


def usable(free: int) -> int:
    """The memory a plan may count its peaks in, of `free` bytes the driver reports free: all
    of it on a driver that refuses; on one that spills, what torch is held to (hold: HELD_BACK
    less) less RESERVE, and said, since the plan's lines and its refusals then count in less
    than the driver reports. The planner's budget is this less its own margin
    (runtime/planner.py)."""
    if not _spills:
        return free
    counted = max(free - HELD_BACK - RESERVE, 0)
    logger.info(
        "the plan counts on %.2f GiB of the GPU's memory, of the %.2f GiB its driver reports"
        " free: %.1f GiB are left to what torch doesn't count on the card, %.1f to what torch's"
        " own allocator holds beyond a peak",
        counted / GIB,
        free / GIB,
        HELD_BACK / GIB,
        RESERVE / GIB,
    )
    return counted


def hold(device: Any) -> int | None:
    """On a driver that spills, limit torch on `device` (a torch.device) to what it holds now
    and the memory free now less HELD_BACK: more is refused, an OutOfMemoryError. Called once
    the plan is made and before the models load. Returns the limit, in bytes; None, nothing
    done, on a driver that refuses."""
    global _limit
    if not _spills:
        return None
    import torch

    free, total = torch.cuda.mem_get_info(device)
    limit = torch.cuda.memory_reserved(device) + max(free - HELD_BACK, 0)
    fraction = min(limit / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device)  # pyright: ignore[reportUnknownMemberType]
    _limit = limit
    logger.info(
        "torch held to %.2f GiB on the GPU: the %.2f GiB free less %.1f, and what it holds",
        limit / GIB,
        free / GIB,
        HELD_BACK / GIB,
    )
    return limit


def check(device: Any) -> None:
    """On a driver that spills, once torch is held: refuse (torch.OutOfMemoryError, as torch
    refuses what passes its limit) when what torch may still take under its limit no longer fits
    in the memory the driver reports free. Called before every unit (model.Models.enter).

    The limit is worth the memory free when it was set, less HELD_BACK, and that figure left
    the other programs' memory out (the module's docstring): it falls as the card fills, and
    once it is under what torch may still take, torch at its limit passes the card with nothing
    to refuse it. So the run stops, its units kept. Measured on the RTX 5090 (2026-10-11, the
    command line at 432p, 30.20 GiB reported free at the start):
    - alone, the figure stayed 2.85 GiB above what torch could still take with 12 or 36 blocks
      swapped, and 0.77 at the least with none (torch holding 22 GiB);
    - beside a program holding 8.6 GiB and working on it, a run with no block swapped loaded
      its 15.8 GiB, then read 3.89 GiB free where it could still take 11.36, and stopped at its
      first unit; the card had 28.6 GiB of its 31.8 in use at the most (nvidia-smi), never more;
    - beside the same program, a run with 36 blocks swapped, 8 GiB at its most, read the card
      as if alone and finished, its frames right: nothing passed the card there.

    Nothing is checked under an allocator of the environment's, whose own slack torch doesn't
    count (startup)."""
    if not _native or _limit is None:
        return
    import torch

    free = torch.cuda.mem_get_info(device)[0]
    room = _limit - torch.cuda.memory_reserved(device)
    if free < room:
        raise torch.OutOfMemoryError(
            f"the GPU has {free / GIB:.2f} GiB free where the run may still take"
            f" {room / GIB:.2f}: other programs hold the rest of the card's memory, or the"
            f" run's own work took more than the {HELD_BACK / GIB:.1f} GiB left to it outside"
            " torch's allocator"
        )


def paravirtual() -> bool:
    """Whether the GPU is reached through WSL2's device: Windows' driver under a Linux process."""
    return os.path.exists(PARAVIRTUAL)


def probe(index: int = 0, library: str = "libcuda.so.1") -> Driver | None:
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
    than before it, and is asked once more otherwise."""
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
