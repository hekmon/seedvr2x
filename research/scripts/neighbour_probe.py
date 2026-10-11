#!/usr/bin/env python3
"""Neighbour probe for seedvr2x: what one program reads of another's memory on the card, under
a driver that gives system memory in the card's place (Windows', WSL2 included), where the free
memory CUDA reports turned out to be the process's own while the card has room.

Two roles, each a process of its own, to run side by side:

  holder GIB SECONDS [active]   takes GIB of the card, touches it, then prints every 5 s for
                                SECONDS the free memory it reads; with `active`, it works on
                                its memory all the while (every second)
  watcher SECONDS GIB           takes nothing but a context, prints every 5 s for SECONDS the
                                free memory it reads, then takes GIB, touches it, and reads
                                the free memory 3 more times

nvidia-smi --query-gpu=memory.used beside them gives the whole card's use. To see the figures
fall as the card fills, run a holder beside a seedvr2x run that takes the rest: seedvr2x reads
the card's free memory before every unit there, and stops when what it may still take no
longer fits (seedvr2x's runtime/memory.py, check).

Measured on an RTX 5090 under WSL2 (driver 616.92, 2026-10-11, 30.20 GiB read free at rest):
the watcher read 30.20 GiB while the holder held 8 GiB, idle or active, and 26.08 once it had
taken 4 itself; the holder read 22.08 throughout; nvidia-smi showed 11.6 GiB in use with the
holder there, 2.8 without. Beside a seedvr2x run holding 15.8 GiB, the holder's reading fell
from 22.06 to 11.72 and the run read 3.89 GiB free (nvidia-smi: 28.6 of 31.8 GiB in use).

Usage (seedvr2x's venv python, under the GPU's lock; sizes far under the card pass nothing):
  python neighbour_probe.py holder 8 100 active &
  sleep 12; python neighbour_probe.py watcher 50 4; wait
"""

import sys
import time

GIB = 2**30


def main() -> int:
    import torch

    role = sys.argv[1]
    start = time.time()

    def say(what: str) -> None:
        free = torch.cuda.mem_get_info(0)[0] / GIB
        print(f"{time.time() - start:6.1f} s {role}: {what}: {free:.2f} GiB free", flush=True)

    if role == "holder":
        gib, seconds, active = float(sys.argv[2]), float(sys.argv[3]), "active" in sys.argv[4:]
        say("before")
        held = torch.empty(int(gib * GIB), dtype=torch.uint8, device="cuda")
        held.fill_(1)
        torch.cuda.synchronize()
        say(f"{gib:g} GiB taken")
        end, last = time.time() + seconds, 0.0
        while time.time() < end:
            if active:
                held.add_(1)
                torch.cuda.synchronize()
            if time.time() - last >= 5:
                last = time.time()
                say("holding" + (", at work" if active else ", idle"))
            time.sleep(1)
        say("leaving")
        return 0
    if role == "watcher":
        seconds, gib = float(sys.argv[2]), float(sys.argv[3])
        torch.cuda.init()
        end = time.time() + seconds
        while time.time() < end:
            say("watching")
            time.sleep(5)
        taken = torch.empty(int(gib * GIB), dtype=torch.uint8, device="cuda")
        taken.fill_(1)
        torch.cuda.synchronize()
        for _ in range(3):
            say(f"{gib:g} GiB taken")
            time.sleep(5)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
