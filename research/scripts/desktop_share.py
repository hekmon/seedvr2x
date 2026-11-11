"""A desktop's share of the card played by a program of its own, beside a run on an emulated card.

    desktop_share.py --ready FILE [--go FILE] [--log FILE] [--base 0.5] [--peak 1.1]
        [--step 0.0625] [--every 2] [--rest 30] [--hold 40]

A desktop on the card a run upscales on holds some of its memory and moves it: on Windows one at
work moved by 0.44 to 0.58 GiB within a 4-minute job, and gave up 455 MiB at once as the card
filled (research/docs/hold.md, "The desktop alone"). small_card.py's ballast plays the card's
size and keeps nothing moving; this program plays the desktop. It is a process of its own with a
CUDA context of its own and no torch, holding memory through the driver API (cuMemAlloc, which
NVML counts as another program's, as small_card.py's ballast does), in pieces of `--step` GiB, as
a compositor asks for surfaces:

  - `--base` GiB held before the run, this process's CUDA context counted in it (FILE `--ready`
    written once held, with what was held);
  - from `--go` FILE's appearance (the run's start; at once without it), a cycle until SIGTERM or
    SIGINT: `--rest` s at the base, then one piece every `--every` s up to `--peak` GiB, `--hold`
    s there (a piece refused is asked again at the next tick, the desktop still wanting it), then
    every piece over the base given back at once;
  - every allocation the driver refuses logged with its time and CUDA's error (2 is out of
    memory): the desktop starved, where a driver that refuses has given the card to the run;
  - the card's free and used memory by NVML once a second, its least free said at the end with
    the counts.

Each line of `--log` is `time action size_bytes held_bytes status free_bytes`, the time in epoch
seconds: action `alloc`, `refused`, `free` or `nvml` (size 0)."""

import argparse
import ctypes
import json
import os
import signal
import sys
import threading
import time

GIB = 1024**3
OUT_OF_MEMORY = 2  # CUDA_ERROR_OUT_OF_MEMORY


class _Nvml:
    """The card's free and used memory, through NVML's own library (no torch, no seedvr2x)."""

    class _Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise SystemExit("desktop_share: nvmlInit failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise SystemExit("desktop_share: no NVML handle for card 0")

    def read(self) -> tuple[int, int]:
        memory = self._Memory()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            return -1, -1
        return memory.free, memory.used


class _Driver:
    """The CUDA driver API: a context of this process's own, allocations that may be refused."""

    def __init__(self) -> None:
        self.cuda = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        for name, status in (
            ("cuInit", lambda: self.cuda.cuInit(0)),
            ("cuDeviceGet", lambda: self.cuda.cuDeviceGet(ctypes.byref(device), 0)),
            (
                "cuDevicePrimaryCtxRetain",
                lambda: self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
            ),
            ("cuCtxSetCurrent", lambda: self.cuda.cuCtxSetCurrent(context)),
        ):
            found = status()
            if found != 0:
                raise SystemExit(f"desktop_share: {name} failed: CUDA error {found}")

    def alloc(self, size: int) -> tuple[int, int]:
        """(pointer, 0), or (0, CUDA's error) where the driver refuses."""
        pointer = ctypes.c_uint64()
        status = self.cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(size))
        return (pointer.value, 0) if status == 0 else (0, status)

    def free(self, pointer: int) -> None:
        self.cuda.cuMemFree_v2(ctypes.c_uint64(pointer))


class Desktop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.piece = round(args.step * GIB)
        self.base = round(args.base * GIB)
        self.peak = round(args.peak * GIB)
        self.nvml = _Nvml()
        before, _ = self.nvml.read()
        self.driver = _Driver()
        time.sleep(0.5)
        after, _ = self.nvml.read()
        # The context is the desktop's too: counted in what it holds, so that the base is the
        # process's whole share as NVML sees it.
        self.context = max(before - after, 0)
        self.held: list[int] = []
        self.lock = threading.Lock()
        self.log = open(args.log, "a") if args.log else None
        self.counts = {"alloc": 0, "refused": 0, "free": 0, "cycles": 0, "peaks_reached": 0}
        self.first_refusal: float | None = None
        self.least = (1 << 62, 0.0)
        self.stop = threading.Event()

    def held_bytes(self) -> int:
        return self.context + len(self.held) * self.piece

    def line(self, action: str, size: int, status: int) -> None:
        free, _ = self.nvml.read()
        if self.log:
            self.log.write(
                f"{time.time():.2f} {action} {size} {self.held_bytes()} {status} {free}\n"
            )
            self.log.flush()

    def grow(self) -> bool:
        """One piece more; False where the driver refused it."""
        with self.lock:
            pointer, status = self.driver.alloc(self.piece)
            if status:
                self.counts["refused"] += 1
                if self.first_refusal is None:
                    self.first_refusal = time.time()
                    print(
                        f"desktop_share: REFUSED at {time.strftime('%H:%M:%S')}: a piece of"
                        f" {self.piece / 2**20:.0f} MiB, CUDA error {status}, holding"
                        f" {self.held_bytes() / GIB:.3f} GiB",
                        flush=True,
                    )
                self.line("refused", self.piece, status)
                return False
            self.held.append(pointer)
            self.counts["alloc"] += 1
            self.line("alloc", self.piece, 0)
            return True

    def give_back(self, down_to: int) -> None:
        with self.lock:
            freed = 0
            while self.held and self.held_bytes() > down_to:
                self.driver.free(self.held.pop())
                freed += self.piece
            if freed:
                self.counts["free"] += 1
                self.line("free", freed, 0)

    def watch(self) -> None:
        while not self.stop.wait(1.0):
            free, used = self.nvml.read()
            if 0 <= free < self.least[0]:
                self.least = (free, time.time())
            if self.log:
                with self.lock:
                    self.log.write(f"{time.time():.2f} nvml 0 {self.held_bytes()} {used} {free}\n")
                    self.log.flush()

    def wait(self, seconds: float) -> bool:
        """True where the stop came first."""
        return self.stop.wait(seconds)

    def reach(self, target: int, seconds: float | None) -> None:
        """Grow a piece a tick towards target, asking again for a piece refused; with seconds,
        give up once they are spent (the hold at the peak), else until reached."""
        started = time.time()
        while self.held_bytes() < target and not self.stop.is_set():
            if seconds is not None and time.time() - started >= seconds:
                return
            self.grow()
            if self.held_bytes() >= target:
                return
            if self.wait(self.args.every):
                return

    def run(self) -> int:
        threading.Thread(target=self.watch, daemon=True).start()
        # The base before the run, as fast as the driver gives it.
        while self.held_bytes() < self.base and not self.stop.is_set():
            if not self.grow() and self.wait(self.args.every):
                break
        free, used = self.nvml.read()
        info = {
            "base": self.base,
            "peak": self.peak,
            "piece": self.piece,
            "held": self.held_bytes(),
            "context": self.context,
            "free_after": free,
            "used_after": used,
            "pid": os.getpid(),
            "time": time.time(),
        }
        print(f"desktop_share: ready {json.dumps(info)}", flush=True)
        with open(self.args.ready + ".tmp", "w") as handle:
            json.dump(info, handle)
        os.replace(self.args.ready + ".tmp", self.args.ready)
        if self.args.go:
            while not os.path.exists(self.args.go) and not self.wait(0.2):
                pass
        while not self.stop.is_set():
            self.counts["cycles"] += 1
            # At the base: a piece lost to a refusal before is asked for again.
            self.reach(self.base, None)
            if self.wait(self.args.rest):
                break
            # Up to the peak, a piece a tick, then held there for --hold seconds: what of the
            # climb the driver refused is asked again through the hold.
            climb = (self.peak - self.base) / self.piece * self.args.every
            self.reach(self.peak, climb + self.args.hold)
            if self.held_bytes() >= self.peak:
                self.counts["peaks_reached"] += 1
            if self.wait(self.args.hold):
                break
            self.give_back(self.base)
        least, when = self.least
        summary = {
            **self.counts,
            "held_at_end": self.held_bytes(),
            "first_refusal": self.first_refusal,
            "least_free": least,
            "least_free_at": when,
        }
        self.give_back(0)
        print(f"desktop_share: ends {json.dumps(summary)}", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="desktop_share.py")
    parser.add_argument("--ready", required=True, help="written once the base is held")
    parser.add_argument("--go", default="", help="the cycle starts once this file exists")
    parser.add_argument("--log", default="")
    parser.add_argument("--base", type=float, default=0.5, help="GiB held before the run")
    parser.add_argument("--peak", type=float, default=1.1, help="GiB at the top of a cycle")
    parser.add_argument("--step", type=float, default=0.0625, help="GiB a piece")
    parser.add_argument("--every", type=float, default=2.0, help="seconds between pieces")
    parser.add_argument("--rest", type=float, default=30.0, help="seconds at the base a cycle")
    parser.add_argument("--hold", type=float, default=40.0, help="seconds at the peak a cycle")
    args = parser.parse_args()
    desktop = Desktop(args)
    signal.signal(signal.SIGTERM, lambda *_: desktop.stop.set())
    signal.signal(signal.SIGINT, lambda *_: desktop.stop.set())
    return desktop.run()


if __name__ == "__main__":
    sys.exit(main())
