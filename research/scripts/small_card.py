"""A smaller card emulated on a card that drives a desktop, the desktop's share kept free.

Two halves, one method (place-study, 2026-11-05), for seedvr2x's runs and for this repository's
measuring scripts alike:

  small_card.py ballast --card N|whole [--keep 4] [--alarm GIB] [--watch PIDFILE] --ready FILE
      [--log FILE]
  small_card.py run [--keep 4] [--attention sdpa] [--pidfile FILE] [--ballast READY] -- \
      (cli ARGS... | SCRIPT.py ARGS...)

`ballast` is a process of its own, with a CUDA context of its own and no torch: F0 is the card's
free memory by NVML before it, T the free memory an N GB card leaves a program, N - 0.65 GiB
(0.35 GiB the driver hides and 0.3 a desktop holds: vram.md, "Emulating a smaller GPU"; the
figure comfy-study's emulated cards used on 2026-11-04, so that its runs and these compare). It
holds F0 - T - keep through the driver API (cuMemAlloc, which NVML counts as another program's),
so that the card has T + keep free, then reads the card every second: where the card has less
than keep - 0.25 GiB free for two readings in a row, it frees what it holds and sends SIGTERM to
the process whose pid the file `--watch` holds (`run --pidfile`), and logs ALARM (the
desktop's share is the rule). `--alarm GIB` sets that floor itself: with `--keep 0`, a card
left exactly T, nothing kept for a desktop that is idle, `--alarm 0.3` stops everything under
0.3 GiB free (place-study's night, 2026-11-05). `--card whole` holds nothing and only watches.

`run` starts the measured process: seedvr2x's command (`cli`) or a research script, in this
process, with seedvr2x's reading of the card through NVML (memory._read_all, which card_free and
card_state read) less `keep` and its programs' memory more by as much, so that the run counts on
T and torch is held under T less seedvr2x's 0.5 GiB (memory.hold): what an N GB card's run counts
on, the desktop's keep never in it. The driver's own figure (torch's mem_get_info) is left as it
is: under WSL2 it is a fixed budget less the process's own use (hold.md, "What the driver's free
figure counts"), over T, and seedvr2x plans in the smaller of the two. `--attention sdpa` hides
FlashAttention 2 from the run (the vendored model and the plan then take PyTorch's SDPA).
`--ballast READY` is the ballast's `--ready` file: on Linux seedvr2x holds a run for another
program on the card (memory.share, by NVML), and the ballast's CUDA context is one NVML lists, so
that every emulated run was taken for shared and held, which a card nothing else uses is not. With
the option, `run` leaves the pid in that file (its "pid") out of the share's processes, so that
the card is planned as that card alone; a desktop, or desktop_share.py (a CUDA process too),
still makes it shared, as it should.
`--set module.NAME=value` sets one of the library's figures before the run (a number keeps its
type), `--lean` installs dit_lean.py's attention (with `--lean-chunk`, 4096 tokens by default)
and `--gelu-inplace` its MLP, `--tail-frames` slice_probe.py's decoder tail a frame at a time,
`--decode-fit F` the plan's decode peaks times F: what a proposal would plan and run, measured
without editing the library.

The emulation is of memory alone: the speed, the PCIe link and the system memory are the RTX
5090's and its host's."""

import argparse
import ctypes
import dataclasses
import json
import os
import runpy
import signal
import sys
import threading
import time
from typing import Any

GIB = 1024**3
HIDDEN_AND_DESKTOP = 0.65  # GiB: an N GB card leaves N - 0.65 (vram.md; comfy-study)
ALARM_MARGIN = round(0.25 * GIB)
TOLERANCE = 64 * 1024**2


def _nvml() -> tuple[int, int]:
    """(free, used) of the card by NVML, seedvr2x's own reading (memory.card_state)."""
    from seedvr2x import memory

    found = memory._read_all(0)
    if isinstance(found, str):
        raise SystemExit(f"small_card: NVML gave no reading: {found}")
    return found


class _Driver:
    """The CUDA driver API through ctypes: a context of this process's own, and allocations."""

    def __init__(self) -> None:
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self._check("cuInit", self.cuda.cuInit(0))
        device = ctypes.c_int()
        self._check("cuDeviceGet", self.cuda.cuDeviceGet(ctypes.byref(device), 0))
        context = ctypes.c_void_p()
        self._check(
            "cuDevicePrimaryCtxRetain",
            self.cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device),
        )
        self._check("cuCtxSetCurrent", self.cuda.cuCtxSetCurrent(context))

    @staticmethod
    def _check(name: str, status: int) -> None:
        if status != 0:
            raise SystemExit(f"small_card: {name} failed: CUDA error {status}")

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(
            "cuMemAlloc", self.cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(size))
        )
        return pointer.value

    def free(self, pointer: int) -> None:
        self._check("cuMemFree", self.cuda.cuMemFree_v2(ctypes.c_uint64(pointer)))


def ballast(args: argparse.Namespace) -> int:
    keep = round(args.keep * GIB)
    f0, _ = _nvml()
    whole = args.card == "whole"
    target = f0 - keep if whole else round((float(args.card) - HIDDEN_AND_DESKTOP) * GIB)
    held: list[int] = []
    driver = None
    if not whole:
        hold = f0 - target - keep
        if hold < 0:
            print(
                f"small_card: refused: the card has {f0 / GIB:.2f} GiB free, under T"
                f" {target / GIB:.2f} and the desktop's {args.keep:g}",
                file=sys.stderr,
            )
            return 3
        driver = _Driver()
        # The context itself takes some of the card: read again, and hold what is left to hold.
        for _ in range(4):
            free, _ = _nvml()
            excess = free - (target + keep)
            if abs(excess) <= TOLERANCE:
                break
            if excess > 0:
                held.append(driver.alloc(excess))
            else:
                break  # the context took more than the hold left: under T + keep, said below
            time.sleep(0.3)
    free, used = _nvml()
    info = {
        "card": args.card,
        "keep_gib": args.keep,
        "F0": f0,
        "T": target,
        "held_allocations": len(held),
        "free_after": free,
        "used_after": used,
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"small_card: ballast {json.dumps(info)}", flush=True)
    with open(args.ready + ".tmp", "w") as handle:
        json.dump(info, handle)
    os.replace(args.ready + ".tmp", args.ready)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log = open(args.log, "a") if args.log else None
    low, least = 0, free
    floor = keep - ALARM_MARGIN if args.alarm is None else round(args.alarm * GIB)
    while not stop.wait(1.0):
        free, used = _nvml()
        least = min(least, free)
        if log:
            log.write(f"{time.time():.1f} {free} {used}\n")
            log.flush()
        low = low + 1 if free < floor else 0
        if low >= 2:
            for pointer in held:
                driver.free(pointer)  # type: ignore[union-attr]
            held = []
            print(
                f"small_card: ALARM: the card had {free / GIB:.2f} GiB free, under"
                f" {floor / GIB:.2f} GiB: ballast freed",
                flush=True,
            )
            if args.watch:
                try:
                    with open(args.watch) as handle:
                        os.kill(int(handle.read().strip()), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
            break
    for pointer in held:
        driver.free(pointer)  # type: ignore[union-attr]
    print(f"small_card: ballast ends, the card's least free {least / GIB:.3f} GiB", flush=True)
    return 0


def install(keep_gib: float) -> None:
    """seedvr2x's reading of the card through NVML, in this process, less keep_gib GiB free and
    more as much used (the desktop's share, never the run's)."""
    from seedvr2x import memory

    keep = round(keep_gib * GIB)
    original = memory._read_all

    def capped(index: int, **more: Any) -> tuple[int, ...] | str:
        # The reading's other figures (memory.activity's, `busy`) come after the two capped.
        found = original(index, **more)
        if isinstance(found, str):
            return found
        free, used, *rest = found
        return max(free - keep, 0), used + keep, *rest

    memory._read_all = capped


def leave_out(pid: int) -> None:
    """memory.share (seedvr2x's reading of who else uses the card, startup and the early reading
    call it by the module's name) without the process `pid` among the processes it lists: the
    ballast's, whose CUDA context NVML counts as another program's. The display and the unread
    reason are kept, and any other process stays."""
    from seedvr2x import memory

    original = memory.share

    def apart(index: int = 0) -> Any:
        found = original(index)
        return dataclasses.replace(
            found, processes=tuple(each for each in found.processes if each[0] != pid)
        )

    memory.share = apart


def hide_flash_attention() -> None:
    """FlashAttention 2 not importable in this process: the model and the plan take SDPA."""
    sys.modules["flash_attn"] = None  # type: ignore[assignment]
    sys.modules["flash_attn_interface"] = None  # type: ignore[assignment]


def run(args: argparse.Namespace) -> int:
    rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    if not rest:
        raise SystemExit("small_card: run what? `cli ARGS` or `SCRIPT.py ARGS`")
    if args.pidfile:
        with open(args.pidfile, "w") as handle:
            handle.write(str(os.getpid()))
    if args.attention == "sdpa":
        hide_flash_attention()
    install(args.keep)
    if args.ballast:
        with open(args.ballast) as handle:
            leave_out(int(json.load(handle)["pid"]))
    for setting in args.set:
        name, _, value = setting.partition("=")
        module_name, _, attribute = name.rpartition(".")
        import importlib

        module = importlib.import_module(module_name)
        old = getattr(module, attribute)
        setattr(
            module, attribute, type(old)(float(value)) if isinstance(old, (int, float)) else value
        )
        print(f"small_card: {name} = {getattr(module, attribute)!r} (was {old!r})", flush=True)
    if args.decode_fit:
        # The plan's decode peaks (peaks._fit's decode column) times the factor: what a decoder
        # whose peak is that share of today's would be planned in.
        from seedvr2x import peaks

        fit = peaks._fit
        factor = args.decode_fit

        def scaled(padded: Any, tiles: Any, column: int) -> float:
            return fit(padded, tiles, column) * (factor if column == 1 else 1.0)

        peaks._fit = scaled
    if args.lean or args.tail_frames:
        # dit_lean.py and slice_probe.py import torch: installed once seedvr2x has chosen the
        # allocator.
        from seedvr2x import memory

        startup = memory.startup

        def started_up() -> str:
            said = startup()
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            if args.lean:
                import dit_lean

                dit_lean.install(lean=True, chunk=args.lean_chunk)
                if args.gelu_inplace:
                    dit_lean.gelu_inplace()
            if args.tail_frames:
                import slice_probe

                slice_probe.install_tail()
            return said

        memory.startup = started_up
    if rest[0] == "cli":
        import main  # the seedvr2x command's own module (its script is main:main)

        return main.main(rest[1:])
    sys.argv = rest
    runpy.run_path(rest[0], run_name="__main__")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="small_card.py")
    sub = parser.add_subparsers(dest="what", required=True)
    b = sub.add_parser("ballast")
    b.add_argument("--card", required=True, help="N (GB) or whole")
    b.add_argument("--keep", type=float, default=4.0, help="GiB left free to the desktop")
    b.add_argument("--alarm", type=float, help="GiB free under which it stops (keep - 0.25)")
    b.add_argument("--watch", default="", help="a file holding the run's pid, stopped on an alarm")
    b.add_argument("--ready", required=True)
    b.add_argument("--log", default="")
    r = sub.add_parser("run")
    r.add_argument("--keep", type=float, default=4.0)
    r.add_argument("--attention", choices=("default", "sdpa"), default="default")
    r.add_argument("--pidfile", default="")
    r.add_argument("--ballast", default="", help="the ballast's --ready file: its pid left out")
    r.add_argument("--set", action="append", default=[], help="module.NAME=value, before the run")
    r.add_argument("--lean", action="store_true", help="dit_lean.py's attention")
    r.add_argument("--lean-chunk", type=int, default=4096)
    r.add_argument("--gelu-inplace", action="store_true", help="dit_lean.py's in-place GELU")
    r.add_argument("--tail-frames", action="store_true", help="slice_probe.py's decoder tail")
    r.add_argument("--decode-fit", type=float, default=0.0, help="the plan's decode peaks x F")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return ballast(args) if args.what == "ballast" else run(args)


if __name__ == "__main__":
    sys.exit(main())
