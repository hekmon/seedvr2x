#!/usr/bin/env python3
"""Card watch for seedvr2x: what a card holds, second by second, while nothing of ours runs on
it or beside a run: the movement of a desktop's own memory, which a run held to the card's
memory has to leave room for (seedvr2x's runtime/memory.py: HELD_BACK), told apart from what
the run itself holds.

Two records, each line stamped by the clock of whoever wrote it:
  --out FILE.jsonl   the whole card by NVML, the driver's own library (what nvidia-smi prints),
                     read through ctypes every --every seconds: total, reserved (the driver's
                     own share), used and free, in bytes, with the time (`t`, seconds since the
                     epoch; `at`, local time)
  --windows FILE.csv under WSL2 only: Windows' own count, by its performance counters, read by
                     typeperf.exe once a second, its CSV kept as it comes: the adapter's
                     dedicated and shared memory in use (`GPU Adapter Memory`: the shared part
                     is system memory given in the card's place, where a driver spills) and each
                     process's (`GPU Process Memory`), a Linux process's memory counted under
                     the virtual machine's. nvidia-smi names no process there, under WSL2 or on
                     the Windows side (WDDM: N/A). The counters' instances are those alive when
                     typeperf starts: it is started again every --renew seconds, a new header
                     line each time, so that a program started since is counted.
  --names FILE.csv   with --windows: the processes' names by their identifiers, tasklist.exe's
                     CSV, written at each start of typeperf after a line `# TIME`
  --windows-dir DIR  with --windows: the directory Windows' programs start in, on a Windows drive
                     as WSL mounts it; by default the first such drive in /proc/self/mounts
                     (a mount of type drvfs, or 9p with aname=drvfs), as the measures ran them

The files name the machine and its programs: they are a measure's log, not for a repository.

`summary` reads a --out file back: the range of the used memory, and its steps (a change
between two readings a second apart or less: how many, how large, how often), over the whole
file or between --since and --until (seconds since the epoch, or HH:MM:SS of the file's first
day).

Usage:
  python card_watch.py watch --out card.jsonl [--windows win.csv --names names.csv]
  python card_watch.py summary card.jsonl [--since 01:00:00 --until 02:00:00] [--least 1]
(watch runs until it is interrupted or terminated, and stops typeperf with it)
"""

import argparse
import ctypes
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time

MIB = 1 << 20
COUNTERS = (
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Adapter Memory(*)\Shared Usage",
    r"\GPU Adapter Memory(*)\Total Committed",
    r"\GPU Process Memory(*)\Dedicated Usage",
    r"\GPU Process Memory(*)\Shared Usage",
)


class Memory(ctypes.Structure):
    """NVML's nvmlMemory_v2_t (nvml.h), in bytes."""

    _fields_ = (
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


class Card:
    """A card's memory by NVML, initialised once and read as often as asked."""

    def __init__(self, index: int) -> None:
        self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
        self.handle = ctypes.c_void_p()
        self.check("nvmlInit", self.nvml.nvmlInit_v2())
        found = self.nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle))
        self.check("nvmlDeviceGetHandleByIndex", found)

    @staticmethod
    def check(name: str, status: int) -> None:
        if status != 0:
            sys.exit(f"card_watch: {name}: {status}")

    def read(self) -> dict:
        memory = Memory(version=ctypes.sizeof(Memory) | 2 << 24)
        status = self.nvml.nvmlDeviceGetMemoryInfo_v2(self.handle, ctypes.byref(memory))
        if status != 0:
            return {"error": f"nvmlDeviceGetMemoryInfo: {status}"}
        return {name: int(getattr(memory, name)) for name in ("total", "reserved", "used", "free")}


def windows_drive() -> str:
    """The first Windows drive WSL has mounted, by /proc/self/mounts: a mount of type drvfs, or 9p
    with aname=drvfs among its options (a space in its path written \\040 there)."""
    with open("/proc/self/mounts") as mounts:
        for line in mounts:
            fields = line.split()
            if len(fields) > 3 and (fields[2] == "drvfs" or "aname=drvfs" in fields[3]):
                return fields[1].replace("\\040", " ")
    sys.exit("card_watch.py: no Windows drive mounted here: give --windows-dir")


def windows(csv: str, names: str | None, renew: float, stop: threading.Event, where: str) -> None:
    """typeperf.exe's CSV into `csv`, started again every `renew` seconds, until `stop`; Windows'
    programs started in `where`."""
    with open(csv, "ab") as out:
        while not stop.is_set():
            if names:
                listed = subprocess.run(
                    ["tasklist.exe", "/fo", "csv", "/nh"], capture_output=True, cwd=where
                )
                with open(names, "ab") as file:
                    file.write(f"# {time.time():.3f}\n".encode() + listed.stdout)
            out.write(f"# {time.time():.3f}\n".encode())
            out.flush()
            child = subprocess.Popen(
                ["typeperf.exe", *COUNTERS, "-si", "1"],
                stdout=out,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                cwd=where,
            )
            stop.wait(renew)
            child.terminate()
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def watch(args: argparse.Namespace) -> int:
    card = Card(args.gpu)
    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    side = None
    if args.windows:
        if not os.path.exists("/dev/dxg"):
            sys.exit("card_watch: --windows needs WSL2 (Windows' counters, through typeperf.exe)")
        where = args.windows_dir or windows_drive()
        side = threading.Thread(
            target=windows, args=(args.windows, args.names, args.renew, stop, where)
        )
        side.start()
    with open(args.out, "a") as out:
        following = time.time()
        while not stop.is_set():
            now = time.time()
            at = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"
            out.write(json.dumps({"t": round(now, 3), "at": at, **card.read()}) + "\n")
            out.flush()
            following = max(following + args.every, now)
            stop.wait(max(following - time.time(), 0))
    if side:
        side.join()
    return 0


def moment(text: str, first: float) -> float:
    """--since or --until as seconds since the epoch: a number, or HH:MM:SS of `first`'s day,
    the day after where that falls before `first`."""
    if ":" not in text:
        return float(text)
    day = time.localtime(first)
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    at = time.mktime((day.tm_year, day.tm_mon, day.tm_mday, hours, minutes, seconds, 0, 0, -1))
    return at if at >= first - 1 else at + 86400


def summary(args: argparse.Namespace) -> int:
    with open(args.file) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    rows = [row for row in rows if "used" in row]
    if not rows:
        sys.exit("card_watch: no reading in the file")
    first = rows[0]["t"]
    since = moment(args.since, first) if args.since else first
    until = moment(args.until, first) if args.until else rows[-1]["t"]
    rows = [row for row in rows if since <= row["t"] <= until]
    if len(rows) < 2:
        sys.exit("card_watch: fewer than two readings between --since and --until")
    used = [row["used"] / MIB for row in rows]
    span = rows[-1]["t"] - rows[0]["t"]
    steps = [
        (b["used"] - a["used"]) / MIB
        for a, b in itertools.pairwise(rows)
        if b["t"] - a["t"] <= 2 * args.every and abs(b["used"] - a["used"]) >= args.least * MIB
    ]
    print(
        f"{rows[0]['at']} to {rows[-1]['at']}: {len(rows)} readings over {span / 60:.1f} min;"
        f" used {min(used):,.0f} to {max(used):,.0f} MiB (a range of {max(used) - min(used):,.0f}),"
        f" {used[0]:,.0f} at the start and {used[-1]:,.0f} at the end;"
        f" free {min(row['free'] for row in rows) / MIB:,.0f} MiB at the least;"
        f" the driver's own share {rows[0]['reserved'] / MIB:,.0f} MiB"
    )
    up, down = [step for step in steps if step > 0], [-step for step in steps if step < 0]
    print(
        f"steps of {args.least:g} MiB or more between two readings: {len(steps)}"
        f" ({len(steps) / span * 3600:.0f} an hour), {len(up)} up and {len(down)} down"
    )
    for name, group in (("up", up), ("down", down)):
        if group:
            group.sort()
            print(
                f"  {name}: {group[0]:.1f} to {group[-1]:.1f} MiB,"
                f" median {group[len(group) // 2]:.1f}, {sum(group):,.0f} in all;"
                f" of 16 MiB or more: {sum(1 for step in group if step >= 16)},"
                f" of 64 or more: {sum(1 for step in group if step >= 64)},"
                f" of 256 or more: {sum(1 for step in group if step >= 256)}"
            )
    # The widest rise and fall within a sliding minute, 5 minutes and 15: what a run that set
    # its limit at the window's start would have seen taken from it, or given back, by its end.
    times = [row["t"] for row in rows]
    for width in (60, 300, 900):
        rise = fall = 0.0
        start = 0
        for end in range(len(rows)):
            while times[end] - times[start] > width:
                start += 1
            window = used[start : end + 1]
            rise, fall = max(rise, used[end] - min(window)), max(fall, max(window) - used[end])
        print(f"  within {width // 60} min: up by {rise:,.0f} MiB at the most, down by {fall:,.0f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parts = parser.add_subparsers(dest="part", required=True)
    one = parts.add_parser("watch")
    one.add_argument("--out", required=True)
    one.add_argument("--every", type=float, default=1.0)
    one.add_argument("--gpu", type=int, default=0)
    one.add_argument("--windows")
    one.add_argument("--names")
    one.add_argument("--renew", type=float, default=600.0)
    one.add_argument("--windows-dir")
    two = parts.add_parser("summary")
    two.add_argument("file")
    two.add_argument("--since")
    two.add_argument("--until")
    two.add_argument("--every", type=float, default=1.0)
    two.add_argument("--least", type=float, default=1.0)
    args = parser.parse_args()
    return watch(args) if args.part == "watch" else summary(args)


if __name__ == "__main__":
    sys.exit(main())
