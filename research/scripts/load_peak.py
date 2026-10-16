#!/usr/bin/env python3
"""The load's peak in system memory: what a seedvr2x process holds while its weights load, sampled
from outside it, and whether another way of reading the weights' file takes less for the same
tensors.

seedvr2x's loader as measured (runtime/model.py, _load_weights, at 0393413) read a weights file two
ways: a tensor bound for the device came from safetensors' mapping of the file and was copied to
the device from it; a tensor of a part that leaves the device (runtime/blockswap.py: a swapped
block, the VAE) was read at the file's own offsets straight into its part's pinned host copy. A
mapped page stays in the process once read, until the mapping goes: the question was what that
makes of a job's resident memory, and what a host short of memory makes of it
(docs/load-memory.md). On a tree whose loader no longer reads a whole model through safetensors'
load_file, `load` measures that loader as it is, and the other readers are refused.

  watch   --out SAMPLES.tsv [--interval S] [--rollup S] [--give-up S] -- COMMAND ...
          Runs COMMAND, and samples it from outside every S seconds (0.05): /proc/PID/status
          (VmRSS, VmHWM, RssAnon, RssFile, RssShmem, VmLck, VmPin, VmSwap), the host's
          /proc/meminfo (MemAvailable, Cached, Unevictable, Mlocked) and, in a control group
          with the memory controller, the group's memory.current, memory.peak and memory.stat
          (anon, file); and every --rollup seconds (1), /proc/PID/smaps_rollup (Rss, Anonymous,
          Locked, the clean and dirty parts), which walks the process's pages: tens of
          milliseconds a read with 16 GiB resident, where status takes a counter. One line a
          sample, kB, with the wall clock, written as it is made. Its last lines give the
          command's exit code, the group's memory.events and what a sample cost. Exits as the
          command did; --give-up S kills a command still running after S seconds, and says so;
          asked to stop (systemd ends a scope one of whose processes the kernel's OOM killer
          took), it still writes those lines. Run it inside the group whose limit is tried:
            systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
                python load_peak.py watch --out S.tsv -- COMMAND ...
  load    --dit DIT.safetensors --vae VAE.safetensors [--swapped S] [--resident-encode R]
          [--resident-decode R] [--vae-away] [--whole] [--reader NAME] [--threads N]
          [--digests DIGESTS.json]
          Loads the models as a job does (memory.startup, memory.hold, model.load_models) with
          a plan of S swapped blocks (36: every block pinned in system memory; 0 and nothing
          else away: the DiT whole on the device), R blocks kept in each of the VAE's phases
          (0: the DiT away) and the VAE away in the DiT's phase; --whole: without a plan. Prints
          a mark with the wall clock at each step of the load (the VAE's weights, the DiT's),
          for `table`; then, with --digests, the SHA-256 of every parameter and buffer of the
          two models, read back from wherever it is.
          --reader replaces the loader's two reads (model.load_file, blockswap.read) for the
          run, the product untouched:
            today            the loader as it is
            madvise          mapped as today; each tensor's pages given back once it is copied
                             (madvise MADV_DONTNEED on its span of the mapping)
            madvise-fadvise  the same, and the span dropped from the page cache
                             (posix_fadvise POSIX_FADV_DONTNEED), the parts' spans too
            st-pread         safetensors' own backend="pread": no mapping, each tensor read
                             into memory of its own
            pread            no mapping, no safetensors: each tensor read at its offsets into
                             one staging buffer of the largest tensor's size, pageable, and
                             copied to the device from it; a part's tensors straight into its
                             pinned copy, as today
            pread-pinned     the same, the staging buffer page-locked (blockswap.HostCopy)
            pread-fadvise    pread, each span dropped from the page cache once read
          --threads N reads each span of the pread readers in N pieces at once (1).
  table   SAMPLES.tsv LOG [...]   per run: the peak and what it is made of, per step of the load
          (LOG holds the run's marks, or a job's own log lines); --json for the figures
  rows    DIRECTORY [REFERENCE.json]   the runs of a directory as one table, a row per plan,
          cache state and reader: each run is NAME.tsv, NAME.log and NAME.json (watch's samples,
          load's output and digests), NAME = READER[-tTHREADS]-PLAN[-STATE]-rROUND (PLAN: whole,
          mid, 36, as the one who ran them names plans; STATE: cold, warm). The figures are the
          least and the most over a row's runs, GiB and seconds; the digests are compared with
          REFERENCE.json's, or with the first of today's reader in the directory; the group's
          columns are its memory.peak, and memory.events' max (the times its limit was met)
          and oom_kill
  same    DIGESTS.json DIGESTS.json [...]   whether every tensor has the same bytes in each
  cache   [--drop] FILE [...]   how much of each file the page cache holds (mincore); --drop
          asks the kernel to drop it first (POSIX_FADV_DONTNEED), for a load that finds none
  read    FILE [--threads N] [--own] [--drop] [--times K]   the file's tensors read as the
          pread reader reads them, into one buffer, no torch and no GPU: the seconds and the
          rate, K times; each span in N pieces at once, through one descriptor or, with --own,
          through one per thread (the kernel's read-ahead keeps its state per descriptor)

Usage (seedvr2x's venv python; `load` under the GPU's lock):
  python load_peak.py watch --out today-36.tsv -- python load_peak.py load --dit DIT --vae VAE \
      --swapped 36 --digests today-36.json > today-36.log
  python load_peak.py table today-36.tsv today-36.log
"""

import argparse
import ctypes
import hashlib
import itertools
import json
import mmap
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GIB = 1 << 30
KIB_PER_GIB = 1 << 20
STATUS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin", "VmSwap")
ROLLUP = (
    "Rss",
    "Anonymous",
    "Locked",
    "Shared_Clean",
    "Shared_Dirty",
    "Private_Clean",
    "Private_Dirty",
)
MEMINFO = ("MemAvailable", "Cached", "Unevictable", "Mlocked")
GROUP = ("current", "peak", "anon", "file")
COLUMNS = (
    "time",
    *STATUS,
    *(f"rollup_{name}" for name in ROLLUP),
    *(f"host_{name}" for name in MEMINFO),
    *(f"group_{name}" for name in GROUP),
)
READERS = (
    "today",
    "madvise",
    "madvise-fadvise",
    "st-pread",
    "pread",
    "pread-pinned",
    "pread-fadvise",
)
MADV_DONTNEED = 4
MARK = re.compile(r"^load_peak: mark (\d+\.\d+) (.*)$")
# A job's own log line: Python's logging time, local, to the millisecond.
LOGGED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) \w+ [\w.]+: (.*)$")


def mark(label):
    print(f"load_peak: mark {time.time():.3f} {label}", flush=True)


# watch


def fields(path, wanted):
    """The kB figures of a /proc file of `Name:   123 kB` lines, by name; {} once it is gone."""
    found = {}
    try:
        with open(path) as file:
            for line in file:
                name, _, rest = line.partition(":")
                if name in wanted:
                    found[name] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return found


def group_of(pid):
    """The process's control group's directory (cgroup v2), where it holds the memory
    controller's files; else None."""
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            number, controllers, path = line.split(":", 2)
            if number == "0" and controllers == "":
                directory = Path("/sys/fs/cgroup") / path.lstrip("/")
                if (directory / "memory.current").exists():
                    return directory
    except (OSError, ValueError):
        pass
    return None


def group_sample(directory):
    """The group's memory.current and memory.peak, and memory.stat's anon and file, in kB."""
    found = {}
    for name in ("current", "peak"):
        try:
            found[name] = int((directory / f"memory.{name}").read_text()) // 1024
        except (OSError, ValueError):
            pass
    try:
        for line in (directory / "memory.stat").read_text().splitlines():
            name, value = line.split()
            if name in ("anon", "file"):
                found[name] = int(value) // 1024
    except (OSError, ValueError):
        pass
    return found


def watch(args):
    command = args.run[1:] if args.run[:1] == ["--"] else args.run
    if not command:
        sys.exit("watch: no command after --")
    child = subprocess.Popen(command)
    directory = group_of(child.pid)
    cost, samples, walked, walks, due = 0.0, 0, 0.0, 0, 0.0
    launched, given_up = time.time(), False
    # Asked to stop (systemd ends a scope one of whose processes the kernel's OOM killer took),
    # the watch still says how the command ended. Each line is written as it is made.
    stopped = []
    signal.signal(signal.SIGTERM, lambda number, frame: stopped.append(number))
    with open(args.out, "w", buffering=1) as out:
        out.write("\t".join(COLUMNS) + "\n")
        while child.poll() is None:
            started = time.time()
            if args.give_up and started - launched > args.give_up and not given_up:
                child.kill()
                given_up = True
            status = fields(f"/proc/{child.pid}/status", STATUS)
            host = fields("/proc/meminfo", MEMINFO)
            group = group_sample(directory) if directory is not None else {}
            rollup = {}
            if started >= due:
                before = time.time()
                rollup = fields(f"/proc/{child.pid}/smaps_rollup", ROLLUP)
                walked, walks = walked + time.time() - before, walks + 1
                due = started + args.rollup
            taken = time.time() - started
            if status:
                row = [f"{started:.3f}"]
                row += [str(status.get(name, "")) for name in STATUS]
                row += [str(rollup.get(name, "")) for name in ROLLUP]
                row += [str(host.get(name, "")) for name in MEMINFO]
                row += [str(group.get(name, "")) for name in GROUP]
                out.write("\t".join(row) + "\n")
                cost, samples = cost + taken, samples + 1
            time.sleep(max(0.0, args.interval - taken))
        out.write(f"# exit {child.returncode}\n")
        if given_up:
            out.write(f"# killed by the watch after {args.give_up:g} s\n")
        if stopped:
            out.write("# the watch was asked to stop (SIGTERM) while the command ran\n")
        if directory is not None:
            try:
                events = " ".join((directory / "memory.events").read_text().split())
                out.write(f"# group {directory.name}: memory.max")
                out.write(f" {(directory / 'memory.max').read_text().strip()}, events {events}\n")
            except OSError:
                pass
        mean = 1000 * (cost - walked) / (samples - walks) if samples > walks else 0.0
        walk = 1000 * walked / walks if walks else 0.0
        out.write(
            f"# {samples} samples, {mean:.2f} ms each on average; {walks} with smaps_rollup,"
            f" {walk:.2f} ms each\n"
        )
    return child.returncode


# load: the readers compared


def dtypes(torch):
    """torch's dtype for each of safetensors' names a SeedVR2 or TransNetV2 file holds."""
    return {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }


class Source:
    """A safetensors file's tensors, one at a time, on the CPU, read one of the ways compared
    (READERS but today's, which is the product's own code).

    tensor(key) gives the tensor, valid until release(key), which gives its memory back where
    the way has something to give; into(key, target) reads its bytes straight into target, a
    part's pinned copy, as the product does."""

    def __init__(self, torch, blockswap, path, how, threads):
        self.torch, self.blockswap, self.path, self.how = torch, blockswap, Path(path), how
        self.mapped = how.startswith("madvise")
        self.drop = how.endswith("fadvise")
        self.threads = max(1, threads)

    def __enter__(self):
        from safetensors import safe_open

        torch = self.torch
        self.raw = open(self.path, "rb", buffering=0)
        length = int.from_bytes(self.raw.read(8), "little")
        header = json.loads(self.raw.read(length))
        header.pop("__metadata__", None)
        self.base = 8 + length
        names = dtypes(torch)
        self.entries = {
            key: (names[entry["dtype"]], tuple(entry["shape"]), *entry["data_offsets"])
            for key, entry in header.items()
        }
        self.file = None
        if self.mapped or self.how == "st-pread":
            backend = "pread" if self.how == "st-pread" else "mmap"
            self.file = safe_open(self.path, framework="pt", device="cpu", backend=backend)
            self.file.__enter__()
            # The file's own order of names, as the product's loader takes them.
            self.keys = list(self.file.keys())
        else:
            # By offset: the file read from its start to its end.
            self.keys = sorted(self.entries, key=lambda key: self.entries[key][2])
        if self.mapped:
            self.span = self._mapping()
            self.libc = ctypes.CDLL(None, use_errno=True)
        largest = max(end - start for _, _, start, end in self.entries.values())
        self.staging = None
        if self.how in ("pread", "pread-fadvise"):
            self.staging = torch.empty(largest, dtype=torch.uint8)
        elif self.how == "pread-pinned":
            self.pinned = self.blockswap.HostCopy(largest, pin=True)
            self.staging = self.pinned.bytes
        self.pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        return self

    def __exit__(self, *error):
        if self.pool is not None:
            self.pool.shutdown()
        if self.file is not None:
            self.file.__exit__(*error)
        self.raw.close()

    def _mapping(self):
        """Where safetensors mapped the file, (start, end) in this process: the one mapping of
        the file's device and inode in /proc/self/maps. madvise is only ever called inside it:
        MADV_DONTNEED on anonymous memory would zero it."""
        stat = os.stat(self.path)
        device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
        found = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) >= 5 and parts[3] == device and int(parts[4]) == stat.st_ino:
                start, end = (int(text, 16) for text in parts[0].split("-"))
                found.append((start, end, parts[1]))
        if len(found) != 1:
            raise RuntimeError(f"{self.path.name}: {len(found)} mappings of the file, not 1")
        start, end, permissions = found[0]
        print(f"load_peak: {self.path.name} mapped {permissions}, {end - start:,} bytes")
        return start, end

    def tensor(self, key):
        dtype, shape, start, end = self.entries[key]
        if self.file is not None:
            return self.file.get_tensor(key)
        size = end - start
        self._read(memoryview(self.staging.numpy())[:size], start)
        return self.staging[:size].view(dtype).view(shape)

    def release(self, key, tensor):
        _, _, start, end = self.entries[key]
        if self.mapped and end > start:
            address = tensor.data_ptr()
            first = address // mmap.PAGESIZE * mmap.PAGESIZE
            last = -(-(address + end - start) // mmap.PAGESIZE) * mmap.PAGESIZE
            if not self.span[0] <= first < last <= self.span[1]:
                raise RuntimeError(f"{key}: not in the file's mapping")
            span = ctypes.c_void_p(first), ctypes.c_size_t(last - first)
            if self.libc.madvise(*span, MADV_DONTNEED):
                raise OSError(ctypes.get_errno(), f"madvise: {os.strerror(ctypes.get_errno())}")
        if self.drop:
            self._dropped(start, end)

    def into(self, key, target):
        _, _, start, end = self.entries[key]
        if len(target) != end - start:
            raise RuntimeError(f"{key} takes {end - start} bytes, not {len(target)}")
        self._read(target, start)
        if self.drop:
            self._dropped(start, end)

    def _read(self, target, start):
        """The file's data from `start` on into target, whole: in one piece, or in as many as
        there are threads, each by pread at its own offset."""
        descriptor, base = self.raw.fileno(), self.base + start

        def piece(first, last):
            done = first
            while done < last:
                got = os.preadv(descriptor, [target[done:last]], base + done)
                if not got:
                    raise RuntimeError(f"{self.path.name}: cut short at {base + done}")
                done += got

        size = len(target)
        if self.pool is None or size < (1 << 22):
            piece(0, size)
            return
        step = -(-size // self.threads)
        starts = range(0, size, step)
        list(self.pool.map(lambda first: piece(first, min(first + step, size)), starts))

    def _dropped(self, start, end):
        os.posix_fadvise(self.raw.fileno(), self.base + start, end - start, os.POSIX_FADV_DONTNEED)


def replace_readers(torch, model, blockswap, how, threads):
    """Replace the loader's two reads for this process: model.load_file, the whole-file read of
    a model with no part away, and blockswap.read, the read of a model with parts."""
    if how == "today":
        return
    if not hasattr(model, "load_file"):
        sys.exit(
            "load_peak: this tree's loader no longer reads a whole model through"
            " safetensors' load_file: its own read is measured as --reader today, alone"
        )
    from safetensors.torch import load_file

    def whole(path, device):
        if how == "st-pread":
            # The product's own call, with safetensors' other backend.
            return load_file(path, device=device, backend="pread")
        state = {}
        with Source(torch, blockswap, path, how, threads) as source:
            for key in source.keys:
                tensor = source.tensor(key)
                state[key] = tensor.to(device)
                source.release(key, tensor)
        return state

    def read(model_, path, device, dtype, part_of, backend=None):
        # blockswap.read's steps, the file read through a Source.
        backend = blockswap.Cuda(device) if backend is None else backend
        parameters = {name for name, _ in model_.named_parameters()}
        with Source(torch, blockswap, path, how, threads) as source:
            layouts = {}
            for key in source.keys:
                name = part_of(key) if key in parameters else None
                if name is not None:
                    found, shape = source.entries[key][:2]
                    final = found if dtype is None or not found.is_floating_point else dtype
                    layouts.setdefault(name, {})[key] = (final, shape)
            parts = blockswap._make(layouts, backend)
            placed = {field.key: (part, field) for part in parts.values() for field in part.fields}
            state = {}
            for key in source.keys:
                recast = dtype is not None and source.entries[key][0].is_floating_point
                if key not in placed:
                    tensor = source.tensor(key)
                    state[key] = tensor.to(device).to(dtype) if recast else tensor.to(device)
                    source.release(key, tensor)
                    continue
                part, field = placed[key]
                state[key] = part.view(part.host.bytes, field)
                if recast:
                    tensor = source.tensor(key)
                    state[key].copy_(tensor.to(device).to(dtype))
                    source.release(key, tensor)
                    continue
                span = part.host.bytes[field.offset : field.offset + field.size]
                source.into(key, memoryview(span.numpy()))
        return state, parts

    model.load_file = whole
    blockswap.read = read


def digests(torch, models):
    """The SHA-256 of every parameter and buffer of the two models, read back from where it
    is, with its dtype and shape: {name: [dtype, shape, digest]}."""
    found = {}
    for label, module in (("dit", models.runner.dit), ("vae", models.runner.vae)):
        named = [*module.named_parameters(), *module.named_buffers()]
        for name, tensor in named:
            data = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
            digest = hashlib.sha256(memoryview(data.numpy())).hexdigest()
            found[f"{label}.{name}"] = [str(tensor.dtype), list(tensor.shape), digest]
    return found


def load(args):
    mark("start")
    from seedvr2x.runtime import memory

    allocator = memory.startup()
    import torch

    from seedvr2x.runtime import blockswap, model
    from seedvr2x.runtime.plan import Plan

    device = torch.device("cuda:0")
    plan = None
    if not args.whole:
        plan = Plan(
            None,
            swapped=args.swapped,
            resident_encode=args.resident_encode,
            resident_decode=args.resident_decode,
            vae_in_dit=not args.vae_away,
        )
    replace_readers(torch, model, blockswap, args.reader, args.threads)
    inner = model._load_weights

    def marked(module, path, *rest, **more):
        which = "vae" if path == Path(args.vae) else "dit"
        mark(f"{which}: reading")
        result = inner(module, path, *rest, **more)
        mark(f"{which}: read")
        return result

    model._load_weights = marked
    memory.hold(device)
    torch.zeros(1, device=device)
    mark("held")
    started = time.perf_counter()
    models = model.load_models(Path(args.dit), Path(args.vae), device, None, plan)
    seconds = time.perf_counter() - started
    mark("loaded")
    pinned = (models.placement.host_bytes if models.placement is not None else 0) + (
        models.vae.host.bytes.numel() if models.vae is not None else 0
    )
    print(
        f"load_peak: reader {args.reader}, threads {args.threads}, plan {plan}, allocator"
        f" {allocator}: loaded in {seconds:.2f} s, {pinned / GIB:.3f} GiB pinned,"
        f" {torch.cuda.memory_allocated(device) / GIB:.3f} GiB allocated on the device",
        flush=True,
    )
    time.sleep(args.settle)
    mark("settled")
    if args.digests:
        found = digests(torch, models)
        with open(args.digests, "w") as file:
            json.dump(found, file, indent=0)
        whole = hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest()
        print(f"load_peak: {len(found)} tensors digested, the digests' own SHA-256 {whole}")
        mark("digested")
    return 0


# table


def marks_of(path):
    """A log's marks, (time, label): load's own, and a job's log lines that tell the load's
    steps (the plan made, the models loaded, a shot's units)."""
    found = []
    with open(path, errors="replace") as file:
        for line in file:
            match = MARK.match(line)
            if match:
                found.append((float(match[1]), match[2]))
                continue
            match = LOGGED.match(line)
            if match:
                when = time.mktime(time.strptime(match[1], "%Y-%m-%d %H:%M:%S"))
                when += int(match[2]) / 1000
                said = match[3]
                for needle, label in (
                    ("SHA-256", "job: a file hashed"),
                    ("torch held to", "job: held"),
                    ("plan: predicted peaks", "job: planned"),
                    ("host memory pinned", "job: pinned"),
                    ("models loaded in", "job: loaded"),
                    ("encoded in", "job: a shot encoded"),
                    ("decoded in", "job: a shot decoded"),
                    ("upscaled in", "job: done"),
                ):
                    if needle in said:
                        found.append((when, label))
                        break
    return found


def samples_of(path):
    rows, notes = [], []
    with open(path) as file:
        columns = file.readline().rstrip("\n").split("\t")
        for line in file:
            if line.startswith("#"):
                notes.append(line[2:].rstrip("\n"))
                continue
            values = line.rstrip("\n").split("\t")
            row = {
                name: (float(value) if name == "time" else int(value)) if value else None
                for name, value in zip(columns, values, strict=False)
            }
            rows.append(row)
    return rows, notes


def gib(value):
    return None if value is None else round(value / KIB_PER_GIB, 3)


def summary(samples, log):
    """A run's figures, in GiB: the peak by the kernel's own high-water mark and by the
    samples, what the sampled peak is made of, and each step's own peak."""
    rows, notes = samples_of(samples)
    marks = marks_of(log)
    if not rows:
        return {"samples": samples, "notes": notes, "rows": 0}
    # smaps_rollup is read once in a while: each sample carries the last one read.
    walked = {name: None for name in rows[0] if name.startswith("rollup_")}
    for row in rows:
        if row["rollup_Rss"] is not None:
            walked = {name: row[name] for name in walked}
        else:
            row.update(walked)
    top = max(rows, key=lambda row: row["VmRSS"])

    def made(row):
        return {
            "rss": gib(row["VmRSS"]),
            "anon": gib(row["RssAnon"]),
            "file": gib(row["RssFile"]),
            "shmem": gib(row["RssShmem"]),
            "locked": gib(row["rollup_Locked"]),
            "vmlck": gib(row["VmLck"]),
            "vmpin": gib(row["VmPin"]),
            "rollup_rss": gib(row["rollup_Rss"]),
            "rollup_anon": gib(row["rollup_Anonymous"]),
            "host_available": gib(row["host_MemAvailable"]),
            "host_cached": gib(row["host_Cached"]),
            "group_current": gib(row["group_current"]),
            "group_anon": gib(row["group_anon"]),
            "group_file": gib(row["group_file"]),
        }

    steps = []
    bounds = [(rows[0]["time"], "before the first mark"), *marks, (rows[-1]["time"] + 1, "end")]
    for (start, label), (end, _) in itertools.pairwise(bounds):
        inside = [row for row in rows if start <= row["time"] < end]
        if not inside:
            continue
        peak = max(inside, key=lambda row: row["VmRSS"])
        steps.append(
            {
                "from": label,
                "seconds": round(end - start, 2) if label != "before the first mark" else None,
                "samples": len(inside),
                "peak": made(peak),
                "last": made(inside[-1]),
            }
        )
    cached = [row["host_Cached"] for row in rows if row["host_Cached"] is not None]
    available = [row["host_MemAvailable"] for row in rows if row["host_MemAvailable"] is not None]
    groups = [row["group_current"] for row in rows if row["group_current"] is not None]
    return {
        "samples": samples,
        "rows": len(rows),
        "notes": notes,
        "hwm": gib(max(row["VmHWM"] for row in rows)),
        "peak": made(top),
        "peak_at": round(top["time"] - rows[0]["time"], 2),
        "peak_file": gib(max(row["RssFile"] for row in rows)),
        "peak_anon": gib(max(row["RssAnon"] for row in rows)),
        "last": made(rows[-1]),
        "host_cached_first": gib(cached[0]) if cached else None,
        "host_cached_most": gib(max(cached)) if cached else None,
        "host_available_least": gib(min(available)) if available else None,
        "group_most": gib(max(groups)) if groups else None,
        "steps": steps,
    }


def table(args):
    if len(args.files) % 2:
        sys.exit("table: SAMPLES.tsv LOG pairs")
    found = [summary(args.files[i], args.files[i + 1]) for i in range(0, len(args.files), 2)]
    if args.json:
        json.dump(found, sys.stdout, indent=1)
        print()
        return 0
    for run in found:
        print(f"## {run['samples']}: {run['rows']} samples")
        for note in run["notes"]:
            print(f"- {note}")
        if not run["rows"]:
            continue
        peak = run["peak"]
        print(
            f"- peak: VmHWM {run['hwm']} GiB; sampled {peak['rss']} at {run['peak_at']} s"
            f" (anonymous {peak['anon']}, file {peak['file']}, shmem {peak['shmem']}; smaps_rollup:"
            f" Rss {peak['rollup_rss']}, Anonymous {peak['rollup_anon']}, Locked {peak['locked']};"
            f" VmLck {peak['vmlck']}, VmPin {peak['vmpin']}); the most file-backed"
            f" {run['peak_file']}, the most anonymous {run['peak_anon']}"
        )
        print(
            f"- host: Cached {run['host_cached_first']} at the start, {run['host_cached_most']}"
            f" at the most; MemAvailable {run['host_available_least']} at the least; the group's"
            f" memory.current at the most: {run['group_most']}"
        )
        print("\n| from | s | peak RSS | anonymous | file | at its end: RSS | anonymous | file |")
        print("|---|---|---|---|---|---|---|---|")
        for step in run["steps"]:
            peak, last = step["peak"], step["last"]
            print(
                f"| {step['from']} | {step['seconds']} | {peak['rss']} | {peak['anon']} |"
                f" {peak['file']} | {last['rss']} | {last['anon']} | {last['file']} |"
            )
        print()
    return 0


# rows

RUN = re.compile(
    r"^(?P<reader>[a-z0-9-]+?)(?:-t(?P<threads>\d+))?-(?P<plan>whole|mid|36)"
    r"(?:-(?P<state>cold|warm))?-r(?P<round>\d+)$"
)
LOADED = re.compile(r"loaded in ([\d.]+) s, ([\d.]+) GiB pinned, ([\d.]+) GiB allocated")


def span(values, digits=2):
    """The least and the most of the values, or the one figure when they round alike."""
    values = [value for value in values if value is not None]
    if not values:
        return "-"
    low, high = min(values), max(values)
    if round(low, digits) == round(high, digits):
        return f"{low:.{digits}f}"
    return f"{low:.{digits}f} to {high:.{digits}f}"


def of_step(run, label, part, key=None):
    for step in run["steps"]:
        if step["from"] == label:
            return step[part] if key is None else step[part][key]
    return None


def grouped(directory, reference=None):
    """The runs of a directory by (plan, cache state, reader), each run its summary with its
    exit code, the group's oom_kill count, the load's seconds and its digests; and the
    reference digests with their name: REFERENCE.json's, or the first of today's reader."""
    directory = Path(directory)
    groups, digests, named = {}, None, None
    for samples in sorted(directory.glob("*.tsv")):
        match = RUN.match(samples.stem)
        log = samples.with_suffix(".log")
        if not match or not log.exists():
            continue
        run = summary(str(samples), str(log))
        if not run.get("rows") or not any(note.startswith("exit") for note in run["notes"]):
            print(f"(left out, still running or empty: {samples.name})", file=sys.stderr)
            continue
        text = log.read_text(errors="replace")
        run["exit"] = next(note for note in run["notes"] if note.startswith("exit")).split()[1]
        events = next((note for note in run["notes"] if "events" in note), "")
        killed = re.search(r"oom_kill (\d+)", events)
        run["oom"] = killed[1] if killed else "-"
        met = re.search(r"events .*\bmax (\d+)", events)
        run["met"] = met[1] if met else "-"
        loaded = LOADED.search(text)
        run["seconds"] = float(loaded[1]) if loaded else None
        found = samples.with_suffix(".json")
        run["digests"] = json.loads(found.read_text()) if found.exists() else None
        reader = match["reader"] + (f" x{match['threads']}" if match["threads"] else "")
        groups.setdefault((match["plan"], match["state"] or "", reader), []).append(run)
        if match["reader"] == "today" and run["digests"] and digests is None:
            digests, named = run["digests"], samples.stem
    if reference:
        digests, named = json.loads(Path(reference).read_text()), reference
    for runs in groups.values():
        for run in runs:
            run["unlike"] = None
            if run["digests"] is not None and digests is not None:
                names = set(run["digests"]) | set(digests)
                run["unlike"] = sum(run["digests"].get(n) != digests.get(n) for n in names)
    return groups, digests, named


def rows(args):
    groups, digests, named = grouped(args.directory, args.reference)
    print(f"reference digests: {named} ({len(digests or {})} tensors)\n")
    print(
        "| plan | cache | reader | runs | peak (VmHWM) | at the sampled peak: anonymous | file |"
        " the most file | resident once loaded | Cached's rise | load s | DiT's read s |"
        " tensors unlike the reference | exit | the group's most | its limit met | its oom_kill |"
    )
    print("|" + "---|" * 17)
    order = {"whole": 0, "mid": 1, "36": 2}
    for key in sorted(groups, key=lambda key: (order[key[0]], key[1], key[2] != "today", key[2])):
        runs = groups[key]
        unlike = ["-" if run["unlike"] is None else str(run["unlike"]) for run in runs]
        rise = [
            run["host_cached_most"] - run["host_cached_first"]
            for run in runs
            if run["host_cached_most"] is not None
        ]
        print(
            f"| {key[0]} | {key[1] or '-'} | {key[2]} | {len(runs)} |"
            f" {span([run['hwm'] for run in runs])} |"
            f" {span([run['peak']['anon'] for run in runs])} |"
            f" {span([run['peak']['file'] for run in runs])} |"
            f" {span([run['peak_file'] for run in runs])} |"
            f" {span([of_step(run, 'loaded', 'last', 'rss') for run in runs])} |"
            f" {span(rise, 1)} | {span([run['seconds'] for run in runs], 1)} |"
            f" {span([of_step(run, 'dit: reading', 'seconds') for run in runs], 1)} |"
            f" {', '.join(unlike)} | {', '.join(run['exit'] for run in runs)} |"
            f" {span([run['group_most'] for run in runs])} |"
            f" {', '.join(run['met'] for run in runs)} | {', '.join(run['oom'] for run in runs)} |"
        )
    return 0


# same


def same(args):
    with open(args.files[0]) as file:
        first = json.load(file)
    different = 0
    for path in args.files[1:]:
        with open(path) as file:
            other = json.load(file)
        names = sorted(set(first) | set(other))
        unlike = [name for name in names if first.get(name) != other.get(name)]
        different += len(unlike)
        print(
            f"{path}: {len(other)} tensors, {len(unlike)} unlike {args.files[0]}'s {len(first)}"
            + (f": {unlike[:5]}" if unlike else "")
        )
    return 1 if different else 0


# cache


def in_cache(path):
    """The bytes of the file at path the page cache holds, by mincore on a mapping of it, no
    page touched: (held, size)."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [ctypes.c_int] * 3 + [ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    size = os.path.getsize(path)
    if not size:
        return 0, 0
    with open(path, "rb") as file:
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, file.fileno(), 0)
    if address in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), f"mmap: {os.strerror(ctypes.get_errno())}")
    pages = -(-size // mmap.PAGESIZE)
    held = (ctypes.c_ubyte * pages)()
    try:
        if libc.mincore(address, size, held):
            raise OSError(ctypes.get_errno(), f"mincore: {os.strerror(ctypes.get_errno())}")
    finally:
        libc.munmap(address, size)
    return sum(byte & 1 for byte in bytes(held)) * mmap.PAGESIZE, size


def cache(args):
    for path in args.files:
        if args.drop:
            with open(path, "rb") as file:
                os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        held, size = in_cache(path)
        print(
            f"load_peak: cache: {os.path.basename(path)}: {held / GIB:.3f} of {size / GIB:.3f}"
            f" GiB in the page cache{', asked to drop it' if args.drop else ''}",
            flush=True,
        )
    return 0


# read


def read(args):
    path = Path(args.file)
    with open(path, "rb") as raw:
        length = int.from_bytes(raw.read(8), "little")
        header = json.loads(raw.read(length))
    header.pop("__metadata__", None)
    spans = sorted(tuple(entry["data_offsets"]) for entry in header.values())
    base, data = 8 + length, sum(end - start for start, end in spans)
    buffer = memoryview(bytearray(max(end - start for start, end in spans)))
    threads = max(1, args.threads)
    shared = os.open(path, os.O_RDONLY)
    local = threading.local()
    opened = []

    def descriptor():
        if not args.own:
            return shared
        if not hasattr(local, "descriptor"):
            local.descriptor = os.open(path, os.O_RDONLY)
            opened.append(local.descriptor)
        return local.descriptor

    def piece(bounds):
        first, last, offset = bounds
        done = first
        while done < last:
            got = os.preadv(descriptor(), [buffer[done:last]], offset + done)
            if not got:
                raise RuntimeError(f"{path.name}: cut short at {offset + done}")
            done += got

    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    for _ in range(args.times):
        if args.drop:
            os.posix_fadvise(shared, 0, 0, os.POSIX_FADV_DONTNEED)
        held = in_cache(path)[0]
        started = time.perf_counter()
        for start, end in spans:
            size = end - start
            if pool is None or size < (1 << 22):
                piece((0, size, base + start))
                continue
            step = -(-size // threads)
            bounds = [
                (first, min(first + step, size), base + start) for first in range(0, size, step)
            ]
            list(pool.map(piece, bounds))
        seconds = time.perf_counter() - started
        print(
            f"load_peak: read: {path.name}: {data / GIB:.3f} GiB in {seconds:.2f} s,"
            f" {data / 1e6 / seconds:.0f} MB/s; {threads} at once,"
            f" {'a descriptor each' if args.own else 'one descriptor'};"
            f" {held / GIB:.3f} GiB of the file in the page cache before",
            flush=True,
        )
    if pool is not None:
        pool.shutdown()
    for each in [shared, *opened]:
        os.close(each)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="mode", required=True)
    watching = commands.add_parser("watch")
    watching.add_argument("--out", required=True)
    watching.add_argument("--interval", type=float, default=0.05)
    watching.add_argument("--rollup", type=float, default=1.0)
    watching.add_argument("--give-up", type=float, default=0.0)
    watching.add_argument("run", nargs=argparse.REMAINDER)
    loading = commands.add_parser("load")
    loading.add_argument("--dit", required=True)
    loading.add_argument("--vae", required=True)
    loading.add_argument("--swapped", type=int, default=0)
    loading.add_argument("--resident-encode", type=int, default=None)
    loading.add_argument("--resident-decode", type=int, default=None)
    loading.add_argument("--vae-away", action="store_true")
    loading.add_argument("--whole", action="store_true")
    loading.add_argument("--reader", choices=READERS, default="today")
    loading.add_argument("--threads", type=int, default=1)
    loading.add_argument("--settle", type=float, default=1.0)
    loading.add_argument("--digests")
    tabling = commands.add_parser("table")
    tabling.add_argument("--json", action="store_true")
    tabling.add_argument("files", nargs="+")
    rowing = commands.add_parser("rows")
    rowing.add_argument("directory")
    rowing.add_argument("reference", nargs="?")
    comparing = commands.add_parser("same")
    comparing.add_argument("files", nargs="+")
    caching = commands.add_parser("cache")
    caching.add_argument("--drop", action="store_true")
    caching.add_argument("files", nargs="+")
    reading = commands.add_parser("read")
    reading.add_argument("file")
    reading.add_argument("--threads", type=int, default=1)
    reading.add_argument("--own", action="store_true")
    reading.add_argument("--drop", action="store_true")
    reading.add_argument("--times", type=int, default=1)
    args = parser.parse_args()
    modes = {"watch": watch, "load": load, "table": table, "rows": rows, "same": same}
    modes.update(cache=cache, read=read)
    return modes[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
