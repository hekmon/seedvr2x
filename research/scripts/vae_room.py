#!/usr/bin/env python3
"""The VAE's room (seedvr2x, milestone 3): what a VAE pass needs between its peak and the memory
free, and what cures it. seedvr2x's planner holds every VAE pass a room further under its budget
(runtime/planner.py, VAE_ROOM), because passes packed to the margin ran out of memory at one large
request while the allocator's pool held more than was asked. This script runs seedvr2x's own
command line on cards emulated by vram_cap.py's ballast, in a process where the planner's room
(and ENCODE_EXTRA) is set as a study needs before the plan is made, with a cure applied by a patch
of the process, and with probes that say, pass by pass and slice by slice, what torch had
allocated, what its pool held and what the card had free. Nothing of seedvr2x is edited.

  run    one seedvr2x run, in this process (the launcher). In order: the emulated card (--card N,
         vram_cap.py's setup, its ballast held to the end; `bare` for none); the planner's
         constants (--room, --encode-extra, in GiB); the cure (--cure); the probes; then
         seedvr2x's command line with the arguments after `--`, its output in --work/NAME. Its
         first lines say the tree it was told (--tree), the constants and the cure in force. At
         the end: --records/NAME.json (the plan, every unit and slice, the failure if any, the
         frames' checksums), --records/NAME.crc32 (the checksum files), and the output directory
         removed (--keep keeps it).
         --stop-after encode|windows asks the run's own stop before the first window or before
         the decode (exit 130), for a pass run alone; --seed-dir DIR copies a job so stopped into
         the output directory first, which the run then resumes: a decode alone.
         --dit-unjudged leaves the DiT's phase out of the plan (its peak taken as 0), for a pass
         alone whose pinned shape must plan the same on every card (floor.py's rungs). The record
         keeps the plan as the planner made it in the process ("planned"): a resumed job's
         manifest holds its earlier plan until the run finishes its segment.
  queue  FILE: the runs of a queue file, one hold of the GPU's lock each (`flock LOCK`), each
         one's output in --records/NAME.log. A line is `NAME | arguments of run`; a line
         `common | arguments` gives arguments every later run takes first; `#` starts a comment.
         The file is read again before each run: lines may be appended while it runs. Before each
         run: waits while FILE.pause exists, stops if FILE.skip exists, and checks the tree of
         the code (--copy, --want: git's tree of the directory, nothing written in it). One line
         per run in FILE.progress. --wait-for PATH waits for that file before the first run.
  pool   the allocator alone, no model: on a card emulated at --card GB, a block of --block GiB
         allocated and freed, then a small tensor allocated, which the pool carves from the
         freed block or not (by their addresses); the pool emptied, which gives the block back
         or can't; then one request of all that is free less --left GiB, as a pass's largest
         request finds the card. Twice, the small tensor made after the block is freed, and
         before. Each step's allocated and reserved bytes and the card's free memory. A queue
         line `NAME | pool ...` runs it.
  conv   one convolution alone, no model: a 3x3x3 convolution of 256 channels in bfloat16 over
         one piece of a decode's input (the VAE cuts a large convolution in pieces of under 0.5
         GiB), run with the card's memory free, then again and again with less of it left (a
         filler tensor takes the rest): at each step, whether its output is still the first
         one's, bit for bit, the seconds it took and what torch allocated at the most. torch
         tries cuDNN's engines in turn and swallows a refused workspace: this shows whether a
         convolution's bits follow the memory left, and from where; and, the memory free again
         at the end, whether the process then gives the first bits back.
  plan   no GPU: what the planner makes of cases under a room, RES:CARD:MODE as
         planner_matrix.py names them (or RES:CARD:FRAMES), with pins: each pass's tiles, what
         of the DiT stays, the predicted peaks and what each pass leaves under the budget.
         RES:least:FRAMES searches the smallest card, by 0.05 GB, that takes the pinned job.
  table  RECORDS_DIR [...]: the Markdown tables of the runs' records, one row per VAE pass.

The probes (run):
- every unit (a shot's encode, a window, a decode), by run.py's own start and end of a unit:
  torch's peak (max_memory_allocated, the "VRAM peak" seedvr2x logs) and the pool's
  (max_memory_reserved: under cudaMallocAsync torch reads both from the driver's memory pool,
  its used and its reserved memory), what was allocated and reserved before, the least memory
  the card had free during the unit (NVML's figure, nvidia-smi's, sampled every 20 ms by a
  thread of its own), and the driver's own (cudaMemGetInfo) at its start; and torch's
  allocator counters at its start and end (memory_stats's num_ooms, num_alloc_retries,
  num_oom_rejections, num_sync_all_streams), which torch's own allocator moves when it refuses
  or retries a request, and cudaMallocAsync leaves at 0;
- every slice of a VAE pass (the VAE's _encode and _decode calls): the same readings at its
  start and end, in the log as `vae_room: slice` lines, so that torch's own lines about its
  retries ("[cudaMallocAsync] recovered from an allocation failure by trimming the pool and
  retrying") fall between the slices they belong to;
- every concatenation of 0.25 GiB or more in the VAE's causal convolutions
  (causal_inflation_lib's torch.cat calls, by call site and size): how many, and the least
  memory the driver had free before one, with what torch had allocated and reserved then;
- a request refused (torch.OutOfMemoryError): where (the call site and the traceback's frames
  in seedvr2x), the request's size, the exception's own figures, torch's counters and
  memory_stats, the driver's and NVML's free memory then and over the second before.

The cures (--cure), each a patch of this process:
  enter    the pool emptied (torch.cuda.synchronize, then empty_cache) whenever a unit enters a
           VAE phase (after Models.enter);
  slice    the pool emptied before every slice of a VAE pass (each _encode and _decode call);
  request  the pool emptied before each concatenation of memory_limit_conv's pieces of
           --request-floor GiB or more (default 1.0), a pass's largest requests;
  output   memory_limit_conv's pieces written into one output allocated before the first of
           them is computed, in place of their concatenation at the end; with --output-floor
           GIB, only where that output is so large, a smaller one concatenated as before;
  output-norm  that, and a group norm's chunks (causal_norm_wrapper) written the same way.
Several at once: their names joined by `+` (output+slice).

Run with the python of seedvr2x's environment, planner_matrix.py beside this file, under the
GPU's lock (queue takes it). Sizes in bytes in the records, GiB (2**30) in the tables."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import linecache
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

GIB = 1024**3
BIG = GIB // 4  # the concatenations counted, in bytes
TRIM = "recovered from an allocation failure"
PHASES = ("encode", "dit", "decode")
CURES = {"enter", "slice", "request", "output", "output-norm"}


def say(text: str) -> None:
    """A line of this script's in the run's log, on the stream torch's own warnings take."""
    print(f"vae_room: {text}", file=sys.stderr, flush=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- the probes


class _NvmlMemory(ctypes.Structure):
    _fields_ = (
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


class Card:
    """The card's free memory by NVML, nvidia-smi's figure, sampled by a thread of its own: the
    least seen since a mark, and the last second's samples. NVML alone: nothing of CUDA or torch
    is called from the thread."""

    PERIOD = 0.02

    def __init__(self) -> None:
        # Without NVML every reading is -1, said once: the run goes on.
        self.nvml = None
        self.handle = ctypes.c_void_p()
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1")
            if nvml.nvmlInit_v2() != 0:
                raise OSError("NVML's initialisation failed")
            if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                raise OSError("NVML knows no device 0")
            self.nvml = nvml
        except OSError as error:
            say(f"NO NVML ({error}): the card's free memory isn't read, -1 in its place")
        self.least = {"unit": self.free(), "slice": self.free()}
        self.last: deque[tuple[float, int]] = deque(maxlen=int(1.5 / self.PERIOD))
        self.ended = False
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def free(self) -> int:
        memory = _NvmlMemory()
        if self.nvml is None:
            return -1
        if self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) != 0:
            return -1
        return int(memory.free)

    def _sample(self) -> None:
        while not self.ended:
            free = self.free()
            if free >= 0:
                self.last.append((time.monotonic(), free))
                for mark in self.least:
                    if free < self.least[mark]:
                        self.least[mark] = free
            time.sleep(self.PERIOD)

    def mark(self, name: str) -> None:
        self.least[name] = self.free()

    def taken(self, name: str) -> int:
        """The least free memory since mark(name), the reading now included."""
        return min(self.least[name], self.free())


class Probes:
    """What a run's probes found: its record (the JSON's `units` and `oom`)."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.torch = torch
        self.args = args
        self.card = Card()
        self.units: list[dict] = []
        self.unit: dict | None = None
        self.oom: dict | None = None
        self.emptied = {"count": 0, "seconds": 0.0}
        self.broken = 0
        self.planned: dict | None = None  # the run's own plan, as the planner made it here

    def quiet(self, probe, *given):
        """A probe's own work, which must never end the run it watches: what fails is said,
        with its traceback the first time, and the run goes on without that reading."""
        try:
            return probe(*given)
        except Exception:
            if not self.broken:
                traceback.print_exc()
            self.broken += 1
            say(f"A PROBE FAILED ({probe.__name__}, {self.broken} so far): its reading is missing")
            return None

    # -- readings

    def counters(self) -> dict:
        """torch's counters now, the driver's free memory and NVML's."""
        torch = self.torch
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "driver_free": torch.cuda.mem_get_info()[0],
            "card_free": self.card.free(),
        }

    def refusals(self) -> dict:
        """torch's allocator counters that a refused allocation moves (memory_stats): torch's
        own allocator counts them; cudaMallocAsync leaves every one at 0."""
        stats = self.torch.cuda.memory_stats()
        names = ("num_ooms", "num_alloc_retries", "num_oom_rejections", "num_sync_all_streams")
        return {name: stats.get(name) for name in names}

    def empty(self) -> None:
        """The pool emptied, as a cure does it: the work queued waited for, then what the pool
        holds free given back to the driver."""
        started = time.monotonic()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        self.emptied["count"] += 1
        self.emptied["seconds"] += time.monotonic() - started

    # -- units

    def begin(self) -> None:
        self.quiet(self._begin)

    def entered(self, phase: str) -> None:
        self.quiet(self._entered, phase)

    def end(self, failed: bool = False) -> None:
        self.quiet(self._end, failed)
        self.unit = None

    def _begin(self) -> None:
        self.unit = {
            "number": len(self.units) + 1,
            "phase": None,
            "began": now(),
            "before": self.counters(),
            "slices": [],
            "requests": {},
            "emptied": dict(self.emptied),
            "counters_before": self.refusals(),
        }
        self.units.append(self.unit)
        self.card.mark("unit")
        self._started = time.monotonic()
        say(f"unit {self.unit['number']} begins: {self._said(self.unit['before'])}")

    def _entered(self, phase: str) -> None:
        if self.unit is not None and self.unit["phase"] is None:
            self.unit["phase"] = phase
            self.unit["entered"] = self.counters()
            say(f"unit {self.unit['number']} entered {phase}: {self._said(self.unit['entered'])}")

    def _end(self, failed: bool) -> None:
        unit, torch = self.unit, self.torch
        if unit is None:
            return
        unit["seconds"] = round(time.monotonic() - self._started, 1)
        unit["peak_allocated"] = torch.cuda.max_memory_allocated()
        unit["peak_reserved"] = torch.cuda.max_memory_reserved()
        unit["card_free_least"] = self.card.taken("unit")
        unit["after"] = self.counters()
        unit["failed"] = failed
        unit["counters_after"] = self.refusals()
        before = unit.pop("emptied")
        unit["emptied"] = {
            "count": self.emptied["count"] - before["count"],
            "seconds": round(self.emptied["seconds"] - before["seconds"], 3),
        }
        say(
            f"unit {unit['number']} {unit['phase']} {'FAILED' if failed else 'ends'}:"
            f" {unit['seconds']} s, peak allocated {unit['peak_allocated'] / GIB:.3f},"
            f" peak reserved {unit['peak_reserved'] / GIB:.3f}, card free least"
            f" {unit['card_free_least'] / GIB:.3f}"
        )

    @staticmethod
    def _said(found: dict) -> str:
        return (
            f"allocated {found['allocated'] / GIB:.3f} reserved {found['reserved'] / GIB:.3f}"
            f" driver free {found['driver_free'] / GIB:.3f}"
            f" card free {found['card_free'] / GIB:.3f}"
        )

    # -- slices

    def slice(self, kind: str, call):
        """One slice of a VAE pass, `call` the VAE's own _encode or _decode."""
        if self.unit is None:
            return call()
        if "slice" in self.args.cures:
            self.empty()
        before = self.quiet(self._slice_before)
        try:
            result = call()
        except self.torch.OutOfMemoryError as error:
            self.refused(error, f"{kind} slice", None)
            raise
        if before is not None:
            self.quiet(self._slice_after, kind, *before)
        return result

    def _slice_before(self) -> tuple[dict, float]:
        before = self.counters()
        self.card.mark("slice")
        return before, time.monotonic()

    def _slice_after(self, kind: str, before: dict, started: float) -> None:
        torch, unit = self.torch, self.unit
        assert unit is not None
        found = {
            "number": len(unit["slices"]) + 1,
            "seconds": round(time.monotonic() - started, 2),
            "allocated": before["allocated"],
            "reserved": before["reserved"],
            "driver_free": before["driver_free"],
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "card_free_least": self.card.taken("slice"),
        }
        unit["slices"].append(found)
        say(
            f"slice {kind} {found['number']}: {found['seconds']} s, before: allocated"
            f" {found['allocated'] / GIB:.3f} reserved {found['reserved'] / GIB:.3f} driver free"
            f" {found['driver_free'] / GIB:.3f}; the unit's peaks so far: allocated"
            f" {found['peak_allocated'] / GIB:.3f} reserved {found['peak_reserved'] / GIB:.3f};"
            f" card free least {found['card_free_least'] / GIB:.3f}"
        )

    # -- large requests

    def request(self, site: str, size: int, make, floor: int | None = None):
        """One large request of the VAE's causal convolutions, `make` the call that allocates
        it: counted by site and size, with the least memory the driver had free before one.
        With floor, the pool is emptied first when the request is that large (--cure request)."""
        if floor is not None and size >= floor:
            self.empty()
        if self.unit is not None:
            self.quiet(self._requested, site, size)
        try:
            return make()
        except self.torch.OutOfMemoryError as error:
            self.refused(error, site, size)
            raise

    def _requested(self, site: str, size: int) -> None:
        torch, unit = self.torch, self.unit
        assert unit is not None
        free = torch.cuda.mem_get_info()[0]
        allocated = torch.cuda.memory_allocated()
        seen = unit["requests"].setdefault(
            f"{site} {size}", {"count": 0, "driver_free_least": free, "allocated_most": 0}
        )
        seen["count"] += 1
        # What torch has allocated once the request is given, at the most: the site a pass
        # peaks at has the largest.
        seen["allocated_most"] = max(seen["allocated_most"], allocated + size)
        if free <= seen["driver_free_least"]:
            seen["driver_free_least"] = free
            seen["allocated"] = allocated
            seen["reserved"] = torch.cuda.memory_reserved()

    def refused(self, error: BaseException, site: str, size: int | None) -> None:
        """A request torch refused: recorded once, where it was first caught, every tensor of
        the failing call still alive (the traceback holds its frames)."""
        if self.oom is None:
            self.quiet(self._refused, error, site, size)

    def _refused(self, error: BaseException, site: str, size: int | None) -> None:
        torch = self.torch
        # The readings first, before anything else moves.
        counters = self.counters()
        peaks = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        last = list(self.card.last)
        stats = {key: value for key, value in torch.cuda.memory_stats().items() if value}
        frames = [
            f"{Path(frame.filename).name}:{frame.lineno} {frame.name}: {(frame.line or '').strip()}"
            for frame in traceback.extract_tb(error.__traceback__)
            if "seedvr2x" in frame.filename and "site-packages" not in frame.filename
        ]
        self.oom = {
            "unit": None if self.unit is None else self.unit["number"],
            "phase": None if self.unit is None else self.unit["phase"],
            "slice": None if self.unit is None else len(self.unit["slices"]) + 1,
            "caught_at": site,
            "request": size,
            "message": str(error),
            "frames": frames,
            **counters,
            "peak_allocated": peaks[0],
            "peak_reserved": peaks[1],
            "memory_stats": stats,
            "card_free_last_second": [
                [round(1000 * (when - last[-1][0])), free] for when, free in last[-50:]
            ],
        }
        say(
            f"REFUSED at {site}, request {'?' if size is None else f'{size / GIB:.3f}'} GiB, in"
            f" unit {self.oom['unit']} ({self.oom['phase']}), slice {self.oom['slice']}:"
            f" {self._said(self.oom)}"
        )
        for line in str(error).splitlines():
            say(f"  torch: {line}")
        for frame in frames:
            say(f"  at {frame}")
        say(f"  memory_stats (non-zero): {json.dumps(stats)}")


def install(probes: Probes, args: argparse.Namespace) -> list[str]:
    """Patch this process: the probes, the cure and the stop asked. What was patched, in words."""
    import torch

    from seedvr2x.runtime import model
    from seedvr2x.runtime import run as running
    from seedvr2x.vendor.models.video_vae_v3.modules import attn_video_vae, causal_inflation_lib

    done = []
    floor = round(args.request_floor * GIB) if "request" in args.cures else None

    # A unit's start and end, run.py's own: the peaks are read where seedvr2x reads its own,
    # before the next unit resets them.
    started, spent = running._started, running._spent

    def unit_started(models):
        probes.begin()
        return started(models)

    def unit_spent(models, since):
        said = spent(models, since)
        probes.end()
        return said

    running._started, running._spent = unit_started, unit_spent

    # The run's own plan, as the planner makes it in this process: a resumed job's manifest
    # keeps its earlier plan until the run finishes its segment, so that a decode refused
    # before then leaves the seed's plan there, made on another card.
    from seedvr2x.runtime import planner as planning

    made = planning.plan

    def planned_here(*given, **named):
        plan, peaks = made(*given, **named)
        found = plan.record()
        found["memory"].update(peaks.record())
        probes.planned = found
        return plan, peaks

    planning.plan = planned_here

    enter = model.Models.enter

    def entered(self, phase):
        enter(self, phase)
        if "enter" in args.cures and phase in ("encode", "decode"):
            probes.empty()
        probes.entered(phase)

    model.Models.enter = entered
    if "enter" in args.cures:
        done.append("the pool emptied after Models.enter of a VAE phase, every unit")
    if "slice" in args.cures:
        done.append("the pool emptied before every slice of a VAE pass")

    vae = attn_video_vae.VideoAutoencoderKL
    encode, decode = vae._encode, vae._decode

    def encode_slice(self, *given, **named):
        return probes.slice("encode", lambda: encode(self, *given, **named))

    def decode_slice(self, *given, **named):
        return probes.slice("decode", lambda: decode(self, *given, **named))

    vae._encode, vae._decode = encode_slice, decode_slice

    # The causal convolutions' concatenations: the module's own `torch` seen through a stand-in
    # whose cat counts the large ones, and empties the pool before memory_limit_conv's last.
    library = causal_inflation_lib

    class Seen:
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def cat(*given, **named):
            tensors = given[0] if given else named["tensors"]
            size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            if size < BIG:
                return torch.cat(*given, **named)
            frame = sys._getframe(1)
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
            site = f"{frame.f_code.co_name}:{frame.f_lineno}"
            pieces = line.startswith("output = torch.cat(x, split_dim)")
            return probes.request(
                site, size, lambda: torch.cat(*given, **named), floor if pieces else None
            )

    library.torch = Seen()
    if "request" in args.cures:
        done.append(
            f"the pool emptied before each concatenation of memory_limit_conv's pieces of"
            f" {args.request_floor} GiB or more"
        )

    if args.cures & {"output", "output-norm"}:
        library.InflatedCausalConv3d.memory_limit_conv = written_into_one(
            probes, library, round(args.output_floor * GIB)
        )
        done.append(
            "memory_limit_conv's pieces written into one output allocated before the first"
            + (f", an output of {args.output_floor} GiB or more" if args.output_floor else "")
        )
    if "output-norm" in args.cures:
        vendored = library.causal_norm_wrapper
        cured = normed_into_one(probes, library)
        # Wherever the vendored modules took the function by name.
        holders = [
            module
            for name, module in list(sys.modules.items())
            if name.startswith("seedvr2x.vendor")
            and getattr(module, "causal_norm_wrapper", None) is vendored
        ]
        for module in holders:
            module.causal_norm_wrapper = cured
        done.append(
            f"causal_norm_wrapper's chunks written into one output allocated before the first"
            f" (in {len(holders)} modules)"
        )

    # A failure outside a slice (a tiled pass's sums, the colour correction): caught at the pass.
    def guarded(pass_):
        def call(*given, **named):
            try:
                return pass_(*given, **named)
            except torch.OutOfMemoryError as error:
                probes.refused(error, pass_.__name__, None)
                probes.end(failed=True)
                raise

        return call

    running.encode_shot = guarded(running.encode_shot)
    running.decode_shot = guarded(running.decode_shot)

    if args.stop_after:
        begin = running._begin
        before = {"encode": "'s window ", "windows": "'s decode and write"}[args.stop_after]

        def begun(stop, unit, cost):
            if stop is not None and before in unit:
                say(f"the stop asked before {unit} (--stop-after {args.stop_after})")
                stop.asked = True
            return begin(stop, unit, cost)

        running._begin = begun
        done.append(f"the run's stop asked after its {args.stop_after}")
    return done


def written_into_one(probes: Probes, library, floor: int = 0):
    """memory_limit_conv with its pieces written into one output allocated before the first of
    them is computed, where the vendored one concatenates them at the end (--cure output): the
    same splits, the same convolutions in the same order, the same values copied. The output's
    shape is the convolution's over the whole padded input; a piece that doesn't fit its place
    in it, or an output left partly unwritten, is a RuntimeError. With floor, only an output of
    that many bytes or more: a smaller one is concatenated as the vendored code does it."""
    import torch
    from torch.nn import Conv3d

    vendored = library.InflatedCausalConv3d.memory_limit_conv
    get_cache_size = library.get_cache_size

    def memory_limit_conv(self, x, *, split_dim=3, padding=(0, 0, 0, 0, 0, 0), prev_cache=None):
        if math.isinf(self.memory_limit) or tuple(self.dilation) != (1, 1, 1):
            return vendored(self, x, split_dim=split_dim, padding=padding, prev_cache=prev_cache)
        # The vendored code's own arithmetic, to the letter: the same splits.
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            return vendored(self, x, split_dim=split_dim, padding=padding, prev_cache=prev_cache)

        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        # The output, before any piece: the convolution's over the whole padded input.
        padded = [int(side) for side in shape]
        made = [padded[0], self.out_channels]
        made += [(padded[2 + i] - self.kernel_size[i]) // self.stride[i] + 1 for i in range(3)]
        size = math.prod(made) * x.element_size()
        if size < floor:
            # A small output: the vendored concatenation (--output-floor).
            return vendored(self, x, split_dim=split_dim, padding=padding, prev_cache=prev_cache)
        output = probes.request(
            "memory_limit_conv:output",
            size,
            lambda: torch.empty(made, dtype=x.dtype, device=x.device),
        )
        filled = 0

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        cache = None
        for idx in range(len(x)):
            if prev_cache is not None:
                # Through the module's own name, counted as the vendored one's are.
                x[idx] = library.torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)
                )

            piece = self.memory_limit_conv(
                x[idx], split_dim=split_dim + 1, padding=padding, prev_cache=cache
            )
            x[idx] = None
            cache = next_cache

            length = piece.size(split_dim)
            place = list(output.shape)
            place[split_dim] = length
            if (
                list(piece.shape) != place
                or piece.dtype != output.dtype
                or filled + length > output.size(split_dim)
            ):
                raise RuntimeError(
                    f"memory_limit_conv: a piece of {tuple(piece.shape)} {piece.dtype} at"
                    f" {filled} along dimension {split_dim} of an output of"
                    f" {tuple(output.shape)} {output.dtype}"
                )
            output.narrow(split_dim, filled, length).copy_(piece)
            filled += length
            del piece
        if filled != output.size(split_dim):
            raise RuntimeError(
                f"memory_limit_conv: {filled} of {output.size(split_dim)} written along"
                f" dimension {split_dim}"
            )
        return output

    assert Conv3d in library.InflatedCausalConv3d.__mro__
    return memory_limit_conv


def normed_into_one(probes: Probes, library):
    """causal_norm_wrapper with a group norm's chunks written into one output allocated before
    the first of them is computed, where the vendored one concatenates them at the end (--cure
    output-norm): the same chunks, normalised by the same calls, the same values copied."""
    from einops import rearrange
    from torch import nn
    from torch.nn import functional

    vendored = library.causal_norm_wrapper

    def causal_norm_wrapper(norm_layer, x):
        if not (isinstance(norm_layer, nn.GroupNorm) and x.ndim == 5):
            return vendored(norm_layer, x)
        t = x.size(2)
        flat = rearrange(x, "b c t h w -> (b t) c h w")
        memory_occupy = flat.numel() * flat.element_size() / 1024**3
        if not memory_occupy > library.get_norm_limit():
            return vendored(norm_layer, x)
        num_chunks = min(4 if flat.element_size() == 2 else 2, norm_layer.num_groups)
        assert norm_layer.num_groups % num_chunks == 0
        num_groups_per_chunk = norm_layer.num_groups // num_chunks
        chunks = flat.chunk(num_chunks, dim=1)
        weights = norm_layer.weight.chunk(num_chunks, dim=0)
        biases = norm_layer.bias.chunk(num_chunks, dim=0)
        output = probes.request(
            "causal_norm_wrapper:output",
            flat.numel() * flat.element_size(),
            lambda: flat.new_empty(flat.shape),
        )
        filled = 0
        for chunk, w, b in zip(chunks, weights, biases, strict=True):
            piece = functional.group_norm(chunk, num_groups_per_chunk, w, b, norm_layer.eps)
            if piece.dtype != output.dtype or piece.shape != chunk.shape:
                raise RuntimeError(
                    f"causal_norm_wrapper: a chunk of {tuple(chunk.shape)} normalised as"
                    f" {tuple(piece.shape)} {piece.dtype}, the output {output.dtype}"
                )
            output.narrow(1, filled, piece.size(1)).copy_(piece)
            filled += piece.size(1)
            del piece
        if filled != output.size(1):
            raise RuntimeError(f"causal_norm_wrapper: {filled} of {output.size(1)} channels")
        return rearrange(output, "(b t) c h w -> b c t h w", t=t)

    return causal_norm_wrapper


# ---------------------------------------------------------------- run


def set_planner(planner, args: argparse.Namespace) -> dict[str, int]:
    """The planner's constants as a run or a dry run asks them, set in this process: VAE_ROOM
    (--room) and ENCODE_EXTRA (--encode-extra), in GiB. With --room-encode or --room-decode, a
    room by the pass: the two functions the planner judges a VAE pass by, _vae_need and
    Peaks.need, then take the pass's own room where they take VAE_ROOM. The rooms in force, in
    bytes, by phase."""
    planner.VAE_ROOM = round(args.room * GIB)
    if hasattr(planner, "_ROOM_HELD"):
        # A refusal's words for the room, made when the module was imported, where it has them.
        planner._ROOM_HELD = (
            f", {planner.VAE_ROOM / GIB:.2f} of them the room a VAE pass is held to"
        )
    if args.encode_extra is not None:
        planner.ENCODE_EXTRA = round(args.encode_extra * GIB)
    if getattr(args, "dit_unjudged", False):
        # A pass alone runs no window: the DiT's phase isn't judged, its peak taken as 0, so
        # that a pinned shape plans the same on every card (floor.py's rungs).
        def nothing(*given, **named):
            return 0

        for name in ("dit_peak", "_dit_least", "dit_draw"):
            if hasattr(planner, name):
                setattr(planner, name, nothing)
    rooms = {"encode": planner.VAE_ROOM, "decode": planner.VAE_ROOM}
    for phase, given in (("encode", args.room_encode), ("decode", args.room_decode)):
        if given is not None:
            rooms[phase] = round(given * GIB)
    if rooms["encode"] != planner.VAE_ROOM or rooms["decode"] != planner.VAE_ROOM:
        peak = planner._vae_peak

        def need(job, phase, tiles, resident, swapped):
            return peak(job, phase, tiles, resident, swapped) + rooms[phase]

        def judged(self, phase):
            return self.peak(phase) + (0 if phase == "dit" else rooms[phase])

        planner._vae_need = need
        planner.Peaks.need = judged
    return rooms


def launch(args: argparse.Namespace) -> int:
    started, began = time.monotonic(), now()
    out = args.work / args.name
    record_path = args.records / f"{args.name}.json"
    args.records.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    command = list(args.seedvr2x)
    say(f"run {args.name}, started {began}; tree {args.tree}")
    say(
        f"constants: card {args.card}, VAE_ROOM {args.room} GiB (the encode's"
        f" {args.room_encode}, the decode's {args.room_decode}), ENCODE_EXTRA"
        f" {'as shipped' if args.encode_extra is None else f'{args.encode_extra} GiB'};"
        f" cure: {args.cure}; the DiT's phase {'UNJUDGED' if args.dit_unjudged else 'judged'};"
        f" CUDNN_FRONTEND_LOG_INFO {os.environ.get('CUDNN_FRONTEND_LOG_INFO')}"
    )
    say(f"seedvr2x {' '.join(command)} -o {out}")
    for query in (
        "--query-compute-apps=pid,process_name,used_memory",
        "--query-gpu=memory.used,memory.total",
    ):
        found = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True
        )
        say(f"nvidia-smi {query}: {found.stdout.strip() or 'nothing'}")
    if out.exists():
        shutil.rmtree(out)
    if args.seed_dir is not None:
        shutil.copytree(args.seed_dir, out)
        say(f"{args.seed_dir} copied to {out}: the run resumes it")

    # As vram_cap.py and seedvr2x's own start have it, before torch is imported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.card != "bare":
        os.environ["VRAM_CAP_GIB"] = args.card
        spec = importlib.util.spec_from_file_location("vram_cap", args.vram_cap)
        assert spec is not None and spec.loader is not None
        cap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cap)
        cap.setup()

    import torch

    from seedvr2x import cli
    from seedvr2x.runtime import planner

    shipped = {"VAE_ROOM": planner.VAE_ROOM, "ENCODE_EXTRA": planner.ENCODE_EXTRA}
    rooms = set_planner(planner, args)
    probes = Probes(args)
    patched = install(probes, args)
    say(
        f"in force: planner.VAE_ROOM {planner.VAE_ROOM} (shipped {shipped['VAE_ROOM']}),"
        f" the encode's room {rooms['encode']}, the decode's {rooms['decode']},"
        f" planner.ENCODE_EXTRA {planner.ENCODE_EXTRA} (shipped {shipped['ENCODE_EXTRA']});"
        f" patched: {'; '.join(patched) or 'nothing but the probes'}; torch {torch.__version__},"
        f" allocator {os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
    )

    status: int | str | None
    try:
        status = cli.main([*command, "-o", str(out)])
    except SystemExit as error:
        status = error.code
    except Exception as error:  # the record is written whatever ended the run
        traceback.print_exc()
        status = f"raised {error!r}"
    if probes.unit is not None:
        probes.end(failed=True)
    probes.card.ended = True

    record: dict = {
        "name": args.name,
        "tree": args.tree,
        "began": began,
        "seconds": round(time.monotonic() - started),
        "exit": status,
        "card": args.card,
        "room": args.room,
        "rooms": {phase: room / GIB for phase, room in rooms.items()},
        "encode_extra": planner.ENCODE_EXTRA / GIB,
        "cure": args.cure,
        "request_floor": args.request_floor if "request" in args.cures else None,
        "output_floor": args.output_floor if "output" in args.cures else None,
        "stop_after": args.stop_after,
        "dit_unjudged": args.dit_unjudged,
        "cudnn_frontend_log": os.environ.get("CUDNN_FRONTEND_LOG_INFO"),
        "seeded": args.seed_dir is not None,
        "command": command,
        "torch": torch.__version__,
        "units": probes.units,
        "planned": probes.planned,
        "oom": probes.oom,
        "emptied": probes.emptied,
        "plan": None,
        "checksums": None,
    }
    manifest = out / "manifest.json"
    if manifest.exists():
        record["plan"] = json.loads(manifest.read_text())["plan"]
    sums = sorted((out / "checksums").glob("*.crc32")) if out.is_dir() else []
    if sums:
        whole = b"".join(path.name.encode() + b"\n" + path.read_bytes() for path in sums)
        record["checksums"] = {"files": len(sums), "md5": hashlib.md5(whole).hexdigest()}
        (args.records / f"{args.name}.crc32").write_bytes(whole)
    record_path.write_text(json.dumps(record, indent=1))
    if out.exists() and not args.keep:
        shutil.rmtree(out)
    say(
        f"run {args.name} ended {now()}: exit {status} in {record['seconds']} s;"
        f" {outcome(record)}; checksums {record['checksums']}"
    )
    return status if isinstance(status, int) else 1


def outcome(record: dict) -> str:
    """A run's record in a clause: each VAE unit's peaks, or where it was refused."""
    said = []
    for unit in record["units"]:
        if "peak_allocated" not in unit:
            continue
        if unit["phase"] in ("encode", "decode"):
            said.append(
                f"{unit['phase']} {'FAILED at' if unit['failed'] else 'ran,'} peak"
                f" {unit['peak_allocated'] / GIB:.2f} reserved {unit['peak_reserved'] / GIB:.2f}"
            )
        elif unit["failed"]:
            said.append(f"unit {unit['number']} ({unit['phase']}) FAILED")
    oom = record["oom"]
    if oom is not None:
        request = "?" if oom["request"] is None else f"{oom['request'] / GIB:.2f}"
        said.append(
            f"REFUSED in the {oom['phase']}, slice {oom['slice']}, at {oom['caught_at']}:"
            f" {request} GiB asked with {oom['allocated'] / GIB:.2f} allocated,"
            f" {oom['reserved'] / GIB:.2f} reserved, card free {oom['card_free'] / GIB:.3f}"
        )
    return "; ".join(said) or "no VAE unit"


# ---------------------------------------------------------------- pool


def pool(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    os.environ["VRAM_CAP_GIB"] = args.card
    spec = importlib.util.spec_from_file_location("vram_cap", args.vram_cap)
    assert spec is not None and spec.loader is not None
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    cap.setup()
    import torch

    say(f"pool: torch {torch.__version__}, allocator {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    def state(what: str) -> int:
        torch.cuda.synchronize()
        free = torch.cuda.mem_get_info()[0]
        say(
            f"pool: {what}: allocated {torch.cuda.memory_allocated()} reserved"
            f" {torch.cuda.memory_reserved()} driver free {free}"
        )
        return free

    def make(size: int):
        return torch.empty(size, dtype=torch.uint8, device="cuda")

    block, small = round(args.block * GIB), round(args.small * GIB)
    for order in ("the small tensor after the block is freed", "the small tensor first"):
        say(f"pool: --- {order}")
        torch.cuda.empty_cache()
        start = state("start")
        kept = make(small) if order.endswith("first") else None
        big = make(block)
        first, last = big.data_ptr(), big.data_ptr() + block
        state(f"a block of {block} allocated at {first:#x}")
        del big
        state("the block freed")
        if kept is None:
            kept = make(small)
        inside = first <= kept.data_ptr() < last
        state(
            f"a tensor of {small} at {kept.data_ptr():#x},"
            f" {'INSIDE' if inside else 'outside'} the freed block"
        )
        torch.cuda.empty_cache()
        free = state("the pool emptied")
        # Twice: of what is free now, then of what the card had free at the start less the
        # tensor kept, which a pool that gave everything back would give.
        for asked, of in (
            (free - round(args.left * GIB), "what is free now"),
            (start - small - round(args.left * GIB), "what was free at the start, the tensor off"),
        ):
            try:
                large = make(asked)
                state(f"a request of {asked}, {of} less {args.left} GiB: GIVEN")
                del large
            except torch.OutOfMemoryError as error:
                state(f"a request of {asked}, {of} less {args.left} GiB: REFUSED")
                for line in str(error).splitlines():
                    say(f"pool:   torch: {line}")
        del kept
    return 0


# ---------------------------------------------------------------- conv


def conv(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    if args.card != "bare":
        os.environ["VRAM_CAP_GIB"] = args.card
        spec = importlib.util.spec_from_file_location("vram_cap", args.vram_cap)
        assert spec is not None and spec.loader is not None
        cap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cap)
        cap.setup()
    import torch

    cudnn = torch.backends.cudnn
    say(
        f"conv: torch {torch.__version__}, cuDNN {cudnn.version()}, benchmark {cudnn.benchmark},"
        f" deterministic {cudnn.deterministic}, allocator {os.environ['PYTORCH_CUDA_ALLOC_CONF']}"
    )
    torch.manual_seed(0)
    layer = torch.nn.Conv3d(args.channels, args.channels, 3, padding=(0, 1, 1))
    layer = layer.to("cuda", torch.bfloat16).requires_grad_(False)
    shape = (1, args.channels, args.frames, args.rows, args.columns)
    x = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to("cuda", torch.bfloat16)
    say(f"conv: input {shape}, {x.numel() * 2 / GIB:.3f} GiB")

    def run() -> tuple[object, float, int]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.monotonic()
        with torch.no_grad():
            out = layer(x)
        torch.cuda.synchronize()
        return out, time.monotonic() - started, torch.cuda.max_memory_allocated() - before

    reference, seconds, peak = run()
    reference = reference.cpu()
    again, seconds, peak = run()
    say(
        f"conv: free: {seconds:.3f} s, {peak / GIB:.3f} GiB allocated over the input at the most;"
        f" the same bits twice: {bool(torch.equal(again.cpu(), reference))}"
    )
    del again
    for left in args.left:
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        fill = free - round(left * GIB)
        if fill <= 0:
            say(f"conv: {left} GiB left: more than is free ({free / GIB:.3f})")
            continue
        filler = torch.empty(fill, dtype=torch.uint8, device="cuda")
        found = []
        for _ in range(args.repeats):
            try:
                out, seconds, peak = run()
                same = bool(torch.equal(out.cpu(), reference))
                found.append(
                    f"{'the same bits' if same else 'OTHER BITS'} in {seconds:.3f} s"
                    f" ({peak / GIB:.3f} GiB at the most)"
                )
                del out
            except torch.OutOfMemoryError as error:
                asked = [line.strip() for line in str(error).splitlines() if "Requested" in line]
                found.append(f"REFUSED ({asked[0] if asked else 'no size said'})")
        say(
            f"conv: {left} GiB left (driver free {torch.cuda.mem_get_info()[0] / GIB:.3f}): "
            + "; ".join(found)
        )
        del filler
    # With the memory free again, after the steps short of it: the first bits, or the others?
    torch.cuda.empty_cache()
    found = []
    for _ in range(args.repeats):
        out, seconds, peak = run()
        same = bool(torch.equal(out.cpu(), reference))
        found.append(
            f"{'the same bits' if same else 'OTHER BITS'} in {seconds:.3f} s"
            f" ({peak / GIB:.3f} GiB at the most)"
        )
        del out
    say(
        f"conv: free again (driver free {torch.cuda.mem_get_info()[0] / GIB:.3f}),"
        " after the steps short of memory: " + "; ".join(found)
    )
    return 0


# ---------------------------------------------------------------- queue


def tree_of(copy: Path, scratch: Path) -> str:
    """git's tree of the files in `copy` (a git repository's working directory), nothing
    written in it: the index and any new object go to `scratch`."""
    scratch.mkdir(parents=True, exist_ok=True)
    objects = scratch / "objects"
    objects.mkdir(exist_ok=True)
    index = scratch / "index"
    index.unlink(missing_ok=True)
    env = dict(
        os.environ,
        GIT_INDEX_FILE=str(index),
        GIT_OBJECT_DIRECTORY=str(objects),
        GIT_ALTERNATE_OBJECT_DIRECTORIES=str(copy / ".git" / "objects"),
    )
    subprocess.run(["git", "add", "-A", "."], cwd=copy, env=env, check=True)
    found = subprocess.run(
        ["git", "write-tree"], cwd=copy, env=env, check=True, capture_output=True, text=True
    )
    return found.stdout.strip()


def halves(arguments: list[str]) -> tuple[list[str], list[str]]:
    """A queue line's arguments: run's own, and seedvr2x's after `--`."""
    cut = arguments.index("--") if "--" in arguments else len(arguments)
    return arguments[:cut], arguments[cut + 1 :]


def queue(args: argparse.Namespace) -> int:
    progress = Path(f"{args.file}.progress")
    pause, skip = Path(f"{args.file}.pause"), Path(f"{args.file}.skip")
    args.records.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)

    def note(line: str) -> None:
        print(line, flush=True)
        with progress.open("a") as file:
            file.write(line + "\n")

    note(
        f"# queue {args.file} started {now()};"
        " name | exit | seconds | began | ended | queued | outcome"
    )
    if args.wait_for is not None:
        while not args.wait_for.exists():
            time.sleep(30)
        note(f"# {args.wait_for} is there {now()}")
    done = 0
    common: list[str] = []
    while True:
        lines = [line.strip() for line in args.file.read_text().splitlines()]
        lines = [line for line in lines if line and not line.startswith("#")]
        if done >= len(lines):
            break
        name, _, rest = (part.strip() for part in lines[done].partition("|"))
        done += 1
        if name == "common":
            common = shlex.split(rest)
            continue
        while pause.exists():
            time.sleep(30)
        if skip.exists():
            note(f"# SKIPPED ({skip.name}): {name} and what follows")
            break
        tree = tree_of(args.copy, args.records / "tree-check") if args.copy else "unchecked"
        if args.want and tree != args.want:
            note(
                f"# WRONG TREE {tree} in {args.copy} (wanted {args.want}):"
                f" {name} not run, queue stopped"
            )
            return 3
        if (args.records / f"{name}.json").exists():
            note(f"# {name}: done already")
            continue
        queued = now()
        # run's own arguments, the common ones then the line's; then seedvr2x's, the line's then
        # the common ones'.
        given = shlex.split(rest)
        (common_ours, common_theirs), (ours, theirs) = halves(common), halves(given)
        command = ["flock", str(args.lock), sys.executable, str(Path(__file__).resolve())]
        if ours[:1] == ["pool"]:
            # The allocator's probe: no model, no record.
            command += ours
        else:
            command += ["run", "--name", name, "--records", str(args.records), "--tree", tree]
            command += [*common_ours, *ours, "--", *theirs, *common_theirs]
        log = args.records / f"{name}.log"
        began = time.monotonic()
        with log.open("w") as file:
            file.write(f"# queued {queued}: {shlex.join(command)}\n")
            file.flush()
            status = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors="replace")
        first = re.search(r"vae_room: run \S+, started (\S+);", text)
        said = "no record"
        record = args.records / f"{name}.json"
        if record.exists():
            said = outcome(json.loads(record.read_text()))
        note(
            f"{name} | {status} | {time.monotonic() - began:.0f} | {first[1] if first else '?'}"
            f" | {now()} | {queued} | {said}"
        )
    note(f"# queue ended {now()}")
    return 0


# ---------------------------------------------------------------- plan


def matrix():
    """planner_matrix.py, beside this file: the cases' clips and the planner's calls."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import planner_matrix

    return planner_matrix


def case_of(driver, spec: str) -> tuple[tuple[int, int], str, int]:
    """A case's target, card and clip length in latents: RES:CARD:MODE as planner_matrix.py
    takes it, the clip's length from the longest shot the card runs in one window under the
    room in force, or RES:CARD:FRAMES."""
    resolution, card, mode = spec.split(":")
    target = (int(resolution), int(resolution) * 16 // 9)
    if mode.isdigit():
        return target, card, (int(mode) - 1) // 4 + 1
    if card == "least":
        raise SystemExit(f"{spec}: the smallest card is searched for a clip of FRAMES frames")
    cap = driver.one_window(driver.free_of(card), target)
    return target, card, {"2w": 2 * cap - 2, "w1": cap + 1, "w": cap}[mode]


def planned(args: argparse.Namespace) -> int:
    driver = matrix()
    from seedvr2x.runtime import planner, tilegrid

    rooms = set_planner(planner, args)
    pins = planner.Pins(args.window, args.encode_tile, args.decode_tile, args.swap_blocks)
    for spec in args.case:
        target, card, latents = case_of(driver, spec)
        if card == "least":
            # The smallest card, by 0.05 GB, on which the planner takes the job with its pins:
            # the pass that bounds it is then packed to the budget.
            for hundredths in range(600, 9600, 5):
                card = f"{hundredths / 100:g}"
                try:
                    driver.planned(driver.free_of(card), target, latents, pins)
                    break
                except planner.PlanError:
                    continue
            spec = f"{spec} = {card}"
        free = driver.free_of(card)
        try:
            plan, peaks = driver.planned(free, target, latents, pins)
        except planner.PlanError as error:
            print(f"{spec} room {args.room}: REFUSED: {error}")
            continue
        found = peaks.record()
        padded = planner._padded(target, False)
        passes = []
        for phase, tiles, resident in (
            ("encode", plan.encode_tiles, plan.resident_encode),
            ("decode", plan.decode_tiles, plan.resident_decode),
        ):
            shapes = "untiled"
            if tiles is not None:
                laid = tilegrid.grid(tilegrid.latent_size(padded), tiles)
                each = sorted({(t.rows * 8, t.columns * 8) for t in laid}, reverse=True)
                shapes = f"{tiles.size} px " + " ".join(
                    f"{rows}x{columns}{'' if (rows, columns) in planner.MEASURED else '*'}"
                    for rows, columns in each
                )
            peak = found[f"{phase}_peak"]
            passes.append(
                f"{phase} {shapes}, resident {resident}, predicted {peak / GIB:.2f},"
                f" {(found['budget'] - peak - rooms[phase]) / GIB:.2f} under the budget"
                f" with its room"
            )
        print(
            f"{spec} room {rooms['encode'] / GIB:g}/{rooms['decode'] / GIB:g}:"
            f" {latents} latents ({4 * latents - 3} frames), free"
            f" {found['free'] / GIB:.2f} budget {found['budget'] / GIB:.2f}; window"
            f" {plan.window}, {plan.swapped} swapped, VAE {'in' if plan.vae_in_dit else 'away'};"
            f" DiT {found['dit_peak'] / GIB:.2f}; "
            + "; ".join(passes)
            + f"; pins: --window {plan.window} --swap-blocks {plan.swapped} --encode-tile"
            f" {plan.encode_tiles.size if plan.encode_tiles else 0} --decode-tile"
            f" {plan.decode_tiles.size if plan.decode_tiles else 0}"
        )
    return 0


# ---------------------------------------------------------------- table


def trims(log: Path) -> dict[int, int]:
    """torch's retries after trimming the pool, by the unit they fell in: the log's lines
    between a unit's `begins` and its end."""
    found: dict[int, int] = {}
    unit = 0
    if not log.exists():
        return found
    for line in log.read_text(errors="replace").splitlines():
        began = re.match(r"vae_room: unit (\d+) begins", line)
        if began:
            unit = int(began[1])
        elif re.match(r"vae_room: unit \d+ \S+ (ends|FAILED)", line):
            unit = 0
        elif TRIM in line:
            found[unit] = found.get(unit, 0) + 1
    return found


def rows(directory: Path) -> list[dict]:
    """One row per VAE unit of every record in a directory."""
    found = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if "units" not in record:
            continue
        plan = record["plan"]
        retried = trims(path.with_suffix(".log"))
        oom = record["oom"]
        for unit in record["units"]:
            phase = unit["phase"]
            if phase not in ("encode", "decode") or "peak_allocated" not in unit:
                continue
            row = {
                "run": record["name"],
                "path": str(path),
                "card": record["card"],
                "room": record["room"],
                "extra": record["encode_extra"],
                "cure": record["cure"],
                "exit": record["exit"],
                "seconds": record["seconds"],
                "phase": phase,
                "unit_seconds": unit["seconds"],
                "slices": len(unit["slices"]),
                "peak": unit["peak_allocated"],
                "reserved": unit["peak_reserved"],
                "card_free_least": unit["card_free_least"],
                "trims": retried.get(unit["number"], 0),
                "emptied": unit["emptied"]["count"],
                "emptied_seconds": unit["emptied"]["seconds"],
                "failed": unit["failed"],
                "checksums": (record["checksums"] or {}).get("md5"),
            }
            if plan is not None:
                memory = plan["memory"]
                tiles = plan[f"{phase}_tiles"]
                row |= {
                    "tiles": "untiled" if tiles is None else str(tiles["size"]),
                    "resident": memory[f"resident_{phase}"],
                    "swapped": memory["swapped_blocks"],
                    "window": plan["window"],
                    "free": memory["free"],
                    "budget": memory["budget"],
                    "predicted": memory[f"{phase}_peak"],
                }
            if unit["failed"] and oom is not None:
                row |= {
                    "oom_slice": oom["slice"],
                    "oom_site": oom["caught_at"],
                    "oom_request": oom["request"],
                    "oom_allocated": oom["allocated"],
                    "oom_reserved": oom["reserved"],
                    "oom_driver_free": oom["driver_free"],
                    "oom_card_free": oom["card_free"],
                }
            largest = max((int(key.split()[-1]) for key in unit["requests"]), default=None)
            row["largest_request"] = largest
            found.append(row)
    return found


def gib(count: int | None, digits: int = 2) -> str:
    return "" if count is None else f"{count / GIB:.{digits}f}"


def table(args: argparse.Namespace) -> int:
    head = (
        "| run | card | room | cure | pass | tiles | DiT blocks there | free | predicted | peak |"
        " free - peak | pool's peak | pool over peak | card free least | largest request |"
        " retries | result | pass s | frames' md5 |"
    )
    for directory in args.records:
        print(f"\n{directory}\n\n{head}\n|" + "---|" * 19)
        for row in rows(directory):
            if row["failed"] and "oom_slice" in row:
                result = (
                    f"REFUSED at slice {row['oom_slice']} ({row['oom_site']}):"
                    f" {gib(row['oom_request'])} asked, {gib(row['oom_allocated'])} allocated,"
                    f" {gib(row['oom_reserved'])} reserved, card free"
                    f" {gib(row['oom_card_free'], 3)}"
                )
            else:
                result = "FAILED" if row["failed"] else "ran"
            free = row.get("free")
            left = None if free is None else free - row["peak"]
            print(
                f"| {row['run']} | {row['card']} | {row['room']} | {row['cure']} | {row['phase']}"
                f" | {row.get('tiles', '')} | {row.get('resident', '')} | {gib(free)}"
                f" | {gib(row.get('predicted'))} | {gib(row['peak'])} | {gib(left)}"
                f" | {gib(row['reserved'])} | {gib(row['reserved'] - row['peak'])}"
                f" | {gib(row['card_free_least'], 3)} | {gib(row['largest_request'])}"
                f" | {row['trims']} | {result} | {row['unit_seconds']}"
                f" | {(row['checksums'] or '')[:8]} |"
            )
    return 0


# ---------------------------------------------------------------- command line


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--records", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--tree", default="not told")
    run.add_argument("--card", required=True, help="a size in GB, or bare")
    run.add_argument("--vram-cap", type=Path, help="vram_cap.py, for an emulated card")
    run.add_argument("--room", type=float, required=True, help="planner.VAE_ROOM, GiB")
    run.add_argument("--room-encode", type=float, help="the encode's own room, GiB")
    run.add_argument("--room-decode", type=float, help="the decode's own room, GiB")
    run.add_argument("--encode-extra", type=float, help="planner.ENCODE_EXTRA, GiB")
    run.add_argument(
        "--cure",
        default="none",
        help="none, or enter, slice, request, output, output-norm; several joined by +",
    )
    run.add_argument("--request-floor", type=float, default=1.0)
    run.add_argument(
        "--output-floor", type=float, default=0.0, help="cure output: only outputs this large, GiB"
    )
    run.add_argument("--stop-after", choices=("encode", "windows"))
    run.add_argument("--dit-unjudged", action="store_true")
    run.add_argument("--seed-dir", type=Path)
    run.add_argument("--keep", action="store_true")
    run.add_argument("seedvr2x", nargs=argparse.REMAINDER, help="-- then seedvr2x's arguments")

    lined = commands.add_parser("queue")
    lined.add_argument("file", type=Path)
    lined.add_argument("--records", type=Path, required=True)
    lined.add_argument("--lock", type=Path, required=True)
    lined.add_argument("--copy", type=Path, help="the code's directory, a git repository's")
    lined.add_argument("--want", help="the tree --copy must be")
    lined.add_argument("--wait-for", type=Path)

    probe = commands.add_parser("pool")
    probe.add_argument("--card", required=True)
    probe.add_argument("--vram-cap", type=Path, required=True)
    probe.add_argument("--block", type=float, default=4.0, help="GiB")
    probe.add_argument("--small", type=float, default=0.0625, help="GiB")
    probe.add_argument("--left", type=float, default=0.5, help="GiB")

    one = commands.add_parser("conv")
    one.add_argument("--card", required=True)
    one.add_argument("--vram-cap", type=Path)
    one.add_argument("--channels", type=int, default=256)
    one.add_argument("--frames", type=int, default=6)
    one.add_argument("--rows", type=int, default=160)
    one.add_argument("--columns", type=int, default=640)
    one.add_argument("--repeats", type=int, default=3)
    one.add_argument(
        "--left",
        type=float,
        nargs="+",
        default=[4, 3, 2, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        help="GiB left free at each step",
    )

    dry = commands.add_parser("plan")
    dry.add_argument("--case", action="append", default=[], required=True)
    dry.add_argument("--room", type=float, required=True)
    dry.add_argument("--room-encode", type=float)
    dry.add_argument("--room-decode", type=float)
    dry.add_argument("--encode-extra", type=float)
    dry.add_argument("--window", type=int)
    dry.add_argument("--swap-blocks", type=int)
    dry.add_argument("--encode-tile", type=int)
    dry.add_argument("--decode-tile", type=int)
    dry.add_argument("--dit-unjudged", action="store_true")

    tabled = commands.add_parser("table")
    tabled.add_argument("records", type=Path, nargs="+")

    args = parser.parse_args()
    if args.command == "run":
        if args.seedvr2x[:1] == ["--"]:
            args.seedvr2x = args.seedvr2x[1:]
        if args.card != "bare" and args.vram_cap is None:
            parser.error("--vram-cap is required for an emulated card")
        args.cures = set(args.cure.split("+")) - {"none"}
        if args.cures - CURES:
            parser.error(f"--cure {args.cure}: known cures are {', '.join(sorted(CURES))}")
        return launch(args)
    if args.command == "queue":
        return queue(args)
    if args.command == "pool":
        return pool(args)
    if args.command == "conv":
        return conv(args)
    if args.command == "plan":
        return planned(args)
    return table(args)


if __name__ == "__main__":
    sys.exit(main())
