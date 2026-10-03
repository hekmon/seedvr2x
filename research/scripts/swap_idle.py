#!/usr/bin/env python3
"""Idle pauses in place of BlockSwap's moves, and NVML power and clock sampling, for the SeedVR2 CLI.

BlockSwap (src/optimization/blockswap.py) moves each swapped DiT block to the GPU before its
forward and back to the CPU after it, synchronously: the GPU computes nothing meanwhile. With
the 7B Q4_K_M model, swapping all 36 blocks showed no end-to-end cost although the moves take
about 0.9 s per batch (swap_probe.py). One explanation: the GPU runs at its power cap, its compute
is energy-bound, and the idle gaps of the copies let the following kernels run at higher clocks.
This wrapper tests that without moving anything: it keeps every block on the GPU and inserts,
where each swapped block would move, a synchronous idle pause as long as that move. If the
pauses cost much less than their own duration, compute is energy-bound; if they cost their full
duration, it is not. It can also sample NVML while the CLI runs and time every DiT forward
between two synchronizations, for the energy, power and clocks of each batch.

Environment (nothing set: nothing is patched, the run is the plain CLI's):
  SWAP_IDLE_BLOCKS=N    pause around the forward of DiT blocks 0..N-1 (the blocks
                        --blocks_to_swap N would swap); meant for --blocks_to_swap 0, and
                        blocks BlockSwap swaps itself are left alone. Before each block's forward:
                        torch.cuda.synchronize(), then wait for the block's in-move time; after it:
                        synchronize, then wait for its out-move time. Nothing is queued on the GPU
                        during a wait, so it idles, as during BlockSwap's synchronous copies.
  SWAP_IDLE_FROM=FILE   per-block move times from a swap_probe.py .swap.json: the mean of each
                        block's CPU -> GPU moves (in) and of its GPU -> CPU moves (out)
  SWAP_IDLE_IN_MS, SWAP_IDLE_OUT_MS
                        constant in / out pauses (ms) instead, and for a block the file lacks
  SWAP_IDLE_SCALE=1     multiplies every pause; 0 keeps the synchronizations only
  SWAP_IDLE_NVML_HZ=0   sample NVML at this rate (e.g. 20) from start to exit, through ctypes
                        (libnvidia-ml, no CUDA context): instant and 1 s average board power,
                        the energy counter, SM and memory clocks, the clock-event (throttle)
                        reasons, temperature, utilization, P-state, and whether a DiT forward or
                        a pause is running. Each DiT forward (VideoDiffusionInfer.inference: one
                        per batch) is then timed between two torch.cuda.synchronize() calls, as
                        it is with SWAP_IDLE_BLOCKS.
  SWAP_IDLE_GPU=0       NVML device index
  SWAP_IDLE_OUT=PREFIX  output prefix (default: $BENCH_LOG without .log, else ./swap_idle)

Output: PREFIX.idle.json (settings, pauses, one record per DiT forward: synchronized time,
pauses, energy, mean power, SM clock in and out of pauses, share of samples under each
clock-event reason) and PREFIX.power.csv (the samples); one line per forward on stderr.
The CLI's own "DiT inference N" timer is not synchronized: it can end while the batch's last
kernels still run.

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python swap_idle.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap swap_idle.py --env SWAP_IDLE_NVML_HZ=20 -- <CLI args> \\
      --dit_offload_device cpu --blocks_to_swap 36                 # BlockSwap, sampled
  python3 bench.py run NAME --wrap swap_idle.py --env SWAP_IDLE_NVML_HZ=20 \\
      --env SWAP_IDLE_BLOCKS=36 --env SWAP_IDLE_FROM=runs/PROBE.swap.json -- <CLI args> \\
      --dit_offload_device cpu --blocks_to_swap 0                  # the pauses, no moves
  python swap_idle.py --nvml-test [SECONDS [CSV]]   # NVML only: fields, sampling rate, cost
  python swap_idle.py --selftest              # CPU only: pause accuracy, move-time parsing, energy
"""
import atexit
import ctypes
import importlib.abc
import json
import os
import runpy
import statistics
import sys
import threading
import time

REASONS = [(0x1, "gpu_idle"), (0x2, "app_clocks"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"),
           (0x10, "sync_boost"), (0x20, "sw_thermal"), (0x40, "hw_thermal"), (0x80, "hw_power_brake"),
           (0x100, "display_clock")]
FI_POWER_AVERAGE, FI_POWER_INSTANT = 185, 186  # nvml.h field ids (mW)
COLUMNS = ["t_s", "power_inst_w", "power_avg_w", "energy_j", "sm_mhz", "mem_mhz", "reasons", "temp_c",
           "util_gpu", "pstate", "dit_forward", "pause", "read_ms"]
T0 = time.perf_counter()


def log(msg):
    print(f"swap_idle: {msg}", file=sys.stderr, flush=True)


def env_float(name, default):
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


class S:  # settings
    blocks = int(env_float("SWAP_IDLE_BLOCKS", 0))
    moves_file = os.environ.get("SWAP_IDLE_FROM", "").strip()
    in_ms = env_float("SWAP_IDLE_IN_MS", 0.0)
    out_ms = env_float("SWAP_IDLE_OUT_MS", 0.0)
    scale = env_float("SWAP_IDLE_SCALE", 1.0)
    hz = env_float("SWAP_IDLE_NVML_HZ", 0.0)
    gpu = int(env_float("SWAP_IDLE_GPU", 0))
    moves = {}  # block -> (in_s, out_s) from the probe file


class State:
    cur = None        # the DiT forward running (dict) or None
    windows = []      # finished DiT forwards
    in_pause = False
    wrapped = 0       # blocks given pauses
    skipped = 0       # blocks left to BlockSwap


# ---------------------------------------------------------------- move times and pauses

def load_moves(path):
    """swap_probe.py JSON -> {block: (mean in-move s, mean out-move s)}."""
    with open(path, encoding="utf-8") as f:
        calls = json.load(f)["calls"]
    acc = {}
    for c in calls:
        if c["src"] == "cpu" and c["dst"] != "cpu":
            k = 0
        elif c["src"] != "cpu" and c["dst"] == "cpu":
            k = 1
        else:
            continue
        acc.setdefault(int(c["block"]), ([], []))[k].append(float(c["s"]))
    return {b: (statistics.mean(v[0]) if v[0] else None, statistics.mean(v[1]) if v[1] else None)
            for b, v in acc.items()}


def block_pauses(i):
    tin, tout = S.moves.get(i, (None, None))
    tin = S.in_ms / 1000 if tin is None else tin
    tout = S.out_ms / 1000 if tout is None else tout
    return tin * S.scale, tout * S.scale


def wait(d):
    """Wait d seconds: sleep, then spin over the last 3 ms (sleep alone overshoots)."""
    end = time.perf_counter() + d
    if d > 0.004:
        time.sleep(d - 0.003)
    while time.perf_counter() < end:
        pass


def pause(d, sync=True):
    if sync:
        import torch
        torch.cuda.synchronize()
    if d <= 0:
        return
    t = time.perf_counter()
    State.in_pause = True
    wait(d)
    State.in_pause = False
    w = State.cur
    if w is not None:
        w["pauses"] += 1
        w["pause_s"] += time.perf_counter() - t


def paused_forward(f, tin, tout):
    def forward(*a, **k):
        pause(tin)
        out = f(*a, **k)
        pause(tout)
        return out
    return forward


def add_pauses(dit):
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        if not getattr(add_pauses, "warned", False):
            add_pauses.warned = True
            log("the DiT has no .blocks: no pauses")
        return
    swapped = getattr(dit, "blocks_to_swap", -1)  # numz: index of the last swapped block, -1 if none
    new = 0
    for i, blk in enumerate(blocks):
        if i >= S.blocks:
            break
        if getattr(blk, "_swap_idle", None) is not None:
            continue
        if hasattr(blk, "_original_forward") and i <= swapped:
            blk._swap_idle = "blockswap"
            State.skipped += 1
            continue
        tin, tout = block_pauses(i)
        blk.forward = paused_forward(blk.forward, tin, tout)
        blk._swap_idle = (tin, tout)
        new += 1
    if new:
        State.wrapped += new
        tot = sum(sum(block_pauses(i)) for i in range(min(S.blocks, len(blocks))))
        log(f"pauses around {new} blocks of {len(blocks)} ({State.skipped} left to BlockSwap): "
            f"{tot:.3f} s per forward, scale {S.scale:g}")


# ---------------------------------------------------------------- DiT forwards

def patch_infer(module):
    cls = getattr(module, "VideoDiffusionInfer", None)
    if cls is None or getattr(cls.inference, "_swap_idle", False):
        return
    orig = cls.inference

    def inference(self, *a, **k):
        import torch
        dit = getattr(self, "dit", None)
        dit = getattr(dit, "dit_model", dit)
        if S.blocks and dit is not None:
            add_pauses(dit)
        torch.cuda.synchronize()
        w = {"index": len(State.windows) + 1, "t0": time.perf_counter() - T0, "pauses": 0, "pause_s": 0.0}
        State.cur = w
        try:
            return orig(self, *a, **k)
        finally:
            torch.cuda.synchronize()
            w["t1"] = time.perf_counter() - T0
            w["s"] = w["t1"] - w["t0"]
            State.cur = None
            State.windows.append(w)
            log(f"DiT forward {w['index']}: {w['s']:.3f} s (synchronized), {w['pauses']} pauses, "
                f"{w['pause_s']:.3f} s paused")

    inference._swap_idle = True
    cls.inference = inference
    log("DiT forwards timed between synchronizations (VideoDiffusionInfer.inference)")


HOOKS = [(".core.infer", patch_infer)]


class Finder(importlib.abc.MetaPathFinder):
    """Lets the module load normally, then applies the patch right after it executes."""

    def find_spec(self, name, path, target=None):
        hook = next((h for suffix, h in HOOKS if ("." + name).endswith(suffix)), None)
        if hook is None:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            hook(module)
        loader.exec_module = exec_module
        return spec


# ---------------------------------------------------------------- NVML

class _Value(ctypes.Union):
    _fields_ = [("d", ctypes.c_double), ("ui", ctypes.c_uint), ("ul", ctypes.c_ulong),
                ("ull", ctypes.c_ulonglong), ("sll", ctypes.c_longlong), ("si", ctypes.c_int),
                ("us", ctypes.c_ushort)]


class FieldValue(ctypes.Structure):  # nvmlFieldValue_t
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", _Value)]


def field_value(f):
    return [f.value.d, f.value.ui, f.value.ul, f.value.ull, f.value.sll, f.value.si, f.value.us][f.valueType] \
        if 0 <= f.valueType <= 6 else None


class Nvml:
    """One read = every available field; unavailable ones read None."""

    def __init__(self, index):
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(h)) != 0:
            raise OSError(f"no NVML device {index}")
        self.lib, self.h = lib, h
        self.fv = (FieldValue * 2)()
        self.u = ctypes.c_uint()
        self.ull = ctypes.c_ulonglong()
        self.i = ctypes.c_int()
        self.util = (ctypes.c_uint * 2)()
        self.reasons_fn = None
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(lib, name, None)
            if fn is not None and fn(h, ctypes.byref(self.ull)) == 0:
                self.reasons_fn, self.reasons_name = fn, name
                break
        self.has_fields = self._fields() != (None, None)
        self.info = {"name": self._string(lib.nvmlDeviceGetName),
                     "driver": self._string(lambda b, n: lib.nvmlSystemGetDriverVersion(b, n), device=False),
                     "power_limit_w": self._uint(lib.nvmlDeviceGetEnforcedPowerLimit, scale=1e-3),
                     "reasons_call": getattr(self, "reasons_name", None), "power_fields": self.has_fields}

    def _string(self, fn, device=True):
        buf = ctypes.create_string_buffer(96)
        rc = fn(self.h, buf, 96) if device else fn(buf, 96)
        return buf.value.decode() if rc == 0 else None

    def _uint(self, fn, *args, scale=1.0):
        return self.u.value * scale if fn(self.h, *args, ctypes.byref(self.u)) == 0 else None

    def _fields(self):
        fv = self.fv
        fv[0].fieldId, fv[1].fieldId = FI_POWER_INSTANT, FI_POWER_AVERAGE
        fv[0].scopeId = fv[1].scopeId = 0  # NVML_POWER_SCOPE_GPU
        if self.lib.nvmlDeviceGetFieldValues(self.h, 2, fv) != 0:
            return None, None
        return tuple(field_value(f) / 1000 if f.nvmlReturn == 0 and field_value(f) is not None else None
                     for f in fv)

    def read(self):
        lib, h = self.lib, self.h
        p_inst, p_avg = self._fields() if self.has_fields else (None, None)
        if p_avg is None:
            p_avg = self._uint(lib.nvmlDeviceGetPowerUsage, scale=1e-3)
        energy = self.ull.value / 1000 if lib.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(self.ull)) == 0 \
            else None
        sm = self._uint(lib.nvmlDeviceGetClockInfo, 1)    # NVML_CLOCK_SM
        mem = self._uint(lib.nvmlDeviceGetClockInfo, 2)   # NVML_CLOCK_MEM
        reasons = self.ull.value if self.reasons_fn and self.reasons_fn(h, ctypes.byref(self.ull)) == 0 else None
        temp = self._uint(lib.nvmlDeviceGetTemperature, 0)  # NVML_TEMPERATURE_GPU
        util = self.util[0] if lib.nvmlDeviceGetUtilizationRates(h, self.util) == 0 else None
        pstate = self.i.value if lib.nvmlDeviceGetPerformanceState(h, ctypes.byref(self.i)) == 0 else None
        return p_inst, p_avg, energy, sm, mem, reasons, temp, util, pstate


class Sampler(threading.Thread):
    def __init__(self, nvml, hz):
        super().__init__(daemon=True, name="swap_idle-nvml")
        self.nvml, self.dt = nvml, 1.0 / hz
        self.rows = []
        self.halt = threading.Event()

    def run(self):
        nxt = time.perf_counter()
        while not self.halt.is_set():
            t = time.perf_counter()
            vals = self.nvml.read()
            w = State.cur
            self.rows.append((t - T0, *vals, w["index"] if w else 0, 1 if State.in_pause else 0,
                              1000 * (time.perf_counter() - t)))
            nxt += self.dt
            d = nxt - time.perf_counter()
            if d > 0:
                self.halt.wait(d)
            else:
                nxt = time.perf_counter()

    def stop(self):
        self.halt.set()
        self.join(timeout=2)


# ---------------------------------------------------------------- statistics

def interp(rows, col, t):
    """Value of a cumulative column at time t, linear between the samples around it."""
    pts = [(r[0], r[col]) for r in rows if r[col] is not None]
    if len(pts) < 2 or t < pts[0][0] or t > pts[-1][0]:
        return None
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, v0), (t1, v1) = pts[lo], pts[hi]
    return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def mean_of(rows, col):
    v = [r[col] for r in rows if r[col] is not None]
    return round(statistics.mean(v), 1) if v else None


def window_stats(w, rows):
    c = {k: i for i, k in enumerate(COLUMNS)}
    inside = [r for r in rows if w["t0"] <= r[0] <= w["t1"]]
    e0, e1 = interp(rows, c["energy_j"], w["t0"]), interp(rows, c["energy_j"], w["t1"])
    out = dict(w)
    out["samples"] = len(inside)
    out["energy_j"] = round(e1 - e0, 1) if e0 is not None and e1 is not None else None
    out["power_w"] = round(out["energy_j"] / w["s"], 1) if out["energy_j"] is not None and w["s"] > 0 else None
    out["power_inst_mean_w"] = mean_of(inside, c["power_inst_w"])
    p = [(r[0], r[c["power_inst_w"]]) for r in inside if r[c["power_inst_w"]] is not None]
    out["energy_inst_j"] = round(sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(p, p[1:])), 1) \
        if len(p) > 1 else None  # trapezoid over the instant-power samples inside the forward
    out["sm_mhz_mean"] = mean_of(inside, c["sm_mhz"])
    sm = [r[c["sm_mhz"]] for r in inside if r[c["sm_mhz"]] is not None]
    out["sm_mhz_min"], out["sm_mhz_max"] = (min(sm), max(sm)) if sm else (None, None)
    out["sm_mhz_mean_outside_pauses"] = mean_of([r for r in inside if not r[c["pause"]]], c["sm_mhz"])
    out["sm_mhz_mean_in_pauses"] = mean_of([r for r in inside if r[c["pause"]]], c["sm_mhz"])
    out["power_inst_mean_in_pauses_w"] = mean_of([r for r in inside if r[c["pause"]]], c["power_inst_w"])
    out["mem_mhz_mean"] = mean_of(inside, c["mem_mhz"])
    out["temp_c_mean"] = mean_of(inside, c["temp_c"])
    out["util_mean"] = mean_of(inside, c["util_gpu"])
    rs = [r[c["reasons"]] for r in inside if r[c["reasons"]] is not None]
    seen = 0
    for x in rs:
        seen |= x
    names = dict(REASONS)
    out["reasons_share"] = {names.get(1 << b, hex(1 << b)): round(sum(1 for x in rs if x >> b & 1) / len(rs), 3)
                            for b in range(seen.bit_length()) if seen >> b & 1}
    return out


# ---------------------------------------------------------------- output

def prefix():
    p = os.environ.get("SWAP_IDLE_OUT")
    if p:
        return p
    log_path = os.environ.get("BENCH_LOG")
    return (log_path[:-4] if log_path and log_path.endswith(".log") else log_path) or "swap_idle"


def write_out():
    if getattr(write_out, "done", False):
        return
    write_out.done = True
    smp = getattr(install, "sampler", None)
    rows = []
    if smp is not None:
        smp.stop()
        rows = smp.rows
    e_col = COLUMNS.index("energy_j")
    e_first = next((r[e_col] for r in rows if r[e_col] is not None), None)
    windows = [window_stats(w, rows) for w in State.windows]
    pre = prefix()
    span = rows[-1][0] - rows[0][0] if len(rows) > 1 else 0
    res = {"settings": {"blocks": S.blocks, "from": S.moves_file or None, "in_ms": S.in_ms, "out_ms": S.out_ms,
                        "scale": S.scale, "nvml_hz": S.hz, "gpu": S.gpu},
           "pauses": {"wrapped_blocks": State.wrapped, "left_to_blockswap": State.skipped,
                      "per_block_s": {str(i): [round(x, 6) for x in block_pauses(i)] for i in range(S.blocks)}},
           "nvml": {**(getattr(install, "nvml_info", None) or {}), "samples": len(rows),
                    "rate_hz": round((len(rows) - 1) / span, 1) if span else None,
                    "read_ms_mean": round(statistics.mean(r[-1] for r in rows), 3) if rows else None,
                    "epoch_at_t0": getattr(install, "epoch_t0", None)},
           "forwards": windows}
    with open(pre + ".idle.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    if rows:
        with open(pre + ".power.csv", "w", encoding="utf-8") as f:
            f.write(",".join(COLUMNS) + "\n")
            for r in rows:
                r = list(r)
                if r[e_col] is not None and e_first is not None:
                    r[e_col] = r[e_col] - e_first
                if r[6] is not None:
                    r[6] = hex(r[6])
                f.write(",".join("" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
                                 for x in r) + "\n")
    for w in windows:
        log(f"forward {w['index']}: {w['s']:.3f} s, paused {w['pause_s']:.3f} s ({w['pauses']}), "
            f"{w['energy_j']} J, {w['power_w']} W, SM {w['sm_mhz_mean']} MHz "
            f"(out of pauses {w['sm_mhz_mean_outside_pauses']}), {w['temp_c_mean']} C, {w['reasons_share']}")
    log(f"-> {pre}.idle.json" + (f", {pre}.power.csv ({len(rows)} samples)" if rows else ""))


# ---------------------------------------------------------------- entry points

def install():
    if S.moves_file:
        S.moves = load_moves(S.moves_file)
        log(f"move times of {len(S.moves)} blocks from {S.moves_file}")
    if S.blocks and not S.moves and not (S.in_ms or S.out_ms) and S.scale:
        sys.exit("swap_idle: SWAP_IDLE_BLOCKS needs SWAP_IDLE_FROM or SWAP_IDLE_IN_MS / SWAP_IDLE_OUT_MS")
    if not (S.blocks or S.hz > 0):
        return False
    sys.meta_path.insert(0, Finder())
    if S.hz > 0:
        try:
            nv = Nvml(S.gpu)
        except OSError as e:
            log(f"NVML unavailable ({e}): no samples")
        else:
            install.nvml_info = nv.info
            install.epoch_t0 = time.time() - (time.perf_counter() - T0)
            install.sampler = Sampler(nv, S.hz)
            install.sampler.start()
            log(f"NVML sampling at {S.hz:g} Hz: {nv.info}")
    log(f"settings: blocks {S.blocks}, scale {S.scale:g}, in/out {S.in_ms:g}/{S.out_ms:g} ms"
        + (f", from {S.moves_file}" if S.moves_file else ""))
    return True


def cmd_nvml_test(seconds, csv_path=None):
    nv = Nvml(S.gpu)
    print("device:", nv.info)
    hz = S.hz or 20.0
    smp = Sampler(nv, hz)
    smp.start()
    time.sleep(seconds)
    smp.stop()
    rows = smp.rows
    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(COLUMNS) + "\n")
            for r in rows:
                f.write(",".join("" if x is None else str(x) for x in r) + "\n")
    span = rows[-1][0] - rows[0][0]
    print(f"{len(rows)} samples in {span:.2f} s at a target of {hz:g} Hz: {(len(rows) - 1) / span:.1f} Hz; "
          f"read {statistics.mean(r[-1] for r in rows):.3f} ms mean, {max(r[-1] for r in rows):.3f} max")
    for i, k in enumerate(COLUMNS[1:10], 1):
        v = [r[i] for r in rows if r[i] is not None]
        if k == "reasons":
            print(f"  {k}: {sorted({hex(x) for x in v})}")
        elif v:
            print(f"  {k}: min {min(v):.1f}, mean {statistics.mean(v):.1f}, max {max(v):.1f} ({len(v)} values)")
        else:
            print(f"  {k}: unavailable")
    e = [(r[0], r[3]) for r in rows if r[3] is not None]
    if len(e) > 1:
        steps = [b[1] - a[1] for a, b in zip(e, e[1:])]
        print(f"  energy counter: monotonic {all(s >= 0 for s in steps)}, unchanged between samples "
              f"{sum(1 for s in steps if s == 0)} of {len(steps)}; mean power from it "
              f"{(e[-1][1] - e[0][1]) / (e[-1][0] - e[0][0]):.1f} W")


def cmd_selftest():
    import tempfile
    ok = True
    # pause accuracy (no CUDA: sync off)
    for d in (0.006, 0.018):
        errs = []
        for _ in range(50):
            t = time.perf_counter()
            pause(d, sync=False)
            errs.append(time.perf_counter() - t - d)
        good = max(errs) < 0.0005 and min(errs) >= 0
        ok &= good
        print(f"pause {d * 1000:.0f} ms x50: error mean {statistics.mean(errs) * 1e6:.0f} us, "
              f"max {max(errs) * 1e6:.0f} us {'ok' if good else 'TOO LARGE'}")
    # move times from a swap_probe.py-like file
    calls = []
    for b in range(3):
        for k in range(2):
            calls.append({"block": b, "src": "cpu", "dst": "cuda:0", "bytes": 1, "s": 0.006 + 0.001 * k, "gap_s": None})
            calls.append({"block": b, "src": "cuda:0", "dst": "cpu", "bytes": 1, "s": 0.018 + 0.002 * k, "gap_s": 0.2})
    calls.append({"block": 0, "src": "cpu", "dst": "cpu", "bytes": 1, "s": 9.0, "gap_s": None})
    with tempfile.NamedTemporaryFile("w", suffix=".swap.json", delete=False) as f:
        json.dump({"summary": {}, "calls": calls}, f)
    m = load_moves(f.name)
    os.unlink(f.name)
    good = all(abs(m[b][0] - 0.0065) < 1e-12 and abs(m[b][1] - 0.019) < 1e-12 for b in range(3)) and len(m) == 3
    ok &= good
    print(f"move times from a probe file: {m[0]} {'ok' if good else 'WRONG'}")
    S.moves, S.in_ms, S.out_ms, S.scale = m, 5.0, 7.0, 2.0
    good = all(abs(a - b) < 1e-12 for a, b in zip(block_pauses(1) + block_pauses(7), (0.013, 0.038, 0.01, 0.014)))
    ok &= good
    print(f"pauses with scale 2: block 1 {block_pauses(1)}, unlisted block 7 {block_pauses(7)} "
          f"{'ok' if good else 'WRONG'}")
    # energy and clock statistics of a window over synthetic samples: 500 W, SM 2000 MHz, power cap bit
    rows = [(0.05 * i, 500.0, 500.0, 1000.0 + 25.0 * i, 2000 + (i % 2) * 100, 13365, 0x4 if i % 4 else 0x0,
             60, 100, 1, 1, i % 5 == 0, 0.5) for i in range(200)]
    w = window_stats({"index": 1, "t0": 1.0125, "t1": 6.0125, "s": 5.0, "pauses": 0, "pause_s": 0.0}, rows)
    good = (abs(w["energy_j"] - 2500.0) < 1e-6 and w["power_w"] == 500.0 and w["energy_inst_j"] == 2475.0
            and w["reasons_share"] == {"sw_power_cap": 0.75})
    ok &= good
    print(f"window stats: {w['energy_j']} J over 5 s = {w['power_w']} W (instant samples: {w['energy_inst_j']} J "
          f"over 4.95 s), SM {w['sm_mhz_mean']} MHz, {w['reasons_share']} {'ok' if good else 'WRONG'}")
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        return cmd_selftest()
    if sys.argv[1] == "--nvml-test":
        return cmd_nvml_test(float(sys.argv[2]) if len(sys.argv) > 2 else 5.0,
                             sys.argv[3] if len(sys.argv) > 3 else None)
    if sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    if install():
        atexit.register(write_out)
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            write_out()
    else:
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
