"""Measurement harness for the SeedVR2 CLI (inference_cli.py).

Runs one CLI invocation with --debug forced on, tees its output to a log, samples device
memory through NVML while it runs, then parses the log into one structured JSON record per
run (per-phase time, torch VRAM figures, NVML peak, sub-timings, OOM/retry events...).

Standard library only. Run it with any Python >= 3.9 on the GPU host; the CLI itself runs
with the SeedVR2 venv's python.

Usage:
  bench.py run <name> [--seedvr2-dir D] [--runs-dir R] [--force] [--overwrite]
               [--env K=V ...] [--interval S] [--gpu I] -- <CLI args>
  bench.py parse <log>... [--append] [--results F] [--json]
  bench.py table [--results F] [--all] [names...]

`run` writes <runs-dir>/<name>.log (the CLI output), <name>.nvml.csv (memory samples) and
appends a record to results.jsonl. `parse` re-reads logs; it picks up <stem>.nvml.csv and the
original `run` record when they exist, so re-parsing after a parser change loses nothing.

Defaults: --seedvr2-dir $SEEDVR2_DIR or ".", --runs-dir $BENCH_RUNS_DIR or "./runs",
--results <runs-dir>/results.jsonl. --env values go through os.path.expandvars, so
--env 'PATH=$HOME/.local/bin:$PATH' works (quote it).

Memory figures are in GiB (SeedVR2 prints GiB and labels them "GB").
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import re
import resource
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

GIB = 1024 ** 3
CLI_DEFAULT_ALLOC_CONF = "backend:cudaMallocAsync"  # inference_cli.py setdefault()s it
PHASE_NAMES = {1: "VAE encode", 2: "DiT", 3: "VAE decode", 4: "Post-processing"}


# ---------------------------------------------------------------- GPU memory sampling

class _NvmlMemV2(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("total", ctypes.c_ulonglong),
                ("reserved", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _NvmlMemV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class Samples:
    """Timestamped device-memory samples: (epoch seconds, used bytes)."""

    def __init__(self, samples=None, interval=0.1, backend=None):
        self.samples = samples or []
        self.interval, self.backend = interval, backend

    @classmethod
    def from_csv(cls, path):
        """Load a <name>.nvml.csv written by `run`."""
        samples = []
        with open(path, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                epoch, _, _, mib = line.strip().split(",")
                samples.append((float(epoch), float(mib) * 2**20))
        gaps = sorted(b[0] - a[0] for a, b in zip(samples, samples[1:]))
        return cls(samples, round(gaps[len(gaps) // 2], 3) if gaps else 0.1, "csv")

    def peak(self, t0=None, t1=None):
        """(peak bytes, time) over [t0, t1], or (None, None) without samples."""
        sel = [s for s in self.samples if (t0 is None or s[0] >= t0) and (t1 is None or s[0] <= t1)]
        if not sel:
            return None, None
        t, used = max(sel, key=lambda s: s[1])
        return used, t

    def nearest(self, t):
        if not self.samples:
            return None
        return min(self.samples, key=lambda s: abs(s[0] - t))[1]


class NvmlSampler(Samples):
    """Samples device memory used (bytes) every `interval` s in a thread.

    Uses libnvidia-ml through ctypes (nvmlDeviceGetMemoryInfo_v2: "used" excludes the
    driver-reserved memory, same as nvidia-smi's memory.used). Falls back to
    `nvidia-smi -lms` if the library can't be loaded.
    """

    def __init__(self, gpu=0, interval=0.1):
        super().__init__(interval=interval)
        self.gpu = gpu
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._read = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 failed")
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(gpu, ctypes.byref(handle)) != 0:
                raise OSError(f"no NVML device {gpu}")
            mem2 = _NvmlMemV2(version=ctypes.sizeof(_NvmlMemV2) | (2 << 24))
            if lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(mem2)) == 0:
                self._read = lambda: (lib.nvmlDeviceGetMemoryInfo_v2(handle, ctypes.byref(mem2)), mem2.used)[1]
                self.backend = "nvml_v2"
            else:
                mem1 = _NvmlMemV1()
                self._read = lambda: (lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem1)), mem1.used)[1]
                self.backend = "nvml_v1"
        except (OSError, AttributeError):
            self.backend = "nvidia-smi"

    def read_now(self):
        return self._read() if self._read else None

    def start(self):
        target = self._loop_nvml if self._read else self._loop_smi
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop_nvml(self):
        while not self._stop.is_set():
            self.samples.append((time.time(), self._read()))
            self._stop.wait(self.interval)

    def _loop_smi(self):
        cmd = ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=timestamp,memory.used",
               "--format=csv,noheader,nounits", f"-lms={max(1, int(self.interval * 1000))}"]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.backend = None
            return
        for line in self._proc.stdout:
            try:
                ts, used = (s.strip() for s in line.split(","))
                t = dt.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((t, int(used) * 1024 * 1024))
            except ValueError:
                continue


def nvidia_smi(*args):
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_compute_apps(gpu):
    out = nvidia_smi("-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader,nounits")
    if out is None:
        return None
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            apps.append({"pid": parts[0], "process": parts[1], "used_mib": parts[2]})
    return apps


def gpu_state(gpu):
    fields = ["name", "driver_version", "memory.used", "memory.reserved", "memory.total",
              "pstate", "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem"]
    out = nvidia_smi("-i", str(gpu), f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
    state = dict(zip(fields, (v.strip() for v in out.split(",")))) if out else {}
    state["compute_apps"] = gpu_compute_apps(gpu)
    return state


def git_info(path):
    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except OSError:
            return None
    rev = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"rev": rev, "describe": git("describe", "--tags", "--always", "--dirty"),
            "dirty": bool(status) if status is not None else None}


# ---------------------------------------------------------------- log parsing

TS_RE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d{3})\] ?(.*)$")
NUM = r"([\d.]+)"
RX = {
    "phase_start": re.compile(r"━+ Phase (\d): (.+?) ━+"),
    "phase_done": re.compile(r"Phase (\d): (.+?) complete: " + NUM + "s"),
    "timing": re.compile(r"⚡(\s+)└─ (.+): " + NUM + r"s\s*$"),
    "mem_label": re.compile(r"📊 ([^\[].*):\s*$"),
    "vram": re.compile(r"\[VRAM\] " + NUM + r"GB allocated / " + NUM + r"GB reserved / Peak: " + NUM
                       + r"GB / " + NUM + r"GB free / " + NUM + r"GB total"),
    "ram": re.compile(r"\[RAM\] " + NUM + r"GB process / " + NUM + r"GB others / " + NUM + r"GB free / "
                      + NUM + r"GB total"),
    "args_start": re.compile(r"🔧 Arguments:\s*$"),
    "arg": re.compile(r"^\s+(\w+): ?(.*)$"),
    "version": re.compile(r"CLI · v([\w.]+)"),
    "os_gpu": re.compile(r"OS: (.+?) \| GPU: (.+?) \((\d+)GB\)"),
    "python": re.compile(r"Python: (\S+) \| PyTorch: (\S+) \| FlashAttn: (.+?) \| SageAttn: (.+?) \| Triton: (\S+)"),
    "cuda": re.compile(r"CUDA: (\S+) \| cuDNN: (\S+)"),
    "conv3d": re.compile(r"Conv3d workaround active"),
    "initial_mem": re.compile(r"Initial CUDA memory: " + NUM + r"GB free / " + NUM + r"GB total"),
    "video_info": re.compile(r"Video info: (\d+) frames, (\d+)x(\d+), " + NUM + " FPS"),
    "target": re.compile(r"Target dimensions: (\d+)x(\d+) \(padded to (\d+)x(\d+)"),
    "gen_input": re.compile(r"Input: (\d+) frames, (\d+)x(\d+)px → Padded: (\d+)x(\d+)px → Output: (\d+)x(\d+)px"),
    "batch": re.compile(r"Batch size: (\d+), Seed: (\d+)"),
    "batch_progress": re.compile(r"(Encoding|Upscaling|Decoding|Post-processing) batch (\d+)/(\d+)"),
    "chunk": re.compile(r"Chunk (\d+)/(\d+): (\d+) new \+ (\d+) context"),
    "latents": re.compile(r"Latents shape: torch\.Size\(\[(.+?)\]\)"),
    "done": re.compile(r"All upscaling processes completed successfully in " + NUM + "s"),
    "fps": re.compile(r"Average FPS: " + NUM),
    "output": re.compile(r"Output saved to: (.+)$"),
    "pid": re.compile(r"Process (\d+) terminating"),
    # /usr/bin/time -v trailer, if the log has one
    "time_cmd": re.compile(r'Command being timed: "(.*)"'),
    "time_rss": re.compile(r"Maximum resident set size \(kbytes\): (\d+)"),
    "time_wall": re.compile(r"Elapsed \(wall clock\) time .*: ([\d:.]+)"),
    "time_exit": re.compile(r"^\s*Exit status: (\d+)"),
}
EVENT_RE = re.compile(r"out of memory|OutOfMemory|\bOOM\b|allocation on device|retry|Traceback|error|"
                      r"⚠️|❌|\[WARNING\]|\[ERROR\]", re.IGNORECASE)
PHASE_BATCH_KEY = {"Encoding": 1, "Upscaling": 2, "Decoding": 3, "Post-processing": 4}


def _hms(s):
    secs = 0.0
    for part in s.split(":"):
        secs = secs * 60 + float(part)
    return secs


def _event_kind(text):
    low = text.lower()
    if "out of memory" in low or "outofmemory" in low or re.search(r"\boom\b", low) or "allocation on device" in low:
        return "oom"
    if "retry" in low:
        return "retry"
    if "traceback" in low:
        return "traceback"
    if "error" in low or "❌" in text:
        return "error"
    return "warning"


def parse_log(path, anchor=None):
    """Parse a SeedVR2 --debug log. `anchor` (datetime) gives the date of the first
    timestamp (logs only carry HH:MM:SS.mmm, local time of the host); epoch times in the
    result are only meaningful when it's right. Midnight rollovers are handled."""
    text = Path(path).read_bytes().decode("utf-8", errors="replace")
    anchor = anchor or dt.datetime.fromtimestamp(Path(path).stat().st_mtime)
    day = dt.datetime.combine(anchor.date(), dt.time())
    last_t = None

    rec = {"args": {}, "platform": {}, "input": {}, "generation": {}, "events": [],
           "checkpoints": [], "phase_runs": [], "batches": {}, "chunks": None}
    cur_phase = None          # phase run dict currently open
    after_done = None         # phase run whose "complete" line we just saw (for └─ lines)
    pending_label = None      # memory checkpoint label awaiting its VRAM/RAM lines
    in_args = False
    plat, inp, gen = rec["platform"], rec["input"], rec["generation"]

    lines = []
    for raw in text.split("\n"):
        lines.extend(seg for seg in raw.split("\r") if seg.strip())  # tqdm uses \r

    for line in lines:
        m = TS_RE.match(line)
        t = None
        if m:
            hh, mm, ss, ms, rest = m.groups()
            t = day + dt.timedelta(hours=int(hh), minutes=int(mm), seconds=int(ss), milliseconds=int(ms))
            if last_t and t < last_t - dt.timedelta(hours=12):
                day += dt.timedelta(days=1)
                t += dt.timedelta(days=1)
            last_t = t
            tstamp = f"{hh}:{mm}:{ss}.{ms}"
            epoch = t.timestamp()
        else:
            rest, tstamp, epoch = line, None, None

        # Arguments block: "[ts]    key: value" lines right after "🔧 Arguments:"
        if in_args:
            am = RX["arg"].match(rest) if m else None
            if am and not rest.lstrip()[:1] in ("🔧", "🖥", "ℹ", "📊"):
                rec["args"][am.group(1)] = am.group(2)
                continue
            in_args = False
        if RX["args_start"].search(rest):
            in_args = True
            continue

        if EVENT_RE.search(rest):
            rec["events"].append({"t": tstamp, "kind": _event_kind(rest), "phase": cur_phase["phase"] if cur_phase else None,
                                  "line": rest.strip()[:300]})

        # timing breakdown lines follow the "Phase N: ... complete" line
        tm = RX["timing"].search(rest)
        if tm and after_done is not None:
            after_done["timings"].append({"label": tm.group(2), "s": float(tm.group(3)),
                                          "depth": max(1, (len(tm.group(1)) - 1) // 2)})
            continue
        after_done = None

        if (mm_ := RX["phase_start"].search(rest)):
            cur_phase = {"phase": int(mm_.group(1)), "label": mm_.group(2), "start": tstamp, "start_epoch": epoch,
                         "end": None, "end_epoch": None, "time_s": None, "timings": [], "checkpoints": []}
            rec["phase_runs"].append(cur_phase)
            continue
        if (mm_ := RX["phase_done"].search(rest)):
            n = int(mm_.group(1))
            pr = cur_phase if cur_phase and cur_phase["phase"] == n else None
            if pr is None:  # no header seen; create one
                pr = {"phase": n, "label": mm_.group(2), "start": None, "start_epoch": None, "end": None,
                      "end_epoch": None, "time_s": None, "timings": [], "checkpoints": []}
                rec["phase_runs"].append(pr)
            pr["time_s"] = float(mm_.group(3))
            pr["end"], pr["end_epoch"] = tstamp, epoch
            after_done = pr
            continue

        if (mm_ := RX["mem_label"].search(rest)):
            pending_label = {"label": mm_.group(1), "t": tstamp, "epoch": epoch,
                             "phase": cur_phase["phase"] if cur_phase else None}
            continue
        if pending_label and (mm_ := RX["vram"].search(rest)):
            a, r, p, fr, tot = map(float, mm_.groups())
            pending_label.update(alloc=a, reserved=r, peak_alloc=p, free=fr, total=tot)
            rec["checkpoints"].append(pending_label)
            if cur_phase is not None:
                cur_phase["checkpoints"].append(pending_label)
                if pending_label["label"].startswith(f"After phase {cur_phase['phase']} "):
                    # the phase window ends with its final memory snapshot
                    cur_phase["end"], cur_phase["end_epoch"] = tstamp, epoch
            continue
        if pending_label and (mm_ := RX["ram"].search(rest)):
            pending_label["ram_process"] = float(mm_.group(1))
            if pending_label["label"].startswith("After phase "):
                cur_phase = None
            pending_label = None
            continue
        pending_label = None  # a "label:" line not followed by a memory snapshot

        for key in ("version", "os_gpu", "python", "cuda", "conv3d", "initial_mem", "video_info", "target",
                    "gen_input", "batch", "batch_progress", "chunk", "latents", "done", "fps", "output", "pid",
                    "time_cmd", "time_rss", "time_wall", "time_exit"):
            mm_ = RX[key].search(rest)
            if not mm_:
                continue
            g = mm_.groups()
            if key == "version":
                plat["seedvr2_version"] = g[0]
            elif key == "os_gpu":
                plat.update(os=g[0], gpu=g[1], gpu_mem_gb=int(g[2]))
            elif key == "python":
                plat.update(python=g[0], torch=g[1], flash_attn=g[2], sage_attn=g[3], triton=g[4])
            elif key == "cuda":
                plat.update(cuda=g[0], cudnn=g[1])
            elif key == "conv3d":
                plat["conv3d_workaround"] = True
            elif key == "initial_mem":
                plat.update(initial_free_gib=float(g[0]), total_gib=float(g[1]))
            elif key == "video_info":
                inp.update(frames=int(g[0]), width=int(g[1]), height=int(g[2]), fps=float(g[3]))
            elif key == "target":
                gen.update(target=f"{g[0]}x{g[1]}", padded=f"{g[2]}x{g[3]}")
            elif key == "gen_input":
                gen.setdefault("frames", 0)
                gen["frames"] += int(g[0])  # summed over chunks (includes context frames)
                gen.update(input_res=f"{g[1]}x{g[2]}", padded=f"{g[3]}x{g[4]}", output_res=f"{g[5]}x{g[6]}")
                gen["generations"] = gen.get("generations", 0) + 1
            elif key == "batch":
                gen.update(batch_size=int(g[0]), seed=int(g[1]))
            elif key == "batch_progress":
                ph = PHASE_BATCH_KEY[g[0]]
                rec["batches"][str(ph)] = max(rec["batches"].get(str(ph), 0), int(g[2]))
            elif key == "chunk":
                rec["chunks"] = int(g[1])
            elif key == "latents":
                gen.setdefault("latents_shape", g[0])
            elif key == "done":
                rec["total_s"] = float(g[0])
            elif key == "fps":
                rec["avg_fps"] = float(g[0])
            elif key == "output":
                rec["output"] = g[0].strip()
            elif key == "pid":
                rec["cli_pid"] = int(g[0])
            elif key == "time_cmd":
                rec["time_v_command"] = g[0]
            elif key == "time_rss":
                rec["max_rss_gib"] = round(int(g[0]) * 1024 / GIB, 2)
            elif key == "time_wall":
                rec["wall_s"] = round(_hms(g[0]), 2)
            elif key == "time_exit":
                rec["exit_status"] = int(g[0])
            break

    rec["completed"] = "total_s" in rec
    rec["oom_events"] = sum(1 for e in rec["events"] if e["kind"] == "oom")
    rec["retries"] = sum(1 for e in rec["events"] if re.search(r"retrying", e["line"], re.I))
    rec["phases"] = aggregate_phases(rec["phase_runs"])
    return rec


def _group_label(label):
    return re.sub(r"\b\d+\b", "#", label)


def aggregate_phases(runs):
    """Aggregate phase runs (several when streaming with --chunk_size) per phase number."""
    out = {}
    for n in sorted({r["phase"] for r in runs}):
        rs = [r for r in runs if r["phase"] == n]
        cps = [c for r in rs for c in r["checkpoints"]]
        end_cps = [c for c in cps if c["label"].startswith("After phase")]
        grouped = {}
        for r in rs:
            seen = {}
            for tmg in r["timings"]:
                label = tmg["label"]
                if not re.search(r"\b\d+\b", label) and label in seen:
                    # SeedVR2 reuses one timer name per batch (e.g. "VAE decode") and prints its
                    # last value under every batch: count it once
                    seen[label]["repeated"] = seen[label].get("repeated", 1) + 1
                    continue
                g = grouped.setdefault(_group_label(label),
                                       {"depth": tmg["depth"], "n": 0, "total_s": 0.0, "values": []})
                seen[label] = g
                g["n"] += 1
                g["total_s"] = round(g["total_s"] + tmg["s"], 2)
                num = re.search(r"\b(\d+)\b", label)
                g["values"].append((int(num.group(1)) if num else 0, tmg["s"]))
        for g in grouped.values():
            # breakdown lines are sorted by duration: put them back in batch order
            v = [s for _, s in sorted(g.pop("values"), key=lambda x: x[0])]
            g.update(first_s=v[0], min_s=min(v), max_s=max(v))
            if len(v) > 1:
                g["mean_rest_s"] = round(sum(v[1:]) / (len(v) - 1), 3)
            if "repeated" in g:
                g["note"] = f"timer shared by {g['repeated']} batches: only the last batch's value is known"
        times = [r["time_s"] for r in rs if r["time_s"] is not None]
        out[str(n)] = {
            "name": PHASE_NAMES.get(n, rs[0]["label"]),
            "runs": len(rs),
            "time_s": round(sum(times), 2) if times else None,
            # torch peak allocated: max of the "Peak" of every snapshot in the phase
            # (SeedVR2 resets the peak after each snapshot)
            "torch_peak_alloc_gib": max((c["peak_alloc"] for c in cps), default=None),
            "torch_alloc_end_gib": end_cps[-1]["alloc"] if end_cps else None,
            "torch_reserved_end_gib": end_cps[-1]["reserved"] if end_cps else None,
            "torch_reserved_max_snapshot_gib": max((c["reserved"] for c in cps), default=None),
            "ram_process_end_gib": end_cps[-1].get("ram_process") if end_cps else None,
            "ram_process_max_snapshot_gib": max((c.get("ram_process", 0) for c in cps), default=None),
            "snapshots": [{k: c.get(k) for k in ("label", "t", "alloc", "reserved", "peak_alloc", "free", "ram_process")}
                          for c in cps],
            "timings": grouped,
        }
    return out


def attach_nvml(rec, sampler, t_start, t_end):
    to_gib = lambda b: round(b / GIB, 2) if b is not None else None  # noqa: E731
    peak, peak_t = sampler.peak(t_start, t_end)
    used = [s[1] for s in sampler.samples]
    first_phase = min((r["start_epoch"] for r in rec["phase_runs"] if r["start_epoch"]), default=None)
    pre_peak, _ = sampler.peak(t_start, first_phase) if first_phase else (None, None)
    rec["nvml"] = {
        "backend": sampler.backend, "interval_s": sampler.interval, "samples": len(used),
        "peak_gib": to_gib(peak),
        "peak_at": dt.datetime.fromtimestamp(peak_t).strftime("%H:%M:%S.%f")[:-3] if peak_t else None,
        "before_phases_peak_gib": to_gib(pre_peak),
    }
    for r in rec["phase_runs"]:
        if r["start_epoch"] and r["end_epoch"]:
            # +interval on the end: the last snapshot is logged after the work it measures
            p, pt = sampler.peak(r["start_epoch"], r["end_epoch"] + sampler.interval)
            r["nvml_peak_gib"] = to_gib(p)
            r["nvml_start_gib"] = to_gib(sampler.nearest(r["start_epoch"]))
            r["nvml_end_gib"] = to_gib(sampler.nearest(r["end_epoch"]))
    for c in rec["checkpoints"]:
        if c.get("epoch"):
            c["nvml_gib"] = to_gib(sampler.nearest(c["epoch"]))
    for n, ph in rec["phases"].items():
        rs = [r for r in rec["phase_runs"] if str(r["phase"]) == n and r.get("nvml_peak_gib") is not None]
        ph["nvml_peak_gib"] = max((r["nvml_peak_gib"] for r in rs), default=None)
        ph["nvml_start_gib"] = rs[0]["nvml_start_gib"] if rs else None
        ph["nvml_end_gib"] = rs[-1]["nvml_end_gib"] if rs else None
        # growth during the phase: the peak also holds memory kept from earlier phases
        ph["nvml_growth_gib"] = (round(max(r["nvml_peak_gib"] - r["nvml_start_gib"] for r in rs), 2)
                                 if rs and rs[0]["nvml_start_gib"] is not None else None)
        by_label = {c["label"]: c.get("nvml_gib") for r in rec["phase_runs"] if str(r["phase"]) == n
                    for c in r["checkpoints"]}
        for s in ph["snapshots"]:
            s["nvml_gib"] = by_label.get(s["label"])


def finalize(rec):
    if rec.get("exit_status") not in (None, 0):
        rec["status"] = "oom" if rec["oom_events"] else "failed"
    elif rec["completed"]:
        rec["status"] = "ok" if not rec["oom_events"] else "ok-after-oom"
    else:
        rec["status"] = "oom" if rec["oom_events"] else "incomplete"
    # epoch fields are internal
    for r in rec["phase_runs"]:
        r.pop("start_epoch", None), r.pop("end_epoch", None)
        r.pop("checkpoints", None)
    # snapshots are kept per phase; keep the top-level list only for those outside any phase
    rec["checkpoints"] = [{k: v for k, v in c.items() if k != "epoch"} for c in rec["checkpoints"]
                          if c.get("phase") is None]
    return rec


# ---------------------------------------------------------------- subcommands

def results_path(a):
    return Path(a.results) if a.results else Path(a.runs_dir) / "results.jsonl"


def append_record(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_run(a):
    seedvr2 = Path(a.seedvr2_dir).resolve()
    runs = Path(a.runs_dir).resolve()
    python = a.python or str(seedvr2 / ".venv" / "bin" / "python")
    if not (seedvr2 / "inference_cli.py").is_file() or not os.access(python, os.X_OK):
        sys.exit(f"bench: no inference_cli.py in {seedvr2} or no interpreter {python}")
    runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / f"{a.name}.log"
    if log_path.exists() and not a.overwrite:
        sys.exit(f"bench: {log_path} exists (use --overwrite or another name)")
    cli_args = list(a.cli_args)
    if not cli_args:
        sys.exit("bench: no CLI arguments given (put them after --)")
    if "--debug" not in cli_args:
        cli_args.append("--debug")

    apps = gpu_compute_apps(a.gpu)
    if apps:
        msg = "bench: GPU busy: " + "; ".join(f"{x['pid']} {x['process']} {x['used_mib']} MiB" for x in apps)
        if not a.force:
            sys.exit(msg + " (use --force to run anyway)")
        print(msg + " (--force)", file=sys.stderr)
    gpu_before = gpu_state(a.gpu)

    extra_env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep:
            sys.exit(f"bench: --env expects K=V, got {kv!r}")
        extra_env[k] = os.path.expandvars(v)
    env = dict(os.environ, **extra_env, PYTHONUNBUFFERED="1")
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF")

    cmd = [python, "inference_cli.py", *cli_args]

    sampler = NvmlSampler(a.gpu, a.interval)
    baseline = sampler.read_now()
    started = dt.datetime.now().astimezone()
    sampler.start()
    t0_epoch, t0 = time.time(), time.monotonic()
    print(f"bench: {shlex.join(cmd)}  (cwd {seedvr2}, log {log_path}, sampler {sampler.backend})", file=sys.stderr)

    interrupted = False
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, cwd=seedvr2, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = sys.stdout.buffer
        while True:
            try:
                chunk = os.read(proc.stdout.fileno(), 65536)
            except KeyboardInterrupt:  # the terminal's SIGINT reaches the CLI too
                interrupted = True
                continue
            if not chunk:
                break
            log.write(chunk)
            log.flush()
            out.write(chunk)
            out.flush()
        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:
                interrupted = True
    wall = time.monotonic() - t0
    t1_epoch = time.time()
    time.sleep(min(0.5, 3 * a.interval))
    sampler.stop()
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)

    rec = parse_log(log_path, anchor=started.replace(tzinfo=None))
    attach_nvml(rec, sampler, t0_epoch, t1_epoch)
    rec["nvml"]["baseline_gib"] = round(baseline / GIB, 2) if baseline is not None else None
    rec.update(exit_status=rc, wall_s=round(wall, 2), max_rss_gib=round(ru.ru_maxrss * 1024 / GIB, 2),
               cpu_user_s=round(ru.ru_utime, 1), cpu_sys_s=round(ru.ru_stime, 1), interrupted=interrupted)
    if rec.get("total_s"):
        rec["startup_s"] = round(wall - rec["total_s"], 2)  # interpreter + imports before the CLI's timer
    rec = finalize(rec)
    head = {"name": a.name, "timestamp": started.isoformat(timespec="seconds"), "source": "run",
            "log": str(log_path), "command": cmd, "cli_args": cli_args, "extra_env": extra_env,
            "alloc_conf": alloc_conf or CLI_DEFAULT_ALLOC_CONF,
            "alloc_conf_source": "env" if alloc_conf else "cli-default",
            "seedvr2_git": git_info(seedvr2), "gpu_before": gpu_before, "gpu_after": gpu_state(a.gpu)}
    rec = {**head, **rec}
    write_nvml_csv(runs / f"{a.name}.nvml.csv", sampler.samples, t0_epoch)
    append_record(results_path(a), rec)
    print("\n" + summary_table([rec]), file=sys.stderr)
    print(f"bench: record appended to {results_path(a)}", file=sys.stderr)
    sys.exit(rc if rc >= 0 else 128 - rc)


def write_nvml_csv(path, samples, t0):
    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch,time,elapsed_s,used_mib\n")
        for t, used in samples:
            hms = dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]
            f.write(f"{t:.3f},{hms},{t - t0:.3f},{used / 2**20:.0f}\n")


# fields only `run` can measure; `parse` carries them over from the run's record
RUN_FIELDS = ("exit_status", "wall_s", "max_rss_gib", "cpu_user_s", "cpu_sys_s", "interrupted")
RUN_HEAD = ("name", "timestamp", "log", "command", "cli_args", "extra_env", "alloc_conf", "alloc_conf_source",
            "seedvr2_git", "gpu_before", "gpu_after")


def cmd_parse(a):
    """Parse logs. If <stem>.nvml.csv sits next to a log, its samples are used, and if the
    results file holds a `run` record for the same log, its run-only fields are kept: a
    re-parse after a parser fix gives the same record as the original run."""
    runs = {}
    if results_path(a).exists():
        runs = {r["log"]: r for r in load_results(results_path(a)) if r.get("source") == "run"}
    recs = []
    for p in a.logs:
        path = Path(p).resolve()
        csv = path.with_suffix(".nvml.csv")
        samples = Samples.from_csv(csv) if csv.exists() else None
        anchor = dt.datetime.fromtimestamp(samples.samples[0][0]) if samples and samples.samples else None
        rec = parse_log(path, anchor)
        old = runs.get(str(path))
        if samples:
            attach_nvml(rec, samples, samples.samples[0][0], samples.samples[-1][0])
            if old:
                rec["nvml"].update(backend=old["nvml"]["backend"], baseline_gib=old["nvml"].get("baseline_gib"))
        if old:
            rec.update({k: old[k] for k in RUN_FIELDS if k in old})
            if rec.get("total_s") and rec.get("wall_s"):
                rec["startup_s"] = round(rec["wall_s"] - rec["total_s"], 2)
        rec = finalize(rec)
        head = {k: old[k] for k in RUN_HEAD if k in old} if old else {
            "name": path.stem, "log": str(path),
            "timestamp": dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")}
        rec = {**head, "source": "reparse" if old else "parse", **rec}
        recs.append(rec)
        if a.json:
            print(json.dumps(rec, ensure_ascii=False, indent=1))
        if a.append:
            append_record(results_path(a), rec)
    print(summary_table(recs))


def load_results(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def cmd_table(a):
    recs = load_results(results_path(a))
    if a.names:
        recs = [r for r in recs if r["name"] in a.names]
    if not a.all:  # keep the latest record per name, in first-seen order
        latest = {}
        for r in recs:
            latest[r["name"]] = r
        recs = list(latest.values())
    print(summary_table(recs))


# ---------------------------------------------------------------- table

# options shown in "Key args" when they differ from these CLI defaults
NOTABLE_DEFAULTS = {
    "uniform_batch_size": "False", "chunk_size": "0", "temporal_overlap": "0", "prepend_frames": "0",
    "color_correction": "lab", "input_noise_scale": "0.0", "latent_noise_scale": "0.0",
    "dit_offload_device": "none", "vae_offload_device": "none", "tensor_offload_device": "cpu",
    "blocks_to_swap": "0", "swap_io_components": "False",
    "vae_encode_tiled": "False", "vae_decode_tiled": "False",
    "compile_dit": "False", "compile_vae": "False", "cache_dit": "False", "cache_vae": "False",
}


def key_args(r):
    a = r.get("args", {})
    model = a.get("dit_model", "?").replace("seedvr2_ema_", "").replace(".safetensors", "")
    parts = [model, f"res {a.get('resolution', '?')}", f"bs {a.get('batch_size', '?')}", a.get("attention_mode", "?")]
    for k, d in NOTABLE_DEFAULTS.items():
        if k in a and a[k] != d:
            parts.append(f"{k}={a[k]}")
    if a.get("vae_encode_tiled") == "True":
        parts.append(f"enc tile {a.get('vae_encode_tile_size')}/{a.get('vae_encode_tile_overlap')}")
    if a.get("vae_decode_tiled") == "True":
        parts.append(f"dec tile {a.get('vae_decode_tile_size')}/{a.get('vae_decode_tile_overlap')}")
    if r.get("alloc_conf_source") == "env":
        parts.append(f"alloc={r['alloc_conf']}")
    return ", ".join(parts)


def _g(x):
    return f"{x:.1f}" if isinstance(x, (int, float)) else "–"


def phase_cell(ph):
    if not ph:
        return "–"
    s = f"{ph['time_s']:.2f} s" if ph.get("time_s") is not None else "– s"
    return f"{s} · {_g(ph.get('torch_peak_alloc_gib'))} / {_g(ph.get('torch_reserved_end_gib'))} / {_g(ph.get('nvml_peak_gib'))}"


def summary_table(recs):
    hdr = ["Run", "Key args", "Frames", "Batches", "Encode", "DiT", "DiT inference", "Decode", "Post",
           "Total s", "FPS", "NVML peak", "Max RSS", "OOM / retries", "Status"]
    rows = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in recs:
        ph = r.get("phases", {})
        inf = ph.get("2", {}).get("timings", {}).get("DiT inference #", {})
        gen = r.get("generation", {})
        frames = gen.get("frames")
        res = gen.get("output_res")
        nb = r.get("batches", {}).get("2")
        rows.append("| " + " | ".join([
            r["name"], key_args(r),
            f"{frames} @ {res}" if frames else "–",
            str(nb) if nb else "–",
            phase_cell(ph.get("1")), phase_cell(ph.get("2")),
            f"{inf['total_s']:.2f} s ({inf['n']}×, first {inf['first_s']:.2f})" if inf else "–",
            phase_cell(ph.get("3")), phase_cell(ph.get("4")),
            f"{r['total_s']:.2f}" if r.get("total_s") is not None else "–",
            f"{r['avg_fps']:.2f}" if r.get("avg_fps") is not None else "–",
            _g((r.get("nvml") or {}).get("peak_gib")),
            _g(r.get("max_rss_gib")),
            f"{r.get('oom_events', 0)} / {r.get('retries', 0)}",
            r.get("status", "?"),
        ]) + " |")
    rows.append("")
    rows.append("Phase cells: time · torch peak allocated / torch reserved at phase end / NVML device peak (GiB).")
    return "\n".join(rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--runs-dir", default=os.environ.get("BENCH_RUNS_DIR", "runs"))
        p.add_argument("--results", help="results JSONL (default: <runs-dir>/results.jsonl)")

    p = sub.add_parser("run", help="run the CLI and record one result")
    p.add_argument("name")
    p.add_argument("--seedvr2-dir", default=os.environ.get("SEEDVR2_DIR", "."))
    p.add_argument("--python", help="interpreter for the CLI (default: <seedvr2-dir>/.venv/bin/python)")
    p.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment (repeatable)")
    p.add_argument("--gpu", type=int, default=0, help="NVML/nvidia-smi index of the GPU to watch")
    p.add_argument("--interval", type=float, default=0.1, help="NVML sampling period in seconds")
    p.add_argument("--force", action="store_true", help="run even if other compute processes hold the GPU")
    p.add_argument("--overwrite", action="store_true", help="overwrite an existing <name>.log")
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("parse", help="parse existing logs (NVML data from <stem>.nvml.csv if present)")
    p.add_argument("logs", nargs="+")
    p.add_argument("--append", action="store_true", help="append the records to the results file")
    p.add_argument("--json", action="store_true", help="print the full records")
    common(p)
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("table", help="Markdown summary of recorded runs")
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true", help="every record, not only the latest per name")
    common(p)
    p.set_defaults(func=cmd_table)

    argv = sys.argv[1:]
    cli_args = []
    if "--" in argv:  # everything after the first "--" goes to the CLI untouched
        i = argv.index("--")
        argv, cli_args = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    a.cli_args = cli_args
    a.func(a)


if __name__ == "__main__":
    main()
