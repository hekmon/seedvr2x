#!/usr/bin/env python3
"""DiT compile probe for seedvr2x: what torch.compile gives and costs across window lengths.

seedvr2x runs its DiT on windows of a shot's latents (runtime/shot.py, sample_windows): a window's
length L, in latents, takes every value up to a cap, where numz's batches had one length. This
probe measures, on seedvr2x's own runtime, what compiling the DiT gives and costs per length,
whether one compilation can serve every length, whether it works over seedvr2x's BlockSwap, and
whether the output keeps its bits.

One step is what sample_windows does for one window, on random latents of a given output size
and L latents (not an encoded video: N(0, 1) values, which the DiT's timing and memory do not
depend on): a latent (L, h, w, 16) bfloat16 in channel-major memory, as the encode gives it, drawn
from a seeded CPU generator (the same for every variant), goes through shot.draw (the shot's
noise, on the CPU, from the seed), both go to the device, the "sr" condition is built
(model.condition), then model.sample runs one Euler step under bfloat16 autocast; the output goes
back to the CPU. That is shot._window's body, with the sample timed alone. Before each step,
models.enter("dit") is called as the pipeline does. The models are loaded as the CLI loads them
(runtime/model.py, load_models), the allocator chosen as the CLI chooses it (runtime/memory.py,
startup, before torch is imported), and the attention backend it reports is recorded.

Variants (--variant, or the names of --seq), put on the DiT between steps of one process:
  plain   the module as loaded
  whole   torch.compile(dit, backend="inductor", mode="default", fullgraph=False, dynamic=False):
          numz's settings (src/core/model_configuration.py, compile_dit); the compiled module
          runs as the runner's DiT
  blocks  each block of dit.blocks compiled on its own forward, the rest of the DiT plain. The
          class's forward is compiled once and bound to every block (types.MethodType), as the
          block's own `forward`: BlockSwap's forward hooks (runtime/blockswap.py, registered on
          the blocks) run in Module.__call__, OUTSIDE the compiled code, where
          torch.compile(block) or block.compile() would trace them. Whether that one compiled
          function serves all the blocks is read from the cache entries per code object.
Only one compiled variant per process (its caches would be shared with the next).

install_dit(dit, variant, *, dynamic=None, mode="default", recompile_limit=None,
mark_dynamic=False, backend="inductor") -> handle is importable (no argparse or global state):
handle.module is what runs as the runner's DiT, handle.restore() takes the variant off (the
blocks' forward, torch's recompile limit), handle.activate() puts it on again. dynamic: None or
False the static graphs numz sets (one compile a shape), True one dynamic graph, "auto" torch's
automatic dynamic shapes (a recompile with the dimensions that changed dynamic). mark_dynamic
marks the token dimension of the block's (or the DiT's) video input dynamic before each call:
the tokens are the one dimension that changes with L (text stays 58 tokens), so a graph
specialised on it is what a length costs; it implies "auto" where dynamic is None.

Options of `run`: --attention sdpa|flash_attn_2 (sdpa: the vendor modules' FLASH_ATTN_2_AVAILABLE
flags put False before the load, so that the backend runs on SDPA where FlashAttention 2 is
installed; flash_attn_2 is refused where it is not), --swap N (a Plan with the last N blocks
swapped, as tests/test_placement.py loads one: the blocks' weights in pinned copies, streamed
through two slots by the placement's hooks), --dynamic, --auto-dynamic, --mark-dynamic,
--mode, --recompile-limit N (torch's default 8 where not given), --numz-limits (numz's two
limits: cache_size_limit 64, recompile_limit 128), --accumulated-limit.

Per step: GPU seconds of model.sample (CUDA events) and its wall seconds (the compiling call's
holds the compilation), torch peak allocated (reset before), allocated after, device memory
used, host RSS (the process and its children, the compile workers) and the CPU seconds both
used. Compilations are counted from torch itself (torch._dynamo.utils.get_compilation_metrics:
frames compiled, per code object, their time, the on-disk cache's hits and misses), and by a
logging handler on torch's loggers, which COUNTS the recompilations, the graph breaks and the
recompile-limit warnings and keeps their distinct reasons, never printing them. The cache
entries are read per code object of the DiT's model code after each step
(torch._C._dynamo.eval_frame._debug_get_cache_entry_list). With --frames, sys.monitoring counts
the starts of the model's Python functions per step: a function that runs compiled no longer
starts, so the counts tell a compiled step from one fallen back to plain code, whatever the clock.

Every output is compared bit for bit (torch.equal on the bfloat16 output) with the first plain
output of its length, and with the first output of its own variant; when they differ: max and
mean absolute difference, both over the reference output's standard deviation, and the share of
elements that differ. Each output's SHA-256 is recorded, and `table` compares them across files.

A reference step (--ref SIZE:L, default none) runs at the start and end of an invocation when
asked: the GPU may be power-capped and its clock drift, so times are valid as ratios of steps
run back to back in one process (`table` pairs each compiled step with the plain one before it).

Subcommands:
  run    --model-dir DIR --size SIZE --lengths L1,L2,... [--seq V1,V2,... | --variant V]
         [--passes N] [--seq-next ...] [--dynamic | --auto-dynamic] [--mark-dynamic] [--mode M]
         [--attention A] [--swap N] [--recompile-limit N | --numz-limits] [--accumulated-limit N]
         [--frames] [--save-artifacts FILE] [--load-artifacts FILE] [--ref SIZE:L] [--out FILE]
         For each length in turn, the variants of --seq run in order (default V,plain,V,plain,V:
         the compiling call, then two plain/compiled pairs); --passes repeats the lengths, later
         passes following --seq-next.
  limit  the recompile-limit probe: run with the variant (default blocks), torch's default limit,
         lengths 1..12 at 270x480, each length plain, compiling call, steady call; then says
         at which length torch stopped compiling (its "hit config.recompile_limit" warning) and
         what each length cost before that. Accepts the options of run.
  table  FILE [FILE...]    a compact Markdown summary per file, then the outputs' SHA-256 of
         the files compared with the first's (a swapped run against an unswapped one).

SIZE = 270 | 540 | 720 | 1080 | 2160 (16:9 output) or HxW (rows, then columns), padded as
seedvr2x pads it (runtime/job.py, padded_size: 1080x1920 to 1104x1920, a latent of 138 x 240,
8,280 tokens).

Environment: TORCHINDUCTOR_CACHE_DIR must name the on-disk cache's directory when a compiled
variant runs (torch's default is under the system's temporary directory; a fresh directory is a
cold run); TRITON_CACHE_DIR likewise. The allocator is memory.startup()'s (the CLI's call).

Usage (seedvr2x's venv python; GPU commands under the shared GPU lock):
  TORCHINDUCTOR_CACHE_DIR=CACHE python compile_dit.py run --model-dir MODELS --size 1080 \\
      --lengths 2,3,4 --variant whole --numz-limits --out whole-2-4.json
  TORCHINDUCTOR_CACHE_DIR=CACHE python compile_dit.py run --model-dir MODELS --size 540 \\
      --lengths 2,3 --variant blocks --swap 20 --out blocks-swap20.json
  TORCHINDUCTOR_CACHE_DIR=CACHE python compile_dit.py limit --model-dir MODELS --out limit.json
  python compile_dit.py table whole-2-4.json
"""
import argparse
import collections
import hashlib
import importlib
import inspect
import json
import logging
import os
import pathlib
import pkgutil
import re
import statistics
import sys
import time
import types

GIB = 1024 ** 3
SIZES = {"270": (270, 480), "540": (540, 960), "720": (720, 1280), "1080": (1080, 1920),
         "2160": (2160, 3840)}
VARIANTS = ("plain", "whole", "blocks")
# numz's defaults for the two limits it sets (inference_cli.py:1464-1467, in this order)
NUMZ_LIMITS = (("cache_size_limit", 64), ("recompile_limit", 128))
DYNAMO_CONFIG = ("recompile_limit", "accumulated_recompile_limit", "fail_on_recompile_limit_hit",
                 "assume_static_by_default", "automatic_dynamic_shapes", "automatic_dynamic_local_pgo",
                 "capture_scalar_outputs", "nested_graph_breaks", "trace_numpy", "specialize_int",
                 "caching_precompile")
INDUCTOR_CONFIG = ("fx_graph_cache", "compile_threads", "force_disable_caches", "autotune_local_cache",
                   "bundle_triton_into_fx_graph_cache")


def log(msg):
    print(f"[compile_dit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_size(text):
    """'1080' (16:9) or 'HxW' -> the output's (height, width), before padding."""
    t = text.strip().lower()
    if t in SIZES:
        return SIZES[t]
    h, w = (int(v) for v in t.split("x"))
    return h, w


def parse_ref(text):
    if text.strip().lower() == "none":
        return None
    size, sep, length = text.partition(":")
    if not sep:
        sys.exit(f"compile_dit: --ref takes SIZE:L or none, got {text!r}")
    return parse_size(size), int(length)


def strip_paths(text):
    """Directories removed from every path of text: results name no machine."""
    return re.sub(r"(?:/[\w.+@~-]+)+/", "", text)


def short(text, limit=400):
    text = strip_paths(" | ".join(line.strip() for line in text.strip().splitlines() if line.strip()))
    return text if len(text) <= limit else text[:limit] + "..."


# ---------------------------------------------------------------- torch's log lines, counted

class Events(logging.Handler):
    """Counts torch's recompilation, graph-break and recompile-limit lines and its other warnings,
    and keeps the distinct reasons of each kind (numbers aside), with one example and a count."""

    KINDS = ("recompiles", "graph_breaks", "limit", "warnings")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.counts = collections.Counter()
        self.reasons = {k: {} for k in self.KINDS}

    def emit(self, record):
        if getattr(record, "_compile_dit_seen", False):
            return
        record._compile_dit_seen = True
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001  (a log line that doesn't format is still counted)
            text = str(record.msg)
        if record.name.endswith("__recompiles") or record.name.endswith("__recompiles_verbose"):
            kind = "recompiles"
        elif record.name.endswith("__graph_breaks"):
            kind = "graph_breaks"
            text = text.split("User code traceback")[0]
        elif "hit config." in text:
            kind = "limit"
            text = text.split("To log all recompilation reasons")[0]
        elif record.levelno >= logging.WARNING and ".__" not in record.name:
            kind = "warnings"
        else:
            return
        self.counts[kind] += 1
        examples = [short(text)]
        if kind == "recompiles":
            # One line per cache entry whose guards failed, "- 0/3: reason": each reason counted
            # under the function recompiled, so that a longer cache doesn't make new reasons.
            lines = [line.strip() for line in text.strip().splitlines()]
            failures = [re.sub(r"^- \d+/\d+: ", "", line) for line in lines if re.match(r"- \d+/\d+: ", line)]
            if failures:
                examples = [short(f"{lines[0]}: {failure}") for failure in failures]
        for example in examples:
            key = re.sub(r"\d+", "N", example)
            entry = self.reasons[kind].setdefault(key, {"count": 0, "example": example})
            entry["count"] += 1

    def snapshot(self):
        return dict(self.counts)

    def reason_counts(self):
        return {k: {key: e["count"] for key, e in reasons.items()} for k, reasons in self.reasons.items()}

    def new_reasons(self, before, n=4):
        """The reasons counted since reason_counts() gave `before`, per kind: example and count."""
        out = {}
        for kind, reasons in self.reasons.items():
            fresh = [{"count": e["count"] - before.get(kind, {}).get(key, 0), "example": e["example"]}
                     for key, e in reasons.items() if e["count"] > before.get(kind, {}).get(key, 0)]
            if fresh:
                out[kind] = sorted(fresh, key=lambda e: -e["count"])[:n]
        return out

    def delta(self, before):
        return {k: self.counts[k] - before.get(k, 0) for k in self.KINDS if self.counts[k] - before.get(k, 0)}

    def top(self, n=10):
        out = {}
        for kind, reasons in self.reasons.items():
            ranked = sorted(reasons.values(), key=lambda e: -e["count"])
            out[kind] = {"distinct": len(ranked), "top": ranked[:n]}
        return out


def install_events(torch):
    """The handler on every logger torch registers, its own stderr handlers silenced for the
    artifact lines (recompiles, graph breaks) and the recompile-limit warnings, which are counted
    and never printed."""
    import torch._logging._internal as internal

    torch._logging.set_logs(recompiles=True, graph_breaks=True)
    events = Events()

    def quiet(record):
        return ".__" not in record.name and "hit config." not in str(record.msg)

    for qname in internal.log_registry.get_log_qnames():
        logger = logging.getLogger(qname)
        for handler in logger.handlers:
            handler.addFilter(quiet)
        logger.addHandler(events)
    return events


# ---------------------------------------------------------------- compilations, from torch's metrics

class Compilations:
    """The frames torch compiled since the last look (torch._dynamo.utils.get_compilation_metrics)."""

    def __init__(self, torch):
        self.utils = torch._dynamo.utils
        self.utils.set_compilation_metrics_limit(1_000_000)
        self.seen = 0
        self.by_code = collections.Counter()
        self.total = 0

    def take(self):
        metrics = list(self.utils.get_compilation_metrics())
        new, self.seen = metrics[self.seen:], len(metrics)
        frames = [m for m in new if getattr(m, "co_name", None)]
        if not frames:
            return None

        def total(field, scale=1.0):
            return round(sum((getattr(m, field, None) or 0) for m in frames) * scale, 3)

        by_code = collections.Counter(
            f"{m.co_name}@{os.path.basename(m.co_filename or '?')}:{m.co_firstlineno}" for m in frames)
        self.by_code.update(by_code)
        self.total += len(frames)
        fails = collections.Counter(short(f"{m.fail_type}: {m.fail_reason}", 200) for m in frames
                                    if getattr(m, "fail_type", None))
        return {
            "frames": len(frames), "code_objects": len(by_code),
            "frame_s": total("entire_frame_compile_time_s"), "backend_s": total("backend_compile_time_s"),
            "inductor_s": total("inductor_compile_time_s"), "triton_s": total("triton_compile_time_us", 1e-6),
            "guards": int(total("guard_count")), "graph_ops": int(total("graph_op_count")),
            "max_cache_size": max((getattr(m, "cache_size", None) or 0) for m in frames),
            "fx_cache_hit": int(total("inductor_fx_local_cache_hit_count")),
            "fx_cache_miss": int(total("inductor_fx_local_cache_miss_count")),
            "aot_cache_hit": int(total("aotautograd_local_cache_hit_count")),
            "aot_cache_miss": int(total("aotautograd_local_cache_miss_count")),
            "restarts": sum(len(getattr(m, "restart_reasons", None) or ()) for m in frames),
            "failed": sum(fails.values()), "fail_reasons": dict(fails.most_common(3)),
            "by_code": dict(by_code.most_common(8)),
        }


# ---------------------------------------------------------------- Python frames started (sys.monitoring)

class Frames:
    """Counts the starts of the model's own Python functions: one that runs compiled no longer
    starts, its code replaced, so a forward's counts tell compiled from fallen back to plain."""

    TOOL = 2  # sys.monitoring.PROFILER_ID

    def __init__(self, enabled):
        self.enabled, self.names, self.counts = enabled, {}, collections.Counter()
        if not enabled:
            return
        from seedvr2x.vendor.models.dit_7b import attention, na, nadit, window
        from seedvr2x.vendor.models.dit_7b.nablocks import mmsr_block

        targets = {
            "dit": nadit.NaDiT.forward, "block": mmsr_block.NaMMSRTransformerBlock.forward,
            "swin": mmsr_block.NaSwinAttention.forward, "attn": attention.FlashAttentionVarlen.forward,
            "win": window.make_720Pwindows_bysize, "swin_win": window.make_shifted_720Pwindows_bysize,
            "unflatten": na.unflatten,
        }
        mon = sys.monitoring
        mon.use_tool_id(self.TOOL, "compile_dit")
        mon.register_callback(self.TOOL, mon.events.PY_START, self._start)
        for name, function in targets.items():
            self.names[function.__code__] = name
            mon.set_local_events(self.TOOL, function.__code__, mon.events.PY_START)

    def _start(self, code, offset):
        self.counts[self.names.get(code, "?")] += 1

    def take(self):
        if not self.enabled:
            return None
        counts, self.counts = dict(self.counts), collections.Counter()
        return counts


# ---------------------------------------------------------------- host and disk figures

def host_stats():
    """This process and its descendants (the compile workers): resident memory, GiB, and the CPU
    seconds they have used, which don't depend on what else loads the machine as wall time does."""
    page, tick = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_CLK_TCK")
    stats = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8", errors="replace") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            # ppid, rss, utime + stime
            stats[int(entry)] = (int(fields[1]), int(fields[21]) * page, (int(fields[11]) + int(fields[12])) / tick)
        except (OSError, IndexError, ValueError):
            continue
    me = os.getpid()
    rss = cpu = 0
    frontier, seen = [me], {me}
    while frontier:
        parent = frontier.pop()
        for pid, (ppid, pid_rss, pid_cpu) in stats.items():
            if ppid == parent and pid not in seen:
                seen.add(pid)
                frontier.append(pid)
                rss += pid_rss
                cpu += pid_cpu
    return {"rss_gib": round(stats.get(me, (0, 0, 0))[1] / GIB, 3), "children_rss_gib": round(rss / GIB, 3),
            "children": len(seen) - 1, "cpu_main_s": round(time.process_time(), 2), "cpu_children_s": round(cpu, 2)}


def dir_stats(path):
    files = size = 0
    if path and os.path.isdir(path):
        for root, _, names in os.walk(path):
            for name in names:
                try:
                    size += os.lstat(os.path.join(root, name)).st_size
                    files += 1
                except OSError:
                    continue
    return {"files": files, "mib": round(size / 2 ** 20, 2)}


# ---------------------------------------------------------------- cache entries per code object

def tracked_codes():
    """The code objects of the DiT's own Python functions (seedvr2x.vendor.models.dit_7b, every
    module and class method), by a short name: where torch's cache entries are counted."""
    package = importlib.import_module("seedvr2x.vendor.models.dit_7b")
    names = [package.__name__] + [i.name for i in pkgutil.walk_packages(package.__path__, package.__name__ + ".")]
    codes = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001, S112  (a module that doesn't import has no code to count)
            continue
        short_name = name.rsplit(".", 1)[-1]
        for obj in list(vars(module).values()):
            if inspect.isfunction(obj):
                functions = [obj]
            elif inspect.isclass(obj) and obj.__module__ == name:
                functions = [f for f in vars(obj).values() if inspect.isfunction(f)]
            else:
                continue
            for f in functions:
                if f.__module__ == name:
                    codes.setdefault(f.__code__, f"{f.__qualname__}@{short_name}:{f.__code__.co_firstlineno}")
    return codes


def cache_entries(torch, codes):
    """The entries torch's cache holds per code object of codes, those with any: one entry is one
    compiled graph and its guards, for one set of conditions."""
    get = torch._C._dynamo.eval_frame._debug_get_cache_entry_list
    out = {}
    for code, name in codes.items():
        try:
            n = len(get(code))
        except Exception:  # noqa: BLE001, S112
            continue
        if n:
            out[name] = n
    return out


def main_entries(entries):
    """The cache entries of the code the variant compiles: a block's forward, else the DiT's."""
    for key in entries:
        if key.startswith("NaMMSRTransformerBlock.forward"):
            return entries[key]
    for key in entries:
        if key.startswith("NaDiT.forward"):
            return entries[key]
    return 0


# ---------------------------------------------------------------- attention backend

def force_attention(backend):
    """The vendor modules' FlashAttention 2 flags (what runtime/model.py's attention_backend reads)
    put False for sdpa, before the load; flash_attn_2 only where they are true."""
    from seedvr2x.vendor.models.dit_3b import attention as attention_3b
    from seedvr2x.vendor.models.dit_7b import attention as attention_7b

    if backend == "sdpa":
        attention_7b.FLASH_ATTN_2_AVAILABLE = attention_3b.FLASH_ATTN_2_AVAILABLE = False
    elif backend == "flash_attn_2" and not (
            attention_7b.FLASH_ATTN_2_AVAILABLE and attention_3b.FLASH_ATTN_2_AVAILABLE):
        sys.exit("compile_dit: --attention flash_attn_2: FlashAttention 2 is not installed")


# ---------------------------------------------------------------- variants

class Handle:
    """A variant put on a DiT (install_dit): `module` runs as the runner's DiT; restore() takes
    it off (the blocks' forward as they were, torch's recompile limit as it was), activate() puts
    it on again, the compiled functions and their caches kept."""

    def __init__(self, module, variant, apply, undo, settings):
        self.module, self.variant, self.settings = module, variant, settings
        self._apply, self._undo, self.active = apply, undo, False

    def activate(self):
        if not self.active:
            self._apply()
            self.active = True
        return self

    def restore(self):
        if self.active:
            self._undo()
            self.active = False

    def __enter__(self):
        return self.activate()

    def __exit__(self, *exc):
        self.restore()


def install_dit(dit, variant, *, dynamic=None, mode="default", recompile_limit=None,
                mark_dynamic=False, backend="inductor"):
    """Put `variant` (plain, whole, blocks) on `dit`, a NaDiT; returns its Handle, active.

    whole: torch.compile(dit, ...), numz's settings; handle.module is the compiled module, to
    run as the runner's DiT. blocks: the class's forward of each kind of block compiled once and
    bound to every block as its `forward` (the DiT itself and its other modules plain, handle.module
    is dit): Module.__call__ of a block, hooks included, stays outside the compiled code.
    dynamic: None or False, the static graphs numz sets (torch.compile's dynamic=False); True,
    dynamic=True; "auto", torch's automatic dynamic shapes (dynamic=None). mark_dynamic: the
    token dimension of the video input marked dynamic before each call, which gives "auto"
    where dynamic is None. recompile_limit: torch._dynamo.config.recompile_limit while the
    variant is on (None: as it is). Importable: no global state besides torch's own config,
    which restore() puts back."""
    import torch

    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}: one of {VARIANTS}")
    if dynamic not in (None, True, False, "auto"):
        raise ValueError(f"dynamic {dynamic!r}: None, False, True or 'auto'")
    if mark_dynamic and dynamic is None:
        dynamic = "auto"
    settings = {"backend": backend, "mode": mode, "fullgraph": False,
                "dynamic": {None: False, False: False, True: True, "auto": None}[dynamic]}
    saved = {}

    def set_limit():
        if recompile_limit is not None:
            saved["limit"] = torch._dynamo.config.recompile_limit
            torch._dynamo.config.recompile_limit = recompile_limit

    def reset_limit():
        if "limit" in saved:
            torch._dynamo.config.recompile_limit = saved.pop("limit")

    def marked(function):
        def forward(self, vid, *args, **kwargs):
            torch._dynamo.mark_dynamic(vid, 0)
            return function(self, vid, *args, **kwargs)
        return forward

    if variant == "plain":
        return Handle(dit, variant, lambda: None, lambda: None, {}).activate()
    if variant == "whole":
        compiled = torch.compile(dit, **settings)
        module = compiled
        if mark_dynamic:
            class Marked(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inner = compiled

                def forward(self, vid, *args, **kwargs):
                    torch._dynamo.mark_dynamic(vid, 0)
                    return self.inner(vid, *args, **kwargs)
            module = Marked()
        return Handle(module, variant, set_limit, reset_limit, settings).activate()
    blocks = list(dit.blocks)
    functions = {}
    for cls in dict.fromkeys(type(b) for b in blocks):
        functions[cls] = torch.compile(cls.forward, **settings)
        if mark_dynamic:
            functions[cls] = marked(functions[cls])

    def apply():
        set_limit()
        for b in blocks:
            b.forward = types.MethodType(functions[type(b)], b)

    def undo():
        for b in blocks:
            b.__dict__.pop("forward", None)
        reset_limit()

    return Handle(dit, variant, apply, undo, settings).activate()


# ---------------------------------------------------------------- one step

def make_latent(torch, dtype, length, h, w):
    """The latent of a window of `length` latents: (length, h, w, 16) in channel-major memory, as
    the encode gives it (runtime/model.py, encode_stream), from a seeded CPU generator."""
    generator = torch.Generator().manual_seed(1000003 * length + 1009 * h + w)
    values = torch.randn((16, length, h, w), generator=generator, dtype=torch.float32)
    return values.to(dtype).permute(1, 2, 3, 0)


def one_step(torch, model, shot, models, module, latent_cpu, seed):
    """shot._window's body for one window, on `module` as the runner's DiT, with the
    noise from shot.draw, the sample timed alone: the measurements and the output on the CPU."""
    device = models.device
    models.runner.dit = module
    models.enter("dit")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    rec = {"alloc_before_gib": torch.cuda.memory_allocated(device) / GIB}
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    with torch.no_grad():
        noise_cpu = shot.draw(models, latent_cpu, seed)
        latent, noise = latent_cpu.to(device), noise_cpu.to(device)
        cond = model.condition(models, noise, latent)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        marks[0].record()
        out = model.sample(models, noise, cond)
        marks[1].record()
        torch.cuda.synchronize(device)
        rec["wall_s"] = time.perf_counter() - started
        rec["gpu_s"] = marks[0].elapsed_time(marks[1]) / 1000
        result = out.to("cpu")
    rec["peak_gib"] = torch.cuda.max_memory_allocated(device) / GIB
    del latent, noise, cond, out
    torch.cuda.synchronize(device)
    rec["alloc_after_gib"] = torch.cuda.memory_allocated(device) / GIB
    free, total = torch.cuda.mem_get_info(device)
    rec["dev_used_gib"] = (total - free) / GIB
    for key, value in rec.items():
        rec[key] = round(value, 4)
    return rec, result


def compare(torch, out, ref):
    """out against ref, both bfloat16 on the CPU: equal bit for bit, or how far apart."""
    if out.shape != ref.shape:
        return {"equal": False, "shape": [list(out.shape), list(ref.shape)]}
    if torch.equal(out, ref):
        return {"equal": True}
    a, b = out.float(), ref.float()
    diff = (a - b).abs()
    std = float(b.std())
    return {"equal": False, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "ref_std": std, "max_abs_over_std": float(diff.max()) / std,
            "mean_abs_over_std": float(diff.mean()) / std,
            "share_differing": float((a != b).float().mean())}


def digest(torch, out):
    """SHA-256 of the output's bytes, bfloat16 read as int16."""
    return hashlib.sha256(out.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- run

def keep_deterministic(torch, how):
    """Keeps torch._inductor.config.deterministic (TORCHINDUCTOR_DETERMINISTIC=1) on while compiling.

    Dynamo calls torch.use_deterministic_algorithms(<prior global state>) after every trace
    (torch/_dynamo/convert_frame.py, torch 2.14), and that call sets the inductor flag as well
    (torch/__init__.py), so the environment variable alone is reset to False at the first
    trace. "inductor": the flag is set again after each such call, eager ops unchanged;
    "torch": torch.use_deterministic_algorithms(True, warn_only=True) once, eager ops too."""
    if how is None:
        return
    import torch._inductor.config as inductor_config
    if how == "torch":
        torch.use_deterministic_algorithms(True, warn_only=True)
        return
    original = torch.use_deterministic_algorithms

    def keep(mode, *, warn_only=False):
        original(mode, warn_only=warn_only)
        inductor_config.deterministic = True

    torch.use_deterministic_algorithms = keep
    inductor_config.deterministic = True


def cmd_run(args):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    variant = args.variant
    if args.seq:
        seq = [v.strip() for v in args.seq.split(",") if v.strip()]
    elif args.cmd == "limit":
        seq = ["plain", variant, variant]
    else:
        seq = [variant, "plain", variant, "plain", variant]
    seq_next = [v.strip() for v in args.seq_next.split(",") if v.strip()] if args.seq_next else seq
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    used = list(dict.fromkeys(v for s in (seq, seq_next) for v in s))
    for v in used:
        if v not in VARIANTS:
            sys.exit(f"compile_dit: unknown variant {v!r} (one of {', '.join(VARIANTS)})")
    compiled_variants = [v for v in used if v != "plain"]
    if len(compiled_variants) > 1:
        sys.exit("compile_dit: one compiled variant per process (their caches would be shared)")
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if compiled_variants and not cache_dir:
        sys.exit("compile_dit: set TORCHINDUCTOR_CACHE_DIR (torch's default is under the temporary directory)")
    target = parse_size(args.size)
    ref = parse_ref(args.ref)
    dynamic = True if args.dynamic else "auto" if args.auto_dynamic else None

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("seedvr2x").setLevel(logging.INFO)
    # The allocator as the CLI chooses it, once, before torch is imported (cli._run).
    from seedvr2x.runtime import memory
    if "torch" in sys.modules:
        sys.exit("compile_dit: torch was imported before the allocator was chosen")
    allocator = memory.startup()
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

    import torch
    from seedvr2x.runtime import job, model, shot

    keep_deterministic(torch, args.keep_deterministic)
    force_attention(args.attention)
    events = install_events(torch)
    compilations = Compilations(torch)
    dynamo, inductor = torch._dynamo.config, torch._inductor.config
    limits_set = {}
    if args.numz_limits:
        for name, value in NUMZ_LIMITS:
            setattr(dynamo, name, value)
            limits_set[name] = value
    if args.accumulated_limit is not None:
        dynamo.accumulated_recompile_limit = limits_set["accumulated_recompile_limit"] = args.accumulated_limit

    device = torch.device("cuda", 0)
    torch.zeros(1, device=device)
    free0, total = torch.cuda.mem_get_info(device)
    log(f"mem_get_info at start: free {free0 / GIB:.2f} of {total / GIB:.2f} GiB; allocator {allocator} "
        f"(PYTORCH_CUDA_ALLOC_CONF={alloc_conf})")
    plan = None
    if args.swap:
        from seedvr2x.runtime.plan import Plan
        plan = Plan(None, swapped=args.swap)
    started = time.perf_counter()
    model_dir = pathlib.Path(os.path.abspath(args.model_dir))
    models = model.load_models(model_dir / args.dit_model, model_dir / args.vae_model, device, plan=plan)
    torch.cuda.synchronize(device)
    load_s = time.perf_counter() - started
    free1, _ = torch.cuda.mem_get_info(device)
    log(f"mem_get_info after load: free {free1 / GIB:.2f} GiB; allocated "
        f"{torch.cuda.memory_allocated(device) / GIB:.2f} GiB")
    dit = models.runner.dit
    if args.attention and models.attention != args.attention:
        sys.exit(f"compile_dit: --attention {args.attention} but the backend is {models.attention}")
    frames = Frames(args.frames)
    codes = tracked_codes()

    try:
        import flash_attn
        flash = flash_attn.__version__
    except Exception:  # noqa: BLE001
        flash = None
    try:
        import triton
        triton_version = triton.__version__
    except Exception:  # noqa: BLE001
        triton_version = None
    res = {
        "argv": [os.path.basename(sys.argv[0]), *sys.argv[1:]], "t_start": time.time(), "cmd": args.cmd,
        "size": list(target), "padded": list(job.padded_size(target)), "lengths": lengths,
        "seq": seq, "seq_next": seq_next, "passes": args.passes, "seed": args.seed,
        "variant": variant, "dynamic": dynamic, "mark_dynamic": args.mark_dynamic, "mode": args.mode,
        "recompile_limit_arg": args.recompile_limit, "swap": args.swap, "backend": args.backend,
        "frames_probe": args.frames,
        "env": {
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "python": sys.version.split()[0],
            "triton": triton_version, "flash_attn": flash, "gpu": torch.cuda.get_device_name(0),
            "attention": models.attention, "dit_model": args.dit_model, "vae_model": args.vae_model,
            "dit_param_dtype": str(next(dit.parameters()).dtype), "compute_dtype": str(model.COMPUTE_DTYPE),
            "blocks": len(dit.blocks),
            "block_types": dict(collections.Counter(type(b).__name__ for b in dit.blocks)),
            "allocator": allocator, "alloc_conf": alloc_conf,
            "TORCHINDUCTOR_CACHE_DIR": cache_dir, "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
            "TORCH_LOGS": os.environ.get("TORCH_LOGS"),
            "dynamo": {k: getattr(dynamo, k, None) for k in DYNAMO_CONFIG},
            "inductor": {k: getattr(inductor, k, None) for k in INDUCTOR_CONFIG},
            "limits_set": limits_set, "loadavg_start": list(os.getloadavg()), "cpus": os.cpu_count(),
        },
        "placement": None if models.placement is None else {
            "swapped": plan.swapped, "device_gib": round(models.placement.device_bytes() / GIB, 3),
            "host_gib": round(models.placement.host_bytes / GIB, 3)},
        "free_at_start_gib": round(free0 / GIB, 3), "free_after_load_gib": round(free1 / GIB, 3),
        "total_gib": round(total / GIB, 3), "load_s": round(load_s, 1),
        "loaded_alloc_gib": round(torch.cuda.memory_allocated(device) / GIB, 4),
        "cache_dir_start": dir_stats(cache_dir), "steps": [], "error": None,
    }
    log(f"loaded in {res['load_s']} s, attention {models.attention}, swap {args.swap}: "
        f"{res['loaded_alloc_gib']} GiB allocated, free {res['free_at_start_gib']} then "
        f"{res['free_after_load_gib']} of {res['total_gib']} GiB; torch {res['env']['torch']}, "
        f"blocks {res['env']['block_types']}, dynamo {res['env']['dynamo']}")
    out_path = os.path.abspath(args.out)

    def write():
        with open(out_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        os.replace(out_path + ".tmp", out_path)

    if args.load_artifacts:
        with open(args.load_artifacts, "rb") as f:
            blob = f.read()
        t0 = time.perf_counter()
        info = torch.compiler.load_cache_artifacts(blob)
        res["artifacts_loaded"] = {
            "mib": round(len(blob) / 2 ** 20, 2), "load_s": round(time.perf_counter() - t0, 2),
            "artifacts": {k: len(v) for k, v in info.artifacts.items()} if info is not None else None,
            "cache_dir_after": dir_stats(cache_dir)}
        log(f"cache artifacts loaded: {res['artifacts_loaded']}")
        del blob
    write()

    handles = {}
    firsts, late, broken = {}, collections.defaultdict(list), set()

    def module_for(name, rec):
        """The DiT module for a step of variant `name`: the other variants taken off first."""
        for other, handle in handles.items():
            if other != name:
                handle.restore()
        if name == "plain":
            return dit
        if name not in handles:
            t0 = time.perf_counter()
            handles[name] = install_dit(dit, name, dynamic=dynamic, mode=args.mode,
                                        recompile_limit=args.recompile_limit,
                                        mark_dynamic=args.mark_dynamic, backend=args.backend)
            rec["wrap_s"] = round(time.perf_counter() - t0, 3)
        else:
            handles[name].activate()
        rec["recompile_limit"] = dynamo.recompile_limit
        return handles[name].module

    def step(role, size, length, name, pass_index=0):
        padded = job.padded_size(size)
        h, w = padded[0] // 8, padded[1] // 8
        rec = {"seq": len(res["steps"]), "role": role, "pass": pass_index, "size": list(size), "L": length,
               "tokens": length * (h // 2) * (w // 2), "variant": name, "t": round(time.time(), 2)}
        res["steps"].append(rec)
        if (name, tuple(size), length) in broken:
            rec["error"] = "skipped: this variant failed at this length"
            return rec
        module = module_for(name, rec)
        latent = make_latent(torch, model.COMPUTE_DTYPE, length, h, w)
        before, before_reasons, host = events.snapshot(), events.reason_counts(), host_stats()
        frames.take()
        try:
            measured, out = one_step(torch, model, shot, models, module, latent, args.seed)
        except Exception as e:  # (a variant that doesn't compile is a result)
            models.runner.dit = dit
            rec["error"] = f"{type(e).__name__}: {short(str(e), 900)}"
            rec["compiles"], rec["events"] = compilations.take(), events.delta(before)
            broken.add((name, tuple(size), length))
            log(f"{role} {name} L={length}: FAILED {rec['error'][:300]}")
            write()
            if "CUDA error" in str(e) or isinstance(e, torch.OutOfMemoryError):
                raise
            return rec
        finally:
            models.runner.dit = dit
        rec.update(measured)
        rec["compiles"], rec["events"], rec["frames"] = compilations.take(), events.delta(before), frames.take()
        rec["reasons"] = events.new_reasons(before_reasons)
        rec["cache_entries"] = cache_entries(torch, codes)
        after = host_stats()
        rec.update(after)
        rec["cpu_main_s"] = round(after["cpu_main_s"] - host["cpu_main_s"], 2)
        rec["cpu_children_s"] = round(after["cpu_children_s"] - host["cpu_children_s"], 2)
        rec["finite"] = bool(torch.isfinite(out.float()).all())
        rec["sha256"] = digest(torch, out)
        key = (tuple(size), length)
        if (key, name) in firsts:
            rec["vs_first"] = compare(torch, out, firsts[key, name])
            rec["equal_first"] = rec["vs_first"]["equal"]
        else:
            firsts[key, name] = out
        if name == "plain":
            if "equal_first" not in rec:  # the first plain output of this length: the ones before it
                for other, other_out in late.pop(key, []):
                    other["vs_plain"] = compare(torch, other_out, out)
        elif (key, "plain") in firsts:
            rec["vs_plain"] = compare(torch, out, firsts[key, "plain"])
        else:
            late[key].append((rec, out))
        c = rec["compiles"]
        log(f"{role} p{pass_index} {name} {size[0]}x{size[1]} L={length} ({rec['tokens']} tokens): "
            f"GPU {rec['gpu_s']:.3f} s, wall {rec['wall_s']:.2f} s, peak {rec['peak_gib']:.2f} GiB, "
            f"allocated after {rec['alloc_after_gib']:.2f}, RSS {rec['rss_gib']}+{rec['children_rss_gib']} GiB, "
            f"CPU {rec['cpu_main_s']}+{rec['cpu_children_s']} s; "
            f"compiled {compiled_ok(rec)} frames"
            + (f" in {c['frame_s']} s (cache hits {c['fx_cache_hit']}, misses {c['fx_cache_miss']})" if c else "")
            + f"; events {rec['events']}; cache entries {rec['cache_entries']}; frames {rec['frames']}; "
            f"bits vs plain {rec.get('vs_plain', {}).get('equal')}, vs own first {rec.get('equal_first')}")
        write()
        return rec

    try:
        step("warmup", parse_size("270"), 2, "plain")
        if ref:
            step("ref", *ref, "plain")
        for pass_index in range(args.passes):
            for length in lengths:
                for name in (seq, seq_next)[min(pass_index, 1)]:
                    step("step", target, length, name, pass_index)
        if ref:
            step("ref", *ref, "plain")
    except Exception as e:  # (recorded, then raised: the context may be unusable)
        res["error"] = f"{type(e).__name__}: {short(str(e), 600)}"
        write()
        raise
    finally:
        res["events"], res["event_reasons"] = dict(events.counts), events.top(10)
        res["compiled_frames"] = compilations.total
        res["compiled_by_code"] = dict(compilations.by_code.most_common(40))
        res["cache_dir_end"] = dir_stats(cache_dir)
        res["env"]["loadavg_end"] = list(os.getloadavg())
        res["t_end"] = time.time()
        write()
    if args.save_artifacts:
        t0 = time.perf_counter()
        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            res["artifacts_saved"] = None
        else:
            blob, info = saved
            with open(args.save_artifacts, "wb") as f:
                f.write(blob)
            res["artifacts_saved"] = {"mib": round(len(blob) / 2 ** 20, 2),
                                      "save_s": round(time.perf_counter() - t0, 2),
                                      "artifacts": {k: len(v) for k, v in info.artifacts.items()}}
        log(f"cache artifacts saved: {res['artifacts_saved']}")
        write()
    log(f"done -> {out_path}: {res['compiled_frames']} frames compiled, events {res['events']}, "
        f"cache directory {res['cache_dir_start']} -> {res['cache_dir_end']}")
    print_table(res, os.path.basename(out_path))
    if args.cmd == "limit":
        print_limit(res)


# ---------------------------------------------------------------- table

def spread(values, digits=3):
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.median(values):.{digits}f} [{min(values):.{digits}f}-{max(values):.{digits}f}]"


def bits(steps):
    """'=' when every output equals the plain one, else how far the worst is."""
    seen = [s["vs_plain"] for s in steps if s.get("vs_plain")]
    if not seen:
        return "?"
    own = [s["equal_first"] for s in steps if "equal_first" in s]
    own_text = "equal" if own and all(own) else "DIFFER" if own else "?"
    if all(v["equal"] for v in seen):
        return f"= ({len(seen)}); own runs {own_text} ({len(own)})"
    worst = max((v for v in seen if not v["equal"]), key=lambda v: v.get("max_abs", 0))
    return (f"differ: max {worst.get('max_abs_over_std', 0):.3g} std, mean {worst.get('mean_abs_over_std', 0):.3g} std, "
            f"{100 * worst.get('share_differing', 0):.1f}% of elements; own runs {own_text} ({len(own)})")


def compiled_ok(step):
    """The frames a step compiled, those refused (the recompile limit, an error) left out."""
    c = step.get("compiles")
    return c["frames"] - c["failed"] if c else 0


def peak(steps):
    return f"{max(s['peak_gib'] for s in steps):.2f}" if steps else "-"


def print_table(res, name):
    steps = [s for s in res["steps"] if "gpu_s" in s]
    env = res["env"]
    print(f"\n### {name}: {res['size'][0]}x{res['size'][1]} (padded {res['padded'][0]}x{res['padded'][1]}), "
          f"seq {','.join(res['seq'])}" + (f" then {','.join(res['seq_next'])}" if res["passes"] > 1 else "")
          + f", passes {res['passes']}; dynamic {res.get('dynamic')}, mark_dynamic {res.get('mark_dynamic')}, "
          f"swap {res.get('swap')}")
    print(f"torch {env['torch']}, attention {env['attention']}, allocator {env.get('alloc_conf')}, limits set "
          f"{env['limits_set']}, recompile_limit arg {res.get('recompile_limit_arg')} (dynamo "
          f"{env['dynamo']['recompile_limit']} / accumulated {env['dynamo']['accumulated_recompile_limit']}); "
          f"free {res['free_at_start_gib']} -> {res['free_after_load_gib']} GiB, loaded {res['loaded_alloc_gib']} GiB; "
          f"placement {res.get('placement')}; load average {env['loadavg_start'][0]:.1f}")
    refs = [s for s in steps if s["role"] == "ref"]
    if refs:
        print("reference (plain) GPU s: " + ", ".join(f"{s['gpu_s']:.2f}" for s in refs))
    print("\n| pass | L | tokens | variant | plain s | steady s | ratio to plain | compiling call wall s (CPU s: process + workers) "
          "| frames compiled (cache hit/miss) | recompiles / graph breaks / limit (compiling call) "
          "| peak plain | peak steady | peak compiling | allocated after | bits |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    keys = list(dict.fromkeys((s["pass"], s["L"]) for s in steps if s["role"] == "step"))
    for pass_index, length in keys:
        group = [s for s in steps if s["role"] == "step" and s["pass"] == pass_index and s["L"] == length]
        tokens = group[0]["tokens"]
        plain = [s for s in group if s["variant"] == "plain"]
        plain_s = [s["gpu_s"] for s in plain]
        for variant in dict.fromkeys(s["variant"] for s in group if s["variant"] != "plain"):
            mine = [s for s in group if s["variant"] == variant]
            compiling = [s for s in mine if compiled_ok(s) or s.get("events", {}).get("limit")]
            steady = [s for s in mine if s not in compiling]
            ratios, last_plain = [], None
            for s in group:
                if s["variant"] == "plain":
                    last_plain = s
                elif s["variant"] == variant and s in steady and last_plain is not None:
                    ratios.append(s["gpu_s"] / last_plain["gpu_s"])
            walls = " + ".join(f"{s['wall_s']:.1f} ({s.get('cpu_main_s', 0):.0f}+{s.get('cpu_children_s', 0):.0f})"
                               for s in compiling) or "-"
            done = [s["compiles"] for s in mine if s.get("compiles")]
            cell = (f"{sum(compiled_ok(s) for s in mine)} "
                    f"({sum(c['fx_cache_hit'] for c in done)}/{sum(c['fx_cache_miss'] for c in done)})")
            failed = sum(c["failed"] for c in done)
            cell += f", {failed} failed" if failed else ""
            ev = collections.Counter()
            for s in compiling:
                ev.update(s.get("events", {}))
            counts = f"{ev['recompiles']} / {ev['graph_breaks']} / {ev['limit']}"
            cells = [pass_index, length, tokens, variant, spread(plain_s),
                     spread([s["gpu_s"] for s in steady]), f"{spread(ratios)} ({len(ratios)})", walls, cell, counts,
                     peak(plain), peak(steady), peak(compiling), f"{mine[-1]['alloc_after_gib']:.2f}", bits(mine)]
            print("| " + " | ".join(str(c) for c in cells) + " |")
        if not all(s.get("equal_first", True) for s in plain):
            print(f"| {pass_index} | {length} | {tokens} | plain | PLAIN OUTPUTS DIFFER BETWEEN RUNS |")
    for s in res["steps"]:
        if s.get("error"):
            print(f"- error, {s['variant']} L={s['L']} pass {s['pass']}: {s['error'][:500]}")
    first_entries = [s for s in steps if s["role"] == "step" and s["variant"] != "plain" and compiled_ok(s)
                     and s.get("cache_entries")]
    if first_entries:
        print("\ncache entries of the compiled code after each compiled step (L: entries): "
              + "; ".join(f"{s['L']}: " + ", ".join(f"{k.split('@')[0]} {v}" for k, v in s["cache_entries"].items()
                                                    if v > 0)[:300] for s in first_entries[:12]))
    print(f"\nframes compiled {res.get('compiled_frames')}, events {res.get('events')}, cache directory "
          f"{res.get('cache_dir_start')} -> {res.get('cache_dir_end')}; artifacts saved {res.get('artifacts_saved')}, "
          f"loaded {res.get('artifacts_loaded')}")
    by_code = res.get("compiled_by_code") or {}
    if by_code:
        print("compiled per code object: " + ", ".join(f"{k} x{v}" for k, v in list(by_code.items())[:14]))
    for kind, info in (res.get("event_reasons") or {}).items():
        if info["distinct"]:
            print(f"{kind}: {info['distinct']} distinct reasons; the first: "
                  + "; ".join(f"[x{e['count']}] {e['example'][:220]}" for e in info["top"][:3]))
    if any(s.get("frames") for s in steps):
        seen = {}
        for s in steps:
            if s["role"] == "step" and s.get("frames") is not None:
                seen.setdefault((s["variant"], bool(s.get("compiles")), json.dumps(s["frames"], sort_keys=True)), []).append(s["L"])
        print("Python frames started per step (variant, compiling, counts: lengths):")
        for (variant, compiling, counts), ls in seen.items():
            print(f"- {variant}{' (compiling)' if compiling else ''} {counts}: L {sorted(set(ls))} x{len(ls)}")


def print_limit(res):
    """The recompile-limit probe's answer from a `limit` run: per length (plain, compiling call,
    steady call), then the first length whose compiling call logged the recompile-limit warning."""
    variant = res["variant"]
    steps = [s for s in res["steps"] if s["role"] == "step" and "gpu_s" in s]
    print(f"\n#### recompile-limit probe: {variant}, dynamic {res.get('dynamic')}, size "
          f"{res['size'][0]}x{res['size'][1]}, recompile_limit {res['env']['dynamo']['recompile_limit']} "
          f"(arg {res.get('recompile_limit_arg')})")
    print("| L | tokens | plain s | compiling call wall s | frames compiled | frame compile s | recompiles / graph breaks "
          "/ limit lines | entries of the main code | steady s | steady / plain | runs compiled |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    first_limit, compiled_lengths, ratios = None, [], {}
    for length in res["lengths"]:
        group = [s for s in steps if s["L"] == length]
        plain = [s for s in group if s["variant"] == "plain"]
        mine = [s for s in group if s["variant"] == variant]
        if not plain or not mine:
            continue
        first = mine[0]
        later = mine[1:]
        c = first.get("compiles") or {}
        ev = first.get("events", {})
        limit = ev.get("limit", 0) + sum(s.get("events", {}).get("limit", 0) for s in later)
        ok = compiled_ok(first)
        if ok:
            compiled_lengths.append(length)
        if limit and first_limit is None:
            first_limit = length
        ratio = later[-1]["gpu_s"] / plain[-1]["gpu_s"] if later else None
        ratios[length] = ratio
        runs = "yes" if ok and not limit else "NO (limit hit)" if limit else "from the cache" if later else "?"
        steady_s = f"{later[-1]['gpu_s']:.3f}" if later else "-"
        ratio_s = f"{ratio:.3f}" if ratio is not None else "-"
        print(f"| {length} | {first['tokens']} | {plain[-1]['gpu_s']:.3f} | {first['wall_s']:.1f} | {ok} | "
              f"{c.get('frame_s', 0)} | {ev.get('recompiles', 0)} / {ev.get('graph_breaks', 0)} / {limit} | "
              f"{main_entries(first.get('cache_entries', {}))} | {steady_s} | {ratio_s} | {runs} |")
    if first_limit is None:
        print(f"\nNo recompile-limit warning in lengths {res['lengths'][0]}..{res['lengths'][-1]}; "
              f"compiled at lengths {compiled_lengths}.")
    else:
        print(f"\nFirst recompile-limit warning (any code object) at L={first_limit}; lengths whose compiling call "
              f"compiled frames: {compiled_lengths}; steady/plain ratios from it on: "
              + ", ".join(f"L={k}: {v:.3f}" for k, v in ratios.items() if k >= first_limit and v is not None))
    # Which code hit the limit, at which length first: the DiT's own block (or the DiT) is the
    # code that decides whether a length runs compiled; the small functions inside it fall back
    # to plain code one by one, their frames only.
    hits = {}
    for s in steps:
        for text in (s.get("reasons") or {}).get("limit", []):
            m = re.search(r"function: '([^']+)' \(([^)]+)\)", text["example"])
            hits.setdefault(f"{m.group(1)} ({m.group(2)})" if m else text["example"][:80], s["L"])
    if hits:
        print("first length at which each code object hit the limit: "
              + "; ".join(f"{k}: L={v}" for k, v in hits.items()))


def cmd_table(args):
    results = []
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            results.append((os.path.basename(path), json.load(f)))
    for name, res in results:
        print_table(res, name)
        if res.get("cmd") == "limit":
            print_limit(res)
    if len(results) > 1:
        print("\n#### outputs compared across files (SHA-256 of each variant's first output at a length, "
              "against the first file's)")
        first_name, first = results[0]
        seen = {}
        for s in first["steps"]:
            if "sha256" in s and s["role"] == "step":
                seen.setdefault((s["variant"], tuple(s["size"]), s["L"]), s["sha256"])
        for name, res in results[1:]:
            mine = {}
            for s in res["steps"]:
                if "sha256" in s and s["role"] == "step":
                    mine.setdefault((s["variant"], tuple(s["size"]), s["L"]), s["sha256"])
            for key, sha in mine.items():
                if key in seen:
                    print(f"- {name} vs {first_name}: {key[0]} {key[1][0]}x{key[1][1]} L={key[2]}: "
                          + ("EQUAL" if sha == seen[key] else "differs"))


def add_options(parser, limit=False):
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dit-model", default="seedvr2_ema_7b_fp16.safetensors")
    parser.add_argument("--vae-model", default="ema_vae_fp16.safetensors")
    parser.add_argument("--size", default="270x480" if limit else "1080")
    parser.add_argument("--lengths", default=",".join(str(i) for i in range(1, 13)) if limit else None,
                        required=not limit, help="window lengths in latents, in order: 2,3,4")
    parser.add_argument("--variant", default="blocks" if limit else None, choices=VARIANTS,
                        help="the compiled variant of the default sequence")
    parser.add_argument("--seq", help="the variants run per length, in order (default: VARIANT,plain,VARIANT,...)")
    parser.add_argument("--seq-next", help="the variants run per length in the passes after the first")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="dynamic=True: one graph for every length if torch can")
    parser.add_argument("--auto-dynamic", action="store_true", help="torch's automatic dynamic shapes")
    parser.add_argument("--mark-dynamic", action="store_true",
                        help="mark the token dimension of the video input dynamic before each call")
    parser.add_argument("--mode", default="default", help="torch.compile's mode")
    parser.add_argument("--attention", choices=("sdpa", "flash_attn_2"),
                        help="force the backend (sdpa where FlashAttention 2 is installed)")
    parser.add_argument("--swap", type=int, default=0, help="BlockSwap: the last N blocks swapped, as a Plan")
    parser.add_argument("--numz-limits", action="store_true",
                        help="set dynamo's limits as numz does: cache_size_limit 64, then recompile_limit 128")
    parser.add_argument("--recompile-limit", type=int, help="torch's default (8) where not given")
    parser.add_argument("--accumulated-limit", type=int)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--frames", action="store_true", help="count the model's Python frames per step")
    parser.add_argument("--save-artifacts", metavar="FILE", help="torch.compiler.save_cache_artifacts at the end")
    parser.add_argument("--load-artifacts", metavar="FILE", help="torch.compiler.load_cache_artifacts first")
    parser.add_argument("--ref", default="none", help="a reference step, SIZE:L, at the start and the end, or none")
    parser.add_argument("--keep-deterministic", choices=("inductor", "torch"),
                        help="keep inductor's deterministic mode on past dynamo's reset of it "
                        "(see keep_deterministic); inductor: the flag only, torch: torch's too")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="compile_dit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="measure plain and compiled steps per window length")
    add_options(run)
    run.set_defaults(func=cmd_run)
    limit = sub.add_parser("limit", help="the recompile-limit probe: lengths 1..12 at 270x480, default limit")
    add_options(limit, limit=True)
    limit.set_defaults(func=cmd_run)
    table = sub.add_parser("table", help="Markdown summary of results")
    table.add_argument("results", nargs="+")
    table.set_defaults(func=cmd_table)
    args = parser.parse_args()
    if args.cmd != "table" and not args.seq and not args.variant:
        parser.error("give --variant or --seq")
    args.func(args)


if __name__ == "__main__":
    main()
