#!/usr/bin/env python3
"""Attention probe for SeedVR2's DiT: what each varlen attention call contains and what runs.

Runs inference_cli.py in-process after installing an import hook that wraps, without
touching the checkout:

- FlashAttentionVarlen.forward (dit_7b and dit_3b): one record per attention call, with the
  number of sequences (windows), their length distribution, whether they are all equal, the
  kernel(s) that really ran, and the call's GPU time (CUDA events)
- NaSwinAttention.forward (mmsr_block): the layer's window method, video grid and text length
- the kernels the compatibility wrappers look up at call time (sageattn_varlen,
  sageattn_blackwell, flash_attn_{2,3}_varlen_func): to tell SA3 from its SA2 fallback
- NaDiT.forward: GPU time of each DiT forward, to put attention time in proportion

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python attn_probe.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap attn_probe.py -- <CLI args>     # writes <runs-dir>/NAME.attn.json
  python3 attn_probe.py --summary runs/*.attn.json                   # Markdown tables

Output: $ATTN_PROBE_OUT, else <$BENCH_LOG without .log>.attn.json, else attn_probe.json.
ATTN_PROBE_TIMING=0 skips the CUDA events (statistics only).
ATTN_PROBE_PROFILE=N runs DiT forward N (0 = first) under torch.profiler and writes the
kernels sorted by self GPU time to <output without .json>.prof.txt.

Timing: the call's lengths are read with a device sync *before* its start event, so the
event pair brackets the attention work only (dtype casts, the wrapper's own .item() syncs and
kernels). Each DiT forward ends with a sync. Expect the instrumented run to be a little slower
than a clean one: compare DiT times with uninstrumented runs.
"""
import atexit
import importlib.abc
import json
import os
import runpy
import sys
import time
from collections import Counter, defaultdict

TIMING = os.environ.get("ATTN_PROBE_TIMING", "1") != "0"
PROFILE = int(os.environ["ATTN_PROBE_PROFILE"]) if os.environ.get("ATTN_PROBE_PROFILE") else None


class State:
    def __init__(self):
        self.calls = []          # one dict per attention call
        self.forwards = []       # one dict per DiT forward
        self.pending = []        # (start, end, record) whose elapsed time is not read yet
        self.kernels = []        # kernels hit during the current attention call
        self.ctx = {}            # set by NaSwinAttention for the call that follows
        self.forward_idx = -1
        self.layer_idx = 0
        self.depth = 0
        self.patched = []
        self.written = False
        self.profile = None      # torch.profiler table of the profiled DiT forward


S = State()


def _torch():
    import torch
    return torch


# ---------------------------------------------------------------- patches

def patch_compat(mod):
    """Tag the kernels the call_*_varlen wrappers look up as module globals at call time."""
    tags = {"sageattn_varlen": "sa2_varlen", "sageattn_blackwell": "sa3_blackwell",
            "flash_attn_2_varlen_func": "fa2_varlen", "flash_attn_3_varlen_func": "fa3_varlen"}
    for name, tag in tags.items():
        fn = getattr(mod, name, None)
        if fn is None:
            continue

        def wrapped(*a, __fn=fn, __tag=tag, **k):
            S.kernels.append(__tag)
            return __fn(*a, **k)
        setattr(mod, name, wrapped)
        S.patched.append(f"{mod.__name__}.{name}")


def patch_attention(mod):
    cls = getattr(mod, "FlashAttentionVarlen", None)
    if cls is None:
        return
    orig = cls.forward

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
        torch = _torch()
        lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()   # syncs the device
        lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()
        S.kernels = []
        if TIMING:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        out = orig(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
        if TIMING:
            ev1.record()
        cnt = Counter(lens_q)
        kern = "+".join(dict.fromkeys(S.kernels)) or ("sdpa_loop" if self.attention_mode == "sdpa" else "?")
        rec = {
            "forward": S.forward_idx, "layer": S.layer_idx, "mode": self.attention_mode, "kernel": kern,
            "n_seq": len(lens_q), "tokens": sum(lens_q), "min": min(lens_q), "max": max(lens_q),
            "distinct": len(cnt), "uniform": len(cnt) == 1 and len(set(lens_k)) == 1,
            "lens": sorted(cnt.items()), "heads": q.shape[1], "head_dim": q.shape[2],
            "sum_l2": sum(n * n for n in lens_q), "dtype": str(q.dtype).replace("torch.", ""),
            **S.ctx,
        }
        S.calls.append(rec)
        S.ctx = {}
        S.layer_idx += 1
        if TIMING:
            S.pending.append((ev0, ev1, rec))
        return out

    cls.forward = forward
    S.patched.append(f"{mod.__name__}.FlashAttentionVarlen.forward")


def patch_block(mod):
    cls = getattr(mod, "NaSwinAttention", None)
    if cls is None:
        return
    orig = cls.forward

    def forward(self, vid, txt, vid_shape, txt_shape, cache, *a, **k):
        S.ctx = {"window_method": getattr(self, "window_method", None),
                 "window": list(getattr(self, "window", []) or []),
                 "vid_shape": vid_shape.tolist(), "txt_shape": txt_shape.tolist()}
        return orig(self, vid, txt, vid_shape, txt_shape, cache, *a, **k)

    cls.forward = forward
    S.patched.append(f"{mod.__name__}.NaSwinAttention.forward")


def patch_nadit(mod):
    for cname in ("NaDiT", "NaDiTUpscaler"):
        cls = getattr(mod, cname, None)
        if cls is None:
            continue
        orig = cls.forward

        def forward(self, *a, __orig=orig, __cname=cname, **k):
            if S.depth:  # nested DiT class: the outer one is timed
                return __orig(self, *a, **k)
            torch = _torch()
            S.depth += 1
            S.forward_idx += 1
            S.layer_idx = 0
            rec = {"forward": S.forward_idx, "class": __cname}
            if TIMING:
                torch.cuda.synchronize()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            prof = None
            if PROFILE == S.forward_idx:
                from torch.profiler import profile, ProfilerActivity
                prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                prof.__enter__()
            t0 = time.perf_counter()
            try:
                out = __orig(self, *a, **k)
            finally:
                S.depth -= 1
                if prof is not None:
                    torch.cuda.synchronize()
                    prof.__exit__(None, None, None)
                    S.profile = prof.key_averages().table(sort_by="self_device_time_total", row_limit=60)
                    rec["profiled"] = True
            if TIMING:
                ev1.record()
                torch.cuda.synchronize()
                rec["gpu_ms"] = round(ev0.elapsed_time(ev1), 3)
                resolve()
            rec["wall_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            calls = [c for c in S.calls if c["forward"] == rec["forward"]]
            rec["attn_calls"] = len(calls)
            rec["attn_ms"] = round(sum(c.get("gpu_ms", 0) for c in calls), 3)
            S.forwards.append(rec)
            return out

        cls.forward = forward
        S.patched.append(f"{mod.__name__}.{cname}.forward")


def resolve():
    if not S.pending:
        return
    _torch().cuda.synchronize()
    for ev0, ev1, rec in S.pending:
        rec["gpu_ms"] = round(ev0.elapsed_time(ev1), 4)
    S.pending = []


HOOKS = (
    (".optimization.compatibility", patch_compat),
    (".dit_7b.attention", patch_attention), (".dit_3b.attention", patch_attention),
    (".dit_7b.nablocks.mmsr_block", patch_block), (".dit_3b.nablocks.mmsr_block", patch_block),
    (".dit_7b.nadit", patch_nadit), (".dit_3b.nadit", patch_nadit),
)


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


# ---------------------------------------------------------------- summary

def txt_tokens(c):
    """Text tokens per window: SeedVR2 appends the whole (per-video) text to every window."""
    out = 0
    for row in c.get("txt_shape", []):
        n = 1
        for x in row:
            n *= x
        out += n
    return out


def summarize(calls, forwards):
    s = {"calls": len(calls)}
    if not calls:
        return s
    d = calls[0]["heads"] * calls[0]["head_dim"]
    s["modes"] = dict(Counter(c["mode"] for c in calls))
    s["kernels"] = dict(Counter(c["kernel"] for c in calls))
    s["uniform_calls"] = sum(c["uniform"] for c in calls)
    s["uniform_share"] = round(s["uniform_calls"] / len(calls), 4)
    sa3 = sum("sa3" in c["kernel"] for c in calls)
    s["sa3_calls"] = sa3
    s["sa3_share"] = round(sa3 / len(calls), 4)
    hist, hist_vid = Counter(), Counter()
    for c in calls:
        txt = txt_tokens(c) if len(c.get("vid_shape", [])) == 1 else None
        for L, n in c["lens"]:
            hist[L] += n
            if txt is not None:
                hist_vid[L - txt] += n
    s["seqlen_hist"] = sorted(hist.items())
    s["vid_tokens_hist"] = sorted(hist_vid.items())
    s["txt_tokens"] = sorted({txt_tokens(c) for c in calls if "txt_shape" in c})
    s["vid_shape"] = sorted({json.dumps(c.get("vid_shape")) for c in calls})
    # one entry per distinct call layout (window method + length multiset)
    sig = defaultdict(lambda: {"calls": 0, "gpu_ms": 0.0})
    for c in calls:
        key = (c.get("window_method"), c["n_seq"], json.dumps(c["lens"]))
        e = sig[key]
        e["calls"] += 1
        e["gpu_ms"] += c.get("gpu_ms", 0)
        e.update(window_method=key[0], n_seq=c["n_seq"], min=c["min"], max=c["max"], distinct=c["distinct"],
                 uniform=c["uniform"], kernel=c["kernel"], lens=c["lens"], tokens=c["tokens"])
    med = {}
    for c in calls:
        if "gpu_ms" in c:
            med.setdefault((c.get("window_method"), json.dumps(c["lens"])), []).append(c["gpu_ms"])
    med = {k: sorted(v)[len(v) // 2] for k, v in med.items()}
    for v in sig.values():
        v["median_ms"] = med.get((v["window_method"], json.dumps(v["lens"])))
    s["layouts"] = [dict(v, gpu_ms=round(v["gpu_ms"], 3)) for v in sig.values()]
    # FLOP model: per layer, linear/MLP ~ 24 d^2 per token, attention 4 L^2 d per sequence
    attn_fl = sum(4 * c["sum_l2"] * d for c in calls)
    lin_fl = sum(24 * d * d * c["tokens"] for c in calls)
    s["flops_attn_share_model"] = round(attn_fl / (attn_fl + lin_fl), 4)
    s["attn_tflop"] = round(attn_fl / 1e12, 3)
    if any("gpu_ms" in c for c in calls):
        att_ms = sum(c.get("gpu_ms", 0) for c in calls)
        s["attn_gpu_ms"] = round(att_ms, 2)
        s["attn_tflops_effective"] = round(attn_fl / (att_ms / 1e3) / 1e12, 1) if att_ms else None
        # the very first call JIT-compiles / autotunes (Triton): estimate the steady state by
        # charging every call the median time of its layout
        est = sum(med[(c.get("window_method"), json.dumps(c["lens"]))] for c in calls if "gpu_ms" in c)
        s["first_call_ms"] = calls[0].get("gpu_ms")
        s["attn_gpu_ms_steady"] = round(est, 2)
        s["attn_tflops_steady"] = round(attn_fl / (est / 1e3) / 1e12, 1) if est else None
    if forwards:
        s["dit_forwards"] = len(forwards)
        fw = [f for f in forwards if "gpu_ms" in f]
        if fw:
            s["dit_gpu_ms"] = round(sum(f["gpu_ms"] for f in fw), 2)
            s["attn_share_of_dit"] = round(sum(f["attn_ms"] for f in fw) / s["dit_gpu_ms"], 4)
            if "attn_gpu_ms_steady" in s:  # DiT time with the attention warmup replaced by the steady estimate
                dit_est = s["dit_gpu_ms"] - s["attn_gpu_ms"] + s["attn_gpu_ms_steady"]
                s["attn_share_of_dit_steady"] = round(s["attn_gpu_ms_steady"] / dit_est, 4)
            rest = fw[1:]  # without the first forward (warmup)
            if rest:
                s["attn_share_of_dit_after_first"] = round(
                    sum(f["attn_ms"] for f in rest) / sum(f["gpu_ms"] for f in rest), 4)
    return s


def write_out():
    if S.written:
        return
    S.written = True
    try:
        resolve()
    except Exception as e:  # CUDA may be unusable after an error
        print(f"attn_probe: could not resolve pending events: {e}", file=sys.stderr)
    out = os.environ.get("ATTN_PROBE_OUT")
    if not out:
        log = os.environ.get("BENCH_LOG")
        out = (log[:-4] if log and log.endswith(".log") else log or "attn_probe") + ".attn.json"
    data = {"argv": sys.argv, "timing": TIMING, "patched": S.patched,
            "summary": summarize(S.calls, S.forwards), "forwards": S.forwards, "calls": S.calls}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    if S.profile:
        with open(out[:-5] + ".prof.txt", "w", encoding="utf-8") as f:
            f.write(S.profile)
    sm = data["summary"]
    print(f"attn_probe: {sm['calls']} attention calls, kernels {sm.get('kernels')}, "
          f"uniform {sm.get('uniform_share')}, attention/DiT GPU {sm.get('attn_share_of_dit')} "
          f"(steady {sm.get('attn_share_of_dit_steady')}) -> {out}",
          file=sys.stderr)


# ---------------------------------------------------------------- report

def report(paths):
    rows1 = ["| Run | DiT fwd | Calls | Seqs/call | Seq length min–max (distinct) | Text tokens | Uniform calls "
             "| Kernels | Attn GPU ms (steady) | DiT GPU ms | Attn share (steady) | Attn share (FLOP model) "
             "| Attn TFLOPS (steady) |",
             "|" + "---|" * 13]
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        s = summarize(data["calls"], data["forwards"])  # recomputed: older files get new fields
        name = os.path.basename(p).replace(".attn.json", "")
        if not s.get("calls"):
            rows1.append(f"| {name} | – | 0 | | | | | | | | | | |")
            continue
        ns = sorted({c["n_seq"] for c in data["calls"]})
        lens = [L for L, _ in s["seqlen_hist"]]
        share = s.get("attn_share_of_dit_steady")
        rows1.append("| " + " | ".join([
            name, str(s.get("dit_forwards", "–")), str(s["calls"]),
            "–".join(map(str, (ns[0], ns[-1]))) if len(ns) > 1 else str(ns[0]),
            f"{lens[0]}–{lens[-1]} ({len(lens)})", ",".join(map(str, s.get("txt_tokens", []))),
            f"{s['uniform_calls']} ({s['uniform_share']:.0%})",
            ", ".join(f"{k} {v}" for k, v in s["kernels"].items()),
            f"{s.get('attn_gpu_ms', 0):.0f} ({s.get('attn_gpu_ms_steady', 0):.0f})", f"{s.get('dit_gpu_ms', 0):.0f}",
            f"{share:.1%}" if share is not None else "–",
            f"{s['flops_attn_share_model']:.1%}", str(s.get("attn_tflops_steady", "–")),
        ]) + " |")
    print("\n".join(rows1))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        s = summarize(data["calls"], data["forwards"])
        if not s.get("calls"):
            continue
        print(f"\n{os.path.basename(p)}: video grid {', '.join(s['vid_shape'])}")
        print("| Window method | Calls | Seqs | Lengths (length×count) | Uniform | Kernel | Median GPU ms/call |")
        print("|---|---|---|---|---|---|---|")
        for e in s["layouts"]:
            lens = " ".join(f"{L}×{n}" for L, n in e["lens"])
            print(f"| {e['window_method']} | {e['calls']} | {e['n_seq']} | {lens} | {e['uniform']} | {e['kernel']} "
                  f"| {e['median_ms'] or 0:.2f} |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        report(sys.argv[2:])
        return
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    sys.meta_path.insert(0, Finder())
    atexit.register(write_out)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        write_out()


if __name__ == "__main__":
    main()
