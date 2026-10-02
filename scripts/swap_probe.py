#!/usr/bin/env python3
"""BlockSwap probe for SeedVR2: what each swapped block's moves really cost.

BlockSwap (src/optimization/blockswap.py) wraps the forward of every swapped DiT block:
block.to(gpu), forward, block.to(offload device). This probe wraps torch.nn.Module.to, without
touching the checkout, and for every .to() call on a swapped block (blockswap tags them with
`_block_idx`) records, synchronized: the direction, the bytes held by the block's parameters
and buffers, the time, and the gap since the previous move (= the block's compute). It also
notes the largest tensor of a block and whether the CPU copies are pinned.

Output: <$BENCH_LOG without .log>.swap.json (or SWAP_PROBE_OUT) and a summary on stderr.

Usage (cwd = the SeedVR2 checkout, its venv's python):
  python swap_probe.py inference_cli.py <CLI args>
  python3 bench.py run NAME --wrap swap_probe.py -- <CLI args>
"""
import atexit
import json
import os
import runpy
import sys
import time

GIB = 1024 ** 3
calls = []
info = {}
last_end = [None]


def _tensors(m):
    yield from m.parameters()
    yield from m.buffers()


def install():
    # the allocator backend is fixed when torch is imported: apply the CLI's default first
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    import torch
    orig = torch.nn.Module.to

    def to(self, *args, **kwargs):
        idx = getattr(self, "_block_idx", None)
        if idx is None or not torch.cuda.is_available():
            return orig(self, *args, **kwargs)
        ts = list(_tensors(self))
        src = str(ts[0].device) if ts else "?"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(self, *args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts = list(_tensors(self))
        dst = str(ts[0].device) if ts else "?"
        nbytes = sum(t.numel() * t.element_size() for t in ts)
        if idx not in info:
            biggest = max(ts, key=lambda t: t.numel() * t.element_size())
            info[idx] = {"bytes": nbytes, "n_tensors": len(ts),
                         "max_tensor_mib": biggest.numel() * biggest.element_size() / 2**20,
                         "types": sorted({type(t).__name__ for t in ts})}
        if dst == "cpu" and "pinned" not in info[idx]:
            info[idx]["pinned"] = all(t.is_pinned() for t in ts)
        calls.append({"block": idx, "src": src, "dst": dst, "bytes": nbytes, "s": t1 - t0,
                      "gap_s": (t0 - last_end[0]) if last_end[0] else None})
        last_end[0] = t1
        return out

    torch.nn.Module.to = to


def summary():
    out = {}
    for d in ("h2d", "d2h", "same"):
        sel = [c for c in calls if (d == "h2d" and c["src"] == "cpu" and c["dst"] != "cpu")
               or (d == "d2h" and c["src"] != "cpu" and c["dst"] == "cpu")
               or (d == "same" and c["src"] == c["dst"])]
        if sel:
            b = sum(c["bytes"] for c in sel)
            s = sum(c["s"] for c in sel)
            out[d] = {"n": len(sel), "gib": round(b / GIB, 3), "s": round(s, 3),
                      "ms_per_call": round(1000 * s / len(sel), 2), "gb_per_s": round(b / s / 1e9, 2) if s else None}
    gaps = [c["gap_s"] for c in calls if c["dst"] == "cpu" and c["gap_s"] is not None]
    out["compute_between_moves_s"] = round(sum(gaps), 3)
    out["blocks"] = {str(k): v for k, v in sorted(info.items())[:2]}
    out["n_blocks"] = len(info)
    return out


def write_out():
    if getattr(write_out, "done", False):
        return
    write_out.done = True
    path = os.environ.get("SWAP_PROBE_OUT")
    if not path:
        log = os.environ.get("BENCH_LOG")
        path = (log[:-4] if log and log.endswith(".log") else log or "swap_probe") + ".swap.json"
    s = summary()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": s, "calls": calls}, f, separators=(",", ":"))
    print(f"swap_probe: {json.dumps(s)} -> {path}", file=sys.stderr)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    sys.path[0] = os.path.dirname(script)
    install()
    atexit.register(write_out)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        write_out()


if __name__ == "__main__":
    main()
