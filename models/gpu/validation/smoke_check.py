#!/usr/bin/env python3
"""The GPU smoke's checks on the CPU (model conversation, 2026-10-07; metrics venv, CUDA hidden).

  python smoke_check.py report      # every smoke run's ck_patch.json: backends, ops, per-layer check figures;
                                    # each decode against the 7B fp16 s42 decode: PSNR, non-finite values;
                                    # int8-swap against int8, bit for bit
  python smoke_check.py equal DIR_A DIR_B   # two dumps' decode.pt and latents.pt, bit for bit

PSNR: both decodes as numz's Phase 4 brings them to [0, 1] with --color_correction none (clamp to [-1, 1],
x 0.5 + 0.5), in float32, peak 1, the squared error pooled over the 45 frames (and the worst frame's).
"""

import json
import math
import os
import sys

import torch
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

torch.set_num_threads(8)
RUNS = env("VAL_STATE") + "/smoke/runs"
DUMPS = env("VAL_DATA") + "/dumps/smoke/anime-clean-d1"
REF = env("COLOUR_OUT") + "/dumps/anime-clean-d1"
TAGS = ["fp8-w8a16", "fp8-w8a8", "int8", "nvfp4-w4a4", "nvfp4-w4a16", "q4k", "int8-swap"]
# mode -> (comfy-kitchen functions that must run on its cuda backend, dequantizations wanted (True) or none (False))
EXPECT = {
    "w8a16": (["dequantize_per_tensor_fp8"], True),
    "w8a8": (["quantize_per_tensor_fp8"], False),
    "int8": (["int8_linear"], False),
    "w4a4": (["quantize_nvfp4", "scaled_mm_nvfp4"], False),
    "w4a16": (["dequantize_nvfp4"], True),
}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def equal(da, db):
    out = {}
    for f in ("decode.pt", "latents.pt"):
        a, b = load(os.path.join(da, f)), load(os.path.join(db, f))
        out[f] = same(a, b)
        if f == "decode.pt" and not out[f]:
            x, y = a["final_video"].float(), b["final_video"].float()
            if x.shape == y.shape:
                d = (x - y).abs()
                out["decode_diff"] = {
                    "values_different": int((d > 0).sum()),
                    "of": d.numel(),
                    "max_abs": float(d.max()),
                    "psnr_db": psnr(a["final_video"], b["final_video"])[0],
                }
    return out


def to01(x):
    return x.float().clamp(-1, 1) * 0.5 + 0.5


def psnr(a, b):
    """Pooled PSNR (dB) of a against b over all frames, the worst frame's, a's non-finite count."""
    assert a.shape == b.shape, (a.shape, b.shape)
    mses = [
        float(((to01(a[t]) - to01(b[t])) ** 2).mean(dtype=torch.float64)) for t in range(a.shape[0])
    ]
    m = sum(mses) / len(mses)
    db = lambda e: math.inf if e == 0 else 10 * math.log10(1 / e)  # noqa: E731
    return db(m), db(max(mses)), int((~torch.isfinite(a)).sum())


def report():
    ref = load(os.path.join(REF, "s42", "decode.pt"))["final_video"]
    print(
        f"reference: {REF}/s42/decode.pt {tuple(ref.shape)} {ref.dtype}, non-finite {int((~torch.isfinite(ref)).sum())}"
    )
    for s in ("s43", "s1234"):
        p = os.path.join(REF, s, "decode.pt")
        if os.path.isfile(p):
            db, worst, nf = psnr(load(p)["final_video"], ref)
            print(
                f"yardstick: 7B fp16 {s} vs s42: PSNR {db:.2f} dB (worst frame {worst:.2f}), non-finite {nf}"
            )
    summary = {}
    for tag in TAGS:
        print(f"\n===== {tag}")
        jp = os.path.join(RUNS, f"model-smoke-{tag}.ck_patch.json")
        r = json.load(open(jp)) if os.path.isfile(jp) else None
        s = summary[tag] = {}
        if r is None:
            print("  no ck_patch.json")
        else:
            ck = r.get("comfy_kitchen") or {}
            ld = r.get("load") or {}
            print(
                f"  dit {r.get('dit', {}).get('realpath')} format {r.get('dit', {}).get('format')} mode {r.get('mode')}"
                f" aborted {r.get('aborted')}"
            )
            print(f"  loaded DiT {r.get('loaded', {}).get('DiT')}")
            if ck:
                print(
                    f"  comfy-kitchen {ck.get('version')} available {ck.get('available')} cuda ext {ck.get('cuda_extension_loaded')}"
                    f" cublaslt {ck.get('cublaslt')} unavailable {ck.get('unavailable')}"
                )
            if ld:
                print(f"  load {ld}")
            if "fp16_compare" in r:
                print(
                    f"  fp16_compare tensors {r['fp16_compare']['tensors']} different {r['fp16_compare']['n_different']}"
                )
            print(
                f"  compute dtypes {r.get('compute_dtypes')} refused casts {r.get('refused_dtype_casts')}"
            )
            c = r.get("check")
            if c:
                print(
                    f"  check: layers {c['layers']} proof_ok {c['proof_ok']} finite {c['finite']}"
                )
                mode = r.get("mode")
                fns, deq = EXPECT[mode]
                ok_paths = 0
                for p in c.get("paths", []):
                    cuda_ok = all(f"cuda:{f}" in p["ck"] for f in fns)
                    others = [x for x in p["ck"] if not x.startswith("cuda:")]
                    deq_ok = (p["dequantize"] >= 1) if deq else (p["dequantize"] == 0)
                    good = cuda_ok and not others and deq_ok
                    ok_paths += p["layers"] if good else 0
                    print(
                        f"  path x{p['layers']}: ck {p['ck']} torch {p['torch']} dequantize {p['dequantize']}"
                        f" aten {p['aten']} -> {'OK' if good else 'NOT AS EXPECTED'}"
                    )
                print(
                    f"  layers on the expected path ({', '.join('cuda:' + f for f in fns)}, dequantize "
                    f"{'>= 1' if deq else '0'}): {ok_paths} of {c['layers']}"
                )
                s["paths_ok"] = ok_paths
                pk = c["per_kind"]
                print(
                    f"  {'kind':28s} {'n':>3s}  "
                    + "  ".join(
                        f"{m + ' med/worst %':>22s}"
                        for m in ("kernel", "total", "weights", "floor")
                    )
                )
                for k in sorted(pk, key=lambda k: (k == "all", k)):
                    v = pk[k]
                    print(
                        f"  {k:28s} {v['layers']:3d}  "
                        + "  ".join(
                            f"{100 * v[m]['median']:10.4f}/{100 * v[m]['worst']:<10.4f}"
                            for m in ("kernel", "total", "weights", "floor")
                        )
                    )
                a = pk["all"]
                s["check_all"] = {
                    m: (a[m]["median"], a[m]["worst"], a[m]["worst_layer"])
                    for m in ("kernel", "total", "weights", "floor")
                }
        dp = os.path.join(DUMPS, tag, "decode.pt")
        if os.path.isfile(dp):
            db, worst, nf = psnr(load(dp)["final_video"], ref)
            s["psnr"] = (db, worst, nf)
            print(
                f"  decode vs 7B fp16 s42: PSNR {db:.2f} dB (worst frame {worst:.2f}), non-finite {nf}"
            )
        else:
            print(f"  no decode {dp}")
    if all(os.path.isfile(os.path.join(DUMPS, t, "decode.pt")) for t in ("int8", "int8-swap")):
        e = equal(os.path.join(DUMPS, "int8"), os.path.join(DUMPS, "int8-swap"))
        print(f"\nint8-swap vs int8: {e}")
    print("\nSUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    if sys.argv[1:2] == ["equal"]:
        print(json.dumps(equal(sys.argv[2], sys.argv[3])))
    else:
        report()
