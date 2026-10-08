#!/usr/bin/env python3
"""S13: the dynamic GGUF's type per matrix, both models, grouped by layer kind and block ranges; where they differ."""

import json
import re
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

D = env("VAL_DYN")
FILES = {
    "7B": f"{D}/seedvr2x_ema_7b_dyn.report.json",
    "sharp": f"{D}/seedvr2x_ema_7b_sharp_dyn.report.json",
}
KINDS = [
    "attn.proj_qkv.vid",
    "attn.proj_qkv.txt",
    "attn.proj_out.vid",
    "attn.proj_out.txt",
    "mlp.vid.proj_in",
    "mlp.vid.proj_out",
    "mlp.txt.proj_in",
    "mlp.txt.proj_out",
]


def ranges(bs):
    bs = sorted(bs)
    out, s, p = [], None, None
    for b in bs:
        if s is None:
            s = p = b
        elif b == p + 1:
            p = b
        else:
            out.append((s, p))
            s = p = b
    if s is not None:
        out.append((s, p))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


choices = {}
for m, p in FILES.items():
    r = json.load(open(p))
    ch = r["choice"]
    choices[m] = ch
    tot = {}
    for t in ch.values():
        tot[t] = tot.get(t, 0) + 1
    f = [v for k, v in r["files"].items() if "_dyn" in k][0]
    print(
        f"== {m}: {len(ch)} matrices, types {dict(sorted(tot.items()))}; file {f['bytes']:,} B sha {f['sha256'][:8]}; "
        f"plain err median/worst {f['err']['median'] * 100:.3f}/{f['err']['worst'] * 100:.3f}%, weighted "
        f"{f['werr']['median'] * 100:.3f}/{f['werr']['worst'] * 100:.3f}%; dead {r['summary'].get('dead')}; "
        f"dyn lower werr than static on {r['summary'].get('dyn_lower_werr_than_static_on')}"
    )
    for k in KINDS:
        by = {}
        for name, t in ch.items():
            mm = re.fullmatch(r"blocks\.(\d+)\.(.+)\.weight", name)
            if mm and mm.group(2) == k:
                by.setdefault(t, []).append(int(mm.group(1)))
        n = sum(len(v) for v in by.values())
        print(
            f"  {k:<20} ({n:2d}): "
            + "; ".join(f"{t} {ranges(v)}" for t, v in sorted(by.items(), key=lambda x: -len(x[1])))
        )
    for key in ("summary",):
        s = r[key]
        for kk in ("budget", "q4k_bytes", "last_price"):
            if kk in s:
                print(f"  {kk}: {s[kk]}")
    q = [v for k, v in r["files"].items() if "imatrix" in k][0]
    print(
        f"  Q4_K_imatrix: {q['bytes']:,} B sha {q['sha256'][:8]}; plain {q['err']['median'] * 100:.3f}/{q['err']['worst'] * 100:.3f}%, "
        f"weighted {q['werr']['median'] * 100:.3f}/{q['werr']['worst'] * 100:.3f}%"
    )
    st = r["summary"].get("static") or r["summary"].get("static_Q4_K")
    if st:
        print(f"  static Q4_K: {json.dumps(st)[:300]}")
    print("  summary keys:", list(r["summary"].keys()))
print("== differences 7B vs sharp:")
nd = 0
for name in sorted(choices["7B"], key=lambda n: (int(n.split(".")[1]), n)):
    a, b = choices["7B"][name], choices["sharp"].get(name)
    if a != b:
        nd += 1
        print(f"  {name}: 7B {a}, sharp {b}")
print(f"  {nd} differ")
