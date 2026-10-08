#!/usr/bin/env python3
"""S13: count the cells past the strict / calibrated line in a 4K summary's per-shot table (ms_sum4k.py /
ms_sum4ksh.py pages): guard columns only (CAMBI+, T-err, T-err lf, DeltaE00 lf, Detail, Finest band, LPIPS, DISTS:
45f where present, else 5f); a strict cell holds W, an up or down arrow; a calibrated one is bold.
  count4k.py PAGE.md [...]   (--dists both: count DISTS 5f and 45f each)"""

import re
import sys

GUARDS = ["CAMBI+", "T-err", "T-err lf", "ΔE00 lf", "Detail", "Finest band", "LPIPS"]
both = "--dists" in sys.argv and sys.argv[sys.argv.index("--dists") + 1] == "both"
pages = [a for a in sys.argv[1:] if a.endswith(".md")]


def strict(c):
    return bool(re.search(r"(^|\s|\*)W(\s|\*|$)|[↑↓]", c))


for p in pages:
    lines = open(p, encoding="utf-8").read().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith("## Per shot"))
    except StopIteration:
        print(f"{p}: no per-shot table")
        continue
    head = None
    rows = []
    for l in lines[i + 1 :]:
        if l.startswith("| Shot"):
            head = [c.strip() for c in l.strip("|").split("|")]
            continue
        if head and l.startswith("|---"):
            continue
        if head and l.startswith("|"):
            rows.append([c.strip() for c in l.strip("|").split("|")])
        elif head and rows and not l.startswith("|"):
            break
    col = {}
    for n, h in enumerate(head):
        for g in GUARDS:
            if (
                h == g
                or h.startswith(g + " ")
                or (g == "Finest band" and h.startswith("Finest band"))
            ):
                col.setdefault(g, n)
        if h.startswith("Detail"):
            col["Detail"] = n
    d5 = head.index("DISTS 5f") if "DISTS 5f" in head else None
    d45 = head.index("DISTS 45f") if "DISTS 45f" in head else None
    ns = nc = 0
    cells = []
    shots = set()
    for r in rows:
        shots.add(r[0])
        use = dict(col)
        if both:
            use["DISTS 5f"] = d5
            if d45 is not None:
                use["DISTS 45f"] = d45
        else:
            use["DISTS"] = d45 if (d45 is not None and r[d45]) else d5
        for g, n in use.items():
            if n is None or n >= len(r):
                continue
            c = r[n]
            if strict(c):
                ns += 1
                b = "**" in c
                nc += b
                cells.append(f"{r[0]} {r[1]} {g}: {c}{' [CAL]' if b else ''}")
    print(f"== {p.split('/')[-1]}: shots {len(shots)}, strict {ns}, calibrated {nc}")
    for c in cells:
        print("   " + c)
