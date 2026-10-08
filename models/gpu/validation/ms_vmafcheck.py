#!/usr/bin/env python3
"""Model conversation, slice A scoring (colour's b2_vmafcheck.py's check, unchanged): is a colour_eval.py vmaf JSON
whole? FRAMES finite VMAF and CAMBI-added values, "frames" = FRAMES, and the VMAF v1 model for the ground truth's
size (the 2160p model from 2160 rows or 3840 columns up, else the 1080p one). One line; exit 1 when anything is off.

  ms_vmafcheck.py JSON FRAMES
"""

import json
import math
import sys

path, n = sys.argv[1], int(sys.argv[2])
try:
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
except Exception as e:  # noqa: BLE001
    print(f"{path}: unreadable ({e})")
    sys.exit(1)
pf = r.get("per_frame", {})
info = r.get("vmaf_info", {})
w, h = (info.get("gt_size") or [0, 0])[:2]
want = "vmaf_v1.0.16_1d5h_2160" if (h >= 2160 or w >= 3840) else "vmaf_v1.0.16_3d0h"
errs = []
for k in ("vmaf", "cambi_added"):
    v = pf.get(k) or []
    if len(v) != n or any(x is None or not math.isfinite(x) for x in v):
        errs.append(f"{k}: {len(v)} values for {n} frames, or not all finite")
if r.get("frames") != n:
    errs.append(f"frames {r.get('frames')}, {n} expected")
if info.get("models") != [want]:
    errs.append(f"models {info.get('models')}, [{want}] expected for a {w}x{h} ground truth")
if errs:
    print(f"{path}: " + "; ".join(errs))
    sys.exit(1)
print(
    f"{path}: ok, {n} frames, VMAF {sum(pf['vmaf']) / n:.2f}, CAMBI added {sum(pf['cambi_added']) / n:.4f}, "
    f"{want} ({w}x{h})"
)
