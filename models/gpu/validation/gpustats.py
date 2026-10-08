#!/usr/bin/env python3
"""S13: the GPU over the model conversation's baton (2026-10-07): clocks under load from the 5 s sampler, run
durations per kind from the runner's gpu-jobs.csv, per-job clock, failures and not-run jobs."""

import csv
import os
import re
import statistics as st
from datetime import datetime, timedelta, timezone
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

G = env("VAL_STATE") + "/gpu"
UTC = timezone.utc
PARIS = timezone(timedelta(hours=2))
REASONS = {
    0x1: "GpuIdle",
    0x2: "ApplicationsClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def num(s):
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else float("nan")


def paris(t):
    return t.astimezone(PARIS).strftime("%H:%M")


rows = []
with open(f"{G}/gpu-samples.csv") as f:
    r = csv.reader(f)
    head = next(r)
    for x in r:
        if len(x) < 7:
            continue
        try:
            t = datetime.strptime(x[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
        except ValueError:
            continue
        rows.append(
            (t, num(x[1]), int(x[2].strip(), 16), num(x[3]), num(x[4]), num(x[5]), num(x[6]))
        )
print(
    f"samples: {len(rows)}, {paris(rows[0][0])} to {paris(rows[-1][0])} Paris "
    f"({rows[0][0]:%H:%M:%S} to {rows[-1][0]:%H:%M:%S} UTC)"
)
gaps = [(a[0], b[0]) for a, b in zip(rows, rows[1:]) if (b[0] - a[0]).total_seconds() > 30]
print(
    f"sampler gaps > 30 s: {len(gaps)}" + "".join(f"; {paris(a)}-{paris(b)}" for a, b in gaps[:10])
)
load = [x for x in rows if x[6] >= 90]
print(
    f"under load (utilization >= 90%): {len(load)} samples = {len(load) * 5 / 3600:.2f} h at 5 s "
    f"({100 * len(load) / len(rows):.1f}% of the samples)"
)
for name, i in (("SM clock MHz", 1), ("power W", 3), ("temperature C", 4), ("memory MiB", 5)):
    xs = [x[i] for x in load]
    print(
        f"  {name}: median {pct(xs, 50):.0f}, p5 {pct(xs, 5):.0f}, p95 {pct(xs, 95):.0f}, min {min(xs):.0f}, "
        f"max {max(xs):.0f}"
    )
cnt = {}
for x in load:
    cnt[x[2]] = cnt.get(x[2], 0) + 1
print("  clock-event reasons under load (value: samples):")
for v, n in sorted(cnt.items(), key=lambda kv: -kv[1]):
    names = "+".join(nm for b, nm in REASONS.items() if v & b) or "none"
    rest = v & ~sum(REASONS)
    print(
        f"    0x{v:x} ({names}{'+0x%x' % rest if rest else ''}): {n} ({100 * n / len(load):.1f}%)"
    )
clk = [x[1] for x in load]
print(
    f"  SM clock bands under load: <=600 MHz {sum(c <= 600 for c in clk)}, 601-1500 {sum(600 < c <= 1500 for c in clk)}, "
    f">1500 {sum(c > 1500 for c in clk)}"
)
hi = [x for x in load if x[1] > 1500]
if hi:
    print(
        f"  samples under load above 1500 MHz: first {paris(hi[0][0])}, last {paris(hi[-1][0])}; reasons "
        + ", ".join(f"0x{v:x}" for v in sorted({x[2] for x in hi}))
    )
idle = [x for x in rows if x[6] == 0]
if idle:
    ci = {}
    for x in idle:
        ci[x[2]] = ci.get(x[2], 0) + 1
    print(
        f"idle (utilization 0%): {len(idle)} samples; SM clock median {pct([x[1] for x in idle], 50):.0f} MHz; power "
        f"median {pct([x[3] for x in idle], 50):.0f} W; reasons "
        + ", ".join(f"0x{v:x}: {n}" for v, n in sorted(ci.items(), key=lambda kv: -kv[1]))
    )

# jobs
CK = {"fp8a16", "fp8a8", "int8", "nv4a4", "nv4a16"}
GG = {"q4k", "q80", "nq4km", "dyn", "q4ki"}
jobs = []
with open(f"{G}/gpu-jobs.csv") as f:
    for x in csv.DictReader(f):
        jobs.append(x)


def kind(i):
    if i.startswith("c-imx"):
        return "1080p fp16 (imatrix hooks)"
    m = re.match(r"([ab])-(.+)-d1-(.+)$", i)
    if not m:
        return "other"
    res = "1080p" if m.group(1) == "a" else "4K"
    lab = m.group(3)
    base = lab[3:] if lab.startswith("sh-") else lab
    if base in CK:
        return f"{res} CK"
    if base in GG:
        return f"{res} GGUF"
    if base.startswith("sharp-s"):
        return f"{res} fp16 (sharp seeds)"
    return f"{res} ?"


def jclock(j):
    t0 = datetime.strptime(j["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    t1 = datetime.strptime(j["end_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    xs = [x for x in load if t0 <= x[0] <= t1]
    return (
        pct([x[1] for x in xs], 50) if xs else float("nan"),
        pct([x[3] for x in xs], 50) if xs else float("nan"),
        len(xs) * 5,
    )


print(
    f"\nruns by the runner: {len(jobs)}; exit statuses: "
    + ", ".join(
        f"{k}: {v}"
        for k, v in sorted(
            {
                j["exit_status"]: sum(1 for y in jobs if y["exit_status"] == j["exit_status"])
                for j in jobs
            }.items()
        )
    )
)
first_nan = {
    "b-digital-cockpit-d1-sh-fp8a8",
    "b-ouatia-face-d1-sh-fp8a8",
    "b-digital-space-d1-sh-fp8a8",
}
seen = set()
by = {}
for j in jobs:
    k = kind(j["id"])
    tag = ""
    if j["id"] in first_nan and j["id"] not in seen:
        tag = " (first run: all-NaN output)"
    seen.add(j["id"])
    by.setdefault(k, []).append((int(j["seconds"]), j, tag))
for k in sorted(by):
    xs = [s for s, _, _ in by[k]]
    print(
        f"  {k}: {len(xs)} runs, mean {st.mean(xs):.0f} s ({st.mean(xs) / 60:.1f} min), range {min(xs)}-{max(xs)} s"
    )
# per label
lab = {}
for j in jobs:
    m = re.match(r"([ab])-(.+)-d1-(.+)$", j["id"])
    key = (m.group(1), m.group(3)) if m else ("c", "imx-sharp" if "sharp" in j["id"] else "imx-7b")
    lab.setdefault(key, []).append(j)
print(
    "\nper label (res, label: n, mean s, range; median SM MHz / power W under load over its runs):"
)
for key in sorted(lab, key=lambda k: (k[0], lab[k][0]["start_utc"])):
    js = lab[key]
    xs = [int(j["seconds"]) for j in js]
    cl = [jclock(j) for j in js]
    print(
        f"  {'1080p' if key[0] in 'ac' else '4K'} {key[1]}: {len(xs)}, {st.mean(xs):.0f} s, {min(xs)}-{max(xs)}; "
        f"{st.median([c[0] for c in cl]):.0f} MHz, {st.median([c[1] for c in cl]):.0f} W; first {paris(datetime.strptime(js[0]['start_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=UTC))}"
    )
# clock per period vs durations (the 1080p slowdown)
print("\n1080p runs, by start hour (Paris): n, mean s, median SM MHz under load")
hb = {}
for j in jobs:
    if kind(j["id"]).startswith("1080p"):
        t0 = (
            datetime.strptime(j["start_utc"], "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=UTC)
            .astimezone(PARIS)
        )
        hb.setdefault(t0.hour, []).append(j)
for h in sorted(hb):
    js = hb[h]
    print(
        f"  {h:02d}h: {len(js)}, {st.mean(int(j['seconds']) for j in js):.0f} s, "
        f"{st.median([jclock(j)[0] for j in js]):.0f} MHz"
    )
# markers
print("\nfailed markers:")
for n in sorted(os.listdir(f"{G}/failed")):
    with open(f"{G}/failed/{n}") as f:
        txt = " | ".join(l.strip() for l in f if l.strip())
    print(f"  {n}: {txt[:160]}")
print(f"done markers: {len(os.listdir(G + '/done'))}")
