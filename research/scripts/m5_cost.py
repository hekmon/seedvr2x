#!/usr/bin/env python3
"""lab's cost from milestone 5's runs (m5_run.sh): per run, each unit's time and VRAM peak as the
seedvr2x log gives them, the wall time and peak RSS (/usr/bin/time -v), and the peak size of each
file of the lab work directory (sampled every 0.5 s).

  m5_cost.py LOG_DIR NAME...    e.g. m5_cost.py M5_OUT/logs m1-lab1 m1-none1 m1-lab2 m1-none2

LOG_DIR: m5_run.sh's logs/ (ours-NAME.log and ours-NAME.sizes); NAME: a run of its cost holds,
<m1|b>-<lab|none><1|2>.

Historical: the record of how milestone 5 measured lab's cost (2026-10-03/04), kept to re-run it:
m5_run.sh's cost holds, lab, none, lab, none back to back in one hold of the GPU lock on
milestone 1's input and on clip B, seedvr2x at 27ce6ba, whose log lines it parses ("encoded in",
"window i/n, latents a to b,", "decoded in", each with its VRAM peak). seedvr2x/DESIGN.md's input
copy figure (about 32 MiB per second of 1080p input, 1.38 MB per frame on clip B) comes from its
file sizes. seedvr2x's `--color-correction lab` went with 68b1529, when split replaced lab. Needs
Python alone, and calls no research script.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

UNIT = re.compile(
    r"(?P<what>encoded|window \d+/\d+, latents \d+ to \d+,|decoded) in (?P<s>[\d.]+) s, "
    r"VRAM peak (?P<gib>[\d.]+) GiB(?:; RAM (?P<ram>[\d.]+) GiB, peak (?P<peak>[\d.]+) GiB)?")


def run(directory, name):
    log = (directory / f"ours-{name}.log").read_text()
    units = []
    for m in UNIT.finditer(log):
        what = m["what"].split(",")[0]
        units.append((what, float(m["s"]), float(m["gib"]), m["ram"], m["peak"]))
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    sizes = defaultdict(int)
    samples = directory / f"ours-{name}.sizes"
    if samples.exists():
        for line in samples.read_text().splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                sizes[parts[2]] = max(sizes[parts[2]], int(parts[1]))
    peak_total = 0
    if samples.exists():
        by_time = defaultdict(int)
        for line in samples.read_text().splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                by_time[parts[0]] += int(parts[1])
        peak_total = max(by_time.values(), default=0)
    return units, wall[1] if wall else None, int(rss[1]) / 2**20 if rss else None, sizes, peak_total


def main():
    if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    directory = Path(sys.argv[1])
    print("| Run | Unit | Time (s) | VRAM peak (GiB) | RAM / peak (GiB) |")
    print("|---|---|---|---|---|")
    files = {}
    for name in sys.argv[2:]:
        units, wall, rss, sizes, peak_total = run(directory, name)
        for what, s, gib, ram, peak in units:
            print(f"| {name} | {what} | {s:.1f} | {gib:.2f} | {ram or ''} / {peak or ''} |")
        print(f"| {name} | wall (time -v) | {wall} | | max RSS {rss:.2f} |" if rss else f"| {name} | wall | {wall} | | |")
        files[name] = (sizes, peak_total)
    print()
    print("| Run | File | Peak size (bytes) |")
    print("|---|---|---|")
    for name, (sizes, peak_total) in files.items():
        for path, size in sorted(sizes.items()):
            print(f"| {name} | {path} | {size:,} |")
        if sizes:
            print(f"| {name} | all at once (peak of the sampled sums) | {peak_total:,} |")


if __name__ == "__main__":
    main()
