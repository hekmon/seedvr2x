#!/usr/bin/env python3
"""A command's host memory with every process it starts: torch.compile's compile workers.

Inductor compiles in worker processes (TORCHINDUCTOR_COMPILE_THREADS of them), forked from the
compiling process, so that their summed resident sizes count the pages they share with it many
times over. This runs a command and samples, every 0.5 s from /proc (smaps_rollup), its process
tree's summed RSS and PSS (proportional set size: each shared page split between its sharers),
the largest process's RSS, and the rest's RSS and PSS, and writes the peaks to a JSON file. Linux
only; no GPU of its own (the command may take one).

Usage:
  compile_mem.py OUT.json -- COMMAND [ARGS...]
"""

import json
import os
import subprocess
import sys
import time


def tree(root):
    children = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(entry))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def mem(pid):
    rss = pss = 0
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Rss:"):
                    rss = int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss = int(line.split()[1])
    except OSError:
        pass
    return rss, pss


out = sys.argv[1]
cmd = sys.argv[sys.argv.index("--") + 1:]
proc = subprocess.Popen(cmd)
peak = {"rss_gib": 0.0, "pss_gib": 0.0, "main_rss_gib": 0.0, "others_rss_gib": 0.0,
        "others_pss_gib": 0.0, "processes": 0}
start = time.time()
while proc.poll() is None:
    pids = tree(proc.pid)
    figures = {pid: mem(pid) for pid in pids}
    # The main process: the largest of the tree by RSS (the compiling python).
    main = max(figures, key=lambda pid: figures[pid][0])
    rss = sum(v[0] for v in figures.values()) / 2**20
    pss = sum(v[1] for v in figures.values()) / 2**20
    others_rss = (sum(v[0] for v in figures.values()) - figures[main][0]) / 2**20
    others_pss = (sum(v[1] for v in figures.values()) - figures[main][1]) / 2**20
    for key, value in (("rss_gib", rss), ("pss_gib", pss), ("main_rss_gib", figures[main][0] / 2**20),
                       ("others_rss_gib", others_rss), ("others_pss_gib", others_pss),
                       ("processes", len(pids))):
        peak[key] = max(peak[key], round(value, 3))
    time.sleep(0.5)
peak["seconds"] = round(time.time() - start, 1)
peak["returncode"] = proc.returncode
peak["cmd"] = cmd
with open(out, "w") as f:
    json.dump(peak, f, indent=1)
sys.exit(proc.returncode)
