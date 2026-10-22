#!/usr/bin/env python3
"""seedvr2x's first pass on one decoder thread against ffmpeg's default threads (CPU only).

The first pass's ffmpeg command is seedvr2x's own (`seedvr2x.media.scan.command`, the shot detector's
48x27 output on fd 3, drained here), run as is or with `-threads 1` as an input option, the decoder's
(placed just before `-noautorotate -i SRC`, where seedvr2x's input options begin); and a decode alone
(`-map 0:v:0 -f null -`) both ways. Per run: wall time, user and system CPU (the child's rusage), the
thread count 2 s in, the load average at start and end, and the CRC-32 of every frame (the pass's own
framehash). `compare` tabulates the runs and the frames whose CRC-32 differs between them. `seek` reads
given frames the way seedvr2x's reader seeks (`-copyts -noaccurate_seek -ss` at a keyframe's time after
the file's start, a `select=gte(pts, target)` before the hashed output), on one decoder thread, from the
frame's own keyframe (the last whose pts is at most the frame's, as FrameIndex.own_keyframe) and from 1
to N keyframes before it, against the CRC-32 a run's pass gave the frame.

Run it with seedvr2x's Python (it imports seedvr2x for the command):

  first_pass_threads.py cmd SRC MODE                    # MODE: default | t1 | decode-default | decode-t1
  first_pass_threads.py run OUT LABEL SRC MODE RUN      # OUT/LABEL-MODE-RUN.{cmd,crc,err,json}
  first_pass_threads.py compare OUT LABEL               # Markdown table and the CRC-32 differences
  first_pass_threads.py seek SRC CRC TSV --frames 101466,142475 [--back 3] [--runs 3]
"""
import argparse
import itertools
import json
import os
import resource
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


def build(src, mode):
    """The command for MODE, as a list."""
    if mode.startswith("decode"):
        thr = ["-threads", "1"] if mode.endswith("t1") else []
        return ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", *thr,
                "-noautorotate", "-i", src, "-map", "0:v:0", "-f", "null", "-"]
    from seedvr2x.media.scan import command

    cmd = command(Path(src), 3)
    if mode == "t1":
        i = cmd.index("-noautorotate")  # input_args begin here: -threads 1 before -i is the decoder's
        cmd[i:i] = ["-threads", "1"]
    elif mode != "default":
        raise SystemExit(f"unknown mode {mode}")
    return cmd


def loadavg():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def run(out, label, src, mode, n):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"{label}-{mode}-{n}"
    cmd = build(src, mode)
    base.with_suffix(".cmd").write_text(shlex.join(cmd) + "\n")
    meta = {"label": label, "mode": mode, "run": n, "load_start": loadavg()}
    detector = not mode.startswith("decode")
    r, w = os.pipe() if detector else (None, None)

    def child():  # the detector's frames on fd 3, as seedvr2x gives them
        if detector:
            os.dup2(w, 3)

    def drain():
        while os.read(r, 1 << 20):
            pass

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    with open(base.with_suffix(".crc"), "wb") as crc, open(base.with_suffix(".err"), "wb") as err:
        p = subprocess.Popen(cmd, stdout=crc if detector else subprocess.DEVNULL, stderr=err,
                             stdin=subprocess.DEVNULL, preexec_fn=child, close_fds=False)  # fd 3 kept
        if detector:
            os.close(w)
            th = threading.Thread(target=drain, daemon=True)
            th.start()
        try:  # the thread count 2 s in, without holding a shorter run
            rc = p.wait(timeout=2)
            meta["threads_at_2s"] = None
        except subprocess.TimeoutExpired:
            try:
                meta["threads_at_2s"] = len(os.listdir(f"/proc/{p.pid}/task"))
            except OSError:
                meta["threads_at_2s"] = None
            rc = p.wait()
        if detector:
            th.join()
            os.close(r)
    meta["wall_s"] = time.monotonic() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta["user_s"] = after.ru_utime - before.ru_utime
    meta["sys_s"] = after.ru_stime - before.ru_stime
    meta["rc"] = rc
    meta["load_end"] = loadavg()
    if detector:
        meta["frames"] = len(crcs(base.with_suffix(".crc")))
    else:
        base.with_suffix(".crc").unlink()
    base.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def crcs(path):
    """[(pts, crc)] of a framehash file, frame n on row n."""
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = [x.strip() for x in line.split(",")]
        rows.append((int(f[2]), f[5]))
    return rows


def compare(out, label):
    out = Path(out)
    print("| run | wall s | frames | fps | user s | sys s | CPU/wall | threads at 2 s | load at start | load at end |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    lists = {}
    for mode in ("default", "t1", "decode-default", "decode-t1"):
        for js in sorted(out.glob(f"{label}-{mode}-*.json")):
            m = json.loads(js.read_text())
            n = m.get("frames")
            fps = f"{n / m['wall_s']:.1f}" if n else "-"
            cpu = m["user_s"] + m["sys_s"]
            print(f"| {mode} #{m['run']} | {m['wall_s']:.3f} | {n or '-'} | {fps} | {m['user_s']:.2f} | "
                  f"{m['sys_s']:.2f} | {cpu / m['wall_s']:.2f} | {m['threads_at_2s']} | {m['load_start']} | {m['load_end']} |")
            if n:
                lists[f"{mode} #{m['run']}"] = crcs(js.with_suffix(".crc"))
    print()
    for a, b in itertools.combinations(sorted(lists), 2):
        A, B = lists[a], lists[b]
        if len(A) != len(B):
            print(f"- {a} vs {b}: {len(A)} against {len(B)} frames")
            continue
        d = [i for i in range(len(A)) if A[i] != B[i]]
        print(f"- {a} vs {b}: {len(d)} frames differ" + "".join(
            f"\n  - frame {i}, pts {A[i][0]}: {A[i][1]} against {B[i][1]}" for i in d[:40]))


def probe(src, entries, select="v:0"):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", select, *entries, "-of", "csv=p=0", src],
                          capture_output=True, text=True, check=True).stdout


def seek(src, crcfile, tsv, frames, back, runs):
    rows = crcs(crcfile)
    tb = probe(src, ["-show_entries", "stream=time_base"]).strip()  # e.g. 1/1000
    num, den = map(int, tb.split("/"))
    start = probe(src, ["-show_entries", "format=start_time"], select="v:0").strip()
    start_us = round(float(start) * 1_000_000) if start not in ("", "N/A") else 0
    kf = sorted({int(p.split(",")[0]) for p in probe(src, ["-show_entries", "packet=pts,flags"]).splitlines()
                 if "," in p and "K" in p.split(",")[1] and p.split(",")[0] not in ("", "N/A")})
    out = ["target\tback\tkeyframe_pts\tfirst_frame_decoded\tseek\trun\tgot_pts\tcrc\tstart_crc\trc"]
    for t in frames:
        tpts, tcrc = rows[t]
        own = max(i for i, k in enumerate(kf) if k <= tpts)
        for b in range(back + 1):
            k = kf[own - b]
            us = max(0, k * num * 1_000_000 // den - start_us)
            ss = f"{us // 1_000_000}.{us % 1_000_000:06d}"
            first = next(i for i, (p, _) in enumerate(rows) if p >= k)
            cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error", "-copyts",
                   "-noaccurate_seek", "-ss", ss, "-threads", "1", "-noautorotate", "-i", src,
                   "-map", "0:v:0", "-vf", f"select=gte(pts\\,{tpts})", "-fps_mode", "passthrough",
                   "-frames:v", "1", "-enc_time_base:v", "demux", "-threads", "1", "-map_metadata", "-1",
                   "-map_chapters", "-1", "-c:v", "rawvideo", "-f", "framehash", "-hash", "crc32", "-"]
            for n in range(1, runs + 1):
                r = subprocess.run(cmd, capture_output=True, text=True)
                data = [x for x in r.stdout.splitlines() if x and not x.startswith("#")]
                f = [x.strip() for x in data[0].split(",")] if data else [""] * 5 + ["none"]
                out.append(f"{t}\t{b}\t{k}\t{first}\t{ss}\t{n}\t{f[2]}\t{f[5]}\t{tcrc}\t{r.returncode}")
                print(out[-1], flush=True)
    Path(tsv).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cmd"); c.add_argument("src"); c.add_argument("mode")
    r = sub.add_parser("run")
    for a in ("out", "label", "src", "mode"):
        r.add_argument(a)
    r.add_argument("run", type=int)
    m = sub.add_parser("compare"); m.add_argument("out"); m.add_argument("label")
    s = sub.add_parser("seek"); s.add_argument("src"); s.add_argument("crc"); s.add_argument("tsv")
    s.add_argument("--frames", required=True); s.add_argument("--back", type=int, default=3)
    s.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "cmd":
        print(shlex.join(build(a.src, a.mode)))
    elif a.cmd == "run":
        run(a.out, a.label, a.src, a.mode, a.run)
    elif a.cmd == "compare":
        compare(a.out, a.label)
    else:
        seek(a.src, a.crc, a.tsv, [int(x) for x in a.frames.split(",")], a.back, a.runs)


if __name__ == "__main__":
    sys.exit(main())
