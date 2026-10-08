#!/usr/bin/env python3
"""Model conversation, slice A: crops for the eyes, every file beside the 7B fp16, at colour's 1080p d1 windows
(crops-b2-1080p's d1 folders: anime-bright, anime-clean, cartoon-bright, live-slow, live-vfx; 2 windows each,
384x216 at frame 22 or 44, placed by hand in colour's baton 1), cut by colour's colour_crops.py as it is.

  ms_crops.py [--labels fp8a16,fp8a8,q4k,q80,int8,nv4a4,nq4km] [--out DIR] [--name crops-a] [--threads 8]

One strip per window: GT | bicubic | 7B fp16 | each label's decode, all through split:ycc:4:3 with the 7B s42
run's reference, 1:1, no stretched twins (--no-stretch). A strip is made only when every decode of its clip
exists (the 7B's and each label's that runs on that clip: nq4km only on anime-sky, anime-bright, live-vfx,
live-slow; a label's decode = the GPU runner's done marker a-<clip>-d1-<label> and its decode.pt); strips already
made with the same labels are skipped, so it can be re-run as runs land. Writes DIR/<name>/<clip>/<window>.png,
DIR/<name>/<clip>/crops.tsv (each panel's mean CIEDE2000 to the GT over the window: colour_crops.py's), and
DIR/<name>/README.txt; checks the 7B panel's dE against colour's own strip of that window (same decode, same
correction: must be equal). Label valsharp = colour's sharp 7B sh42 decodes (the test).
S9 (2026-10-07): the sharp 7B's files (labels sh-*) beside the sharp 7B fp16 (colour's sh42 decodes, panel "sharp
fp16", checked against colour's sharp panel of the same window): GT | bicubic | sharp fp16 | each sh-* label, e.g.
  ms_crops.py --labels sh-fp8a16,sh-fp8a8,sh-int8,sh-q80,sh-q4k,sh-q4ki,sh-dyn,sh-nv4a16,sh-nv4a4 --name crops-a-sharp
(a mixed list shows both references: 7B fp16, then sharp fp16).
S17 (2026-10-08): --refs sets the reference panels by hand ('7B fp16', 'sharp fp16'; default: as above); a label
in CAPTION is captioned in words: 3b-cur "3B current" (the SeedVR2 3B from ByteDance's current 3B weights, our
float16), 3b-first "3B first" (numz's 3B fp16: the first 3B weights). The 3B beside the sharp 7B and its 4 GB file:
  ms_crops.py --refs "sharp fp16" --labels sh-dyn,3b-cur,3b-first --name crops-3b
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

O = env("COLOUR_OUT")
TOOL = env("COLOUR_SCRIPTS") + "/colour_crops.py"
WIN = f"{O}/crops-b2-1080p"
CLIPS = env("MEAS_CLIPS")
G = env("VAL_DATA") + "/gpu"
GPUDONE = env("VAL_STATE") + "/gpu/done"
PY = env("METRICS_PY")
VARIANT = "split:ycc:4:3"
FOLDERS = ("anime-bright", "anime-clean", "cartoon-bright", "live-slow", "live-vfx")
NQ4KM = ("anime-sky", "anime-bright", "live-vfx", "live-slow")
REFS = ("7B fp16", "sharp fp16")  # the reference panels: colour's 7B / sharp 7B s42 decodes
CAPTION = {
    "3b-cur": "3B current",
    "3b-first": "3B first",
}  # S17: panels captioned in words (else the label)
PARIS = timezone(timedelta(hours=2))
ENV = dict(
    os.environ,
    COLOUR_BASELINE=env("COLOUR_BASELINE"),
    MEAS_SCRIPTS=env("MEAS_SCRIPTS"),
    PATH=env("FFMPEG_BIN") + ":" + os.environ.get("PATH", ""),
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="8",
    MKL_NUM_THREADS="8",
    TMPDIR=f"{G}/tmp",
)
WHAT = {  # what each window shows (colour's README for crops-b2-1080p)
}


def say(*lines):
    for line in lines:
        print(f"{datetime.now(PARIS).strftime('%H:%M:%S')} {line}", flush=True)


def read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def cap(label):
    """A panel's caption (colour_crops.py's LABEL): CAPTION's words, else the label itself."""
    return CAPTION.get(label, label)


def decode(label, cd):
    """(path, ready?) of a label's decode of clip cd."""
    if label == "7B fp16":
        p = f"{O}/dumps/{cd}/s42/decode.pt"
        return p, os.path.exists(p)
    if label in (
        "valsharp",
        "sharp fp16",
    ):  # S9: "sharp fp16" = the sharp's reference panel, the same decodes
        p = f"{O}/dumps/{cd}/sh42/decode.pt"
        return p, os.path.exists(p)
    p = f"{G}/dumps/{cd}/{label}-s42/decode.pt"
    return p, os.path.exists(p) and os.path.exists(f"{GPUDONE}/a-{cd}-{label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="fp8a16,fp8a8,q4k,q80,int8,nv4a4,nq4km")
    ap.add_argument("--out", default=f"{G}/crops")
    ap.add_argument("--name", default="crops-a")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--refs",
        help="the reference panels, comma-separated: '7B fp16' and/or 'sharp fp16' (default: "
        "the 7B fp16 for the 7B's files, the sharp 7B fp16 for the sharp's sh-*)",
    )
    a = ap.parse_args()
    labels = [x for x in a.labels.split(",") if x]
    # the reference panels (S9): the 7B fp16 for the 7B's files, the sharp 7B fp16 for the sharp's (sh-*)
    refs = (["7B fp16"] if any(not L.startswith("sh-") for L in labels) or not labels else []) + (
        ["sharp fp16"] if any(L.startswith("sh-") for L in labels) else []
    )
    if a.refs is not None:  # S17: the reference panels by hand
        refs = [r for r in a.refs.split(",") if r]
        if not refs or len(set(refs)) < len(refs) or any(r not in REFS for r in refs):
            raise SystemExit(
                f"--refs {a.refs!r}: one or both of {', '.join(REFS)}, comma-separated"
            )
    sharp_only = refs == ["sharp fp16"]
    top = os.path.join(a.out, a.name)
    os.makedirs(top, exist_ok=True)
    made, waiting, failed, checks = [], [], [], []
    for folder in FOLDERS:
        cd = f"{folder}-d1"
        rows = read_tsv(f"{WIN}/{folder}/crops.tsv")
        head, wins = rows[0], [r for r in rows[1:] if r and r[0] != "name"]
        cols = {
            r: head.index(f"{r.split(' fp16')[0]} {VARIANT}") for r in refs
        }  # colour's 7B / sharp panel's dE
        mine = refs + [L for L in labels if L != "nq4km" or folder in NQ4KM]
        decs = {L: decode(L, cd) for L in mine}
        missing = [L for L, (_, ok) in decs.items() if not ok]
        dst = os.path.join(top, folder)
        os.makedirs(dst, exist_ok=True)
        tsv = os.path.join(dst, "crops.tsv")
        have = {}
        if os.path.exists(tsv):
            t = read_tsv(tsv)
            for r in t[1:]:
                have[r[0]] = (t[0], r)
        sig = ["name", "kind", "frame", "x", "y", "score", "bicubic"] + [
            f"{cap(L)} {VARIANT}" for L in mine
        ]
        todo = [
            w
            for w in wins
            if not (
                w[0] in have
                and have[w[0]][0] == sig
                and os.path.exists(os.path.join(dst, w[0] + ".png"))
            )
        ]
        if not todo:
            say(f"{folder}: {len(wins)} strip(s) already made")
            continue
        if missing:
            waiting.append(f"{folder} (waits for {', '.join(missing)})")
            say(f"{folder}: waits for {', '.join(missing)}")
            continue
        by_frame = {}
        for w in todo:
            by_frame.setdefault(w[2], []).append(w)
        for frame, ws in by_frame.items():
            stage = os.path.join(a.out, "stage", a.name, f"{folder}-f{frame}")
            shutil.rmtree(stage, ignore_errors=True)
            os.makedirs(stage)
            cmd = [
                "/usr/bin/time",
                "-f",
                "%e s, max RSS %M KB",
                "taskset",
                "-c",
                "0-31",
                "nice",
                "-n",
                "19",
                PY,
                TOOL,
                "--clip",
                folder,
                "--gt",
                f"{CLIPS}/{folder}.gt.mkv",
                "--bicubic",
                f"{CLIPS}/{folder}.d1.bicubic.mkv",
                "--ref",
                f"{O}/dumps/{cd}/s42/ref_f32.pt",
            ]
            for L in mine:
                cmd += ["--content", f"{cap(L)}={decs[L][0]}"]
            cmd += [
                "--variants",
                VARIANT,
                "--no-stretch",
                "--per-kind",
                "0",
                "--at-frame",
                frame,
                "--size",
                "384x216",
                "--threads",
                str(a.threads),
                "--out",
                stage,
            ]
            for w in ws:
                cmd += ["--at", f"{w[3]},{w[4]}"]
            say(f"{folder} f{frame}: {len(ws)} window(s), {len(mine)} decodes")
            p = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV
            )
            for line in p.stdout.splitlines()[-4:]:
                say(f"  {line}")
            if p.returncode:
                failed.append(f"{folder} f{frame} (exit {p.returncode})")
                continue
            out = read_tsv(os.path.join(stage, "crops.tsv"))
            if out[0] != sig:
                failed.append(f"{folder} f{frame}: header {out[0]}")
                continue
            for w in ws:
                r = next((r for r in out[1:] if r[0] == w[0]), None)
                if r is None:
                    failed.append(f"{w[0]}: not cut")
                    continue
                shutil.move(os.path.join(stage, w[0] + ".png"), os.path.join(dst, w[0] + ".png"))
                have[w[0]] = (sig, r)
                made.append(w[0])
                for ref, col in cols.items():
                    d7, twin = float(r[sig.index(f"{ref} {VARIANT}")]), float(w[col])
                    checks.append(
                        f"{w[0]}: {ref.split(' fp16')[0]} dE {d7:.3f}, colour's strip {twin:.3f}: "
                        + ("same" if abs(d7 - twin) < 0.0015 else "DIFFERENT")
                    )
            shutil.rmtree(stage, ignore_errors=True)
        with open(tsv + ".tmp", "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t", lineterminator="\n")
            wr.writerow(sig)
            for w in wins:
                if w[0] in have and have[w[0]][0] == sig:
                    wr.writerow(have[w[0]][1])
        os.replace(tsv + ".tmp", tsv)
    # README
    three = [L for L in labels if L in CAPTION]  # S17: the 3B's panels
    rest = [L for L in labels if L not in CAPTION]
    who = (
        "the sharp 7B's phase-2 model files beside the sharp 7B fp16"
        if sharp_only
        else "the phase-2 model files beside the 7B fp16"
    )
    if three:
        who = (
            f"the SeedVR2 3B ({', '.join(cap(L) for L in three)}) beside "
            + " and ".join(
                {"7B fp16": "the 7B fp16", "sharp fp16": "the sharp 7B fp16"}[r] for r in refs
            )
            + (f" and {', '.join(rest)}" if rest else "")
        )
    lines = [
        f"{a.name}: {who} at 1080p, for the eyes",
        f"Model conversation, slice A (night of 2026-10-07). Updated {datetime.now(PARIS).strftime('%Y-%m-%d %H:%M')} Paris.",
        "",
        "PANELS, left to right, each 384x216 at 1:1 (no scaling), under a label strip:",
        "  GT | bicubic | " + " | ".join(f"{cap(L)} split:ycc:4:3" for L in refs + labels),
    ]
    if "7B fp16" in refs:
        lines += [
            "- 7B fp16 = SeedVR2 7B (seedvr2_ema_7b_fp16, numz's file), seed 42: colour's dumps of steps 1-3."
        ]
    if "sharp fp16" in refs:
        lines += [
            "- sharp fp16 = SeedVR2 7B sharp (seedvr2_ema_7b_sharp_fp16, numz's file), seed 42: colour's baton-2",
            "  decodes (crops-b2-1080p's sharp panel). sh-<x> = the sharp 7B from our phase-2 file <x> below.",
        ]
    if "3b-cur" in three:  # S17
        lines += [
            "- 3B current = SeedVR2 3B from ByteDance's current 3B weights (SeedVR2-3B at 37255ff, seedvr2_ema_3b.pth)",
            "  in float16: our seedvr2x_ema_3b_fp16 (seedvr2_fp16.py's rounding; research, not uploaded), seed 42, the",
            "  same input and flags.",
        ]
    if "3b-first" in three:
        lines += [
            "- 3B first = SeedVR2 3B from the first 3B weights: seedvr2_ema_3b_fp16, numz's file, seed 42, the same",
            "  input and flags.",
        ]
    other = "the same model" if "sharp fp16" in refs else "the same 7B"
    if three:  # S17: "the same model" would read as the 3B
        other = (
            "the sharp 7B"
            if refs == ["sharp fp16"]
            else "the 7B"
            if refs == ["7B fp16"]
            else "the 7B or the sharp 7B"
        )
    lines += [
        f"- every other panel = {other} from one of our phase-2 files, seed 42, the same input and flags:",
        "  fp8a16 / fp8a8 = fp8 e4m3 with one scale per tensor, weights only / weights and activations; q4k, q80 =",
        "  GGUF Q4_K / Q8_0 (ggml's quantizer); int8 = INT8 with comfy-kitchen's rotation; nv4a4 = NVFP4 weights and",
        "  activations; nq4km = numz's own Q4_K_M (only on anime-sky, anime-bright, live-vfx, live-slow);",
        "  nv4a16 = NVFP4 weights only (activations 16-bit); dyn = dynamic GGUF (a ggml type per matrix, Q3_K to",
        "  Q8_0, chosen with an importance matrix within Q4_K's bytes); q4ki = Q4_K made with that importance",
        "  (dyn's control);",
        "  valsharp = colour's sharp 7B decodes (the pipeline's test).",
        "- x2 from 960x540 to 1920x1080, from d1 (x1/2 Mitchell, x264 CRF 20). split:ycc:4:3 = seedvr2x's default",
        "  colour correction, applied to each decode with the 7B run's reference. Bicubic: the input upscaled x2.",
        '- "dE x" in a label: the panel\'s mean CIEDE2000 colour difference to the GT over the window (also in',
        "  <folder>\\crops.tsv). No luma-stretched .stretch.png twins. View at 100%.",
        "WINDOWS: crops-b2-1080p's 10 d1 windows (colour's, placed by hand in baton 1), names unchanged:",
        "  <clip>-at-f<frame>-x<x>-y<y> (top-left corner x, y; frames 0-based).",
        "",
        f"STRIPS: {sum(1 for f in FOLDERS for n in os.listdir(os.path.join(top, f)) if n.endswith('.png')) if all(os.path.isdir(os.path.join(top, f)) for f in FOLDERS) else '?'} of 10 made",
    ]
    for f in FOLDERS:
        d = os.path.join(top, f)
        pngs = sorted(n for n in os.listdir(d) if n.endswith(".png")) if os.path.isdir(d) else []
        lines.append(f"- {f}: " + (", ".join(pngs) if pngs else "none yet"))
    if waiting:
        lines += ["", "WAITING: " + "; ".join(waiting)]
    with open(os.path.join(top, "README.txt") + ".tmp", "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(os.path.join(top, "README.txt") + ".tmp", os.path.join(top, "README.txt"))
    say(*checks)
    say(
        f"made {len(made)}, waiting {len(waiting)}, failed {len(failed)}"
        + (": " + "; ".join(failed) if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
