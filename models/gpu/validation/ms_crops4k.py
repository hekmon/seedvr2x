#!/usr/bin/env python3
"""Model conversation, slice B: crops for the eyes at 4K, every file beside the 7B fp16, at colour's 4K windows
(crops-b2, placed by hand in colour's baton 1 = "at", or its automatic pick = "diff": sollevante-painted 3,
sollevante-line 3, sollevante-action 2, digital-sunrise 3, digital-space 3) and, for sollevante-dark (no colour
window), 3 placed by hand here on frame 22 (a line, a flat area, a texture); 512x288 each, cut by colour's
colour_crops.py as it is.

  ms_crops4k.py [--labels fp8a16,fp8a8,q4k,q80,int8,nv4a4] [--out DIR] [--name crops-b] [--shots a,b] [--threads 8]
                [--landed]
  test labels (decodes that exist already): sharp = colour's sharp 7B sh42, 7b43 = the 7B fp16 at seed 43

One strip per window: GT | bicubic | 7B fp16 | each label's decode, all through split:ycc:4:3 with the 7B s42 run's
reference, 1:1, no stretched twins (--no-stretch). A strip is made only when every decode of its shot exists (the
7B's, and each label's: the GPU runner's done marker b-<shot>-d1-<label> and its decode.pt); strips already made
with the same labels are skipped, so it can be re-run as runs land. Writes DIR/<name>/<shot>/<window>.png (colour's
names: <shot>-<kind>-f<frame>-x<x>-y<y>), DIR/<name>/<shot>/crops.tsv (each panel's mean CIEDE2000 to the GT over
the window: colour_crops.py's), DIR/<name>/<shot>/README.txt, DIR/<name>/README.txt; checks the 7B panel's dE
against colour's own crops-b2 strip of that window (same decode, same correction: must be equal).
S9 (2026-10-07): the sharp 7B's files (labels sh-*) beside the sharp 7B fp16 (colour's sh42 decodes, panel "sharp
fp16", checked against colour's crops-b2 sharp panel of the same window) on the sharp's 5 two-seed shots
(digital-cockpit, ouatia-face, digital-space, sollevante-painted, cel4k-detail: colour's crops-b2 windows, 3 each),
the default shots when every label is sh-* (test label sh43 = colour's sharp sh43 decode); --landed: each shot's
strips with the labels whose run of that shot has landed (the others left out, in the given order), re-made when
more land (slice B's strips as its runs land):
  ms_crops4k.py --labels sh-int8,sh-dyn,sh-q4k,sh-fp8a8,sh-q80,sh-fp8a16,sh-nv4a4,sh-q4ki --name crops-b-sharp --landed
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
WIN = f"{O}/crops-b2"
C4 = env("MEAS_SHOTS")
G = env("VAL_DATA") + "/gpu"
GPUDONE = env("VAL_STATE") + "/gpu/done"
PY = env("METRICS_PY")
VARIANT = "split:ycc:4:3"
SIZE = "512x288"
SHOTS = (
    "sollevante-painted",
    "sollevante-line",
    "sollevante-action",
    "sollevante-dark",
    "digital-sunrise",
    "digital-space",
)
SHOTS_SH = (
    "digital-cockpit",
    "ouatia-face",
    "digital-space",
    "sollevante-painted",
    "cel4k-detail",
)  # S9
SHOT = {  # the shot, colour's baton-1 verdict (crops-b2's README)
    "sollevante-painted": (
        "Sol Levante (painted anime), 3840x2160",
        "preferred A = 0.25, the least bad; nothing "
        "satisfies against the GT (one verdict for the three Sol Levante shots)",
    ),
    "sollevante-line": (
        "Sol Levante (painted anime, line art), 3840x2160",
        "preferred A = 0.25, the least bad; nothing "
        "satisfies against the GT (one verdict for the three Sol Levante shots)",
    ),
    "sollevante-action": (
        "Sol Levante (painted anime), 3840x2160, fast motion",
        "preferred A = 0.25, the least bad; "
        "nothing satisfies against the GT (one verdict for the three Sol Levante shots)",
    ),
    "sollevante-dark": (
        "Sol Levante (painted anime), 3840x2160, a dark space shot, 41 frames",
        "none: no window in colour's crops (these 3 placed by the model conversation, 2026-10-07)",
    ),
    "digital-sunrise": (
        "digital, 3840x2016",
        "preferred A = 0.5; no GT grain, a clean, coherent gradient",
    ),
    "digital-space": (
        "digital, 3840x2016",
        "preferred A = 0.25; the split and ycc:3:3 destroy the clouds with bright, "
        "over-sharpened artefacts",
    ),
    "digital-cockpit": (
        "digital (live action), 3840x2016",
        "preferred none; bicubic loses the GT's pores, the split "
        "over-emphasises them, the blends attenuate them without reaching the GT's sharpness",
    ),
    "ouatia-face": (
        "the first film's close-up (live action), 3840x2160, letterboxed; the windows inside the picture",
        'preferred A = 0.25; the split and ycc:3:3 make the skin "almost reptilian"',
    ),
    "cel4k-detail": (
        "cel (animation, film scan), 3840x2048",
        "preferred the split; far better: redrawn, crisper, the "
        "style kept; A = 0.25 too close to the washed-out GT",
    ),
}
WHAT = {  # what each window shows (crops-b2's README; sollevante-dark's: ours)
    "sollevante-painted-at-f22-x1340-y1000": "painted standing stone (vertical brush texture), mist, bush",
    "sollevante-painted-at-f22-x2580-y500": "character in profile: hair, face tattoo, hood; painted branches behind",
    "sollevante-painted-diff-f42-x120-y150": "painted foliage (leaves against light), top-left corner",
    "sollevante-line-at-f22-x1700-y560": "eye, lashes, eyeshadow, hair strands, corner of the tattoo",
    "sollevante-line-at-f22-x1780-y1000": "lips with highlights, purple tattoo, flat skin, hair, thin hair strands "
    "across the cheek",
    "sollevante-line-diff-f41-x2130-y0": "white hair strands over blue sky",
    "sollevante-action-at-f22-x2480-y600": "purple flame bird over cyan",
    "sollevante-action-diff-f43-x2400-y1830": "cel creature head: teeth, red eye, spots, spray specks",
    "sollevante-dark-at-f22-x768-y1336": "a line: the planet's limb, a thin bright atmosphere rim; stars above, "
    "cloud texture below",
    "sollevante-dark-at-f22-x896-y1792": "a flat area: the dark, smooth planet surface (gradient: banding, noise)",
    "sollevante-dark-at-f22-x1280-y560": "a texture: pink nebula and stars",
    "digital-sunrise-at-f22-x1900-y280": "sky gradient, blue to teal, with film grain",
    "digital-sunrise-at-f22-x3144-y776": "sun disc and glow with rays, aircraft edge on the right",
    "digital-sunrise-diff-f30-x2100-y810": "horizon: haze and cloud layer over the dark ground",
    "digital-space-at-f22-x2240-y440": "rocket head and its contrail curve over the dark Earth",
    "digital-space-at-f22-x1560-y820": "sunlit cumulus tops over dark ground",
    "digital-space-diff-f41-x570-y540": "field of bright cloud rows",
    "digital-cockpit-at-f22-x1840-y700": "cheek under the eye behind the visor: sweat highlights, creases, green/yellow "
    "HUD reflections, nose wing",
    "digital-cockpit-at-f22-x2020-y430": "forehead, eyebrows, top of the eye",
    "digital-cockpit-diff-f42-x1620-y420": "helmet visor rim with rivet holes, temple and forehead skin",
    "ouatia-face-at-f22-x2400-y1000": "close-up cheek: nose wing, cheek, shadow above the lip, edge of the ear",
    "ouatia-face-at-f22-x2080-y600": "forehead: hairline, forehead skin, eyebrows",
    "ouatia-face-diff-f16-x2190-y750": "eye and eyebrow, skin between the brows and under the eye",
    "cel4k-detail-at-f22-x2160-y740": "no-U-turn sign, pole, wire fence",
    "cel4k-detail-diff-f25-x2940-y1350": "wrecked red car: grille and headlight",
    "cel4k-detail-flat-f30-x1020-y1740": "flat pavement with film grain",
}
EYES2 = {  # the user's eyes on colour's crops-b2 (sharp vs 7B, 2026-10-07: colour's eyes-verdicts-2026-10-07.txt)
    "cel4k-detail-at-f22-x2160-y740": "both models alike",
    "cel4k-detail-diff-f25-x2940-y1350": "both models alike",
    "cel4k-detail-flat-f30-x1020-y1740": "a bit more noise on the sharp but more consistent across the image",
    "digital-cockpit-at-f22-x1840-y700": "the sharp really reduces the light artefact, much better",
    "digital-cockpit-at-f22-x2020-y430": "the sharp a bit better, same reason",
    "digital-cockpit-diff-f42-x1620-y420": "alike",
    "digital-space-at-f22-x1560-y820": "both bad, but the source image is hard",
    "digital-space-at-f22-x2240-y440": "same (both bad, the source hard)",
    "digital-space-diff-f41-x570-y540": "the sharp a bit better",
    "ouatia-face-at-f22-x2080-y600": "the sharp WAY better: the reptilian texture is gone",
    "ouatia-face-at-f22-x2400-y1000": "same (the sharp way better)",
    "ouatia-face-diff-f16-x2190-y750": "same (the sharp way better)",
    "sollevante-painted-at-f22-x1340-y1000": "the sharp less blocky, less light artefact",
    "sollevante-painted-at-f22-x2580-y500": "alike",
    "sollevante-painted-diff-f42-x120-y150": "alike, leaning to the sharp: less light artefact",
}
OWN = {"sollevante-dark": [("at", 22, 768, 1336), ("at", 22, 896, 1792), ("at", 22, 1280, 560)]}
LABELS = {
    "fp8a16": "fp8 e4m3, one scale per tensor, weights only (W8A16)",
    "fp8a8": "fp8 e4m3, one scale per tensor, weights and activations (W8A8)",
    "q4k": "GGUF Q4_K (ggml's quantizer)",
    "q80": "GGUF Q8_0",
    "int8": "INT8 with comfy-kitchen's rotation (W8A8)",
    "nv4a4": "NVFP4 weights and activations (W4A4)",
    "nv4a16": "NVFP4 weights only (W4A16)",
    "nq4km": "numz's own Q4_K_M",
    "dyn": "dynamic GGUF (per-matrix types)",
    "q4ki": "GGUF Q4_K with the dynamic GGUF's importance matrix (dyn's control)",
    "sharp": "TEST: colour's sharp 7B fp16, seed 42",
    "7b43": "TEST: the 7B fp16 at seed 43",
    "sh43": "TEST: colour's sharp 7B fp16, seed 43",
}
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


def say(*lines):
    for line in lines:
        print(f"{datetime.now(PARIS).strftime('%H:%M:%S')} {line}", flush=True)


def read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def describe(L):
    if L.startswith("sh-"):
        return "the sharp 7B, " + LABELS.get(L[3:], L[3:])
    return LABELS.get(L, L)


def decode(label, cd):
    """(path, ready?) of a label's decode of shot cd."""
    own = {"7B fp16": "s42", "sharp": "sh42", "7b43": "s43", "sharp fp16": "sh42", "sh43": "sh43"}
    if label in own:
        p = f"{O}/dumps/{cd}/{own[label]}/decode.pt"
        return p, os.path.exists(p)
    p = f"{G}/dumps/{cd}/{label}-s42/decode.pt"
    return p, os.path.exists(p) and os.path.exists(f"{GPUDONE}/b-{cd}-{label}")


def windows(shot, refs):
    """[(name, kind, frame, x, y, {ref: colour's dE of that reference's panel, or None})]"""
    if shot in OWN:
        return [
            (f"{shot}-{k}-f{f}-x{x}-y{y}", k, str(f), str(x), str(y), {r: None for r in refs})
            for k, f, x, y in OWN[shot]
        ]
    rows = read_tsv(f"{WIN}/{shot}/crops.tsv")
    head = rows[0]
    cols = {
        r: head.index(f"{r.split(' fp16')[0]} {VARIANT}") for r in refs
    }  # colour's "7B ..." / "sharp ..." panel
    return [
        (r[0], r[1], r[2], r[3], r[4], {ref: float(r[c]) for ref, c in cols.items()})
        for r in rows[1:]
        if r and r[0] != "name"
    ]


def write_readme(path, lines):
    with open(path + ".tmp", "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="fp8a16,fp8a8,q4k,q80,int8,nv4a4")
    ap.add_argument("--out", default=f"{G}/crops")
    ap.add_argument("--name", default="crops-b")
    ap.add_argument(
        "--shots",
        default=None,
        help="default: the sharp's 5 shots if every label is sh-*, else the 7B's 6",
    )
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--landed", action="store_true", help="per shot, only the labels whose run of it has landed"
    )
    a = ap.parse_args()
    labels = [x for x in a.labels.split(",") if x]
    sharp = bool(labels) and all(L.startswith("sh-") or L == "sh43" for L in labels)
    refs = ["sharp fp16"] if sharp else ["7B fp16"]
    allshots = SHOTS_SH if sharp else SHOTS
    shots = [x for x in a.shots.split(",") if x] if a.shots else list(allshots)
    top = os.path.join(a.out, a.name)
    os.makedirs(top, exist_ok=True)
    made, waiting, failed, checks, used = [], [], [], [], {}
    for shot in shots:
        cd = f"{shot}-d1"
        wins = windows(shot, refs)
        decs = {L: decode(L, cd) for L in refs + labels}
        if (
            a.landed
        ):  # the labels landed on this shot, in the given order (the references must exist)
            mine = refs + [L for L in labels if decs[L][1]]
        else:
            mine = refs + labels
        used[shot] = mine
        missing = [L for L in mine if not decs[L][1]]
        sig = ["name", "kind", "frame", "x", "y", "score", "bicubic"] + [
            f"{L} {VARIANT}" for L in mine
        ]
        dst = os.path.join(top, shot)
        os.makedirs(dst, exist_ok=True)
        tsv = os.path.join(dst, "crops.tsv")
        have = {}
        if os.path.exists(tsv):
            t = read_tsv(tsv)
            for r in t[1:]:
                have[r[0]] = (t[0], r)
        todo = [
            w
            for w in wins
            if not (
                w[0] in have
                and have[w[0]][0] == sig
                and os.path.exists(os.path.join(dst, w[0] + ".png"))
            )
        ]
        if a.landed and len(mine) == len(refs):
            waiting.append(f"{shot} (no label landed yet)")
            say(f"{shot}: no label landed yet")
        elif not todo:
            say(f"{shot}: {len(wins)} strip(s) already made")
        elif missing:
            waiting.append(f"{shot} (waits for {', '.join(missing)})")
            say(f"{shot}: waits for {', '.join(missing)}")
        else:
            by_frame = {}
            for w in todo:
                by_frame.setdefault(w[2], []).append(w)
            for frame, ws in by_frame.items():
                stage = os.path.join(a.out, "stage", a.name, f"{shot}-f{frame}")
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
                    shot,
                    "--gt",
                    f"{C4}/{shot}.gt.mkv",
                    "--bicubic",
                    f"{C4}/{shot}.d1.bicubic.mkv",
                    "--ref",
                    f"{O}/dumps/{cd}/s42/ref_f32.pt",
                ]
                for L in mine:
                    cmd += ["--content", f"{L}={decs[L][0]}"]
                cmd += [
                    "--variants",
                    VARIANT,
                    "--no-stretch",
                    "--per-kind",
                    "0",
                    "--at-frame",
                    frame,
                    "--size",
                    SIZE,
                    "--threads",
                    str(a.threads),
                    "--out",
                    stage,
                ]
                for w in ws:
                    cmd += ["--at", f"{w[3]},{w[4]}"]
                say(f"{shot} f{frame}: {len(ws)} window(s), {len(mine)} decodes")
                p = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV
                )
                for line in p.stdout.splitlines()[-3:]:
                    say(f"  {line}")
                if p.returncode:
                    failed.append(f"{shot} f{frame} (exit {p.returncode})")
                    continue
                out = read_tsv(os.path.join(stage, "crops.tsv"))
                if out[0] != sig:
                    failed.append(f"{shot} f{frame}: header {out[0]}")
                    continue
                for w in ws:
                    tool = f"{shot}-at-f{w[2]}-x{w[3]}-y{w[4]}"  # the tool calls every given window "at"
                    r = next((r for r in out[1:] if r[0] == tool), None)
                    if r is None or not os.path.exists(os.path.join(stage, tool + ".png")):
                        failed.append(f"{w[0]}: not cut")
                        continue
                    shutil.move(
                        os.path.join(stage, tool + ".png"), os.path.join(dst, w[0] + ".png")
                    )
                    r = [w[0], w[1]] + r[2:]
                    have[w[0]] = (sig, r)
                    made.append(w[0])
                    for ref in refs:
                        d7, twin = float(r[sig.index(f"{ref} {VARIANT}")]), w[5].get(ref)
                        if twin is not None:
                            checks.append(
                                f"{w[0]}: {ref.split(' fp16')[0]} dE {d7:.3f}, colour's crops-b2 strip "
                                f"{twin:.3f}: "
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
        # the shot's README
        desc, verdict = SHOT[shot]
        pngs = sorted(n for n in os.listdir(dst) if n.endswith(".png"))
        shown = [L for L in mine if L not in refs]
        lines = [
            f"{a.name}\\{shot}: {desc}",
            "",
            f"Panels, left to right, 512x288 each at 1:1: GT | bicubic | {refs[0]} | "
            + " | ".join(shown)
            + f" (all {VARIANT} but GT and bicubic)."
            + (
                ""
                if not a.landed or len(shown) == len(labels)
                else f" The labels whose run of this shot has landed so far ({len(shown)} of {len(labels)}); the strips are "
                "re-made as more land."
            ),
            f"Colour's baton-1 verdict on the shot (7B, colour variants): {verdict}.",
        ]
        if sharp:
            lines += [
                "The user's eyes on colour's crops-b2 (sharp vs 7B, 2026-10-07): per window below."
            ]
        lines += [
            "",
            "Windows (x, y = top-left corner in the 4K frame; frames 0-based; kind at = placed by hand, diff = "
            "colour's automatic pick, flat = colour's flattest GT window):",
        ]
        for w in wins:
            lines.append(
                f"  {w[0]}.png: {'made' if w[0] + '.png' in pngs else 'not made yet'}; x {w[3]}, y {w[4]}, "
                f"frame {w[2]}, {w[1]}: {WHAT.get(w[0], '?')}"
                + (
                    f"; the user's eyes, sharp vs 7B: {EYES2[w[0]]}"
                    if sharp and w[0] in EYES2
                    else ""
                )
            )
        write_readme(os.path.join(dst, "README.txt"), lines)
    # the top README
    nwin = {s: len(windows(s, refs)) for s in allshots}
    nmade = {
        s: (
            len([n for n in os.listdir(os.path.join(top, s)) if n.endswith(".png")])
            if os.path.isdir(os.path.join(top, s))
            else 0
        )
        for s in allshots
    }
    if sharp:
        lines = [
            f"{a.name}: the sharp 7B's phase-2 model files beside the sharp 7B fp16 at 4K, for the eyes",
            f"Model conversation, slice B of the sharp's files (2026-10-07). Updated "
            f"{datetime.now(PARIS).strftime('%Y-%m-%d %H:%M')} Paris.",
            "",
            "PANELS, left to right, each 512x288 at 1:1 (no scaling), under a label strip:",
            "  GT | bicubic | sharp fp16 split:ycc:4:3 | "
            + " | ".join(f"{L} split:ycc:4:3" for L in labels),
            "- sharp fp16 = SeedVR2 7B sharp (seedvr2_ema_7b_sharp_fp16, numz's file), seed 42: colour's baton-2",
            "  runs (crops-b2's sharp panel).",
            "- every other panel = the same sharp 7B from one of our phase-2 files, seed 42, the same input and",
            "  numz flags:",
        ]
    else:
        lines = [
            f"{a.name}: the phase-2 model files beside the 7B fp16 at 4K, for the eyes",
            f"Model conversation, slice B (night of 2026-10-07). Updated {datetime.now(PARIS).strftime('%Y-%m-%d %H:%M')} Paris.",
            "",
            "PANELS, left to right, each 512x288 at 1:1 (no scaling), under a label strip:",
            "  GT | bicubic | 7B fp16 split:ycc:4:3 | "
            + " | ".join(f"{L} split:ycc:4:3" for L in labels),
            "- 7B fp16 = SeedVR2 7B (seedvr2_ema_7b_fp16, numz's file), seed 42: colour's batch-A runs.",
            "- every other panel = the same 7B from one of our phase-2 files, seed 42, the same input and numz flags:",
        ]
    lines += [f"    {L} = {describe(L)}" for L in labels]
    if a.landed:
        lines += [
            "- Each shot's strips hold the labels whose run of that shot has landed (in this order); they are",
            "  re-made as more land: see each folder's README.txt for the panels it has now.",
        ]
    lines += [
        "- x2 to 4K from d1 (x1/2 Mitchell, x264 CRF 20). split:ycc:4:3 = seedvr2x's default colour correction,",
        "  applied to each decode with the 7B run's reference"
        + (" (the sharp's: the same file, md5-checked by colour)." if sharp else ".")
        + " Bicubic: the d1 input upscaled x2 (Catmull-Rom).",
        '- "dE x" in a label: the panel\'s mean CIEDE2000 colour difference to the GT over the window (also in',
        "  <shot>\\crops.tsv). No luma-stretched .stretch.png twins. The strips are "
        f"{512 * (3 + len(labels))} px wide{' when every label is in' if a.landed else ''}: view at 100%.",
    ]
    if sharp:
        lines += [
            "- Kinds of source: digital = live-action digital (cockpit, space); first film = live action, film",
            "  (the face close-up); Sol Levante = painted anime (painted); cel = animation film scan (detail).",
            "WINDOWS: colour's crops-b2 windows (baton 1's, unchanged: position, size, frame, name), 3 per shot.",
            "  Each folder's README.txt says what each window shows and the user's eyes on it (sharp vs 7B).",
            "",
        ]
    else:
        lines += [
            "- Kinds of source: Sol Levante = painted anime (painted, line, action, dark); digital = live-action",
            "  digital (sunrise, space).",
            "WINDOWS: colour's crops-b2 windows (baton 1's, unchanged: position, size, frame, name) for 5 shots;",
            "  sollevante-dark has none there: 3 placed by hand here on frame 22 (a line, a flat area, a texture).",
            "  Each folder's README.txt says what each window shows.",
            "",
        ]
    lines += [f"STRIPS: {sum(nmade.values())} of {sum(nwin.values())} made"]
    lines += [
        f"- {s}: {nmade[s]} of {nwin[s]}"
        + (f" (panels: {', '.join(used[s][1:]) or 'none yet'})" if a.landed and s in used else "")
        for s in allshots
    ]
    if waiting:
        lines += ["", "WAITING: " + "; ".join(waiting)]
    write_readme(os.path.join(top, "README.txt"), lines)
    say(*checks)
    say(
        f"made {len(made)}, waiting {len(waiting)}, failed {len(failed)}"
        + (": " + "; ".join(failed) if failed else "")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
