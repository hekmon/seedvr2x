#!/usr/bin/env python3
"""Review sheets for scene-cut candidates (scd_scores.py analyse): one row per candidate with
frames c-2, c-1, c, c+1 (a real cut falls between c-1 and c, marked red), the scdet score and
PySceneDetect's verdicts (and TransNetV2's, once analysed with it); an index (CSV and Markdown)
with an empty label column.

  scd_review.py sheets DIR... --out REVIEW [--rows 15] [--width 320] [--cap 120] [--sure-frac 0.1]
  scd_review.py extend REVIEW DIR... [--cap 30] [--seed 1] [--skip NAME...]  # TransNetV2's own
  scd_review.py index REVIEW         # add REVIEW/*/index.csv to REVIEW/index.{csv,md}

An episode already in REVIEW is skipped (--force makes it again). The combined index is only
ever appended to: the episodes it lacks are added at its end, its existing rows (and the labels
written in them) are left byte for byte, in the delimiter and line ends the file already uses.

Selection (sheets), per episode, among the candidates of scdet, PySceneDetect and segment joins
(those of TransNetV2 alone are extend's): a random --sure-frac of the sure candidates (to check
the auto-class), and every doubtful one up to --cap. Above it, --max-rows doubtful rows at
most, sampled by stratum: the decision group first (scdet in [--keep-lo, --keep-hi], the scores
that decide a threshold in 8-14, plus PySceneDetect detections scoring below --keep-lo, the
possible misses, and segment joins scoring below --keep-lo), whole if it fits, then the rest
(scdet 4-6 that PySceneDetect ignores, scdet above --keep-hi): the unused rows, and at least
--rest-frac of it (between --rest-min and --rest-cap rows). A stratum is a group, a scdet band
(4, 6, 8, 10, 12, 14, 17, 20, 30) and PySceneDetect's agreement (both / one / none); samples are
allocated in proportion to stratum sizes, one row per stratum at least. The index gives each
row's stratum, its size and its weight (candidates the row stands for): scd_scores.py summary
--labels turns labels into estimates, even when only part of the rows are labelled. Sampling is
seeded (--seed) and repeatable.

Thumbnails come from one sequential decode of the source, the score pass's: same files, frames
counted from 0 in decode order (ffmpeg's trim filter keeps the span from the first frame needed
to the last; no seeking), scaled to --width at the display aspect (the stream's sample aspect
ratio: anamorphic sources are shown unsqueezed, interlaced ones as decoded) with the BT.709
matrix (--matrix). As a check of the frame indexing, the luma change between the thumbnails of
each sampled sure cut must be largest between c-1 and c.

hints.csv (per episode, kept apart from the sheets and the index so as not to steer the labels):
mean absolute luma changes between the thumbnails (before = c-2 to c-1, jump = c-1 to c, after =
c to c+1, back = c-1 to c+1), mean luma, and a heuristic hint: transient (c+1 back near c-1: a
flash or an odd frame), motion (change all along, no peak), step (an isolated change that stays:
cut-like), tiny (hardly any change), plus dark (both frames dim) and brightness (the change is
mostly a global luma shift: fade, flash, exposure) and burst (2 or more other frames scoring
>= 6 within +-6 frames). Unverified: a first look before the labels.

extend: the candidates of TransNetV2 alone (scd_scores.py analyse with tnet.npz: its aligned
peaks >= 0.1 with no candidate of scdet, PySceneDetect or a join within +-1 frame) added to the
episodes already in REVIEW, without touching a row or a page there. Strata "tnet <band>" by the
candidate's TransNetV2 probability (0.1-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, >=0.9); all of them shown
up to --cap per episode, else --cap rows sampled by stratum (one at least each, seeded), weight
= stratum size / rows taken, selection tnet-all (the whole stratum) or tnet-sampled. Thumbnails
as sheets takes them. The pages are new files numbered on from the episode's last page and
headed as TransNetV2-only candidates; the rows are appended to the episode's index.csv and
index.md and to REVIEW/index.csv and index.md, same columns, in the delimiter and line ends each
file uses, every existing byte kept; hints-tnet.csv/md and note-tnet.txt hold the rest. Frame
indexing check: of the rows whose thumbnails show one clear change (>= 8 luma levels, twice the
next largest), the share where it falls between c-1 and c. Running it again adds nothing: an
episode whose index.csv holds tnet rows gets no page, REVIEW/index.csv only the tnet rows it
lacks.

Labels: cut / flash / pan / fade / dissolve / other / not-a-cut.

Needs ffmpeg, numpy and Pillow; scd_scores.py next to it.
"""
import argparse
import csv
import io
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scd_scores import (LABELS, decode_cmd, load_json, load_scores, log, old_population,  # noqa: E402
                        pysd_agreement, read_full, score_band, segments_of, timecode, tnet_band)

INDEX_COLS = ["page", "row", "episode", "frame", "timecode", "class", "stratum", "stratum_size", "weight",
              "selection", "scdet", "mafd", "prev", "next", "burst", "adaptive", "content", "ad_ratio",
              "content_val", "join", "label"]
TNET_PREFIX = "tnet "  # strata of the candidates of TransNetV2 alone (extend)
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/TTF/DejaVuSans.ttf")


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_candidates(d):
    with open(os.path.join(d, "candidates.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(r["frame"])
        for k in ("scdet", "mafd", "prev", "next", "ad_ratio", "content_val"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else math.nan
    return rows


def allocate(groups, n):
    """Rows per stratum: proportional to its size, one at least, n in all (largest remainders)."""
    total = sum(len(g) for g in groups.values())
    if n >= total:
        return {k: len(g) for k, g in groups.items()}
    share = {k: n * len(g) / total for k, g in groups.items()}
    alloc = {k: max(1, min(len(groups[k]), int(share[k]))) for k in groups}
    order = sorted(groups, key=lambda k: share[k] - int(share[k]), reverse=True)
    while sum(alloc.values()) < n:
        grew = False
        for k in order:
            if sum(alloc.values()) < n and alloc[k] < len(groups[k]):
                alloc[k] += 1
                grew = True
        if not grew:
            break
    return alloc


def select_rows(cands, a, rng):
    for c in cands:
        c["band"], c["pysd"] = score_band(c["scdet"]), pysd_agreement(c)
        if c["class"] == "sure":
            c["group"] = "sure"
        elif a.keep_lo <= c["scdet"] <= a.keep_hi or (
                (c["pysd"] != "none" or c["join"]) and c["scdet"] < a.keep_lo):
            c["group"] = "decision"
        else:
            c["group"] = "rest"
        c["stratum"] = "sure" if c["group"] == "sure" else f"{c['group']} {c['band']}/{c['pysd']}"
    sure = [c for c in cands if c["group"] == "sure"]
    dec = [c for c in cands if c["group"] == "decision"]
    rest = [c for c in cands if c["group"] == "rest"]
    n_doubt = len(dec) + len(rest)
    n_sure = min(len(sure), max(1, round(a.sure_frac * len(sure)))) if sure else 0
    if n_doubt <= a.cap:
        n_dec, n_rest = len(dec), len(rest)
        note = f"{n_doubt} doubtful, all shown"
    else:
        n_rest = min(len(rest), max(a.rest_min, min(a.rest_cap, math.ceil(a.rest_frac * len(rest)))))
        n_dec = min(len(dec), a.max_rows - n_rest)
        n_rest = min(len(rest), a.max_rows - n_dec)
        note = (f"{n_doubt} doubtful, {n_dec + n_rest} shown: {n_dec} of the {len(dec)} of the decision "
                f"group (scdet {a.keep_lo:g}-{a.keep_hi:g}; PySceneDetect or joins below {a.keep_lo:g}) "
                f"and {n_rest} of the other {len(rest)}, sampled by stratum")
    picked = []
    for part, n, sel in ((sure, n_sure, "sure-sample"), (dec, n_dec, None), (rest, n_rest, None)):
        groups = defaultdict(list)
        for c in part:
            groups[c["stratum"]].append(c)
        alloc = allocate(groups, n)
        for k in sorted(groups):
            g = groups[k]
            take = rng.sample(g, alloc[k]) if alloc[k] < len(g) else list(g)
            for c in take:
                c["stratum_size"], c["weight"] = len(g), len(g) / len(take)
                c["selection"] = sel or ("all" if len(take) == len(g) else "sampled")
                picked.append(c)
    note += f"; {n_sure} of {len(sure)} sure sampled"
    return sorted(picked, key=lambda c: c["frame"]), note


def grab(src, frames, width, matrix, threads):
    """Thumbnails {frame: PIL image} of the given frames, from sequential decodes."""
    w, h = src["width"], src["height"]
    try:
        sar = Fraction(*(int(x) for x in str(src.get("sar") or "1:1").split(":")))
    except (ValueError, ZeroDivisionError, TypeError):
        sar = Fraction(1)
    sar = sar if sar > 0 else Fraction(1)
    th = int(round(h * width / float(w * sar) / 2)) * 2
    size = width * th * 3
    conv = (f"scale={width}:{th}:flags=bicubic:in_color_matrix={matrix}:in_range=tv:out_range=pc,"
            "format=rgb24")
    want = sorted(set(f for f in frames if 0 <= f < src["frames"]))
    out = {}
    buf = bytearray(size)
    for path, off, cnt in segments_of(src):
        local = [f - off for f in want if f >= off and (cnt is None or f < off + cnt)]
        if not local:
            continue
        lo, hi = local[0], local[-1] + 1
        cmd = decode_cmd(path, "rgb24", threads, vf=f"trim=start_frame={lo}:end_frame={hi},{conv}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        need = set(local)
        try:
            for f in range(lo, hi):
                if read_full(p.stdout, buf) != size:
                    raise RuntimeError(f"{path}: frame {f} missing from the decode")
                if f in need:
                    out[off + f] = Image.frombytes("RGB", (width, th), bytes(buf))
        finally:
            p.stdout.close()
            p.wait()
    return out, (width, th)


def luma(img):
    a = np.asarray(img, np.float32)
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


HINT_COLS = ["episode", "frame", "class", "stratum", "scdet", "adaptive", "content", "burst", "before",
             "jump", "after", "back", "lum_prev", "lum_cur", "hint"]
BAND_ORDER = {b: i for i, b in enumerate(["<4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-17", "17-20",
                                          "20-30", ">=30"])}


def stratum_order(key):
    """sure last, then by group and scdet band."""
    band = key.split()[-1].split("/")[0]
    return key.split()[0], BAND_ORDER.get(band, 99), key


def hint(ims, burst):
    """Luma changes around a candidate (thumbnails c-2, c-1, c, c+1) and a heuristic hint."""
    y = [luma(t) for t in ims]

    def mad(a, b):
        return float(np.abs(a - b).mean())

    before, jump, after, back = mad(y[0], y[1]), mad(y[1], y[2]), mad(y[2], y[3]), mad(y[1], y[3])
    lum = [float(v.mean()) for v in y]
    kind = []
    if jump < 2:
        kind.append("tiny")
    elif back < 0.5 * jump:
        kind.append("transient")
    elif min(before, after) > 0.5 * jump:
        kind.append("motion")
    elif jump > 2.5 * max(before, after, 1.0):
        kind.append("step")
    if max(lum[1], lum[2]) < 40:
        kind.append("dark")
    if jump >= 2 and abs(lum[2] - lum[1]) > 0.6 * jump:
        kind.append("brightness")
    if burst >= 2:
        kind.append("burst")
    return {"before": before, "jump": jump, "after": after, "back": back, "lum_prev": lum[1],
            "lum_cur": lum[2], "hint": "+".join(kind)}


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def verdict(c, key, metric, nd):
    hit = c[key]
    if hit:
        off = int(hit) - c["frame"]
        where = "c" if off == 0 else f"c{off:+d}"
        return f"cut at {where}  ({metric} {fmt(c['ad_ratio'] if key == 'adaptive' else c['content_val'], nd)})"
    return f"no  ({metric} {fmt(c['ad_ratio'] if key == 'adaptive' else c['content_val'], nd)})"


def tnet_verdict(c):
    pk = c.get("tnet_peak")
    where = "" if pk in (None, "") else ("  at c" if int(pk) == c["frame"] else f"  at c{int(pk) - c['frame']:+d}")
    return f"TransNetV2: {float(c['tnet']):.3f}{where}  (all-frames {float(c['tnet_all']):.3f})"


def draw_pages(name, rows, thumbs, tsize, fps, pts, out_dir, per_page, note, first=1, what=None):
    """page_NNN.jpg from number first; what (extend's pages) heads them, and they never overwrite."""
    tw, th = tsize
    f_lab, f_cap, f_head = font(15), font(13), font(18)
    label_w, gap, mid_gap, margin, cap_h, pad = 330, 6, 14, 10, 20, 14
    row_h = th + cap_h + pad
    head_h = 34
    page_w = margin * 2 + label_w + 4 * tw + 2 * gap + mid_gap
    pages = [rows[i:i + per_page] for i in range(0, len(rows), per_page)]
    for p, page in enumerate(pages, 1):
        num = first + p - 1
        path = os.path.join(out_dir, f"page_{num:03d}.jpg")
        if what and os.path.exists(path):
            raise RuntimeError(f"{path} exists: a page is never overwritten")
        img = Image.new("RGB", (page_w, head_h + len(page) * row_h + margin), "white")
        dr = ImageDraw.Draw(img)
        span = f"rows {(p - 1) * per_page + 1}-{(p - 1) * per_page + len(page)} of {len(rows)}"
        head = (f"{name}   page {p}/{len(pages)}   {span}" if not what else
                f"{name}   page {num}: {what} {p}/{len(pages)}, {span}")
        dr.text((margin, 8), f"{head}   frames c-2, c-1 | c, c+1: a real cut falls on the red mark", fill="black",
                font=f_head)
        for r, c in enumerate(page, 1):
            c["page"], c["row"] = num, r
            y = head_h + (r - 1) * row_h
            dr.line([(margin, y - 4), (page_w - margin, y - 4)], fill=(200, 200, 200), width=1)
            sel = f"{c['stratum']}: {c['selection']}" + (f", w {c['weight']:.1f}" if c["weight"] != 1 else "")
            lines = [
                (f"#{r}  {c['class'].upper()}", "black"),
                (f"frame {c['frame']}  {c['timecode']}", "black"),
                (f"scdet {fmt(c['scdet'])}   mafd {fmt(c['mafd'], 2)}", "black"),
                (f"  prev {fmt(c['prev'], 2)}   next {fmt(c['next'], 2)}   burst {c['burst']}", (90, 90, 90)),
                ("Adaptive: " + verdict(c, "adaptive", "ratio", 2), "black"),
                ("Content: " + verdict(c, "content", "val", 1), "black"),
            ]
            if c.get("tnet") not in (None, ""):
                lines.append((tnet_verdict(c), "black"))
            if c["join"]:
                lines.append((f"segment join at {c['join']}", (160, 0, 0)))
            lines.append((sel, (90, 90, 90)))
            for k, (txt, col) in enumerate(lines):
                dr.text((margin, y + 2 + k * 19), txt, fill=col, font=f_lab)
            x = margin + label_w
            for k, f in enumerate((c["frame"] - 2, c["frame"] - 1, c["frame"], c["frame"] + 1)):
                if k == 2:
                    dr.rectangle([x - mid_gap + 4, y, x - 5, y + th], fill=(220, 0, 0))
                t = thumbs.get(f)
                if t is not None:
                    img.paste(t, (x, y))
                else:
                    dr.rectangle([x, y, x + tw, y + th], fill=(128, 128, 128))
                tag = ("c-2", "c-1", "c", "c+1")[k]
                dr.text((x + 2, y + th + 2), f"{tag}  {f}  {timecode(f, fps, pts)}" if f >= 0 else tag,
                        fill="black", font=f_cap)
                x += tw + (mid_gap if k == 1 else gap)
        img.save(path, quality=85, optimize=True)
    return len(pages)


def write_index(rows, csv_path, md_path, title, note):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(INDEX_COLS)
        for c in rows:
            w.writerow([index_value(c, k) for k in INDEX_COLS])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("Fill the label column (in the CSV) with one of: " + " / ".join(LABELS) + ".\n"
                "Frames c-2, c-1 | c, c+1 of each row; a real cut falls between c-1 and c (red mark). "
                "adaptive / content: the frame where PySceneDetect's detector cuts, if it does (within +-1). "
                "class: sure (scdet >= 30 and both PySceneDetect detectors), else doubtful. stratum: group "
                "(sure; decision: scdet 6-20, PySceneDetect or joins below 6; rest), scdet band and "
                "PySceneDetect agreement; selection: all (the whole stratum), sampled, sure-sample; weight: "
                "candidates the row stands for. Rows labelled in part still give estimates "
                "(scd_scores.py summary --labels).\n\n")
        if note:
            f.write(note + "\n\n")
        f.write(md_table(rows))


MD_COLS = ["page", "row", "episode", "frame", "timecode", "class", "scdet", "adaptive", "content", "join",
           "stratum", "weight", "label"]


def md_table(rows):
    out = "| " + " | ".join(MD_COLS) + " |\n|" + "---|" * len(MD_COLS) + "\n"
    for c in rows:
        out += "| " + " | ".join(str(index_value(c, k)) for k in MD_COLS) + " |\n"
    return out


def index_value(c, k):
    v = c.get(k, "")
    if isinstance(v, float):
        return "" if math.isnan(v) else (f"{v:.1f}" if k == "weight" else f"{v:.3f}")
    return "" if v is None else v


def write_hints(ep_dir, stem, name, rows, thumbs, cols=HINT_COLS, what=""):
    """stem.csv: each row's thumbnail luma changes and hint; stem.md: the hints per stratum, weighted."""
    kinds = ("step", "transient", "motion", "tiny", "")
    tally = defaultdict(lambda: defaultdict(float))
    with open(os.path.join(ep_dir, stem + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in rows:
            ims = [thumbs.get(c["frame"] + k) for k in (-2, -1, 0, 1)]
            if any(t is None for t in ims):
                continue
            h = {**c, **hint(ims, int(c["burst"]))}
            w.writerow([f"{h[k]:.2f}" if isinstance(h[k], float) else h[k] for k in cols])
            tags = h["hint"].split("+")
            t = tally[c["stratum"]]
            t[next((k for k in kinds if k in tags), "")] += c["weight"]
            t["dark"] += c["weight"] * ("dark" in tags)
            t["brightness"] += c["weight"] * ("brightness" in tags)
            t["burst"] += c["weight"] * ("burst" in tags)
            t["n"] += 1
    with open(os.path.join(ep_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(f"# Heuristic hints, {name}{what} (unverified, before any label)\n\nCandidates per stratum "
                "(weighted: what the sampled rows stand for) by thumbnail pattern; rows = sampled "
                "rows.\n\n| stratum | rows | step | transient | motion | tiny | other | dark | brightness "
                "| burst |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for key in sorted(tally, key=stratum_order):
            t = tally[key]
            f.write(f"| {key} | {int(t['n'])} | " + " | ".join(f"{t[k]:.0f}" for k in kinds) +
                    f" | {t['dark']:.0f} | {t['brightness']:.0f} | {t['burst']:.0f} |\n")


def cmd_sheets(a):
    os.makedirs(a.out, exist_ok=True)
    for d in a.dirs:
        st = load_json(os.path.join(d, "stats.json"))
        src = load_json(os.path.join(d, "source.json"))
        name = st["name"]
        fps = src["fps"]
        ep_dir = os.path.join(a.out, name)
        if os.path.exists(os.path.join(ep_dir, "index.csv")) and not a.force:
            log(f"{name}: already in {a.out}, skipped (--force makes it again)")
            continue
        cands = [c for c in load_candidates(d) if old_population(c)]
        rng = random.Random(f"{a.seed}:{name}")
        rows, note = select_rows(cands, a, rng)
        for c in rows:
            c["episode"] = name
            c["label"] = ""
        frames = [f for c in rows for f in range(c["frame"] - 2, c["frame"] + 2)]
        log(f"{name}: {note}; {len(rows)} rows, {len(set(frames))} frames to grab")
        thumbs, tsize = grab(src, frames, a.width, a.matrix, a.threads)
        # frame indexing check: on sure cuts, the largest thumbnail change is between c-1 and c
        ok = bad = 0
        for c in rows:
            if c["class"] != "sure":
                continue
            ims = [thumbs.get(c["frame"] + k) for k in (-2, -1, 0, 1)]
            if any(t is None for t in ims):
                continue
            y = [luma(t) for t in ims]
            dif = [float(np.abs(y[i + 1] - y[i]).mean()) for i in range(3)]
            if dif[1] > dif[0] and dif[1] > dif[2]:
                ok += 1
            else:
                bad += 1
                log(f"  sure cut {c['frame']}: thumbnail changes {['%.1f' % x for x in dif]}")
        note += f"; thumbnail check on sampled sure cuts: {ok} aligned, {bad} not"
        if a.note:
            note += f"; {a.note}"
        os.makedirs(ep_dir, exist_ok=True)
        write_hints(ep_dir, "hints", name, rows, thumbs)
        for old in os.listdir(ep_dir):
            if old.startswith("page_") and old.endswith(".jpg"):
                os.remove(os.path.join(ep_dir, old))
        n_pages = draw_pages(name, rows, thumbs, tsize, fps, load_scores(d)["pts_time"], ep_dir, a.rows, note)
        note = f"{len(rows)} rows on {n_pages} pages: {note}"
        with open(os.path.join(ep_dir, "note.txt"), "w", encoding="utf-8") as f:
            f.write(note + "\n")
        write_index(rows, os.path.join(ep_dir, "index.csv"), os.path.join(ep_dir, "index.md"),
                    f"Scene-cut review: {name}", note)
        log(f"{name}: {note} ({ep_dir})")
    merge_index(a.out)


def episode_rows(review, name):
    with open(os.path.join(review, name, "index.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    q = os.path.join(review, name, "note.txt")
    note = ""
    if os.path.isfile(q):
        with open(q, encoding="utf-8") as f:
            note = f"- {name}: {f.read().strip()}"
    return rows, note


def read_csv_keep(path):
    """A CSV's rows, and the delimiter, line end and columns it uses (a spreadsheet may have saved it
    with ';' or other line ends): what append_rows needs to add rows without touching a byte."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig")
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    eol = "\r\n" if b"\r\n" in raw[:65536] else "\n"
    header = next(csv.reader([first], delimiter=delim)) if first else INDEX_COLS
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    return rows, {"delim": delim, "eol": eol, "header": header, "open_line": bool(raw) and not raw.endswith(b"\n")}


def append_rows(path, rows, fmt):
    """Append rows (dicts) to a CSV read by read_csv_keep, in its delimiter, line ends and columns."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        if fmt["open_line"]:
            f.write(fmt["eol"])
        w = csv.writer(f, delimiter=fmt["delim"], lineterminator=fmt["eol"])
        for r in rows:
            w.writerow([r.get(k, "") for k in fmt["header"]])


def merge_index(review):
    """Create REVIEW/index.{csv,md}, or append the episodes they lack; never rewrite a row."""
    names = sorted(n for n in os.listdir(review) if os.path.isfile(os.path.join(review, n, "index.csv")))
    csv_path, md_path = os.path.join(review, "index.csv"), os.path.join(review, "index.md")
    if not os.path.exists(csv_path):
        rows, notes = [], []
        for n in names:
            r, note = episode_rows(review, n)
            rows += r
            notes.append(note)
        write_index(rows, csv_path, md_path, "Scene-cut review: all episodes", "\n".join(notes))
        log(f"index: {len(rows)} rows from {len(names)} episodes in {csv_path} and index.md")
        return
    have, fmt = read_csv_keep(csv_path)
    present = {r.get("episode") for r in have}
    new = [n for n in names if n not in present]
    if not new:
        log(f"index: {csv_path} already lists every episode, nothing appended")
        return
    rows, notes = [], []
    for n in new:
        r, note = episode_rows(review, n)
        rows += r
        notes.append(note)
    append_rows(csv_path, rows, fmt)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Added {time.strftime('%Y-%m-%d')}: {', '.join(new)}\n\n" + "\n".join(notes) +
                "\n\n" + md_table(rows))
    log(f"index: {len(rows)} rows of {len(new)} episodes ({', '.join(new)}) appended to {csv_path} and index.md")


TNET_NOTE = ("Candidates of TransNetV2 alone: its aligned peaks >= 0.1 with no candidate of scdet, PySceneDetect or "
             "a segment join within +-1 frame. stratum: tnet and the band of TransNetV2's probability (its highest "
             "within +-1 frame); selection: tnet-all (the whole stratum) or tnet-sampled; weight: candidates the row "
             "stands for. Frames, marks and labels as above.")


def tnet_section(title, notes):
    return f"\n## Added {time.strftime('%Y-%m-%d')}: {title}\n\n{TNET_NOTE}\n\n{notes}\n\n"


def tnet_rows(cands, cap, rng):
    """The candidates of TransNetV2 alone to review, in strata by TransNetV2 band: all of them up to
    cap, else cap rows sampled by stratum (one at least each)."""
    new = [c for c in cands if not old_population(c)]
    groups = defaultdict(list)
    for c in new:
        c["tnet"], c["tnet_all"] = float(c["tnet"]), float(c["tnet_all"])
        c["stratum"] = TNET_PREFIX + tnet_band(c["tnet"])
        groups[c["stratum"]].append(c)
    alloc = allocate(groups, min(cap, len(new)))
    picked = []
    for k in sorted(groups):
        g = groups[k]
        take = rng.sample(g, alloc[k]) if alloc[k] < len(g) else list(g)
        for c in take:
            c["stratum_size"], c["weight"] = len(g), len(g) / len(take)
            c["selection"] = "tnet-all" if len(take) == len(g) else "tnet-sampled"
            picked.append(c)
    note = (f"{len(new)} candidates of TransNetV2 alone (" +
            ", ".join(f"{k[len(TNET_PREFIX):]}: {len(groups[k])}" for k in sorted(groups)) + "), " +
            ("all shown" if len(picked) == len(new) else f"{len(picked)} shown, sampled by stratum"))
    return sorted(picked, key=lambda c: c["frame"]), note


def thumb_check(rows, thumbs):
    """Where the thumbnails' one clear change falls (>= 8 luma levels, twice the next largest):
    counts at c-2|c-1, c-1|c, c|c+1, for every row and for those with TransNetV2 >= 0.5. A cut found
    at the right frame changes between c-1 and c."""
    at, at_hi = [0, 0, 0], [0, 0, 0]
    for c in rows:
        ims = [thumbs.get(c["frame"] + k) for k in (-2, -1, 0, 1)]
        if any(t is None for t in ims):
            continue
        y = [luma(t) for t in ims]
        dif = [float(np.abs(y[i + 1] - y[i]).mean()) for i in range(3)]
        i = int(np.argmax(dif))
        if dif[i] >= 8 and dif[i] >= 2 * sorted(dif)[1]:
            at[i] += 1
            at_hi[i] += c["tnet"] >= 0.5
            if i != 1:
                log(f"  {c['frame']} (TransNetV2 {c['tnet']:.2f}): thumbnail changes {['%.1f' % x for x in dif]}")
    return at, at_hi


def cmd_extend(a):
    done = []
    for d in a.dirs:
        st = load_json(os.path.join(d, "stats.json"))
        src = load_json(os.path.join(d, "source.json"))
        name = st["name"]
        ep_dir = os.path.join(a.review, name)
        idx = os.path.join(ep_dir, "index.csv")
        if name in a.skip:
            log(f"{name}: skipped (--skip)")
            continue
        if not os.path.isfile(idx):
            log(f"{name}: not in {a.review} (sheets makes an episode's first rows), skipped")
            continue
        have, fmt = read_csv_keep(idx)
        done.append(name)
        if any((r.get("stratum") or "").startswith(TNET_PREFIX) for r in have):
            log(f"{name}: {idx} already holds TransNetV2 rows, no page added")
            continue
        cands = load_candidates(d)
        if cands and "tnet" not in cands[0]:
            sys.exit(f"{d}/candidates.csv has no tnet column: run scd_scores.py tnet, then analyse")
        rows, note = tnet_rows(cands, a.cap, random.Random(f"{a.seed}:{name}:tnet"))
        if not rows:
            log(f"{name}: {note}, nothing to add")
            continue
        last = max((int(r["page"]) for r in have if (r.get("page") or "").isdigit()), default=0)
        files = [int(m.group(1)) for m in (re.match(r"page_(\d+)\.jpg$", f) for f in os.listdir(ep_dir)) if m]
        if max(files, default=0) > last:
            sys.exit(f"{ep_dir}: page files beyond the index's last page ({last}), from an interrupted extend? "
                     "Remove them first")
        for c in rows:
            c["episode"], c["label"] = name, ""
        frames = [f for c in rows for f in range(c["frame"] - 2, c["frame"] + 2)]
        log(f"{name}: {note}; {len(rows)} rows, {len(set(frames))} frames to grab")
        thumbs, tsize = grab(src, frames, a.width, a.matrix, a.threads)
        at, at_hi = thumb_check(rows, thumbs)
        note += (f"; thumbnail check, rows with one clear change at c-2|c-1, c-1|c, c|c+1: {'/'.join(map(str, at))}, "
                 f"of them with TransNetV2 >= 0.5: {'/'.join(map(str, at_hi))}")
        n_pages = draw_pages(name, rows, thumbs, tsize, src["fps"], load_scores(d)["pts_time"], ep_dir, a.rows,
                             note, first=last + 1, what="TransNetV2-only candidates")
        write_hints(ep_dir, "hints-tnet", name, rows, thumbs, HINT_COLS + ["tnet", "tnet_all"],
                    ", TransNetV2-only candidates")
        note = f"{len(rows)} rows on {n_pages} pages ({last + 1}-{last + n_pages}): {note}"
        with open(os.path.join(ep_dir, "note-tnet.txt"), "w", encoding="utf-8") as f:
            f.write(note + "\n")
        with open(os.path.join(ep_dir, "index.md"), "a", encoding="utf-8") as f:
            f.write(tnet_section("TransNetV2-only candidates", note) + md_table(rows))
        append_rows(idx, [{k: index_value(c, k) for k in INDEX_COLS} for c in rows], fmt)
        log(f"{name}: {note} ({ep_dir})")
    merge_tnet(a.review, done)


def merge_tnet(review, names):
    """Append to REVIEW/index.{csv,md} the TransNetV2 rows of the episodes' indexes that it lacks."""
    csv_path, md_path = os.path.join(review, "index.csv"), os.path.join(review, "index.md")
    if not os.path.exists(csv_path):
        merge_index(review)
        return
    have, fmt = read_csv_keep(csv_path)
    present = {(r.get("episode"), r.get("frame")) for r in have if (r.get("stratum") or "").startswith(TNET_PREFIX)}
    rows, notes, eps = [], [], []
    for n in names:
        er, _ = read_csv_keep(os.path.join(review, n, "index.csv"))
        add = [r for r in er if (r.get("stratum") or "").startswith(TNET_PREFIX) and
               (r.get("episode"), r.get("frame")) not in present]
        if not add:
            continue
        rows += add
        eps.append(n)
        q = os.path.join(review, n, "note-tnet.txt")
        if os.path.isfile(q):
            with open(q, encoding="utf-8") as f:
                notes.append(f"- {n}: {f.read().strip()}")
    if not rows:
        log(f"index: {csv_path} already holds the TransNetV2 rows of {', '.join(names) or 'no episode'}, "
            "nothing appended")
        return
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(tnet_section(f"TransNetV2-only candidates of {', '.join(eps)}", "\n".join(notes)) + md_table(rows))
    append_rows(csv_path, rows, fmt)
    log(f"index: {len(rows)} TransNetV2 rows of {len(eps)} episodes ({', '.join(eps)}) appended to {csv_path} "
        "and index.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheets")
    p.add_argument("dirs", nargs="+", help="episode directories of scd_scores.py (analysed)")
    p.add_argument("--out", required=True, help="review directory (one subdirectory per episode)")
    p.add_argument("--rows", type=int, default=15, help="rows per page")
    p.add_argument("--width", type=int, default=320, help="thumbnail width")
    p.add_argument("--cap", type=int, default=120, help="doubtful candidates shown in full up to this")
    p.add_argument("--max-rows", type=int, default=120, help="doubtful rows at most, above --cap")
    p.add_argument("--keep-lo", type=float, default=6.0)
    p.add_argument("--keep-hi", type=float, default=20.0)
    p.add_argument("--rest-frac", type=float, default=0.1)
    p.add_argument("--rest-min", type=int, default=10)
    p.add_argument("--rest-cap", type=int, default=24)
    p.add_argument("--sure-frac", type=float, default=0.1)
    p.add_argument("--seed", default="1")
    p.add_argument("--matrix", default="bt709")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--note", help="added to the episodes' notes in the indexes")
    p.add_argument("--force", action="store_true", help="make the sheets of an episode already reviewed again")
    p = sub.add_parser("extend")
    p.add_argument("review", help="review directory made by sheets")
    p.add_argument("dirs", nargs="+", help="episode directories of scd_scores.py (analysed with tnet.npz)")
    p.add_argument("--cap", type=int, default=30, help="candidates of TransNetV2 alone shown per episode")
    p.add_argument("--seed", default="1")
    p.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="episodes left out")
    p.add_argument("--rows", type=int, default=15, help="rows per page")
    p.add_argument("--width", type=int, default=320, help="thumbnail width")
    p.add_argument("--matrix", default="bt709")
    p.add_argument("--threads", type=int, default=16)
    p = sub.add_parser("index")
    p.add_argument("review")
    a = ap.parse_args()
    if a.cmd == "sheets":
        cmd_sheets(a)
    elif a.cmd == "extend":
        cmd_extend(a)
    else:
        merge_index(a.review)


if __name__ == "__main__":
    main()
