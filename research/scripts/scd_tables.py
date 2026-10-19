#!/usr/bin/env python3
"""Tables across episodes for scene-detection.md's unlabelled statistics, read from the episode
directories of scd_scores.py (each analysed with TransNetV2) and from the review index of
scd_review.py. Nothing is detected here: sums, shares and rates per hour of what analyse wrote.

  scd_tables.py sources DIR... [--md OUT.md]                     # every source, every threshold
  scd_tables.py compact DIR... [--skip NAME...] [--md OUT.md]    # per source, per hour by kind
  scd_tables.py groups DIR... [--index INDEX.csv] [--md OUT.md]  # candidates by agreement group

DIR is an episode directory of scd_scores.py (stats.json, source.json, pysd.json, tnet.npz,
candidates.csv). A source is shown under the name its stats.json holds (analyse --name, else its
directory's), or under another with --label NAME=LABEL; its kind is scd_scores.py's (KINDS, or
--kind NAME=KIND; any other source is animation); --skip leaves sources out (an original scored
again as its segments stays out of every pooled figure). Rows follow the directories' order.

sources: one row per source, one column per threshold. The analysed range, frames and minutes
(scene-detection.md's Sources table); the candidates, sure and doubtful, by detector (its
Candidates section); scdet per threshold T: detections, bursts and their share, shots under
0.5 s, the median shot, then the PySceneDetect cuts it leaves below T (both detectors / one),
one-frame shots and the share of detections right after a held frame (the bullets of Statistics
and of the summary); PySceneDetect's detectors alone, with their shortest shot; TransNetV2's
alignment on the sure cuts and its own candidates by band (the TransNetV2 and Candidates
sections); TransNetV2 per threshold p, with the sure cuts and the PySceneDetect cuts below
T = 10 it takes; scdet T against TransNetV2 at 0.5 on the candidates; and what each detection
pass took as it ran (wall seconds, minutes per hour of source; pooled over the 1080p sources of
one file, a list of segments being decoded through a pipe). The doc's Cost per detector is not
from these: it was measured apart, back to back on one source.

compact: the tables of Statistics (unlabelled). Per source at a few thresholds (scdet at SHOW_T,
TransNetV2 at SHOW_P): detections / bursts / shots under 0.5 s / the shortest shot in frames /
one-frame shots. Per source at the working thresholds, scdet at sptenc's T and TransNetV2 at
scd_review.py's ROUND_TN, with the sure cuts and the PySceneDetect cuts below T that TransNetV2
takes (the doc's "Per source"). The hours of each kind and, per hour of source by kind, every
detector's detections / bursts / shots under 0.5 s, with for scdet the cuts both PySceneDetect
detectors find below T (the doc's "Per hour, by kind", and the pooled hours of its Sources).
Then, pooled by kind, the sure cuts and the PySceneDetect cuts below T that TransNetV2 takes at
each p (the summary's pooled counts).

groups: the candidates of each kind by agreement group at the working thresholds (the doc's
"Where the detectors agree"): scd_review.py's agreement_group on every row of the directories'
candidates.csv. G5 is counted with its lone scdet hits inside a burst (G5b), which are also
given apart, as are its lone scdet hits outside one. --index adds the rows the review index
holds in each group, joined on (episode, frame).

A burst is a detection 1-3 frames after the previous one (stats.json's gaps 1, 2 and 3).

Needs numpy, and Pillow through scd_review.py; scd_scores.py and scd_review.py next to it.
"""
import argparse
import os
import sys
from collections import defaultdict
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scd_review import (
    ROUND_SCDET,
    ROUND_TN,
    ROUND_TN_LOW,
    agreement_group,
    load_candidates,
)
from scd_scores import KINDS, SPTENC_T, load_json, load_tnet, read_index

KIND_ORDER = ("animation", "live action", "DVD")  # the doc's columns; any other kind after them
SHOW_T, SHOW_P = ("8", "10", "12", "14"), ("0.3", "0.5", "0.7")  # compact's per-source thresholds
GROUPS = (  # scd_review.agreement_group's groups, worded as the doc's table
    ("G1", "G1: TN, S and both P"),
    ("G2", "G2: TN and S or P, not G1"),
    ("G3", "G3: TN alone"),
    ("G4", f"G4: S or P, TN at {ROUND_TN_LOW:g}-{ROUND_TN:g}"),
    ("G5", f"G5: S or P, TN under {ROUND_TN_LOW:g}"),
    ("G5b", "... of which scdet alone in a burst"),
    ("G5a", "... of which scdet alone outside a burst"),
    ("G6", "G6: none of them"),
)


def load_sources(a):
    """The episodes of the directories given, in their order: name (stats.json's), label, kind,
    directory and stats."""
    labels = dict(x.split("=", 1) for x in a.label or [])
    kinds = {**KINDS, **dict(x.split("=", 1) for x in a.kind or [])}
    srcs = []
    for d in a.dirs:
        st = load_json(os.path.join(d, "stats.json"))
        name = st["name"]
        if name in a.skip:
            continue
        if "tnet" not in st or "pysd_detectors" not in st:
            sys.exit(f"{d}: analysed without PySceneDetect or TransNetV2 (pysd.json, tnet.npz)")
        srcs.append({"name": name, "label": labels.get(name, name), "kind": kinds.get(name, "animation"),
                     "dir": d, "st": st})
    if not srcs:
        sys.exit("no episode left")
    return srcs


def kinds_of(srcs):
    """The kinds present, the doc's three first."""
    present = {s["kind"] for s in srcs}
    return [k for k in KIND_ORDER if k in present] + sorted(present - set(KIND_ORDER))


def bursts(x):
    """Detections 1-3 frames after the previous one."""
    return x["gaps"]["1"] + x["gaps"]["2"] + x["gaps"]["3"]


def hours(frames, fps):
    return frames / fps / 3600.0


def section(out, title, head, align, rows):
    """Add a titled Markdown table to out; align: one letter per column, l or r."""
    out += ([""] if out else []) + [f"## {title}", "", "| " + " | ".join(head) + " |",
                                    "|" + "|".join("---:" if c == "r" else "---" for c in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def emit(out, a):
    md = "\n".join(out) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


def cmd_sources(a):
    srcs = load_sources(a)
    ts, ps = list(srcs[0]["st"]["per_threshold"]), list(srcs[0]["st"]["tnet"]["per_p"])
    t_head, p_head = [f"T={t}" for t in ts], [f"p={p}" for p in ps]
    out = []

    def per(key, cell):
        """One row per source: its label, then cell(stats, x) for each x of key(stats)."""
        return [[s["label"]] + [cell(s["st"], x) for x in key(s["st"])] for s in srcs]

    def share(x):
        return bursts(x) / x["detections"] if x["detections"] else float("nan")

    rows = []
    for s in srcs:
        st, src = s["st"], load_json(os.path.join(s["dir"], "source.json"))
        rows.append([s["label"], s["kind"], st["size"], st["fps"], st["frames_scored"],
                     f"{st['range_tc'][0]}–{st['range_tc'][1]} ({st['range'][0]}–{st['range'][1]})", st["frames"],
                     f"{st['duration_s'] / 60:.1f}", st["files"], src.get("color_space"), src.get("field_order")])
    section(out, "Sources", ["name", "kind", "size", "fps", "frames scored", "analysed range", "frames", "min",
                             "files", "color_space", "field"], "llllrlrrrll", rows)
    rows = []
    for s in srcs:
        st, c = s["st"], s["st"]["candidates_by_source"]
        rows.append([s["label"], st["candidates"], st["sure"], st["doubtful"], c["scdet_only"], c["scdet_and_pysd"],
                     c["pysd_only"], c["join_only"], st["pysd_cuts"]["adaptive"], st["pysd_cuts"]["content"]])
    section(out, "Candidates", ["name", "candidates", "sure", "doubtful", "scdet only", "scdet+pysd", "pysd only",
                                "joins", "pysd adaptive", "content"], "lrrrrrrrrr", rows)
    section(out, "scdet per threshold: detections; burst (gaps 1+2+3) count/share; shots <0.5 s; median shot s",
            ["name"] + t_head, "l" * (len(ts) + 1),
            per(lambda st: [st["per_threshold"][t] for t in ts],
                lambda st, x: f"{x['detections']}; {bursts(x)} ({share(x):.0%}); {x['under_0.5s']}; {x['median_s']}"))
    section(out, "scdet: possible misses (PySD both / one below T); one-frame shots; after-hold share",
            ["name"] + t_head, "l" * (len(ts) + 1),
            per(lambda st: [st["per_threshold"][t] for t in ts],
                lambda st, x: f"{x['possible_misses_both']}/{x['possible_misses_one']}; 1f {x['one_frame']}; "
                              f"h {x['after_hold']}"))
    section(out, "PySD detectors alone: detections; burst; shots <0.5/<1/<2 s; median; min frames",
            ["name", "adaptive", "content"], "lll",
            per(lambda st: [st["pysd_detectors"][k] for k in ("adaptive", "content")],
                lambda st, x: f"{x['detections']}; {bursts(x)}; {x['under_0.5s']}/{x['under_1s']}/{x['under_2s']}; "
                              f"{x['median_s']}; min {x['min_frames']}"))
    rows = []
    for s in srcs:
        t = s["st"]["tnet"]
        rows.append([s["label"], t["offset_on_sure"], t["offset"], t["offset_share"],
                     f"{t['sure_with_peak']}/{t['sure_without_peak']}", t["tnet_only"],
                     " / ".join(str(v) for v in t["tnet_only_by_band"].values()), t["tnet_only_2_3_from_old"]])
    section(out, "TransNetV2 alignment and own candidates",
            ["name", "offsets on sure", "offset", "share", "sure with/without peak", "TN-only", "by band",
             "2-3 from old"], "llrrlrlr", rows)
    section(out, "TransNetV2 per p: detections; burst count; shots <0.5 s; median; sure detected/sure",
            ["name"] + p_head, "l" * (len(ps) + 1),
            per(lambda st: [st["tnet"]["per_p"][p] for p in ps],
                lambda st, x: f"{x['detections']}; {bursts(x)}; {x['under_0.5s']}; {x['median_s']}; "
                              f"{x['sure_detected']}/{st['sure']}"))
    t10 = f"T={SPTENC_T:g}"
    rows = []
    for s in srcs:
        t = s["st"]["tnet"]
        m = t["possible_misses_t10"]
        rows.append([s["label"], f"{m['both']}/{m['one']}"] +
                    [f"{t['per_p'][p]['possible_misses_t10_detected']['both']}/"
                     f"{t['per_p'][p]['possible_misses_t10_detected']['one']}" for p in ps])
    section(out, f"TransNetV2 per p: PySD possible misses at {t10} detected (both/one of total both/one)",
            ["name", f"misses {t10} both/one"] + p_head, "l" * (len(ps) + 2), rows)
    rows = []
    for s in srcs:
        ag = s["st"]["tnet"]["per_p"][f"{ROUND_TN:g}"]["agreement"]
        rows.append([s["label"]] + [f"{ag[t]['both']}/{ag[t]['scdet_only']}/{ag[t]['tnet_only']}" if t in ag else "-"
                                    for t in ts])
    section(out, f"Agreement scdet T x TN p={ROUND_TN:g} on candidates: both / scdet only / TN only",
            ["name"] + t_head, "l" * (len(ts) + 1), rows)
    # what each pass took as it ran: its own log of wall seconds, over the frames it decoded
    pool = {k: [0, 0] for k in ("scdet", "pysd", "tn", "tn decode", "tn inference")}
    rows = []
    for s in srcs:
        st, d = s["st"], s["dir"]
        src, py, tn = load_json(os.path.join(d, "source.json")), load_json(os.path.join(d, "pysd.json")), load_tnet(d)
        if tn is None:
            sys.exit(f"{d}: no tnet.npz")
        fps = float(Fraction(st["fps"]))
        sc_f, sc_s = src["frames"], src["score_seconds"]
        py_f, py_s = py["frames_read"], py["seconds"]
        tn_f, td, ti = tn["frames"], float(tn["decode_seconds"]), float(tn["inference_seconds"])
        sc_h, py_h, tn_h = hours(sc_f, fps), hours(py_f, fps), hours(tn_f, fps)
        rows.append([s["label"], st["size"], sc_f, sc_s, f"{sc_s / 60 / sc_h:.2f}", py_f, py_s,
                     f"{py_s / 60 / py_h:.2f}", tn_f, td, ti,
                     f"{(td + ti) / 60 / tn_h:.2f} ({td / 60 / tn_h:.2f}+{ti / 60 / tn_h:.2f})",
                     f"{tn['meta'].get('decode_cpu_seconds')}/{tn['meta'].get('inference_cpu_seconds')}"])
        if st["size"] == "1920x1080" and st["files"] == 1:
            for k, sec, h in (("scdet", sc_s, sc_h), ("pysd", py_s, py_h), ("tn", td + ti, tn_h),
                              ("tn decode", td, tn_h), ("tn inference", ti, tn_h)):
                pool[k][0] += sec
                pool[k][1] += h
    section(out, "Costs (wall seconds of each pass as it ran; minutes per hour of source)",
            ["name", "size", "scdet frames", "scdet s", "scdet min/h", "pysd frames", "pysd s", "pysd min/h",
             "tn frames", "tn dec s", "tn inf s", "tn min/h (dec+inf)", "tn cpu s dec/inf"], "llrrrrrrrrrll", rows)
    out += ["", "Pooled over the 1080p sources of one file (min per hour): " +
            ", ".join(f"{k} {v[0] / 60 / v[1]:.2f} ({v[1]:.2f} h)" for k, v in pool.items() if v[1])]
    emit(out, a)


def cmd_compact(a):
    srcs = load_sources(a)
    ts, ps = list(srcs[0]["st"]["per_threshold"]), list(srcs[0]["st"]["tnet"]["per_p"])
    wt, wp = f"{SPTENC_T:g}", f"{ROUND_TN:g}"  # the working thresholds
    out = []

    def shots(x):
        return f"{x['detections']} / {bursts(x)} / {x['under_0.5s']}"

    def shortest(x):
        return f"{shots(x)} / {x['min_frames']} / {x['one_frame']}"

    what = "detections / bursts / shots under 0.5 s / shortest shot (frames) / one-frame shots"
    section(out, f"scdet per source: {what}", ["source"] + [f"T = {k}" for k in SHOW_T], "l" * (len(SHOW_T) + 1),
            [[s["label"]] + [shortest(s["st"]["per_threshold"][k]) for k in SHOW_T] for s in srcs])
    section(out, f"TransNetV2 per source: {what}", ["source"] + [f"p = {k}" for k in SHOW_P],
            "l" * (len(SHOW_P) + 1),
            [[s["label"]] + [shortest(s["st"]["tnet"]["per_p"][k]) for k in SHOW_P] for s in srcs])
    rows = []
    for s in srcs:
        st = s["st"]
        tn, py = st["tnet"]["per_p"][wp], st["pysd_detectors"]
        rows.append([s["label"], shots(st["per_threshold"][wt]),
                     f"{py['adaptive']['detections']}, {py['content']['detections']}", shots(tn),
                     f"{tn['sure_detected']} of {st['sure']}",
                     f"{tn['possible_misses_t10_detected']['both']} of {st['tnet']['possible_misses_t10']['both']}"])
    section(out, "Per source at the working thresholds: detections / bursts / shots under 0.5 s",
            ["source", f"scdet T = {wt}", "PySceneDetect adaptive, content", f"TransNetV2 p = {wp}",
             "sure cuts TransNetV2 takes", f"PySceneDetect cuts below T = {wt}: TransNetV2 takes"], "llllll", rows)

    kinds = kinds_of(srcs)
    of = {k: [s["st"] for s in srcs if s["kind"] == k] for k in kinds}
    hrs = {k: sum(st["duration_s"] for st in of[k]) / 3600 for k in kinds}
    dets = [(f"scdet T = {x}", lambda st, x=x: st["per_threshold"][x], True) for x in ts]
    dets += [(f"PySceneDetect {x}", lambda st, x=x: st["pysd_detectors"][x], False) for x in ("adaptive", "content")]
    dets += [(f"TransNetV2 p = {x}", lambda st, x=x: st["tnet"]["per_p"][x], False) for x in ps]
    rows = []
    for name, get, below in dets:
        cells = []
        for k in kinds:
            xs = [get(st) for st in of[k]]
            v = [sum(x["detections"] for x in xs), sum(bursts(x) for x in xs), sum(x["under_0.5s"] for x in xs)]
            if below:
                v.append(sum(x["possible_misses_both"] for x in xs))
            cells.append(" / ".join(f"{x / hrs[k]:.0f}" for x in v))
        rows.append([name] + cells)
    section(out, "Per hour of source, by kind: detections / bursts / shots under 0.5 s, and for scdet the cuts both "
            "PySceneDetect detectors find below T",
            ["detector"] + [f"{k} ({hrs[k]:.2f} h, {len(of[k])} source{'s' if len(of[k]) > 1 else ''})"
                            for k in kinds], "l" * (len(kinds) + 1), rows)
    rows = []
    for k in kinds:
        tn = [st["tnet"] for st in of[k]]
        rows.append([k, sum(st["sure"] for st in of[k]), sum(x["possible_misses_t10"]["both"] for x in tn)] +
                    [f"{sum(x['per_p'][p]['sure_detected'] for x in tn)} / "
                     f"{sum(x['per_p'][p]['possible_misses_t10_detected']['both'] for x in tn)}" for p in ps])
    section(out, f"Pooled by kind, TransNetV2 >= p takes: sure cuts / PySceneDetect cuts below T = {wt} (both "
            "detectors)", ["kind", "sure cuts", f"PySceneDetect cuts below T = {wt}"] + [f"p = {p}" for p in ps],
            "lrr" + "l" * len(ps), rows)
    emit(out, a)


def cmd_groups(a):
    srcs = load_sources(a)
    kinds = kinds_of(srcs)
    held = set()  # the review's rows of these episodes
    if a.index:
        names = {s["name"] for s in srcs}
        held = {(r["episode"], int(float(r["frame"]))) for r in read_index(a.index) if r["episode"] in names}
    size, rows_in, found = defaultdict(int), defaultdict(int), 0
    for s in srcs:
        for c in load_candidates(s["dir"]):
            g = agreement_group(c)
            keys = ["G5", "G5b"] if g == "G5b" else [g]
            if g == "G5" and c["scdet"] >= ROUND_SCDET and c["adaptive"] == "" and c["content"] == "":
                keys.append("G5a")  # scdet alone, outside a burst
            row = (s["name"], c["frame"]) in held
            found += row
            for k in keys:
                size[(k, s["kind"])] += 1
                rows_in[(k, s["kind"])] += row
    out, table = [], []
    for g, title in GROUPS:
        n = [size[(g, k)] for k in kinds]
        r = [rows_in[(g, k)] for k in kinds]
        table.append([title] + n + [sum(n)] + (r + [sum(r)] if a.index else []))
    section(out, f"Candidates by agreement group and kind (TN = TransNetV2 >= {ROUND_TN:g}, S = scdet >= "
            f"{ROUND_SCDET:g}, P = a detection of either PySceneDetect detector)" +
            (", and the review's rows among them" if a.index else ""),
            ["group"] + kinds + ["all"] + ([f"review rows: {kinds[0]}"] + kinds[1:] + ["all"] if a.index else []),
            "l" + "r" * (len(kinds) + 1) * (2 if a.index else 1), table)
    if a.index:
        out += ["", f"{len(held)} review rows of these episodes, {len(held) - found} of them on no candidate."]
    emit(out, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sources", "compact", "groups"):
        p = sub.add_parser(name)
        p.add_argument("dirs", nargs="+", help="episode directories of scd_scores.py (analysed with tnet.npz)")
        p.add_argument("--label", action="append", metavar="NAME=LABEL",
                       help="show an episode under this label (default: its name)")
        p.add_argument("--kind", action="append", metavar="NAME=KIND",
                       help="pool an episode with this kind (default: KINDS, else animation)")
        p.add_argument("--skip", nargs="*", default=[], metavar="NAME", help="episodes left out")
        p.add_argument("--md", help="also write the tables to this file")
    p.add_argument("--index", help="review index CSV (scd_review.py): count its rows in each group")
    a = ap.parse_args()
    {"sources": cmd_sources, "compact": cmd_compact, "groups": cmd_groups}[a.cmd](a)


if __name__ == "__main__":
    main()
