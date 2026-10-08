#!/usr/bin/env python3
"""S16 (model conversation, 2026-10-08): the 4K floor's effect, read only on the pool's outputs (writes only into
--out, default $VAL_STATE/s16/floor; the pool's pairing cache is copied there first, never written).

  $METRICS_PY $VAL_GLUE/floor_report.py 4k|4ksh|1080 [--out DIR]

4k: ms_sum4k.py's labels (the 7B's slice B), 4ksh: ms_sum4ksh.py's (the sharp's), 1080: ms_sum.py's (applied there
too since S23, 2026-10-08). Per label, data = the script's label_data() as it was (before) and with
ms_floor.apply() (after): every printed verdict cell whose strict or calibrated verdict changes (shot or clip, variant,
metric; guard cells and the reported ones), the guards per kind that change, and the VALIDATION.md row before and
after (cells past the strict / calibrated line, guards past the calibrated line with their kinds, the worst multiple,
dB to the reference s42 mean (worst), PSNR-Y and VMAF split means). 1080: also the band cells (clip, variant, metric,
band source) whose seed spread is under the floor. Writes OUT/<mode>.md and OUT/<mode>-changes.tsv.
"""

import argparse
import copy
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glue_env import env  # noqa: E402  the paths: glue.env (models/gpu/validation/glue.env.example)

SUM = env("VAL_STATE") + "/sum"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("4k", "4ksh", "1080"))
    ap.add_argument("--out", default=env("VAL_STATE") + "/s16/floor")
    a = ap.parse_args()
    if a.mode == "4k":
        import ms_sum4k as M

        S, ld, labels, gtl = M.S, M.S.label_data, M.labels_present4(), "gt_lap4k.json"
    elif a.mode == "4ksh":
        import ms_sum4ksh as M

        S, ld, labels, gtl = M.S, M.label_data, M.labels_present(), "gt_lap4ksh.json"
    else:
        import ms_sum as S

        ld, labels, gtl = S.label_data, S.labels_present(), "gt_lap.json"
    import ms_floor as F

    F.install(S)
    work = os.path.join(a.out, f"work-{a.mode}")
    os.makedirs(work, exist_ok=True)
    if os.path.isdir(f"{SUM}/cache"):
        shutil.copytree(f"{SUM}/cache", os.path.join(work, "cache"), dirs_exist_ok=True)
    if os.path.exists(f"{SUM}/{gtl}"):
        shutil.copy(f"{SUM}/{gtl}", os.path.join(work, gtl))
    S.X["out"] = work
    gl = S.GtLap(os.path.join(work, gtl))
    GK = S.GUARD_KEYS
    SHOWN = [k for k, _ in S.COLS] + ["lap"]
    NAME = dict(S.COLS, lap="detail (Laplacian)", band="finest band")

    def mtxt(tag, mean, band):
        if not tag:
            return "within"
        return f"{tag} {S.mult_txt(S.mult_of(mean, band))}"

    def cells(data):
        out = {}
        for c in data:
            for short in ("none", "split"):
                e = data[c].get(short)
                if not e:
                    continue
                P = e["pairs"]
                dk = "dists_all" if "dists_all" in P else "dists"
                for k in SHOWN:
                    if k not in P:
                        continue
                    p = P[k]
                    out[(c, short, k)] = {
                        "guard": k in GK or k == dk,
                        "s": p["tag"],
                        "c": S.verdict(p, "tagc"),
                        "txt": mtxt(p["tag"], p["mean"], p["band"]),
                        "ctxt": mtxt(S.verdict(p, "tagc"), p["mean"], p["band"]),
                        "raw": p.get("band_raw", p["band"]),
                        "band": float(p["band"]),
                        "mean": p["mean"],
                    }
                bl, bref = (e.get("band") or (None,))[0], e.get("band7ref")
                if bl is not None and bref is not None and len(e.get("band7") or {}) > 1:
                    t = e["btag"]
                    out[(c, short, "band")] = {
                        "guard": True,
                        "s": t,
                        "c": e.get("btagc", ""),
                        "txt": f"{t} {S.mult_txt(e.get('bmult'))}" if t else "within",
                        "ctxt": f"{e['btagc']} {S.mult_txt(e.get('bmult'))}"
                        if e.get("btagc")
                        else "within",
                        "raw": e.get("bspread_raw"),
                        "mean": bl - bref,
                    }
        return out

    def vrow(data):
        st = ca = 0
        for c in data:
            for short in ("none", "split"):
                e = data[c].get(short)
                if not e:
                    continue
                P = e["pairs"]
                dk = "dists_all" if "dists_all" in P else "dists"
                for k in GK + (dk,):
                    if k in P:
                        st += P[k]["tag"] in ("W", "↑", "↓")
                        ca += S.verdict(P[k], "tagc") in ("W", "↑", "↓")
                st += bool(e.get("btag"))
                ca += bool(e.get("btagc"))
        gs = []
        for g, title in S.GUARDS:
            ks = {}
            for short in ("none", "split"):
                kc = S.kind_cells(data, short, "tagc")
                for kd in S.KINDS:
                    if kc[g][kd].startswith("FAIL"):
                        arrows = "".join(sorted({x for x in kc[g][kd] if x in "↑↓"}))
                        ks.setdefault(kd, []).append(short + (f" {arrows}" if arrows else ""))
            if ks:
                gs.append(
                    f"{title.split(':')[0]}: "
                    + ", ".join(f"{kd} ({'; '.join(v)})" for kd, v in ks.items())
                )
        wm = S.worst_mult(data)
        dbs = [data[c]["db"] for c in data if data[c].get("db") is not None]
        sp = {
            k: [
                data[c]["split"]["pairs"][k]["mean"]
                for c in data
                if "split" in data[c] and k in data[c]["split"]["pairs"]
            ]
            for k in ("psnr_y", "vmaf")
        }
        return {
            "n": len(data),
            "strict": st,
            "calib": ca,
            "guards": "; ".join(gs) or "none",
            "worst": S.mult_txt(wm[0]) + (f" ({wm[1]})" if wm[1] else ""),
            "db": f"{S.f2(np.mean(dbs))} ({S.f2(min(dbs))})" if dbs else "–",
            "fid": ", ".join(
                S.fmt(k, float(np.mean(sp[k]))) if sp[k] else "–" for k in ("psnr_y", "vmaf")
            ),
        }

    lines = [
        f"# The 4K floor's effect: {a.mode} ({'applied at 1080p too since S23' if a.mode == '1080' else 'applied'})",
        "",
        f"Generated {S.stamp()} by floor_report.py. Floors: {F.FLOOR_LIST}.",
        "",
    ]
    tsv = [
        [
            "label",
            "shot",
            "variant",
            "metric",
            "guard",
            "strict before",
            "strict after",
            "calibrated before",
            "calibrated after",
            "difference",
            "seed spread",
            "floor",
        ]
    ]
    vrows = []
    under = {}  # 1080: (band source, clip, variant, metric) -> raw spread under the floor
    nunder_cells = 0
    for L in labels:
        raw = ld(L, gl)
        flo = copy.deepcopy(raw)
        n = F.apply(S, flo)
        c0, c1 = cells(raw), cells(flo)
        ch = []
        for key in c0:
            x, y = c0[key], c1.get(key)
            if y is None:
                continue
            if a.mode == "1080" and y.get("raw") is not None and key[2] != "band":
                fl = F.floor_of(key[2], raw[key[0]][key[1]].get("gtlap"))
                if fl is not None and y["raw"] < fl:
                    nunder_cells += 1
                    src = (raw[key[0]][key[1]].get("bandsrc") or {}).get(
                        "fr" if key[2] == "dists_all" else "scores", "7B"
                    )
                    under[(src, key[0], key[1], key[2])] = y["raw"]
            if a.mode == "1080" and key[2] == "band" and x.get("raw") is None:
                e = raw[key[0]][key[1]]
                b7 = [v for v in (e.get("band7") or {}).values() if v is not None]
                if len(b7) > 1 and max(b7) - min(b7) < F.BAND_RATIO:
                    nunder_cells += 1
                    src = (e.get("bandsrc") or {}).get("bands", "7B")
                    under[(src, key[0], key[1], "band")] = max(b7) - min(b7)
            if x["s"] != y["s"] or x["c"] != y["c"]:
                ch.append((key, x, y))
                fl = (
                    F.BAND_RATIO
                    if key[2] == "band"
                    else F.floor_of(key[2], raw[key[0]][key[1]].get("gtlap"))
                )
                tsv.append(
                    [
                        L,
                        key[0],
                        key[1],
                        NAME.get(key[2], key[2]),
                        "guard" if x["guard"] else "reported",
                        x["txt"],
                        y["txt"],
                        x["ctxt"],
                        y["ctxt"],
                        f"{y['mean']:+.6g}",
                        "" if y.get("raw") is None else f"{y['raw']:.6g}",
                        "" if fl is None else f"{fl:.6g}",
                    ]
                )
        gch = []
        for mode in ("tag", "tagc"):
            for short in ("none", "split"):
                k0, k1 = S.kind_cells(raw, short, mode), S.kind_cells(flo, short, mode)
                for g, title in S.GUARDS:
                    for kd in S.KINDS:
                        f0, f1 = k0[g][kd].startswith("FAIL"), k1[g][kd].startswith("FAIL")
                        if f0 != f1:
                            gch.append(
                                f"{'strict' if mode == 'tag' else 'calibrated'}, {short}, {title.split(':')[0]}, "
                                f"{kd}: {'FAIL' if f0 else 'PASS'} → {'FAIL' if f1 else 'PASS'}"
                            )
        r0, r1 = vrow(raw), vrow(flo)
        vrows.append((L, r0, r1))
        lines += [
            f"## {L}: {len(ch)} verdict cells change ({sum(1 for _, x, _ in ch if x['guard'])} guard cells); "
            f"floor raised {n['pairs']} pair spreads, {n['bands']} band-energy spreads",
            "",
        ]
        if ch:
            rows = [
                [
                    c,
                    s,
                    NAME.get(k, k),
                    "guard" if x["guard"] else "reported",
                    f"{x['txt']} → {y['txt']}",
                    f"{x['ctxt']} → {y['ctxt']}",
                    f"{y['mean']:+.6g}",
                    "–" if y.get("raw") is None else f"{y['raw']:.3g}",
                ]
                for (c, s, k), x, y in ch
            ]
            lines += S.table(
                [
                    "Shot",
                    "Var",
                    "Metric",
                    "Kind",
                    "Strict before → after",
                    "Calibrated before → after",
                    "Difference",
                    "Seed spread",
                ],
                rows,
            ) + [""]
        if gch:
            lines += ["Guards per kind that change: " + "; ".join(gch) + ".", ""]
    lines += ["## VALIDATION.md rows, before → after", ""]
    rows = []
    for L, r0, r1 in vrows:
        rows.append(
            [
                L,
                str(r1["n"]),
                f"{r0['strict']} / {r0['calib']} → {r1['strict']} / {r1['calib']}",
                f"{r0['guards']} → {r1['guards']}"
                if r0["guards"] != r1["guards"]
                else r1["guards"],
                f"{r0['worst']} → {r1['worst']}",
                r1["db"],
                r1["fid"],
            ]
        )
    lines += S.table(
        [
            "Label",
            "Shots",
            "Cells past the strict / calibrated line",
            "Guards past the calibrated line (kinds)",
            "Worst multiple of the band",
            "dB to its reference s42: mean (worst)",
            "PSNR-Y, VMAF (split)",
        ],
        rows,
    )
    if a.mode == "1080":
        bym = {}
        for src, c, s, k in under:
            bym.setdefault((src, k), []).append(f"{c} {s}")
        lines += [
            "",
            f"## 1080p band cells under the floor: {len(under)} distinct (band source, clip, variant, metric); "
            f"{nunder_cells} label cells use one",
            "",
        ]
        lines += S.table(
            ["Band", "Metric", "Clip-variants"],
            [
                [src, NAME.get(k, k), f"{len(v)}: " + ", ".join(sorted(v))]
                for (src, k), v in sorted(bym.items())
            ],
        )
    os.makedirs(a.out, exist_ok=True)
    S.write(os.path.join(a.out, f"{a.mode}.md"), lines)
    with open(os.path.join(a.out, f"{a.mode}-changes.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join("\t".join(r) for r in tsv) + "\n")
    print(
        f"floor_report {a.mode}: {len(labels)} labels, {len(tsv) - 1} changed cells; {os.path.join(a.out, a.mode)}.md"
    )


if __name__ == "__main__":
    main()
