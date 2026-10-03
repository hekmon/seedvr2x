#!/usr/bin/env python3
"""Frame-exact access into long-GOP sources: ffmpeg's input seek against decode-and-count.

seedvr2x reads a shot's frames again (resume, colour correction) from any source a user gives
it: Blu-ray, web or DVD files with B-frames and open GOPs. Is ffmpeg's accurate input seek
(-ss before -i) frame-exact on such files, checked by md5 against a decode from the start, and
how fast is each way? Python standard library only; it drives ffmpeg and ffprobe (CPU only).

One output directory per source file (OUT). Steps, in order:

  seek_test.py info    FILE OUT   stream facts, packet scan (decode order: pts, dts, flags),
                                  GOP analysis: keyframe interval, B-frame runs, reorder delay,
                                  leading pictures (displayed before their keyframe, decoded
                                  after it: the open-GOP signature), timestamp regularity
  seek_test.py nal     FILE OUT   picture headers of a few GOPs (trace_headers bsf): H.264 IDR
                                  vs non-IDR I + recovery point SEI, HEVC IDR/CRA/RASL/RADL,
                                  MPEG-2 closed_gop/broken_link
  seek_test.py ref     FILE OUT   the reference: one full decode from the start, framemd5 with
                                  the source pts (-copyts, encoder time base = the stream's)
  seek_test.py same    OUT A B    compare two reference decodes (determinism)
  seek_test.py targets OUT        target frames: the first and last 3, keyframes with the
                                  frames around them, open-GOP leading frames, random ones
  seek_test.py seek    FILE OUT   every method on every target: K frames' md5 against the
                                  reference at n..n+K-1 (exact / shifted by d / corrupted), time
                                  (median of R runs) next to the 1-minute load average
  seek_test.py report  OUT...     tables over several sources
  seek_test.py keycheck FILE OUT  is each flagged keyframe a clean random access point? decode
                                  from exactly that keyframe (stream copy piped to a decoder)
                                  and compare every frame with the reference
  seek_test.py paramsets FILE OUT in-band SPS/PPS (VPS) over the whole stream against the
                                  container's extradata: keyframes a decoder cannot start from
  seek_test.py hashcost           throughput of per-frame content hashes (first-pass cost)
  seek_test.py synth   SRC OUT    short test files with open GOPs (x264, x265, MPEG-2) in
                                  MKV/MP4/TS/PS, made from an excerpt of SRC (--remux: a
                                  stream copy of another source into M2TS)

Methods (decoder threads --threads; K frames, -frames:v K, -fps_mode passthrough):
  a_naive       -ss n/fps: n times the nominal frame duration (relative to the file start)
  b_pts         -ss (pts_n - start_time): frame n's exact pts from the reference, relative
  b_copyts      the same with -copyts (the output keeps the source timestamps)
  b_abs         -seek_timestamp 1 -ss pts_n (absolute)          } only when the file's
  b_abs_copyts  the same with -copyts                           } start_time is not 0
  b_abs_wrong   -ss pts_n without -seek_timestamp (the pitfall) }
  c_1s          -copyts -ss (pts_n - start_time - 1 s), then select='gte(pts,PTS_n)': seek
                early, keep the frames from the target pts on
  c_gop         the same from one GOP back: -ss at the keyframe before the target's own
                keyframe (both known from the packet scan)
  e_pipe        b_pts read through a pipe as rawvideo (how seedvr2x reads), md5 taken here
  d_count       decode from the start and count: select='gte(n,N)' (exact by construction;
                run on a few targets only, its time grows with N)

Usage on one file:
  python3 seek_test.py info FILE OUT && python3 seek_test.py nal FILE OUT &&
  python3 seek_test.py ref FILE OUT && python3 seek_test.py targets OUT --count 40 &&
  python3 seek_test.py seek FILE OUT --repeats 3

Environment: SEEK_FFMPEG, SEEK_FFPROBE (default: ffmpeg / ffprobe on PATH).
"""
import argparse
import bisect
import collections
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from fractions import Fraction

FFMPEG = os.environ.get("SEEK_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("SEEK_FFPROBE", "ffprobe")
VMAP = "0:V:0"  # the first video stream that is not a cover picture
VSEL = "V:0"
ALL_METHODS = ("a_naive", "b_pts", "b_copyts", "b_abs", "b_abs_copyts", "b_abs_wrong",
               "c_1s", "c_gop", "e_pipe", "d_count")
DEFAULT_METHODS = ("a_naive", "b_pts", "b_copyts", "c_1s", "c_gop", "e_pipe", "d_count")
START_METHODS = ("b_abs", "b_abs_copyts", "b_abs_wrong")  # added when start_time != 0


# ----------------------------------------------------------------------------- helpers

def jload(path):
    with open(path) as f:
        return json.load(f)


def jsave(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def num(s):
    return None if s in (None, "", "N/A") else int(s)


def tstr(t):
    """Seconds (Fraction) as a decimal string rounded to the microsecond, as -ss parses it."""
    q = round(t * 1000000)
    sign = "-" if q < 0 else ""
    q = abs(q)
    return f"{sign}{q // 1000000}.{q % 1000000:06d}"


def load1():
    return round(os.getloadavg()[0], 2)


def med(xs):
    return statistics.median(xs) if xs else None


def stats3(xs):
    return {"min": min(xs), "median": med(xs), "max": max(xs)} if xs else None


def p(path, *names):
    return os.path.join(path, *names)


def ffversion():
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-version"], capture_output=True,
                             text=True).stdout
        return out.splitlines()[0]
    except OSError:
        return None


def parse_framemd5(text):
    """framemd5 (format 2) -> (time base, pts list, md5 list) of stream 0, in output order."""
    tb, pts, md5 = None, [], []
    for line in text.splitlines():
        if line.startswith("#tb 0:"):
            tb = Fraction(line.split(":", 1)[1].strip())
        elif line and line[0] != "#":
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 6 and f[0] == "0":
                pts.append(int(f[2]))
                md5.append(f[5])
    return tb, pts, md5


def read_ref(out, name="ref"):
    with open(p(out, name + ".framemd5")) as f:
        return parse_framemd5(f.read())


def frame_bytes(pix_fmt, w, h):
    """Bytes of one rawvideo frame (planar YUV/gray, or NV12/P010-like semi-planar)."""
    bps = 2 if re.search(r"(9|10|12|14|16)(le|be)$", pix_fmt) or pix_fmt.startswith("p0") \
        or pix_fmt.startswith("p2") or pix_fmt.startswith("p4") else 1
    if pix_fmt.startswith("gray"):
        return w * h * bps
    sub = {"420": (1, 1), "422": (1, 0), "444": (0, 0), "440": (0, 1), "411": (2, 0),
           "410": (2, 2), "nv12": (1, 1), "nv21": (1, 1), "p010": (1, 1), "p016": (1, 1),
           "nv16": (1, 0), "p210": (1, 0), "nv24": (0, 0), "p410": (0, 0)}
    for key, (sx, sy) in sub.items():
        if key in pix_fmt:
            cw, ch = -(-w >> sx), -(-h >> sy)
            return (w * h + 2 * cw * ch) * bps
    raise SystemExit(f"unsupported pix_fmt for the pipe method: {pix_fmt}")


# ----------------------------------------------------------------------------- info

def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", VSEL, "-show_streams",
                          "-show_format", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s, f = j["streams"][0], j["format"]
    keep = ("index", "codec_name", "profile", "level", "width", "height", "pix_fmt",
            "has_b_frames", "refs", "field_order", "time_base", "r_frame_rate",
            "avg_frame_rate", "start_pts", "start_time", "nb_frames", "duration",
            "color_range", "color_space", "color_transfer", "color_primaries")
    st = {k: s[k] for k in keep if k in s}
    fm = {k: f[k] for k in ("format_name", "start_time", "duration", "size", "bit_rate",
                            "nb_streams") if k in f}
    return st, fm


def scan_packets(path, csv_path):
    cmd = [FFPROBE, "-v", "error", "-select_streams", VSEL, "-show_entries",
           "packet=pts,dts,duration,size,pos,flags", "-of", "compact=p=0", path]
    pk = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc, \
            open(csv_path, "w") as w:
        w.write("pts,dts,duration,size,pos,flags\n")
        for line in proc.stdout:
            d = dict(kv.split("=", 1) for kv in line.rstrip("\n").split("|") if "=" in kv)
            if not d:
                continue
            row = (num(d.get("pts")), num(d.get("dts")), num(d.get("duration")),
                   num(d.get("size")), num(d.get("pos")), d.get("flags", ""))
            pk.append(row)
            w.write(",".join("" if v is None else str(v) for v in row) + "\n")
    if proc.returncode:
        raise SystemExit(f"ffprobe failed ({proc.returncode})")
    return pk


def read_packets(csv_path):
    pk = []
    with open(csv_path) as f:
        next(f)
        for line in f:
            a = line.rstrip("\n").split(",")
            pk.append((num(a[0]), num(a[1]), num(a[2]), num(a[3]), num(a[4]), a[5]))
    return pk


def analyse(pk, tb, fd):
    """GOP structure from the packets, in decode order (pts, dts, duration, size, pos, flags)."""
    n = len(pk)
    keys = [i for i, q in enumerate(pk) if "K" in q[5]]
    with_pts = [(q[0], i) for i, q in enumerate(pk) if q[0] is not None]
    order = sorted(with_pts)
    rank = {i: r for r, (_, i) in enumerate(order)}
    # reorder delay: how many packets a frame is decoded after its display position
    delay = max((i - rank[i] for _, i in with_pts), default=0)
    # anchors: displayed after everything decoded before them (I/P, the top of a B pyramid);
    # the frames between two anchors in display order are B-frames
    anchors, top = [], None
    for i, q in enumerate(pk):
        if q[0] is not None and (top is None or q[0] > top):
            anchors.append(rank[i])
            top = q[0]
    anchors.sort()
    runs = collections.Counter(b - a - 1 for a, b in zip(anchors, anchors[1:]))
    key_ranks = sorted(rank[i] for i in keys if i in rank)
    kint = [b - a for a, b in zip(key_ranks, key_ranks[1:])]
    # leading pictures: decoded after a keyframe (before the next one), displayed before it
    leading = []
    for j, k in enumerate(keys):
        kp = pk[k][0]
        if kp is None:
            continue
        end = keys[j + 1] if j + 1 < len(keys) else n
        lp = [pk[i][0] for i in range(k + 1, end) if pk[i][0] is not None and pk[i][0] < kp]
        if lp:
            leading.append([kp, sorted(lp)])
    # timestamps: deviation from a regular grid at the nominal frame rate
    ps = [x for x, _ in order]
    nom = fd / tb
    diffs = collections.Counter(b - a for a, b in zip(ps, ps[1:]))
    dev = max((abs(x - ps[0] - k * nom) for k, x in enumerate(ps)), default=Fraction(0))
    dts = [q[1] for q in pk if q[1] is not None]
    fps_nom = 1 / fd
    res = {
        "packets": n, "keyframes": len(keys), "no_pts": n - len(with_pts),
        "no_dts": n - len(dts),
        "discard_flag": sum("D" in q[5] for q in pk), "corrupt_flag": sum("C" in q[5] for q in pk),
        "dts_increasing": all(b > a for a, b in zip(dts, dts[1:])),
        "first_pts": ps[0] if ps else None, "last_pts": ps[-1] if ps else None,
        "first_dts": dts[0] if dts else None,
        "reorder_delay": delay,
        "b_runs": {str(k): v for k, v in sorted(runs.items())},
        "max_b_run": max(runs, default=0),
        "keyint_frames": stats3(kint),
        "keyint_s": {k: round(float(v / fps_nom), 3) for k, v in stats3(kint).items()}
        if kint else None,
        "keyint_hist": {str(k): v for k, v in collections.Counter(kint).most_common(8)},
        "leading": {"keyframes_with": len(leading),
                    "max": max((len(x[1]) for x in leading), default=0),
                    "total": sum(len(x[1]) for x in leading)},
        "pts_steps": {str(k): v for k, v in diffs.most_common(6)},
        "pts_step_nominal": str(nom),
        "pts_grid_dev_ms": round(float(dev * tb * 1000), 3),
        "pts_gaps": sum(c for d, c in diffs.items() if d > Fraction(3, 2) * nom),
        "pts_dups": sum(c for d, c in diffs.items() if d <= 0),
    }
    keylist = [pk[i][0] for i in keys if pk[i][0] is not None]
    return res, keylist, leading


def cmd_info(a):
    os.makedirs(a.out, exist_ok=True)
    st, fm = probe(a.file)
    t0, l0 = time.perf_counter(), load1()
    pk = scan_packets(a.file, p(a.out, "packets.csv"))
    scan_s = time.perf_counter() - t0
    tb = Fraction(st["time_base"])
    fd = 1 / Fraction(st["r_frame_rate"])
    gop, keylist, leading = analyse(pk, tb, fd)
    info = {"file": a.file, "ffmpeg": ffversion(), "stream": st, "format": fm, "gop": gop,
            "packet_scan_s": round(scan_s, 1), "packet_scan_load": l0,
            "key_pts": keylist, "leading_pts": leading}
    jsave(info, p(a.out, "info.json"))
    show = {k: v for k, v in info.items() if k not in ("key_pts", "leading_pts")}
    print(json.dumps(show, indent=1))


# ----------------------------------------------------------------------------- nal

TRACE_RE = re.compile(r"^\[trace_headers @ [^\]]+\]\s?(.*)$")
FIELD_RE = re.compile(r"^(\d+)\s+(\S+)\s+([01]*)\s+=\s+(-?\d+)\s*$")
PACKET_RE = re.compile(r"Packet: (\d+) bytes(.*)\.$")  # an error message may precede it
SEI_DROP = {"h264": "filter_units=remove_types=6,trace_headers",
            "hevc": "filter_units=remove_types=39|40,trace_headers"}
HEVC_NAL = {0: "TRAIL_N", 1: "TRAIL_R", 2: "TSA_N", 3: "TSA_R", 4: "STSA_N", 5: "STSA_R",
            6: "RADL_N", 7: "RADL_R", 8: "RASL_N", 9: "RASL_R", 16: "BLA_W_LP",
            17: "BLA_W_RADL", 18: "BLA_N_LP", 19: "IDR_W_RADL", 20: "IDR_N_LP", 21: "CRA"}
KEEP_FIELDS = ("recovery_frame_cnt", "exact_match_flag", "broken_link_flag", "closed_gop",
               "broken_link", "picture_coding_type", "temporal_reference")


def parse_trace(text):
    pkts, cur = [], None
    for line in text.splitlines():
        pm = PACKET_RE.search(line)
        m = TRACE_RE.match(line)
        if not m and not pm:
            continue
        body = m.group(1).strip() if m else ""
        if pm:
            rest = pm.group(2)
            g = re.search(r", pts (-?\d+)", rest)
            h = re.search(r", dts (-?\d+)", rest)
            cur = {"size": int(pm.group(1)), "key": "key frame" in rest,
                   "pts": int(g.group(1)) if g else None, "dts": int(h.group(1)) if h else None,
                   "nal": [], "slice": [], "sei": [], "f": {}}
            pkts.append(cur)
            continue
        if cur is None:
            continue
        fm = FIELD_RE.match(body)
        if not fm:
            continue
        name, val = fm.group(2), int(fm.group(4))
        if name == "nal_unit_type":
            cur["nal"].append(val)
        elif name == "pic_parameter_set_id":  # in a PPS, then in each slice header
            cur.setdefault("pps", []).append(val)
        elif name == "slice_type":
            cur["slice"].append(val)
        elif "payload_type" in name:
            cur["sei"].append(val)
        elif name in KEEP_FIELDS and name not in cur["f"]:
            cur["f"][name] = val
    return pkts


def picture_kind(codec, q):
    """A short label for one packet: picture type and, for H.264/HEVC, its NAL type."""
    if codec == "h264":
        vcl = [t for t in q["nal"] if t in (1, 5)]
        st = "PBISS"[q["slice"][0] % 5] if q["slice"] else "?"
        lab = ("IDR" if 5 in vcl else "") + st
        if len(set(q["slice"])) > 1:
            lab += "(" + "".join(sorted({"PBISS"[t % 5] for t in q["slice"]})) + ")"
        if "recovery_frame_cnt" in q["f"]:
            lab += "+RP" + str(q["f"]["recovery_frame_cnt"])
        if 7 in q["nal"] or 8 in q["nal"]:
            lab += "+SPS/PPS"
        return lab
    if codec == "hevc":
        vcl = sorted({t for t in q["nal"] if t < 32})
        st = "BPI"[q["slice"][0]] if q["slice"] and q["slice"][0] < 3 else "?"
        return "/".join(HEVC_NAL.get(t, str(t)) for t in vcl) + ":" + st
    if codec in ("mpeg2video", "mpeg1video"):
        t = q["f"].get("picture_coding_type")
        lab = {1: "I", 2: "P", 3: "B"}.get(t, "?")
        if "closed_gop" in q["f"]:
            lab = f"GOP(closed={q['f']['closed_gop']},broken={q['f'].get('broken_link')})" + lab
        return lab
    return "?"


def cmd_nal(a):
    info = jload(p(a.out, "info.json"))
    codec = info["stream"]["codec_name"]
    tb = Fraction(info["stream"]["time_base"])
    start = Fraction(info["format"].get("start_time", "0"))
    keys = info["key_pts"]
    if a.at:
        picks = [None if x == "start" else int(x) for x in a.at]
    else:
        picks = [None] + ([keys[len(keys) * j // 3] for j in (1, 2)] if len(keys) > 6 else [])
    kmax = (info["gop"]["keyint_frames"] or {}).get("max") or 60
    npk = a.packets or int(min(400, 2 * kmax + 8))
    samples = []
    for kp in picks:
        # CBS cannot parse every SEI message (x265's fail): on an error, drop the SEI NAL
        # units before tracing (the picture NAL types are what matter)
        for bsf in ("trace_headers", SEI_DROP.get(codec)):
            if bsf is None:
                break
            cmd = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "repeat+info",
                   "-copyts"]
            if kp is not None:
                cmd += ["-ss", tstr(Fraction(kp) * tb - start)]
            cmd += ["-i", a.file, "-map", VMAP, "-c", "copy", "-bsf:v", bsf,
                    "-frames:v", str(npk), "-f", "null", "-"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if "Error applying bitstream filters" not in r.stderr:
                break
        pk = parse_trace(r.stderr)
        for q in pk:
            q["kind"] = picture_kind(codec, q)
        # leading pictures of each keyframe in this sample
        lead = []
        for i, q in enumerate(pk):
            if q["key"] and q["pts"] is not None:
                ls = []
                for r2 in pk[i + 1:]:
                    if r2["key"]:
                        break
                    if r2["pts"] is not None and r2["pts"] < q["pts"]:
                        ls.append(r2["kind"])
                lead.append({"pts": q["pts"], "kind": q["kind"], "leading": ls})
        samples.append({"at_pts": kp, "cmd": cmd, "rc": r.returncode, "packets": pk,
                        "kinds": dict(collections.Counter(q["kind"] for q in pk)),
                        "keyframes": lead})
        print(f"== sample at pts {kp if kp is not None else 'start'}: {len(pk)} packets, "
              f"rc {r.returncode}")
        print("   kinds:", dict(collections.Counter(q["kind"] for q in pk)))
        for k in lead:
            print(f"   key pts {k['pts']}: {k['kind']}, leading {len(k['leading'])}: "
                  f"{dict(collections.Counter(k['leading']))}")
        print("   decode order:", " ".join(
            f"{q['kind']}@{q['pts']}" + ("*" if q["key"] else "") for q in pk[:a.show]))
    jsave({"codec": codec, "samples": samples}, p(a.out, a.name + ".json"))


# ----------------------------------------------------------------------------- ref

def run_sampled(cmd, stdout=None, stderr=None, every=5.0):
    """Run cmd, sampling the 1-minute load average; returns (rc, seconds, loads)."""
    loads = [load1()]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    while True:
        try:
            rc = proc.wait(timeout=every)
            break
        except subprocess.TimeoutExpired:
            loads.append(load1())
    return rc, time.perf_counter() - t0, loads


def cmd_ref(a):
    os.makedirs(a.out, exist_ok=True)
    md5_path = p(a.out, a.name + ".framemd5")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "warning",
           "-progress", p(a.out, a.name + ".progress"), "-stats_period", "30",
           "-threads", str(a.threads), "-copyts", "-i", a.file, "-map", VMAP,
           "-fps_mode", "passthrough", "-enc_time_base:v", "demux", "-f", "framemd5", "-y", md5_path]
    with open(p(a.out, a.name + ".stderr.log"), "w") as err:
        rc, secs, loads = run_sampled(cmd, stderr=err)
    if rc:
        raise SystemExit(f"ffmpeg failed ({rc}), see {a.name}.stderr.log")
    tb, pts, md5 = read_ref(a.out, a.name)
    res = {"cmd": cmd, "frames": len(md5), "seconds": round(secs, 1),
           "fps": round(len(md5) / secs, 1), "load_median": med(loads),
           "load_range": [min(loads), max(loads)], "tb": str(tb),
           "pts_increasing": all(b > a for a, b in zip(pts, pts[1:])),
           "distinct_md5": len(set(md5))}
    with open(p(a.out, a.name + ".stderr.log")) as f:
        lines = f.read().splitlines()
    res["stderr_lines"] = len(lines)
    res["stderr_head"] = lines[:5]
    if os.path.exists(p(a.out, "info.json")):
        info = jload(p(a.out, "info.json"))
        g = info["gop"]
        res["packets"] = g["packets"]
        res["packets_minus_discard"] = g["packets"] - g["discard_flag"]
        stb = Fraction(info["stream"]["time_base"])
        if tb == stb:
            pk = read_packets(p(a.out, "packets.csv"))
            pset = {q[0] for q in pk if q[0] is not None and "D" not in q[5]}
            fset = set(pts)
            res["packet_pts_not_decoded"] = len(pset - fset)
            res["frame_pts_not_in_packets"] = len(fset - pset)
        else:
            res["note"] = f"framemd5 tb {tb} != stream tb {stb}"
    jsave(res, p(a.out, a.name + ".json"))
    print(json.dumps(res, indent=1))


def cmd_same(a):
    ta, pa, ma = read_ref(a.out, a.a)
    tb_, pb, mb = read_ref(a.out, a.b)
    first = next((i for i, (x, y) in enumerate(zip(ma, mb)) if x != y), None)
    res = {"a": a.a, "b": a.b, "frames": [len(ma), len(mb)], "md5_equal": ma == mb,
           "pts_equal": pa == pb, "first_md5_difference": first,
           "md5_differences": sum(x != y for x, y in zip(ma, mb))}
    jsave(res, p(a.out, f"same_{a.a}_{a.b}.json"))
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- targets

def cmd_targets(a):
    info = jload(p(a.out, "info.json"))
    _, pts, md5 = read_ref(a.out)
    nf = len(pts)
    pos = {x: i for i, x in enumerate(pts)}
    keys = sorted(pos[x] for x in info["key_pts"] if x in pos)
    rng = random.Random(a.seed)
    tg = {}

    def add(n, kind):
        if 0 <= n < nf and n not in tg:
            tg[n] = kind

    for n in (0, 1, 2):
        add(n, "first")
    for n in (nf - 3, nf - 2, nf - 1):
        add(n, "last")
    nk = max(2, a.count // 10)
    inner = [k for k in keys if 16 <= k <= nf - 16]
    for k in sorted(rng.sample(inner, min(nk, len(inner)))):
        add(k, "key")
        add(k + 1, "after-key")
        add(k - 1, "before-key")
    lead = []
    for kp, lps in info["leading_pts"]:
        if kp in pos and 16 <= pos[kp] <= nf - 16:
            ls = sorted(pos[x] for x in lps if x in pos)
            if ls:
                lead.append((pos[kp], ls))
    for k, ls in sorted(rng.sample(lead, min(nk, len(lead)))):
        add(ls[0], "leading-first")
        add(ls[-1], "leading-last")
    while len(tg) < min(a.count, nf):
        add(rng.randrange(3, max(4, nf - 3)), "random")
    out = []
    for n in sorted(tg):
        i = bisect.bisect_right(keys, n) - 1
        out.append({"n": n, "kind": tg[n], "pts": pts[n],
                    "key_before": keys[i] if i >= 0 else None,
                    "from_key": n - keys[i] if i >= 0 else None})
    jsave({"frames": nf, "seed": a.seed, "targets": out}, p(a.out, "targets.json"))
    kinds = collections.Counter(t["kind"] for t in out)
    print(f"{len(out)} targets over {nf} frames: {dict(kinds)}")
    print("n:", " ".join(f"{t['n']}{t['kind'][0]}" for t in out))


# ----------------------------------------------------------------------------- seek

class Ctx:
    pass


def build(m, n, c):
    """ffmpeg command for method m and target n; returns (cmd, reads_rawvideo)."""
    pts = c.P[n]
    ab = Fraction(pts) * c.tb
    rel = ab - c.start
    pre, vf = [], None
    if m == "a_naive":
        pre = ["-ss", tstr(n * c.fd)]
    elif m in ("b_pts", "e_pipe"):
        pre = ["-ss", tstr(rel)]
    elif m == "b_copyts":
        pre = ["-copyts", "-ss", tstr(rel)]
    elif m == "b_abs":
        pre = ["-seek_timestamp", "1", "-ss", tstr(ab)]
    elif m == "b_abs_copyts":
        pre = ["-copyts", "-seek_timestamp", "1", "-ss", tstr(ab)]
    elif m == "b_abs_wrong":
        pre = ["-ss", tstr(ab)]
    elif m in ("c_1s", "c_gop"):
        if m == "c_1s":
            early = rel - 1
        else:  # the keyframe before the target's own keyframe, from the packet index
            i = bisect.bisect_right(c.keys, n) - 1
            early = Fraction(c.P[c.keys[i - 1]] if i >= 1 else c.P[0]) * c.tb - c.start
        pre = ["-copyts", "-ss", tstr(max(Fraction(0), early))]
        vf = f"select='gte(pts,{pts})'"
    elif m == "d_count":
        pre = ["-copyts"]
        vf = f"select='gte(n,{n})'"
    else:
        raise SystemExit(f"unknown method {m}")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "error",
           "-threads", str(c.threads)] + pre + ["-i", c.file, "-map", VMAP]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-fps_mode", "passthrough", "-frames:v", str(c.k)]
    if m == "e_pipe":
        return cmd + ["-f", "rawvideo", "-pix_fmt", c.pix_fmt, "pipe:1"], True
    return cmd + ["-enc_time_base:v", "demux", "-f", "framemd5", "pipe:1"], False


def execute(cmd, pipe, c):
    l0 = load1()
    with tempfile.TemporaryFile() as err:
        t0 = time.perf_counter()
        if pipe:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
            md5, pts = [], []
            while len(md5) < c.k:
                buf = proc.stdout.read(c.frame_size)
                if len(buf) < c.frame_size:
                    break
                md5.append(hashlib.md5(buf).hexdigest())
            proc.stdout.close()
            rc = proc.wait()
        else:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err)
            rc = r.returncode
            _, pts, md5 = parse_framemd5(r.stdout.decode())
        secs = time.perf_counter() - t0
        err.seek(0)
        errs = err.read().decode(errors="replace").splitlines()
    return {"rc": rc, "secs": secs, "load": l0, "md5": md5, "pts": pts,
            "err_n": len(errs), "err": errs[:3]}


def classify(out, n, c):
    """Compare the frames we got with the reference at n.. -> (status, shift, foreign)."""
    R = c.R
    want = min(c.k, len(R) - n)
    if not out:
        return "none", None, []
    foreign = [i for i, h in enumerate(out) if h not in c.idx]
    if out == R[n:n + len(out)]:
        return ("exact" if len(out) == want else "short"), 0, foreign
    if not foreign:
        for cand in sorted(c.idx[out[0]], key=lambda x: (abs(x - n), x))[:5000]:
            if R[cand:cand + len(out)] == out:
                return "shifted", cand - n, foreign
        return "mixed", None, foreign
    # some frames are not in the reference at all (decoded without their references?)
    good = [i for i in range(len(out)) if i not in foreign]
    shift = None
    if good:
        for cand in sorted(c.idx[out[good[0]]], key=lambda x: (abs(x - n - good[0]), x))[:5000]:
            base = cand - good[0]
            if all(0 <= base + i < len(R) and R[base + i] == out[i] for i in good):
                shift = base - n
                break
    return "corrupt", shift, foreign


def cmd_seek(a):
    info = jload(p(a.out, "info.json"))
    tj = jload(p(a.out, "targets.json"))
    targets = tj["targets"]
    if a.only_kinds:
        targets = [t for t in targets if t["kind"] in a.only_kinds.split(",")]
    if a.max_targets:
        targets = targets[:: max(1, len(targets) // a.max_targets)][:a.max_targets]
    c = Ctx()
    c.file, c.k, c.threads = a.file, a.frames, a.threads
    c.tb, c.P, c.R = read_ref(a.out)
    c.idx = collections.defaultdict(list)
    for i, h in enumerate(c.R):
        c.idx[h].append(i)
    st = info["stream"]
    if c.tb != Fraction(st["time_base"]):
        print(f"warning: reference tb {c.tb} != stream tb {st['time_base']}")
    c.start = Fraction(info["format"].get("start_time", "0"))
    c.fd = 1 / Fraction(st["r_frame_rate"])
    pos = {x: i for i, x in enumerate(c.P)}
    c.keys = sorted(pos[x] for x in info["key_pts"] if x in pos)  # display indices
    c.pix_fmt = st["pix_fmt"]
    c.frame_size = frame_bytes(c.pix_fmt, int(st["width"]), int(st["height"]))
    methods = a.methods.split(",") if a.methods else list(DEFAULT_METHODS) + (
        list(START_METHODS) if c.start != 0 else [])
    # decode-and-count runs on a few targets only (nearest to the given fractions of the file)
    nf = len(c.R)
    count_set = set()
    for x in (float(v) for v in a.count_at.split(",")):
        best = min(targets, key=lambda t: abs(t["n"] - x * (nf - 1)))
        count_set.add(best["n"])
    runs_path = p(a.out, a.name + "_runs.jsonl")
    if os.path.exists(runs_path) and not a.append:
        os.remove(runs_path)
    print(f"{len(targets)} targets x {methods}, {a.repeats} repeats, K={c.k}, "
          f"d_count on n in {sorted(count_set)} ({a.count_repeats} repeats), "
          f"start_time {float(c.start)}, {len(c.keys)} keyframes", flush=True)
    res = collections.defaultdict(list)
    t_all = time.perf_counter()
    for rep in range(a.repeats):
        for t in targets:
            for m in methods:
                if m == "d_count" and (t["n"] not in count_set or rep >= a.count_repeats):
                    continue
                cmd, pipe = build(m, t["n"], c)
                r = execute(cmd, pipe, c)
                status, shift, foreign = classify(r["md5"], t["n"], c)
                rec = {"n": t["n"], "kind": t["kind"], "from_key": t["from_key"], "method": m,
                       "rep": rep, "status": status, "shift": shift, "foreign": foreign,
                       "got": len(r["md5"]), "secs": round(r["secs"], 3), "load": r["load"],
                       "rc": r["rc"], "first_pts": r["pts"][0] if r["pts"] else None,
                       "err_n": r["err_n"], "err": r["err"], "md5": r["md5"]}
                if rep == 0 and t is targets[0]:
                    rec["cmd"] = cmd
                res[(t["n"], m)].append(rec)
                with open(runs_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        print(f"repeat {rep + 1}/{a.repeats} done at {time.perf_counter() - t_all:.0f} s, "
              f"load {load1()}", flush=True)
    summarise(a, c, targets, methods, res)


def summarise(a, c, targets, methods, res):
    per = {}
    for m in methods:
        rows = [(t, res[(t["n"], m)]) for t in targets if res.get((t["n"], m))]
        if not rows:
            continue
        status = collections.Counter()
        shifts = collections.Counter()
        unstable, times, loads, foreign = 0, [], [], 0
        for t, rr in rows:
            r0 = rr[0]
            status[r0["status"]] += 1
            if r0["status"] in ("shifted", "corrupt") and r0["shift"] is not None:
                shifts[r0["shift"]] += 1
            foreign += len(r0["foreign"])
            if any(r["md5"] != r0["md5"] for r in rr[1:]):
                unstable += 1
            times.append(med([r["secs"] for r in rr]))
            loads += [r["load"] for r in rr]
        e = {"targets": len(rows), "status": dict(status),
             "shifts": {str(k): v for k, v in sorted(shifts.items())},
             "foreign_frames": foreign, "unstable": unstable,
             "time_median": round(med(times), 3), "time_max": round(max(times), 3),
             "load_median": med(loads), "load_range": [min(loads), max(loads)],
             "runs": sum(len(rr) for _, rr in rows)}
        if m == "d_count":
            e["by_n"] = [{"n": t["n"], "secs": round(med([r["secs"] for r in rr]), 2),
                          "fps": round(t["n"] / med([r["secs"] for r in rr]), 1)}
                         for t, rr in rows]
        per[m] = e
    out = {"file": a.file, "frames": len(c.R), "k": c.k, "repeats": a.repeats,
           "methods": per, "ffmpeg": ffversion()}
    jsave(out, p(a.out, a.name + "_summary.json"))
    # grid: one line per target, one column per method
    code = {"exact": "=", "short": "s", "none": "0", "mixed": "?"}
    print(f"{'n':>8} {'kind':<13} {'fromK':>6} " + " ".join(f"{m:>12}" for m in methods))
    for t in targets:
        cells = []
        for m in methods:
            rr = res.get((t["n"], m))
            if not rr:
                cells.append(f"{'.':>12}")
                continue
            r0 = rr[0]
            s = code.get(r0["status"])
            if s is None:
                s = ("X" if r0["status"] == "corrupt" else "") + (
                    f"{r0['shift']:+d}" if r0["shift"] is not None else "")
                if r0["status"] == "corrupt":
                    s += f"/{len(r0['foreign'])}f"
            tm = med([r["secs"] for r in rr])
            cells.append(f"{s + ' ' + format(tm, '.2f'):>12}")
        print(f"{t['n']:>8} {t['kind']:<13} {str(t['from_key']):>6} " + " ".join(cells))
    for m, e in per.items():
        print(f"{m:>12}: {e['status']} shifts {e['shifts']} foreign {e['foreign_frames']} "
              f"unstable {e['unstable']} time median {e['time_median']} s max {e['time_max']} s"
              f" load {e['load_median']} {e['load_range']}"
              + (f" by_n {e['by_n']}" if "by_n" in e else ""))


# ----------------------------------------------------------------------------- keycheck

def cmd_keycheck(a):
    """Is each keyframe the container flags a clean random access point? Decode starting
    exactly at keyframe K (a stream copy from K, piped as NUT into a second ffmpeg: no seek
    heuristics) and compare every frame with the reference, by pts."""
    info = jload(p(a.out, "info.json"))
    tb, P, R = read_ref(a.out)
    pos = {x: i for i, x in enumerate(P)}
    start = Fraction(info["format"].get("start_time", "0"))
    keys = [k for k in info["key_pts"] if k in pos]
    picks = set()
    if a.range:
        lo, hi = (Fraction(x) for x in a.range.split(":"))
        picks |= {k for k in keys if lo <= k * tb <= hi}
    if a.sample:
        picks |= set(random.Random(a.seed).sample(keys, min(a.sample, len(keys))))
    rows = []
    for kp in sorted(picks):
        n = pos[kp]
        # -copypriorss 0: drop what the demuxer seek returns before K (-copyts copies it)
        c1 = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "error", "-copyts",
              "-ss", tstr(Fraction(kp) * tb - start), "-i", a.file, "-map", VMAP, "-c", "copy",
              "-copypriorss", "0", "-frames:v", str(a.frames), "-f", "nut", "pipe:1"]
        c2 = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "error", "-threads",
              str(a.threads), "-copyts", "-i", "pipe:0", "-map", "0:v:0", "-fps_mode",
              "passthrough", "-enc_time_base:v", "demux", "-f", "framemd5", "pipe:1"]
        with tempfile.TemporaryFile() as e1, tempfile.TemporaryFile() as e2:
            p1 = subprocess.Popen(c1, stdout=subprocess.PIPE, stderr=e1)
            r2 = subprocess.run(c2, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=e2)
            p1.stdout.close()
            p1.wait()
            e2.seek(0)
            errs = e2.read().decode(errors="replace").splitlines()
        otb, opts, omd5 = parse_framemd5(r2.stdout.decode())
        if otb is not None and otb != tb:  # NUT may pick another time base: rescale
            opts = [round(Fraction(x) * otb / tb) for x in opts]
        before = [x for x in opts if x < kp]  # leading pictures, decoded after K
        from_k = [(x, h) for x, h in zip(opts, omd5) if x >= kp]
        bad = [x for x, h in from_k if x not in pos or R[pos[x]] != h]
        lead_bad = [x for x, h in zip(opts, omd5) if x < kp and (x not in pos or R[pos[x]] != h)]
        first_good = next((i for i, (x, h) in enumerate(from_k)
                           if x in pos and R[pos[x]] == h), None)
        got = set(opts)
        row = {"key_pts": kp, "n": n, "first_out_pts": opts[0] if opts else None,
               "out": len(opts), "leading_out": len(before), "leading_bad": len(lead_bad),
               "from_k": len(from_k), "bad_from_k": len(bad), "first_good": first_good,
               "missing_from_k": sum(1 for x in P[n:n + len(from_k)] if x not in got),
               "decode_errors": len(errs), "err": errs[:2]}
        rows.append(row)
        flag = "CLEAN" if not bad else f"BAD {len(bad)}/{len(from_k)}"
        print(f"key pts {kp} (n {n}): out {len(opts)} (leading {len(before)}, "
              f"{len(lead_bad)} bad), from K: {flag}, first good +{first_good}, "
              f"decode errors {len(errs)}", flush=True)
    bad_keys = [r for r in rows if r["bad_from_k"]]
    pred = None
    if os.path.exists(p(a.out, "paramsets.json")):  # its prediction against what we saw
        unsafe = set(jload(p(a.out, "paramsets.json"))["unsafe_keyframe_pts"])
        pred = collections.Counter(
            ("predicted unsafe" if r["key_pts"] in unsafe else "predicted safe") + ", " +
            ("bad" if r["bad_from_k"] else "clean") for r in rows)
        print("paramsets prediction:", dict(pred))
    summary = {"checked": len(rows), "unclean": len(bad_keys), "prediction": pred,
               "unclean_pts": [r["key_pts"] for r in bad_keys][:200],
               "leading_bad_keys": sum(1 for r in rows if r["leading_bad"]),
               "frames": a.frames, "rows": rows}
    jsave(summary, p(a.out, a.name + ".json"))
    print(f"{len(rows)} keyframes checked: {len(bad_keys)} not clean (frames from K differ), "
          f"{summary['leading_bad_keys']} with bad leading pictures in the output")


# ----------------------------------------------------------------------------- paramsets

PS_TYPES = {"h264": "7|8", "hevc": "32|33|34"}
PS_UNITS = {"Video Parameter Set": "VPS", "Sequence Parameter Set": "SPS",
            "Picture Parameter Set": "PPS"}
PS_SKIP = {"forbidden_zero_bit", "nal_ref_idc", "nal_unit_type", "nuh_layer_id",
           "nuh_temporal_id_plus1", "rbsp_stop_one_bit", "rbsp_alignment_zero_bit"}


def cmd_paramsets(a):
    """In-band parameter sets over the whole stream, against the container's extradata
    (avcC/hvcC). After a seek the decoder only knows the extradata ones: a keyframe that comes
    while an in-band set with the same id but other content is in effect, and carries no copy
    of it, decodes wrongly until the next keyframe that carries one."""
    info = jload(p(a.out, "info.json"))
    codec = info["stream"]["codec_name"]
    if codec not in PS_TYPES:
        raise SystemExit(f"paramsets: {codec} not handled (H.264/HEVC only)")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "repeat+info", "-copyts",
           "-i", a.file, "-map", VMAP, "-c", "copy", "-bsf:v",
           f"filter_units=pass_types={PS_TYPES[codec]},trace_headers", "-f", "null", "-"]
    extra, sigs = {}, collections.defaultdict(set)
    inband = collections.defaultdict(list)  # pts -> [(kind, id, signature)]
    unit, fields, cur, in_extra = None, [], None, False

    def close_unit():
        if unit is None:
            return
        ident = next((v for k, v in fields if k.endswith("parameter_set_id")), 0)
        if in_extra:
            extra[(unit, ident)] = tuple(fields)
        elif cur is not None:
            inband[cur].append((unit, ident, tuple(fields)))

    # filter_units drops the packets left empty: only those with parameter sets are traced;
    # the others come from the packet scan
    t0 = time.perf_counter()
    with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, errors="replace") as proc:
        for line in proc.stderr:
            pm = PACKET_RE.search(line)
            m = TRACE_RE.match(line.rstrip("\n"))
            if pm:
                close_unit()
                unit, fields, in_extra = None, [], False
                g = re.search(r", pts (-?\d+)", pm.group(2))
                cur = int(g.group(1)) if g else None
                continue
            if not m:
                continue
            body = m.group(1).strip()
            if body == "Extradata":
                in_extra = True
                continue
            if body in PS_UNITS:
                close_unit()
                unit, fields = PS_UNITS[body], []
                continue
            fm = FIELD_RE.match(body)
            if fm and unit is not None and fm.group(2) not in PS_SKIP:
                fields.append((fm.group(2), int(fm.group(4))))
        close_unit()
    secs = time.perf_counter() - t0
    active, pk = {}, []
    for q in read_packets(p(a.out, "packets.csv")):  # decode order
        sets = [(k, i, extra.get((k, i)) == sig) for k, i, sig in inband.get(q[0], [])]
        for k, i, sig in inband.get(q[0], []):
            sigs[(k, i)].add(sig)
        rec = {"pts": q[0], "key": "K" in q[5], "sets": sets,
               "stale_before": sorted(f"{k}{i}" for (k, i), s in active.items() if not s)}
        for k, i, same in sets:  # its own sets apply to its slices and what follows
            active[(k, i)] = same
        rec["stale_after"] = sorted(f"{k}{i}" for (k, i), s in active.items() if not s)
        rec["carried"] = {f"{k}{i}" for k, i, _ in sets}
        pk.append(rec)
    with_sets = [q for q in pk if q["sets"]]
    differ = [q for q in with_sets if any(not s for _, _, s in q["sets"])]
    # unsafe keyframes: an in-band set that differs from the extradata is in effect, and the
    # keyframe carries no copy of it: a decoder starting there uses the extradata's instead
    unsafe = [q for q in pk if q["key"] and any(x not in q["carried"] for x in q["stale_before"])]
    keys = [q for q in pk if q["key"]]
    zones, z0 = [], None  # pts ranges where some differing set is in effect
    for q in pk:
        if q["stale_after"] and z0 is None:
            z0 = q["pts"]
        elif not q["stale_after"] and z0 is not None:
            zones.append([z0, q["pts"]])
            z0 = None
    if z0 is not None:
        zones.append([z0, pk[-1]["pts"]])
    res = {"file": a.file, "seconds": round(secs, 1), "packets": len(pk),
           "extradata_sets": sorted(f"{k}{i}" for k, i in extra),
           "packets_with_inband_sets": len(with_sets),
           "keyframes_with_inband_sets": sum(1 for q in with_sets if q["key"]),
           "packets_with_sets_differing_from_extradata": len(differ),
           "keyframes": len(keys), "unsafe_keyframes": len(unsafe),
           "keyframes_without_inband_sets": sum(1 for q in keys if not q["sets"]),
           "unsafe_keyframe_pts": [q["pts"] for q in unsafe],
           "stale_zones_pts": zones[:200], "stale_zones": len(zones),
           "first_differing": [{"pts": q["pts"], "key": q["key"], "sets": q["sets"]}
                               for q in differ[:20]],
           "distinct_inband": {f"{k}{i}": len(s) for (k, i), s in sigs.items()}}
    jsave(res, p(a.out, a.name + ".json"))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("unsafe_keyframe_pts", "first_differing")}, indent=1)[:4000])


# ----------------------------------------------------------------------------- report

def cmd_report(a):
    for out in a.outs:
        info = jload(p(out, "info.json"))
        st, g = info["stream"], info["gop"]
        print(f"\n## {os.path.basename(os.path.normpath(out))}: {st['codec_name']} "
              f"{st.get('profile')} {st['width']}x{st['height']} {st['pix_fmt']}, "
              f"{info['format']['format_name']}, tb {st['time_base']}, "
              f"start {info['format'].get('start_time')}")
        print(f"packets {g['packets']} (K {g['keyframes']}, D {g['discard_flag']}, no pts "
              f"{g['no_pts']}), keyint {g['keyint_frames']} frames {g['keyint_s']} s, "
              f"max B run {g['max_b_run']}, reorder delay {g['reorder_delay']}, leading "
              f"{g['leading']}, pts grid dev {g['pts_grid_dev_ms']} ms, steps {g['pts_steps']}")
        if os.path.exists(p(out, "packets.csv")) and g["first_pts"] is not None:
            size = sum(q[3] or 0 for q in read_packets(p(out, "packets.csv")))
            span = (g["last_pts"] - g["first_pts"]) * Fraction(st["time_base"]) \
                + 1 / Fraction(st["r_frame_rate"])
            print(f"video bitrate {float(size * 8 / span) / 1e6:.1f} Mbit/s "
                  f"({size / 1e9:.2f} GB over {float(span):.1f} s)")
        for name in ("ref", "ref2"):
            if os.path.exists(p(out, name + ".json")):
                r = jload(p(out, name + ".json"))
                print(f"{name}: {r['frames']} frames in {r['seconds']} s = {r['fps']} fps "
                      f"(load {r['load_median']} {r['load_range']}), "
                      f"stderr lines {r['stderr_lines']}")
        for fn in sorted(os.listdir(out)):
            if fn.startswith("same_"):
                print(fn, jload(p(out, fn)))
        for fn in sorted(os.listdir(out)):
            if fn.endswith("_summary.json"):
                s = jload(p(out, fn))
                print(f"\n{fn}: K={s['k']}, repeats {s['repeats']}")
                print("| method | targets | exact | shifted (d: count) | corrupt | other | "
                      "median s | max s | load |")
                print("|---|---|---|---|---|---|---|---|---|")
                for m, e in s["methods"].items():
                    stt = e["status"]
                    other = {k: v for k, v in stt.items()
                             if k not in ("exact", "shifted", "corrupt")}
                    print(f"| {m} | {e['targets']} | {stt.get('exact', 0)} | "
                          f"{stt.get('shifted', 0)} {e['shifts'] or ''} | "
                          f"{stt.get('corrupt', 0)} | {other or ''} | {e['time_median']} | "
                          f"{e['time_max']} | {e['load_median']} |")
                    if "by_n" in e:
                        print(f"|  | d_count by n: {e['by_n']} |||||||| ")


# ----------------------------------------------------------------------------- hashcost

def cmd_hashcost(a):
    """What a per-frame content hash costs in a first pass: throughput of the standard
    library's hashes on frame-sized buffers (distinct buffers, so caches don't flatter)."""
    import concurrent.futures
    import zlib
    size = frame_bytes(a.pix_fmt, a.width, a.height)
    bufs = [os.urandom(size) for _ in range(32)]
    algos = {"md5": lambda b: hashlib.md5(b).digest(),
             "sha1": lambda b: hashlib.sha1(b).digest(),
             "blake2b-128": lambda b: hashlib.blake2b(b, digest_size=16).digest(),
             "crc32": zlib.crc32, "adler32": zlib.adler32}
    print(f"{a.pix_fmt} {a.width}x{a.height}: {size} bytes per frame, {a.frames} frames per "
          f"measure, load {load1()}")
    for name, fn in algos.items():
        for th in (int(x) for x in a.threads.split(",")):
            work = [bufs[i % len(bufs)] for i in range(a.frames)]
            t0 = time.perf_counter()
            if th == 1:
                for b in work:
                    fn(b)
            else:
                with concurrent.futures.ThreadPoolExecutor(th) as ex:
                    list(ex.map(fn, work))
            dt = time.perf_counter() - t0
            print(f"{name:>12} {th:>2} thread(s): {a.frames / dt:7.0f} frames/s "
                  f"{a.frames * size / dt / 1e6:6.0f} MB/s {dt / a.frames * 1000:6.2f} ms/frame")


# ----------------------------------------------------------------------------- synth

SYNTH_ENCODES = {
    # open GOPs on purpose: x264 non-IDR I + recovery point, x265 CRA + RASL, MPEG-2 open GOP
    "h264_open.mkv": ["-c:v", "libx264", "-preset", "veryfast", "-threads", "16", "-x264-params",
                      "keyint=48:min-keyint=24:open-gop=1:bframes=3:b-pyramid=normal"],
    "hevc_open.mkv": ["-c:v", "libx265", "-preset", "veryfast", "-x265-params",
                      "keyint=48:min-keyint=24:open-gop=1:bframes=4:pools=16:log-level=error"],
    "mpeg2_open.vob": ["-vf", "scale=720:480", "-c:v", "mpeg2video", "-b:v", "6M", "-maxrate",
                       "9M", "-bufsize", "1835k", "-g", "15", "-bf", "2", "-threads", "16",
                       "-f", "vob"],
}
SYNTH_REMUX = {"h264_open.mp4": "h264_open.mkv", "h264_open.ts": "h264_open.mkv",
               "hevc_open.mp4": "hevc_open.mkv", "hevc_open.ts": "hevc_open.mkv",
               "mpeg2_open.ts": "mpeg2_open.vob"}


def cmd_synth(a):
    """Short test files with open GOPs and in index-less containers (TS, PS), made from an
    excerpt of SRC; with --remux, a stream copy of another source's excerpt into M2TS (a
    Blu-ray stream as found on the disc)."""
    os.makedirs(a.out, exist_ok=True)
    ff = [FFMPEG, "-hide_banner", "-nostdin", "-nostats", "-v", "error", "-y"]
    src = ["-ss", str(a.start), "-t", str(a.duration), "-i", a.src, "-map", VMAP,
           "-an", "-sn", "-dn"]
    for name, args in SYNTH_ENCODES.items():
        t0 = time.perf_counter()
        subprocess.run(ff + src + args + [p(a.out, name)], check=True)
        print(f"{name}: encoded in {time.perf_counter() - t0:.1f} s", flush=True)
    for name, base in SYNTH_REMUX.items():
        subprocess.run(ff + ["-i", p(a.out, base), "-map", "0", "-c", "copy", p(a.out, name)],
                       check=True)
        print(f"{name}: stream copy of {base}", flush=True)
    if a.remux:
        subprocess.run(ff + ["-ss", str(a.remux_start), "-t", str(a.duration), "-i", a.remux,
                             "-map", VMAP, "-c", "copy", "-f", "mpegts",
                             "-mpegts_m2ts_mode", "1", p(a.out, "bluray_excerpt.m2ts")],
                       check=True)
        print("bluray_excerpt.m2ts: stream copy", flush=True)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info")
    s.add_argument("file")
    s.add_argument("out")
    s = sub.add_parser("nal")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--at", nargs="*", help="keyframe pts to start from ('start' allowed)")
    s.add_argument("--packets", type=int, default=0)
    s.add_argument("--show", type=int, default=60, help="packets listed per sample")
    s.add_argument("--name", default="nal")
    s = sub.add_parser("ref")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--name", default="ref")
    s.add_argument("--threads", type=int, default=16)
    s = sub.add_parser("same")
    s.add_argument("out")
    s.add_argument("a")
    s.add_argument("b")
    s = sub.add_parser("targets")
    s.add_argument("out")
    s.add_argument("--count", type=int, default=40)
    s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("seek")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--methods", help="comma list (default: " + ",".join(DEFAULT_METHODS) +
                   ", plus " + ",".join(START_METHODS) + " when start_time != 0)")
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--frames", type=int, default=8)
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--count-at", default="0.1,0.5,0.9",
                   help="fractions of the file where d_count runs (nearest target)")
    s.add_argument("--count-repeats", type=int, default=1)
    s.add_argument("--only-kinds", help="comma list of target kinds")
    s.add_argument("--max-targets", type=int, default=0)
    s.add_argument("--name", default="seek")
    s.add_argument("--append", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("outs", nargs="+")
    s = sub.add_parser("keycheck")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--range", help="keyframes with pts in LO:HI seconds")
    s.add_argument("--sample", type=int, default=0, help="also N random keyframes")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--frames", type=int, default=30, help="packets copied from K")
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--name", default="keycheck")
    s = sub.add_parser("paramsets")
    s.add_argument("file")
    s.add_argument("out")
    s.add_argument("--name", default="paramsets")
    s = sub.add_parser("hashcost")
    s.add_argument("--width", type=int, default=1920)
    s.add_argument("--height", type=int, default=1080)
    s.add_argument("--pix-fmt", default="yuv420p")
    s.add_argument("--frames", type=int, default=300)
    s.add_argument("--threads", default="1,4,8")
    s = sub.add_parser("synth")
    s.add_argument("src")
    s.add_argument("out")
    s.add_argument("--start", type=float, default=60)
    s.add_argument("--duration", type=float, default=120)
    s.add_argument("--remux", help="a second source, stream-copied into bluray_excerpt.m2ts")
    s.add_argument("--remux-start", type=float, default=600)
    a = ap.parse_args()
    {"info": cmd_info, "nal": cmd_nal, "ref": cmd_ref, "same": cmd_same,
     "targets": cmd_targets, "seek": cmd_seek, "report": cmd_report,
     "keycheck": cmd_keycheck, "paramsets": cmd_paramsets, "hashcost": cmd_hashcost,
     "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
