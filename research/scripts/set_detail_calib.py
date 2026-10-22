#!/usr/bin/env python3
"""The 'recognisable window' score of quality_set.py, on windows the user's eyes judged or couldn't: the GT window's
8-bit luma, Sobel 3x3 magnitude / 8 (levels per pixel); printed: mean, 90th percentile, share of pixels >= 2 and >= 4
levels/px (quality_set.py's score, REC = 10%), mean luma. Windows: the 9 places of the tiles set on anime-sky (7 of
them unjudged, a flat sky), 3 judged 1080p places, 9 judged 4K places of the first --quality set, and hand-picked
4K windows (a flat sky, near-black, clouds, a jet and motion blur, a face, a dark crater, a debris-strewn street).

  set_detail_calib.py CLIPS CLIPS4K      (fr_clips.py's 1080p and 4K clip directories: <clip>.gt.mkv)
"""
import subprocess, sys
import numpy as np
import cv2

def frame_y(path, t):
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip().split(","))
    b = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-i", path, "-vf", f"select=eq(n\\,{t})",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"], capture_output=True, check=True).stdout
    g, bb, r = np.frombuffer(b, np.uint16).reshape(3, h, w).astype(np.float32) / 65535 * 255
    return 0.2126 * r + 0.7152 * g + 0.0722 * bb

def score(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy) / 8
    return m.mean(), np.percentile(m, 90), (m >= 2).mean(), (m >= 4).mean(), y.mean()

C1, C4 = sys.argv[1:3]
P = [("t1 anime-sky", f"{C1}/anime-sky.gt.mkv", t, x, y) for t, x, y in
     [(3, 1536, 672), (1, 1344, 696), (1, 736, 576), (19, 1536, 480), (4, 1536, 696), (37, 1152, 0), (5, 1024, 696),
      (2, 224, 696), (39, 0, 696)]]
P += [("t1 anime-dark", f"{C1}/anime-dark.gt.mkv", 2, 1536, 256), ("t1 live-slow", f"{C1}/live-slow.gt.mkv", 21, 1344, 320),
      ("t1 cartoon-bright", f"{C1}/cartoon-bright.gt.mkv", 35, 128, 384)]
P += [(f"t6 {s}", f"{C4}/{s}.gt.mkv", t, x, y) for s, t, x, y in
      [("sollevante-painted", 33, 960, 1024), ("sollevante-painted", 41, 448, 0), ("ouatia-motion", 5, 928, 640),
       ("ouatia-motion", 13, 896, 1216), ("ouatia-dark", 34, 1792, 1280), ("ouatia-dark", 42, 1760, 1216),
       ("ouatia-street", 7, 1408, 672), ("ouatia-street", 22, 1344, 672), ("ouatia-face", 14, 2048, 832)]]
P += [("sunrise sky", f"{C4}/digital-sunrise.gt.mkv", 22, 1000, 200), ("sunrise dark", f"{C4}/digital-sunrise.gt.mkv", 22, 1000, 1500),
      ("space clouds", f"{C4}/digital-space.gt.mkv", 22, 1600, 700), ("space dark", f"{C4}/digital-space.gt.mkv", 22, 3200, 1200),
      ("dogfight jet", f"{C4}/digital-dogfight.gt.mkv", 22, 1800, 900), ("dogfight blur", f"{C4}/digital-dogfight.gt.mkv", 22, 2600, 100),
      ("cockpit face", f"{C4}/digital-cockpit.gt.mkv", 22, 2000, 600), ("cockpit black", f"{C4}/digital-cockpit.gt.mkv", 22, 200, 1000),
      ("night crater", f"{C4}/cel4k-night.gt.mkv", 22, 1000, 1000), ("detail street", f"{C4}/cel4k-detail.gt.mkv", 22, 2160, 740)]
cache = {}
for lab, p, t, x, y in P:
    if (p, t) not in cache:
        cache[(p, t)] = frame_y(p, t)
    s = score(cache[(p, t)][y:y + 384, x:x + 384])
    print(f"{lab:22s} f{t:<3d} x{x:<5d} y{y:<5d} mean {s[0]:6.2f} p90 {s[1]:6.2f} >=2 {s[2]:.3f} >=4 {s[3]:.3f} Y {s[4]:6.1f}", flush=True)
