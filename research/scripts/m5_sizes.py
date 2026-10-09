#!/usr/bin/env python3
"""What an FFV1 master's size comes from, for milestone 5's report: per gbrp16le master, its bytes
per frame, the 16-bit codes its first frame uses, and re-encodes with seedvr2x's writer options
(media/writer.py: ffv1 level 3, intra, 16 slices, slice CRCs):
- same:        the frames as they are (the encoder's own size, without the container's)
- bf16:        each value rounded to the nearest bf16 (round to nearest even) first, as numz's output
               is before ffv1_out.py quantises it; with the PSNR that rounding costs (8-bit scale)
- yuv420p10le: through the writer's yuv420p10le chain (zscale, BT.709, limited, chroma left,
               bilinear, no dither)

  m5_sizes.py MASTER...      (numpy, ffmpeg/ffprobe on PATH; CPU only)

Historical: the record of how milestone 5 measured the masters' sizes (2026-10-04), kept to re-run
it, on m5_run.sh's and m5_run3b.sh's gbrp16le masters (ours with lab or none, numz's): the
figures in seedvr2x/DESIGN.md, Output (master sizes: gbrp16le with colour correction 2.0-2.5
times numz's lab, tens of thousands of codes in a frame against 300-400 in numz's bf16 output,
rounding ours to bf16 saving 34-43%) and Colour correction. The writer options and the
yuv420p10le chain are seedvr2x's media/writer.py's at eca0ff1 (zscale on ffmpeg's default
slices: seedvr2x runs it on one since). seedvr2x's `--color-correction lab` went with 68b1529,
when split replaced lab. It calls no research script.
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

FFV1 = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]
YUV = ("zscale=min=gbr:rin=full:m=709:r=limited:c=left:pin=unspecified:p=unspecified"
       ":tin=unspecified:t=unspecified:d=none:f=bilinear,format=yuv420p10le")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,pix_fmt,nb_read_packets",
                          "-of", "json", path], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["pix_fmt"], int(s["nb_read_packets"])


def planes(path, w, h):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
                          "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, "<u2").reshape(-1, 3, h, w)


def encode(frames, w, h, out, vf=None, pix_fmt="gbrp16le"):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gbrp16le",
           "-s", f"{w}x{h}", "-framerate", "24000/1001", "-i", "-"]
    if vf:
        cmd += ["-vf", vf]
    cmd += [*FFV1, "-pix_fmt", pix_fmt, "-f", "matroska", out]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    return os.path.getsize(out)


def bf16(codes):
    x = (codes.astype(np.float32) / np.float32(65535)).view(np.uint32)
    x = (x + (0x7FFF + ((x >> 16) & 1))) & np.uint32(0xFFFF0000)
    return np.rint(np.clip(x.view(np.float32), 0, 1) * 65535).astype(np.uint16)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    rows = []
    for path in sys.argv[1:]:
        w, h, fmt, n = probe(path)
        if fmt != "gbrp16le":
            raise SystemExit(f"{path}: {fmt}")
        f = planes(path, w, h)
        first = f[0]
        with tempfile.TemporaryDirectory() as tmp:
            same = encode(f, w, h, os.path.join(tmp, "same.mkv"))
            rounded = bf16(f)
            grid = encode(rounded, w, h, os.path.join(tmp, "bf16.mkv"))
            yuv = encode(f, w, h, os.path.join(tmp, "yuv.mkv"), YUV, "yuv420p10le")
        d = (f.astype(np.float64) - rounded) * 255 / 65535
        mse = float(np.mean(d * d))
        rows.append({
            "master": path, "frames": n, "bytes": os.path.getsize(path),
            "MB_per_frame": round(os.path.getsize(path) / n / 1e6, 3),
            "codes_frame0": int(np.unique(first).size),
            "odd_frame0": round(float((first & 1).mean()), 4),
            "same_MB_per_frame": round(same / n / 1e6, 3),
            "bf16_MB_per_frame": round(grid / n / 1e6, 3),
            "bf16_psnr": round(10 * math.log10(255**2 / mse), 2) if mse else None,
            "bf16_unchanged": bool(np.array_equal(rounded, f)),
            "yuv420p10le_MB_per_frame": round(yuv / n / 1e6, 3),
        })
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
