#!/usr/bin/env python3
"""The DiT's peak per token and per attention window, refitted on the lean attention.

Reads the .jsonl files research/scripts/hold_probe.py writes (the sharp 7B fp16, one sampling step
per window, the VAE's weights on the GPU): the size and the placement (blocks swapped) from each
file's first record, then every step record that has "ok": true, "latents" and "peak_bytes".
Each point is peak - (loaded(FP16) - swapped_saving(FP16, swapped)), the model's own terms from
seedvr2x/peaks.py; least squares on the tokens and the attention windows (peaks.tokens and
peaks.attention_windows, as the planner counts them) with a constant. Prints the table and the
fit, then the constant that bounds every point for the fitted slopes.

Run with the CLI's environment, apps/seedvr2x/.venv/bin/python, which imports seedvr2x:

    lean_fit.py [--leave-out ROWSxLATENTS] FILE.jsonl [FILE.jsonl ...]

The left-out point (default 1104x1, the 1-latent 1080p window, swapped or not, whose peak is elsewhere in the
step) is in the table and counted by the bounding constant, not by the fit. The same window
measured by several files (FlashAttention 2 and SDPA) is one point per file; the fit takes all.
Needs numpy.
"""

import argparse
import json
import sys

import numpy as np

try:
    from seedvr2x import peaks
except ImportError:
    sys.exit("seedvr2x is not importable: run with the CLI environment's python")

GIB, KIB, MIB = 1024**3, 1024, 1024**2
PADDED = {"1080": (1104, 1920), "2160": (2192, 3840)}


def points(path):
    size = swapped = None
    found = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if "placement" in record and "size" in record and size is None:
                size, swapped = record["size"], int(record["placement"])
            elif (
                record.get("ok") is True
                and "peak_bytes" in record
                and "latents" in record
            ):
                if size is None:
                    sys.exit(f"{path}: a step before the first record")
                found.append(
                    (
                        PADDED[size],
                        swapped,
                        int(record["latents"]),
                        record["peak_bytes"],
                    )
                )
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leave-out", default="1104x1")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    out_rows, out_latents = (int(v) for v in args.leave_out.split("x"))
    rows = []
    for path in args.files:
        for padded, swapped, latents, peak in points(path):
            base = peaks.loaded(peaks.FP16) - peaks.swapped_saving(peaks.FP16, swapped)
            rows.append(
                (
                    padded,
                    swapped,
                    latents,
                    peaks.tokens(padded, latents),
                    peaks.attention_windows(padded, latents),
                    peak - base,
                    peak,
                )
            )
    if not rows:
        sys.exit("no step record found")
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    fit_rows = [r for r in rows if (r[0][0], r[2]) != (out_rows, out_latents)]
    design = np.array([[r[3], r[4], 1.0] for r in fit_rows])
    target = np.array([r[5] for r in fit_rows], dtype=float)
    (token, window, constant), *_ = np.linalg.lstsq(design, target, rcond=None)
    print(
        f"{'rows':>5} {'swapped':>7} {'latents':>7} {'tokens':>8} {'windows':>7} {'peak GiB':>9}"
        f" {'resid GiB':>10}"
    )
    worst = -np.inf
    for padded, swapped, latents, tok, win, extra, peak in rows:
        resid = (extra - (token * tok + window * win + constant)) / GIB
        worst = max(worst, (extra - (token * tok + window * win)) / GIB)
        mark = "" if (padded[0], latents) != (out_rows, out_latents) else " *"
        print(
            f"{padded[0]:>5} {swapped:>7} {latents:>7} {tok:>8} {win:>7} {peak / GIB:>9.3f}"
            f" {resid:>+10.4f}{mark}"
        )
    print(f"\npoints fitted: {len(fit_rows)} of {len(rows)} (* left out)")
    print(f"per token : {token / KIB:.3f} KiB")
    print(f"per window: {window / MIB:.3f} MiB")
    print(f"constant  : {constant / GIB:+.4f} GiB")
    print(f"constant that bounds every point: {worst:.4f} GiB")


if __name__ == "__main__":
    main()
