#!/usr/bin/env python3
"""The brightness bias of numz's encoder input, predicted from each input's own 8-bit codes: every
code k becomes k/255 in float32, then float16 (CPU), then bfloat16 (device) before the resize
(numz generation_phases.py:380-388, inference_cli.py:613-618); numz's lab reference keeps float16
(:130-168). Per input: the mean BT.601 luma of (rounded - exact), 8-bit levels, over every pixel
of every frame, for both paths; a resize keeps the mean.

  m5_bias.py INPUT.mkv...     (numpy + torch; ffmpeg on PATH; CPU only)

Historical: the record of milestone 5's diagnosis (2026-10-04), kept to re-run it: ours as first
built (seedvr2x 27ce6ba) took numz's encoder input as lab's reference, so its output came out
brighter than numz's by each input's own bias; this predicts it from the inputs' codes (milestone
1's, clip B's, the full-reference clips' d1), against which the measured differences were set:
seedvr2x/DESIGN.md, Colour correction ("by 0.10–0.16 level, as predicted from their codes"),
which made lab's reference float32 (eca0ff1). numz 4490bd1's lines cited. seedvr2x's
`--color-correction lab` went with 68b1529, when split replaced lab. It calls no research script.
"""
import json
import subprocess
import sys

import numpy as np
import torch

k = torch.arange(256, dtype=torch.float32) / 255
BF16 = ((k.to(torch.float16).to(torch.bfloat16).float() - k) * 255).double().numpy()
FP16 = ((k.to(torch.float16).float() - k) * 255).double().numpy()
LUMA = np.array([0.299, 0.587, 0.114])


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             check=True, capture_output=True).stdout
        codes = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        counts = np.stack([np.bincount(codes[:, c], minlength=256) for c in range(3)])
        total = counts[0].sum()
        bf16 = float((counts * BF16).sum(1) @ LUMA / total)
        fp16 = float((counts * FP16).sum(1) @ LUMA / total)
        mean = float((counts * np.arange(256)).sum(1) @ LUMA / total)
        print(json.dumps({"input": path, "mean_luma": round(mean, 2),
                          "bias_bf16_path": round(bf16, 4), "bias_fp16_path": round(fp16, 4)}))


if __name__ == "__main__":
    main()
