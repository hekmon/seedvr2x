#!/usr/bin/env python3
"""Step 3b: tests/test_lab.py's own figures for a seedvr2x lab master of milestone 1's input, on the
CPU, from the test's own functions (M5_SEEDVR2X's tests/test_lab.py): the PSNR against numz's
lab master, whole and per frame, and the test's scores of both, to set its PSNR floor and comments.

    M5_SEEDVR2X=SEEDVR2X CUDA_VISIBLE_DEVICES= nice -n 19 SEEDVR2X/.venv/bin/python \
      m5_lab_fig.py OURS.mkv M1_DIR

M5_SEEDVR2X, required: seedvr2x's project directory at eca0ff1, whose tests/test_lab.py this
imports (with its venv's python). OURS.mkv: ours' lab master of milestone 1's input
(m5_run3b.sh's ours3b/m1.mkv); M1_DIR: the directory of milestone 1's input_rgb.mkv and numz's lab
master, numz_lab.mkv (m5_run.sh m1).

Historical: the record of how milestone 5's step 3b set test_lab.py's figures (2026-10-04), kept
to re-run it: PSNR to numz 60.18 dB, floor 59.5 (seedvr2x/DESIGN.md, Validation milestones 5).
test_lab.py retired with lab when split replaced it (68b1529).
"""

import os
import sys
from pathlib import Path

import numpy as np

if len(sys.argv) != 3 or not os.environ.get("M5_SEEDVR2X"):
    sys.exit(__doc__)
sys.path.insert(0, str(Path(os.environ["M5_SEEDVR2X"]) / "tests"))
import test_lab as t  # noqa: E402

master = Path(sys.argv[1])
m1 = Path(sys.argv[2])
source = t.planes(m1 / "input_rgb.mkv", "gbrp", "u1")
ours = t.planes(master, "gbrp16le", "<u2")
numz = t.planes(m1 / "numz_lab.mkv", "gbrp16le", "<u2")
frames = [t.psnr(ours[i : i + 1], numz[i : i + 1]) for i in range(len(ours))]
worst = int(np.argmin(frames))
diff = int(np.abs(ours.astype(np.int32) - numz.astype(np.int32)).max())
print(
    f"PSNR ours against numz {t.psnr(ours, numz):.3f} dB; worst frame {frames[worst]:.2f} dB"
    f" (frame {worst}), best {max(frames):.2f}; largest difference {diff} codes"
)
print("ours", t.scores(ours, source))
print("numz", t.scores(numz, source))
