#!/usr/bin/env python3
"""Compare two runs bit for bit: their FFV1 masters, or their float32 frame dumps.

Milestone 1 holds seedvr2x's output against numz's (DESIGN.md, Validation milestones), and the
resume milestone an interrupted run against an uninterrupted one. Both ask for identical frames,
so this reports equality first, then, for the frames that differ, how much.

Usage, from seedvr2x/:
  uv run tools/compare.py masters A.mkv B.mkv   decoded as gbrp16le, compared frame by frame
  uv run tools/compare.py frames DIR_A DIR_B    frame_NNNNNN.npy float32 dumps (ffv1_out.py's
                                                FFV1_OUT_DUMP, seedvr2x's --dump-frames)
"""

import argparse
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt


def master_frames(path: Path) -> Iterator[npt.NDArray[np.uint16]]:
    """The frames of an RGB FFV1 master, (3, H, W) uint16 planes, as decoded."""
    probe = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "csv=p=0"),
            *("-show_entries", "stream=width,height", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = (int(x) for x in probe.stdout.strip().split(","))
    frame_bytes = 3 * width * height * 2
    decoder = subprocess.Popen(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0"),
            *("-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"),
        ],
        stdout=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    while data := decoder.stdout.read(frame_bytes):
        if len(data) < frame_bytes:
            raise ValueError(f"{path}: truncated frame")
        yield np.frombuffer(data, dtype="<u2").reshape(3, height, width)
    if decoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed decoding {path}")


def dump_frames(directory: Path) -> Iterator[npt.NDArray[np.float32]]:
    for path in sorted(directory.glob("frame_*.npy")):
        yield np.load(path)


def report(pairs: Iterator[tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]]) -> int:
    """Print the comparison of frame pairs; 0 when every frame is identical."""
    count, identical, worst = 0, 0, 0.0
    for index, (a, b) in enumerate(pairs):
        count += 1
        if a.shape != b.shape:
            print(f"frame {index}: shapes {a.shape} and {b.shape}")
            continue
        if a.dtype == np.float32 and b.dtype == np.float32:
            same = np.array_equal(a.view(np.uint32), b.view(np.uint32))  # -0.0 vs 0.0, NaNs
        else:
            same = np.array_equal(a, b)
        if same:
            identical += 1
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        worst = max(worst, float(diff.max()))
        print(
            f"frame {index}: {int(np.count_nonzero(diff))} of {diff.size} values differ, "
            f"max {diff.max():.6g}, mean {diff.mean():.3g}"
        )
    print(
        f"{identical} of {count} frames identical"
        + (f", max difference {worst:.6g}" if worst else "")
    )
    return 0 if count and identical == count else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    masters = commands.add_parser("masters", help="two RGB FFV1 masters")
    masters.add_argument("a", type=Path)
    masters.add_argument("b", type=Path)
    frames = commands.add_parser("frames", help="two directories of frame_NNNNNN.npy")
    frames.add_argument("a", type=Path)
    frames.add_argument("b", type=Path)
    args = parser.parse_args()
    if args.command == "masters":
        pairs = zip(master_frames(args.a), master_frames(args.b), strict=True)
    else:
        names_a = sorted(p.name for p in args.a.glob("frame_*.npy"))
        names_b = sorted(p.name for p in args.b.glob("frame_*.npy"))
        if names_a != names_b:
            print(f"different frame files: {len(names_a)} and {len(names_b)}", file=sys.stderr)
            return 1
        pairs = zip(dump_frames(args.a), dump_frames(args.b), strict=True)
    return report(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
