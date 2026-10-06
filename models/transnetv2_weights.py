# /// script
# requires-python = "==3.12.*"
# dependencies = ["tensorflow-cpu==2.21.0", "torch==2.14.1", "numpy==2.5.2", "safetensors==0.8.0"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""TransNetV2's weights for seedvr2x, as safetensors.

TransNetV2 (github.com/soCzech/TransNetV2, MIT) publishes its weights as a TensorFlow SavedModel,
with a PyTorch port of the model and the script converting the weights to it,
inference-pytorch/convert_weights.py. This script runs that converter, unchanged, on the official
weights at COMMIT; checks its output byte for byte against the one measurement made
(transnetv2-pytorch-weights.pth, PTH_SHA256: research/docs/scene-detection.md); and saves the
same tensors as safetensors, values unchanged.

    uv run models/transnetv2_weights.py [--out DIR] [--work DIR] [--threads N]
                                        [--video FILE START COUNT]...

- Inputs: TransNetV2's files at COMMIT, each checked by its SHA-256 (the weights are Git LFS
  objects, whose SHA-256 is their object id), downloaded into --work (default
  ~/.cache/seedvr2x-models/transnetv2), where the converter runs.
- The check: TransNetV2's PyTorch model loading the safetensors file, and its official TensorFlow
  model (inference/transnetv2.py), on the same frames: 1,000 random ones, then each --video
  excerpt, frames START to START+COUNT-1 decoded from the first as the official extraction
  decodes them (ffmpeg, rgb24, scaled to 48x27 by its default scaler; needs ffmpeg on PATH). Both
  run predict_frames' windows: 100 frames every 50, frames 25-74 of each kept. It passes when the
  probabilities differ by less than TOLERANCE and the detections are the same: each run of frames
  at or above 0.1, 0.3 and 0.5, at its peak (DESIGN.md, Shot detection), and the official
  predictions_to_scenes at 0.5. measurement found 5.1e-7 at most on 1,000 random frames and two
  3,000-frame excerpts (torch 2.14.1, TensorFlow 2.21.0, 16 threads).
- The output, in --out (default models/dist): transnetv2.safetensors, read back with the
  safetensors library and compared with the converter's tensors bit for bit, its SHA-256 the one
  pinned in OUTPUT; and transnetv2.LICENSE beside it, TransNetV2's MIT licence. A file failing a
  check is deleted.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import safetensors
import torch
from common import CACHE, DIST, Entry, check_file, check_output, fetch, log, write_safetensors
from safetensors.torch import load_file

COMMIT = "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed"  # 2021-07-28, the repository's last commit
RAW = f"https://raw.githubusercontent.com/soCzech/TransNetV2/{COMMIT}/"
MEDIA = f"https://media.githubusercontent.com/media/soCzech/TransNetV2/{COMMIT}/"  # Git LFS files
WEIGHTS = "inference/transnetv2-weights"
SOURCES = {  # path in the repository: (URL prefix, size, SHA-256)
    "LICENSE": (RAW, 1072, "a8d7a056688ccedebe89f18fd60f1a47128df94cb82669cd02459934919cbb6f"),
    "inference-pytorch/transnetv2_pytorch.py": (
        RAW,
        12475,
        "f7c1d437465579a8ec28a5add19853d2cb2755248ea4a4207678210a609428e1",
    ),
    "inference-pytorch/convert_weights.py": (
        RAW,
        4199,
        "3572d76ddccc92e7ccca13ac3c47aaf39c86f56ce4e392d4d0d6a60da05b42fa",
    ),
    "inference/transnetv2.py": (
        RAW,
        8045,
        "f55b3a75727d1502438707ac15e8f6257a736817e713e2113e4b84176500ca65",
    ),
    f"{WEIGHTS}/saved_model.pb": (
        MEDIA,
        5933260,
        "8ac2a52c5719690d512805b6eaf5ce12097c1d8860b3d9de245dcbbc3100f554",
    ),
    f"{WEIGHTS}/variables/variables.data-00000-of-00001": (
        MEDIA,
        30516656,
        "b8c9dc3eb807583e6215cabee9ca61737b3eb1bceff68418b43bf71459669367",
    ),
    f"{WEIGHTS}/variables/variables.index": (
        MEDIA,
        5526,
        "8b99e28b4ad11372d9a1ad9703298c2e370df14859da4245fdbe818e92dd403f",
    ),
}
PTH_SHA256 = "eed5336d5d6a013c67f5863505a26e7e835053e64a9ce413d6b089ccba07bb53"
NAME = "transnetv2.safetensors"
LICENSE = "transnetv2.LICENSE"
# The output of the pinned inputs and versions: a run must give these bytes.
OUTPUT = "bb8c838811a5e52e23be70e2794646a758d2bf4dcbd110dec8211ae7b8cbefdf"
TOLERANCE = 1e-6
THRESHOLDS = (0.1, 0.3, 0.5)
SIZE = (27, 48, 3)  # frames, height x width x RGB
DTYPES = {torch.float32: "F32", torch.int64: "I64"}


def metadata() -> dict[str, str]:
    weights = ", ".join(
        f"{p.removeprefix(WEIGHTS + '/')} {s[2]}"
        for p, s in SOURCES.items()
        if p.startswith(WEIGHTS)
    )
    return {
        "format": "pt",
        "license": "mit",
        "copyright": "TransNetV2: Copyright (c) 2020 Tomáš Souček",
        "source": f"TransNetV2's TensorFlow weights ({WEIGHTS}) of soCzech/TransNetV2, "
        f"commit {COMMIT}",
        "source_url": f"https://github.com/soCzech/TransNetV2/tree/{COMMIT}/{WEIGHTS}",
        "source_sha256": weights,
        "change": "converted to PyTorch by TransNetV2's own inference-pytorch/convert_weights.py "
        f"(its output transnetv2-pytorch-weights.pth, sha256 {PTH_SHA256}), saved as "
        "safetensors; values unchanged",
        "conversion": "unofficial, not TransNetV2's authors': made by seedvr2x's "
        "models/transnetv2_weights.py",
    }


def convert(src: Path) -> dict[str, torch.Tensor]:
    """The official converter, run unchanged in its directory: its .pth, checked byte for byte."""
    pth = src / "inference-pytorch" / "transnetv2-pytorch-weights.pth"
    pth.unlink(missing_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "1"}
    subprocess.run(
        [sys.executable, "convert_weights.py", "--tf_weights", f"../{WEIGHTS}/"],
        cwd=src / "inference-pytorch",
        env=env,
        check=True,
    )
    check_file(pth, PTH_SHA256)
    log(f"convert_weights.py: {pth.name} is measurement's, byte for byte (sha256 {PTH_SHA256})")
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    if any(v.dtype not in DTYPES for v in sd.values()):
        raise SystemExit(f"{pth}: dtypes {sorted({str(v.dtype) for v in sd.values()})}")
    return sd


def as_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy())


def read_back(path: Path, sd: dict[str, torch.Tensor], md: dict[str, str]) -> None:
    """The file as the safetensors library reads it: the metadata, and every tensor's dtype, shape
    and bytes equal to the converter's."""
    with safetensors.safe_open(path, framework="pt") as f:
        if f.metadata() != md:
            raise SystemExit(f"{path}: metadata read back differs")
    got = load_file(path)
    if set(got) != set(sd):
        raise SystemExit(f"{path}: tensor names read back differ")
    for k, v in sd.items():
        if (
            got[k].dtype != v.dtype
            or got[k].shape != v.shape
            or bytes(as_bytes(got[k])) != bytes(as_bytes(v))
        ):
            raise SystemExit(f"{path}: {k} read back differs")


def predict(model: torch.nn.Module, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """predict_frames of inference/transnetv2.py with the PyTorch model: 25 copies of the first
    frame before, 25 to 74 of the last after, windows of 100 frames every 50, frames 25-74 of each
    kept; the sigmoids of the single-frame and all-frames outputs."""
    n = len(frames)
    idx = np.concatenate(
        (np.zeros(25, np.int64), np.arange(n), np.full(75 - (n % 50 or 50), n - 1))
    )
    single, many = [], []
    with torch.inference_mode():
        for s in range(0, len(idx) - 99, 50):
            logits, d = model(torch.from_numpy(frames[idx[s : s + 100]][None]))
            single.append(torch.sigmoid(logits)[0, 25:75, 0].numpy())
            many.append(torch.sigmoid(d["many_hot"])[0, 25:75, 0].numpy())
    return np.concatenate(single)[:n], np.concatenate(many)[:n]


def detections(x: np.ndarray, p: float) -> list[int]:
    """One per run of frames at or above p, at its highest frame (the first if tied)."""
    m = np.diff(np.concatenate(([False], x >= p, [False])).astype(np.int8))
    return [
        int(a + np.argmax(x[a:b]))
        for a, b in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1), strict=True)
    ]


def compare(official, model: torch.nn.Module, frames: np.ndarray) -> dict[str, object]:
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):  # its progress line
        tf_single, tf_many = official.predict_frames(frames)
    t1 = time.time()
    single, many = predict(model, frames)
    r: dict[str, object] = {
        "frames": len(frames),
        "seconds_tf": round(t1 - t0, 1),
        "seconds_torch": round(time.time() - t1, 1),
        "max_abs_single": float(np.abs(tf_single - single).max()),
        "max_abs_all": float(np.abs(tf_many - many).max()),
        "mean_abs_single": float(np.abs(tf_single - single).mean()),
    }
    same = r["max_abs_single"] < TOLERANCE and r["max_abs_all"] < TOLERANCE  # type: ignore[operator]
    for p in THRESHOLDS:
        d_tf, d_pt = detections(tf_single, p), detections(single, p)
        r[f"detections_{p}"] = len(d_tf)
        r[f"same_detections_{p}"] = d_tf == d_pt
        same = same and d_tf == d_pt
    sc_tf = official.predictions_to_scenes(tf_single, 0.5)
    sc_pt = official.predictions_to_scenes(single, 0.5)
    r["scenes_0.5"] = len(sc_tf)
    r["same_scenes_0.5"] = bool(np.array_equal(sc_tf, sc_pt))
    r["pass"] = bool(same and r["same_scenes_0.5"])
    return r


def decode(path: str, count: int, threads: int) -> tuple[np.ndarray, list[str]]:
    """The first count frames as the official extraction decodes them (inference/transnetv2.py's
    predict_video: -f rawvideo -pix_fmt rgb24 -s 48x27), quiet, with -threads."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-v",
        "error",
        "-threads",
        str(threads),
        "-i",
        path,
        "-frames:v",
        str(count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "48x27",
        "pipe:",
    ]
    r = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, *SIZE), cmd


def check(src: Path, out: Path, videos: list[list[str]], threads: int) -> dict[str, object]:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path[:0] = [str(src / "inference-pytorch"), str(src / "inference")]
    from transnetv2 import TransNetV2 as Official  # the official TensorFlow inference
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2()
    model.load_state_dict(load_file(out))  # strict: every parameter and buffer, nothing else
    model.eval()
    official = Official(str(src / WEIGHTS))
    res: dict[str, object] = {
        "torch": torch.__version__,
        "tensorflow": tf.__version__,
        "threads": threads,
        "tolerance": TOLERANCE,
    }
    rnd = np.random.default_rng(1).integers(0, 256, (1000, *SIZE), np.uint8)
    res["random"] = compare(official, model, rnd)
    log(f"random frames: {json.dumps(res['random'])}")
    for path, start, count in videos:
        a, n = int(start), int(count)
        frames, cmd = decode(path, a + n, threads)
        if len(frames) != a + n:
            raise SystemExit(f"{path}: {len(frames)} frames decoded, {a + n} wanted")
        r = {
            "file": path,
            "start": a,
            "decode": " ".join(cmd),
            **compare(official, model, frames[a:]),
        }
        res.setdefault("videos", []).append(r)  # type: ignore[union-attr]
        log(f"{path} frames {a}..{a + n - 1}: {json.dumps(r)}")
    res["pass"] = bool(res["random"]["pass"]) and all(  # type: ignore[index]
        v["pass"] for v in res.get("videos", [])
    )  # type: ignore[union-attr]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DIST)
    ap.add_argument("--work", type=Path, default=CACHE / "transnetv2")
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument(
        "--video",
        nargs=3,
        action="append",
        default=[],
        metavar=("FILE", "START", "COUNT"),
        help="an excerpt of real frames for the check",
    )
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    log(
        f"torch {torch.__version__}, numpy {np.__version__}, "
        f"safetensors {safetensors.__version__}, {a.threads} threads"
    )
    src = a.work / f"TransNetV2-{COMMIT[:7]}"
    for path, (prefix, size, sha256) in SOURCES.items():
        fetch(prefix + path, src / path, sha256, size)
    sd = convert(src)
    md = metadata()
    out = a.out / NAME
    write_safetensors(
        out,
        [Entry(k, DTYPES[v.dtype], tuple(v.shape), lambda v=v: as_bytes(v)) for k, v in sd.items()],
        md,
    )
    try:
        read_back(out, sd, md)
        log(
            f"{NAME}: {len(sd)} tensors, read back by safetensors {safetensors.__version__}: "
            f"metadata and every tensor equal to the converter's"
        )
        res = check(src, out, a.video, a.threads)
        (a.work / "check.json").write_text(json.dumps(res, indent=1) + "\n")
        if not res["pass"]:
            raise SystemExit(f"PyTorch and TensorFlow disagree: {a.work / 'check.json'}")
        log(
            f"PyTorch with {NAME} = TensorFlow within {TOLERANCE:g}, same detections "
            f"({a.work / 'check.json'})"
        )
        check_output(out, OUTPUT)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    shutil.copyfile(src / "LICENSE", a.out / LICENSE)
    log(f"{LICENSE}: TransNetV2's MIT licence, beside {NAME}")


if __name__ == "__main__":
    main()
