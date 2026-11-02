"""Where the cuts fall, over the shot detector's probabilities (cuts.py; seedvr2x/detector.py is the
library's, test_detector.py's): on a synthetic clip of four shots decoded to 48x27 as TransNetV2's
official extraction decodes (ffmpeg), with TransNetV2's weights (SEEDVR2X_MODEL_DIR holding
transnetv2.safetensors, skipped without): each cut's probability peaks on the frame before it,
above 0.5, every other frame's under 0.05; the cuts at 0.3 are the clip's; the streamed windows
equal measurement's tnet_predict; the same bits twice. On the GPU (pytest -m gpu): the GPU's
probabilities against the CPU's, the largest difference printed (-rP), the same detections at 0.3,
the same bits twice, its memory given back and torch's settings as found."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_detector import MODELS, official, settings, streamed, tnet_predict, trained  # noqa: F401

from cuts import detections
from seedvr2x.detector import BATCH

# The synthetic clip: four lavfi sources at 25 fps, each moving, 73, 77, 62 and 68 frames, its cuts
# the first frames of the shots but the first.
SHOTS = (
    ("testsrc2=s=320x180:r=25", 73),
    ("mandelbrot=s=320x180:r=25", 77),
    ("life=s=320x180:r=25:seed=1:mold=10:ratio=0.1:death_color=#202060:life_color=#e0c040", 62),
    ("sierpinski=s=320x180:r=25:seed=1", 68),
)
CUTS = [73, 150, 212]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> npt.NDArray[np.uint8]:
    """The synthetic clip, an FFV1 file, decoded with the official extraction's arguments
    (inference/transnetv2.py's predict_video: -f rawvideo -pix_fmt rgb24 -s 48x27): (280, 27, 48,
    3) uint8."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    path = tmp_path_factory.mktemp("clip") / "clip.mkv"
    inputs = [arg for source, _ in SHOTS for arg in ("-f", "lavfi", "-i", source)]
    trims = "".join(
        f"[{k}]trim=end_frame={count},setpts=PTS-STARTPTS,format=yuv420p[s{k}];"
        for k, (_, count) in enumerate(SHOTS)
    )
    joined = "".join(f"[s{k}]" for k in range(len(SHOTS)))
    graph = f"{trims}{joined}concat=n={len(SHOTS)}:v=1:a=0[v]"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[v]"),
            *("-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    decoded = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-i", str(path)),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "pipe:"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(decoded, np.uint8).reshape(-1, 27, 48, 3)
    assert len(frames) == sum(count for _, count in SHOTS)
    return frames


def test_cuts_found(trained: Path, clip: npt.NDArray[np.uint8]) -> None:  # noqa: F811
    # Each cut peaks on the frame before it, the outgoing shot's last, above 0.5; every other
    # frame stays under 0.05 (measured 2026-10-09: 0.963, 0.994 and 0.999 at the cuts, 0.024 at
    # most elsewhere).
    found = streamed(trained, clip)
    peaks = [cut - 1 for cut in CUTS]
    assert all(found[peak] > 0.5 for peak in peaks)
    assert float(np.delete(found, peaks).max()) < 0.05
    assert detections(found, 0.3) == detections(found, 0.5) == peaks


def test_measurement_reproduced(trained: Path, clip: npt.NDArray[np.uint8]) -> None:  # noqa: F811
    # Bit for bit tnet_predict's at one window a forward, the weights measurement ran; the same
    # bits twice, at the GPU's batch too.
    reference, _ = tnet_predict(official(trained), clip)
    first = streamed(trained, clip)
    assert first.tobytes() == reference.tobytes()
    assert streamed(trained, clip, (17,)).tobytes() == first.tobytes()
    batched = streamed(trained, clip, batch=BATCH)
    assert streamed(trained, clip, (250, 30), batch=BATCH).tobytes() == batched.tobytes()
    assert detections(batched, 0.3) == detections(first, 0.3)


@pytest.mark.gpu
def test_gpu_against_cpu(trained: Path, clip: npt.NDArray[np.uint8]) -> None:  # noqa: F811
    # The GPU's probabilities, at its batch, against the CPU's at one window a forward: the
    # largest difference printed, the same detections at 0.3, the same bits twice; its memory
    # given back, as torch counts it, and every setting as found.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda", 0)
    cpu = streamed(trained, clip)
    torch.cuda.init()
    found = settings()
    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    free = torch.cuda.mem_get_info(device)[0]

    gpu = streamed(trained, clip, batch=BATCH, device=device)
    again = streamed(trained, clip, batch=BATCH, device=device)
    torch.cuda.synchronize(device)
    difference = float(np.abs(gpu.astype(np.float64) - cpu).max())
    print(
        f"GPU against CPU: {difference:.3g} at most; free memory {free / 2**30:.2f} GiB before,"
        f" {torch.cuda.mem_get_info(device)[0] / 2**30:.2f} after; torch's allocated"
        f" {allocated:,} then {torch.cuda.memory_allocated(device):,} bytes, reserved"
        f" {reserved:,} then {torch.cuda.memory_reserved(device):,}"
    )
    assert detections(gpu, 0.3) == detections(cpu, 0.3) == [cut - 1 for cut in CUTS]
    assert gpu.tobytes() == again.tobytes()
    assert torch.cuda.memory_allocated(device) <= allocated
    assert torch.cuda.memory_reserved(device) <= reserved
    assert settings() == found
