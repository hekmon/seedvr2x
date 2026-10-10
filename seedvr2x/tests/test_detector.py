"""The shot detector (runtime/detector.py; DESIGN.md, Input, Shot detection): TransNetV2's official
PyTorch model, vendored byte for byte (transnetv2/), run on frames pushed as they come, in
predict_frames' windows as measurement ran them.

On the CPU, without the weights: the vendored file and its licence, their bytes as pinned; the
model built and run with the locked torch; the GPU's settings of a forward set within it and put
back exactly. With a file of the untrained model's weights, written here, and again with
TransNetV2's, the windows: pushed in chunks of odd sizes, they equal measurement's tnet_predict
over the whole array (copied below), bit for bit at one window a forward, within float rounding
at more, at every length about a window's edges; the frames kept bounded; wrong frames and a
wrong or changed file refused.

With TransNetV2's weights (SEEDVR2X_MODEL_DIR holding transnetv2.safetensors, skipped without), on
a synthetic clip of four shots decoded to 48x27 as TransNetV2's official extraction decodes
(ffmpeg): each cut's probability peaks on the frame before it, above 0.5, every other frame's
under 0.05; the streamed windows equal tnet_predict, bit for bit at one window a forward; the same
bits twice. On the GPU (pytest -m gpu): the GPU's probabilities against the CPU's, the largest
difference printed (-rP), the same detections at 0.3, the same bits twice, its memory given back
and torch's settings as found."""

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from safetensors.torch import load_file, save_file
from test_weights import as_dtype, write

import seedvr2x.transnetv2
from seedvr2x.runtime import detector, pull
from seedvr2x.runtime.detector import BATCH, STEP, WINDOW, Detector
from seedvr2x.runtime.weights import ModelError
from seedvr2x.transnetv2.transnetv2_pytorch import TransNetV2

VENDORED = Path(seedvr2x.transnetv2.__file__).parent
# TransNetV2's files at 85cef72 (github.com/soCzech/TransNetV2), as models/transnetv2_weights.py
# pins them (SOURCES), by their name here: size and SHA-256.
PINS = {
    "transnetv2_pytorch.py": (
        12_475,
        "f7c1d437465579a8ec28a5add19853d2cb2755248ea4a4207678210a609428e1",
    ),
    "LICENSE": (1_072, "a8d7a056688ccedebe89f18fd60f1a47128df94cb82669cd02459934919cbb6f"),
}
CPU = torch.device("cpu")
# Chunks of odd sizes, cycled, an empty one among them.
ODD = (1, 0, 7, 13, 64, 3, 101, 29)


def test_vendored_as_pinned() -> None:
    # TransNetV2's own bytes, unchanged: a change is deliberate, its pin with it (AGENTS.md, The
    # vendored TransNetV2); nothing else there but seedvr2x's __init__.py.
    for name, (size, digest) in PINS.items():
        data = (VENDORED / name).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (size, digest), name
    assert {path.name for path in VENDORED.iterdir() if path.name != "__pycache__"} == {
        *PINS,
        "__init__.py",
    }
    licence = (VENDORED / "LICENSE").read_text()
    assert licence.startswith("MIT License\n\nCopyright (c) 2020 Tomáš Souček\n")


def test_vendored_runs() -> None:
    # Built and run with the locked torch, untrained: a window of 100 frames in, a logit per
    # frame out for each head, in eval mode as built.
    model = TransNetV2()
    assert not model.training
    with torch.inference_mode():
        logits, heads = model(torch.zeros((2, WINDOW, 27, 48, 3), dtype=torch.uint8))
    assert logits.shape == heads["many_hot"].shape == (2, WINDOW, 1)
    assert logits.dtype == torch.float32


def settings() -> dict[str, object]:
    """The settings of torch's that _exact sets, read through its legacy flags and its newer
    fp32_precision ones, the CPU's (mkldnn) and the generic one too."""
    # torch.backends and torch.backends.mkldnn are modules torch replaces as it imports them,
    # whose attributes typing doesn't see.
    backends: Any = torch.backends
    cudnn, matmul, mkldnn = backends.cudnn, backends.cuda.matmul, backends.mkldnn
    return {
        "cudnn": (cudnn.enabled, cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32),
        "cudnn precision": (
            cudnn.fp32_precision,
            cudnn.conv.fp32_precision,
            cudnn.rnn.fp32_precision,
        ),
        "matmul": (matmul.allow_tf32, matmul.fp32_precision),
        "generic": backends.fp32_precision,
        "mkldnn": (mkldnn.matmul.fp32_precision, mkldnn.conv.fp32_precision),
    }


@pytest.mark.parametrize("tf32", [False, True], ids=["defaults", "cublas-tf32"])
def test_exact_within_a_forward(tf32: bool) -> None:
    # On CUDA, within the block: cuDNN's deterministic algorithms, no benchmark and no TF32, nor
    # cuBLAS's; after it, every setting as found, with torch's defaults and with cuBLAS's TF32 on
    # (as TORCH_ALLOW_TF32_CUBLAS_OVERRIDE sets it). Nothing on the CPU. Torch's flags are read
    # and set without a GPU.
    matmul = torch.backends.cuda.matmul
    precision = matmul.fp32_precision
    if tf32:
        matmul.allow_tf32 = True
    try:
        found = settings()
        with detector._exact(torch.device("cuda", 0)):
            inside = settings()
        assert settings() == found
        with detector._exact(CPU):
            assert settings() == found
    finally:
        if tf32:
            matmul.allow_tf32 = False
            matmul.fp32_precision = precision  # torch's default, "none", as before the test
    assert inside["cudnn"] == (True, False, True, False)
    assert inside["matmul"] == (False, "ieee" if tf32 else precision)
    assert found["matmul"] == (tf32, "tf32" if tf32 else precision)


# measurement's run of TransNetV2, research/scripts/scd_scores.py's tnet_predict (its progress log
# left out), itself TransNetV2's predict_frames (inference/transnetv2.py at 85cef72, MIT) on the
# PyTorch model: the reference the streamed windows are held to.
TNET_WINDOW, TNET_STEP, TNET_PAD = 100, 50, 25


def tnet_predict(model: Any, frames: Any, batch: int = 1) -> tuple[Any, Any]:
    """TransNetV2's predict_frames with the PyTorch model: 25 copies of the first frame before, 25
    to 74 of the last after, windows of 100 frames every 50 (batch windows per forward;
    officially one), frames 25-74 of each kept; sigmoid of the single-frame and all-frames
    logits."""
    n = len(frames)
    pad_end = TNET_PAD + TNET_STEP - (n % TNET_STEP or TNET_STEP)
    idx = np.concatenate((np.zeros(TNET_PAD, np.int64), np.arange(n), np.full(pad_end, n - 1)))
    starts = list(range(0, len(idx) - TNET_WINDOW + 1, TNET_STEP))
    one = np.empty(len(starts) * TNET_STEP, np.float32)
    many = np.empty_like(one)
    with torch.inference_mode():
        for b in range(0, len(starts), batch):
            sl = starts[b : b + batch]
            x = torch.from_numpy(np.stack([frames[idx[s : s + TNET_WINDOW]] for s in sl]))
            logits, d = model(x)
            keep = slice(TNET_PAD, TNET_PAD + TNET_STEP)
            one[b * TNET_STEP : (b + len(sl)) * TNET_STEP] = (
                torch.sigmoid(logits)[:, keep, 0].reshape(-1).numpy()
            )
            many[b * TNET_STEP : (b + len(sl)) * TNET_STEP] = (
                torch.sigmoid(d["many_hot"])[:, keep, 0].reshape(-1).numpy()
            )
    return one[:n], many[:n]


def tnet_detections(x: npt.NDArray[np.float32], p: float) -> list[int]:
    """One detection per run of frames >= p, at the run's highest frame (the first if tied):
    measurement's tnet_detections (scd_scores.py), DESIGN.md's detection, until the cuts' commit
    brings seedvr2x's own."""
    m = np.concatenate(([False], np.nan_to_num(x, nan=-1.0) >= p, [False]))
    d = np.diff(m.astype(np.int8))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [int(a + np.argmax(x[a:b])) for a, b in zip(starts, ends, strict=True)]


def official(path: Path) -> torch.nn.Module:
    """TransNetV2 as measurement built it (scd_scores.py's tnet_model): the official model, the
    weights loaded into it, on the CPU, in eval mode."""
    model = TransNetV2()
    model.load_state_dict(load_file(path))
    return model.eval()


def streamed(
    path: Path,
    frames: npt.NDArray[np.uint8],
    chunks: Sequence[int] = ODD,
    batch: int = 1,
    device: torch.device = CPU,
) -> npt.NDArray[np.float32]:
    """frames pushed into a Detector in chunks of these sizes, cycled, then finished: every
    frame's probability. Each push gives whole windows' frames, and keeps fewer than
    STEP * (batch + 1) frames; finish gives the rest, once."""
    out: list[npt.NDArray[np.float32]] = []
    with Detector(path, device, batch=batch) as found:
        start, turn = 0, 0
        while start < len(frames):
            size = chunks[turn % len(chunks)]
            turn += 1
            got = found.push(frames[start : start + size])
            start += size
            assert got.dtype == np.float32 and len(got) % STEP == 0
            assert len(found._held) < STEP * (batch + 1)
            out.append(got)
            assert found.scored == sum(len(each) for each in out)
        out.append(found.finish())
        assert len(found.finish()) == 0
        assert found.pushed == found.scored == len(frames)
    result = np.concatenate(out)
    assert result.shape == (len(frames),)
    return result


def untrained_weights(directory: Path) -> Path:
    """A transnetv2.safetensors of the untrained model's weights, in directory, drawn from a
    generator of its own: no cut to find, but every window's output depends on its frames as the
    trained one's."""
    path = directory / "transnetv2.safetensors"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TransNetV2()
    save_file(model.state_dict(), path)
    return path


@pytest.fixture(scope="module")
def untrained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The untrained model's weights (untrained_weights)."""
    return untrained_weights(tmp_path_factory.mktemp("untrained"))


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")


@pytest.fixture(scope="module")
def trained() -> Path:
    """TransNetV2's weights in SEEDVR2X_MODEL_DIR, checked by their pin; skipped without."""
    if not MODELS:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    path = Path(MODELS) / pull.DETECTOR
    if not path.is_file():
        pytest.skip(f"{pull.DETECTOR}: not in SEEDVR2X_MODEL_DIR")
    pull.check_pin(pull.DETECTOR, path, pull.PINNED[pull.DETECTOR], pull.Hashes(), None)
    return path


@pytest.fixture(params=["untrained", "trained"])
def weights_file(request: pytest.FixtureRequest) -> Path:
    """The untrained model's weights, then TransNetV2's (trained, skipped without them)."""
    return request.getfixturevalue(request.param)


def frames_of(count: int, seed: int) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 256, (count, 27, 48, 3), np.uint8)


def test_windows_as_measurement_ran_them(weights_file: Path) -> None:
    # Bit for bit tnet_predict's at one window a forward; at more, within float rounding, the
    # largest difference printed (-rP).
    frames = frames_of(437, 1)
    reference, _ = tnet_predict(official(weights_file), frames)
    assert streamed(weights_file, frames).tobytes() == reference.tobytes()
    for batch in (2, 3, BATCH):
        found = streamed(weights_file, frames, batch=batch)
        difference = float(np.abs(found.astype(np.float64) - reference).max())
        print(f"batch {batch}: {difference:.3g} at most from one window a forward")
        assert difference < 1e-6
    # However the frames are pushed, the windows run in the same whole batches: the same bits. Run
    # as their frames came, the batches cut by the pushes rounded otherwise, 5.96e-08 apart on the
    # untrained model's weights at 4 windows a forward (a review's mutation, 2026-10-09).
    batched = streamed(weights_file, frames, batch=BATCH)
    for chunks in ((1,), (250, 30), (len(frames),)):
        again = streamed(weights_file, frames, chunks, batch=BATCH)
        assert again.tobytes() == batched.tobytes(), chunks


@pytest.mark.parametrize("count", [1, 49, 50, 51, 99, 100, 101])
def test_lengths_about_the_windows(weights_file: Path, count: int) -> None:
    # Every length about a window's edges, pushed a frame at a time and at once: bit for bit
    # tnet_predict's, one probability per frame.
    frames = frames_of(count, count)
    reference, _ = tnet_predict(official(weights_file), frames)
    for chunks in ((1,), (count,)):
        assert streamed(weights_file, frames, chunks).tobytes() == reference.tobytes()


def test_frames_refused(untrained: Path) -> None:
    # Frames of another shape or dtype; none, given nothing; frames after finish, or once closed.
    found = Detector(untrained, CPU)
    for frames in (np.zeros((2, 27, 48, 3), np.float32), np.zeros((2, 48, 27, 3), np.uint8)):
        with pytest.raises(ValueError, match="where TransNetV2 takes"):
            found.push(frames)  # pyright: ignore[reportArgumentType]
    assert found.push(np.zeros((0, 27, 48, 3), np.uint8)).shape == (0,)
    assert found.finish().shape == (0,)
    with pytest.raises(ValueError, match="after finish"):
        found.push(frames_of(1, 0))
    found.close()
    found.close()
    with pytest.raises(ValueError, match="closed"):
        found.finish()
    with pytest.raises(ValueError, match="at least 1"):
        Detector(untrained, CPU, batch=0)


def test_model_file_checked(tmp_path: Path, untrained: Path) -> None:
    # The file is checked before anything loads: another model refused, saying what it is; a
    # file changed since its hash refused, before its load.
    vae = write(tmp_path / "transnetv2.safetensors", as_dtype("vae", "F16"))
    with pytest.raises(ModelError, match="SeedVR2's VAE in fp16, given as the shot detector"):
        Detector(vae, CPU)
    copy = tmp_path / "copy" / "transnetv2.safetensors"
    copy.parent.mkdir()
    shutil.copyfile(untrained, copy)
    hashes = pull.Hashes()
    hashes.sha256(copy)
    Detector(copy, CPU, hashes=hashes).close()
    state = copy.stat()
    os.utime(copy, ns=(state.st_atime_ns, state.st_mtime_ns + 1))
    with pytest.raises(ModelError, match="changed since its SHA-256 was read"):
        Detector(copy, CPU, hashes=hashes)


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


def test_cuts_found(trained: Path, clip: npt.NDArray[np.uint8]) -> None:
    # Each cut peaks on the frame before it, the outgoing shot's last, above 0.5; every other
    # frame stays under 0.05 (measured 2026-10-09: 0.963, 0.994 and 0.999 at the cuts, 0.024 at
    # most elsewhere).
    found = streamed(trained, clip)
    peaks = [cut - 1 for cut in CUTS]
    assert all(found[peak] > 0.5 for peak in peaks)
    assert float(np.delete(found, peaks).max()) < 0.05
    assert tnet_detections(found, 0.3) == tnet_detections(found, 0.5) == peaks


def test_measurement_reproduced(trained: Path, clip: npt.NDArray[np.uint8]) -> None:
    # Bit for bit tnet_predict's at one window a forward, the weights measurement ran; the same
    # bits twice, at the GPU's batch too.
    reference, _ = tnet_predict(official(trained), clip)
    first = streamed(trained, clip)
    assert first.tobytes() == reference.tobytes()
    assert streamed(trained, clip, (17,)).tobytes() == first.tobytes()
    batched = streamed(trained, clip, batch=BATCH)
    assert streamed(trained, clip, (250, 30), batch=BATCH).tobytes() == batched.tobytes()
    assert tnet_detections(batched, 0.3) == tnet_detections(first, 0.3)


# A Detector built on the GPU before anything else of its process touched CUDA, then run.
FIRST_ON_THE_GPU = """
import sys
from pathlib import Path

import numpy as np
import torch

from seedvr2x.runtime.detector import Detector

assert not torch.cuda.is_initialized()
with Detector(Path(sys.argv[1]), torch.device("cuda", 0)) as found:
    frames = np.zeros((120, 27, 48, 3), np.uint8)
    print(len(found.push(frames)) + len(found.finish()))
"""


@pytest.mark.gpu
@pytest.mark.parametrize("allocator", ["backend:cudaMallocAsync", "backend:native"])
def test_gpu_detector_first_on_the_gpu(untrained: Path, allocator: str) -> None:
    # The detector doesn't depend on who initialised CUDA: built first thing in a process of its
    # own, under the command line's allocator (cli.main sets cudaMallocAsync) and under torch's
    # own. Under cudaMallocAsync, the reset of torch's peak statistics before CUDA's
    # initialisation raised "RuntimeError: Invalid device argument." on the GPU box
    # (2026-10-10, runtime/detector.py), which no test saw: each initialised CUDA before it built
    # one, as the command line's runs do. With the untrained model's weights: no file needed.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    environment = os.environ | {"PYTORCH_CUDA_ALLOC_CONF": allocator}
    result = subprocess.run(
        [sys.executable, "-c", FIRST_ON_THE_GPU, str(untrained)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["120"]


@pytest.mark.gpu
def test_gpu_against_cpu(trained: Path, clip: npt.NDArray[np.uint8]) -> None:
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
    assert tnet_detections(gpu, 0.3) == tnet_detections(cpu, 0.3) == [cut - 1 for cut in CUTS]
    assert gpu.tobytes() == again.tobytes()
    assert torch.cuda.memory_allocated(device) <= allocated
    assert torch.cuda.memory_reserved(device) <= reserved
    assert settings() == found
