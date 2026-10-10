"""The shot detector, TransNetV2 (DESIGN.md, Input, Shot detection): its official PyTorch model,
vendored in transnetv2/, with seedvr2x's own weights (transnetv2.safetensors, pinned in
runtime/pull.py), run on a source's frames as the first pass decodes them, each frame's
probability given as soon as the window keeping it has run.

The windows are TransNetV2's predict_frames' (inference/transnetv2.py at 85cef72), as measurement
ran it on the CPU (research/scripts/scd_scores.py's tnet_predict; research/docs/
scene-detection.md, TransNetV2), followed exactly: the frames padded with PAD copies of the first
before them and copies of the last after them, as many as fill the last window; windows of WINDOW
frames every STEP, each keeping its frames PAD to PAD + STEP - 1; the sigmoid of the single-frame
head, float32. Window k thus scores frames STEP * k to STEP * k + STEP - 1 from frames
STEP * k - PAD to STEP * k + WINDOW - PAD - 1, and can run once that last frame has come. A cut at
frame c peaks at c - 1, the outgoing shot's last frame (offset -1 on 98% of measurement's sure
cuts)."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import torch

# Strict typing finds load_file's signature partly unknown, for its bare os.PathLike.
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]

from seedvr2x.runtime import weights
from seedvr2x.runtime.pull import Hashes
from seedvr2x.transnetv2.transnetv2_pytorch import TransNetV2

logger = logging.getLogger(__name__)

# TransNetV2's frames: 48x27 RGB, as its official extraction scales them (DESIGN.md, Shot
# detection), (HEIGHT, WIDTH, 3) uint8, the shape its forward asserts.
HEIGHT, WIDTH = 27, 48
# predict_frames' windows: WINDOW frames every STEP, each keeping its frames PAD to PAD + STEP - 1,
# the first window's first PAD frames copies of the source's first.
WINDOW, STEP, PAD = 100, 50, 25
# Windows per forward on the GPU. Provisional (implementation, 2026-10-09; DESIGN.md doesn't say):
# a forward's peak, measured on the CPU as its process's peak resident memory (2026-10-09), takes
# about 125 MiB a window, 0.5 GiB for 4 (167 MiB for 1, 496 for 4, 1,012 for 8, 2,041 for 16),
# little on the first pass's idle GPU, whose memory the detector gives back before the planner
# reads its budget; 4 windows a forward to spare the GPU launches, its time measured on the box.
# On the CPU, a batch of 2 to 16 changes a probability by float rounding alone from one window a
# forward: 3.0e-8 at most with TransNetV2's weights on the tests' clip and on random frames,
# 6.0e-8 with the untrained model's, near 0.5, where float32's step is 6.0e-8
# (tests/test_detector.py).
BATCH = 4


class Detector:
    """TransNetV2 on a source's frames, pushed in order (push), the last ones flushed (finish),
    then its memory given back (close): each frame's single-frame probability, float32 in [0, 1],
    given once, in order, push's and finish's together one per frame pushed.

    The windows run batch by batch, batch windows per forward, as their last frames come; finish
    runs those left. Only the frames the next windows read are kept: from the first frame of the
    next window to run (PAD before the first frame it scores) to the last pushed, fewer than
    STEP * (batch + 1) between two pushes, 3,888 bytes a frame (under 1 MB at the GPU's batch),
    where a 2-hour film at 24 fps would take 672 MB whole.

    The device is the caller's: CUDA when a job runs, which the CLI refuses without; the CPU in
    the tests, deterministic run to run. On CUDA each forward computes in float32, with
    deterministic cuDNN algorithms and no TF32, within the forward only (_exact), so that nothing
    else of the job runs otherwise.

    One thread drives it: push, finish and close aren't thread-safe, and _exact's settings are
    the process's while a forward runs, so no other CUDA work may run then."""

    def __init__(
        self, path: Path, device: torch.device, *, batch: int = BATCH, hashes: Hashes | None = None
    ) -> None:
        """Load TransNetV2 with the weights in the file at path, checked first (weights.check:
        ModelError says what the file is otherwise), onto device, to run batch windows per forward
        (from 1). A file hashed this run (hashes) must still be the one hashed (Hashes.unchanged,
        ModelError), before the load and right after it."""
        if batch < 1:
            raise ValueError(f"batch {batch}: at least 1 window per forward")
        weights.check(path, "detector")
        if hashes is not None:
            hashes.unchanged(path)
        if device.type == "cuda":
            # Its peak, from here, said as close gives it back (DESIGN.md, Memory planner, Budget).
            # CUDA initialised first, whoever built the detector: torch initialises it at the
            # first allocation, and the reset of its statistics asks the allocator at once
            # (torch/cuda/memory.py, where memory_stats returns {} until then), which
            # cudaMallocAsync, the command line's allocator (cli.main), refuses for a device it
            # hasn't met: "RuntimeError: Invalid device argument." on the GPU box, a Detector
            # built before anything else touched the GPU (2026-10-10; the command line's own
            # runs record their environment, the GPU's, before).
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        # Built on the meta device, then given memory of its own on device and the file's values:
        # no random initialisation, which would draw from torch's generator, and every tensor the
        # file holds, nothing else (strict).
        with torch.device("meta"):
            model = TransNetV2()
        model.to_empty(device=device)
        state = load_file(path)
        model.load_state_dict(state, strict=True)
        del state
        if hashes is not None:
            hashes.unchanged(path)
        self._model: torch.nn.Module | None = model.requires_grad_(False).eval()
        self._device = device
        self._batch = batch
        # The frames kept, from frame self._first on, and how many were pushed.
        self._held = np.empty((0, HEIGHT, WIDTH, 3), np.uint8)
        self._first = 0
        self._pushed = 0
        self._windows = 0  # the windows run
        self._finished = False
        logger.info(
            "shot detector: TransNetV2 (%s) on %s, %d windows of %d frames per forward",
            path.name,
            device,
            batch,
            WINDOW,
        )

    @property
    def pushed(self) -> int:
        """The frames pushed so far."""
        return self._pushed

    @property
    def scored(self) -> int:
        """The frames whose probabilities were given so far, from the first."""
        return min(self._windows * STEP, self._pushed)

    def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Take the source's next frames, (n, 27, 48, 3) uint8 RGB, any n from 0, and give the
        probabilities of the frames the windows that could run then score: (m,) float32 in [0, 1],
        the frames from scored on, m a multiple of STEP, 0 until batch windows can run.
        ValueError for frames of another shape or dtype, or after finish."""
        model = self._live()
        if self._finished:
            raise ValueError("frames pushed after finish")
        if frames.dtype != np.uint8 or frames.shape[1:] != (HEIGHT, WIDTH, 3):
            raise ValueError(
                f"frames {frames.shape} {frames.dtype}, where TransNetV2 takes"
                f" (n, {HEIGHT}, {WIDTH}, 3) uint8"
            )
        if len(frames) == 0:
            return np.empty(0, np.float32)
        self._held = np.concatenate((self._held, frames))
        self._pushed += len(frames)
        # Window k's last frame is STEP * k + WINDOW - PAD - 1: the windows whose last frames have
        # come, run in whole batches, so that the batches, which round differently (on the CPU,
        # one window from two), don't depend on how the frames were pushed.
        ready = max(0, (self._pushed - (WINDOW - PAD) + STEP) // STEP)
        whole = self._windows + (ready - self._windows) // self._batch * self._batch
        return self._run(model, whole)

    def finish(self) -> npt.NDArray[np.float32]:
        """Run the windows left, the last one filled with copies of the last frame, and give their
        frames' probabilities: (m,) float32 in [0, 1], the frames from scored to the last pushed;
        after which every frame pushed has had its probability, once. Nothing can be pushed
        after; finish again gives nothing more."""
        model = self._live()
        self._finished = True
        return self._run(model, -(-self._pushed // STEP))

    def close(self) -> None:
        """Give the model's memory back: on the GPU, every block torch's allocator holds for it
        (torch.cuda.empty_cache) and the workspaces it keeps for cuBLAS, so that the planner's
        budget, read before anything else allocates on the GPU, isn't taken by the detector
        (DESIGN.md, Memory planner, Budget). What the process then keeps on the GPU isn't the
        detector's to give back: the CUDA context, cuDNN's and cuBLAS's handles and the kernels
        they loaded, which stay until the process ends, outside torch's allocator, and which the
        models' own load would have brought anyway. On the GPU box, the free memory was 94.50 GiB
        with CUDA initialised and 94.34 after two detections, torch's counters at 0 both times
        (2026-10-09): 0.16 GiB of the libraries' own; a budget read then is the smaller one, on
        the safe side. The detector takes nothing more after; close again does nothing."""
        if self._model is None:
            return
        self._model = None
        self._held = np.empty((0, HEIGHT, WIDTH, 3), np.uint8)
        if self._device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self._device)
            # torch keeps the workspace it gives cuBLAS for a handle and stream, allocated from its
            # allocator at the first matrix product, until told otherwise: empty_cache frees what
            # is cached, not what is held. torch's own memory-leak check clears them so before it
            # compares (torch/testing/_internal/common_utils.py, CudaMemoryLeakCheck).
            torch._C._cuda_clearCublasWorkspaces()  # pyright: ignore[reportPrivateUsage]
            torch.cuda.empty_cache()
            logger.info(
                "shot detector: VRAM peak %.2f GiB, given back (%.0f MiB still reserved by torch)",
                peak / 2**30,
                torch.cuda.memory_reserved(self._device) / 2**20,
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _live(self) -> torch.nn.Module:
        """The model, ValueError once closed."""
        if self._model is None:
            raise ValueError("the shot detector is closed")
        return self._model

    def _run(self, model: torch.nn.Module, upto: int) -> npt.NDArray[np.float32]:
        """Run the windows from the next one to upto, excluded, batch windows per forward, and
        give the probabilities of the frames they score, but those past the last frame pushed,
        which only the last window holds. The frames no later window reads are then dropped."""
        start = self._windows
        out: list[npt.NDArray[np.float32]] = []
        while self._windows < upto:
            count = min(self._batch, upto - self._windows)
            # Each window's frames: positions PAD before its first frame scored, clamped to the
            # first and last frames pushed, as predict_frames pads them.
            first = (self._windows + np.arange(count)) * STEP - PAD
            frames = np.clip(first[:, None] + np.arange(WINDOW), 0, self._pushed - 1)
            windows = torch.as_tensor(self._held[frames - self._first], device=self._device)
            with _exact(self._device), torch.inference_mode():
                logits = cast(torch.Tensor, model(windows)[0])
                kept = torch.sigmoid(logits)[:, PAD : PAD + STEP, 0].reshape(-1)
            out.append(kept.cpu().numpy())
            self._windows += count
        keep_from = max(0, self._windows * STEP - PAD)
        if keep_from > self._first:
            # A copy, so that the frames dropped go with the array they were pushed in.
            self._held = self._held[keep_from - self._first :].copy()
            self._first = keep_from
        if not out:
            return np.empty(0, np.float32)
        return np.concatenate(out)[: max(0, self._pushed - start * STEP)]


@contextmanager
def _exact(device: torch.device) -> Generator[None]:
    """On CUDA, float32 computed in float32 by deterministic algorithms, within the block only,
    the settings found put back after it, exactly: cuDNN's deterministic convolutions, without
    benchmarking or TF32; and cuBLAS without TF32. Nothing on the CPU, deterministic as it is."""
    if device.type != "cuda":
        yield
        return
    # torch 2.14.1's defaults: cuDNN may compute float32 convolutions in TF32
    # (torch.backends.cudnn.allow_tf32 True), cuBLAS may not (torch.backends.cuda.matmul.allow_tf32
    # False, unless TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set). cuBLAS's flag is set only when on,
    # through its own setter and back: torch tells its legacy flags from its newer fp32_precision
    # ones, and a write through one API then a read through the other raises (the "mix of the
    # legacy and new APIs" RuntimeError), where these left every flag as found (measured
    # 2026-10-09, with the defaults, TF32 set on and the override). cuDNN's are torch's own
    # scope. Provisional (implementation, 2026-10-09; DESIGN.md says deterministic, not how).
    matmul = torch.backends.cuda.matmul
    tf32 = bool(matmul.allow_tf32)
    if tf32:
        matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ):
            yield
    finally:
        if tf32:
            matmul.allow_tf32 = True
