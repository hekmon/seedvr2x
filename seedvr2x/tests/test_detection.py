"""The shot detector in the first pass (media/scan.py, media/source.py's first_pass; DESIGN.md,
Input, Shot detection): every frame the first pass decodes, scaled to 48x27 as TransNetV2's
official extraction scales them, on one thread, fed to the detector as it comes, its
probabilities kept in the frame index.

On the CPU, a recording stand-in in the detector's place: the frames it gets are the official
extraction command's (ffmpeg -i F -f rawvideo -pix_fmt rgb24 -s 48x27 pipe:), byte for byte, on
constant-rate samples, which give the same bytes with their scaler forced to 1 and 16 threads; as
many as the index's frames, in order, each probability recorded; the scaler's filter graph on one
thread of ffmpeg's; on a stream tagged from its 15th frame, where the graph is rebuilt, the bytes
of measurement's command keeping one graph (-reinit_filter 0) and of the official one. A detector
slow on purpose holds ffmpeg back, the pass ending with every frame once; one that stops moving
is stopped by the watchdog, which says so at once; a thread reading one of ffmpeg's pipes that
raises stops the pass at once too, the detector's queue full or not. With the untrained model's
weights (test_detector.py's), the probabilities recorded are runtime/detector.py's on the official
extraction's frames, bit for bit.

With TransNetV2's weights (SEEDVR2X_MODEL_DIR holding transnetv2.safetensors, skipped without),
on test_detector.py's clip of four shots: the probabilities the first pass records are
runtime/detector.py's on the official extraction's frames, bit for bit at the same batch, peaking
on the frame before each cut. On the GPU (pytest -m gpu): the first pass's probabilities against
the CPU's, on that clip and on milestone 1's input (SEEDVR2X_REFERENCE_DIR's m1/input_rgb.mkv, as
tests/test_regression.py reads it), the largest difference printed (-rP), the same detections at
0.3, the GPU's memory given back."""

import logging
import os
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from test_detector import CUTS, SHOTS, streamed, tnet_detections, tnet_predict, untrained_weights
from test_detector import official as measurements_model
from test_index import make
from test_probe import has_encoder

from seedvr2x.media import ffmpeg
from seedvr2x.media import scan as first
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.runtime import pull
from seedvr2x.runtime.detector import BATCH, Detector


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

CPU = torch.device("cpu")
HEIGHT, WIDTH = 27, 48


class Recorder:
    """A stand-in for the shot detector: keeps every frame pushed, in order, and gives each one's
    probability at once, its mean component value over 255; `pause`, seconds, slept at each push
    (each push's, or the first's alone with `once`); with `count`, ffmpeg's threads counted at
    the first push (ffmpeg_threads), for the test that looks at them alone."""

    def __init__(self, pause: float = 0.0, once: bool = False, count: bool = False) -> None:
        self.pushes: list[npt.NDArray[np.uint8]] = []
        self.pause, self.once, self.count = pause, once, count
        self.threads: Counter[str] | None = None  # ffmpeg's, by name, at the first push

    @property
    def frames(self) -> npt.NDArray[np.uint8]:
        return (
            np.concatenate(self.pushes)
            if self.pushes
            else np.empty((0, HEIGHT, WIDTH, 3), np.uint8)
        )

    def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        if self.count and self.threads is None:
            self.threads = ffmpeg_threads()
        if self.pause and (not self.once or not self.pushes):
            time.sleep(self.pause)
        self.pushes.append(frames.copy())
        return (frames.mean(axis=(1, 2, 3), dtype=np.float64) / 255).astype(np.float32)

    def finish(self) -> npt.NDArray[np.float32]:
        return np.empty(0, np.float32)


def ffmpeg_threads() -> Counter[str]:
    """The threads of the ffmpeg this process runs, counted by the names ffmpeg gives them
    (fftools: a filter graph's are named after it, "vf#<output>:<stream>"); empty when none
    runs. Linux's /proc, whose files of a thread or process that ended meanwhile fail each in its
    way: FileNotFoundError opening one, ProcessLookupError (ESRCH) reading one opened before its
    thread ended, which failed 2 runs in 10 (a review's count): any OSError is one gone."""
    found: Counter[str] = Counter()
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            children = (task / "children").read_text().split()
        except OSError:
            continue  # a thread ended meanwhile
        for child in children:
            try:
                if Path(f"/proc/{child}/comm").read_text().strip() != "ffmpeg":
                    continue
                for thread in Path(f"/proc/{child}/task").iterdir():
                    try:
                        found[(thread / "comm").read_text().strip()] += 1
                    except OSError:
                        continue  # that thread ended meanwhile, the others still counted
            except OSError:
                continue  # ended meanwhile
    return found


def ffmpeg_alive() -> bool:
    """Whether an ffmpeg this process runs is alive, not a zombie whose exit status waits to be
    read (/proc's state Z). Linux's /proc."""
    for task in Path(f"/proc/{os.getpid()}/task").iterdir():
        try:
            children = (task / "children").read_text().split()
        except OSError:
            continue  # a thread ended meanwhile (ffmpeg_threads)
        for child in children:
            try:
                state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if Path(f"/proc/{child}/comm").read_text().strip() == "ffmpeg" and state != "Z":
                    return True
            except OSError:
                continue  # ended meanwhile
    return False


def official(path: Path, *options: str) -> npt.NDArray[np.uint8]:
    """The official extraction's frames (inference/transnetv2.py's predict_video at 85cef72:
    ffmpeg-python's -i FILE -f rawvideo -pix_fmt rgb24 -s 48x27 pipe:, quieter), options added to
    its output's, as measurement's tnet_official_extraction ran it."""
    decoded = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-i", str(path), *options),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "pipe:"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(decoded, np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """test_detector.py's clip of four lavfi shots at 25 fps, 280 frames, its cuts at CUTS, as an
    FFV1 file."""
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
    return path


@pytest.fixture(scope="module")
def x264(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1280x720 H.264 file at 24000/1001, 120 frames of moving noise, B-frames and all."""
    if not has_encoder("libx264"):
        pytest.skip("needs ffmpeg with libx264")
    path = tmp_path_factory.mktemp("x264") / "x264.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i"),
            "mandelbrot=s=1280x720:r=24000/1001,noise=alls=20:allf=t",
            *("-frames:v", "120", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20"),
            *("-pix_fmt", "yuv420p", str(path)),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize("kind", ["FFV1", "x264"])
def test_frames_as_the_official_extraction(request: pytest.FixtureRequest, kind: str) -> None:
    # The detector's frames, the official command's byte for byte, which its scaler forced to 1
    # thread and to 16 doesn't change; every frame the index holds, each pushed once, in order,
    # each one's probability recorded.
    path: Path = request.getfixturevalue("clip" if kind == "FFV1" else "x264")
    recorder = Recorder()
    scanned = first.scan(path, recorder)
    found = scanned.index
    assert found is not None and found.probabilities is not None
    reference = official(path)
    assert len(recorder.frames) == len(reference) == scanned.frames == found.frames
    assert recorder.frames.tobytes() == reference.tobytes()
    for threads in ("1", "16"):
        assert official(path, "-threads", threads).tobytes() == reference.tobytes(), threads
    means = (reference.mean(axis=(1, 2, 3), dtype=np.float64) / 255).astype(np.float32)
    assert found.probabilities.dtype == np.float32
    assert found.probabilities.tobytes() == means.tobytes()


def test_scaled_on_one_thread(clip: Path) -> None:
    # The detector's output, the first pass's third (scan.command), its filter graph named vf#2:0
    # by ffmpeg, which inserts the scaler there: one thread, where ffmpeg would give it one per
    # CPU and swscale as many again (media/scan.py, _detector_output). The other graphs as ffmpeg
    # sets them.
    recorder = Recorder(count=True)
    first.scan(clip, recorder)
    assert recorder.threads is not None
    graphs = {name: count for name, count in recorder.threads.items() if name.startswith("vf#")}
    assert set(graphs) == {"vf#0:0", "vf#1:0", "vf#2:0"}, recorder.threads
    assert graphs["vf#2:0"] == 1, graphs


def test_without_detector_no_third_output(clip: Path) -> None:
    # --cuts: the first pass as before the detector, two outputs, no probabilities.
    arguments = first.command(clip)
    assert arguments.count("-map") == 2 and "rgb24" not in arguments
    with_frames = first.command(clip, 3)
    assert with_frames[: len(arguments)] == arguments
    assert with_frames[len(arguments) :] == [
        *("-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux"),
        *("-map_metadata", "-1", "-map_chapters", "-1"),
        *("-threads", "1", "-s", "48x27", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:3"),
    ]
    # No output takes the source's metadata, whose text ffmpeg would print among the lines the
    # pass reads (ffmpeg.NO_METADATA): each of the three.
    assert with_frames.count("-map_metadata") == with_frames.count("-map_chapters") == 3
    scanned = first.scan(clip)
    assert scanned.index is not None and scanned.index.probabilities is None


def test_a_rebuilt_graph_gives_the_same_frames(tmp_path: Path) -> None:
    # A stream tagged BT.601 from its 15th frame (test_index.py's), where ffmpeg rebuilds every
    # filter graph: the detector's frames, scaled by a rebuilt scaler that takes the frames' new
    # tags, are the bytes of measurement's command keeping one graph for every frame, as it ran on
    # its DVD (-reinit_filter 0: research/scripts/scd_scores.py's decode_cmd), and the official
    # command's.
    path = make(tmp_path, "tags.mkv")
    recorder = Recorder()
    scanned = first.scan(path, recorder)
    assert scanned.frames == len(recorder.frames) == 120
    measurement = subprocess.run(
        [
            *("ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-threads", "16"),
            *("-filter_threads", "16", "-reinit_filter", "0", "-i", str(path)),
            *("-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough"),
            *("-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "pipe:1"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert recorder.frames.tobytes() == measurement == official(path).tobytes()


def test_slow_detector_holds_ffmpeg_back(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A detector slower than the decode, two reads of 7 frames held at most: ffmpeg waits for it,
    # the frames read ahead of it never more than those two, the one being read and the one
    # pushed, its other outputs read all the while; the pass ends with every frame pushed once,
    # in order, in reads of 7.
    monkeypatch.setattr(first, "HELD", 2)
    monkeypatch.setattr(first, "CHUNK", 7)
    made: list[Any] = []
    frames_class = first._Frames

    def making(*arguments: Any) -> Any:
        made.append(frames_class(*arguments))
        return made[-1]

    monkeypatch.setattr(first, "_Frames", making)
    ahead: list[int] = []

    class Slow(Recorder):
        def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            ahead.append(made[0].read - len(self.frames))
            return super().push(frames)

    recorder = Slow(pause=0.01)
    scanned = first.scan(clip, recorder)
    assert scanned.frames == 280 and scanned.index is not None
    assert [len(frames) for frames in recorder.pushes] == [7] * 40
    assert max(ahead) <= (2 + 2) * 7, ahead
    assert recorder.frames.tobytes() == official(clip).tobytes()
    assert scanned.index.probabilities is not None and len(scanned.index.probabilities) == 280


def test_a_pass_that_stops_moving_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Nothing moving for WATCHDOG seconds, here a detector stuck in its first push until ffmpeg is
    # gone, 30 s at most, one read held: said at once, ffmpeg stopped, the pass refused, saying
    # how far each part went. ffmpeg fills its queues and pipes before it stops moving: a loaded
    # machine took over 3 s to.
    monkeypatch.setattr(first, "WATCHDOG", 1.0)
    monkeypatch.setattr(first, "HELD", 1)
    monkeypatch.setattr(first, "CHUNK", 7)
    path = tmp_path / "long.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25"),
            *("-frames:v", "3000", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )

    class Stuck(Recorder):
        def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            waited = time.monotonic()
            while not self.pushes and ffmpeg_alive() and time.monotonic() - waited < 30:
                time.sleep(0.05)
            return super().push(frames)

    recorder = Stuck()
    started = time.monotonic()
    with pytest.raises(MediaError) as stopped:
        first.scan(path, recorder)
    assert time.monotonic() - started < 30
    said = str(stopped.value)
    assert said.startswith(f"{path}: the first pass stopped, nothing moved in 1 s: ")
    assert " scored" in said and " scaled for the shot detector" in said
    assert said.split(": ", 1)[1] in caplog.text  # said by the watchdog itself
    assert len(recorder.frames) < 3000
    assert not ffmpeg_threads()


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3,000 frames, more than ffmpeg's queues and pipes hold: with a pipe left unread, ffmpeg
    stops moving before their end."""
    path = tmp_path_factory.mktemp("long") / "long.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25"),
            *("-frames:v", "3000", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize(
    ("reader", "what"),
    [
        ("_Decoded.read", "the thread reading ffmpeg's messages"),
        ("_Hashed.read", "the thread reading ffmpeg's hashes"),
        ("_Frames.drain", "the thread reading the shot detector's frames"),
    ],
)
def test_a_reading_thread_that_raises_stops_the_pass(
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reader: str,
    what: str,
) -> None:
    # A thread reading one of ffmpeg's pipes raises, here once it has read some: ffmpeg is
    # stopped and the pass fails at once, saying which thread and what it raised, the detector
    # given no more frames. The thread gone, its pipe was read no more, ffmpeg blocked on it, and
    # the pass was stopped by the watchdog alone, as one in which nothing moved, 120 s later
    # (WATCHDOG, as it is here).
    kind, name = reader.split(".")
    reading = getattr(getattr(first, kind), name)
    reads = [0]

    def some(lines: Any) -> Any:
        for line in lines:
            reads[0] += 1
            if reads[0] > 200:
                raise RuntimeError("made to fail")
            yield line

    def failing(self: Any, *lines: Any) -> None:
        if lines:
            reading(self, some(*lines))
            return
        # The frames' reader takes no lines: it raises once frames were read.
        monkeypatch.setattr(first, "CHUNK", 7)
        stream = self._stream

        class Short:
            def readinto(self, into: Any) -> int:
                reads[0] += 1
                if reads[0] > 20:
                    raise RuntimeError("made to fail")
                return stream.readinto(into)

            def close(self) -> None:
                stream.close()

        self._stream = Short()
        reading(self)

    monkeypatch.setattr(getattr(first, kind), name, failing)
    recorder = Recorder(pause=0.01)
    started = time.monotonic()
    with pytest.raises(MediaError) as stopped:
        first.scan(long_clip, recorder)
    assert time.monotonic() - started < 30 < first.WATCHDOG
    said = f"the first pass stopped, {what} read no further: RuntimeError('made to fail')"
    assert str(stopped.value) == f"{long_clip}: {said}"
    assert isinstance(stopped.value.__cause__, RuntimeError)
    assert said in caplog.text
    assert len(recorder.frames) < 3000 and not ffmpeg_threads()


def test_a_reading_thread_that_raises_stops_the_pass_its_queue_full(
    long_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same with the detector's queue full, one read of 7 frames held and a detector slower
    # than the decode: the thread reading its frames, waiting for room in the queue, is stopped
    # with the pass, which fails at once. It waited there for good, and the pass with it, joining
    # it: no end at the watchdog's 120 s, which stops ffmpeg alone (a review's finding). Bounded
    # here, a hang failing the test. And the detector is given no more frames, but a read it was
    # being given as the thread raised: not the reads queued, nor the 16 frames left in the pipe.
    monkeypatch.setattr(first, "HELD", 1)
    monkeypatch.setattr(first, "CHUNK", 7)
    reading = first._Hashed.read
    failed = threading.Event()
    late: list[int] = []  # the reads pushed once the thread had raised

    def failing(self: Any, lines: Any) -> None:
        def some() -> Any:
            for read, line in enumerate(lines):
                if read > 200:
                    failed.set()
                    raise RuntimeError("made to fail")
                yield line

        reading(self, some())

    class Slow(Recorder):
        def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            if failed.is_set():
                late.append(len(frames))
            return super().push(frames)

    monkeypatch.setattr(first._Hashed, "read", failing)
    recorder = Slow(pause=0.05)
    raised: list[BaseException] = []

    def scanning() -> None:
        try:
            first.scan(long_clip, recorder)
        except BaseException as error:
            raised.append(error)

    worker = threading.Thread(target=scanning, daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), "the first pass hangs"
    assert time.monotonic() - started < 30 < first.WATCHDOG
    [stopped] = raised
    assert isinstance(stopped, MediaError) and str(stopped) == (
        f"{long_clip}: the first pass stopped, the thread reading ffmpeg's hashes read no further:"
        " RuntimeError('made to fail')"
    )
    assert 0 < len(recorder.frames) < 3000 and not ffmpeg_threads()
    assert len(late) <= 1, late


@pytest.mark.parametrize("value", [float("nan"), 1.5, -0.25])
def test_a_probability_outside_its_range_refused(clip: Path, value: float) -> None:
    # A probability no sigmoid gives, a NaN among them: refused by the pass, as the index's
    # reader refuses it (media/index.py), where it was recorded, the next resume then refusing the
    # index and running the pass again.
    class Wrong(Recorder):
        def push(self, frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
            given = super().push(frames)
            if len(self.pushes) == 2:
                given[3] = value
            return given

    with pytest.raises(ValueError, match="outside") as refused:
        first.scan(clip, Wrong())
    said = f"the shot detector gave frame 53 the probability {np.float32(value)}, outside [0, 1]"
    assert str(refused.value) == said


@pytest.fixture(scope="module")
def untrained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """test_detector.py's file of the untrained model's weights."""
    return untrained_weights(tmp_path_factory.mktemp("untrained"))


MODELS = os.environ.get("SEEDVR2X_MODEL_DIR")
REFERENCE = os.environ.get("SEEDVR2X_REFERENCE_DIR")


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


def detected(
    path: Path, weights: Path, device: torch.device, batch: int
) -> npt.NDArray[np.float32]:
    """The probabilities the first pass over path records with TransNetV2 on device."""
    with Detector(weights, device, batch=batch) as detector:
        scanned = first.scan(path, detector)
    assert scanned.index is not None and scanned.index.probabilities is not None
    assert len(scanned.index.probabilities) == scanned.frames
    return scanned.index.probabilities


def test_the_first_pass_records_what_the_detector_gives(untrained: Path, clip: Path) -> None:
    # The detector itself in the pass, on the untrained model's weights, no file needed: the
    # probabilities recorded are its own on the official extraction's frames, bit for bit, those
    # its finish gives with those its pushes gave, the last windows' (a stand-in gives every one
    # as it is pushed: finish's dropped, no test without the weights saw it).
    found = detected(clip, untrained, CPU, BATCH)
    assert found.tobytes() == streamed(untrained, official(clip), batch=BATCH).tobytes()


@pytest.mark.parametrize("batch", [1, BATCH])
def test_the_first_pass_records_the_detectors_probabilities(
    trained: Path, clip: Path, batch: int
) -> None:
    # Bit for bit runtime/detector.py's on the official extraction's frames, at the same batch,
    # and at one window a forward measurement's tnet_predict's; each cut peaking on the frame
    # before it, the only detection at 0.3.
    found = detected(clip, trained, CPU, batch)
    frames = official(clip)
    assert found.tobytes() == streamed(trained, frames, batch=batch).tobytes()
    if batch == 1:
        reference, _ = tnet_predict(measurements_model(trained), frames)
        assert found.tobytes() == reference.tobytes()
    assert tnet_detections(found, 0.3) == [cut - 1 for cut in CUTS]


@pytest.mark.gpu
@pytest.mark.parametrize("source", ["clip", "milestone 1"])
def test_gpu_first_pass_against_cpu(
    trained: Path, clip: Path, source: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The first pass's probabilities with the detector on the GPU, at its batch, against the
    # CPU's at one window a forward: the largest difference printed, the same detections at 0.3;
    # the GPU's memory given back, as torch counts it, and its peak said.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if source == "clip":
        path = clip
    else:
        if not REFERENCE:
            pytest.skip("needs SEEDVR2X_REFERENCE_DIR")
        path = Path(REFERENCE) / "m1" / "input_rgb.mkv"
        assert path.is_file(), f"{path}: missing; tests/test_regression.py says how to make it"
    device = torch.device("cuda", 0)
    cpu = detected(path, trained, CPU, 1)
    torch.cuda.init()
    allocated = torch.cuda.memory_allocated(device)
    caplog.set_level(logging.INFO, logger="seedvr2x")
    gpu = detected(path, trained, device, BATCH)
    torch.cuda.synchronize(device)
    difference = float(np.abs(gpu.astype(np.float64) - cpu).max())
    peaks = [record.getMessage() for record in caplog.records if "VRAM peak" in record.getMessage()]
    print(f"{source}: GPU against CPU {difference:.3g} at most, {len(cpu)} frames; {peaks}")
    assert tnet_detections(gpu, 0.3) == tnet_detections(cpu, 0.3)
    if source == "clip":
        assert tnet_detections(gpu, 0.3) == [cut - 1 for cut in CUTS]
    assert torch.cuda.memory_allocated(device) <= allocated
