"""Upscale Video on a VIDEO read from its file (stream.py, video.py; DESIGN.md, Repository and apps,
the nodes' VIDEO path), on the CPU with the stand-in models: the file's frames, as the node reads
them, through the library's pass (clip.upscale), shot by shot, are the planes of the node's file
bit for bit, at every quality, tiled and not, corrected and not; the source's audio copied packet
for packet; the trim, the crop and the rate as ComfyUI gives them; seedvr2x's record in the file;
the memory the run holds the same for a clip ten times as long; nothing left by a Cancel or an
error. Skipped without PyAV, which ComfyUI ships."""

import shutil
import subprocess
import threading
import time
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any

import comfy.model_management as management
import numpy as np
import numpy.typing as npt
import pytest
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl
from comfyui import nodes, plans, stream, video, words
from standins import RESOLUTION, StandIn, execute, handle, master_of, upscale_video
from test_nodes import TARGET, fresh, planned_clip
from test_progress import stop_at, tiled_plans
from test_still import TILES

from seedvr2x import clip, record
from seedvr2x import frames as sizes
from seedvr2x.shots import Shot

av = pytest.importorskip("av")

# The stand-in VAE's picture, 40x24, which the tests' resolution makes 80x48.
WIDTH, HEIGHT = 40, 24


def rgb_frames(count: int, seed: int = 3) -> npt.NDArray[np.uint8]:
    """Frames of `count`, (T, H, W, 3) uint8: noise over a gradient, each its own."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 200, WIDTH, dtype=np.float32)[None, :, None]
    frames = rng.integers(0, 56, (count, HEIGHT, WIDTH, 3)).astype(np.float32) + ramp
    return frames.astype(np.uint8)


def source(
    path: Path,
    count: int,
    *,
    rate: Fraction = Fraction(24),
    audio: bool = True,
    pix_fmt: str = "yuv420p",
    codec: str = "ffv1",
    times: list[int] | None = None,
) -> Path:
    """A source file at `path` (its container by its suffix): `count` frames of rgb_frames at
    `rate`, FFV1 (or `codec`) in `pix_fmt`, tagged BT.709 limited, with a FLAC tone of the same
    length where `audio`."""
    frames = rgb_frames(count)
    with av.open(str(path), "w") as container:
        stream_: Any = container.add_stream(codec, rate=rate)
        stream_.width, stream_.height, stream_.pix_fmt = WIDTH, HEIGHT, pix_fmt
        rgb = pix_fmt.startswith("gbr")
        stream_.codec_context.colorspace, stream_.codec_context.color_range = (
            (0, 2) if rgb else (1, 1)
        )
        sound: Any = None
        if audio:
            sound = container.add_stream("flac", rate=48000)
            sound.layout = "stereo"
        for index, values in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(values, format="rgb24")
            frame = (
                frame.reformat(format=pix_fmt)
                if rgb
                else frame.reformat(format=pix_fmt, dst_colorspace=1, dst_color_range=1)
            )
            frame.pts = index if times is None else times[index]
            for packet in stream_.encode(frame):
                container.mux(packet)
        for packet in stream_.encode():
            container.mux(packet)
        if sound is not None:
            samples = int(48000 * count / rate)
            tone = (np.sin(np.arange(samples) / 20) * 8000).astype(np.int16)
            for start in range(0, samples, 4608):
                part = np.ascontiguousarray(np.stack([tone[start : start + 4608]] * 2))
                chunk = av.AudioFrame.from_ndarray(
                    part.reshape(1, -1), format="s16", layout="stereo"
                )
                chunk.sample_rate, chunk.pts = 48000, start
                for packet in sound.encode(chunk):
                    container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
    return path


WHOLE = video.Trim()


def read_frames(
    path: Path | BytesIO, trim: video.Trim = WHOLE, crop: Any = None
) -> npt.NDArray[np.float32]:
    """The frames the node reads of a file, whole: video.probe and video.Decoded's."""
    found = video.probe(path if isinstance(path, BytesIO) else str(path), trim, crop)
    if isinstance(path, BytesIO):
        path.seek(0)
    with video.Decoded(
        path if isinstance(path, BytesIO) else str(path), found, torch.device("cpu")
    ) as decoded:
        return decoded.next_frames(found.frames)


def libraries(
    machine: StandIn,
    frames: npt.NDArray[np.float32],
    cuts: tuple[Shot, ...],
    settings: plans.Settings,
    quality: str = "standard",
) -> npt.NDArray[np.uint16]:
    """The master's planes of the library's pass over each shot of `frames` (clip.upscale, the
    shot's seed), the plan the node makes for them."""
    made = planned_clip(machine, settings, cuts)
    alone = fresh(machine)
    parts = [
        clip.upscale(
            alone,
            frames[each.start : each.end],
            TARGET,
            settings.seed + each.start,
            window=made.window,
            encode_tiles=made.encode_tiles,
            decode_tiles=made.decode_tiles,
            correct=settings.split,
            quality=quality,
        )
        for each in cuts
    ]
    return master_of(np.concatenate(parts))


def run(
    tmp_path: Path, given: Any, settings: plans.Settings, **kwargs: Any
) -> tuple[npt.NDArray[np.uint16], str]:
    out, text = upscale_video(handle(tmp_path), settings, given, RESOLUTION, "1", **kwargs)
    return out, text


def run_video(tmp_path: Path, given: Any, settings: plans.Settings, **kwargs: Any) -> Any:
    """The VIDEO the node gives."""
    inputs: dict[str, Any] = {"color_correction": settings.split, "seed": settings.seed}
    if settings.window is not None:
        inputs["window"] = settings.window
    return execute(
        nodes.UpscaleVideo,
        "1",
        video=given,
        models=handle(tmp_path),
        resolution=RESOLUTION,
        plan_only=False,
        quality="standard",
        **inputs,
        **kwargs,
    )[0]


CUTS = {
    "1 shot": (Shot(0, 21),),
    "2 shots": (Shot(0, 9), Shot(9, 21)),
    "3 shots": (Shot(0, 5), Shot(5, 18), Shot(18, 21)),
}


@pytest.mark.parametrize("quality", ["standard", "enhanced", "maximum"])
@pytest.mark.parametrize("cuts", list(CUTS), ids=list(CUTS))
def test_a_files_video_is_the_librarys_pass_on_its_frames(
    machine: StandIn, tmp_path: Path, cuts: str, quality: str
) -> None:
    path = source(tmp_path / "in.mkv", 21)
    settings = plans.Settings(seed=4, window=5)
    out, text = run(
        tmp_path, InputImpl.VideoFromFile(str(path)), settings, cuts=CUTS[cuts], quality=quality
    )
    expected = libraries(machine, read_frames(path), CUTS[cuts], settings, quality)
    assert out.shape == (21, 3, 48, 80)
    assert out.tobytes() == expected.tobytes()
    assert "the video is read from its file shot by shot" in text


@pytest.mark.parametrize("correct", [True, False], ids=["corrected", "uncorrected"])
@pytest.mark.parametrize("quality", ["standard", "maximum"])
def test_a_tiled_files_video_is_the_librarys_tiled_pass(
    machine: StandIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, correct: bool, quality: str
) -> None:
    tiled_plans(monkeypatch)
    path = source(tmp_path / "in.mkv", 13)
    settings = plans.Settings(seed=2, window=5, color_correction="split" if correct else "none")
    cuts = (Shot(0, 6), Shot(6, 13))
    out, _ = run(tmp_path, InputImpl.VideoFromFile(str(path)), settings, cuts=cuts, quality=quality)
    made = planned_clip(machine, settings, cuts)
    assert made.encode_tiles == TILES and made.decode_tiles == TILES
    expected = libraries(machine, read_frames(path), cuts, settings, quality)
    assert out.tobytes() == expected.tobytes()


def test_a_video_of_bytes_in_memory_is_read_as_its_file(machine: StandIn, tmp_path: Path) -> None:
    path = source(tmp_path / "in.mkv", 9)
    settings = plans.Settings(window=5)
    from_path, _ = run(tmp_path, InputImpl.VideoFromFile(str(path)), settings)
    from_bytes, _ = run(tmp_path, InputImpl.VideoFromFile(BytesIO(path.read_bytes())), settings)
    assert from_path.tobytes() == from_bytes.tobytes()


def packets(path: Path, kind: str) -> list[tuple[bytes, int | None, Any]]:
    with av.open(str(path)) as container:
        chosen = [each for each in container.streams if each.type == kind]
        return [
            (bytes(packet), packet.pts, packet.time_base)
            for packet in container.demux(*chosen)
            if packet.size
        ]


def times(found: list[tuple[bytes, int | None, Any]]) -> list[Fraction]:
    return [Fraction(pts) * Fraction(base) for _, pts, base in found if pts is not None]


def test_the_sources_audio_is_copied_packet_for_packet(machine: StandIn, tmp_path: Path) -> None:
    path = source(tmp_path / "in.mkv", 21)
    given = run_video(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings(window=5))
    out = Path(given.get_stream_source())
    copied, original = packets(out, "audio"), packets(path, "audio")
    assert [each[0] for each in copied] == [each[0] for each in original]
    # Matroska keeps a packet's time to the millisecond, as the source's own file did.
    assert times(copied) == times(original)
    with av.open(str(out)) as container:
        assert [each.type for each in container.streams] == ["video", "audio"]


def test_a_trimmed_video_is_its_windows_frames_and_the_audio_overlapping_it(
    machine: StandIn, tmp_path: Path
) -> None:
    path = source(tmp_path / "in.mkv", 48)
    trimmed = InputImpl.VideoFromFile(str(path)).as_trimmed(start_time=0.5, duration=0.5)
    settings = plans.Settings(window=5)
    out, _ = run(tmp_path, trimmed, settings)
    # ComfyUI's window: frames 12 to 23 at 24 fps (start_pts <= pts < end_pts).
    frames = read_frames(path, video.Trim(0.5, 0.5))
    assert frames.shape[0] == out.shape[0] == 12
    assert np.array_equal(frames, read_frames(path)[12:24])
    assert out.tobytes() == libraries(machine, frames, (Shot(0, 12),), settings).tobytes()
    given = run_video(tmp_path, trimmed, settings)
    out = Path(given.get_stream_source())
    copied = times(packets(out, "audio"))
    # Matroska moves every stream alike where one would begin before 0: the times are said from
    # the video's first frame.
    first = times(packets(out, "video"))[0]
    original = packets(path, "audio")
    # The packets overlapping the window (0.5 s to 1 s), whole, a FLAC packet 4608 samples long.
    starts = times(original)
    kept = [each for each in starts if each + Fraction(4608, 48000) > Fraction(1, 2) and each < 1]
    assert kept[0] < Fraction(1, 2) and kept[-1] + Fraction(4608, 48000) > 1
    assert [each - first for each in copied] == [each - Fraction(1, 2) for each in kept]


def test_a_cropped_video_is_cropped_as_comfyui_crops_it(machine: StandIn, tmp_path: Path) -> None:
    path = source(tmp_path / "in.mkv", 9)
    cropped = InputImpl.VideoFromFile(str(path)).as_cropped(x=3, y=1, width=33, height=21)
    whole = read_frames(path)
    # Floored to even numbers: (2, 0, 32, 20).
    assert video.crop_rect((3, 1, 33, 21), WIDTH, HEIGHT) == (2, 0, 32, 20)
    frames = read_frames(path, crop=(3, 1, 33, 21))
    assert np.array_equal(frames, whole[:, 0:20, 2:34])
    out, text = run(tmp_path, cropped, plans.Settings(window=5))
    assert "a clip of 9 frames 32x20 to" in text and out.shape[0] == 9


def test_a_video_whose_crop_cant_be_read_is_refused(machine: StandIn, tmp_path: Path) -> None:
    class Elsewhere(InputImpl.VideoFromFile):
        def __getattribute__(self, name: str) -> Any:
            if name == "_VideoFromFile__crop":
                raise AttributeError(name)
            return super().__getattribute__(name)

    path = source(tmp_path / "in.mkv", 5)
    with pytest.raises(words.Refused, match="crop"):
        run(tmp_path, Elsewhere(str(path)), plans.Settings())
    assert machine.loads == []


def test_the_files_record_is_its_streams_tags(machine: StandIn, tmp_path: Path) -> None:
    path = source(tmp_path / "in.mkv", 9)
    cuts = plans.CutList((Shot(0, 4), Shot(4, 9)), "", 0.3)
    given = run_video(
        tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings(seed=5, window=5), cuts=cuts
    )
    with av.open(given.get_stream_source()) as container:
        tags = {key.lower(): value for key, value in container.streams.video[0].metadata.items()}
    assert tags["seedvr2x_app"] == "seedvr2x-comfyui"
    assert tags["seedvr2x_digital_source_type"] == record.DIGITAL_SOURCE_TYPE
    assert tags["seedvr2x_source_size"] == "40x24" and tags["seedvr2x_output_size"] == "80x48"
    assert tags["seedvr2x_seed"] == "5" and tags["seedvr2x_window"] == "5"
    assert tags["seedvr2x_cut_threshold"] == "0.3" and tags["seedvr2x_quality"] == "standard"
    assert tags["seedvr2x_dit_model"] == "w.safetensors"
    assert tags["seedvr2x_dit_sha256"] == "d" * 64 and tags["seedvr2x_vae_sha256"] == "e" * 64
    # The file's own tags hold it too, which Save Video's copy keeps where the stream's go: its
    # copy (ComfyUI 0.45.3, VideoFromFile.save_to) replayed, the file's tags copied first and each
    # stream added from its template.
    saved = tmp_path / "saved.mkv"
    with av.open(given.get_stream_source()) as source_, av.open(str(saved), "w") as output:
        for key, value in source_.metadata.items():
            output.metadata[key] = value
        outputs = {
            each.index: output.add_stream_from_template(template=each, opaque=True)
            for each in source_.streams
        }
        for packet in source_.demux(*source_.streams):
            if packet.dts is None:
                continue
            packet.stream = outputs[packet.stream.index]
            output.mux(packet)
    with av.open(str(saved)) as container:
        kept = {key.lower(): value for key, value in container.metadata.items()}
    record_tags = {key: value for key, value in tags.items() if key.startswith("seedvr2x_")}
    assert len(record_tags) > 20
    assert {key: kept.get(key) for key in record_tags} == record_tags


@pytest.mark.parametrize(
    "rate", [Fraction(24), Fraction(24000, 1001), Fraction(60000, 1001), Fraction(120000, 1001)]
)
def test_the_video_given_has_the_sources_own_rate(
    machine: StandIn, tmp_path: Path, rate: Fraction
) -> None:
    """Matroska reads 60000/1001 back as 19001/317 (libavformat's fraction of a frame's duration):
    the node reads the source's NTSC rate as itself and gives it to what reads its VIDEO."""
    path = source(tmp_path / "in.mkv", 5, rate=rate, audio=False)
    given = run_video(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings())
    assert given.get_frame_rate() == rate
    assert given.get_components().frame_rate == rate
    assert video.probe(str(path), video.Trim(), None).rate == rate


def test_ntsc_readings_are_taken_back_to_their_rate() -> None:
    assert video.ntsc_reading(Fraction(19001, 317), "matroska,webm") == Fraction(60000, 1001)
    assert video.ntsc_reading(Fraction(29011, 242), "matroska,webm") == Fraction(120000, 1001)
    assert video.ntsc_reading(Fraction(7001, 146), "matroska,webm") == Fraction(48000, 1001)
    assert video.ntsc_reading(Fraction(19001, 317), "mov,mp4,m4a,3gp,3g2,mj2") == Fraction(
        19001, 317
    )
    for kept in (Fraction(25), Fraction(24000, 1001), Fraction(50), Fraction(1001, 6)):
        assert video.ntsc_reading(kept, "matroska,webm") == kept


def test_quarter_turns_are_comfyuis() -> None:
    examples = {
        90: 1, -90: 3, 180: 2, -180: 2, 270: 3, 360: 0, 45: 0, 89: 0, -1: 3, -45: 3,
        359.9: 3, 450: 1, 450.5: 1, 0: 0, None: 0,
    }  # fmt: skip
    for rotation, turns in examples.items():
        found = video.quarter_turns(rotation)
        assert found == turns and type(found) is int, rotation


def test_crop_rect_is_comfyuis_rectangle() -> None:
    on_64x48 = {
        (3, 1, 33, 21): (2, 0, 32, 20),
        (0, 0, 64, 48): None,
        (0, 0, 100, 100): None,
        (1, 0, 64, 48): None,
        (1, 1, 100, 100): None,
        (0, 0, 63, 48): (0, 0, 62, 48),
        (0, 0, 63, 47): (0, 0, 62, 46),
        (63, 47, 10, 10): (62, 46, 2, 2),
        (70, 70, 10, 10): (62, 46, 2, 2),
        (-5, -5, 10, 10): (0, 0, 10, 10),
        (10, 10, 1, 1): None,
        (5, 0, 1, 48): None,
        (0, 0, 0, 10): None,
        (10, 10, 0, 5): None,
        (2.9, 3.9, 20.5, 10.2): (2, 2, 20, 10),
    }
    for crop, rectangle in on_64x48.items():
        assert video.crop_rect(crop, 64, 48) == rectangle, crop
    assert video.crop_rect(None, 64, 48) is None
    assert video.crop_rect((3, 1, 33, 21), 48, 32) == (2, 0, 32, 20)


def test_the_stand_ins_trim_window_is_the_one_the_video_was_made_with(tmp_path: Path) -> None:
    path = source(tmp_path / "ten.mkv", 240, audio=False)  # 10 s at 24 fps

    def window(start_time: float, duration: float) -> tuple[float, float]:
        made = InputImpl.VideoFromFile(str(path), start_time=start_time, duration=duration)
        return made.get_active_trim_window()

    assert window(start_time=2, duration=3) == (2.0, 3.0)
    assert window(start_time=0, duration=0) == (0.0, 0.0)
    for start_time, start in ((-1, 9.0), (-4, 6.0)):
        found = window(start_time=start_time, duration=0)
        assert found[0] == pytest.approx(start, abs=0.1) and found[1] == 0.0
    assert window(start_time=-20, duration=3) == (0.0, 3.0)


def test_the_stand_ins_as_trimmed_keeps_the_source_and_adds_the_starts(tmp_path: Path) -> None:
    path = source(tmp_path / "ten.mkv", 240, audio=False)
    first = InputImpl.VideoFromFile(str(path), start_time=2, duration=5, crop=(0, 0, 8, 8))
    second = first.as_trimmed(start_time=1, duration=3)
    assert second is not None and second.get_stream_source() == str(path)
    assert second.get_active_trim_window() == (3.0, 3.0)
    assert second._VideoFromFile__crop == (0, 0, 8, 8)  # pyright: ignore[reportAttributeAccessIssue]
    assert first.as_trimmed(start_time=9, duration=3) is None
    loose = first.as_trimmed(start_time=9, duration=3, strict_duration=False)
    assert loose is not None and loose.get_active_trim_window() == (11.0, 3.0)
    buffer = BytesIO(path.read_bytes())
    buffer.seek(7)
    kept = InputImpl.VideoFromFile(buffer)
    assert kept.get_stream_source() is buffer and buffer.tell() == 0


@pytest.mark.parametrize("pix_fmt", ["nv12", "yuv422p", "yuv444p10le", "gbrp10le"])
def test_each_format_is_read_by_the_librarys_conversion(tmp_path: Path, pix_fmt: str) -> None:
    """A frame's planes, a semi-planar one repacked to its planar twin, through yuv.to_rgb."""
    codec = "libx264" if pix_fmt == "nv12" else "ffv1"
    path = source(tmp_path / "in.mkv", 3, pix_fmt=pix_fmt, codec=codec, audio=False)
    found = video.probe(str(path), video.Trim(), None)
    with av.open(str(path)) as container:
        frame = next(container.decode(container.streams.video[0]))
        planes = [
            torch.as_tensor(np.ascontiguousarray(each))[None]
            for each in video.planes_of(frame, found.reading)
        ]
    from seedvr2x import yuv

    expected = yuv.to_rgb(planes, found.reading)[0].numpy()
    assert np.array_equal(read_frames(path)[0], expected)


def test_a_plan_only_writes_no_file(machine: StandIn, tmp_path: Path, comfy_temp: Path) -> None:
    path = source(tmp_path / "in.mkv", 9)
    run(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings(), plan_only=True)
    assert list(comfy_temp.rglob("*.mkv")) == []


def left(comfy_temp: Path) -> list[Path]:
    return [each for each in comfy_temp.rglob("*") if each.is_file()]


def writers() -> list[threading.Thread]:
    return [each for each in threading.enumerate() if each.name == "seedvr2x writer"]


@pytest.mark.parametrize("update", [4, 8])
def test_cancel_leaves_no_file_and_no_writer(
    machine: StandIn, tmp_path: Path, comfy_temp: Path, update: int
) -> None:
    path = source(tmp_path / "in.mkv", 21)
    stop_at(update)
    with pytest.raises(management.InterruptProcessingException):
        run(
            tmp_path,
            InputImpl.VideoFromFile(str(path)),
            plans.Settings(window=5),
            cuts=(Shot(0, 9), Shot(9, 21)),
            quality="enhanced",
        )
    assert left(comfy_temp) == [] and writers() == []
    ProgressBar.on_update = None


def test_a_writer_that_fails_ends_the_run_leaving_nothing(
    machine: StandIn, tmp_path: Path, comfy_temp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = source(tmp_path / "in.mkv", 21)
    real = nodes.Master.write_planes
    written: list[int] = []

    def failing(self: Any, planes: Any) -> None:
        written.append(planes.shape[0])
        if len(written) > 3:
            raise OSError(28, "No space left on device")
        real(self, planes)

    monkeypatch.setattr(nodes.Master, "write_planes", failing)
    with pytest.raises(words.Refused, match=r"could not be written .* No space left on device"):
        run(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings(window=5))
    assert left(comfy_temp) == [] and writers() == []


def resident() -> int:
    with open("/proc/self/statm") as file:
        return int(file.read().split()[1]) * 4096


def peak_while(action: Any) -> int:
    """The process's resident memory at its most while `action` runs, over what it was before,
    sampled every millisecond."""
    import gc

    gc.collect()
    before, most, done = resident(), 0, threading.Event()

    def sample() -> None:
        nonlocal most
        while not done.is_set():
            most = max(most, resident())
            time.sleep(0.001)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        action()
    finally:
        done.set()
        sampler.join()
    return most - before


@pytest.mark.skipif(not Path("/proc/self/statm").exists(), reason="reads /proc")
def test_a_clip_ten_times_as_long_holds_no_more_memory(machine: StandIn, tmp_path: Path) -> None:
    """The run holds a slice of the source's frames, a shot's windows and the writer's queue,
    never the clip: a clip of 900 frames (100 shots of 9) holds what one of 90 (10 shots) does,
    within a shot's frames (its input and output in float32), where the clip's output alone would
    be 10 times its own."""
    shots = 10
    short = source(tmp_path / "short.mkv", 9 * shots, audio=False)
    long = source(tmp_path / "long.mkv", 9 * shots * 10, audio=False)
    settings = plans.Settings(window=5)

    def go(path: Path, count: int) -> None:
        cuts = tuple(Shot(start, start + 9) for start in range(0, count, 9))
        run_video(tmp_path, InputImpl.VideoFromFile(str(path)), settings, cuts=cuts)

    go(short, 9 * shots)  # the models and the process's own buffers made once
    first = peak_while(lambda: go(short, 9 * shots))
    longer = peak_while(lambda: go(long, 9 * shots * 10))
    shot = 9 * (WIDTH * HEIGHT + 4 * WIDTH * HEIGHT) * 3 * 4
    output = 9 * shots * 10 * 4 * WIDTH * HEIGHT * 3 * 4
    assert longer - first < max(shot, 2 * 2**20), (first, longer, output)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg to tag a sample aspect")
def test_a_files_sample_aspect_shapes_the_output_and_the_record(
    machine: StandIn, tmp_path: Path
) -> None:
    """A file's pixels of sample aspect 8:9 (a DVD's) are made square, as the command line makes
    them (DESIGN.md, Colour and shape): the record says the aspect used. The file is ffmpeg's,
    PyAV writing no sample aspect."""
    path = tmp_path / "in.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=40x24:rate=24"),
            *("-frames:v", "5", "-vf", "setsar=8/9", "-c:v", "ffv1", "-pix_fmt", "yuv420p"),
            str(path),
        ],
        check=True,
    )
    assert video.probe(str(path), video.Trim(), None).sample_aspect == Fraction(8, 9)
    given = run_video(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings())
    with av.open(given.get_stream_source()) as container:
        stream_ = container.streams.video[0]
        size = (stream_.width, stream_.height)
        tags = {key.lower(): value for key, value in stream_.metadata.items()}
    height, width = sizes.output_size(sizes.target_size(40, 24, Fraction(8, 9), RESOLUTION))
    assert size == (width, height) == (70, 48) and tags["seedvr2x_sample_aspect"] == "8:9"


def test_frames_off_their_rate_are_refused_before_anything_loads(
    machine: StandIn, tmp_path: Path
) -> None:
    """A frame half a frame or more from its place on the rate's timeline (a variable rate) is
    refused, as the command line refuses it: the node numbers its frames at the rate."""
    path = source(tmp_path / "in.mkv", 6, audio=False, times=[0, 1, 2, 4, 5, 6])
    with pytest.raises(words.Refused, match="don't keep its frame rate"):
        run(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings())
    assert machine.loads == []


def test_the_frames_are_counted_by_decoding_them(tmp_path: Path) -> None:
    path = source(tmp_path / "in.mkv", 7, audio=False, codec="libx264")
    assert video.probe(str(path), video.Trim(), None).frames == 7


def test_the_run_says_its_disk_and_keeps_only_its_file(
    machine: StandIn, tmp_path: Path, comfy_temp: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = source(tmp_path / "in.mkv", 9)
    with caplog.at_level("INFO", logger="seedvr2x_comfyui.nodes"):
        given = run_video(tmp_path, InputImpl.VideoFromFile(str(path)), plans.Settings(window=5))
    said = [each for each in caplog.messages if "disk: the upscaled file takes" in each]
    assert len(said) == 1 and "free there" in said[0]
    assert [each.relative_to(comfy_temp) for each in comfy_temp.rglob("*")] == [
        Path("seedvr2x"),
        Path("seedvr2x") / Path(given.get_stream_source()).name,
    ]


def test_a_tiled_scratch_beyond_its_allowance_is_a_file_removed_once_unmapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store over the allowance is a mapped file of the run's directory; one Windows refuses to
    remove while mapped (faked: the first removal fails) is removed at the next store, as the
    command line's scratch removes it."""
    with stream.Workspace(tmp_path / "seedvr2x") as workspace:
        scratch = stream.FileScratch(workspace, 0)
        real = Path.unlink
        refused: list[Path] = []

        def unlink(self: Path, missing_ok: bool = False) -> None:
            if self.name.startswith("scratch-") and not refused:
                refused.append(self)
                raise PermissionError(13, "the file is mapped")
            real(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)
        with scratch.store(2, (16, 16)) as store:
            store[0, 0, 0, 0] = 1.0
            assert list(workspace.root.glob("scratch-*")) and "a file of" in scratch.where(
                2, (16, 16)
            )
        del store
        assert refused and refused[0].exists() and stream._PENDING == refused  # pyright: ignore[reportPrivateUsage]
        with scratch.store(2, (16, 16)):
            assert not refused[0].exists()
        assert stream._PENDING == []  # pyright: ignore[reportPrivateUsage]
    assert not workspace.root.exists()
