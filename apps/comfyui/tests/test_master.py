"""The nodes' lossless file (master.py): FFV1 in Matroska, 16-bit 4:4:4 YCbCr tagged BT.709
limited, written through PyAV a shot at a time into one open file, read back by PyAV and by
ffmpeg: the planes written bit for bit (yuv.master_planes of the frames), so the frames read
through the master's reading (yuv.to_rgb, MASTER) those of the planes written, within two codes of
the RGB given (the master's limited range spans fewer codes than RGB's: the shares said); the tags
set; the frame rate exact, each frame's time within Matroska's millisecond. Skipped without PyAV,
and ffmpeg's half without an ffmpeg on PATH."""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from seedvr2x import yuv

av = pytest.importorskip("av", reason="needs PyAV (av), which ComfyUI ships")

from comfyui.master import BT709, LIMITED, UNSPECIFIED, Master  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH")

WIDTH, HEIGHT = 48, 32
# Shots of unequal lengths, each written by one call.
SHOTS = (5, 1, 3)


def shots(seed: int = 0) -> list[torch.Tensor]:
    """Frames (T, H, W, 3) float32 in [0, 1], whole 16-bit codes over 65535, a shot a tensor: the
    first row black, white and the primaries, the rest random."""
    rng = np.random.default_rng(seed)
    made: list[torch.Tensor] = []
    for length in SHOTS:
        codes = rng.integers(0, 65536, (length, HEIGHT, WIDTH, 3))
        codes[:, 0, :6] = [
            [0, 0, 0],
            [65535] * 3,
            [65535, 0, 0],
            [0, 65535, 0],
            [0, 0, 65535],
            [32768] * 3,
        ]
        made.append(torch.from_numpy(codes.astype(np.float32)) / 65535)
    return made


def written(path: Path, rate: Fraction, primaries: int = BT709, transfer: int = BT709) -> Path:
    with Master(path, WIDTH, HEIGHT, rate, primaries, transfer) as master:
        for shot in shots():
            master.write(shot)
        assert master.frames == sum(SHOTS)
    return path


def expected_planes() -> npt.NDArray[np.uint16]:
    return torch.cat([yuv.master_planes(shot) for shot in shots()]).numpy()


def pyav_planes(path: Path) -> tuple[npt.NDArray[np.uint16], list[object]]:
    planes: list[npt.NDArray[np.uint16]] = []
    frames: list[object] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            planes.append(
                np.stack(
                    [
                        np.frombuffer(memoryview(plane), dtype="<u2").reshape(
                            plane.height, plane.line_size // 2
                        )[:, : plane.width]
                        for plane in frame.planes
                    ]
                )
            )
            frames.append(frame)
    return np.stack(planes), frames


def test_the_planes_come_back_bit_for_bit_through_pyav(tmp_path: Path) -> None:
    path = written(tmp_path / "master.mkv", Fraction(25))
    planes, frames = pyav_planes(path)
    assert np.array_equal(planes, expected_planes())
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        context = stream.codec_context
        assert (stream.codec_context.name, stream.format.name) == ("ffv1", "yuv444p16le")
        assert (context.colorspace, context.color_range) == (BT709, LIMITED)
        assert (context.color_primaries, context.color_trc) == (BT709, BT709)
        assert (stream.width, stream.height) == (WIDTH, HEIGHT)
    assert len(frames) == sum(SHOTS)


def test_the_frames_read_back_through_the_master_s_reading(tmp_path: Path) -> None:
    # The planes read back, read as the master (yuv.to_rgb, MASTER), give the frames of the planes
    # written exactly, and the RGB given within two codes, most within one.
    path = written(tmp_path / "master.mkv", Fraction(25))
    planes, _ = pyav_planes(path)
    tensors = [torch.from_numpy(planes[:, plane].astype(np.int32)) for plane in range(3)]
    back = torch.round(yuv.to_rgb(tensors, yuv.MASTER) * 65535).to(torch.int64).numpy()
    given = torch.round(torch.cat(shots()) * 65535).to(torch.int64).numpy()
    expected = expected_planes()
    of_the_written = [torch.from_numpy(expected[:, plane].astype(np.int32)) for plane in range(3)]
    written_back = torch.round(yuv.to_rgb(of_the_written, yuv.MASTER) * 65535).to(torch.int64)
    assert np.array_equal(back, written_back.numpy())
    difference = np.abs(back - given)
    print(
        f"the master's RGB read back: {np.mean(difference == 0):.2%} equal,"
        f" {np.mean(difference == 1):.2%} one code off, {np.mean(difference == 2):.2%} two"
    )
    assert difference.max() <= 2
    assert np.mean(difference == 2) < 0.03
    # Black, white and grey exactly: the master's neutral chroma, its black and white.
    assert (back[:, 0, :2] == given[:, 0, :2]).all() and (back[:, 0, 5] == given[:, 0, 5]).all()


def ffprobe(path: Path) -> Any:
    output = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames"),
            *("-show_entries", "stream=codec_name,pix_fmt,color_space,color_range,color_primaries"),
            *("-show_entries", "stream=color_transfer,r_frame_rate,avg_frame_rate"),
            *("-show_entries", "frame=pts_time", "-of", "json", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


@needs_ffmpeg
def test_ffmpeg_reads_the_planes_and_the_tags(tmp_path: Path) -> None:
    path = written(tmp_path / "master.mkv", Fraction(25))
    data = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv444p16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    planes = np.frombuffer(data, dtype="<u2").reshape(-1, 3, HEIGHT, WIDTH)
    assert np.array_equal(planes, expected_planes())
    stream = ffprobe(path)["streams"][0]
    assert {key: stream[key] for key in stream if key.startswith(("codec", "pix", "color"))} == {
        "codec_name": "ffv1",
        "pix_fmt": "yuv444p16le",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
    }


@needs_ffmpeg
def test_a_source_s_primaries_and_transfer_are_copied_and_none_stays_none(tmp_path: Path) -> None:
    # H.273: 9 BT.2020's primaries, 14 its 10-bit transfer; 6 SMPTE 170M's.
    path = written(tmp_path / "wide.mkv", Fraction(25), primaries=9, transfer=6)
    stream = ffprobe(path)["streams"][0]
    assert (stream["color_primaries"], stream["color_transfer"]) == ("bt2020", "smpte170m")
    assert (stream["color_space"], stream["color_range"]) == ("bt709", "tv")
    path = written(tmp_path / "none.mkv", Fraction(25), UNSPECIFIED, UNSPECIFIED)
    stream = ffprobe(path)["streams"][0]
    assert "color_primaries" not in stream and "color_transfer" not in stream


# The rates written and the rate read back. Matroska keeps a frame's duration in whole nanoseconds,
# which libavformat's reader turns back into a fraction of terms up to 30,000 (matroskadec.c's
# av_reduce): exact for the usual rates, and for 60000/1001 and 120000/1001 its nearest such
# fraction, 5 and 38 parts in 100 million off, ffmpeg's own Matroska files the same
# (2026-11-13), each frame's time still within the millisecond.
RATES = {
    Fraction(24000, 1001): Fraction(24000, 1001),
    Fraction(24): Fraction(24),
    Fraction(25): Fraction(25),
    Fraction(30000, 1001): Fraction(30000, 1001),
    Fraction(50): Fraction(50),
    Fraction(60): Fraction(60),
    Fraction(60000, 1001): Fraction(19001, 317),
    Fraction(120000, 1001): Fraction(29011, 242),
}


@needs_ffmpeg
@pytest.mark.parametrize("rate", RATES, ids=str)
def test_the_rate_read_back_and_each_time_within_a_millisecond(
    tmp_path: Path, rate: Fraction
) -> None:
    path = written(tmp_path / "master.mkv", rate)
    probed = ffprobe(path)
    stream = probed["streams"][0]
    read = RATES[rate]
    assert Fraction(stream["r_frame_rate"]) == Fraction(stream["avg_frame_rate"]) == read
    assert abs(read / rate - 1) < Fraction(1, 10**6)
    times = [Fraction(frame["pts_time"]) for frame in probed["frames"]]
    assert len(times) == sum(SHOTS)
    for index, time in enumerate(times):
        assert abs(time - index / rate) <= Fraction(1, 2000)  # Matroska's millisecond, rounded
    with av.open(str(path)) as container:
        assert container.streams.video[0].average_rate == read


def test_a_frame_of_another_size_is_refused(tmp_path: Path) -> None:
    with Master(tmp_path / "master.mkv", WIDTH, HEIGHT, Fraction(25)) as master:
        with pytest.raises(ValueError, match="48x32"):
            master.write(torch.zeros((1, HEIGHT, WIDTH + 2, 3)))
        master.write(torch.zeros((1, HEIGHT, WIDTH, 3)))
    planes, _ = pyav_planes(tmp_path / "master.mkv")
    assert planes.shape == (1, 3, HEIGHT, WIDTH)
    assert (planes[0, 0] == 4096).all() and (planes[0, 1:] == 32768).all()


def test_an_exception_closes_the_file_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "master.mkv"
    with pytest.raises(RuntimeError, match="stopped"):
        with Master(path, WIDTH, HEIGHT, Fraction(25)) as master:
            master.write(shots()[0])
            raise RuntimeError("stopped")
    assert path.exists()  # a partial file, the caller's to remove


def test_each_frame_waits_for_the_card_through_the_library_s_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # model.synchronize before each copy to the host, so that a stop and Cancel reach the wait.
    from seedvr2x import model

    waited: list[torch.device] = []
    monkeypatch.setattr(model, "synchronize", waited.append)
    with Master(tmp_path / "master.mkv", WIDTH, HEIGHT, Fraction(25)) as master:
        master.write(shots()[0])
    assert waited == [torch.device("cpu")] * SHOTS[0]
