"""A VIDEO's frames for Upscale Video and Cuts (DESIGN.md, Repository and apps, the nodes' VIDEO
path): a file's, read forward through PyAV, which ComfyUI ships, a frame at a time, its planes
converted to RGB by the command line's rules in torch (`seedvr2x.yuv`), never the clip whole; or
frames ComfyUI holds already, a VIDEO with no file of its own (Create Video's, a concatenation),
taken whole as ComfyUI gives them.

A file's frames are those ComfyUI's own decode of the same VIDEO gives (ComfyUI 0.45.3,
comfy_api/latest/_input_impl/video_types.py, VideoFromFile._decode_frames): its trim window's
frames by their timestamps (`start_pts <= pts < end_pts`, the window's seconds over the stream's
time base, truncated), each turned by its rotation's quarter turns and then cropped as Crop Video
asked, after the conversion. Only the conversion differs: ComfyUI's swscale reads an untagged
source as BT.601 whatever its size and puts a limited white at 0.996, where seedvr2x reads it as
the command line does (impl-videostudy, 2026-11-12).

PyAV is imported inside the functions that use it, never at import: the nodes' other modules and
their tests import this one without it."""

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from seedvr2x import yuv

from . import words

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("seedvr2x_comfyui.video")

# FFmpeg's names of its colour enums (libavutil/pixdesc.c, av_color_space_name, av_color_range_name,
# av_chroma_location_name), which PyAV gives as integers and the library's reading takes as ffprobe
# writes them.
COLOR_SPACES = {
    0: "gbr",
    1: "bt709",
    2: "unknown",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}
COLOR_RANGES = {0: "unknown", 1: "tv", 2: "pc"}
CHROMA_LOCATIONS = {
    0: "unspecified",
    1: "left",
    2: "center",
    3: "topleft",
    4: "top",
    5: "bottomleft",
    6: "bottom",
}

# The highest N of an NTSC rate N x 1000/1001 a Matroska file's reading is taken back to (the
# command line's media/rate.py, MOST_NTSC), and how near the reading is to it: libavformat reads a
# frame's duration in whole nanoseconds back as the nearest fraction of terms up to 30,000,
# 60000/1001 as 19001/317, 5 parts in 10^8 off, 120000/1001 as 29011/242, 38 (impl-video-1), and
# 100000/1001 as 999/10, 1 in 10^6 (the command line reads 999/10 as 100000/1001 too).
MOST_NTSC = 240
NTSC_NEAR = Fraction(2, 10**6)


def ntsc_reading(declared: Fraction, container: str) -> Fraction:
    """The rate a stream is read at: its declared rate, but for a Matroska or WebM file's that is
    libavformat's reading of an NTSC rate's frame duration, N x 1000/1001 for N from 1 to
    MOST_NTSC, which is taken back to that rate, as the command line takes it (its media/rate.py,
    matroska_reading; DESIGN.md, Input, the frame rate). The command line computes each reading
    exactly; this takes a rate within NTSC_NEAR of N x 1000/1001, relative, that is not itself a
    multiple of 1/12 fps (a rate a file declares as such): the readings are 5 to 100 parts in
    10^8 off theirs, and no rate of a whole number of thousandths but 999/10, 999/5 and 999/8 is
    that near one. PROVISIONAL: the command line's exact rule is not shared with the nodes."""
    if not container.startswith(("matroska", "webm")) or declared <= 0:
        return declared
    n = round(declared * 1001 / 1000)
    if not 1 <= n <= MOST_NTSC:
        return declared
    ntsc = Fraction(n * 1000, 1001)
    if declared == ntsc or (declared * 12).denominator == 1:
        return declared
    if abs(declared - ntsc) / ntsc <= NTSC_NEAR:
        return ntsc
    return declared


def crop_rect(
    crop: tuple[float, float, float, float] | None, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """The rectangle (x, y, width, height) ComfyUI crops an upright frame of `width` x `height`
    to, given Crop Video's rectangle, or None for no crop (none given, a side of 0 or less, the
    whole frame, or a side left at 0 after the even rounding). The values are whole numbers as
    int() makes them (toward zero); the corner is brought into the frame and down to an even
    number, the sides to what the frame has from the corner, then down to even numbers."""
    if crop is None:
        return None
    x, y, wide, high = (int(each) for each in crop)
    if wide <= 0 or high <= 0:
        return None
    x = min(max(x, 0), width - 1) // 2 * 2
    y = min(max(y, 0), height - 1) // 2 * 2
    wide = min(wide, width - x)
    high = min(high, height - y)
    if (x, y, wide, high) == (0, 0, width, height):
        return None
    wide, high = wide // 2 * 2, high // 2 * 2
    if wide <= 0 or high <= 0:
        return None
    return x, y, wide, high


def quarter_turns(rotation: float | None) -> int:
    """The quarter turns (0 to 3) ComfyUI turns a decoded frame by, from the frame's rotation in
    degrees (PyAV's, from the display matrix): the rotation over 90 rounded down, modulo 4."""
    if not rotation:
        return 0
    return math.floor(rotation / 90) % 4


@dataclass(frozen=True)
class Trim:
    """A VIDEO's trim window, as ComfyUI gives it (VideoFromFile.get_active_trim_window): its start
    and duration in seconds, 0 for none."""

    start: float = 0.0
    duration: float = 0.0


@dataclass(frozen=True)
class Probe:
    """What a file's video stream is, read before its frames: the stream's index, its picture as
    stored (width, height), its pixel format as decoded, the reading of its planes (yuv.reading),
    its colour primaries and transfer as H.273 code points (copied to the master as they are), its
    rate (ntsc_reading), the frames of the trim window (decoded and counted), the quarter turns
    and crop applied after the conversion, the size of the frames given (upright and cropped), the
    container's name and the stream's time base; the window in it (start_pts, end_pts: None for no
    end) and its first frame's timestamp (first_pts); its pixels' sample aspect, the stream's
    (1 where it says none), of the frames given (turned with them)."""

    index: int
    stored: tuple[int, int]
    pix_fmt: str
    reading: yuv.Reading
    primaries: int
    transfer: int
    rate: Fraction
    frames: int
    turns: int
    crop: tuple[int, int, int, int] | None
    size: tuple[int, int]
    container: str
    time_base: Fraction
    start_pts: int
    end_pts: int | None
    first_pts: int
    sample_aspect: Fraction = Fraction(1)


def _tag(names: dict[int, str], *values: Any) -> str:
    """The first of `values` (PyAV's integers, or None where it has no such attribute) that names a
    tag, by FFmpeg's names; "" where none does."""
    for value in values:
        if value is None:
            continue
        name = names.get(int(value), "")
        if name and name not in yuv.UNTAGGED:
            return name
    return ""


def _window(stream: Any, trim: Trim) -> tuple[int, int | None]:
    """The trim window in the stream's time base, as ComfyUI computes it: (start_pts, end_pts),
    end_pts None for no end."""
    time_base = stream.time_base
    start_pts = int(trim.start / time_base)
    end_pts = int((trim.start + trim.duration) / time_base) if trim.duration else None
    return start_pts, end_pts


def _within(pts: int | None, start_pts: int, end_pts: int | None) -> bool:
    return pts is not None and pts >= start_pts and (end_pts is None or pts < end_pts)


def probe(source: str | BytesIO, trim: Trim, crop: tuple[int, int, int, int] | None) -> Probe:
    """The file's first video stream as Probe says, its first frame of the window decoded for its
    tags and rotation, and its window's frames counted by decoding each once.
    SourceRefused (seedvr2x.errors) as yuv.reading refuses the stream, and for frames whose sides
    the chroma subsampling doesn't divide (yuv.check_size, as yuv.to_rgb refuses them); ValueError
    for a file with no video stream or no frame in the window."""
    import av

    if isinstance(source, BytesIO):
        source.seek(0)
    with av.open(source, "r") as container:
        if not container.streams.video:
            raise ValueError("the file has no video stream")
        stream = container.streams.video[0]
        start_pts, end_pts = _window(stream, trim)
        first = _first_frame(container, stream, start_pts, end_pts)
        if first is None:
            raise ValueError("the file has no frame in the video's trim window")
        context = stream.codec_context
        pix_fmt = first.format.name
        width, height = first.width, first.height
        reading = yuv.reading(
            pix_fmt,
            width,
            height,
            _tag(COLOR_SPACES, getattr(first, "colorspace", None), context.colorspace),
            _tag(COLOR_RANGES, getattr(first, "color_range", None), context.color_range),
            _chroma_location(pix_fmt, stream, first),
        )
        yuv.check_size(reading.planar, width, height)
        turns = quarter_turns(getattr(first, "rotation", None))
        upright = (height, width) if turns % 2 else (width, height)
        rect = crop_rect(crop, *upright)
        size = upright if rect is None else (rect[2], rect[3])
        primaries = int(getattr(first, "color_primaries", None) or context.color_primaries or 2)
        transfer = int(getattr(first, "color_trc", None) or context.color_trc or 2)
        aspect = stream.sample_aspect_ratio or getattr(first, "sample_aspect_ratio", None)
        sample_aspect = Fraction(cast(Any, aspect)) if aspect else Fraction(1)
        if turns % 2:
            sample_aspect = 1 / sample_aspect
        declared = stream.average_rate or stream.guessed_rate
        if not declared:
            raise ValueError("the file's video stream has no frame rate")
        container_name = container.format.name
        rate = ntsc_reading(Fraction(cast(Any, declared)), container_name)
        time_base = Fraction(cast(Any, stream.time_base))
        index = stream.index
        first_pts = int(first.pts)
    frames = count_frames(source, index, start_pts, end_pts, rate)
    return Probe(
        index=index,
        stored=(width, height),
        pix_fmt=pix_fmt,
        reading=reading,
        primaries=primaries,
        transfer=transfer,
        rate=rate,
        frames=frames,
        turns=turns,
        crop=rect,
        size=size,
        container=container_name,
        time_base=time_base,
        start_pts=start_pts,
        end_pts=end_pts,
        first_pts=first_pts,
        sample_aspect=sample_aspect,
    )


def _chroma_location(pix_fmt: str, stream: Any, frame: Any) -> str:
    """The stream's chroma siting, the command line's order (DESIGN.md, Input: the stream's, else
    the first frame's), by FFmpeg's names; "" where neither says. PyAV before 19 gives neither,
    which leaves the command line's guess, left, but for the JPEG family, which ffmpeg sites at
    the centre itself (the CLI's tests/test_decode.py)."""
    found = _tag(
        CHROMA_LOCATIONS,
        getattr(stream.codec_context, "chroma_sample_location", None),
        getattr(frame, "chroma_location", None),
    )
    if found:
        return found
    if not hasattr(frame, "chroma_location"):
        if pix_fmt.startswith("yuvj"):
            return "center"
        if yuv.PIXEL_FORMATS.get(pix_fmt) is not None and yuv.PIXEL_FORMATS[pix_fmt].subsampled:
            logger.info(
                "this ComfyUI's PyAV gives no chroma siting: the video's is taken as untagged"
                " (left, the H.264, HEVC and MPEG-2 default); PyAV 19 and later give it"
            )
    return ""


def _first_frame(container: Any, stream: Any, start_pts: int, end_pts: int | None) -> Any:
    """The window's first decoded frame, None where it has none."""
    if start_pts:
        container.seek(start_pts, stream=stream)
    for frame in container.decode(stream):
        if frame.pts is None or frame.pts < start_pts:
            continue
        if end_pts is not None and frame.pts >= end_pts:
            return None
        return frame
    return None


def count_frames(
    source: str | BytesIO, index: int, start_pts: int, end_pts: int | None, rate: Fraction
) -> int:
    """The frames of the window, counted by decoding each once (no conversion), as the command
    line's first pass counts a source's: a packet count is no frame count where a decoder drops
    frames (a stream cut before a keyframe) or makes more. Each frame is held to the rate the
    stream is read at, as the command line holds a source (DESIGN.md, Input, the frame rate:
    half a frame or more from its place on the rate's timeline, counted from the first frame, is
    a source that doesn't keep its rate): the node numbers its file's frames at that rate and
    copies the audio by its own times, so a variable rate would drift from it; refused
    (words.off_the_rate). A decode that fails is refused naming its error (words.undecodable)."""
    import av

    if isinstance(source, BytesIO):
        source.seek(0)
    count = 0
    first: int | None = None
    container: Any
    with av.open(source, "r") as container:
        stream = container.streams[index]
        stream.thread_type = "AUTO"
        time_base = Fraction(stream.time_base)
        if start_pts:
            container.seek(start_pts, stream=stream)
        try:
            for frame in container.decode(stream):
                pts: int | None = frame.pts
                if pts is None or not _within(pts, start_pts, None):
                    continue
                if end_pts is not None and pts >= end_pts:
                    break
                if first is None:
                    first = pts
                at = (pts - first) * time_base
                if abs(at - count / rate) * 2 >= 1 / rate:
                    raise words.off_the_rate(count, at, rate)
                count += 1
        except av.FFmpegError as error:  # pyright: ignore[reportPrivateImportUsage]
            raise words.undecodable(str(error)) from error
    return count


def planes_of(frame: Any, reading: yuv.Reading) -> list[npt.NDArray[np.integer]]:
    """A decoded frame's three planes in `reading.planar`, each (h, w) cropped from its line size:
    a packed or semi-planar format repacked to its planar twin first (frame.reformat, which only
    moves the samples: impl-video-1's review, PyAV 17 and 19)."""
    if frame.format.name != reading.planar:
        frame = frame.reformat(format=reading.planar)
    depth = reading.format.depth
    dtype = np.dtype(np.uint8) if depth == 8 else np.dtype("<u2")
    found: list[npt.NDArray[np.integer]] = []
    for plane in frame.planes[:3]:
        values = np.frombuffer(plane, dtype=dtype)
        rows = values.reshape(plane.height, plane.line_size // dtype.itemsize)
        found.append(rows[:, : plane.width])
    return found


class Decoded:
    """A file's frames read forward from its window's start, a frame at a time (next_frames): each
    (H, W, 3) float32 in [0, 1], 16-bit codes over 65535, as the command line's reader gives them
    (yuv.to_rgb), turned and cropped as ComfyUI turns and crops them; converted on `device` (the
    card ComfyUI runs on, the processor without one) and brought to the host behind the library's
    poll (model.synchronize), which a stop and ComfyUI's Cancel reach. The window's frames are
    probe.frames; a file that gives another count raises ValueError when found. Closed by close(),
    or as a context manager."""

    def __init__(self, source: str | BytesIO, probe: Probe, device: "torch.device") -> None:
        import av

        if isinstance(source, BytesIO):
            source.seek(0)
        self.probe, self.device = probe, device
        self.given = 0
        self._container: Any = av.open(source, "r")
        try:
            stream = self._container.streams[probe.index]
            stream.thread_type = "AUTO"
            if probe.start_pts:
                self._container.seek(probe.start_pts, stream=stream)
            self._frames: Iterator[Any] = self._container.decode(stream)
        except BaseException:
            self._container.close()
            raise

    def _next_decoded(self) -> Any:
        """The window's next decoded frame, None past its end."""
        probe = self.probe
        for frame in self._frames:
            if frame.pts is None or frame.pts < probe.start_pts:
                continue
            if probe.end_pts is not None and frame.pts >= probe.end_pts:
                return None
            return frame
        return None

    def next_frames(self, count: int) -> npt.NDArray[np.float32]:
        """The next `count` frames, (count, H, W, 3) float32 in [0, 1]. ValueError where the window
        has fewer than its count, or more once the last is given."""
        import torch

        from seedvr2x import model

        height, width = self.probe.size[1], self.probe.size[0]
        out = np.empty((count, height, width, 3), dtype=np.float32)
        for index in range(count):
            frame = self._next_decoded()
            if frame is None:
                raise ValueError(
                    f"the file gave {self.given} frames of its window, where the count found"
                    f" {self.probe.frames}"
                )
            rgb = self.convert(frame)
            model.synchronize(rgb.device)
            out[index] = rgb.to("cpu", torch.float32).numpy()
            self.given += 1
        if self.given == self.probe.frames and self._next_decoded() is not None:
            raise ValueError(f"the file gave more frames than the count found, {self.probe.frames}")
        return out

    def convert(self, frame: Any) -> "torch.Tensor":
        """One decoded frame as RGB on the device, (H, W, 3) float32: its planes through
        yuv.to_rgb, turned, then cropped."""
        import torch

        probe = self.probe
        planes = [
            torch.as_tensor(np.ascontiguousarray(plane)).to(self.device)[None]
            for plane in planes_of(frame, probe.reading)
        ]
        rgb = yuv.to_rgb(planes, probe.reading)[0]
        if probe.turns:
            rgb = torch.rot90(rgb, k=probe.turns, dims=(0, 1))
        if probe.crop is not None:
            x, y, w, h = probe.crop
            rgb = rgb[y : y + h, x : x + w]
        return rgb

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> "Decoded":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def frames_reader(decoded: Decoded) -> Callable[[int], npt.NDArray[np.float32]]:
    """read(n), the next n frames of the file (Decoded.next_frames)."""
    return decoded.next_frames
