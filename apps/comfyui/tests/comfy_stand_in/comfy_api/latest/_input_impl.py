"""The stand-in for ComfyUI 0.45.3's VIDEO objects (comfy_api/latest/_input_impl/video_types.py
and _util/video_types.py), written from their reading in the tests' own words, no ComfyUI code:
what seedvr2x's nodes call on them.

- VideoComponents: the frames (T, H, W, C) float32, the rate as a Fraction, the audio.
- VideoFromFile: a path or a BytesIO, a trim (start_time, duration in seconds; a negative start
  from the end, resolved by get_active_trim_window) and a crop (x, y, width, height), each kept in
  a name-mangled private attribute as the real one keeps them (`_VideoFromFile__crop`: the crop
  has no getter); get_stream_source, get_active_trim_window, get_frame_rate (the first video
  stream's average rate, through PyAV), as_trimmed and as_cropped (a new object on the same
  source, as the real ones), and get_components, which decodes the window whole through PyAV
  (`gbrpf32le`, as the real one reads a YUV or 16-bit source; the tests only read their rate and
  shape from it).
- VideoFromComponents: frames ComfyUI holds, get_components alone (the real one's
  get_stream_source encodes them to H.264 in memory; the nodes never call it)."""

from dataclasses import dataclass
from fractions import Fraction
from io import BytesIO
from typing import Any, cast


@dataclass
class VideoComponents:
    images: Any
    frame_rate: Fraction
    audio: Any = None
    metadata: dict[str, Any] | None = None
    alpha: Any = None


class VideoFromFile:
    def __init__(
        self,
        file: str | BytesIO,
        *,
        start_time: float = 0,
        duration: float = 0,
        crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.__file = file
        self.__start_time = start_time
        self.__duration = duration
        self.__crop = crop

    def get_stream_source(self) -> str | BytesIO:
        """The source itself, the same object; a bytes buffer is rewound to its start first."""
        if isinstance(self.__file, BytesIO):
            self.__file.seek(0)
        return self.__file

    def _raw_duration(self) -> float:
        import av

        with av.open(self.get_stream_source(), "r") as container:
            return float(container.duration or 0) / 1_000_000

    def get_active_trim_window(self) -> tuple[float, float]:
        """(start, duration) in seconds as the VIDEO was made with them (0 and 0: no trim; a
        duration of 0 runs to the end); a negative start counts back from the end of the untrimmed
        video and is never below 0."""
        start = float(self.__start_time)
        if start < 0:
            start = max(0.0, self._raw_duration() + start)
        return start, float(self.__duration)

    def get_duration(self) -> float:
        """The trimmed video's length in seconds: the window's, cut short by the end of the file."""
        start, duration = self.get_active_trim_window()
        left = max(0.0, self._raw_duration() - start)
        return min(duration, left) if duration else left

    def get_frame_rate(self) -> Fraction:
        import av

        with av.open(self.get_stream_source(), "r") as container:
            return Fraction(cast(Any, container.streams.video[0].average_rate))

    def as_trimmed(
        self, start_time: float = 0, duration: float = 0, strict_duration: bool = True
    ) -> "VideoFromFile | None":
        """A new VIDEO on the same source, its start this one's plus `start_time`, its duration
        `duration`, this one's crop; None when `strict_duration` and a duration other than 0 is
        given and the new VIDEO is shorter than it."""
        trimmed = VideoFromFile(
            self.get_stream_source(),
            start_time=self.__start_time + start_time,
            duration=duration,
            crop=self.__crop,
        )
        if strict_duration and duration and trimmed.get_duration() < duration:
            return None
        return trimmed

    def as_cropped(
        self, x: int = 0, y: int = 0, width: int = 0, height: int = 0
    ) -> "VideoFromFile":
        return VideoFromFile(
            self.__file,
            start_time=self.__start_time,
            duration=self.__duration,
            crop=(x, y, width, height),
        )

    def get_components(self) -> VideoComponents:
        import av
        import numpy as np
        import torch

        start, duration = self.get_active_trim_window()
        frames: list[Any] = []
        with av.open(self.get_stream_source(), "r") as container:
            stream = container.streams.video[0]
            start_pts = int(start / stream.time_base)
            end_pts = int((start + duration) / stream.time_base)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < start_pts:
                    continue
                if duration and frame.pts >= end_pts:
                    break
                frames.append(frame.to_ndarray(format="gbrpf32le"))
            rate = Fraction(cast(Any, stream.average_rate))
        images = torch.from_numpy(np.stack(frames))
        return VideoComponents(images, rate)


class VideoFromComponents:
    def __init__(self, components: VideoComponents) -> None:
        self.__components = components

    def get_components(self) -> VideoComponents:
        return self.__components

    def get_frame_rate(self) -> Fraction:
        return self.__components.frame_rate
