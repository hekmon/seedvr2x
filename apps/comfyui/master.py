"""The lossless file Upscale Video will write (DESIGN.md, Repository and apps, the nodes' VIDEO
path): FFV1 in Matroska, 16-bit 4:4:4 YCbCr tagged BT.709 limited range, its planes the library's
(`seedvr2x.yuv.master_planes`), written a shot at a time into one file kept open for the run,
through PyAV, which ComfyUI ships (its requirements.txt: av 17.0.0 and later). Save Video re-encodes
such a file right, where a 16-bit RGB master came out coded as BT.601 and tagged wrong, and remuxes
it into Matroska as it is (impl-videostudy, 2026-11-12).

FFV1 as the command line writes its masters (its media/writer.py, from research/scripts/
ffv1_out.py): level 3, every frame a keyframe, 16 slices each with its CRC. The frames are numbered
at the frame rate given, an exact fraction; Matroska keeps their times to the millisecond, and a
frame's duration in whole nanoseconds, which libavformat reads back as a fraction of terms up to
30,000: the usual rates exact (24000/1001 read back as 24000/1001: impl-videostudy), 60000/1001 as
19001/317 and 120000/1001 as 29011/242, ffmpeg's own Matroska files the same (tests/
test_master.py). A file Matroska has closed can't be opened again to append to (PyAV
opens "r" or "w"), so one Master is the run's whole file.

Upscale Video writes it (stream.py): seedvr2x's record as the video stream's tags (DESIGN.md, The
record in each output, its app `seedvr2x-comfyui`), and the source's other streams, audio and
subtitles, copied into it packet for packet as the frames are written (Others), as the command
line's assembly copies them (DESIGN.md, Output)."""

import logging
from collections.abc import Sequence
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt

from seedvr2x import yuv

from . import words

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("seedvr2x_comfyui.master")

# FFV1's settings, the command line's (its media/writer.py, _ffv1_matroska).
OPTIONS = {"level": "3", "g": "1", "slices": "16", "slicecrc": "1"}

# The tags' values, ITU-T H.273's code points, which FFmpeg's enums are: BT.709's matrix and a
# limited range, the master's own (yuv.MASTER); 2, unspecified, for a
# source's primaries or transfer it doesn't say.
BT709 = 1
LIMITED = 1
UNSPECIFIED = 2


class Master:
    """The master being written at `path`: frames of `width` x `height` at `rate` frames per
    second, its primaries and transfer the source's own, H.273's code points as PyAV gives them
    (`stream.codec_context.color_primaries`, `color_trc`), copied as they are, never converted
    (DESIGN.md, Colour and shape), UNSPECIFIED where the source says none.

    Used as a context manager: on a normal exit the encoder is flushed and the file closed, on an
    exception the file closed as it is, whatever PyAV raises; a file left by an exception is a
    partial one, the caller's to remove."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        rate: Fraction,
        primaries: int = UNSPECIFIED,
        transfer: int = UNSPECIFIED,
        tags: Sequence[tuple[str, str]] = (),
        others: "Others | None" = None,
    ) -> None:
        import av

        self.path, self.width, self.height, self.rate = path, width, height, rate
        self.frames = 0  # the frames written
        self._others = others
        self._container: Any = av.open(str(path), "w", format="matroska")
        try:
            stream: Any = self._container.add_stream("ffv1", rate=rate, options=OPTIONS)
            stream.width, stream.height = width, height
            stream.pix_fmt = yuv.MASTER.planar
            context = stream.codec_context
            context.colorspace, context.color_range = BT709, LIMITED
            context.color_primaries, context.color_trc = primaries, transfer
            # The record twice: as the video stream's tags, as the command line's masters carry
            # it (DESIGN.md, The record in each output), and as the file's, which ComfyUI 0.45.3's
            # Save Video copies into the file it saves whether it copies the streams or encodes
            # them again (video_types.py, write_output_metadata), where a stream copied from its
            # template keeps no tag of its own (impl-video-2's measure, PyAV 19.0.2).
            for name, value in tags:
                stream.metadata[name] = value
                self._container.metadata[name] = value
            if others is not None:
                others.add_to(self._container)
        except BaseException:
            self._container.close()
            raise
        self._stream = stream

    def write(self, rgb: "torch.Tensor") -> None:
        """Frames `rgb`, (T, H, W, 3) float in [0, 1] on any device, appended: a shot, or any
        part of one. Their planes are made on the frames' device (yuv.master_planes) and brought
        to the host a frame at a time, the card's work waited for first through the library's
        poll (model.synchronize), which a stop and ComfyUI's Cancel reach and which marks a card a
        CUDA error failed, where a copy to the host would wait in C (AGENTS.md, the DiT's waits on
        the host)."""
        from seedvr2x import model

        if tuple(rgb.shape[1:]) != (self.height, self.width, 3):
            raise ValueError(
                f"frames of {tuple(rgb.shape)} for a master of {self.width}x{self.height}"
            )
        for index in range(rgb.shape[0]):
            made = yuv.master_planes(rgb[index : index + 1])[0]
            model.synchronize(made.device)
            self.write_planes(made.cpu().numpy()[None])

    def write_planes(self, planes: npt.NDArray[np.uint16]) -> None:
        """Frames given as their planes, (T, 3, H, W) uint16, Y, Cb, Cr, as yuv.master_planes makes
        them, appended; the source's other packets up to each frame's time muxed before it
        (Others.advance)."""
        import av

        if tuple(planes.shape[1:]) != (3, self.height, self.width):
            raise ValueError(
                f"planes of {tuple(planes.shape)} for a master of {self.width}x{self.height}"
            )
        for values in planes:
            if self._others is not None:
                self._others.advance(self._container, Fraction(self.frames) / self.rate)
            frame = av.VideoFrame(self.width, self.height, yuv.MASTER.planar)
            for plane, each in zip(frame.planes, values, strict=True):
                # A plane's rows may be padded past its width (its line size).
                padded = np.zeros((self.height, plane.line_size // 2), dtype="<u2")
                padded[:, : self.width] = each
                plane.update(padded.tobytes())
            frame.pts = self.frames
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self.frames += 1

    def close(self) -> None:
        """The encoder flushed, the source's other packets left muxed, and the file closed: whole
        once this returns. The file is closed whatever the flush raises."""
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
            if self._others is not None:
                self._others.advance(self._container, None)
        finally:
            self._close()

    def _close(self) -> None:
        try:
            self._container.close()
        finally:
            if self._others is not None:
                self._others.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if kind is None:
            self.close()
            return
        try:
            self._close()
        except Exception as closing:
            # The exception that ended the writing is the one raised; the close's said beside it.
            logger.warning("the partial master %s could not be closed: %s", self.path, closing)


# The kinds of the source's streams copied into the master beside its video: audio and subtitles,
# packet for packet. A data stream, an attachment and another video stream (a cover picture among
# them) are left out, each said: Matroska holds no data stream, and the command line's assembly
# attaches a cover where the nodes' file has none (DESIGN.md, Output).
COPIED = ("audio", "subtitle")


def _described(stream: Any) -> str:
    codec = getattr(stream.codec_context, "name", None) or "unknown codec"
    language = stream.metadata.get("language")
    said = f"stream {stream.index} ({stream.type}, {codec}"
    return said + (f", {language})" if language else ")")


def other_streams(source: str | BytesIO, video: int) -> tuple[list[str], list[str]]:
    """The source's streams but its video stream `video`: (those copied, those left out), each
    described ("stream 1 (audio, aac, eng)"). Each copied one is tried into Matroska's muxer first,
    in memory: one it refuses is refused in the nodes' words (words.stream_unholdable), as the
    command line refuses it."""
    import av

    if isinstance(source, BytesIO):
        source.seek(0)
    copied: list[str] = []
    left: list[str] = []
    with av.open(source, "r") as container:
        for stream in container.streams:
            if stream.index == video:
                continue
            if stream.type not in COPIED:
                left.append(_described(stream))
                continue
            trial: Any = av.open(BytesIO(), "w", format="matroska")
            try:
                trial.add_stream_from_template(stream)
                trial.start_encoding()
            except Exception as error:
                raise words.stream_unholdable(_described(stream), str(error)) from error
            finally:
                try:
                    trial.close()
                except Exception:  # a trial never started: nothing to keep
                    logger.debug("a trial muxer not closed cleanly", exc_info=True)
            copied.append(_described(stream))
    return copied, left


class Others:
    """The source's audio and subtitle streams (COPIED), copied into the master packet for packet,
    their packets muxed in their order as the frames' times reach them (advance), so that the
    file is interleaved and nothing waits in memory but a packet. `first` is the time of the
    video's first frame in the source, in seconds, subtracted from every packet's, so that the
    streams keep their place against the video (Matroska shifts all of them alike where one would
    start before 0). `window`, a trimmed VIDEO's (start, end) in seconds (end None: to the file's
    end), copies the packets that overlap it, whole: a copy cuts at its packets, so that the
    audio may begin a packet before the trim's start and end a packet after its end (said:
    `copied`). None: every packet."""

    def __init__(
        self,
        source: str | BytesIO,
        video: int,
        first: Fraction,
        window: tuple[Fraction, Fraction | None] | None = None,
    ) -> None:
        import av

        if isinstance(source, BytesIO):
            source.seek(0)
        self._first, self._window = first, window
        self._container: Any = av.open(source, "r")
        self.streams: list[Any] = [
            stream
            for stream in self._container.streams
            if stream.index != video and stream.type in COPIED
        ]
        self._outputs: dict[int, Any] = {}
        self._packets: Any = self._container.demux(*self.streams) if self.streams else iter(())
        self._waiting: Any = None
        # Per stream copied: (packets, first time, last end), in seconds of the source.
        self.copied: dict[int, tuple[int, Fraction, Fraction]] = {}
        # The packets left out for having no time to place them by, which the node says.
        self.untimed = 0

    def add_to(self, container: Any) -> None:
        """The streams added to the master's container, from their templates."""
        for stream in self.streams:
            self._outputs[stream.index] = container.add_stream_from_template(stream)

    def _time(self, packet: Any) -> Fraction | None:
        stamp = packet.pts if packet.pts is not None else packet.dts
        if stamp is None:
            return None
        return Fraction(stamp) * Fraction(packet.time_base)

    def advance(self, container: Any, until: Fraction | None) -> None:
        """Mux the packets whose time, less `first`, is before `until` seconds (None: every one
        left)."""
        while True:
            packet = self._waiting
            self._waiting = None
            if packet is None:
                packet = next(self._packets, None)
                if packet is None:
                    return
            time = self._time(packet)
            if packet.size == 0:
                continue  # a demuxer's flush: nothing to copy
            if time is None:
                self.untimed += 1
                continue
            if until is not None and time - self._first >= until:
                self._waiting = packet
                return
            self._copy(container, packet, time)

    def _copy(self, container: Any, packet: Any, time: Fraction) -> None:
        time_base = Fraction(packet.time_base)
        end = time + Fraction(packet.duration or 0) * time_base
        if self._window is not None:
            start, stop = self._window
            if (end <= start and packet.duration) or (time < start and not packet.duration):
                return
            if stop is not None and time >= stop:
                return
        index = packet.stream.index
        shift = round(self._first / time_base)
        if packet.pts is not None:
            packet.pts -= shift
        if packet.dts is not None:
            packet.dts -= shift
        packet.stream = self._outputs[index]
        container.mux(packet)
        count, first, last = self.copied.get(index, (0, time, end))
        self.copied[index] = (count + 1, min(first, time), max(last, end))

    def close(self) -> None:
        self._container.close()
