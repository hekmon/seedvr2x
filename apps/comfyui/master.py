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

No node writes it yet: Upscale Video takes it with the VIDEO input, in the pass after this one."""

import logging
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from seedvr2x import yuv

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
    ) -> None:
        import av

        self.path, self.width, self.height = path, width, height
        self.frames = 0  # the frames written
        self._container: Any = av.open(str(path), "w", format="matroska")
        try:
            stream: Any = self._container.add_stream("ffv1", rate=rate, options=OPTIONS)
            stream.width, stream.height = width, height
            stream.pix_fmt = yuv.MASTER.planar
            context = stream.codec_context
            context.colorspace, context.color_range = BT709, LIMITED
            context.color_primaries, context.color_trc = primaries, transfer
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
        import av

        from seedvr2x import model

        if tuple(rgb.shape[1:]) != (self.height, self.width, 3):
            raise ValueError(
                f"frames of {tuple(rgb.shape)} for a master of {self.width}x{self.height}"
            )
        for index in range(rgb.shape[0]):
            made = yuv.master_planes(rgb[index : index + 1])[0]
            model.synchronize(made.device)
            planes = made.cpu().numpy()
            frame = av.VideoFrame(self.width, self.height, yuv.MASTER.planar)
            for plane, values in zip(frame.planes, planes, strict=True):
                # A plane's rows may be padded past its width (its line size).
                padded = np.zeros((self.height, plane.line_size // 2), dtype="<u2")
                padded[:, : self.width] = values
                plane.update(padded.tobytes())
            frame.pts = self.frames
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self.frames += 1

    def close(self) -> None:
        """The encoder flushed and the file closed: whole once this returns. The file is closed
        whatever the flush raises."""
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()

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
            self._container.close()
        except Exception as closing:
            # The exception that ended the writing is the one raised; the close's said beside it.
            logger.warning("the partial master %s could not be closed: %s", self.path, closing)
