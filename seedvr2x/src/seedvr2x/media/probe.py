"""What seedvr2x reads of a video stream, through ffprobe."""

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from seedvr2x.media.ffmpeg import MediaError

logger = logging.getLogger(__name__)

# ffprobe's words for an absent tag.
UNTAGGED = frozenset({"", "unknown", "unspecified"})

# The video packets decoded for the first frame. The stream's own ffprobe call decodes the first
# alone, which gives a frame unless the stream starts away from a keyframe or an MP4 edit list
# hides its first frames: ffmpeg -ss with -c copy keeps the frames from the keyframe before the
# cut, which the mov demuxer marks to be dropped once decoded (libavformat/mov.c:4567,
# 11680-11682 at n9.0.2) and the decoder drops, however many (libavcodec/decode.c:464,
# 1584-1585). Should it give none, a second call decodes up to 250 packets, a provisional bound
# (DESIGN.md sets none): x264's and x265's default longest GOP (keyframes 250 and 248 packets
# apart through ffmpeg n9.0.2's libx264 and libx265), so that a stream starting anywhere in such a
# GOP, an MPEG-TS recording or a raw stream cut, still reaches a frame. That call ignores MP4's
# edit list (the mov demuxer's ignore_editlist, mov.c:6503, which other demuxers skip with a
# warning, fftools/ffprobe.c:2590-2591), so that the frames it hides decode from the cut's
# keyframe on, a GOP longer than 250 included (SVT-AV1's default: keyframes 321 packets apart at
# 60 fps through n9.0.2's libsvtav1); and it decodes on every CPU, where ffprobe leaves its
# decoders on libavcodec's default, one thread (libavcodec/options_table.h:216). The stream stays
# as the first call declares it, an edit list ignored changing its start and duration. Measured
# with n9.0.2: the first packet adds 2-4 ms to the stream's probe on 1080p HEVC and H.264, 15-18
# ms on 2160p HEVC, 56-57 ms on 1080p 10-bit FFV1 (a frame decoded on one thread); the second call
# takes 0.16 s on 1080p H.264, 0.21-0.23 s on 1080p HEVC, 0.96 s on 2160p 10-bit HEVC and
# 0.36-0.38 s on 1080p AV1 (libdav1d), film encoded with a single keyframe and cut 249 to 1000
# frames in; on one thread, the edit list kept, it took 0.54-0.58, 0.67-0.73, 6.99 and 0.26-0.27 s.
FIRST_PACKETS = 1
MOST_PACKETS = 250
LATER = ("-ignore_editlist", "1", "-threads", "0")

# What to do with a stream whose start doesn't decode.
CUT_OR_ENCODE = "cut it at a keyframe, or re-encode it"

# The side data a decoder gives a frame carrying Dolby Vision's metadata, its RPU, as found and
# as parsed (ffprobe's names, libavutil/side_data.c:44-45 at n9.0.2), either of which shows Dolby
# Vision where no configuration record says it. ffmpeg's HEVC decoder attaches the first to every
# frame whose access unit holds an RPU, whatever the stream declares, and the second once the
# RPU's metadata is complete (libavcodec/hevc/hevcdec.c:3740-3769, 3131-3140,
# dovi_rpudec.c:37-38); AV1 frames decoded by libdav1d carry the second alone, parsed from an
# ITU-T T.35 metadata OBU (libavcodec/libdav1d.c:498, itut35.c:357-358).
DOVI_FRAME = frozenset({"Dolby Vision RPU Data", "Dolby Vision Metadata"})

# ffprobe's name for a Dolby Vision configuration record in a stream's side data
# (libavcodec/packet.c:301 at n9.0.2), which ffmpeg's demuxers read from MP4's dvcC, dvvC and dvwC
# boxes and Matroska's block addition mappings (libavformat/dovi_isom.c:32, called from
# mov.c:8854 and matroskadec.c:2510), and from MPEG-TS's descriptor (mpegts.c:2444).
DOVI = "DOVI configuration record"

# What seedvr2x reads of it, ffprobe's keys (fftools/ffprobe.c:1150-1168 at n9.0.2), each with its
# width in the record, in bits (libavformat/dovi_isom.c:52-62): ffprobe writes them as integers.
DOVI_FIELDS = {
    "dv_profile": 7,
    "rpu_present_flag": 1,
    "el_present_flag": 1,
    "bl_present_flag": 1,
    "dv_bl_signal_compatibility_id": 4,
}


@dataclass(frozen=True)
class DoviRecord:
    """What seedvr2x reads of a stream's Dolby Vision configuration record, with ffprobe's names
    (ffmpeg's AVDOVIDecoderConfigurationRecord, libavutil/dovi_meta.h)."""

    dv_profile: int
    rpu_present_flag: bool  # Dolby Vision's metadata, the RPU
    el_present_flag: bool  # an enhancement layer
    bl_present_flag: bool  # a base layer, what a player without Dolby Vision shows
    dv_bl_signal_compatibility_id: int  # what the base layer is: 2 for SDR (media/source.py)


@dataclass(frozen=True)
class FirstFrame:
    """What the stream's first frame decoded carries, where the stream's tags can hide it: a
    decoder tags its frames with the bitstream's own (HEVC: libavcodec/hevc/hevcdec.c:350-373 at
    n9.0.2), while the stream gets the container's primaries, transfer and matrix, all three,
    when the container declares any one of them, and its range and chroma location each when
    declared (libavformat/demux.c:2584-2598). A Matroska file tagging the matrix alone, or BT.709
    throughout, thus hides a PQ bitstream. Behind an MP4 edit list, it is the first frame the
    list hides, from the same keyframe on (probe). Its colour tags, read with the stream's
    (media/source.py, COLOUR), are VideoStream's, "" when untagged."""

    color_transfer: str
    dovi: bool  # Dolby Vision's metadata, its RPU (DOVI_FRAME)
    color_space: str = ""
    color_range: str = ""
    chroma_location: str = ""
    color_primaries: str = ""


@dataclass(frozen=True)
class VideoStream:
    """The first video stream of a file, as declared: ffprobe's stream level, whose colour tags a
    source resolves with its first frame's (media/source.py, resolved). An absent tag reads "",
    whatever word ffprobe has for it ("unknown", "unspecified")."""

    codec_name: str  # ffprobe's name: "h264", "hevc", "mpeg2video"...
    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction  # r_frame_rate, exact (24000/1001, not 23.976)
    avg_frame_rate: Fraction | None  # None when ffprobe has none (0/0)
    field_order: str  # "progressive", "tt", "bb", "tb", "bt", or "" (unknown)
    sample_aspect: Fraction | None  # None when undeclared (0:1)
    color_space: str  # the matrix: "bt709", "smpte170m", "gbr"...
    color_range: str  # "tv" or "pc"
    color_primaries: str
    color_transfer: str
    chroma_location: str  # "left", "center", "topleft", "top", "bottomleft", "bottom"
    display_matrix: tuple[int, ...] | None  # the 9 values of a display matrix, if declared
    cropped: bool  # a crop declared by the container (MP4 clap, Matroska PixelCrop)
    dovi: DoviRecord | None  # a Dolby Vision configuration record, if declared
    first_frame: FirstFrame | None  # None when not read (probe's first_frame, verify's)


def probe(path: Path, first_frame: bool = True) -> VideoStream:
    """The first video stream of path, as declared, and, when `first_frame`, what its first frame
    decoded carries. verify reads no frame here: it decodes every frame of what seedvr2x wrote.

    Raises MediaError when ffprobe fails, for no video stream, one ffprobe has no decoder for or
    found no pixel format or size for, or, when `first_frame`, no frame decoded from the
    stream's first MOST_PACKETS packets."""
    if not first_frame:
        return parse(path, _ffprobe(path, "-show_streams"), first_frame=False)
    output = _ffprobe(path, "-show_streams", *_frames(FIRST_PACKETS))
    probed = cast(dict[str, Any], json.loads(output))
    streams = cast(list[dict[str, Any]], probed.get("streams") or [])
    # Without a video stream, ffprobe has read the whole file for its packet: parse refuses it.
    if streams and not probed.get("frames"):
        logger.info(
            "%s: no frame decoded from its first packet: decoding up to %d for one",
            path,
            MOST_PACKETS,
        )
        later = json.loads(
            _ffprobe(
                path, *LATER, "-show_entries", "stream=nb_read_packets", *_frames(MOST_PACKETS)
            )
        )
        # Its frames and the count of its packets read; its stream the first call's.
        [counted] = cast(list[dict[str, Any]], later.get("streams") or [{}])
        streams[0]["nb_read_packets"] = counted.get("nb_read_packets", 0)
        probed["frames"] = later.get("frames") or []
        output = json.dumps(probed)
    return parse(path, output)


def _frames(packets: int) -> tuple[str, ...]:
    """ffprobe's options for the frames decoded from the stream's first `packets` packets, with
    the count of those read (nb_read_packets, fftools/ffprobe.c:2052-2053 at n9.0.2): ffprobe
    drains the decoder at the end of the packets read (ffprobe.c:1798-1803), so a frame-threaded
    decoder gives them too."""
    return ("-count_packets", "-show_frames", "-read_intervals", f"%+#{packets}")


def _ffprobe(path: Path, *options: str) -> str:
    """ffprobe's JSON for the first video stream of path, with `options`."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *options, "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise MediaError(f"{path}: ffprobe failed: {result.stderr.strip()}")
    return result.stdout


def parse(path: Path, output: str, first_frame: bool = True) -> VideoStream:
    """The first video stream of path and, when `first_frame`, its first frame, from ffprobe's
    JSON output on it (probe's commands).

    Raises MediaError for no video stream, one ffprobe has no decoder for or found no pixel
    format or size for, no frame when `first_frame`, or an unreadable Dolby Vision record."""
    probed = json.loads(output)
    streams = cast(list[dict[str, Any]], probed.get("streams", []))
    if not streams:
        raise MediaError(f"{path}: no video stream")
    stream = streams[0]
    frames = cast(list[dict[str, Any]], probed.get("frames") or [])
    unread = _unread(stream, frames if first_frame else None)
    if unread:
        raise MediaError(f"{path}: {unread}")
    side_data = cast(list[dict[str, Any]], stream.get("side_data_list") or [])
    matrices = [d for d in side_data if d.get("side_data_type") == "Display Matrix"]
    records = [d for d in side_data if d.get("side_data_type") == DOVI]
    frame_rate = _fraction(stream.get("r_frame_rate"))
    if frame_rate is None:
        raise MediaError(f"{path}: no frame rate declared")
    return VideoStream(
        codec_name=_tag(stream.get("codec_name")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=str(stream["pix_fmt"]),
        frame_rate=frame_rate,
        avg_frame_rate=_fraction(stream.get("avg_frame_rate")),
        field_order=_tag(stream.get("field_order")),
        sample_aspect=_fraction(stream.get("sample_aspect_ratio")),
        color_space=_tag(stream.get("color_space")),
        color_range=_tag(stream.get("color_range")),
        color_primaries=_tag(stream.get("color_primaries")),
        color_transfer=_tag(stream.get("color_transfer")),
        chroma_location=_tag(stream.get("chroma_location")),
        display_matrix=_display_matrix(str(matrices[0].get("displaymatrix"))) if matrices else None,
        cropped=any(d.get("side_data_type") == "Frame Cropping" for d in side_data),
        dovi=_dovi(path, records[0]) if records else None,
        first_frame=_first_frame(frames[0]) if first_frame else None,
    )


def _unread(stream: dict[str, Any], frames: list[dict[str, Any]] | None) -> str:
    """Why ffprobe's stream can't be read, or "": no decoder for it, no pixel format or size
    found, or, unless `frames` is None (not read), no frame decoded."""
    sized = all(stream.get(key) for key in ("pix_fmt", "width", "height"))
    if sized and (frames is None or frames):
        return ""
    # ffprobe writes a coded size only for a stream it has a decoder for, and no codec name for a
    # codec it doesn't know (fftools/ffprobe.c:1945-1947, 1905-1912 at n9.0.2).
    if "coded_width" not in stream:
        codec = stream.get("codec_name")
        missing = (
            f"no decoder for its video codec, {codec}, in this ffmpeg build"
            if codec
            else "its video codec unknown to this ffmpeg build"
        )
        return f"{missing}: use a build that reads it, or re-encode it with a tool that does"
    # ffprobe finds them in the stream's start, 5 s of it, 7 in MPEG-TS
    # (libavformat/demux.c:2639-2647), which gives nothing before a keyframe.
    if not sized:
        return (
            "no pixel format or size found for its video stream in the start ffprobe analyses,"
            f" as when it starts away from a keyframe: {CUT_OR_ENCODE}"
        )
    read = int(stream.get("nb_read_packets") or 0)  # absent when none
    if not read:
        return "no packet in its video stream"
    packets = (
        f"the first {read} packets"
        if read >= MOST_PACKETS
        else "the only packet"
        if read == 1
        else f"all {read} packets"
    )
    return (
        f"no frame decoded from {packets} of its video stream, so what its frames carry, an HDR"
        f" transfer or Dolby Vision's metadata, can't be checked: {CUT_OR_ENCODE}"
    )


def _first_frame(frame: dict[str, Any]) -> FirstFrame:
    side_data = cast(list[dict[str, Any]], frame.get("side_data_list") or [])
    return FirstFrame(
        color_transfer=_tag(frame.get("color_transfer")),
        dovi=any(d.get("side_data_type") in DOVI_FRAME for d in side_data),
        color_space=_tag(frame.get("color_space")),
        color_range=_tag(frame.get("color_range")),
        chroma_location=_tag(frame.get("chroma_location")),
        color_primaries=_tag(frame.get("color_primaries")),
    )


def _tag(value: object) -> str:
    text = "" if value is None else str(value)
    return "" if text in UNTAGGED else text


def _fraction(value: object) -> Fraction | None:
    """A positive ratio written "num/den" (frame rates) or "num:den" (sample aspect), else None:
    ffprobe writes 0/0 and 0:1 for none."""
    match = re.fullmatch(r"(\d+)[/:](\d+)", "" if value is None else str(value))
    if match is None or int(match[1]) == 0 or int(match[2]) == 0:
        return None
    return Fraction(int(match[1]), int(match[2]))


def _display_matrix(dump: str) -> tuple[int, ...]:
    """The 9 values of ffprobe's display matrix dump, three rows "0000000N: a b c"."""
    values = tuple(int(v) for line in dump.splitlines() if ":" in line for v in line.split()[1:])
    if len(values) != 9:
        raise MediaError(f"unreadable display matrix: {dump!r}")
    return values


def _dovi(path: Path, record: dict[str, Any]) -> DoviRecord:
    """What seedvr2x reads of a Dolby Vision configuration record, as ffprobe writes it.

    Raises MediaError for a field missing, or not a whole number its width holds: the base layer
    is then unknown, so the source is refused, never read on a guess."""
    values = [record.get(key) for key in DOVI_FIELDS]
    if not all(
        type(value) is int and 0 <= value < 1 << bits
        for value, bits in zip(values, DOVI_FIELDS.values(), strict=True)
    ):
        raise MediaError(
            f"{path}: unreadable Dolby Vision configuration record, its base layer unknown:"
            f" {json.dumps(record)}"
        )
    profile, rpu, el, bl, compatibility = cast(list[int], values)
    return DoviRecord(profile, bool(rpu), bool(el), bool(bl), compatibility)
