"""What probe reads of ffprobe's JSON, the Dolby Vision configuration record and the first frame
among it, and what they make of a source: HDR refused by the stream's transfer or its first
frame's; Dolby Vision read through its base layer when its record says SDR, refused otherwise, and
without a record (DESIGN.md, Not in the first version); the colour tags read from the stream and
from its first frame, one side's taken, both sides' contradicting refused (DESIGN.md, Input), but
the chroma location's, the stream's then taken (provisional: media/source.py, SITING). A stream
probe can't read is refused: no decoder for it, no pixel format or size found, or no frame
decoded.

The fixtures need no ffmpeg. A file made for a test, read by the installed ffprobe, skips without
a build seedvr2x accepts; one made by an encoder a build may lack (libx265, libx264, libopenh264,
or ffmpeg's own mpeg1video, mpeg2video or mpeg4), without it."""

import json
import logging
import re
import shlex
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from seedvr2x.media import ffmpeg
from seedvr2x.media import source as sources
from seedvr2x.media.conversion import Conversion, conversion_for
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.probe import DoviRecord, FirstFrame, parse, probe
from seedvr2x.media.source import SITING, declare, declared_refusal, resolved


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


def has_encoder(name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    )
    return f" {name} " in listed.stdout


def has_x265() -> bool:
    return has_encoder("libx265")


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, scdet and ffv1"
)
needs_x265 = pytest.mark.skipif(not has_x265(), reason="needs ffmpeg with libx265")
needs_mpeg4 = pytest.mark.skipif(not has_encoder("mpeg4"), reason="needs ffmpeg's mpeg4 encoder")


def needs_encoders(*names: str) -> pytest.MarkDecorator:
    """Skips a test without any of ffmpeg's encoders `names`."""
    missing = [name for name in names if not has_encoder(name)]
    return pytest.mark.skipif(bool(missing), reason=f"needs ffmpeg with {', '.join(missing)}")


def written(profile: int, el: int, bl: int, compatibility: int) -> dict[str, Any]:
    """A Dolby Vision configuration record with its RPU, version 1.0 at level 6, as ffprobe n9.0.2
    writes it in a stream's side data (fftools/ffprobe.c:1150-1168)."""
    return {
        "side_data_type": "DOVI configuration record",
        "dv_version_major": 1,
        "dv_version_minor": 0,
        "dv_profile": profile,
        "dv_level": 6,
        "rpu_present_flag": 1,
        "el_present_flag": el,
        "bl_present_flag": bl,
        "dv_bl_signal_compatibility_id": compatibility,
        "dv_md_compression": "none",
    }


# ffprobe n9.0.2's JSON (probe's command) for 2160p HEVC Main 10 streams with a Dolby Vision
# configuration record, cut to the codec and the keys probe reads, its side data whole; and for
# their first frame (FRAMES), cut to its transfer and the types of its side data. Profile 7's is
# ffmpeg's FATE sample mkv/dovi-p7-hvce.mkv: base layer, enhancement layer and RPU in one track,
# as UHD Blu-ray remuxes carry them, the RPU on every frame. The others are x265 encodes tagged as
# their base layer is, given the record in their MP4 sample entry (dvcC, dvvC), 8.1 HDR10's
# mastering display and light level as well (mdcv, clli), and remuxed to Matroska by ffmpeg: no
# RPU on their frames.
HEVC = {
    "codec_name": "hevc",
    "profile": "Main 10",
    "width": 3840,
    "height": 2160,
    "sample_aspect_ratio": "1:1",
    "pix_fmt": "yuv420p10le",
    "color_range": "tv",
    "chroma_location": "left",
    "field_order": "progressive",
    "r_frame_rate": "24000/1001",
    "avg_frame_rate": "24000/1001",
}
PQ = {"color_space": "bt2020nc", "color_transfer": "smpte2084", "color_primaries": "bt2020"}
HDR10 = [
    {"side_data_type": "Content light level metadata", "max_content": 1000, "max_average": 400},
    {
        "side_data_type": "Mastering display metadata",
        "red_x": "177/250",
        "red_y": "73/250",
        "green_x": "17/100",
        "green_y": "797/1000",
        "blue_x": "131/1000",
        "blue_y": "23/500",
        "white_point_x": "3127/10000",
        "white_point_y": "329/1000",
        "min_luminance": "1/10000",
        "max_luminance": "1000/1",
    },
]
STREAMS: dict[str, dict[str, Any]] = {
    "5": HEVC | {"side_data_list": [written(5, 0, 1, 0)]},
    "7": {key: value for key, value in HEVC.items() if key != "field_order"}
    | PQ
    | {
        "chroma_location": "topleft",
        "side_data_list": [
            written(7, 1, 1, 6),
            {"side_data_type": "HEVC enhancement-layer decoder configuration"},
        ],
    },
    "8.1": HEVC | PQ | {"side_data_list": [*HDR10, written(8, 0, 1, 1)]},
    "8.2": HEVC
    | {"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}
    | {"side_data_list": [written(8, 0, 1, 2)]},
    "8.4": HEVC
    | {"color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020"}
    | {"side_data_list": [written(8, 0, 1, 4)]},
}
SEI = {"side_data_type": "H.26[45] User Data Unregistered SEI message"}  # x265's settings
MASTERING = {"side_data_type": "Mastering display metadata"}
LIGHT = {"side_data_type": "Content light level metadata"}
RPU = [{"side_data_type": "Dolby Vision RPU Data"}, {"side_data_type": "Dolby Vision Metadata"}]
RPU_DATA, METADATA = RPU
FRAMES: dict[str, dict[str, Any]] = {
    "5": {"side_data_list": [SEI]},
    "7": {
        "color_transfer": "smpte2084",
        "side_data_list": [MASTERING, LIGHT, {"side_data_type": "SMPTE 12-1 timecode"}, *RPU],
    },
    "8.1": {"color_transfer": "smpte2084", "side_data_list": [MASTERING, LIGHT, SEI]},
    "8.2": {"color_transfer": "bt709", "side_data_list": [SEI]},
    "8.4": {"color_transfer": "arib-std-b67", "side_data_list": [SEI]},
}

# The refusals' reasons and their end, quoted whole.
TRAINED = "the model was trained on SDR video"
WHY_PQ = (
    f"{TRAINED}, what it makes of PQ-coded pixels is unmeasured, and HDR10's metadata (mastering"
    " display, MaxCLL) isn't carried"
)
WHY_HLG = f"{TRAINED}, and what it makes of HLG-coded pixels is unmeasured"
TONE_MAP = (
    "for an SDR upscale, tone-map the source to SDR first, a grading choice for your own tools, as"
    " a gamut conversion is"
)
NO_RECORD = (
    "Dolby Vision's metadata on its first frame, but no configuration record to say what its base"
    " layer is: not supported: only a base layer known to be SDR is read, never one guessed; read"
    " the file it came from if that has the record (ffmpeg keeps it remuxing to Matroska, to MP4"
    " only with -strict unofficial, never to MPEG-TS), or, if its base layer is known to be SDR,"
    " drop the metadata with ffmpeg's dovi_rpu bitstream filter (-c copy -bsf:v dovi_rpu=strip=1)"
)


def output(stream: dict[str, Any], frame: dict[str, Any]) -> str:
    """ffprobe's JSON for a stream and the one frame decoded from its first packet."""
    return json.dumps({"frames": [frame], "streams": [stream]})


@pytest.mark.parametrize(
    ("profile", "record", "refusal"),
    [
        (
            "5",
            DoviRecord(5, True, False, True, 0),
            "Dolby Vision profile 5, its base layer Dolby's own, viewable only through Dolby's"
            f" processing: not supported: {TRAINED}",
        ),
        (
            "7",
            DoviRecord(7, True, True, True, 6),
            "Dolby Vision profile 7, its base layer UHD Blu-ray's HDR10 (PQ): not supported:"
            f" {WHY_PQ}",
        ),
        (
            "8.1",
            DoviRecord(8, True, False, True, 1),
            f"Dolby Vision profile 8, its base layer HDR10 (PQ): not supported: {WHY_PQ}",
        ),
        ("8.2", DoviRecord(8, True, False, True, 2), ""),
        (
            "8.4",
            DoviRecord(8, True, False, True, 4),
            f"Dolby Vision profile 8, its base layer HLG: not supported: {WHY_HLG}",
        ),
    ],
)
def test_dolby_vision_through_its_base_layer(
    profile: str, record: DoviRecord, refusal: str
) -> None:
    # The record decides, the RPU on profile 7's frames or not.
    stream = parse(Path("dv.mkv"), output(STREAMS[profile], FRAMES[profile]))
    assert stream.dovi == record
    assert stream.first_frame == FirstFrame(
        FRAMES[profile].get("color_transfer", ""), dovi=profile == "7"
    )
    assert declared_refusal(stream) == (f"{refusal}; {TONE_MAP}" if refusal else "")


def test_no_record() -> None:
    sdr = {key: value for key, value in STREAMS["8.2"].items() if key != "side_data_list"}
    stream = parse(Path("sdr.mkv"), output(sdr, FRAMES["8.2"]))
    assert (stream.dovi, declared_refusal(stream)) == (None, "")


# ffprobe n9.0.2's JSON for streams whose first frame says what their tags don't, cut likewise, the
# frames' other colour tags left out (test_hdr_on_the_first_frame reads them, which contradict
# pq_vui_709.mkv's stream).
# The first two are x265 encodes in Matroska tagged BT.2020 PQ in their bitstream: hdr10.mkv given
# -colorspace, -color_primaries and -color_trc too, of which ffmpeg writes the matrix alone to the
# container, and pq_vui_709.mkv's frames tagged BT.709 before the encoder, which the container
# says; both made by test_hdr_on_the_first_frame. dv84.ts is ffmpeg's FATE sample hevc/dv84.mov,
# profile 8.4, its RPU on every frame, remuxed to MPEG-TS by ffmpeg, which writes no record;
# dv_untagged.ts the same stream, its colour tags dropped and its range made full, as a profile 5
# base layer's are; dv_709.ts dv84.ts tagged BT.709, stream and frame, as profile 8.2's would be.
# Each member of probe.DOVI_FRAME alone: av1dv709.ivf, an AV1 Dolby Vision profile 10 stream in
# IVF, which holds no record, tagged BT.709, its frames decoded by libdav1d carrying the metadata
# alone; rpu_alone.ts, dv_untagged.ts with its first frame given the RPU alone, as ffmpeg's HEVC
# decoder gives a frame whose RPU leaves the metadata incomplete (libavcodec/dovi_rpudec.c:37-38
# at n9.0.2).
MAIN10 = {"codec_name": "hevc", "profile": "Main 10", "pix_fmt": "yuv420p10le"}
TINY = MAIN10 | {
    "sample_aspect_ratio": "1:1",
    "color_range": "tv",
    "chroma_location": "left",
    "field_order": "progressive",
    "r_frame_rate": "25/1",
    "avg_frame_rate": "25/1",
}
DV84 = MAIN10 | {
    "width": 1920,
    "height": 1080,
    "color_range": "tv",
    "chroma_location": "left",
    "r_frame_rate": "30/1",
    "avg_frame_rate": "30/1",
}
BT709 = {"color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}
HLG = {"color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020"}
AMBIENT = {"side_data_type": "Ambient viewing environment"}
AV1 = {
    "codec_name": "av1",
    "profile": "Main",
    "width": 1080,
    "height": 1920,
    "pix_fmt": "yuv420p10le",
    "sample_aspect_ratio": "1:1",
    "color_range": "tv",
    "chroma_location": "left",
    "r_frame_rate": "30000/1001",
    "avg_frame_rate": "0/0",
}
FIRST_FRAMES: dict[str, tuple[dict[str, Any], dict[str, Any], FirstFrame, str]] = {
    "hdr10.mkv": (
        TINY | {"width": 128, "height": 72, "color_space": "bt2020nc"},
        {"color_transfer": "smpte2084", "side_data_list": [SEI, SEI, LIGHT]},
        FirstFrame("smpte2084", dovi=False),
        "HDR, transfer smpte2084 (PQ) on its first frame, where the stream declares none: not"
        f" supported: {WHY_PQ}; {TONE_MAP}",
    ),
    "pq_vui_709.mkv": (
        TINY | {"width": 64, "height": 48} | BT709,
        {"color_transfer": "smpte2084", "side_data_list": [SEI]},
        FirstFrame("smpte2084", dovi=False),
        "HDR, transfer smpte2084 (PQ) on its first frame, where the stream declares bt709: not"
        f" supported: {WHY_PQ}; {TONE_MAP}",
    ),
    # HLG, which says it all: the metadata without a record adds nothing.
    "dv84.ts": (
        DV84 | HLG,
        {"color_transfer": "arib-std-b67", "side_data_list": [SEI, AMBIENT, *RPU]},
        FirstFrame("arib-std-b67", dovi=True),
        f"HDR, transfer arib-std-b67 (HLG): not supported: {WHY_HLG}; {TONE_MAP}",
    ),
    # Untagged, its base layer would be read as SDR.
    "dv_untagged.ts": (
        DV84 | {"color_range": "pc"},
        {"side_data_list": [SEI, AMBIENT, *RPU]},
        FirstFrame("", dovi=True),
        NO_RECORD,
    ),
    "dv_709.ts": (
        DV84 | BT709,
        {"color_transfer": "bt709", "side_data_list": [SEI, AMBIENT, *RPU]},
        FirstFrame("bt709", dovi=True),
        NO_RECORD,
    ),
    "av1dv709.ivf": (
        AV1 | BT709,
        {"color_transfer": "bt709", "side_data_list": [METADATA]},
        FirstFrame("bt709", dovi=True),
        NO_RECORD,
    ),
    "rpu_alone.ts": (
        DV84 | {"color_range": "pc"},
        {"side_data_list": [SEI, AMBIENT, RPU_DATA]},
        FirstFrame("", dovi=True),
        NO_RECORD,
    ),
}


@pytest.mark.parametrize("name", FIRST_FRAMES)
def test_first_frame(name: str) -> None:
    # A PQ or HLG transfer refused from the first frame as from the stream; Dolby Vision's
    # metadata there, without a record, refused whatever the tags (provisional:
    # source.NO_RECORD).
    stream, frame, read, refusal = FIRST_FRAMES[name]
    probed = parse(Path(name), output(stream, frame))
    assert (probed.dovi, probed.first_frame) == (None, read)
    assert declared_refusal(probed) == refusal


# ffprobe n9.0.2's JSON (probe's commands, the second call's frames and count of packets read
# merged in) for streams probe can't read, cut likewise. UNSIZED: H.264 in MPEG-TS, cut by bytes
# 210 packets before a keyframe, which the 7 s ffprobe analyses of an MPEG-TS start don't reach;
# the second call decodes 40 frames from that keyframe on. UNKNOWN: an FFV1 Matroska file whose
# codec id ffmpeg doesn't know (test_unknown_codec_refused). The others: STREAMS["8.2"] given the
# coded size ffprobe writes with a decoder, or not, as without one.
UNSIZED = {
    "codec_name": "h264",
    "width": 0,
    "height": 0,
    "coded_width": 0,
    "r_frame_rate": "24000/1001",
    "avg_frame_rate": "24000/1001",
}
UNKNOWN = {
    "width": 64,
    "height": 48,
    "sample_aspect_ratio": "1:1",
    "color_range": "tv",
    "field_order": "progressive",
    "r_frame_rate": "25/1",
    "avg_frame_rate": "25/1",
}
DECODED = STREAMS["8.2"] | {"coded_width": 3840}
UNCHECKED = "so what its frames carry, an HDR transfer or Dolby Vision's metadata, can't be checked"
CUT_OR_ENCODE = "cut it at a keyframe, or re-encode it"
OTHER_BUILD = "use a build that reads it, or re-encode it with a tool that does"


@pytest.mark.parametrize(
    ("stream", "frames", "refusal"),
    [
        # The second call reads MOST_PACKETS at most (test_first_frame_behind_an_edit_list).
        (
            DECODED | {"nb_read_packets": "250"},
            0,
            f"no frame decoded from the first 250 packets of its video stream, {UNCHECKED}:"
            f" {CUT_OR_ENCODE}",
        ),
        (
            DECODED | {"nb_read_packets": "37"},
            0,
            f"no frame decoded from all 37 packets of its video stream, {UNCHECKED}:"
            f" {CUT_OR_ENCODE}",
        ),
        (
            DECODED | {"nb_read_packets": "1"},
            0,
            f"no frame decoded from the only packet of its video stream, {UNCHECKED}:"
            f" {CUT_OR_ENCODE}",
        ),
        (DECODED, 0, "no packet in its video stream"),  # none read: no count written
        # No pixel format or size, whatever the second call decoded.
        (
            UNSIZED | {"nb_read_packets": "250"},
            40,
            "no pixel format or size found for its video stream in the start ffprobe analyses, as"
            f" when it starts away from a keyframe: {CUT_OR_ENCODE}",
        ),
        (
            UNKNOWN | {"nb_read_packets": "5"},
            0,
            f"its video codec unknown to this ffmpeg build: {OTHER_BUILD}",
        ),
        (
            STREAMS["8.2"] | {"nb_read_packets": "250"},
            0,
            f"no decoder for its video codec, hevc, in this ffmpeg build: {OTHER_BUILD}",
        ),
    ],
    ids=["250", "37", "1", "0", "unsized", "unknown codec", "no decoder"],
)
def test_unreadable_stream_refused(stream: dict[str, Any], frames: int, refusal: str) -> None:
    probed = json.dumps({"frames": [FRAMES["8.2"]] * frames, "streams": [stream]})
    with pytest.raises(MediaError) as refused:
        parse(Path("cut.ts"), probed)
    assert str(refused.value) == f"cut.ts: {refusal}"


def test_stream_alone() -> None:
    # verify's read (media/probe.py): no frame read, nor needed; a stream ffprobe found no pixel
    # format or size for refused all the same.
    stream = parse(Path("one.mkv"), json.dumps({"streams": [DECODED]}), first_frame=False)
    assert (stream.pix_fmt, stream.first_frame) == ("yuv420p10le", None)
    with pytest.raises(MediaError, match="no pixel format or size found"):
        parse(Path("one.mkv"), json.dumps({"streams": [UNSIZED]}), first_frame=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dv_profile", ...),  # missing
        ("bl_present_flag", ...),
        ("dv_bl_signal_compatibility_id", ...),
        ("dv_bl_signal_compatibility_id", "2"),
        ("dv_bl_signal_compatibility_id", None),
        ("dv_bl_signal_compatibility_id", 2.0),
        ("dv_bl_signal_compatibility_id", 16),  # 4 bits
        ("dv_profile", 128),  # 7 bits
        ("bl_present_flag", True),
        ("el_present_flag", 2),
        ("rpu_present_flag", -1),
    ],
)
def test_unreadable_record_refused(key: str, value: object) -> None:
    # What the base layer is then unknown: refused, with the record as ffprobe gave it.
    record = {name: field for name, field in written(8, 0, 1, 2).items() if name != key}
    if value is not ...:
        record[key] = value
    with pytest.raises(MediaError) as refused:
        parse(Path("dv.mkv"), output(STREAMS["8.2"] | {"side_data_list": [record]}, FRAMES["8.2"]))
    assert str(refused.value) == (
        f"dv.mkv: unreadable Dolby Vision configuration record, its base layer unknown:"
        f" {json.dumps(record)}"
    )


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def record_box(profile: int, compatibility: int, el: int = 0, bl: int = 1) -> bytes:
    """A Dolby Vision configuration record with its RPU, version 1.0 at level 6, as an MP4 dvcC box
    holds it (libavformat/dovi_isom.c:96-110 writes it so): 24 bytes."""
    fields = profile << 9 | 6 << 3 | 1 << 2 | el << 1 | bl
    return struct.pack(">BBHB", 1, 0, fields, compatibility << 4) + bytes(19)


def with_record(path: Path, record: bytes) -> Path:
    """path, an MP4 file ffmpeg wrote, with record added to its video's sample entry as a dvcC
    box, which ffmpeg's mov demuxer reads (libavformat/mov.c:8837 at n9.0.2): its encoders write
    one only for frames carrying Dolby Vision's metadata (libavcodec/libx265.c:848-860). ffmpeg
    writes the moov after the mdat, so no chunk offset moves."""

    def rebuilt(data: bytes) -> bytes:
        out, at = b"", 0
        while at < len(data):
            size, kind = struct.unpack(">I4s", data[at : at + 8])
            body = data[at + 8 : at + size]
            if kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                body = rebuilt(body)
            elif kind == b"stsd":  # its version and flags, its entry count, then the entry
                (end,) = struct.unpack(">I", body[8:12])
                entry = box(body[12:16], body[16 : 8 + end] + box(b"dvcC", record))
                body = body[:8] + entry + body[8 + end :]
            out += box(kind, body)
            at += size
        return out

    data = path.read_bytes()
    assert data.find(b"mdat") < data.find(b"moov")
    path.write_bytes(rebuilt(data))
    return path


SDR_TAGS = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv,"


def dolby_vision_file(path: Path, record: bytes, tags: str = "") -> Path:
    """A tiny FFV1 MP4 file, its frames tagged by setparams's `tags`, with record."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2"),
            *("-vf", f"{tags}format=yuv420p", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    return with_record(path, record)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("record", "read", "refusal"),
    [
        (
            record_box(5, 0),
            DoviRecord(5, True, False, True, 0),
            "profile 5, its base layer Dolby's",
        ),
        (record_box(8, 2), DoviRecord(8, True, False, True, 2), ""),
        # Cut before its id, which ffmpeg then reads as 0, none (libavformat/dovi_isom.c:59-68):
        # Dolby's own only for profiles 5 and 10 (source.DOLBYS_OWN).
        (
            record_box(8, 2)[:4],
            DoviRecord(8, True, False, True, 0),
            "profile 8, its base layer compatible with no other display (id 0): not supported:"
            f" only an SDR base layer, id 2, is read; {TONE_MAP}",
        ),
    ],
    ids=["5", "8.2", "cut short"],
)
def test_record_as_ffprobe_writes_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    record: bytes,
    read: DoviRecord,
    refusal: str,
) -> None:
    # Tagged BT.709 throughout: the record alone decides.
    path = dolby_vision_file(tmp_path / "dv.mp4", record, SDR_TAGS)
    assert probe(path).dovi == read
    caplog.set_level(logging.INFO)
    if refusal:
        with pytest.raises(MediaError) as refused:
            declare(path)
        assert str(refused.value).startswith(f"{path}: Dolby Vision {refusal}")
    else:
        declare(path)
        assert f"{path}: Dolby Vision profile 8, read through its SDR base layer" in caplog.text


# x265 encodes whose container hides the PQ or HLG transfer of their bitstream (FIRST_FRAMES):
# x265's parameters tag the bitstream; ffmpeg's options, or the frames' own tags before the
# encoder, the container. Each with the transfer of its bitstream and the one its stream declares.
BT2020 = "colorprim=bt2020:colormatrix=bt2020nc"
HIDDEN = {
    "hdr10.mkv": (
        "",
        f"{BT2020}:transfer=smpte2084:hdr10=1",
        ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084"],
        "smpte2084",
        "",
    ),
    "pq_vui_709.mkv": (SDR_TAGS, f"{BT2020}:transfer=smpte2084", [], "smpte2084", "bt709"),
    "hlg_vui_709.mkv": (SDR_TAGS, f"{BT2020}:transfer=arib-std-b67", [], "arib-std-b67", "bt709"),
}


def x265_file(path: Path, tags: str, parameters: str, *options: str, size: str = "128x72") -> Path:
    """A short x265 encode in 10 bits, `size` (width x height), its frames tagged by setparams's
    `tags` before the encoder, its bitstream by x265's `parameters`, with ffmpeg's output
    `options`."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi"),
            *("-i", f"testsrc2=s={size}:r=25:d=0.4", "-vf", f"{tags}format=yuv420p10le"),
            *("-c:v", "libx265", "-x265-params", f"{parameters}:log-level=error", *options),
            *("-color_range", "tv", str(path)),
        ],
        check=True,
    )
    return path


@needs_ffmpeg
@needs_x265
@pytest.mark.parametrize("name", HIDDEN)
def test_hdr_on_the_first_frame(tmp_path: Path, name: str) -> None:
    # The stream's tags are the container's, the frames' the bitstream's (media/probe.py,
    # FirstFrame): refused all the same, before the first pass.
    tags, parameters, options, transfer, declared = HIDDEN[name]
    path = x265_file(tmp_path / name, tags, parameters, *options)
    stream = probe(path)
    if name == "hdr10.mkv" and stream.color_transfer == transfer:
        # ffmpeg n9.0.2 writes the matrix alone of the three options to Matroska: FIRST_FRAMES
        # keeps its case.
        pytest.skip("this ffmpeg writes the transfer to Matroska: nothing hidden")
    bitstream = FirstFrame(transfer, False, "bt2020nc", "tv", "left", "bt2020")
    assert (stream.color_transfer, stream.first_frame) == (declared, bitstream)
    with pytest.raises(MediaError) as refused:
        declare(path)
    hdr, why = ("PQ", WHY_PQ) if transfer == "smpte2084" else ("HLG", WHY_HLG)
    pixels = (
        f"HDR, transfer {transfer} ({hdr}) on its first frame, where the stream declares"
        f" {declared or 'none'}: not supported: {why}; {TONE_MAP}"
    )
    if name == "hdr10.mkv":
        # The container declaring the matrix alone over an HDR10 bitstream: the stream's
        # primaries and transfer unspecified with it (libavformat/demux.c:2590-2596), the frame's
        # taken, the PQ refusal alone.
        assert (stream.color_space, stream.color_primaries) == ("bt2020nc", "")
        assert str(refused.value) == f"{path}: {pixels}"
        return
    # A container saying BT.709 throughout contradicts the bitstream's matrix and primaries too;
    # its transfer, which the HDR reason names with the stream's, isn't said twice.
    tags_reason = contradiction(
        "matrix bt709 against bt2020nc on its first frame, primaries bt709 against bt2020",
        "-colorspace:v bt2020nc -color_primaries:v bt2020",
        matrix=True,
    )
    assert str(refused.value) == f"{path}: 2 reasons: (1) {pixels}; (2) {tags_reason}"


# The colour tags (source.COLOUR) as ffprobe n9.0.2 writes them: for each, a stream's value and a
# frame's contradicting it, what the refusal says of them and the option its remux sets; the chroma
# location's never refused nor remuxed (provisional: source.SITING), what its info line says of
# them. Untagged, ffprobe writes "unknown", but "unspecified" for the chroma location
# (fftools/ffprobe.c:1240-1288).
TAGS = {
    "color_space": ("bt470bg", "bt709", "matrix bt470bg against bt709", "-colorspace:v bt709"),
    "color_range": ("pc", "tv", "range pc (full) against tv (limited)", "-color_range:v tv"),
    "chroma_location": (
        "topleft",
        "left",
        "chroma location topleft against left",
        "-chroma_sample_location:v left",
    ),
    "color_primaries": (
        "bt470bg",
        "bt709",
        "primaries bt470bg against bt709",
        "-color_primaries:v bt709",
    ),
    "color_transfer": (
        "smpte170m",
        "bt709",
        "transfer smpte170m against bt709",
        "-color_trc:v bt709",
    ),
}
UNTAGGED = {name: "unspecified" if name == "chroma_location" else "unknown" for name in TAGS}
STREAM = UNTAGGED | {
    "codec_name": "hevc",
    "width": 64,
    "height": 48,
    "pix_fmt": "yuv420p",
    "sample_aspect_ratio": "1:1",
    "field_order": "progressive",
    "r_frame_rate": "25/1",
    "avg_frame_rate": "25/1",
}


def contradiction(said: str, options: str, matrix: bool) -> str:
    """The reason for colour tags contradicted, quoted whole: `said` names them, `options` are its
    remux's, `matrix` whether the matrix is among them, which --input-matrix settles."""
    return (
        f"its colour tags contradicted: {said}: a bad file, not read on a guess: correct its"
        " container's tags with a remux, to its frames' with ffmpeg -i SOURCE -map 0 -map -0:d -c"
        f" copy {options} retagged.mkv or with mkvpropedit on a Matroska file"
        + (", or give the matrix with --input-matrix" if matrix else "")
    )


@pytest.mark.parametrize("name", TAGS)
@pytest.mark.parametrize(
    ("sides", "read"),
    [
        (("", ""), ""),  # untagged: conversion_for's guesses
        (("stream", ""), "stream"),
        (("", "frame"), "frame"),
        (("stream", "stream"), "stream"),
        (("stream", "frame"), None),  # refused, but the chroma location
    ],
    ids=["untagged", "stream", "frame", "equal", "different"],
)
def test_colour_tag_resolved(name: str, sides: tuple[str, str], read: str | None) -> None:
    # A tag declared on one side only is taken; on both sides, differently, the source is a bad
    # file, refused naming both (DESIGN.md, Input); but the chroma location, the stream's then
    # taken, a decoder's default being no declaration (provisional: source.SITING).
    declared, decoded, said, option = TAGS[name]
    values = {"": UNTAGGED[name], "stream": declared, "frame": decoded}
    on_stream, on_frame = (values[side] for side in sides)
    probed = parse(Path("in.mkv"), output(STREAM | {name: on_stream}, UNTAGGED | {name: on_frame}))
    if read is None and name == SITING:
        read = "stream"
    if read is None:
        refusal = contradiction(f"{said} on its first frame", option, name == "color_space")
        assert declared_refusal(probed) == refusal
        return
    assert declared_refusal(probed) == ""
    taken = resolved(probed)
    assert getattr(taken, name) == {"": "", "stream": declared, "frame": decoded}[read]
    # That tag alone resolved: the rest of the stream as declared.
    assert replace(taken, **{name: getattr(probed, name)}) == probed


def test_input_matrix_settles_the_matrix_alone() -> None:
    # Every tag contradicted: one reason naming each, in the order of COLOUR, the first "on its
    # first frame", the remux setting each; but the chroma location, never refused nor remuxed
    # (provisional: source.SITING). --input-matrix settles the matrix, its contradiction then not
    # said, nor its way out; the others are still refused. Settled, the matrix alone is no reason.
    frame = UNTAGGED | {name: values[1] for name, values in TAGS.items()}
    probed = parse(
        Path("in.mkv"), output(STREAM | {name: values[0] for name, values in TAGS.items()}, frame)
    )
    others = (
        "range pc (full) against tv (limited){}, primaries bt470bg against bt709, transfer"
        " smpte170m against bt709"
    )
    remux = "-color_primaries:v bt709 -color_trc:v bt709"
    assert declared_refusal(probed) == contradiction(
        f"matrix bt470bg against bt709 on its first frame, {others.format('')}",
        f"-colorspace:v bt709 -color_range:v tv {remux}",
        matrix=True,
    )
    assert declared_refusal(probed, "bt709") == contradiction(
        others.format(" on its first frame"), f"-color_range:v tv {remux}", matrix=False
    )
    matrix = parse(
        Path("in.mkv"),
        output(STREAM | {"color_space": "bt470bg"}, UNTAGGED | {"color_space": "bt709"}),
    )
    assert declared_refusal(matrix, "smpte170m") == ""
    assert declared_refusal(matrix) == contradiction(
        "matrix bt470bg against bt709 on its first frame", "-colorspace:v bt709", matrix=True
    )


def test_chroma_location_apart_said(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The stream's chroma location and its first frame's, both declared and different, the other
    # tags alike: no refusal, the stream's siting taken, both said in an info line, a decoder
    # giving its codec's default where the bitstream carries none (provisional: source.SITING).
    # With the range contradicted as well, the range alone is refused and remuxed. Through declare,
    # its probe giving the parsed stream.
    sited, default, said, _ = TAGS[SITING]
    stream = STREAM | BT709 | {"color_range": "tv", SITING: sited}
    frame = UNTAGGED | BT709 | {"color_range": "tv", SITING: default}
    probed = parse(Path("in.mkv"), output(stream, frame))
    monkeypatch.setattr(sources, "probe", lambda path: probed)
    caplog.set_level(logging.INFO)
    assert declare(Path("in.mkv")).conversion == Conversion("yuv420p", "709", "limited", sited)
    assert (
        f"in.mkv: {said} on its first frame: the stream's taken, a decoder giving its codec's"
        " default where the bitstream carries none"
    ) in caplog.text
    probed = parse(Path("in.mkv"), output(stream, frame | {"color_range": "pc"}))
    assert declared_refusal(probed) == contradiction(
        "range tv (limited) against pc (full) on its first frame", "-color_range:v pc", matrix=False
    )


def test_conversion_of_the_tags_resolved(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The conversion reads the stream resolved (source.resolved), not as declared: a 720x576
    # stream declaring no matrix, range or siting, its first frame tagged bt709, limited range and
    # sited top left, is read as its frame says, nothing guessed, where the stream alone gives
    # conversion_for's guesses, the matrix smpte170m from that size among them. Through declare,
    # its probe giving the parsed stream.
    frame = UNTAGGED | {"color_space": "bt709", "color_range": "tv", SITING: "topleft"}
    probed = parse(Path("in.mkv"), output(STREAM | {"width": 720, "height": 576}, frame))
    monkeypatch.setattr(sources, "probe", lambda path: probed)
    caplog.set_level(logging.INFO)
    assert declare(Path("in.mkv")).conversion == Conversion("yuv420p", "709", "limited", "topleft")
    assert "guessed" not in caplog.text
    assert conversion_for(probed).guessed == (
        "matrix smpte170m (from 720x576)",
        "limited range",
        "chroma sited left",
    )


def test_contradiction_among_the_other_reasons() -> None:
    # After the pixels' reason, before the geometry's, numbered. The transfer of an HDR first
    # frame, which the HDR reason names with the stream's, isn't said twice; the stream's own HDR
    # transfer names only its own, so a first frame's other one is.
    cropped = {"side_data_list": [{"side_data_type": "Frame Cropping"}]}
    pq = {"color_space": "bt2020nc", "color_primaries": "bt2020", "color_transfer": "smpte2084"}
    probed = parse(
        Path("in.mkv"), output(STREAM | {"field_order": "tt"} | cropped | BT709, UNTAGGED | pq)
    )
    tags = contradiction(
        "matrix bt709 against bt2020nc on its first frame, primaries bt709 against bt2020",
        "-colorspace:v bt2020nc -color_primaries:v bt2020",
        matrix=True,
    )
    assert declared_refusal(probed) == (
        "4 reasons: (1) interlaced (field order tt): not supported; (2) HDR, transfer smpte2084"
        f" (PQ) on its first frame, where the stream declares bt709: not supported: {WHY_PQ};"
        f" {TONE_MAP}; (3) {tags}; (4) cropped by its container: not supported"
    )
    hlg = {"color_transfer": "arib-std-b67"}
    probed = parse(Path("in.mkv"), output(STREAM | pq, UNTAGGED | pq | hlg))
    tags = contradiction(
        "transfer smpte2084 against arib-std-b67 on its first frame",
        "-color_trc:v arib-std-b67",
        matrix=False,
    )
    assert declared_refusal(probed) == (
        f"2 reasons: (1) HDR, transfer smpte2084 (PQ): not supported: {WHY_PQ}; {TONE_MAP};"
        f" (2) {tags}"
    )


def test_rgb_matrix_contradicted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # An RGB source's matrix tag contradicted: the remux is its one way out, its matrix gbr to
    # ffprobe and rgb to -colorspace; --input-matrix is none, RGB having no matrix. Given, it
    # waives the contradiction and ends in conversion_for's refusal of the option, the one
    # refusal, no line saying it settled anything. Through declare, its probe giving the parsed
    # stream.
    rgb = STREAM | {"pix_fmt": "gbrp", "color_space": "bt709"}
    probed = parse(Path("in.mkv"), output(rgb, UNTAGGED | {"color_space": "gbr"}))
    monkeypatch.setattr(sources, "probe", lambda path: probed)
    refusal = contradiction(
        "matrix bt709 against gbr on its first frame", "-colorspace:v rgb", matrix=False
    )
    assert declared_refusal(probed) == refusal
    with pytest.raises(MediaError) as refused:
        declare(Path("in.mkv"))
    assert str(refused.value) == f"in.mkv: {refusal}"
    caplog.set_level(logging.INFO)
    with pytest.raises(MediaError) as refused:
        declare(Path("in.mkv"), "bt709")
    assert str(refused.value) == "in.mkv: --input-matrix bt709: the source is RGB, it has no matrix"
    assert "settled" not in caplog.text


def remuxed(path: Path, out: Path, *options: str) -> Path:
    """path remuxed to out, every stream copied, ffmpeg's output `options` setting the
    container's tags (a stream copy takes them, fftools/ffmpeg_mux_init.c:1015-1024 at n9.0.2)."""
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0", "-c", "copy"),
            *options,
            str(out),
        ],
        check=True,
    )
    return out


# x265's parameters tagging a bitstream BT.709 throughout, in limited range.
BT709_VUI = "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"


def contradicted_file(path: Path, *options: str) -> Path:
    """An x265 encode tagged BT.709 in limited range, container and bitstream, then remuxed to
    path, its container given the tags of ffmpeg's `options`."""
    consistent = x265_file(path.with_name(f"consistent_{path.name}"), SDR_TAGS, BT709_VUI)
    return remuxed(consistent, path, *options)


def retagged(refusal: str, path: Path, directory: Path) -> Path:
    """The remux a refusal gives, run on path in a new directory: the file it writes."""
    found = re.search(r"ffmpeg -i SOURCE .* retagged\.mkv", refusal)
    assert found is not None
    command = [str(path) if word == "SOURCE" else word for word in shlex.split(found[0])]
    directory.mkdir()
    subprocess.run(command, cwd=directory, check=True, capture_output=True)
    return directory / "retagged.mkv"


def codec_types(path: Path) -> list[str]:
    """The types of path's streams, in order: "video", "audio", "data"... (ffprobe's streams, not
    its stream groups, which list a MOV timecode track's video and data streams again)."""
    listed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json"),
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [stream["codec_type"] for stream in json.loads(listed.stdout)["streams"]]


@needs_ffmpeg
@needs_x265
def test_contradicted_tags_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # A Matroska file whose container declares bt470bg and full range over a BT.709 limited-range
    # bitstream, which ffmpeg alone would read as bt470bg full: refused, naming both values of
    # both. --input-matrix settles the matrix, not the range. The remux the refusal gives, run on
    # the file, makes it read as its frames say; and on a MOV file with a timecode track, a data
    # stream, which it leaves out, Matroska having no place for it (media/source.py).
    tags = ("-colorspace:v", "bt470bg", "-color_range:v", "pc")
    path = contradicted_file(tmp_path / "in.mkv", *tags)
    stream = probe(path)
    assert (stream.color_space, stream.color_range) == ("bt470bg", "pc")
    assert stream.first_frame == FirstFrame("bt709", False, "bt709", "tv", "left", "bt709")
    refusal = contradiction(
        "matrix bt470bg against bt709 on its first frame, range pc (full) against tv (limited)",
        "-colorspace:v bt709 -color_range:v tv",
        matrix=True,
    )
    with pytest.raises(MediaError) as refused:
        declare(path)
    assert str(refused.value) == f"{path}: {refusal}"
    with pytest.raises(MediaError) as refused:
        declare(path, "bt709")
    assert str(refused.value) == f"{path}: " + contradiction(
        "range pc (full) against tv (limited) on its first frame", "-color_range:v tv", matrix=False
    )
    bt709 = Conversion("yuv420p10le", "709", "limited", "left")
    assert declare(retagged(refusal, path, tmp_path / "mkv")).conversion == bt709
    bt470bg = ("-colorspace:v", "bt470bg", "-color_primaries:v", "bt470bg")
    tags = (*bt470bg, "-color_trc:v", "smpte170m", "-timecode", "00:00:00:00")
    path = contradicted_file(tmp_path / "timecode.mov", *tags)
    assert codec_types(path) == ["video", "data"]
    refusal = contradiction(
        "matrix bt470bg against bt709 on its first frame, primaries bt470bg against bt709,"
        " transfer smpte170m against bt709",
        "-colorspace:v bt709 -color_primaries:v bt709 -color_trc:v bt709",
        matrix=True,
    )
    with pytest.raises(MediaError) as refused:
        declare(path)
    assert str(refused.value) == f"{path}: {refusal}"
    remuxed_file = retagged(refusal, path, tmp_path / "mov")
    assert codec_types(remuxed_file) == ["video"]
    assert declare(remuxed_file).conversion == bt709
    # The matrix alone contradicted: --input-matrix settles it, and says so.
    path = contradicted_file(tmp_path / "matrix.mkv", "-colorspace:v", "bt470bg")
    with pytest.raises(MediaError, match="its colour tags contradicted: matrix bt470bg against"):
        declare(path)
    caplog.set_level(logging.INFO)
    assert declare(path, "bt709").conversion.matrix == "709"
    said = f"{path}: matrix bt470bg against bt709 on its first frame, settled by --input-matrix"
    assert f"{said}: read as bt709" in caplog.text


@needs_ffmpeg
@needs_x265
def test_one_side_taken(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # One side's tags taken, both ways ffmpeg makes it. The container declaring the matrix alone
    # over a bitstream tagged BT.709 throughout: the stream's primaries and transfer unspecified
    # with it (libavformat/demux.c:2590-2596), the frame's taken. A bitstream without colour
    # description under a container tagged BT.709: HEVC's frames untagged
    # (libavcodec/hevc/hevcdec.c:360-364), the container's taken. Nothing guessed either way.
    caplog.set_level(logging.INFO)
    matrix = x265_file(tmp_path / "matrix.mkv", "", BT709_VUI, "-colorspace", "bt709")
    stream = probe(matrix)
    assert (stream.color_space, stream.color_primaries, stream.color_transfer) == ("bt709", "", "")
    assert stream.first_frame == FirstFrame("bt709", False, "bt709", "tv", "left", "bt709")
    untagged = x265_file(tmp_path / "untagged.mkv", "", "")
    bt709 = ("-colorspace:v", "bt709", "-color_primaries:v", "bt709", "-color_trc:v", "bt709")
    container = remuxed(untagged, tmp_path / "container.mkv", *bt709)
    stream = probe(container)
    assert (stream.color_space, stream.color_primaries, stream.color_transfer) == ("bt709",) * 3
    assert stream.first_frame == FirstFrame("", False, "", "tv", "left", "")
    for path in (matrix, container):
        declared = declare(path)
        tags = (declared.stream.color_primaries, declared.stream.color_transfer)
        assert (declared.stream.color_space, *tags) == ("bt709",) * 3
        assert declared.conversion == Conversion("yuv420p10le", "709", "limited", "left")
    # The conversion reads the matrix taken: a 720x576 bitstream tagged BT.709 throughout, its
    # container declaring the primaries alone, the stream's matrix unspecified with them, is read
    # as its frames' bt709, where the stream alone gave a guess from its size, smpte170m.
    sd = x265_file(tmp_path / "sd.mkv", "", BT709_VUI, size="720x576")
    unknown = ("-colorspace:v", "unknown", "-color_trc:v", "unknown")
    primaries = remuxed(sd, tmp_path / "primaries.mkv", "-color_primaries:v", "bt709", *unknown)
    stream = probe(primaries)
    assert (stream.color_space, stream.color_primaries, stream.color_transfer) == ("", "bt709", "")
    assert stream.first_frame == FirstFrame("bt709", False, "bt709", "tv", "left", "bt709")
    declared = declare(primaries)
    assert declared.conversion == Conversion("yuv420p10le", "709", "limited", "left")
    assert "guessed" not in caplog.text


@needs_ffmpeg
@pytest.mark.parametrize(
    ("encoder", "pix_fmt", "sited", "default"),
    [
        # The decoders' defaults (libavcodec at n9.0.2): MPEG-2's in 4:2:0 (mpeg12dec.c:958).
        pytest.param(
            "mpeg2video", "yuv420p", "topleft", "left", marks=needs_encoders("mpeg2video")
        ),
        # MPEG-4 Part 2's (mpeg4videodec.c:4032).
        pytest.param("mpeg4", "yuv420p", "topleft", "left", marks=needs_encoders("mpeg4")),
        # H.264's from a VUI without chroma_loc_info (h2645_vui.c:99).
        pytest.param(
            "libopenh264", "yuv420p", "topleft", "left", marks=needs_encoders("libopenh264")
        ),
        # MPEG-1's (mpeg12dec.c:948).
        pytest.param("mpeg1video", "yuv420p", "left", "center", marks=needs_encoders("mpeg1video")),
        # MPEG-2's in 4:2:2 (mpeg12dec.c:959-960), the same siting as left without vertical
        # subsampling; its source an x264 4:2:2 encode, sited left by its decoder (h2645_vui.c:99).
        pytest.param(
            "mpeg2video",
            "yuv422p",
            "left",
            "topleft",
            marks=needs_encoders("mpeg2video", "libx264"),
        ),
    ],
    ids=["mpeg2video", "mpeg4", "libopenh264", "mpeg1video", "mpeg2video-422"],
)
def test_chroma_location_of_a_decoder_default(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    encoder: str,
    pix_fmt: str,
    sited: str,
    default: str,
) -> None:
    # Consistent files ffmpeg makes, tagged BT.709 throughout, their bitstream carrying no chroma
    # siting: their Matroska container's is their source's frames' (media/source.py, SITING), their
    # first frame's their decoder's default. Accepted, the container's siting read, both said.
    ffmpeg_command = ("ffmpeg", "-v", "error", "-nostdin")
    testsrc = ("-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=0.2")
    if pix_fmt == "yuv422p":
        x264 = tmp_path / "x264.mkv"
        tagged = f"{SDR_TAGS}format={pix_fmt}"
        subprocess.run(
            [*ffmpeg_command, *testsrc, "-vf", tagged, "-c:v", "libx264", str(x264)], check=True
        )
        assert probe(x264).first_frame == FirstFrame("bt709", False, "bt709", "tv", sited, "bt709")
        source: tuple[str, ...] = ("-i", str(x264))
    else:
        tagged = f"{SDR_TAGS}setparams=chroma_location={sited},format={pix_fmt}"
        source = (*testsrc, "-vf", tagged)
    path = tmp_path / f"{encoder}.mkv"
    subprocess.run([*ffmpeg_command, *source, "-c:v", encoder, str(path)], check=True)
    stream = probe(path)
    assert stream.first_frame is not None
    assert (stream.chroma_location, stream.first_frame.chroma_location) == (sited, default)
    caplog.set_level(logging.INFO)
    assert declare(path).conversion == Conversion(pix_fmt, "709", "limited", sited)
    said = f"{path}: chroma location {sited} against {default} on its first frame: the stream's"
    assert f"{said} taken" in caplog.text


def dropped(path: Path) -> int:
    """The packets of path's video marked to be dropped once decoded (an MP4 edit list's)."""
    listed = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "packet=flags", "-of", "csv=p=0", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum("D" in flags for flags in listed.stdout.split())


@needs_ffmpeg
@needs_mpeg4
def test_first_frame_behind_an_edit_list(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # An MP4 file cut by ffmpeg -ss with -c copy keeps the frames from the keyframe before the cut,
    # dropped once decoded: none from the first packet, so probe decodes on, the edit list ignored,
    # however many frames it hides (media/probe.py). One 300-frame GOP, by ffmpeg's own MPEG-4
    # encoder, cut 10 and 265 frames in: more than the 250 packets probe's second call reads.
    source, near, far = tmp_path / "gop.mp4", tmp_path / "near.mp4", tmp_path / "far.mp4"
    ffmpeg_command = ("ffmpeg", "-v", "error", "-nostdin")
    subprocess.run(
        [
            *(*ffmpeg_command, "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25:d=10.8"),
            *("-c:v", "mpeg4", "-g", "300", str(source)),
        ],
        check=True,
    )
    for cut, at in ((near, "0.4"), (far, "10.6")):
        subprocess.run(
            [*ffmpeg_command, "-ss", at, "-i", str(source), "-c", "copy", str(cut)], check=True
        )
    assert (dropped(near), dropped(far)) == (10, 265)
    caplog.set_level(logging.INFO)
    for cut in (near, far):
        stream = probe(cut)
        # MPEG-4 Part 2's frames, untagged but for their chroma location, which its decoder sets
        # (libavcodec/mpeg4videodec.c:4032 at n9.0.2).
        assert stream.first_frame == FirstFrame("", dovi=False, chroma_location="left")
        retried = f"{cut}: no frame decoded from its first packet: decoding up to 250 for one"
        assert retried in caplog.text
        # The stream as declared, the edit list kept: as read without its frames.
        assert replace(stream, first_frame=None) == probe(cut, first_frame=False)


@needs_ffmpeg
def test_unknown_codec_refused(tmp_path: Path) -> None:
    # ffprobe declares the track's size, but no codec, coded size or pixel format: no decoder
    # (media/probe.py). An FFV1 Matroska file, its codec id changed to one ffmpeg doesn't know.
    path = tmp_path / "unknown.mkv"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi"),
            *("-i", "testsrc2=s=64x48:r=25:d=0.2", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    data = path.read_bytes()
    assert data.count(b"V_FFV1") == 1
    path.write_bytes(data.replace(b"V_FFV1", b"V_FFVX"))  # the same length: no size moves
    with pytest.raises(MediaError) as refused:
        probe(path)
    assert (
        str(refused.value) == f"{path}: its video codec unknown to this ffmpeg build: {OTHER_BUILD}"
    )


@needs_ffmpeg
def test_no_video_stream(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Refused, without a second read: ffprobe has read the whole file looking for a video packet.
    path = tmp_path / "audio.mka"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "sine=d=1"),
            *("-c:a", "flac", str(path)),
        ],
        check=True,
    )
    caplog.set_level(logging.INFO)
    with pytest.raises(MediaError) as refused:
        probe(path)
    assert (str(refused.value), caplog.text) == (f"{path}: no video stream", "")
