"""A source, examined before any GPU work: what it declares, what seedvr2x refuses, how its frames
become RGB, and the first pass (DESIGN.md, Input). A source is one video file, a job's only input.
A job resumed checks its record between what the source declares and its first pass, whose record
it then trusts when nothing changed (DESIGN.md, Pause and resume)."""

import logging
import threading
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from seedvr2x.media.conversion import PIXEL_FORMATS, Conversion, conversion_for
from seedvr2x.media.decode import Decoder
from seedvr2x.media.ffmpeg import MediaError, input_args
from seedvr2x.media.files import sha256
from seedvr2x.media.probe import DoviRecord, VideoStream, probe
from seedvr2x.media.scan import scan, timing_error

logger = logging.getLogger(__name__)

# The first two rows of a display matrix that neither rotates, flips nor scales (a b u, c d v;
# a to d in 16.16 fixed point). The third row, a translation, changes no pixel.
IDENTITY = (65536, 0, 0, 0, 65536, 0)

# HDR's transfers, ffprobe's names (libavutil/pixdesc.c:3317, 3319 at n9.0.2), refused in v1; why,
# and how to get an SDR upscale (DESIGN.md, Not in the first version).
HDR = {"smpte2084": "PQ", "arib-std-b67": "HLG"}
TRAINED = "the model was trained on SDR video"
WHY = {
    "PQ": f"{TRAINED}, what it makes of PQ-coded pixels is unmeasured, and HDR10's metadata"
    " (mastering display, MaxCLL) isn't carried",
    "HLG": f"{TRAINED}, and what it makes of HLG-coded pixels is unmeasured",
}
TONE_MAP = (
    "for an SDR upscale, tone-map the source to SDR first, a grading choice for your own tools,"
    " as a gamut conversion is"
)

# Dolby Vision is read through its base layer, accepted when that layer is SDR (DESIGN.md, Not in
# the first version), which its configuration record says by its signal compatibility id
# (media/probe.py). ffmpeg n9.0.2 gives the records it writes 2 over BT.709 SDR, 1 over BT.2020
# PQ, 4 over BT.2020 HLG, and 0 to profile 5 and to profile 10's like of it, whose base layer is
# Dolby's own IPTPQc2 (libavcodec/dovi_rpuenc.c:97-101, 119-153). 0 stands for none, compatible
# with no other display: Dolby's own said of profiles 5 and 10 alone, since 0 is also what ffmpeg
# reads from a record too short to hold an id (libavformat/dovi_isom.c:59-68), and profile 8's
# base layer is cross-compatible by definition. MediaInfoLib names 6 Blu-ray (153ad67a,
# Source/MediaInfo/File__Analyze_Streams.cpp:775-784), profile 7's: in ffmpeg's FATE sample
# mkv/dovi-p7-hvce.mkv, a profile 7 base layer is tagged PQ, HDR10 as on UHD Blu-ray. Any other
# id is refused, as unknown.
SDR_BASE_LAYER = 2
DOLBYS_OWN = frozenset({5, 10})  # the profiles whose id 0 is Dolby's own IPTPQc2
ONLY_SDR = f"only an SDR base layer, id {SDR_BASE_LAYER}, is read"
BASE_LAYERS = {
    0: ("compatible with no other display (id 0)", ONLY_SDR),
    1: ("HDR10 (PQ)", WHY["PQ"]),
    4: ("HLG", WHY["HLG"]),
    6: ("UHD Blu-ray's HDR10 (PQ)", WHY["PQ"]),
}

# Dolby Vision's metadata on the first frame, with no configuration record in the stream: in an
# MPEG-TS file ffmpeg wrote (its muxer has no Dolby Vision descriptor, libavformat/mpegtsenc.c at
# n9.0.2, which its demuxer reads, mpegts.c:2444), in an MP4 file it wrote without -strict
# unofficial (movenc.c:3000-3008), or in a raw HEVC stream. Provisional (DESIGN.md accepts Dolby
# Vision when its base layer is SDR, and says nothing of a stream without its record): refused,
# since what the base layer is can't be told. The RPU holds no compatibility id, ffmpeg only
# guesses a profile from it (libavcodec/dovi_rpu.c:72-92), and profile 5's base layer, untagged,
# would pass for an untagged SDR one. The ways out, ffmpeg n9.0.2's: the file it came from, its
# record kept by a remux to Matroska (matroskaenc.c:1753-1786), or, for a base layer known to be
# SDR, the metadata dropped by the dovi_rpu bitstream filter (libavcodec/bsf/dovi_rpu.c:91-94,
# 210-213, 268).
NO_RECORD = (
    "Dolby Vision's metadata on its first frame, but no configuration record to say what its base"
    " layer is: not supported: only a base layer known to be SDR is read, never one guessed; read"
    " the file it came from if that has the record (ffmpeg keeps it remuxing to Matroska, to MP4"
    " only with -strict unofficial, never to MPEG-TS), or, if its base layer is known to be SDR,"
    " drop the metadata with ffmpeg's dovi_rpu bitstream filter (-c copy -bsf:v dovi_rpu=strip=1)"
)

# The colour tags, read from the stream and from its first frame, since neither side alone says
# what a file declares (DESIGN.md, Input). ffmpeg gives the stream the container's primaries,
# transfer and matrix, all three, as soon as the container declares one of them, and its range and
# chroma location each when declared, the decoder's otherwise (libavformat/demux.c:2584-2598 at
# n9.0.2). The decoders tag the frames with the bitstream's own: HEVC's always, untagged without a
# colour description, limited range and 4:2:0 chroma sited left without theirs
# (libavcodec/hevc/hevcdec.c:350-373); H.264's range and colour description when its VUI has them,
# else the stream's (h264_slice.c:1116-1126, decode.c:578-587), its chroma location always
# (h264_slice.c:1134); MJPEG's BT.601 and centre siting, JPEG's, whatever the container says
# (mjpegdec.c:144-145). VideoStream's and FirstFrame's fields, each with what a refusal calls it
# and ffmpeg's option setting it in a remux, which a stream copy takes
# (fftools/ffmpeg_mux_init.c:1015-1024).
COLOUR = {
    "color_space": ("matrix", "-colorspace:v"),
    "color_range": ("range", "-color_range:v"),
    "chroma_location": ("chroma location", "-chroma_sample_location:v"),
    "color_primaries": ("primaries", "-color_primaries:v"),
    "color_transfer": ("transfer", "-color_trc:v"),
}
RANGES = {"tv": "tv (limited)", "pc": "pc (full)"}
# ffprobe's names are those ffmpeg's options take, but RGB's: gbr, which -colorspace calls rgb
# (libavutil/pixdesc.c:3327, libavcodec/options_table.h:321).
OPTION_VALUES = {"gbr": "rgb"}

# PROVISIONAL (DESIGN.md, Input, reads the chroma location with the other tags, a contradiction
# refused; design decides): the chroma location is out of the contradiction rule, since a decoder's
# default is no declaration. Many bitstreams carry no siting, and their decoders give their codec's
# default: MPEG-1's centre, MPEG-2's left in 4:2:0 and top left in 4:2:2 and 4:4:4
# (libavcodec/mpeg12dec.c:948, 958-960 at n9.0.2), MPEG-4 Part 2's left (mpeg4videodec.c:4032),
# DV's top left (dvdec.c:253), H.264's and HEVC's left from a VUI without chroma_loc_info
# (h2645_vui.c:99, hevc/hevcdec.c:366-372). Yet ffmpeg's encoders take the siting of the frames
# they are given (fftools/ffmpeg_enc.c:274-275), which its Matroska muxer writes
# (libavformat/matroskaenc.c:1398-1405). So mpeg2video, mpeg4 and libopenh264 encodes of a source
# sited top left, mpeg1video and dvvideo ones of a source sited left, and a 4:2:2 one sited left
# that its decoder calls top left (the same siting, without vertical subsampling) are consistent
# files that the rule would refuse, its remux to the frames' siting mis-siting them. The stream's
# siting is taken, the container's when it declares one, else the decoder's
# (libavformat/demux.c:2597-2598), as when the stream alone was read; the first frame's where the
# stream has none (resolved). Both declared and different are said (declare), never refused nor in
# the remux: COLOUR's option for the chroma location goes unused while it is out.
SITING = "chroma_location"


@dataclass(frozen=True)
class Declared:
    """A video file as it declares itself, accepted: its first video stream, its colour tags
    resolved (resolved), how its frames become RGB, and its sample aspect; its frames not counted
    yet (first_pass)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)


@dataclass(frozen=True)
class FirstPass:
    """What the first pass found in a file: its frames, decoded and counted, at the constant rate
    declared; and its content's SHA-256, when hashed."""

    frames: int
    sha256: str | None = None


@dataclass(frozen=True)
class Source:
    """A video file seedvr2x reads: its first video stream, its colour tags resolved (resolved)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    frames: int  # counted by the first pass
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)
    sha256: str | None = None  # the file's content, hashed for a manifest

    @property
    def display_aspect(self) -> Fraction:
        return self.sample_aspect * self.stream.width / self.stream.height

    def decoder(self) -> Decoder:
        """A decoder of every frame, from the first, checked against the first pass's count."""
        return Decoder(
            input_args(self.path),
            self.conversion,
            self.stream.width,
            self.stream.height,
            self.frames,
        )


def examine(path: Path, matrix: str | None = None, sample_aspect: Fraction | None = None) -> Source:
    """Probe path, refuse what seedvr2x doesn't read (declare), then decode it once to count and
    time its frames (first_pass). `matrix` (ffprobe's name) and `sample_aspect` override the
    source's tags.

    Raises MediaError with the reason for a refusal."""
    declared = declare(path, matrix, sample_aspect)
    return counted(declared, first_pass(declared))


def declare(
    path: Path, matrix: str | None = None, sample_aspect: Fraction | None = None
) -> Declared:
    """Probe path and refuse what seedvr2x doesn't read, by what it declares and its first frame
    carries; its colour tags resolved (resolved). `matrix` (ffprobe's name) and `sample_aspect`
    override the source's tags.

    Raises MediaError with the reason for a refusal."""
    probed = probe(path)
    try:
        refusal = declared_refusal(probed, matrix)
        if refusal:
            raise MediaError(refusal)
        stream = resolved(probed)
        conversion = conversion_for(stream, matrix)
    except MediaError as error:
        raise MediaError(f"{path}: {error}") from None
    waived = contradicted(probed).get("color_space")
    if waived is not None:
        # Not refused (declared_refusal): said, the conversion reading --input-matrix's.
        logger.info(
            "%s: matrix %s against %s on its first frame, settled by --input-matrix: read as %s",
            path,
            *waived,
            matrix,
        )
    sited, decoded = _sides(probed)[SITING]
    if sited and decoded and sited != decoded:
        # Never refused (SITING): said, the stream's taken.
        logger.info(
            "%s: chroma location %s against %s on its first frame: the stream's taken, a decoder"
            " giving its codec's default where the bitstream carries none",
            path,
            sited,
            decoded,
        )
    if stream.dovi is not None:
        # Said, since what Dolby Vision adds to that layer is lost.
        logger.info(
            "%s: Dolby Vision profile %d, read through its SDR base layer, as a player without"
            " Dolby Vision shows it: the output has no Dolby Vision",
            path,
            stream.dovi.dv_profile,
        )
    if sample_aspect is None:
        sample_aspect = stream.sample_aspect
        if sample_aspect is None:
            logger.info("%s: no sample aspect declared, read as square pixels", path)
            sample_aspect = Fraction(1)
    if conversion.guessed:
        # --input-matrix sets the matrix alone: named when the matrix is among the guesses, not
        # for a range or a chroma siting, which no option sets.
        logger.warning(
            "%s: untagged, guessed: %s%s",
            path,
            ", ".join(conversion.guessed),
            " (--input-matrix sets the matrix)" if conversion.matrix_guessed else "",
        )
    return Declared(path, stream, conversion, sample_aspect)


def first_pass(declared: Declared, hashed: bool = False) -> FirstPass:
    """Decode every frame of the file once, to count and time them, refused unless at the constant
    rate declared; and, when `hashed`, hash its content meanwhile, from another thread: the file
    read a second time, at the speed of the disk, while ffmpeg decodes it.

    Raises MediaError with the reason for a refusal."""
    path, stream = declared.path, declared.stream
    hashed_as: list[str | Exception] = []

    def hash_content() -> None:
        try:
            hashed_as.append(sha256(path))
        except Exception as error:  # raised again by the caller, below
            hashed_as.append(error)

    # A daemon: should the scan fail, the hash isn't waited for, and ends with the process.
    reader = threading.Thread(target=hash_content, daemon=True)
    if hashed:
        reader.start()
    scanned = scan(path)
    refusal = timing_error(scanned, stream.frame_rate, stream.avg_frame_rate)
    if refusal:
        raise MediaError(f"{path}: {refusal}")
    if not hashed:
        return FirstPass(scanned.frames)
    reader.join()
    [digest] = hashed_as
    if isinstance(digest, OSError):
        raise MediaError(f"{path}: not readable: {digest}") from digest
    if isinstance(digest, Exception):
        raise digest
    return FirstPass(scanned.frames, digest)


def counted(declared: Declared, found: FirstPass) -> Source:
    """The source declared, its frames counted by its first pass, or by the record of one."""
    stream = declared.stream
    source = Source(
        declared.path,
        stream,
        declared.conversion,
        found.frames,
        declared.sample_aspect,
        found.sha256,
    )
    logger.info(
        "%s: %d frames, %dx%d %s at %s fps, sample aspect %s (display %s), read as %s",
        source.path,
        source.frames,
        stream.width,
        stream.height,
        stream.pix_fmt,
        stream.frame_rate,
        _ratio(source.sample_aspect),
        _ratio(source.display_aspect),
        source.conversion.describe(),
    )
    return source


def declared_refusal(stream: VideoStream, matrix: str | None = None) -> str:
    """Why a stream is refused for what it declares and its first frame carries, every such reason
    in one, numbered when there are several, or "": interlacing; HDR, Dolby Vision unless over an
    SDR base layer; colour tags its first frame contradicts, the matrix's but when `matrix`
    (--input-matrix) settles it, the chroma location's never (SITING); a rotation or flip; a
    declared crop (DESIGN.md, Input, Not in the first version)."""
    # Every declared reason at once, so that dealing with one doesn't end on another of them
    # (conversion_for's refusals, parse's and the first pass's come apart): interlacing first,
    # refused in every version; then what the pixels are, and the tags saying how to read them;
    # then the geometry, which a later version may read.
    reasons: list[str] = []
    # sptenc's rule (ffmpeg/probe.go, IsInterlaced): "unknown" is not interlaced, or every file
    # ffprobe can't tell anything about would be refused.
    if stream.field_order not in ("", "progressive"):
        reasons.append(f"interlaced (field order {stream.field_order}): not supported")
    pixels, both_transfers = _pixels(stream)
    if pixels:
        reasons.append(pixels)
    tags = _contradiction(stream, matrix, both_transfers)
    if tags:
        reasons.append(tags)
    if stream.display_matrix is not None and stream.display_matrix[:6] != IDENTITY:
        reasons.append(
            f"rotated or flipped by its display matrix {stream.display_matrix}: not supported"
        )
    if stream.cropped:
        # ffmpeg applies the crop on decode, while ffprobe declares the uncropped size.
        reasons.append("cropped by its container: not supported")
    if len(reasons) < 2:
        return "".join(reasons)
    # Numbered, since a reason can hold "; " itself (TONE_MAP's).
    listed = "; ".join(f"({number}) {reason}" for number, reason in enumerate(reasons, 1))
    return f"{len(reasons)} reasons: {listed}"


def _pixels(stream: VideoStream) -> tuple[str, bool]:
    """Why a stream's pixels aren't read, or "": HDR, by the stream's transfer or its first
    frame's, or Dolby Vision unless over an SDR base layer, then read as a player without Dolby
    Vision shows it; and whether that reason names both transfers, the stream's and its first
    frame's."""
    # Provisional (DESIGN.md doesn't rank them): one reason, the first of the record's refusal,
    # which names the profile; the stream's transfer; the first frame's, which a container's tags
    # can hide (media/probe.py, FirstFrame); Dolby Vision's metadata without a record, which an
    # HDR transfer refuses already.
    frame = stream.first_frame
    assert frame is not None, "a source's first frame is read (probe)"
    if stream.dovi is not None:
        refusal = _dolby_vision(stream.dovi)
        if refusal:
            return refusal, False
    declared, decoded = stream.color_transfer, frame.color_transfer
    if declared in HDR:
        hdr = HDR[declared]
        return f"HDR, transfer {declared} ({hdr}): not supported: {WHY[hdr]}; {TONE_MAP}", False
    if decoded in HDR:
        hdr = HDR[decoded]
        return (
            f"HDR, transfer {decoded} ({hdr}) on its first frame, where the stream declares"
            f" {declared or 'none'}: not supported: {WHY[hdr]}; {TONE_MAP}"
        ), True
    if stream.dovi is None and frame.dovi:
        return NO_RECORD, False
    return "", False


def contradicted(stream: VideoStream) -> dict[str, tuple[str, str]]:
    """The colour tags (COLOUR) the stream and its first frame both declare, differently, but the
    chroma location (SITING): each field with the stream's value and the frame's, in COLOUR's
    order."""
    return {
        name: (declared, decoded)
        for name, (declared, decoded) in _sides(stream).items()
        if declared and decoded and declared != decoded and name != SITING
    }


def resolved(stream: VideoStream) -> VideoStream:
    """The stream, each of its colour tags (COLOUR) as the source declares it: the stream's, or
    its first frame's where the stream has none, a tag declared on one side only being taken
    (DESIGN.md, Input). Every reader after takes it from there: the conversion, the output's tags,
    the manifest. Both sides declaring a tag differently are refused before (declared_refusal), but
    for the matrix when --input-matrix settles it, which conversion_for reads in place of either,
    and for the chroma location, the stream's then taken (SITING)."""
    return replace(
        stream,
        **{name: declared or decoded for name, (declared, decoded) in _sides(stream).items()},
    )


def _sides(stream: VideoStream) -> dict[str, tuple[str, str]]:
    """Each colour tag (COLOUR), the stream's value and its first frame's, "" when untagged."""
    frame = stream.first_frame
    assert frame is not None, "a source's first frame is read (probe)"
    return {
        "color_space": (stream.color_space, frame.color_space),
        "color_range": (stream.color_range, frame.color_range),
        "chroma_location": (stream.chroma_location, frame.chroma_location),
        "color_primaries": (stream.color_primaries, frame.color_primaries),
        "color_transfer": (stream.color_transfer, frame.color_transfer),
    }


def _contradiction(stream: VideoStream, matrix: str | None, both_transfers: bool) -> str:
    """Why a stream's colour tags are refused, or "": each its first frame contradicts
    (contradicted, the chroma location never), named with both values; but the matrix when
    `matrix` (--input-matrix) settles it, and the transfer when the HDR reason names both already
    (`both_transfers`, _pixels). A bad file, as a contradicted frame rate is (DESIGN.md, Input);
    its ways out a remux, and --input-matrix for a YUV source's matrix."""
    found = contradicted(stream)
    if matrix is not None:
        found.pop("color_space", None)
    if both_transfers:
        found.pop("color_transfer", None)
    if not found:
        return ""
    said = [
        f"{COLOUR[name][0]} {_said(name, declared)} against {_said(name, decoded)}"
        for name, (declared, decoded) in found.items()
    ]
    said[0] += " on its first frame"
    # The way out is the user's (DESIGN.md, Input). Provisional (DESIGN.md names the remux, not
    # its values): the command given sets the container's tags to the frames', since a stream
    # copy leaves the bitstream's as they are; and writes Matroska, which keeps each tag, where MP4
    # keeps none of the primaries, transfer and matrix unless all three are declared
    # (libavformat/movenc.c:2945-2948). It copies every stream but the data streams (-map -0:d),
    # which Matroska refuses (matroskaenc.c:2163-2165): -map 0 alone failed on a MOV file's
    # timecode track, exit 234. A subtitle codec Matroska has no id for, MP4's mov_text, still
    # fails it (matroskaenc.c:2151-2154), left to the user to convert or drop.
    options = " ".join(
        f"{COLOUR[name][1]} {OPTION_VALUES.get(decoded, decoded)}"
        for name, (_, decoded) in found.items()
    )
    # --input-matrix is a way out for YUV alone: RGB has no matrix, and conversion_for refuses the
    # option there. Given for RGB, it still waives the contradiction above, so that this refusal
    # of the option is the one said.
    pixel_format = PIXEL_FORMATS.get(stream.pix_fmt)
    rgb = pixel_format is not None and pixel_format.family == "rgb"
    matrix_way = "color_space" in found and not rgb
    override = ", or give the matrix with --input-matrix" if matrix_way else ""
    return (
        f"its colour tags contradicted: {', '.join(said)}: a bad file, not read on a guess: correct"
        " its container's tags with a remux, to its frames' with ffmpeg -i SOURCE -map 0 -map -0:d"
        f" -c copy {options} retagged.mkv or with mkvpropedit on a Matroska file{override}"
    )


def _said(name: str, value: str) -> str:
    """A colour tag's value as a refusal says it: a range with its meaning."""
    return RANGES[value] if name == "color_range" and value in RANGES else value


def _dolby_vision(record: DoviRecord) -> str:
    """Why a Dolby Vision stream is refused, or "" when its base layer is SDR, then read as a
    player without Dolby Vision shows it (its transfer checked after)."""
    profile = f"Dolby Vision profile {record.dv_profile}"
    if not record.bl_present_flag:
        return (
            f"{profile} with no base layer: not supported: only an SDR base layer is read;"
            f" {TONE_MAP}"
        )
    compatibility = record.dv_bl_signal_compatibility_id
    if compatibility == SDR_BASE_LAYER:
        return ""
    if compatibility == 0 and record.dv_profile in DOLBYS_OWN:
        layer, why = "Dolby's own, viewable only through Dolby's processing", TRAINED
    else:
        layer, why = BASE_LAYERS.get(
            compatibility, (f"of compatibility id {compatibility}, unknown", ONLY_SDR)
        )
    return f"{profile}, its base layer {layer}: not supported: {why}; {TONE_MAP}"


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}:{value.denominator}"
