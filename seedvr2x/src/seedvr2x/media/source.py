"""A source, examined before any GPU work: what it declares, what seedvr2x refuses, how its frames
become RGB, and the first pass (DESIGN.md, Input). A source is one video file, a job's only input.
A job resumed checks its record between what the source declares and its first pass, whose record
it then trusts when nothing changed (DESIGN.md, Pause and resume)."""

import logging
import math
import threading
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from seedvr2x.media import bitstream, grid, rate
from seedvr2x.media.conversion import PIXEL_FORMATS, Conversion, conversion_for
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import sha256
from seedvr2x.media.grid import Filling, Places
from seedvr2x.media.index import NOPTS, FrameIndex
from seedvr2x.media.probe import DoviRecord, VideoStream, probe
from seedvr2x.media.rate import Timeline
from seedvr2x.media.reader import Reader
from seedvr2x.media.scan import TOLERANCE_US, Detects, Idet, Scan, name_error, scan, timing_error

logger = logging.getLogger(__name__)

# The first two rows of a display matrix that neither rotates, flips nor scales (a b u, c d v;
# a to d in 16.16 fixed point). The third row, a translation, changes no pixel.
IDENTITY = (65536, 0, 0, 0, 65536, 0)

# The highest frame rate seedvr2x writes, frames per second: every FFV1 output is Matroska,
# whose timestamps count milliseconds, so that above 1,000 fps two frames can't be told apart
# (DESIGN.md, Input, Not in the first version: "A frame rate above 1,000 fps", decided on
# 2026-10-12, worded on 2026-10-28). Measured then: seedvr2x's own master at 2,000 fps, read back,
# had 15 of its 30 frames a whole frame off; 1,000 fps itself holds.
MOST_FPS = 1000
FINER_THAN_MATROSKA = (
    "its frames written to Matroska, whose timestamps count milliseconds, couldn't be told apart"
)

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

# Two names of one meaning, the stream's and its first frame's, are no contradiction (DESIGN.md,
# Input): the stream's name is kept (resolved). ITU-T H.273 (07/2024) says which code points are
# "functionally the same": the matrices 5 and 6 (Table 4), ffmpeg's bt470bg and smpte170m, whose
# coefficients are equal (libavutil/csp.c:48-49 at n9.0.2; "functionally identical", pixfmt.h:707);
# the transfers 1, 6, 14 and 15 (Table 3), its bt709, smpte170m, bt2020-10 and bt2020-12
# (pixfmt.h:668, 673, 681-682), one function in its own tables (csp.c:445-457); the primaries 6
# and 7 (Table 2), its smpte170m and smpte240m, whose chromaticities and white point are equal
# (csp.c:80-81; "identical", pixfmt.h:645). The primaries bt470bg and smpte170m differ (Table 2,
# csp.c:79-80).
SYNONYMS = {
    "color_space": frozenset({"bt470bg", "smpte170m"}),
    "color_transfer": frozenset({"bt709", "smpte170m", "bt2020-10", "bt2020-12"}),
    "color_primaries": frozenset({"smpte170m", "smpte240m"}),
}

# The chroma location is the exception to the contradiction rule (DESIGN.md, Input): a decoder's
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
# (libavformat/demux.c:2597-2598); the first frame's where the stream has none (resolved). Both
# declared and different are said in an info line (declare), never refused nor fixed: COLOUR's
# option for the chroma location goes unused.
SITING = "chroma_location"

# DESIGN.md, Not in the first version: "finds 10% of them or more combed (its multiple-frame
# detection) in a source not declared interlaced". That 10% is the share of the frames idet's
# multiple-frame detection finds interlaced, top or bottom field first (Idet.combed), from which a
# source is warned of. Measured on 2026-10-09, n9.0.2, on the first 6,000 frames of every real
# source the gate accepts (27, declared progressive or with no field order) and of 9 Blu-ray and
# web films and episodes: at most 2.1% (126 TFF on a live-action film with film grain, the next
# 0.8%), repeated fields at most 17 of 6,001; the DVD control, telecined content (MPEG-2 480i,
# refused for its field order anyway), 88% (5,302 BFF of 6,001), 403 repeated fields; a clip
# hard-telecined here (tests/test_idet.py), 98%, 2 frames in 5 with a repeated field. Single-frame
# detection is noisier (1.9% TFF on that film, many undetermined): not used. Sharp synthetic
# pictures fool it: lavfi's testsrc2 at 320x240 reads 92% combed.
COMBED = Fraction(1, 10)

# The remake's cost, as DESIGN.md gives it (Input): FFV1 took 0.53-0.55 MB a frame on S9, a 1080p
# Blu-ray film (research/docs/seeking.md, mechanism 7), 80 GB against its source's 14.5 GB.
REMAKE_COST = "lossless, so large: 0.53-0.55 MB a frame on a 1080p film; and a full encode"

# The share of a source's places filled from which its warning says what such a file is (_missing;
# DESIGN.md, Input: "A file whose encoder dropped repeated frames on a constant grid is read the
# same way, each drawing held as it was. The share of filled places is said, however large").
# PROVISIONAL (implementation, 2026-10-12, a question for design: DESIGN.md gives no share): a
# tenth, where a decoder's dropped pictures are a few frames of a film (2 of 99,585 on the source
# this was found on, research/docs/seeking.md, mechanism 1).
DROPPED_REPEATS = Fraction(1, 10)


@dataclass(frozen=True)
class Declared:
    """A video file as it declares itself, accepted: its first video stream, its colour tags
    resolved (resolved), how its frames become RGB, and its sample aspect; its frames not counted
    yet (first_pass), nor --frame-rate's checked against them (timing_refusal)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)
    rate_override: Fraction | None = None  # --frame-rate, frames per second


@dataclass(frozen=True)
class FirstPass:
    """What the first pass found in a file: its frames, decoded and counted, at the constant rate
    declared, the places they leave empty on that rate's grid counted with them, each filled
    (in_places); its content's SHA-256, when hashed; its frame index, in places (media/index.py);
    idet's counts (scan.Idet)."""

    frames: int  # the places: the frames decoded and the missing ones filled
    sha256: str | None = None
    index: FrameIndex | None = None
    idet: Idet | None = None


@dataclass(frozen=True)
class Source:
    """A video file seedvr2x reads: its first video stream, its colour tags resolved (resolved)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    # Counted by the first pass, in places: the frames decoded and the missing ones filled, the
    # frames of everything after the pass (in_places); the index's rows.
    frames: int
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)
    sha256: str | None = None  # the file's content, hashed for a manifest
    index: FrameIndex | None = None  # the first pass's frame index, in places, or its record's
    rate_override: Fraction | None = None  # --frame-rate, which its first pass accepted
    idet: Idet | None = None  # idet's counts in its first pass, or its record's

    @property
    def display_aspect(self) -> Fraction:
        return self.sample_aspect * self.stream.width / self.stream.height

    @property
    def frame_rate(self) -> Fraction:
        """The job's frame rate, frames per second: --frame-rate's, else the one the stream
        declares. Every use downstream takes it from here: the writers' rate, the segments' merge
        rule, the manifest's output, split's input copies (DESIGN.md, Input: R replaces the
        declared rate)."""
        return self.stream.frame_rate if self.rate_override is None else self.rate_override

    @property
    def filled(self) -> Filling | None:
        """What was filled of its rate's grid, the places its frames leave empty there
        (grid.filling), for its warning and its record; None for a source without such a place,
        or without an index."""
        found = None if self.index is None else grid.filling(self.index)
        return found if found is not None and found.filled else None

    def decoder(self) -> Reader:
        """A reader of every frame, from the first (reader)."""
        return self.reader(0)

    def reader(self, first: int, md5: bool = False) -> Reader:
        """A reader of the frames from frame `first` on, each checked against the frame index,
        reached through it (media/reader.py); `frames` of them in all, the first pass's count.
        md5 is for the tests (Reader)."""
        if self.index is None:
            raise ValueError(f"{self.path}: no frame index to read its frames by")
        stream = self.stream
        return Reader(
            *(self.path, self.conversion, stream.width, stream.height, self.index),
            first,
            self.frames,
            md5=md5,
        )


def examine(
    path: Path,
    matrix: str | None = None,
    sample_aspect: Fraction | None = None,
    frame_rate: Fraction | None = None,
) -> Source:
    """Probe path, refuse what seedvr2x doesn't read (declare), then decode it once to count and
    time its frames (first_pass). `matrix` (ffprobe's name), `sample_aspect` and `frame_rate`
    (frames per second) override what the source declares.

    Raises MediaError with the reason for a refusal."""
    declared = declare(path, matrix, sample_aspect, frame_rate)
    return counted(declared, first_pass(declared))


def declare(
    path: Path,
    matrix: str | None = None,
    sample_aspect: Fraction | None = None,
    frame_rate: Fraction | None = None,
) -> Declared:
    """Probe path and refuse what seedvr2x doesn't read, by what it declares and its first frame
    carries; its colour tags resolved (resolved). `matrix` (ffprobe's name) and `sample_aspect`
    override the source's tags; `frame_rate` (--frame-rate, frames per second) its rate, checked
    against its frames by the first pass (timing_refusal).

    Raises MediaError with the reason for a refusal."""
    if frame_rate is not None and frame_rate > MOST_FPS:
        # The rate the job would write at is --frame-rate's, whatever the source declares:
        # refused from the command line alone, before the file is probed. PROVISIONAL (impl-v010,
        # 2026-10-28: DESIGN.md says "whatever gives the rate" is refused once, before the first
        # pass, and doesn't say whether --frame-rate's refusal joins the numbered list of a
        # source's reasons): apart from it, as a command-line refusal, the source's own reasons
        # unread until the option is fixed.
        raise MediaError(
            f"--frame-rate {rate.fraction(frame_rate)}: above {MOST_FPS:,} fps: not supported:"
            f" {FINER_THAN_MATROSKA}"
        )
    # A name the first pass refuses (scan.name_error), refused here, first: the command line
    # checks what a source declares before it fetches, checks and hashes the model files, and
    # the pass ran after them, a 17 GB pull and a 16.5 GB hash before a refusal its name alone
    # gives (a review's finding, 2026-10-10).
    named = name_error(path)
    if named:
        raise MediaError(named)
    probed = probe(path)
    try:
        refusal = declared_refusal(probed, matrix, frame_rate)
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
    for name, (declared_as, decoded_as) in _sides(probed).items():
        settled = name == "color_space" and matrix is not None  # read as --input-matrix's
        if _one_meaning(name, declared_as, decoded_as) and not settled:
            # Said, as design accepted it built (2026-10-10), since the name kept shows after: the
            # conversion, the output's tags and the manifest take the stream's name (resolved),
            # not the bitstream's.
            logger.info(
                "%s: %s %s against %s on its first frame: two names of one %s, the stream's kept",
                path,
                COLOUR[name][0],
                declared_as,
                decoded_as,
                COLOUR[name][0],
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
    if stream.ffprobe_rate is not None and rate.matroska_guessed(stream.avg_frame_rate):
        # Said once, as the reading below is (design, 2026-10-12): a Matroska stream under 6
        # fps or above 1,000 is read by its avg_frame_rate (media/probe.py, parse).
        logger.info(
            "%s: declares %s fps, ffprobe's guess from the timestamps of a Matroska stream under"
            " 6 fps or above 1,000, whose frame duration gives %s (its avg_frame_rate): read as %s",
            path,
            rate.fraction(stream.ffprobe_rate),
            rate.fraction(stream.frame_rate),
            rate.fraction(stream.frame_rate),
        )
    elif stream.ffprobe_rate is not None:
        # Said once, here (design's decision of 2026-10-12: "is read as that NTSC rate, said in
        # an info line"): the rate the file says is not the job's (media/probe.py, parse).
        logger.info(
            "%s: declares %s fps, Matroska's reading of %s (its frame duration in whole"
            " nanoseconds): read as %s",
            path,
            rate.fraction(stream.ffprobe_rate),
            rate.fraction(stream.frame_rate),
            rate.fraction(stream.frame_rate),
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
    return Declared(path, stream, conversion, sample_aspect, frame_rate)


def first_pass(
    declared: Declared, hashed: bool = False, detector: Detects | None = None
) -> FirstPass:
    """Decode every frame of the file once, to count, time and index them (media/scan.py), refused
    unless at the constant rate declared, or --frame-rate's (timing_refusal), the places its
    frames leave empty on that rate's grid then filled (in_places): the frames and the index
    returned are in places; given the shot detector, feed it every frame, its probabilities kept
    in the index; and, when `hashed`, hash its content meanwhile, from another thread: the file
    read a second time, at the speed of the disk, while ffmpeg decodes it.

    Raises MediaError with the reason for a refusal."""
    path = declared.path
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
    scanned = scan(path, detector)
    refusal = timing_refusal(declared, scanned)
    if refusal:
        raise MediaError(f"{path}: {refusal}")
    # The grid is built here, after the judge: the scan's index and probabilities stay the
    # decoded frames', what the detector saw and what is compared with measured arrays. The
    # frames counted are the places: those the scan counted, and the places filled.
    index = in_places(declared, scanned)
    frames = scanned.frames
    if index is not None and scanned.index is not None:
        frames += index.frames - scanned.index.frames
    if not hashed:
        return FirstPass(frames, index=index, idet=scanned.idet)
    reader.join()
    [digest] = hashed_as
    if isinstance(digest, OSError):
        raise MediaError(f"{path}: not readable: {digest}") from digest
    if isinstance(digest, Exception):
        raise digest
    return FirstPass(frames, digest, index, scanned.idet)


def counted(declared: Declared, found: FirstPass) -> Source:
    """The source declared, its frames counted and indexed by its first pass, or by the record of
    one; warned of when idet found combed frames there (combing), and when places of its rate's
    grid were left empty and filled (_missing), on every run. Their timing is the first pass's
    to judge (timing_refusal), never judged again here: a resume that takes the pass from its
    record reads on at the rate recorded, its index in places already (DESIGN.md, Pause and
    resume)."""
    stream = declared.stream
    source = Source(
        declared.path,
        stream,
        declared.conversion,
        found.frames,
        declared.sample_aspect,
        found.sha256,
        found.index,
        declared.rate_override,
        found.idet,
    )
    declares = rate.fraction(stream.frame_rate)
    without = "" if stream.ffprobe_rate is None else " without it"
    forced = (
        ""
        if source.rate_override is None
        else f" (--frame-rate; it {_declares(stream)} {declares}{without})"
    )
    filled = source.filled
    logger.info(
        "%s: %d frames%s, %dx%d %s at %s fps%s, sample aspect %s (display %s), read as %s",
        source.path,
        source.frames,
        "" if filled is None else f" ({filled.decoded} decoded, {filled.filled} filled)",
        stream.width,
        stream.height,
        stream.pix_fmt,
        source.frame_rate,
        forced,
        _ratio(source.sample_aspect),
        _ratio(source.display_aspect),
        source.conversion.describe(),
    )
    if filled is not None:
        logger.warning("%s: %s", source.path, _missing(source, filled))
    combing(source)
    return source


def _missing(source: Source, found: Filling) -> str:
    """What a source's warning says of the places its frames leave empty on its rate's grid,
    `found` (grid.filling), each filled by the frame before it, held (DESIGN.md, Input: "Missing
    frames are filled, not refused ... says so (the frame, the hole's length, how many holes)";
    "The share of filled places is said, however large"): the first hole's frame and length, with
    the timestamps on both sides of it, the file's own, in seconds, as a jump's are said
    (scan.timing_error); how many holes and the longest; the places, the filled ones, their share
    and the frames decoded. From DROPPED_REPEATS of the places up, what such a file is. Frames
    are numbered as everything after the first pass numbers them, by place."""
    index, first = source.index, found.first
    assert index is not None and first is not None
    on = f"the rate it {_declares(source.stream)}, {rate.fraction(source.frame_rate)}"
    if source.rate_override is not None:
        on = f"the rate it is taken at, {rate.fraction(source.frame_rate)} (--frame-rate)"
    # The file's own timestamps around the hole: the frame held's and the next frame's, which
    # the index has on those frames' own rows.
    after = first.first + first.length
    stepped = ""
    if after < index.frames and NOPTS not in (int(index.pts[first.held]), int(index.pts[after])):
        before, then = (int(index.pts[row]) * index.time_base for row in (first.held, after))
        stepped = (
            f"its timestamps step from {float(before):.3f} to {float(then):.3f} s there,"
            f" {first.length + 1} frames of {on}"
        )
    if found.holes == 1:
        empty = (
            f"place {first.first}"
            if first.length == 1
            else f"places {first.first} and {first.first + 1}"
            if first.length == 2
            else f"places {first.first} to {after - 1}"
        )
        their = "its" if first.length == 1 else "their"
        what = (
            f"{_frames(first.length)} missing after frame {first.held}: "
            + (f"{stepped}, leaving" if stepped else f"its timestamps leave, at {on},")
            + f" {empty} empty on that rate's grid; read as a player shows it, frame {first.held}"
            f" held in {their} place"
        )
    else:
        what = (
            f"{_frames(found.filled)} missing in {found.holes} holes, the first after frame"
            f" {first.held}, {_frames(first.length)} long"
            + (f" ({stepped})" if stepped else "")
            + f", the longest {_frames(found.longest)}: its timestamps leave those places empty on"
            f" the grid of {on}; read as a player shows it, the frame before each hole held"
            " through it"
        )
    share = float(found.share) * 100
    decimals = max(1, 1 - math.floor(math.log10(share)))
    said = (
        f"{what}: {found.places} frames, {found.filled} of them filled ({share:.{decimals}f}%),"
        f" {found.decoded} decoded"
    )
    if found.share >= DROPPED_REPEATS:
        said += (
            "; so large a share is what an encoder leaves that drops repeated frames, each"
            " picture held as it was until the next"
        )
    return said


def _frames(count: int) -> str:
    """A count of frames, as messages say it: 1 frame, 2 frames."""
    return f"{count} frame{'' if count == 1 else 's'}"


def combing(source: Source) -> None:
    """Warn when idet's multiple-frame detection finds combed frames in the first pass, COMBED of
    them or more: telecined or interlaced content in a source declared progressive, or declaring
    no field order, the interlaced ones being refused (DESIGN.md, Not in the first version). The
    run goes on."""
    idet = source.idet
    if idet is None or not idet.analysed or idet.combed < COMBED * idet.analysed:
        return
    # DESIGN.md, Not in the first version: the warning is for "a source not declared interlaced":
    # one that declares no field order is warned of too, since one that declares nothing could be
    # telecined as well, and sptenc's rule reads it as it reads a progressive one
    # (declared_refusal).
    declared = source.stream.field_order == "progressive"
    logger.warning(
        "%s: ffmpeg's idet finds combed frames in a source %s: %d of %d (%.0f%%) interlaced by its"
        " multiple-frame detection, %d top field first and %d bottom field first, %d with a"
        " repeated field: likely telecined or interlaced content, which seedvr2x reads as"
        " progressive frames, combing and all; for a clean upscale, inverse-telecine or"
        " deinterlace it first with a tool made for it; the run goes on",
        source.path,
        "declared progressive" if declared else "that declares no field order",
        idet.combed,
        idet.analysed,
        100 * idet.combed / idet.analysed,
        idet.multiple[0],
        idet.multiple[1],
        idet.repeated[1] + idet.repeated[2],
    )


def timing_refusal(declared: Declared, scanned: Scan) -> str:
    """Why the first pass's frames are refused for their timing, or "" (DESIGN.md, Input: exact
    rational frame rate).

    A jump in the timestamps is refused first, whatever the rate (scan.Discontinuity). Without
    --frame-rate, sptenc's rule against the rate declared (scan.timing_error). A file whose
    timestamps contradict it is a bad file, refused, never reinterpreted on seedvr2x's own; when
    they are a constant rate's exactly, or follow the declared rate's NTSC twin a frame or more
    away from the declared timeline (_guided), the refusal gives that rate, how far they stray
    from its timeline, and two fixes, a remake at that rate or --frame-rate; any other gets
    sptenc's refusal as it is, no rate named, no --frame-rate offered (design's decision of
    2026-10-12). Then seedvr2x's own test of a file the rule passes: every frame
    strictly within half a frame of its place on the declared rate's timeline, counted from the
    first frame, or refused too (_leaves). With --frame-rate R: every frame within half a frame
    of R's timeline (_off).

    Frames that sit on the rate's grid with places left empty are no refusal (_on_grid; DESIGN.md,
    Input: "Missing frames are filled, not refused"): looked for where the rule above fails, by
    the spread of the frames' durations without --frame-rate, by R's timeline with it, never
    where it passes; read with those places filled (in_places), which the source's warning says
    (counted)."""
    path, stream, index = declared.path, declared.stream, scanned.index
    error = timing_error(
        scanned, stream.frame_rate, stream.avg_frame_rate, read_at=stream.ffprobe_rate is not None
    )
    if scanned.discontinuity is not None:
        return error
    forced = declared.rate_override
    if forced is None:
        if index is None:
            return error
        if not error:
            return _leaves(stream, index)
        if _on_grid(declared, scanned) is not None:
            return ""
        guided = _guided(index, stream.frame_rate)
        if guided is None:
            return error
        found, twinned = guided
        last = f"{_ms(scanned.shortest)} to {_ms(scanned.longest)} ms"
        if found.rate == stream.frame_rate:
            what = (
                f"its timestamps follow {rate.said(found.rate)}, the rate it {_declares(stream)},"
                f" {_within(found)}, but its frames last from {last}, where a constant rate's"
                f" lie within {_ms(TOLERANCE_US)} ms of each other"
            )
        else:
            what = (
                f"its timestamps run at {rate.said(found.rate)}, {_within(found)}, where it"
                f" {_declares(stream)} {rate.fraction(stream.frame_rate)}, its frames lasting"
                f" from {last}"
            )
        fixes = _fixes(found.rate, index, stream.frame_rate, twinned)
        return f"{what}: a bad file, not read on a guess: {fixes}"
    if index is None or scanned.frames == 0:
        return "no frame decoded"
    # R replaces the declared rate, and sptenc's bound of 1 ms on the frames' durations with it:
    # S9's frames step 41 to 63 ms (research/docs/seeking.md, mechanism 7), yet each lies within
    # 10.9 ms of 24000/1001's timeline. Within half a frame, a frame's nearest place on R's
    # timeline is its own, so that, taken at R, none is dropped or doubled: the bound that keeps
    # every frame once.
    on = rate.timeline(index, forced)
    declares = rate.fraction(stream.frame_rate)
    where = (
        f"the rate it {_declares(stream)}"
        if forced == stream.frame_rate
        else f"where it {_declares(stream)} {declares}"
    )
    holed = None
    if not on.holds:
        # Frames on R's grid with places left empty are read, those places filled (_on_grid),
        # the 1 ms bound holding the time a place there as it does without the option: R's
        # waiver of it is for frames that hold R's timeline frame after frame, above.
        holed = _on_grid(declared, scanned)
        if holed is None:
            without = _on_grid(replace(declared, rate_override=None), scanned)
            return _off(on, declared, index, passes=not error, holed=without)
    logger.info(
        "%s: --frame-rate %s: every frame %s, %s: taken at that rate",
        path,
        rate.fraction(forced),
        _within(on, "its")
        if holed is None
        else _within(holed, "its place on that rate's") + ", places left empty between them",
        where,
    )
    if holed is None:
        too = not error and rate.timeline(index, stream.frame_rate).holds
    else:
        # With places left empty: on the declared rate's grid as well, by the same judge.
        too = _on_grid(replace(declared, rate_override=None), scanned) is not None
    if forced != stream.frame_rate and too:
        # Accepted by design, 2026-10-12 (DESIGN.md has R accepted when every frame lies within
        # half a frame of its timeline, and says nothing of a file whose frames lie on the
        # timeline of the rate it declares as well): --frame-rate R on a short file that also
        # lies on its declared timeline is taken at R, as asked, and warned of, since its output
        # then lasts another time than its source, off the source's other streams. Only a short
        # file's frames lie on both: 24 and 24000/1001 part by half a frame within 500 frames. By
        # the declared rate's timeline, never by sptenc's rule alone, which doesn't tell: frames
        # at 24000/1001 last within 1 ms of 24's (_leaves), and a file of them declared 24/1 was
        # told its frames were at 24/1 "too" when given its own rate.
        change = stream.frame_rate / forced - 1
        logger.warning(
            "%s: its frames are at the rate it %s, %s, too: taken at %s as asked, the output"
            " lasting %.3f%% %s than the source, off its other streams",
            path,
            _declares(stream),
            rate.fraction(stream.frame_rate),
            rate.fraction(forced),
            abs(float(change)) * 100,
            "shorter" if change < 0 else "longer",
        )
    return ""


def _on_grid(declared: Declared, scanned: Scan) -> Places | None:
    """Each frame's place on the grid of the source's rate, the declared one or --frame-rate's,
    when its frames sit on that grid with places left empty, which are then filled (in_places);
    None otherwise, for the rule of before the places to judge the file alone (timing_refusal):
    DESIGN.md's "Frames that sit on the declared rate's grid with places left empty ... have one
    reading, the one a player shows: the frame before the hole, held", against "A contradicted
    rate has two readings, which is why it is refused; so is a frame off the grid, a variable
    frame rate" (Input). The judge of the holes, and the one place that says what a hole is.

    Looked for only where the rule of before fails, so that a file read until now is read as it
    was, and a constant cadence stays what it was: never after a jump in the timestamps
    (scan.Discontinuity; a hole past scan.JUMP_FORWARD's 10 s is one); without --frame-rate, when
    sptenc's rule fails by the spread of the frames' durations, shortest and longest more than
    TOLERANCE_US apart (scan.timing_error's first test: frames every 2/R seconds declared R pass
    it, and are refused as the contradicted rate they are, R/2 named); with --frame-rate R, when
    a frame lies half a frame or more from R's timeline counted frame after frame
    (rate.Timeline.holds).

    Each frame's place is then the nearest on the rate's timeline, counted from the first frame
    (grid.places), and the frames sit on the grid when all of:

    - every frame has a timestamp, none lies exactly between two places, and no two share a
      place, the places strictly increasing: each frame then lies strictly within half a frame
      of its own place, the nearest being the one within half a frame, DESIGN.md's bound for a
      frame's place (Input: "when a frame lies half a frame or more from its place");
    - a step spans two places or more: else no place is empty, and the refusal is today's;
    - the steps are the grid's own, a hole's step lasting the frames it skips: every step
      spanning k places, 2 or more, lasts k frames of the rate to within half a frame, strictly;
      and sptenc's rule on each step divided by the places it spans, the time a place: shortest
      and longest within TOLERANCE_US of each other, and the rate's period within them
      (scan.timing_error's two tests, of whole steps there), with --frame-rate R as without,
      against R's period: R waives that bound for frames that hold its timeline counted frame
      after frame (timing_refusal), not once places are skipped. A step of 1.5 to 2.5 frames
      would else be a missing frame: segments joined by ffmpeg's concat demuxer alone, each
      starting 64 ms, 1.6 frames of 25 fps, after the last frame of the one before, were read
      under --frame-rate 25 with a frame doubled at two joins of four and none at the others,
      and 8 real segments of 24 frames at 24000/1001 whose AAC audio lasts 1.012 s, joined so
      (steps of 74 ms, 1.77 frames), with one at 5 joins of 7, where no frame is missing (a
      review's findings, 2026-10-12). Nearest places alone would
      fill a retagged file's drift, a place skipped every 1,000 frames of 24000/1001 material
      taken at 24, by a step of one frame: ffmpeg's constant-rate judgement, and the
      contradicted rate DESIGN.md refuses. S9's long steps, 62-63 ms where 24/1 is declared, are
      1.5 frames, no whole number of them: refused, and guided to 24000/1001, as before
      (research/docs/seeking.md, mechanism 7). The time a place alone would let a long hole take
      the drift's place with it: 70 frames of 24000/1001 missing where the frames come half a
      frame off 24/1's grid span 71 places, 41.1 ms each, within 1 ms of the others;
    - the spans have a step that isn't a multiple of the others' common divisor: frames whose
      every step spans a multiple of g places, g above 1, sit on the grid of R/g, a contradicted
      rate (two readings: R/g with fewer places empty, R with more), refused as before. One
      step that spans no multiple of g shows the grid to be R's, and the file is read, its
      repeats filled.

    PROVISIONAL (implementation, 2026-10-12, questions for design: DESIGN.md has "sit on the
    declared rate's grid", "within half a frame" and "no two frames in one place", not the steps'
    tests, which are the reading above. --frame-rate's waiver of the 1 ms bound is DESIGN.md's,
    for a file that holds R's timeline; that it stops where places are skipped is ours, on
    DESIGN.md's "--frame-rate can't hold there" of a join: frames whose steps wander more than
    1 ms a place with a frame really missing, S9's with a hole, are then refused with the
    option as without, and have the remake alone, which numbers the frames as they come, the
    hole closed. That a hole's step is held to half a frame is ours. And the common divisor is
    ours: by the rules before it alone, frames at 12 fps declared 24/1 with one of theirs
    missing, a step of 4 places among steps of 2, and a 24 fps file given --frame-rate 48, would
    be read with half their places filled or more, every frame doubled, where the same cadence
    without that step is refused as a contradicted rate; it holds no further: with one step of
    3 places among them, the frames are read at 24/1, 7 places of 14 filled, as an encoder
    that drops repeated frames leaves a file, DESIGN.md's "however many places are filled")."""
    index, forced = scanned.index, declared.rate_override
    if index is None or scanned.discontinuity is not None or scanned.frames < 2:
        return None
    if forced is None:
        if not scanned.durations or scanned.longest - scanned.shortest <= TOLERANCE_US:
            return None
    elif rate.timeline(index, forced).holds:
        return None
    on = declared.stream.frame_rate if forced is None else forced
    found = grid.places(index, on)
    if not found.timed.all() or found.tie.any():
        return None
    spans = np.diff(found.place)
    # The common divisor refuses only while every step spans a multiple of it: one step that
    # doesn't shows the grid, and the file is read, its repeats filled (pts 0, 2, 4, 6, 9, 11
    # and 13 in 1/24 s declared 24/1: 14 places, 7 filled, where 0 to 12 by 2 is refused, 12/1
    # named; DESIGN.md, Input: "read the same way, however many places are filled").
    if int(spans.min()) < 1 or int(spans.max()) < 2 or int(np.gcd.reduce(spans)) != 1:
        return None
    steps = np.diff(index.pts)
    # A step of `ticks` ticks of p/q seconds lasts ticks p N / (q D) frames of the rate, N/D:
    # within half a frame of its span, strictly, when |2 ticks p N - 2 span q D| < q D, integers.
    long = spans >= 2
    p, q = index.time_base.numerator, index.time_base.denominator
    exact = steps[long].astype(object) * (2 * p * on.numerator)
    whole = spans[long].astype(object) * (2 * q * on.denominator)
    if any(abs(int(off)) >= q * on.denominator for off in exact - whole):
        return None
    # The time a place, exact: each step, in ticks, over the places it spans; the shortest and
    # the longest among those of each span. With --frame-rate too (PROVISIONAL, the docstring's):
    # returning the places here for R read a join's step of 1.6 frames as a missing frame.
    each: list[Fraction] = []
    for span in np.unique(spans).tolist():
        ticks = steps[spans == span]
        each += [Fraction(int(ticks.min()), span), Fraction(int(ticks.max()), span)]
    microseconds = index.time_base * 1_000_000
    shortest, longest = min(each) * microseconds, max(each) * microseconds
    period = Fraction(1_000_000) / on
    if longest - shortest > TOLERANCE_US:
        return None
    if not shortest - TOLERANCE_US <= period <= longest + TOLERANCE_US:
        return None
    return found


def in_places(declared: Declared, scanned: Scan) -> FrameIndex | None:
    """The first pass's frame index in places (media/grid.py), for a source timing_refusal
    accepts: the scan's own index, of the decoded frames, when they leave no place empty on the
    rate's grid, the same object, its file the bytes of before the places; else each frame at
    the place the judge gives it (_on_grid), every place left empty a held row, which shows the
    frame before it again. Everything after the first pass counts in its rows: the shots, the
    detector's probabilities, the cuts, the segments, the writers, a resume's reads (DESIGN.md,
    Input). None when the scan has no index."""
    index = scanned.index
    found = _on_grid(declared, scanned)
    if index is None or found is None:
        return index
    return grid.filled(index, found.place, found.rate)


def _guided(index: FrameIndex, declares: Fraction) -> tuple[Timeline, bool] | None:
    """The rate a refusal names for a file sptenc's rule refuses at the rate it declares,
    `declares`, None without one, with whether the twin rule named it (rate.twin, not rate.exact):
    the rate its timestamps are exactly (rate.exact), whatever it
    is, the declared one included, in a time base too coarse for sptenc's 1 ms (24 fps in 10 ms
    ticks steps 40 and 50 ms); else the declared rate's NTSC twin, once the frames have followed
    it a frame or more away from the declared timeline (rate.twin), S9's case, whose timestamps
    are no rate's exactly, their long steps catching up, and lie within 10.9 ms of 24000/1001's
    timeline where it declares 24/1 (research/docs/seeking.md, mechanism 7). The twins are the
    declared rate's under --frame-rate too (_off): the guidance says what the file is. A twin
    is N x 1000/1001 for a whole number N and N for N x 1000/1001, no other (rate.twins;
    design, 2026-10-12, after the decision below), and the refusal says what else reads so
    where the twin rule names one (TWIN_JOIN).

    Design's decision of 2026-10-12: "a refusal names another rate only (a) when the timestamps
    are that rate's EXACTLY, to the time base's rounding, or (b) where sptenc's rule fails, when
    it is the DECLARED RATE'S NTSC TWIN (R x 1000/1001 or R x 1001/1000) holding every frame
    strictly within half a frame AND the two timelines have parted by a frame or more over the
    file, so that one late frame of a short file is never read as a rate. Everything else that
    sptenc's rule refuses and that isn't frames missing on the grid gets the plain
    variable-frame-rate refusal, no rate named, no --frame-rate offered." Before it, any of
    ffmpeg's standard rates that held the frames within half a frame was named (rate.follows,
    removed), by the file's length: 33 ms a frame declared 25/1 ran "at 91/3 fps (30.333),
    within 3.9 ms" over 120 frames, --frame-rate 91/3 then accepted, and at no rate over 3,000;
    a frame every 11 s, 6 frames, "at 1/12 fps (0.083), within 5000 ms" (a review's findings,
    2026-10-10: exactly, they are 1000/33 fps at both lengths, and 1/11); 8 segments of 24
    frames at 24000/1001 joined by ffmpeg's concat demuxer, their audio outlasting them, the
    video stepping 63 ms at each join, "at 47/2 fps (23.5), within 19.7 ms", and read with
    --frame-rate 47/2, 2% off their audio (a review's finding, 2026-10-12)."""
    # (a) before (b), the decision's order: the two name different rates of one file where a
    # rate the timestamps are exactly lies near enough the twin for the twin's timeline to hold
    # them too, in a time base too coarse for sptenc's 1 ms: 1,001 frames 41.7 ms apart in
    # ticks of 10 ms, declared 24/1, are 10000/417 fps exactly, within 5 ms, and lie within
    # 13.2 ms of 24000/1001's timeline, a frame from 24/1's at their end (tests/test_rate.py).
    exact = rate.exact(index)
    if exact is not None:
        return exact, False
    twin = rate.twin(index, declares)
    return None if twin is None else (twin, True)


def _leaves(stream: VideoStream, index: FrameIndex) -> str:
    """Why a source that sptenc's rule passes is refused all the same, or "": seedvr2x's own test
    (DESIGN.md, Input: "or when a frame lies half a frame or more from its place on the declared
    rate's timeline, counted from the first frame"; "The second test is seedvr2x's own (decided
    2026-10-10)"). Every frame must lie strictly within half a frame of its place on the timeline
    of the rate it declares, `stream`'s, as --frame-rate's frames on R's (rate.Timeline.holds);
    else how many don't, from which frame, how far (_strayed), and what the file is, with its
    ways out (_way_out).

    sptenc's rule bounds each frame's duration, not where the frames end up: frames at 24000/1001
    declared 24/1 last 41.7 ms, within 1 ms of 24's 41.667, yet lie half a frame off 24's timeline
    from frame 500 on, 0.1% further each frame, and frames at 48 fps declared 50/1 last 20.8 ms
    for 20, 4%: both were read, their output written at the rate declared, 0.1% and 4% shorter
    than their frames' own time, off the source's other streams (a review's finding, 2026-10-09).

    By the declared rate's timeline alone, whatever other rate may hold the frames. The refusal
    tried first, of the files whose frames followed another rate within half a frame (one of
    ffmpeg's standard rates, a guess since removed: rate.TWELFTHS), refused files by their
    length and named joins a rate: 2,400 frames at 24000/1001 joined by ffmpeg's concat demuxer
    every 20 or 24 frames, as running at 24/1; 33 ms a frame declared 30/1 over 120 and 300
    frames, as running at 91/3 fps, and read over 3,000 (a review's findings, 2026-10-10). Such
    files were then read and warned of, until design made the rule strict: real sources hold
    their declared timeline, none of 34 Blu-ray, DVD and web files measured leaving it
    (DESIGN.md), and a join by ffmpeg's concat demuxer, read until then however far it drifted,
    is refused once a frame is half a frame off. Measured at 24000/1001 in Matroska: segments
    split by stream copy join 0.77 ms early each, half a frame after 24 joins; segments encoded
    on their own 0.31 ms late, after 68; segments of exactly 24 or 48 frames 1 ms early, after
    21. At 30000/1001, 0.35 to 0.37 ms early, 41 to 45 joins. None at 25 fps, nor from MP4
    segments."""
    # On the rate the stream is read at, which for a Matroska file is no longer always the one
    # ffprobe gives: Matroska holds a frame's duration in whole nanoseconds, read back as the
    # nearest fraction of terms up to 30000, 60000/1001 as 19001/317, 48000/1001 as 7001/146 and
    # 120000/1001 as 29011/242, 0.9, 3.0 and 3.2 ns a frame off (ffmpeg n9.0.2), and timestamps
    # in milliseconds at those rates left the timeline of the rate so declared at frames
    # 8,939,970, 3,335,976 and 1,161,780, 41 h, 19 h and 2 h 41 min in, where sptenc's files and
    # seedvr2x's own were refused, against DESIGN.md's "a file either of them builds is never
    # refused". Settled by design (2026-10-12): such a rate is read as the NTSC rate it is the
    # reading of, where the stream is probed (media/probe.py, parse; rate.matroska_reading), and
    # this test runs on that rate's timeline. A rate Matroska reads worse that is no NTSC
    # rate's reading, 1000000/33333 as 30/1, 1e-5 off, still leaves at frame 48,503, 27 min in,
    # as no rate's exactly.
    on = rate.timeline(index, stream.frame_rate)
    if on.holds:
        return ""
    return f"{_strayed(on, index.frames, stream)}, and {_way_out(on, index, stream)}"


def _strayed(on: Timeline, frames: int, stream: VideoStream) -> str:
    """How a source's `frames` frames leave the timeline of the rate it declares, `on`, `stream`'s:
    how many lie half a frame or more from their place there, from which frame, and how far the
    furthest wanders, in ms and of a frame."""
    lie, their = ("lies", "its") if on.off == 1 else ("lie", "their")
    return (
        f"{on.off} of its {frames} frames {lie} half a frame or more from {their} place on the"
        f" timeline of the rate it {_declares(stream)}, {rate.fraction(on.rate)}, from frame"
        f" {on.first_off}, up to {_ms(on.largest * 1_000_000)} ms"
        f" ({float(on.largest * on.rate):.2f} of a frame)"
    )


def _way_out(on: Timeline, index: FrameIndex, stream: VideoStream) -> str:
    """What a source whose frames leave the timeline of the rate it declares, `on`, `stream`'s,
    is, and its ways out, as its refusal ends (_leaves, _off).

    Its timestamps another constant rate's exactly (rate.exact): a file declaring a rate that
    isn't its frames', named with what the declared one would have cost, the output that much
    shorter or longer than its timestamps' own time, and the two fixes at the rate named
    (_fixes; DESIGN.md, Input: "When the timestamps follow another rate").

    No one rate's exactly: what a plain join by ffmpeg's concat demuxer gives, each segment
    starting where the one before ends by its declared duration, 0.77 to 1 ms early a join at
    24000/1001 for segments split by stream copy, 0.31 ms late for segments encoded on their own
    (cli._directory_refused: 155 ms over 155 joins; 24.5 ms over 29 joins of 20 frames; 39 ms
    over 127 joins of encoded segments). Timestamps alone can't tell such a join from a retagged
    file (DESIGN.md,
    Input: "When the timestamps follow no rate exactly"): the same join every 20 or 24 frames
    lies within 1.7 and 1.3 ms of 24/1's timeline, and 24.03 fps declared 24/1, rounded to
    milliseconds, is no rate's exactly either (rate.exact), nor is a file of a few frames that
    several rates hold so. Its ways out are a join that keeps the frame grid (rate.grid_join,
    its commands at the rate declared, the segments' own) and the remake at the rate declared,
    which numbers the frames as they come, the place each had before the join: the remake of
    such a join gave every frame back in order, on the grid, for each kind of segment whose
    plain join was whole (cli._directory_refused's materials, 29 of 31: a program stream's
    segments and a transport stream's of one frame joined into no whole file). --frame-rate is
    no way out and isn't named: no rate is the frames' own
    there, and one that holds them within half a frame, as 24/1 holds those joins, would take
    frames meant for 24000/1001 at 24."""
    declares = on.rate
    found = rate.exact(index)
    if found is not None:
        return (
            f"its output, written at that rate, would be {_cost(found.rate / declares - 1)} than"
            " its timestamps' own time, off the source's other streams: its timestamps follow"
            f" {rate.said(found.rate)} exactly, to the rounding of its time base,"
            f" {index.time_base} s: a bad file, not read on a guess:"
            f" {_fixes(found.rate, index, declares)}"
        )
    command = rate.remake(declares, rate.first_time(index))
    return (
        "its timestamps follow no one rate exactly: a file joined by ffmpeg's concat demuxer"
        " alone drifts so, up to 1 ms a join, and timestamps alone can't tell a drifting join"
        " from a retagged file: a bad file, not read on a guess: join its segments again so that"
        f" every frame keeps its place on the frame grid, {rate.grid_join(declares=declares)};"
        f" or remake it at the rate it {_declares(stream)}, an FFV1 master numbering its frames at"
        f" that rate as they come, every one kept, its other streams copied ({REMAKE_COST}), with"
        f" {command}"
    )


def _cost(change: Fraction) -> str:
    """What a rate does to an output's duration, `change` its relative change, not 0, as a
    refusal gives it: in percent to 3 decimals, "0.100% shorter", "4.000% shorter", or to as
    many as show its first two digits: "0.000038% shorter" for 120000/1001 declared 29011/242,
    Matroska's reading of it, by a file in another container (_leaves), where 3 decimals said
    0.000%."""
    percent = abs(float(change)) * 100
    decimals = max(3, 1 - math.floor(math.log10(percent)))
    return f"{percent:.{decimals}f}% {'shorter' if change < 0 else 'longer'}"


def _exactly(found: Timeline, index: FrameIndex) -> str:
    """The rate a source's timestamps are exactly (rate.exact), as messages give it, with the
    option that takes its frames at it. Named by the exact rule, never the twin rule, it says
    nothing of a join (_twin_join)."""
    return (
        f"its timestamps follow {rate.said(found.rate)} exactly, to the rounding of its time"
        f" base, {index.time_base} s: --frame-rate"
        f" {rate.fraction(found.rate)} takes its frames at that rate"
    )


def _off(
    on: Timeline, declared: Declared, index: FrameIndex, passes: bool, holed: Places | None = None
) -> str:
    """Why --frame-rate R is refused, its frames not all within half a frame of R's timeline: the
    first one off, how far, how many. Then what the file is without the option. One sptenc's rule
    passes at the rate it declares (`passes`) and that holds that rate's timeline: read without
    the option, at that rate; and the rate its timestamps are exactly (rate.exact), when they
    are another's. One the rule passes that leaves that timeline: refused without the option
    too, with the fix that applies (_leaves, _way_out), the exact rate's when there is one. One
    the rule refuses: read without the option all the same when its frames sit on the declared
    rate's grid with places left empty (`holed`, their places there, _on_grid's), which is said,
    no rate named; else the rate its refusal names without the option (_guided: its timestamps'
    exactly, or the NTSC twin of the rate it declares, never of R), and the two fixes at it; and
    nothing more when that refusal names none, a variable frame rate's (design's decision of
    2026-10-12, _guided's).

    A file the rule passes is never given a rate that holds its frames within half a frame
    alone, its declared rate's twin no more than another (DESIGN.md, Input: "When the timestamps
    follow no rate exactly ... `--frame-rate` can't hold there"; design's decision: the twin is
    named "where sptenc's rule fails"). 2,400 frames at 24000/1001 joined by ffmpeg's concat
    demuxer every 20, given the rate they declare, were told "its timestamps run at 24/1 fps
    (24), within 1.7 ms of that rate's timeline", and --frame-rate 24/1 then read them at 24,
    0.1% off, without a warning (a review's finding, 2026-10-10)."""
    half = 1000 / (2 * float(on.rate))
    by = float(on.first_off_by) * 1000
    more = on.off - 1
    reason = (
        f"--frame-rate {rate.fraction(on.rate)}: frame {on.first_off} lies {abs(by):.1f} ms"
        f" {'after' if by > 0 else 'before'} its place on that rate's timeline"
        f" ({abs(by) / half / 2:.3f} of a frame, half a frame being {half:.1f} ms)"
        + (
            f", and {more} more frame{'s' if more > 1 else ''}, up to"
            f" {_ms(on.largest * 1_000_000)} ms"
            if more
            else ""
        )
        + ": taken at that rate, frames would be dropped or doubled"
    )
    stream = declared.stream
    declares = stream.frame_rate
    if passes:
        if on.rate == declares:
            # R is the rate declared: its frames are said already, the first one off and how many.
            return (
                f"{reason}; without --frame-rate, the file is refused too, that rate being the"
                f" one it {_declares(stream)}, and {_way_out(on, index, stream)}"
            )
        left = _leaves(stream, index)
        if left:
            return f"{reason}; without --frame-rate, the file is refused too: {left}"
        reason += (
            f"; without --frame-rate, the file is read at {_read_at(stream)}, whose timeline"
            " holds every frame within half a frame"
        )
        exact = rate.exact(index)
        if exact is None or exact.rate == declares:
            return reason
        return f"{reason}; {_exactly(exact, index)}"
    if holed is not None:
        missing = int(holed.place[-1]) + 1 - len(holed.place)
        return (
            f"{reason}; without --frame-rate, the file is read at {_read_at(stream)}, on whose"
            f" grid its frames sit, the {_frames(missing)} missing there filled"
        )
    guided = _guided(index, declares)
    if guided is None or guided[0].rate == on.rate:
        return reason
    found, twinned = guided
    return (
        f"{reason}; its timestamps run at {rate.said(found.rate)}, {_within(found)}:"
        f" {_fixes(found.rate, index, declares, twinned)}"
    )


# What a refusal adds where the twin rule names a whole rate N for a file declared N x 1000/1001
# (rate.twin), before the fixes at N: DESIGN.md, Input, "The twin and a join" (decided on
# 2026-10-12, worded on 2026-10-28), "only where the twin rule names the whole rate N for a file
# declared N x 1000/1001, never the exact rule, the one way a join was seen to read". N frames at
# N x 1000/1001 fps last 1.001 s, and ffmpeg's concat demuxer starts each such segment 1.000 s
# after the one before, a millisecond early (cli._directory_refused): N frames a second, which
# N/1's timeline holds, 24/1's fifty segments within 1.3 ms. Segments at a whole rate last whole
# milliseconds and join whole: a file declared 24/1 that runs at 24000/1001, a Blu-ray remux, is
# no join. "otherwise" is ours, leading to the fixes of a file that is no such join.
TWIN_JOIN = (
    "a join of one-second segments by ffmpeg's concat demuxer reads so too, and such a file is"
    " joined again on the frame grid (a directory given to seedvr2x says how); otherwise, "
)


def _twin_join(found: Fraction, declares: Fraction, twinned: bool) -> str:
    """TWIN_JOIN, what a refusal says before the fixes at the rate it names, `found`, when the
    twin rule named it (`twinned`), it is a whole number and the file declares it x 1000/1001
    (`declares`); else ""."""
    return TWIN_JOIN if twinned and found.denominator == 1 and declares == found * rate.NTSC else ""


def _declares(stream: VideoStream) -> str:
    """What messages say a file does with its stream's rate, `stream`.frame_rate: "declares"
    it, "the rate it declares, 24/1", "where it declares 24/1"; and "is read at" it for a
    stream read at another rate than ffprobe gives for it (VideoStream.ffprobe_rate: a Matroska
    stream's, media/probe.py, parse), which the file doesn't declare to ffprobe: "the rate it
    is read at, 60000/1001", of a file that declares 19001/317, said once with both rates
    (declare). Design, 2026-10-12: "the rate it declares" where it is the rate it is read at.
    A stream read at the rate ffprobe gives keeps every message to the byte."""
    return "declares" if stream.ffprobe_rate is None else "is read at"


def _read_at(stream: VideoStream) -> str:
    """The rate a file is read at without --frame-rate, after "the file is read at" (_off): "the
    rate it declares, 24/1", and the rate alone, "60000/1001", for a stream read at another
    rate than ffprobe gives for it (_declares)."""
    said = rate.fraction(stream.frame_rate)
    return said if stream.ffprobe_rate is not None else f"the rate it declares, {said}"


def _within(found: Timeline | Places, whose: str = "that rate's") -> str:
    """How far the frames stray from a rate's timeline, `whose`: within 10.8 ms of that rate's
    timeline (0.26 of a frame)."""
    return (
        f"within {_ms(found.largest * 1_000_000)} ms of {whose} timeline"
        f" ({float(found.largest * found.rate):.2f} of a frame)"
    )


def _fixes(found: Fraction, index: FrameIndex, declares: Fraction, twinned: bool = False) -> str:
    """The two fixes of a file a refusal names the rate `found` for, its timestamps' exactly or
    the NTSC twin of the one it declares, `declares` (_guided, _way_out; DESIGN.md, Input): a
    remake at that rate (rate.remake), or --frame-rate; before them, where the twin rule named
    `found` (`twinned`), what else reads so (_twin_join)."""
    command = rate.remake(found, rate.first_time(index))
    return (
        f"{_twin_join(found, declares, twinned)}"
        "remake it at that rate, an FFV1 master with its other streams copied, a clean file for"
        f" every tool ({REMAKE_COST}), with {command}; or give --frame-rate {rate.fraction(found)},"
        " which takes its frames at that rate, each checked to lie within half a frame of its"
        " timeline: the same frames on the same timeline, without the intermediate file"
    )


def _ms(microseconds: Fraction | int) -> str:
    """A time in microseconds, in milliseconds to a tenth, a whole one without its .0: 10.8, 41,
    62.5."""
    return f"{float(microseconds) / 1000:.1f}".removesuffix(".0")


def declared_refusal(
    stream: VideoStream, matrix: str | None = None, frame_rate: Fraction | None = None
) -> str:
    """Why a stream is refused for what it declares and its first frame carries, every such reason
    in one, numbered when there are several, or "": interlacing; HDR, Dolby Vision unless over an
    SDR base layer; colour tags its first frame contradicts in meaning (SYNONYMS), the matrix's but
    when `matrix` (--input-matrix) settles it, the chroma location's never (SITING); a rotation or
    flip; a declared crop; a frame rate above MOST_FPS, the stream's own (declared, or Matroska's
    reading of it: VideoStream.frame_rate), unless `frame_rate` (--frame-rate) replaces it, whose
    refusal is the command line's (declare) (DESIGN.md, Input, Not in the first version)."""
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
    if frame_rate is None and stream.frame_rate > MOST_FPS:
        reasons.append(
            f"frame rate above {MOST_FPS:,} fps, the rate it {_declares(stream)},"
            f" {rate.fraction(stream.frame_rate)}: not supported: {FINER_THAN_MATROSKA}"
        )
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
    """The colour tags (COLOUR) the stream and its first frame both declare, differently in
    meaning, not two names of one (SYNONYMS), but the chroma location (SITING): each field with the
    stream's value and the frame's, in COLOUR's order."""
    return {
        name: (declared, decoded)
        for name, (declared, decoded) in _sides(stream).items()
        if declared
        and decoded
        and declared != decoded
        and not _one_meaning(name, declared, decoded)
        and name != SITING
    }


def _one_meaning(name: str, declared: str, decoded: str) -> bool:
    """Whether a colour tag's two values, the stream's and its first frame's, are two names of one
    meaning (SYNONYMS)."""
    return declared != decoded and {declared, decoded} <= SYNONYMS.get(name, frozenset())


def resolved(stream: VideoStream) -> VideoStream:
    """The stream, each of its colour tags (COLOUR) as the source declares it: the stream's, or
    its first frame's where the stream has none, a tag declared on one side only being taken
    (DESIGN.md, Input), and the stream's name of two of one meaning (SYNONYMS). Every reader after
    takes it from there: the conversion, the output's tags, the manifest. Both sides declaring a
    tag differently in meaning are refused before (declared_refusal), but for the matrix when
    --input-matrix settles it, which conversion_for reads in place of either, and for the chroma
    location, the stream's then taken (SITING)."""
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
    its ways out a remux when its frames are right, the bitstream's own fix when its container is,
    for a codec that has one, and --input-matrix for a YUV source's matrix."""
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
    # The way out is the user's (DESIGN.md, Input): when the frames are right, a remux setting the
    # container's tags to theirs, a stream copy leaving the bitstream's as they are; when the
    # container is, the bitstream's own fix, its codec's filter setting the bitstream's tags to the
    # container's (media/bitstream.py), or to another name of the container's meaning where the
    # filter doesn't take its own (SYNONYMS: the file is then read, the stream's name kept), named
    # only for a filter setting each tag contradicted, the file otherwise still refused ("whose
    # command the refusal gives when the codec's bitstream filter can set every contradicted
    # tag"). The two commands are the code's, accepted by design as built (2026-10-10): both write
    # Matroska, which keeps each tag, where MP4 keeps none of the primaries, transfer and matrix
    # unless all three are declared (libavformat/movenc.c:2945-2948), and copy every stream but the
    # data streams (-map -0:d), which Matroska refuses (matroskaenc.c:2163-2165): -map 0 alone
    # failed on a MOV file's timecode track, exit 234. A subtitle codec Matroska has no id for,
    # MP4's mov_text, still fails them (matroskaenc.c:2151-2154), left to the user to convert or
    # drop. The filter is given to the first video stream alone (-bsf:v:0), the one seedvr2x reads
    # (media/probe.py): given to every video stream (-bsf:v), it failed on an MP4 file with a
    # cover, a PNG stream, exit 234.
    options = " ".join(
        f"{COLOUR[name][1]} {OPTION_VALUES.get(decoded, decoded)}"
        for name, (_, decoded) in found.items()
    )
    remux = (
        "correct its container's tags with a remux, to its frames' with ffmpeg -i SOURCE -map 0"
        f" -map -0:d -c copy {options} retagged.mkv or with mkvpropedit on a Matroska file"
    )
    fix = bitstream.setting(
        stream.codec_name, {name: declared for name, (declared, _) in found.items()}, SYNONYMS
    )
    ways = (
        f"if its frames are right, {remux}; if its container is right, correct its bitstream's to"
        f" its container's with ffmpeg -i SOURCE -map 0 -map -0:d -c copy -bsf:v:0 {fix} fixed.mkv"
        if fix
        else remux
    )
    # --input-matrix is a way out for YUV alone: RGB has no matrix, and conversion_for refuses the
    # option there. Given for RGB, it still waives the contradiction above, so that this refusal
    # of the option is the one said.
    pixel_format = PIXEL_FORMATS.get(stream.pix_fmt)
    rgb = pixel_format is not None and pixel_format.family == "rgb"
    matrix_way = "color_space" in found and not rgb
    separator = "; " if fix else ", "  # a third clause after the two ways', as they are parted
    override = f"{separator}or give the matrix with --input-matrix" if matrix_way else ""
    return (
        f"its colour tags contradicted: {', '.join(said)}: a bad file, not read on a guess: {ways}"
        f"{override}"
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
