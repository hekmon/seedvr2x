"""A source, examined before any GPU work: what it declares, what seedvr2x refuses, how its frames
become RGB, and the first pass (DESIGN.md, Input). A source is one video file, a job's only input.
A job resumed checks its record between what the source declares and its first pass, whose record
it then trusts when nothing changed (DESIGN.md, Pause and resume)."""

import logging
import threading
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from seedvr2x.media import bitstream, rate
from seedvr2x.media.conversion import PIXEL_FORMATS, Conversion, conversion_for
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.files import sha256
from seedvr2x.media.index import FrameIndex
from seedvr2x.media.probe import DoviRecord, VideoStream, probe
from seedvr2x.media.rate import Timeline
from seedvr2x.media.reader import Reader
from seedvr2x.media.scan import TOLERANCE_US, Scan, scan, timing_error

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

# Two names of one meaning, the stream's and its first frame's, are no contradiction (DESIGN.md,
# Input): the stream's name is kept (resolved). ITU-T H.273 (07/2024) says which code points are
# "functionally the same": the matrices 5 and 6 (Table 4), ffmpeg's bt470bg and smpte170m, whose
# coefficients are equal (libavutil/csp.c:48-49 at n9.0.2; "functionally identical", pixfmt.h:707);
# the transfers 1, 6, 14 and 15 (Table 3), its bt709, smpte170m, bt2020-10 and bt2020-12
# (pixfmt.h:668, 673, 681-682), one function in its own tables (csp.c:445-457). The primaries
# bt470bg and smpte170m differ (Table 2, csp.c:79-80). Provisional (DESIGN.md names the matrix's
# and the transfer's alone): H.273's primaries 6 and 7, smpte170m and smpte240m, are "functionally
# the same" too (Table 2; "identical", pixfmt.h:645), and still refused.
SYNONYMS = {
    "color_space": frozenset({"bt470bg", "smpte170m"}),
    "color_transfer": frozenset({"bt709", "smpte170m", "bt2020-10", "bt2020-12"}),
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

# The remake's cost, as DESIGN.md gives it (Input): FFV1 took 0.53-0.55 MB a frame on S9, a 1080p
# Blu-ray film (research/docs/seeking.md, mechanism 7), 80 GB against its source's 14.5 GB.
REMAKE_COST = "lossless, so large: 0.53-0.55 MB a frame on a 1080p film; and a full encode"


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
    declared; its content's SHA-256, when hashed; and its frame index (media/index.py)."""

    frames: int
    sha256: str | None = None
    index: FrameIndex | None = None


@dataclass(frozen=True)
class Source:
    """A video file seedvr2x reads: its first video stream, its colour tags resolved (resolved)."""

    path: Path
    stream: VideoStream
    conversion: Conversion
    frames: int  # counted by the first pass
    sample_aspect: Fraction  # the override, else the declared one, else 1 (square pixels)
    sha256: str | None = None  # the file's content, hashed for a manifest
    index: FrameIndex | None = None  # the first pass's frame index, or its record's
    rate_override: Fraction | None = None  # --frame-rate, which its first pass accepted

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
    for name, (declared_as, decoded_as) in _sides(probed).items():
        settled = name == "color_space" and matrix is not None  # read as --input-matrix's
        if _one_meaning(name, declared_as, decoded_as) and not settled:
            # Provisional (DESIGN.md doesn't say whether it is said): the conversion, the output's
            # tags and the manifest take the stream's name (resolved), not the bitstream's.
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


def first_pass(declared: Declared, hashed: bool = False) -> FirstPass:
    """Decode every frame of the file once, to count, time and index them (media/scan.py), refused
    unless at the constant rate declared, or --frame-rate's (timing_refusal); and, when `hashed`,
    hash its content meanwhile, from another thread: the file read a second time, at the speed of
    the disk, while ffmpeg decodes it.

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
    scanned = scan(path)
    refusal = timing_refusal(declared, scanned)
    if refusal:
        raise MediaError(f"{path}: {refusal}")
    if not hashed:
        return FirstPass(scanned.frames, index=scanned.index)
    reader.join()
    [digest] = hashed_as
    if isinstance(digest, OSError):
        raise MediaError(f"{path}: not readable: {digest}") from digest
    if isinstance(digest, Exception):
        raise digest
    return FirstPass(scanned.frames, digest, scanned.index)


def counted(declared: Declared, found: FirstPass) -> Source:
    """The source declared, its frames counted and indexed by its first pass, or by the record of
    one; warned of when, read at the rate it declares, its frames leave that rate's timeline
    (_strays)."""
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
    )
    declares = rate.fraction(stream.frame_rate)
    forced = "" if source.rate_override is None else f" (--frame-rate; it declares {declares})"
    logger.info(
        "%s: %d frames, %dx%d %s at %s fps%s, sample aspect %s (display %s), read as %s",
        source.path,
        source.frames,
        stream.width,
        stream.height,
        stream.pix_fmt,
        source.frame_rate,
        forced,
        _ratio(source.sample_aspect),
        _ratio(source.display_aspect),
        source.conversion.describe(),
    )
    if source.rate_override is None:
        # Here, not in the first pass's own check (timing_refusal): a resume that takes the pass
        # from its record reads on at the declared rate, and says so again from the recorded
        # index. Said by the pass alone, a job resumed read on without it (a review's finding,
        # 2026-10-10).
        _strays(source)
    return source


def timing_refusal(declared: Declared, scanned: Scan) -> str:
    """Why the first pass's frames are refused for their timing, or "" (DESIGN.md, Input: exact
    rational frame rate).

    A jump in the timestamps is refused first, whatever the rate (scan.Discontinuity). Without
    --frame-rate, sptenc's rule against the rate declared (scan.timing_error). A file whose
    timestamps contradict it is a bad file, refused, never reinterpreted on seedvr2x's own; when
    they are a constant rate's exactly, or else follow one within half a frame (_guided), the
    refusal gives that rate, how far they stray from its timeline, and two fixes, a remake at
    that rate or --frame-rate. A file the rule passes is read at the rate it declares, warned of
    when its frames leave that rate's timeline (counted, _strays). With --frame-rate R: every
    frame within half a frame of R's timeline (_off)."""
    path, stream, index = declared.path, declared.stream, scanned.index
    error = timing_error(scanned, stream.frame_rate, stream.avg_frame_rate)
    if scanned.discontinuity is not None:
        return error
    forced = declared.rate_override
    if forced is None:
        if index is None or not error:
            return error
        found = _guided(index)
        if found is None:
            return error
        last = f"{_ms(scanned.shortest)} to {_ms(scanned.longest)} ms"
        if found.rate == stream.frame_rate:
            what = (
                f"its timestamps follow {rate.said(found.rate)}, the rate it declares,"
                f" {_within(found)}, but its frames last from {last}, where a constant rate's"
                f" lie within {_ms(TOLERANCE_US)} ms of each other"
            )
        else:
            what = (
                f"its timestamps run at {rate.said(found.rate)}, {_within(found)}, where it"
                f" declares {rate.fraction(stream.frame_rate)}, its frames lasting from {last}"
            )
        return f"{what}: a bad file, not read on a guess: {_fixes(found.rate, index)}"
    if index is None or scanned.frames == 0:
        return "no frame decoded"
    # R replaces the declared rate, and sptenc's bound of 1 ms on the frames' durations with it:
    # S9's frames step 41 to 63 ms (research/docs/seeking.md, mechanism 7), yet each lies within
    # 10.9 ms of 24000/1001's timeline. Within half a frame, a frame's nearest place on R's
    # timeline is its own, so that, taken at R, none is dropped or doubled: the bound that keeps
    # every frame once.
    on = rate.timeline(index, forced)
    if not on.holds:
        return _off(on, declared, index, passes=not error)
    declares = rate.fraction(stream.frame_rate)
    logger.info(
        "%s: --frame-rate %s: every frame %s, %s: taken at that rate",
        path,
        rate.fraction(forced),
        _within(on, "its"),
        "the rate it declares" if forced == stream.frame_rate else f"where it declares {declares}",
    )
    if not error and forced != stream.frame_rate and rate.timeline(index, stream.frame_rate).holds:
        # PROVISIONAL (DESIGN.md has R accepted when every frame lies within half a frame of its
        # timeline, and says nothing of a file whose frames lie on the timeline of the rate it
        # declares as well): taken at R all the same, as asked, said, since its output then lasts
        # another time than its source, off the source's other streams. Only a short file's
        # frames lie on both: 24 and 24000/1001 part by half a frame within 500 frames. By the
        # declared rate's timeline, never by sptenc's rule alone, which doesn't tell: frames at
        # 24000/1001 last within 1 ms of 24's (_strays), and a file of them declared 24/1 was
        # told its frames were at 24/1 "too" when given its own rate.
        change = stream.frame_rate / forced - 1
        logger.warning(
            "%s: its frames are at the rate it declares, %s, too: taken at %s as asked, the output"
            " lasting %.3f%% %s than the source, off its other streams",
            path,
            rate.fraction(stream.frame_rate),
            rate.fraction(forced),
            abs(float(change)) * 100,
            "shorter" if change < 0 else "longer",
        )
    return ""


def _guided(index: FrameIndex) -> Timeline | None:
    """The rate a refusal names for a file sptenc's rule refuses at the rate it declares, None
    without one: the rate its timestamps are exactly (rate.exact); else the one they follow
    within half a frame (rate.follows), DESIGN.md's own case, S9's, whose timestamps are no
    rate's exactly, their long steps catching up, and lie within 10.9 ms of 24000/1001's timeline
    (research/docs/seeking.md, mechanism 7).

    PROVISIONAL (implementation, 2026-10-10: DESIGN.md's Input has "its timestamps follow that
    rate", of S9, and not how the rate is found). Half a frame's rate alone named one by the
    file's length: 33 ms a frame declared 25/1 ran "at 91/3 fps (30.333), within 3.9 ms" over
    120 frames, --frame-rate 91/3 then accepted, and at no rate over 3,000, refused without
    guidance; a frame every 11 s, 6 frames, "at 1/12 fps (0.083), within 5000 ms". Exactly, they
    are 1000/33 fps at both lengths, and 1/11 (a review's findings, 2026-10-10)."""
    return rate.exact(index) or rate.follows(index)


def _strays(source: Source) -> None:
    """Warn of a source that passes sptenc's rule and yet leaves the timeline of the rate it
    declares, some frame half a frame or more from its place there: how many frames, from which,
    how far; that it is read at the rate it declares all the same, frame after frame; and, when
    its timestamps are another constant rate's exactly (rate.exact), that rate, which
    --frame-rate takes its frames at, and what the declared one does to the output's duration.
    From the source's frame index, the first pass's or its record's (counted).

    sptenc's rule bounds each frame's duration, not where the frames end up: frames at 24000/1001
    declared 24/1 last 41.7 ms, within 1 ms of 24's 41.667, yet lie half a frame off 24's timeline
    from frame 500 on, 0.1% further each frame, and frames at 48 fps declared 50/1 last 20.8 ms
    for 20, 4%: both are read, their output written at the rate declared, 0.1% and 4% shorter
    than their frames' own time, off the source's other streams (a review's finding, 2026-10-09).

    PROVISIONAL (implementation, 2026-10-10, a question for design: DESIGN.md's Input has sptenc's
    rule, "the declared rate must also match the measured durations", and accepts the joins of
    ffmpeg's concat demuxer, "1 ms early per join, off the frame grid"): no file the rule passes
    is refused, as before the frame index, and what is read on is said, with its reason
    (AGENTS.md: no silent fallback). The refusal tried first, of the files whose frames followed
    another rate within half a frame (rate.follows), refused files by their length, and the very
    joins the refusal of a directory advises (cli._directory_refused), all read before: 2,400
    frames at 24000/1001 joined every 20 or 24 frames, as running at 24/1; 33 ms a frame
    declared 30/1 over 120 and 300 frames, as running at 91/3 fps, and read over 3,000 (a
    review's findings, 2026-10-10). A rate is named only when the timestamps are that rate's to
    their time base's rounding (rate.exact), which no join measured is but one at every frame, a
    constant step."""
    stream, index = source.stream, source.index
    if index is None:
        return
    on = rate.timeline(index, stream.frame_rate)
    if on.holds:
        return
    said = (
        f"{on.off} of its {source.frames} frames half a frame or more from their place on the"
        f" timeline of the rate it declares, {rate.fraction(stream.frame_rate)}, from frame"
        f" {on.first_off}, up to {_ms(on.largest * 1_000_000)} ms"
        f" ({float(on.largest * on.rate):.2f} of a frame): read at the rate it declares all the"
        " same, frame after frame, as a file joined by ffmpeg's concat demuxer is"
    )
    found = rate.exact(index)
    if found is not None:
        change = found.rate / stream.frame_rate - 1
        said += (
            f"; {_exactly(found, index)}, where the rate it declares makes its output"
            f" {abs(float(change)) * 100:.3f}% {'shorter' if change < 0 else 'longer'} than its"
            " timestamps' own time, off the source's other streams"
        )
    logger.warning("%s: %s", source.path, said)


def _exactly(found: Timeline, index: FrameIndex) -> str:
    """The rate a source's timestamps are exactly (rate.exact), as messages give it, with the
    option that takes its frames at it."""
    return (
        f"its timestamps follow {rate.said(found.rate)} exactly, to the rounding of its time"
        f" base, {index.time_base} s: --frame-rate {rate.fraction(found.rate)} takes its frames"
        " at that rate"
    )


def _off(on: Timeline, declared: Declared, index: FrameIndex, passes: bool) -> str:
    """Why --frame-rate R is refused, its frames not all within half a frame of R's timeline: the
    first one off, how far, how many. Then, of a file sptenc's rule passes at the rate it
    declares (`passes`): that it is read without the option, at that rate, frame after frame;
    and the rate its timestamps are exactly (rate.exact), when they are another's. Of a file
    the rule refuses: the rate its refusal names (_guided), and the two fixes at it.

    PROVISIONAL (implementation, 2026-10-10, where DESIGN.md doesn't say): a file the rule
    passes is never given a rate that holds its frames within half a frame alone (rate.follows).
    The join a directory's refusal advises (cli._directory_refused), 2,400 frames at 24000/1001
    joined every 20, given the rate it declares, was told "its timestamps run at 24/1 fps (24),
    within 1.7 ms of that rate's timeline", and --frame-rate 24/1 then read it at 24, 0.1% off,
    without a warning, where it is read at 24000/1001 without the option (a review's finding,
    2026-10-10)."""
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
    if passes:
        declares = declared.stream.frame_rate
        reason += (
            "; without --frame-rate, the file is read at the rate it declares,"
            f" {rate.fraction(declares)}, frame after frame, whatever their place on that rate's"
            " timeline"
        )
        exact = rate.exact(index)
        if exact is None or exact.rate == declares:
            return reason
        return f"{reason}; {_exactly(exact, index)}"
    found = _guided(index)
    if found is None or found.rate == on.rate:
        return reason
    return (
        f"{reason}; its timestamps run at {rate.said(found.rate)}, {_within(found)}:"
        f" {_fixes(found.rate, index)}"
    )


def _within(found: Timeline, whose: str = "that rate's") -> str:
    """How far the frames stray from a rate's timeline, `whose`: within 10.8 ms of that rate's
    timeline (0.26 of a frame)."""
    return (
        f"within {_ms(found.largest * 1_000_000)} ms of {whose} timeline"
        f" ({float(found.largest * found.rate):.2f} of a frame)"
    )


def _fixes(found: Fraction, index: FrameIndex) -> str:
    """The two fixes of a file whose timestamps follow the rate `found`, not the one it declares
    (DESIGN.md, Input): a remake at that rate (rate.remake), or --frame-rate."""
    command = rate.remake(found, rate.first_time(index))
    return (
        "remake it at that rate, an FFV1 master with its other streams copied, a clean file for"
        f" every tool ({REMAKE_COST}), with {command}; or give --frame-rate {rate.fraction(found)},"
        " which takes its frames at that rate, each checked to lie within half a frame of its"
        " timeline: the same frames on the same timeline, without the intermediate file"
    )


def _ms(microseconds: Fraction | int) -> str:
    """A time in microseconds, in milliseconds to a tenth, a whole one without its .0: 10.8, 41,
    62.5."""
    return f"{float(microseconds) / 1000:.1f}".removesuffix(".0")


def declared_refusal(stream: VideoStream, matrix: str | None = None) -> str:
    """Why a stream is refused for what it declares and its first frame carries, every such reason
    in one, numbered when there are several, or "": interlacing; HDR, Dolby Vision unless over an
    SDR base layer; colour tags its first frame contradicts in meaning (SYNONYMS), the matrix's but
    when `matrix` (--input-matrix) settles it, the chroma location's never (SITING); a rotation or
    flip; a declared crop (DESIGN.md, Input, Not in the first version)."""
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
    # only for a filter setting each tag contradicted, the file otherwise still refused.
    # Provisional (DESIGN.md names neither command): both write
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
