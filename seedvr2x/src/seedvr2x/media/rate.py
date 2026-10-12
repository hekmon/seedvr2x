"""A source's frames against the timeline of a constant rate (DESIGN.md, Input: exact rational
frame rate): each frame's deviation from that timeline, anchored at the first frame, every one
within half a frame of the declared rate's or the source is refused (media/source.py,
timing_refusal); the rate their timestamps are exactly (exact), which a refusal of a contradicted
rate gives with two fixes; the declared rate's NTSC twin (twin), the one other rate a refusal
names, where sptenc's rule fails, once the frames have followed it a frame away from the declared
rate's timeline; the remake a refusal gives (remake) and the join that keeps the frame grid
(grid_join); and --frame-rate R, accepted when every frame lies within half a frame of R's
timeline. Exact throughout: the frames' pts in their stream's time base, as the frame index holds
them, and rates as fractions."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from seedvr2x.media.index import NOPTS, FrameIndex

# Beyond it, a deviation's numerator could overflow numpy's int64 (_numerators): Python's exact
# integers then, slower.
INT64_SAFE = 2**62

# The rates a file declares, among which the rate its timestamps are exactly is looked for where
# their rounding hides it (exact): the rates ffmpeg itself guesses a stream's r_frame_rate from
# (libavformat/demux.c:2287-2303 at n9.0.2, get_std_framerate), every multiple of 1/12 fps and the
# NTSC forms N * 1000/1001; here without ffmpeg's bounds (multiples of 1/12 up to 30 fps, integers
# up to 60, then 80, 120 and 240, six NTSC forms), so that 100 and 120000/1001 are found too. The
# list names no rate by itself: one is given only when the timestamps are its own to the time
# base's rounding, and no other of the list's holds them so. It was also the list a rate holding
# the frames within half a frame was looked for in, the steadiest given (follows, removed): so
# dense a list that a file was named a rate by its length, 24000/1001 segments joined with 63 ms
# steps "47/2 fps", six frames at 12 fps with one missing "109/12 fps", a frame 20 ms late in 10
# at 25 fps "25000/1001 fps", each then read with the --frame-rate its refusal advised (a
# review's finding and design's check of it, 2026-10-12). Design's decision of 2026-10-12: "a
# refusal names another rate only (a) when the timestamps are that rate's EXACTLY, to the time
# base's rounding, or (b) where sptenc's rule fails, when it is the DECLARED RATE'S NTSC TWIN"
# (twin); "rate.follows' list goes".
TWELFTHS = 12
NTSC = Fraction(1000, 1001)
# The most rates exact tries: more means a span too short to tell one rate from the next (two
# steps over 3 ms allow any from 400 to 2,000 fps, some 19,000 of them), so none is given.
MOST_RATES = 1000

# The FFV1 master the remake makes, as seedvr2x writes its own (media/writer.py, FFV1Writer):
# every frame a keyframe, each slice with its CRC.
FFV1 = ("-c:v:0", "ffv1", "-level:v:0", "3", "-g:v:0", "1", "-slicecrc:v:0", "1")


@dataclass(frozen=True)
class Timeline:
    """How a source's frames lie on the timeline of `rate`, anchored at its first frame that has a
    pts: each frame's deviation, its time less its place's on that timeline, in seconds, exact;
    frames without a pts have none."""

    rate: Fraction  # frames per second
    lowest: Fraction  # the lowest deviation, in seconds, 0 or below
    highest: Fraction  # the highest deviation, in seconds, 0 or above
    first_off: int | None  # the first frame half a frame or more off, None when none is
    first_off_by: Fraction  # that frame's deviation, in seconds; 0 when none is off
    off: int  # how many frames are half a frame or more off

    @property
    def largest(self) -> Fraction:
        """The largest deviation either way, in seconds."""
        return max(-self.lowest, self.highest)

    @property
    def holds(self) -> bool:
        """Whether every frame lies strictly within half a frame of its place on the timeline: the
        nearest place is then each frame's own, so that, taken at that rate, no frame is dropped
        or doubled."""
        return self.first_off is None


def timeline(index: FrameIndex, rate: Fraction) -> Timeline:
    """The frames of index on the timeline of `rate` (frames per second, positive), anchored at
    the first frame that has a pts."""
    places, pts = _timed(index)
    if not len(places):
        return Timeline(rate, Fraction(0), Fraction(0), None, Fraction(0), 0)
    numerators, scale = _numerators(index.time_base, places, pts, rate)
    # A deviation is numerator / scale seconds, scale q N for a time base p/q and a rate N/D, and
    # half a frame D / 2N seconds: a frame is half a frame or more off when |2 numerator| >= q D,
    # integers.
    off = np.abs(2 * numerators) >= index.time_base.denominator * rate.denominator
    found = np.flatnonzero(off)
    first = int(found[0]) if len(found) else None
    return Timeline(
        rate,
        Fraction(int(numerators.min()), scale),
        Fraction(int(numerators.max()), scale),
        None if first is None else int(places[first]),
        Fraction(0) if first is None else Fraction(int(numerators[first]), scale),
        len(found),
    )


def twin(index: FrameIndex, declares: Fraction) -> Timeline | None:
    """The NTSC twin of the rate a file declares that its frames follow, or None: `declares`
    (frames per second) times 1000/1001 or 1001/1000, when its timeline holds every frame
    strictly within half a frame (Timeline.holds) and has parted from the declared rate's by a
    frame or more at the file's last frame. The one rate a refusal names that the timestamps
    aren't exactly (media/source.py, _guided), where sptenc's rule fails: design's decision of
    2026-10-12, "when it is the DECLARED RATE'S NTSC TWIN (R x 1000/1001 or R x 1001/1000)
    holding every frame strictly within half a frame AND the two timelines have parted by a
    frame or more over the file, so that one late frame of a short file is never read as a
    rate". S9's case (research/docs/seeking.md, mechanism 7): 142,477 frames declared 24/1,
    stepping at 24 fps with one step in 500 of 62-63 ms, no rate's exactly, within 10.9 ms of
    24000/1001's timeline, 142 frames from 24's at their end.

    Parted by a frame: the last frame, k frames after the first, has its two places k |1/R -
    1/twin| seconds apart, R the declared rate, k |1 - R/twin| frames of R: k/1000 for R x
    1000/1001, a frame from the 1,001st frame of a file on, and k/1001 for R x 1001/1000, from
    the 1,002nd. A frame, since a timeline holds a frame within half a frame of its place: under
    a frame apart, one frame can lie within half a frame of both places, and the twin's timeline
    holding the file says nothing against the rate declared (a frame 20 ms late in 10 at 25 fps
    lies within 19.8 ms of 25000/1001's timeline; S9's first 260 frames, one long step in them,
    within 21.3 ms of 24/1's); from a frame apart, the last frame, within half a frame of the
    twin's place, lies half a frame from the declared rate's or more (0.4995 of one at the
    least, the twins' frames 0.1% apart): no late frame, but a file that left the timeline it
    declares for its twin's, every frame held there on the way. The twins part from each other
    twice as fast: one at most holds a file so."""
    places, _ = _timed(index)
    if len(places) < 2:
        return None
    steps = int(places[-1] - places[0])
    for other in (declares * NTSC, declares / NTSC):
        if steps * abs(1 - declares / other) < 1:
            continue  # under a frame apart at the last frame
        found = timeline(index, other)
        if found.holds:
            return found
    return None


def exact(index: FrameIndex) -> Timeline | None:
    """The constant rate the frames' timestamps are exactly, to their time base's rounding, or
    None: a rate whose timeline holds every frame within half a frame, their deviations from it
    spread over one tick of the time base at most, the highest less the lowest.

    DESIGN.md's "exactly (to one tick of the file's time base)" (Input: "When the timestamps
    follow another rate"), the rate a refusal names (media/source.py, _guided, _leaves);
    without one, a file that leaves its declared rate's timeline is refused as the drifting join
    it may be ("When the timestamps follow no rate exactly"). A constant
    rate's times, rounded to the time base's ticks, each lie within half a tick of a timeline of
    that rate (rounded to the nearest tick, a tie either way; rounded down, of one half a tick
    earlier): whatever the timeline's anchor, their deviations from it spread over one tick at
    most. Measured in Matroska, whose tick is 1 ms (2026-10-10): frames at 24000/1001 and at 48
    fps, declared 24/1 and 50/1, spread 0.96 and 0.83 of a tick on their own rate's timeline,
    within 0.5 ms of it, 600 and 3,000 frames alike; 33 ms a frame, 0 on 1000/33's; 2,400 frames
    joined by ffmpeg's concat demuxer 1.67 to 2 ticks on the rate looked for nearest them
    (30000/1001 joined every 10 or 12 frames on 30/1's, within 1.0 ms; 24000/1001 every 24 or 20
    on 24/1's, within 1.3 and 1.7 ms), and 2 to 2.6 on their own mean rate's. Half a frame, the
    bound a rate was first looked for by among the same rates (follows, removed: TWELFTHS), took
    those joins for files at 30/1 and 24/1; 33 and 42 ms a frame for 91/3 and 143/6 fps over
    120 to 300 frames, within 3.9 to 12.5 ms, rates that 3,000 frames of each leave; and 17 ms a
    frame for 353/6 fps, within 0.85 ms over 300 frames and 8.5 over 3,000 (a review's findings,
    2026-10-10).

    The rates tried: the frames' own mean rate, their steps over their span, given when every
    frame lies on its timeline, a constant step in ticks, whatever the rate (1000/33 fps for 33
    ms a frame, at any length; 17 ms a frame, 1000/17, lies within 0.85 ms of 353/6's timeline
    too over 300 frames); then the rates looked for (TWELFTHS, NTSC) that the span allows, its
    last frame within a tick of its place, when one alone holds the frames so; then the mean
    rate again, when none of them does and its timeline holds the frames so. A rate that is
    none of those, rounded to ticks, isn't always found: 24.03 fps over 1,000 frames spread
    1.03 ticks on its mean rate's timeline, which the rounding of the first and last frames
    tilts; no rate is then given.

    The rates looked for stay, both families, where the half-frame guess went (TWELFTHS): here
    they give a rate its name and nothing more. 600 frames of 24000/1001 in milliseconds have
    599000/24983 fps for their mean rate, and the simplest fraction that holds them to a tick's
    spread, 2997/125, is no rate a file declares; and the denser the list, the more short files
    it leaves unnamed, by the rule below: without the multiples of 1/12, 18 frames of 48000/1001
    in milliseconds would be named a rate that 575/12 fps holds as exactly.

    Nor is one when several of the rates looked for hold the frames so, a file too short to
    tell them apart: the one the frames lay steadiest on was given, and of 12,872 synthetic
    files 222, all under 83 frames, were named a neighbour of their rate (a review's finding,
    2026-10-10): 120000/1001 as 120/1, 479/4, 719/6 or 1439/12 over 4 to 41 frames, 48000/1001
    as 48/1 or 575/12 over 12 to 13. In milliseconds, several hold the first 12 frames of
    24000/1001, 18 of 48000/1001 and 60 of 120000/1001, 24/1, 575/12 and 120/1 among them, and
    one alone any more of them, up to 1,500."""
    places, pts = _timed(index)
    if len(places) < 2:
        return None
    ticks = int(pts[-1] - pts[0])
    steps = int(places[-1] - places[0])
    if ticks <= 0:
        return None
    tick = index.time_base
    mean = timeline(index, steps / (ticks * tick))
    if mean.largest == 0:
        return mean
    if ticks <= 2:
        return None  # so short a span allows any rate
    # The rates that put the last frame within two ticks of its place: a tick more than a rate
    # that holds the frames so allows, _rates leaving its bounds out.
    low, high = steps / ((ticks + 2) * tick), steps / ((ticks - 2) * tick)
    if (high - low) * TWELFTHS > MOST_RATES:
        return None
    fitting = [
        found for rate in _rates(low, high) if _rounded(found := timeline(index, rate), tick)
    ]
    if len(fitting) > 1:
        return None  # too short to tell them apart
    if fitting:
        return fitting[0]
    return mean if _rounded(mean, tick) else None


def _rounded(found: Timeline, tick: Fraction) -> bool:
    """Whether the frames lie on a timeline as that rate's times rounded to `tick`, seconds, do
    (exact): within half a frame, their deviations spread over one tick at most."""
    return found.holds and found.highest - found.lowest <= tick


def _rates(low: Fraction, high: Fraction) -> list[Fraction]:
    """The rates looked for (TWELFTHS, NTSC) strictly between low and high, the simplest fractions
    first."""
    found = {
        Fraction(k, TWELFTHS)
        for k in range(math.floor(low * TWELFTHS) + 1, math.ceil(high * TWELFTHS))
    }
    found.update(n * NTSC for n in range(math.floor(low / NTSC) + 1, math.ceil(high / NTSC)))
    return sorted((rate for rate in found if low < rate < high), key=lambda r: (r.denominator, r))


def _timed(index: FrameIndex) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The places of the frames that have a pts, and their pts."""
    places = np.flatnonzero(index.pts != NOPTS).astype(np.int64)
    return places, index.pts[places]


def _numerators(
    time_base: Fraction,
    places: npt.NDArray[np.int64],
    pts: npt.NDArray[np.int64],
    rate: Fraction,
) -> tuple[npt.NDArray[Any], int]:
    """Each frame's deviation from the timeline of `rate` anchored at the first, as integer
    numerators over the scale returned: (pts - pts_0) time_base - (place - place_0) / rate seconds,
    time_base p/q and rate N/D, times q N: (pts - pts_0) p N - (place - place_0) q D; int64, or
    Python's integers where twice a numerator could overflow it (INT64_SAFE)."""
    p, q = time_base.numerator, time_base.denominator
    n, d = rate.numerator, rate.denominator
    ticks = pts - pts[0]
    steps = places - places[0]
    widest = int(np.abs(ticks).max()) * p * n + int(steps.max()) * q * d
    if 2 * widest >= INT64_SAFE:
        return ticks.astype(object) * (p * n) - steps.astype(object) * (q * d), q * n
    return ticks * (p * n) - steps * (q * d), q * n


def said(rate: Fraction) -> str:
    """A rate as messages give it, N/D and its decimal to 3 places, its zeros dropped: 24000/1001
    fps (23.976), 25/1 fps (25)."""
    decimal = f"{float(rate):.3f}".rstrip("0").removesuffix(".")
    return f"{fraction(rate)} fps ({decimal})"


def fraction(rate: Fraction) -> str:
    """A rate written N/D, as ffprobe and --frame-rate write it: 24/1, 24000/1001."""
    return f"{rate.numerator}/{rate.denominator}"


def remake(rate: Fraction, start: Fraction) -> str:
    """The ffmpeg command remaking a source at `rate` (DESIGN.md, Input: "Remake the file"): its
    first video stream, the one seedvr2x reads, an FFV1 master numbering its frames at that rate,
    its other streams copied; `start`, its first frame's time after the file's start, in seconds,
    where that frame stays, the other streams' sync with it kept. At the rate the timestamps
    follow, for a file that declares another; at the rate declared, for one whose frames drift
    off its timeline and follow no rate exactly, as a join by ffmpeg's concat demuxer does
    (DESIGN.md, Input: "the remake at the declared rate"): 600 frames at 24000/1001 joined in
    30 segments, up to 24.5 ms early, came out at the timestamps of the file they were split
    from, every frame in order (tests/test_rate.py, 2026-10-10).

    DESIGN.md, Input: "The message's command gives the rate as an input option (`-r:v:0`), which
    numbers the frames as they come": the input option -r:v:0, which gives every frame
    decoded the next place at that rate, from 0 (fftools/ffmpeg_dec.c:398-409 at n9.0.2), so that
    no frame can be dropped or doubled, whatever its timestamp. Of the three retimings
    research/docs/seeking.md measured on S9: fps=R anchors its timeline at 0, not at the first
    frame, and on a fixture whose video starts 23 ms after its audio, its frames within a quarter
    of a frame of 24000/1001's timeline, it dropped one frame and doubled another; -fps_mode cfr
    -r R doubled the first frame there, filling the 23 ms (fftools/ffmpeg_filter.c:2554-2570);
    settb and setpts=N kept every frame but declared the stream's 24/1 again, the rate a filter
    graph takes from its input. Neither -r:v:0 nor setpts=N keeps the first frame's time: 0 for
    the video, its timestamps then shifted back by `start`, in microseconds (settb=AVTB), the
    encoder taking the stream's own time base, so that the shift isn't rounded to the rate's."""
    words = ["ffmpeg", "-r:v:0", fraction(rate), "-i", "SOURCE"]
    words += ["-map", "0", "-map", "-0:d", "-c", "copy", *FFV1, "-fps_mode:v:0", "passthrough"]
    microseconds = round(start * 1_000_000)
    if microseconds:
        words += ["-filter:v:0", f"settb=AVTB,setpts=PTS{microseconds:+d}"]
        words += ["-enc_time_base:v:0", "demux"]
    return " ".join([*words, "remade.mkv"])


# The lines of a POSIX shell that list a directory's segments for ffmpeg's concat demuxer, in
# list.txt beside them: each with its exact duration, its frames (ffprobe's count of its
# packets) over the rate $N/$D, to the microsecond, and bare, for a remake that numbers the
# frames itself (grid_join).
LISTED = (
    'for f in *.mkv; do echo "file \'$f\'"; echo "duration $(ffprobe -v error -select_streams'
    ' v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$f" | awk -v n=$N'
    ' -v d=$D \'NR==1{printf "%.6f", $1*d/n}\')"; done > list.txt'
)
LISTED_BARE = "for f in *.mkv; do echo \"file '$f'\"; done > list.txt"

# What those lines ask of the segments' names, said after each command that lists them
# (grid_join). The shell gives *.mkv in name order, seg_10.mkv before seg_2.mkv: 12 segments
# named seg_1.mkv to seg_12.mkv, typed so, joined as 1, 10, 11, 12, 2 to 9, every frame on the
# grid, read without a word; and a name with an apostrophe ends its line's quotes in list.txt,
# ffmpeg then opening another name, failing there and leaving the segments before it as the
# join, its exit status 0 (ffmpeg n9.0.2; a review's findings, 2026-10-12).
NAMES = (
    "*.mkv the segments' names, taken in name order, which must be the order they play in:"
    " numbers with leading zeros, as seg_000012.mkv, seg_10.mkv coming before seg_2.mkv"
    " otherwise; check list.txt, or write it by hand; a name must hold no apostrophe"
)

# sptenc's snap (its ffmpeg/concat.go, frameGridSnap), for ffmpeg's setts bitstream filter: a
# timestamp to the nearest frame's time at $N/$D, rounded once to the time base; one a packet
# doesn't have left so.
SNAPPED = "'if(eq({0},NOPTS),NOPTS,round(round({0}*TB*$N/$D)*$D/TB/$N))'"


def grid_join(directory: str = "DIR", declares: Fraction | None = None) -> str:
    """How segments are joined so that every frame keeps its place on the frame grid, as a
    refusal advises it, each way with its command, in one place (DESIGN.md, Input: "a join that
    keeps the frame grid", "The advice names a join by ffmpeg alone that holds the grid"): a
    directory's (cli._directory_refused), and media/source.py's, of a file whose frames drift
    off the timeline of the rate it declares (_leaves), as those of a plain join by ffmpeg's
    concat demuxer do, up to 1 ms a join; `directory`, the segments', as a shell takes it, DIR
    for a refusal that can't know it; `declares`, their frame rate, in the commands, 24000/1001
    as an example, said to be one, for a refusal that can't know it. What each way was measured
    to hold, and the plain join's drift, are with cli._directory_refused.

    By ffmpeg alone first, seedvr2x never needing sptenc, sptenc's own method (its
    ffmpeg/concat.go) in a shell's lines: the concat demuxer given each segment's exact
    duration, its frames over the rate, where it would start each segment by the duration the
    one before declares, off by up to a millisecond; and every timestamp snapped to the grid in
    the same stream copy, the nearest frame's time rounded once to the time base, since a
    segment encoded on its own still comes a millisecond off the others' rounding. -safe 0
    takes names the demuxer's safe mode refuses. No -fflags +genpts: it changes nothing for
    Matroska or MP4 segments, and an MPEG program stream's, whose packets lack timestamps, then
    join with a frame out of place, 1 of 812 (tests/test_rate.py), where without it Matroska's
    muxer stops at the first such packet, ffmpeg saying so.

    Then sptenc's concat, the same method. Last, where no stream copy holds, the segments
    decoded and written again as an FFV1 master, the input option -r numbering their frames at
    the rate as they come (remake), from a list without durations, the video alone.

    Both commands list the segments by the shell's *.mkv, and each says what that asks of
    their names (NAMES)."""
    # PROVISIONAL (implementation, 2026-10-12, questions for design: DESIGN.md asks for a join
    # by ffmpeg alone that holds the grid and names no command). (1) Durations and the snap, in
    # one pass, over the other join that held as widely: the plain join, then a second stream
    # copy counting each packet's distance from the one before in frames (setts with a time
    # base of one frame), two commands and no ffprobe, but a whole second copy of the film, and
    # timestamps that depend on every packet before them. (2) The rate as the shell's N and D,
    # an example where the refusal can't know it, rather than read from a segment by ffprobe,
    # which gives Matroska's 60000/1001 as 19001/317. (3) Program streams sent to the remake
    # rather than -fflags +genpts advised for them (see the docstring). (4) The segments listed
    # by the shell's glob, its name order and the apostrophe said (NAMES), rather than sorted by
    # their numbers (ls -v and sort -V are GNU's, and a listing parsed breaks on other names)
    # or escaped for the demuxer by a sed in the line, which nobody could read.
    rate = Fraction(24000, 1001) if declares is None else declares
    example = "" if declares is not None else f", {fraction(rate)} here"
    snap = f"setts=pts={SNAPPED.format('PTS')}:dts={SNAPPED.format('DTS')}"
    there = f"cd {directory} &&"
    ways = [
        "with ffmpeg alone, each segment listed with its exact duration, its frames over the"
        " rate, and every timestamp snapped to the grid in one stream copy:"
        f" {there} N={rate.numerator} D={rate.denominator} && {LISTED} && ffmpeg -f concat"
        f' -safe 0 -i list.txt -c copy -bsf:v "{snap}" ../joined.mkv ({NAMES}; N/D their'
        f" frame rate{example})",
        f"with sptenc, for segments of its split, sptenc concat {directory} joined.mkv, which"
        " snaps every timestamp to the grid",
        "where no stream copy holds, for segments that carry audio, MPEG program streams, or"
        " transport streams of a single frame, their pictures remade as an FFV1 master, numbered"
        f" at that rate as they come, every one kept: {there} {LISTED_BARE} && ffmpeg -f concat"
        f" -safe 0 -r {fraction(rate)} -i list.txt -map 0:v:0 {' '.join(FFV1)} -fps_mode:v:0"
        f" passthrough ../remade.mkv ({NAMES})",
    ]
    return ", or ".join(ways[:2]) + ", or, " + ways[2]


def first_time(index: FrameIndex) -> Fraction:
    """The first frame's time after the file's start, in seconds: where ffmpeg puts it when it
    reads the file without -copyts (fftools/ffmpeg_demux.c, ts_offset); 0 without a pts."""
    pts = int(index.pts[0]) if index.frames else NOPTS
    if pts == NOPTS:
        return Fraction(0)
    return pts * index.time_base - Fraction(index.start_time or 0, 1_000_000)
