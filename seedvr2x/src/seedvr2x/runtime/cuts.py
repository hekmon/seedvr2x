"""A source's cuts (DESIGN.md, Shot detection): where its shots start, detected from the shot
detector's per-frame probabilities, the first pass's record (media/index.py), at a threshold, so
that another threshold gives its cuts without a second decode or detection; and the cut list
--plan writes, the cuts and the possible ones, for editing and --cuts (DESIGN.md, Input: the cut
list format, read by job.read_cuts), with a playlist of each beside it, for VLC.

A detection is a run of frames whose single-frame probability reaches the threshold, one per run,
at its peak. The cut is the frame after the peak: TransNetV2 marks the outgoing shot's last frame
(offset -1 on 98% of measurement's sure cuts). Every detection starts a shot, however short: no
burst filter and no minimum shot length, a short shot being better run alone than merged into its
neighbour, from 1 frame on, and real cuts coming 1-3 frames apart in action anime."""

import logging
import math
import os
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

import numpy as np
import numpy.typing as npt

from seedvr2x.media import files
from seedvr2x.media.grid import filling
from seedvr2x.media.index import NOPTS, FrameIndex

logger = logging.getLogger(__name__)

# DESIGN.md, Shot detection: 0.3 by default, confirmed by the user's second round of labels; below
# it each cut found costs 6 to 14 false cuts on animation, 14 to 21 on live action
# (research/docs/scene-detection.md, The threshold below 0.3).
THRESHOLD = 0.3
# DESIGN.md, Shot detection: the possible cuts are the runs peaking from 0.1 up to the threshold,
# about 250 an hour of animation at 0.3, one real cut in 7 to 18 of them; most misses score under
# 0.1, beyond any threshold (77 to 81 of animation's 92 an hour).
POSSIBLE = 0.1


@dataclass(frozen=True)
class Cut:
    """A cut: `frame`, the first frame of the shot it starts, counted from 0 in the source; and
    `probability`, the peak of the run that detected it, on the frame before (float32 in [0, 1],
    as recorded), None for a cut list's."""

    frame: int
    probability: float | None = None


def detections(probabilities: npt.NDArray[np.floating], threshold: float) -> list[int]:
    """The detections at threshold, each the frame of its peak: one per run of frames whose
    probability reaches the threshold, at the first of its highest frames, as measurement detects
    (research/scripts/scd_scores.py's tnet_detections).

    probabilities: (frames,), each frame's, float32 as the first pass records them (others are
    rounded to it); threshold in (0, 1]. Compared exactly, each float32 value against the
    threshold as Python reads it, a double (_exact): "reaches" taken at its word, and
    measurement's own comparison, whose analysis reads its recorded float32 values as float64
    before it detects (scd_scores.py's load_tnet and tnet_align). So float32's 0.7, 0.699999988,
    doesn't reach 0.7, where float32's 0.3, 0.300000012, reaches 0.3. ValueError on a NaN, which
    no sigmoid gives: a seedvr2x bug."""
    return [peak for _, _, peak in _runs(_exact(probabilities), threshold)]


def detected(probabilities: npt.NDArray[np.floating], threshold: float) -> list[Cut]:
    """The cuts detected at threshold (detections): the frame after each peak, with the peak's
    probability. A peak on the last frame gives none: no frame comes after it to start a shot."""
    values = _exact(probabilities)
    return [
        Cut(peak + 1, float(values[peak]))
        for _, _, peak in _runs(values, threshold)
        if peak + 1 < len(values)
    ]


def possible(probabilities: npt.NDArray[np.floating], threshold: float) -> list[Cut]:
    """The possible cuts at threshold, which --plan's cut list gives commented, to be checked: the
    cut of each run of frames from POSSIBLE up whose peak stays under the threshold, the frame
    after its peak, with the peak's probability; none on the last frame, as detected. Never a
    detection's: a run peaking at the threshold or above holds one. None when the threshold is
    POSSIBLE or under. Both bounds compared exactly, as detections compares."""
    values = _exact(probabilities)
    return [
        Cut(peak + 1, float(values[peak]))
        for _, _, peak in _runs(values, POSSIBLE)
        if values[peak] < threshold and peak + 1 < len(values)
    ]


def _exact(probabilities: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """The probabilities as recorded, float32, each widened to the double of its exact value, so
    that a comparison with a threshold, a double, is exact. Compared in float32, the threshold
    was rounded to float32 first: a probability of float32's 0.7, 0.699999988, then reached 0.7,
    as it does for 0.35, 0.45, 0.65 and 0.9, where measurement's analysis, in float64, finds no
    detection; and a threshold of 1e-50 became 0 (a review's finding, 2026-10-10)."""
    values = np.asarray(probabilities, dtype=np.float32).astype(np.float64)
    nan = np.flatnonzero(np.isnan(values))
    if len(nan):
        raise ValueError(f"a NaN shot probability at frame {int(nan[0])}: a seedvr2x bug")
    return values


def _runs(values: npt.NDArray[np.float64], threshold: float) -> list[tuple[int, int, int]]:
    """Each run of frames whose probability (_exact) reaches threshold, in order: (first, end,
    peak), its frames [first, end), its peak the first of its highest frames (np.argmax)."""
    reached = np.concatenate(([0], (values >= threshold).astype(np.int8), [0]))
    edges = np.diff(reached)
    firsts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [
        (int(first), int(end), int(first) + int(np.argmax(values[first:end])))
        for first, end in zip(firsts, ends, strict=True)
    ]


def time_of(index: FrameIndex, frame: int) -> Fraction | None:
    """Frame `frame`'s time in seconds, as a player shows it, from its timestamp in the frame
    index: from the file's start, the start time ffprobe reads, which mpv takes for 0:00
    (--rebase-start-time, on by default); from the first frame's when the file declares none.
    None for a frame without a timestamp (index.NOPTS, which ffmpeg n9.0.2 never gives); from
    the stream's own zero when the file declares no start and its first frame has none either,
    as rate.first_time counts.

    A frame is a place of the index, its row (media/index.py). A filled place, where a player
    still shows the frame before its hole, has the time of that place: the timestamp its row
    holds, that frame's plus the frame periods since (media/grid.py, filled).

    DESIGN.md, Shot detection: a line holds the cut's "time from the file's start": the file's
    start, not the first frame's, so that a source whose video starts after its audio
    (by 23 ms, say, on a fixture of S9's mechanism 7: research/docs/seeking.md) gets the times a
    player shows; from the first frame, every time would be early by that much, more than half a
    frame at 24 fps, and the jump would land on the outgoing shot's last frame."""
    pts = int(index.pts[frame])
    if pts == NOPTS:
        return None
    if index.start_time is not None:
        origin = Fraction(index.start_time, 1_000_000)
    else:
        first = int(index.pts[0])
        # NOPTS is the lowest int64: as an origin, it gave "102481911520608:37:12.400".
        origin = 0 if first == NOPTS else first * index.time_base
    return pts * index.time_base - origin


def timecode(seconds: Fraction | None) -> str:
    """H:MM:SS.mmm, rounded up to the millisecond; "-" without a time.

    DESIGN.md, Shot detection: the cut's "time from the file's start, rounded up to the
    millisecond", for the player that checks the cuts: "The player is VLC, the user's choice
    (2026-10-10)", which "lands within 3 frames of the cut on a file with B-frames, mostly before
    it, which a time rounded up serves better than one rounded down; mpv lands on the cut frame".
    Rounded up, the time lies in its frame's own display, at its start or within a millisecond
    after. A seek to the first frame at or after the time lands one frame late instead, wherever
    the frame's time isn't a whole millisecond: ffmpeg -ss T -i FILE gave the next frame for 45
    of the 48 frames of a 24000/1001 MP4 (time base 1/24000), and the frame itself for all 48
    with the time rounded down; on Matroska, whose times are milliseconds, the frame itself
    either way (measured 2026-10-10, n9.0.2)."""
    if seconds is None:
        return "-"
    milliseconds = math.ceil(seconds * 1000)
    sign, value = ("-" if milliseconds < 0 else ""), abs(milliseconds)
    hours, minutes = value // 3_600_000, value // 60_000 % 60
    return f"{sign}{hours}:{minutes:02d}:{value // 1000 % 60:02d}.{value % 1000:03d}"


def probability(value: float) -> str:
    """A probability to 3 decimals: the float32 value's exact decimal, cut, never rounded up, so
    that a possible cut never shows at the threshold it stays under, nor a cut under the one it
    reaches (a threshold of 3 decimals or fewer), as they are compared (_exact): 0.29999998 is
    0.299; float32's 0.7, 0.699999988, which doesn't reach 0.7, is 0.699; float32's 0.3,
    0.300000012, which reaches 0.3, is 0.300. DESIGN.md (Shot detection) asks for "its
    probability" and no more; cut to 3 decimals, accepted by design as built (2026-10-10): a peak
    scoring 0.9939 shows 0.993, where a rounded figure would say 0.994."""
    exact = Decimal(float(np.float32(value)))
    return str(exact.quantize(Decimal("0.001"), rounding=ROUND_DOWN))


def cut_list(
    header: Sequence[str], index: FrameIndex, cuts: Sequence[Cut], possible: Sequence[Cut] = ()
) -> str:
    """The cut list --plan writes (DESIGN.md, Shot detection), in the cut list format, so that
    --cuts reads it back (job.read_cuts: a frame number per line, # comments, the fields after the
    number ignored): the header's lines as comments, then a line per cut and per possible cut, in
    frame order: a cut `<frame> <probability> <time>`, a possible cut commented, `# <frame>
    <probability> <time>`, so that keeping one is uncommenting its line; a cut list's own cut
    `<frame> <time>`. The probability is its run's peak, on the frame before (probability); the
    time the cut frame's, as a player shows it (time_of, timecode), a jump to it in a player
    checking the cut.

    The header's lines hold names, the source's and a cut list's: whatever in them isn't
    printable is written escaped (a line break, which would start a line of the name's own, read
    back as a cut when it begins with a number: job.read_cuts splits as str.splitlines does). No
    header line may begin with a digit: uncommenting every "# <digit>" line, as an editor's
    search does to keep every possible cut, would make a cut of it (ValueError: a seedvr2x
    bug)."""
    lines: list[str] = []
    for line in header:
        escaped = "".join(
            char if char.isprintable() else char.encode("unicode_escape").decode() for char in line
        )
        if escaped[:1].isdigit():
            raise ValueError(f"a cut list's header line begins with a digit: {escaped!r}")
        lines.append(f"# {escaped}" if escaped else "#")
    found = [(cut, False) for cut in cuts] + [(cut, True) for cut in possible]
    for cut, commented in sorted(found, key=lambda item: item[0].frame):
        fields = [str(cut.frame)]
        if cut.probability is not None:
            fields.append(probability(cut.probability))
        fields.append(timecode(time_of(index, cut.frame)))
        lines.append(("# " if commented else "") + " ".join(fields))
    return "\n".join(lines) + "\n"


def places_note(index: FrameIndex) -> list[str]:
    """What the cut list's header says of a source with missing frames, as header lines
    (cut_list), wrapped as cli._write_plan wraps its own, at 86 columns; none for a source
    without a filled place, whose header stays as it was.

    Such a source's index is in places (media/grid.py), and a cut's frame number is its place,
    the missing frames' filled ones counted: the list says so where its user reads it, with how
    many are filled, in how many holes and where the first is, and that a list numbered by a tool
    counting the frames it decodes is off by the holes before each cut (design, 2026-10-12:
    "FRAMES COUNT PLACES, SAID WHERE THE USER READS IT: the cut list's header when places are
    filled"). Frames are numbered as the source's warning numbers them (source._missing)."""
    found = filling(index)
    first = found.first
    if first is None:
        return []
    missing = f"{found.filled} frame{'' if found.filled == 1 else 's'} missing"
    where = (
        f"after frame {first.held}"
        if found.holes == 1
        else f"in {found.holes} holes, the first after frame {first.held}"
    )
    text = (
        "Frame numbers count places, the missing frames' filled ones among them:"
        f" {missing} {where}, {found.places} places for {found.decoded} frames decoded."
        " A list numbered by a tool that counts the decoded frames is off by the holes before"
        " each cut, its number lower by the frames missing before it."
    )
    # No header line may begin with a digit (cut_list): a figure stays on the line of the word
    # before it, the space between them no place to break at.
    lines = textwrap.wrap(re.sub(r" (?=\d)", "\x00", text), 86)
    return [line.replace("\x00", " ") for line in lines]


# The playlists of the cuts (DESIGN.md, Shot detection): "an entry per possible cut starting a
# second before it, so that VLC goes from one to the next with no time to type". Measured in VLC
# 3.0.24 for Windows, stock settings but software decoding and no window, on H.264 at 24000/1001
# with a keyframe every 48 frames, 19 entries a file, each as a list of its own and all in one
# list played through. MP4 with B-frames: the first frame shown is 0 to 3 frames before the
# entry's start (1 after, once), 20 to 26 frames before the cut on 18 entries of 19 (the cut at
# frame 5 has the 5 before it), and the cut is inside the entry for 19 of 19 alone, 18 of 19 in
# the list played through. Matroska with B-frames: 0 to 2 frames before the start, the cut
# inside 19 of 19 both ways. Without B-frames: on the start's frame, 19 of 19. A file of one
# long GOP: within a frame of the start, 19 of 19. MPEG-TS is rough: the cut inside 14 of 19,
# the entry beginning up to 31 frames late or 19 early. An entry stops 0 to 9 frames after its
# stop, and VLC goes on to the next by itself. Not measured: hardware decoding, VLC's default;
# HEVC; VLC 4.
BEFORE = 1
# PROVISIONAL (implementation, 2026-10-10, a question for design: how long an entry plays after
# its cut; DESIGN.md gives its start alone): 3 seconds, the new shot seen long enough to tell it
# from the one before, an entry 4 seconds in all, 250 of them, an hour of animation's, 17 minutes.
AFTER = 3
# VLC 3.0.24 reads a .m3u8 file as UTF-8 (modules/demux/playlist/m3u.c:93-95), and Windows opens
# it with VLC as it does .m3u (the installer's file types); these are ASCII, read the same under
# both.
PLAYLIST_SUFFIX = ".m3u8"


@dataclass(frozen=True)
class Playlist:
    """One of the two playlists --plan FILE writes beside FILE when its shots are detected
    (design, 2026-10-12: "A second playlist, of the DETECTED cuts, beside the first, named so
    that the two can't be taken for each other"): `name`, what its file's name holds between
    FILE's and .m3u8 (playlist_path); `cut`, one of its cuts as a message and its own comments
    name it; `title`, what each entry's title calls its cut, the detected one's the plain word,
    as the cut list has its line plain and a possible cut's commented."""

    name: str
    cut: str
    title: str


# PROVISIONAL (implementation, 2026-10-12, a question for design: the two names, DESIGN.md
# saying "beside the list" and design asking only that they can't be taken for each other):
# FILE.possible.m3u8 and FILE.cuts.m3u8, each named by what it lists, over FILE.m3u8 kept for
# the possible cuts' beside a FILE.cuts.m3u8: a name that says nothing is the one taken for the
# other, and no user has a FILE.m3u8 yet, the first playlist landing with the second.
OF_POSSIBLE = Playlist("possible", "possible cut", "possible cut")
OF_DETECTED = Playlist("cuts", "detected cut", "cut")
# In the order they are written and said, the cut list's own: its cuts, then the possible ones.
PLAYLISTS = (OF_DETECTED, OF_POSSIBLE)


def playlist_path(plan: Path, which: Playlist) -> Path:
    """Where one of --plan FILE's playlists goes: beside FILE, FILE's name with .possible.m3u8
    added for the possible cuts' and .cuts.m3u8 for the detected cuts' (plan.txt.cuts.m3u8,
    x.cuts.possible.m3u8, plan.m3u8.cuts.m3u8), no option naming them. Added, never in place of
    FILE's suffix: a name is then never FILE's own (a FILE named plan.m3u8 would be), nor
    another FILE's cut list, nor the other playlist's; that it is neither the source nor the
    output is checked with FILE's own place, before the first pass (cli._plan_file)."""
    return plan.parent / f"{plan.name}.{which.name}{PLAYLIST_SUFFIX}"


def _entries(index: FrameIndex, cuts: Sequence[Cut]) -> list[tuple[Cut, Fraction, int, int]]:
    """An entry per cut, in frame order: the cut, its time (time_of), and where the entry
    starts and stops, in milliseconds from the source's start: BEFORE seconds before the cut's
    time as the cut list writes it, rounded up to the millisecond (timecode), never before 0;
    AFTER seconds after it, never past the last frame's time. None for a cut whose frame has
    no time, which the list writes "-"."""
    last = time_of(index, index.frames - 1)
    entries: list[tuple[Cut, Fraction, int, int]] = []
    for cut in sorted(cuts, key=lambda cut: cut.frame):
        seconds = time_of(index, cut.frame)
        if seconds is None:
            continue
        at = math.ceil(seconds * 1000)
        stop = at + AFTER * 1000
        if last is not None:
            stop = min(stop, math.ceil(last * 1000))
        entries.append((cut, seconds, max(0, at - BEFORE * 1000), stop))
    return entries


def playlist(
    index: FrameIndex, cuts: Sequence[Cut], source_name: str, which: Playlist = OF_POSSIBLE
) -> str:
    """One of the playlists for VLC (DESIGN.md, Shot detection), the possible cuts' or the
    detected cuts' (`which`, `cuts` its own): an extended M3U, an entry per cut in frame order
    (_entries), which VLC plays one after the other, each from a second before its cut, so that
    checking them takes no time typed.

    An entry is its title, `possible cut <frame> at <time> (<probability>)`, or `cut <frame> at
    <time> (<probability>)` for a detected one, the frame, time and probability of its line in
    the cut list (cut_list), which VLC shows in its playlist and its window's title; VLC's own
    options for its start and its stop, in seconds to the millisecond (VLC reads them as 32-bit
    floats, src/input/input.c:943-946 at 3.0.24: to the millisecond up to 4.5 hours, 2 ms up to
    9); and the source, by `source_name`, its file name alone, which VLC looks for beside the
    playlist, wherever it is started from: so the playlist plays from the source's directory, on
    Windows as on Linux, where a path of the machine seedvr2x ran on names nothing on another.
    PROVISIONAL (implementation, 2026-10-10, a question for design: the title's words and the
    name alone, DESIGN.md saying neither).

    VLC reads an entry as a URL relative to the playlist's own (modules/demux/playlist/
    playlist.c:141-202), so the name is written as one, every byte but RFC 3986's unreserved
    ones percent-encoded. Measured in VLC 3.0.24, the playlist opened on Windows: a file name
    alone opens the file beside the playlist, and nothing from another directory;
    percent-encoded, 13 odd names of 13 open (spaces, an apostrophe, #, %, brackets, non-ASCII
    letters, a leading space), where written raw those holding #, %20 or a leading space open
    nothing; a Linux absolute path opens nothing, a Windows absolute path or a file URL opens;
    and .m3u8 is read as UTF-8, a Latin-1 name in it lost. A stop
    at or before its start, which VLC ignores, playing on to the file's end (input.c:970-974), is
    left out: none comes of a source whose times go up from 0. The comments say what the file
    is, and which of the two; none begins with a tag VLC reads (EXTINF:, EXTVLCOPT:, PLAYLIST:,
    EXTALBUMARTURL:)."""
    lines = [
        "#EXTM3U",
        f"# seedvr2x: the {which.cut}s of the cut list beside this file, to check in VLC, which",
        f"# plays an entry per {which.title}, from {BEFORE} s before its time to {AFTER} s after.",
        "# It names the source by its file name alone: it plays from the source's directory.",
    ]
    for cut, seconds, start, stop in _entries(index, cuts):
        peak = "" if cut.probability is None else f" ({probability(cut.probability)})"
        lines.append(f"#EXTINF:-1,{which.title} {cut.frame} at {timecode(seconds)}{peak}")
        lines.append(f"#EXTVLCOPT:start-time={start // 1000}.{start % 1000:03d}")
        if stop > start:
            lines.append(f"#EXTVLCOPT:stop-time={stop // 1000}.{stop % 1000:03d}")
        # The name's own bytes, those of a name that isn't UTF-8 too (os.fsencode).
        lines.append(quote(os.fsencode(source_name), safe=""))
    return "\n".join(lines) + "\n"


def write_playlist(
    plan: Path, index: FrameIndex, cuts: Sequence[Cut], source: Path, which: Playlist = OF_POSSIBLE
) -> bool:
    """One of --plan FILE's playlists (playlist), written beside FILE (playlist_path) as FILE
    is: whole, an earlier one replaced (files.write_whole); and said, with what its user does:
    open it in VLC, from the source's directory, where it is moved first when FILE isn't there.
    False when it can't be written, said, the cut list being written.

    No playlist, and True, without a cut of its own, a cut list's plan among them: nothing was
    to be written. A file at its place is then left as it is, and said: PROVISIONAL
    (implementation, 2026-10-10, a question for design: left or removed), an earlier plan's
    playlist there listing cuts this plan doesn't have, but seedvr2x removes no file it didn't
    write in this run, nor reads this one to tell whose it is."""
    path = playlist_path(plan, which)
    count = len(_entries(index, cuts))
    if not count:
        if path.exists() or path.is_symlink():
            logger.warning(
                "%s: left as it is: this plan has no %s, so no playlist of them; a playlist"
                " there is an earlier plan's, its entries not this plan's",
                path,
                which.cut,
            )
        return True
    try:
        files.write_whole(path, playlist(index, cuts, source.name, which).encode())
    except OSError as error:
        logger.error("--plan %s: its playlist, %s, not written: %s", plan, path, error)
        return False
    said = (
        f"a playlist of the {count} {which.cut}{'' if count == 1 else 's'} for VLC, an entry"
        f" each, from {BEFORE} second before its cut to {AFTER} seconds after"
    )
    if path.parent.resolve() == source.parent.resolve():
        logger.info(
            "%s: %s; beside the source, which it names by its file name alone: open it in VLC",
            path,
            said,
        )
    else:
        logger.info(
            "%s: %s; it names the source by its file name alone, %s: move it beside the source,"
            " then open it in VLC",
            path,
            said,
            source.name,
        )
    return True


def write_playlists(
    plan: Path, index: FrameIndex, detected: Sequence[Cut], possible: Sequence[Cut], source: Path
) -> bool:
    """--plan FILE's two playlists, the detected cuts' then the possible cuts' (PLAYLISTS), each
    written and said on its own (write_playlist), the second whatever became of the first; False
    when either can't be written. `detected` is empty for a cut list's plan, whose cuts no
    detection found, as its possible cuts are: no playlist at all."""
    written = [
        write_playlist(plan, index, cuts, source, which)
        for which, cuts in zip(PLAYLISTS, (detected, possible), strict=True)
    ]
    return all(written)
