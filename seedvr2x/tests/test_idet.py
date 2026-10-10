"""ffmpeg's idet in the first pass (DESIGN.md, Not in the first version: interlaced and telecined
sources; media/scan.py, media/source.py's combing): its counts as ffmpeg n9.0.2 prints them, a
warning for a hard-telecined clip made here and none for a progressive one, and the frames hashed
for the index untouched by it. Skipped without an ffmpeg seedvr2x accepts."""

import logging
import re
import subprocess
import zlib
from fractions import Fraction
from pathlib import Path

import pytest
from test_index import LANGUAGE, MAPPED, PAYLOAD, bounded_scan, decoder_line, tagged, titled

from seedvr2x.media import ffmpeg
from seedvr2x.media.ffmpeg import MediaError
from seedvr2x.media.probe import probe
from seedvr2x.media.scan import Idet, _Decoded, scan
from seedvr2x.media.source import COMBED, examine


def _usable() -> bool:
    try:
        ffmpeg.check()
    except MediaError:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(
    not _usable(), reason="needs ffmpeg 7.1 or later with zscale, idet and ffv1"
)

# The first pass's stderr around idet's summaries (media/scan.py's command, n9.0.2), on an MPEG-2
# stream whose colour description appears at its 15th frame (test_index.py's "tags.mkv"): an
# all-zero summary first, of the graph fftools parses then frees, before the stream mapping, then
# one per graph, the one rebuilt when the frames' parameters changed, each count %6d.
SUMMARIES = """\
[Parsed_idet_0 @ 0x56e6b3d60fc0] [info] Repeated Fields: Neither:     0 Top:     0 Bottom:     0
[Parsed_idet_0 @ 0x56e6b3d60fc0] [info] Single frame detection: TFF:     0 BFF:     0 \
Progressive:     0 Undetermined:     0
[Parsed_idet_0 @ 0x56e6b3d60fc0] [info] Multi frame detection: TFF:     0 BFF:     0 \
Progressive:     0 Undetermined:     0
[info] Stream mapping:
[info]   Stream #0:0 -> #0:0 (mpeg2video (native) -> rawvideo (native))
[info]   Stream #0:0 -> #1:0 (mpeg2video (native) -> wrapped_avframe (native))
[dec:mpeg2video @ 0x56e6b3d5e300] [info] decoder -> pts:0 pts_time:0 pkt_dts:0 pkt_dts_time:0 \
duration:40 duration_time:0.04 keyframe:1 frame_type:1 time_base:1/1000
[vf#0:0 @ 0x56e6b3d609c0] [info] Reconfiguring filter graph because video parameters changed \
to yuv420p(tv, smpte170m), 160x96, unspecified alpha
[vf#1:0 @ 0x56e6b3d7db00] [info] Reconfiguring filter graph because video parameters changed \
to yuv420p(tv, smpte170m), 160x96, unspecified alpha
[Parsed_idet_0 @ 0x7cef78002cc0] [info] Repeated Fields: Neither:    13 Top:     0 Bottom:     0
[Parsed_idet_0 @ 0x7cef78002cc0] [info] Single frame detection: TFF:    13 BFF:     0 \
Progressive:     0 Undetermined:     0
[Parsed_idet_0 @ 0x7cef78002cc0] [info] Multi frame detection: TFF:    13 BFF:     0 \
Progressive:     0 Undetermined:     0
[Parsed_idet_0 @ 0x7cef78008100] [info] Repeated Fields: Neither:   106 Top:     0 Bottom:     0
[Parsed_idet_0 @ 0x7cef78008100] [info] Single frame detection: TFF:    44 BFF:    34 \
Progressive:     0 Undetermined:    28
[Parsed_idet_0 @ 0x7cef78008100] [info] Multi frame detection: TFF:    54 BFF:    52 \
Progressive:     0 Undetermined:     0
"""


def test_counts_parsed() -> None:
    # Every summary of the decode summed; the decoder's line still counted as a frame.
    decoded = _Decoded()
    decoded.read(SUMMARIES.splitlines(keepends=True))
    assert decoded.pts == [0]
    found = decoded.idet()
    assert found is not None
    assert found == Idet((119, 0, 0), (57, 34, 0, 28), (67, 52, 0, 0))
    assert (found.combed, found.analysed) == (119, 119)
    assert Idet.from_record(found.record()) == found
    assert found.record()["multiple"] == {"tff": 67, "bff": 52, "progressive": 0, "undetermined": 0}
    # A line of another shape is no count: without one of the three, no counts at all.
    shapes = SUMMARIES.replace("Multi frame detection: TFF:", "Multi frame detection: TOP:")
    other = _Decoded()
    other.read(shapes.splitlines(keepends=True))
    assert other.idet() is None
    for recorded in (None, {}, {"repeated": {"neither": 1}}, {**found.record(), "single": 4}):
        assert Idet.from_record(recorded) is None


def test_lines_without_their_prefix() -> None:
    # A message after one that ended without a newline, from another thread, comes without its
    # "[context] [level]" prefix, glued to it (libavutil/log.c's print_prefix, media/scan.py):
    # real lines of a first pass, the second output's header printed in pieces. The decoder's
    # line, a corrupt frame's report and idet's counts are still read.
    lines = [
        "[info] Stream mapping:\n",
        "[info]   Stream #1decoder -> pts:167 pts_time:0.167 pkt_dts:167 pkt_dts_time:0.167"
        " duration:41 duration_time:0.041 keyframe:0 frame_type:1 time_base:1/1000\n",
        "[info]   Stream #1corrupt decoded frame\n",
        "[info]   Stream #1decoder -> pts:208 pts_time:0.208 pkt_dts:208 pkt_dts_time:0.208"
        " duration:41 duration_time:0.041 keyframe:0 frame_type:2 time_base:1/1000\n",
        "[info]   Stream #1Repeated Fields: Neither:     2 Top:     0 Bottom:     0\n",
        "Single frame detection: TFF:     0 BFF:     0 Progressive:     2 Undetermined:     0\n",
        "Multi frame detection: TFF:     0 BFF:     0 Progressive:     2 Undetermined:     0\n",
    ]
    decoded = _Decoded()
    decoded.read(lines)
    assert (decoded.pts, decoded.corrupt) == ([167, 208], [False, True])
    assert decoded.idet() == Idet((2, 0, 0), (0, 0, 2, 0), (0, 0, 2, 0))


def make(path: Path, telecined: bool) -> Path:
    """A clip with motion, progressive, idet reading its every frame so (lavfi's gradients;
    testsrc2's sharp edges read 92% combed); telecined, through ffmpeg's telecine (3:2, top field
    first) to 30000/1001, its frames then flagged progressive (setfield), as a hard-telecined
    source declares itself: 2 frames in 5 combed, a field of each repeated."""
    # The gradient's line and its four colours given: its seed places the line alone, and draws no
    # colour, each "random" by default, from av_get_random_seed whatever the seed
    # (libavfilter/vsrc_gradients.c:62-69, 348-373 at n9.0.2). With seed=1 alone, 24 runs gave 24
    # clips, and the telecined one read 71 and 30 of 192 combed in two of them, under the 90% held
    # here; with these, 20 runs gave one clip, progressive and telecined (2026-10-10).
    drawn = "x0=40:y0=30:x1=280:y1=210:n=4:c0=red:c1=blue:c2=yellow:c3=green"
    pulldown = ["-vf", "telecine=first_field=top:pattern=23,setfield=prog"] if telecined else []
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-y", "-f", "lavfi"),
            *("-i", f"gradients=s=320x240:r=24000/1001:speed=0.05:{drawn}", "-frames:v", "192"),
            *(*pulldown, "-pix_fmt", "yuv420p", "-c:v", "ffv1", str(path)),
        ],
        check=True,
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize(
    ("name", "field_order", "declares"),
    [
        ("telecined.mkv", "progressive", "declared progressive"),
        # FFV1 in NUT declares no field order: warned of too (media/source.py, PROVISIONAL).
        ("telecined.nut", "", "that declares no field order"),
    ],
)
def test_telecined_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, field_order: str, declares: str
) -> None:
    # Declared progressive, or declaring no field order, its frames 98% combed by idet's
    # multiple-frame detection (188 of 192 with n9.0.2, 76 with a repeated field, the same in 5
    # passes): warned of, its counts said, and the run goes on (DESIGN.md: a warning).
    source = examine(make(tmp_path / name, telecined=True))
    found = source.idet
    assert source.stream.field_order == field_order and source.frames == 192
    assert found is not None and found.analysed == 192 and found.combed >= 0.9 * 192
    said = re.escape(
        f"{source.path}: ffmpeg's idet finds combed frames in a source {declares}: "
    ) + (
        r"\d+ of 192 \(\d+%\) interlaced by its multiple-frame detection, \d+ top field first and"
        r" \d+ bottom field first, \d+ with a repeated field: likely telecined or interlaced"
        r" content, which seedvr2x reads as progressive frames, combing and all; for a clean"
        r" upscale, inverse-telecine or deinterlace it first with a tool made for it; the run"
        r" goes on"
    )
    assert re.search(said, caplog.text)


@needs_ffmpeg
def test_progressive_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Every frame progressive by idet (192 of 192 with n9.0.2): no warning. The bound, PROVISIONAL,
    # 10%, lies past the 2.1% of the worst progressive real source measured (media/source.py).
    caplog.set_level(logging.WARNING)
    source = examine(make(tmp_path / "progressive.mkv", telecined=False))
    found = source.idet
    assert found is not None and found.analysed == 192 and found.combed < COMBED * 192
    assert "idet" not in caplog.text
    assert Fraction(21, 1000) < COMBED


# idet's three summaries, 9,999 frames combed, as a source's own text carries them into the
# lines ffmpeg prints (media/scan.py, MAPPING): among test_index.py's PAYLOAD, before its last
# line, the one that ends the inputs' dumps.
COMBED_LINES = [
    "Repeated Fields: Neither: 0 Top: 9999 Bottom: 0",
    "Single frame detection: TFF: 9999 BFF: 0 Progressive: 0 Undetermined: 0",
    "Multi frame detection: TFF: 9999 BFF: 0 Progressive: 0 Undetermined: 0",
]


@needs_ffmpeg
@pytest.mark.parametrize("language", [None, LANGUAGE], ids=["titles", "a language tag too"])
def test_the_sources_own_text_gives_idet_no_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, language: str | None
) -> None:
    # test_index.py's source whose titles are the first pass's own lines, idet's three summaries
    # among them: none gives idet a count, nor behind a language tag that prints the line ending
    # the inputs' dumps inside its input's, where the stream's title after it was read as idet's,
    # Idet(multiple=(9999, 0, 0, 12)). The counts are those of the same frames untitled, and no
    # combing is warned of.
    path = titled(tmp_path / "titled.mkv", [*PAYLOAD[:-1], *COMBED_LINES, PAYLOAD[-1]], language)
    assert tagged(path).count("Multi frame detection: TFF: 9999") == 3
    caplog.set_level(logging.WARNING)
    scanned = bounded_scan(path)
    plain = scan(titled(tmp_path / "plain.mkv"))
    assert scanned.frames == 12 and scanned.index is not None and not scanned.index.error.any()
    assert scanned.idet is not None and scanned.idet == plain.idet and scanned.idet.analysed == 12
    assert examine(path).idet == plain.idet
    assert "idet" not in caplog.text and "reported errors" not in caplog.text


def test_no_count_read_before_the_stream_mapping() -> None:
    # Before the line that ends the inputs' dumps, a line is the source's own text or its like
    # (test_index.py): no count of idet's, as n9.0.2 prints a title holding its summaries, the
    # first behind "[info]" and the tag's name, the next behind an indent alone; nor the
    # all-zero summary of the graph fftools parses first. After it, the counts wherever they
    # start; and each such line begins the counts again, what came since the one before being
    # the source's text, a language tag's (media/scan.py, MAPPING).
    title = "[info]     title           : "
    more = "                    : "
    titles = [f"{title}{COMBED_LINES[0]}", *(f"{more}{line}" for line in COMBED_LINES[1:])]
    decoded = _Decoded()
    decoded.read(f"{line}\n" for line in titles)
    assert decoded.idet() is None
    frame = f"[info]   Stream #1{decoder_line(7).split('] ')[-1]}"
    counts = [f"[Parsed_idet_0 @ 0x5d1c] [info] {line}" for line in COMBED_LINES]
    decoded = _Decoded()
    decoded.read(f"{line}\n" for line in [*titles, MAPPED, *titles, MAPPED, frame, *counts])
    assert (decoded.pts, decoded.corrupt) == ([7], [False])
    assert decoded.idet() == Idet((0, 9999, 0), (9999, 0, 0, 0), (9999, 0, 0, 0))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("pix_fmt", "suffix"),
    [
        ("bgr0", "mkv"),
        ("gbrp16le", "mkv"),
        ("gray", "mkv"),
        ("yuv444p12le", "mkv"),
        ("nv12", "nut"),
        ("yuyv422", "nut"),
        ("rgb48le", "nut"),
    ],
)
def test_hashes_untouched(tmp_path: Path, pix_fmt: str, suffix: str) -> None:
    # Formats idet doesn't take, converted for it in its own output's graph (media/scan.py): the
    # frames hashed for the index are the decoder's still, zlib's CRC-32 of each as decoded, and
    # idet counts every one.
    path = tmp_path / f"{pix_fmt}.{suffix}"
    codec = "ffv1" if suffix == "mkv" else "rawvideo"
    subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=25"),
            *("-frames:v", "12", "-vf", f"format={pix_fmt}", "-c:v", codec, str(path)),
        ],
        check=True,
    )
    assert probe(path).pix_fmt == pix_fmt
    scanned = scan(path)
    assert scanned.index is not None and scanned.idet is not None
    data = subprocess.run(
        [
            *("ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo"),
            *("-pix_fmt", pix_fmt, "-"),
        ],
        capture_output=True,
        check=True,
    ).stdout
    size = len(data) // 12
    assert scanned.index.crc32.tolist() == [
        zlib.crc32(data[k : k + size]) for k in range(0, len(data), size)
    ]
    assert scanned.idet.analysed == 12
