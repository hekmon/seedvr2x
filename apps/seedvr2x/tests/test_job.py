"""A job's layout: cut lists, shots and their seeds."""

from codecs import BOM_UTF8
from pathlib import Path

import pytest

from job import read_cuts, shots_from_cuts
from seedvr2x.errors import JobError
from seedvr2x.shots import Shot


def test_cut_list(tmp_path: Path) -> None:
    path = tmp_path / "cuts.txt"
    path.write_text("# shots of the trailer\n120\n\n 250  # a flash before\n300\n")
    assert read_cuts(path) == ([120, 250, 300], {})
    path.write_text("120\n12.5\n")
    with pytest.raises(JobError, match=r"cuts.txt:2: '12.5' is not a frame number"):
        read_cuts(path)
    # Numbers and times after the frame number are ignored (DESIGN.md, Input): scores, say, and
    # the times --plan FILE writes.
    path.write_text("120 13.7\n250\t4.2 0:00:10.427 # doubtful\n300 -1e-3 .5 \n")
    assert read_cuts(path) == ([120, 250, 300], {})
    path.write_text("00:00:05.005 120\n")
    with pytest.raises(JobError, match=r"cuts.txt:1: '00:00:05.005' is not a frame number"):
        read_cuts(path)
    with pytest.raises(JobError, match=r"missing\.txt"):
        read_cuts(tmp_path / "missing.txt")


def test_a_level_after_the_frame_marks_its_shot(tmp_path: Path) -> None:
    # DESIGN.md, A level per shot, marked in the cut list (decided on 2026-11-24): a level's name
    # after the frame number runs the shot from that cut at that level, the numbers and the time
    # around it ignored; a line `0 LEVEL` marks the first shot; each of the four levels may be.
    path = tmp_path / "cuts.txt"
    path.write_text(
        "# marked\n0 maximum\n120 0.598 0:00:05.005 deghost\n250\n300 enhanced # fast\n"
        "400 standard 0.31\n"
    )
    assert read_cuts(path) == (
        [120, 250, 300, 400],
        {0: "maximum", 120: "deghost", 300: "enhanced", 400: "standard"},
    )
    # The first shot alone marked, and none: no cut, the marks only the first's.
    path.write_text("0 deghost\n")
    assert read_cuts(path) == ([], {0: "deghost"})


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        # A word that is no level, misspelt, in another case, or a level's prefix.
        ("120\n250 deghots\n", r"cuts.txt:2: 'deghots' is neither a level \(standard,"),
        ("250 Deghost\n", r"cuts.txt:1: 'Deghost' is neither a level"),
        ("250 max\n", r"cuts.txt:1: 'max' is neither a level"),
        ("250 flash\n", r"cuts.txt:1: 'flash' is neither a level .* nor a number or a time"),
        # Two levels on a line, and the first shot marked twice.
        ("250 deghost maximum\n", r"cuts.txt:1: two levels on one line, deghost and maximum"),
        ("0 deghost\n0 maximum\n", r"cuts.txt:2: the first shot marked again, deghost before"),
    ],
)
def test_a_word_that_is_no_level_is_refused_naming_its_line(
    tmp_path: Path, text: str, refusal: str
) -> None:
    path = tmp_path / "cuts.txt"
    path.write_text(text)
    with pytest.raises(JobError, match=refusal):
        read_cuts(path)


def test_a_cut_list_is_read_as_utf_8_its_byte_order_mark_taken(tmp_path: Path) -> None:
    # A Windows editor's byte order mark at the start, which the locale's codec read there as
    # part of the first frame number ('ï»¿215'), and a comment out of ASCII: read as UTF-8 on
    # every platform, as --plan FILE writes the list.
    path = tmp_path / "cuts.txt"
    path.write_bytes("\ufeff215 deghost\n# plans d'été\n300\n".encode())
    assert read_cuts(path) == ([215, 300], {215: "deghost"})
    path.write_bytes("215 # été\n".encode())
    assert read_cuts(path) == ([215], {})


def test_a_cut_list_that_isnt_utf_8_is_refused_naming_it(tmp_path: Path) -> None:
    # Latin-1's é, no UTF-8, with a byte order mark and without: a cut list that can't be read,
    # the file and the line named, no traceback.
    path = tmp_path / "cuts.txt"
    for start in (b"", BOM_UTF8):
        path.write_bytes(start + "215\n300 # été\n".encode("latin-1"))
        with pytest.raises(
            JobError, match=r"cuts\.txt:2: a cut list that can't be read: not UTF-8 text \("
        ):
            read_cuts(path)


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("120\n120\n", r"cuts.txt:2: frame 120 cut again, line 1 cutting there already: each"),
        ("120 deghost\n250\n120 maximum\n", r"cuts.txt:3: frame 120 cut again, line 1 cutting"),
        ("120 0:00:05.005\n\n# again\n120\n", r"cuts.txt:4: frame 120 cut again, line 1"),
    ],
)
def test_a_frame_cut_twice_is_refused_naming_both_lines(
    tmp_path: Path, text: str, refusal: str
) -> None:
    # read_cuts has the lines; check_cuts' "must increase" stays for cuts out of order.
    path = tmp_path / "cuts.txt"
    path.write_text(text)
    with pytest.raises(JobError, match=refusal):
        read_cuts(path)


def test_a_cut_at_0_alone_stays_refused_saying_how_to_mark_the_first_shot(tmp_path: Path) -> None:
    path = tmp_path / "cuts.txt"
    path.write_text("0\n120\n")
    cuts, marks = read_cuts(path)
    assert cuts == [0, 120] and marks == {}
    with pytest.raises(
        JobError,
        match=r"a cut at frame 0: the first shot starts there without one; a line `0 LEVEL`, a"
        r" level's name after the 0 \(standard, enhanced, maximum or deghost\), marks the first"
        r" shot's level",
    ):
        shots_from_cuts(cuts, 500)
    # A score after it is no level: still a cut at 0.
    path.write_text("0 0.9\n")
    assert read_cuts(path) == ([0], {})


def test_shots_from_cuts() -> None:
    assert shots_from_cuts([], 45) == [Shot(0, 45)]
    shots = shots_from_cuts([10, 11, 30], 45)
    assert shots == [Shot(0, 10), Shot(10, 11), Shot(11, 30), Shot(30, 45)]
    assert [shot.frames for shot in shots] == [10, 1, 19, 15]
    for cuts, message in (
        ([0, 10], "frame 0"),
        ([10, 10], "must increase"),
        ([20, 10], "must increase"),
        ([10, 45], "has 45 frames"),
    ):
        with pytest.raises(JobError, match=message):
            shots_from_cuts(cuts, 45)


def test_seed_per_shot() -> None:
    # The plain seed at frame 0, as numz's single batch; then the seed plus the first frame.
    assert [shot.seed(42) for shot in shots_from_cuts([10, 11], 20)] == [42, 52, 53]
