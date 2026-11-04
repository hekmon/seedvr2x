"""What a request asks and what the server answers in words (answers.py; DESIGN.md, The image
server, A request): the settings in the Seedvr2x- headers (required, optional, unknown and
repeated ones refused, numbers digits only and bounded), the answer's format by Accept (PNG by
default, WebP where asked for by name, PNG kept as the fallback where the client takes it, 406
where it takes neither), and the server's words for a refused picture, held on refusals the
library really raises, read from bytes under a name the words must not repeat."""

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

import answers
from answers import BadRequest, NotAcceptable
from seedvr2x import frames, picture
from seedvr2x.errors import Refusal
from seedvr2x.passes import ENCODE_SEED_OFFSET

Pairs = list[tuple[str, str]]


def headers(**given: str) -> Pairs:
    return [(name.replace("_", "-"), value) for name, value in given.items()]


# ---- Settings ------------------------------------------------------------------------------------


def test_the_largest_seed_is_the_one_numpy_takes_after_the_offset() -> None:
    # numz seeds NumPy's generator with the seed plus ENCODE_SEED_OFFSET (1,000,000), and NumPy
    # takes seeds under 2^32.
    assert ENCODE_SEED_OFFSET == 1_000_000
    assert answers.SEED_MAX == 2**32 - 1 - 1_000_000
    np.random.default_rng(answers.SEED_MAX + ENCODE_SEED_OFFSET)


def test_a_resolution_alone_is_a_request() -> None:
    assert answers.settings([("Seedvr2x-Resolution", "1080")]) == answers.Settings(1080, None)


def test_a_seed_is_taken_from_zero_to_the_largest() -> None:
    for seed in (0, 1, 12345, answers.SEED_MAX):
        got = answers.settings([("Seedvr2x-Resolution", "64"), ("Seedvr2x-Seed", str(seed))])
        assert got == answers.Settings(64, seed)


def test_a_seed_over_the_largest_is_refused_with_the_largest() -> None:
    with pytest.raises(BadRequest, match=str(answers.SEED_MAX)):
        answers.settings(
            [("Seedvr2x-Resolution", "64"), ("Seedvr2x-Seed", str(answers.SEED_MAX + 1))]
        )


def test_the_names_are_read_in_any_case() -> None:
    got = answers.settings([("SEEDVR2X-RESOLUTION", "720"), ("seedvr2x-seed", "3")])
    assert got == answers.Settings(720, 3)


def test_other_headers_are_ignored() -> None:
    got = answers.settings(
        [
            ("Host", "localhost"),
            ("Accept", "image/png"),
            ("Content-Type", "image/png"),
            ("X-Seedvr2x-Resolution", "5"),
            ("Seedvr2x-Resolution", "720"),
        ]
    )
    assert got == answers.Settings(720, None)


def test_the_resolution_is_required() -> None:
    with pytest.raises(BadRequest, match="Seedvr2x-Resolution: missing"):
        answers.settings([("Seedvr2x-Seed", "1")])
    with pytest.raises(BadRequest, match="missing"):
        answers.settings([])


def test_a_resolution_of_zero_is_refused() -> None:
    with pytest.raises(BadRequest, match="Seedvr2x-Resolution"):
        answers.settings([("Seedvr2x-Resolution", "0")])
    with pytest.raises(BadRequest, match="Seedvr2x-Resolution"):
        answers.settings([("Seedvr2x-Resolution", "0000")])


@pytest.mark.parametrize(
    "value",
    [
        "-1",
        "+5",
        "1 0",
        "1_000",
        "1.5",
        "1e3",
        "0x10",
        "abc",
        "",
        " ",
        "١٢",  # Arabic-Indic digits: str.isdigit() is true, int() reads them
        "²",  # a superscript digit: isdigit() true, int() raises
        "1" * (answers.DIGITS + 1),
    ],
)
@pytest.mark.parametrize("name", ["Seedvr2x-Resolution", "Seedvr2x-Seed"])
def test_a_number_is_digits_only(name: str, value: str) -> None:
    given = [("Seedvr2x-Resolution", "64")]
    if name == "Seedvr2x-Seed":
        given.append((name, value))
    else:
        given = [(name, value)]
    with pytest.raises(BadRequest, match=name):
        answers.settings(given)


def test_a_number_may_have_the_spaces_a_header_value_is_trimmed_of() -> None:
    # HTTP strips optional whitespace around a value; spaces inside a number are refused above.
    assert answers.settings([("Seedvr2x-Resolution", " 64 ")]).resolution == 64


def test_the_longest_number_is_taken() -> None:
    assert answers.settings([("Seedvr2x-Resolution", "9" * answers.DIGITS)]).resolution == (
        10**answers.DIGITS - 1
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Seedvr2x-Resolution", "Seedvr2x-Resolution"),
        ("Seedvr2x-Resolution", "seedvr2x-RESOLUTION"),
        ("Seedvr2x-Seed", "Seedvr2x-Seed"),
        ("Seedvr2x-Seed", "SEEDVR2X-SEED"),
    ],
)
def test_a_header_given_twice_is_refused(first: str, second: str) -> None:
    # Even with the same value, and whatever the case of the name.
    given = [("Seedvr2x-Resolution", "64"), (first, "1"), (second, "1")]
    if first == "Seedvr2x-Resolution":
        given = [(first, "64"), (second, "64")]
    with pytest.raises(BadRequest, match="twice"):
        answers.settings(given)


@pytest.mark.parametrize(
    "name", ["Seedvr2x-Colour", "Seedvr2x-Output-Size", "seedvr2x-seeds", "Seedvr2x-", "SEEDVR2X-X"]
)
def test_any_other_seedvr2x_header_is_refused_naming_it(name: str) -> None:
    with pytest.raises(BadRequest) as error:
        answers.settings([("Seedvr2x-Resolution", "64"), (name, "1")])
    assert name in str(error.value)


# ---- Accept --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("accept", "first", "fallback"),
    [
        (None, answers.PNG, None),
        ("", answers.PNG, None),
        ("   ", answers.PNG, None),
        ("image/png", answers.PNG, None),
        ("image/webp", answers.WEBP, None),
        ("image/webp, image/png", answers.WEBP, answers.PNG),
        ("image/png, image/webp", answers.WEBP, answers.PNG),
        ("image/avif,image/webp,*/*;q=0.8", answers.WEBP, answers.PNG),
        (
            "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            answers.WEBP,
            answers.PNG,
        ),
        ("*/*", answers.PNG, None),
        ("image/*", answers.PNG, None),
        ("image/png;q=0, image/webp", answers.WEBP, None),
        ("image/png;q=0.5, image/webp;q=0.9", answers.WEBP, answers.PNG),
        ("image/png;q=0.9, image/webp;q=0.5", answers.PNG, None),
        ("image/webp;q=0, image/png", answers.PNG, None),
        ("image/webp;q=0, */*", answers.PNG, None),
        ("IMAGE/WebP, Image/PNG", answers.WEBP, answers.PNG),
        ("Image/Webp;Q=0.5", answers.WEBP, None),
        ("image/webp ; q=1 , image/png ; q=1", answers.WEBP, answers.PNG),
    ],
)
def test_the_format_by_accept(accept: str | None, first: str, fallback: str | None) -> None:
    assert answers.formats(accept) == answers.Formats(first, fallback)


@pytest.mark.parametrize(
    "accept",
    [
        "image/jpeg",
        "text/html",
        "image/webp;q=0, image/png;q=0",
        "image/png;q=0",
        "image/webp;q=0",
        "image/*;q=0",
        "*/*;q=0",
    ],
)
def test_an_accept_the_server_cannot_meet_is_not_acceptable(accept: str) -> None:
    with pytest.raises(NotAcceptable) as error:
        answers.formats(accept)
    assert answers.PNG in str(error.value)
    assert answers.WEBP in str(error.value)


@pytest.mark.parametrize("accept", ["image/png;q=2", "image/png;q=x", "image/webp;q=-1", "*/*;q="])
def test_a_weight_that_is_no_number_from_0_to_1_is_a_bad_request(accept: str) -> None:
    with pytest.raises(BadRequest, match="weight"):
        answers.formats(accept)


def test_a_specific_range_beats_a_wildcard() -> None:
    # RFC 9110, 12.5.1: image/png;q=0 stands against image/*.
    with pytest.raises(NotAcceptable):
        answers.formats("image/png;q=0, image/webp;q=0, image/*")


# ---- A refused picture in the server's words -----------------------------------------------------

SECRET = "secret-name.png"


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def saved(img: Image.Image, fmt: str, **options: object) -> bytes:
    out = io.BytesIO()
    img.save(out, fmt, **options)
    return out.getvalue()


def noise(width: int = 64, height: int = 64) -> bytes:
    pixels = np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return saved(Image.fromarray(pixels), "PNG")


def refusals() -> dict[str, bytes]:
    """Real files for every kind of the library's refusals that Pillow can write, by kind."""
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[..., 3] = 100
    still = [Image.fromarray(np.full((16, 16, 3), 50 * each, np.uint8)) for each in range(3)]
    png = noise()
    made = {
        "transparent": saved(Image.fromarray(rgba), "PNG"),
        "frames": saved(still[0], "GIF", save_all=True, append_images=still[1:]),
        "not-a-picture": b"hello, this is no picture at all\n" * 5,
        "damaged": png[: len(png) // 2],
        "tiff16": saved(Image.fromarray(np.zeros((8, 8), np.uint16)), "TIFF"),
        "colour-mode": saved(Image.new("CMYK", (16, 16)), "JPEG"),
        # cICP says transfer 16: PQ, high dynamic range.
        "hdr": png[:33] + chunk(b"cICP", bytes([9, 16, 0, 1])) + png[33:],
        # A TIFF header with CR after it: Canon CR2.
        "raw": b"II*\x00" + struct.pack("<I", 16) + b"CR\x02\x00" + b"\0" * 40,
        "too-large": png,
    }
    return made


def read_refusal(kind: str, monkeypatch: pytest.MonkeyPatch) -> Refusal:
    if kind == "too-large":
        # A 64x64 picture over twice a limit of 100 pixels.
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(Refusal) as error:
        picture.read(refusals()[kind], SECRET)
    return error.value


def test_every_kind_the_library_refuses_with_has_a_way_out() -> None:
    assert set(picture.KINDS) | {"too-small"} <= set(answers.WAYS)
    assert set(answers.WAYS) == set(picture.KINDS) | {"too-small"}
    for way in answers.WAYS.values():
        assert way.endswith(".")


def test_the_refusals_made_here_cover_every_kind_the_library_has() -> None:
    assert set(refusals()) == set(picture.KINDS)


@pytest.mark.parametrize("kind", picture.KINDS)
def test_a_refused_picture_is_worded_for_a_client(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = read_refusal(kind, monkeypatch)
    assert error.kind == kind
    # The library's message names the file, and the way it holds may name ffmpeg: the server's
    # words are the reason and its own way out.
    words = answers.refused(error)
    assert "secret-name" not in words
    assert "ffmpeg" not in words.lower()
    assert "--" not in words
    assert words[0].isupper()
    assert words.endswith(".")
    assert error.reason
    assert error.reason[1:] in words
    assert answers.WAYS[kind] in words


def test_the_command_lines_message_is_not_the_servers() -> None:
    error = picture_refusal("transparent")
    assert "ffmpeg" in str(error)
    assert "ffmpeg" not in answers.refused(error)


def picture_refusal(kind: str) -> Refusal:
    monkeypatch = pytest.MonkeyPatch()
    try:
        return read_refusal(kind, monkeypatch)
    finally:
        monkeypatch.undo()


def test_a_picture_too_small_for_its_resolution_is_worded_too() -> None:
    with pytest.raises(Refusal) as error:
        frames.check_target((8, 8), 8)
    assert error.value.kind == "too-small"
    words = answers.refused(error.value)
    assert "--" not in words
    assert "ffmpeg" not in words
    assert words[0].isupper()
    assert words.endswith(".")
    assert "too small" in words
    assert answers.WAYS["too-small"] in words


def test_a_refusal_with_no_reason_is_worded_by_its_message() -> None:
    assert answers.refused(Refusal("something odd")) == "something odd"


def test_a_refusal_of_a_kind_with_no_way_is_its_reason_alone() -> None:
    words = answers.refused(Refusal("m", kind="new-kind", reason="the picture is odd"))
    assert words == "The picture is odd."


def test_a_reason_ending_in_a_stop_is_not_given_a_second() -> None:
    words = answers.refused(Refusal("m", kind="raw", reason="it is raw."))
    assert words.startswith("It is raw. ")
    assert ".." not in words


# ---- Accept-Post ---------------------------------------------------------------------------------


def test_accept_post_lists_the_formats_the_library_reads_in_order() -> None:
    assert answers.ACCEPT_POST == (
        "image/png, image/jpeg, image/webp, image/tiff, image/bmp, image/gif"
    )
    assert set(answers.MEDIA_TYPES) == set(picture.FORMATS)


def test_a_format_with_no_media_type_is_not_built() -> None:
    with pytest.raises(ValueError, match="AVIF"):
        answers.accept_post((*picture.FORMATS, "AVIF"))
