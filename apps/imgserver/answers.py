"""What a request asks and what the server answers in words (DESIGN.md, The image server, A
request): the settings in its Seedvr2x- headers, the answer's format by Accept, and the server's
own words for a refused picture, the library's reason with a way out of the server's, with no
file name and no command (the user, 2026-11-03: Pillow alone, no ffmpeg, in production). Plain
Python, no torch."""

from collections.abc import Iterable
from dataclasses import dataclass

from seedvr2x.errors import Refusal
from seedvr2x.passes import ENCODE_SEED_OFFSET

RESOLUTION = "seedvr2x-resolution"
SEED = "seedvr2x-seed"
SETTINGS = (RESOLUTION, SEED)
PREFIX = "seedvr2x-"

# The seeds taken: numz seeds NumPy's generator with the seed plus ENCODE_SEED_OFFSET before the
# encode (model.set_seed; numz src/core/generation_phases.py:329 at 4490bd1), and NumPy takes
# seeds under 2^32, so a seed is at most 2^32 - 1 - ENCODE_SEED_OFFSET, as the CLI checks it
# (apps/seedvr2x/job.py, SEED_LIMIT).
SEED_MAX = 2**32 - 1 - ENCODE_SEED_OFFSET
# The longest number a setting is read from: a resolution past 8 digits is past any card.
DIGITS = 10

PNG = "image/png"
WEBP = "image/webp"


class BadRequest(Exception):
    """A request's header wrong or unknown: answered 400 with the message."""


class NotAcceptable(Exception):
    """An Accept the server can't meet: answered 406 with the message."""


@dataclass(frozen=True)
class Settings:
    """What a request sets: the output's short side, in pixels, and its seed, None for one drawn
    at random."""

    resolution: int
    seed: int | None


def _number(name: str, value: str) -> int:
    text = value.strip()
    if not text or len(text) > DIGITS or not text.isascii() or not text.isdigit():
        raise BadRequest(f"{name}: {value!r} is not a whole number of at most {DIGITS} digits")
    return int(text)


def settings(headers: Iterable[tuple[str, str]]) -> Settings:
    """The settings of a request's headers, (name, value) pairs as received: Seedvr2x-Resolution,
    required, a positive whole number; Seedvr2x-Seed, optional, from 0 to SEED_MAX. BadRequest for
    a setting missing, given twice or out of its range, and for any other Seedvr2x- header, never
    ignored."""
    found: dict[str, str] = {}
    for name, value in headers:
        key = name.lower()
        if not key.startswith(PREFIX):
            continue
        if key not in SETTINGS:
            raise BadRequest(
                f"{name}: not a header the server takes; it takes Seedvr2x-Resolution and"
                " Seedvr2x-Seed, everything else being set at its start"
            )
        if key in found:
            raise BadRequest(f"{name}: given twice")
        found[key] = value
    if RESOLUTION not in found:
        raise BadRequest("Seedvr2x-Resolution: missing; it is the output's short side, in pixels")
    resolution = _number("Seedvr2x-Resolution", found[RESOLUTION])
    if resolution < 1:
        raise BadRequest("Seedvr2x-Resolution: 0; it is the output's short side, in pixels")
    seed = None
    if SEED in found:
        seed = _number("Seedvr2x-Seed", found[SEED])
        if seed > SEED_MAX:
            raise BadRequest(f"Seedvr2x-Seed: {seed}, over the largest seed, {SEED_MAX}")
    return Settings(resolution, seed)


@dataclass(frozen=True)
class Formats:
    """The answer's format by the request's Accept: `first` the one to write, `fallback` the one
    to write where `first` can't hold the picture (PNG where the client takes it), None where
    there is none."""

    first: str
    fallback: str | None


def _ranges(accept: str) -> list[tuple[str, float]]:
    """An Accept header's media ranges with their weights (RFC 9110, 12.5.1), lower-case; a
    weight that isn't a number from 0 to 1 is BadRequest."""
    ranges: list[tuple[str, float]] = []
    for part in accept.split(","):
        pieces = [piece.strip() for piece in part.split(";")]
        media = pieces[0].lower()
        if not media:
            continue
        weight = 1.0
        for parameter in pieces[1:]:
            key, _, value = parameter.partition("=")
            if key.strip().lower() == "q":
                try:
                    weight = float(value.strip())
                except ValueError:
                    raise BadRequest(f"Accept: a weight of {value.strip()!r}") from None
                if not 0.0 <= weight <= 1.0:
                    raise BadRequest(f"Accept: a weight of {value.strip()!r}, not from 0 to 1")
        ranges.append((media, weight))
    return ranges


def _weight(ranges: list[tuple[str, float]], media: str) -> tuple[float, bool]:
    """A format's weight under the ranges, the most specific range that matches it deciding, and
    whether it is named itself rather than by a wildcard; (0.0, False) where none matches."""
    kind = media.split("/")[0]
    for candidate, named in ((media, True), (f"{kind}/*", False), ("*/*", False)):
        weights = [weight for each, weight in ranges if each == candidate]
        if weights:
            return max(weights), named
    return 0.0, False


def formats(accept: str | None) -> Formats:
    """The answer's formats for an Accept header (None: absent): PNG by default, lossless WebP
    where it is asked for by name with no lower weight than PNG's; PNG the fallback where the
    client takes it. NotAcceptable where it takes neither."""
    if accept is None or not accept.strip():
        return Formats(PNG, None)
    ranges = _ranges(accept)
    png, _ = _weight(ranges, PNG)
    webp, named = _weight(ranges, WEBP)
    if webp > 0 and ((named and webp >= png) or png == 0):
        return Formats(WEBP, PNG if png > 0 else None)
    if png > 0:
        return Formats(PNG, None)
    raise NotAcceptable(
        f"Accept: {accept.strip()}: the server answers a PNG ({PNG}) or a lossless WebP ({WEBP})"
    )


# The server's way out for each kind of the library's refusals (seedvr2x.picture.KINDS, and
# frames.check_target's), with no file name and no command: a client sends another picture.
WAYS = {
    "raw": "Develop it first in a raw developer, save it as a 16-bit PNG, and send the PNG.",
    "too-large": "Send a smaller copy of it.",
    "not-a-picture": "Send a PNG, JPEG, WebP, TIFF, BMP or GIF picture.",
    "damaged": "Send a good copy of it, or save it again from the program that made it.",
    "hdr": "Tone-map it to SDR first, and send the result.",
    "frames": "Send one frame of it as a picture of its own: seedvr2x's command line upscales a"
    " video.",
    "tiff16": "Send it as a 16-bit PNG, which keeps its colour profile.",
    "transparent": "Flatten it onto a background first, and send the result.",
    "colour-mode": "Convert it to RGB first, keeping its colour profile, and send the result.",
    "too-small": "Ask for a larger Seedvr2x-Resolution.",
}


def refused(error: Refusal) -> str:
    """A library refusal in the server's words: its reason and the server's way out by its kind;
    its message where it carries no reason (a refusal the library hasn't split), which names no
    file, the server reading bytes under the name "the picture"."""
    if not error.reason:
        return str(error)
    way = WAYS.get(error.kind, "")
    return f"{_sentence(error.reason)} {way}".strip()


def _sentence(text: str) -> str:
    text = text.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."
