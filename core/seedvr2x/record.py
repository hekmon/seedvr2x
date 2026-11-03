"""seedvr2x's record in each output (DESIGN.md, The record in each output): what made it, and
nothing of the source's but its size. seedvr2x's version and repository's address, the source's
and the output's sizes, the DiT's and the VAE's files with their SHA-256, the seed, the colour
correction, the encode's and the decode's tiles, the attention kernel, and for a video its window
cap, --quality and --compile: what changes the output's bits, the GPU and the libraries' versions
aside. With them, IPTC's digital source type as the AI mark. Always written, no option.

One record, two forms: an XMP packet for a picture (png.py's iTXt chunk, webp.py's XMP chunk) and
stream tags for a video (the CLI's FFV1 writer), the same fields in the same order. The writing
app gives its own distribution's version: the CLI's `seedvr2x`, the image server's its own. No date
or time, nor anything of the machine: the bytes of a record are a function of its fields."""

import re
from dataclasses import dataclass
from xml.sax.saxutils import escape

from seedvr2x.peaks import FLASH_ATTN_2, SDPA

# The repository's address, in the creator's line and the video's seedvr2x_url.
URL = "https://github.com/hekmon/seedvr2x"

# IPTC's digital source type for "augmentation, correction or enhancement using a Generative AI
# model" (IPTC NewsCodes, digitalsourcetype): the field platforms read to label a picture so.
DIGITAL_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
)

# seedvr2x's own XMP namespace, under the repository's address, versioned: a field's meaning
# changed or removed is a new version of the URI, a field added is not.
NAMESPACE = f"{URL}/xmp/1.0/"
PREFIX = "seedvr2x"
# The standard namespaces the packet names: XMP's basic schema (CreatorTool) and the IPTC Photo
# Metadata Standard's extension (DigitalSourceType).
XMP_BASIC = "http://ns.adobe.com/xap/1.0/"
IPTC_EXT = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
# The packet wrapper's id, XMP's own constant (XMP Specification Part 1, 7.3.2).
PACKET_ID = "W5M0MpCehiHzreSzNTczkc9d"

# The video's stream tags' prefix, as sptenc names its own (sptenc_url, sptenc_version).
TAG_PREFIX = "seedvr2x_"

# The characters XML 1.0 can't hold (its Char production), which a model file's name may: C0
# controls but tab, line feed and carriage return, lone surrogates (a name's undecodable bytes,
# Python's surrogateescape), and U+FFFE and U+FFFF. Each is written as U+FFFD, in the XMP and in
# the tags alike; the SHA-256 beside the name identifies the file all the same.
_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_SHA256 = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True)
class Model:
    """A model file as the record holds it: the name it was given (--dit-model, --vae-model) and
    its SHA-256, in lower-case hex. The record writes the name's last part alone (fields): a path
    given for it would put the user's directories in every output, which holds nothing personal
    (DESIGN.md, The record in each output)."""

    name: str
    sha256: str


@dataclass(frozen=True)
class Video:
    """What a video's record holds besides a picture's: the cap on the DiT windows the plan chose,
    in latents (0: each shot in one window, uncapped), --quality's level (standard, enhanced or
    maximum) and whether the VAE was compiled (--compile)."""

    window: int
    quality: str
    compiled: bool


@dataclass(frozen=True)
class Record:
    """An output's record. Sizes are (width, height) in pixels: the source's as its file stores
    it, the output's as written (a picture's turned upright). Tiles are the VAE passes' tile
    sides, in output pixels, 0 for an untiled pass; their overlap is passes.OVERLAP, which no
    plan chooses. attention is peaks' name of the kernel the DiT ran (FLASH_ATTN_2 or SDPA).
    ValueError for a field out of its range."""

    version: str
    source: tuple[int, int]
    output: tuple[int, int]
    dit: Model
    vae: Model
    seed: int
    color_correction: str
    encode_tile: int
    decode_tile: int
    attention: str
    video: Video | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("a record's version is not empty")
        for what, size in (("source", self.source), ("output", self.output)):
            if len(size) != 2 or min(size) < 1:
                raise ValueError(f"a record's {what} size is two sides of 1 pixel or more: {size}")
        for model in (self.dit, self.vae):
            if _SHA256.fullmatch(model.sha256) is None:
                raise ValueError(f"{model.name}: not a SHA-256 in lower-case hex: {model.sha256!r}")
        if self.seed < 0 or self.encode_tile < 0 or self.decode_tile < 0:
            raise ValueError("a record's seed and tiles are 0 or more")
        if self.attention not in (FLASH_ATTN_2, SDPA):
            raise ValueError(f"attention {self.attention!r}: {FLASH_ATTN_2} or {SDPA}")
        if self.video is not None and self.video.window < 0:
            raise ValueError(f"a record's window cap is 0 or more: {self.video.window}")

    @property
    def creator(self) -> str:
        """What made the output, in one line: `seedvr2x VERSION (URL)`, XMP's CreatorTool and a
        PNG's Software text."""
        return f"seedvr2x {self.version} ({URL})"

    def fields(self) -> list[tuple[str, str]]:
        """The record's fields but the version, the address and the mark, as (name, value), in
        their order: the names are the video's tags' less TAG_PREFIX, and in the XMP the
        namespace's properties (seedvr2x:source_size). Sizes as WIDTHxHEIGHT, numbers in
        decimal, --compile as true or false."""

        def size(sides: tuple[int, int]) -> str:
            return f"{sides[0]}x{sides[1]}"

        found = [
            ("source_size", size(self.source)),
            ("output_size", size(self.output)),
            ("dit_model", _file_name(self.dit.name)),
            ("dit_sha256", self.dit.sha256),
            ("vae_model", _file_name(self.vae.name)),
            ("vae_sha256", self.vae.sha256),
            ("seed", str(self.seed)),
            ("color_correction", self.color_correction),
            ("encode_tile", str(self.encode_tile)),
            ("decode_tile", str(self.decode_tile)),
            ("attention", self.attention),
        ]
        if self.video is not None:
            found += [
                ("window", str(self.video.window)),
                ("quality", self.video.quality),
                ("compile", "true" if self.video.compiled else "false"),
            ]
        return found

    def xmp(self) -> bytes:
        """The record as an XMP packet, UTF-8: xmp:CreatorTool (creator), the mark in
        Iptc4xmpExt:DigitalSourceType, and the fields under seedvr2x's namespace (NAMESPACE),
        each a simple property in element form. A read-only packet ("r"), with no padding."""
        properties = [
            f"   <xmp:CreatorTool>{_text(self.creator)}</xmp:CreatorTool>",
            "   <Iptc4xmpExt:DigitalSourceType>"
            f"{DIGITAL_SOURCE_TYPE}"
            "</Iptc4xmpExt:DigitalSourceType>",
            *(
                f"   <{PREFIX}:{name}>{_text(value)}</{PREFIX}:{name}>"
                for name, value in self.fields()
            ),
        ]
        lines = [
            f'<?xpacket begin="\ufeff" id="{PACKET_ID}"?>',
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
            f' <rdf:RDF xmlns:rdf="{RDF}">',
            '  <rdf:Description rdf:about=""',
            f'    xmlns:xmp="{XMP_BASIC}"',
            f'    xmlns:Iptc4xmpExt="{IPTC_EXT}"',
            f'    xmlns:{PREFIX}="{NAMESPACE}">',
            *properties,
            "  </rdf:Description>",
            " </rdf:RDF>",
            "</x:xmpmeta>",
            '<?xpacket end="r"?>',
        ]
        return "\n".join(lines).encode()

    def tags(self) -> list[tuple[str, str]]:
        """The record as a video stream's tags, (name, value) in order: seedvr2x_url,
        seedvr2x_version, seedvr2x_digital_source_type (the mark), then each field prefixed
        (TAG_PREFIX). Matroska's muxer writes the names in upper case."""
        head = [
            ("url", URL),
            ("version", self.version),
            ("digital_source_type", DIGITAL_SOURCE_TYPE),
        ]
        return [(TAG_PREFIX + name, value) for name, value in head + self.fields()]


def tag_names() -> frozenset[str]:
    """Every tag name a video's record may hold (Record.tags), a video's fields included: what
    the CLI's assembly keeps of the first segment's tags."""
    model = Model("", "0" * 64)
    sample = Record("0", (1, 1), (1, 1), model, model, 0, "", 0, 0, SDPA, Video(0, "", False))
    return frozenset(name for name, _ in sample.tags())


def _file_name(name: str) -> str:
    """A model's name as the record writes it: its last part, after any "/" (a path's file name),
    held to what XML can hold (_held)."""
    return _held(name.rsplit("/", 1)[-1])


def _held(text: str) -> str:
    """text with each character XML 1.0 can't hold written as U+FFFD (_NOT_XML)."""
    return _NOT_XML.sub("\ufffd", text)


def _text(value: str) -> str:
    """value as an element's text: &, < and > escaped, and a carriage return as a character
    reference, which a parser would otherwise read as a line feed (XML 1.0, 2.11)."""
    return escape(value, {"\r": "&#13;"})
