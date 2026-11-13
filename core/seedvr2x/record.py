"""seedvr2x's record in each output (DESIGN.md, The record in each output): what made it, and
nothing of the source's but its size. The app that made it, its version and seedvr2x's address, the
source's and the output's sizes, the short side asked and the sample aspect used, the DiT's and the
VAE's files with their SHA-256, the seed, the colour correction, the encode's and the decode's
tiles, the attention kernel, and for a video its window cap, --quality, --compile, whether it was
planned again on a resume, its cuts, and the deband its frames were read through, when they were:
what changes the output's bits, the GPU and the libraries' versions aside. With them, IPTC's
digital source type as the AI mark. Always written, no option.

One record, two forms: an XMP packet for a picture (png.py's iTXt chunk, webp.py's XMP chunk) and
stream tags for a video (the CLI's FFV1 writer and assembly), the same fields in the same order.
The writing app gives its name and its distribution's version: `seedvr2x` for the CLI,
`seedvr2x-imgserver` for the image server, the apps being released at one version. No date or
time, nor anything of the machine: the bytes of a record are a function of its fields."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from xml.sax.saxutils import escape

# The repository's address, in the creator's line and the video's seedvr2x_url.
URL = "https://github.com/hekmon/seedvr2x"

# The attention kernel the DiT runs, in the record's attention field: PyTorch's SDPA, the one
# kernel since FlashAttention 2 went (2026-11-08; DESIGN.md, Vendored model code). The field stays,
# constant, as provenance: an output made before says what made it (`flash_attn_2` or `sdpa`).
# Also the name a job's environment records (the CLI's environment.py) and the image server's
# info gives.
ATTENTION = "sdpa"

# IPTC's digital source type for "augmentation, correction or enhancement using a Generative AI
# model" (IPTC NewsCodes, digitalsourcetype): the field platforms read to label a picture so.
DIGITAL_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
)

# seedvr2x's own XMP namespace, under the repository's address, versioned: a field's meaning
# changed or removed is a new version of the URI, a field added is not (DESIGN.md, The record in
# each output).
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
# An app's name in the creator's line: a distribution's name as PyPI normalises it.
_APP = re.compile("[a-z0-9]+(-[a-z0-9]+)*")


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
    maximum), whether the VAE was compiled (--compile), whether the job was planned again on a
    resume (--accept-env-change: its values are then the job's plan, each unit's own in its
    manifest), and its cuts: cut_threshold, the threshold they were detected at as the job's
    settings write it ("0.3"), or cuts_sha256, the SHA-256 of the list --cuts named
    (cuts_sha256()), one of the two; and deband, the filter the frames were read through
    (--deband: "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0",
    the CLI's DEBAND), None for frames read as they are, the field then not written, as in every
    record before it."""

    window: int
    quality: str
    compiled: bool
    replanned: bool
    cut_threshold: str | None = None
    cuts_sha256: str | None = None
    deband: str | None = None


@dataclass(frozen=True)
class Record:
    """An output's record. app is the writing app's name (`seedvr2x`, `seedvr2x-imgserver`) and
    version its version. Sizes are (width, height) in pixels: the source's as its file stores
    it, the output's as written (a picture's turned upright); resolution is the short side asked
    (--resolution, given or its default) and sample_aspect the one used (the source's, or
    --input-sar's), which the sizes alone don't pin. Tiles are the VAE passes' tile sides, in
    output pixels, 0 for an untiled pass; their overlap is passes.OVERLAP, which no plan
    chooses. The attention field is ATTENTION, the one kernel. ValueError for a field out of its
    range."""

    app: str
    version: str
    source: tuple[int, int]
    output: tuple[int, int]
    resolution: int
    sample_aspect: Fraction
    dit: Model
    vae: Model
    seed: int
    color_correction: str
    encode_tile: int
    decode_tile: int
    video: Video | None = None

    def __post_init__(self) -> None:
        if _APP.fullmatch(self.app) is None:
            raise ValueError(f"an app's name is a distribution's, lower case: {self.app!r}")
        if not self.version:
            raise ValueError("a record's version is not empty")
        for what, size in (("source", self.source), ("output", self.output)):
            if len(size) != 2 or min(size) < 1:
                raise ValueError(f"a record's {what} size is two sides of 1 pixel or more: {size}")
        for model in (self.dit, self.vae):
            if _SHA256.fullmatch(model.sha256) is None:
                raise ValueError(f"{model.name}: not a SHA-256 in lower-case hex: {model.sha256!r}")
        if self.resolution < 1 or self.sample_aspect <= 0:
            raise ValueError("a record's resolution and sample aspect are positive")
        if self.seed < 0 or self.encode_tile < 0 or self.decode_tile < 0:
            raise ValueError("a record's seed and tiles are 0 or more")
        if self.video is not None:
            if self.video.window < 0:
                raise ValueError(f"a record's window cap is 0 or more: {self.video.window}")
            if (self.video.cut_threshold is None) == (self.video.cuts_sha256 is None):
                raise ValueError("a video's cuts are a threshold or a list's SHA-256, one of two")
            if self.video.cuts_sha256 is not None and not _SHA256.fullmatch(self.video.cuts_sha256):
                raise ValueError(f"not a SHA-256 in lower-case hex: {self.video.cuts_sha256!r}")
            if self.video.deband is not None and not self.video.deband:
                raise ValueError("a video's deband is a filter, or None for none")

    @property
    def creator(self) -> str:
        """What made the output, in one line: `APP VERSION (URL)`, XMP's CreatorTool and a PNG's
        Software text."""
        return f"{self.app} {self.version} ({URL})"

    def fields(self) -> list[tuple[str, str]]:
        """The record's fields but the app, the version, the address and the mark, as (name,
        value), in their order: the names are the video's tags' less TAG_PREFIX, and in the XMP
        the namespace's properties (seedvr2x:source_size). Sizes as WIDTHxHEIGHT, the sample
        aspect as N:D (ffprobe's form), numbers in decimal, flags as true or false."""

        def size(sides: tuple[int, int]) -> str:
            return f"{sides[0]}x{sides[1]}"

        found = [
            ("source_size", size(self.source)),
            ("output_size", size(self.output)),
            ("resolution", str(self.resolution)),
            ("sample_aspect", f"{self.sample_aspect.numerator}:{self.sample_aspect.denominator}"),
            ("dit_model", _file_name(self.dit.name)),
            ("dit_sha256", self.dit.sha256),
            ("vae_model", _file_name(self.vae.name)),
            ("vae_sha256", self.vae.sha256),
            ("seed", str(self.seed)),
            ("color_correction", self.color_correction),
            ("encode_tile", str(self.encode_tile)),
            ("decode_tile", str(self.decode_tile)),
            ("attention", ATTENTION),
        ]
        if self.video is not None:
            found += [
                ("window", str(self.video.window)),
                ("quality", self.video.quality),
                ("compile", _flag(self.video.compiled)),
                ("replanned", _flag(self.video.replanned)),
            ]
            if self.video.cut_threshold is not None:
                found.append(("cut_threshold", self.video.cut_threshold))
            else:
                assert self.video.cuts_sha256 is not None
                found.append(("cuts_sha256", self.video.cuts_sha256))
            if self.video.deband is not None:
                found.append(("deband", self.video.deband))
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
        seedvr2x_app, seedvr2x_version, seedvr2x_digital_source_type (the mark), then each field
        prefixed (TAG_PREFIX). Matroska's muxer writes the names in upper case."""
        head = [
            ("url", URL),
            ("app", self.app),
            ("version", self.version),
            ("digital_source_type", DIGITAL_SOURCE_TYPE),
        ]
        return [(TAG_PREFIX + name, value) for name, value in head + self.fields()]


def tag_names() -> frozenset[str]:
    """Every tag name a video's record may hold (Record.tags), a video's fields, both forms of
    its cuts and its deband included."""
    model = Model("", "0" * 64)
    names: set[str] = set()
    for video in (
        Video(0, "", False, False, "0", deband="deband"),
        Video(0, "", False, False, None, "0" * 64),
    ):
        sample = Record("a", "0", (1, 1), (1, 1), 1, Fraction(1), model, model, 0, "", 0, 0, video)
        names.update(name for name, _ in sample.tags())
    return frozenset(names)


def cuts_sha256(cuts: Sequence[int]) -> str:
    """The SHA-256 of a cut list as a video's record holds it: of its text one frame number per
    line, each line ended by a line feed, which is the file --cuts read where it holds the numbers
    alone (job.read_cuts ignores blank lines, comments and fields after the number, so a file
    holding them has another hash and the same cuts). Made from the cuts, not the file: the job
    records the cuts, and a job resumed reads its file again."""
    return hashlib.sha256("".join(f"{cut}\n" for cut in cuts).encode()).hexdigest()


def _flag(value: bool) -> str:
    return "true" if value else "false"


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
