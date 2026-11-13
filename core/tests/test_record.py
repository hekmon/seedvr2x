"""seedvr2x's record (seedvr2x/record.py; DESIGN.md, The record in each output): its XMP packet
parsed back by the standard library's XML parser, every field at its place under its namespace,
the mark's URI in full; its video tags, the same fields in the same order; the bytes a function of
the fields alone; a field out of range refused; a model's name XML can't hold written so it parses.

read_xmp is the helper test_png.py, test_webp.py and the CLI's test_still_cli.py read a file's
packet with."""

import dataclasses
import hashlib
import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

from seedvr2x import record
from seedvr2x.record import Model, Record, Video

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
OURS = "{https://github.com/hekmon/seedvr2x/xmp/1.0/}"
CREATOR_TOOL = "{http://ns.adobe.com/xap/1.0/}CreatorTool"
SOURCE_TYPE = "{http://iptc.org/std/Iptc4xmpExt/2008-02-29/}DigitalSourceType"
MARK = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"

DIT = Model("seedvr2x_ema_7b_sharp_fp16.safetensors", "1" * 64)
VAE = Model("ema_vae_fp16.safetensors", "2" * 64)
PICTURE = Record(
    "seedvr2x", "0.1.0", (4032, 3024), (1440, 1080), 1080, Fraction(1), DIT, VAE, 42, "split", 0,
    1024,
)  # fmt: skip
MOVIE = dataclasses.replace(PICTURE, video=Video(20, "maximum", True, True, cut_threshold="0.3"))

# The CLI's DEBAND (record only carries the string).
DEBAND = "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:blur=0"

# What PICTURE's packet holds under seedvr2x's namespace, written out here, in order.
FIELDS = [
    ("source_size", "4032x3024"),
    ("output_size", "1440x1080"),
    ("resolution", "1080"),
    ("sample_aspect", "1:1"),
    ("dit_model", "seedvr2x_ema_7b_sharp_fp16.safetensors"),
    ("dit_sha256", "1" * 64),
    ("vae_model", "ema_vae_fp16.safetensors"),
    ("vae_sha256", "2" * 64),
    ("seed", "42"),
    ("color_correction", "split"),
    ("encode_tile", "0"),
    ("decode_tile", "1024"),
    ("attention", "sdpa"),
]


def read_xmp(packet: bytes) -> tuple[str, str, list[tuple[str, str]]]:
    """An XMP packet parsed by the standard library: its CreatorTool, its DigitalSourceType and
    seedvr2x's properties, (name, text) in the packet's order. Asserts the shape: one
    rdf:Description inside x:xmpmeta's rdf:RDF, about the file itself, no property elsewhere."""
    root = ET.fromstring(packet)
    assert root.tag == "{adobe:ns:meta/}xmpmeta"
    (rdf,) = list(root)
    assert rdf.tag == f"{RDF}RDF"
    (description,) = list(rdf)
    assert description.tag == f"{RDF}Description" and description.get(f"{RDF}about") == ""
    found = {element.tag: element.text or "" for element in description}
    ours = [
        (element.tag[len(OURS) :], element.text or "")
        for element in description
        if element.tag.startswith(OURS)
    ]
    assert len(ours) + 2 == len(description)
    return found[CREATOR_TOOL], found[SOURCE_TYPE], ours


def test_the_packet_parses_back_to_every_field() -> None:
    tool, mark, fields = read_xmp(PICTURE.xmp())
    assert tool == "seedvr2x 0.1.0 (https://github.com/hekmon/seedvr2x)"
    assert mark == MARK
    assert fields == FIELDS


def test_a_videos_packet_adds_its_window_quality_compile_replan_and_cuts() -> None:
    _, _, fields = read_xmp(MOVIE.xmp())
    assert fields == [
        *FIELDS,
        ("window", "20"),
        ("quality", "maximum"),
        ("compile", "true"),
        ("replanned", "true"),
        ("cut_threshold", "0.3"),
    ]
    listed = Video(0, "standard", False, False, cuts_sha256=record.cuts_sha256([24, 96]))
    _, _, fields = read_xmp(dataclasses.replace(PICTURE, video=listed).xmp())
    assert fields[-5:] == [
        ("window", "0"),
        ("quality", "standard"),
        ("compile", "false"),
        ("replanned", "false"),
        ("cuts_sha256", hashlib.sha256(b"24\n96\n").hexdigest()),
    ]


def test_a_cut_lists_hash_is_that_of_its_plain_file() -> None:
    # One frame number a line: the file --cuts reads where it holds them alone; an empty list,
    # a source of one shot, the hash of nothing.
    assert record.cuts_sha256([120, 250]) == hashlib.sha256(b"120\n250\n").hexdigest()
    assert record.cuts_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_the_sample_aspect_is_written_as_ffprobe_writes_it() -> None:
    anamorphic = dataclasses.replace(PICTURE, sample_aspect=Fraction(32, 27))
    assert dict(anamorphic.fields())["sample_aspect"] == "32:27"


def test_the_creator_names_the_app() -> None:
    server = dataclasses.replace(PICTURE, app="seedvr2x-imgserver")
    assert server.creator == "seedvr2x-imgserver 0.1.0 (https://github.com/hekmon/seedvr2x)"
    assert read_xmp(server.xmp())[0] == server.creator
    assert dict(server.tags())["seedvr2x_app"] == "seedvr2x-imgserver"


def test_the_packet_is_wrapped_as_xmp_asks() -> None:
    packet = PICTURE.xmp()
    assert packet.startswith(b'<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>')
    assert packet.endswith(b'<?xpacket end="r"?>')
    assert packet.decode("utf-8")  # UTF-8 throughout


def test_the_namespace_is_under_the_repositorys_address_with_a_version() -> None:
    assert record.NAMESPACE == "https://github.com/hekmon/seedvr2x/xmp/1.0/"
    assert record.NAMESPACE.startswith(record.URL + "/")
    assert b'xmlns:seedvr2x="https://github.com/hekmon/seedvr2x/xmp/1.0/"' in PICTURE.xmp()


def test_the_tags_hold_the_same_fields_prefixed() -> None:
    assert MOVIE.tags() == [
        ("seedvr2x_url", "https://github.com/hekmon/seedvr2x"),
        ("seedvr2x_app", "seedvr2x"),
        ("seedvr2x_version", "0.1.0"),
        ("seedvr2x_digital_source_type", MARK),
        *((f"seedvr2x_{name}", value) for name, value in FIELDS),
        ("seedvr2x_window", "20"),
        ("seedvr2x_quality", "maximum"),
        ("seedvr2x_compile", "true"),
        ("seedvr2x_replanned", "true"),
        ("seedvr2x_cut_threshold", "0.3"),
    ]
    assert record.tag_names() == {name for name, _ in MOVIE.tags()} | {
        "seedvr2x_cuts_sha256",
        "seedvr2x_deband",
    }


def test_a_video_read_through_a_deband_says_its_filter_and_one_read_plain_nothing() -> None:
    # --deband's filter as the frames were read through it, last, in the packet and the tags;
    # a video read without it writes no such field, the record of before it (MOVIE's above).
    video = Video(20, "maximum", True, True, cut_threshold="0.3", deband=DEBAND)
    debanded = dataclasses.replace(MOVIE, video=video)
    _, _, fields = read_xmp(debanded.xmp())
    assert fields[-2:] == [("cut_threshold", "0.3"), ("deband", DEBAND)]
    assert debanded.tags() == [*MOVIE.tags(), ("seedvr2x_deband", DEBAND)]
    assert not [name for name, _ in MOVIE.tags() if "deband" in name]
    assert not [name for name, _ in MOVIE.fields() if "deband" in name]


def test_the_bytes_are_the_fields_alone() -> None:
    # No date, time or machine: two records of the same fields are the same bytes, and a
    # field changed changes them.
    again = dataclasses.replace(PICTURE)
    assert again.xmp() == PICTURE.xmp() and again.tags() == PICTURE.tags()
    assert dataclasses.replace(PICTURE, seed=43).xmp() != PICTURE.xmp()


# test_a_name_xml_cant_hold_is_written_so_the_packet_parses's name as the record writes it.
WRITTEN = "a<b>&c\ufffdd\ufffde\nf\rg.safetensors"


def test_a_name_xml_cant_hold_is_written_so_the_packet_parses() -> None:
    # A model's name is the user's: markup is escaped, and what XML 1.0 can't hold at all (a
    # control character, an undecodable byte's surrogate) is U+FFFD.
    named = dataclasses.replace(
        PICTURE, dit=Model("a<b>&c\x01d\udcffe\nf\rg.safetensors", "1" * 64)
    )
    _, _, fields = read_xmp(named.xmp())
    assert dict(fields)["dit_model"] == WRITTEN
    assert dict(named.tags())["seedvr2x_dit_model"] == WRITTEN


def test_a_path_given_for_a_model_is_written_as_its_file_name() -> None:
    # --dit-model may name a path: the record holds nothing personal, so its file name alone.
    named = dataclasses.replace(PICTURE, vae=Model("models/fp16/vae.safetensors", "2" * 64))
    assert dict(read_xmp(named.xmp())[2])["vae_model"] == "vae.safetensors"
    assert dict(named.tags())["seedvr2x_vae_model"] == "vae.safetensors"


@pytest.mark.parametrize(
    "change",
    [
        {"version": ""},
        {"source": (0, 1080)},
        {"output": (1920,)},
        {"dit": Model("x", "A" * 64)},
        {"vae": Model("x", "1" * 63)},
        {"seed": -1},
        {"encode_tile": -1},
        {"app": "SeedVR2X"},
        {"app": ""},
        {"resolution": 0},
        {"sample_aspect": Fraction(0)},
        {"video": Video(-1, "standard", False, False, "0.3")},
        {"video": Video(5, "standard", False, False)},
        {"video": Video(5, "standard", False, False, "0.3", "1" * 64)},
        {"video": Video(5, "standard", False, False, None, "not hex")},
        {"video": Video(5, "standard", False, False, "0.3", deband="")},
    ],
)
def test_a_field_out_of_range_is_a_value_error(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(PICTURE, **change)  # pyright: ignore[reportArgumentType]
