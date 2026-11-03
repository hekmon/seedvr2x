"""seedvr2x's record (seedvr2x/record.py; DESIGN.md, The record in each output): its XMP packet
parsed back by the standard library's XML parser, every field at its place under its namespace,
the mark's URI in full; its video tags, the same fields in the same order; the bytes a function of
the fields alone; a field out of range refused; a model's name XML can't hold written so it parses.

read_xmp is the helper test_png.py, test_webp.py and the CLI's test_still_cli.py read a file's
packet with."""

import dataclasses
import xml.etree.ElementTree as ET

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
PICTURE = Record("0.1.0", (4032, 3024), (1440, 1080), DIT, VAE, 42, "split", 0, 1024, "sdpa")
MOVIE = dataclasses.replace(PICTURE, video=Video(20, "maximum", True))

# What PICTURE's packet holds under seedvr2x's namespace, written out here, in order.
FIELDS = [
    ("source_size", "4032x3024"),
    ("output_size", "1440x1080"),
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


def test_a_videos_packet_adds_its_window_quality_and_compile() -> None:
    _, _, fields = read_xmp(MOVIE.xmp())
    assert fields == [*FIELDS, ("window", "20"), ("quality", "maximum"), ("compile", "true")]
    _, _, fields = read_xmp(dataclasses.replace(PICTURE, video=Video(0, "standard", False)).xmp())
    assert fields[-3:] == [("window", "0"), ("quality", "standard"), ("compile", "false")]


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
        ("seedvr2x_version", "0.1.0"),
        ("seedvr2x_digital_source_type", MARK),
        *((f"seedvr2x_{name}", value) for name, value in FIELDS),
        ("seedvr2x_window", "20"),
        ("seedvr2x_quality", "maximum"),
        ("seedvr2x_compile", "true"),
    ]
    assert record.tag_names() == {name for name, _ in MOVIE.tags()}


def test_the_bytes_are_the_fields_alone() -> None:
    # No date, time or machine: two records of the same fields are the same bytes, and a
    # field changed changes them.
    again = Record("0.1.0", (4032, 3024), (1440, 1080), DIT, VAE, 42, "split", 0, 1024, "sdpa")
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
    named = dataclasses.replace(
        PICTURE, vae=Model("/home/someone/models/vae.safetensors", "2" * 64)
    )
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
        {"attention": "flash"},
        {"video": Video(-1, "standard", False)},
    ],
)
def test_a_field_out_of_range_is_a_value_error(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(PICTURE, **change)  # pyright: ignore[reportArgumentType]
