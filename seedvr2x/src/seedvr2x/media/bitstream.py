"""The bitstream filters that set a codec's own colour tags, the way out of a contradiction when a
source's container is right and its bitstream wrong (DESIGN.md, Input): ffmpeg n9.0.2's, run in a
stream copy. They rewrite the bitstream alone, the stream's parameters, its colour tags among them,
passing through (libavcodec/bsf.c:175 at n9.0.2, the extradata alone rewritten,
cbs_bsf.c:167-178), so the container keeps its tags and the frames then say the same."""

from collections.abc import Mapping
from dataclasses import dataclass

# H.273's code points (ITU-T H.273 (07/2024): Tables 2, 3 and 4, and 8.3's VideoFullRangeFlag) of
# ffprobe's names for the tags a filter sets. The names are ffmpeg's (libavutil/pixdesc.c:3272-3345
# at n9.0.2), its enums' values H.273's (pixfmt.h:634, 664, 698). Left out: "reserved" and
# "unknown", ffprobe's names of no value (media/probe.py, UNTAGGED), and ffmpeg's extensions beyond
# H.273 (vgamut and vlog, from 256 on), which no filter takes.
MATRICES = {
    "gbr": 0,
    "bt709": 1,
    "fcc": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "ycgco": 8,
    "bt2020nc": 9,
    "bt2020c": 10,
    "smpte2085": 11,
    "chroma-derived-nc": 12,
    "chroma-derived-c": 13,
    "ictcp": 14,
    "ipt-c2": 15,
    "ycgco-re": 16,
    "ycgco-ro": 17,
}
RANGES = {"tv": 0, "pc": 1}
PRIMARIES = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "film": 8,
    "bt2020": 9,
    "smpte428": 10,
    "smpte431": 11,
    "smpte432": 12,
    "ebu3213": 22,
}
TRANSFERS = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "linear": 8,
    "log100": 9,
    "log316": 10,
    "iec61966-2-4": 11,
    "bt1361e": 12,
    "iec61966-2-1": 13,
    "bt2020-10": 14,
    "bt2020-12": 15,
    "smpte2084": 16,
    "smpte428": 17,
    "arib-std-b67": 18,
}


@dataclass(frozen=True)
class Filter:
    """A bitstream filter setting a codec's own colour tags."""

    name: str  # ffmpeg's
    # Each tag it sets, a VideoStream field (media/source.py, COLOUR): its option, and the values
    # it takes, ffprobe's names to the filter's numbers.
    options: Mapping[str, tuple[str, Mapping[str, int]]]


def _only(values: Mapping[str, int], *taken: int) -> dict[str, int]:
    """values' names whose number is among `taken`."""
    return {name: number for name, number in values.items() if number in taken}


# H.264's and HEVC's filters: their options (h264_metadata.c:609-620, h265_metadata.c:508-519 at
# n9.0.2) take H.273's numbers, those of H.264's and H.265's own Tables E-3 to E-5.
# Their chroma_sample_loc_type goes unused: the chroma location is never contradicted
# (media/source.py, SITING).
H26X = {
    "color_space": ("matrix_coefficients", MATRICES),
    "color_range": ("video_full_range_flag", RANGES),
    "color_primaries": ("colour_primaries", PRIMARIES),
    "color_transfer": ("transfer_characteristics", TRANSFERS),
}

# Every filter of ffmpeg n9.0.2 setting a colour tag (libavcodec/bsf/), by ffprobe's name of its
# codec. None for the others: VVC's sets none (h266_metadata.c:124-128), nor MPEG-4 Part 2's,
# MJPEG's or DV's; LCEVC's sets an enhancement layer's, not a stream seedvr2x reads. Each was run
# on a file whose container and bitstream disagree, the frames then agreeing with the container
# (tests/test_probe.py, test_bitstream_fix).
FILTERS = {
    "h264": Filter("h264_metadata", H26X),
    "hevc": Filter("hevc_metadata", H26X),
    # AV1's and APV's take H.273's numbers too (av1_metadata.c:170-186, apv_metadata.c:94-110).
    "av1": Filter(
        "av1_metadata",
        {
            "color_space": ("matrix_coefficients", MATRICES),
            "color_range": ("color_range", RANGES),
            "color_primaries": ("color_primaries", PRIMARIES),
            "color_transfer": ("transfer_characteristics", TRANSFERS),
        },
    ),
    "apv": Filter(
        "apv_metadata",
        {
            "color_space": ("matrix_coefficients", MATRICES),
            "color_range": ("full_range_flag", RANGES),
            "color_primaries": ("color_primaries", PRIMARIES),
            "color_transfer": ("transfer_characteristics", TRANSFERS),
        },
    ),
    # MPEG-2's (mpeg2_metadata.c:211-219) as ffmpeg's decoder reads them, H.273's numbers
    # (mpeg12dec.c:1075-1077), 0 refused (mpeg2_metadata.c:182-191), so no RGB matrix. No range:
    # MPEG-2's frames are always limited range (mpeg12dec.c:786).
    "mpeg2video": Filter(
        "mpeg2_metadata",
        {
            "color_space": ("matrix_coefficients", {n: v for n, v in MATRICES.items() if v}),
            "color_primaries": ("colour_primaries", PRIMARIES),
            "color_transfer": ("transfer_characteristics", TRANSFERS),
        },
    ),
    # VP9's matrix by VP9's own numbers (cbs_vp9.h:57-64), as its decoder names them
    # (vp9.c:461-463), and its range (vp9_metadata.c:97-122). RGB left out: the filter sets it in
    # profiles 1 and 3 alone, warning in the others (vp9_metadata.c:47-55). VP9 has no primaries or
    # transfer: its frames carry the stream's.
    "vp9": Filter(
        "vp9_metadata",
        {
            "color_space": (
                "color_space",
                {"bt470bg": 1, "bt709": 2, "smpte170m": 3, "smpte240m": 4, "bt2020nc": 5},
            ),
            "color_range": ("color_range", RANGES),
        },
    ),
    # ProRes's: the primaries and matrices its filter accepts (prores_metadata.c:102-127) and the
    # transfers it names (145-150), H.273's numbers, but 0, which it calls unknown and ffmpeg's
    # decoder reads as RGB's matrix (proresdec.c:282-284). No range: ProRes's frames are always
    # limited range (proresdec.c:285).
    "prores": Filter(
        "prores_metadata",
        {
            "color_space": ("colorspace", _only(MATRICES, 1, 6, 9)),
            "color_primaries": ("color_primaries", _only(PRIMARIES, 1, 5, 6, 9, 11, 12)),
            "color_transfer": ("color_trc", _only(TRANSFERS, 1, 16, 18)),
        },
    ),
}


def setting(
    codec: str, tags: Mapping[str, str], synonyms: Mapping[str, frozenset[str]] | None = None
) -> str:
    """The bitstream filter with its options, as -bsf takes it, setting a stream's own colour
    `tags` (VideoStream's fields, media/source.py, COLOUR, to ffprobe's names of their values):
    "hevc_metadata=matrix_coefficients=1", the options in the order of `tags`. A value the filter
    doesn't take is set by another name of its meaning that it does (`synonyms`: each field's
    names of one meaning, source.SYNONYMS), the lowest of its numbers: the bitstream then says
    what its container does, by another name. "" for a codec (ffprobe's name) ffmpeg has no such
    filter for, or whose filter can't set every one of `tags` to its value or to one of its
    meaning, which would leave the bitstream contradicting its container still."""
    found = FILTERS.get(codec)
    if found is None:
        return ""
    options: list[str] = []
    for name, value in tags.items():
        option, values = found.options.get(name, ("", {}))
        # The container's own value where the filter takes it, so that it sets the same name;
        # else another name of its meaning: ProRes's filter takes the matrix smpte170m and not
        # bt470bg, the transfer bt709 and neither smpte170m nor bt2020's (FILTERS).
        same = (synonyms or {}).get(name, frozenset[str]())
        others = sorted(values[other] for other in same if other in values and value in same)
        numbers = [values[value]] if value in values else others
        if not numbers:
            return ""
        options.append(f"{option}={numbers[0]}")
    return f"{found.name}={':'.join(options)}"
