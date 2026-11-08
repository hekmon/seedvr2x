"""The example workflows (workflows/*.json), in ComfyUI's UI workflow format: each parses, every
node is seedvr2x's or a stock node named below, each seedvr2x node's sockets, outputs and widgets
are those of its V3 schema (count, order, values) at their defaults, every optional input left
unwired (the plan chooses), every link joins an output to an input of its type, seedvr2x Models is
wired into each upscale node and no node holds the plan's settings but the upscale node.

The frontend builds a node's sockets in the order of the schema's input list, required before
optional (the loader hands it the schema as INPUT_TYPES with the keys in that order, and the
executor names the arguments of `execute` by id), then its widgets' slots; a socket is an input of
a type that is no widget's (a custom type or IMAGE) or one with forceInput, a widget an input with
a default (WidgetInput). The files are in the form ComfyUI_frontend 1.57.6 exports (read in its
bundle, 2026-11-05): the sockets alone under `inputs` (the unlinked widget slots left out, the
links' target slots numbered in that list), the widgets' values in their order. Its load rebuilds
the inputs from the node's definition and takes each saved entry's link by name.

The stock nodes' sockets and widgets are the table below, read from ComfyUI v0.44.0's schemas
(LoadImage, SaveImage, PreviewAny, LoadVideo, GetVideoComponents, CreateVideo, SaveVideo) and
loaded in it by hand on 2026-11-05: the tests hold the files to what they were written against."""

import json
from pathlib import Path
from typing import Any

import pytest
from comfy_api.latest import io
from comfyui import nodes

from seedvr2x import shotcuts

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"
NAMES = ("picture", "clip")

type Node = dict[str, Any]

# What each stock node takes on a socket (the inputs a workflow lists, in the order the frontend
# saves them: a widget fed by a link comes after the sockets), what it gives, and how many widget
# values it saves. SaveVideo's four are the prefix, the format, the format's codec and the
# hidden codec; CreateVideo's five are fps, bit depth, colour space, chroma subsampling, codec.
STOCK: dict[str, dict[str, Any]] = {
    "LoadImage": {"inputs": [], "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")], "widgets": 2},
    "SaveImage": {"inputs": [("images", "IMAGE")], "outputs": [], "widgets": 1},
    "PreviewAny": {
        "inputs": [("source", "*")],
        "outputs": [("STRING", "STRING")],
        "widgets": 0,
    },
    "LoadVideo": {"inputs": [], "outputs": [("VIDEO", "VIDEO")], "widgets": 2},
    "GetVideoComponents": {
        "inputs": [("video", "VIDEO")],
        "outputs": [
            ("images", "IMAGE"),
            ("audio", "AUDIO"),
            ("fps", "FLOAT"),
            ("bit_depth", "COMBO"),
            ("color_space", "COMBO"),
        ],
        "widgets": 0,
    },
    "CreateVideo": {
        "inputs": [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")],
        "outputs": [("VIDEO", "VIDEO")],
        "widgets": 5,
    },
    "SaveVideo": {
        "inputs": [("video", "VIDEO")],
        "outputs": [("video", "VIDEO")],
        "widgets": 4,
    },
}
UPSCALES = ("seedvr2xUpscaleImage", "seedvr2xUpscaleVideo")
SCHEMAS = {each.GET_SCHEMA().node_id: each.GET_SCHEMA() for each in nodes.NODES}


def load(name: str) -> dict[str, Any]:
    return json.loads((WORKFLOWS / f"{name}.json").read_text(encoding="utf-8"))


def by_id(workflow: dict[str, Any]) -> dict[int, Node]:
    return {node["id"]: node for node in workflow["nodes"]}


def of_type(workflow: dict[str, Any], kind: str) -> list[Node]:
    return [node for node in workflow["nodes"] if node["type"] == kind]


def widgets_with_controls(widgets: list[io.WidgetInput]) -> list[io.WidgetInput | None]:
    """The widgets in the order the frontend saves their values: a seed's control widget right
    after it (None stands for it)."""
    found: list[io.WidgetInput | None] = []
    for widget in widgets:
        found.append(widget)
        if getattr(widget, "control_after_generate", None):
            found.append(None)
    return found


def definition(kind: str) -> tuple[list[tuple[str, str]], list[io.WidgetInput]]:
    """A seedvr2x node's sockets (name, type) and its widgets, each in the order of its schema's
    inputs."""
    sockets: list[tuple[str, str]] = []
    widgets: list[io.WidgetInput] = []
    for each in SCHEMAS[kind].inputs:
        if isinstance(each, io.WidgetInput) and not each.force_input:
            widgets.append(each)
        else:
            sockets.append((each.id, each.io_type))
    return sockets, widgets


def optional(kind: str) -> list[str]:
    """A seedvr2x node's optional inputs, by id."""
    return [each.id for each in SCHEMAS[kind].inputs if each.optional]


@pytest.mark.parametrize("name", NAMES)
def test_the_file_parses_as_a_workflow_of_the_frontends_format(name: str) -> None:
    workflow = load(name)
    assert workflow["version"] == 0.4
    ids = [node["id"] for node in workflow["nodes"]]
    assert len(set(ids)) == len(ids) and workflow["last_node_id"] >= max(ids)
    link_ids = [link[0] for link in workflow["links"]]
    assert len(set(link_ids)) == len(link_ids) and workflow["last_link_id"] >= max(link_ids)
    orders = [node["order"] for node in workflow["nodes"]]
    assert sorted(orders) == list(range(len(orders)))
    for node in workflow["nodes"]:
        assert {"id", "type", "pos", "size", "inputs", "outputs", "widgets_values"} <= node.keys()
        assert node["properties"]["Node name for S&R"] == node["type"]
        assert len(node["pos"]) == len(node["size"]) == 2
    for group in workflow["groups"]:
        assert len(group["bounding"]) == 4


@pytest.mark.parametrize("name", NAMES)
def test_every_node_is_seedvr2xs_or_a_named_stock_node(name: str) -> None:
    for node in load(name)["nodes"]:
        assert node["type"] in SCHEMAS or node["type"] in STOCK, node["type"]


@pytest.mark.parametrize("name", NAMES)
def test_the_files_name_no_machine_and_no_user(name: str) -> None:
    text = (WORKFLOWS / f"{name}.json").read_text(encoding="utf-8")
    for word in ("/home", "/mnt", "/Users", "C:\\", "D:\\", "localhost"):
        assert word not in text


@pytest.mark.parametrize("name", NAMES)
def test_seedvr2x_widgets_are_those_of_the_schema_at_their_defaults(name: str) -> None:
    seen = 0
    for node in load(name)["nodes"]:
        if node["type"] not in SCHEMAS:
            continue
        seen += 1
        _, widgets = definition(node["type"])
        values = node["widgets_values"]
        # A widget saves one value, in the order of the schema's inputs, and the seed's control
        # widget a second one after it ("fixed" by default).
        controls = [
            widget.id for widget in widgets if getattr(widget, "control_after_generate", None)
        ]
        assert controls in ([], ["seed"]), node["type"]
        saved = [
            (widget, value)
            for widget, value in zip(widgets_with_controls(widgets), values, strict=True)
        ]
        assert len(values) == len(widgets) + len(controls), node["type"]
        for widget, value in saved:
            if widget is None:
                assert value == "fixed"
                continue
            assert value == widget.default, (node["type"], widget.id)
            if isinstance(widget, io.Combo.Input):
                assert value in widget.options, (node["type"], widget.id)
            elif isinstance(widget, io.Int.Input):
                assert type(value) is int and widget.min is not None and widget.max is not None
                assert widget.min <= value <= widget.max
                if widget.id == "seed":
                    assert widget.control_after_generate == io.ControlAfterGenerate.fixed
            elif isinstance(widget, io.Boolean.Input):
                assert type(value) is bool
            elif isinstance(widget, io.Float.Input):
                assert widget.min is not None and widget.max is not None
                assert type(value) is float and widget.min <= value <= widget.max
            else:
                assert isinstance(value, str)
        if node["type"] == "seedvr2xCuts":
            # The model folder is an optional input and no widget: no saved value for it.
            assert [widget.id for widget, _ in saved if widget] == ["mode", "threshold"]
            assert values == ["detect", shotcuts.THRESHOLD]
        elif node["type"] != "seedvr2xModels":
            # No widget of a number the plan makes: the workflow shows the plan's own choices.
            by_id_ = {widget.id: value for widget, value in saved if widget is not None}
            assert set(by_id_) == {
                "resolution",
                "attention",
                "color_correction",
                "seed",
                "plan_only",
                *(["quality"] if node["type"] == "seedvr2xUpscaleVideo" else []),
            }
            if node["type"] == "seedvr2xUpscaleVideo":
                assert by_id_["quality"] == "standard"
            assert by_id_["attention"] == "auto" and by_id_["color_correction"] is True
            assert by_id_["plan_only"] is False and by_id_["resolution"] == 1080
        else:
            assert values[-1] == "auto", "keep in RAM between runs is auto"
            assert len(values) == 2, "the model folder is no widget: the cache, unwired"
    assert seen == {"picture": 2, "clip": 3}[name]


@pytest.mark.parametrize("name", NAMES)
def test_seedvr2x_sockets_and_outputs_are_those_of_the_classes(name: str) -> None:
    for node in load(name)["nodes"]:
        schema = SCHEMAS.get(node["type"])
        if schema is None:
            continue
        sockets, _ = definition(node["type"])
        assert [(each["name"], each["type"]) for each in node["inputs"]] == sockets
        # No saved label, which the frontend would show in place of the input's own.
        assert all("label" not in each and "widget" not in each for each in node["inputs"])
        assert [(each["name"], each["type"]) for each in node["outputs"]] == [
            (each.display_name, each.io_type) for each in schema.outputs
        ]


@pytest.mark.parametrize("name", NAMES)
def test_stock_nodes_have_the_sockets_outputs_and_widgets_they_were_written_for(
    name: str,
) -> None:
    for node in load(name)["nodes"]:
        stock = STOCK.get(node["type"])
        if stock is None:
            continue
        assert [(each["name"], each["type"]) for each in node["inputs"]] == stock["inputs"]
        assert [(each["name"], each["type"]) for each in node["outputs"]] == stock["outputs"]
        assert len(node["widgets_values"]) == stock["widgets"], node["type"]


@pytest.mark.parametrize("name", NAMES)
def test_every_link_joins_an_output_to_an_input_of_its_type(name: str) -> None:
    workflow = load(name)
    found = by_id(workflow)
    for link_id, origin, origin_slot, target, target_slot, kind in workflow["links"]:
        output = found[origin]["outputs"][origin_slot]
        socket = found[target]["inputs"][target_slot]
        assert output["type"] == kind, (link_id, "origin")
        assert socket["type"] in (kind, "*"), (link_id, "target")
        assert link_id in output["links"] and socket["link"] == link_id
    # And nothing refers to a link that is not listed.
    listed = {link[0] for link in workflow["links"]}
    for node in workflow["nodes"]:
        for socket in node["inputs"]:
            assert socket["link"] is None or socket["link"] in listed
        for output in node["outputs"]:
            assert set(output["links"] or []) <= listed
    # A seedvr2x node's required socket is wired (the nodes take no default for a handle or the
    # pictures), and every optional one left unwired: the plan chooses.
    for node in workflow["nodes"]:
        if node["type"] in SCHEMAS:
            unwired = optional(node["type"])
            for socket in node["inputs"]:
                assert (socket["link"] is None) == (socket["name"] in unwired), socket["name"]
            assert {each["name"] for each in node["inputs"]} >= set(unwired)


def source(workflow: dict[str, Any], node: Node, socket: str) -> Node:
    """The node whose output feeds `socket` of `node`."""
    found = by_id(workflow)
    link = next(each["link"] for each in node["inputs"] if each["name"] == socket)
    origin = next(each[1] for each in workflow["links"] if each[0] == link)
    return found[origin]


@pytest.mark.parametrize("name", NAMES)
def test_models_is_wired_into_each_upscale_node_and_there_is_no_tuning_node(name: str) -> None:
    workflow = load(name)
    upscales = [node for node in workflow["nodes"] if node["type"] in UPSCALES]
    assert len(upscales) == 1
    assert len(of_type(workflow, "seedvr2xModels")) == 1
    assert not of_type(workflow, "seedvr2xTuning")
    assert not [link for link in workflow["links"] if "TUNING" in str(link[5])]
    for node in upscales:
        assert source(workflow, node, "models")["type"] == "seedvr2xModels"


@pytest.mark.parametrize("name", NAMES)
def test_the_plan_is_shown_as_text(name: str) -> None:
    workflow = load(name)
    (upscale,) = [node for node in workflow["nodes"] if node["type"] in UPSCALES]
    (preview,) = of_type(workflow, "PreviewAny")
    assert source(workflow, preview, "source") is upscale
    link = next(each for each in workflow["links"] if each[3] == preview["id"])
    assert upscale["outputs"][link[2]]["name"] == "plan"


def test_the_picture_workflow_is_a_picture_in_and_out() -> None:
    workflow = load("picture")
    (upscale,) = of_type(workflow, "seedvr2xUpscaleImage")
    assert source(workflow, upscale, "images")["type"] == "LoadImage"
    (save,) = of_type(workflow, "SaveImage")
    assert source(workflow, save, "images") is upscale
    assert len(workflow["nodes"]) == 5


def test_the_clip_workflow_takes_the_frame_rate_and_audio_of_the_video_it_loads() -> None:
    workflow = load("clip")
    (upscale,) = of_type(workflow, "seedvr2xUpscaleVideo")
    (components,) = of_type(workflow, "GetVideoComponents")
    (create,) = of_type(workflow, "CreateVideo")
    (save,) = of_type(workflow, "SaveVideo")
    assert source(workflow, components, "video")["type"] == "LoadVideo"
    assert source(workflow, upscale, "images") is components
    assert source(workflow, create, "images") is upscale
    assert source(workflow, save, "video") is create
    # The frame rate is the loaded video's own, not CreateVideo's widget.
    assert source(workflow, create, "fps") is components
    fps = next(each for each in workflow["links"] if each[3] == create["id"] and each[5] == "FLOAT")
    assert components["outputs"][fps[2]]["name"] == "fps"
    audio = source(workflow, create, "audio")
    assert audio is components
    # Frames come from the first output, the frame rate from the third.
    frames_link = next(
        each for each in workflow["links"] if each[3] == upscale["id"] and each[5] == "IMAGE"
    )
    assert components["outputs"][frames_link[2]]["name"] == "images"


def test_the_clip_workflow_finds_the_cuts_of_the_frames_it_upscales() -> None:
    """Load Video, Cuts, Models, Upscale Video: the same frames go to Cuts and to the upscale, the
    cut list is wired to the upscale's required cuts input, and Cuts is at its defaults (the
    detector at the command line's threshold)."""
    workflow = load("clip")
    (upscale,) = of_type(workflow, "seedvr2xUpscaleVideo")
    (cuts,) = of_type(workflow, "seedvr2xCuts")
    (components,) = of_type(workflow, "GetVideoComponents")
    assert source(workflow, cuts, "images") is components
    assert source(workflow, upscale, "cuts") is cuts
    assert source(workflow, upscale, "images") is components
    assert cuts["order"] < upscale["order"]
    assert not of_type(load("picture"), "seedvr2xCuts")
