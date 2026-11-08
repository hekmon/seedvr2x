"""What the nodes offer and how it reaches the plan: the DiT files by their
names, the combos, the widgets' choices as the library's pins, and the pins taken exactly (or
refused in the nodes' words) by the planners the upscale nodes make."""

import asyncio
import inspect
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from comfy_api.latest import io
from comfyui import models, nodes, plans, words
from standins import StandIn, execute, upscale_image

from seedvr2x import peaks, pull, weights
from seedvr2x.memory import Reading
from seedvr2x.shots import Shot

GIB = peaks.GIB
BIG = Reading(80 * GIB, None, 80 * GIB, False)
FULL_HD = (1080, 1920)


def test_the_dit_files_are_named_as_a_user_reads_them_the_sharp_float16_first() -> None:
    labels = list(plans.DIT_LABELS.values())
    assert labels == [
        "Sharp 7B, float16 (16.5 GB, 0.02%)",
        "Sharp 7B, int8 (8.3 GB, 0.86%)",
        "Sharp 7B, dynamic GGUF (4.8 GB, 7.55%)",
        "7B, float16 (16.5 GB, 0.02%)",
        "7B, int8 (8.3 GB, 0.86%)",
        "7B, dynamic GGUF (4.8 GB, 7.55%)",
    ]
    assert plans.DEFAULT_DIT == next(iter(plans.DIT_LABELS))
    # Every name is a file seedvr2x pins as a DiT (pull.PINNED), the table the CLI downloads by.
    for name in plans.DIT_LABELS:
        assert pull.PINNED[name].role == "dit" and name in pull.MODEL_OF
    assert set(plans.DIT_LABELS) == {n for n, pin in pull.PINNED.items() if pin.role == "dit"}


def schema_inputs(node: type[io.ComfyNode]) -> dict[str, Any]:
    """A node's inputs by id, as its V3 schema lists them."""
    return {each.id: each for each in node.GET_SCHEMA().inputs}


def test_the_models_node_gives_a_light_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_folder = tmp_path / "models"
    models_folder.mkdir()
    for name in ("seedvr2x_ema_7b_sharp_int8_convrot.safetensors", plans.VAE_FILE):
        (models_folder / name).write_bytes(b"x")
    spec = schema_inputs(nodes.Models)
    assert spec["model"].default == "Sharp 7B, float16 (16.5 GB, 0.02%)"
    assert spec["model"].options == list(plans.DIT_LABELS.values())
    # The folder is an optional input and no widget: no default, nothing to fill in.
    assert spec["model_dir"].optional and spec["model_dir"].force_input
    assert spec["model_dir"].io_type == "STRING"
    assert list(spec) == ["model", "keep_in_ram_between_runs", "model_dir"]
    # auto, on, off: auto first and the default; the handle carries the mode, nothing is read.
    keep = spec["keep_in_ram_between_runs"]
    assert (
        keep.options == ["auto", "on", "off"] == list(plans.KEEP_MODES) and keep.default == "auto"
    )
    (handle,) = execute(
        nodes.Models,
        model="Sharp 7B, int8 (8.3 GB, 0.86%)",
        model_dir=f" {models_folder} ",
        keep_in_ram_between_runs="off",
    ).args
    assert handle == models.Handle(
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors", str(models_folder), "off"
    )
    assert models.Handle("a").directory is None and models.Handle("a").keep == "auto"
    assert str(models.Handle("a", "/m").directory) == "/m"
    # The files, not the mode, are what the kept weights are filed under.
    assert models.Handle("a", "/m", "on").key == models.Handle("a", "/m", "off").key


def test_the_models_nodes_labels_placeholder_and_tooltips_say_what_they_are() -> None:
    """The user's Desktop test, 2026-11-05: the folder's id showed, the % unsaid, the RAM's
    choice unclear."""
    spec = schema_inputs(nodes.Models)
    folder = spec["model_dir"]
    # ComfyUI_frontend 1.57.6 draws no placeholder in a single-line STRING: the tooltip says it.
    assert folder.display_name == "model folder" and "placeholder" not in folder.as_dict()
    assert folder.tooltip.startswith("Optional. A folder that has seedvr2x's model files")
    assert "Not connected: Hugging Face's cache, shared with seedvr2x's command line" in (
        folder.tooltip
    )
    assert "Nothing is downloaded from a folder" in folder.tooltip
    assert "Connect any string node" in folder.tooltip
    # What the folder must hold, by the files' names, every DiT file's name made by its rule.
    assert plans.VAE_FILE in folder.tooltip and plans.DEFAULT_DIT in folder.tooltip
    for name in plans.DIT_LABELS:
        made = plans.DEFAULT_DIT
        if "int8" in name:
            made = made.replace("_fp16.safetensors", "_int8_convrot.safetensors")
        elif name.endswith(".gguf"):
            made = made.replace("_fp16.safetensors", "_dyn.gguf")
        if "_sharp" not in name:
            made = made.replace("_sharp", "")
        assert made == name
    for part in ("_int8_convrot.safetensors", "_dyn.gguf", "_fp16.safetensors", "no _sharp"):
        assert part in folder.tooltip, part
    model = spec["model"].tooltip
    # The % is the model card's error per weight (models/hf/README.md, "How to read it").
    assert "error per weight" in model and "the median over the 7B's 288" in model
    assert "else the 8.3 GB file, else the 4.8 GB one" in model
    keep = spec["keep_in_ram_between_runs"].tooltip
    assert "(the float16 file is 16.5 GB)" in keep and "on Windows, 40% of your RAM" in keep
    for word in ("auto:", "on:", "off:"):
        assert word in keep
    assert spec["keep_in_ram_between_runs"].display_name == "keep in RAM between runs"


def test_there_is_no_tuning_node() -> None:
    assert not hasattr(nodes, "Tuning")
    assert [each.GET_SCHEMA().node_id for each in nodes.NODES] == [
        "seedvr2xModels",
        "seedvr2xCuts",
        "seedvr2xUpscaleImage",
        "seedvr2xUpscaleVideo",
        "seedvr2xTile",
        "seedvr2xBlocksSwapped",
        "seedvr2xWindowLength",
    ]


PLANS_NUMBERS = ("window", "encode_tile", "decode_tile", "swap_blocks")
# Each socket's type, and the one value node that gives it.
TYPES = {
    "window": "SEEDVR2X_WINDOW",
    "encode_tile": "SEEDVR2X_TILE",
    "decode_tile": "SEEDVR2X_TILE",
    "swap_blocks": "SEEDVR2X_BLOCKS_SWAPPED",
}
VALUE_NODES = {
    "SEEDVR2X_TILE": nodes.Tile,
    "SEEDVR2X_BLOCKS_SWAPPED": nodes.BlocksSwapped,
    "SEEDVR2X_WINDOW": nodes.WindowLength,
}


def test_the_pictures_come_first_then_the_models_then_the_widgets_then_the_plans_numbers() -> None:
    video = schema_inputs(nodes.UpscaleVideo)
    image = schema_inputs(nodes.UpscaleImage)
    # What does not apply to a node is not on it: a picture is one latent and has no window.
    assert "window" in video and "window" not in image
    assert list(image) == [
        "images",
        "models",
        "resolution",
        "attention",
        "color_correction",
        "seed",
        "plan_only",
        "encode_tile",
        "decode_tile",
        "swap_blocks",
    ]
    # A clip's cut list is the clip's fact: a required socket, right after its frames.
    # and the clip's quality, the video's alone (a picture is one frame in a latent of its own),
    # right after the resolution.
    assert "quality" in video and "quality" not in image
    assert list(video) == [
        "images",
        "cuts",
        *list(image)[1:3],
        "quality",
        *list(image)[3:7],
        "window",
        *list(image)[7:],
    ]
    assert video["cuts"].io_type == "SEEDVR2X_CUTS" and not video["cuts"].optional
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        spec = schema_inputs(node)
        assert spec["images"].io_type == "IMAGE" and spec["models"].io_type == "SEEDVR2X_MODELS"
        for name in ("resolution", "seed"):
            assert spec[name].io_type == "INT", name
        for name, kind in TYPES.items():
            if name in spec:
                assert spec[name].io_type == kind, name
        assert spec["attention"].io_type == "COMBO"
        assert spec["color_correction"].io_type == spec["plan_only"].io_type == "BOOLEAN"
        # The required ones first, as INPUT_TYPES lists them and the frontend builds them.
        optional = [each.optional for each in node.GET_SCHEMA().inputs]
        assert optional == sorted(optional)


def test_the_plans_numbers_are_optional_sockets_each_given_by_one_value_node() -> None:
    """The user's Desktop test, 2026-11-05: a tile of 64 px could be chosen, the INT's min being
    0 for auto; then: each setting a type of its own, so that a link dragged from its socket
    makes ComfyUI offer the node that gives it. Unconnected, the plan chooses."""
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        spec = schema_inputs(node)
        for name in PLANS_NUMBERS:
            if name not in spec:
                continue
            each = spec[name]
            assert each.optional and not isinstance(each, io.WidgetInput), name
            assert "Unconnected, the plan chooses" in each.tooltip and "auto" not in each.tooltip
            giver = VALUE_NODES[each.io_type]
            assert each.tooltip.startswith(
                f"Optional: connect a {giver.GET_SCHEMA().display_name} node"
            )
    # Each type is given by its value node alone, whose inputs are widgets.
    outputs = [each.io_type for node in nodes.NODES for each in node.GET_SCHEMA().outputs]
    for kind, giver in VALUE_NODES.items():
        assert outputs.count(kind) == 1
        assert [each.io_type for each in giver.GET_SCHEMA().outputs] == [kind]
        assert all(isinstance(each, io.WidgetInput) for each in giver.GET_SCHEMA().inputs)


def value(node: type[io.ComfyNode], **inputs: Any) -> Any:
    """A value node's output, run as the executor runs it."""
    return execute(node, None, **inputs)[0]


def test_seedvr2x_tile_gives_a_side_on_the_planners_grid_or_the_whole_image() -> None:
    spec = schema_inputs(nodes.Tile)
    size, whole = spec["tile_size"], spec["whole_image"]
    assert (size.default, size.min, size.max, size.step) == (1024, 512, 2048, 64)
    assert size.display_name == "tile size (px)" and whole.display_name == "whole image at once"
    assert whole.default is False and whole.io_type == "BOOLEAN"
    # Every side on the widget's grid is a tile the plan makes; the switch is untiled (0).
    for side in range(size.min, size.max + 1, size.step):
        assert value(nodes.Tile, tile_size=side, whole_image=False) == side
        assert plans.problems(encode_tile=side) == []
    assert value(nodes.Tile, tile_size=1024, whole_image=True) == 0
    assert plans.Settings.of(encode_tile=0).encode_tile == 0


def test_seedvr2x_blocks_swapped_gives_0_to_36() -> None:
    blocks = schema_inputs(nodes.BlocksSwapped)["blocks"]
    assert (blocks.default, blocks.min, blocks.max, blocks.step) == (0, 0, peaks.BLOCKS, 1)
    for count in range(peaks.BLOCKS + 1):
        assert value(nodes.BlocksSwapped, blocks=count) == count
        assert plans.problems(swap_blocks=count) == []


def test_seedvr2x_window_length_is_set_in_frames_on_the_models_grid_and_gives_latents() -> None:
    frames = schema_inputs(nodes.WindowLength)["frames"]
    assert (frames.default, frames.min, frames.max, frames.step) == (41, 17, 509, 4)
    assert frames.display_name == "window length (frames)"
    # The planner's own range, 5 to 128 latents, in frames: 4(L - 1) + 1.
    assert (plans.window_frames(5), plans.window_frames(128)) == (17, 509)
    for count in range(frames.min, frames.max + 1, frames.step):
        latents = value(nodes.WindowLength, frames=count)
        assert plans.window_frames(latents) == count and 5 <= latents <= 128
        assert plans.problems(window=latents) == []
    assert value(nodes.WindowLength, frames=41) == 11
    # The frontend rounds an INT to its step from its min: 17 + 4k, the model's grid.
    assert frames.min % frames.step == 1


@pytest.mark.parametrize(
    ("node", "inputs", "said"),
    [
        (nodes.Tile, {"tile_size": 1000, "whole_image": False}, "tile size 1000: must be 512"),
        (nodes.Tile, {"tile_size": 448, "whole_image": False}, "tile size 448"),
        (nodes.Tile, {"tile_size": 2112, "whole_image": True}, "tile size 2112"),
        (nodes.Tile, {"tile_size": 1024.0, "whole_image": False}, "must be a whole number"),
        (nodes.Tile, {"tile_size": 1024, "whole_image": 1}, "whole image at once 1"),
        (nodes.BlocksSwapped, {"blocks": 37}, "blocks swapped 37: must be from 0 to 36"),
        (nodes.BlocksSwapped, {"blocks": -1}, "blocks swapped -1"),
        (nodes.WindowLength, {"frames": 20}, "window length 20: must be 17 to 509 frames"),
        (nodes.WindowLength, {"frames": 13}, "window length 13"),
        (nodes.WindowLength, {"frames": 513}, "window length 513"),
    ],
)
def test_a_value_node_refuses_a_number_off_its_grid_queued_or_linked(
    node: type[io.ComfyNode], inputs: dict[str, Any], said: str
) -> None:
    from standins import validate

    # A link brings the number when the node runs: refused there, in the nodes' words.
    with pytest.raises(words.Refused, match=re.escape(said)) as refused:
        value(node, **inputs)
    assert "Change the number connected to this node" in str(refused.value)
    # A number the prompt holds is refused when it is queued (ComfyUI checks no step).
    number = next(each for key, each in inputs.items() if key != "whole_image")
    names = list(inspect.signature(node.validate_inputs).parameters)
    if isinstance(number, int) and not said.startswith("whole image"):
        queued = validate(node, **{names[0]: number})
        assert isinstance(queued, str) and said.split(":")[0] in queued
    assert all(
        each.default is None for each in inspect.signature(node.validate_inputs).parameters.values()
    )
    assert validate(node) is True


def test_every_input_and_output_has_a_label_and_a_tooltip() -> None:
    """The user's Desktop test, 2026-11-05: model_dir showed its id. Every input and output of
    each node has a label in plain words, lower case, as ComfyUI's V3 nodes name theirs."""
    for node in nodes.NODES:
        schema = node.GET_SCHEMA()
        for each in [*schema.inputs, *schema.outputs]:
            assert each.display_name and each.tooltip, (schema.node_id, each.id)
            assert each.display_name[0].islower() and "_" not in each.display_name, each.id
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        schema = node.GET_SCHEMA()
        names = [each.display_name for each in schema.inputs]
        assert names[:2] == (
            ["images", "cuts"] if node is nodes.UpscaleVideo else ["images", "models"]
        )
        assert [(each.id, each.display_name) for each in schema.outputs] == [
            ("images", "images"),
            ("plan", "plan"),
        ]


# What a ComfyUI user reads has none of the library's own terms (the user's Desktop test,
# 2026-11-05): checked on every label, tooltip and description, and on the lines the nodes write.
LIBRARY_TERMS = re.compile(
    r"\b(pins?|pinned|caps?|held|holds?|phases?|latents?|untiled|host)\b", re.IGNORECASE
)
# The launch options a text may name: ComfyUI's own, never one of seedvr2x's command line.
COMFY_OPTIONS = {"--disable-smart-memory"}


def plain(text: str) -> bool:
    """Whether a text a user reads is free of the library's terms and names no option but
    ComfyUI's own."""
    return not LIBRARY_TERMS.search(text) and set(re.findall(r"--[\w-]+", text)) <= COMFY_OPTIONS


def test_no_label_tooltip_or_description_uses_the_librarys_terms() -> None:
    for node in nodes.NODES:
        schema = node.GET_SCHEMA()
        said = [schema.description]
        for each in [*schema.inputs, *schema.outputs]:
            said += [each.display_name, each.tooltip, *each.as_dict().values()]
        for text in said:
            if isinstance(text, str):
                assert plain(text), (schema.node_id, text)
                assert "{" not in text and "}" not in text, (schema.node_id, text)


def test_the_seed_is_comfys_with_its_control_fixed_at_the_command_lines_seed() -> None:
    seed = schema_inputs(nodes.UpscaleImage)["seed"]
    assert (seed.default, seed.min, seed.max) == (42, 0, 2**32 - 1 - 1_000_000)
    # A string starts the frontend's control widget at that mode, where True starts at randomize.
    assert seed.control_after_generate == io.ControlAfterGenerate.fixed == "fixed"
    assert seed.as_dict()["control_after_generate"] == "fixed"


def test_attention_is_a_combo_and_the_colour_correction_a_boolean_split_or_none() -> None:
    spec = schema_inputs(nodes.UpscaleImage)
    assert spec["attention"].options == ["auto", "sdpa", "flash_attn_2"]
    assert spec["attention"].default == "auto"
    # What helps a user choose, measured on the lean attention (research/docs/small-cards.md,
    # FlashAttention 2 against the lean SDPA): the same peak and about the same time, so no
    # speed claim.
    assert "the same memory and about the same time" in spec["attention"].tooltip
    assert "faster" not in spec["attention"].tooltip
    correction = spec["color_correction"]
    assert correction.default is True and correction.display_name == "colour correction"
    assert (correction.label_on, correction.label_off) == ("split", "none")
    assert plans.Settings.of(color_correction=True).color_correction == "split"
    assert plans.Settings.of(color_correction=False).color_correction == "none"


def test_quality_is_a_combo_of_the_levels_on_upscale_video_alone_in_a_users_words() -> None:
    from seedvr2x.shotruns import QUALITY

    assert "quality" not in schema_inputs(nodes.UpscaleImage)
    quality = schema_inputs(nodes.UpscaleVideo)["quality"]
    assert quality.io_type == "COMBO" and quality.display_name == "quality"
    assert quality.options == list(QUALITY) == ["standard", "enhanced", "maximum"]
    assert quality.default == "standard" and not quality.optional
    # MANUAL.md's section, in a ComfyUI user's words: the runs, each frame from the run that
    # rebuilds it best, the gain where there is motion, none on a still shot, the cost in time.
    said = quality.tooltip
    for point in (
        "standard: once",
        "enhanced: twice",
        "maximum: four times",
        "each frame is kept from the run that rebuilds it best",
        "sharper lines and less flicker where there is motion",
        "changes little on a still shot",
        "2 (enhanced) or 4 (maximum) times as long as standard, and no more memory",
    ):
        assert point in said, point
    assert plain(said) and "--" not in said


def test_plan_only_is_a_boolean_off_by_default_on_each_upscale_node() -> None:
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        plan_only = schema_inputs(node)["plan_only"]
        assert plan_only.io_type == "BOOLEAN" and plan_only.default is False
        assert "no model is loaded" in plan_only.tooltip


def settings_of(**inputs_: Any) -> plans.Settings:
    return plans.Settings.of(**inputs_)


def test_the_inputs_values_are_the_settings_as_pins() -> None:
    settings = settings_of(
        window=9,
        swap_blocks=30,
        encode_tile=0,
        decode_tile=1024,
        attention="sdpa",
        color_correction=False,
        seed=7,
    )
    assert settings == plans.Settings(9, 30, 0, 1024, "sdpa", "none", 7)
    assert settings.clip_pins().window == 9 and settings.clip_pins().encode_tile == 0
    assert settings.picture_pins().swap_blocks == 30 and settings.picture_pins().decode_tile == 1024
    # Unconnected: nothing given, the plan's choice.
    auto = settings_of()
    assert auto == plans.Settings()
    assert auto.clip_pins().window is None and auto.attention is None and auto.split
    assert auto.clip_pins().swap_blocks is None and auto.clip_pins().encode_tile is None
    assert not settings.split
    # 0 is a plain value for the blocks: swap none.
    assert settings_of(swap_blocks=0).swap_blocks == 0


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        # What reaches a socket in the library's units: what no value node gives is refused.
        ({"window": 4}, "window length 4: not a value seedvr2x Window Length gives"),
        ({"window": 129}, "window length 129"),
        ({"swap_blocks": 37}, "blocks swapped 37: not a value seedvr2x Blocks Swapped gives"),
        ({"swap_blocks": -1}, "blocks swapped -1"),
        ({"encode_tile": 256}, "encode tile 256: not a value seedvr2x Tile gives; connect one"),
        ({"encode_tile": 2112}, "encode tile 2112"),
        ({"decode_tile": 1000}, "decode tile 1000"),
        ({"encode_tile": 1024.0}, "encode tile 1024.0"),
        ({"swap_blocks": True}, "blocks swapped True"),
        ({"attention": "flash"}, "attention 'flash'"),
        ({"color_correction": "split"}, "colour correction 'split': must be on or off"),
        ({"seed": -1}, "seed -1"),
        ({"seed": 2**32}, "seed 4294967296"),
    ],
)
def test_a_value_no_input_takes_is_refused_naming_it(bad: dict[str, Any], said: str) -> None:
    with pytest.raises(plans.BadSetting, match=re.escape(said)):
        plans.Settings.of(**bad)


def test_a_value_off_the_grid_at_an_upscale_nodes_socket_is_refused_before_anything_loads(
    machine: StandIn, tmp_path
) -> None:
    from standins import handle

    # A node of another pack that gives the same type could give anything.
    with pytest.raises(words.Refused) as refused:
        execute(
            nodes.UpscaleImage,
            "1",
            images=torch.zeros(1, 24, 40, 3),
            models=handle(tmp_path),
            resolution=48,
            attention="auto",
            color_correction=True,
            seed=42,
            plan_only=False,
            encode_tile=300,
        )
    assert "encode tile 300: not a value seedvr2x Tile gives" in str(refused.value)
    assert machine.loads == [] and machine.probes == [] and machine.made == []


def test_the_value_nodes_outputs_reach_the_plan_as_set(machine: StandIn, tmp_path) -> None:
    from standins import handle

    out = execute(
        nodes.UpscaleVideo,
        "1",
        images=torch.zeros(33, 24, 40, 3),
        cuts=(Shot(0, 33),),
        models=handle(tmp_path),
        resolution=48,
        quality="standard",
        attention="auto",
        color_correction=True,
        seed=42,
        plan_only=True,
        window=value(nodes.WindowLength, frames=21),
        decode_tile=value(nodes.Tile, tile_size=512, whole_image=True),
        swap_blocks=value(nodes.BlocksSwapped, blocks=0),
    )
    text = out[1]
    assert "window length: at most 21 frames, as you set it" in text
    assert "decode tile: the whole image at once, no tiles (needs up to " in text
    assert "blocks swapped: none, the whole model on the card while it works, as you set it" in (
        text
    )
    assert "encode tile: auto -> " in text


def test_the_optional_inputs_unwired_are_not_given_and_the_plan_chooses(
    machine: StandIn, tmp_path
) -> None:
    from standins import execute, handle

    # As ComfyUI's executor calls a node: an optional input left unwired is not given at all.
    out = execute(
        nodes.UpscaleImage,
        "1",
        images=torch.zeros(1, 24, 40, 3),
        models=handle(tmp_path),
        resolution=48,
        attention="auto",
        color_correction=True,
        seed=42,
        plan_only=True,
    )
    text = out[1]
    assert "encode tile: auto -> " in text and "decode tile: auto -> " in text
    assert "blocks swapped: auto -> " in text
    with pytest.raises(TypeError, match="optionally"):
        execute(nodes.UpscaleImage, "1", images=torch.zeros(1, 24, 40, 3))
    # Nothing to check when the prompt is queued: a socket holds no number of its own.
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        assert "validate_inputs" not in node.__dict__


@pytest.mark.parametrize(
    ("bad", "widget"),
    [
        ({"attention": "flash"}, "attention"),
        ({"color_correction": "lab"}, "colour correction"),
    ],
)
def test_a_combo_value_no_option_offers_is_refused_naming_the_widget(
    bad: dict[str, object], widget: str
) -> None:
    with pytest.raises(plans.BadSetting, match=widget):
        plans.Settings.of(**bad)  # type: ignore[arg-type]


def picture_plan(settings: plans.Settings):  # type: ignore[no-untyped-def]
    planned = nodes.picture_planner(settings, FULL_HD, FULL_HD)(
        BIG, 64 * GIB, peaks.FLASH_ATTN_2, weights.DIT_7B, False
    )
    return planned.plan


def clip_plan(settings: plans.Settings, frames: int = 100):  # type: ignore[no-untyped-def]
    planned = nodes.clip_planner(settings, FULL_HD, [Shot(0, frames)])(
        BIG, 64 * GIB, peaks.FLASH_ATTN_2, weights.DIT_7B, False
    )
    return planned.plan


def test_the_pins_of_the_settings_reach_a_pictures_plan_exactly() -> None:
    auto, auto_tiles = picture_plan(plans.Settings())
    assert auto.models.swapped == 0
    plan, tiles = picture_plan(plans.Settings(swap_blocks=12, encode_tile=0, decode_tile=1024))
    assert plan.models.swapped == 12 and plan.pins.swap_blocks == 12
    assert tiles.encode is None and tiles.decode is not None and tiles.decode.size == 1024
    assert auto_tiles.decode != tiles.decode
    assert plan.split and not picture_plan(plans.Settings(color_correction="none"))[0].split


def test_the_pins_of_the_settings_reach_a_clips_plan_exactly() -> None:
    plan, expected = clip_plan(plans.Settings(window=9, swap_blocks=10, encode_tile=1024))
    assert plan.window == 9 and plan.swapped == 10
    assert plan.encode_tiles is not None and plan.encode_tiles.size == 1024
    assert expected.window <= 9  # the longest window the layout really runs
    untiled, _ = clip_plan(plans.Settings(decode_tile=0))
    assert untiled.decode_tiles is None
    assert clip_plan(plans.Settings())[0].window != 9


def test_a_pin_the_plan_cant_take_is_refused_in_the_nodes_words() -> None:
    from seedvr2x.errors import PlanRefused

    with pytest.raises(PlanRefused) as refused:
        picture_plan(plans.Settings(swap_blocks=0, decode_tile=4096))
    text = str(words.refusal(refused.value))
    assert "Decode tile 4096 px is as large as the whole image" in text and "--" not in text
    assert "seedvr2x Tile with whole image at once on to decode tile" in text
    with pytest.raises(PlanRefused) as clip:
        clip_plan(plans.Settings(swap_blocks=99), frames=10)
    said = str(words.refusal(clip.value))
    assert said.startswith("Blocks swapped 99 is more than the model has: 36 blocks")
    assert "--" not in said


def test_the_registered_nodes() -> None:
    import comfyui
    from comfy_api.latest import ComfyExtension

    # V3: an entrypoint and no dictionary (ComfyUI's loader reads a dictionary first, and would
    # take the V1 path for it).
    assert not hasattr(comfyui, "NODE_CLASS_MAPPINGS")
    extension = asyncio.run(comfyui.comfy_entrypoint())
    assert isinstance(extension, ComfyExtension)
    listed = asyncio.run(extension.get_node_list())
    assert listed == list(nodes.NODES)
    # The ids a saved workflow names, with the names the menu shows, kept from the V1 nodes.
    schemas = {each.GET_SCHEMA().node_id: each.GET_SCHEMA() for each in listed}
    assert {key: value.display_name for key, value in schemas.items()} == {
        "seedvr2xModels": "seedvr2x Models",
        "seedvr2xCuts": "seedvr2x Cuts",
        "seedvr2xUpscaleImage": "seedvr2x Upscale Image",
        "seedvr2xUpscaleVideo": "seedvr2x Upscale Video",
        "seedvr2xTile": "seedvr2x Tile",
        "seedvr2xBlocksSwapped": "seedvr2x Blocks Swapped",
        "seedvr2xWindowLength": "seedvr2x Window Length",
    }
    for schema in schemas.values():
        assert schema.description
    # Models and the upscale nodes in seedvr2x's category, the value nodes in its subcategory.
    assert {key: value.category for key, value in schemas.items()} == {
        "seedvr2xModels": "seedvr2x",
        "seedvr2xCuts": "seedvr2x",
        "seedvr2xUpscaleImage": "seedvr2x",
        "seedvr2xUpscaleVideo": "seedvr2x",
        "seedvr2xTile": "seedvr2x/plan settings",
        "seedvr2xBlocksSwapped": "seedvr2x/plan settings",
        "seedvr2xWindowLength": "seedvr2x/plan settings",
    }
    upscale = nodes.UpscaleImage.GET_SCHEMA()
    assert [each.id for each in upscale.inputs][:3] == ["images", "models", "resolution"]
    assert upscale.hidden == [io.Hidden.unique_id]
    assert nodes.UpscaleImage.INPUT_TYPES()["hidden"] == {"unique_id": ("UNIQUE_ID",)}
    assert schema_inputs(nodes.UpscaleImage)["resolution"].default == 1080
    assert [(each.id, each.io_type) for each in upscale.outputs] == [
        ("images", "IMAGE"),
        ("plan", "STRING"),
    ]
    video = nodes.UpscaleVideo.GET_SCHEMA()
    assert [(each.id, each.io_type) for each in video.outputs] == [
        ("images", "IMAGE"),
        ("plan", "STRING"),
    ]
    assert [(each.id, each.io_type) for each in nodes.Models.GET_SCHEMA().outputs] == [
        ("models", "SEEDVR2X_MODELS")
    ]
    with pytest.raises(TypeError, match="takes the output of seedvr2x Models"):
        upscale_image(
            "models",  # type: ignore[arg-type]
            plans.Settings(),
            torch.zeros(1, 40, 24, 3),
            1080,
        )


def test_the_hidden_node_id_is_read_from_the_executors_clone_of_the_class() -> None:
    # On the class itself the hidden values are absent; on the clone ComfyUI runs, set.
    assert nodes.UpscaleImage.hidden is None and nodes._node_id(nodes.UpscaleImage) is None  # pyright: ignore[reportPrivateUsage]
    clone = nodes.UpscaleImage.PREPARE_CLASS_CLONE({io.Hidden.unique_id: "17"})
    assert nodes._node_id(clone) == "17"  # pyright: ignore[reportPrivateUsage]
    # A hidden value the schema doesn't ask for is not handed over.
    tile_clone = nodes.Tile.PREPARE_CLASS_CLONE({io.Hidden.unique_id: "17"})
    assert tile_clone.hidden is not None and tile_clone.hidden.unique_id is None
    # The Models node asks for it: its status line and its download's bar are on its node.
    models_clone = nodes.Models.PREPARE_CLASS_CLONE({io.Hidden.unique_id: "17"})
    assert nodes._node_id(models_clone) == "17"  # pyright: ignore[reportPrivateUsage]


ROOM = plans.Room("a card", 40 * GIB, 0, 20 * GIB, False, 40 * GIB, host=48 * GIB)
KEPT = plans.Keeping("on", True, 8 * GIB, 12 * GIB, "x")


def picture_text(settings: plans.Settings, keeping: plans.Keeping = KEPT, **kwargs: Any) -> str:
    made, tiles = picture_plan(settings)
    return plans.picture_text(
        count=3,
        source=(1920, 1080),
        output=(1920, 1080),
        resolution=1080,
        dit="seedvr2x_ema_7b_sharp_fp16.safetensors",
        attention=peaks.SDPA,
        plan=made,
        tiles=tiles,
        room=ROOM,
        settings=settings,
        keeping=keeping,
        **kwargs,
    )


def clip_text(settings: plans.Settings, **kwargs: Any) -> str:
    made, expected = clip_plan(settings, 100)
    return plans.clip_text(
        cuts=[Shot(0, 100)],
        source=(1920, 1080),
        output=(1920, 1080),
        resolution=1080,
        dit="seedvr2x_ema_7b_sharp_fp16.safetensors",
        attention=peaks.SDPA,
        plan=made,
        expected=expected,
        room=ROOM,
        settings=settings,
        keeping=KEPT,
        **kwargs,
    )


def test_a_clip_that_fits_this_card_only_with_the_last_rungs_says_so_in_a_users_words() -> None:
    # The plan's last rungs (shot_plan.plan): one line, in ComfyUI's terms (no tile, no option).
    from dataclasses import replace

    settings = plans.Settings()
    made, expected = clip_plan(settings, 100)
    assert "slower decode" not in clip_text(settings)

    def text(plan: Any) -> str:
        return plans.clip_text(
            cuts=[Shot(0, 100)],
            source=(1920, 1080),
            output=(1920, 1080),
            resolution=1080,
            dit="seedvr2x_ema_7b_sharp_fp16.safetensors",
            attention=peaks.SDPA,
            plan=plan,
            expected=expected,
            room=ROOM,
            settings=settings,
            keeping=KEPT,
        )

    tail = [line for line in text(replace(made, tail_frames=True)).splitlines() if "slower" in line]
    assert tail == [
        "slower decode to fit this card: the end of the video decoder runs one frame at a time"
        " (its memory about a quarter lower, the decode about 1.1 times as long; at tiles of 512,"
        " 576 and 960 px its pictures differ a little from the usual decoder's, the same in every"
        " run)"
    ]
    both = text(replace(made, tail_frames=True, vae_floor=peaks.VAE_ROOM_LEAST))
    (line,) = [each for each in both.splitlines() if "slower" in each]
    assert line.endswith(
        "and the least spare memory kept free around the decode is lowered to 0.50 GiB (the"
        " decode about 1.2 times as long; a pass that needs more keeps it)"
    )
    assert "--" not in line and "tile_" not in line


def test_the_text_says_what_each_auto_became_and_what_a_value_wired_in_was_taken_as() -> None:
    text = picture_text(plans.Settings(decode_tile=1024, swap_blocks=0))
    lines = text.splitlines()
    assert "attention: auto -> PyTorch's SDPA" in lines
    encode = next(each for each in lines if each.startswith("encode tile: "))
    # "encode tile: auto -> 1024 px (up to ...)", or one pass where it fits.
    assert re.fullmatch(
        r"encode tile: auto -> (\d+ px|the whole image at once, no tiles) \(needs up to"
        r" \d+\.\d\d GiB\)",
        encode,
    ), encode
    decode = next(each for each in lines if each.startswith("decode tile: "))
    assert decode.startswith("decode tile: 1024 px (needs up to ")
    assert decode.endswith(", as you set it")
    assert "blocks swapped: none, the whole model on the card while it works, as you set it" in (
        lines
    )
    assert "colour correction: on (split)" in lines and "seed: 42" in lines
    pinned = picture_text(plans.Settings(swap_blocks=12))
    assert any(
        each.startswith("blocks swapped: 12 of the model's 36") and each.endswith(", as you set it")
        for each in pinned.splitlines()
    )
    assert "blocks swapped: auto -> " in picture_text(plans.Settings())
    clipped = clip_text(plans.Settings(window=9, encode_tile=0))
    assert "window length: at most 33 frames, as you set it" in clipped.splitlines()
    assert "encode tile: the whole image at once, no tiles (needs up to " in clipped
    assert "window length: auto -> " in clip_text(plans.Settings())
    for said in (text, pinned, clipped, clip_text(plans.Settings())):
        assert plain(said), said


def test_a_clips_plan_says_the_quality_level_and_its_runs() -> None:
    assert (
        "quality: standard, the model runs once over each shot"
        in clip_text(plans.Settings()).splitlines()
    )
    enhanced = clip_text(plans.Settings(), quality="enhanced").splitlines()
    assert (
        "quality: enhanced, the model runs 2 times over each shot, 2 runs in all, each frame kept"
        " from the run that rebuilds it best: about 2 times standard's time, and no more memory"
    ) in enhanced
    maximum = clip_text(plans.Settings(), quality="maximum").splitlines()
    assert any(
        each.startswith("quality: maximum, the model runs 4 times over each shot, 4 runs in all,")
        for each in maximum
    )
    # A shot too short for a run's first frame has no such run: in all counts the runs made.
    made, expected = clip_plan(plans.Settings(), 100)
    cuts = [Shot(0, 100), Shot(100, 102), Shot(102, 103)]
    text = plans.clip_text(
        cuts=cuts,
        source=(1920, 1080),
        output=(1920, 1080),
        resolution=1080,
        dit="seedvr2x_ema_7b_sharp_fp16.safetensors",
        attention=peaks.SDPA,
        plan=made,
        expected=expected,
        room=ROOM,
        settings=plans.Settings(),
        keeping=KEPT,
        quality="maximum",
    )
    assert "4 times over each shot, 7 runs in all," in text  # 4 + 2 + 1
    for said in (*enhanced, *maximum, text):
        assert plain(said), said


def test_the_text_counts_the_ram_and_says_what_the_computer_has() -> None:
    text = picture_text(plans.Settings(), held=8 * GIB)
    line = next(each for each in text.splitlines() if each.startswith("RAM: "))
    # 3 pictures of 1920x1080 float32: 23.8 MiB in, the same out; 0.02 GiB each, 0.07 GiB.
    assert "(the input's 0.07 GiB, the output's 0.07 GiB and a working copy of one image)" in line
    assert line.startswith("RAM: this run needs about ")
    assert "the computer has 48.00 GiB available" in line
    assert "the model already kept, 8.00 GiB, is not counted in that" in line
    assert "keeping the model takes 8.00 GiB of that" in picture_text(plans.Settings())
    unread = plans.Room("a card", 1, 0, 1, False, 1, host=None)
    assert "the RAM available was not read, so it is not checked" in plans._host(
        unread.host, KEPT, 0, ["x"]
    )  # pyright: ignore[reportPrivateUsage]
    clip = next(each for each in clip_text(plans.Settings()).splitlines() if each.startswith("RAM"))
    # The comparison has the output and the run's own work; the input, ComfyUI's, is said and
    # left out of it (its memory is out of the RAM available already).
    assert "for the clip's output, kept whole," in clip and "2.32 GiB for the clip's" in clip
    assert "the input's frames, 2.32 GiB, are not in that" in clip
    assert "ComfyUI has them in memory already, so they are out of the RAM available" in clip
    assert "the computer has 48.00 GiB available" in clip
    off = plans.Keeping("off", False, 8 * GIB, 12 * GIB, "x")
    assert "the model is not kept" in picture_text(plans.Settings(), off)
    assert "keep in RAM between runs: off -> not kept" in picture_text(plans.Settings(), off)


def test_the_text_leads_with_the_notes_to_say_once() -> None:
    text = picture_text(plans.Settings(), notes=("ComfyUI had dropped the weights",))
    assert text.splitlines()[1] == "ComfyUI had dropped the weights"


def test_nodes_py_is_the_one_module_that_uses_comfy_api() -> None:
    """comfy_api.latest is marked unstable (STABLE = False, v0.44.0): every use of it is in
    nodes.py, the one place to change for another version of it."""
    from pathlib import Path

    folder = Path(nodes.__file__).resolve().parent
    for each in folder.glob("*.py"):
        text = each.read_text(encoding="utf-8")
        uses = "from comfy_api" in text or "import comfy_api" in text
        assert uses == (each.name == "nodes.py"), each.name
