"""What the nodes offer and how it reaches the plan: the DiT files by their
names, the combos, the widgets' choices as the library's pins, and the pins taken exactly (or
refused in the nodes' words) by the planners the upscale nodes make."""

import asyncio
import re
from typing import Any

import pytest
import torch
from comfy_api.latest import io
from comfyui import models, nodes, plans, words
from standins import StandIn, execute, upscale_image

from seedvr2x import peaks, pull, weights
from seedvr2x.memory import Reading

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


def test_the_models_node_gives_a_light_handle() -> None:
    spec = schema_inputs(nodes.Models)
    assert spec["model"].default == "Sharp 7B, float16 (16.5 GB, 0.02%)"
    assert spec["model"].options == list(plans.DIT_LABELS.values())
    assert spec["model_dir"].default == "" and not spec["model_dir"].multiline
    # auto, on, off: auto first and the default; the handle carries the mode, nothing is read.
    keep = spec["keep_in_ram_between_runs"]
    assert (
        keep.options == ["auto", "on", "off"] == list(plans.KEEP_MODES) and keep.default == "auto"
    )
    (handle,) = execute(
        nodes.Models,
        model="Sharp 7B, int8 (8.3 GB, 0.86%)",
        model_dir=" /models ",
        keep_in_ram_between_runs="off",
    ).args
    assert handle == models.Handle(
        "seedvr2x_ema_7b_sharp_int8_convrot.safetensors", "/models", "off"
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
    assert folder.tooltip.startswith("Leave empty to download the files from Hugging Face")
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
        "seedvr2xUpscaleImage",
        "seedvr2xUpscaleVideo",
    ]


PLANS_NUMBERS = ("window", "encode_tile", "decode_tile", "swap_blocks")


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
    assert list(video) == [*list(image)[:7], "window", *list(image)[7:]]
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        spec = schema_inputs(node)
        assert spec["images"].io_type == "IMAGE" and spec["models"].io_type == "SEEDVR2X_MODELS"
        for name in ("resolution", "seed", *[each for each in PLANS_NUMBERS if each in spec]):
            assert spec[name].io_type == "INT", name
        assert spec["attention"].io_type == "COMBO"
        assert spec["color_correction"].io_type == spec["plan_only"].io_type == "BOOLEAN"
        # The required ones first, as INPUT_TYPES lists them and the frontend builds them.
        optional = [each.optional for each in node.GET_SCHEMA().inputs]
        assert optional == sorted(optional)


def test_the_plans_numbers_are_optional_inputs_with_no_widget_on_the_planners_grid() -> None:
    """The user's Desktop test, 2026-11-05: a tile of 64 px could be chosen, the INT's min being
    0 for auto. Unwired, the plan chooses; wired, the value is taken; no value stands for auto."""
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        spec = schema_inputs(node)
        for name in PLANS_NUMBERS:
            if name not in spec:
                continue
            each = spec[name]
            assert each.optional and each.force_input and each.default is None, name
            assert each.as_dict()["forceInput"] is True
            assert "Unconnected, the plan chooses" in each.tooltip and "auto" not in each.tooltip
            assert each.tooltip.startswith("Optional: connect a number (from an Int node, for one)")
            assert "auto" not in each.display_name and "=" not in each.display_name
            # Every value on the input's grid is one the plan takes, and nothing off it.
            grid = range(each.min, each.max + 1, each.step)
            assert all(plans.problems(**{name: value}) == [] for value in grid), name
            assert plans.problems(**{name: each.min - each.step}) != [], name
            assert plans.problems(**{name: each.max + each.step}) != [], name
        for name in ("encode_tile", "decode_tile"):
            tile = spec[name]
            assert (tile.min, tile.max, tile.step) == (512, 2112, 64)
            assert tile.max == peaks.TILE_MAX + peaks.TILE_STEP == plans.UNTILED_TILE
            assert "2112 for no tiles, the whole image at once" in tile.tooltip
        swap = spec["swap_blocks"]
        assert (swap.min, swap.max, swap.step) == (0, peaks.BLOCKS, 1)
        assert "from 0 (none: the whole model on the card)" in swap.tooltip
    window = schema_inputs(nodes.UpscaleVideo)["window"]
    assert (window.min, window.max, window.step) == (plans.WINDOW_MIN, plans.WINDOW_MAX, 1)
    assert plans.WINDOW_MIN == 5 and "from 5 (about 20 frames) to 128" in window.tooltip


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
        assert [each.display_name for each in schema.inputs][:2] == ["images", "models"]
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
    # What helps a user choose: the speed (research/docs/small-cards.md), and no memory claim.
    assert "about a fifth off the model's step for a 4K picture" in spec["attention"].tooltip
    assert "memory" not in spec["attention"].tooltip
    correction = spec["color_correction"]
    assert correction.default is True and correction.display_name == "colour correction"
    assert (correction.label_on, correction.label_off) == ("split", "none")
    assert plans.Settings.of(color_correction=True).color_correction == "split"
    assert plans.Settings.of(color_correction=False).color_correction == "none"


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
        encode_tile=plans.UNTILED_TILE,
        decode_tile=1024,
        attention="sdpa",
        color_correction=False,
        seed=7,
    )
    assert settings == plans.Settings(9, 30, 0, 1024, "sdpa", "none", 7)
    assert settings.clip_pins().window == 9 and settings.clip_pins().encode_tile == 0
    assert settings.picture_pins().swap_blocks == 30 and settings.picture_pins().decode_tile == 1024
    # Unwired: nothing given, the plan's choice.
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
        ({"window": 4}, "window length 4: must be from 5 to 128"),
        ({"window": 129}, "window length 129"),
        ({"window": 0}, "window length 0"),
        ({"window": -1}, "window length -1"),
        ({"swap_blocks": 37}, "blocks swapped 37: must be from 0 to 36"),
        ({"swap_blocks": -1}, "blocks swapped -1"),
        (
            {"encode_tile": 256},
            "encode tile 256: must be 512 to 2048 px in steps of 64, or 2112 for no tiles",
        ),
        ({"encode_tile": 64}, "encode tile 64"),
        ({"encode_tile": 0}, "encode tile 0"),
        ({"decode_tile": 1000}, "decode tile 1000"),
        ({"decode_tile": 4096}, "decode tile 4096"),
        ({"encode_tile": -1}, "encode tile -1"),
        # A link may carry what is no whole number.
        ({"encode_tile": 1024.0}, "encode tile 1024.0: must be a whole number"),
        ({"swap_blocks": True}, "blocks swapped True: must be a whole number"),
        ({"attention": "flash"}, "attention 'flash'"),
        ({"color_correction": "split"}, "colour correction 'split': must be on or off"),
        ({"seed": -1}, "seed -1"),
        ({"seed": 2**32}, "seed 4294967296"),
    ],
)
def test_a_value_no_input_takes_is_refused_naming_it(bad: dict[str, Any], said: str) -> None:
    with pytest.raises(plans.BadSetting, match=re.escape(said)):
        plans.Settings.of(**bad)


def test_the_values_on_the_grid_the_plan_tries_are_taken() -> None:
    for tile in range(512, 2113, 64):
        assert plans.tile_problem("encode_tile", tile) is None, tile
    for window in range(5, 129):
        assert plans.window_problem(window) is None, window
    for blocks in range(37):
        assert plans.swap_problem(blocks) is None, blocks
    assert plans.UNTILED_TILE == 2112


def test_numbers_the_prompt_holds_are_checked_when_it_is_queued() -> None:
    from standins import validate

    # A value the prompt holds (a value node the frontend folds into it): refused in the nodes'
    # words, before anything runs; the constants only are given, and only the arguments the node
    # names (the executor hands over what validate_inputs takes).
    text = validate(nodes.UpscaleVideo, window=3, encode_tile=256, decode_tile=2112, swap_blocks=99)
    assert isinstance(text, str)
    assert "window length 3" in text and "encode tile 256" in text and "blocks swapped 99" in text
    assert "decode tile" not in text and "--" not in text
    assert validate(nodes.UpscaleImage, encode_tile=1024, decode_tile=2112, swap_blocks=0) is True
    # A linked number is not among the constants, an unwired one not given: nothing to check.
    assert validate(nodes.UpscaleImage, resolution=1080) is True
    assert validate(nodes.UpscaleVideo) is True
    # The window is the video node's alone: a picture node does not take it.
    assert validate(nodes.UpscaleImage, window=3) is True
    for node in (nodes.UpscaleImage, nodes.UpscaleVideo):
        import inspect

        names = inspect.signature(node.validate_inputs).parameters
        assert all(each.default is None for each in names.values()), "each has a default"
        assert set(names) <= set(schema_inputs(node))


def test_a_linked_number_off_the_grid_is_refused_at_the_nodes_start_before_anything_loads(
    machine: StandIn, tmp_path
) -> None:
    from standins import execute, handle

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
    assert "encode tile 300" in str(refused.value)
    assert "Change the number connected to this node" in str(refused.value)
    assert machine.loads == [] and machine.probes == [] and machine.made == []


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
    planned = nodes.clip_planner(settings, FULL_HD, frames)(
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
    assert "decode tile 4096 on this node" in text and "--" not in text
    assert "decode tile 2112 (no tiles) on this node" in text
    with pytest.raises(PlanRefused) as clip:
        clip_plan(plans.Settings(swap_blocks=99), frames=10)
    said = str(words.refusal(clip.value))
    assert said.startswith("blocks swapped 99 on this node: the DiT has 36 blocks")
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
    assert listed == [nodes.Models, nodes.UpscaleImage, nodes.UpscaleVideo]
    # The ids a saved workflow names, with the names the menu shows, kept from the V1 nodes.
    schemas = {each.GET_SCHEMA().node_id: each.GET_SCHEMA() for each in listed}
    assert {key: value.display_name for key, value in schemas.items()} == {
        "seedvr2xModels": "seedvr2x Models",
        "seedvr2xUpscaleImage": "seedvr2x Upscale Image",
        "seedvr2xUpscaleVideo": "seedvr2x Upscale Video",
    }
    for schema in schemas.values():
        assert schema.category == "seedvr2x" and schema.description
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
    models_clone = nodes.Models.PREPARE_CLASS_CLONE({io.Hidden.unique_id: "17"})
    assert models_clone.hidden is not None and models_clone.hidden.unique_id is None


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
        frames=100,
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


def test_the_text_says_what_each_auto_became_and_what_a_value_wired_in_was_taken_as() -> None:
    text = picture_text(plans.Settings(decode_tile=1024, swap_blocks=0))
    lines = text.splitlines()
    assert "attention: auto -> PyTorch's SDPA" in lines
    encode = next(each for each in lines if each.startswith("encode tile: "))
    # "encode tile: auto -> 1024 px (up to ...)", or one pass where it fits.
    assert re.fullmatch(
        r"encode tile: auto -> (\d+ px|no tiles, the whole image at once) \(needs up to"
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
    assert (
        "window length: at most 9 groups of 4 frames (about 36 frames), as you set it"
        in clipped.splitlines()
    )
    assert "encode tile: no tiles, the whole image at once (needs up to " in clipped
    assert "window length: auto -> " in clip_text(plans.Settings())
    for said in (text, pinned, clipped, clip_text(plans.Settings())):
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
    assert "the clip's output, 2.32 GiB, is taken from what the computer has" in clip
    assert "the input's frames, 2.32 GiB, are already in memory" in clip
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
