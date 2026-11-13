"""The stand-in for comfy_api.latest.io: the parts of ComfyUI v0.44.0's V3 schema that seedvr2x's
nodes use, and what the executor does around a node, as the tests need them.

What it keeps of the real one:
- a type (Boolean, Int, Float, String, Combo, Image, Video, and Custom(io_type) for a node's own)
  has an `Input` and an `Output` class; an input has an id, a display name, a tooltip, whether it is
  optional, and for a widget (Int, Float, Boolean, String, Combo, a WidgetInput) a default and its
  options, among them `force_input` (forceInput: a socket and no widget) and a BOOLEAN's labels;
  `as_dict` is the options dictionary the frontend reads, with every field that is None left out,
  False kept;
- a Schema lists the inputs and outputs in order (the order the frontend builds its widgets in:
  `v1_inputs`, the INPUT_TYPES the loader reads off a V3 node, keeps it), its hidden inputs, its
  category and description; an output with no id gets `_<index>_<TYPE>_`, and ids must be unique;
- a node is a class with a `define_schema` and an `execute` classmethod, both required; the
  executor runs `execute` on a clone of the class whose `hidden` holds the values the schema
  asked for (UNIQUE_ID), looked up as `cls.hidden.unique_id` (None for what is not asked), takes
  the inputs by keyword (an optional input left unwired is not given at all), and normalises what
  comes back to a NodeOutput.

What it leaves out: dynamic inputs, the other types, the clone's lock against setting an
attribute, lazy and list inputs, the UI outputs, and the validation of values."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, cast


class ControlAfterGenerate(StrEnum):
    """What the frontend's control widget next to an INT does to it after a run; given as the
    INT's `control_after_generate`, the one it starts at (a plain True starts at randomize)."""

    fixed = "fixed"
    increment = "increment"
    decrement = "decrement"
    randomize = "randomize"


class Hidden(StrEnum):
    """The values ComfyUI can hand a node beside its inputs (the V1 hidden types, by name)."""

    unique_id = "UNIQUE_ID"
    prompt = "PROMPT"
    extra_pnginfo = "EXTRA_PNGINFO"


class HiddenHolder:
    """The hidden values a run holds, `cls.hidden`: one not asked for reads as None."""

    def __init__(self, values: dict[Hidden, Any]) -> None:
        self._values = values

    @property
    def unique_id(self) -> str | None:
        return self._values.get(Hidden.unique_id)


class Input:
    """A node's input: a socket, unless the type is a widget's."""

    io_type: ClassVar[str] = ""

    def __init__(
        self,
        id: str,
        display_name: str | None = None,
        optional: bool = False,
        tooltip: str | None = None,
    ) -> None:
        self.id = id
        self.display_name = display_name
        self.optional = optional
        self.tooltip = tooltip

    def as_dict(self) -> dict[str, Any]:
        found = {"display_name": self.display_name, "tooltip": self.tooltip}
        return {key: value for key, value in found.items() if value is not None}


class WidgetInput(Input):
    """An input the frontend draws as a widget, with a default; with `force_input`, a socket and
    no widget."""

    def __init__(
        self,
        id: str,
        display_name: str | None = None,
        optional: bool = False,
        tooltip: str | None = None,
        default: Any = None,
        force_input: bool | None = None,
    ) -> None:
        super().__init__(id, display_name, optional, tooltip)
        self.default = default
        self.force_input = force_input

    def as_dict(self) -> dict[str, Any]:
        found = super().as_dict()
        if self.default is not None:
            found["default"] = self.default
        if self.force_input is not None:
            found["forceInput"] = self.force_input
        return found


class Output:
    """A node's output: its id, which is the name the frontend shows when it has no other."""

    io_type: ClassVar[str] = ""

    def __init__(
        self, id: str | None = None, display_name: str | None = None, tooltip: str | None = None
    ) -> None:
        self.id = id
        self.display_name = display_name if display_name else id
        self.tooltip = tooltip

    def as_dict(self) -> dict[str, Any]:
        found = {"display_name": self.display_name, "tooltip": self.tooltip}
        return {key: value for key, value in found.items() if value is not None}


class Boolean:
    io_type = "BOOLEAN"

    class Input(WidgetInput):
        io_type = "BOOLEAN"

        def __init__(
            self,
            id: str,
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: bool | None = None,
            label_on: str | None = None,
            label_off: str | None = None,
            force_input: bool | None = None,
        ) -> None:
            super().__init__(id, display_name, optional, tooltip, default, force_input)
            self.label_on = label_on
            self.label_off = label_off

        def as_dict(self) -> dict[str, Any]:
            found = super().as_dict()
            for key in ("label_on", "label_off"):
                if getattr(self, key) is not None:
                    found[key] = getattr(self, key)
            return found

    class Output(Output):
        io_type = "BOOLEAN"


class Int:
    io_type = "INT"

    class Input(WidgetInput):
        io_type = "INT"

        def __init__(
            self,
            id: str,
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: int | None = None,
            min: int | None = None,
            max: int | None = None,
            step: int | None = None,
            control_after_generate: bool | ControlAfterGenerate | None = None,
            force_input: bool | None = None,
        ) -> None:
            super().__init__(id, display_name, optional, tooltip, default, force_input)
            self.min = min
            self.max = max
            self.step = step
            self.control_after_generate = control_after_generate

        def as_dict(self) -> dict[str, Any]:
            found = super().as_dict()
            for key in ("min", "max", "step", "control_after_generate"):
                if getattr(self, key) is not None:
                    found[key] = getattr(self, key)
            return found

    class Output(Output):
        io_type = "INT"


class Float:
    io_type = "FLOAT"

    class Input(WidgetInput):
        io_type = "FLOAT"

        def __init__(
            self,
            id: str,
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: float | None = None,
            min: float | None = None,
            max: float | None = None,
            step: float | None = None,
            round: float | bool | None = None,
            force_input: bool | None = None,
        ) -> None:
            super().__init__(id, display_name, optional, tooltip, default, force_input)
            self.min = min
            self.max = max
            self.step = step
            self.round = round

        def as_dict(self) -> dict[str, Any]:
            found = super().as_dict()
            for key in ("min", "max", "step", "round"):
                if getattr(self, key) is not None:
                    found[key] = getattr(self, key)
            return found

    class Output(Output):
        io_type = "FLOAT"


class String:
    io_type = "STRING"

    class Input(WidgetInput):
        io_type = "STRING"

        def __init__(
            self,
            id: str,
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: str | None = None,
            multiline: bool = False,
            force_input: bool | None = None,
        ) -> None:
            super().__init__(id, display_name, optional, tooltip, default, force_input)
            self.multiline = multiline

        def as_dict(self) -> dict[str, Any]:
            return super().as_dict() | {"multiline": self.multiline}

    class Output(Output):
        io_type = "STRING"


class Combo:
    io_type = "COMBO"

    class Input(WidgetInput):
        io_type = "COMBO"

        def __init__(
            self,
            id: str,
            options: list[str],
            display_name: str | None = None,
            optional: bool = False,
            tooltip: str | None = None,
            default: str | None = None,
        ) -> None:
            super().__init__(id, display_name, optional, tooltip, default)
            self.options = options

        def as_dict(self) -> dict[str, Any]:
            return super().as_dict() | {"options": self.options}


class Image:
    io_type = "IMAGE"

    class Input(Input):
        io_type = "IMAGE"

    class Output(Output):
        io_type = "IMAGE"


class Video:
    """ComfyUI 0.45.3's VIDEO type (comfy_api/latest/_io.py:660): a VideoInput object, as Load
    Video, Create Video and the other video nodes give it (_input_impl.py here)."""

    io_type = "VIDEO"

    class Input(Input):
        io_type = "VIDEO"

    class Output(Output):
        io_type = "VIDEO"


def Custom(io_type: str) -> Any:
    """A type of the node's own, a socket: a class with an `Input` and an `Output` of that name
    (the real one gives a ComfyTypeIO class; typed Any here, as the real one's are)."""

    class CustomInput(Input):
        pass

    class CustomOutput(Output):
        pass

    CustomInput.io_type = CustomOutput.io_type = io_type
    return type(
        "CustomType", (), {"io_type": io_type, "Input": CustomInput, "Output": CustomOutput}
    )


@dataclass
class Schema:
    node_id: str
    display_name: str | None = None
    category: str = "sd"
    inputs: list[Input] = field(default_factory=list[Input])
    outputs: list[Output] = field(default_factory=list[Output])
    hidden: list[Hidden] = field(default_factory=list[Hidden])
    description: str = ""

    def finalize(self) -> None:
        """An output with no id gets one from its place and type."""
        for index, output in enumerate(self.outputs):
            if output.id is None:
                output.id = f"_{index}_{output.io_type}_"

    def validate(self) -> None:
        for each, kind in ((self.inputs, "input"), (self.outputs, "output")):
            ids = [one.id for one in each]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{kind} ids must be unique: {ids}")

    def v1_inputs(self) -> dict[str, dict[str, tuple[str, dict[str, Any]]]]:
        """The INPUT_TYPES the loader reads off the schema: each input's type and options, in the
        schema's order, under required or optional, the hidden ones by name under hidden."""
        found: dict[str, dict[str, Any]] = {"required": {}}
        for one in self.inputs:
            found.setdefault("optional" if one.optional else "required", {})[one.id] = (
                one.io_type,
                {key: value for key, value in one.as_dict().items() if key != "optional"},
            )
        for each in self.hidden:
            found.setdefault("hidden", {})[each.name] = (each.value,)
        return found


class NodeOutput:
    """What a node gives: its outputs in order."""

    def __init__(self, *args: Any, ui: Any = None) -> None:
        self.args = args
        self.ui = ui

    def __getitem__(self, index: int) -> Any:
        return self.args[index]


class ComfyNode(ABC):
    """A V3 node: a schema and an `execute`, both classmethods."""

    hidden: ClassVar[HiddenHolder | None] = None
    SCHEMA: ClassVar[Schema | None] = None

    @classmethod
    @abstractmethod
    def define_schema(cls) -> Schema: ...

    @classmethod
    @abstractmethod
    def execute(cls, **kwargs: Any) -> NodeOutput: ...

    @classmethod
    def validate_inputs(cls, **kwargs: Any) -> bool | str:
        """Optional, True or the reason a value is refused; the executor calls it at queue time
        with the constants (not the linked inputs) of the arguments it names."""
        raise NotImplementedError

    @classmethod
    def GET_SCHEMA(cls) -> Schema:
        """The schema, finalised and validated: what the loader reads for the node's id."""
        schema = cls.define_schema()
        schema.finalize()
        schema.validate()
        cls.SCHEMA = schema
        return schema

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return cls.GET_SCHEMA().v1_inputs()

    @classmethod
    def PREPARE_CLASS_CLONE(cls, hidden_inputs: dict[Hidden, Any] | None) -> type["ComfyNode"]:
        """The class the executor runs `execute` on: a copy whose `hidden` holds the values the
        schema asked for and nothing else."""
        asked = cls.GET_SCHEMA().hidden
        held = {key: value for key, value in (hidden_inputs or {}).items() if key in asked}
        return cast(type[ComfyNode], type(cls.__name__, (cls,), {"hidden": HiddenHolder(held)}))

    @classmethod
    def EXECUTE_NORMALIZED(
        cls, hidden_inputs: dict[Hidden, Any] | None = None, **inputs: Any
    ) -> NodeOutput:
        """One run as the executor makes it: the inputs must be the schema's, by name, every
        required one given and an optional one given only where it is wired; `execute` is called
        on the clone; what it returns is a NodeOutput or a tuple of the outputs."""
        schema = cls.GET_SCHEMA()
        known = {one.id for one in schema.inputs}
        required = {one.id for one in schema.inputs if not one.optional}
        if not required <= set(inputs) <= known:
            raise TypeError(
                f"{cls.__name__} takes {sorted(required)} and optionally"
                f" {sorted(known - required)}, given {sorted(inputs)}"
            )
        clone = cls.PREPARE_CLASS_CLONE(hidden_inputs)
        returned: Any = clone.execute(**inputs)
        if isinstance(returned, tuple):
            returned = NodeOutput(*cast(tuple[Any, ...], returned))
        if not isinstance(returned, NodeOutput):
            raise TypeError(f"{cls.__name__} returned {type(returned).__name__}, not a NodeOutput")
        if len(returned.args) != len(schema.outputs):
            raise ValueError(f"{cls.__name__} gave {len(returned.args)} of its outputs")
        return returned
