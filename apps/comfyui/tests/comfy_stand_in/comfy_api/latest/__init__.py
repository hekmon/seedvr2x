"""The stand-in for ComfyUI v0.44.0's comfy_api.latest, the V3 schema namespace: `io` (the types,
Schema, ComfyNode, NodeOutput, Hidden) and ComfyExtension, the two names a node file imports.
Written from the reading of that version, in the tests' own words: no ComfyUI code."""

from abc import ABC, abstractmethod

from . import _io as io


class ComfyExtension(ABC):
    """What `comfy_entrypoint` returns: ComfyUI's loader awaits `on_load` (nothing by default),
    then `get_node_list`, and registers each class under its schema's node_id."""

    async def on_load(self) -> None:
        return None

    @abstractmethod
    async def get_node_list(self) -> list[type[io.ComfyNode]]: ...


__all__ = ["ComfyExtension", "io"]
