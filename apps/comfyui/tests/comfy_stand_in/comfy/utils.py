"""The stand-in for ComfyUI's comfy.utils: the progress bar a node moves, recording what it is
told (ComfyUI v0.44.0's ProgressBar(total, node_id=None) with update_absolute and update)."""

from collections.abc import Callable
from typing import ClassVar


class ProgressBar:
    created: ClassVar["list[ProgressBar]"] = []
    # Called with each value the bar is moved to, after it is recorded: the tests' Cancel.
    on_update: ClassVar["Callable[[float], None] | None"] = None

    def __init__(self, total: float, node_id: str | None = None) -> None:
        self.total = total
        self.node_id = node_id
        self.current = 0.0
        self.updates: list[float] = []
        ProgressBar.created.append(self)

    def update_absolute(
        self, value: float, total: float | None = None, preview: object = None
    ) -> None:
        if total is not None:
            self.total = total
        self.current = value
        self.updates.append(value)
        if ProgressBar.on_update is not None:
            ProgressBar.on_update(value)

    def update(self, value: float) -> None:
        self.update_absolute(self.current + value)
