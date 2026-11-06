"""The stand-in for ComfyUI's comfy.utils: the progress bar a node moves, recording what it is
told (ComfyUI v0.44.0's ProgressBar(total, node_id=None) with update_absolute(value, total,
preview) and update): the values, the totals they came with and the previews, a tuple (format, a
PIL image, the largest side) as the sampler nodes send it."""

from collections.abc import Callable
from typing import ClassVar

from comfy import model_management as management


class ProgressBar:
    created: ClassVar["list[ProgressBar]"] = []
    # Called with each value the bar is moved to, after it is recorded: the tests' Cancel.
    on_update: ClassVar["Callable[[float], None] | None"] = None

    def __init__(self, total: float, node_id: str | None = None) -> None:
        self.total = total
        self.node_id = node_id
        self.current = 0.0
        self.updates: list[float] = []
        self.totals: list[float] = []
        self.previews: list[tuple[float, object]] = []
        ProgressBar.created.append(self)

    def update_absolute(
        self, value: float, total: float | None = None, preview: object = None
    ) -> None:
        # The hook ComfyUI installs checks Cancel first (main.py:467): it clears the flag and
        # raises, on whatever thread reports, before anything is sent.
        management.throw_exception_if_processing_interrupted()
        if total is not None:
            self.total = total
        self.current = value
        self.updates.append(value)
        self.totals.append(self.total)
        if preview is not None:
            self.previews.append((value, preview))
        if ProgressBar.on_update is not None:
            ProgressBar.on_update(value)

    def update(self, value: float) -> None:
        self.update_absolute(self.current + value)
