"""The stand-in for comfy_execution.graph_utils (ComfyUI v0.44.0): ExecutionBlocker, the value a
node gives in place of an output to block the nodes that take it. With no message they are skipped
silently; a message is reported as an error by the executor."""


class ExecutionBlocker:
    def __init__(self, message: str | None) -> None:
        self.message = message
