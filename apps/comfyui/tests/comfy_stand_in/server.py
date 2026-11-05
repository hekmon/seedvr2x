"""The stand-in for ComfyUI's server module: the one call a node makes of it, the text shown on a
node while it runs (PromptServer.instance.send_progress_text), recorded."""


class PromptServer:
    instance: "PromptServer | None" = None

    def __init__(self) -> None:
        self.texts: list[tuple[str, str | None]] = []

    def send_progress_text(self, text: str, node_id: str | None) -> None:
        self.texts.append((text, node_id))
