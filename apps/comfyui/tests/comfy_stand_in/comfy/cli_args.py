"""The stand-in for ComfyUI's comfy.cli_args: the one option a node reads, the largest side of a
preview (--preview-size, 512 by default in v0.44.0), which a test sets."""

from types import SimpleNamespace

args = SimpleNamespace(preview_size=512)
