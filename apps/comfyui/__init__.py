"""seedvr2x's nodes for ComfyUI: seedvr2x Models, Cuts, Upscale Image and Upscale Video, which run
SeedVR2 with seedvr2x's own loaders and placement (the library, `seedvr2x`, installed in ComfyUI's
Python environment), and the value nodes of the plan's settings (Tile, Blocks Swapped, Window
Length).

ComfyUI imports this folder from its custom_nodes directory, whatever the folder is called: it
keys a custom node by the folder's full path and loads its `__init__.py`. The folder is not on
sys.path while it is imported, so the modules import each other by relative imports.

The registration is V3, ComfyUI's current schema (`comfy_api.latest`: `io.ComfyNode`,
`io.Schema`, `comfy_entrypoint`; the user, 2026-11-05): ComfyUI's own nodes use it, and its
documentation says new node features come to it alone, the older dictionaries
(NODE_CLASS_MAPPINGS) still loading. ComfyUI v0.44.0 marks `comfy_api.latest` unstable
(`STABLE = False`), which is one more reason for the list of versions the nodes are tested
against (README.md). The loader looks for NODE_CLASS_MAPPINGS first and for `comfy_entrypoint`
only where it finds none, so this module defines the entrypoint alone. The nodes' ids, once
published, are kept: a saved workflow names them.
"""

import logging

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]

# Said once, when ComfyUI loads the nodes, never at a run (the user, 2026-11-06): where the nodes
# stop and the command line begins. The repository's folder and its README, no address.
logging.getLogger("seedvr2x_comfyui").info(
    "seedvr2x's nodes hold a clip whole in RAM, as ComfyUI does. For a whole film, seedvr2x's"
    " command line is the tool: it reads a film shot by shot, resumes after a stop and writes"
    " lossless files. It is in this repository's apps/seedvr2x folder, beside these nodes, and"
    " its README says how to use it."
)
