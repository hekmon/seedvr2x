"""The stand-in for ComfyUI's model patcher (comfy.model_patcher), the few calls the nodes' one
registration class and ComfyUI's model manager make of each other, as ComfyUI v0.44.0's patcher
answers them, in the tests' own words: written from the reading of that version, no ComfyUI code."""

from typing import Any


class ModelPatcher:
    """A patcher over a stub `model`: the six attributes ComfyUI's patcher sets on it, the sizes it
    reports, and the calls the manager makes (partially_load, partially_unload, detach, is_clone,
    model_patches_models, model_patches_to). A plain patcher is not dynamic and holds no weights
    of its own: `loaded` is the bytes the tests say it has on the card."""

    def __init__(
        self,
        model: Any,
        load_device: Any,
        offload_device: Any,
        size: int = 0,
        weight_inplace_update: bool = False,
        fast_disk: bool = False,
    ) -> None:
        self.size = size
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.parent: ModelPatcher | None = None
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        if getattr(model, "device", None) is None:
            model.device = offload_device
        for name, value in (
            ("model_loaded_weight_memory", 0),
            ("lowvram_patch_counter", 0),
            ("model_lowvram", False),
            ("current_weight_patches_uuid", None),
            ("model_offload_buffer_memory", 0),
        ):
            if not hasattr(model, name):
                setattr(model, name, value)

    def is_dynamic(self) -> bool:
        return False

    def model_size(self) -> int:
        return self.size

    def loaded_size(self) -> int:
        return self.model.model_loaded_weight_memory

    def current_loaded_device(self) -> Any:
        return self.model.device

    def is_clone(self, other: Any) -> bool:
        return hasattr(other, "model") and self.model is other.model

    def model_patches_models(self) -> list[Any]:
        return []

    def model_patches_to(self, device: Any) -> None:
        return None

    def model_dtype(self) -> Any:
        return None

    def partially_load(
        self, device_to: Any, extra_memory: float = 0, force_patch_weights: bool = False
    ) -> int:
        """The base brings weights to the device up to `extra_memory`: here the bytes the stub
        says it has on the card are the model's `size`, brought whole."""
        self.model.model_loaded_weight_memory = self.size
        self.model.device = device_to
        return self.size

    def partially_unload(
        self, device_to: Any, memory_to_free: float = 0, force_patch_weights: bool = False
    ) -> int:
        """Some of the bytes on the card sent away, at most what is asked: the stub gives back
        the whole of a request under what it holds."""
        freed = min(int(memory_to_free), self.model.model_loaded_weight_memory)
        self.model.model_loaded_weight_memory -= freed
        return freed

    def detach(self, unpatch_all: bool = True) -> Any:
        self.model.model_loaded_weight_memory = 0
        self.model.device = self.offload_device
        return self.model

    def partially_unload_ram(self, ram_to_unload: float) -> int:
        return 0

    def pinned_memory_size(self) -> int:
        return 0

    def loaded_ram_size(self) -> int:
        return 0
