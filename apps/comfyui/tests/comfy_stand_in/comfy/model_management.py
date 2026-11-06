"""The stand-in for ComfyUI's model manager (comfy.model_management), the calls the nodes make of
it and the way it answers them, as ComfyUI v0.44.0's does, in the tests' own words: written from
the reading of that version, no ComfyUI code.

What it keeps of the real one: `current_loaded_models` holds weak references to a model's patcher
and its stub (LoadedModel); load_models_gpu registers a patcher and, for a model on a CUDA device,
frees the card down to a constant of free memory first; free_memory is asked for the free amount
wanted (not the bytes to free), visits the entries of the device in the real order (the most
offloaded first, then the fewest references, the smallest), frees each by what is short at that
moment, partly when the patcher can and whole (detach and removal) when it cannot, and never an
entry in `keep_loaded`; unload_all_models asks for everything; Cancel is a flag read by
throw_exception_if_processing_interrupted, which raises a BaseException and clears the flag.

The card is `card`: its total, what programs other than ComfyUI's models hold (`outside`) and what
the node under test holds while it runs (`running`, set by the tests' machine); the free memory is
the total less those and the bytes the loaded models say they have on it. Every call is recorded."""

import sys
import threading
import weakref
from typing import Any

import torch

GIB = 1024**3
# What a registration on a CUDA device frees the card to, whatever its size: 0.8 GiB for a
# minimum inference and 400 MiB reserved (the reading's 1.2 GiB, ComfyUI v0.44.0).
EXTRA = round(0.8 * GIB) + 400 * 1024**2


class Card:
    """The stand-in card's figures, in bytes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 32 * GIB
        self.outside = 0
        self.running = 0
        # Bytes freed by a model's partial unload that torch's cache still holds: the driver
        # counts them used until soft_empty_cache gives them back.
        self.cached = 0


card = Card()
current_loaded_models: "list[LoadedModel]" = []
DISABLE_SMART_MEMORY = False
# ComfyUI's cap on the memory it pins, in bytes: -1 where pinned memory is off (v0.44.0 sets it at
# import from the RAM and the platform). The stand-in's is large, a machine with room.
MAX_PINNED_MEMORY: float = 64 * GIB
interrupt_processing = False
# The bytes a partial unload keeps back from what it was asked to free: ComfyUI v0.44.0's reserves
# a buffer (93.84 MB in the log of the proof run) and frees to the figure less it. 0 unless a test
# sets it.
PARTIAL_BUFFER = 0
# What the tests read back.
free_calls: "list[tuple[float, Any, list[LoadedModel]]]" = []
loads: list[int] = []  # the size of each batch load_models_gpu was asked for
empties: list[int] = []
unloads: list[str] = []
device = torch.device("cuda", 0)


def reset() -> None:
    """The stand-in as new, between tests."""
    global DISABLE_SMART_MEMORY, MAX_PINNED_MEMORY, PARTIAL_BUFFER, interrupt_processing, device
    current_loaded_models.clear()
    free_calls.clear()
    loads.clear()
    empties.clear()
    unloads.clear()
    DISABLE_SMART_MEMORY = False
    MAX_PINNED_MEMORY = 64 * GIB
    PARTIAL_BUFFER = 0
    interrupt_processing = False
    device = torch.device("cuda", 0)
    card.reset()


class InterruptProcessingException(BaseException):
    pass


# ComfyUI v0.44.0 guards the flag with a re-entrant lock (model_management.py:2209-2230): a thread
# that must leave a pressed Cancel for another holds it while it reads the flag.
interrupt_processing_mutex = threading.RLock()


def interrupt_current_processing(value: bool = True) -> None:
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value


def processing_interrupted() -> bool:
    with interrupt_processing_mutex:
        return interrupt_processing


def throw_exception_if_processing_interrupted() -> None:
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()


def get_torch_device() -> Any:
    return device


class LoadedModel:
    def __init__(self, model: Any) -> None:
        self._model = weakref.ref(model)
        self.device = model.load_device
        self.real_model: weakref.ref[Any] | None = None
        self.currently_used = True
        self.model_finalizer: weakref.finalize[..., Any] | None = None

    @property
    def model(self) -> Any:
        return self._model()

    def model_memory(self) -> int:
        return self.model.model_size()

    def model_loaded_memory(self) -> int:
        return self.model.loaded_size()

    def model_offloaded_memory(self) -> int:
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, dev: Any) -> int:
        if dev == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        return self.model_memory()

    def model_load(self) -> Any:
        self.model.model_patches_to(self.device)
        self.model.partially_load(self.device, 1e32)
        real = self.model.model
        self.real_model = weakref.ref(real)
        self.model_finalizer = weakref.finalize(real, cleanup_models)
        self.model_finalizer.atexit = False
        return real

    def model_unload(
        self, memory_to_free: float | None = None, unpatch_weights: bool = True
    ) -> bool:
        if memory_to_free is not None and memory_to_free < self.model.loaded_size():
            memory_to_free = max(memory_to_free - PARTIAL_BUFFER, 0)
            freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
            if freed >= memory_to_free:
                return False
        self.model.detach(unpatch_weights)
        if self.model_finalizer is not None:
            self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LoadedModel) and self.model is other.model

    __hash__ = object.__hash__

    def is_dead(self) -> bool:
        return self.real_model is not None and self.real_model() is not None and self.model is None


def cleanup_models() -> None:
    """The entries whose stub has died, dropped."""
    current_loaded_models[:] = [
        each
        for each in current_loaded_models
        if each.real_model is None or each.real_model() is not None
    ]


# The system's RAM, which ComfyUI's cap on pinned memory is a fraction of.
SYSTEM_RAM = 32 * GIB


def get_total_memory(dev: Any = None, torch_total_too: bool = False) -> Any:
    total = SYSTEM_RAM if dev is not None and dev.type == "cpu" else card.total
    return (total, total) if torch_total_too else total


def get_free_memory(dev: Any = None, torch_free_too: bool = False) -> Any:
    held = sum(
        each.model_loaded_memory()
        for each in current_loaded_models
        if each.model is not None and (dev is None or each.device == dev)
    )
    free = card.total - card.outside - card.running - card.cached - held
    return (free, 0) if torch_free_too else free


def soft_empty_cache(force: bool = False) -> None:
    empties.append(1)
    card.cached = 0


def free_memory(
    memory_required: float, dev: Any, keep_loaded: "list[LoadedModel] | None" = None, **_: Any
) -> "list[LoadedModel]":
    kept = [] if keep_loaded is None else keep_loaded
    free_calls.append((memory_required, dev, list(kept)))
    candidates: list[tuple[int, int, int, int]] = []
    for index in range(len(current_loaded_models) - 1, -1, -1):
        each = current_loaded_models[index]
        if (dev is None or each.device == dev) and each not in kept and each.model is not None:
            candidates.append(
                (
                    -each.model_offloaded_memory(),
                    sys.getrefcount(each.model),
                    each.model_memory(),
                    index,
                )
            )
    gone: list[int] = []
    for *_rest, index in sorted(candidates):
        each = current_loaded_models[index]
        if DISABLE_SMART_MEMORY:
            short: float = 1e32
        else:
            short = 0 if dev is None else memory_required - get_free_memory(dev)
        if short > 0 and each.model_unload(short):
            gone.append(index)
    unloaded = [current_loaded_models.pop(index) for index in sorted(gone, reverse=True)]
    if unloaded:
        soft_empty_cache()
    return unloaded


def unload_all_models() -> None:
    unloads.append("all")
    free_memory(1e30, device)


def load_models_gpu(models: "list[Any]", memory_required: float = 0, **_: Any) -> None:
    loads.append(len(models))
    for patcher in models:
        loaded = LoadedModel(patcher)
        if any(each == loaded for each in current_loaded_models):
            continue
        if patcher.load_device.type == "cuda":
            required = loaded.model_memory_required(patcher.load_device) * 1.1 + EXTRA
            free_memory(required, patcher.load_device)
            if get_free_memory(patcher.load_device) < EXTRA:
                free_memory(EXTRA, patcher.load_device)
        loaded.model_load()
        current_loaded_models.insert(0, loaded)
