"""The models of a node run (DESIGN.md, Repository and apps, the ComfyUI nodes, "seedvr2x's own
loaders and placement"): found and checked as the CLI and the server check them, loaded with the
library, kept in pinned host memory between runs or read for one run, registered with ComfyUI's
model manager for that memory alone, and put on the card only while a node runs, by a plan made
for the card as it would be with ComfyUI's own models gone.

The node classes (V3) hold no state, `__init__` being ignored and the class sanitised before
`execute`: what lives between runs is here, the registry `_entries` (the kept weights, pinned in
host memory, one set at a time) with the notes of the weights ComfyUI dropped (`_dropped`) and
the probes done (`_probed`). The Models node's output, the `Handle`, is light and cached by
ComfyUI as it likes.

One run is `session`, a context manager both upscale nodes use:
0. whether the weights are kept (`decide_keeping`: auto keeps them where they fit under ComfyUI's
   own cap on pinned memory, on keeps them whatever the cap, off does not), said in the
   console and in the plan; where they are not, the kept set of an earlier run is freed before
   this run reads its own, and a kept set of another file before the new one is pinned;
1. the files found and checked (pull.resolve, weights.check_models, pull.check_pinned: their
   header, then each pinned file's size, its SHA-256 left to the load, which proves it), and the
   environment probe (probe.check) for that DiT layout, once per process;
2. the models: kept ones are read once, every weight in its pinned copy and none on the card
   (passes.HELD), after a plan for the card alone refused nothing (a refusal comes before a
   16 GB read), and registered with ComfyUI; ones not kept are read for the run, after its plan;
3. the card read as ComfyUI leaves it, and what ComfyUI's own models hold of it added: the plan
   is made for the card as it would be without them;
4. ComfyUI asked to free only what the plan needs, and only if the card has less free (`room`);
5. the models placed (Models.place), the pass run by the caller; torch's memory limit is not
   set (CudaMachine's docstring);
   A plan only (`plan_only`) stops after 3: nothing is probed, read whole, registered or placed,
   and ComfyUI is asked to free nothing;
6. in every case, the end, a Cancel or an error, the card freed: the weights back in their
   pinned copies (Models.release) or dropped (after model.forget, and the exception's frames
   cleared of what they hold), torch's cache emptied, the global random generators as they were.

The card, the load and the device are behind `Machine`, the real one on torch.cuda (CudaMachine),
the tests' a stand-in on the CPU. ComfyUI's own calls are made on `comfy.model_management`, which
the tests stand in for (tests/comfy_stand_in), each of them described where it is made."""

import gc
import logging
import os
import random
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import comfy.model_management as management
import numpy as np
import torch
from comfy.model_patcher import ModelPatcher

from seedvr2x import blockswap, memory, model, pull, waiting, weights
from seedvr2x.errors import ModelError, ModelMissing, ModelsRefused, PlanRefused, Refusal
from seedvr2x.memory import Reading
from seedvr2x.model import Models
from seedvr2x.passes import HELD, ModelPlan
from seedvr2x.pull import ModelFiles
from seedvr2x.weights import Architecture

from . import console, plans, words
from .progress import (
    FAILED,
    FREEING,
    FREEING_SAID,
    STOPPED,
    STOPPED_SAID,
    Preparing,
    file_size,
)

logger = logging.getLogger("seedvr2x_comfyui.models")


@dataclass(frozen=True)
class Handle:
    """What seedvr2x Models gives and the upscale nodes take (the SEEDVR2X_MODELS type): the DiT's
    file by its pinned name, the directory the files are read from ("" for Hugging Face's cache)
    and whether the weights stay in RAM between runs (plans.KEEP_MODES: "auto" keeps them where
    they fit under ComfyUI's cap on pinned memory). A light handle, nothing of the weights: they
    are the registry's (`_entries`), ComfyUI's caches dropping a node's output without dropping
    them."""

    dit: str
    model_dir: str = ""
    keep: str = "auto"

    @property
    def directory(self) -> Path | None:
        return Path(self.model_dir) if self.model_dir else None

    @property
    def key(self) -> tuple[str, str]:
        """What the kept weights are filed under: the files, not the mode that keeps them."""
        return (self.dit, self.model_dir)


def pinned_cap() -> tuple[int, str]:
    """ComfyUI's cap on pinned memory in bytes, and where it was read from: the manager's own
    MAX_PINNED_MEMORY where it exposes one (v0.44.0 sets it at import: 40% of the RAM on Windows,
    else the larger of 40% and the smaller of 90%, the RAM less 4 GiB and the RAM plus the disk
    swap less 16 GiB; -1 where pinned memory is off, an integrated GPU or --disable-pinned-memory),
    else that rule without the swap, which only lowers it (said). The weights kept in RAM are
    pinned, and ComfyUI's own models are held to the same cap."""
    found = getattr(management, "MAX_PINNED_MEMORY", None)
    if found is not None:
        return int(found), "ComfyUI's MAX_PINNED_MEMORY"
    ram = int(management.get_total_memory(torch.device("cpu")))
    if getattr(management, "WINDOWS", False):
        cap = int(ram * 0.40)
    else:
        cap = int(max(ram * 0.40, min(ram * 0.90, ram - 4 * memory.GIB)))
    logger.warning(
        "ComfyUI exposes no MAX_PINNED_MEMORY: its rule is applied to the RAM, less the disk swap"
        " it also counts, which can only lower it"
    )
    return cap, "ComfyUI's rule for its cap, as MAX_PINNED_MEMORY is not exposed"


def pinned_size(handle: Handle) -> int:
    """The bytes the kept weights pin, known before any file is read: the DiT's and the VAE's
    files (the weights are pinned whole, passes.HELD) by their pins' sizes, a file seedvr2x does
    not pin by its size on disk in the model directory (0 where it is not there)."""
    total = 0
    for name in (handle.dit, plans.VAE_FILE):
        pin = pull.PINNED.get(name)
        if pin is not None:
            total += pin.size
        elif handle.directory is not None and (handle.directory / name).is_file():
            total += (handle.directory / name).stat().st_size
    return total


def decide_keeping(handle: Handle) -> plans.Keeping:
    """Whether the run keeps the weights in RAM: "on" keeps them whatever the cap (said), "off"
    does not, "auto" keeps them where the set fits under ComfyUI's cap on pinned memory. A kept
    set already in the registry is counted by what it pins. ComfyUI run with --disable-smart-memory
    unloads every model after each prompt, so nothing is kept (said)."""
    mode = handle.keep
    entry = _entries.get(handle.key)
    size = entry.host if entry is not None and entry.models is not None else pinned_size(handle)
    cap, source = pinned_cap()
    if mode == "off":
        kept = False
    elif mode == "on":
        kept = True
    else:
        kept = 0 < cap and size <= cap
    because = ""
    if kept and management.DISABLE_SMART_MEMORY:
        because = (
            "ComfyUI runs with --disable-smart-memory, which unloads every model after each"
            " prompt: the models are read again for each run"
        )
        _once("smart-memory", "%s", because)
        kept = False
    return plans.Keeping(mode, kept, size, cap, source, because)


# ---- The card and the machine behind a run -------------------------------------------------------


class Machine(Protocol):
    """What a run reads of the machine and does on its card: the real one (CudaMachine) and the
    tests' on the CPU. `device` is the card, the one ComfyUI loads its models on."""

    device: Any

    def name(self) -> str: ...
    def spills(self) -> bool: ...
    def allocator(self) -> str: ...
    def free(self) -> int: ...
    def reading(self, free: int) -> Reading: ...
    def host(self) -> int | None: ...
    def dit_model(self, path: Path) -> Architecture: ...
    def probe(self, architecture: Architecture) -> float | None: ...
    def load(
        self,
        files: ModelFiles,
        plan: ModelPlan,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> Models: ...


class CudaMachine:
    """The card ComfyUI runs on, through torch: the library's readings (memory, blockswap, model,
    probe) in ComfyUI's process, which is not seedvr2x's. memory.startup finds whether the driver
    lends system memory and what allocator torch runs, as the CLI's and the server's runs ask it
    once, with torch's allocator setting as it is, not as startup would choose it (_startup).

    It sets no limit on torch's memory (the CLI's and the server's runs hold themselves to the
    card where the driver lends system memory, memory.hold, by torch's per-process memory
    fraction). In ComfyUI's process that cannot be undone: the allocator keeps a fraction once
    set, and 1.0 is a cap at the card's total, not no limit, where before the first node the
    process could spill into system memory. And under cudaMallocAsync, ComfyUI's default on its
    CUDA builds, the limit does not hold anyway: a run went 2.06 GiB past it (DESIGN.md,
    Allocator). The plan keeps the reading's reserve (Needs), and the allocator note says the way
    to torch's own allocator."""

    def __init__(self) -> None:
        device = management.get_torch_device()
        if getattr(device, "type", "") != "cuda":
            raise words.no_cuda("ComfyUI is running on the CPU")
        if not torch.cuda.is_available():
            raise words.no_cuda("torch finds no CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise words.no_cuda(
                "this graphics card can't compute in bfloat16, the number format the model runs in"
            )
        self.torch = torch
        self.device = device
        _startup()

    def name(self) -> str:
        return str(self.torch.cuda.get_device_name(self.device))

    def spills(self) -> bool:
        return memory.spills()

    def allocator(self) -> str:
        return str(self.torch.cuda.get_allocator_backend())

    def free(self) -> int:
        """The driver's free memory with a first allocation and the convolution's and the matrix
        product's libraries loaded, as the CLI reads it (apps/seedvr2x/main.py, _free_memory: the
        planner's margin was bisected on such readings)."""
        torch = self.torch
        first = torch.empty(1, device=self.device)
        volume = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32, device=self.device)
        convolved = torch.nn.functional.conv3d(volume, volume)
        matrix = torch.zeros(2, 2, dtype=torch.float32, device=self.device)
        product = torch.mm(matrix, matrix)
        free = int(torch.cuda.mem_get_info(self.device)[0])
        del first, convolved, product
        return free

    def reading(self, free: int) -> Reading:
        return memory.reading(free)

    def host(self) -> int | None:
        found = blockswap.available()
        if found is None:
            logger.warning("the system memory available is unread: the plan doesn't check it")
        return found

    def dit_model(self, path: Path) -> Architecture:
        return weights.check(path, "dit")

    def probe(self, architecture: Architecture) -> float | None:
        from seedvr2x import probe

        return probe.check(self.device, architecture).allocator_margin

    def load(
        self,
        files: ModelFiles,
        plan: ModelPlan,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> Models:
        return model.load_models(
            files.dit.path, files.vae.path, self.device, files.hashes, plan, progress=progress
        )


# The start of the library's warnings that are the command line's words (memory.startup): about an
# allocator "of the environment's", worded for a user who set it, which in ComfyUI's process is the
# allocator ComfyUI chose (the backend torch reports is what _startup gives startup to read); that
# seedvr2x holds itself to the card by a limit on torch, which the nodes never set (the limit
# cannot be taken back in a process shared with other nodes); and the driver that lends system
# memory (memory.LENDS), worded for a run that passes the card, which the nodes' plan keeps from
# (the user, 2026-11-08: the nodes' own INFO line alone says it), and WSL2's driver that didn't
# answer, taken for one that lends (memory.UNANSWERED), the same finding in other words. The nodes
# say their own words, once (_allocator_note).
_COMMAND_LINE_WORDS = (
    "the allocator is the environment's",
    "seedvr2x holds itself to the card's own memory there",
    memory.LENDS,
    memory.UNANSWERED,
)


class _CommandLineWords(logging.Filter):
    """Drops the library's warnings that describe a limit, a setting or a risk the nodes don't
    have, by the message made (LENDS is logged through "%s")."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_COMMAND_LINE_WORDS)


def _startup() -> None:
    """memory.startup, in a process whose allocator is already chosen. It reads the allocator's
    setting from the environment, and sets its own default there when there is none (backend:native
    with expandable segments where the driver lends system memory): in ComfyUI's process torch
    runs what ComfyUI chose, so an environment with no setting is given the backend torch reports,
    for startup to read, and left as it was after (DESIGN.md, Repository and apps: "ComfyUI's
    process is not seedvr2x's"). The warnings it gives about that allocator and the limit on
    torch are the command line's: left out, the nodes' own words say what applies
    (_allocator_note)."""
    names = memory.VARIABLES
    before = {name: os.environ.get(name) for name in names}
    if not any(before.values()):
        os.environ[names[0]] = f"backend:{torch.cuda.get_allocator_backend()}"
    words_off = _CommandLineWords()
    library = logging.getLogger(memory.logger.name)
    library.addFilter(words_off)
    try:
        memory.startup()
    finally:
        library.removeFilter(words_off)
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_machine: Machine | None = None


def machine() -> Machine:
    """The machine the run uses: the card, made at the first run."""
    global _machine
    if _machine is None:
        _machine = CudaMachine()
    return _machine


def use(replacement: Machine | None) -> None:
    """Replace the machine (None: the card again at the next run), for the tests."""
    global _machine
    _machine = replacement


# ---- What ComfyUI is told about the memory held between runs -------------------------------------


class _Stub:
    """The model ComfyUI's patcher wraps: nothing, but the attributes the patcher sets on its model
    and the weak reference ComfyUI's manager takes of it (a plain class is weak-referenceable)."""

    device: Any = None


# ComfyUI's console names a model by its class's name (v0.44.0, model_management.py's
# load_models_gpu: "Requested to load %s"): "Requested to load seedvr2x", not "_Stub".
_Stub.__name__ = _Stub.__qualname__ = "seedvr2x"


class Registration(ModelPatcher):
    """seedvr2x's one registration with ComfyUI's model manager: a patcher over a stub that holds
    nothing, so that ComfyUI counts the node's memory and can free it (its "Unload models" button,
    POST /free, an out-of-memory error, `free_memory` when its own models need the room). It is
    honest about the card: size 0, nothing loaded, no partial unload, so that ComfyUI never
    believes it frees VRAM by it; what it holds is host memory, which `loaded_ram_size` says.
    Its `load_device` is the card (else ComfyUI's unload loops, which visit the CUDA devices, never
    reach it), its offload device the CPU.

    ComfyUI's detach, the only way it frees the entry whole, drops the weights (Entry.drop): the
    next run finds them gone and loads them again. A model of ComfyUI's is dropped by detach
    whenever `free_memory` asks for more than a partial unload gives; with nothing loaded here,
    any shortage of VRAM in a batch that is not all dynamic drops this entry too, freeing no VRAM
    (read in v0.44.0, "spurious eviction"): the nodes' own calls protect it with `keep_loaded`, a
    foreign one's drop costs a read of the files."""

    def __init__(self, entry: "Entry", load_device: Any, offload_device: Any) -> None:
        super().__init__(_Stub(), load_device, offload_device)
        self.entry = entry

    def is_dynamic(self) -> bool:
        return False

    def model_size(self) -> int:
        return 0

    def loaded_size(self) -> int:
        return 0

    def partially_load(
        self, device_to: Any, extra_memory: float = 0, force_patch_weights: bool = False
    ) -> int:
        """Nothing is brought to the card: the weights wait in pinned host memory until a run."""
        return 0

    def partially_unload(
        self, device_to: Any, memory_to_free: float = 0, force_patch_weights: bool = False
    ) -> int:
        return 0

    def loaded_ram_size(self) -> int:
        return self.entry.host

    def detach(self, unpatch_all: bool = True) -> Any:
        self.entry.drop("ComfyUI unloaded it", by_comfy=True)
        return self.model


@dataclass(eq=False)
class Entry:
    """The models kept between runs: loaded with passes.HELD, registered with ComfyUI, held by this
    module alone (the registry), with the references to the patcher and its stub that ComfyUI's
    manager, which holds weak ones only, relies on."""

    handle: Handle
    files: ModelFiles
    architecture: Architecture
    models: Models | None
    host: int = 0  # bytes of pinned host memory the weights take
    busy: bool = False  # a run is using it: never dropped under it
    patcher: Registration | None = field(default=None, repr=False)

    def drop(self, why: str, by_comfy: bool = False) -> None:
        """The weights given up: the models deleted (their pinned copies freed), the entry
        forgotten. Not while a run uses it: ComfyUI's nodes run one at a time, so that a drop
        under a run would be a bug, logged and refused."""
        if self.busy:
            logger.warning("seedvr2x's models are in use by a run: %s, and are kept", why)
            return
        logger.info("seedvr2x's models are dropped (%s): the next run reads the files again", why)
        lost = self.models is not None and waiting.abandoned(model.card_index(self.models.device))
        if lost and self.models is not None:
            # After an abandoned wait the finalizers of the pinned copies (cudaHostUnregister)
            # would wait on the card: held, not collected (_left_on_card).
            _hold(self.models)
        self.models = None
        if _entries.get(self.handle.key) is self:
            del _entries[self.handle.key]
        if by_comfy:
            _dropped.add(self.handle.key)
        if not lost:
            gc.collect()


_entries: dict[tuple[str, str], Entry] = {}
# The probes done, by DiT layout, and the allocator's margin each found (a
# share of a step's P, probe.Probed.allocator_margin; None: none measured, the measurement being
# unbuilt), which the plans of this process count with the rule's share, the larger of the two.
_probed: dict[str, float | None] = {}

# The models of runs not kept that ended in an abandoned wait on the GPU (waiting.abandoned),
# held on purpose: dropping them runs their finalizers (blockswap.HostCopy's cudaHostUnregister),
# which wait on the card that stopped answering, as the hang of 2026-11-06 did (DESIGN.md,
# Allocator). The process is for restarting.
_left_on_card: list[Models] = []
# The CUDA errors other than out of memory that failed a card (waiting.fail), each with its
# traceback, whose frames hold the tensors the run had on the card: kept for the process's life,
# since their free would report the error again, which under cudaMallocAsync aborts the process
# (the nodes on Windows, 2026-11-07: "Fatal Python error: Aborted" at a free).
_failures: list[BaseException] = []


def _hold(models: Models) -> None:
    """Keep `models` alive for the process (_left_on_card), once."""
    if not any(each is models for each in _left_on_card):
        _left_on_card.append(models)


# The handles whose weights were dropped (Entry.drop): the next load of one says so.
_dropped: set[tuple[str, str]] = set()
_told: set[str] = set()


def reset() -> None:
    """The module as new (the tests): the entries dropped, the probe and the notes forgotten, the
    models left on the card by an abandoned wait let go."""
    for entry in list(_entries.values()):
        entry.models = None
    _entries.clear()
    _dropped.clear()
    _probed.clear()
    _told.clear()
    _left_on_card.clear()
    _failures.clear()


def kept(handle: Handle) -> Entry | None:
    """The entry that holds this handle's models, if ComfyUI hasn't dropped it."""
    entry = _entries.get(handle.key)
    return entry if entry is not None and entry.models is not None else None


def _once(key: str, text: str, *args: object) -> None:
    """A warning said once per process."""
    if key not in _told:
        _told.add(key)
        logger.warning(text, *args)


# ---- A run ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Needs:
    """What a plan asks of the card, in bytes, for the room: its largest phase (a VAE pass with
    the room it is held to), the margin it leaves and the reading it was made in."""

    peak: int
    margin: int
    reading: Reading


@dataclass(frozen=True)
class Planned[T]:
    """A plan made by a node's planner: the plan itself (T), the models' placement and its needs."""

    plan: T
    models: ModelPlan
    needs: Needs


# A node's planner: the plan for a card as it would be with ComfyUI's models gone (the reading),
# the system memory available (None: unchecked), the DiT's format and
# whether the weights' pinned copies exist already. PlanRefused as the library raises it.
type Planner[T] = Callable[[Reading, int | None, Architecture, bool], Planned[T]]


class Run[T]:
    """What a node gets in a session: the models placed for its plan, the plan, the room it took,
    the handle, what keeping the weights came to (`keeping`) and the
    notes to say in the plan once (`notes`: ComfyUI having dropped the weights). The models are
    the session's: over when it ends, they are no longer reachable from here, so that a node still
    holding its Run after the `with` keeps no weight alive. A plan only has none: `models` is
    refused, the files having been read for their headers alone."""

    def __init__(
        self,
        models: Models | None,
        planned: Planned[T],
        room: plans.Room,
        handle: Handle,
        keeping: plans.Keeping,
        notes: tuple[str, ...] = (),
        held: int = 0,
    ) -> None:
        self._models: Models | None = models
        self.planned = planned
        self.room = room
        self.handle = handle
        self.keeping = keeping
        self.notes = notes
        self.held = held  # bytes of the weights already pinned when the plan was made

    @property
    def models(self) -> Models:
        if self._models is None:
            raise RuntimeError("the run is over, or was a plan only: no models are placed")
        return self._models

    def over(self) -> None:
        self._models = None


def _taken(reading: Reading) -> int:
    """The free memory a reading counts from: the smaller of the driver's figure and the whole
    card's where that was read (memory.reading)."""
    return reading.reported if reading.card is None else min(reading.reported, reading.card)


def room_beyond(reading: Reading, reserve: int) -> int:
    """What ComfyUI's `reserve` asks of the card beyond what the reading already holds back
    (`reading.left`, torch's limit on a held run), in bytes. The room left to the rest of the card
    is the larger of the two, one need (DESIGN.md, Allocator, Windows itself): the plan counts
    `reading.left` itself (memory.counted), and this is the rest of the room. 0 where seedvr2x's
    least is the larger; the whole reserve where the run is not held (`left` is 0 there)."""
    return max(reserve - reading.left, 0)


def with_comfy_gone(reading: Reading, held: int, reserve: int = 0) -> Reading:
    """`reading` with the `held` bytes ComfyUI's own models have on the card counted free and the
    room left to the rest of the card counted gone: the reading the plan is made in, for the card
    as it would be without those models and with that room left. The room is the larger of the
    `reserve` ComfyUI keeps for the system and the reading's own `left` (room_beyond): `left` is
    taken off by memory.counted, so each figure of the reading moves by `held` less what the
    reserve asks beyond it (never below 0), and a held run's counted memory is taken again from
    them: the driver's and the card's figures less what torch is held back by (the room) and,
    under torch's own allocator without the segments, RESERVE besides."""
    delta = held - room_beyond(reading, reserve)
    if delta == 0:
        return reading
    if not reading.held:
        return replace(
            reading,
            reported=max(reading.reported + delta, 0),
            counted=max(reading.counted + delta, 0),
        )
    reported = max(reading.reported + delta, 0)
    card = None if reading.card is None else max(reading.card + delta, 0)
    taken = reported if card is None else min(reported, card)
    return replace(
        reading,
        reported=reported,
        card=card,
        counted=memory.counted(taken, reading),
    )


def system_reserve() -> int:
    """The bytes ComfyUI keeps of the card for the system and other programs, read at each run
    (ComfyUI's startup option `--reserve-vram` replaces its default): its own loads leave them
    free, and the plan does too."""
    return max(int(management.extra_reserved_memory()), 0)


def comfy_holds(device: Any, ours: Registration | None) -> int:
    """The bytes ComfyUI's own models say they have on `device`: the sum of the loaded size of
    every entry of its manager on it, this registration's excluded (it holds none)."""
    held = 0
    for each in management.current_loaded_models:
        patcher = each.model
        if patcher is None or each.device != device or patcher is ours:
            continue
        held += each.model_loaded_memory()
    return held


def _read(
    machine: Machine, ours: Registration | None, margin: float | None = None
) -> tuple[Reading, Reading, int, int]:
    """The card as the plan counts it: ComfyUI's cache of freed blocks given back first (what
    torch keeps cached and unused is free by definition, and the driver's figure leaves it out),
    then the driver's free memory read as the CLI reads it, ComfyUI's models added to it and the
    room ComfyUI keeps for the system taken off (`system_reserve`).
    `margin`: the allocator's margin the probe found, which the plan's reading carries
    (Reading.margin) so that its reserves count the larger of it and the rule's share.
    Returns (the reading for the plan, the card's own reading, what ComfyUI's models hold, the
    room kept for the system)."""
    management.soft_empty_cache()
    own = machine.reading(machine.free())
    held = comfy_holds(machine.device, ours)
    reserve = system_reserve()
    return with_comfy_gone(replace(own, margin=margin), held, reserve), own, held, reserve


def _prepare(
    handle: Handle, machine: Machine, preparing: Preparing | None = None
) -> tuple[ModelFiles, Architecture]:
    """The model files found and checked as the server checks them (service.check_files): the
    directory or Hugging Face's cache, pulled at the pinned revision where missing, their headers
    checked, each pinned file's size checked and its SHA-256 left to the load. The detector is not
    needed here (the Cuts node loads it). A refusal is worded for the nodes (words.refusal). The
    bytes of a download are told to `preparing`, which shows them on the node."""
    try:
        if handle.directory is None:
            # The Models node fetched what was missing when it ran, and ComfyUI keeps its output
            # between prompts: a file the cache has lost since is fetched here, said before it is.
            for name in (handle.dit, plans.VAE_FILE):
                if name in pull.PINNED and pull.cached(name) is None:
                    logger.info(
                        "%s is no longer in Hugging Face's cache: downloading it again",
                        plans.short_label(name),
                    )
        files = pull.resolve(
            handle.directory,
            handle.dit,
            plans.VAE_FILE,
            detector=False,
            progress=None if preparing is None else preparing.download,
        )
        if preparing is not None:
            # A Cancel pressed during a download that reported from another thread is still
            # pending: honoured here, on the node's own thread (progress.Preparing).
            management.throw_exception_if_processing_interrupted()
        dit, vae = (None if each.absent else each.path for each in (files.dit, files.vae))
        weights.check_models(dit, vae, pull.advice(files), None)
        pull.check_pinned(files, load=True)
        assert dit is not None
        return files, machine.dit_model(dit)
    except Refusal as error:
        raise words.refusal(error, model_dir=bool(handle.model_dir)) from error


def model_folder(value: str | None, node: str = words.MODELS) -> str:
    """The model folder the Models node's optional input gave, as the Handle holds it: "" for
    Hugging Face's cache. Whitespace and one pair of double quotes around the value are taken off
    (Windows' "Copy as path" adds the quotes); a value empty after that is the same as an unwired
    input, said at INFO (an empty string from a primitive node would otherwise read as a folder
    named ""); a "~" that begins it is the user's home folder (os.path.expanduser's rule: "~" or
    "~user" before the first separator; the home of USERPROFILE on Windows, HOME elsewhere), as
    a shell would have it; one that is no existing folder is refused (words.not_a_folder), naming
    `node`."""
    if value is None:
        return ""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    if not text:
        logger.info("the model folder given is empty: Hugging Face's cache is used")
        return ""
    text = os.path.expanduser(text)
    if not Path(text).is_dir():
        raise words.not_a_folder(text, node)
    return text


def detector_folder(handle: Handle) -> Path | None:
    """The folder the shot detector's file is read from: the handle's model folder where it holds
    the file under its name, else None, Hugging Face's cache. The one place that decides it (the
    Models node provides the file and the Cuts node loads it by this), because a folder is never
    written into: where it lacks the file, the cache's is used and fetched there when missing."""
    folder = handle.directory
    if folder is not None and (folder / pull.DETECTOR).is_file():
        return folder
    return None


def said_where(folder: Path | None, names: tuple[str, ...], downloaded: set[str]) -> None:
    """One INFO line for each of the model files, where it is and how large: a file of the cache
    by its pin's size (the cache holds nothing else under seedvr2x's names), a file of the model
    folder `folder` by its own (one of another size is refused by its pin when the upscale node
    checks it): the cache's directory, which HF_HOME moves, or the folder. A file the folder lacks
    is the cache's (the shot detector's, detector_folder)."""
    for name in names:
        pin = pull.PINNED.get(name)
        if folder is not None and (folder / name).is_file():
            size = (folder / name).stat().st_size
            where = f"in the model folder: {folder}"
        else:
            size = 0 if pin is None else pin.size
            how = "downloaded into" if name in downloaded else "in"
            where = f"{how} Hugging Face's cache: {pull.cache()}; HF_HOME moves it"
        logger.info("%s, %s, %s", plans.short_label(name), plans.size_text(size), where)


def provide(handle: Handle, show: Callable[[str], None]) -> None:
    """The files the Models node's choice needs, there when it has run: the DiT's, the VAE's and
    the shot detector's. With a model folder, the DiT's file and the VAE's present in it, nothing
    read; without, what Hugging Face's cache lacks of them downloaded into it. The detector's is
    the folder's where it has it, else Hugging Face's cache's, downloaded there when missing
    (detector_folder). The bytes on a bar of the node's own, the downloads' sizes all known from
    the first (`expected`), its lines told to `show`; ComfyUI's Cancel honoured as the upscale
    nodes' downloads are (progress.Preparing). Where each is is said in the console. The files'
    headers and pins are checked by the nodes that read them (`_prepare`, the Cuts node). A
    refusal is worded for the nodes (words.refusal), the Models node's.

    The detector's file, which seedvr2x Cuts alone reads, holds no other node up: where it can't
    be had (offline, the Hub out of reach, no room), the node says so at WARNING and gives its
    handle, and Cuts fetches it when it runs (review M2 of impl-nodes-6: an offline folder of the
    DiT and the VAE, which ran Upscale Image before, was refused for it)."""
    names = (handle.dit, plans.VAE_FILE)
    detector = pull.DETECTOR
    folder = handle.directory
    absent: set[str] = set()
    try:
        if folder is not None:
            missing: list[ModelError] = [
                ModelMissing(f"{folder / name}: {weights.MISSING}", file=name)
                for name in names
                if not (folder / name).is_file()
            ]
            if missing:
                raise missing[0] if len(missing) == 1 else ModelsRefused(missing)
        else:
            absent = {name for name in names if pull.cached(name) is None}
        fetching = detector_folder(handle) is None and pull.cached(detector) is None
        sizes = {name: pull.PINNED[name].size for name in absent}
        if fetching:
            sizes[detector] = pull.PINNED[detector].size
        downloading = Preparing(show, plans.file_label, expected=sizes)
        if folder is None:
            pull.resolve(
                None, handle.dit, plans.VAE_FILE, detector=False, progress=downloading.download
            )
            # A Cancel pressed during a download reported from another thread is still pending:
            # honoured here, on the node's own thread (progress.Preparing).
            management.throw_exception_if_processing_interrupted()
    except Refusal as error:
        raise words.refusal(error, model_dir=folder is not None) from error
    there = [*names, detector]
    if fetching:
        try:
            pull.fetch({detector: pull.PINNED[detector]}, progress=downloading.download)
            management.throw_exception_if_processing_interrupted()
            absent.add(detector)
        except Refusal as error:
            there.remove(detector)
            console.say(
                logger,
                logging.WARNING,
                "the shot detector's file, which seedvr2x Cuts reads, was not downloaded, and Cuts"
                f" downloads it when it runs: {words.refusal(error, model_dir=folder is not None)}",
            )
    said_where(folder, tuple(there), absent)


def _probe(machine: Machine, architecture: Architecture) -> float | None:
    """The environment probe, before the first load in this process for this DiT layout
    (probe.check: each kernel the file needs, held to the CPU's numbers; cached
    on disk by the environment's fingerprint). A refusal is worded for the nodes. Returns the
    allocator's margin the cache holds for the environment (None: none measured, which is every
    run until the probe measures it; probe.py)."""
    key = architecture.name
    if key in _probed:
        return _probed[key]
    try:
        margin = machine.probe(architecture)
    except Refusal as error:
        raise words.refusal(error) from error
    _probed[key] = margin
    return margin


def _plan[T](
    planner: Planner[T],
    reading: Reading,
    host: int | None,
    architecture: Architecture,
    held: bool,
    handle: Handle,
    clip: words.ClipSize | None = None,
) -> Planned[T]:
    """The planner's plan; a refusal for memory is worded with the smaller files of the same 7B
    that would plan the run (pull.smaller_planning: the file never changed by itself), and with
    the clip's size where the run is a clip's (words.ClipSize)."""
    try:
        return planner(reading, host, architecture, held)
    except PlanRefused as refused:
        smaller: list[pull.Smaller] = []
        if refused.kind in ("memory", "host", "input"):

            def plans_with(other: Architecture) -> bool | None:
                try:
                    planner(reading, host, other, held)
                except PlanRefused:
                    return None
                return True

            smaller = [
                each for each, _ in pull.smaller_planning(handle.dit, architecture, plans_with)
            ]
        raise words.refusal(
            refused,
            smaller=smaller,
            model_dir=bool(handle.model_dir),
            keep=held,
            clip=clip,
        ) from refused


def _load(
    machine: Machine, files: ModelFiles, plan: ModelPlan, preparing: Preparing | None
) -> Models:
    """machine.load, the bytes it reads told to `preparing` (the sizes of both files first, so that
    the bar's total is known before the first is read)."""
    if preparing is None:
        return machine.load(files, plan)
    preparing.will_read(file_size(files.dit.path) + file_size(files.vae.path))
    return machine.load(files, plan, preparing.read)


def _load_kept(
    handle: Handle,
    machine: Machine,
    files: ModelFiles,
    architecture: Architecture,
    preparing: Preparing | None = None,
) -> tuple[Entry, str]:
    """The models read once, every weight in its pinned host copy and none on the card
    (passes.HELD), the buffers and embeddings the load puts on the card sent back (Models.release),
    and registered with ComfyUI; and the note to say in the plan where ComfyUI had dropped the
    weights and they are read again. Any other entry is dropped first, before this one is read:
    two DiTs pinned would take the host's memory twice."""
    for other in list(_entries.values()):
        if other.handle.key != handle.key:
            other.drop("another model was chosen")
    dropped = handle.key in _dropped
    again = " again, ComfyUI having dropped them" if dropped else ""
    _dropped.discard(handle.key)
    logger.info("loading %s into RAM%s, to keep between runs", handle.dit, again)
    note = (
        "ComfyUI's Unload models had dropped seedvr2x's weights from RAM: they are read from the"
        " files again"
        if dropped
        else ""
    )
    try:
        models = _load(machine, files, HELD, preparing)
    except Refusal as error:
        raise words.refusal(error, model_dir=bool(handle.model_dir)) from error
    models.release()
    entry = Entry(handle, files, architecture, models, host_bytes(models))
    entry.patcher = Registration(entry, machine.device, torch.device("cpu"))
    _entries[handle.key] = entry
    _register(entry)
    return entry, note


def host_bytes(models: Models) -> int:
    """The pinned host memory the held models take: the DiT's parts and the VAE's."""
    held = sum(part.size for part in (models.parts or {}).values())
    return held + (0 if models.vae is None else models.vae.size)


def _register(entry: Entry) -> None:
    """The entry registered with ComfyUI's manager, whose only way to count a node's memory and to
    free it this is. Registering a model on the card makes ComfyUI free VRAM down to about 1.2 GiB
    free (0.8 GiB for a minimum inference and 400 MiB reserved, v0.44.0), whatever the model's
    size: the cost of being listed, paid once, before the run's card is read."""
    assert entry.patcher is not None
    management.load_models_gpu([entry.patcher])


def _registered(entry: Entry) -> bool:
    return any(each.model is entry.patcher for each in management.current_loaded_models)


def _ours(entry: Entry | None) -> list[Any]:
    """The manager's own record of seedvr2x's registration, to hand `free_memory` as kept."""
    if entry is None or entry.patcher is None:
        return []
    return [each for each in management.current_loaded_models if each.model is entry.patcher]


def _allocator_note(machine: Machine) -> None:
    """Where the driver lends system memory, ComfyUI's allocator and the plan's reserve said once
    per process, with the way to torch's own allocator with expandable segments, the one that
    holds a run to the card best (memory.py, DESIGN.md, Allocator). Under cudaMallocAsync,
    ComfyUI's default, nothing slows and no bit moves, but its pool takes more of the card than
    torch's limit counts, which the plan counts; on an 8 GB card a picture fits either way and a
    1080p clip only under torch's own allocator with its segments (place-study, 2026-11-05):
    ComfyUI's --disable-cuda-malloc and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, set
    before ComfyUI starts, ComfyUI setting no segments of its own (impl-nodes' reading of
    v0.44.0). Torch's own allocator without the segments plans with the 4.5 GiB reserve alone,
    and a step short of room fails there where the segments slow it down. On Windows, whose torch
    has no segments, the allocator ComfyUI runs is said with no way to them."""
    if not machine.spills():
        return
    if "lends" not in _told:
        _told.add("lends")
        logger.info(
            "this graphics driver borrows your computer's RAM when the card is full"
            " (Windows' shared GPU memory), which is slow; seedvr2x's plan leaves room on the card"
        )
    allocator = machine.allocator()
    if memory.windows():
        # Torch's Windows build has no expandable segments (memory._allocator): under its own
        # allocator a step short of room fails, as ComfyUI's --disable-cuda-malloc did on an RTX
        # 5090 with 8.02 GiB held unused at a 1080p decode (2026-11-08), so cudaMallocAsync,
        # ComfyUI's default, is the one to run, the plan leaving its pool room step by step.
        if "allocator" in _told:
            return
        _told.add("allocator")
        if allocator == "cudaMallocAsync":
            logger.info(
                "ComfyUI runs torch on cudaMallocAsync, whose memory holds more of the"
                " card than it uses: seedvr2x's plan leaves room for it beside each step"
            )
        else:
            logger.warning(
                "ComfyUI runs torch's own allocator, which has no expandable segments"
                " on Windows: a step short of room fails there, and the plan keeps %.1f GiB of the"
                " card free beside the model's work. ComfyUI's default, cudaMallocAsync (ComfyUI"
                " started without --disable-cuda-malloc), leaves the model more",
                memory.RESERVE / memory.GIB,
            )
    elif allocator == "cudaMallocAsync":
        _once(
            "allocator",
            "this driver lends system memory where the card is full, and ComfyUI runs"
            " torch on cudaMallocAsync, its default allocator. Nothing slows down and no picture"
            " changes for it, but it holds more of the card than it uses, which the plan counts:"
            " on an 8 GB card a picture fits either way, a 1080p clip only under torch's own"
            " allocator with its expandable segments: start ComfyUI with"
            " --disable-cuda-malloc, and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            " before it starts (ComfyUI sets no segments of its own)",
        )
    elif not machine.reading(machine.free()).expandable:
        _once(
            "allocator",
            "this driver lends system memory where the card is full, and ComfyUI runs"
            " torch's own allocator without expandable segments, under which a step short of"
            " room fails and the plan keeps %.1f GiB of the card free beside the model's work: set"
            " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting ComfyUI, and a"
            " short step slows down instead",
            memory.RESERVE / memory.GIB,
        )


# What ComfyUI is asked to free beyond the plan's room, in bytes, only where the card is short and
# ComfyUI is asked at all: its partial unload of a model frees to the figure it is given by its own
# count and keeps a buffer, so the card then reads below it. Measured on the RTX 5090 under WSL2
# in ComfyUI v0.44.0 (2026-11-05): "Unloaded partially: 3221.73 MB freed, 76.23 MB remains loaded,
# 93.84 MB buffer reserved", and in four cases asked exactly the plan's room (19.82 to 21.92 GiB)
# the card read 0.04 to 0.11 GiB short and the run was refused; asked 256 MiB more, the same case
# had 19.99 GiB free for a plan of 19.82 and ran. The plan itself never counts it (the output must
# not depend on what else was loaded), and the check after the call is against the plan's own need.
ASK_MARGIN = 256 * 1024**2


def needed_free(needs: Needs) -> int:
    """The free memory the plan needs, in bytes: its largest phase and its margin, and what the
    reading holds back (the reserve a held run leaves torch's allocator, what torch is held back
    from): the plan holds on a card with that much free, in the figure the reading counts from."""
    holdback = max(_taken(needs.reading) - needs.reading.counted, 0)
    return needs.peak + needs.margin + holdback


def rng_states() -> Callable[[], None]:
    """The global random generators' states, and the call that puts them back: Python's, NumPy's
    and torch's on the CPU, and torch's on each CUDA device where CUDA is initialised.
    model.set_seed seeds all of them (the library's passes call it), and ComfyUI's other nodes
    draw from them: a run leaves them as it found them."""
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )

    def restore() -> None:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

    return restore


def _clear_frames(error: BaseException) -> None:
    """The frames of an exception's traceback, and of those it was raised from or while handling,
    cleared of their local variables: a node's error or ComfyUI's Cancel keeps its traceback
    until ComfyUI drops the exception, and the frames of the passes it went through hold the
    models and the tensors on the card (as the library's clip.upscale does for its own)."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        each = pending.pop()
        if each is None or id(each) in seen:
            continue
        seen.add(id(each))
        traceback.clear_frames(each.__traceback__)
        pending += [each.__cause__, each.__context__]


def _settle(card: int) -> None:
    """The mark an earlier run's left wait put on the card, cleared by the card itself (DESIGN.md,
    the nodes' Progress): asked, without waiting (waiting.settle), whether all the work the wait
    was for has finished. Finished: the models held are given back now (Models.release, whose
    waits are immediate once all streams are done, and which clears the mark), those of runs not
    kept synchronized in the poll and let go, said at INFO. Both run as a release, with ComfyUI's
    Cancel as the check.
    Not finished: words.card_busy, the mark kept. A card a CUDA error failed is never cleared
    (waiting.settle), and _session refuses it before this."""
    if not waiting.settle(card):
        if waiting.failed(card) is not None:
            raise words.card_failed(waiting.failed(card) or "", earlier=True)
        raise words.card_busy()
    held = list(_left_on_card)
    kept_ones = [each.models for each in _entries.values() if each.models is not None]
    logger.info("the card finished the work an earlier run left; giving the models back")
    device = machine().device
    try:
        # Outside a run, before ComfyUI's Cancel is the check (Progress.running): the release is
        # one (a second Cancel leaves it, the mark kept) and the check is ComfyUI's.
        with (
            waiting.releasing(card),
            waiting.checking(management.throw_exception_if_processing_interrupted),
        ):
            for each in held:
                # Models loaded without passes.HELD are never placed or released: a run not kept
                # frees them by forget alone (below), as its end does.
                if each.parts is not None:
                    each.release()
                if not any(each is other for other in kept_ones):
                    # The device's work waited for in the poll before the frees (HostCopy's
                    # cudaHostUnregister, torch's deleters), as a run's end does.
                    model.synchronize(device)
                _left_on_card.remove(each)
                if not any(each is other for other in kept_ones):
                    model.forget(each)
    except BaseException as error:
        # A release that waits again and is left, or a card that failed: marked by the wait.
        if waiting.failed(card) is not None:
            raise words.card_failed(waiting.failed(card) or "", earlier=True) from error
        if waiting.abandoned(card):
            raise words.card_busy() from error
        raise
    gc.collect()
    if waiting.abandoned(card):
        raise words.card_busy()


@contextmanager
def _freeing(show: Callable[[str], None] | None) -> Generator[None]:
    """The release of the models after a run an exception ended, in the with block: the node says
    what it waits for (FREEING), ComfyUI's Cancel is the check of the release's waits
    (waiting.checking: the first Cancel's flag is gone, ComfyUI's check clearing it as it raises,
    so only a second press raises), and a release that completes puts back the text the node
    showed before, a healthy card taking a moment. A wait there that an exception ended (a second
    Cancel, a CUDA error) leaves the card behind: the node says so (STOPPED) and the exception goes
    through; another exception goes through with the text before put back."""
    last: str | None = getattr(show, "last", None)
    if show is not None:
        show(FREEING)
        logger.info("%s", FREEING_SAID)
    try:
        with waiting.checking(management.throw_exception_if_processing_interrupted):
            yield
    except BaseException as error:
        if show is not None:
            # A wait left (a second Cancel) or failed: the card is not given back. Any other
            # error says nothing of the card.
            stopped = waiting.ended_by(error) is not None
            show(STOPPED if stopped else "" if last is None else last)
            if stopped:
                logger.warning("%s", STOPPED_SAID)
        raise
    if show is not None:
        # The text before, or none where the run had said nothing yet (an error before its first
        # event): never FREEING left on a card that answered.
        show("" if last is None else last)


@contextmanager
def session[T](
    handle: Handle,
    planner: Planner[T],
    *,
    plan_only: bool = False,
    clip: words.ClipSize | None = None,
    preparing: Preparing | None = None,
) -> Generator[Run[T]]:
    """One run of a node (the module's docstring): yields the models placed for the plan the
    planner makes, and frees the card whatever happens after. The Refused of words.py for a
    refusal, with the way out in the nodes' terms; ComfyUI's Cancel and the node's errors pass
    through, the card freed, and the global random generators put back as they were.

    `plan_only`: the plan alone. The files are found and their headers and pins' sizes checked
    (a file missing from Hugging Face's cache is fetched, as the command line's --plan does), the
    card and the host read and the plan made; nothing is read whole, probed, registered, loaded
    or placed, ComfyUI is asked to free nothing and the run's `models` is refused.

    `clip`: the size of the clip the run upscales, for its refusals to be worded with (a host
    that can't hold it whole is told what the command line does instead).

    `preparing`: where the bytes of a download of seedvr2x's files and of their reading into memory
    are told (progress.Preparing), for the node to show; observing only."""
    restore_rng = rng_states()
    # A card an earlier run failed refuses this one at its start (_session), with nothing to keep.
    before = _machine
    earlier = before is not None and waiting.failed(model.card_index(before.device)) is not None
    try:
        with _session(handle, planner, plan_only, clip, preparing) as run:
            yield run
    except BaseException as error:
        made = _machine
        if made is None or earlier:
            raise
        card = model.card_index(made.device)
        if waiting.cuda_failed(error):
            # Met before the run's own handling (the card's reading, the probe, the load): marked
            # here, so that the next run is refused.
            waiting.fail(card, error)
        failed = waiting.failed(card)
        if failed is None:
            raise
        # The card failed, whatever exception arrived here (another may have replaced the CUDA
        # error on its way out).
        # The error and its frames kept, nothing they hold freed (_failures); the node, the
        # console and ComfyUI's error say what failed and the way out.
        _failures.append(error)
        refused = words.card_failed(failed)
        console.say(logger, logging.ERROR, str(refused))
        refused.logged = True
        if preparing is not None:
            preparing.show(FAILED)
        raise refused from error
    finally:
        restore_rng()


@contextmanager
def _session[T](
    handle: Handle,
    planner: Planner[T],
    plan_only: bool,
    clip: words.ClipSize | None,
    preparing: Preparing | None = None,
) -> Generator[Run[T]]:
    """session, less the random generators."""
    machine_ = machine()
    card = model.card_index(machine_.device)
    failed = waiting.failed(card)
    if failed is not None:
        # Nothing is read from the card, placed, freed or waited on: an earlier run's CUDA error
        # failed it (waiting.fail).
        raise words.card_failed(failed, earlier=True)
    # Process-wide until runs go to other cards: models left on any card refuse every run.
    if waiting.abandoned(card) or _left_on_card:
        # An earlier run left a wait on the GPU (a second Cancel): the work it was waiting for is
        # asked about without waiting, and nothing is read from the card, placed or waited on
        # while it is pending.
        _settle(card)
    _allocator_note(machine_)
    keeping = decide_keeping(handle)
    keep = keeping.kept
    logger.info("%s", keeping.line)
    if not keep and not plan_only:
        # The weights of an earlier run, kept, are freed before this run reads its own: never
        # two sets pinned for one choice.
        for other in list(_entries.values()):
            other.drop("they are not kept for this run")
    entry = kept(handle) if keep else None
    notes: list[str] = []
    if entry is None:
        if preparing is not None and not plan_only:
            # The reading's bytes are on the bar before the first download's, so that its total
            # holds: the pinned files' sizes (a file of a model folder is announced by its load).
            preparing.will_read(
                sum(
                    pull.PINNED[each].size
                    for each in (handle.dit, plans.VAE_FILE)
                    if each in pull.PINNED
                )
            )
        files, architecture = _prepare(handle, machine_, preparing)
    else:
        files, architecture = entry.files, entry.architecture
    if plan_only:
        ours = None if entry is None else entry.patcher
        # No probe runs for a plan only; the margin an earlier run of this process found counts.
        reading, own, held, reserve = _read(machine_, ours, _probed.get(architecture.name))
        host = machine_.host()
        # Weights to be pinned by the run come out of the memory the host has available.
        counted = (
            host if entry is not None or not keep or host is None else max(host - keeping.size, 0)
        )
        planned = _plan(planner, reading, counted, architecture, keep, handle, clip)
        room = plans.Room(
            card=machine_.name(),
            free=own.reported,
            comfy=held,
            reserve=reserve,
            needed=needed_free(planned.needs),
            asked=False,
            after=own.reported,
            host=host,
            plan_only=True,
        )
        yield Run(None, planned, room, handle, keeping, (), 0 if entry is None else entry.host)
        return
    margin = _probe(machine_, architecture)
    if keep and entry is None:
        # A refusal comes before the read of the files: the plan for the card alone, the pinned
        # copies not counted (the host is checked by the plan that follows).
        early, _, _, _ = _read(machine_, None, margin)
        _plan(planner, early, None, architecture, True, handle, clip)
        entry, note = _load_kept(handle, machine_, files, architecture, preparing)
        if note:
            notes.append(note)
    ours = None if entry is None else entry.patcher
    if entry is not None and not _registered(entry):
        logger.info("seedvr2x's models are not listed with ComfyUI any more: registering again")
        _register(entry)

    reading, own, held, reserve = _read(machine_, ours, margin)
    host = machine_.host()
    planned = _plan(planner, reading, host, architecture, keep, handle, clip)
    needed = needed_free(planned.needs)
    # ComfyUI's own free memory counts the room it keeps for the system: the run needs that much
    # free beyond its plan, and is asked for it.
    beyond = room_beyond(own, reserve)
    asked = max(_taken(own) - beyond, 0) < needed
    if asked:
        # The free amount wanted, in the figure ComfyUI reads (the driver's), which is above the
        # card's where NVML reports less.
        wanted = needed + beyond + ASK_MARGIN + max(own.reported - _taken(own), 0)
        logger.info(
            "asking ComfyUI for %.2f GiB free on the card for this run (%.2f GiB free, %.2f GiB"
            " used by its models; %.2f GiB of that over what the run needs, for what its unloading"
            " keeps)",
            wanted / memory.GIB,
            _taken(own) / memory.GIB,
            held / memory.GIB,
            ASK_MARGIN / memory.GIB,
        )
        management.free_memory(wanted, machine_.device, keep_loaded=_ours(entry))
    after = own
    if asked:
        # ComfyUI's partial unloads leave what they free in torch's cache, which it trims only
        # where an entry was dropped or the cache is large; the driver's free memory leaves it
        # out, though this process can use it (readings, as _read).
        management.soft_empty_cache()
        after = machine_.reading(machine_.free())
        there = max(after.counted - beyond, 0)
        if there < planned.needs.peak + planned.needs.margin:
            raise words.not_freed(
                need=planned.needs.peak + planned.needs.margin,
                there=there,
                comfy=comfy_holds(machine_.device, ours),
            )
    room = plans.Room(
        card=machine_.name(),
        free=own.reported,
        comfy=held,
        reserve=reserve,
        needed=needed,
        asked=asked,
        after=after.reported,
        host=host,
        margin=ASK_MARGIN if asked else 0,
    )
    models: Models | None = None
    ended = False
    if entry is not None:
        entry.busy = True
    try:
        try:
            if entry is not None:
                models = entry.models
                assert models is not None
                models.place(planned.models)
            else:
                models = _load(machine_, files, planned.models, preparing)
        except Refusal as error:
            raise words.refusal(error, model_dir=bool(handle.model_dir), keep=keep) from error
        run = Run(
            models,
            planned,
            room,
            handle,
            keeping,
            tuple(notes),
            0 if entry is None else entry.host,
        )
        try:
            yield run
        except BaseException as error:
            if waiting.cuda_failed(error):
                # A CUDA error other than out of memory: the card is marked before anything
                # below frees on it, and the frames keep what they hold (_failures).
                waiting.fail(card, error)
            if waiting.failed(card) is None:
                # The frames of the node and the passes the error went through hold the models
                # and their tensors on the card: cleared before the models are given up, whether
                # the error is the node's, a Cancel between its steps or a refusal.
                _clear_frames(error)
            raise
        finally:
            run.over()
    except BaseException as error:
        ended = True
        if waiting.cuda_failed(error):
            # Placing or loading the models met it: marked before the release below.
            waiting.fail(card, error)
        raise
    finally:
        # After a wait on the GPU that was abandoned, nothing here waits on the card: the kept
        # models' release leaves them as they are (Models.release), models not kept are held, not
        # dropped, and torch's cache is not emptied (it synchronises). A release after a run an
        # exception ended (a Cancel, an error) may wait on a card that stopped answering: it is
        # shown on the node and left by a second Cancel (_freeing).
        lost = waiting.abandoned(card)
        show = None if preparing is None else preparing.show
        try:
            if entry is not None:
                if models is not None:
                    try:
                        if ended and not lost:
                            with _freeing(show):
                                models.release()
                        else:
                            models.release()
                    finally:
                        if waiting.abandoned(card):
                            _hold(models)
            elif lost:
                if models is not None:
                    if waiting.failed(card) is None:
                        # Everything queued behind the wait left, on every stream (waiting.pend).
                        waiting.pend(card, *models.queries())
                    _hold(models)
                    logger.warning(
                        "the models of this run are left on the card: %s; restart ComfyUI",
                        "a wait on the GPU was abandoned, and giving them back would wait on it"
                        " again"
                        if waiting.failed(card) is None
                        else "the card failed, and giving them back would free memory there",
                    )
                models = None
            else:
                if models is not None:
                    # Nothing below goes through the poll: the models' finalizers
                    # (blockswap.HostCopy's cudaHostUnregister, in gc.collect) and torch's cache
                    # (soft_empty_cache) synchronise in C, and their frees report a kernel's
                    # failure where it can't raise (under cudaMallocAsync, an abort). The device's
                    # work is waited for first, in the poll, at a run's end too: a failure comes
                    # back here as a CUDA error, which marks the card (waiting.wait), and a card
                    # that stopped answering in a kind's first unit (no bound, no mark) can be
                    # left by a second Cancel.
                    try:
                        with waiting.releasing(card), model.marking(machine_.device):
                            # What the release the next run completes (_settle) would wait for.
                            waiting.expect(*models.queries())
                            if ended:
                                with _freeing(show):
                                    model.synchronize(machine_.device)
                            else:
                                model.synchronize(machine_.device)
                    finally:
                        if waiting.abandoned(card):
                            _hold(models)
                if waiting.abandoned(card):
                    models = None
                else:
                    if models is not None:
                        # What the models keep on the device besides their parameters (the
                        # vendored RoPE's lru_cache, the VAE's convolution memory) would outlive
                        # them.
                        model.forget(models)
                    models = None
                    gc.collect()
        finally:
            if not waiting.abandoned(card):
                management.soft_empty_cache()
            if entry is not None:
                entry.busy = False
