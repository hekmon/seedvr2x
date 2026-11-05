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
   own cap on pinned memory, on keeps them whatever the cap, off does not), said on the node and
   in the console; where they are not, the kept set of an earlier run is freed before this run
   reads its own, and a kept set of another file before the new one is pinned;
1. the files found and checked (pull.resolve, weights.check_models, pull.check_pinned: their
   header, then each pinned file's size, its SHA-256 left to the load, which proves it), and the
   environment probe (probe.check) for that DiT layout and attention kernel, once per process;
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

from seedvr2x import blockswap, memory, model, peaks, pull, weights
from seedvr2x.errors import PlanRefused, Refusal
from seedvr2x.memory import Reading
from seedvr2x.model import Models
from seedvr2x.passes import HELD, ModelPlan
from seedvr2x.pull import ModelFiles
from seedvr2x.weights import Architecture

from . import plans, words

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
        _once("smart-memory", "seedvr2x: %s", because)
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
    def attention(self) -> str: ...
    def dit_model(self, path: Path) -> Architecture: ...
    def probe(self, architecture: Architecture, attention: str) -> None: ...
    def load(self, files: ModelFiles, plan: ModelPlan) -> Models: ...


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

    def attention(self) -> str:
        return model.attention_backend()

    def dit_model(self, path: Path) -> Architecture:
        return weights.check(path, "dit")

    def probe(self, architecture: Architecture, attention: str) -> None:
        from seedvr2x import probe

        probe.check(self.device, architecture, attention)

    def load(self, files: ModelFiles, plan: ModelPlan) -> Models:
        return model.load_models(files.dit.path, files.vae.path, self.device, files.hashes, plan)


# The start of the library's warnings that are the command line's words (memory.startup): about an
# allocator "of the environment's", worded for a user who set it, which in ComfyUI's process is the
# allocator ComfyUI chose (the backend torch reports is what _startup gives startup to read); and
# that seedvr2x holds itself to the card by a limit on torch, which the nodes never set (the
# limit cannot be taken back in a process shared with other nodes). The nodes say their own words,
# once (_allocator_note).
_COMMAND_LINE_WORDS = (
    "the allocator is the environment's",
    "seedvr2x holds itself to the card's own memory there",
)


class _CommandLineWords(logging.Filter):
    """Drops the library's warnings that describe a limit or a setting the nodes don't make."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith(_COMMAND_LINE_WORDS)


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
        self.models = None
        if _entries.get(self.handle.key) is self:
            del _entries[self.handle.key]
        if by_comfy:
            _dropped.add(self.handle.key)
        gc.collect()


_entries: dict[tuple[str, str], Entry] = {}
_probed: set[tuple[str, str]] = set()
# The handles whose weights were dropped (Entry.drop): the next load of one says so.
_dropped: set[tuple[str, str]] = set()
_told: set[str] = set()


def reset() -> None:
    """The module as new (the tests): the entries dropped, the probe and the notes forgotten."""
    for entry in list(_entries.values()):
        entry.models = None
    _entries.clear()
    _dropped.clear()
    _probed.clear()
    _told.clear()


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
# the system memory available (None: unchecked), the attention kernel, the DiT's format and
# whether the weights' pinned copies exist already. PlanRefused as the library raises it.
type Planner[T] = Callable[[Reading, int | None, str, Architecture, bool], Planned[T]]


class Run[T]:
    """What a node gets in a session: the models placed for its plan, the plan, the room it took,
    the attention kernel it runs, the handle, what keeping the weights came to (`keeping`) and the
    notes to say on the node once (`notes`: ComfyUI having dropped the weights). The models are
    the session's: over when it ends, they are no longer reachable from here, so that a node still
    holding its Run after the `with` keeps no weight alive. A plan only has none: `models` is
    refused, the files having been read for their headers alone."""

    def __init__(
        self,
        models: Models | None,
        planned: Planned[T],
        room: plans.Room,
        attention: str,
        handle: Handle,
        keeping: plans.Keeping,
        notes: tuple[str, ...] = (),
        held: int = 0,
    ) -> None:
        self._models: Models | None = models
        self.planned = planned
        self.room = room
        self.attention = attention
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


def with_comfy_gone(reading: Reading, held: int) -> Reading:
    """`reading` with the `held` bytes ComfyUI's own models have on the card counted free: the
    reading the plan is made in, for the card as it would be without them. Each figure of the
    reading moves by `held`, and a held run's counted memory is taken again from them
    (memory.reading): the driver's and the card's figures less what torch is held back by and
    the reserve."""
    if held <= 0:
        return reading
    if not reading.held:
        return replace(reading, reported=reading.reported + held, counted=reading.counted + held)
    reported = reading.reported + held
    card = None if reading.card is None else reading.card + held
    taken = reported if card is None else min(reported, card)
    return replace(
        reading,
        reported=reported,
        card=card,
        counted=max(taken - reading.left - reading.reserve, 0),
    )


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


def _read(machine: Machine, ours: Registration | None) -> tuple[Reading, Reading, int]:
    """The card as the plan counts it: ComfyUI's cache of freed blocks given back first (what
    torch keeps cached and unused is free by definition, and the driver's figure leaves it out),
    then the driver's free memory read as the CLI reads it, and ComfyUI's models added to it.
    Returns (the reading for the plan, the card's own reading, what ComfyUI's models hold)."""
    management.soft_empty_cache()
    own = machine.reading(machine.free())
    held = comfy_holds(machine.device, ours)
    return with_comfy_gone(own, held), own, held


def _prepare(handle: Handle, machine: Machine) -> tuple[ModelFiles, Architecture]:
    """The model files found and checked as the server checks them (service.check_files): the
    directory or Hugging Face's cache, pulled at the pinned revision where missing, their headers
    checked, each pinned file's size checked and its SHA-256 left to the load. The detector is not
    needed. A refusal is worded for the nodes (words.refusal)."""
    try:
        files = pull.resolve(handle.directory, handle.dit, plans.VAE_FILE, detector=False)
        dit, vae = (None if each.absent else each.path for each in (files.dit, files.vae))
        weights.check_models(dit, vae, pull.advice(files), None)
        pull.check_pinned(files, load=True)
        assert dit is not None
        return files, machine.dit_model(dit)
    except Refusal as error:
        raise words.refusal(error, model_dir=bool(handle.model_dir)) from error


def _attention(machine: Machine, requested: str | None) -> str:
    """The attention kernel of the run: the environment's (FlashAttention 2 where it imports, else
    SDPA) unless the node's attention setting asked for one; SDPA is always there, FlashAttention
    2 only where it imports."""
    available = machine.attention()
    if requested is None:
        return available
    if requested == peaks.FLASH_ATTN_2 and available != peaks.FLASH_ATTN_2:
        raise words.attention_missing()
    return requested


def _set_attention(models: Models, attention: str) -> None:
    """The attention kernel set on the DiT's attention modules, as the loader sets the
    environment's (model.load_models), where the node's setting asked for the other one."""
    if models.attention == attention:
        return
    from seedvr2x.vendor.models.dit_3b import attention as attention_3b
    from seedvr2x.vendor.models.dit_7b import attention as attention_7b

    dit = getattr(models.runner, "dit", None)
    for module in () if dit is None else dit.modules():
        if isinstance(
            module, attention_7b.FlashAttentionVarlen | attention_3b.FlashAttentionVarlen
        ):
            module.attention_mode = attention
    models.attention = attention


def _probe(machine: Machine, architecture: Architecture, attention: str) -> None:
    """The environment probe, before the first load in this process for this DiT layout and this
    attention kernel (probe.check: each kernel the file needs, held to the CPU's numbers; cached
    on disk by the environment's fingerprint). A refusal is worded for the nodes."""
    key = (architecture.name, attention)
    if key in _probed:
        return
    try:
        machine.probe(architecture, attention)
    except Refusal as error:
        raise words.refusal(error) from error
    _probed.add(key)


def _plan[T](
    planner: Planner[T],
    reading: Reading,
    host: int | None,
    attention: str,
    architecture: Architecture,
    held: bool,
    handle: Handle,
) -> Planned[T]:
    """The planner's plan; a refusal for memory is worded with the smaller files of the same 7B
    that would plan the run (pull.smaller_planning: the file never changed by itself)."""
    try:
        return planner(reading, host, attention, architecture, held)
    except PlanRefused as refused:
        smaller: list[pull.Smaller] = []
        if refused.kind in ("memory", "host", "input"):

            def plans_with(other: Architecture) -> bool | None:
                try:
                    planner(reading, host, attention, other, held)
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
        ) from refused


def _load_kept(
    handle: Handle, machine: Machine, files: ModelFiles, architecture: Architecture
) -> tuple[Entry, str]:
    """The models read once, every weight in its pinned host copy and none on the card
    (passes.HELD), the buffers and embeddings the load puts on the card sent back (Models.release),
    and registered with ComfyUI; and the note to say on the node where ComfyUI had dropped the
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
        models = machine.load(files, HELD)
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
    """Where the driver lends system memory and ComfyUI's torch runs an allocator that holds a
    run to the card less well than torch's own with expandable segments (memory.py, DESIGN.md,
    Allocator, "Expandable segments"), said once per process, with the way to it: cudaMallocAsync,
    ComfyUI's default, gives way to torch's own allocator with ComfyUI's --disable-cuda-malloc;
    torch's own without the segments takes them from PYTORCH_CUDA_ALLOC_CONF set before ComfyUI
    starts, so that a step short of room slows down instead of failing."""
    if not machine.spills():
        return
    if "lends" not in _told:
        _told.add("lends")
        logger.info(
            "seedvr2x: this driver lends system memory where the card is full: the plan keeps the"
            " reserve for it, and no limit is set on torch's memory (ComfyUI's process is shared"
            " with its other nodes)"
        )
    allocator = machine.allocator()
    if allocator == "cudaMallocAsync":
        _once(
            "allocator",
            "seedvr2x: this driver lends system memory where the card is full, and ComfyUI runs"
            " torch on cudaMallocAsync, which holds a run to the card less well than torch's own"
            " allocator does: start ComfyUI with --disable-cuda-malloc for it",
        )
    elif not machine.reading(machine.free()).expandable:
        _once(
            "allocator",
            "seedvr2x: this driver lends system memory where the card is full, and ComfyUI runs"
            " torch's own allocator without expandable segments, under which a step short of"
            " room fails: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting"
            " ComfyUI, and a short step slows down instead",
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


@contextmanager
def session[T](
    handle: Handle, attention: str | None, planner: Planner[T], *, plan_only: bool = False
) -> Generator[Run[T]]:
    """One run of a node (the module's docstring): yields the models placed for the plan the
    planner makes, and frees the card whatever happens after. The Refused of words.py for a
    refusal, with the way out in the nodes' terms; ComfyUI's Cancel and the node's errors pass
    through, the card freed, and the global random generators put back as they were.

    `plan_only`: the plan alone. The files are found and their headers and pins' sizes checked
    (a file missing from Hugging Face's cache is fetched, as the command line's --plan does), the
    card and the host read and the plan made; nothing is read whole, probed, registered, loaded
    or placed, ComfyUI is asked to free nothing and the run's `models` is refused."""
    restore_rng = rng_states()
    try:
        with _session(handle, attention, planner, plan_only) as run:
            yield run
    finally:
        restore_rng()


@contextmanager
def _session[T](
    handle: Handle, attention: str | None, planner: Planner[T], plan_only: bool
) -> Generator[Run[T]]:
    """session, less the random generators."""
    machine_ = machine()
    _allocator_note(machine_)
    keeping = decide_keeping(handle)
    keep = keeping.kept
    logger.info("seedvr2x: %s", keeping.line)
    if not keep and not plan_only:
        # The weights of an earlier run, kept, are freed before this run reads its own: never
        # two sets pinned for one choice.
        for other in list(_entries.values()):
            other.drop("they are not kept for this run")
    entry = kept(handle) if keep else None
    notes: list[str] = []
    if entry is None:
        files, architecture = _prepare(handle, machine_)
    else:
        files, architecture = entry.files, entry.architecture
    chosen = _attention(machine_, attention)
    if plan_only:
        ours = None if entry is None else entry.patcher
        reading, own, held = _read(machine_, ours)
        host = machine_.host()
        # Weights to be pinned by the run come out of the memory the host has available.
        counted = (
            host if entry is not None or not keep or host is None else max(host - keeping.size, 0)
        )
        planned = _plan(planner, reading, counted, chosen, architecture, keep, handle)
        room = plans.Room(
            card=machine_.name(),
            free=own.reported,
            comfy=held,
            needed=needed_free(planned.needs),
            asked=False,
            after=own.reported,
            host=host,
            plan_only=True,
        )
        yield Run(
            None, planned, room, chosen, handle, keeping, (), 0 if entry is None else entry.host
        )
        return
    _probe(machine_, architecture, chosen)
    if keep and entry is None:
        # A refusal comes before the read of the files: the plan for the card alone, the pinned
        # copies not counted (the host is checked by the plan that follows).
        early, _, _ = _read(machine_, None)
        _plan(planner, early, None, chosen, architecture, True, handle)
        entry, note = _load_kept(handle, machine_, files, architecture)
        if note:
            notes.append(note)
    ours = None if entry is None else entry.patcher
    if entry is not None and not _registered(entry):
        logger.info("seedvr2x's models are not listed with ComfyUI any more: registering again")
        _register(entry)

    reading, own, held = _read(machine_, ours)
    host = machine_.host()
    planned = _plan(planner, reading, host, chosen, architecture, keep, handle)
    needed = needed_free(planned.needs)
    asked = _taken(own) < needed
    if asked:
        # The free amount wanted, in the figure ComfyUI reads (the driver's), which is above the
        # card's where NVML reports less.
        wanted = needed + ASK_MARGIN + max(own.reported - _taken(own), 0)
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
        if after.counted < planned.needs.peak + planned.needs.margin:
            raise words.not_freed(
                need=planned.needs.peak + planned.needs.margin,
                there=after.counted,
                comfy=comfy_holds(machine_.device, ours),
            )
    room = plans.Room(
        card=machine_.name(),
        free=own.reported,
        comfy=held,
        needed=needed,
        asked=asked,
        after=after.reported,
        host=host,
        margin=ASK_MARGIN if asked else 0,
    )
    models: Models | None = None
    if entry is not None:
        entry.busy = True
    try:
        try:
            if entry is not None:
                models = entry.models
                assert models is not None
                models.place(planned.models)
            else:
                models = machine_.load(files, planned.models)
        except Refusal as error:
            raise words.refusal(error, model_dir=bool(handle.model_dir), keep=keep) from error
        _set_attention(models, chosen)
        run = Run(
            models,
            planned,
            room,
            chosen,
            handle,
            keeping,
            tuple(notes),
            0 if entry is None else entry.host,
        )
        try:
            yield run
        except BaseException as error:
            # The frames of the node and the passes the error went through hold the models and
            # their tensors on the card: cleared before the models are given up, whether the
            # error is the node's, a Cancel between its steps or a refusal.
            _clear_frames(error)
            raise
        finally:
            run.over()
    finally:
        try:
            if entry is not None:
                if models is not None:
                    models.release()
            else:
                if models is not None:
                    # What the models keep on the device besides their parameters (the vendored
                    # RoPE's lru_cache, the VAE's convolution memory) would outlive them.
                    model.forget(models)
                models = None
                gc.collect()
        finally:
            management.soft_empty_cache()
            if entry is not None:
                entry.busy = False
