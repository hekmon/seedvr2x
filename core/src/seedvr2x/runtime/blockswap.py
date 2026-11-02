"""Where the models' weights are: BlockSwap, rewritten (DESIGN.md, Memory planner, BlockSwap,
rewritten).

numz swaps its first N blocks, each copied to the GPU for its forward pass and back to pageable
memory after it, synchronously (research/bugs/17). Weights never change, so here:
- what leaves the GPU has one page-locked host copy (HostCopy), made once, and only ever goes
  host to device;
- the last S blocks of the DiT are the swapped ones, and stream through SLOTS buffers on the GPU,
  allocated once and written in place: a swapped block's parameters are views of its slot, bound
  once, and only the slot's bytes change;
- with the prefetch, a slot's next block is copied on a side stream while the other slot's block
  computes, the copy waiting for the slot's last reader and the block for its copy by CUDA events,
  the host never waiting. Without it, each block is copied on the compute stream right before
  its forward: the reference the prefetch is checked against;
- between phases, what the plan sends away goes back to its host copy (Placement.enter), and
  comes back from it in one copy per part.

Only parameters move. Buffers stay on the GPU: RoPE's tables (3 per block of the 7B, 0.7 MiB)
and the angle tables each block caches from them, so numz's device-aware RoPE wrapper, and its
late-binding bug (research/bugs/24), have no place here.

A part (Part) is a set of parameters that leaves and comes back together: one block of the DiT,
the DiT's parameters outside its blocks, or the VAE. Its parameters are laid out in one run of
bytes, so that a part moves in one copy, and a module's parameters are views into whichever
buffer holds those bytes at the time: the host copy, a buffer on the device, or a slot.
"""

import json
import logging
import mmap
import re
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from seedvr2x.runtime.job import JobError
from seedvr2x.runtime.plan import Phase
from seedvr2x.runtime.weights import AGAIN, SLOTS, ModelError

logger = logging.getLogger(__name__)

GIB = 1 << 30

# Each parameter starts on a multiple of this in its part's bytes, as torch's CUDA allocator
# aligns its own blocks (512 bytes): 30 parameters waste under 16 KiB of a block's 0.42 GiB.
ALIGN = 512
# The part of the DiT's parameters outside its blocks.
OTHER = "other"

_BLOCK = re.compile(r"blocks\.(\d+)\.")
# torch's dtype for each of safetensors' names the loader reads (read), each held to
# safetensors' own reading by tests/test_blockswap.py. A file seedvr2x runs holds F16 alone
# (weights.check); the other names of weights.DTYPE_BYTES, the unsigned ones over 8 bits and
# fp8's two, which no such file holds, are refused there rather than read untested.
_DTYPES: dict[str, torch.dtype] = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class HostMemoryError(JobError):
    """The host can't hold the pinned copies a plan needs: the message names the RAM they take."""


class HostCopy:
    """`size` bytes of host memory, page-locked for CUDA when `pin`: a private anonymous mapping
    of its own, registered with cudaHostRegister, so that it takes its size and no more. torch's
    own pinned allocator rounds each allocation up to a power of two: a 7B block's 0.4220 GiB
    took 0.5 GiB there, 18.0 GiB for the 36 blocks where their bytes are 15.19 (torch 2.14.1).
    Copies from it to the device are as fast as from torch's (54.6 against 53.4 GB/s on a desktop
    PCIe 5.0 link, 18.2 from pageable memory) and return at once (0.12 ms): measured by
    research/scripts/pin_probe.py, on such a mapping, 2026-10-10, on an RTX 5090 under WSL.
    DESIGN.md's 57 GB/s pinned and 25 pageable (BlockSwap, rewritten) are research/docs/vram.md's,
    standalone copies on the 96 GB card's PCIe 5.0 x16 link: another machine, the same ratio.

    bytes is the buffer, (size rounded up to a page,) uint8, zeros at first. Raises
    HostMemoryError when the host refuses the mapping or its page-locking, or when this torch
    doesn't bind CUDA's call for it (torch.cuda.cudart(), which torch binds without documenting
    it): a refusal before the run, never a pageable copy in the pinned one's place."""

    def __init__(self, size: int, pin: bool) -> None:
        length = -(-max(size, 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        try:
            area = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            # Not a forked child's: the kernel would copy every pinned page for it at the fork,
            # the 15.81 GiB pinned for a plan that keeps the two models apart (the whole DiT and
            # the VAE, measured, milestone 3, 2026-10-11). seedvr2x forks nothing (its
            # ffmpeg processes are spawned); a child that did would have no weights to read.
            area.madvise(mmap.MADV_DONTFORK)
        except OSError as error:
            raise HostMemoryError(f"the host refuses {_gib(length)} of memory: {error}") from error
        self.bytes: Tensor = torch.frombuffer(area, dtype=torch.uint8)
        self.pinned = False
        if pin:
            try:
                cudart = _cudart()
                register, unregister = cudart.cudaHostRegister, cudart.cudaHostUnregister
                words = cudart.cudaGetErrorString
            except (AttributeError, RuntimeError, AssertionError) as error:
                # No binding in this torch, or no CUDA for it to bind: the error says which.
                raise HostMemoryError(
                    f"{_gib(length)} of host memory can't be pinned: CUDA's cudaHostRegister"
                    f" isn't to be had from this torch ({torch.__version__}): {error}"
                ) from error
            address: int = self.bytes.data_ptr()
            code: Any = register(address, length, 0)
            if int(code) != 0:
                # torch names no code but success: CUDA's own words for it.
                raise HostMemoryError(
                    f"the host refuses to pin {_gib(length)} of memory: {words(code)}"
                )
            self.pinned = True
            # Unregistered when this object goes: the mapping itself lives as long as a tensor
            # views it. Not at exit, where CUDA is torn down first.
            self.unregistering = weakref.finalize(self, unregister, address)
            self.unregistering.atexit = False


class Cuda:
    """The CUDA calls a placement makes on its device, in one place: the tests on the CPU give
    stand-ins that record them."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def host(self, size: int) -> HostCopy:
        return HostCopy(size, pin=True)

    def compute(self) -> Any:
        """The stream the models compute on: the caller's current one."""
        return torch.cuda.current_stream(self.device)

    def side(self) -> Any:
        """A stream of its own, for the prefetch's copies."""
        return torch.cuda.Stream(self.device)

    def event(self) -> Any:
        return torch.cuda.Event()

    def allocate(self, size: int) -> Tensor:
        return torch.empty(size, dtype=torch.uint8, device=self.device)

    def copy(self, target: Tensor, source: Tensor, stream: Any, blocking: bool) -> None:
        """source's bytes (host) into target (device), queued on stream; with blocking, the host
        waits until they are there."""
        with torch.cuda.stream(stream):
            target.copy_(source, non_blocking=not blocking)

    def trim(self) -> None:
        """Hand the driver back what the allocator holds free (torch.cuda.empty_cache), after
        weights have left: freed, their memory is the allocator's, not the driver's, and the
        next phase finds the device as full as before unless the allocator can make its own
        buffers of it. A card that refuses makes the allocator give it back when it must; one
        whose driver spills into system memory never does (research/scripts/spill_probe.py):
        there the driver had 5.5 GiB free in the VAE's phase after the whole DiT had left, and
        29.3 with this, for 0.28 s (DESIGN.md, BlockSwap, rewritten, Per phase)."""
        torch.cuda.empty_cache()

    def wait(self, *streams: Any) -> None:
        """Wait for the work queued on streams, polling as model.synchronize does, so that a
        signal's handler runs meanwhile (runtime/stop.py)."""
        for stream in streams:
            done = torch.cuda.Event()
            done.record(stream)
            while not done.query():
                time.sleep(0.001)


@dataclass
class _Field:
    """One parameter in its part's bytes: `size` bytes at `offset`."""

    key: str
    offset: int
    size: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    parameter: torch.nn.Parameter | None = None


class Part:
    """Parameters that leave the device and come back together, laid out in one run of `size`
    bytes: `host` holds them always, and `device` is the device buffer holding them, None while
    they are away. `tensors` gives each parameter's dtype and shape by its name in the model."""

    def __init__(
        self, name: str, tensors: Mapping[str, tuple[torch.dtype, tuple[int, ...]]], backend: Cuda
    ) -> None:
        self.name = name
        self.fields: list[_Field] = []
        offset = 0
        for key, (dtype, shape) in tensors.items():
            size = dtype.itemsize * prod(shape)
            self.fields.append(_Field(key, offset, size, dtype, shape))
            offset += _aligned(size)
        self.size = offset
        self.host = backend.host(self.size)
        self.device: Tensor | None = None

    def view(self, buffer: Tensor, field: _Field) -> Tensor:
        """field's parameter as buffer holds it: no copy."""
        return buffer[field.offset : field.offset + field.size].view(field.dtype).view(field.shape)

    def adopt(self, model: torch.nn.Module) -> None:
        """Take model's parameters, loaded as views of the host copy (read), as this part's."""
        for field in self.fields:
            parameter = model.get_parameter(field.key)
            if parameter.data_ptr() != self.view(self.host.bytes, field).data_ptr():
                raise RuntimeError(f"{field.key} isn't loaded in its part's host copy")
            field.parameter = parameter

    def bind(self, buffer: Tensor) -> None:
        """Make the part's parameters views of buffer, which holds its bytes, or will before
        they are read."""
        for field in self.fields:
            assert field.parameter is not None, "adopted first"
            field.parameter.data = self.view(buffer, field)

    def enter(self, backend: Cuda) -> None:
        """Bring the part to the device: one buffer, one copy from the host's, queued on the
        compute stream, where what computes next comes after it."""
        buffer = backend.allocate(self.size)
        backend.copy(buffer, self.host.bytes[: self.size], backend.compute(), blocking=False)
        self.bind(buffer)
        self.device = buffer

    def leave(self) -> None:
        """Send the part away: its parameters are the host copy's again, and its device buffer
        goes. The caller has waited for the device's work first."""
        self.bind(self.host.bytes)
        self.device = None


def dit_parts(blocks: int, stay: int, other: bool) -> Callable[[str], str | None]:
    """The part of each parameter of a DiT of `blocks` blocks, by its name, None for one that
    never leaves the device: each block from the `stay`-th on is a part, and with `other` the
    parameters outside the blocks are one more."""

    def part_of(key: str) -> str | None:
        match = _BLOCK.match(key)
        if match is None:
            return OTHER if other else None
        return _block(int(match[1])) if stay <= int(match[1]) < blocks else None

    return part_of


def read(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None,
    part_of: Callable[[str], str | None] | None,
    backend: Cuda | None = None,
) -> tuple[dict[str, Tensor], dict[str, Part]]:
    """The tensors of the safetensors checkpoint at path, for model.load_state_dict(assign=True),
    on device, every floating one cast to dtype if given, there, as numz casts a model loaded
    whole; but the parameters part_of names a part for, if it is given: those go to their part's
    host copy, pinned, and never to the device whole (DESIGN.md, BlockSwap, rewritten: the
    loader). Returns the tensors and the parts, by name, each to adopt the model once loaded.
    Raises HostMemoryError before anything is pinned when the host has too little memory
    available.

    The file is read, never mapped, from its start to its end, each tensor's bytes at the
    file's own offsets: a part's straight into its pinned copy, any other tensor's into one
    buffer the size of the largest of them, which it leaves for the device before the next is
    read. A file cut short since weights.check passed it is a ModelError, as the check's own.
    Mapped, as safetensors gives a file's tensors, a page stays in the process from the copy
    that reads it until the mapping goes, at the end of the file's read: with every weight of
    the 7B bound for the device, its file's 15.35 GiB, the process 16.5 GiB resident at the
    load's peak where it holds 1.25 once loaded; and as much whatever a plan pins, the pinned
    copies in the mapped pages' place: 16.8 GiB with 6.38 pinned, 16.3 with the 36 blocks'
    15.19 (and 31.4 when the blocks' copies were made from the mapping too). Read so, the
    three loads peak at 1.1, 7.5 and 16.3 GiB, every tensor's bytes the same (measured on
    the 96 GB card with research/scripts/load_peak.py, 2026-10-12;
    research/docs/load-memory.md)."""
    backend = Cuda(device) if backend is None else backend
    parameters = {name for name, _ in model.named_parameters()}
    with open(path, "rb") as raw:

        def fill(target: memoryview, at: int, whole: int | None = None) -> None:
            # target, from the file's byte at on. weights.check has passed the file whole: one
            # cut short since, as it loads, is refused as the check refuses one, in a line,
            # with the bytes it holds and those it should (whole, by its header; inside the
            # header, the least it can), rather than read into wrong tensors or a traceback.
            raw.seek(at)
            done = 0
            while done < len(target):
                got = raw.readinto(target[done:])
                if not got:
                    of = f"at least {at + len(target):,}" if whole is None else f"{whole:,}"
                    raise ModelError(
                        f"{path}: cut short as it loaded, {at + done:,} of {of} bytes; {AGAIN}"
                    )
                done += got

        # The file's layout, as safetensors lays it: the header's length, the header, the data,
        # each tensor at its offsets from the header's end. weights.check has validated it.
        size = bytearray(8)
        fill(memoryview(size), 0)
        length = int.from_bytes(size, "little")
        text = bytearray(length)
        fill(memoryview(text), 8)
        header = cast("dict[str, Any]", json.loads(text))
        header.pop("__metadata__", None)
        tensors: dict[str, tuple[torch.dtype, tuple[int, ...], int, int]] = {}
        # By their offsets: the order the bytes are in, whatever the header's.
        for key, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            if entry["dtype"] not in _DTYPES:
                raise RuntimeError(f"{path.name}: {key} is {entry['dtype']}, which isn't read")
            found, shape = _DTYPES[entry["dtype"]], tuple(cast("list[int]", entry["shape"]))
            start, end = cast("tuple[int, int]", tuple(entry["data_offsets"]))
            if end - start != found.itemsize * prod(shape):
                raise RuntimeError(
                    f"{path.name}: {key} takes {end - start} bytes, not"
                    f" {found.itemsize * prod(shape)}"
                )
            tensors[key] = (found, shape, start, end)
        whole = 8 + length + max((end for *_, end in tensors.values()), default=0)
        layouts: dict[str, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
        for key, (found, shape, _, _) in tensors.items():
            name = part_of(key) if part_of is not None and key in parameters else None
            if name is not None:
                final = found if dtype is None or not found.is_floating_point else dtype
                layouts.setdefault(name, {})[key] = (final, shape)
        parts = _make(layouts, backend)
        fields = {field.key: (part, field) for part in parts.values() for field in part.fields}
        # What isn't read straight into a pinned copy passes through here, one tensor at a
        # time: the largest of those tensors' size. For the 7B, its largest, 108 MiB (outside
        # the blocks), or at the most a block's largest, 72 MiB, when a plan pins the
        # parameters outside the blocks; for the VAE, cast as it loads, 13.5 MiB (the files'
        # headers).
        through = [
            end - start
            for key, (found, _, start, end) in tensors.items()
            if key not in fields or (dtype is not None and found.is_floating_point)
        ]
        staging = torch.empty(max(through, default=0), dtype=torch.uint8)
        state: dict[str, Tensor] = {}
        for key, (found, shape, start, end) in tensors.items():
            recast = dtype is not None and found.is_floating_point
            placed = fields.get(key)
            if placed is not None and not recast:
                part, field = placed
                state[key] = part.view(part.host.bytes, field)
                target = part.host.bytes[field.offset : field.offset + field.size]
            else:
                target = staging[: end - start]
            fill(memoryview(target.numpy()), 8 + length + start, whole)
            if placed is not None and not recast:
                continue
            # A copy on the CPU too, where to() alone gives the buffer's own memory, which the
            # next tensor's bytes replace. Cast on the device, as the whole-model load casts
            # there: the same values.
            moved = target.view(found).view(shape).to(device, copy=True)
            moved = moved.to(dtype) if recast else moved
            if placed is None:
                state[key] = moved
            else:
                part, field = placed
                state[key] = part.view(part.host.bytes, field)
                state[key].copy_(moved)
    return state, parts


def host(
    model: torch.nn.Module, part_of: Callable[[str], str | None], backend: Cuda
) -> dict[str, Part]:
    """The parts of a loaded model, by name, as part_of names them: each parameter is copied to
    its part's host copy from where it is, and is that copy's view from then on, away from the
    device. For a model loaded whole, on a card that holds it; read is the loader's way."""
    layouts: dict[str, dict[str, tuple[torch.dtype, tuple[int, ...]]]] = {}
    for key, parameter in model.named_parameters():
        name = part_of(key)
        if name is not None:
            layouts.setdefault(name, {})[key] = (parameter.dtype, tuple(parameter.shape))
    parts = _make(layouts, backend)
    for part in parts.values():
        for field in part.fields:
            field.parameter = model.get_parameter(field.key)
            part.view(part.host.bytes, field).copy_(field.parameter.data)
        part.bind(part.host.bytes)
    return parts


def available(proc: Path = Path("/proc"), cgroups: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The host memory this process can still take, in bytes: what the kernel estimates
    available (MemAvailable of /proc/meminfo), or less in a container held to a limit, where
    /proc/meminfo is the host's and knows nothing of it: what is left under the nearest limit
    of its control group and of the groups above it. A group's limit is cgroup v2's memory.max
    or memory.high (past which the kernel reclaims from it and slows it down), v1's
    memory.limit_in_bytes; what it holds, memory.current (v1: memory.usage_in_bytes) less its
    file cache (memory.stat's active_file and inactive_file; v1: total_ of each), which the
    kernel gives back when asked, as MemAvailable counts the host's: a model file just hashed
    is 16 GiB of it. None where nothing can be read. The planner counts the job's host memory
    with it before the load (runtime/planner.py)."""
    left: list[int] = []
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                left.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    try:
        groups = (proc / "self" / "cgroup").read_text().splitlines()
    except OSError:
        groups = []
    for group in groups:
        fields = group.split(":", 2)  # the hierarchy's number, its controllers, the group's path
        if len(fields) != 3:
            continue
        controllers, path = fields[1], fields[2]
        if controllers == "":  # v2: one hierarchy
            directory, used, prefix = cgroups / path.lstrip("/"), "memory.current", ""
            limits = ("memory.max", "memory.high")
        elif "memory" in controllers.split(","):  # v1: the memory controller's own
            directory = cgroups / "memory" / path.lstrip("/")
            limits, used, prefix = ("memory.limit_in_bytes",), "memory.usage_in_bytes", "total_"
        else:
            continue
        while True:
            for limit in limits:
                try:
                    text = (directory / limit).read_text().strip()
                    if text == "max":
                        continue
                    most, held = int(text), int((directory / used).read_text())
                except (OSError, ValueError):
                    continue  # no such file there: the root, or a group without the controller
                cache = 0
                try:
                    for line in (directory / "memory.stat").read_text().splitlines():
                        if line.split()[0] in (f"{prefix}active_file", f"{prefix}inactive_file"):
                            cache += int(line.split()[1])
                except (OSError, ValueError, IndexError):
                    cache = 0  # unknown: all of what the group holds counts as taken
                left.append(max(most - max(held - cache, 0), 0))
            if directory in (cgroups, cgroups / "memory") or directory == directory.parent:
                break
            directory = directory.parent
    return min(left) if left else None


def _make(
    layouts: Mapping[str, Mapping[str, tuple[torch.dtype, tuple[int, ...]]]], backend: Cuda
) -> dict[str, Part]:
    """The parts laid out by layouts, each with its host copy. Refused before the first is
    pinned when their sum passes the memory available: page-locking faults every page in, and a
    host short of memory kills the process rather than refuse (no pageable fallback in v1)."""
    sizes = {
        name: sum(_aligned(dtype.itemsize * prod(shape)) for dtype, shape in layout.values())
        for name, layout in layouts.items()
    }
    needed, free = sum(sizes.values()), available()
    if free is not None and needed > free:
        raise HostMemoryError(
            f"the weights that leave the GPU need {_gib(needed)} of host memory, pinned"
            f" ({listing(sizes)}), and the host has {_gib(free)} available"
        )
    parts: dict[str, Part] = {}
    try:
        for name, layout in layouts.items():
            parts[name] = Part(name, layout, backend)
    except HostMemoryError as error:
        raise HostMemoryError(
            f"the weights that leave the GPU need {_gib(needed)} of host memory, pinned"
            f" ({listing(sizes)}): {error}, after {_gib(sum(p.size for p in parts.values()))}"
        ) from error
    return parts


def listing(names: Iterable[str]) -> str:
    """What the parts of these names are, in words: the DiT's blocks counted, the others named."""
    names = list(names)
    blocks = [name for name in names if _BLOCK.match(name + ".")]
    others = [name for name in names if name not in blocks]
    words = [f"{len(blocks)} of the DiT's blocks"] if blocks else []
    words += ["the DiT's other parameters" if name == OTHER else name for name in others]
    return ", ".join(words)


def _cudart() -> Any:
    """CUDA's runtime as torch binds it, untyped: cudaHostRegister and cudaHostUnregister."""
    return torch.cuda.cudart()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _aligned(size: int) -> int:
    return -(-size // ALIGN) * ALIGN


def _gib(size: int) -> str:
    return f"{size / GIB:.2f} GiB"


def _block(index: int) -> str:
    return f"blocks.{index}"


@dataclass
class _Slot:
    """A device buffer the swapped blocks take turns in: `holds` is the block whose bytes it
    holds, or will once `ready` (its copy done, on the side stream); `done` marks the end of its
    last reader's work on the compute stream."""

    bytes: Tensor
    holds: int | None = None
    ready: Any = None
    done: Any = None


class Placement:
    """The DiT's weights between the device and the host: the last `swapped` of its blocks
    streaming through SLOTS device buffers in its own phase, and what each phase keeps on the
    device (enter).

    dit is the model, loaded: `parts` gives the parts read into host copies (read), by name;
    without it, the swapped blocks' are made here from the model as it is (host). A block that is
    neither swapped nor a part stays on the device for good. After the constructor, and until
    enter is called, every part is away.

    prefetch: the next block of a slot is copied on a side stream while the other slot's block
    computes; without it, each block is copied on the compute stream before its forward, the
    host waiting. verify, for the tests only, compares each slot with its block's host copy
    before the block runs (verified counts them).

    The swapped blocks are numbered from 0, in the DiT's order: block j runs in slot j % SLOTS.
    """

    def __init__(
        self,
        dit: torch.nn.Module,
        device: torch.device,
        swapped: int,
        prefetch: bool = True,
        *,
        parts: Mapping[str, Part] | None = None,
        verify: bool = False,
        backend: Cuda | None = None,
    ) -> None:
        blocks = cast("list[torch.nn.Module]", list(cast(Any, dit).blocks))
        if not 0 <= swapped <= len(blocks):
            raise ValueError(f"{swapped} blocks to swap of the DiT's {len(blocks)}")
        self.device = device
        self.swapped = swapped
        self.prefetch = prefetch
        self.verify = verify
        self.verified = 0
        self.uploads = 0
        self._backend = Cuda(device) if backend is None else backend
        self._fixed = len(blocks) - swapped
        if parts is None:
            parts = host(dit, dit_parts(len(blocks), self._fixed, other=False), self._backend)
        self._parts = dict(parts)
        missing = [j for j in range(self._fixed, len(blocks)) if _block(j) not in self._parts]
        if missing:
            raise ValueError(f"no host copy for the swapped blocks {missing}")
        # Every block's bytes on the device, and the other parameters': a part's are its run of
        # bytes, alignment included, as its device buffer takes them.
        named = dict(dit.named_parameters())
        raw = {name: 0 for name in [*map(_block, range(len(blocks))), OTHER]}
        for key, parameter in named.items():
            match = _BLOCK.match(key)
            raw[OTHER if match is None else _block(int(match[1]))] += (
                parameter.numel() * parameter.element_size()
            )
        self._sizes = {
            name: self._parts[name].size if name in self._parts else size
            for name, size in raw.items()
        }
        self._streaming = [self._parts[_block(j)] for j in range(self._fixed, len(blocks))]
        self._slots: list[_Slot] = []
        self._side: Any = None
        self._closed = False
        self._hooks: list[Any] = []
        for j, block in enumerate(blocks[self._fixed :]):
            self._hooks.append(block.register_forward_pre_hook(self._before(j)))
            # always_call: a forward that raises still marks its slot's reader done.
            self._hooks.append(block.register_forward_hook(self._after(j), always_call=True))

    @property
    def block_bytes(self) -> int:
        """The device bytes of a slot: the largest swapped block's; 0 with none swapped."""
        return max((part.size for part in self._streaming), default=0)

    @property
    def host_bytes(self) -> int:
        """The host memory the parts' pinned copies take, in bytes."""
        return sum(int(part.host.bytes.numel()) for part in self._parts.values())

    def device_bytes(self, resident: int | None = None) -> int:
        """The bytes of the DiT's parameters on the device in a phase entered with `resident`
        (enter): None, the DiT's own phase: the blocks that aren't swapped, the other
        parameters and the slots; else that many blocks, and the other parameters unless 0."""
        kept, other, slots = self._kept("encode", resident)
        names = [*map(_block, range(kept)), *([OTHER] if other else [])]
        return sum(self._sizes[name] for name in names) + (
            min(SLOTS, self.swapped) * self.block_bytes if slots else 0
        )

    def enter(self, phase: Phase, resident: int | None = None) -> None:
        """Bring the DiT to what a phase keeps of it on the device. The DiT's own phase, "dit",
        keeps the blocks that aren't swapped, the other parameters and the slots, the first
        swapped blocks on their way into them. A VAE phase, "encode" or "decode", keeps
        `resident`: None, what the DiT's phase keeps, nothing moving; else its first `resident`
        blocks, of those that aren't swapped, with the other parameters, and no slot; or, at 0,
        nothing: the VAE finds the DiT gone (research/bugs/12, 13: numz's leaves the GPU in
        name only).

        What leaves goes first, once the device's work is done, and goes to its host copy by
        name alone: nothing is copied back. What comes back is queued on the compute stream,
        one copy per part: the caller needn't wait."""
        if self._closed:
            raise RuntimeError("the placement is closed: its hooks are off the blocks")
        kept, other, slots = self._kept(phase, resident)
        fixed = [self._parts.get(_block(j)) for j in range(self._fixed)]
        rest = self._parts.get(OTHER)
        leaving = [part for part in fixed[kept:] if part is not None and part.device is not None]
        if rest is not None and rest.device is not None and not other:
            leaving.append(rest)
        for j, part in enumerate(fixed[kept:], kept):
            if part is None:
                raise ValueError(f"block {j} has no host copy to leave the device for")
        if rest is None and not other:
            raise ValueError("the DiT's other parameters have no host copy to leave the device for")
        if leaving or (self._slots and not slots):
            self._wait()
        for part in leaving:
            part.leave()
        if self._slots and not slots:
            for part in self._streaming:
                part.bind(part.host.bytes)
            self._slots = []
            self._backend.trim()
        elif leaving:
            self._backend.trim()
        for part in [*fixed[:kept], *([rest] if other else [])]:
            if part is not None and part.device is None:
                part.enter(self._backend)
        if slots and not self._slots:
            self._slots = [
                _Slot(self._backend.allocate(self.block_bytes))
                for _ in range(min(SLOTS, self.swapped))
            ]
            for j, part in enumerate(self._streaming):
                part.bind(self._slots[j % len(self._slots)].bytes)
            # A new buffer may be memory the compute stream freed a moment ago, which work still
            # queued there reads or writes: the allocator gives it back to that stream's next
            # allocation at once (torch's Tensor.record_stream says so of its own allocator).
            # The slots' first copies come from the side stream: they wait for the compute
            # stream to be past this point, as later ones wait for the slot's last reader.
            behind = self._backend.event()
            behind.record(self._backend.compute())
            for slot in self._slots:
                slot.done = behind
            if self.prefetch:
                # The first blocks' uploads, behind the blocks that stay: every block of the DiT
                # before them computes meanwhile.
                self._side = self._backend.side() if self._side is None else self._side
                for j in range(len(self._slots)):
                    self._fetch(j)

    def close(self) -> None:
        """Send every part away, the slots with them, and take the hooks off the blocks: the
        model's parts are their host copies again, for another placement to take; this one
        enters no phase again. To call before a placement is dropped while the device may
        still work: only the waits here and in enter keep a slot, or a host copy's pinning,
        from going under a copy still queued."""
        self._wait()
        for part in self._parts.values():
            part.leave()
        self._slots = []
        self._backend.trim()
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._closed = True

    def _kept(self, phase: Phase, resident: int | None) -> tuple[int, bool, bool]:
        """What a phase keeps on the device: how many of the blocks that aren't swapped, whether
        the other parameters, whether the slots."""
        if phase == "dit" or resident is None:
            if phase == "dit" and resident not in (None, self._fixed):
                raise ValueError(f"the DiT's phase keeps its {self._fixed} blocks, not {resident}")
            return self._fixed, True, self.swapped > 0
        if not 0 <= resident <= self._fixed:
            raise ValueError(f"{resident} blocks to keep of the {self._fixed} that aren't swapped")
        return resident, resident > 0, False

    def _wait(self) -> None:
        """Wait for everything that reads or writes the device buffers: the blocks' work on the
        compute stream, the copies on the side stream. A buffer freed under a copy still queued
        would be written to after its memory is given to another tensor: the allocator knows
        nothing of the side stream's use of it."""
        streams = [self._backend.compute(), *([] if self._side is None else [self._side])]
        self._backend.wait(*streams)

    def _fetch(self, j: int) -> None:
        """Queue swapped block j's copy into its slot on the side stream, after the end of the
        slot's last reader, and mark its end: the host waits for neither."""
        slot, part = self._slots[j % len(self._slots)], self._streaming[j]
        if slot.done is not None:
            self._side.wait_event(slot.done)
        self._backend.copy(
            slot.bytes[: part.size], part.host.bytes[: part.size], self._side, blocking=False
        )
        slot.ready = self._backend.event()
        slot.ready.record(self._side)
        slot.holds = j
        self.uploads += 1

    def _before(self, j: int) -> Callable[..., None]:
        """The hook that runs before swapped block j's forward: its slot holds its bytes before
        the block's work starts on the compute stream."""

        def before(module: torch.nn.Module, args: Any) -> None:
            if not self._slots:
                raise RuntimeError("the DiT runs outside its phase: its swapped blocks are away")
            slot, part = self._slots[j % len(self._slots)], self._streaming[j]
            compute = self._backend.compute()
            if self.prefetch:
                if slot.holds != j:
                    # Out of step, after a forward that raised: fetched now, on the side stream.
                    self._fetch(j)
                compute.wait_event(slot.ready)
            elif slot.holds != j:
                # On the compute stream, behind the slot's last reader, the host waiting.
                host = part.host.bytes[: part.size]
                self._backend.copy(slot.bytes[: part.size], host, compute, blocking=True)
                slot.holds = j
                self.uploads += 1
            if self.verify:
                self._wait()
                if not torch.equal(slot.bytes[: part.size].cpu(), part.host.bytes[: part.size]):
                    raise RuntimeError(f"slot {j % len(self._slots)} doesn't hold {part.name}")
                self.verified += 1

        return before

    def _after(self, j: int) -> Callable[..., None]:
        """The hook that runs after swapped block j's forward, its work queued on the compute
        stream: the end of that work is marked, and the slot's next block fetched behind it,
        block j + SLOTS, or past the last one the slot's first, for the next forward."""

        def after(module: torch.nn.Module, args: Any, output: Any) -> None:
            if not self.prefetch or not self._slots:
                return
            slot = self._slots[j % len(self._slots)]
            slot.done = self._backend.event()
            slot.done.record(self._backend.compute())
            following = j + len(self._slots)
            following = following if following < self.swapped else j % len(self._slots)
            if following != j:
                self._fetch(following)

        return after
