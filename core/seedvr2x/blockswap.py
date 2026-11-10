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

Only parameters move. Buffers stay on the GPU: the rotary buffers (3 per block of the 7B, 0.7 MiB)
and the angle tables each block caches from them, so numz's device-aware RoPE wrapper, and its
late-binding bug (research/bugs/24), have no place here.

A part (Part) is a set of parameters that leaves and comes back together: one block of the DiT,
the DiT's parameters outside its blocks, or the VAE. Its parameters are laid out in one run of
bytes, so that a part moves in one copy, and a module's parameters are views into whichever
buffer holds those bytes at the time: the host copy, a buffer on the device, or a slot.
"""

import ctypes
import hashlib
import json
import logging
import mmap
import os
import re
import sys
import weakref
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from io import BufferedReader, BytesIO
from math import prod
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

import torch
from torch import Tensor

from seedvr2x import gguf_file, memory, waiting
from seedvr2x.errors import HostMemoryError, ModelChanged, ModelDamaged
from seedvr2x.gguf_file import GgufError
from seedvr2x.passes import Phase
from seedvr2x.weights import AGAIN, SLOTS

logger = logging.getLogger(__name__)

GIB = 1 << 30

# Each parameter starts on a multiple of this in its part's bytes, as torch's CUDA allocator
# aligns its own blocks (512 bytes): 30 parameters waste under 16 KiB of a block's 0.42 GiB.
ALIGN = 512
# The part of the DiT's parameters outside its blocks.
OTHER = "other"

_BLOCK = re.compile(r"blocks\.(\d+)\.")
# torch's dtype for each of safetensors' names the loader reads (read), each held to
# safetensors' own reading by tests/test_blockswap.py. A file seedvr2x runs holds F16 alone, or in
# an int8 file I8 and F32 besides, the block matrices and their scales, and U8, their markers,
# which the loader doesn't read (weights.check; model.load_models); the other names of
# weights.DTYPE_BYTES, the unsigned ones over 8 bits and fp8's two, which no such file holds, are
# refused there rather than read untested. A GGUF file's tensors are named by ggml's types
# (gguf_file.TYPES), the same names for the plain ones (F16, F32...); its quantised ones, in
# blocks of bytes, are read as U8 (_gguf_tensors).
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


def _anonymous(length: int) -> mmap.mmap:
    """An anonymous mapping of `length` bytes (a multiple of the page size), zeros, of the
    platform's own kind: Python's mmap takes flags and madvise on POSIX only, and on Windows
    mmap.mmap(-1, length) is the anonymous mapping, which no child process inherits unless it is
    made to (seedvr2x starts none that takes it)."""
    if sys.platform == "win32":
        return mmap.mmap(-1, length)
    area = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    # Not a forked child's: the kernel would copy every pinned page for it at the fork,
    # the 15.81 GiB pinned for a plan that keeps the two models apart (the whole DiT and
    # the VAE, measured, milestone 3, 2026-10-11). seedvr2x forks nothing (its
    # ffmpeg processes are spawned); a child that did would have no weights to read.
    area.madvise(mmap.MADV_DONTFORK)
    return area


class HostCopy:
    """`size` bytes of host memory, page-locked for CUDA when `pin`: a private anonymous mapping
    of its own (_anonymous: POSIX's, or Windows' anonymous mmap), registered with
    cudaHostRegister, so that it takes its size and no more. torch's own pinned allocator rounds
    each allocation up to a power of two: a 7B block's 0.4220 GiB
    took 0.5 GiB there, 18.0 GiB for the 36 blocks where their bytes are 15.19 (torch 2.14.1).
    Copies from it to the device are as fast as from torch's (54.6 against 53.4 GB/s on a desktop
    PCIe 5.0 link, 18.2 from pageable memory) and return at once (0.12 ms): measured by
    research/scripts/pin_probe.py, on such a mapping, 2026-10-10, on an RTX 5090 under WSL.
    DESIGN.md's 57 GB/s pinned and 25 pageable (BlockSwap, rewritten) are research/docs/vram.md's,
    standalone copies on the 96 GB card's PCIe 5.0 x16 link: another machine, the same ratio.

    bytes is the buffer, (size rounded up to a page,) uint8, zeros at first. Raises
    HostMemoryError when the host refuses the mapping or its page-locking, or when this torch
    doesn't bind CUDA's call for it (torch.cuda.cudart(), which torch binds without documenting
    it): a refusal before the run, never a pageable copy in the pinned one's place. The pinning
    is torch's binding of the runtime's cudaHostRegister on every platform (no POSIX call in it);
    the Windows mapping is checked by its call alone, on the CPU (tests/test_blockswap.py), never
    pinned on a Windows card by seedvr2x's tests."""

    def __init__(self, size: int, pin: bool) -> None:
        length = -(-max(size, 1) // mmap.PAGESIZE) * mmap.PAGESIZE
        try:
            area = _anonymous(length)
        except OSError as error:
            raise HostMemoryError(
                f"the host refuses {_gib(length)} of memory: {error}", need=length
            ) from error
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
                    f" isn't to be had from this torch ({torch.__version__}): {error}",
                    need=length,
                ) from error
            address: int = self.bytes.data_ptr()
            code: Any = register(address, length, 0)
            if int(code) != 0:
                # torch names no code but success: CUDA's own words for it.
                raise HostMemoryError(
                    f"the host refuses to pin {_gib(length)} of memory: {words(code)}", need=length
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
        29.3 with this, for 0.28 s (DESIGN.md, BlockSwap, rewritten, Per phase).

        Nothing where a caller keeps a reservation (memory.keep, the image server's): the freed
        weights' memory is then the reservation's, which the next phase's work is counted in;
        given back to the card, it would leave the reservation at every phase that sends
        weights away (an RTX 5090 under WSL2, 2026-11-04: at every picture of 12). Under the
        reservation's limit torch gives its cache back itself where an allocation can't
        otherwise be served (memory.keep)."""
        if memory.kept() is not None:
            logger.debug("torch's cache kept as weights leave the GPU: the reservation holds it")
            return
        torch.cuda.empty_cache()

    def wait(self, *streams: Any) -> None:
        """Wait for the work queued on streams, polling as model.synchronize does, so that a
        signal's handler runs meanwhile (stop.py), timed (waiting.wait)."""
        named: Any = self.device.index  # None for a bare "cuda", whatever the stubs say
        index = torch.cuda.current_device() if named is None else named
        for stream in streams:
            done = torch.cuda.Event()
            done.record(stream)
            waiting.wait(done.query, 0.001, index)

    def queries(self, *streams: Any) -> list[Callable[[], bool]]:
        """For each of streams, an event recorded now behind its queued work, as the query that
        says whether that work has finished: the call waits for nothing (the wait on the same
        events is wait's), so that a caller can ask later, without waiting, what a wait that was
        left would have waited for (waiting.expect)."""
        found: list[Callable[[], bool]] = []
        for stream in streams:
            done = self.event()
            done.record(stream)
            found.append(done.query)
        return found


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


# The bytes read at a time into a tensor's memory, each piece then hashed on the hash thread while
# the next is read (_Tee): the wait for the last piece's hash, before a reused buffer is read into
# again, about 3 ms at hashlib's 2.7 GB/s on the Ryzen 9 9950X3D beside the RTX 5090, where WSL2's
# 9p reads at 170 to 250 MB/s whatever the piece. Measured there (DESIGN.md, Still images, Model
# files): the DiT's 16.5 GB read and hashed from the Windows drive in 66.6 s, the read the bound;
# 3.93 GB of it from Linux's disk in 1.7 s, the hash the bound, 2.5 s hashed in turn on the
# read's own thread.
PIECE = 8 << 20


class _Tee:
    """A file read from its start to its end, each byte hashed (SHA-256) in the file's order as
    it is read, as Go's io.TeeReader: the hash is that of the very bytes read, which a whole-file
    hash of the same file gives (hashlib.file_digest). Read into the tensors' own memory: hashlib
    takes any buffer, with no copy, and releases the GIL over 2 KiB, as readinto does, so one
    thread hashes each piece while the next is read. One file's SHA-256 can't be split across
    cores: from a disk faster than hashlib (2.7 GB/s on the Ryzen 9 9950X3D, design's
    measurement of 2026-11-03), the read waits for the hash.

    Without `hashing`, nothing is hashed, the bytes not read skipped by a seek (read's callers
    that ask for no SHA-256: tests).

    `progress`, when given, is told (bytes of the file read so far, the file's size) after each
    piece read (read's callback): it only observes, and may raise to stop the read."""

    def __init__(
        self,
        raw: BufferedReader,
        path: Path,
        hashing: bool,
        progress: Callable[[int, int], None] | None = None,
        size: int = 0,
    ) -> None:
        self.raw = raw
        self.path = path
        self.progress = progress
        self.size = size
        self.position = 0
        self.hash = hashlib.sha256() if hashing else None
        self.thread = ThreadPoolExecutor(1, "seedvr2x-sha256") if hashing else None
        self.pending: list[Future[None]] = []

    def fill(
        self, target: memoryview, at: int, whole: int | None = None, reused: bool = False
    ) -> None:
        """target, from the file's byte at on, which is at or after the bytes read so far: the
        bytes between read and hashed (or skipped, without hashing). `reused`: target is a buffer
        read into before, whose bytes the hash thread may still be reading: waited for first.
        weights.check has passed the file whole: one cut short since, as it loads, is refused as
        the check refuses one, in a line, with the bytes it holds and those it should (whole, by
        its header; inside the header, the least it can), rather than read into wrong tensors or
        a traceback."""
        if at < self.position:
            raise RuntimeError(f"{self.path.name}: bytes at {at:,} read after {self.position:,}")
        self.skip(at, whole)
        if reused:
            self.wait()
        done = 0
        while done < len(target):
            piece = target[done : done + PIECE] if self.hash is not None else target[done:]
            got = self.raw.readinto(piece)
            if not got:
                self.cut(at + done, at + len(target), whole)
            self.hashed(piece[:got])
            done += got
            self.told(at + done)
        self.position = at + len(target)

    def skip(self, to: int, whole: int | None) -> None:
        """The bytes from those read so far up to the file's byte `to`, which no tensor takes (the
        tensors skip leaves out; weights.check refuses a gap): read and hashed, or skipped by a
        seek without hashing."""
        if self.hash is None:
            self.raw.seek(to)
        while self.position < to and self.hash is not None:
            data = self.raw.read(min(PIECE, to - self.position))
            if not data:
                self.cut(self.position, to, whole)
            self.hashed(memoryview(data))
            self.position += len(data)
            self.told(self.position)
        self.position = max(self.position, to)

    def finish(self) -> str | None:
        """The SHA-256 of the whole file, in hex, its bytes after the last tensor's read and hashed
        to its end; None without hashing."""
        if self.hash is None:
            return None
        while data := self.raw.read(PIECE):
            self.hashed(memoryview(data))
            self.position += len(data)
            self.told(self.position)
        self.wait()
        return self.hash.hexdigest()

    def told(self, position: int) -> None:
        """The progress callback told the bytes of the file read, up to byte `position`."""
        if self.progress is not None:
            self.progress(position, self.size)

    def hashed(self, data: memoryview) -> None:
        """data, just read, hashed on the hash thread after every piece before it."""
        if self.thread is not None and self.hash is not None:
            self.pending.append(self.thread.submit(self.hash.update, data))

    def wait(self) -> None:
        """Wait until every piece read so far is hashed, an error of the hash thread raised."""
        for future in self.pending:
            future.result()
        self.pending.clear()

    def close(self) -> None:
        """The hash thread ended, the pieces it has yet to hash dropped."""
        if self.thread is not None:
            self.thread.shutdown(wait=True, cancel_futures=True)

    def cut(self, at: int, least: int, whole: int | None) -> NoReturn:
        """Refuse the file, cut short at byte `at` where `least` bytes at least were to be read."""
        of = f"at least {least:,}" if whole is None else f"{whole:,}"
        raise ModelDamaged(
            f"{self.path}: cut short as it loaded, {at:,} of {of} bytes; {AGAIN}",
            file=self.path.name,
            figure="size",
            expected=whole,
            found=at,
        )


def read(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None,
    part_of: Callable[[str], str | None] | None,
    backend: Cuda | None = None,
    *,
    skip: Callable[[str], bool] | None = None,
    hashed: Callable[[str, os.stat_result], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Part]]:
    """The tensors of the safetensors or GGUF checkpoint at path (told by its signature; a GGUF
    file's quantised matrices as the uint8 bytes of ggml's blocks, _gguf_tensors), for
    model.load_state_dict(assign=True), on device, every floating one cast to dtype if given,
    there, as numz casts a model loaded whole; but the parameters part_of names a part for, if
    it is given: those go to their part's host copy, pinned, and never to the device whole
    (DESIGN.md, BlockSwap, rewritten: the loader); and none of those skip says True of, if it is
    given, which aren't kept: an int8 file's markers, which no model holds (model.load_models).
    Returns the tensors and the parts, by name, each to adopt the model once loaded. Raises
    HostMemoryError before anything is pinned when the host has too little memory available.

    With `hashed`, every byte of the file is hashed as it is read, in the file's order, the
    header and the tensors skipped included (_Tee), and hashed is called at the end with the
    file's SHA-256, in hex, and the file's status as it was opened (os.fstat): the SHA-256 of the
    very bytes loaded, the file read once (DESIGN.md, Weights, decided on 2026-11-03). Whatever
    hashed raises (a ModelError: pull.Hashes.proved) ends the load. With `progress`, it is told the
    bytes of the file read so far and the file's size after each piece (PIECE with a hash, else
    each tensor), the header and the bytes skipped included: it only observes, and may raise to
    stop the read, the pinned copies made so far given up with the exception.

    The file is read, never mapped, from its start to its end, each tensor's bytes at the file's
    own offsets: a part's straight into its pinned copy, any other tensor's into one buffer the
    size of the largest of them, which it leaves for the device before the next is read. A file
    cut short since weights.check passed it is a ModelError, as the check's own. Mapped, as
    safetensors gives a file's tensors, a page stays in the process from the copy that reads it
    until the mapping goes, at the end of the file's read: with every weight of the 7B bound for
    the device, its file's 15.35 GiB, the process 16.5 GiB resident at the load's peak where it
    holds 1.25 once loaded; and as much whatever a plan pins, the pinned copies in the mapped
    pages' place: 16.8 GiB with 6.38 pinned, 16.3 with the 36 blocks' 15.19 (and 31.4 when the
    blocks' copies were made from the mapping too). Read so, the three loads peak at 1.1, 7.5
    and 16.3 GiB, every tensor's bytes the same (measured on the 96 GB card with
    research/scripts/load_peak.py, 2026-10-12; research/docs/load-memory.md)."""
    backend = Cuda(device) if backend is None else backend
    parameters = {name for name, _ in model.named_parameters()}
    with open(path, "rb") as raw:
        opened = os.fstat(raw.fileno())
        tee = _Tee(raw, path, hashed is not None, progress, opened.st_size)
        try:
            state, parts = _read(
                tee, model, parameters, path, device, dtype, part_of, backend, skip
            )
            digest = tee.finish()
        finally:
            tee.close()
    if hashed is not None:
        assert digest is not None
        hashed(digest, opened)
    return state, parts


# A tensor as a file lays it out: torch's dtype and shape as the model holds it, and where its
# bytes begin and end, in bytes from the layout's base.
_Entry = tuple[torch.dtype, tuple[int, ...], int, int]


def _read(
    tee: _Tee,
    model: torch.nn.Module,
    parameters: set[str],
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None,
    part_of: Callable[[str], str | None] | None,
    backend: Cuda,
    skip: Callable[[str], bool] | None,
) -> tuple[dict[str, Tensor], dict[str, Part]]:
    """read's tensors and parts, read through tee. The file is a GGUF file or a safetensors one,
    told by its first 8 bytes; either way its tensors are read in the order of their bytes, each
    at its offset, and every byte between them hashed where the file is."""
    fill = tee.fill
    first = bytearray(8)
    fill(memoryview(first), 0)
    # A safetensors file starts with its header's length, which "GGUF" read as one (1.18 GB)
    # would put over weights.HEADER_LIMIT.
    if bytes(first[: len(gguf_file.MAGIC)]) == gguf_file.MAGIC:
        tensors, base, whole = _gguf_tensors(tee, path, skip, bytes(first))
    else:
        tensors, base, whole = _safetensors_tensors(tee, path, skip, first)
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
            reused = False
        else:
            # The staging buffer, whose last tensor's bytes may still be hashing.
            target, reused = staging[: end - start], True
        fill(memoryview(target.numpy()), base + start, whole, reused)
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
    # The tensors skipped after the last one read, up to the data's end.
    tee.skip(whole, whole)
    return state, parts


def _safetensors_tensors(
    tee: _Tee, path: Path, skip: Callable[[str], bool] | None, first: bytearray
) -> tuple[dict[str, _Entry], int, int]:
    """The tensors to read of the safetensors file at path, by name, in the order of their bytes;
    where its data starts (the base of their offsets: 8 and the header's length); and the file's
    whole size by its header, the tensors skip leaves out counted. `first` is the file's first 8
    bytes, read through tee already; the header is read through it, hashed with the rest.
    weights.check has validated the file."""
    fill = tee.fill
    # The file's layout, as safetensors lays it: the header's length, the header, the data,
    # each tensor at its offsets from the header's end.
    length = int.from_bytes(first, "little")
    text = bytearray(length)
    fill(memoryview(text), 8)
    header = cast("dict[str, Any]", json.loads(text))
    header.pop("__metadata__", None)
    tensors: dict[str, _Entry] = {}
    # By their offsets: the order the bytes are in, whatever the header's. The tensors skipped
    # are read past (_Tee.skip), hashed where the file is.
    for key, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        if skip is not None and skip(key):
            continue
        if entry["dtype"] not in _DTYPES:
            raise RuntimeError(f"{path.name}: {key} is {entry['dtype']}, which isn't read")
        found, shape = _DTYPES[entry["dtype"]], tuple(cast("list[int]", entry["shape"]))
        start, end = cast("tuple[int, int]", tuple(entry["data_offsets"]))
        if end - start != found.itemsize * prod(shape):
            raise RuntimeError(
                f"{path.name}: {key} takes {end - start} bytes, not {found.itemsize * prod(shape)}"
            )
        tensors[key] = (found, shape, start, end)
    # The file's whole size by its header, the tensors skipped counted.
    ends = (cast("list[int]", entry["data_offsets"])[1] for entry in header.values())
    return tensors, 8 + length, 8 + length + max(ends, default=0)


def _gguf_tensors(
    tee: _Tee, path: Path, skip: Callable[[str], bool] | None, first: bytes
) -> tuple[dict[str, _Entry], int, int]:
    """The tensors to read of the GGUF file at path, by name, in the order of their bytes; 0, the
    base of their offsets, which are absolute (gguf_file); and the file's size. `first`: the
    file's first 8 bytes, read through tee already. The header is parsed by gguf_file on the file
    itself to learn where its data starts (74,272 bytes, and the padding to the first tensor),
    those bytes are read again through tee, hashed with the rest, and parsed again from the very
    bytes hashed, so that the offsets the tensors are read at are those of the bytes the SHA-256
    covers, as a safetensors file's header is. A file whose header read twice differs changed
    as it loaded, and is refused.

    A plain type (F16...) is read as torch's dtype of that name, in its shape. A quantised one is
    read as the bytes the file holds, uint8 (rows, bytes of a row): the weight of a
    kquant.GGUFLinear, which the placement moves as it moves any parameter. A file cut short since
    weights.check passed it is refused in safetensors' words (_Tee.cut)."""
    size = os.fstat(tee.raw.fileno()).st_size
    try:
        learned = gguf_file.header_of(tee.raw, size)
    except GgufError as error:
        if error.kind == "cut":
            detail = str(error).removeprefix("cut short")
            raise ModelDamaged(
                f"{path}: cut short as it loaded{detail}; {AGAIN}", file=path.name
            ) from None
        raise ModelDamaged(f"{path}: a damaged GGUF file: {error}", file=path.name) from None
    # The header's bytes, from the 8th to where the data starts (padding included), read through
    # tee from where it is; the data's start is bounded by gguf_file's limits.
    tee.raw.seek(tee.position)
    bytes_of = bytearray(learned.data_start - len(first))
    tee.fill(memoryview(bytes_of), len(first))
    try:
        header = gguf_file.header_of(BytesIO(first + bytes_of), size)
    except GgufError:
        header = None
    # What the offsets and the sizes below come from; the metadata is not read here.
    if header is None or (header.data_start, header.tensors) != (
        learned.data_start,
        learned.tensors,
    ):
        raise ModelChanged(
            f"{path}: changed as it loaded, its header read twice differently; {AGAIN}",
            file=path.name,
        )
    tensors: dict[str, _Entry] = {}
    for each in sorted(header.tensors, key=lambda tensor: tensor.offset):
        if skip is not None and skip(each.name):
            continue
        if each.type in _DTYPES:
            found, shape = _DTYPES[each.type], each.shape
            if each.nbytes != found.itemsize * prod(shape):
                raise RuntimeError(f"{path.name}: {each.name} takes {each.nbytes} bytes")
        elif len(each.shape) == 2:
            # weights.check pins the shapes of the 7B's matrices, so a quantised tensor of no rows
            # never gets here from a checked file; a refusal in words all the same, not a
            # ZeroDivisionError, for a file read without the check (blockswap.read's own callers).
            if each.shape[0] == 0:
                raise ModelDamaged(
                    f"{path}: {each.name} is {each.type} with no rows; {AGAIN}", file=path.name
                )
            found, shape = torch.uint8, (each.shape[0], each.nbytes // each.shape[0])
        else:
            raise RuntimeError(f"{path.name}: {each.name} is {each.type}, which isn't read")
        tensors[each.name] = (found, shape, each.offset, each.offset + each.nbytes)
    return tensors, 0, size


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


class _MemoryStatus(ctypes.Structure):
    """kernel32's MEMORYSTATUSEX, which GlobalMemoryStatusEx fills once dwLength holds its size."""

    _fields_ = (
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    )


def _windows_status(kernel32: Any, what: str) -> _MemoryStatus | None:
    """kernel32's GlobalMemoryStatusEx, or None where the call fails, said at debug as `what`
    unread (the callers' own lines are the user's). `kernel32` is the library object, loaded here
    where None and replaced by the tests."""
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    # WinDLL, WinError and get_last_error are ctypes' on Windows alone, and its stubs say so: by
    # getattr, so that pyright, which reads the module for Linux, takes the calls.
    try:
        if kernel32 is None:
            kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)  # noqa: B009
        done = kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (OSError, AttributeError) as error:
        logger.debug("%s is unread: kernel32 failed: %s", what, error)
        return None
    if not done:
        reason = getattr(ctypes, "WinError")(getattr(ctypes, "get_last_error")())  # noqa: B009
        logger.debug("%s is unread: GlobalMemoryStatusEx: %s", what, reason)
        return None
    return status


def _windows_available(kernel32: Any = None) -> int | None:
    """The physical memory Windows has available, in bytes, or None where the call fails (said at
    debug: the callers' own warning that the memory is unread is the user's line). Source:
    kernel32's GlobalMemoryStatusEx; its ullAvailPhys is what Task Manager calls "Available"
    (standby, free and zeroed pages)."""
    status = _windows_status(kernel32, "the system memory available")
    return None if status is None else int(status.ullAvailPhys)


def commit(kernel32: Any = None) -> tuple[int, int] | None:
    """On Windows, the system's commit charge and its limit, in bytes (GlobalMemoryStatusEx: the
    limit ullTotalPageFile, what is left of it ullAvailPageFile); None elsewhere, or where the
    call fails. Windows charges the GPU memory a process holds to its commit, as it does its own
    memory: the process's private bytes grew as the pool's reserved bytes did, by 8.86 GiB for
    8.84 as the encode's phase came in, and as torch's own allocator's did, by 6.37 for 6.36 (an
    RTX 5090 natively, impl-win-5, 2026-11-10). A job's memory limit, which bounds the
    same charge, is not in these figures: in a job of 10 GiB they were the system's."""
    if sys.platform != "win32":
        return None
    status = _windows_status(kernel32, "the system's commit")
    if status is None:
        return None
    return int(status.ullTotalPageFile - status.ullAvailPageFile), int(status.ullTotalPageFile)


def available(proc: Path = Path("/proc"), cgroups: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The host memory this process can still take, in bytes. On Windows, the physical memory
    available (GlobalMemoryStatusEx, Task Manager's "Available"; `proc` and `cgroups` unused). On
    Linux, what the kernel estimates available (MemAvailable of /proc/meminfo), or less in a
    container held to a limit, where /proc/meminfo is the host's and knows nothing of it: what
    is left under the nearest limit of its control group and of the groups above it. A group's
    limit is cgroup v2's memory.max or memory.high (past which the kernel reclaims from it and
    slows it down), v1's memory.limit_in_bytes; what it holds, memory.current (v1:
    memory.usage_in_bytes) less its file cache (memory.stat's active_file and inactive_file; v1:
    total_ of each), which the kernel gives back when asked, as MemAvailable counts the host's:
    a model file just hashed is 16 GiB of it. None where nothing can be read. The planner counts
    the job's host memory with it before the load (shot_plan.py)."""
    if sys.platform == "win32":
        return _windows_available()
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
            f" ({listing(sizes)}), and the host has {_gib(free)} available",
            need=needed,
            available=free,
        )
    parts: dict[str, Part] = {}
    try:
        for name, layout in layouts.items():
            parts[name] = Part(name, layout, backend)
    except HostMemoryError as error:
        raise HostMemoryError(
            f"the weights that leave the GPU need {_gib(needed)} of host memory, pinned"
            f" ({listing(sizes)}): {error}, after {_gib(sum(p.size for p in parts.values()))}",
            need=needed,
            available=free,
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
    def parts(self) -> Mapping[str, Part]:
        """The parts this placement moves, by name, read-only: the same parts, host copies and
        all, that another placement takes after close (Models.place builds its next from
        them)."""
        return MappingProxyType(self._parts)

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
        self._backend.wait(*self.streams())

    def streams(self) -> list[Any]:
        """The streams this placement's work is queued on: the compute stream and, once a
        prefetch has begun, the side stream of its copies. All that a wait for it covers."""
        return [self._backend.compute(), *([] if self._side is None else [self._side])]

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
