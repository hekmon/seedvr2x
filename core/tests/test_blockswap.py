"""The DiT's placement (seedvr2x/blockswap.py) on the CPU: a toy model on simulated CUDA streams.

On a GPU, a block's forward only queues its kernels: they read its weights later, in the
stream's order, and a copy queued on another stream runs whenever the events it waits for
allow. The simulation (Sim) keeps that: a toy block's forward queues one kernel, which reads the
memory its parameters were views of when it was queued, and nothing runs until the host waits;
then the streams advance in an order the test chooses: the side stream as early as its events
let it, as late, or at random. A copy that doesn't wait for its slot's last reader, or a block
that doesn't wait for its copy, then gives other bits under one of them, and the simulation
itself refuses a buffer written under a kernel queued to read it that nothing holds back.

The swap on a real card, the DiT's bits and the figures: tests/test_placement.py (GPU).
"""

import gc
import hashlib
import io
import itertools
import json
import os
import random
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from test_gguf_file import Spec, entry, header, placed
from test_kquant import blocks as kquant_blocks
from torch import Tensor

from seedvr2x import blockswap, gguf_file, int8, kquant, model, weights
from seedvr2x.blockswap import OTHER, HostMemoryError, Part, Placement, dit_parts
from seedvr2x.errors import ModelChanged, ModelDamaged
from seedvr2x.passes import ModelPlan
from seedvr2x.weights import SLOTS

CPU = torch.device("cpu")
WIDTH = 8
POLICIES = ["side-first", "compute-first", "random-1", "random-2", "random-3"]


class Stream:
    """A simulated CUDA stream: a queue of kernels, copies, waits for events and records of
    events, run in order."""

    def __init__(self, sim: "Sim", name: str) -> None:
        self.sim, self.name = sim, name
        self.queue: deque[tuple[Any, ...]] = deque()

    def wait_event(self, event: "Event") -> None:
        self.queue.append(("wait", event))


class Event:
    """A simulated CUDA event: set once the stream it was recorded on reaches its record."""

    def __init__(self) -> None:
        self.set = False

    def record(self, stream: Stream) -> None:
        self.set = False
        stream.queue.append(("record", self))

    def query(self) -> bool:
        """Whether the stream has reached the record: asked, never waited for."""
        return self.set


class Sim(blockswap.Cuda):
    """The placement's CUDA calls, simulated on the CPU (the module's docstring). policy is the
    order the streams advance in when the host waits. ran lists what ran, in order: kernels
    ("kernel", name), copies ("copy", stream, buffer, source address) and the host's waits."""

    def __init__(self, policy: str = "side-first") -> None:
        super().__init__(CPU)
        self.policy = policy
        self.random = random.Random(policy)
        self.main = Stream(self, "compute")
        self.sides: list[Stream] = []
        self.buffers: list[weakref.ref[Tensor]] = []
        self.sizes: dict[int, int] = {}
        self.ran: list[tuple[Any, ...]] = []
        self.peak = 0  # the most bytes of device buffers alive at once
        self.trims = 0  # the times the allocator was told to give its free memory back
        self.refuse_host = False

    def host(self, size: int) -> blockswap.HostCopy:
        if self.refuse_host:
            raise HostMemoryError("the host refuses to pin: a stand-in")
        return blockswap.HostCopy(size, pin=False)

    def compute(self) -> Any:
        return self.main

    def side(self) -> Any:
        self.sides.append(Stream(self, f"side-{len(self.sides)}"))
        return self.sides[-1]

    def event(self) -> Any:
        return Event()

    def allocate(self, size: int) -> Tensor:
        # Not zeros: a block run on a buffer nothing was copied into gives other bits.
        buffer = torch.full((size,), 0xA5, dtype=torch.uint8)
        self.buffers.append(weakref.ref(buffer))
        self.sizes[buffer.data_ptr()] = size
        self.peak = max(self.peak, self.alive())
        return buffer

    def copy(self, target: Tensor, source: Tensor, stream: Any, blocking: bool) -> None:
        stream.queue.append(("copy", target, source))
        if blocking:
            self.drain(stream)

    def trim(self) -> None:
        self.trims += 1

    def wait(self, *streams: Any) -> None:
        self.ran.append(("wait", *[stream.name for stream in streams]))
        for stream in streams:
            self.drain(stream)

    def kernel(self, name: str, run: Callable[[], None], reads: list[Tensor]) -> None:
        """Queue a kernel on the compute stream: run, which reads the memory of reads."""
        self.main.queue.append(("kernel", name, run, reads))

    def settle(self) -> None:
        """Run everything queued, as the host does when it reads a result."""
        for stream in self.streams():
            self.drain(stream)

    def idle(self) -> bool:
        return not any(stream.queue for stream in self.streams())

    def streams(self) -> list[Stream]:
        return [self.main, *self.sides]

    def live(self) -> int:
        """The bytes of the device buffers still alive."""
        gc.collect()
        return self.alive()

    def alive(self) -> int:
        alive = [buffer for buffer in (ref() for ref in self.buffers) if buffer is not None]
        return sum(buffer.numel() for buffer in alive)

    def drain(self, target: Stream) -> None:
        """Advance the streams, in the policy's order, until target's queue is empty."""
        while target.queue:
            order = self.streams()
            if self.policy == "side-first":
                order = [*self.sides, self.main]
            elif self.policy.startswith("random"):
                self.random.shuffle(order)
            for stream in order:
                if self.step(stream):
                    break
            else:
                raise AssertionError(f"deadlock: {[(s.name, len(s.queue)) for s in order]}")

    def step(self, stream: Stream) -> bool:
        """Run the stream's next operation, if it can run now."""
        if not stream.queue:
            return False
        kind, *rest = stream.queue[0]
        if kind == "wait" and not rest[0].set:
            return False
        stream.queue.popleft()
        if kind == "record":
            rest[0].set = True
        elif kind == "copy":
            target, source = rest
            for name, reads in self.pending(stream):
                for read in reads:
                    assert not overlap(target, read), f"written under {name}, still to run"
            target.copy_(source)
            self.ran.append(("copy", stream.name, self.buffer_of(target), source.data_ptr()))
        elif kind == "kernel":
            name, run, _ = rest
            run()
            self.ran.append(("kernel", name))
        return True

    def pending(self, writer: Stream) -> list[tuple[str, list[Tensor]]]:
        """The kernels queued that could run before a copy at the head of writer does, each with
        the memory it reads: those of the other streams that no event still to come holds back.
        A kernel behind a wait for an event not yet set reads what is written meanwhile, as it
        is meant to: a block behind its copy; and so does one queued behind the copy on its own
        stream: a block behind its part's return."""
        free: list[tuple[str, list[Tensor]]] = []
        for stream in self.streams():
            if stream is writer:
                continue
            for kind, *rest in stream.queue:
                if kind == "wait" and not rest[0].set:
                    break
                if kind == "kernel":
                    free.append((rest[0], rest[2]))
        return free

    def buffer_of(self, tensor: Tensor) -> int | None:
        """The address of the device buffer tensor is a view of, None for other memory."""
        address = tensor.data_ptr()
        for start, size in self.sizes.items():
            if start <= address < start + size:
                return start
        return None


def overlap(a: Tensor, b: Tensor) -> bool:
    a_start, b_start = a.data_ptr(), b.data_ptr()
    return a_start < b_start + b.numel() * b.element_size() and b_start < a_start + a.nbytes


class Block(torch.nn.Module):
    """A toy block: a matrix, a bias and a scalar gain, parameters, and a table, a buffer, as a
    DiT block's rotary buffers are. Its forward queues one kernel and returns where its result will
    be."""

    def __init__(self, name: str, sim: Sim) -> None:
        super().__init__()
        self.name, self.sim = name, sim
        self.mix = torch.nn.Linear(WIDTH, WIDTH)
        self.gain = torch.nn.Parameter(torch.rand(()) + 0.5)
        self.table: Tensor
        self.register_buffer("table", torch.randn(WIDTH))
        self.fail = False

    def forward(self, x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        # The memory this forward's kernel reads, as a GPU's kernel holds its addresses.
        weight, bias, gain = self.mix.weight.data, self.mix.bias.data, self.gain.data
        table = self.table

        def run() -> None:
            out.copy_(torch.tanh(torch.nn.functional.linear(x, weight, bias)) * gain + table)

        self.sim.kernel(self.name, run, [weight, bias, gain])
        if self.fail:
            # After its kernel is queued, as a GPU forward fails: what reads the slot is on its
            # way, and the slot's next copy must still wait for it.
            raise RuntimeError("a forward that raises")
        return out


class Toy(torch.nn.Module):
    """A toy DiT: parameters before and after its blocks, as the DiT's emb_in, txt_in, vid_in
    and vid_out."""

    def __init__(self, blocks: int, sim: Sim) -> None:
        super().__init__()
        self.before = Block("before", sim)
        self.blocks = torch.nn.ModuleList(Block(f"blocks.{i}", sim) for i in range(blocks))
        self.after = Block("after", sim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.before(x)
        for block in self.blocks:
            x = block(x)
        return self.after(x)


def toy(blocks: int, sim: Sim) -> Toy:
    """The same toy every time: its weights by a seed."""
    torch.manual_seed(blocks)
    return Toy(blocks, sim).requires_grad_(False).eval()


def run(model: Toy, sim: Sim, forwards: int = 3) -> list[Tensor]:
    """The model's outputs for `forwards` inputs, each read once everything queued has run."""
    outputs: list[Tensor] = []
    for index in range(forwards):
        torch.manual_seed(1000 + index)
        out = model(torch.randn(4, WIDTH))
        sim.settle()
        outputs.append(out.clone())
    return outputs


def reference(blocks: int, forwards: int = 3) -> list[Tensor]:
    sim = Sim()
    return run(toy(blocks, sim), sim, forwards)


def same(a: list[Tensor], b: list[Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_a_part_is_one_run_of_bytes() -> None:
    sim = Sim()
    tensors = {
        "matrix": (torch.float16, (3, 5)),
        "gain": (torch.float32, ()),
        "count": (torch.int64, (2,)),
    }
    part = Part("blocks.0", tensors, sim)
    # Each parameter on a multiple of ALIGN, in the order given; the host copy a whole page.
    assert [field.offset for field in part.fields] == [0, 512, 1024]
    assert [field.size for field in part.fields] == [30, 4, 16]
    assert part.size == 1536 and part.host.bytes.numel() == 4096 and not part.host.pinned
    assert not part.host.bytes.any()
    buffer = sim.allocate(part.size)
    views = [part.view(buffer, field) for field in part.fields]
    assert [(view.dtype, tuple(view.shape)) for view in views] == list(tensors.values())
    assert [view.data_ptr() - buffer.data_ptr() for view in views] == [0, 512, 1024]
    views[0].fill_(1.5)
    assert buffer[:30].view(torch.float16).eq(1.5).all() and buffer[30] == 0xA5


def test_the_parts_of_a_dit_by_name() -> None:
    part_of = dit_parts(36, 33, other=False)
    assert part_of("blocks.32.attn.proj_qkv.vid.weight") is None
    assert part_of("blocks.33.attn.proj_qkv.vid.weight") == "blocks.33"
    assert part_of("blocks.35.mlp.vid.proj_in.weight") == "blocks.35"
    assert part_of("emb_in.proj_in.weight") is None and part_of("vid_out.proj.bias") is None
    part_of = dit_parts(36, 0, other=True)
    assert part_of("blocks.0.ada.txt.attn_gate") == "blocks.0"
    assert part_of("emb_in.proj_in.weight") == OTHER and part_of("vid_in.proj.weight") == OTHER
    # A name that only starts like a block's isn't one.
    assert part_of("blocks_extra.0.weight") == OTHER


@pytest.mark.parametrize("architecture", [weights.DIT_7B, weights.DIT_3B], ids=["7b", "3b"])
def test_a_dit_s_blocks_are_weights_py_s(architecture: weights.Architecture) -> None:
    # By the model's shapes, in fp16, as the files hold them: the bytes the planner plans with
    # and the placement moves, in seedvr2x/weights.py, which imports no torch.
    import seedvr2x.vendor
    from seedvr2x.vendor.common.config import create_object, load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    assert architecture.config is not None
    config: Any = load_config(str(vendor / architecture.config / "main.yaml"))
    with torch.device("meta"):
        dit = create_object(config.dit.model)
    sizes = dict.fromkeys(range(len(dit.blocks)), 0)
    other = 0
    for key, parameter in dit.named_parameters():
        match = blockswap._BLOCK.match(key)  # pyright: ignore[reportPrivateUsage]
        if match is None:
            other += 2 * parameter.numel()
        else:
            sizes[int(match[1])] += 2 * parameter.numel()
    assert tuple(sizes.values()) == architecture.blocks and other == architecture.other
    # Three buffers a block, RoPE's, which stay on the device.
    buffers = [key for key, _ in dit.named_buffers() if key.startswith("blocks.0.")]
    assert len(buffers) == 3 and all(".rope." in key for key in buffers)


def test_the_7b_s_figures() -> None:
    block = weights.DIT_7B.blocks[0]
    assert set(weights.DIT_7B.blocks) == {453_133_312} and len(weights.DIT_7B.blocks) == 36
    assert f"{block / blockswap.GIB:.4f}" == "0.4220"
    assert f"{sum(weights.DIT_7B.blocks) / blockswap.GIB:.2f}" == "15.19"
    assert f"{weights.DIT_7B.other / blockswap.GIB:.3f}" == "0.155"
    assert weights.SLOTS == 2 == blockswap.SLOTS


def test_hosting_moves_parameters_only() -> None:
    sim = Sim()
    model = toy(5, sim)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    tables = {key: buffer.data_ptr() for key, buffer in model.named_buffers()}
    kept = {key: p.data_ptr() for key, p in model.named_parameters() if key.startswith("before")}
    parts = blockswap.host(model, dit_parts(5, 3, other=False), sim)
    assert sorted(parts) == ["blocks.3", "blocks.4"]
    for part in parts.values():
        assert [field.key.removeprefix(part.name + ".") for field in part.fields] == [
            "gain",
            "mix.weight",
            "mix.bias",
        ]
        for field in part.fields:
            parameter = model.get_parameter(field.key)
            assert field.parameter is parameter
            assert parameter.data_ptr() == part.host.bytes.data_ptr() + field.offset
    after = model.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert {key: buffer.data_ptr() for key, buffer in model.named_buffers()} == tables
    assert all(model.get_parameter(key).data_ptr() == address for key, address in kept.items())
    assert not model.get_parameter("blocks.3.mix.weight").requires_grad


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_read_puts_the_parts_in_their_host_copies(
    tmp_path: Path, dtype: torch.dtype | None
) -> None:
    sim = Sim()
    source = toy(4, sim).to(torch.float16)
    path = tmp_path / "toy.safetensors"
    save_file(source.state_dict(), path)
    with torch.device("meta"):
        model = Toy(4, sim)
    state, parts = blockswap.read(model, path, CPU, dtype, dit_parts(4, 1, other=True), sim)
    assert sorted(parts) == ["blocks.1", "blocks.2", "blocks.3", OTHER]
    hosts = {part.host.bytes.data_ptr(): part for part in parts.values()}
    for key, tensor in state.items():
        part = next(
            (p for start, p in hosts.items() if start <= tensor.data_ptr() < start + p.size), None
        )
        # The buffers and the block that stays aren't in a host copy; every other tensor is.
        assert (part is None) == (key.endswith(".table") or key.startswith("blocks.0.")), key
    model.load_state_dict(state, strict=True, assign=True)
    for part in parts.values():
        part.adopt(model)
    expected = source.state_dict()
    for key, tensor in model.state_dict().items():
        assert tensor.dtype == (torch.float16 if dtype is None else dtype)
        assert torch.equal(tensor, expected[key].to(tensor.dtype)), key
    assert len(parts[OTHER].fields) == 6  # the parameters outside the blocks: no buffer


def mappings(path: Path) -> list[str]:
    """This process's mappings of the file at path, /proc/self/maps' lines for it."""
    name = str(path.resolve())
    return [line for line in Path("/proc/self/maps").read_text().splitlines() if name in line]


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
@pytest.mark.parametrize("with_parts", [False, True])
def test_read_never_maps_the_file(
    tmp_path: Path, with_parts: bool, dtype: torch.dtype | None
) -> None:
    # A mapped page stays in the process from the copy that reads it until the mapping goes:
    # the 7B's 15.35 GiB at a load's peak (research/scripts/load_peak.py). The file is read.
    sim = Sim()
    path = tmp_path / "toy.safetensors"
    save_file(toy(4, sim).to(torch.float16).state_dict(), path)
    # A mapping is seen where there is one: safetensors' own, while a tensor of it lives.
    with cast(Any, safe_open(path, framework="pt", device="cpu")) as mapped:
        held = mapped.get_tensor("after.gain")
        assert mappings(path)
    del held, mapped
    gc.collect()
    assert not mappings(path)
    with torch.device("meta"):
        model = Toy(4, sim)
    names = dit_parts(4, 1, other=True)
    seen: list[bool] = []

    def part_of(key: str) -> str | None:
        seen.append(bool(mappings(path)))
        return names(key) if with_parts else None

    state, parts = blockswap.read(model, path, CPU, dtype, part_of, sim)
    assert seen and not any(seen)
    assert not mappings(path)
    assert bool(parts) == with_parts and len(state) == len(model.state_dict())


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_read_gives_what_safetensors_reads(tmp_path: Path, dtype: torch.dtype | None) -> None:
    # Every dtype read, a scalar and a tensor of no value among them, in a file safetensors
    # wrote, against safetensors' own reading of it; no part: a model loaded whole.
    torch.manual_seed(7)
    saved = {
        "matrix": torch.randn(5, 3).to(torch.float16),
        "scalar": torch.randn(()).to(torch.float16),
        "none": torch.empty(0, 4, dtype=torch.float16),
        "brain": torch.randn(2, 3).to(torch.bfloat16),
        "single": torch.randn(3, 7),
        "double": torch.randn(2, dtype=torch.float64),
        "counter": torch.tensor(123_456_789_012, dtype=torch.int64),
        "whole": torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32),
        "short": torch.tensor([-(2**15), 2**15 - 1], dtype=torch.int16),
        "tiny": torch.tensor([-128, 127], dtype=torch.int8),
        "bytes": torch.arange(13, dtype=torch.uint8),
        "flags": torch.tensor([True, False, True]),
    }
    path = tmp_path / "assorted.safetensors"
    save_file(saved, path)
    state, parts = blockswap.read(torch.nn.Module(), path, CPU, dtype, None)
    expected = load_file(path)
    assert not parts and state.keys() == expected.keys() == saved.keys()
    for key, tensor in state.items():
        want = expected[key]
        if dtype is not None and want.is_floating_point():
            want = want.to(dtype)
        assert tensor.dtype == want.dtype and tensor.shape == want.shape, key
        assert torch.equal(tensor, want), key
    # Each in memory of its own: they are read through one buffer, where the next tensor's
    # bytes replace the last's.
    held = [tensor for tensor in state.values() if tensor.numel()]
    assert not any(overlap(a, b) for a, b in itertools.combinations(held, 2))


def written(path: Path, entries: dict[str, Any], data: bytes) -> Path:
    """A safetensors file of this header and these bytes, as the format lays them."""
    header = json.dumps(entries).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + data)
    return path


def test_read_refuses_what_it_cannot_read(tmp_path: Path) -> None:
    # weights.check has refused such files before a job's load reads one: a file that passes
    # it unchecked, or changes under the load, is still never read into wrong tensors.
    module = torch.nn.Module()
    half = {"dtype": "F16", "shape": [3], "data_offsets": [0, 6]}
    # A dtype torch has no reading for here.
    odd = written(tmp_path / "odd.safetensors", {"x": {**half, "dtype": "F8_E4M3"}}, bytes(6))
    with pytest.raises(RuntimeError, match=r"^odd\.safetensors: x is F8_E4M3, which isn't read$"):
        blockswap.read(module, odd, CPU, None, None)
    # Data of another size than its shape's.
    long = written(tmp_path / "long.safetensors", {"x": {**half, "data_offsets": [0, 8]}}, bytes(8))
    with pytest.raises(RuntimeError, match=r"^long\.safetensors: x takes 8 bytes, not 6$"):
        blockswap.read(module, long, CPU, None, None)
    # A file cut short since its check, as it loads, in a tensor bound for the device and in
    # a part's: refused as weights.check refuses one, in a line (cli logs a ModelError so:
    # tests/test_cli_run.py, test_model_replaced_before_its_load), the bytes it holds and
    # those its header gives.
    after = {"dtype": "F16", "shape": [2], "data_offsets": [6, 10]}
    short = written(tmp_path / "short.safetensors", {"x": half, "y": after}, bytes(8))
    size = short.stat().st_size
    said = f"{short}: cut short as it loaded, {size:,} of {size + 2:,} bytes; fetch it again"
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(module, short, CPU, None, None)
    assert str(error.value) == said
    module.register_parameter("y", torch.nn.Parameter(torch.zeros(2, dtype=torch.float16)))
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(module, short, CPU, None, lambda key: "part", Sim())
    assert str(error.value) == said
    # Cut in a tensor before the last: still the whole file's size, by its header.
    shorter = written(tmp_path / "shorter.safetensors", {"x": half, "y": after}, bytes(4))
    size = shorter.stat().st_size
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(module, shorter, CPU, None, lambda key: "part", Sim())
    assert str(error.value) == (
        f"{shorter}: cut short as it loaded, {size:,} of {size + 6:,} bytes; fetch it again"
    )
    # Cut inside its header: the least it can hold, its header's length or the header's end.
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(bytes(5))
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(module, tiny, CPU, None, None)
    assert (
        str(error.value) == f"{tiny}: cut short as it loaded, 5 of at least 8 bytes; fetch it again"
    )
    cut = tmp_path / "cut.safetensors"
    cut.write_bytes(short.read_bytes()[:20])
    end = 8 + int.from_bytes(short.read_bytes()[:8], "little")
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(module, cut, CPU, None, None)
    assert (
        str(error.value)
        == f"{cut}: cut short as it loaded, 20 of at least {end:,} bytes; fetch it again"
    )
    # Whole, it reads: the same file with its last two bytes.
    whole = written(tmp_path / "whole.safetensors", {"x": half, "y": after}, bytes(range(10)))
    state, parts = blockswap.read(module, whole, CPU, None, lambda key: "part", Sim())
    assert bytes(state["x"].view(torch.uint8)) == bytes(range(6))
    assert bytes(state["y"].view(torch.uint8)) == bytes(range(6, 10)) and list(parts) == ["part"]
    assert state["y"].data_ptr() == parts["part"].host.bytes.data_ptr()


def test_read_takes_the_file_from_its_start_to_its_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each byte of the file once, in its order, the data's whatever the header's: one pass
    # over the file, 15 GiB of which take over a minute where a file reads slowly.
    late = {"dtype": "F16", "shape": [3], "data_offsets": [4, 10]}
    early = {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}
    last = {"dtype": "U8", "shape": [5], "data_offsets": [10, 15]}
    path = written(tmp_path / "order.safetensors", {"b": late, "c": last, "a": early}, bytes(15))
    spans: list[tuple[int, int]] = []
    real = open

    class Recorded:
        def __init__(self, file: Any) -> None:
            self.file = file

        def __enter__(self) -> "Recorded":
            return self

        def __exit__(self, *error: object) -> None:
            self.file.close()

        def seek(self, position: int) -> int:
            return self.file.seek(position)

        def fileno(self) -> int:
            return self.file.fileno()

        def readinto(self, target: memoryview) -> int:
            start = self.file.tell()
            got = self.file.readinto(target)
            spans.append((start, start + got))
            return got

        def read(self, size: int) -> bytes:
            start = self.file.tell()
            data = self.file.read(size)
            spans.append((start, start + len(data)))
            return data

    def recorded(name: Any, *args: Any, **more: Any) -> Any:
        opened = real(name, *args, **more)
        return Recorded(opened) if Path(name) == path else opened

    monkeypatch.setattr("builtins.open", recorded)
    module = torch.nn.Module()
    module.register_parameter("b", torch.nn.Parameter(torch.zeros(3, dtype=torch.float16)))
    state, _ = blockswap.read(module, path, CPU, None, lambda key: "part", Sim())
    monkeypatch.undo()
    assert list(state) == ["a", "b", "c"]
    # End to end, nothing twice, nothing left: the header, the device's tensors and the
    # part's alike.
    assert spans[0][0] == 0 and spans[-1][1] == path.stat().st_size
    assert all(end == start for (_, end), (start, _) in itertools.pairwise(spans))


def hashing(
    module: torch.nn.Module,
    path: Path,
    dtype: torch.dtype | None,
    part_of: Callable[[str], str | None] | None = None,
    backend: blockswap.Cuda | None = None,
    skip: Callable[[str], bool] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Part], str, os.stat_result]:
    """blockswap.read over the file at path, on the CPU, with hashed: its tensors, parts, SHA-256
    and status."""
    given: list[tuple[str, os.stat_result]] = []
    state, parts = blockswap.read(
        *(module, path, CPU, dtype, part_of, backend),
        skip=skip,
        hashed=lambda d, s: given.append((d, s)),
    )
    assert len(given) == 1
    return state, parts, *given[0]


def test_read_hashes_every_byte_of_the_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DESIGN.md, Weights (decided on 2026-11-03): a load reads its file once, hashing every byte
    # as it reads it, in the file's order: the header, the tensors kept, those skipped (an int8
    # file's markers, first, between and last) and any byte after the data, so that the SHA-256
    # is hashlib's of the whole file, the bytes put on the device among them. The tensors are
    # today's, read without the hash; the file read once, front to back, and never mapped.
    entries = {
        "m0": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
        "a": {"dtype": "F16", "shape": [2], "data_offsets": [3, 7]},
        "m1": {"dtype": "U8", "shape": [2], "data_offsets": [7, 9]},
        "b": {"dtype": "F16", "shape": [3], "data_offsets": [9, 15]},
        "m2": {"dtype": "U8", "shape": [4], "data_offsets": [15, 19]},
    }
    data = bytes(range(1, 20))
    for tail in (b"", b"after the data"):
        path = written(tmp_path / f"marked{len(tail)}.safetensors", entries, data)
        with path.open("ab") as file:
            file.write(tail)
        module = torch.nn.Module()
        module.register_parameter("b", torch.nn.Parameter(torch.zeros(3, dtype=torch.float16)))
        spans: list[tuple[int, int]] = []
        real = blockswap._Tee.hashed  # pyright: ignore[reportPrivateUsage]

        def recorded(
            tee: Any, piece: memoryview, real: Any = real, spans: list[tuple[int, int]] = spans
        ) -> None:
            spans.append((tee.raw.tell() - len(piece), tee.raw.tell()))
            real(tee, piece)

        monkeypatch.setattr(blockswap._Tee, "hashed", recorded)  # pyright: ignore[reportPrivateUsage]
        state, parts, digest, opened = hashing(
            module, path, None, lambda key: "part", Sim(), lambda key: key[0] == "m"
        )
        monkeypatch.undo()
        whole = path.read_bytes()
        assert digest == hashlib.sha256(whole).hexdigest()
        assert (opened.st_ino, opened.st_size) == (path.stat().st_ino, len(whole))
        # Each byte hashed once, in the file's order, end to end.
        assert spans[0][0] == 0 and spans[-1][1] == len(whole)
        assert all(end == start for (_, end), (start, _) in itertools.pairwise(spans))
        # The tensors those read without the hash; the markers skipped, not kept.
        plain, _ = blockswap.read(
            module, path, CPU, None, lambda key: "part", Sim(), skip=lambda key: key[0] == "m"
        )
        assert list(state) == list(plain) == ["a", "b"] and list(parts) == ["part"]
        assert all(torch.equal(state[key], plain[key]) for key in state)
        assert bytes(state["b"].view(torch.uint8)) == data[9:15]
        assert not mappings(path)


def test_the_hash_waits_for_nothing_but_the_buffer_it_reads_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The hash runs on a thread of its own, a piece behind the read (blockswap._Tee): the
    # staging buffer, read into again for each tensor bound for the device, waits for its last
    # piece's hash, so that the bytes hashed are the file's even where the hash is slow. Pieces
    # of 5 bytes, a hash that sleeps on each and keeps a copy of what it was given: the copies
    # are the file, its SHA-256 hashlib's, the tensors the file's.
    saved = {f"t{i}": torch.arange(i, i + 7 + i, dtype=torch.int16) for i in range(6)}
    path = tmp_path / "many.safetensors"
    save_file(saved, path)
    monkeypatch.setattr(blockswap, "PIECE", 5)
    seen: list[bytes] = []
    threads: set[str] = set()
    real = hashlib.sha256

    class Slow:
        def __init__(self) -> None:
            self.real = real()

        def update(self, data: memoryview) -> None:
            time.sleep(0.002)
            threads.add(threading.current_thread().name)
            seen.append(bytes(data))
            self.real.update(data)

        def hexdigest(self) -> str:
            return self.real.hexdigest()

    monkeypatch.setattr(blockswap.hashlib, "sha256", Slow)
    state, parts, digest, _ = hashing(torch.nn.Module(), path, None)
    monkeypatch.undo()
    whole = path.read_bytes()
    assert b"".join(seen) == whole and digest == hashlib.sha256(whole).hexdigest()
    assert not parts and all(torch.equal(state[key], saved[key]) for key in saved)
    assert len(threads) == 1 and next(iter(threads)).startswith("seedvr2x-sha256")
    # The thread ends with the read.
    assert not [t for t in threading.enumerate() if t.name.startswith("seedvr2x-sha256")]


def test_a_file_cut_short_is_refused_as_it_hashes_too(tmp_path: Path) -> None:
    # The same refusal with the hash as without (test_read_refuses_what_it_cannot_read), in a
    # tensor read and in one skipped, its thread ended; nothing hashed is given.
    half = {"dtype": "F16", "shape": [3], "data_offsets": [0, 6]}
    after = {"dtype": "F16", "shape": [2], "data_offsets": [6, 10]}
    short = written(tmp_path / "short.safetensors", {"x": half, "y": after}, bytes(8))
    size = short.stat().st_size
    given: list[str] = []
    for skip in (None, lambda key: key == "y"):
        with pytest.raises(weights.ModelError) as error:
            blockswap.read(
                *(torch.nn.Module(), short, CPU, None, None),
                skip=skip,
                hashed=lambda d, s: given.append(d),
            )
        assert str(error.value) == (
            f"{short}: cut short as it loaded, {size:,} of {size + 2:,} bytes; fetch it again"
        )
    assert not given
    assert not [t for t in threading.enumerate() if t.name.startswith("seedvr2x-sha256")]


# seedvr2x's VAE and TransNetV2's weights, read on the CPU where SEEDVR2X_MODEL_DIR has them.
REAL = {
    "seedvr2x_ema_vae_fp16.safetensors": "b9c6ebf0",
    "transnetv2.safetensors": "bb8c8388",
}


@pytest.mark.parametrize("name", list(REAL))
def test_a_real_file_s_hash_is_its_pin_and_its_tensors_safetensors(name: str) -> None:
    # The model files themselves, where the test's model directory has them: the read's SHA-256
    # the file's pin (pull.PINNED), its tensors those of safetensors' own load, bit for bit, the
    # VAE cast as a run loads it. The 7B's 16.5 GB are held by the GPU regression, bit for bit.
    directory = os.environ.get("SEEDVR2X_MODEL_DIR")
    if not directory or not (Path(directory) / name).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {name}")
    from seedvr2x import pull

    path = Path(directory) / name
    for dtype in (None, torch.bfloat16) if "vae" in name else (None,):
        state, parts, digest, _ = hashing(torch.nn.Module(), path, dtype)
        assert digest == pull.PINNED[name].sha256 and digest.startswith(REAL[name])
        expected = load_file(path)
        assert not parts and state.keys() == expected.keys()
        for key, tensor in state.items():
            want = expected[key]
            want = want.to(dtype) if dtype is not None and want.is_floating_point() else want
            assert tensor.dtype == want.dtype and torch.equal(tensor, want), key


def test_a_model_loaded_whole_is_read_too(tmp_path: Path) -> None:
    # model._load_weights without a part, the load of a run without a plan: the file's
    # values, through the same read, the file never mapped; the model's parameters are the
    # tensors read, which a mapping would be the memory of.
    sim = Sim()
    source = toy(3, sim).to(torch.float16)
    path = tmp_path / "toy.safetensors"
    save_file(source.state_dict(), path)
    with torch.device("meta"):
        loaded = Toy(3, sim)
    assert model._load_weights(loaded, path, CPU, None, None) == {}  # pyright: ignore[reportPrivateUsage]
    assert not mappings(path)
    expected = source.state_dict()
    for key, tensor in loaded.state_dict().items():
        assert tensor.dtype == torch.float16 and torch.equal(tensor, expected[key]), key


def test_too_little_host_memory_is_refused_before_anything_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = Sim()
    model = toy(36, sim)
    hosted: list[int] = []
    monkeypatch.setattr(sim, "host", lambda size: hosted.append(size))
    monkeypatch.setattr(blockswap, "available", lambda: 1000)
    with pytest.raises(HostMemoryError) as refusal:
        blockswap.host(model, dit_parts(36, 0, other=True), sim)
    assert str(refusal.value) == (
        "the weights that leave the GPU need 0.00 GiB of host memory, pinned (36 of the DiT's"
        " blocks, the DiT's other parameters), and the host has 0.00 GiB available"
    )
    # The facts, in bytes: what was asked to be pinned and what the host has.
    assert refusal.value.kind == "host-memory" and refusal.value.available == 1000
    assert (refusal.value.need or 0) > 1000
    assert not hosted
    # What the kernel can't say isn't held against the plan: the pinning itself refuses then.
    monkeypatch.undo()
    monkeypatch.setattr(blockswap, "available", lambda: None)
    sim.refuse_host = True
    with pytest.raises(HostMemoryError, match=r"need 0\.00 GiB of host memory, pinned \(2 of"):
        blockswap.host(model, dit_parts(36, 34, other=False), sim)


def test_the_host_memory_available_is_read() -> None:
    assert (blockswap.available() or 0) > 0


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_swapping_changes_no_bit(policy: str, prefetch: bool) -> None:
    blocks = 7
    expected = reference(blocks)
    for swapped in range(blocks + 1):
        sim = Sim(policy)
        model = toy(blocks, sim)
        placement = Placement(model, CPU, swapped, prefetch, backend=sim)
        placement.enter("dit")
        assert same(run(model, sim), expected), swapped
        # Two slots at most, whatever the number of blocks swapped; the blocks that stay and the
        # other parameters are where the load put them, 292 bytes a block.
        slots = min(SLOTS, swapped) * placement.block_bytes
        assert sim.live() == slots
        assert placement.device_bytes() == slots + (blocks - swapped + 2) * 292


class Int8Block(torch.nn.Module):
    """A toy block of seedvr2x's int8 files: an Int8Linear (seedvr2x/int8.py), its int8 weight,
    float32 scale and float16 bias, and a table, a buffer. Its forward queues one kernel, the W8A8
    linear on the memory its parameters were views of when it was queued, as a GPU's kernel
    holds its addresses."""

    def __init__(self, name: str, sim: Sim) -> None:
        super().__init__()
        self.name, self.sim = name, sim
        generator = torch.Generator().manual_seed(len(name) * 31 + sum(map(ord, name)))
        self.mix = int8.Int8Linear(INT8_WIDTH, INT8_WIDTH)
        weight = torch.randint(-128, 128, (INT8_WIDTH, INT8_WIDTH), generator=generator)
        self.mix.weight.data = weight.to(torch.int8)
        self.mix.weight_scale.data = torch.rand(INT8_WIDTH, 1, generator=generator) * 0.02
        assert self.mix.bias is not None
        self.mix.bias.data = torch.randn(INT8_WIDTH, generator=generator).to(torch.float16)
        self.table: Tensor
        self.register_buffer("table", torch.randn(INT8_WIDTH, generator=generator))

    def forward(self, x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        weight, scale = self.mix.weight.data, self.mix.weight_scale.data
        assert self.mix.bias is not None
        bias, table = self.mix.bias.data, self.table

        def run() -> None:
            y = int8.linear(x, weight, scale, bias, torch.bfloat16).float()
            out.copy_(torch.tanh(y) + table)

        self.sim.kernel(self.name, run, [weight, scale, bias])
        return out


# The int8 toy's width: a multiple of the rotation's 256 columns.
INT8_WIDTH = 256


class Int8Toy(torch.nn.Module):
    """A toy DiT of int8 blocks, an int8 layer before and after them."""

    def __init__(self, blocks: int, sim: Sim) -> None:
        super().__init__()
        self.before = Int8Block("before", sim)
        self.blocks = torch.nn.ModuleList(Int8Block(f"blocks.{i}", sim) for i in range(blocks))
        self.after = Int8Block("after", sim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.before(x)
        for block in self.blocks:
            x = block(x)
        return self.after(x)


def int8_run(model: Int8Toy, sim: Sim, forwards: int = 2) -> list[Tensor]:
    outputs: list[Tensor] = []
    for index in range(forwards):
        torch.manual_seed(2000 + index)
        out = model(torch.randn(5, INT8_WIDTH))
        sim.settle()
        outputs.append(out.clone())
    return outputs


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", ["side-first", "compute-first", "random-1"])
def test_int8_blocks_swap_as_fp16_ones(policy: str, prefetch: bool) -> None:
    # The int8 file's blocks, their int8 weights, float32 scales and float16 biases moved as any
    # parameters: every count of swapped blocks gives the bits of none swapped. A part's bytes are
    # its parameters' own, each on ALIGN: 65,536 for the weight, 1,024 for the scale and 512 for
    # the bias, a block's 67,072 in a slot.
    blocks = 5
    plain = Sim(policy)
    expected = int8_run(Int8Toy(blocks, plain).requires_grad_(False).eval(), plain)
    for swapped in range(blocks + 1):
        sim = Sim(policy)
        model = Int8Toy(blocks, sim).requires_grad_(False).eval()
        placement = Placement(model, CPU, swapped, prefetch, backend=sim)
        placement.enter("dit")
        assert same(int8_run(model, sim), expected), swapped
        assert placement.block_bytes == (67_072 if swapped else 0)
        assert sim.live() == min(SLOTS, swapped) * placement.block_bytes
        fields = [field.dtype for field in placement._streaming[0].fields] if swapped else []  # pyright: ignore[reportPrivateUsage]
        assert fields == ([torch.int8, torch.float32, torch.float16] if swapped else [])


def test_an_int8_file_is_read_into_its_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # model._load_weights over an int8 file: the int8 weights and float32 scales read as the file
    # holds them, into the parts' host copies for the blocks a plan swaps, onto the device for
    # the rest; the markers never read (skip), no key missing or unexpected. The host copies
    # pinned by the simulator, as the load's other tests pin them: no CUDA.
    sim = Sim()
    monkeypatch.setattr(blockswap, "Cuda", lambda device: sim)
    source = Int8Toy(3, sim)
    tensors = {key: value.contiguous() for key, value in source.state_dict().items()}
    marker = torch.frombuffer(bytearray(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
    for layer in ("before.mix", "blocks.1.mix", "after.mix"):
        tensors[f"{layer}.comfy_quant"] = marker.clone()
    path = tmp_path / "int8.safetensors"
    save_file(tensors, path)
    with torch.device("meta"):
        target = Int8Toy(3, sim)
    parts = model._load_weights(  # pyright: ignore[reportPrivateUsage]
        target,
        path,
        CPU,
        None,
        dit_parts(3, 1, other=False),
        model._marker,  # pyright: ignore[reportPrivateUsage]
    )
    assert sorted(parts) == ["blocks.1", "blocks.2"]
    for key, value in source.state_dict().items():
        loaded = target.state_dict()[key]
        assert loaded.dtype == value.dtype and torch.equal(loaded, value), key
    assert target.get_parameter("blocks.1.mix.weight").dtype == torch.int8
    host = parts["blocks.1"].host.bytes
    for parameter in target.get_submodule("blocks.1").parameters():
        assert overlap(parameter.data, host)


# seedvr2x's int8 files, the sharp 7B's and the 7B's, the first in SEEDVR2X_MODEL_DIR taken.
INT8_DITS = (
    "seedvr2x_ema_7b_sharp_int8_convrot.safetensors",
    "seedvr2x_ema_7b_int8_convrot.safetensors",
)


@pytest.mark.gpu
def test_a_real_int8_dit_s_blocks_swap_and_change_no_bit() -> None:
    # The fp16 gate of tests/test_placement.py on seedvr2x's int8 file, its helpers taken: the
    # DiT loaded whole, then every parameter to a pinned host copy and 0, 3, 12 and 36 blocks
    # swapped, without the prefetch and with it, the windows' bits those of the DiT loaded whole;
    # its peak each swapped block's bytes less, but the two in the slots (peaks.block_bytes:
    # 226,862,080 in int8), from the whole load's less its allocator's rounding (below); the
    # device holding the placement's bytes as each enters. Needs CUDA with the int8 DiT and a
    # window beside it (10 GiB free), SEEDVR2X_MODEL_DIR holding an int8 file and the VAE, and
    # 8 GiB of host memory to pin.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from test_placement import MODELS, SLACK, VAES, allocated, found, windows

    if not MODELS:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    from seedvr2x import peaks

    device = torch.device("cuda:0")
    models = model.load_models(found(INT8_DITS), found(VAES), device)
    try:
        dit = models.runner.dit
        assert sum(isinstance(module, int8.Int8Linear) for module in dit.modules()) == 288
        digests, peak = windows(models)
        whole = allocated(device)
        parts = blockswap.host(
            dit, dit_parts(len(dit.blocks), 0, other=True), blockswap.Cuda(device)
        )
        torch.cuda.empty_cache()
        away = allocated(device)
        pinned = sum(part.host.bytes.numel() for part in parts.values())
        # Loaded whole, the DiT holds 72 MiB more than its parameters' bytes, which a placement's
        # packed parts don't: torch's allocator gives a request of 10 MiB or more a block rounded
        # up to 2 MiB, left whole when 1 MiB or less would remain (c10/cuda/
        # CUDACachingAllocator.cpp at v2.14.1: kMinLargeAlloc, kRoundLarge, should_split), and
        # each of the int8 file's 72 QKV matrices, 9,216 x 3,072 bytes, 27 MiB, takes 28: 72
        # blocks of 28,311,552 bytes, each 1 MiB over (memory_snapshot, RTX 5090, 2026-11-03).
        # fp16's matrices are whole multiples of 2 MiB. The placements' peaks are held to the
        # whole load's without it.
        rounding = whole - away - pinned
        assert abs(rounding - 72 * 2**20) < 2 * 2**20, rounding / 2**20
        block = peaks.block_bytes(weights.DIT_7B_INT8)
        for swapped, prefetch in [
            (0, False),
            (3, False),
            (3, True),
            (12, True),
            (36, False),
            (36, True),
        ]:
            said = f"{swapped} swapped, prefetch {prefetch}"
            placement = Placement(dit, device, swapped, prefetch, parts=parts)
            placement.enter("dit")
            assert abs(allocated(device) - away - placement.device_bytes()) < SLACK, said
            got, got_peak = windows(models)
            assert got == digests, said
            expected = peak - rounding - max(swapped - SLOTS, 0) * block
            assert abs(got_peak - expected) < SLACK, (said, got_peak / 2**30, expected / 2**30)
            placement.enter("encode", 0)
            assert abs(allocated(device) - away) < SLACK, said
            placement.close()
    finally:
        models.runner.dit = models.runner.vae = None
        gc.collect()
        torch.cuda.empty_cache()


# ---- The dynamic GGUF's blocks ---------------------------------------------------------------

# The toy's matrices have 256 inputs and outputs: a row is one ggml block, so a block's weight is
# (256, the type's block bytes) uint8, 28,160, 36,864 or 45,056 bytes for Q3_K, Q4_K and Q5_K (each
# on ALIGN), and its bias 512: three sizes of block, as the dynamic files' differ.
GGUF_WIDTH = 256
GGUF_KINDS = ("Q3_K", "Q4_K", "Q5_K")


def gguf_part_bytes(qtype: str) -> int:
    """A toy block's part in bytes: its matrix's quantised bytes and its float16 bias, on ALIGN."""
    return GGUF_WIDTH * kquant.BLOCK_BYTES[qtype] + GGUF_WIDTH * 2


class GgufBlock(torch.nn.Module):
    """A toy block of seedvr2x's dynamic GGUF files: a kquant.GGUFLinear, its quantised bytes
    (uint8) and float16 bias, and a table, a buffer. Its forward queues one kernel, the linear on
    the memory its parameters were views of when it was queued, as a GPU's kernel holds its
    addresses. `fill`: random blocks with sane scales (test_kquant.blocks) in the weight, False
    for a model built to be loaded."""

    def __init__(self, name: str, sim: Sim, qtype: str, fill: bool = True) -> None:
        super().__init__()
        self.name, self.sim = name, sim
        self.mix = kquant.GGUFLinear(GGUF_WIDTH, GGUF_WIDTH, qtype)
        self.table: Tensor
        self.register_buffer("table", torch.zeros(GGUF_WIDTH), persistent=False)
        if fill:
            seed = sum(map(ord, name))
            generator = torch.Generator().manual_seed(seed)
            self.mix.weight.data = kquant_blocks(qtype, GGUF_WIDTH, seed).reshape(GGUF_WIDTH, -1)
            assert self.mix.bias is not None
            self.mix.bias.data = (torch.randn(GGUF_WIDTH, generator=generator) * 0.5).to(
                torch.float16
            )
            self.table = torch.randn(GGUF_WIDTH, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.empty_like(x)
        weight, qtype, table = self.mix.weight.data, self.mix.qtype, self.table
        assert self.mix.bias is not None
        bias = self.mix.bias.data

        def run() -> None:
            y = kquant.linear(x, weight, bias, qtype, torch.bfloat16).float()
            out.copy_(torch.tanh(y) + table)

        self.sim.kernel(self.name, run, [weight, bias])
        return out


class GgufToy(torch.nn.Module):
    """A toy DiT of GGUF blocks of different types, with float16 parameters before them and after
    (`scale`, 100 values, 200 bytes, so that the file pads before its first matrix; `tail`, 3)."""

    def __init__(self, blocks: int, sim: Sim, fill: bool = True) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(100, dtype=torch.float16))
        self.blocks = torch.nn.ModuleList(
            GgufBlock(f"blocks.{i}", sim, GGUF_KINDS[2 * i % 3], fill) for i in range(blocks)
        )
        self.tail = torch.nn.Parameter(torch.zeros(3, dtype=torch.float16))
        if fill:
            generator = torch.Generator().manual_seed(blocks)
            self.scale.data = torch.randn(100, generator=generator).to(torch.float16)
            self.tail.data = torch.randn(3, generator=generator).to(torch.float16)

    def forward(self, x: Tensor) -> Tensor:
        # Before the blocks: their kernels are queued, so that what they return is only written
        # once the simulator runs them.
        x = x * (1 + self.scale.float().sum() * 1e-3) + self.tail.float().sum()
        for block in self.blocks:
            x = block(x)
        return x


def gguf_toy(blocks: int, sim: Sim) -> GgufToy:
    return GgufToy(blocks, sim).requires_grad_(False).eval()


def gguf_run(model: GgufToy, sim: Sim, forwards: int = 2) -> list[Tensor]:
    outputs: list[Tensor] = []
    for index in range(forwards):
        torch.manual_seed(3000 + index)
        out = model(torch.randn(5, GGUF_WIDTH))
        sim.settle()
        outputs.append(out.clone())
    return outputs


def write_gguf(path: Path, source: GgufToy) -> Path:
    """The toy's GGUF file, as seedvr2x's are laid out: a header (version 3, a metadata entry, each
    tensor of the model under its name, a matrix in its ggml type, its shape PyTorch's), padding
    to 32, each tensor's data at its aligned offset, 0xEE in the padding between and after, so
    that the hash must read it."""
    specs: list[Spec] = []
    data: dict[str, bytes] = {}
    for key, value in source.state_dict().items():
        layer = key.rpartition(".")[0]
        if value.dtype == torch.uint8:
            qtype = cast("kquant.GGUFLinear", source.get_submodule(layer)).qtype
            specs.append((key, qtype, (GGUF_WIDTH, GGUF_WIDTH)))
        else:
            specs.append((key, "F16", tuple(value.shape)))
        data[key] = bytes(value.contiguous().view(torch.uint8).numpy().tobytes())
    head = header(specs, [entry("general.architecture", 8, "toy")])
    offsets, length = placed(specs)
    raw = bytearray(b"\xee" * (len(head) + length))
    raw[: len(head)] = head
    for (key, _, _), offset in zip(specs, offsets, strict=True):
        raw[len(head) + offset : len(head) + offset + len(data[key])] = data[key]
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", ["side-first", "compute-first", "random-1"])
def test_gguf_blocks_of_unequal_sizes_swap_as_any_others(policy: str, prefetch: bool) -> None:
    # The quantised bytes of the blocks moved as any parameters: every count of swapped blocks
    # gives the bits of none swapped. The blocks differ in size (a type a block): a slot is the
    # largest swapped block's, the blocks that stay each their own size.
    blocks = 5
    plain = Sim(policy)
    expected = gguf_run(gguf_toy(blocks, plain), plain)
    sizes = [gguf_part_bytes(GGUF_KINDS[2 * i % 3]) for i in range(blocks)]
    assert len(set(sizes)) == 3
    other = (100 + 3) * 2
    for swapped in range(blocks + 1):
        sim = Sim(policy)
        model = gguf_toy(blocks, sim)
        placement = Placement(model, CPU, swapped, prefetch, backend=sim)
        placement.enter("dit")
        assert same(gguf_run(model, sim), expected), swapped
        last = sizes[blocks - swapped :]
        assert placement.block_bytes == max(last, default=0)
        slots = min(SLOTS, swapped) * placement.block_bytes
        assert sim.live() == slots
        assert placement.device_bytes() == sum(sizes[: blocks - swapped]) + other + slots
        if swapped:
            dtypes = [field.dtype for field in placement._streaming[0].fields]  # pyright: ignore[reportPrivateUsage]
            assert dtypes == [torch.uint8, torch.float16]


def test_a_gguf_file_is_read_into_its_parts_and_hashed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # model._load_weights and blockswap.read over a GGUF file: the quantised matrices read as the
    # uint8 bytes the file holds, (rows, bytes of a row), the float16 ones as they are, into the
    # parts' host copies for the blocks a plan swaps and onto the device for the rest; every byte
    # of the file hashed once in its order, the header, the padding between and after the tensors
    # included, so that the SHA-256 is hashlib's of the whole file.
    sim = Sim()
    monkeypatch.setattr(blockswap, "Cuda", lambda device: sim)
    source = gguf_toy(4, sim)
    path = write_gguf(tmp_path / "toy.gguf", source)
    found = gguf_file.read(path)
    gaps = [
        (a.offset + a.nbytes, b.offset)
        for a, b in itertools.pairwise(found.tensors)
        if a.offset + a.nbytes < b.offset
    ]
    assert gaps
    whole = path.read_bytes()
    assert whole[found.tensors[0].offset + found.tensors[0].nbytes] == 0xEE

    spans: list[tuple[int, int]] = []
    real = blockswap._Tee.hashed  # pyright: ignore[reportPrivateUsage]

    def recorded(tee: Any, piece: memoryview) -> None:
        spans.append((tee.raw.tell() - len(piece), tee.raw.tell()))
        real(tee, piece)

    monkeypatch.setattr(blockswap._Tee, "hashed", recorded)  # pyright: ignore[reportPrivateUsage]
    with torch.device("meta"):
        target = GgufToy(4, sim, fill=False)
    given: list[tuple[str, os.stat_result]] = []
    parts = model._load_weights(  # pyright: ignore[reportPrivateUsage]
        target,
        path,
        CPU,
        None,
        dit_parts(4, 1, other=False),
        hashed=lambda digest, status: given.append((digest, status)),
    )
    monkeypatch.undo()
    assert sorted(parts) == ["blocks.1", "blocks.2", "blocks.3"]
    assert len(given) == 1 and given[0][0] == hashlib.sha256(whole).hexdigest()
    assert (given[0][1].st_ino, given[0][1].st_size) == (path.stat().st_ino, len(whole))
    # Each byte once, in the file's order, end to end.
    assert spans[0][0] == 0 and spans[-1][1] == len(whole)
    assert all(end == start for (_, end), (start, _) in itertools.pairwise(spans))
    for key, value in source.state_dict().items():
        loaded = target.state_dict()[key]
        assert loaded.dtype == value.dtype and torch.equal(loaded, value), key
    assert target.get_parameter("blocks.2.mix.weight").dtype == torch.uint8
    assert target.get_parameter("blocks.2.mix.weight").shape == (GGUF_WIDTH, 144 * 1)
    host = parts["blocks.1"].host.bytes
    for parameter in target.get_submodule("blocks.1").parameters():
        assert overlap(parameter.data, host)
    assert not mappings(path)
    # The same tensors read without the hash, and the parts' fields in the model's order.
    plain, plain_parts = blockswap.read(
        GgufToy(4, sim, fill=False), path, CPU, None, dit_parts(4, 1, other=False), sim
    )
    assert list(plain) == list(source.state_dict())
    assert sorted(plain_parts) == sorted(parts)
    assert [field.dtype for field in parts["blocks.1"].fields] == [torch.uint8, torch.float16]
    # The block's matrix and its bias, a part of the toy's three sizes.
    assert [part.size for part in parts.values()] == [
        gguf_part_bytes(GGUF_KINDS[2 * i % 3]) for i in (1, 2, 3)
    ]


def test_a_gguf_file_cut_short_is_refused_in_safetensors_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As test_a_file_cut_short_is_refused_as_it_hashes_too: the same refusal, with the bytes held
    # and the bytes the header claims, whether the file was cut before the load or changes under
    # it; nothing hashed is given, the hash thread ended.
    sim = Sim()
    source = gguf_toy(3, sim)
    path = write_gguf(tmp_path / "toy.gguf", source)
    info = gguf_file.read(path)
    last = max(each.offset + each.nbytes for each in info.tensors)
    given: list[str] = []

    def read_it() -> None:
        blockswap.read(
            torch.nn.Module(),
            path,
            CPU,
            None,
            None,
            hashed=lambda digest, status: given.append(digest),
        )

    with path.open("r+b") as file:
        file.truncate(last - 10)
    with pytest.raises(weights.ModelError) as error:
        read_it()
    assert str(error.value) == (
        f"{path}: cut short as it loaded, {last - 10:,} of {last:,} bytes; {weights.AGAIN}"
    )
    # Cut inside the header: said as cut short as it loaded, with the way out.
    with path.open("r+b") as file:
        file.truncate(100)
    with pytest.raises(weights.ModelError, match=f"^{path}: cut short as it loaded") as error:
        read_it()
    assert str(error.value).endswith(f"; {weights.AGAIN}")
    # Cut after the header was read: the tensors' read meets the end of the file.
    path = write_gguf(tmp_path / "again.gguf", source)
    size = path.stat().st_size
    header_of = gguf_file.header_of

    def then_cut(file: Any, length: int) -> Any:
        found = header_of(file, length)
        with path.open("r+b") as other:
            other.truncate(last - 10)
        return found

    monkeypatch.setattr(gguf_file, "header_of", then_cut)
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(
            torch.nn.Module(),
            path,
            CPU,
            None,
            None,
            hashed=lambda digest, status: given.append(digest),
        )
    assert str(error.value) == (
        f"{path}: cut short as it loaded, {last - 10:,} of {size:,} bytes; {weights.AGAIN}"
    )
    # The kind and the facts: damaged, the bytes it has and the bytes it needs.
    assert isinstance(error.value, ModelDamaged) and error.value.kind == "damaged"
    assert (error.value.file, error.value.figure) == (path.name, "size")
    assert (error.value.found, error.value.expected) == (last - 10, size)
    assert not given
    assert not [t for t in threading.enumerate() if t.name.startswith("seedvr2x-sha256")]


def test_a_gguf_header_is_parsed_from_the_bytes_that_are_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The offsets the tensors are read at come from the header's bytes the SHA-256 covers, as a
    # safetensors file's: the header parsed on the file to learn where the data starts, then its
    # bytes read through the tee (hashed) and parsed again, in memory. A header that reads
    # differently the second time is a file changed as it loaded, refused.
    sim = Sim()
    source = gguf_toy(3, sim)
    path = write_gguf(tmp_path / "toy.gguf", source)
    whole = path.read_bytes()
    info = gguf_file.read(path)
    real = gguf_file.header_of
    calls: list[Any] = []

    def recorded(file: Any, length: int) -> Any:
        calls.append(file)
        return real(file, length)

    monkeypatch.setattr(gguf_file, "header_of", recorded)
    given: list[str] = []
    blockswap.read(
        GgufToy(3, sim, fill=False),
        path,
        CPU,
        None,
        None,
        sim,
        hashed=lambda digest, status: given.append(digest),
    )
    assert len(calls) == 2 and isinstance(calls[1], io.BytesIO)
    assert calls[1].getvalue() == whole[: info.data_start]
    assert given == [hashlib.sha256(whole).hexdigest()]

    # The second reading differs (a tensor under another name, as a file changed under the load
    # would give): refused, nothing hashed given, the hash thread ended.
    def then_another(file: Any, length: int) -> Any:
        found = real(file, length)
        if not isinstance(file, io.BytesIO):
            return found
        first = replace(found.tensors[0], name="other")
        return replace(found, tensors=(first, *found.tensors[1:]))

    monkeypatch.setattr(gguf_file, "header_of", then_another)
    given.clear()
    with pytest.raises(weights.ModelError) as error:
        blockswap.read(
            torch.nn.Module(),
            path,
            CPU,
            None,
            None,
            hashed=lambda digest, status: given.append(digest),
        )
    assert str(error.value) == (
        f"{path}: changed as it loaded, its header read twice differently; {weights.AGAIN}"
    )

    # Or one that no longer parses.
    def then_damaged(file: Any, length: int) -> Any:
        if isinstance(file, io.BytesIO):
            raise gguf_file.GgufError("damaged", kind="damaged")
        return real(file, length)

    monkeypatch.setattr(gguf_file, "header_of", then_damaged)
    with pytest.raises(weights.ModelError, match="changed as it loaded") as changed:
        blockswap.read(torch.nn.Module(), path, CPU, None, None)
    assert isinstance(changed.value, ModelChanged) and changed.value.file == path.name
    assert not given
    assert not [t for t in threading.enumerate() if t.name.startswith("seedvr2x-sha256")]


def test_a_quantised_gguf_tensor_of_no_rows_is_refused_in_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # weights.check pins the matrices' shapes, so a checked file never has one; read without the
    # check, a quantised matrix of 0 rows is a refusal, not a division by zero.
    sim = Sim()
    path = write_gguf(tmp_path / "toy.gguf", gguf_toy(2, sim))
    real = gguf_file.header_of

    def empty(file: Any, length: int) -> Any:
        found = real(file, length)
        at = next(i for i, each in enumerate(found.tensors) if each.type != "F16")
        none = replace(found.tensors[at], shape=(0, GGUF_WIDTH), nbytes=0)
        return replace(found, tensors=(*found.tensors[:at], none, *found.tensors[at + 1 :]))

    monkeypatch.setattr(gguf_file, "header_of", empty)
    with pytest.raises(weights.ModelError, match="with no rows"):
        blockswap.read(torch.nn.Module(), path, CPU, None, None)


@pytest.mark.parametrize(
    "architecture", [weights.DIT_7B_DYN, weights.DIT_7B_SHARP_DYN], ids=["7b", "sharp"]
)
def test_a_gguf_dit_s_blocks_are_weights_py_s(architecture: weights.Architecture) -> None:
    # The 7B swapped to GGUF on the meta device with the architecture's own types (model.
    # _swap_gguf, 288 matrices): each block's parameters, the quantised matrices' bytes and the
    # float16 ones, are the bytes weights.py pins for it, and the parts the placement makes of
    # them (each parameter on ALIGN) the same: the planner's figures (peaks.block_sizes) are the
    # placement's.
    import seedvr2x.vendor
    from seedvr2x.vendor.common.config import create_object, load_config

    vendor = Path(seedvr2x.vendor.__file__).parent
    config: Any = load_config(str(vendor / "configs_7b" / "main.yaml"))
    with torch.device("meta"):
        dit = create_object(config.dit.model)
    model._swap_gguf(dit, architecture)  # pyright: ignore[reportPrivateUsage]
    sizes = dict.fromkeys(range(len(dit.blocks)), 0)
    parts = dict.fromkeys(range(len(dit.blocks)), 0)
    other = 0
    for key, parameter in dit.named_parameters():
        match = blockswap._BLOCK.match(key)  # pyright: ignore[reportPrivateUsage]
        # The files hold the model's other parameters in float16; the meta model's are float32.
        nbytes = parameter.numel() * (1 if parameter.dtype == torch.uint8 else 2)
        if match is None:
            other += nbytes
        else:
            sizes[int(match[1])] += nbytes
            parts[int(match[1])] += blockswap._aligned(nbytes)  # pyright: ignore[reportPrivateUsage]
    assert tuple(sizes.values()) == architecture.blocks and other == architecture.other
    # The parts lay each parameter on ALIGN: a few small ones, 1,024 bytes a block, over (blockswap
    # says under 16 KiB), which the planner's figures leave to its margins as fp16's.
    for part, raw in zip(parts.values(), architecture.blocks, strict=True):
        assert 0 <= part - raw <= 16 << 10
    for key, parameter in dit.named_parameters():
        layer, _, leaf = key.rpartition(".")
        if weights.MATRIX.fullmatch(layer) and leaf == "weight":
            assert parameter.dtype == torch.uint8 and architecture.dtype_of(key) in (
                "Q3_K",
                "Q4_K",
                "Q5_K",
            )
        else:
            assert parameter.dtype != torch.uint8, key
    # The planner's slot is the largest block's.
    from seedvr2x import peaks

    assert 0 <= max(parts.values()) - peaks.block_bytes(architecture) <= 16 << 10


# seedvr2x's dynamic GGUF files, the sharp 7B's and the 7B's, the first in SEEDVR2X_MODEL_DIR taken.
GGUF_DITS = ("seedvr2x_ema_7b_sharp_dyn.gguf", "seedvr2x_ema_7b_dyn.gguf")


@pytest.mark.parametrize("name", ["seedvr2x_ema_7b_dyn.gguf", "seedvr2x_ema_7b_sharp_dyn.gguf"])
def test_a_real_gguf_file_s_hash_is_its_pin_and_its_tensors_ggufs(name: str) -> None:
    # The dynamic files themselves, where the test's model directory has them (4.8 GB read on
    # the CPU): the read's SHA-256 the file's pin (pull.PINNED), hashed with the tensors as one
    # read, and every tensor the bytes of gguf-py's reader (llama.cpp's own, the tests'
    # reference): the F16 ones as float16, the quantised ones as the uint8 rows of ggml's blocks.
    directory = os.environ.get("SEEDVR2X_MODEL_DIR")
    if not directory or not (Path(directory) / name).is_file():
        pytest.skip(f"needs SEEDVR2X_MODEL_DIR holding {name}")
    gguf = pytest.importorskip("gguf", reason="the gguf package is the reference")
    import numpy as np

    from seedvr2x import pull

    path = Path(directory) / name
    state, parts, digest, _ = hashing(torch.nn.Module(), path, None)
    assert digest == pull.PINNED[name].sha256 and not parts
    reader = gguf.GGUFReader(path)
    assert len(state) == len(reader.tensors) == 1128
    for tensor in reader.tensors:
        got = state[tensor.name]
        want = torch.from_numpy(np.array(tensor.data))
        assert got.dtype == want.dtype and got.shape == want.shape, tensor.name
        assert torch.equal(got, want), tensor.name


@pytest.mark.gpu
def test_a_real_gguf_dit_s_blocks_swap_and_change_no_bit() -> None:
    # test_a_real_int8_dit_s_blocks_swap_and_change_no_bit on seedvr2x's dynamic GGUF file: the
    # DiT loaded whole, then every parameter to a pinned host copy and 0, 3, 12 and 36 blocks
    # swapped, without the prefetch and with it, the windows' bits those of the DiT loaded whole;
    # its peak the last swapped blocks' own bytes less, but the two slots, each the largest of
    # them (peaks.swapped_saving: the blocks differ, 113,689,600 to 129,909,760 bytes), from the
    # whole load's less its allocator's rounding (below); the device holding the placement's
    # bytes as each enters. Needs CUDA with the GGUF DiT and a window beside it (10 GiB free),
    # SEEDVR2X_MODEL_DIR holding a dynamic GGUF file and the VAE, and 5 GiB of host memory to pin.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from test_placement import MODELS, SLACK, VAES, allocated, found, windows

    if not MODELS:
        pytest.skip("needs SEEDVR2X_MODEL_DIR")
    from seedvr2x import peaks

    device = torch.device("cuda:0")
    path = found(GGUF_DITS)
    architecture = weights.check(path, "dit")
    models = model.load_models(path, found(VAES), device)
    try:
        dit = models.runner.dit
        assert sum(isinstance(module, kquant.GGUFLinear) for module in dit.modules()) == 288
        digests, peak = windows(models)
        whole = allocated(device)
        parts = blockswap.host(
            dit, dit_parts(len(dit.blocks), 0, other=True), blockswap.Cuda(device)
        )
        torch.cuda.empty_cache()
        away = allocated(device)
        pinned = sum(part.host.bytes.numel() for part in parts.values())
        # Loaded whole, the DiT holds more than its parameters' bytes, which a placement's packed
        # parts don't: torch's allocator gives a request of 10 MiB or more a block rounded up to
        # 2 MiB (c10/cuda/CUDACachingAllocator.cpp at v2.14.1: kMinLargeAlloc, kRoundLarge), as
        # the int8 file's 27 MiB matrices take 28 (the int8 gate above).
        # The GGUF file's matrices are 5.3 to 26 MB, many under the 10 MiB past which a request
        # is rounded alone: what the whole load holds beyond the parameters is at most the
        # rounding each one would take alone, never less than none. Measured on the RTX 5090
        # (2026-11-04, the sharp file): 66.7 MiB, where each matrix rounded alone would be 291.8.
        # The placements' peaks are held to the whole load's without it.
        mib = 2**20
        counted = 0
        for parameter in dit.parameters():
            size = parameter.numel() * parameter.element_size()
            if size > 10 * mib:
                counted += -(-size // (2 * mib)) * 2 * mib - size
        rounding = whole - away - pinned
        assert 0 <= rounding <= counted + 8 * mib, (rounding / mib, counted / mib)
        for swapped, prefetch in [
            (0, False),
            (3, False),
            (3, True),
            (12, True),
            (36, False),
            (36, True),
        ]:
            said = f"{swapped} swapped, prefetch {prefetch}"
            placement = Placement(dit, device, swapped, prefetch, parts=parts)
            placement.enter("dit")
            assert abs(allocated(device) - away - placement.device_bytes()) < SLACK, said
            assert abs(placement.block_bytes - peaks.slot_bytes(architecture, swapped)) < 16 << 10
            got, got_peak = windows(models)
            assert got == digests, said
            expected = peak - rounding - peaks.swapped_saving(architecture, swapped)
            assert abs(got_peak - expected) < SLACK, (said, got_peak / 2**30, expected / 2**30)
            placement.enter("encode", 0)
            assert abs(allocated(device) - away) < SLACK, said
            placement.close()
    finally:
        models.runner.dit = models.runner.vae = None
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("policy", POLICIES)
def test_every_slot_is_verified_against_its_host_copy(policy: str) -> None:
    sim = Sim(policy)
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, verify=True, backend=sim)
    placement.enter("dit")
    assert same(run(model, sim), reference(7))
    assert placement.verified == 3 * 5
    # A slot that doesn't hold its block's bytes is said, before the block runs.
    placement._slots[0].bytes.fill_(0)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match=r"slot 0 doesn't hold blocks\.2"):
        model(torch.randn(4, WIDTH))


def uploads(sim: Sim, placement: Placement) -> list[tuple[str, int, int]]:
    """The copies that ran, in order: the stream, the slot and the swapped block."""
    parts = placement._streaming  # pyright: ignore[reportPrivateUsage]
    blocks = {part.host.bytes.data_ptr(): j for j, part in enumerate(parts)}
    slots = [slot.bytes.data_ptr() for slot in placement._slots]  # pyright: ignore[reportPrivateUsage]
    copies = [op for op in sim.ran if op[0] == "copy"]
    return [(stream, slots.index(buffer), blocks[source]) for _, stream, buffer, source in copies]


@pytest.mark.parametrize("prefetch", [False, True])
def test_the_slot_of_each_step(prefetch: bool) -> None:
    sim = Sim()
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, prefetch, backend=sim)
    placement.enter("dit")
    run(model, sim, forwards=2)
    if prefetch:
        # Blocks 0 and 1 on their way as the DiT's phase starts, then each slot's next block as
        # its block's forward ends: block j + 2, or past the last one the slot's first again,
        # for the next forward.
        order = [0, 1, *[2, 3, 4, 1, 0] * 2]
        expected = [("side-0", block % 2, block) for block in order]
    else:
        # Block j copied into slot j % 2 before its forward, in the DiT's order.
        expected = [("compute", block % 2, block) for block in [*range(5)] * 2]
    assert uploads(sim, placement) == expected
    assert placement.uploads == len(expected)


def test_the_blocks_that_stay_compute_before_the_first_upload_is_waited_for() -> None:
    # Nothing waits on the host: with the compute stream first, the two blocks that stay, and
    # the parameters before them, run before any copy does.
    sim = Sim("compute-first")
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, backend=sim)
    placement.enter("dit")
    model(torch.randn(4, WIDTH))
    assert sim.ran == [] and not sim.idle()
    sim.settle()
    assert [op[:2] for op in sim.ran[:4]] == [
        ("kernel", "before"),
        ("kernel", "blocks.0"),
        ("kernel", "blocks.1"),
        ("copy", "side-0"),
    ]
    assert all(op[0] != "wait" for op in sim.ran)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("swapped", [1, 2])
def test_one_or_two_swapped_blocks_stay_in_their_slots(swapped: int, prefetch: bool) -> None:
    sim = Sim()
    model = toy(7, sim)
    placement = Placement(model, CPU, swapped, prefetch, backend=sim)
    placement.enter("dit")
    assert same(run(model, sim, forwards=4), reference(7, forwards=4))
    assert placement.uploads == swapped


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_all_36_blocks_swapped_wrap_around(policy: str, prefetch: bool) -> None:
    sim = Sim(policy)
    model = toy(36, sim)
    placement = Placement(model, CPU, 36, prefetch, backend=sim)
    placement.enter("dit")
    assert same(run(model, sim, forwards=3), reference(36, forwards=3))
    order = [block for _, _, block in uploads(sim, placement)]
    slots = [slot for _, slot, _ in uploads(sim, placement)]
    # With the prefetch, blocks 0 and 1 are fetched again as each forward ends, for the next.
    assert order == [*range(36)] * 3 + ([0, 1] if prefetch else [])
    assert slots == [block % 2 for block in order]
    assert sim.live() == 2 * placement.block_bytes


def test_a_slot_is_written_only_after_its_last_reader() -> None:
    # The simulation refuses a copy into memory a queued kernel still reads: a placement whose
    # copies don't wait for the slot's last reader is caught by it, not by luck.
    sim = Sim("side-first")
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, backend=sim)
    placement.enter("dit")
    for slot in placement._slots:  # pyright: ignore[reportPrivateUsage]
        slot.done = None
    original = Event.record

    def forgetful(self: Event, stream: Stream) -> None:
        # The end of a reader's work is never marked: nothing holds the next copy back.
        if stream.name == "compute":
            self.set = True
        else:
            original(self, stream)

    Event.record = forgetful  # type: ignore[method-assign]
    try:
        model(torch.randn(4, WIDTH))
        with pytest.raises(AssertionError, match=r"written under blocks\.2, still to run"):
            sim.settle()
    finally:
        Event.record = original  # type: ignore[method-assign]


def test_a_block_waits_for_its_copy() -> None:
    # The same toy with the block's wait taken out: its kernel runs on what the slot held.
    sim = Sim("compute-first")
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, backend=sim)
    placement.enter("dit")
    sim.main.wait_event = lambda event: None  # type: ignore[method-assign]
    assert not same(run(model, sim), reference(7))


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_forward_that_raises_leaves_the_next_one_its_bits(policy: str, prefetch: bool) -> None:
    sim = Sim(policy)
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, prefetch, backend=sim)
    placement.enter("dit")
    expected = reference(7)
    assert same(run(model, sim, forwards=1), expected[:1])
    for failing in (3, 4, 6):
        cast(Block, model.blocks[failing]).fail = True
        with pytest.raises(RuntimeError, match="a forward that raises"):
            model(torch.randn(4, WIDTH))
        cast(Block, model.blocks[failing]).fail = False
        assert same(run(model, sim), expected), failing


def hosted(blocks: int, sim: Sim, swapped: int, **options: Any) -> tuple[Toy, Placement]:
    """A toy whose every parameter has a host copy, as a plan that sends the DiT away has it."""
    model = toy(blocks, sim)
    parts = blockswap.host(model, dit_parts(blocks, 0, other=True), sim)
    return model, Placement(model, CPU, swapped, parts=parts, backend=sim, **options)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("swapped", [0, 3, 7])
@pytest.mark.parametrize("policy", ["side-first", "compute-first", "random-1"])
def test_the_dit_leaves_and_comes_back(policy: str, swapped: int, prefetch: bool) -> None:
    blocks = 7
    sim = Sim(policy)
    model, placement = hosted(blocks, sim, swapped, prefetch=prefetch)
    expected = reference(blocks)
    tables = {key: buffer.data_ptr() for key, buffer in model.named_buffers()}
    block = placement.block_bytes if swapped else placement._sizes["blocks.0"]  # pyright: ignore[reportPrivateUsage]
    other = placement._sizes[OTHER]  # pyright: ignore[reportPrivateUsage]
    fixed = blocks - swapped
    # Away after the load: nothing of the DiT on the device.
    assert sim.live() == 0 == placement.device_bytes(0)
    if swapped:
        with pytest.raises(RuntimeError, match="the DiT runs outside its phase"):
            model(torch.randn(4, WIDTH))
        sim.main.queue.clear()

    placement.enter("dit")
    whole = other + fixed * block + min(SLOTS, swapped) * block
    assert sim.live() == whole == placement.device_bytes()
    assert same(run(model, sim), expected)

    # A VAE phase that keeps nothing: the device's work waited for first, then every buffer
    # gone, the parameters their host copies' again.
    trims = sim.trims
    placement.enter("encode", 0)
    assert sim.ran[-1][0] == "wait" and sim.idle()
    assert sim.live() == 0
    assert sim.trims == trims + 1  # what left, handed back to the driver
    hosts = [part.host.bytes for part in placement._parts.values()]  # pyright: ignore[reportPrivateUsage]
    for key, parameter in model.named_parameters():
        assert any(overlap(parameter.data, bytes_) for bytes_ in hosts), key

    placement.enter("dit")
    assert sim.live() == whole
    assert same(run(model, sim), expected)

    # A VAE phase that keeps what the DiT's phase has: nothing moves.
    ran, trims = len(sim.ran), sim.trims
    placement.enter("decode")
    assert sim.live() == whole and len(sim.ran) == ran and sim.trims == trims

    # One that keeps some of the blocks that aren't swapped: those, the other parameters, no
    # slot.
    for resident in range(fixed + 1):
        placement.enter("decode", resident)
        kept = resident * block + (other if resident else 0)
        assert sim.live() == kept == placement.device_bytes(resident), resident
        placement.enter("dit")
        assert same(run(model, sim, forwards=1), expected[:1]), resident
    assert {key: buffer.data_ptr() for key, buffer in model.named_buffers()} == tables
    for bad in (-1, fixed + 1):
        with pytest.raises(ValueError, match="blocks to keep of the"):
            placement.enter("encode", bad)
    with pytest.raises(ValueError, match="the DiT's phase keeps its"):
        placement.enter("dit", fixed + 1)


def test_nothing_is_queued_when_a_part_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = Sim("compute-first")
    model, placement = hosted(7, sim, 5)
    placement.enter("dit")
    model(torch.randn(4, WIDTH))
    assert not sim.idle()
    leaves: list[bool] = []
    leave = Part.leave

    def checked(self: Part) -> None:
        leaves.append(sim.idle())
        leave(self)

    monkeypatch.setattr(Part, "leave", checked)
    placement.enter("encode", 0)
    assert leaves == [True] * 3  # the two blocks that stay in the DiT's phase, and the rest


def test_a_part_with_no_host_copy_cannot_leave() -> None:
    sim = Sim()
    model = toy(7, sim)
    placement = Placement(model, CPU, 5, backend=sim)  # the swapped blocks' copies only
    placement.enter("dit")
    with pytest.raises(ValueError, match="block 0 has no host copy to leave the device for"):
        placement.enter("encode", 0)
    with pytest.raises(ValueError, match="block 1 has no host copy"):
        placement.enter("decode", 1)
    assert placement.host_bytes == 5 * 4096
    # The blocks that stay count as they are on the device, the swapped ones as their slots.
    raw = sum(p.numel() * p.element_size() for p in model.blocks[0].parameters())
    other = 2 * raw
    assert placement.device_bytes() == other + 2 * raw + 2 * placement.block_bytes
    assert placement.block_bytes == 3 * 512
    # Nothing moved, the DiT still whole in its phase.
    assert same(run(model, sim), reference(7))
    placement.enter("encode")
    assert same(run(model, sim), reference(7))


def test_the_swapped_blocks_need_their_host_copies() -> None:
    sim = Sim()
    model = toy(7, sim)
    parts = blockswap.host(model, dit_parts(7, 4, other=False), sim)
    with pytest.raises(ValueError, match=r"no host copy for the swapped blocks \[2, 3\]"):
        Placement(model, CPU, 5, parts=parts, backend=sim)
    with pytest.raises(ValueError, match="8 blocks to swap of the DiT's 7"):
        Placement(model, CPU, 8, backend=sim)


def test_a_placement_closed_leaves_the_model_to_another() -> None:
    sim = Sim()
    model, placement = hosted(7, sim, 5)
    placement.enter("dit")
    expected = reference(7)
    assert same(run(model, sim), expected)
    parts = placement._parts  # pyright: ignore[reportPrivateUsage]
    trims = sim.trims
    placement.close()
    assert sim.live() == 0 and sim.idle() and sim.trims == trims + 1
    assert not any(block._forward_pre_hooks or block._forward_hooks for block in model.blocks)
    for swapped in (0, 2, 7):
        other = Placement(model, CPU, swapped, parts=parts, backend=sim)
        other.enter("dit")
        assert same(run(model, sim), expected), swapped
        other.close()


def test_a_close_that_fails_can_be_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    # A placement is closed once everything is done, its hooks off the blocks: a close stopped
    # half-way, here at the trim, leaves one that isn't, to close again.
    sim = Sim()
    model, placement = hosted(7, sim, 5)
    placement.enter("dit")
    trim = sim.trim

    def failing() -> None:
        raise RuntimeError("the device said no")

    monkeypatch.setattr(sim, "trim", failing)
    with pytest.raises(RuntimeError, match="the device said no"):
        placement.close()
    assert any(block._forward_pre_hooks for block in model.blocks)
    monkeypatch.setattr(sim, "trim", trim)
    placement.close()
    assert not any(block._forward_pre_hooks or block._forward_hooks for block in model.blocks)
    with pytest.raises(RuntimeError, match="the placement is closed"):
        placement.enter("dit")


def test_a_closed_placement_enters_no_phase() -> None:
    # Its hooks are off the blocks: a forward would run on slots nothing fills.
    sim = Sim()
    _, placement = hosted(7, sim, 5)
    placement.enter("dit")
    placement.close()
    with pytest.raises(RuntimeError, match="the placement is closed"):
        placement.enter("dit")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("leaving", ["a VAE phase that keeps every block", "close"])
def test_the_slots_go_only_once_what_is_queued_is_done(leaving: str, policy: str) -> None:
    # A forward's work still queued, its last prefetch among it, when the slots go: alone, in a
    # VAE phase that keeps every block that isn't swapped, or with everything at close. The
    # device's work is waited for first: nothing is queued after, and the buffers are gone, where
    # a copy still queued would hold its slot.
    sim = Sim(policy)
    model_, placement = hosted(7, sim, 5)
    placement.enter("dit")
    model_(torch.randn(4, WIDTH))
    assert not sim.idle()
    if leaving == "close":
        placement.close()
        assert sim.idle() and sim.live() == 0
    else:
        placement.enter("decode", 2)
        assert sim.idle() and sim.live() == placement.device_bytes(2)
        placement.enter("dit")
        assert same(run(model_, sim), reference(7))


class Reusing(Sim):
    """A simulation whose allocations take memory freed before, as an allocator gives the
    compute stream's next allocation what that stream just freed."""

    def __init__(self, policy: str) -> None:
        super().__init__(policy)
        self.freed: list[Tensor] = []

    def allocate(self, size: int) -> Tensor:
        for index, buffer in enumerate(self.freed):
            if buffer.numel() >= size:
                del self.freed[index]
                view = buffer[:size]
                self.buffers.append(weakref.ref(view))
                self.sizes[view.data_ptr()] = size
                return view
        return super().allocate(size)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_new_slot_waits_for_what_the_compute_stream_has_queued(
    policy: str, prefetch: bool
) -> None:
    # The caller has a kernel still queued that fills a temporary, freed since, when the DiT's
    # phase is entered: the slot is that memory. Its first copy, on the side stream, must come
    # after that kernel, or the kernel overwrites a block's weights, and nothing says so.
    sim = Reusing(policy)
    model_, placement = hosted(7, sim, 7, prefetch=prefetch)
    temporary = torch.zeros(placement.block_bytes, dtype=torch.uint8)

    def fill(filled: Tensor = temporary) -> None:
        filled.fill_(0x33)

    sim.kernel("the caller's kernel", fill, [temporary])
    sim.freed.append(temporary)
    del temporary
    placement.enter("dit")
    assert same(run(model_, sim), reference(7))


def test_every_plan_s_phases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The loader's parts and Models.enter (seedvr2x/model.py) over every plan of a toy of 4
    # blocks with a toy VAE: each number of swapped blocks, each VAE phase keeping the DiT as it
    # is (None), nothing, or any number of its blocks that aren't swapped, the VAE staying for
    # the DiT's phase or not. After each phase entered: the device holds what the plan says, to
    # the byte; the DiT's bits and the VAE's are the whole models'; and for a plan that keeps
    # the two models apart, their weights were never on the device together.
    blocks = 4
    expected = reference(blocks)

    def vae_bits(vae: Block, sim: Sim) -> Tensor:
        torch.manual_seed(77)
        out = vae(torch.randn(4, WIDTH))
        sim.settle()
        return out.clone()

    def toy_vae(sim: Sim) -> Block:
        torch.manual_seed(99)
        return Block("vae", sim).requires_grad_(False).eval()

    plain = Sim()
    vae_expected = vae_bits(toy_vae(plain), plain)
    dit_file, vae_file = tmp_path / "dit.safetensors", tmp_path / "vae.safetensors"
    save_file(toy(blocks, plain).state_dict(), dit_file)
    save_file(toy_vae(plain).state_dict(), vae_file)
    order = ["encode", "dit", "decode", "encode", "dit", "dit", "decode", "decode", "dit", "encode"]
    plans: set[ModelPlan] = set()
    runs = 0
    for swapped in range(blocks + 1):
        fixed = blocks - swapped
        residents = [None, *range(fixed + 1)]
        for kept_encode, kept_decode, vae_in_dit, first in itertools.product(
            residents, residents, (True, False), ("encode", "dit")
        ):
            runs += 1
            plan = ModelPlan(
                swapped=swapped,
                resident_encode=kept_encode,
                resident_decode=kept_decode,
                vae_in_dit=vae_in_dit,
            )
            plans.add(plan)
            sim = Sim("random-1")
            monkeypatch.setattr(blockswap, "Cuda", lambda device, sim=sim: sim)
            with torch.device("meta"):
                dit, vae = Toy(blocks, sim), Block("vae", sim)
            # As load_models loads and assembles them.
            leaves = (lambda key: model.VAE_PART) if not vae_in_dit else None
            vae_parts = model._load_weights(vae, vae_file, CPU, None, leaves)  # pyright: ignore[reportPrivateUsage]
            dit_parts_ = model._load_weights(  # pyright: ignore[reportPrivateUsage]
                dit,
                dit_file,
                CPU,
                None,
                model._dit_parts(plan, blocks),  # pyright: ignore[reportPrivateUsage]
            )
            dit.requires_grad_(False).eval()
            vae.requires_grad_(False).eval()
            runner = SimpleNamespace(dit=dit, vae=vae)
            models = model.Models(runner, torch.empty(0), torch.empty(0), "none", CPU, plan)
            if dit_parts_ or plan.swapped:
                models.placement = Placement(dit, CPU, plan.swapped, parts=dit_parts_)
            models.vae = vae_parts.get(model.VAE_PART)
            placement = models.placement
            vae_bytes = models.vae.size if models.vae is not None else 0

            def on_device(resident: int | None, placement: Placement | None = placement) -> int:
                """The bytes of the DiT's parts and slots on the device in a phase."""
                if placement is None:
                    return 0
                kept, other, slots = placement._kept("encode", resident)  # pyright: ignore[reportPrivateUsage]
                parts = placement._parts  # pyright: ignore[reportPrivateUsage]
                names = [f"blocks.{j}" for j in range(kept)] + ([OTHER] if other else [])
                there = sum(parts[name].size for name in names if name in parts)
                return there + (
                    min(SLOTS, placement.swapped) * placement.block_bytes if slots else 0
                )

            assert sim.alive() == 0, plan
            apart = not vae_in_dit and kept_encode == 0 and kept_decode == 0
            previous: str | None = None
            for phase in ([first] if first == "dit" else []) + order:
                trims = sim.trims
                models.enter(phase)  # type: ignore[arg-type]
                resident = {"encode": kept_encode, "dit": None, "decode": kept_decode}[phase]
                wanted = on_device(resident) + (vae_bytes if phase != "dit" else 0)
                # Reference counts free the buffers: no collection needed, which 4,000 phases
                # would take minutes in.
                assert sim.alive() == wanted, (plan, phase)
                # What a phase sends away is handed back to the driver: a model for the other.
                if apart and previous is not None and (phase == "dit") != (previous == "dit"):
                    assert sim.trims > trims, (plan, phase)
                previous = phase
                assert models.phase == phase
                if phase == "dit":
                    assert same(run(dit, sim, forwards=2), expected[:2]), (plan, phase)
                else:
                    assert torch.equal(vae_bits(vae, sim), vae_expected), (plan, phase)
            if apart:
                assert sim.peak <= max(vae_bytes, on_device(None)), plan
    # 180 plans, each entered twice: from a shot's encode first, and from a window first.
    assert (len(plans), runs) == (180, 360)


def test_a_phase_that_fails_to_enter_is_no_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    # The DiT's return fails, the VAE already gone: the models are in no phase, and the phase
    # they left is entered again in full, not taken for entered.
    sim = Sim()
    monkeypatch.setattr(blockswap, "Cuda", lambda device: sim)
    dit, placement = hosted(4, sim, 2)
    torch.manual_seed(99)
    vae = Block("vae", sim).requires_grad_(False).eval()
    part = blockswap.host(vae, lambda key: model.VAE_PART, sim)[model.VAE_PART]
    plan = ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False)
    runner = SimpleNamespace(dit=dit, vae=vae)
    models = model.Models(
        runner, torch.empty(0), torch.empty(0), "none", CPU, plan, placement, part
    )
    models.enter("decode")
    assert models.phase == "decode" and sim.live() == part.size

    def refuse(size: int) -> Tensor:
        raise torch.OutOfMemoryError("a stand-in")

    with monkeypatch.context() as patch:
        patch.setattr(sim, "allocate", refuse)
        with pytest.raises(torch.OutOfMemoryError):
            models.enter("dit")
    assert models.phase is None and sim.live() == 0
    models.enter("decode")
    assert models.phase == "decode" and sim.live() == part.size
    models.enter("dit")
    assert same(run(dit, sim), reference(4))


def test_a_host_copy_is_a_private_mapping() -> None:
    # One kind of mapping, the one research/scripts/pin_probe.py measured: private, anonymous,
    # as /proc says of the copy's address; and kept out of a child process (MADV_DONTFORK, the
    # kernel's "dc" flag), where a pinned page copied on write would leave the driver the
    # parent's old one.
    copy = blockswap.HostCopy(3 * 4096, pin=False)
    address = copy.bytes.data_ptr()
    inside = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        fields = line.split()
        if "-" in fields[0] and not fields[0].endswith(":"):
            start, end = (int(part, 16) for part in fields[0].split("-"))
            inside = start <= address < end
            if inside:
                assert fields[1] == "rw-p", line
        elif inside and fields[0] == "VmFlags:":
            assert "dc" in fields[1:], line
            break
    else:
        pytest.fail("the copy's address is in no mapping")


def test_a_host_copy_on_windows_is_an_anonymous_mmap_with_no_posix_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Python's mmap has MAP_PRIVATE, MAP_ANONYMOUS and madvise on POSIX only; on Windows
    # mmap.mmap(-1, length) is the anonymous mapping. The platform is faked and mmap's call
    # recorded; the stand-in is a bytearray, which has no madvise, so that a call to it fails.
    calls: list[tuple[Any, ...]] = []

    def anonymous(*args: Any, **kwargs: Any) -> bytearray:
        calls.append((args, kwargs))
        return bytearray(args[1])

    fake = SimpleNamespace(mmap=anonymous, PAGESIZE=4096)
    monkeypatch.setattr(blockswap, "mmap", fake)
    monkeypatch.setattr(blockswap, "sys", SimpleNamespace(platform="win32"))
    copy = blockswap.HostCopy(4096 + 1, pin=False)
    assert calls == [((-1, 2 * 4096), {})]
    assert copy.bytes.numel() == 2 * 4096 and not copy.pinned


def test_a_host_copy_on_linux_is_a_private_anonymous_mmap_kept_from_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other branch, by its calls: the flags and the madvise (the mapping itself is read in
    # /proc by test_a_host_copy_is_a_private_mapping).
    import mmap

    calls: list[tuple[Any, ...]] = []

    class Area(bytearray):
        def madvise(self, option: int) -> None:
            calls.append(("madvise", option))

    def anonymous(*args: Any, **kwargs: Any) -> Area:
        calls.append(("mmap", args, kwargs))
        return Area(args[1])

    fake = SimpleNamespace(
        mmap=anonymous,
        PAGESIZE=mmap.PAGESIZE,
        MAP_PRIVATE=mmap.MAP_PRIVATE,
        MAP_ANONYMOUS=mmap.MAP_ANONYMOUS,
        MADV_DONTFORK=mmap.MADV_DONTFORK,
    )
    monkeypatch.setattr(blockswap, "mmap", fake)
    monkeypatch.setattr(blockswap, "sys", SimpleNamespace(platform="linux"))
    blockswap.HostCopy(1, pin=False)
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    assert calls == [
        ("mmap", (-1, mmap.PAGESIZE), {"flags": flags}),
        ("madvise", mmap.MADV_DONTFORK),
    ]


@pytest.mark.parametrize("missing", ["the binding", "the call", "CUDA"])
def test_pinning_without_torch_s_binding_is_a_refusal(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    # torch binds CUDA's runtime without documenting it: a torch without it, or without CUDA,
    # refuses the plan as a host short of memory does, with no traceback of its own.
    def cudart() -> Any:
        if missing == "the binding":
            raise AttributeError("module 'torch._C' has no attribute '_cudart'")
        if missing == "CUDA":
            raise AssertionError("Torch not compiled with CUDA enabled")
        return SimpleNamespace(cudaHostUnregister=None, cudaGetErrorString=None)

    monkeypatch.setattr(blockswap, "_cudart", cudart)
    with pytest.raises(HostMemoryError, match="CUDA's cudaHostRegister isn't to be had from"):
        blockswap.HostCopy(4096, pin=True)


def test_a_copy_is_registered_whole_and_unregistered_when_it_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A stand-in for torch's binding of CUDA's runtime: the copy's address and its length in
    # whole pages are registered; unregistered when the copy is collected, and not at exit,
    # where CUDA is gone first.
    calls: list[tuple[Any, ...]] = []

    def register(address: int, length: int, flags: int) -> int:
        calls.append(("on", address, length, flags))
        return 0

    cudart = SimpleNamespace(
        cudaHostRegister=register,
        cudaHostUnregister=lambda address: calls.append(("off", address)),
        cudaGetErrorString=lambda code: "unused",
    )
    monkeypatch.setattr(blockswap, "_cudart", lambda: cudart)
    copy = blockswap.HostCopy(4096 + 1, pin=True)
    address = copy.bytes.data_ptr()
    assert copy.pinned and calls == [("on", address, 2 * 4096, 0)]
    assert copy.unregistering.alive and not copy.unregistering.atexit
    del copy
    gc.collect()
    assert calls == [("on", address, 2 * 4096, 0), ("off", address)]


def test_a_refused_registration_says_cuda_s_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    cudart = SimpleNamespace(
        cudaHostRegister=lambda address, length, flags: 2,
        cudaHostUnregister=lambda address: 0,
        cudaGetErrorString=lambda code: "out of memory",
    )
    monkeypatch.setattr(blockswap, "_cudart", lambda: cudart)
    with pytest.raises(HostMemoryError) as refusal:
        blockswap.HostCopy(4096, pin=True)
    assert str(refusal.value) == "the host refuses to pin 0.00 GiB of memory: out of memory"
    assert (refusal.value.kind, refusal.value.need, refusal.value.available) == (
        "host-memory",
        4096,
        None,
    )


def cgroup_tree(root: Path, available_kib: int, groups: str, files: dict[str, str]) -> None:
    (root / "proc" / "self").mkdir(parents=True)
    (root / "proc" / "meminfo").write_text(f"MemTotal: 99 kB\nMemAvailable:   {available_kib} kB\n")
    (root / "proc" / "self" / "cgroup").write_text(groups)
    for name, content in files.items():
        (root / "sys" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "sys" / name).write_text(content)


def test_the_memory_available_is_a_container_s_limit_too(tmp_path: Path) -> None:
    def available(case: str, kib: int, groups: str, files: dict[str, str]) -> int | None:
        cgroup_tree(tmp_path / case, kib, groups, files)
        return blockswap.available(tmp_path / case / "proc", tmp_path / case / "sys")

    # No limit: what the kernel says is available.
    assert available("free", 1000, "0::/\n", {"memory.stat": ""}) == 1000 * 1024
    assert available("max", 1000, "0::/a\n", {"a/memory.max": "max\n"}) == 1000 * 1024
    # A v2 group's limit, less what it holds; the nearest limit above counts as well.
    limited = {"a/b/memory.max": "900000\n", "a/b/memory.current": "100000\n"}
    assert available("v2", 1000, "0::/a/b\n", limited) == 800_000
    above = {**limited, "a/memory.max": "500000\n", "a/memory.current": "150000\n"}
    assert available("above", 1000, "0::/a/b\n", above) == 350_000
    # A limit looser than the host's memory changes nothing; one overrun leaves nothing.
    loose = {"a/memory.max": "9000000\n", "a/memory.current": "0\n"}
    assert available("loose", 1000, "0::/a\n", loose) == 1000 * 1024
    over = {"a/memory.max": "1000\n", "a/memory.current": "5000\n"}
    assert available("over", 1000, "0::/a\n", over) == 0
    # A group's file cache isn't memory taken: the kernel gives it back, as MemAvailable counts
    # the host's. 900,000 allowed, 850,000 held of which 700,000 of cache: 750,000 left.
    cached = {
        "a/memory.max": "900000\n",
        "a/memory.current": "850000\n",
        "a/memory.stat": "anon 150000\nfile 700000\ninactive_file 400000\nactive_file 300000\n",
    }
    assert available("cache", 1000, "0::/a\n", cached) == 750_000
    # A cache counted larger than what the group holds, the two read at different instants:
    # nothing taken, never more left than the limit.
    racing = {
        "a/memory.max": "900000\n",
        "a/memory.current": "100000\n",
        "a/memory.stat": "inactive_file 400000\nactive_file 300000\n",
    }
    assert available("racing", 1000, "0::/a\n", racing) == 900_000
    # A limit that isn't a number is no limit read: the next one counts, here the host's.
    for text in ("", "9e5"):
        odd = {"a/memory.max": f"{text}\n", "a/memory.current": "100000\n"}
        assert available(f"odd {text}", 1000, "0::/a\n", odd) == 1000 * 1024
    # memory.high is a limit as well: past it the kernel reclaims from the group and slows it.
    high = {"a/memory.max": "max\n", "a/memory.high": "600000\n", "a/memory.current": "100000\n"}
    assert available("high", 1000, "0::/a\n", high) == 500_000
    # v1: the memory controller's own tree.
    v1 = {
        "memory/c/memory.limit_in_bytes": "700000\n",
        "memory/c/memory.usage_in_bytes": "300001\n",
        "memory/c/memory.stat": "total_inactive_file 200000\ntotal_active_file 100000\ncache 9\n",
    }
    assert available("v1", 1000, "3:cpu:/c\n7:memory:/c\n", v1) == 699_999
    # Nothing readable: no figure, and the pinning itself refuses if it must.
    assert blockswap.available(tmp_path / "none", tmp_path / "none") is None


@pytest.mark.parametrize(
    ("swapped", "said", "nothing"),
    [
        (1, "the DiT's last block swapped", True),
        (2, "the DiT's last 2 blocks swapped", True),
        (3, "the DiT's last 3 blocks swapped", False),
    ],
)
def test_one_or_two_swapped_blocks_are_said_to_free_nothing(
    caplog: pytest.LogCaptureFixture, swapped: int, said: str, nothing: bool
) -> None:
    sim = Sim()
    dit = toy(7, sim)
    placement = Placement(dit, CPU, swapped, backend=sim)
    plan = ModelPlan(swapped=swapped)
    runner = SimpleNamespace(dit=dit, vae=None)
    models = model.Models(runner, torch.empty(0), torch.empty(0), "none", CPU, plan, placement)
    caplog.set_level("INFO")
    model._say_pinned(models, [f"blocks.{7 - 1 - j}" for j in range(swapped)])  # pyright: ignore[reportPrivateUsage]
    blocks = f"{swapped} of the DiT's blocks"
    assert caplog.messages[0] == (
        f"0.00 GiB of host memory pinned for the weights that leave the GPU: {blocks}; {said}"
    )
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    expected = (
        f"{said}: that frees nothing in the DiT's phase, where swapped blocks take turns in 2"
        " buffers on the GPU; from 3 swapped blocks up, each frees 0.00 GiB there"
    )
    assert warnings == ([expected] if nothing else [])
