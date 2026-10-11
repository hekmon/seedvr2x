"""The DiT's placement (runtime/blockswap.py) on the CPU: a toy model on simulated CUDA streams.

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
import itertools
import random
import weakref
from collections import deque
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

from seedvr2x.runtime import blockswap, model, weights
from seedvr2x.runtime.blockswap import OTHER, HostMemoryError, Part, Placement, dit_parts
from seedvr2x.runtime.plan import Plan
from seedvr2x.runtime.weights import SLOTS

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
    DiT block's RoPE tables are. Its forward queues one kernel and returns where its result will
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
    # and the placement moves, in runtime/weights.py, which imports no torch.
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
    # The loader's parts and Models.enter (runtime/model.py) over every plan of a toy of 4
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
    plans: set[Plan] = set()
    runs = 0
    for swapped in range(blocks + 1):
        fixed = blocks - swapped
        residents = [None, *range(fixed + 1)]
        for kept_encode, kept_decode, vae_in_dit, first in itertools.product(
            residents, residents, (True, False), ("encode", "dit")
        ):
            runs += 1
            plan = Plan(
                None,
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
    plan = Plan(None, swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False)
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
    plan = Plan(None, swapped=swapped)
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
