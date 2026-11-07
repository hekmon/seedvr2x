"""Models kept in their pinned host copies and placed per run (seedvr2x/model.py: Models.place and
Models.release, passes.HELD), on the CPU: toy models on test_blockswap.py's simulated CUDA
streams, loaded as load_models loads them for HELD (every weight in its pinned copy, none on the
device), then placed for a run's plan, run through Models.enter, released and placed again. The
bits are those of models loaded directly for the same plan; nothing is read from the files after
the load; what is on the device after each step is counted to the byte. The loader itself,
load_models with HELD on the real weights: tests/test_placement.py's family (GPU)."""

import gc
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from test_blockswap import CPU, WIDTH, Block, Sim, Toy, overlap, run, same, toy
from torch import Tensor

from seedvr2x import blockswap, model, waiting
from seedvr2x.blockswap import Placement
from seedvr2x.passes import HELD, ModelPlan

BLOCKS = 4


class Files:
    """The toy DiT's and the toy VAE's files, and what the models read from them: counted."""

    def __init__(self, directory: Path, sim: Sim, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sim = sim
        self.dit, self.vae = directory / "dit.safetensors", directory / "vae.safetensors"
        save_file(toy(BLOCKS, sim).state_dict(), self.dit)
        save_file(self.vae_model().state_dict(), self.vae)
        self.reads = 0
        monkeypatch.setattr(blockswap, "Cuda", lambda device: sim)
        reading = blockswap.read

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.reads += 1
            return reading(*args, **kwargs)

        monkeypatch.setattr(blockswap, "read", counted)

    def vae_model(self) -> Block:
        torch.manual_seed(99)
        return Block("vae", self.sim).requires_grad_(False).eval()

    def load(self, plan: ModelPlan) -> model.Models:
        """The models as load_models builds them for `plan`: its parts in their pinned copies,
        the rest on the device, the placement and the VAE's part as the plan has them."""
        with torch.device("meta"):
            dit, vae = Toy(BLOCKS, self.sim), Block("vae", self.sim)
        leaves = (lambda key: model.VAE_PART) if not plan.vae_in_dit else None
        vae_parts = model._load_weights(vae, self.vae, CPU, None, leaves)  # pyright: ignore[reportPrivateUsage]
        dit_parts = model._load_weights(  # pyright: ignore[reportPrivateUsage]
            dit,
            self.dit,
            CPU,
            None,
            model._dit_parts(plan, BLOCKS),  # pyright: ignore[reportPrivateUsage]
        )
        dit.requires_grad_(False).eval()
        vae.requires_grad_(False).eval()
        runner = SimpleNamespace(dit=dit, vae=vae)
        generator = torch.Generator().manual_seed(5)
        text = [torch.randn(3, WIDTH, generator=generator).to(torch.bfloat16) for _ in range(2)]
        models = model.Models(runner, text[0], text[1], "none", CPU, plan)
        if dit_parts or plan.swapped:
            models.placement = Placement(dit, CPU, plan.swapped, parts=dit_parts)
        models.vae = vae_parts.get(model.VAE_PART)
        if plan is HELD:
            models.parts = dict(dit_parts)
        return models


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Files:
    return Files(tmp_path, Sim("random-1"), monkeypatch)


def vae_bits(models: model.Models, sim: Sim) -> Tensor:
    torch.manual_seed(77)
    out = models.runner.vae(torch.randn(4, WIDTH))
    sim.settle()
    return out.clone()


# Plans A and B of a run each: swapped blocks, the VAE's phases keeping the DiT as it is, nothing,
# or some of its blocks, the VAE staying for the DiT's phase or not.
PLANS = [
    ModelPlan(),
    ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False),
    ModelPlan(swapped=1, resident_encode=1, resident_decode=None, vae_in_dit=True),
    ModelPlan(swapped=4, resident_encode=0, resident_decode=0, vae_in_dit=True),
    ModelPlan(swapped=0, resident_encode=2, resident_decode=3, vae_in_dit=False),
]
ORDER = ["encode", "dit", "decode", "dit", "encode", "decode"]


def phases(models: model.Models, files: Files) -> list[tuple[Any, ...]]:
    """What a run through `models` gives in each phase of ORDER: the bytes on the device after
    entering it, and the bits of the DiT's two windows or of the VAE."""
    sim, dit = files.sim, models.runner.dit
    seen: list[tuple[Any, ...]] = []
    for phase in ORDER:
        models.enter(phase)  # type: ignore[arg-type]
        assert models.phase == phase
        if phase == "dit":
            seen.append((phase, sim.live(), run(dit, sim, forwards=2)))
        else:
            seen.append((phase, sim.live(), vae_bits(models, sim)))
    return seen


def compare(
    held: list[tuple[Any, ...]], direct: list[tuple[Any, ...]], said: object, exact: bool
) -> None:
    """The bits equal, phase by phase; and the bytes on the device too where `exact`: models
    loaded for a plan keep the weights it never sends away as plain tensors, which the
    simulation does not count, where held models have every weight in a buffer."""
    assert len(held) == len(direct)
    for (phase, live, bits), (_, expected_live, expected_bits) in zip(held, direct, strict=True):
        assert not exact or live == expected_live, (said, phase)
        if phase == "dit":
            assert same(bits, expected_bits), (said, phase)
        else:
            assert torch.equal(bits, expected_bits), (said, phase)


def on_device(models: model.Models, plan: ModelPlan, phase: str) -> int:
    """The bytes the plan has on the device after `phase` is entered: the DiT's parts and slots
    as the placement keeps them, and the VAE's part where the phase has it."""
    resident = {"encode": plan.resident_encode, "dit": None, "decode": plan.resident_decode}[phase]
    assert models.placement is not None and models.vae is not None
    vae = models.vae.size if phase != "dit" or plan.vae_in_dit else 0
    return models.placement.device_bytes(resident) + vae


@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("backwards", [False, True])
def test_models_placed_for_a_plan_give_the_bits_of_models_loaded_for_it(
    files: Files, backwards: bool, release: bool
) -> None:
    # One set of held models placed for each plan in turn, A then B then the next, a run through
    # Models.enter each: the bits, phase by phase, are those of models loaded directly for that
    # plan, and the bytes on the device those the plan's placement keeps. With a release between
    # two plans, or with place alone sending back what the run left.
    order = list(reversed(range(len(PLANS)))) if backwards else list(range(len(PLANS)))
    direct = {index: phases(files.load(PLANS[index]), files) for index in order}
    models = files.load(HELD)
    assert models.model_plan == HELD and models.phase is None and files.sim.live() == 0
    reads = files.reads
    host = {name: part.host.bytes.data_ptr() for name, part in models.parts.items()}  # type: ignore[union-attr]
    for index in order:
        plan = PLANS[index]
        models.place(plan)
        assert models.model_plan == plan and models.phase is None
        assert files.sim.live() == 0, "placed, nothing on the device"
        seen = phases(models, files)
        for phase, live, _ in seen:
            assert live == on_device(models, plan, phase), (index, phase)
        exact = plan.resident_encode == plan.resident_decode == 0 and not plan.vae_in_dit
        compare(seen, direct[index], index, exact)
        if release:
            models.release()
            assert files.sim.live() == 0 and models.phase is None
    # Nothing read again, and the pinned copies the very same memory.
    assert files.reads == reads
    assert host == {name: part.host.bytes.data_ptr() for name, part in models.parts.items()}  # type: ignore[union-attr]


def test_the_vae_stays_in_the_dit_s_phase_where_the_plan_says_so(files: Files) -> None:
    # The same held models, the VAE a part always: away in the DiT's phase under a plan that
    # says so, there under one that does not (Models.enter).
    models = files.load(HELD)
    vae = models.vae
    assert vae is not None
    models.place(ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False))
    models.enter("encode")
    assert vae.device is not None
    models.enter("dit")
    assert vae.device is None
    models.place(ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=True))
    assert vae.device is None
    models.enter("dit")
    assert vae.device is not None
    dit = models.placement.device_bytes()  # type: ignore[union-attr]
    assert files.sim.live() == dit + vae.size
    models.enter("encode")
    assert files.sim.live() == vae.size
    models.enter("dit")
    assert same(run(models.runner.dit, files.sim), run(toy(BLOCKS, files.sim), files.sim))
    # The first phase entered is the DiT's, on a fresh placement: the VAE comes too.
    models.release()
    models.place(ModelPlan(vae_in_dit=True))
    models.enter("dit")
    assert vae.device is not None


def test_jobs_models_keep_the_vae_where_their_plan_has_it(files: Files) -> None:
    # What Models.enter did before: models loaded for a plan that keeps the VAE away in the DiT's
    # phase have a VAE part, and it leaves there; for one that keeps it there, no part.
    away = files.load(ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False))
    assert away.vae is not None and away.parts is None
    away.enter("encode")
    assert away.vae.device is not None
    away.enter("dit")
    assert away.vae.device is None
    kept = files.load(ModelPlan(swapped=2, resident_encode=0, resident_decode=0))
    assert kept.vae is None and kept.parts is None
    kept.enter("dit")
    kept.enter("encode")


def test_released_models_hold_nothing_on_the_device_and_release_is_idempotent(
    files: Files,
) -> None:
    sim = files.sim
    models = files.load(HELD)
    dit = models.runner.dit
    models.place(ModelPlan(swapped=3, resident_encode=1, resident_decode=0, vae_in_dit=True))
    models.enter("dit")
    run(dit, sim, forwards=1)
    models.enter("decode")
    assert sim.live() > 0 and models.vae is not None and models.vae.device is not None
    trims = sim.trims
    models.release()
    assert sim.live() == 0 and sim.idle() and sim.trims > trims
    assert models.phase is None and models.model_plan is None and models.placement is None
    assert models.vae.device is None
    hosts = [part.host.bytes for part in (*models.parts.values(), models.vae)]  # type: ignore[union-attr]
    for key, parameter in [*dit.named_parameters(), *models.runner.vae.named_parameters()]:
        assert any(overlap(parameter.data, bytes_) for bytes_ in hosts), key
    # The hooks of the swapped blocks are off the blocks.
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in dit.blocks)  # pyright: ignore[reportPrivateUsage]
    # A second and a third release change nothing.
    for _ in range(2):
        models.release()
        assert sim.live() == 0 and models.model_plan is None
    # An enter before the next place is a mistake, said so, not a run on weights that are away.
    with pytest.raises(RuntimeError, match="released"):
        models.enter("dit")
    models.place(ModelPlan(swapped=1))
    models.enter("dit")
    assert same(run(dit, sim, forwards=2), run(toy(BLOCKS, sim), sim, forwards=2))


def test_release_skips_under_its_own_devices_mark_only(
    files: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = files.sim
    models = files.load(HELD)
    models.place(ModelPlan(swapped=3, resident_encode=1, resident_decode=0, vae_in_dit=True))
    models.enter("dit")
    assert sim.live() > 0
    # The device's own card (0, a device that is no card) marked: nothing is sent back.
    monkeypatch.setattr(waiting, "_abandoned", {0})
    models.release()
    assert sim.live() > 0 and models.phase is None and models.model_plan is None
    # Another card marked: this release goes through, and clears nothing of the other's.
    models.place(ModelPlan(swapped=3, resident_encode=1, resident_decode=0, vae_in_dit=True))
    models.enter("dit")
    monkeypatch.setattr(waiting, "_abandoned", {1})
    models.release()
    assert sim.live() == 0 and waiting.abandoned(1)
    # A release that runs on an unmarked card leaves it unmarked; the clear is the block's
    # (test_waiting.py).
    monkeypatch.setattr(waiting, "_abandoned", set())
    models.place(ModelPlan(swapped=1))
    models.release()
    assert not waiting.abandoned(0)


def test_models_are_placed_again_after_a_run_that_raised(files: Files) -> None:
    # A forward that raises inside a window, then the run's end (release), a window's failure
    # leaving the slots mid-stream: the next plan's run has the bits it should.
    sim = files.sim
    models = files.load(HELD)
    dit = models.runner.dit
    expected = run(toy(BLOCKS, sim), sim)
    plan_a = ModelPlan(swapped=3, resident_encode=0, resident_decode=0, vae_in_dit=False)
    models.place(plan_a)
    models.enter("dit")
    block = dit.blocks[2]
    block.fail = True  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(RuntimeError, match="a forward that raises"):
        dit(torch.randn(4, WIDTH))
    block.fail = False  # pyright: ignore[reportAttributeAccessIssue]
    models.release()
    assert sim.live() == 0 and sim.idle()
    models.place(ModelPlan(swapped=2, resident_encode=None, resident_decode=None, vae_in_dit=True))
    models.enter("dit")
    assert same(run(dit, sim), expected)
    # Without a release in between: place does it, whatever the failed run left.
    models.enter("encode")
    block.fail = True  # pyright: ignore[reportAttributeAccessIssue]
    models.enter("dit")
    with pytest.raises(RuntimeError, match="a forward that raises"):
        dit(torch.randn(4, WIDTH))
    block.fail = False  # pyright: ignore[reportAttributeAccessIssue]
    models.place(plan_a)
    assert sim.live() == 0 and models.phase is None
    models.enter("dit")
    assert same(run(dit, sim), expected)
    # And a failure of the release itself leaves the models to be released again.
    original = sim.trim

    def failing() -> None:
        raise RuntimeError("a stand-in")

    sim.trim = failing  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="a stand-in"):
        models.release()
    sim.trim = original  # type: ignore[method-assign]
    models.release()
    assert sim.live() == 0 and models.placement is None
    models.place(HELD)
    models.enter("dit")
    assert same(run(dit, sim), expected)


def test_a_place_that_cannot_be_made_changes_nothing(files: Files) -> None:
    sim = files.sim
    loaded = files.load(ModelPlan(swapped=2, resident_encode=0, resident_decode=0))
    with pytest.raises(ValueError, match=r"weren't loaded with passes\.HELD"):
        loaded.place(ModelPlan())
    with pytest.raises(ValueError, match=r"weren't loaded with passes\.HELD"):
        loaded.release()
    models = files.load(HELD)
    models.place(ModelPlan(swapped=2))
    models.enter("dit")
    live = sim.live()
    for bad in (
        ModelPlan(swapped=BLOCKS + 1),
        ModelPlan(swapped=1, resident_encode=4),
        ModelPlan(compile_vae=True),
    ):
        with pytest.raises(ValueError):
            models.place(bad)
        # The plan in place, and its phase, as they were.
        assert models.model_plan == ModelPlan(swapped=2) and models.phase == "dit"
        assert sim.live() == live
    with pytest.raises(ValueError, match="compile_vae"):
        models.place(ModelPlan(compile_vae=True))


def test_release_drops_what_the_vae_and_the_rope_keep_between_calls(
    files: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The VAE's causal convolutions keep their memory from slice to slice, and the vendored RoPE
    # memoises get_axial_freqs on its class, up to 128 shapes of a window, with the module that
    # asked: both dropped by a release (model._forget, model._forget_rope).
    from seedvr2x.vendor.models.dit_7b.rope import NaRotaryEmbedding3d

    # The cache is the class's, shared by the process: the tables the passes of other test files
    # left (the shot entries clear at their start, not at their end) are cleared first.
    model.forget_rope_tables()
    rope = NaRotaryEmbedding3d(dim=120)
    rope.get_axial_freqs(2, 3, 4)
    rope.get_axial_freqs(3, 3, 4)
    cache = type(rope).get_axial_freqs
    assert cache.cache_info().currsize == 2  # pyright: ignore[reportFunctionMemberAccess]
    called: list[object] = []
    forget: Callable[[Any], None] = model._forget  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(model, "_forget", lambda vae: called.append(vae) or forget(vae))
    models = files.load(HELD)
    models.runner.dit.blocks.append(rope)  # a module of the DiT, as the blocks' own ropes are
    models.release()
    assert cache.cache_info().currsize == 0  # pyright: ignore[reportFunctionMemberAccess]
    assert called == [models.runner.vae]
    # The same values come back from the module's buffer, to the bit.
    again = rope.get_axial_freqs(2, 3, 4)
    assert torch.equal(again, rope.rope.get_axial_freqs(2, 3, 4))


def test_forget_drops_what_any_models_keep_and_a_dropped_job_s_models_leave_no_rope_alive(
    files: Files,
) -> None:
    # Models.release does it for the kept ones; a run's models that are not kept (loaded for the
    # job's plan, or for HELD and dropped) are dropped without a release, and the vendored RoPE's
    # lru_cache would keep the module alive, with its frequencies on the card (review H2).
    from seedvr2x.vendor.models.dit_7b.rope import NaRotaryEmbedding3d

    for plan in (ModelPlan(swapped=2, resident_encode=0, resident_decode=0), HELD):
        rope = NaRotaryEmbedding3d(dim=120)
        rope.get_axial_freqs(2, 3, 4)
        cache = type(rope).get_axial_freqs
        assert cache.cache_info().currsize > 0  # pyright: ignore[reportFunctionMemberAccess]
        models = files.load(plan)
        models.runner.dit.blocks.append(rope)
        alive = weakref.ref(rope)
        # Without forget, the cache holds the module past the models and its own name.
        del rope
        model.forget(models)
        assert cache.cache_info().currsize == 0  # pyright: ignore[reportFunctionMemberAccess]
        del models
        gc.collect()
        assert alive() is None


def test_forget_changes_no_parameter_and_can_be_called_again(files: Files) -> None:
    models = files.load(ModelPlan(swapped=2, resident_encode=0, resident_decode=0))
    before = {name: value.clone() for name, value in models.runner.dit.state_dict().items()}
    model.forget(models)
    model.forget(models)  # idempotent
    after = models.runner.dit.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_only_passes_held_itself_makes_models_kept_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # load_models took a plan equal to HELD by value for HELD (review L7): a job's plan of swapped
    # 0, nothing resident and the VAE away is a job's, its models are placed by no one.
    from dataclasses import replace

    from test_weights import as_dtype, write

    monkeypatch.setattr(model, "_load_weights", lambda *args, **more: {})
    write(tmp_path / "dit.safetensors", as_dtype("7b", "F16"))
    write(tmp_path / "vae.safetensors", as_dtype("vae", "F16"))
    paths = (tmp_path / "dit.safetensors", tmp_path / "vae.safetensors", CPU)
    equal = replace(HELD)
    assert equal == HELD and equal is not HELD
    job = model.load_models(*paths, None, equal)
    assert job.parts is None and job.model_plan == equal
    kept = model.load_models(*paths, None, HELD)
    assert kept.parts == {}


def test_a_held_load_says_what_it_is_not_that_no_blocks_are_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """HELD's swapped is 0, yet every weight sits in its pinned copy and each run's plan places
    the blocks: "none of the DiT's blocks swapped" was wrong for it. A job's plan equal to HELD by
    value (swapped 0 too) is a job's, and its line is the other loads' as before."""
    from dataclasses import replace

    files = Files(tmp_path, Sim("random-1"), monkeypatch)
    caplog.set_level("INFO", logger=model.logger.name)
    parts = ["blocks.0", "blocks.1"]
    prefix = "host memory pinned for the weights that leave the GPU: 2 of the DiT's blocks; "
    for plan, said in (
        (HELD, "every weight in its pinned copy, placed by each run's plan"),
        (replace(HELD), "none of the DiT's blocks swapped"),
    ):
        models = files.load(plan)
        assert models.placement is not None and models.placement.host_bytes > 0
        caplog.clear()
        model._say_pinned(models, parts)  # pyright: ignore[reportPrivateUsage]
        (line,) = caplog.messages
        assert line.endswith(prefix + said), line
        assert "none of the DiT's blocks swapped" not in line or plan is not HELD


class Card:
    """A stand-in for the device where the CPU is the only one: model._to_device and _to_host
    copy, as a copy to a device and back does, and the card's tensors are those _to_device made
    and are still alive."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tensors: dict[int, weakref.ref[Tensor]] = {}
        monkeypatch.setattr(model, "_to_device", self.to_device)
        monkeypatch.setattr(model, "_to_host", self.to_host)

    def to_device(self, tensor: Tensor, device: torch.device) -> Tensor:
        copy = tensor.clone()
        self.tensors[copy.data_ptr()] = weakref.ref(copy)
        return copy

    def to_host(self, tensor: Tensor) -> Tensor:
        return tensor.clone()

    def holds(self, tensor: Tensor) -> bool:
        found = self.tensors.get(tensor.data_ptr())
        return found is not None and found() is not None

    def put(self, models: model.Models) -> None:
        """Everything the load put on the device, there: the buffers of both modules and the text
        embeddings (a load puts them there; the parts are the placement's)."""
        for module in (models.runner.dit, models.runner.vae):
            for each in module.modules():
                for name, buffer in each._buffers.items():  # pyright: ignore[reportPrivateUsage]
                    if buffer is not None:
                        each._buffers[name] = self.to_device(buffer, CPU)  # pyright: ignore[reportPrivateUsage]
        models.text_pos = self.to_device(models.text_pos, CPU)
        models.text_neg = self.to_device(models.text_neg, CPU)


def strays(models: model.Models, card: Card | None = None) -> list[str]:
    """What of the models is on a device other than the CPU, or on the stand-in card, or kept by
    a cache: the name of every tensor reachable from the models that is, parameters (but for the
    pinned copies, which are the CPU's), buffers, the text embeddings, and the RoPE cache."""
    found: list[str] = []

    def check(name: str, tensor: Tensor) -> None:
        if tensor.device.type != "cpu" or (card is not None and card.holds(tensor)):
            found.append(name)

    for name, tensor in (("text_pos", models.text_pos), ("text_neg", models.text_neg)):
        check(name, tensor)
    for label, module in (("dit", models.runner.dit), ("vae", models.runner.vae)):
        for key, parameter in module.named_parameters():
            check(f"{label}.{key}", parameter.data)
        for key, buffer in module.named_buffers():
            check(f"{label}.{key}", buffer)
        for each in module.modules():
            cached = getattr(getattr(type(each), "get_axial_freqs", None), "cache_info", None)
            if cached is not None and cached().currsize:
                found.append(f"{label}: RoPE cache holds {cached().currsize} shapes")
    return found


def test_release_leaves_nothing_of_the_models_on_the_device_and_place_brings_it_back(
    files: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The buffers of both modules and the text embeddings go to system memory at release and
    # come back at place, exactly: dtype, shape, values; the bits of a pass the same through a
    # cycle, whether a pass ran before the release or not.
    card = Card(monkeypatch)
    models = files.load(HELD)
    card.put(models)
    assert len(strays(models, card)) >= 4, "the load's buffers and embeddings are on the card"
    before = {key: value.clone() for key, value in models.runner.dit.named_buffers()}
    text = (models.text_pos.clone(), models.text_neg.clone())
    plan = ModelPlan(swapped=2, resident_encode=0, resident_decode=0, vae_in_dit=False)
    expected = run(toy(BLOCKS, files.sim), files.sim)
    sim = files.sim
    for ran in (True, False, True):
        models.place(plan)
        # Placed: the buffers and embeddings are on the card again, the parts not yet.
        assert models.text_pos is not None and card.holds(models.text_pos)
        assert all(card.holds(b) for b in models.runner.dit.buffers())
        if ran:
            models.enter("dit")
            assert same(run(models.runner.dit, sim), expected)
            models.enter("decode")
        models.release()
        assert strays(models, card) == [], ran
        assert sim.live() == 0
    models.place(plan)
    models.enter("dit")
    assert same(run(models.runner.dit, sim), expected)
    models.release()
    models.release()
    assert strays(models, card) == []
    for key, value in models.runner.dit.named_buffers():
        assert value.dtype == before[key].dtype and torch.equal(value, before[key]), key
    for now, then in zip((models.text_pos, models.text_neg), text, strict=True):
        assert now.dtype == then.dtype == torch.bfloat16 and torch.equal(now, then)


def test_the_helper_finds_what_stays_on_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # The check is not vacuous: a cached RoPE shape, a buffer left on the card and an embedding
    # left there are each found.
    from seedvr2x.vendor.models.dit_7b.rope import NaRotaryEmbedding3d

    card = Card(monkeypatch)
    rope = NaRotaryEmbedding3d(dim=120)
    rope.get_axial_freqs(2, 3, 4)
    holder = torch.nn.ModuleList([rope])
    runner = SimpleNamespace(dit=holder, vae=torch.nn.Linear(2, 2))
    text = torch.zeros(2)
    models = model.Models(runner, text, text.clone(), "none", CPU)
    assert any("RoPE cache" in each for each in strays(models, card))
    type(rope).get_axial_freqs.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]
    assert strays(models, card) == []
    models.text_pos = card.to_device(text, CPU)
    rope.rope.dummy = card.to_device(torch.tensor(0), CPU)
    found = strays(models, card)
    assert "text_pos" in found and "dit.0.rope.dummy" in found
    assert torch.device("meta").type != "cpu"
    models.text_neg = torch.zeros(2, device="meta")
    assert "text_neg" in strays(models)
