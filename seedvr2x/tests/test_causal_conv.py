"""A causal convolution cut in pieces under the VAE's memory limit, on the CPU
(vendor/models/video_vae_v3/modules/causal_inflation_lib.py, memory_limit_conv): its pieces,
written into one output allocated before the first of them is computed where that output is
large, are numz's concatenation of them, bit for bit; and every slice of a VAE pass takes one
path, judged by the output a slice of four frames gives."""

import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import Conv3d

from seedvr2x.vendor.common.config import load_config
from seedvr2x.vendor.common.half_precision_fixes import safe_pad_operation
from seedvr2x.vendor.models.video_vae_v3.modules import attn_video_vae, causal_inflation_lib
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import (
    InflatedCausalConv3d,
    ignore_padding,
)
from seedvr2x.vendor.models.video_vae_v3.modules.context_parallel_lib import get_cache_size
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

NO_PADDING = (0, 0, 0, 0, 0, 0)


@pytest.fixture(autouse=True)
def every_output_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests' outputs are a few kilobytes: the size from which an output is allocated whole
    is brought down to nothing, but where a test sets its own."""
    monkeypatch.setattr(causal_inflation_lib, "ONE_OUTPUT_FROM", 0)


def concatenated(
    conv: Any,
    x: Any,
    *,
    split_dim: int = 3,
    padding: tuple[int, ...] = NO_PADDING,
    prev_cache: Any = None,
) -> Tensor:
    """numz's memory_limit_conv under a memory limit, to the letter but for its retry_on_oom
    (src/models/video_vae_v3/modules/causal_inflation_lib.py:131-218 at numz 4490bd1): the
    pieces' outputs kept in a list and concatenated once all are made. What the vendored one,
    which writes them into one output, is held to."""
    shape = torch.tensor(x.size())
    if prev_cache is not None:
        shape[split_dim - 1] += prev_cache.size(split_dim - 1)
    shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
    memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
    if memory_occupy < conv.memory_limit or split_dim == x.ndim:
        x_concat = x
        if prev_cache is not None:
            x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)
        padded = safe_pad_operation(x_concat, padding, mode="constant", value=0.0)
        with ignore_padding(conv):
            return Conv3d.forward(conv, padded)

    num_splits = math.ceil(memory_occupy / conv.memory_limit)
    size_per_split = x.size(split_dim) // num_splits
    split_sizes = [size_per_split] * (num_splits - 1)
    split_sizes += [x.size(split_dim) - sum(split_sizes)]

    x = list(x.split(split_sizes, dim=split_dim))
    if prev_cache is not None:
        prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
    cache = None
    for idx in range(len(x)):
        if prev_cache is not None:
            x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

        lpad_dim = (x[idx].ndim - split_dim - 1) * 2
        rpad_dim = lpad_dim + 1
        padding = list(padding)  # type: ignore[assignment]
        padding[lpad_dim] = conv.padding[split_dim - 2] if idx == 0 else 0  # type: ignore[index]
        padding[rpad_dim] = conv.padding[split_dim - 2] if idx == len(x) - 1 else 0  # type: ignore[index]
        pad_len = padding[lpad_dim] + padding[rpad_dim]
        padding = tuple(padding)

        next_cache = None
        cache_len = cache.size(split_dim) if cache is not None else 0
        next_catch_size = get_cache_size(
            conv_module=conv,
            input_len=x[idx].size(split_dim) + cache_len,
            pad_len=pad_len,
            dim=split_dim - 2,
        )
        if next_catch_size != 0:
            assert next_catch_size <= x[idx].size(split_dim)
            next_cache = x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)

        x[idx] = concatenated(
            conv, x[idx], split_dim=split_dim + 1, padding=padding, prev_cache=cache
        )
        cache = next_cache

    return torch.cat(x, split_dim)


def conv_of(
    channels: tuple[int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    limit: float,
    dtype: torch.dtype,
) -> Any:
    """A causal convolution with random weights, under a memory limit of `limit` GiB."""
    torch.manual_seed(0)
    conv: Any = InflatedCausalConv3d(
        *channels, kernel, stride=stride, padding=padding, inflation_mode="tail"
    )
    conv.eval().requires_grad_(False).to(dtype)
    conv.set_memory_limit(limit)
    return conv


def noise(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


# The VAE's kinds of causal convolution, small (attn_video_vae.py): 3x3x3 in a block, and a
# downsampling one's strides, over space alone and over time too; one that looks along time only
# and one with no spatial padding, for the splits' arithmetic.
CONVS = [
    ((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
    ((4, 6), (3, 3, 3), (2, 2, 2), (1, 1, 1)),
    ((4, 4), (3, 3, 3), (1, 2, 2), (1, 0, 0)),
    ((4, 4), (1, 3, 3), (1, 2, 2), (0, 1, 1)),
    ((5, 7), (3, 1, 1), (1, 1, 1), (1, 0, 0)),
]
# In GiB, for inputs of a few kilobytes: two that cut along the rows and then along the columns,
# two that cut along the rows alone, in fewer pieces, and one that cuts nothing, the path numz's
# and the vendored code share.
LIMITS = [2e-6, 4e-6, 1e-5, 1.5e-5, 2e-5, 3e-5, 6e-5, 1.0]
SHAPES = [(1, 5, 17, 23), (1, 4, 32, 40), (2, 4, 9, 31)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("channels", "kernel", "stride", "padding"), CONVS)
def test_pieces_written_into_one_output_are_numz_s_concatenation(
    channels: tuple[int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs: list[int] = []
    whole_output = InflatedCausalConv3d.whole_output

    def counted(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor | None:
        outputs.append(1)
        return whole_output(self, like, padded, *path)

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", counted)
    once = twice = 0
    for limit in LIMITS:
        for batch, *rest in SHAPES:
            conv = conv_of(channels, kernel, stride, padding, limit, dtype)
            shape = (batch, channels[0], *rest)
            # As slicing_forward calls it: the padding of the two spatial sides, and a cache of
            # earlier frames ahead of the input where the kernel looks back.
            around = tuple(side for side in reversed(conv.padding) for _ in range(2))
            cache_size = conv.kernel_size[0] - conv.stride[0]
            caches: list[Tensor | None] = [None]
            if cache_size:
                caches.append(noise((batch, channels[0], cache_size, *rest[1:]), 3, dtype))
            for cache in caches:
                x = noise(shape, 1, dtype)
                try:
                    expected = concatenated(conv, x.clone(), padding=around, prev_cache=cache)
                except (AssertionError, RuntimeError):
                    # numz's own refusal: a piece too thin for the kernel, or frames too few
                    # for it without a cache. Nothing to hold the vendored one to.
                    continue
                outputs.clear()
                got = conv.memory_limit_conv(x.clone(), padding=around, prev_cache=cache)
                assert got.dtype == expected.dtype and got.shape == expected.shape
                assert torch.equal(got, expected)
                assert got.is_contiguous() == expected.is_contiguous()
                once += len(outputs) == 1
                twice += len(outputs) > 1
    # The test cuts convolutions, along the rows and along both sides, or it holds nothing.
    assert once >= 2 and twice >= 2, (once, twice)


def test_a_whole_pass_of_slices_is_numz_s() -> None:
    # The causal memory across slices, as the VAE runs a convolution (forward): the first slice
    # initialises it and the next ones take it, each cut in pieces.
    conv = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 1e-5, torch.bfloat16)
    reference = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 1e-5, torch.bfloat16)
    # numz's own, which knows nothing of the output's path (head, whole: whole_output).
    reference.memory_limit_conv = lambda x, head, **named: concatenated(reference, x, **named)
    states = (MemoryState.INITIALIZING, MemoryState.ACTIVE, MemoryState.ACTIVE)
    for seed, state in enumerate(states):
        x = noise((1, 3, 5 if seed == 0 else 4, 17, 23), seed, torch.bfloat16)
        assert torch.equal(
            conv(x.clone(), memory_state=state), reference(x.clone(), memory_state=state)
        )


def test_the_output_is_allocated_before_its_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    # What the change is for: the one large request comes before the pieces' own, not after
    # them all, when the pieces hold as much as it asks.
    conv = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 2e-5, torch.float32)
    events: list[str] = []
    whole_output = InflatedCausalConv3d.whole_output
    forward = Conv3d.forward

    def allocated(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor | None:
        events.append("output")
        return whole_output(self, like, padded, *path)

    def convolved(self: Any, x: Tensor) -> Tensor:
        events.append("piece")
        return forward(self, x)

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", allocated)
    monkeypatch.setattr(Conv3d, "forward", convolved)
    x = noise((1, 3, 5, 17, 23), 1, torch.float32)
    output = conv.memory_limit_conv(x, padding=(1, 1, 1, 1, 0, 0))
    # One output for the cut along the rows, then its pieces, more than one; no piece is cut
    # again along the columns under this limit.
    assert events[0] == "output" and events.count("output") == 1
    assert events.count("piece") > 1
    assert output.shape == (1, 8, 3, 17, 23)


def test_a_piece_that_does_not_fit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # An output of another shape than the pieces fill is an error, never frames partly unwritten.
    conv = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 2e-5, torch.float32)
    whole_output = InflatedCausalConv3d.whole_output

    def longer(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor:
        made = whole_output(self, like, padded, *path)
        assert made is not None
        return made.new_zeros((*made.shape[:3], made.shape[3] + 1, made.shape[4]))

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", longer)
    x = noise((1, 3, 5, 17, 23), 1, torch.float32)
    with pytest.raises(RuntimeError, match="17 of 18 written along dimension 3"):
        conv.memory_limit_conv(x, padding=(1, 1, 1, 1, 0, 0))

    def wider(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor:
        made = whole_output(self, like, padded, *path)
        assert made is not None
        return made.new_zeros((*made.shape[:4], made.shape[4] + 1))

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", wider)
    with pytest.raises(RuntimeError, match=r"a piece of .* at 0 along dimension 3"):
        conv.memory_limit_conv(x, padding=(1, 1, 1, 1, 0, 0))


def test_a_small_output_is_concatenated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under the size the module sets, numz's own path: no output allocated ahead, the pieces
    # concatenated; from it on, the one output. The same values either way.
    conv = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 2e-5, torch.float32)
    x = noise((1, 3, 5, 17, 23), 1, torch.float32)
    around = (1, 1, 1, 1, 0, 0)
    expected = concatenated(conv, x.clone(), padding=around)
    size = expected.numel() * expected.element_size()
    made: list[bool] = []
    whole_output = InflatedCausalConv3d.whole_output

    def noted(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor | None:
        output = whole_output(self, like, padded, *path)
        made.append(output is not None)
        return output

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", noted)
    for floor, at_once in ((size + 1, False), (size, True), (size - 1, True)):
        monkeypatch.setattr(causal_inflation_lib, "ONE_OUTPUT_FROM", floor)
        made.clear()
        assert torch.equal(conv.memory_limit_conv(x.clone(), padding=around), expected)
        assert made == [at_once]


def test_a_pass_s_first_slice_takes_its_other_slices_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pass's first slice holds one frame more than its others at every convolution, five frames
    # here where they hold four: judged by its own output, it took the one output where they
    # didn't (design's review, 2026-10-16). Judged by the output a slice of four frames gives,
    # every slice of the pass takes one path: under a size a byte over that output, none; at it,
    # all; numz's frames either way.
    conv = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 2e-5, torch.float32)
    reference = conv_of((3, 8), (3, 3, 3), (1, 1, 1), (1, 1, 1), 2e-5, torch.float32)
    reference.memory_limit_conv = lambda x, head, **named: concatenated(reference, x, **named)
    taken: list[bool] = []
    whole_output = InflatedCausalConv3d.whole_output

    def noted(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor | None:
        output = whole_output(self, like, padded, *path)
        if path[1] is None:  # the convolution's own call, not a piece cut again
            taken.append(output is not None)
        return output

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", noted)
    states = (MemoryState.INITIALIZING, MemoryState.ACTIVE, MemoryState.ACTIVE)
    slices = [noise((1, 3, 4 + (seed == 0), 17, 23), seed, torch.float32) for seed in range(3)]
    # The output of a slice of four frames, in bytes: 8 channels of 17 x 23, float32; the first
    # slice's is a quarter more.
    four = 8 * 4 * 17 * 23 * 4
    for floor, path in ((four + 1, False), (four, True)):
        monkeypatch.setattr(causal_inflation_lib, "ONE_OUTPUT_FROM", floor)
        taken.clear()
        for x, state in zip(slices, states, strict=True):
            got = conv(x.clone(), memory_state=state)
            assert torch.equal(got, reference(x.clone(), memory_state=state))
        # Each slice judged once, at its convolution's own call.
        assert taken == [path] * 3
        # A pass in one slice, its memory disabled, of five frames: judged as a slice of four.
        taken.clear()
        x = noise((1, 3, 5, 17, 23), 3, torch.float32)
        got = conv(x.clone(), memory_state=MemoryState.DISABLED)
        assert torch.equal(got, reference(x.clone(), memory_state=MemoryState.DISABLED))
        assert taken == [path]
    # A pass of one frame has no slice of four: concatenated, whatever its output (cut in pieces
    # under a limit of its own).
    for one in (conv, reference):
        one.set_memory_limit(1e-5)
    monkeypatch.setattr(causal_inflation_lib, "ONE_OUTPUT_FROM", 1)
    taken.clear()
    x = noise((1, 3, 1, 17, 23), 4, torch.float32)
    got = conv(x.clone(), memory_state=MemoryState.DISABLED)
    assert torch.equal(got, reference(x.clone(), memory_state=MemoryState.DISABLED))
    assert taken == [False]


def narrow_vae() -> Any:
    """The VAE as shipped (s8_c16_t4_inflation_sd3.yaml: its convolutions, kernels, strides,
    causal memory and slices of four frames), narrowed to 32 channels a block, on the CPU in
    float32, a convolution's input cut in pieces from 0.5 MiB where the configuration cuts it
    from 0.5 GiB."""
    vendor = Path(attn_video_vae.__file__).parents[3]
    config = cast(
        Any,
        OmegaConf.to_container(
            load_config(str(vendor / "models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"))
        ),
    )
    config["block_out_channels"] = [32] * 4
    torch.manual_seed(0)
    vae: Any = attn_video_vae.VideoAutoencoderKLWrapper(**config, freeze_encoder=False)
    vae.eval().requires_grad_(False)
    vae.set_causal_slicing(split_size=4, memory_device="same")
    vae.set_memory_limit(conv_max_mem=0.5 / 1024, norm_max_mem=None)
    return vae


def test_every_slice_of_a_vae_pass_takes_one_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The VAE's own convolutions (narrow_vae): an encode of 13 frames in slices of 5, 4 and 4, and
    # a decode of 4 latents in slices of 2, 1 and 1, as the runtime feeds them (runtime/model.py,
    # encode_sliced and decode_sliced), 64 x 64 px. At every convolution the first slice holds one
    # frame more than the others, and the pieces along the rows are cut again along the columns
    # where they still pass the limit. For each size a convolution's first slice or its slices of
    # four give, every call a convolution makes in a pass, every slice and every piece cut again,
    # takes one path, and the frames are numz's.
    vae = narrow_vae()
    names = {module: name for name, module in vae.named_modules()}
    frames = noise((1, 3, 13, 64, 64), 1, torch.float32)
    latents = noise((1, 16, 4, 8, 8), 2, torch.float32)
    states = (MemoryState.INITIALIZING, MemoryState.ACTIVE, MemoryState.ACTIVE)
    # Each call: its convolution, its slice, the path it was given, its output's bytes and frames.
    calls: list[tuple[str, int, Any, int, int]] = []
    whole_output = InflatedCausalConv3d.whole_output
    current = 0

    def noted(self: Any, like: Tensor, padded: Tensor, *path: Any) -> Tensor | None:
        output = whole_output(self, like, padded, *path)
        size = 0 if output is None else output.numel() * output.element_size()
        count = 0 if output is None else output.size(2)
        calls.append((names[self], current, path[1], size, count))
        return output

    def passes(floor: float) -> list[Tensor]:
        nonlocal current
        monkeypatch.setattr(causal_inflation_lib, "ONE_OUTPUT_FROM", floor)
        calls.clear()
        made = []
        encode = (frames[:, :, :5], frames[:, :, 5:9], frames[:, :, 9:])
        for index, (x, state) in enumerate(zip(encode, states, strict=True)):
            current = index
            made.append(vae._encode(x.clone(), memory_state=state))
        decode = (latents[:, :, :2], latents[:, :, 2:3], latents[:, :, 3:])
        for index, (z, state) in enumerate(zip(decode, states, strict=True)):
            current = index
            made.append(vae._decode(z.clone(), memory_state=state))
        return made

    monkeypatch.setattr(InflatedCausalConv3d, "whole_output", noted)
    numz = passes(math.inf)
    # Every output allocated whole: the sizes the convolutions' own calls give, by slice, and
    # their frames, the first slice's one more than the others' at every convolution.
    passes(0)
    sizes: dict[str, dict[int, set[int]]] = {}
    frames_of: dict[str, dict[int, set[int]]] = {}
    for name, index, whole, size, count in calls:
        if whole is None:
            sizes.setdefault(name, {}).setdefault(index, set()).add(size)
            frames_of.setdefault(name, {}).setdefault(index, set()).add(count)
    assert any(len(by_slice) == 3 for by_slice in sizes.values())
    assert any(path is True for _, _, path, _, _ in calls)  # pieces cut again
    for name, by_slice in frames_of.items():
        if len(by_slice) == 3:
            (first,), (second,), (third,) = by_slice[0], by_slice[1], by_slice[2]
            assert first == second + 1 == third + 1, (name, by_slice)
    floors = {size for by_slice in sizes.values() for found in by_slice.values() for size in found}
    split = 0  # sizes at which judging call by call gave a first slice alone the one output
    for floor in sorted(floors):
        made = passes(floor)
        assert all(torch.equal(a, b) for a, b in zip(made, numz, strict=True))
        paths: dict[str, set[bool]] = {}
        for name, _, _, size, _ in calls:
            paths.setdefault(name, set()).add(size > 0)
        assert all(len(found) == 1 for found in paths.values()), (floor, paths)
        split += any(
            max(by_slice[0]) >= floor > max(by_slice.get(1, {0}))
            for by_slice in sizes.values()
            if len(by_slice) == 3
        )
    assert split >= 1, split


def test_the_size_an_output_is_allocated_whole_from(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2.5 GiB as shipped, the fixture aside.
    monkeypatch.undo()
    assert causal_inflation_lib.ONE_OUTPUT_FROM == round(2.5 * 1024**3)
