"""The decoder's tail a frame at a time (Decoder3D.tail_frames, model.tail_frames) on the CPU,
against the plain decoder, bit for bit: cpu_models' VAE (the real configs narrowed, random weights)
in float32 and in bfloat16, the pipeline's dtype (test_still.pipeline_models), a latent of 3 x 5
whose decode is 24 x 40, cut in slices as model.latent_slices cuts it (2 latents with 5 frames, then
1 latent with 4 frames at a time), so that the causal convolutions' memory is carried from slice to
slice and, inside the tail, from frame to frame. The cards' sizes (512 px and up) are measured on
the GPU (research/docs/small-cards.md); these hold the code's arithmetic."""

import copy
from collections.abc import Iterator
from typing import Any, cast

import pytest
import torch
from stand_ins import cpu_models  # noqa: F401
from test_still import pipeline_models
from torch import Tensor

from seedvr2x import model
from seedvr2x.vendor.models.video_vae_v3.modules.attn_video_vae import Decoder3D
from seedvr2x.vendor.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from seedvr2x.vendor.models.video_vae_v3.modules.types import MemoryState

DTYPES = [torch.float32, torch.bfloat16]
# Latents of a decode: 2 + 1 + 1 (frames 5, 4, 4), 2 alone (5 frames), 1 alone (1 frame: the
# tail's single-frame branch), 3 + 1 slices with a last of 1.
LATENTS = [4, 2, 1, 6]


@pytest.fixture
def models(
    cpu_models: model.Models,  # noqa: F811
    request: pytest.FixtureRequest,
) -> Iterator[model.Models]:
    """cpu_models' VAE in the requested dtype, in models of its own, with oneDNN's convolutions
    off for the test in both dtypes. oneDNN picks its algorithm by the tensor's shape and by the
    threads it runs on, so that a frame's convolution and the slice's can differ by an ulp: in
    float32 by 8.2e-6 at most here, at 4 and 6 latents (2026-11-06); in bfloat16 at the shipped
    widths on 8 threads and not on 16 (2026-11-07). What these tests hold is the algorithm's, the
    causal memory carried from frame to frame giving the slices' bits, which PyTorch's own
    kernels gave on 4, 8 and 16 threads at the narrow and the shipped widths (2026-11-08);
    oneDNN is the CPU's, and the pipeline decodes on the card."""
    dtype: torch.dtype = request.param
    made = pipeline_models(cpu_models)
    runner: Any = made.runner
    runner.vae = runner.vae.to(dtype)
    # The switch alone, not mkldnn.flags(), which also sets TF32's precision and warns of it.
    enabled = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        yield made
    finally:
        torch.backends.mkldnn.enabled = enabled


def latent_of(models: model.Models, latents: int) -> Tensor:
    """A latent (T', 3, 5, 16) in the VAE's dtype."""
    generator = torch.Generator().manual_seed(latents)
    dtype = next(cast(Any, models).runner.vae.parameters()).dtype
    return torch.randn(latents, 3, 5, 16, generator=generator).to(dtype)


def decode(models: model.Models, latent: Tensor) -> Tensor:
    """The frames of model.decode_sliced, whole (C, t, H, W)."""
    return torch.cat(list(model.decode_sliced(models, latent)), dim=1)


def memories(vae: Any) -> list[Tensor | None]:
    """Every causal convolution's memory, in module order."""
    return [m.memory for m in vae.modules() if isinstance(m, InflatedCausalConv3d)]


def decoders(vae: Any) -> list[Decoder3D]:
    return [m for m in vae.modules() if isinstance(m, Decoder3D)]


def test_it_is_off_by_default(cpu_models: model.Models) -> None:  # noqa: F811
    vae: Any = cast(Any, cpu_models).runner.vae
    assert decoders(vae)
    assert [d.tail_frames for d in decoders(vae)] == [False]
    assert Decoder3D.tail_frames is False


@pytest.mark.parametrize("models", DTYPES, indirect=True)
@pytest.mark.parametrize("latents", LATENTS)
def test_the_frames_are_the_plain_decodes_bit_for_bit(models: model.Models, latents: int) -> None:
    latent = latent_of(models, latents)
    plain = decode(models, latent)
    model.tail_frames(cast(Any, models).runner.vae, True)
    tail = decode(models, latent)
    assert tail.dtype == plain.dtype
    assert tail.shape == plain.shape
    assert torch.equal(tail, plain)


@pytest.mark.parametrize("models", DTYPES, indirect=True)
def test_the_memory_left_after_each_slice_is_the_plain_decoders(models: model.Models) -> None:
    """Slice by slice, as slicing_decode steps (INITIALIZING then ACTIVE): every causal
    convolution's memory equal to the bit after each, those of the tail's too, which saw its
    frames one at a time."""
    vae: Any = cast(Any, models).runner.vae
    other = copy.deepcopy(vae)
    model.tail_frames(other, True)
    assert [d.tail_frames for d in decoders(other)] == [True]
    assert [d.tail_frames for d in decoders(vae)] == [False]
    latent = latent_of(models, 4)
    z = model.decoder_input(models, latent)
    sizes = model.latent_slices(models, 4)
    assert sizes == [2, 1, 1]
    start = 0
    with torch.no_grad():
        for number, size in enumerate(sizes):
            piece = z[:, :, start : start + size]
            start += size
            state = MemoryState.INITIALIZING if number == 0 else MemoryState.ACTIVE
            expected = vae._decode(piece, memory_state=state)  # pyright: ignore
            got = other._decode(piece, memory_state=state)  # pyright: ignore
            assert torch.equal(got, expected)
            held, kept = memories(vae), memories(other)
            assert len(held) == len(kept) > 0
            for a, b in zip(held, kept, strict=True):
                assert (a is None) == (b is None)
                if a is not None and b is not None:
                    assert a.dtype == b.dtype
                    assert torch.equal(a, b)
    model._forget(vae)  # pyright: ignore
    model._forget(other)  # pyright: ignore
    assert memories(vae) == memories(other) == [None] * len(memories(vae))


@pytest.mark.parametrize("models", DTYPES, indirect=True)
def test_the_switch_on_and_off_again_gives_todays_bits(models: model.Models) -> None:
    vae: Any = cast(Any, models).runner.vae
    latent = latent_of(models, 4)
    first = decode(models, latent)
    model.tail_frames(vae, True)
    assert decoders(vae)[0].tail_frames is True
    on = decode(models, latent)
    model.tail_frames(vae, False)
    assert decoders(vae)[0].tail_frames is False
    assert torch.equal(decode(models, latent), first)
    assert torch.equal(on, first)


def test_a_vae_with_no_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="no Decoder3D"):
        model.tail_frames(torch.nn.Linear(1, 1), True)
