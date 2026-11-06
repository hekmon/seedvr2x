"""The 7B DiT's lean attention (DESIGN.md, Allocator, Small cards), bit for bit against the plain.

The plain reference is the vendored code as it stood on main at 2ba0857, before the lean changes
(nablocks/mmsr_block.py, attention.py, mlp.py), read from the repository's git objects with
tools/vendor.py's `git_show` and loaded beside the vendored modules. It is not numz's own file:
numz's mmsr_block.py imports sequence parallelism (`gather_seq_scatter_heads_qkv`) that the
vendored layer removed, so its forward does not run here; main's differs from numz's by that and
by attention.py's two backends only. Skipped where the commit is not in the repository.

The sizes are stand-ins: few heads, a small head dimension, a latent whose 720p-based windows
(`window.py`) are several and of unequal lengths, and more tokens than 4,096, so that the chunked
norm and RoPE run two whole chunks and a partial one. The pipeline's path is the chunked one
(`model.py` sets `compute_dtype` to bfloat16 on every attention module, the 7B's config has RoPE);
the branch without a compute dtype is held too, and both under autocast to bfloat16 as the
pipeline runs its DiT."""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from seedvr2x.vendor.common.cache import Cache
from seedvr2x.vendor.models.dit_7b import attention, mlp
from seedvr2x.vendor.models.dit_7b.mm import MMModule
from seedvr2x.vendor.models.dit_7b.nablocks import mmsr_block
from seedvr2x.vendor.models.dit_7b.normalization import get_norm_layer
from seedvr2x.vendor.models.dit_7b.rope import NaRotaryEmbedding3d

PROJECT = Path(__file__).resolve().parents[1]
MAIN = "2ba0857"
DIT = "core/seedvr2x/vendor/models/dit_7b"
PACKAGE = "seedvr2x.vendor.models.dit_7b"

HEADS = 2
HEAD_DIM = 24  # RoPE's dim is half of it, split over the three axes
DIM = HEADS * HEAD_DIM
TXT_TOKENS = 7
VID_SHAPE = (3, 50, 70)  # 10,500 tokens: 4,096 + 4,096 + 2,308 in the chunks
TXT_SHAPE = (TXT_TOKENS,)


def load_plain(name: str, path: str, package: str) -> ModuleType:
    """The vendored module at path as main had it, loaded as `package.name`."""
    vendor_tool = sys.modules.get("vendor_tool_under_test")
    if vendor_tool is None:
        spec = importlib.util.spec_from_file_location(
            "vendor_tool_under_test", PROJECT / "tools" / "vendor.py"
        )
        assert spec is not None and spec.loader is not None
        vendor_tool = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = vendor_tool  # dataclasses look their module up there
        spec.loader.exec_module(vendor_tool)
    source = vendor_tool.git_show(PROJECT.parent, MAIN, f"{DIT}/{path}")
    if source is None:
        pytest.skip(f"{MAIN} is not in this repository's history")
    spec = importlib.util.spec_from_loader(f"{package}.{name}", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package  # the relative imports resolve in the vendored package
    sys.modules[spec.name] = module
    exec(compile(source, f"{MAIN}:{DIT}/{path}", "exec"), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def rope_cache_left_empty() -> Iterator[None]:
    # The vendored RoPE memoises its frequencies on the class (up to 128 shapes): a test of
    # test_held.py counts that cache from empty.
    yield
    NaRotaryEmbedding3d.get_axial_freqs.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]


@pytest.fixture(scope="module")
def plain_attention() -> ModuleType:
    return load_plain("_plain_attention", "attention.py", PACKAGE)


@pytest.fixture(scope="module")
def plain_mmsr_block() -> ModuleType:
    return load_plain("_plain_mmsr_block", "nablocks/mmsr_block.py", f"{PACKAGE}.nablocks")


@pytest.fixture(scope="module")
def plain_mlp() -> ModuleType:
    return load_plain("_plain_mlp", "mlp.py", PACKAGE)


def swin(
    module: ModuleType,
    attn: ModuleType,
    *,
    method: str,
    shared: bool,
    mode: str,
    compute: torch.dtype | None,
    norm_dtype: torch.dtype,
) -> Any:
    """A NaSwinAttention of `module`, its attention of `attn`, with the seeded weights."""
    torch.manual_seed(5)
    block = module.NaSwinAttention(
        vid_dim=DIM,
        txt_dim=DIM,
        heads=HEADS,
        head_dim=HEAD_DIM,
        qk_bias=False,
        qk_rope=True,
        qk_norm=get_norm_layer("fusedrms"),
        qk_norm_eps=1e-5,
        window=(4, 3, 3),
        window_method=method,
        shared_qkv=shared,
        attention_mode=mode,
    )
    block.attn = attn.FlashAttentionVarlen(attention_mode=mode, compute_dtype=compute)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=_generator(parameter)))
    for norm in (block.norm_q, block.norm_k):
        norm.to(norm_dtype)
    return block.eval()


def _generator(parameter: torch.Tensor) -> torch.Generator:
    # One draw per parameter by its size, the same in the lean and the plain block.
    return torch.Generator().manual_seed(parameter.numel())


def inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(9)
    vid_tokens = VID_SHAPE[0] * VID_SHAPE[1] * VID_SHAPE[2]
    vid = torch.randn(vid_tokens, DIM, generator=generator).to(dtype)
    txt = torch.randn(TXT_TOKENS, DIM, generator=generator).to(dtype)
    return vid, txt, torch.tensor([VID_SHAPE]), torch.tensor([TXT_SHAPE])


def run(block: Any, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    vid, txt, vid_shape, txt_shape = inputs(dtype)
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        return block(vid, txt, vid_shape, txt_shape, Cache())


METHODS = ["720pwin_by_size_bysize", "720pswin_by_size_bysize"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("norm_dtype", [torch.float32, torch.bfloat16])
def test_lean_attention_equals_the_plain_one(
    method: str,
    norm_dtype: torch.dtype,
    plain_attention: ModuleType,
    plain_mmsr_block: ModuleType,
) -> None:
    # The pipeline's path: compute dtype bfloat16, RoPE on, the DiT under bfloat16 autocast. A
    # float32 norm weight promotes q and k to float32 (the cast then moves before the
    # concatenation); a bfloat16 one leaves them in bfloat16.
    kwargs: dict[str, Any] = dict(
        method=method, shared=False, mode="sdpa", compute=torch.bfloat16, norm_dtype=norm_dtype
    )
    lean = swin(mmsr_block, attention, **kwargs)
    plain = swin(plain_mmsr_block, plain_attention, **kwargs)
    assert lean.state_dict().keys() == plain.state_dict().keys()
    lean_vid, lean_txt = run(lean, torch.bfloat16)
    plain_vid, plain_txt = run(plain, torch.bfloat16)
    assert lean_vid.shape == plain_vid.shape and lean_txt.shape == plain_txt.shape
    assert torch.equal(lean_vid, plain_vid)
    assert torch.equal(lean_txt, plain_txt)
    assert bool(torch.isfinite(lean_vid).all() and torch.isfinite(lean_txt).all())


def test_the_windows_are_several_and_unequal_and_the_chunks_end_partial(
    plain_attention: ModuleType, plain_mmsr_block: ModuleType
) -> None:
    # The stand-in sizes do what the tests above need: several windows of unequal lengths and
    # a last chunk shorter than the others.
    seen: list[torch.Tensor] = []
    real = attention.pytorch_varlen_attention

    def spy(q: Any, k: Any, v: Any, cu_q: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        seen.append(cu_q.clone())
        return real(q, k, v, cu_q, *args, **kwargs)

    lean = swin(
        mmsr_block,
        attention,
        method=METHODS[0],
        shared=False,
        mode="sdpa",
        compute=torch.bfloat16,
        norm_dtype=torch.float32,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(attention, "pytorch_varlen_attention", spy)
        run(lean, torch.bfloat16)
    lengths = seen[0][1:] - seen[0][:-1]
    assert len(lengths) > 4 and len(set(lengths.tolist())) > 1
    tokens = VID_SHAPE[0] * VID_SHAPE[1] * VID_SHAPE[2]
    assert tokens // mmsr_block.LEAN_CHUNK >= 2 and tokens % mmsr_block.LEAN_CHUNK != 0


def test_lean_attention_equals_the_plain_one_with_shared_weights(
    plain_attention: ModuleType, plain_mmsr_block: ModuleType
) -> None:
    # The MM pair's shared-weights variant of the projections (the 7B's config has shared_qkv
    # False). The norms of q and k are never shared (MMWindowAttention builds them without
    # shared_weights): vid_or_all and txt_or_all of a shared pair are held apart below.
    kwargs: dict[str, Any] = dict(
        method=METHODS[1], shared=True, mode="sdpa", compute=torch.bfloat16,
        norm_dtype=torch.float32,
    )  # fmt: skip
    lean = swin(mmsr_block, attention, **kwargs)
    plain = swin(plain_mmsr_block, plain_attention, **kwargs)
    assert hasattr(lean.proj_qkv, "all") and not hasattr(lean.proj_qkv, "vid")
    for ours, theirs in zip(run(lean, torch.bfloat16), run(plain, torch.bfloat16), strict=True):
        assert torch.equal(ours, theirs)


def test_the_pairs_modules_as_forward_picks_them() -> None:
    separate = MMModule(torch.nn.Linear, 4, 4, shared_weights=False)
    shared = MMModule(torch.nn.Linear, 4, 4, shared_weights=True)
    assert separate.vid_or_all is separate.vid and separate.txt_or_all is separate.txt
    assert shared.vid_or_all is shared.all and shared.txt_or_all is shared.all


@pytest.mark.parametrize("autocast", [False, True])
def test_lean_attention_equals_the_plain_one_without_a_compute_dtype(
    autocast: bool, plain_attention: ModuleType, plain_mmsr_block: ModuleType
) -> None:
    # The branch of numz's order, reached where compute_dtype is None (no pipeline leaves it so;
    # the tests build the module alone), with and without bfloat16 autocast.
    kwargs: dict[str, Any] = dict(
        method=METHODS[0], shared=False, mode="sdpa", compute=None, norm_dtype=torch.float32
    )
    lean = swin(mmsr_block, attention, **kwargs)
    plain = swin(plain_mmsr_block, plain_attention, **kwargs)
    vid, txt, vid_shape, txt_shape = inputs(torch.float32)
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16, enabled=autocast):
        ours = lean(vid, txt, vid_shape, txt_shape, Cache())
        theirs = plain(vid, txt, vid_shape, txt_shape, Cache())
    for a, b in zip(ours, theirs, strict=True):
        assert torch.equal(a, b)


def varlen_inputs(
    lengths: list[int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    total = sum(lengths)
    q, k, v = (torch.randn(total, 3, 16, generator=generator).to(dtype) for _ in range(3))
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return q, k, v, cu


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[40], [37, 5, 80, 1, 22], [3, 3, 3]])
def test_varlen_attention_writes_the_same_outputs_into_one_tensor(
    lengths: list[int], dtype: torch.dtype, plain_attention: ModuleType
) -> None:
    q, k, v, cu = varlen_inputs(lengths, dtype)
    ours = attention.pytorch_varlen_attention(q, k, v, cu, cu)
    theirs = plain_attention.pytorch_varlen_attention(q, k, v, cu, cu)
    assert ours.shape == theirs.shape == q.shape and ours.dtype == theirs.dtype
    assert torch.equal(ours, theirs)


def test_varlen_attention_with_a_longer_key_dimension(plain_attention: ModuleType) -> None:
    # The output's last dimension is v's, as the concatenation of the windows gave it.
    q, k, _, cu = varlen_inputs([30, 12], torch.float32)
    v = torch.randn(42, 3, 8, generator=torch.Generator().manual_seed(4))
    ours = attention.pytorch_varlen_attention(q, k, v, cu, cu)
    theirs = plain_attention.pytorch_varlen_attention(q, k, v, cu, cu)
    assert ours.shape == (42, 3, 8)
    assert torch.equal(ours, theirs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_gelu_in_place_equals_the_plain_mlp(dtype: torch.dtype, plain_mlp: ModuleType) -> None:
    torch.manual_seed(1)
    ours, theirs = mlp.MLP(48, 4), plain_mlp.MLP(48, 4)
    theirs.load_state_dict(ours.state_dict())
    assert ours.act.approximate == "tanh"  # the in-place call is the tanh one
    x = torch.randn(300, 48, generator=torch.Generator().manual_seed(3)).to(dtype)
    ours, theirs = ours.to(dtype), theirs.to(dtype)
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))
    assert x.dtype == dtype  # the input itself is not the tensor made in place


def test_mlp_input_is_left_as_it_was(plain_mlp: ModuleType) -> None:
    # The GELU acts on proj_in's output, a tensor of its own, never on the caller's.
    block = mlp.MLP(16, 4)
    x = torch.randn(10, 16, generator=torch.Generator().manual_seed(8))
    before = x.clone()
    with torch.no_grad():
        block(x)
    assert torch.equal(x, before)


def test_the_swiglu_mlp_is_the_plain_one(plain_mlp: ModuleType) -> None:
    torch.manual_seed(2)
    ours, theirs = mlp.SwiGLUMLP(32, 4), plain_mlp.SwiGLUMLP(32, 4)
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))
