"""The 7B DiT's window attention with less memory live at its peak, installed at run time.

A study of place-study (2026-11-05, Q5), never seedvr2x's code: dit_need.py and
picture_phases.py install it with --lean or --sdpa-batched, to measure what it saves and whether
the DiT's output keeps its bits. The vendored attention (seedvr2x/vendor/models/dit_7b/
nablocks/mmsr_block.py, NaSwinAttention.forward; ByteDance's SeedVR as modified by numz and by
seedvr2x, Apache-2.0) is followed line for line, the arithmetic unchanged; what changes:

lean
- the projection's q, k and v (vid_qkv, 0.56 GiB at a 4K latent) is dropped once its windowed
  copy is made, where the original keeps it to the forward's end;
- q and k, float32 after the norm and RoPE (the norm's float32 weight promotes them), are cast
  to the attention's compute dtype before their concatenation with the text by windows, where
  the original concatenates in float32 and the attention casts after: a cast and a copy commute
  bit for bit, and the attention's cast is then nothing;
- each of q, k and v is dropped once concatenated, the windowed copy of the projection (whose
  view v is) with v;
- under SDPA the windows' outputs are written into one tensor as they come, where the original
  keeps them all and concatenates them at the end (vendor/models/dit_7b/attention.py,
  pytorch_varlen_attention).

lean with chunk N
- the norm of q and k, their RoPE and their cast to the compute dtype run N tokens at a time,
  into one tensor of the compute dtype per q and k, where the original keeps their float32
  outputs whole (each acts on each token alone: the bits to be checked, not assumed);

sdpa_batched
- under SDPA, windows of the same length go through one call, batched (ComfyUI v0.44.0 batches
  its windows of equal length so: comfy/ldm/seedvr, read 2026-11-05, described, not copied):
  fewer calls, and the output's bits to be checked, not assumed."""

import itertools
from typing import Any


def install(lean: bool = True, sdpa_batched: bool = False, chunk: int = 0) -> None:
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    from rotary_embedding_torch import apply_rotary_emb

    from seedvr2x.vendor.models.dit_7b import attention, mm, na
    from seedvr2x.vendor.models.dit_7b.nablocks import mmsr_block

    if sdpa_batched:
        original_varlen = attention.pytorch_varlen_attention

        def batched(
            q: Any,
            k: Any,
            v: Any,
            cu_seqlens_q: Any,
            cu_seqlens_k: Any,
            max_seqlen_q: Any = None,
            max_seqlen_k: Any = None,
            dropout_p: float = 0.0,
            softmax_scale: Any = None,
            causal: bool = False,
            deterministic: bool = False,
        ) -> Any:
            if not torch.equal(cu_seqlens_q, cu_seqlens_k):
                return original_varlen(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    dropout_p, softmax_scale, causal, deterministic,
                )  # fmt: skip
            bounds = cu_seqlens_q.long().cpu().tolist()
            out = torch.empty_like(q)
            runs: dict[int, list[int]] = {}
            for start, end in itertools.pairwise(bounds):
                runs.setdefault(end - start, []).append(start)
            for length, starts in runs.items():
                index = torch.tensor(
                    [s + i for s in starts for i in range(length)], device=q.device
                )
                shape = (len(starts), length, q.shape[1], q.shape[2])

                def gathered(x: Any, index: Any = index, shape: Any = shape) -> Any:
                    return x.index_select(0, index).view(shape).permute(0, 2, 1, 3)

                o = F.scaled_dot_product_attention(
                    gathered(q),
                    gathered(k),
                    gathered(v),
                    dropout_p=dropout_p if not deterministic else 0.0,
                    is_causal=causal,
                )
                out.index_copy_(0, index, o.permute(0, 2, 1, 3).reshape(-1, *q.shape[1:]))
            return out

        attention.pytorch_varlen_attention = batched
    elif lean:

        def written(
            q: Any,
            k: Any,
            v: Any,
            cu_seqlens_q: Any,
            cu_seqlens_k: Any,
            max_seqlen_q: Any = None,
            max_seqlen_k: Any = None,
            dropout_p: float = 0.0,
            softmax_scale: Any = None,
            causal: bool = False,
            deterministic: bool = False,
        ) -> Any:
            q_splits = torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0)
            k_splits = torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0)
            v_splits = torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0)
            out = q.new_empty(q.shape[0], q.shape[1], v.shape[2])
            start = 0
            for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits, strict=True):
                o = F.scaled_dot_product_attention(
                    q_i.permute(1, 0, 2).unsqueeze(0),
                    k_i.permute(1, 0, 2).unsqueeze(0),
                    v_i.permute(1, 0, 2).unsqueeze(0),
                    dropout_p=dropout_p if not deterministic else 0.0,
                    is_causal=causal,
                )
                out[start : start + q_i.shape[0]] = o.squeeze(0).permute(1, 0, 2)
                start += q_i.shape[0]
            return out

        attention.pytorch_varlen_attention = written
    if not lean:
        return

    def forward(
        self: Any, vid: Any, txt: Any, vid_shape: Any, txt_shape: Any, cache: Any
    ) -> tuple[Any, Any]:
        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: Any) -> list[Any]:
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform", lambda: na.window_idx(vid_shape, make_window)
        )
        vid_qkv_win = window_partition(vid_qkv)
        del vid_qkv  # lean: the windowed copy is all that is read from here
        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)
        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))
        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )
        compute = self.attn.compute_dtype
        if chunk and self.rope and compute is not None:
            # lean by chunks: the norm, RoPE and the cast token by token (all three act on each
            # token, and on each head, alone), CHUNK tokens at a time into one tensor of the
            # compute dtype, where the original keeps the float32 norm's and RoPE's outputs whole.
            txt_q, txt_k = self.norm_q.txt_or_all(txt_q), self.norm_k.txt_or_all(txt_k)
            out_dtype = txt_q.dtype
            freqs = cache_win("rope_freqs_3d", lambda: self.rope.get_freqs(window_shape))
            freqs = freqs.to(device=vid_q.device, dtype=out_dtype)

            def roped(raw: Any, norm: Any) -> Any:
                done = torch.empty(raw.shape, dtype=compute, device=raw.device)
                for start in range(0, raw.shape[0], chunk):
                    end = min(start + chunk, raw.shape[0])
                    normed = rearrange(norm(raw[start:end]), "L h d -> h L d")
                    turned = apply_rotary_emb(freqs[start:end], normed.float()).to(normed.dtype)
                    done[start:end].copy_(rearrange(turned, "h L d -> L h d"))
                return done

            vid_q = roped(vid_q, self.norm_q.vid_or_all)
            q = concat_win(vid_q, txt_q.to(compute))
            del vid_q
            vid_k = roped(vid_k, self.norm_k.vid_or_all)
            k = concat_win(vid_k, txt_k.to(compute))
            del vid_k
            v = concat_win(vid_v.to(compute), txt_v.to(compute))
        else:
            vid_q, txt_q = self.norm_q(vid_q, txt_q)
            vid_k, txt_k = self.norm_k(vid_k, txt_k)
            if self.rope:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            # The dtype the output is cast to, as `.type_as(vid_q)` casts it in the original.
            out_dtype = vid_q.dtype
            # lean: cast before the concatenation (a cast and a copy commute bit for bit), each
            # part dropped once concatenated.
            if compute is not None and vid_q.dtype != compute:
                q = concat_win(vid_q.to(compute), txt_q.to(compute))
                del vid_q
                k = concat_win(vid_k.to(compute), txt_k.to(compute))
                del vid_k
                v = concat_win(vid_v.to(compute), txt_v.to(compute))
            else:
                q = concat_win(vid_q, txt_q)
                del vid_q
                k = concat_win(vid_k, txt_k)
                del vid_k
                v = concat_win(vid_v, txt_v)
        del vid_v, vid_qkv_win, txt_q, txt_k, txt_v, txt_qkv
        cu_seqlens = cache_win(
            "vid_seqlens_q",
            lambda: mmsr_block.safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int(),
        )
        cu_seqlens_k = cache_win(
            "vid_seqlens_k",
            lambda: mmsr_block.safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int(),
        )
        out = self.attn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=cache_win("vid_max_seqlen_q", lambda: all_len_win.max()),
            max_seqlen_k=cache_win("vid_max_seqlen_k", lambda: all_len_win.max()),
        )
        del q, k, v
        out = out.to(out_dtype)
        vid_out, txt_out = unconcat_win(out)
        del out
        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)
        vid_out, txt_out = self.proj_out(vid_out, txt_out)
        return vid_out, txt_out

    # The vid and txt modules of an MM pair, as MMModule.forward picks them.
    mm.MMModule.vid_or_all = property(lambda self: self.all if self.shared_weights else self.vid)
    mm.MMModule.txt_or_all = property(lambda self: self.all if self.shared_weights else self.txt)
    mmsr_block.NaSwinAttention.forward = forward


def gelu_inplace() -> None:
    """The 7B's MLP (vendor/models/dit_7b/mlp.py, MLP.forward) with its GELU (tanh) in place on
    proj_in's output, which nothing else reads: the same kernel's values, one tensor of 12,288
    values a token fewer at the MLP's peak."""
    import torch

    from seedvr2x.vendor.models.dit_7b import mlp

    def forward(self: Any, x: Any) -> Any:
        x = self.proj_in(x)
        torch._C._nn.gelu_(x, approximate="tanh")
        return self.proj_out(x)

    mlp.MLP.forward = forward
