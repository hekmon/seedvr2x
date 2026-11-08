# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# Modified for seedvr2x: no sequence parallelism; the attention keeps less live at its peak; the
# windows' shapes, lengths and sizes made once a forward, read on the host.

from typing import Tuple, Union
import torch
from einops import rearrange
from rotary_embedding_torch import apply_rotary_emb
from torch.nn import functional as F

# from ..cache import Cache
from ....common.cache import Cache

from .. import na
from ..attention import FlashAttentionVarlen
from ..blocks.mmdit_window_block import MMWindowAttention, MMWindowTransformerBlock
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..rope import NaRotaryEmbedding3d
from ..window import get_window_op
from ....common.half_precision_fixes import safe_pad_operation

# seedvr2x: the tokens whose q and k go through the norm, RoPE and the cast to the compute dtype
# at a time (forward below). 4,096 tokens a chunk lowers a 4K picture's DiT step from 5.53 to 3.08
# GiB together with the other lean changes, the same bits (DESIGN.md, Allocator, Small cards;
# research/docs/small-cards.md, the DiT section; research/scripts/dit_lean.py measured it, with
# the chunk as its parameter).
LEAN_CHUNK = 4096


class NaSwinAttention(MMWindowAttention):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        qk_norm_eps: float,
        window: Union[int, Tuple[int, int, int]],
        window_method: str,
        shared_qkv: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            window=window,
            window_method=window_method,
            shared_qkv=shared_qkv,
        )
        self.rope = NaRotaryEmbedding3d(dim=head_dim // 2) if qk_rope else None
        self.attn = FlashAttentionVarlen(attention_mode=attention_mode)
        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)
        # seedvr2x: no sequence parallelism (DESIGN.md, Vendored model code):
        # gather_seq_scatter_heads_qkv returned vid_qkv and txt_qkv unchanged, as no
        # sequence-parallel group is ever set up.

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        # seedvr2x: the windows' indices, lengths and sizes held for the forward (Cache.held), where
        # every block made them again, each making copying shapes and lengths from the device, a
        # wait in C (DESIGN.md, Allocator, "Out of the stop's reach"). Small: a few int64 values a
        # token. The same values, so the same bits.
        window_partition, window_reverse, window_shape, window_count = cache_win.held(
            "win_transform",
            lambda: na.window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)
        # seedvr2x: lean attention (DESIGN.md, Allocator, Small cards: a 4K picture's DiT step
        # 5.53 to 3.08 GiB, a 1080p window of 5 latents 6.58 to 3.57, the same bits under SDPA
        # and FlashAttention 2): the projection's q, k and v are freed once windowed, where numz
        # keeps them to the forward's end.
        del vid_qkv

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        txt_len = cache.held("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win.held("vid_len", lambda: window_shape.prod(-1))
        txt_len_win = cache_win.held("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win.held("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win.held(
            "mm_pnp", lambda: na.repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        compute = self.attn.compute_dtype
        if self.rope and compute is not None:
            # seedvr2x: the pipeline's path (model.py sets compute_dtype on every attention
            # module). The norm, the window RoPE and the cast to the compute dtype act on each
            # token, and on each head, alone: they run LEAN_CHUNK tokens at a time into one tensor
            # of the compute dtype, where numz keeps the float32 outputs of the norm (its float32
            # weight promotes q and k) and of RoPE whole and casts them in the attention, after
            # their concatenation with the text. A cast and a copy commute, so the attention's
            # own cast is then nothing and the bits are numz's.
            txt_q, txt_k = self.norm_q.txt_or_all(txt_q), self.norm_k.txt_or_all(txt_k)
            out_dtype = txt_q.dtype
            # seedvr2x: the windows' shape read on the host (get_freqs lists it), held; the
            # frequencies, a token's size, made in each block as before.
            shape = cache_win.held("window_shape_host", lambda: window_shape.cpu())
            freqs = cache_win("rope_freqs_3d", lambda: self.rope.get_freqs(shape))
            freqs = freqs.to(device=vid_q.device, dtype=out_dtype)

            def norm_and_rope(raw: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
                done = torch.empty(raw.shape, dtype=compute, device=raw.device)
                for start in range(0, raw.shape[0], LEAN_CHUNK):
                    end = min(start + LEAN_CHUNK, raw.shape[0])
                    normed = rearrange(norm(raw[start:end]), "L h d -> h L d")
                    turned = apply_rotary_emb(freqs[start:end], normed.float()).to(normed.dtype)
                    done[start:end].copy_(rearrange(turned, "h L d -> L h d"))
                return done

            # Each part is freed once concatenated.
            vid_q = norm_and_rope(vid_q, self.norm_q.vid_or_all)
            q = concat_win(vid_q, txt_q.to(compute))
            del vid_q
            vid_k = norm_and_rope(vid_k, self.norm_k.vid_or_all)
            k = concat_win(vid_k, txt_k.to(compute))
            del vid_k
            v = concat_win(vid_v.to(compute), txt_v.to(compute))
        else:
            # seedvr2x: numz's order, for a module whose compute dtype is not set or without
            # RoPE (the 7B's config has RoPE; no pipeline leaves the dtype unset), the casts
            # before the concatenations as above, each part freed once concatenated.
            vid_q, txt_q = self.norm_q(vid_q, txt_q)
            vid_k, txt_k = self.norm_k(vid_k, txt_k)

            # window rope
            if self.rope:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

            out_dtype = vid_q.dtype
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

        # seedvr2x: the attention's lengths held, and given on the host, where SDPA's split reads
        # them (attention.py copied them there in every block), the longest as an int.
        seqlens = cache_win.held(
            "vid_seqlens", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
        )
        seqlens = cache_win.held("vid_seqlens_host", lambda: seqlens.cpu())
        longest = cache_win.held("vid_max_seqlen", lambda: int(all_len_win.max()))
        # Attention handles dtype conversion internally using pipeline compute_dtype
        out = self.attn(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=seqlens,
            cu_seqlens_k=seqlens,
            max_seqlen_q=longest,
            max_seqlen_k=longest,
        )
        # seedvr2x: q, k and v freed before the output is cast, and the cast to the dtype of q
        # after the norm and RoPE (numz's `.type_as(vid_q)`, vid_q being gone by then).
        del q, k, v
        out = out.to(out_dtype)

        # text pooling
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        # seedvr2x: no sequence parallelism, as above: gather_heads_scatter_seq returned vid_out
        # and txt_out unchanged.

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out


class NaMMSRTransformerBlock(MMWindowTransformerBlock):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_rope: bool,
        qk_norm: norm_layer_type,
        shared_qkv: bool,
        shared_mlp: bool,
        mlp_type: str,
        **kwargs,
    ):
        super().__init__(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            emb_dim=emb_dim,
            heads=heads,
            head_dim=head_dim,
            expand_ratio=expand_ratio,
            norm=norm,
            norm_eps=norm_eps,
            ada=ada,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            shared_qkv=shared_qkv,
            shared_mlp=shared_mlp,
            mlp_type=mlp_type,
            **kwargs,
        )

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_rope=qk_rope,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            shared_qkv=shared_qkv,
            **kwargs,
        )

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        # seedvr2x: held for the forward, as the attention's lengths are (NaSwinAttention).
        hid_len = MMArg(
            cache.held("vid_len", lambda: vid_shape.prod(-1)),
            cache.held("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape
