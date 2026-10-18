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
# Modified for seedvr2x: no retry on OOM; no Conv3d workaround; no sequence parallelism; a split
# convolution's pieces written into one output where a slice of four frames makes it large.

import math
from contextlib import contextmanager
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d

from .context_parallel_lib import cache_send_recv, get_cache_size
from .global_config import get_norm_limit
from .types import MemoryState, _inflation_mode_t, _memory_device_t
from ....common.half_precision_fixes import safe_pad_operation

# Single GPU inference - no distributed processing needed
#print("Warning: Using single GPU inference mode - distributed features disabled in causal_inflation_lib")

# seedvr2x: numz's sequence-parallel stubs removed (DESIGN.md, Vendored model code). Only
# get_sequence_parallel_group was called, in InflatedCausalConv3d.forward, and it returned None.


# seedvr2x: the size, in bytes, from which the output of a convolution cut in pieces is allocated
# whole before its pieces are computed, and they are written into it (memory_limit_conv); under
# it the pieces are concatenated once all are made, as numz does. PROVISIONAL (implementation,
# 2026-10-13), between two measures on cards packed to the planner's margin
# (research/docs/vae-room.md). Written into one output, which is then held through its pieces'
# convolutions, a small tile's pass peaks higher: a decode in 640-px tiles, whose largest output
# is 0.78 GiB, by 0.27 GiB, one in 832-px tiles (1.32) by 0.50, one in tiles of 1104 x 1152
# (2.43) by 0.04. A large one's peaks lower, the concatenation's two copies being what it peaked
# at: a decode in tiles of 1104 x 1600 (3.37 GiB) by 0.80 GiB, one in 1664-px tiles (5.28) by
# 2.85. Those are the outputs of a slice of four frames, and so is what the size is held to
# (whole_output; design, 2026-10-16, the size PROVISIONAL): judged at each call, a pass's first
# slice, which holds one frame more than its others at every convolution (five frames where they
# hold four, three where two, two where one), took the path where its others didn't, a decode in
# tiles of 1104 x 1152 on its first slice alone, its largest output 3.03 GiB there by its shape.
ONE_OUTPUT_FROM = round(2.5 * 1024**3)


@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.
        self.memory_limit = float("inf")

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    # seedvr2x: numz's _conv_forward override is gone. It called torch.cudnn_convolution directly,
    # for a Conv3d memory bug (about 3x) with torch >= 2.9 and cuDNN >= 9.10.2. On the pinned
    # stack (torch 2.14.1, cuDNN 9.24) the standard path gives the same output, bit for bit,
    # and the same peak memory and time (research/docs/vram.md:240-254).

    # seedvr2x: head and whole, for the output's path (whole_output): the frames of this call's
    # output that a slice of four frames doesn't give, and the path a piece's own pieces take.
    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
        head=0,
        whole=None,
    ):
        # Compatible with no limit.
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return super().forward(x)

        # Compute tensor shape after concat & padding.
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            x_concat = x
            if prev_cache is not None:
                x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)
            
            def pad_and_forward():
                padded = safe_pad_operation(x_concat, padding, mode='constant', value=0.0)
                with ignore_padding(self):
                    return Conv3d.forward(self, padded)
            
            # seedvr2x: a direct call, without numz's retry_on_oom.
            return pad_and_forward()

        # Exceed memory limit, splitting tensor

        # Split input (& prev_cache).
        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        # seedvr2x: where the output a slice of four frames gives is ONE_OUTPUT_FROM bytes or more
        # (whole_output), the pieces' outputs are written into one output allocated here, before the
        # first of them is computed, where numz concatenates them once all are made (output =
        # torch.cat(x, split_dim)). That concatenation was a large pass's peak and its largest
        # request, a whole tile's output asked for in one piece while the pieces, as large together,
        # were still there: packed to the planner's margin, a 1080p decode in 1600-px tiles was
        # refused it 3 times of 3 (3.37 GiB asked with 26.21 allocated on a card emulated at 32 GB,
        # the allocator's pool holding 30.53 and the card full), and a 4K decode in 1664-px tiles
        # every time (5.28 GiB with 40.55 allocated, the pool at 46.53, on 48 GB). Written so, both
        # ran, with the frames of the concatenation where it ran (2026-10-13;
        # research/docs/vae-room.md).
        output = self.whole_output(x[0], shape, head, whole)
        written = 0
        # Loop Fwd.
        cache = None
        for idx in range(len(x)):
            # Concat prev cache from last dim
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            # Get padding pattern.
            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            # Prepare cache for next slice (this dim).
            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)
                )

            # Recursive.
            # seedvr2x: the piece's output goes to its place in the one output and is let go,
            # with its input, which numz's list kept the output in place of; a piece cut again
            # takes the path its convolution's call took (whole_output).
            piece = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache,
                whole=output is not None,
            )
            if output is None:
                # A small output: kept in numz's list, for its concatenation.
                x[idx] = piece
            else:
                x[idx] = None
                length = piece.size(split_dim)
                place = list(output.shape)
                place[split_dim] = length
                if (
                    list(piece.shape) != place
                    or piece.dtype != output.dtype
                    or written + length > output.size(split_dim)
                ):
                    raise RuntimeError(
                        f"a piece of {tuple(piece.shape)} {piece.dtype} at {written} along"
                        f" dimension {split_dim} of an output of {tuple(output.shape)}"
                        f" {output.dtype}"
                    )
                output.narrow(split_dim, written, length).copy_(piece)
                written += length
            del piece

            # Update cache.
            cache = next_cache

        # seedvr2x: a small output is numz's concatenation, a direct call, without its
        # retry_on_oom; a large one is whole already, and one left partly unwritten would be
        # read as it is.
        if output is None:
            return torch.cat(x, split_dim)
        if written != output.size(split_dim):
            raise RuntimeError(
                f"{written} of {output.size(split_dim)} written along dimension {split_dim}"
            )
        return output

    # seedvr2x: the output of a convolution cut in pieces (memory_limit_conv), allocated before
    # its pieces are computed: uninitialised, in the input's dtype and on its device, of the shape
    # the convolution gives the whole padded input (padded: its shape, as memory_limit_conv counts
    # it, the cache ahead and the padding in). None where the pieces are concatenated, numz's
    # path. Taken where the output a slice of four frames gives is ONE_OUTPUT_FROM bytes or more:
    # this output less its head, the frames a pass's first slice, or a pass in one slice, has more
    # than its slices of four (one, at every convolution: slicing_forward), so that every slice of
    # a pass takes one path. A pass of one frame has no slice of four, and is concatenated
    # (PROVISIONAL, implementation, 2026-10-16). whole, where given, is the path already chosen: a
    # piece cut again takes its convolution's.
    def whole_output(self, like, padded, head=0, whole=None):
        shape = [int(padded[0]), self.out_channels]
        for dim in range(3):
            dilated_kernel_size = self.dilation[dim] * (self.kernel_size[dim] - 1) + 1
            shape.append((int(padded[2 + dim]) - dilated_kernel_size) // self.stride[dim] + 1)
        if whole is None:
            frames = max(shape[2] - head, 0)
            whole = math.prod(shape) // shape[2] * frames * like.element_size() >= ONE_OUTPUT_FROM
        return like.new_empty(shape) if whole else None

    def forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        if (
            math.isinf(self.memory_limit)
            and torch.is_tensor(input)
            # seedvr2x: "and get_sequence_parallel_group() is None" dropped: always True, as numz's
            # stub returned None.
        ):
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")
        return super().forward(input)

    def slicing_forward(
        self,
        input: Union[Tensor, List[Tensor]],
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2
        )

        # Single GPU inference - simplified memory management
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            if cache_size <= input[-1].size(2):
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()
                if self.memory_device == "cpu" and self.memory is not None:
                    self.memory = self.memory.to("cpu")

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            # seedvr2x: head, the output's frames a slice of four frames doesn't give, for the
            # path of its pieces (whole_output): a pass's first slice, and a pass in one slice, the
            # memory not active, hold one frame more than its slices of four at every
            # convolution of the VAE, the first frame (tests/test_causal_conv.py).
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache,
                head=int(i == 0 and memory_state != MemoryState.ACTIVE),
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

    def tflops(self, args, kwargs, output) -> float:
        if torch.is_tensor(output):
            output_numel = output.numel()
        elif isinstance(output, list):
            output_numel = sum(o.numel() for o in output)
        else:
            raise NotImplementedError
        return (2 * math.prod(self.kernel_size) * self.in_channels * (output_numel / 1e6)) / 1e6

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = modify_state_dict(
                self,
                state_dict,
                prefix,
                inflate_weight_fn=inflate_weight,
                inflate_bias_fn=inflate_bias,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def init_causal_conv3d(
    *args,
    inflation_mode: _inflation_mode_t,
    **kwargs,
):
    """
    Initialize a Causal-3D convolution layer.
    Parameters:
        inflation_mode: Listed as below. It's compatible with all the 3D-VAE checkpoints we have.
            - none: No inflation will be conducted.
                    The loading logic of state dict will fall back to default.
            - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
    """
    return InflatedCausalConv3d(*args, inflation_mode=inflation_mode, **kwargs)


def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm_layer, (nn.LayerNorm, RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy > get_norm_limit():
                num_chunks = min(4 if x.element_size() == 2 else 2, norm_layer.num_groups)
                assert norm_layer.num_groups % num_chunks == 0
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                x = list(x.chunk(num_chunks, dim=1))
                weights = norm_layer.weight.chunk(num_chunks, dim=0)
                biases = norm_layer.bias.chunk(num_chunks, dim=0)
                
                for i, (w, b) in enumerate(zip(weights, biases)):
                    def apply_group_norm():
                        return F.group_norm(x[i], num_groups_per_chunk, w, b, norm_layer.eps)
                    
                    # seedvr2x: a direct call, without numz's retry_on_oom.
                    x[i] = apply_group_norm()
                    x[i] = x[i]
                
                # seedvr2x: a direct call, without numz's retry_on_oom.
                x = torch.cat(x, dim=1)
            else:
                # seedvr2x: a direct call, without numz's retry_on_oom.
                x = norm_layer(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x
    raise NotImplementedError


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    # Single GPU inference - always process
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution bias tensor to a 3D one
    Parameters:
        bias_2d:        The bias tensor of 2D conv to be inflated.
        bias_3d:        The bias tensor of 3D conv to be initialized.
        inflation_mode: Placeholder to align `inflate_weight`.
    """
    assert bias_3d.shape == bias_2d.shape
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
            # It's a 3d state dict, should not do inflation on both bias and weight.
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            # Assuming the 2D biases are 1D tensors (out_channels,)
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[bias_name] = bias_3d
    return state_dict
