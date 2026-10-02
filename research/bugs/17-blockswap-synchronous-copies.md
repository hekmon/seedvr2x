# 17. BlockSwap moves every swapped block in and out synchronously, from pageable memory, with no prefetch

| | |
|---|---|
| Severity | performance |
| Status | measured |
| Affected options | `--blocks_to_swap`, `--swap_io_components` |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

For every forward, each swapped block is copied CPU → GPU, run, and copied back GPU → CPU,
synchronously, from and into pageable memory. Nothing overlaps with compute, and the copy back
re-transfers weights that never changed. With 36 swapped 7B fp16 blocks this costs 4.3 s per
batch on a PCIe 5.0 link (+97% DiT time at 1080p batch 5). The copy back, the slow direction, is
not needed at all.

## Reproduction

```bash
python scripts/bench.py run swapprobe-1080-bs5-fp16-swap36 --wrap scripts/swap_probe.py \
  --seedvr2-dir /path/to/seedvr2 --runs-dir runs -- input.mp4 --model_dir /path/to/models \
  --dit_model seedvr2_ema_7b_fp16.safetensors --dit_offload_device cpu --blocks_to_swap 36 \
  --resolution 1080 --batch_size 5 --load_cap 10
```

Measured per block move (synchronized, `swap_probe.py`), 1080p batch 5:

| Model | Block size | In: CPU → GPU | Out: GPU → CPU | Moves per batch (36 blocks) | Block compute per batch |
|---|---|---|---|---|---|
| 7B fp16 | 0.423 GiB | 24.7 ms, 18 GB/s | 95 ms, 4.8 GB/s | 4.3 s | 5.4 s |
| 7B fp8 | 0.212 GiB | 13.4 ms, 17 GB/s | 52 ms, 4.5 GB/s | 2.4 s | 5.3 s |
| 7B Q4_K_M | 0.119 GiB | 6.1 ms, 21 GB/s | 18.2 ms, 7.0 GB/s | 0.88 s | 7.1 s |

End to end (7B fp16): +0.07–0.10 s per block and batch; all 36 blocks add 97% to the DiT time at
batch 5, 20% at batch 21, 12% at batch 45. The same link does 57 GB/s each way from pinned
memory.

- Expected: swapping costs close to nothing while a block's compute (≈ 150 ms for 7B fp16 at
  batch 5) is longer than its upload (25 ms).
- Actual: each block's compute waits for its upload, and is followed by a 95 ms download.

## Root cause

`_wrap_block_forward` (`src/optimization/blockswap.py:487-522`):

```python
current_device = next(self.parameters()).device
target_device = torch.device(model.main_device)

if current_device != target_device:
    self.to(model.main_device, non_blocking=False)

# Execute forward pass with OOM protection
output = original_forward(*args, **kwargs)

# Move back to offload device
self.to(model.offload_device, non_blocking=False)
...
clear_memory(debug=debug, deep=False, force=False, timer_name="wrap_block_forward")
```

- `Module.to(cuda)` allocates new GPU tensors and copies from pageable host memory: the driver
  stages through a bounce buffer, synchronously (18 GB/s instead of 57).
- `Module.to(cpu)` allocates fresh pageable host tensors and copies back (4.5–7 GB/s; new
  pageable allocations are the slow direction), although the CPU already had these exact
  weights before the upload.
- Block k+1's upload only starts after block k's download; nothing runs on a side stream.
- `_wrap_io_forward` does the same for the I/O modules (`blockswap.py:567-599`).
- The timing helpers are `@torch._dynamo.disable`, and the moves happen inside `forward`, so
  `--compile_dit` breaks its graph at every swapped block and gains nothing with BlockSwap
  ([vram.md](../docs/vram.md#torchcompile)).

## Impact

- Every BlockSwap user, i.e. everyone on a GPU too small for the DiT: up to 2× DiT time at small
  batches with fp16 weights, and about twice that on PCIe 4.0 or with slower RAM.
- More host RAM churn: each forward allocates and frees 36 × 0.42 GiB of pageable memory (max RSS
  34 GiB with 36 fp16 blocks against 19 GiB without swap).

## Possible fix

Keep one pinned CPU copy of each swapped block for the whole run, upload ahead on a side stream,
and simply drop the GPU copy after use:

```python
# pseudo-code, at BlockSwap setup
for blk in swapped_blocks:
    blk.cpu_state = {n: t.detach().cpu().pin_memory() for n, t in blk.state_dict(keep_vars=True).items()}
    blk.to_empty(device="meta")           # no GPU or pageable copy kept
copy_stream = torch.cuda.Stream()

def upload(blk):                          # async, on copy_stream
    with torch.cuda.stream(copy_stream):
        gpu = {n: t.to(device, non_blocking=True) for n, t in blk.cpu_state.items()}
        blk.ready = torch.cuda.Event(); blk.ready.record(copy_stream)
    blk.load_state_dict(gpu, assign=True)

# in the wrapped forward of block k
torch.cuda.current_stream().wait_event(blk.ready)
if k + 1 is swapped: upload(next_block)  # overlap with this block's compute
out = original_forward(...)
release(blk)                              # to_empty('meta') after the compute stream is done with it
```

- The download disappears (the weights are read-only at inference).
- Uploads run at pinned-memory bandwidth (≈ 57 GB/s here), overlapped with the previous block's
  compute: for 7B fp16 at batch 5, 8 ms per block hidden behind 150 ms of compute.
- Memory: one extra block in flight on the GPU (0.42 GiB for 7B fp16). Pinned host memory equals
  the swapped weights (which already live in RAM today), but pinned memory is page-locked: cap
  it, or fall back to pageable when pinning fails.
- GGUF blocks hold their weights in buffers (`GGUFTensor`); the same scheme applies to buffers.
- Freeing a GPU copy must wait until the kernels that read it are done (record an event on the
  compute stream, or rely on the caching allocator's stream semantics with `record_stream`).

Test: `swap_probe.py` must show no GPU → CPU moves and uploads overlapping compute; with
`bench.py`, `dit-1080-bs5-swap36` should come within ≈ 10% of the no-swap DiT time (3.7 s per
batch), with the same output bits.

## References

- [vram.md, BlockSwap](../docs/vram.md#blockswap)
- [cli-flags.md, Devices, offload and BlockSwap](../docs/cli-flags.md#devices-offload-and-blockswap)
