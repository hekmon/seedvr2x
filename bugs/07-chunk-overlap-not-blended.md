# 07. `--chunk_size` streaming: the overlap between chunks is context only, never blended

| | |
|---|---|
| Severity | wrong output (hard seam at every chunk boundary) |
| Status | from code, not reproduced |
| Affected options | `--chunk_size` with `--temporal_overlap` |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

With `--chunk_size`, each chunk is a full, independent four-phase run. The code's docstrings and
the help example present `--temporal_overlap` as "overlap between chunks for blending" and
"seamless transitions", but the overlap frames of the previous chunk are only prepended as input
context, and their output is thrown away. Chunk boundaries are hard cuts between two independent
reconstructions, exactly like batch boundaries without overlap.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out/ --output_format png --model_dir /path/to/models \
  --batch_size 21 --chunk_size 42 --temporal_overlap 3 --load_cap 84 --color_correction none
# then compare the change between output frames 41 → 42 with the changes inside a batch,
# on content where the input doesn't move (held frames)
```

- Expected (per the docstrings): the 3 frames around frame 42 cross-faded between chunk 1 and
  chunk 2, as between batches or GPUs.
- Actual (from the code): chunk 2 is computed from input frames 39–83; its first 3 output frames
  are dropped and frame 42 starts a new, independent reconstruction. On held frames the jump at
  frame 42 should be the size of a batch boundary without overlap (≈ 2.3 levels at batch 21 on
  our clip B, against ≈ 1.5 with a real 3-frame blend between batches). Not run.

## Root cause

`_stream_video_chunks` (`inference_cli.py:621-718`). The docstring says (`inference_cli.py:646`):

```python
overlap: Temporal overlap frames between chunks for blending
```

and `process_single_file`'s says "chunks with temporal overlap for seamless transitions between
chunks" (`inference_cli.py:430-431`). The code:

```python
# Prepend context from previous chunk
if prev_raw_tail is not None and overlap > 0:
    context_count = min(overlap, prev_raw_tail.shape[0])
    frames = torch.cat([prev_raw_tail[-context_count:], new_frames], dim=0)
...
# Remove context frames from output
if context_count > 0:
    result = result[context_count:]

# Save tail for next chunk context
prev_raw_tail = new_frames[-overlap:].clone() if overlap > 0 else None
```

(`inference_cli.py:677-709`). The previous chunk's output for those frames has already been
yielded and written (`inference_cli.py:714`, `553-559`), so nothing could be blended anyway.

What the context does buy: the new chunk's first output frame isn't the first frame of its first
batch (that one is a context frame), so it avoids the "first frame less restored" effect
([quality.md](../docs/quality.md#--prepend_frames)). It doesn't smooth the boundary.

Also: batches restart at each chunk, so a `--chunk_size` that isn't a multiple of
`--batch_size − --temporal_overlap` (counting the context frames) ends every chunk with a short,
padded batch ([quality.md](../docs/quality.md#--uniform_batch_size) measured how a short last
batch differs).

## Impact

- Users of streaming mode for long videos (the help's own example is `--chunk_size 330
  --temporal_overlap 3`) get a hard seam every `chunk_size` frames, on top of the batch
  boundaries.
- Workaround: choose `--chunk_size` so that chunk boundaries fall on scene cuts, or as large as
  host RAM allows; make it a multiple of the batch step.

## Possible fix

Keep the previous chunk's last `overlap` output frames unwritten, and blend them with the new
chunk's first `overlap` output frames, which correspond to the same input frames:

```python
# in _stream_video_chunks (pseudo-code)
held_tail = None                       # last `overlap` output frames of the previous chunk
while frames_read < frames_to_process:
    ...
    result = _process_frames_core(frames, ...)
    if held_tail is not None and context_count > 0:
        head = result[:context_count]
        blended = blend_overlapping_frames(held_tail[-context_count:], head, context_count)
        result = torch.cat([blended, result[context_count:]], dim=0)
    is_last = frames_read >= frames_to_process
    if overlap > 0 and not is_last:
        held_tail = result[-overlap:].clone()
        result = result[:-overlap]     # written with the next chunk, after blending
    yield result
```

The context frames are then computed twice (once at the end of chunk k, once at the start of
chunk k+1), as they already are today. The total frame count is unchanged. This relies on the
weights of [06](06-temporal-overlap-blend-weights.md) actually blending; with today's weights an
overlap of 3 would mix one frame.

Alternatively, fix the docstrings and the help to say "context frames, not blended".

Test: the command above on held content; the change across frames 39–42 should drop to the
level of a blended batch boundary, and the output must still have 84 frames, aligned with the
input.

## References

- [cli-flags.md, Resolution, frames and batching](../docs/cli-flags.md#resolution-frames-and-batching) (`--chunk_size`)
- [cli-flags.md, Temporal](../docs/cli-flags.md#temporal)
- [quality.md, `--temporal_overlap`](../docs/quality.md#--temporal_overlap)
